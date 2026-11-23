"""
可微前向光学模型（PyTorch）。

本模块提供两套把 (散射参数, 吸收组分浓度) 映射到反射率谱的前向模型。
两者对「绝对尺度是否可辨识」给出**不同**答案，这正是仿真自检要判定的事。

------------------------------------------------------------------------
模型 A: 半无限浑浊介质（Kubelka-Munk / 扩散近似）
------------------------------------------------------------------------
    mu_s'(lam) = a * (lam/lam0)^(-b)          Mie 散射幂律
    mu_a(lam)  = sum_i c_i * eps_i(lam)       Beer-Lambert 基组分解
    R = f(mu_a, mu_s')

**理论上的关键性质**：半无限介质没有内在长度尺度，R 只能通过无量纲的
输运反照率 a' = mu_s'/(mu_a+mu_s') 依赖光学参数。于是对任意 k>0：

    (mu_a, mu_s') -> (k*mu_a, k*mu_s')   给出完全相同的 R 谱

=> **绝对尺度不可辨识**。可辨识的只有：
     b          散射幂律指数（形状参数，无量纲，完全可辨识）
     c_i / a    浓度与散射幅值之比
     c_i / c_j  浓度之间的比值
   不可辨识：a 单独、c_i 单独。

这正是空间分辨(SR)/时间分辨(TR)/频域(FD)方法存在的理由——它们各自引入
一个长度尺度（源探距离 / 光飞行时间 / 调制波长）来破除这个简并。

**对本课题的影响**：b 与浓度比值都完好，而
    b               <-> 微结构 / 质地 / 硬度
    c_cab / c_car   <-> 成熟度（叶绿素降解、类胡萝卜素累积）
这两条正是 E5 生理验证的靶子，都不依赖绝对尺度。

------------------------------------------------------------------------
模型 B: PROSPECT-D 平板模型
------------------------------------------------------------------------
N 层等效平板，每层光学厚度 k(lam) = sum_i c_i*eps_i(lam) / N。
有限层数 N 引入长度尺度，遥感文献中 PROSPECT 反演能给出绝对浓度
(Cab in ug/cm2 等)。本模块忠实实现 Stokes 解以便与模型 A 直接对照。

参考: Feret, Gitelson, Noble & Jacquemoud (2017), PROSPECT-D,
      Remote Sensing of Environment. 实现参照 jgomezdans/prosail。
"""
from __future__ import annotations

import math

import torch

__all__ = ["scattering_powerlaw", "absorption_from_basis",
           "kubelka_munk_R", "diffusion_R", "prospect_RT", "exp_integral_E1"]

EPS = 1e-9


# ====================================================================== 组件
def scattering_powerlaw(a: torch.Tensor, b: torch.Tensor,
                        wl: torch.Tensor, wl0: float = 700.0) -> torch.Tensor:
    """mu_s'(lam) = a * (lam/lam0)^(-b)。a,b: (...,1) 或标量；wl: (L,) -> (...,L)"""
    if a.dim() == 0:
        a = a.reshape(1)
    if b.dim() == 0:
        b = b.reshape(1)
    if a.shape[-1] != 1:
        a = a.unsqueeze(-1)
    if b.shape[-1] != 1:
        b = b.unsqueeze(-1)
    return a * (wl / wl0).pow(-b)


def absorption_from_basis(c: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """mu_a(lam) = sum_i c_i * eps_i(lam)。c: (...,K), eps: (K,L) -> (...,L)"""
    return c @ eps


# ============================================================== 半无限模型
def kubelka_munk_R(mu_a: torch.Tensor, mu_s: torch.Tensor) -> torch.Tensor:
    """Kubelka-Munk 半无限反射率。

    F(R) = (1-R)^2/(2R) = K/S。令 x = K/S，解得 [0,1] 内的根：
        R = 1 + x - sqrt(x^2 + 2x)

    R 只依赖 x = mu_a/mu_s' —— 绝对尺度简并在此一目了然。
    """
    x = mu_a / (mu_s + EPS)
    return 1.0 + x - torch.sqrt(x * x + 2.0 * x + EPS)


def diffusion_R(mu_a: torch.Tensor, mu_s: torch.Tensor,
                n_rel: float = 1.4) -> torch.Tensor:
    """半无限介质扩散近似漫反射率。

        R(a') = a'/2 * [1 + exp(-4/3 * A * sqrt(3(1-a')))] * exp(-sqrt(3(1-a')))

    a' = mu_s'/(mu_a+mu_s') 为输运反照率，A 由折射率决定的内反射参数。
    与 K-M 一样只通过 a' 依赖参数，同样存在绝对尺度简并。
    """
    r_i = -1.440 / n_rel**2 + 0.710 / n_rel + 0.668 + 0.0636 * n_rel
    A = (1.0 + r_i) / (1.0 - r_i)
    alb = mu_s / (mu_a + mu_s + EPS)
    s = torch.sqrt(torch.clamp(3.0 * (1.0 - alb), min=EPS))
    return 0.5 * alb * (1.0 + torch.exp(-4.0 / 3.0 * A * s)) * torch.exp(-s)


# ================================================================ PROSPECT
def exp_integral_E1(x: torch.Tensor) -> torch.Tensor:
    """指数积分 E1(x)，可微。Abramowitz & Stegun 有理逼近（5.1.53 / 5.1.56）。"""
    x = torch.clamp(x, min=1e-12)
    g = 0.57721566
    small = (-torch.log(x) - g
             + 0.99999193 * x
             - 0.24991055 * x**2
             + 0.05519968 * x**3
             - 0.00976004 * x**4
             + 0.00107857 * x**5)
    num = x * x + 2.334733 * x + 0.250621
    den = x * x + 3.330657 * x + 1.681534
    large = torch.exp(-x) / x * num / den
    return torch.where(x <= 1.0, small, large)


def _tav(theta_deg: float, n: torch.Tensor) -> torch.Tensor:
    """入射角 theta 内的平均透射率 tav(theta, n)（Stern / Allen 公式）。"""
    if theta_deg == 0.0:
        return 4.0 * n / (n + 1.0) ** 2

    theta = math.radians(theta_deg)
    n2 = n * n
    npx = n2 + 1.0
    nm = n2 - 1.0
    a = (n + 1.0) ** 2 / 2.0
    k = -((n2 - 1.0) ** 2) / 4.0
    sa2 = math.sin(theta) ** 2

    b2 = sa2 - npx / 2.0
    # theta=90 时 b2^2 + k 恒等于 0（因 b2=(1-n^2)/2, k=-(n^2-1)^2/4），
    # 故 clamp 后 b1=0，无需特判。
    b1 = torch.sqrt(torch.clamp(b2 * b2 + k, min=0.0))
    # 注意符号：参考实现是 b = b1 - b2。写成 b2 - b1 会让 b 变负，
    # 后面 log(b/a) 立刻产生 NaN —— 这是本实现最初的 bug。
    b = b1 - b2
    b3, a3 = b**3, a**3

    ts = (k**2 / (6.0 * b3) + k / b - b / 2.0) - (k**2 / (6.0 * a3) + k / a - a / 2.0)
    tp1 = -2.0 * n2 * (b - a) / npx**2
    tp2 = -2.0 * n2 * npx * torch.log(b / a) / nm**2
    tp3 = n2 * (1.0 / b - 1.0 / a) / 2.0
    tp4 = (16.0 * n2**2 * (n2**2 + 1.0)
           * torch.log((2.0 * npx * b - nm**2) / (2.0 * npx * a - nm**2))
           / (npx**3 * nm**2))
    tp5 = (16.0 * n2**3 * (1.0 / (2.0 * npx * b - nm**2)
                           - 1.0 / (2.0 * npx * a - nm**2)) / npx**3)
    return (ts + tp1 + tp2 + tp3 + tp4 + tp5) / (2.0 * sa2)


def prospect_RT(N: torch.Tensor, c: torch.Tensor, eps: torch.Tensor,
                n_refr: torch.Tensor) -> tuple:
    """PROSPECT-D 平板模型，返回 (R, T)。

    N   : (...,) 或 (...,1)  等效层数（结构参数），要求 > 1
    c   : (..., K)  各吸收组分浓度
    eps : (K, L)    比吸收系数
    n_refr : (L,)   材料折射率

    实现严格遵循 PROSPECT 的 Stokes 解，与参考实现 jgomezdans/prosail 一致。
    """
    if N.dim() == 0:
        N = N.reshape(1)
    Nv = N.unsqueeze(-1) if N.shape[-1] != 1 else N
    Nv = torch.clamp(Nv, min=1.0 + 1e-4)

    # --- 单层光学厚度与透射率 tau = (1-k)e^{-k} + k^2 E1(k)，k<=0 时 tau=1
    kall = absorption_from_basis(c, eps) / Nv
    kk = torch.clamp(kall, min=1e-12)
    tau = torch.where(kall > 0,
                      (1.0 - kk) * torch.exp(-kk) + kk * kk * exp_integral_E1(kk),
                      torch.ones_like(kk))
    tau = torch.clamp(tau, min=1e-12, max=1.0)

    # --- 单层的界面项（严格对齐 prosail.refl_trans_one_layer）
    nr = n_refr
    talf = _tav(40.0, nr)
    ralf = 1.0 - talf
    t12 = _tav(90.0, nr)
    r12 = 1.0 - t12
    t21 = t12 / (nr * nr)
    r21 = 1.0 - t21

    denom0 = torch.clamp(1.0 - r21 * r21 * tau * tau, min=EPS)
    Ta = talf * tau * t21 / denom0          # 顶面（含 alpha 入射锥）
    Ra = ralf + r21 * tau * Ta
    t = t12 * tau * t21 / denom0            # 内部层
    r = r12 + r21 * tau * t

    # --- N 层堆叠（Stokes 1862），严格对齐 prosail.prospect_d
    Dq = torch.sqrt(torch.clamp((1.0 + r + t) * (1.0 + r - t)
                                * (1.0 - r + t) * (1.0 - r - t), min=0.0))
    rq, tq = r * r, t * t
    va = (1.0 + rq - tq + Dq) / torch.clamp(2.0 * r, min=EPS)
    vb = (1.0 - rq + tq + Dq) / torch.clamp(2.0 * t, min=EPS)

    bNm1 = vb.pow(Nv - 1.0)
    bN2 = bNm1 * bNm1
    a2 = va * va
    den = a2 * bN2 - 1.0
    den = torch.where(den.abs() < EPS, torch.full_like(den, EPS), den)
    Rsub = va * (bN2 - 1.0) / den
    Tsub = bNm1 * (a2 - 1.0) / den

    # 零吸收特例（r+t >= 1）
    zero_abs = (r + t) >= 1.0
    Tsub_z = t / torch.clamp(t + (1.0 - t) * (Nv - 1.0), min=EPS)
    Tsub = torch.where(zero_abs, Tsub_z, Tsub)
    Rsub = torch.where(zero_abs, 1.0 - Tsub_z, Rsub)

    # --- 顶层与其余 N-1 层合成
    den2 = torch.clamp(1.0 - Rsub * r, min=EPS)
    TN = Ta * Tsub / den2
    RN = Ra + Ta * Rsub * t / den2
    return torch.clamp(RN, 0.0, 1.0), torch.clamp(TN, 0.0, 1.0)
