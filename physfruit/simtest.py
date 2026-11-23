"""
仿真自检 —— 判定物理反演的可辨识性（生死线实验）。

要回答的三个问题
----------------
Q1 我写的 PROSPECT 实现对不对？（与官方 prosail 参考实现对拍）
Q2 半无限模型的绝对尺度简并，在数值上确实存在吗？
Q3 用梯度下降从合成光谱反演，哪些参数能恢复、哪些不能？

为什么先做直接优化而不是训练神经网络
------------------------------------
若直接对参数做梯度下降都恢复不出来，那是**可辨识性**问题，
任何编码器网络都救不了。先用直接优化把可辨识性与可学习性分开，
避免把「物理上不可能」误判成「网络没训好」。

用法:
    python physfruit/simtest.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from physfruit.basis import build_basis  # noqa: E402
from physfruit.forward import (  # noqa: E402
    absorption_from_basis, diffusion_R, kubelka_munk_R, prospect_RT,
    scattering_powerlaw,
)

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64            # 反演对数值精度敏感，统一用 float64


def hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ============================================================ Q1 实现正确性
def test_prospect_matches_reference() -> None:
    hr("[Q1] 我的 PyTorch PROSPECT 与官方 prosail 参考实现对拍")
    try:
        import prosail
    except Exception as e:  # noqa: BLE001
        print("  prosail 不可用，跳过 (%s)" % e)
        return

    # prosail.run_prospect 的签名: (n, cab, car, cbrown, cw, cm, ant=...)
    n_, cab, car, cbrown, cw, cm, ant = 1.5, 40.0, 10.0, 0.0, 0.015, 0.009, 2.0
    try:
        wl_ref, r_ref, _t_ref = prosail.run_prospect(
            n_, cab, car, cbrown, cw, cm, ant=ant, prospect_version="D")
    except Exception as e:  # noqa: BLE001
        print("  调用 prosail 失败，跳过 (%s)" % e)
        return

    # 用同一波长网格构造我们的基谱（400-2500 全谱，便于逐点对比）
    from physfruit.basis import _load_raw, COLS
    raw = _load_raw()
    src_wl = raw[:, 0]
    idx = {c: i for i, c in enumerate(COLS)}
    order = ["cab", "car", "anth", "brown", "water", "dm"]
    eps = torch.tensor(np.stack([raw[:, idx[o]] for o in order]), dtype=DT)
    n_refr = torch.tensor(raw[:, idx["n_refr"]], dtype=DT)
    c = torch.tensor([[cab, car, ant, cbrown, cw, cm]], dtype=DT)
    N = torch.tensor([n_], dtype=DT)

    R, _T = prospect_RT(N, c, eps, n_refr)
    R = R.squeeze(0).numpy()

    m = np.isin(np.round(src_wl).astype(int), np.round(wl_ref).astype(int))
    r_mine = R[m]
    r_theirs = np.asarray(r_ref)[: m.sum()]
    n_cmp = min(len(r_mine), len(r_theirs))
    d = np.abs(r_mine[:n_cmp] - r_theirs[:n_cmp])
    print("  比较 %d 个波长" % n_cmp)
    print("  max |diff| = %.3e    mean |diff| = %.3e" % (d.max(), d.mean()))
    print("  -> %s" % ("实现一致 ✓" if d.max() < 1e-4
                       else "*** 存在偏差，需检查实现 ***"))


# ======================================================== Q2 尺度简并验证
def test_scale_degeneracy() -> None:
    hr("[Q2] 半无限模型的绝对尺度简并：(mu_a, mu_s') -> (k*mu_a, k*mu_s')")
    b = build_basis("VIS")
    wl = torch.tensor(b.wl, dtype=DT)
    eps = torch.tensor(b.eps, dtype=DT)

    a0 = torch.tensor([[30.0]], dtype=DT)
    b0 = torch.tensor([[1.2]], dtype=DT)
    c0 = torch.tensor([[20.0, 8.0, 1.0, 0.1, 0.02, 0.008]], dtype=DT)

    mus = scattering_powerlaw(a0, b0, wl)
    mua = absorption_from_basis(c0, eps)

    print("  %-6s %18s %18s" % ("k", "max|dR| (K-M)", "max|dR| (扩散)"))
    R_km0 = kubelka_munk_R(mua, mus)
    R_df0 = diffusion_R(mua, mus)
    for k in [0.1, 0.5, 2.0, 10.0, 100.0]:
        R_km = kubelka_munk_R(k * mua, k * mus)
        R_df = diffusion_R(k * mua, k * mus)
        print("  %-6.1f %18.3e %18.3e"
              % (k, (R_km - R_km0).abs().max().item(),
                 (R_df - R_df0).abs().max().item()))
    print("  -> 差值为机器精度量级即证实简并：单次反射率谱无法定出绝对尺度")


# ============================================ Q3 直接优化反演（半无限模型）
def _fit_semiinf(R_obs: torch.Tensor, wl: torch.Tensor, eps: torch.Tensor,
                 model, steps: int = 4000, lr: float = 0.05) -> dict:
    """对 (a, b, c) 做梯度下降拟合。用 log 参数化保证正性。"""
    K = eps.shape[0]
    log_a = torch.zeros(1, 1, dtype=DT, requires_grad=True)
    raw_b = torch.full((1, 1), 0.2, dtype=DT, requires_grad=True)
    log_c = torch.full((1, K), -1.0, dtype=DT, requires_grad=True)

    opt = torch.optim.Adam([log_a, raw_b, log_c], lr=lr)
    for i in range(steps):
        opt.zero_grad()
        a = log_a.exp()
        bb = torch.nn.functional.softplus(raw_b)
        c = log_c.exp()
        mus = scattering_powerlaw(a, bb, wl)
        mua = absorption_from_basis(c, eps)
        loss = ((model(mua, mus) - R_obs) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        return dict(a=log_a.exp().item(),
                    b=torch.nn.functional.softplus(raw_b).item(),
                    c=log_c.exp().squeeze(0).numpy(),
                    loss=loss.item())


def test_identifiability_semiinf(noise: float = 0.0) -> None:
    hr("[Q3] 半无限模型：从合成光谱直接反演  (噪声 sigma=%.4f)" % noise)
    bs = build_basis("VIS")
    wl = torch.tensor(bs.wl, dtype=DT)
    eps = torch.tensor(bs.eps, dtype=DT)
    names = bs.names

    a_t, b_t = 30.0, 1.20
    c_t = np.array([20.0, 8.0, 1.0, 0.10, 0.020, 0.0080])

    mus = scattering_powerlaw(torch.tensor([[a_t]], dtype=DT),
                              torch.tensor([[b_t]], dtype=DT), wl)
    mua = absorption_from_basis(torch.tensor(c_t[None], dtype=DT), eps)
    R = kubelka_munk_R(mua, mus)
    if noise > 0:
        R = R + noise * torch.randn_like(R)

    fit = _fit_semiinf(R.detach(), wl, eps, kubelka_munk_R)

    print("  重建 MSE = %.3e" % fit["loss"])
    print()
    print("  %-10s %12s %12s %10s" % ("param", "true", "fitted", "rel.err"))
    print("  %-10s %12.4f %12.4f %9.1f%%"
          % ("a", a_t, fit["a"], 100 * abs(fit["a"] - a_t) / a_t))
    print("  %-10s %12.4f %12.4f %9.1f%%   <-- 形状参数"
          % ("b", b_t, fit["b"], 100 * abs(fit["b"] - b_t) / b_t))
    for i, nm in enumerate(names):
        print("  %-10s %12.5g %12.5g %9.1f%%"
              % ("c_" + nm, c_t[i], fit["c"][i],
                 100 * abs(fit["c"][i] - c_t[i]) / max(1e-12, c_t[i])))

    print()
    print("  --- 理论预言：绝对量不可辨识，但比值可辨识 ---")
    print("  %-16s %12s %12s %10s" % ("derived", "true", "fitted", "rel.err"))
    sc = fit["a"] / a_t
    print("  %-16s %12.4f %12.4f %9.1f%%"
          % ("scale k=a_fit/a_true", 1.0, sc, 100 * abs(sc - 1.0)))
    for i, nm in enumerate(names):
        t = c_t[i] / a_t
        f = fit["c"][i] / fit["a"]
        print("  %-16s %12.5g %12.5g %9.1f%%"
              % ("c_%s / a" % nm, t, f, 100 * abs(f - t) / max(1e-12, t)))
    for i, j in [(0, 1), (0, 4)]:
        t = c_t[i] / c_t[j]
        f = fit["c"][i] / max(1e-12, fit["c"][j])
        print("  %-16s %12.5g %12.5g %9.1f%%"
              % ("c_%s/c_%s" % (names[i], names[j]), t, f,
                 100 * abs(f - t) / max(1e-12, t)))


# =========================================== Q4 直接优化反演（PROSPECT）
def test_identifiability_prospect(noise: float = 0.0) -> None:
    hr("[Q4] PROSPECT 平板模型：绝对浓度可辨识吗？ (噪声 sigma=%.4f)" % noise)
    bs = build_basis("VIS")
    wl_np = bs.wl
    eps = torch.tensor(bs.eps, dtype=DT)
    n_refr = torch.tensor(bs.n_refr, dtype=DT)
    names = bs.names

    N_t = 1.6
    c_t = np.array([35.0, 9.0, 2.0, 0.0, 0.012, 0.0070])

    R_t, _ = prospect_RT(torch.tensor([N_t], dtype=DT),
                         torch.tensor(c_t[None], dtype=DT), eps, n_refr)
    R_obs = R_t.detach()
    if noise > 0:
        R_obs = R_obs + noise * torch.randn_like(R_obs)

    raw_N = torch.full((1, 1), 0.5, dtype=DT, requires_grad=True)
    log_c = torch.full((1, eps.shape[0]), 0.0, dtype=DT, requires_grad=True)
    opt = torch.optim.Adam([raw_N, log_c], lr=0.05)
    for _ in range(5000):
        opt.zero_grad()
        N = 1.0 + torch.nn.functional.softplus(raw_N)
        c = log_c.exp()
        R, _ = prospect_RT(N, c, eps, n_refr)
        loss = ((R - R_obs) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        N_f = (1.0 + torch.nn.functional.softplus(raw_N)).item()
        c_f = log_c.exp().squeeze(0).numpy()

    print("  重建 MSE = %.3e" % loss.item())
    print("  波长范围 %.1f-%.1f nm (%d 波段)" % (wl_np[0], wl_np[-1], len(wl_np)))
    print()
    print("  %-10s %12s %12s %10s" % ("param", "true", "fitted", "rel.err"))
    print("  %-10s %12.4f %12.4f %9.1f%%   <-- 结构参数(层数)"
          % ("N", N_t, N_f, 100 * abs(N_f - N_t) / N_t))
    for i, nm in enumerate(names):
        t = c_t[i]
        e = 100 * abs(c_f[i] - t) / max(1e-12, t) if t > 0 else float("nan")
        print("  %-10s %12.5g %12.5g %9.1f%%" % ("c_" + nm, t, c_f[i], e))
    print()
    print("  若绝对浓度误差小 -> 平板模型的有限厚度确实破除了尺度简并")


def test_multi_init_spread() -> None:
    """决定性测试：多个随机初值反演同一条光谱。

    若绝对尺度真的不可辨识，则不同初值会停在简并谷的不同位置 ——
    绝对量 a, c_i 大幅漂移，而 b 与各比值保持稳定。
    这比单次拟合的误差数字更能说明问题。
    """
    hr("[Q5] 多初值反演同一条光谱：绝对量漂移 vs 比值稳定")
    bs = build_basis("VIS")
    wl = torch.tensor(bs.wl, dtype=DT)
    eps = torch.tensor(bs.eps, dtype=DT)
    names = bs.names

    a_t, b_t = 30.0, 1.20
    c_t = np.array([20.0, 8.0, 1.0, 0.10, 0.020, 0.0080])
    mus = scattering_powerlaw(torch.tensor([[a_t]], dtype=DT),
                              torch.tensor([[b_t]], dtype=DT), wl)
    mua = absorption_from_basis(torch.tensor(c_t[None], dtype=DT), eps)
    R = kubelka_munk_R(mua, mus).detach()

    K = eps.shape[0]
    rows = []
    for trial in range(6):
        g = torch.Generator().manual_seed(100 + trial)
        log_a = (torch.randn(1, 1, generator=g, dtype=DT) * 1.5).requires_grad_(True)
        raw_b = (torch.randn(1, 1, generator=g, dtype=DT) * 0.3 + 0.2).requires_grad_(True)
        log_c = (torch.randn(1, K, generator=g, dtype=DT) * 1.0 - 1.0).requires_grad_(True)
        opt = torch.optim.Adam([log_a, raw_b, log_c], lr=0.05)
        for _ in range(4000):
            opt.zero_grad()
            a = log_a.exp()
            bb = torch.nn.functional.softplus(raw_b)
            c = log_c.exp()
            loss = ((kubelka_munk_R(absorption_from_basis(c, eps),
                                    scattering_powerlaw(a, bb, wl)) - R) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            rows.append(dict(
                mse=loss.item(),
                a=log_a.exp().item(),
                b=torch.nn.functional.softplus(raw_b).item(),
                c=log_c.exp().squeeze(0).numpy()))

    print("  %-6s %10s %10s %9s %12s %12s" %
          ("trial", "MSE", "a", "b", "c_cab", "c_cab/a"))
    for i, r in enumerate(rows):
        print("  %-6d %10.2e %10.3f %9.4f %12.4f %12.5f"
              % (i, r["mse"], r["a"], r["b"], r["c"][0], r["c"][0] / r["a"]))

    def spread(vals):
        v = np.array(vals)
        return 100.0 * (v.max() - v.min()) / max(1e-12, abs(v.mean()))

    print()
    print("  真值:  a=%.3f  b=%.4f  c_cab=%.4f  c_cab/a=%.5f  c_cab/c_car=%.4f"
          % (a_t, b_t, c_t[0], c_t[0] / a_t, c_t[0] / c_t[1]))
    print()
    print("  %-18s %12s   %s" % ("quantity", "spread%", "judgement"))
    checks = [
        ("a  (绝对散射幅值)", [r["a"] for r in rows]),
        ("c_cab (绝对浓度)", [r["c"][0] for r in rows]),
        ("b  (幂律指数)", [r["b"] for r in rows]),
        ("c_cab / a", [r["c"][0] / r["a"] for r in rows]),
        ("c_cab / c_car", [r["c"][0] / r["c"][1] for r in rows]),
    ]
    for nm, vals in checks:
        s = spread(vals)
        verdict = "可辨识" if s < 5 else ("边缘" if s < 20 else "*** 不可辨识 ***")
        print("  %-18s %11.1f%%   %s" % (nm, s, verdict))
    del names


def main() -> None:
    print("device=%s dtype=%s" % (DEV, DT))
    test_prospect_matches_reference()
    test_scale_degeneracy()
    test_identifiability_semiinf(noise=0.0)
    test_multi_init_spread()
    test_identifiability_prospect(noise=0.0)
    test_identifiability_prospect(noise=0.002)


if __name__ == "__main__":
    main()
