"""
真实数据上的 PROSPECT 反演（第一步：诊断）。

仿真自检已经证明方法在合成数据上成立。本模块回答仿真答不了的问题：

  D1  真实果实光谱长什么样？叶绿素 677nm 吸收谷在不在？
  D2  同一果实内像元间的光谱变异有多大？（镜面反射、曲率的影响）
  D3  PROSPECT 能拟合真实光谱吗？残差多大？
  D4  真实数据上的可辨识性 —— 仿真预言「VIS 单相机只有比值可辨识」，
      在真实光谱上是否重现？

D4 尤其关键：Kaki/Mango 只有 VIS + VIS_COR（都在色素窗口），
按机制**不应该**能定出绝对量。若真实数据重现这一预言，
就说明整套框架在真实数据上也站得住；等 Avocado/Kiwi 的 NIR 到位后，
融合应当修复它 —— 这构成一条干净的实验叙事。

用法:
    python physfruit/realfit.py diag       # 诊断 D1-D3
    python physfruit/realfit.py ident      # 可辨识性 D4
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from physfruit.basis import build_basis  # noqa: E402
from physfruit.envi import read_cube  # noqa: E402
from physfruit.forward import prospect_RT  # noqa: E402
from physfruit.labels import CUBE_ROOT, load_records  # noqa: E402

DT = torch.float64
ORDER = ["cab", "car", "anth", "brown", "water", "dm"]


def hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ------------------------------------------------------------------ 掩码
def fruit_pixels(cube: np.ndarray,
                 lo_pct: float = 45.0, hi_pct: float = 97.0) -> np.ndarray:
    """果实像元掩码，同时剔除背景与镜面高光。

    背景暗、果实中等、镜面高光极亮。单纯取「高于某分位数」会把高光也算进来，
    而高光是几乎无色的强反射，会严重污染吸收特征。故取一个**中间区间**。
    """
    bright = cube.mean(axis=2)
    lo = np.percentile(bright, lo_pct)
    hi = np.percentile(bright, hi_pct)
    return (bright >= lo) & (bright <= hi)


def mean_spectrum(stem: str, **kw) -> tuple:
    cube, _h = read_cube(stem)
    m = fruit_pixels(cube, **kw)
    px = cube[m]                       # (P, L)
    return px.mean(axis=0), px


# --------------------------------------------------------------- 反演核心
def fit_prospect(R_obs: np.ndarray, basis, seeds=(0,), steps: int = 2500,
                 lr: float = 0.05, fit_gain: bool = False,
                 reparam: bool = True) -> list:
    """对一条实测光谱做 PROSPECT 反演，返回每个种子的结果。

    fit_gain: 是否额外拟合一个乘性增益。真实测量几何与 PROSPECT 假设的
    半球积分不同，增益可吸收这一差异 —— 但它也会吸收掉绝对尺度信息，
    因此默认关闭，只在诊断时用来看残差能降到多少。
    """
    eps = torch.tensor(basis.eps, dtype=DT)
    nr = torch.tensor(basis.n_refr, dtype=DT)
    y = torch.tensor(R_obs, dtype=DT).unsqueeze(0)

    out = []
    for s in seeds:
        g = torch.Generator().manual_seed(1234 + s)
        raw_N = (torch.randn(1, 1, generator=g, dtype=DT) * 0.5 + 0.5).requires_grad_(True)
        # Optimising c directly creates a very flat valley: for a finite layer
        # model, N and c can move together while preserving the per-layer
        # optical thickness c/N.  The log(c/N) parameterisation makes the
        # quantity seen by the forward model explicit and is substantially
        # better conditioned on real spectra.  Keep the old parameterisation
        # available for diagnostics/reproducibility.
        if reparam:
            log_q = (torch.randn(1, len(ORDER), generator=g, dtype=DT) * 0.8).requires_grad_(True)
            params = [raw_N, log_q]
        else:
            log_c = (torch.randn(1, len(ORDER), generator=g, dtype=DT) * 0.8).requires_grad_(True)
            params = [raw_N, log_c]
        log_g = torch.zeros(1, 1, dtype=DT, requires_grad=True)
        if fit_gain:
            params.append(log_g)

        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            N = 1.0 + torch.nn.functional.softplus(raw_N)
            c = N * log_q.exp() if reparam else log_c.exp()
            R, _ = prospect_RT(N, c, eps, nr)
            if fit_gain:
                R = R * log_g.exp()
            loss = ((R - y) ** 2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            N = (1.0 + torch.nn.functional.softplus(raw_N)).item()
            c = (N * log_q.exp() if reparam else log_c.exp()).squeeze(0).numpy()
            R, _ = prospect_RT(torch.tensor([[N]], dtype=DT),
                               torch.tensor(c[None], dtype=DT), eps, nr)
            if fit_gain:
                R = R * log_g.exp()
            resid = (R.squeeze(0).numpy() - R_obs)
            out.append(dict(N=N, c=c, q=c / max(N, 1e-12),
                            gain=log_g.exp().item() if fit_gain else 1.0,
                            rmse=float(np.sqrt((resid ** 2).mean())),
                            fit=R.squeeze(0).numpy()))
    return out


def _pick(df, fruit: str, camera: str, n: int):
    d = df[(df["fruit"] == fruit) & (df["camera"] == camera) & df["cube_exists"]]
    d = d.drop_duplicates(subset=["uid"])
    return d.head(n)


# =============================================================== D1-D3 诊断
def diag() -> None:
    tab = load_records()
    df = tab.df
    basis = build_basis("VIS")
    wl = basis.wl

    hr("[D1] 真实果实平均光谱（VIS 398-1004 nm）")
    for fruit in ("Kaki", "Mango"):
        sub = _pick(df, fruit, "VIS", 3)
        if len(sub) == 0:
            print("  %s: 无可用立方体" % fruit)
            continue
        print("\n  --- %s ---" % fruit)
        for _, row in sub.iterrows():
            stem = os.path.join(CUBE_ROOT, row["data_file"].replace("/", os.sep))
            stem = stem.rsplit(".", 1)[0]
            sp, px = mean_spectrum(stem)
            i450 = int(np.argmin(np.abs(wl - 450)))
            i550 = int(np.argmin(np.abs(wl - 550)))
            i677 = int(np.argmin(np.abs(wl - 677)))
            i800 = int(np.argmin(np.abs(wl - 800)))
            dip = (sp[i550] + sp[i800]) / 2 - sp[i677]
            print("    %-34s ripe=%-9s px=%4d  R450=%.3f R550=%.3f "
                  "R677=%.3f R800=%.3f  chl_dip=%+.4f"
                  % (os.path.basename(stem)[:34], row["ripeness3"], len(px),
                     sp[i450], sp[i550], sp[i677], sp[i800], dip))
    print("\n  chl_dip = (R550+R800)/2 - R677，正值说明存在叶绿素吸收谷")

    hr("[D2] 像元间光谱变异（镜面反射 / 曲率的影响）")
    for fruit in ("Kaki", "Mango"):
        sub = _pick(df, fruit, "VIS", 2)
        for _, row in sub.iterrows():
            stem = os.path.join(CUBE_ROOT, row["data_file"].replace("/", os.sep))
            stem = stem.rsplit(".", 1)[0]
            cube, _ = read_cube(stem)
            m = fruit_pixels(cube)
            px = cube[m]
            rel = px.std(axis=0) / np.maximum(px.mean(axis=0), 1e-6)
            allpx = cube.reshape(-1, cube.shape[2])
            print("    %-34s 掩码内像元 %4d/%4d  相对标准差: 中位 %.1f%% 最大 %.1f%%"
                  % (os.path.basename(stem)[:34], m.sum(), m.size,
                     100 * np.median(rel), 100 * rel.max()))
            print("      %-32s 全图亮度 p5=%.3f p50=%.3f p95=%.3f p99.5=%.3f"
                  % ("", *np.percentile(allpx.mean(axis=1), [5, 50, 95, 99.5])))
    print("\n  p99.5 远高于 p95 即提示存在镜面高光")

    hr("[D3] PROSPECT 能拟合真实光谱吗？")
    print("  %-30s %-9s %10s %10s %8s %9s"
          % ("sample", "ripeness", "RMSE", "RMSE+gain", "N", "c_cab"))
    for fruit in ("Kaki", "Mango"):
        sub = _pick(df, fruit, "VIS", 3)
        for _, row in sub.iterrows():
            stem = os.path.join(CUBE_ROOT, row["data_file"].replace("/", os.sep))
            stem = stem.rsplit(".", 1)[0]
            sp, _ = mean_spectrum(stem)
            r0 = fit_prospect(sp, basis, seeds=(0,))[0]
            r1 = fit_prospect(sp, basis, seeds=(0,), fit_gain=True)[0]
            print("  %-30s %-9s %10.5f %10.5f %8.3f %9.2f"
                  % (os.path.basename(stem)[:30], str(row["ripeness3"])[:9],
                     r0["rmse"], r1["rmse"], r0["N"], r0["c"][0]))
    print("\n  实测光谱自身的像元间标准差量级约 2-10%%，")
    print("  若 RMSE 与之相当，说明 PROSPECT 已拟合到数据噪声水平")


# ================================================================ D4 可辨识性
def ident() -> None:
    tab = load_records()
    df = tab.df
    basis = build_basis("VIS")

    hr("[D4] 真实光谱上的可辨识性 —— 检验仿真预言")
    print("仿真预言: VIS 单相机 -> 绝对量不可辨识，色素比值可辨识")
    print("下面对每个样本用 5 个随机初值反演，看离散度\n")

    print("  %-26s %-9s %9s %9s %11s %11s"
          % ("sample", "ripeness", "N_spr%", "cab_spr%", "cab/car_spr%", "RMSE"))
    agg = {"N": [], "cab": [], "ratio": []}
    for fruit in ("Kaki", "Mango"):
        sub = _pick(df, fruit, "VIS", 4)
        for _, row in sub.iterrows():
            stem = os.path.join(CUBE_ROOT, row["data_file"].replace("/", os.sep))
            stem = stem.rsplit(".", 1)[0]
            sp, _ = mean_spectrum(stem)
            res = fit_prospect(sp, basis, seeds=range(5))

            def spr(v):
                v = np.asarray(v, dtype=float)
                return 100.0 * (v.max() - v.min()) / max(1e-12, abs(v.mean()))

            sN = spr([r["N"] for r in res])
            sC = spr([r["c"][0] for r in res])
            sR = spr([r["c"][0] / max(1e-12, r["c"][1]) for r in res])
            agg["N"].append(sN)
            agg["cab"].append(sC)
            agg["ratio"].append(sR)
            print("  %-26s %-9s %8.1f%% %8.1f%% %10.1f%% %11.5f"
                  % (os.path.basename(stem)[:26], str(row["ripeness3"])[:9],
                     sN, sC, sR, np.median([r["rmse"] for r in res])))

    print()
    print("  %-16s %12s   %s" % ("quantity", "median spr%", "judgement"))
    for k, nm in [("N", "N (结构)"), ("cab", "c_cab (绝对)"),
                  ("ratio", "c_cab/c_car (比值)")]:
        m = float(np.median(agg[k])) if agg[k] else float("nan")
        print("  %-16s %11.1f%%   %s"
              % (nm, m, "可辨识" if m < 15 else "*** 不可辨识 ***"))
    print()
    print("  若 N/c_cab 离散大而比值离散小 -> 真实数据重现了仿真预言，框架成立")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if cmd == "diag":
        diag()
    elif cmd == "ident":
        ident()
    else:
        print(__doc__)
