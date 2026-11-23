"""
光谱范围对 PROSPECT 绝对可辨识性的影响。

动机
----
simtest.py 的 Q4 发现：只用 VIS (398-1004 nm) 反演 PROSPECT 时，
(N, c) 会一起放大约 1.8-2 倍而重建误差仍很小 —— 因为 k = sum(c*eps)/N，
(N, c) -> (kN, kc) 近似不改变 k。

但 VIS 波段里水和干物质几乎不吸收（c_dm 误差高达 670%）。
若把 NIR (920-1727 nm) 的水 1450 nm、C-H 1200/1700 nm 强吸收带纳入，
这些组分就有了独立的光谱特征，可能把绝对尺度钉住。

这个问题直接决定两件事：
  1. 该用半无限模型还是 PROSPECT
  2. 要不要做 VIS+NIR 双相机融合（若融合能破除简并，那是物理论证而非凑数）

用法: python physfruit/simtest_range.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from physfruit.basis import COLS, _load_raw, camera_wavelengths  # noqa: E402
from physfruit.forward import prospect_RT  # noqa: E402

DT = torch.float64
ORDER = ["cab", "car", "anth", "brown", "water", "dm"]
N_TRUE = 1.6
C_TRUE = np.array([35.0, 9.0, 2.0, 0.05, 0.012, 0.0070])


def basis_on(wl_target: np.ndarray):
    raw = _load_raw()
    src = raw[:, 0]
    idx = {c: i for i, c in enumerate(COLS)}
    eps = np.stack([np.interp(wl_target, src, raw[:, idx[o]]) for o in ORDER])
    n_refr = np.interp(wl_target, src, raw[:, idx["n_refr"]])
    return (torch.tensor(eps, dtype=DT), torch.tensor(n_refr, dtype=DT))


def fit(wl: np.ndarray, noise: float, seed: int, steps: int = 6000) -> dict:
    eps, n_refr = basis_on(wl)
    R_t, _ = prospect_RT(torch.tensor([N_TRUE], dtype=DT),
                         torch.tensor(C_TRUE[None], dtype=DT), eps, n_refr)
    R_obs = R_t.detach()
    if noise > 0:
        g = torch.Generator().manual_seed(seed)
        R_obs = R_obs + noise * torch.randn(R_obs.shape, generator=g, dtype=DT)

    g = torch.Generator().manual_seed(1000 + seed)
    raw_N = (torch.randn(1, 1, generator=g, dtype=DT) * 0.5 + 0.5).requires_grad_(True)
    log_c = (torch.randn(1, len(ORDER), generator=g, dtype=DT) * 0.8).requires_grad_(True)
    opt = torch.optim.Adam([raw_N, log_c], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        N = 1.0 + torch.nn.functional.softplus(raw_N)
        c = log_c.exp()
        R, _ = prospect_RT(N, c, eps, n_refr)
        loss = ((R - R_obs) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return dict(N=(1.0 + torch.nn.functional.softplus(raw_N)).item(),
                    c=log_c.exp().squeeze(0).numpy(),
                    mse=loss.item())


def spread(v) -> float:
    v = np.asarray(v, dtype=float)
    return 100.0 * (v.max() - v.min()) / max(1e-12, abs(v.mean()))


def run(tag: str, wl: np.ndarray, noise: float, trials: int = 5) -> None:
    print("\n" + "-" * 78)
    print("%s   |  %d bands  %.1f-%.1f nm  |  noise sigma=%.4f"
          % (tag, len(wl), wl.min(), wl.max(), noise))
    print("-" * 78)
    res = [fit(wl, noise, s) for s in range(trials)]

    print("  %-6s %10s %9s %10s %10s %11s"
          % ("trial", "MSE", "N", "c_cab", "c_water", "c_dm"))
    for i, r in enumerate(res):
        print("  %-6d %10.2e %9.4f %10.3f %10.5f %11.5f"
              % (i, r["mse"], r["N"], r["c"][0], r["c"][4], r["c"][5]))
    print("  %-6s %10s %9.4f %10.3f %10.5f %11.5f"
          % ("TRUE", "", N_TRUE, C_TRUE[0], C_TRUE[4], C_TRUE[5]))

    print()
    print("  %-14s %10s %12s   %s" % ("quantity", "spread%", "median err%", "verdict"))
    items = [("N", [r["N"] for r in res], N_TRUE)]
    for i, nm in enumerate(ORDER):
        if C_TRUE[i] <= 0:
            continue
        items.append(("c_" + nm, [r["c"][i] for r in res], C_TRUE[i]))
    items.append(("c_cab/c_car",
                  [r["c"][0] / r["c"][1] for r in res], C_TRUE[0] / C_TRUE[1]))

    for nm, vals, truth in items:
        sp = spread(vals)
        err = 100.0 * abs(np.median(vals) - truth) / abs(truth)
        ok = sp < 15 and err < 20
        print("  %-14s %9.1f%% %11.1f%%   %s"
              % (nm, sp, err, "可辨识" if ok else "*** 不可辨识 ***"))


def main() -> None:
    vis = camera_wavelengths("VIS")
    nir = camera_wavelengths("NIR")
    both = np.sort(np.concatenate([vis, nir]))
    full = np.arange(400.0, 2501.0, 1.0)

    print("=" * 78)
    print("PROSPECT 绝对可辨识性 vs 光谱范围")
    print("=" * 78)
    print("真值: N=%.2f  c=%s" % (N_TRUE, np.array2string(C_TRUE, precision=4)))
    print("每组 5 个随机初值；spread 小且中位误差小才算可辨识")

    for noise in (0.0, 0.002):
        run("VIS only (Specim FX10)", vis, noise)
        run("NIR only (RedEye 1.7)", nir, noise)
        run("VIS + NIR 融合", both, noise)
        if noise == 0.0:
            run("Full 400-2500nm (遥感常用)", full, noise)


if __name__ == "__main__":
    main()
