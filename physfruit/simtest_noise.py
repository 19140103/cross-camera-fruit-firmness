"""
VIS+NIR 融合反演的噪声鲁棒性。

背景
----
simtest_range.py 得到一个很强的无噪声结果：
    VIS only  -> 只有色素**比值**可辨识，绝对量全部不可辨识
    NIR only  -> N / water / dry matter 完美可辨识，色素完全不可辨识
    VIS+NIR   -> **全部参数完美可辨识**（MSE 到 1e-31）

机制清楚：NIR 的水 1450 nm 与 C-H 1200/1700 nm 强吸收带把绝对光学厚度
（即 N）钉死；N 一旦确定，VIS 段就能解出绝对色素浓度。单相机做不到，
双相机可以 —— 这是多相机融合的**物理必要性**论证，不是凑数。

但无噪声结果可能很脆。本测试给出在真实噪声水平下还剩下什么。
真实高光谱反射率的噪声量级约 0.002-0.01（反射率单位）。

用法: python physfruit/simtest_noise.py
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


def basis_on(wl):
    raw = _load_raw()
    src = raw[:, 0]
    idx = {c: i for i, c in enumerate(COLS)}
    eps = np.stack([np.interp(wl, src, raw[:, idx[o]]) for o in ORDER])
    nr = np.interp(wl, src, raw[:, idx["n_refr"]])
    return torch.tensor(eps, dtype=DT), torch.tensor(nr, dtype=DT)


def fit(wl, noise, seed, steps=3000):
    eps, nr = basis_on(wl)
    R_t, _ = prospect_RT(torch.tensor([N_TRUE], dtype=DT),
                         torch.tensor(C_TRUE[None], dtype=DT), eps, nr)
    g = torch.Generator().manual_seed(seed)
    R_obs = R_t.detach()
    if noise > 0:
        R_obs = R_obs + noise * torch.randn(R_obs.shape, generator=g, dtype=DT)

    g2 = torch.Generator().manual_seed(9000 + seed)
    raw_N = (torch.randn(1, 1, generator=g2, dtype=DT) * 0.5 + 0.5).requires_grad_(True)
    log_c = (torch.randn(1, len(ORDER), generator=g2, dtype=DT) * 0.8).requires_grad_(True)
    opt = torch.optim.Adam([raw_N, log_c], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        N = 1.0 + torch.nn.functional.softplus(raw_N)
        R, _ = prospect_RT(N, log_c.exp(), eps, nr)
        loss = ((R - R_obs) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return ((1.0 + torch.nn.functional.softplus(raw_N)).item(),
                log_c.exp().squeeze(0).numpy())


def main():
    vis = camera_wavelengths("VIS")
    nir = camera_wavelengths("NIR")
    both = np.sort(np.concatenate([vis, nir]))

    print("=" * 78)
    print("VIS+NIR 融合反演的噪声鲁棒性 (476 bands, 397.7-1727.0 nm)")
    print("=" * 78)
    print("真值 N=%.2f  c_cab=%.1f  c_car=%.1f  c_water=%.4f  c_dm=%.4f"
          % (N_TRUE, C_TRUE[0], C_TRUE[1], C_TRUE[4], C_TRUE[5]))
    print("\n%-9s %9s %9s %9s %9s %9s %11s"
          % ("noise", "N", "c_cab", "c_car", "c_water", "c_dm", "cab/car"))
    print("%-9s %9.3f %9.2f %9.2f %9.5f %9.5f %11.3f"
          % ("TRUE", N_TRUE, C_TRUE[0], C_TRUE[1], C_TRUE[4], C_TRUE[5],
             C_TRUE[0] / C_TRUE[1]))
    print("-" * 78)

    for noise in (0.0, 0.001, 0.002, 0.005, 0.010, 0.020):
        res = [fit(both, noise, s) for s in range(4)]
        Ns = np.array([r[0] for r in res])
        cs = np.stack([r[1] for r in res])
        med = np.median(cs, axis=0)
        print("%-9.3f %9.3f %9.2f %9.2f %9.5f %9.5f %11.3f"
              % (noise, np.median(Ns), med[0], med[1], med[4], med[5],
                 med[0] / max(1e-12, med[1])))

        errs = {
            "N": 100 * abs(np.median(Ns) - N_TRUE) / N_TRUE,
            "c_cab": 100 * abs(med[0] - C_TRUE[0]) / C_TRUE[0],
            "c_water": 100 * abs(med[4] - C_TRUE[4]) / C_TRUE[4],
            "c_dm": 100 * abs(med[5] - C_TRUE[5]) / C_TRUE[5],
            "cab/car": 100 * abs(med[0] / max(1e-12, med[1])
                                 - C_TRUE[0] / C_TRUE[1]) / (C_TRUE[0] / C_TRUE[1]),
        }
        print("%-9s %s" % ("  err%",
                           "  ".join("%s=%.1f%%" % (k, v) for k, v in errs.items())))
    print()
    print("真实高光谱反射率噪声量级约 0.002-0.01；看该区间内哪些量仍然站得住。")


if __name__ == "__main__":
    main()
