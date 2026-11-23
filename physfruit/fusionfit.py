"""
VIS+NIR 双相机融合反演 —— 真实数据上的第三步验证。

论证链
------
    仿真   VIS-only -> 只有比值可辨识      VIS+NIR -> 全部可辨识   (simtest_range)
    真实   VIS-only (Kaki/Mango) -> 只有比值可辨识 ✓ 预言命中     (realfit ident)
    真实   VIS+NIR (Avocado/Kiwi) -> 应恢复绝对量                ← 本模块

外加一个可证伪对照：VIS+VIS_COR 两台都在色素窗口，**不应**破除简并。

本模块比仿真多回答两个真实数据才有的问题
--------------------------------------
  F0  跨相机一致性：VIS 与 NIR 在 920-1004 nm 重叠，同一果实两台相机测到的
      反射率一致吗？若差一个常数因子，融合就必须带每相机增益。
  F1  融合是否把 N 拉回 PROSPECT 合理区间 (1-2.5)？
      VIS 单独反演时 N=3.5-4.9，若融合修复它，双层模型就不必要。

配置
----
  vis        只用 VIS 相机                （预期：绝对量不可辨识）
  nir        只用 NIR 相机                （预期：N/水/干物质可辨识，色素不可）
  fused      VIS+NIR 拼接，共享 (N, c)    （预期：全部可辨识）
  fused_gain 同上，但每相机各一个自由增益  （检验增益是否重新引入简并）
  viscor     VIS+VIS_COR 拼接（对照）      （预期：仍不可辨识）

用法:
    python physfruit/fusionfit.py overlap            # F0 跨相机一致性
    python physfruit/fusionfit.py run [--n 10]       # 主实验
    python physfruit/fusionfit.py control [--n 10]   # VIS+VIS_COR 对照
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from physfruit.basis import COLS, _load_raw, camera_wavelengths  # noqa: E402
from physfruit.envi import read_cube  # noqa: E402
from physfruit.forward import prospect_RT  # noqa: E402
from physfruit.labels import CUBE_ROOT, camera_pairs, load_records  # noqa: E402
from physfruit.realfit import fruit_pixels  # noqa: E402

DT = torch.float64
ORDER = ["cab", "car", "anth", "brown", "water", "dm"]


def hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ------------------------------------------------------------------ 基谱
def basis_on(wl: np.ndarray):
    raw = _load_raw()
    src = raw[:, 0]
    idx = {c: i for i, c in enumerate(COLS)}
    eps = np.stack([np.interp(wl, src, raw[:, idx[o]]) for o in ORDER])
    nr = np.interp(wl, src, raw[:, idx["n_refr"]])
    return torch.tensor(eps, dtype=DT), torch.tensor(nr, dtype=DT)


# ------------------------------------------------------------------ 数据
def stem_of(data_file: str) -> str:
    p = os.path.join(CUBE_ROOT, data_file.replace("/", os.sep))
    return p.rsplit(".", 1)[0]


def mean_spec(data_file: str) -> np.ndarray:
    cube, _ = read_cube(stem_of(data_file))
    return cube[fruit_pixels(cube)].mean(axis=0)


def get_pairs(df, cam_a: str, cam_b: str, fruits=("Avocado", "Kiwi"), n=None):
    d = df[df["fruit"].isin(fruits) & df["cube_exists"]]
    m = camera_pairs(d, cam_a, cam_b)
    # 每个果实只取一条（front 优先），避免同一果实重复计数
    m = m.sort_values("side").drop_duplicates(subset=["uid"])
    if n is not None:
        m = m.head(n)
    return m


# ------------------------------------------------------------------ 反演
def fit(segments: list, seeds=range(5), steps: int = 2000, lr: float = 0.05,
        per_cam_gain: bool = False) -> list:
    """对若干光谱段做联合 PROSPECT 反演。

    segments: [(wl, R_obs), ...]  各段可来自不同相机；所有段共享同一组 (N, c)。
    per_cam_gain: 每段各配一个自由乘性增益。
    """
    epss, nrs, ys = [], [], []
    for wl, R in segments:
        e, nr = basis_on(wl)
        epss.append(e)
        nrs.append(nr)
        ys.append(torch.tensor(R, dtype=DT).unsqueeze(0))

    out = []
    for s in seeds:
        g = torch.Generator().manual_seed(777 + s)
        raw_N = (torch.randn(1, 1, generator=g, dtype=DT) * 0.5 + 0.5).requires_grad_(True)
        log_c = (torch.randn(1, len(ORDER), generator=g, dtype=DT) * 0.8).requires_grad_(True)
        log_g = torch.zeros(len(segments), 1, dtype=DT, requires_grad=per_cam_gain)
        params = [raw_N, log_c] + ([log_g] if per_cam_gain else [])
        opt = torch.optim.Adam(params, lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            N = 1.0 + torch.nn.functional.softplus(raw_N)
            c = log_c.exp()
            loss = 0.0
            for k, (e, nr, y) in enumerate(zip(epss, nrs, ys)):
                R, _ = prospect_RT(N, c, e, nr)
                if per_cam_gain:
                    R = R * log_g[k].exp()
                loss = loss + ((R - y) ** 2).mean()
            loss = loss / len(segments)
            loss.backward()
            opt.step()

        with torch.no_grad():
            out.append(dict(
                N=(1.0 + torch.nn.functional.softplus(raw_N)).item(),
                c=log_c.exp().squeeze(0).numpy(),
                gains=log_g.exp().squeeze(1).numpy() if per_cam_gain else None,
                rmse=float(np.sqrt(loss.item()))))
    return out


def spread(v) -> float:
    v = np.asarray(v, dtype=float)
    return 100.0 * (v.max() - v.min()) / max(1e-12, abs(v.mean()))


def summarize(res: list) -> dict:
    return dict(
        N_spr=spread([r["N"] for r in res]),
        cab_spr=spread([r["c"][0] for r in res]),
        water_spr=spread([r["c"][4] for r in res]),
        dm_spr=spread([r["c"][5] for r in res]),
        # 用 cab/(cab+car) 而不是 cab/car：过熟果实两者同趋近 0 时 cab/car 是 0/0，
        # 会出现 300%+ 的伪离散（Mango day_8|09、day_10|02 即此情况）；
        # 有界比例在 [0,1] 内，无此病态。
        ratio_spr=spread([r["c"][0] / max(1e-12, r["c"][0] + r["c"][1]) for r in res]),
        N_med=float(np.median([r["N"] for r in res])),
        cab_med=float(np.median([r["c"][0] for r in res])),
        rmse=float(np.median([r["rmse"] for r in res])),
    )


# ================================================================ F0 重叠区
def overlap(n: int) -> None:
    tab = load_records()
    pairs = get_pairs(tab.df, "VIS", "NIR", n=n)
    wl_v, wl_n = camera_wavelengths("VIS"), camera_wavelengths("NIR")
    lo, hi = max(wl_v.min(), wl_n.min()), min(wl_v.max(), wl_n.max())
    grid = np.linspace(lo + 2, hi - 2, 20)

    hr("[F0] 跨相机一致性：VIS vs NIR 在重叠区 %.0f-%.0f nm" % (lo, hi))
    print("  同一果实、同一侧、两台相机。若定标一致，比值应≈1 且平坦。\n")
    print("  %-28s %-7s %9s %9s %9s %9s"
          % ("fruit", "side", "R_vis", "R_nir", "nir/vis", "slope"))
    ratios = []
    for _, row in pairs.iterrows():
        try:
            sv = mean_spec(row["data_file_VIS"])
            sn = mean_spec(row["data_file_NIR"])
        except Exception as e:  # noqa: BLE001
            print("  %-28s 读取失败: %s" % (row["uid"][:28], e))
            continue
        iv = np.interp(grid, wl_v, sv)
        inn = np.interp(grid, wl_n, sn)
        r = inn / np.maximum(iv, 1e-6)
        slope = np.polyfit(grid, r, 1)[0] * 100  # 每 100 nm 的变化
        ratios.append(r.mean())
        print("  %-28s %-7s %9.4f %9.4f %9.3f %+9.3f"
              % (row["uid"][:28], row["side"], iv.mean(), inn.mean(),
                 r.mean(), slope))
    if ratios:
        ratios = np.array(ratios)
        print("\n  nir/vis 比值：均值 %.3f  标准差 %.3f  范围 [%.3f, %.3f]"
              % (ratios.mean(), ratios.std(), ratios.min(), ratios.max()))
        if abs(ratios.mean() - 1) < 0.1 and ratios.std() < 0.1:
            print("  -> 两相机绝对定标基本一致，可直接拼接")
        else:
            print("  -> 存在系统性相机间偏差，融合需带每相机增益（见 fused_gain）")


# ================================================================ 主实验
def run(n: int, control: bool = False) -> None:
    tab = load_records()
    df = tab.df
    if control:
        cam_b = "VIS_COR"
        fruits = ("Avocado", "Kaki", "Mango", "Papaya")
        title = "[对照] VIS+VIS_COR 融合 —— 预言：两台都在色素窗口，不应破除简并"
    else:
        cam_b = "NIR"
        fruits = ("Avocado", "Kiwi")
        title = "[主实验] VIS+NIR 融合 —— 预言：恢复绝对量，N 回到 1-2.5"

    pairs = get_pairs(df, "VIS", cam_b, fruits=fruits, n=n)
    wl_v, wl_b = camera_wavelengths("VIS"), camera_wavelengths(cam_b)

    hr(title)
    print("  %d 个果实，每个 5 个随机初值\n" % len(pairs))

    configs = ["vis", cam_b.lower(), "fused", "fused_gain"]
    agg = {c: [] for c in configs}
    Ns = {c: [] for c in configs}

    print("  %-22s %-11s %9s %9s %9s %9s %9s %9s"
          % ("fruit", "config", "N_med", "N_spr%", "cab_spr%", "wat_spr%",
             "dm_spr%", "rat_spr%"))
    for _, row in pairs.iterrows():
        try:
            sv = mean_spec(row["data_file_VIS"])
            sb = mean_spec(row["data_file_" + cam_b])
        except Exception as e:  # noqa: BLE001
            print("  %-22s 读取失败: %s" % (row["uid"][:22], e))
            continue

        runs = {
            "vis": fit([(wl_v, sv)]),
            cam_b.lower(): fit([(wl_b, sb)]),
            "fused": fit([(wl_v, sv), (wl_b, sb)]),
            "fused_gain": fit([(wl_v, sv), (wl_b, sb)], per_cam_gain=True),
        }
        for cfg in configs:
            s = summarize(runs[cfg])
            agg[cfg].append(s)
            Ns[cfg].append(s["N_med"])
            print("  %-22s %-11s %9.3f %8.1f%% %8.1f%% %8.1f%% %8.1f%% %8.1f%%"
                  % (row["uid"][:22] if cfg == "vis" else "", cfg, s["N_med"],
                     s["N_spr"], s["cab_spr"], s["water_spr"], s["dm_spr"],
                     s["ratio_spr"]))
        if runs["fused_gain"][0]["gains"] is not None:
            gs = np.median([r["gains"] for r in runs["fused_gain"]], axis=0)
            print("  %-22s %-11s gains VIS=%.3f %s=%.3f"
                  % ("", "", gs[0], cam_b, gs[1]))

    hr("汇总（各果实中位数）")
    print("  %-11s %8s %9s %9s %9s %9s %9s   %s"
          % ("config", "N_med", "N_spr%", "cab_spr%", "wat_spr%", "dm_spr%",
             "rat_spr%", "判定"))
    for cfg in configs:
        if not agg[cfg]:
            continue
        m = {k: float(np.median([s[k] for s in agg[cfg]]))
             for k in agg[cfg][0]}
        ident_abs = m["N_spr"] < 15 and m["cab_spr"] < 15
        n_ok = 1.0 <= m["N_med"] <= 2.5
        verdict = ("绝对量可辨识" if ident_abs else "绝对量不可辨识")
        verdict += "，N %s" % ("在合理区间" if n_ok else "超出 1-2.5")
        print("  %-11s %8.3f %8.1f%% %8.1f%% %8.1f%% %8.1f%% %8.1f%%   %s"
              % (cfg, m["N_med"], m["N_spr"], m["cab_spr"], m["water_spr"],
                 m["dm_spr"], m["ratio_spr"], verdict))

    print()
    if control:
        print("  若 fused 仍不可辨识 -> 对照成立：融合有效靠的是补上水吸收带，不是波段变多")
    else:
        print("  若 fused 可辨识而 vis/nir 单独不可 -> 三步论证链闭合")
        print("  若 fused 的 N 回到 1-2.5 -> 单层 PROSPECT 够用，双层模型不必要")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["overlap", "run", "control"])
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "overlap":
        overlap(a.n)
    elif a.cmd == "run":
        run(a.n)
    else:
        run(a.n, control=True)
