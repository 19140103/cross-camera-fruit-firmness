"""
吸收基谱 epsilon_i(lambda) —— 物理解码器的固定输入。

来源: PROSPECT-D 比吸收系数表 (Feret et al. 2017), 400-2500 nm @ 1 nm。
详见 physfruit/data/SOURCE.md。

这些系数是**固定不训练**的。网络只预测各组分的浓度 c_i，
物理解码器用 mu_a(lambda) = sum_i c_i * eps_i(lambda) 重建吸收谱。
这正是让反演可辨识的关键约束：224 个波段的观测 vs ~8 个未知数。

关键设计决定
-----------
1. **按相机重采样**：三台相机波长网格不同，各自插值一份基谱。
2. **VIS 相机前 2.34 nm 外插**：PROSPECT-D 从 400 nm 起，VIS 从 397.66 起。
   用最近邻外插（400nm 的值），只影响 3 个波段，误差可忽略。
3. **提供“吸收近零窗口”**：辨识性的锚点。果实在 750-900 nm 吸收极弱，
   该区间 R 几乎完全由散射决定，用来钉住幂律散射参数 (a, b)。
   这个窗口是整个反演能工作的数学基础，必须显式暴露出来。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PD = os.path.join(_HERE, "data", "prospect_d_spectra.txt")

# PROSPECT-D 列序（见 SOURCE.md）
COLS = ["wl", "n_refr", "cab", "car", "anth", "brown", "water", "dm"]

# 我们实际使用的吸收组分及其单位
ABSORBERS = {
    "cab":   dict(label="Chlorophyll a+b", unit="ug/cm2",
                  note="430/662 nm 双峰；mu_a670 是文献公认的成熟度指标"),
    "car":   dict(label="Carotenoids", unit="ug/cm2",
                  note="450-500 nm；成熟过程中累积"),
    "anth":  dict(label="Anthocyanins", unit="ug/cm2",
                  note="~530-550 nm；部分果皮显色"),
    "brown": dict(label="Brown pigments", unit="a.u.",
                  note="褐变/衰老；对 overripe 可能重要"),
    "water": dict(label="Water", unit="cm",
                  note="970 / 1450 nm；1450 只在 NIR 相机量程内"),
    "dm":    dict(label="Dry matter", unit="g/cm2",
                  note="细胞壁/糖类 C-H；1200/1700 nm 只在 NIR 量程内"),
}

# 相机波长网格（与 deephs_fruit/core/spectral_io.py 一致）
CAMERA_RANGE = {
    "VIS":     (397.66, 1003.81, 224),
    "VIS_COR": (408.03,  901.26, 249),
    "NIR":     (919.68, 1727.04, 252),
}

# 吸收近零窗口（辨识性锚点）。果实组织在这些区间吸收极弱，
# 反射率近似只由散射决定。
LOW_ABSORPTION_WINDOWS = {
    "VIS":     [(750.0, 900.0)],           # 叶绿素红边之后、水 970 之前
    "VIS_COR": [(750.0, 880.0)],           # 同上，受 901 上限所限
    "NIR":     [(1060.0, 1300.0)],         # 水 970 与 1450 之间的谷
}


def _load_raw() -> np.ndarray:
    """读 PROSPECT-D 表 -> (2101, 8)。"""
    rows = []
    with open(_PD, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    a = np.asarray(rows, dtype=np.float64)
    assert a.shape[1] == 8, "unexpected column count: %s" % (a.shape,)
    return a


def camera_wavelengths(camera: str) -> np.ndarray:
    """返回相机的波长网格。

    优先从 deephs_fruit 仓库读取真实波长表；读不到则按端点线性生成
    （端点与波段数来自官方标注 JSON，已核实）。
    """
    try:
        import sys
        repo = os.path.join(os.path.dirname(_HERE), "deephs_fruit")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from core.spectral_io import VIS_BANDS, VIS_COR_BANDS, NIR_BANDS
        return np.asarray(
            {"VIS": VIS_BANDS, "VIS_COR": VIS_COR_BANDS, "NIR": NIR_BANDS}[camera],
            dtype=np.float64)
    except Exception:  # noqa: BLE001
        lo, hi, n = CAMERA_RANGE[camera]
        return np.linspace(lo, hi, n)


@dataclass
class AbsorberBasis:
    """某一台相机上的吸收基谱。

    属性
    ----
    camera    : 相机名
    wl        : (L,)   波长网格 (nm)
    eps       : (K, L) 比吸收系数矩阵，行序 = names
    names     : list[str] 组分名
    n_refr    : (L,)   叶片材料折射率（PROSPECT 平板模型需要）
    low_abs   : (L,) bool  吸收近零窗口掩码 —— 辨识性锚点
    extrapolated : (L,) bool  哪些波段是外插出来的（VIS 前端）
    """
    camera: str
    wl: np.ndarray
    eps: np.ndarray
    names: list
    n_refr: np.ndarray
    low_abs: np.ndarray
    extrapolated: np.ndarray

    @property
    def K(self) -> int:
        return self.eps.shape[0]

    @property
    def L(self) -> int:
        return self.wl.shape[0]

    def mu_a(self, c: np.ndarray) -> np.ndarray:
        """mu_a(lambda) = sum_i c_i * eps_i(lambda)。c: (K,) 或 (B, K)。"""
        c = np.asarray(c, dtype=np.float64)
        return c @ self.eps if c.ndim > 1 else self.eps.T @ c

    def summary(self) -> None:
        print("AbsorberBasis[%s]  L=%d bands  K=%d absorbers  %.2f-%.2f nm"
              % (self.camera, self.L, self.K, self.wl[0], self.wl[-1]))
        if self.extrapolated.any():
            print("  extrapolated bands: %d (below 400 nm, nearest-neighbour)"
                  % int(self.extrapolated.sum()))
        n_low = int(self.low_abs.sum())
        if n_low:
            print("  low-absorption anchor window: %d bands (%.1f-%.1f nm)"
                  % (n_low, self.wl[self.low_abs][0], self.wl[self.low_abs][-1]))
        else:
            print("  !! NO low-absorption anchor window -- scattering may be unidentifiable")
        print("  %-7s %12s %12s   %s" % ("name", "eps_min", "eps_max", "unit"))
        for i, nm in enumerate(self.names):
            e = self.eps[i]
            print("  %-7s %12.4g %12.4g   %s"
                  % (nm, e.min(), e.max(), ABSORBERS[nm]["unit"]))


def build_basis(camera: str,
                absorbers: tuple = ("cab", "car", "anth", "brown", "water", "dm"),
                ) -> AbsorberBasis:
    """构造某相机上的吸收基谱。"""
    raw = _load_raw()
    src_wl = raw[:, 0]
    idx = {c: i for i, c in enumerate(COLS)}

    wl = camera_wavelengths(camera)
    extrap = (wl < src_wl[0]) | (wl > src_wl[-1])

    def interp(col: str) -> np.ndarray:
        # np.interp 在范围外自动取端点值 = 最近邻外插，正是我们要的
        return np.interp(wl, src_wl, raw[:, idx[col]])

    eps = np.stack([interp(a) for a in absorbers], axis=0)
    n_refr = interp("n_refr")

    low = np.zeros_like(wl, dtype=bool)
    for lo, hi in LOW_ABSORPTION_WINDOWS.get(camera, []):
        low |= (wl >= lo) & (wl <= hi)

    return AbsorberBasis(camera=camera, wl=wl, eps=eps, names=list(absorbers),
                         n_refr=n_refr, low_abs=low, extrapolated=extrap)


def all_bases(**kw) -> dict:
    return {c: build_basis(c, **kw) for c in CAMERA_RANGE}


# ------------------------------------------------------------------ 自检
def _selftest() -> None:
    print("=" * 78)
    print("PROSPECT-D absorber basis -- per-camera sanity check")
    print("=" * 78)
    bases = all_bases()
    for cam, b in bases.items():
        print()
        b.summary()

    print()
    print("=" * 78)
    print("Known absorption peaks -- do they land where physics says?")
    print("=" * 78)
    # 文献锚点：叶绿素 a 在 ~662-670 nm；水在 970 与 1450 nm
    checks = [
        ("VIS", "cab", 600, 720, "chlorophyll a+b red peak (~662-670 nm)"),
        ("VIS", "water", 940, 1000, "water 3rd overtone (~970 nm)"),
        ("NIR", "water", 1400, 1500, "water 1st overtone (~1450 nm)"),
        ("VIS_COR", "cab", 600, 720, "chlorophyll a+b red peak"),
    ]
    for cam, comp, lo, hi, desc in checks:
        b = bases[cam]
        if comp not in b.names:
            continue
        k = b.names.index(comp)
        m = (b.wl >= lo) & (b.wl <= hi)
        if not m.any():
            print("  %-8s %-6s  window %d-%d nm not in range" % (cam, comp, lo, hi))
            continue
        peak_wl = b.wl[m][np.argmax(b.eps[k][m])]
        print("  %-8s %-6s  peak at %7.2f nm   <- %s" % (cam, comp, peak_wl, desc))

    print()
    print("=" * 78)
    print("Identifiability anchor: absorption strength in low-abs window")
    print("=" * 78)
    for cam, b in bases.items():
        if not b.low_abs.any():
            continue
        # 用典型浓度算一下锚点窗口里的 mu_a 相对全谱最大值
        c_typ = np.array([20.0, 8.0, 1.0, 0.1, 0.02, 0.008])[:b.K]
        mua = b.mu_a(c_typ)
        ratio = mua[b.low_abs].mean() / mua.max()
        flag = "GOOD" if ratio < 0.15 else "WEAK -- anchor may not pin scattering"
        print("  %-8s mean(mu_a) in window / max(mu_a) = %.4f   [%s]"
              % (cam, ratio, flag))


if __name__ == "__main__":
    _selftest()
