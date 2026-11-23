"""
轻量 ENVI 读取器 —— 不依赖 `spectral` 包。

为什么自己写
-----------
`spectral` 0.23 在 Windows 上会崩：spyfile.py:199 尝试构造 np.dtype('complex256')，
而 Windows 的 numpy 只有 complex64/complex128 -> TypeError。
DeepHS-Fruit 的 ENVI 文件格式极简（见下），自己读 20 行就够，
顺便去掉一个脆弱依赖，对论文复现也更友好。

DeepHS-Fruit v2 立方体的实际格式（已核实）
----------------------------------------
    ENVI
    samples = 64            宽
    lines   = 64            高
    bands   = 224           波段（VIS=224 / VIS_COR=249 / NIR=252）
    header offset = 0
    data type = 4           4 = float32
    interleave = bip        band-interleaved-by-pixel -> [line][sample][band]
    byte order = 0          小端

尺寸校验: 64*64*224*4 = 3,670,016 bytes，与实际文件精确吻合。

三个重要事实
-----------
1. **头文件不含 wavelength 字段** -> 波长必须来自 physfruit.basis.camera_wavelengths()
2. **立方体尺寸因测量序列而异**（此前"都是 64x64"的说法只对 Kaki 成立，已更正）：
       Kaki / Mango   (M3, 2021)        64x64 预裁剪，单一尺寸，3.5-3.9 MB
       Kiwi / Avocado (M1/M2, 2019-20)  原生分辨率裁剪，~185-350 px 边长，37-108 MB
       Papaya         (M4, 2022)        VIS 75-167 MB；VIS_COR 6-12 MB
   两类都是**以果实为中心**的裁剪（果实占帧 50-70%，四周一圈薄暗边），
   不是果实在角落的整幅场景。`fruit_pixels()` 的分位数掩码在两类上都验证为
   紧凑团块。read_cube() 按头文件维度 reshape，无需特殊处理。
   仓库原始管线用 cv2.resize 统一到 62x62，正是为吸收这一差异。
3. **同目录下无 _White / _Dark 文件** -> 发布的数据已经过白板/暗电流定标，
   即已是漫反射率量纲，不要重复定标（这是物理反演成立的前提）
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

# ENVI data type 编码 -> numpy dtype
ENVI_DTYPE = {
    1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
    5: np.float64, 12: np.uint16, 13: np.uint32,
    14: np.int64, 15: np.uint64,
}


@dataclass
class EnviHeader:
    samples: int
    lines: int
    bands: int
    dtype: np.dtype
    interleave: str
    byte_order: int
    header_offset: int
    raw: dict

    @property
    def expected_bytes(self) -> int:
        return self.samples * self.lines * self.bands * np.dtype(self.dtype).itemsize


def read_header(hdr_path: str) -> EnviHeader:
    with open(hdr_path, encoding="latin-1") as fh:
        txt = fh.read()
    if not txt.lstrip().startswith("ENVI"):
        raise ValueError("not an ENVI header: %s" % hdr_path)

    # 形如 `key = value`。键名可含空格（如 "header offset"）但不可跨行，
    # 所以用 [^\n=] 而不是 \s —— \s 会匹配换行，把 "ENVI\nsamples" 当成键名。
    kv = {}
    for m in re.finditer(r"^[ \t]*([^\n=]+?)[ \t]*=[ \t]*(.+?)[ \t]*$", txt, flags=re.M):
        kv[m.group(1).strip().lower()] = m.group(2).strip()

    dt_code = int(kv["data type"])
    if dt_code not in ENVI_DTYPE:
        raise ValueError("unsupported ENVI data type %d" % dt_code)

    return EnviHeader(
        samples=int(kv["samples"]),
        lines=int(kv["lines"]),
        bands=int(kv["bands"]),
        dtype=np.dtype(ENVI_DTYPE[dt_code]),
        interleave=kv.get("interleave", "bsq").lower(),
        byte_order=int(kv.get("byte order", 0)),
        header_offset=int(kv.get("header offset", 0)),
        raw=kv,
    )


def read_cube(stem: str, dtype=np.float32) -> tuple:
    """读取一个立方体。

    参数
    ----
    stem : 不含扩展名的路径，例如 .../kaki_day_1_m3_01_front
           自动寻找 .hdr 以及 .bin 或 .img

    返回
    ----
    (cube, header)  cube 形状 (lines, samples, bands)
    """
    hdr_path = stem + ".hdr"
    bin_path = None
    for ext in (".bin", ".img", ".dat", ""):
        if os.path.exists(stem + ext) and ext != "":
            bin_path = stem + ext
            break
    if bin_path is None:
        raise FileNotFoundError("no data file for %s" % stem)

    h = read_header(hdr_path)

    actual = os.path.getsize(bin_path) - h.header_offset
    if actual != h.expected_bytes:
        raise ValueError(
            "size mismatch for %s: file has %d bytes, header implies %d "
            "(%dx%dx%d %s)" % (bin_path, actual, h.expected_bytes,
                               h.lines, h.samples, h.bands, h.dtype))

    dt = h.dtype.newbyteorder(">" if h.byte_order == 1 else "<")
    flat = np.fromfile(bin_path, dtype=dt, offset=h.header_offset)

    il = h.interleave
    if il == "bip":      # [line][sample][band]
        cube = flat.reshape(h.lines, h.samples, h.bands)
    elif il == "bil":    # [line][band][sample]
        cube = flat.reshape(h.lines, h.bands, h.samples).transpose(0, 2, 1)
    elif il == "bsq":    # [band][line][sample]
        cube = flat.reshape(h.bands, h.lines, h.samples).transpose(1, 2, 0)
    else:
        raise ValueError("unknown interleave: %s" % il)

    return np.ascontiguousarray(cube, dtype=dtype), h


def fruit_mask(cube: np.ndarray, percentile: float = 55.0) -> np.ndarray:
    """粗略的果实/背景分割掩码。

    按整段光谱的平均反射率取阈值。果实比背景亮，故取高于分位数的像元。
    物理反演只应在果实像元上做——背景像元没有 mu_a / mu_s' 可言。
    """
    bright = cube.mean(axis=2)
    return bright > np.percentile(bright, percentile)


if __name__ == "__main__":
    import glob
    import sys

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DeepHS-Fruit", "data")
    print("=" * 78)
    print("ENVI reader self-test on real DeepHS-Fruit cubes")
    print("=" * 78)

    any_found = False
    for cam, want_bands in [("VIS", 224), ("VIS_COR", 249), ("NIR", 252)]:
        pat = os.path.join(root, "*", cam, "*", "*.hdr")
        files = sorted(glob.glob(pat))
        if not files:
            print("\n[%s] no cubes downloaded yet" % cam)
            continue
        any_found = True
        print("\n[%s] %d cubes found" % (cam, len(files)))

        stats = []
        for f in files[:40]:
            stem = f[:-4]
            try:
                c, h = read_cube(stem)
            except Exception as e:  # noqa: BLE001
                print("   FAIL %s: %s" % (os.path.basename(stem), e))
                continue
            m = fruit_mask(c)
            v = c[m]
            stats.append((c.shape, h.bands, float(v.min()), float(v.max()),
                          float(v.mean()), float((v < 0).mean()), float((v > 1).mean())))

        if not stats:
            continue
        shp = stats[0][0]
        nb = stats[0][1]
        print("   shape %s  bands %d  (expected %d) %s"
              % (shp, nb, want_bands, "OK" if nb == want_bands else "<< MISMATCH"))
        arr = np.array([s[2:] for s in stats])
        print("   over %d cubes, fruit pixels only:" % len(stats))
        print("      min    %8.4f" % arr[:, 0].min())
        print("      max    %8.4f" % arr[:, 1].max())
        print("      mean   %8.4f" % arr[:, 2].mean())
        print("      frac<0 %8.4f%%" % (100 * arr[:, 3].mean()))
        print("      frac>1 %8.4f%%" % (100 * arr[:, 4].mean()))
        hi = arr[:, 1].max()
        frac_neg = arr[:, 3].mean()
        frac_hi = arr[:, 4].mean()
        # 定标后的数据允许极少量负值：光谱两端传感器灵敏度低，
        # 暗电流扣除后会产生接近零的小负数，这是正常的，不是定标问题。
        if hi <= 1.6 and frac_neg < 0.01 and frac_hi < 0.01:
            print("      => 数值落在反射率量纲 [0,~1] -> 已定标，符合物理反演前提")
            if frac_neg > 0:
                print("         (少量负值来自低灵敏度波段的暗电流扣除，属正常)")
        else:
            print("      => 数值超出反射率量纲 -> 需要复查是否已定标")

    if not any_found:
        print("\nno cubes extracted yet; run:  unzip -q zips/Kaki.zip -d data/")
        sys.exit(0)
