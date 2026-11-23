"""
DeepHS-Fruit v2 标签加载器 —— 合并三个来源，产出干净的记录表。

三个来源及其取舍理由
--------------------
1. 仓库硬编码 Python 清单 (deephs_fruit/core/measurements/*.py)
   - 唯一含 **糖度 sugar_content (°Brix)** 的来源
   - 官方发布的 JSON 标注里根本没有 sugar_content 字段（已核实）
2. 官方 JSON 标注 (annotations-upd-2024-01-09)
   - 含官方 train/val/test 切分
   - ReadMe 声明 "fixed one error"，即 2024 年修正过一处错误
   - 字段: init_weight, end_weight, storage_days, firmness, comment,
           ripeness_state, ripeness_state_fine
3. 本模块的数据清洗
   - firmness == 0 视为缺失（硬度不可能为 0；Kaki 有 12/40 = 30% 是 0）
   - 失重率为负视为异常（Avocado 有 -3.99%）

输出的每一行 = 一条高光谱记录 (fruit, day, id, side, camera)，带：
    连续量: firmness_g_cm2, sugar_brix, storage_days, weight_loss_pct
    类别量: ripeness3, firmness3, sugar3
    切分:   split ∈ {train, val, test}
    路径:   data_file, header_file

用法:
    from physfruit.labels import load_records, RecordTable
    tab = load_records()
    tab.summary()
    df = tab.df
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = str(PROJECT_ROOT / "deephs_fruit")
DATA_ROOT = str(PROJECT_ROOT / "DeepHS-Fruit" / "data")
ANN_DIR = os.path.join(DATA_ROOT, "annotations")
CUBE_ROOT = DATA_ROOT

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------- 相机元数据
CAMERAS = {
    "VIS":     dict(name="Specim FX10",                  n_bands=224, lo=397.66, hi=1003.81),
    "VIS_COR": dict(name="Corning microHSI 410 Vis-NIR",  n_bands=249, lo=408.03, hi=901.26),
    "NIR":     dict(name="INNO-SPEC RedEye 1.7",          n_bands=252, lo=919.68, hi=1727.04),
}

# 官方分级阈值（core/name_convention.py），写论文时需列出
FIRMNESS_THRESH = {           # (too_soft_below, too_hard_above)  单位 g/cm^2
    "Avocado": (900, 1200),
    "Kiwi":    (1000, 1500),
    "Kaki":    (500, 1500),
    "Mango":   (5000, 12500),
    "Papaya":  (1000, 2000),
}
SUGAR_THRESH = {              # (not_sweet_below, too_sweet_above)  单位 °Brix
    "Kiwi":   (15.5, 17.0),
    "Kaki":   (20.0, 22.0),
    "Mango":  (14.0, 16.0),
    "Papaya": (11.5, 13.5),
}
# Avocado 无糖度标签（AvocadoLabel 不含 sugar_content）


def _bin3(v: Optional[float], lo: float, hi: float,
          names=("too_soft", "ready", "too_hard")) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if v < lo:
        return names[0]
    if v > hi:
        return names[2]
    return names[1]


# ---------------------------------------------------------- 来源 1: Python 清单
def _sugar_from_repo() -> dict:
    """从仓库硬编码清单提取 {(fruit, day, id): sugar_brix}。

    这是糖度的唯一来源。糖度按「果实」而非「记录」给出，同一果实的
    front/back 与多相机记录共享同一个值。
    """
    from core.measurements import all_fruits_new  # noqa: PLC0415
    from core.name_convention import SweetFruitLabel  # noqa: PLC0415

    out = {}
    for r in all_fruits_new:
        if not r.is_labeled():
            continue
        lab = r.label
        if isinstance(lab, SweetFruitLabel) and lab.sugar_content is not None:
            out[(r.fruit.value, r.day.value, r.id.value)] = float(lab.sugar_content)
    return out


# ---------------------------------------------------------- 来源 2: 官方 JSON
def _parse_ids(data_file: str) -> tuple:
    """Avocado/VIS/day_01/avocado_day_01_07_front.bin -> (id='07', side='front')

    测试日记录形如 avocado_test2_front.bin（无 id 段），此时 id 记为 '?'。
    """
    base = os.path.basename(data_file).rsplit(".", 1)[0]
    parts = base.split("_")
    side = parts[-1]
    fid = parts[-2] if len(parts) >= 2 and parts[-2].isdigit() else "?"
    return fid, side


def _load_split(split_name: str, split_label: str) -> list:
    path = os.path.join(ANN_DIR, split_name + ".json")
    with open(path, encoding="utf-8") as fh:
        j = json.load(fh)
    recs = {r["id"]: r for r in j["records"]}
    rows = []
    for a in j["annotations"]:
        r = recs[a["record_id"]]
        fid, side = _parse_ids(r["files"]["data_file"])
        rows.append(dict(
            fruit=r["fruit"], day=r["day"], fruit_id=fid, side=side,
            camera=r["camera_type"],
            data_file=r["files"]["data_file"],
            header_file=r["files"]["header_file"],
            init_weight=a.get("init_weight"),
            end_weight=a.get("end_weight"),
            storage_days=a.get("storage_days"),
            firmness_raw=a.get("firmness"),
            ripeness3=a.get("ripeness_state"),
            ripeness5=a.get("ripeness_state_fine"),
            comment=a.get("comment"),
            split=split_label,
        ))
    return rows


# ------------------------------------------------------------------ 测量序列
def _series_of(day: str) -> str:
    d = day.lower()
    if d.startswith("day_m2"):
        return "M2_2020"
    if d.endswith("_m3"):
        return "M3_2021"
    if d.startswith("day_m4"):
        return "M4_2022"
    if d.startswith("day_"):
        return "M1_2019"
    return "TEST"


@dataclass
class RecordTable:
    df: pd.DataFrame
    sugar_coverage: float = 0.0
    notes: list = field(default_factory=list)

    def summary(self) -> None:
        d = self.df
        print("=" * 78)
        print("RecordTable: %d records / %d unique fruits"
              % (len(d), d["uid"].nunique()))
        print("=" * 78)
        for n in self.notes:
            print("  ! " + n)
        print()
        print("-- records by split x camera --")
        print(pd.crosstab(d["split"], d["camera"]))
        print()
        print("-- UNIQUE FRUITS by fruit x camera (labeled) --")
        u = d.drop_duplicates(subset=["uid", "camera"])
        print(pd.crosstab(u["fruit"], u["camera"]))
        print()
        print("-- unique fruits by fruit x split --")
        u2 = d.drop_duplicates(subset=["uid", "camera", "split"])
        print(pd.crosstab(u2["fruit"], u2["split"]))
        print()
        print("-- continuous target availability (unique fruits) --")
        uf = d.drop_duplicates(subset=["uid"])
        rows = []
        for fr, g in uf.groupby("fruit"):
            rows.append(dict(
                fruit=fr, n=len(g),
                firmness_ok=int(g["firmness_g_cm2"].notna().sum()),
                firmness_dropped0=int(g["firmness_was_zero"].sum()),
                sugar_ok=int(g["sugar_brix"].notna().sum()),
                wloss_ok=int(g["weight_loss_pct"].notna().sum()),
                wloss_neg=int(g["weight_loss_neg"].sum()),
            ))
        print(pd.DataFrame(rows).to_string(index=False))
        print()
        print("-- degenerate tasks (any class < 3 unique fruits) --")
        for task in ["ripeness3", "firmness3", "sugar3"]:
            for fr, g in uf.groupby("fruit"):
                vc = g[task].value_counts()
                if len(vc) == 0:
                    continue
                if vc.min() < 3 or len(vc) < 3:
                    print("   %-8s %-10s -> %s   << EXCLUDE"
                          % (fr, task, dict(vc)))


def load_records(verbose: bool = False) -> RecordTable:
    notes = []

    rows = []
    rows += _load_split("train_only_labeled_v2", "train")
    rows += _load_split("val_v2", "val")
    rows += _load_split("test_v2", "test")
    df = pd.DataFrame(rows)

    # 唯一果实标识：同一果实的 front/back 与多相机记录共享
    df["uid"] = df["fruit"] + "|" + df["day"] + "|" + df["fruit_id"]
    df["series"] = df["day"].map(_series_of)
    # 物理果实标识：M2 纵向序列中同一 fruit_id 可跨多个 day 出现；
    # 任何交叉验证都必须按此列分组，不能只按 uid 分组。
    df["physical_uid"] = df["fruit"] + "|" + df["series"] + "|" + df["fruit_id"]

    # ---------------- 清洗 1: firmness == 0 视为缺失
    df["firmness_was_zero"] = df["firmness_raw"].fillna(-1) == 0
    df["firmness_g_cm2"] = df["firmness_raw"].where(
        (df["firmness_raw"].notna()) & (df["firmness_raw"] > 0))
    n0 = int(df["firmness_was_zero"].sum())
    if n0:
        notes.append("firmness==0 视为缺失并剔除：%d 条记录 "
                     "(硬度不可能为 0，是缺失值编码)" % n0)

    # ---------------- 来源 1: 补糖度
    sugar = _sugar_from_repo()
    df["sugar_brix"] = [
        sugar.get((f, d_, i)) for f, d_, i in
        zip(df["fruit"], df["day"], df["fruit_id"])
    ]
    cov_fruits = df[df["fruit"] != "Avocado"].drop_duplicates("uid")
    cov = float(cov_fruits["sugar_brix"].notna().mean()) if len(cov_fruits) else 0.0
    notes.append("糖度来自仓库 Python 清单（官方 JSON 无此字段）；"
                 "非 Avocado 果实覆盖率 %.1f%%" % (100 * cov))

    # ---------------- 清洗 2: 失重率
    iw = pd.to_numeric(df["init_weight"], errors="coerce")
    ew = pd.to_numeric(df["end_weight"], errors="coerce")
    wl = 100.0 * (iw - ew) / iw.where(iw > 0)
    df["weight_loss_neg"] = wl < 0
    df["weight_loss_pct"] = wl.where(wl >= 0)
    nneg = int(df["weight_loss_neg"].sum())
    if nneg:
        notes.append("失重率为负视为异常并剔除：%d 条记录（增重物理上不可能）" % nneg)

    overlap_groups = int(df.groupby("physical_uid")["split"].nunique().gt(1).sum())
    if overlap_groups:
        notes.append("官方 split 在 %d 个物理果实组上出现多 split；评测必须使用 physical_uid 分组"
                     % overlap_groups)

    # ---------------- 类别标签（由连续量按官方阈值重算，保证自洽）
    df["firmness3"] = [
        _bin3(v, *FIRMNESS_THRESH[f]) if f in FIRMNESS_THRESH else None
        for v, f in zip(df["firmness_g_cm2"], df["fruit"])
    ]
    df["sugar3"] = [
        _bin3(v, *SUGAR_THRESH[f], names=("not_sweet", "ready", "too_sweet"))
        if f in SUGAR_THRESH else None
        for v, f in zip(df["sugar_brix"], df["fruit"])
    ]

    # ---------------- 立方体是否已下载
    df["cube_exists"] = [
        os.path.exists(os.path.join(CUBE_ROOT, p.replace("/", os.sep)))
        for p in df["data_file"]
    ]

    # 相机元数据
    df["n_bands"] = df["camera"].map(lambda c: CAMERAS[c]["n_bands"])
    df["wl_lo"] = df["camera"].map(lambda c: CAMERAS[c]["lo"])
    df["wl_hi"] = df["camera"].map(lambda c: CAMERAS[c]["hi"])

    tab = RecordTable(df=df, sugar_coverage=cov, notes=notes)
    if verbose:
        tab.summary()
    return tab


# ------------------------------------------------- 跨相机配对（E4 灵魂实验）
def camera_pairs(df: pd.DataFrame, cam_a: str, cam_b: str) -> pd.DataFrame:
    """返回同一果实同一侧同时被 cam_a 与 cam_b 拍到的配对表。

    E4 要检验的是：同一果实的散射参数 (a, b) 在两台相机上应当一致。
    """
    a = df[df["camera"] == cam_a][["uid", "side", "data_file"]]
    b = df[df["camera"] == cam_b][["uid", "side", "data_file"]]
    m = a.merge(b, on=["uid", "side"], suffixes=("_" + cam_a, "_" + cam_b))
    return m


if __name__ == "__main__":
    t = load_records(verbose=True)
    print()
    print("=" * 78)
    print("[E4] cross-camera pairs (same fruit, same side)")
    print("=" * 78)
    for ca, cb in [("VIS", "NIR"), ("VIS", "VIS_COR"), ("VIS_COR", "NIR")]:
        m = camera_pairs(t.df, ca, cb)
        n_fruit = m["uid"].nunique() if len(m) else 0
        print("  %-8s <-> %-8s : %4d record pairs, %3d unique fruits"
              % (ca, cb, len(m), n_fruit))
    print()
    print("cubes downloaded so far: %d / %d records"
          % (int(t.df["cube_exists"].sum()), len(t.df)))
