# 运行与统计说明

## 1. 运行任务

| 任务 | 输入 | 主要输出 |
| --- | --- | --- |
| 结果核验与图表生成 | 仓库内的固定版本 CSV | 汇总核验、PDF/SVG/PNG 和作图数据 |
| 配对光谱实验 | 配对 NPZ、官方 JSON 标注、上游测量元数据 | 各方法逐折或逐重复指标 |
| 原始光谱处理 | 原始 BIN/HDR 图像、官方标注和相机波长表 | 果实平均光谱及配对 NPZ |

Python 依赖见 `requirements.txt`；原始图像处理使用的额外依赖见 `requirements-raw.txt`。

## 2. 数据准备

从仓库根目录获取上游代码：

```bash
git clone https://github.com/cogsys-tuebingen/deephs_fruit.git deephs_fruit
git -C deephs_fruit checkout 5f47812a3fb5bd8f0ed5367cedf9b464f83076c1
```

从 [DeepHS-Fruit v2](https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/) 获取图像与标注，保留官方路径结构：

```text
deephs_fruit/core/...
DeepHS-Fruit/data/annotations/train_only_labeled_v2.json
DeepHS-Fruit/data/annotations/val_v2.json
DeepHS-Fruit/data/annotations/test_v2.json
DeepHS-Fruit/data/<官方标注引用的图像路径>
```

标签加载器使用 JSON 中的测量记录，以及上游 `core.measurements`、`core.name_convention` 中的元数据。光谱重采样使用 `core.spectral_io` 中的官方相机波长表。

### 生成配对光谱缓存

原始图像处理依赖安装完成后，以猕猴桃为例：

```bash
python -m pip install -r requirements-raw.txt
python -m physfruit.inter_transfer --fruit Kiwi --a VIS --b NIR
```

该入口读取配对图像、计算前景平均光谱并重采样至公共波长位置，将 NPZ 和直接传递诊断 CSV 写入 `baseline_cache/`。后续六方法实验由 `inter_transfer_baselines.py` 执行。

六个配对子集及其参数如下：

| 水果 | `--fruit` | `--a` | `--b` |
| --- | --- | --- | --- |
| 鳄梨 | Avocado | VIS | NIR |
| 鳄梨 | Avocado | VIS | VIS_COR |
| 柿 | Kaki | VIS | VIS_COR |
| 猕猴桃 | Kiwi | VIS | NIR |
| 芒果 | Mango | VIS | VIS_COR |
| 番木瓜 | Papaya | VIS | VIS_COR |

配对缓存命名为 `inter_<fruit>_<a>_<b>_overlap.npz`，包含 `Xa`、`Xb`、`uids`。两个光谱矩阵的每一行通过 `uids` 对应同一果实记录。实验入口从仓库根目录下的 `baseline_cache/` 读取这些 NPZ。

## 3. 主要实验入口

| 脚本（位于 `physfruit/`） | 用途 |
| --- | --- |
| `inter_transfer_baselines.py` | Raw、Scale、CORAL、Pair Ridge、PDS(w=1)、PDS(w=2) 六方法比较 |
| `inter_calibration_budget.py` | 配对校准样本量预算 |
| `inter_supervised_reference.py` | 目标相机监督参考 |
| `inter_pairing_ablation.py` | 每折一次打乱配对的消融 |
| `inter_permutation_null.py` | 重复打乱配对的描述性分析 |
| `inter_mechanism_diagnostics.py` | 光谱差异与硬度预测的联合诊断 |
| `inter_storage_holdout.py` | 贮藏后期留出测试及果实身份清单 |
| `inter_plsr_sensitivity.py` | PLSR 源模型敏感性 |
| `inter_band_count_pooled.py` | 缓存选列及重复特征的维度敏感性 |
| `inter_alpha_sensitivity.py` | 正则化敏感性 |

```bash
python -m physfruit.inter_transfer_baselines --fruit Kiwi --a VIS --b NIR --repeats 5 --folds 5 --cache runs/main
python -m physfruit.inter_calibration_budget --fruit Kiwi --a VIS --b NIR --repeats 5 --folds 5 --budgets 4 8 16 24 32 --subsamples 5 --cache runs/budget
python -m physfruit.inter_storage_holdout --fruit Kiwi --a VIS --b NIR --cache runs/storage
```

每个入口同时计算两个传递方向。`--cache` 指定结果输出目录，配对 NPZ 的输入目录为 `baseline_cache/`。建议将新结果保存到 `runs/`，与固定版本结果分别管理。

特征维度实验通过 Python 接口指定输出目录：

```python
from physfruit.inter_band_count_pooled import run

run("Kiwi", "VIS", "NIR", repeats=5, folds=5, cache="runs/features")
```

## 4. 统计口径

- **主验证**：采用五次重复的五折物理果实分组验证。每次重复先汇合各测试折预测，再计算指标；均值与样本 SD 在五次重复间汇总。
- **模型与映射**：源模型为 Elastic Net，目标标签采用对数变换，预测经指数减一还原至原始硬度尺度。Pair Ridge 的正则化系数为 `5 × p`，20 列时为 100。PLSR 用于源模型敏感性分析。
- **校准预算**：实际校准数量为名义预算与该折可用配对果实数的较小值。先在每次划分内平均五次子抽样指标，再跨五次划分计算均值与 SD。
- **两类差值**：配对身份差值为真实配对 R² 减打乱配对 R²；相对基准的改善为 Pair Ridge R² 减 Raw R²。
- **重复置换**：每次划分内平均 50 次置换差值，再跨五次划分汇总；250 个差值的分位数用于描述置换与划分共同形成的变化范围。
- **贮藏测试**：按数据集内部的贮藏日划分早期训练与后期测试样本，所有模型均在早期数据上拟合，并记录训练与测试的果实身份。
- **特征维度**：8、16、32 列设置均从 20 个波长的缓存中选列，其中 32 列设置包含重复特征。

## 5. 结果与来源记录

`main_results_frozen_2026-10-03.csv` 汇总六种设置的主要指标；六个 `inter_transfer_baselines_*.csv` 保留各子集的逐折与逐重复结果。其他 CSV 按实验名称和配对子集命名。

- [结果清单](frozen_results_manifest.csv)：文件大小与 SHA-256。
- [代码来源清单](source_manifest.csv)：归档代码的来源与版本对应关系。
- [代码与结果索引](paper_scope.md)：论文入口、公共模块及扩展分析。
- [验证记录](upload_validation.md)：分组测试、结果汇总与绘图检查。

扩展物理模型的 PROSPECT 系数表按 [来源说明](../physfruit/data/SOURCE.md) 获取，放入 `physfruit/data/`。
