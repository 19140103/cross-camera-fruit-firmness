# 复现说明

## 1. 区分三种运行层级

1. **冻结结果核验与重绘**：只需仓库内的 CSV 和 Python 依赖，无需原始图像、标注或模型训练。
2. **配对光谱缓存上的实验重放**：还需要六个配对 NPZ、官方 JSON 标注和上游代码。此次上传没有执行该层级。
3. **从原始图像重建**：还需要原始 BIN/HDR 文件及 PyTorch。原项目曾清点 392 对 BIN/HDR 文件，完整重建尚未验证。

此次上传只在现有本地环境验证已有测试、冻结结果与绘图入口，不把这些检查称为完整独立复现。

## 2. 主实验与支持实验

主实验入口为 `physfruit/inter_transfer_baselines.py`，提供 Raw、Scale、CORAL、Pair Ridge、PDS(w=1)、PDS(w=2) 六种设置。`inter_domain_final.py` 保留 Raw、Scale、CORAL、Pair Ridge 四种设置；不要把四方法入口当作全量六方法结果。

| 脚本 | 用途 |
| --- | --- |
| `inter_transfer_baselines.py` | 六种设置的主比较 |
| `inter_calibration_budget.py` | 配对校准预算 |
| `inter_supervised_reference.py` | 目标相机监督参考 |
| `inter_pairing_ablation.py` | 每折一次打乱配对的消融 |
| `inter_permutation_null.py` | 重复打乱配对的描述性诊断 |
| `inter_mechanism_diagnostics.py` | 光谱差异与硬度预测的联合诊断 |
| `inter_storage_holdout.py` | 数据集内部贮藏后期留出测试及身份清单 |
| `inter_plsr_sensitivity.py` | PLSR 源模型敏感性 |
| `inter_band_count_pooled.py` | 缓存选列及重复特征的维度敏感性 |
| `inter_alpha_sensitivity.py` | 正则化敏感性 |
| `evaluation.py`、`test_evaluation.py` | 果实分组工具与已有测试 |

表中脚本位于 `physfruit/`。其余代码保留研究过程中的探索实现；尤其是 `inter_nested_*`、`inter_domain_invariant.py` 中的选参入口、`intra_*`、`simtest*`、`fusionfit.py` 等，不应混同于主实验的固定参数方案。测试文件中的嵌套分组检查是工具测试，不表示主实验采用嵌套调参。

## 3. 数据和目录

从仓库根目录获取上游代码：

```bash
git clone https://github.com/cogsys-tuebingen/deephs_fruit.git deephs_fruit
git -C deephs_fruit checkout 5f47812a3fb5bd8f0ed5367cedf9b464f83076c1
```

按上游说明获取数据，保留官方目录结构：

```text
deephs_fruit/core/...
DeepHS-Fruit/data/annotations/train_only_labeled_v2.json
DeepHS-Fruit/data/annotations/val_v2.json
DeepHS-Fruit/data/annotations/test_v2.json
DeepHS-Fruit/data/<官方标注引用的图像路径>
baseline_cache/inter_Avocado_VIS_NIR_overlap.npz
baseline_cache/inter_Avocado_VIS_VIS_COR_overlap.npz
baseline_cache/inter_Kaki_VIS_VIS_COR_overlap.npz
baseline_cache/inter_Kiwi_VIS_NIR_overlap.npz
baseline_cache/inter_Mango_VIS_VIS_COR_overlap.npz
baseline_cache/inter_Papaya_VIS_VIS_COR_overlap.npz
```

NPZ 包含 `Xa`、`Xb`、`uids`；UID 及数组行序必须一致。当前 `paired_table` 从根目录下的 `baseline_cache` 读取 NPZ；大多数脚本的 `--cache` 参数指定的是**输出目录**，不会改变这个输入位置。标签加载器同时引用上游 `core.measurements` 和 `core.name_convention`，不能只复制 JSON 而省略上游代码。

## 4. 实验命令示例（需要数据，会重新拟合）

以下命令用于今后的实验重放，**此次归档未执行**。指定单独输出目录，保留 `baseline_cache` 中的冻结 CSV：

```bash
python -m physfruit.inter_transfer_baselines --fruit Kiwi --a VIS --b NIR --repeats 5 --folds 5 --cache runs/main
python -m physfruit.inter_calibration_budget --fruit Kiwi --a VIS --b NIR --repeats 5 --folds 5 --budgets 4 8 16 24 32 --subsamples 5 --cache runs/budget
python -m physfruit.inter_storage_holdout --fruit Kiwi --a VIS --b NIR --cache runs/storage
```

每个入口会计算两个传递方向。六个配对子集是：Avocado/VIS/NIR、Avocado/VIS/VIS_COR、Kaki/VIS/VIS_COR、Kiwi/VIS/NIR、Mango/VIS/VIS_COR、Papaya/VIS/VIS_COR。

`inter_band_count_pooled.py` 的现有命令行没有 `--cache`；如需重放，可调用其 `run(..., cache=...)` 并显式指定输出目录。不要直接以默认参数运行所有历史脚本。

原始光谱重建入口 `inter_transfer.py` 还会运行历史诊断，并向 `baseline_cache` 写 NPZ 和 CSV；不应把它生成的历史诊断 CSV 当作主结果。应在独立工作副本中操作。它通过 `realfit.py` 导入 PyTorch；完整依赖另见 `requirements-raw.txt`。

`basis.camera_wavelengths` 保留原代码的端点线性回退行为。重建论文输入时必须确认成功加载了上游真实波长表；回退网格不能作为等价复现输入。历史 PROSPECT 物理模型还需要本仓库未分发的系数表，见来源说明。

## 5. 统计口径

- 主实验为五次重复的五折物理果实分组验证。每次重复先汇合各测试折预测，再计算 R²、RMSE、MAE；均值和样本 SD 在五次重复间汇总。
- 源模型为 Elastic Net；配对映射固定正则化系数为 `5 × p`，20 列时为 100。PLSR 是敏感性分析。
- 预算 32 是名义上限，实际数量为预算与该折可用配对果实数的较小值。预算重复层级汇总应先在每次划分内平均五次子抽样，再跨五次划分计算均值与 SD。
- 配对身份差值为真实配对 R² 减打乱配对 R²；相对基准差值为 Pair Ridge R² 减 Raw R²，两者不能混用。
- 重复置换图先在每次划分内平均 50 次置换差值，再跨五次划分汇总。250 个差值的分位数仅是描述性范围，不是置信区间；零分布比例不是 p 值。
- 后期留出是数据集内部时序压力测试，不是外部验证。源模型、映射和时间对照仅拟合早期训练数据；脚本检查训练/测试物理果实身份不重叠。
- “32 维”设置从 20 列缓存选列，含重复特征，不是 32 个不同波长。

## 6. 上传副本的改动范围

`inter_transfer.py`、`inter_overlap.py`、`basis.py`、`envi.py` 中四处旧盘符路径改为项目相对路径。其余原有 Python 源文件逐字节保留。具体哈希见 `source_manifest.csv`。未调整模型、种子、数据筛选或统计计算。

冻结 CSV 按原文件复制并记录哈希。旧的、只有四个子集的 `inter_domain_final_*.csv` 历史结果未收入归档；完整主比较以六个 `inter_transfer_baselines_*.csv` 和 `main_results_frozen_2026-10-03.csv` 为准。

