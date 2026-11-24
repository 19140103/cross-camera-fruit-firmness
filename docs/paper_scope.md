# 论文代码与冻结结果范围

本仓库当前兼顾论文复核与研究过程备份，不能把所有保留文件都解释为论文主实验必需代码。以下范围来自现有 Python 导入依赖检查，不涉及重新训练。

## 直接支撑本文实验：14 个文件

10 个实验入口：

| 文件（均位于 `physfruit/`） | 本文用途 |
| --- | --- |
| `inter_transfer_baselines.py` | 六种设置的主性能比较 |
| `inter_pairing_ablation.py` | 配对身份的单次打乱消融 |
| `inter_permutation_null.py` | 重复打乱配对诊断 |
| `inter_mechanism_diagnostics.py` | 光谱差异与预测性能 |
| `inter_calibration_budget.py` | 校准样本量预算 |
| `inter_supervised_reference.py` | 目标系统监督参考 |
| `inter_storage_holdout.py` | 贮藏后期留出测试 |
| `inter_plsr_sensitivity.py` | 源模型敏感性 |
| `inter_band_count_pooled.py` | 缓存选列及重复特征的维度诊断 |
| `inter_alpha_sensitivity.py` | 正则化敏感性 |

4 个公共模块为 `evaluation.py`、`inter_alignment.py`、`inter_domain_invariant.py`、`labels.py`。`inter_domain_invariant.py` 同时保留探索性函数，本文只使用相应的配对读取和映射实现；不要把文件中的所有函数都当作主实验方案。

## 数据准备与测试：6 个文件

`inter_transfer.py`、`basis.py`、`envi.py`、`realfit.py`、`forward.py` 涉及原始图像读取、波长信息、前景提取和导入依赖。这 5 个文件也含历史功能，并非每个函数都用于当前论文。原始数据重建的条件和限制见 [复现说明](reproduction.md)。

`test_evaluation.py` 是已有分组验证测试。加上上述模块，研究代码中的 20 个文件构成当前论文最相关的代码范围；完整重放仍需未分发的上游数据、标注、代码和配对缓存。

## 历史探索或替代入口：24 个文件

以下文件保留用于研究过程追溯，不是复核当前六方法主结果的必需入口：

```text
baseline_regression.py          evaluate_corrected.py
fusionfit.py                    inter_band_count.py
inter_calibration.py            inter_domain_final.py
inter_nested_alpha.py           inter_nested_supervised_alpha.py
inter_overlap.py                inter_series_holdout.py
inter_stable_selection.py       inter_storage_diagnostics.py
intra_band_selection.py         intra_coverage_selection.py
intra_interval_pool.py          intra_stable_intervals.py
physical_features.py            region_pooling.py
simtest.py                      simtest_noise.py
simtest_range.py                simtest_reparam.py
spectral_views.py               transfer_reliability_diagnostic.py
```

例如 `inter_domain_final.py` 只提供四种设置；当前六方法结果以 `inter_transfer_baselines.py` 对应缓存为准。`inter_nested_*` 不代表本文主实验采用嵌套调参。

## 为什么保留 61 个 CSV

61 个文件是 **10 类实验 × 6 个配对子集 + 1 张主结果汇总表**，总计 1,638,047 字节，约 1.56 MiB。六个配对子集各包含两个传递方向，合计 12 个方向；并不是 61 个独立实验。

每类缓存分别保留逐折、逐重复、预算子抽样或置换层级的指标，以实际 CSV 的 `aggregate` 和其他层级字段为准。这些记录可用于复核均值、SD、作图数据及差值定义，比只保存最终四舍五入的表格更有用。主汇总表是便于阅读的派生表，保留它不代替逐重复证据。

这些 CSV 是本研究产生的数值结果，不是原始高光谱图像，也不包含作者个人资料。文件小、内容可比较，适合随代码在 GitHub 保存。`frozen_results_manifest.csv` 记录字节级校验值；复现实验应输出到独立目录，不覆盖冻结文件。

论文对外发布时宜突出上述 20 个相关研究文件、3 个绘图脚本及核验工具；历史探索已在此明确列出，当前没有删除历史记录，也没有更改任何结果。仓库是否公开与结果是否适合归档是两件事：私有仓库地址不会自动给予审稿人或读者访问权限。
