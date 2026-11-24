# 代码与结果索引

代码按用途分为论文实验、数据准备、验证测试和扩展分析。下表列出各模块的作用及依赖关系。

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

4 个公共模块为 `evaluation.py`、`inter_alignment.py`、`inter_domain_invariant.py`、`labels.py`，分别提供分组验证、分布对齐、配对读取与映射、测量标签加载等功能。

## 数据准备与测试：6 个文件

`inter_transfer.py`、`basis.py`、`envi.py`、`realfit.py`、`forward.py` 提供原始图像读取、波长信息、前景提取及相关依赖。数据准备步骤见 [运行说明](reproduction.md)。

`test_evaluation.py` 验证果实分组与重复划分逻辑。以上共 20 个文件组成论文实验、数据准备和测试的相关模块。实验输入包括上游数据、标注、测量元数据及配对光谱缓存。

## 扩展分析与替代入口：24 个文件

以下模块提供物理模型、相机内建模、特征选择、嵌套选参及其他扩展分析：

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

`inter_domain_final.py` 提供 Raw、Scale、CORAL 和 Pair Ridge 四种设置，六方法比较使用 `inter_transfer_baselines.py`。`inter_nested_*` 提供嵌套选参实验入口；论文主实验使用固定参数方案。

## 固定版本结果 CSV

61 个文件对应 **10 类实验 × 6 个配对子集 + 1 张主结果汇总表**，总计 1,638,047 字节，约 1.56 MiB。每个配对子集包含两个传递方向，合计 12 个方向。

各类缓存保留逐折、逐重复、预算子抽样或置换层级的指标。记录层级由 `aggregate` 和对应实验字段标识，可用于核对均值、SD、作图数据及差值。主汇总表提供六种设置的完整比较。

`frozen_results_manifest.csv` 记录各文件的 SHA-256。新实验结果可输出到 `runs/` 下的独立目录，便于与固定版本结果比较。

3 个绘图脚本位于 `JFE_submission/figures/`，结果核验入口为 `tools/check_frozen_results.py`。具体命令见仓库首页。
