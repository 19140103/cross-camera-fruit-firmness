# 跨相机果实硬度预测的光谱校准效果评估

Evaluation of spectral calibration for cross-camera fruit firmness prediction

本仓库保存基于公开 **DeepHS-Fruit v2** 数据的实验代码、冻结指标 CSV、绘图代码及运行说明。研究问题是：更换高光谱相机后，原有果实硬度预测模型能否继续使用，配对光谱校准在什么条件下有帮助。

**Pair Ridge 是参考光谱校准方法，不是新提出的硬度预测模型。** 主实验用 Elastic Net 建立源相机硬度预测模型，用训练果实的配对光谱拟合目标到源相机的岭回归映射。PLSR 用于源模型敏感性分析。

## 内容与当前状态

- `physfruit/`：44 个研究代码文件，包含主实验、数据处理、已有测试及历史探索模块。
- `baseline_cache/`：61 个冻结结果 CSV；不包含原始图像或配对光谱 NPZ。
- `JFE_submission/figures/`：3 个只读取冻结 CSV 的绘图脚本；目录名保留原项目结构，未包含论文。
- `tools/check_frozen_results.py`：核验 CSV 哈希、主表汇总与主要计数，不训练模型。
- `docs/frozen_results_manifest.csv`：冻结结果的 SHA-256 清单。
- `docs/source_manifest.csv`：来源代码与上传代码的 SHA-256 对照。
- `docs/reproduction.md`：数据准备、实验入口、统计口径和复现范围。

这是代码与冻结结果的归档版本。未包含论文草稿、作者个人资料、原始高光谱数据、第三方代码副本或模型权重。此次归档未重跑模型，也未修改原项目中的冻结结果。完整原始数据重建和干净环境安装复现尚未完成。

## 快速核验：不需要原始数据

已记录的实验环境为 Python 3.11.8。以下命令均在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_frozen_results.py
python -m pytest physfruit/test_evaluation.py -q
```

`requirements.txt` 记录本地核验环境的版本；它不代表已验证所有平台均可安装。原始数据处理和历史物理模型另需要 PyTorch，见 `requirements-raw.txt`。

冻结主表包含 5 类水果、12 个有向相机组合、6 种设置。按未四舍五入的平均 R² 比较：

- Pair Ridge 在 **10/12** 个方向上高于 Raw；在 **6/12** 个方向上为正。
- Pair Ridge 在 **4/12** 个方向上取得六种设置中的最高均值：柿 VIS→VIS_COR、柿 VIS_COR→VIS、猕猴桃 NIR→VIS、芒果 VIS_COR→VIS。
- 六种设置为 Raw、Scale、CORAL、Pair Ridge、PDS(w=1)、PDS(w=2)。均值最高不等于统计显著或预测可用。

这些结果限于所考察的数据集、果种、相机组合与实验设置。光谱差异下降不足以单独证明硬度预测模型传递成功。

## 从冻结 CSV 重绘

以下脚本只汇总已有 CSV，输出到 `outputs/`，不拟合模型：

```bash
python JFE_submission/figures/_figure3_revised.py --output-dir outputs/figure3
python JFE_submission/figures/_figure4_revised.py --output-dir outputs/figure4
python JFE_submission/figures/_figure5_revised.py --output-dir outputs/figure5
```

文件名保留原英文稿编号，内容依次是：光谱差异与绝对 R²、校准预算与后期留出、PLSR 敏感性与配对身份差值。中文改投稿对应图 2、3、4；本仓库保存的是英文标签的数据图脚本，不是中文 Word 排版工具。每个脚本同时导出作图数据供核验。

## 数据来源与使用范围

上游项目：[cogsys-tuebingen/deephs_fruit](https://github.com/cogsys-tuebingen/deephs_fruit)。数据入口：[DeepHS-Fruit v2](https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/)。本地审计时上游提交为 `5f47812a3fb5bd8f0ed5367cedf9b464f83076c1`。

原始数据、第三方代码和 PROSPECT 系数表应从各自来源获取并遵守其使用条款。本仓库没有附加开源许可证；公开发布与授权方式另行确定。历史物理模型的数据来源见 `physfruit/data/SOURCE.md`，相应系数文件未随仓库分发。
