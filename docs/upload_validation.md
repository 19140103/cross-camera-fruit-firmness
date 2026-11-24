# 验证记录

验证环境：Python 3.11.8，依赖版本见仓库根目录的 requirements 文件。

| 检查项 | 结果 |
| --- | --- |
| Python 语法 | 48 个文件解析通过 |
| 果实分组测试 | 5 项测试通过 |
| 固定版本 CSV | 61 个文件的大小和 SHA-256 一致 |
| 主表汇总 | 72 行 R²、RMSE、MAE 均值及 SD 与逐重复缓存汇总一致 |
| 主要比较计数 | 高于 Raw 10/12、正平均 R² 6/12、六方法最高平均 R² 4/12 |
| 图表生成 | 3 个脚本运行成功，导出 PDF、SVG、600 dpi PNG |
| 作图数据 | 6 个作图数据 CSV 与固定版本记录一致 |

分组测试：

```bash
python -m pytest physfruit/test_evaluation.py -q
```

结果核验：

```bash
python tools/check_frozen_results.py
```

代码及结果文件的对应关系分别记录在 `source_manifest.csv` 和 `frozen_results_manifest.csv` 中。
