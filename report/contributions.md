# 小组贡献表（已定稿）

| 姓名 | 学号 | 贡献占比 | 具体工作与可核验产出 |
|---|---|---:|---|
| WANG Chengru（A） | 60254685 | 33% | 数据下载与质量检查；10% 共享子集管理；`explore_data.py` 数据探索图；`spans.py` BIO 解码与无效 I-X 规则；`evaluate.py` 统一评估（四组实验独立复算，与 B/C 报告一致）；`predict.py` 预测入口与三场景演示；`metrics.csv`/`error_cases.csv` 整合；`report_draft.md` 合稿与图表整合 |
| ZHANG Tongwei（B） | 61091750 | 34% | `train_crf.py` CRF 特征/训练/导出（E1：80.65，E3：71.48）；`ner_common.py`；全量 CRF 错例统计与 20 条案例；文档分组交叉验证、特征消融、SVD+k-means 聚类、10% 抽样稳健性扩展实验；CRF 报告章节（Methods / Error Analysis / Additional analyses） |
| Xu Shenghao（C） | 61115726 | 33% | `train_bert.ipynb` BERT 微调与子词对齐（E2：91.17，E4：85.02）；训练 manifest 与日志；BERT 模型发布（Release `bert-v1`）与复现说明；BERT 报告章节 |

合计：33% + 34% + 33% = 100%。

备注：
- 贡献占比已经全员确认，与报告 Appendix A 一致。