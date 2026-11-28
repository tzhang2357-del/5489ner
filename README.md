# 英文命名实体识别（NER）｜CRF 与 BERT 比较

基于 CoNLL-2003 英文数据集，实现 NER 系统并比较 **CRF 基线** 与 **BERT 微调** 在
**100% 全量** 和 **10% 少量** 训练数据下的实体级性能差异。

**状态**：四组实验已完成，全部指标经统一评估器独立复算并与 B/C 报告核对一致。

---

## 1. 结果总览（经 `src/evaluate.py` 统一复算）

| # | 模型 | 训练数据 | P | R | micro-F1 | TP/FP/FN |
|---|---|---|---:|---:|---:|---|
| E1 | CRF | 100% (14,041 句) | 81.75 | 79.57 | **80.65** | 4494/1003/1154 |
| E2 | BERT | 100% (14,041 句) | 90.32 | 92.03 | **91.17** | 5198/557/450 |
| E3 | CRF | 10% (1,404 句) | 74.55 | 68.66 | **71.48** | 3878/1324/1770 |
| E4 | BERT | 10% (1,404 句) | 83.89 | 86.19 | **85.02** | 4868/935/780 |

- **RQ1** 全量条件 BERT − CRF = **+10.52** F1 点
- **RQ2** 数据降至 10%：BERT −6.15 点，CRF −9.17 点 → BERT 数据效率更高
- **RQ3** BERT 优势集中在 ORG（+15.62）与 ≥3 词长实体召回（+17.11）

完整结果与报告见 [report/report_draft.md](report/report_draft.md)。

---

## 2. 数据获取

数据集：[eriktks/conll2003](https://huggingface.co/datasets/eriktks/conll2003)，固定修订版
`e8237f8875e4ecf7f60b07ae49691b85cf893410`，放入 `data/raw/`：

| 文件 | 句数 | SHA-256（前 8 位） |
|---|---|---|
| `conll2003-train.parquet` | 14,041 | `1c14a0d3` |
| `conll2003-validation.parquet` | 3,250 | `48d75b38` |
| `conll2003-test.parquet` | 3,453 | `47156dc0` |

下载方式：

```bash
# 方式一：国内镜像（已验证可用）
# 基址: https://hf-api.gitee.com/datasets/eriktks/conll2003/resolve/e8237f8875e4ecf7f60b07ae49691b85cf893410/conll2003/

# 方式二：标准 Hugging Face（需可访问 huggingface.co）
python -c "
from datasets import load_dataset
ds = load_dataset('eriktks/conll2003')
for split in ['train', 'validation', 'test']:
    ds[split].to_parquet(f'data/raw/conll2003-{split}.parquet')
"
```

> 注意：`hf-mirror.com` 目前会 308 重定向回 huggingface.co，若直连失败可改用
> `hf-api.gitee.com` 镜像直接下载 Parquet。

---

## 3. 环境安装

```bash
pip install -r requirements.txt
# BERT 复现额外需要（C 使用 Colab GPU）：
pip install torch transformers accelerate
```

---

## 4. 快速开始

```bash
# 1) 数据质量检查（句数、标签长度、10% 子集分布）
python src/data.py check

# 2) 预览样本
python src/data.py show test -n 3

# 3) 实体抽取演示（CRF 全量模型，需 models/crf_full.joblib）
python src/predict.py "I study at City University of Hong Kong." --model crf
python src/predict.py "Barack Obama was born in Hawaii." --model crf --json

# 4) 复算任一实验的测试集指标
python src/evaluate.py eval results/predictions/e1_crf_full_test.jsonl --name E1

# 5) 抽取 CRF 与 BERT 的分歧案例（写入 results/error_cases.csv）
python src/evaluate.py diff \
  results/predictions/e1_crf_full_test.jsonl \
  results/predictions/e2_bert_full_test.jsonl --n 20

# 6) 测试集 gold 实体统计
python src/evaluate.py gold-stats
```

演示输出记录见 [results/demo_outputs.txt](results/demo_outputs.txt)（单实体 / 多实体 / 无实体三种场景）。

---

## 5. 目录结构

```
ner-project/
├── README.md
├── requirements.txt
├── plan.md                       组内工作计划
├── data/
│   ├── raw/                      Parquet 数据（不入 git）
│   ├── train_10pct_ids.shared.json   B/C 确认的共享 10% 子集（1,404 句，seed 42）
│   └── train_10pct_ids.proposed.json B 的原始建议稿
├── src/
│   ├── data.py              [A]  数据读取、标签映射、抽样、质量检查
│   ├── spans.py             [A]  BIO 解码、无效 I-X 修正
│   ├── evaluate.py          [A]  统一评估、分类型/分长度指标、分歧抽取
│   ├── predict.py           [A]  预测入口（CRF/BERT 可切换）
│   ├── explore_data.py      [A]  数据探索图（类型/句长/实体长度分布）
│   ├── ner_common.py        [B]  CRF 侧数据加载与指标工具
│   ├── train_crf.py         [B]  CRF 训练与导出
│   ├── analyze_errors.py    [B]  错例对比分析脚本
│   ├── plot_crf.py          [B]  CRF 结果绘图
│   └── train_bert.ipynb     [C]  BERT 微调 notebook（Colab）
├── models/                       训练产物（不入 git）
│   ├── crf_full.joblib            E1 模型
│   ├── crf_10pct.joblib           E3 模型
│   └── bert_full/                 E2 模型（C 通过 GitHub Release `bert-v1` 发布）
├── results/
│   ├── predictions/               四组逐句预测 e1~e4_*_test.jsonl
│   ├── metrics.csv                四组指标汇总（统一复算）
│   ├── error_cases.csv            CRF vs BERT 分歧案例（seed 42 抽 20 条）
│   ├── demo_outputs.txt           演示记录
│   ├── data_exploration/          数据探索图与统计（A）
│   ├── metrics_by_type_bert.csv   C 的分类别指标
│   ├── recall_by_length_bert.csv  C 的长度分组召回
│   ├── final/                     B 汇总的四组最终分数与对比图
│   ├── extensions/                B 的扩展实验（交叉验证/消融/聚类）
│   └── robustness/                B 的 10% 子集抽样稳健性分析
├── deliverables/
│   └── role-b-crf-final.zip       B 的最终 CRF 交付包
└── report/
    ├── report_draft.md            整合报告初稿（A 合稿）
    ├── contributions.md           贡献表模板（待全员确认）
    ├── role_b.md                  B 的英文报告
    └── role_c.md                  C 的英文报告
```

---

## 6. 成员分工与贡献

| 成员 | 主责 | 关键交付物 |
|---|---|---|
| A（数据与集成） | 数据、共享抽样、BIO 解码、统一评估、预测入口、结果整合与合稿 | `data.py` `spans.py` `evaluate.py` `predict.py`、`metrics.csv`、`error_cases.csv`、`report_draft.md` |
| B（CRF 与分析） | CRF 特征与训练（E1/E3）、错例分析 | `train_crf.py` `ner_common.py`、CRF 模型与预测、CRF 报告章节 |
| C（BERT 与复现） | 子词对齐、BERT 微调（E2/E4）、训练环境 | `train_bert.ipynb`、BERT 预测、manifest 与复现说明 |

贡献占比表见报告 Appendix A（待全员确认姓名、学号与占比）。

### 6.1 组员交付物与补充实验

**B 的交付包**：[deliverables/role-b-crf-final.zip](deliverables/role-b-crf-final.zip)，含全量/10% CRF 模型、逐句预测与验证集消融结果；英文报告见 [report/role_b.md](report/role_b.md)，包含交叉验证、消融、分歧案例和稳健性分析。复现命令（装好项目依赖与 matplotlib 后，从仓库根目录执行）：

```bash
python src/extension_experiments.py data/raw/conll2003-train.parquet data/raw/conll2003-validation.parquet results/extensions/train_document_ids.json rerun/extensions
python src/robustness.py data/raw/conll2003-train.parquet data/raw/conll2003-validation.parquet rerun/extensions data/train_10pct_ids.shared.json results/robustness/validation_document_ids.json rerun/robustness
python src/plot_robustness.py rerun/robustness
```

**C 的 BERT 交付**：英文报告见 [report/role_c.md](report/role_c.md)；E2/E4 模型权重通过 GitHub Release 发布：[bert-v1](https://github.com/tzhang2357-del/5489ner/releases/tag/bert-v1)。E2 全量训练，测试 micro-F1 为 91.17%；E4 与 E3 使用同一 10% 训练子集，测试 micro-F1 为 85.02%。下载后解压到 `models/bert_full/` 即可用 `--model bert` 运行预测入口。

---

## 7. 接口约定（组内统一）

- **词级标签接口**：`predict_tags(tokens) -> List[str]`，每个词一个 BIO 标签。
- **预测文件**：JSONL 或 JSON 数组，字段 `{"id", "tokens", "pred_tags"}`；
  C 的 `test_pred_tags.jsonl`（仅 `id` + `pred_tags`）同样被评估器接受。
- **无效 I-X 规则**：`I-X` 在句首或前无同类型开放实体时视为 `B-X`（训练与打分同规则）。
- **严格匹配**：`(样本ID, 起始词, 结束词右开, 类型)` 全一致才算正确。
- **10% 子集**：`data/train_10pct_ids.shared.json`，1,404 句，SHA-256 `3d1748de…a987`，
  两组低资源实验共享同一份 ID。

---

## 8. 复算指引（可追溯性）

| 想复算的内容 | 命令 |
|---|---|
| E1–E4 任一指标 | `python src/evaluate.py eval results/predictions/<file>.jsonl --name <exp>` |
| seqeval 交叉核对 | 同上（默认启用，输出 `micro-F1-seqeval`） |
| 分歧案例 | `python src/evaluate.py diff <crf>.jsonl <bert>.jsonl --n 20` |
| 数据完整性 | `python src/data.py check` |

所有预测文件覆盖 3,453/3,453 句测试样本（评估器会打印覆盖率并断言）。