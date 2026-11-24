# 英文命名实体识别（NER）｜CRF 与 BERT 比较

## 1. 项目简介

基于 CoNLL-2003 英文数据集，训练命名实体识别系统，并比较：
- **CRF 基线**（线性链条件随机场）
- **BERT 微调**（`google-bert/bert-base-cased`）

在 **100% 全量**与 **10% 少量**两种训练数据条件下的实体级性能差异。

---

## 2. 数据获取

本项目使用 [eriktks/conll2003](https://huggingface.co/datasets/eriktks/conll2003) 的公开 Parquet 文件。

将以下三个文件下载至 `data/raw/`：

| 文件名 | 句数 |
|---|---|
| `conll2003-train.parquet` | 14,041 |
| `conll2003-validation.parquet` | 3,250 |
| `conll2003-test.parquet` | 3,453 |

下载方式（任选其一）：

```bash
# 方式一：huggingface hub CLI
pip install huggingface-hub
huggingface-cli download eriktks/conll2003 \
  --repo-type dataset \
  --local-dir data/raw/

# 方式二：Python 脚本
python -c "
from datasets import load_dataset
ds = load_dataset('eriktks/conll2003')
for split in ['train', 'validation', 'test']:
    ds[split].to_parquet(f'data/raw/conll2003-{split}.parquet')
print('Done.')
"
```

---

## 3. 环境安装

```bash
# 基础依赖（CRF + 评估）
pip install -r requirements.txt

# 若需运行 BERT（C 负责）
pip install torch transformers accelerate
```

依赖版本见 `requirements.txt`。

---

## 4. 快速开始

### 4.1 生成 10% 训练子集 ID（A 负责，第 1 天完成）

```bash
python src/data.py subset --seed 42 --frac 0.10
# 输出 → data/train_10pct_ids.json
```

### 4.2 数据质量检查

```bash
python src/data.py check
```

### 4.3 预览样本

```bash
python src/data.py show test -n 3
```

### 4.4 运行预测（需先完成模型训练）

```bash
# CRF
python src/predict.py "I study at City University of Hong Kong." --model crf

# BERT
python src/predict.py "Barack Obama was born in Hawaii." --model bert

# JSON 输出
python src/predict.py "..." --model crf --json
```

### 4.5 评估预测结果

预测文件格式（JSON）：
```json
[
  {"id": "0", "tokens": ["I", "study", "at", ...], "pred_tags": ["O", "O", "O", ...]}
]
```

运行评估：
```bash
python src/evaluate.py eval results/predictions/crf_full_test.json \
  --name E1_CRF_100pct \
  --save

# 对比两个模型的分歧样本
python src/evaluate.py diff \
  results/predictions/crf_full_test.json \
  results/predictions/bert_full_test.json \
  --n 20
```

### 4.6 测试集实体统计

```bash
python src/evaluate.py gold-stats
```

---

## 5. 目录结构

```
ner-project/
├── README.md                   ← 本文件
├── requirements.txt
├── plan.md                     ← 组内工作计划
├── data/
│   ├── raw/                    ← Parquet 文件放此处
│   └── train_10pct_ids.json   ← 10% 子集 ID（自动生成）
├── configs/
│   └── crf_feature_fn.py      ← B 提供 CRF 特征函数（可选）
├── src/
│   ├── data.py                 ← A：数据读取、标签映射、抽样
│   ├── spans.py                ← A：BIO 解码、无效 I-X 修正
│   ├── evaluate.py             ← A：统一评估、指标输出、分歧抽取
│   ├── predict.py              ← A：预测入口，支持 CRF/BERT 切换
│   ├── train_crf.py            ← B：CRF 训练
│   └── train_bert.py           ← C：BERT 微调
├── models/                     ← 训练产物（大文件不入 git）
│   ├── crf_full.joblib
│   ├── crf_10pct.joblib
│   └── bert_full/
├── results/
│   ├── predictions/            ← 逐句预测 JSON
│   ├── metrics.csv             ← 四组实验指标汇总
│   └── error_cases.csv         ← 分歧案例表
└── report/
```

---

## 6. 成员分工（对应 plan.md 第 10 节）

| 成员 | 主责 | 关键文件 |
|---|---|---|
| A（数据与集成） | 数据、抽样、BIO 解码、评估、预测入口、合稿 | `data.py` `spans.py` `evaluate.py` `predict.py` |
| B（CRF 与分析） | CRF 特征与训练（E1/E3）、错误分析 | `train_crf.py`、`configs/crf_feature_fn.py` |
| C（BERT 与复现） | 子词对齐、BERT 微调（E2/E4）、训练环境 | `train_bert.py` |

---

## 7. 四组核心实验（对应 plan.md 第 8 节）

| 编号 | 模型 | 训练数据 | 负责人 |
|---|---|---|---|
| E1 | CRF | 100% | B |
| E2 | BERT | 100% | C |
| E3 | CRF | 固定 10%（seed=42） | B |
| E4 | BERT | 同一 10%（seed=42） | C |

---

## 8. 接口约定

**B/C 训练完成后，预测文件格式（JSON list）**：
```json
{
  "id": "sample_id_string",
  "tokens": ["word1", "word2", ...],
  "pred_tags": ["B-PER", "I-PER", "O", ...]
}
```
长度与 tokens 一致，BIO 标签使用组内统一 LABEL_LIST（见 `src/data.py`）。

**B 的 CRF 模型**：保存为 `.joblib`，路径约定 `models/crf_full.joblib`、`models/crf_10pct.joblib`。

**C 的 BERT 模型**：保存为 HuggingFace 模型目录，路径约定 `models/bert_full/`、`models/bert_10pct/`。

---

## 9. 关键定义

- **严格匹配**：(样本 ID, 起始词索引, 结束词索引, 类型) 完全一致，结束索引右开。
- **无效 I-X 修正**：I-X 在句首或前词非同类时，视为 B-X。
- **micro-F1**：汇总全部类型 TP/FP/FN 后计算 F1。
- **10% 子集**：seed=42 按句无放回抽样，共 1,404 句，ID 存于 `data/train_10pct_ids.json`。
