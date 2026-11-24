"""
src/evaluate.py
统一实体级评估：micro P/R/F1、分类型 F1、按长度召回率。
负责成员：A

接口约定（对应 plan.md 5.4）：
- 预测文件格式：JSON 列表，每条 {"id": ..., "tokens": [...], "pred_tags": [...]}
- Gold 来源：data.py 的 load_split()
- 严格匹配：(样本ID, 起始词索引, 结束词索引, 类型)，结束索引右开
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# 相对导入支持（脚本运行时需保证 src/ 在 sys.path 中）
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data import load_split, Sample, ENTITY_TYPES, ids_to_bio
from spans import bio_to_spans, fix_bio_sequence, Span

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_FILE = RESULTS_DIR / "metrics.csv"


# ---------------------------------------------------------------------------
# 工具：安全计算 Precision / Recall / F1
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """由 TP/FP/FN 计算 P、R、F1。"""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


# ---------------------------------------------------------------------------
# 单句匹配：gold spans vs pred spans
# ---------------------------------------------------------------------------

def _match_spans(
    gold_spans: List[Span],
    pred_spans: List[Span],
) -> Tuple[int, int, int]:
    """
    严格匹配单句中的 gold 与 pred spans。
    返回 (tp, fp, fn)。
    """
    gold_set = set(gold_spans)
    pred_set = set(pred_spans)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# 总体评估：汇总所有句子
# ---------------------------------------------------------------------------

def evaluate_all(
    gold_samples: List[Sample],
    pred_tag_lists: List[List[str]],
) -> Dict:
    """
    对全部样本计算实体级指标。

    Parameters
    ----------
    gold_samples : List[Sample]
        从 data.py 加载的标注样本，顺序与 pred_tag_lists 一致。
    pred_tag_lists : List[List[str]]
        每个元素是一条样本的 BIO 预测序列（pred_tags），
        长度应与 gold_samples 相同。

    Returns
    -------
    dict
        {
          'overall': {'P': ..., 'R': ..., 'F1': ..., 'TP': ..., 'FP': ..., 'FN': ...},
          'per_type': {'PER': {...}, 'ORG': {...}, 'LOC': {...}, 'MISC': {...}},
          'length_recall': {'1': ..., '2': ..., '3+': ...},
          'n_sentences': int,
        }
    """
    assert len(gold_samples) == len(pred_tag_lists), (
        f"gold ({len(gold_samples)}) 与 pred ({len(pred_tag_lists)}) 数量不一致"
    )

    # 累计计数器
    total_tp = total_fp = total_fn = 0
    type_counts: Dict[str, Dict[str, int]] = {
        t: {"tp": 0, "fp": 0, "fn": 0} for t in ENTITY_TYPES
    }

    # 按实体词数统计召回：{word_count_bucket: [gold_n, matched_n]}
    length_stats: Dict[str, List[int]] = {
        "1": [0, 0],
        "2": [0, 0],
        "3+": [0, 0],
    }

    def bucket(n_words: int) -> str:
        if n_words == 1:
            return "1"
        elif n_words == 2:
            return "2"
        else:
            return "3+"

    for gold, pred_tags in zip(gold_samples, pred_tag_lists):
        # 修正预测中的无效 I-X
        pred_tags_fixed = fix_bio_sequence(pred_tags)

        g_spans = bio_to_spans(gold.bio_tags, auto_fix=False)
        p_spans = bio_to_spans(pred_tags_fixed, auto_fix=False)

        # --- 总体 ---
        tp, fp, fn = _match_spans(g_spans, p_spans)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # --- 分类型 ---
        g_by_type: Dict[str, set] = defaultdict(set)
        p_by_type: Dict[str, set] = defaultdict(set)
        for s in g_spans:
            g_by_type[s[2]].add(s)
        for s in p_spans:
            p_by_type[s[2]].add(s)

        all_types = set(g_by_type.keys()) | set(p_by_type.keys())
        for etype in all_types:
            gs = g_by_type.get(etype, set())
            ps = p_by_type.get(etype, set())
            t_tp = len(gs & ps)
            t_fp = len(ps - gs)
            t_fn = len(gs - ps)
            if etype in type_counts:
                type_counts[etype]["tp"] += t_tp
                type_counts[etype]["fp"] += t_fp
                type_counts[etype]["fn"] += t_fn

        # --- 按长度召回率 ---
        matched_spans = set(g_spans) & set(p_spans)
        for g in g_spans:
            n_words = g[1] - g[0]
            b = bucket(n_words)
            length_stats[b][0] += 1   # gold 实体数
            if g in matched_spans:
                length_stats[b][1] += 1  # 命中数

    # 汇总输出
    p, r, f1 = _prf(total_tp, total_fp, total_fn)
    overall = {"P": p, "R": r, "F1": f1, "TP": total_tp, "FP": total_fp, "FN": total_fn}

    per_type = {}
    for etype in ENTITY_TYPES:
        c = type_counts[etype]
        tp_, r_, f1_ = _prf(c["tp"], c["fp"], c["fn"])
        per_type[etype] = {
            "P": tp_, "R": r_, "F1": f1_,
            "n_gold": c["tp"] + c["fn"],
        }

    length_recall = {}
    for b, (n_gold, n_hit) in length_stats.items():
        length_recall[b] = round(n_hit / n_gold, 4) if n_gold > 0 else None

    return {
        "overall": overall,
        "per_type": per_type,
        "length_recall": length_recall,
        "n_sentences": len(gold_samples),
    }


# ---------------------------------------------------------------------------
# seqeval 交叉核对（对应 plan.md 6.1）
# ---------------------------------------------------------------------------

def seqeval_check(
    gold_samples: List[Sample],
    pred_tag_lists: List[List[str]],
) -> Dict:
    """
    使用 seqeval 的 IOB2 严格模式对结果进行交叉核对。
    返回 seqeval 计算的 classification_report 字典。
    """
    try:
        from seqeval.metrics import classification_report, f1_score
        from seqeval.scheme import IOB2
    except ImportError:
        print("[evaluate] seqeval 未安装，跳过交叉核对。pip install seqeval")
        return {}

    y_true_seq = [s.bio_tags for s in gold_samples]
    y_pred_seq = [fix_bio_sequence(t) for t in pred_tag_lists]

    report = classification_report(
        y_true_seq, y_pred_seq,
        mode="strict",
        output_dict=True,
    )
    micro_f1 = f1_score(y_true_seq, y_pred_seq, mode="strict", average="micro")
    report["_micro_f1_seqeval"] = round(micro_f1, 4)
    return report


# ---------------------------------------------------------------------------
# 从预测文件加载并评估
# ---------------------------------------------------------------------------

def load_predictions(pred_file: Path) -> Dict[str, List[str]]:
    """
    加载预测文件，返回 {sample_id: pred_tags} 字典。

    支持两种格式（与 B/C 的输出保持兼容）：
    - JSONL（.jsonl）：每行一个 JSON 对象
    - JSON 数组（.json）：整体为一个 JSON 列表
    每条记录字段：{"id": ..., "tokens": [...], "pred_tags": [...]}
    """
    text = Path(pred_file).read_text(encoding="utf-8-sig")

    # 判断格式：以 '[' 开头的是 JSON 数组，否则按 JSONL 逐行解析
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    result: Dict[str, List[str]] = {}
    for rec in records:
        sid = str(rec["id"])
        tags = rec["pred_tags"]
        # 若 pred_tags 是整数列表，转换为 BIO 字符串
        if tags and isinstance(tags[0], int):
            tags = ids_to_bio(tags)
        result[sid] = tags

    return result


def evaluate_prediction_file(
    pred_file: Path,
    split: str = "test",
    run_seqeval: bool = True,
) -> Dict:
    """
    读取预测文件，与指定划分的 gold 对齐后评估。

    Parameters
    ----------
    pred_file : Path
        预测文件路径。
    split : str
        gold 来源，默认 test。
    run_seqeval : bool
        是否执行 seqeval 交叉核对。

    Returns
    -------
    dict
        evaluate_all() 的输出，可附带 seqeval 结果。
    """
    pred_map = load_predictions(pred_file)
    gold_samples = load_split(split)

    # 按 gold 顺序对齐预测，缺失的视为全 O
    pred_tags_list: List[List[str]] = []
    matched = 0
    for s in gold_samples:
        if s.id in pred_map:
            pred_tags_list.append(pred_map[s.id])
            matched += 1
        else:
            pred_tags_list.append(["O"] * len(s.tokens))

    print(f"[evaluate] 预测覆盖: {matched}/{len(gold_samples)} 句，"
          f"缺失 {len(gold_samples) - matched} 句（按全 O 处理）")

    results = evaluate_all(gold_samples, pred_tags_list)

    if run_seqeval:
        seq = seqeval_check(gold_samples, pred_tags_list)
        results["seqeval"] = seq

    return results


# ---------------------------------------------------------------------------
# 结果写入 CSV
# ---------------------------------------------------------------------------

def save_metrics_to_csv(experiment_name: str, metrics: Dict) -> None:
    """
    将一次实验的指标追加写入 results/metrics.csv。
    字段：experiment, n_sentences, P, R, F1, P_PER, R_PER, F1_PER, ...
    """
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = METRICS_FILE.exists()

    row: Dict[str, object] = {
        "experiment": experiment_name,
        "n_sentences": metrics["n_sentences"],
        "P": metrics["overall"]["P"],
        "R": metrics["overall"]["R"],
        "F1": metrics["overall"]["F1"],
        "TP": metrics["overall"]["TP"],
        "FP": metrics["overall"]["FP"],
        "FN": metrics["overall"]["FN"],
    }

    # 分类型
    for etype in ENTITY_TYPES:
        for k in ["P", "R", "F1"]:
            row[f"{k}_{etype}"] = metrics["per_type"][etype][k]

    # 按长度召回
    for b in ["1", "2", "3+"]:
        row[f"recall_{b}"] = metrics["length_recall"].get(b)

    fieldnames = list(row.keys())

    with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[evaluate] 指标已追加保存至 {METRICS_FILE}")


# ---------------------------------------------------------------------------
# 打印格式化的结果摘要
# ---------------------------------------------------------------------------

def print_summary(name: str, metrics: Dict) -> None:
    """打印人类可读的实验结果摘要。"""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  实验: {name}  |  测试句数: {metrics['n_sentences']}")
    print(sep)

    o = metrics["overall"]
    print(f"\n【总体实体级指标】")
    print(f"  Precision : {o['P']:.4f}")
    print(f"  Recall    : {o['R']:.4f}")
    print(f"  micro-F1  : {o['F1']:.4f}")
    print(f"  TP/FP/FN  : {o['TP']}/{o['FP']}/{o['FN']}")

    print(f"\n【分类型 F1】")
    print(f"  {'类型':<6} {'P':>6} {'R':>6} {'F1':>6} {'n_gold':>7}")
    print(f"  {'-'*36}")
    for etype in ENTITY_TYPES:
        t = metrics["per_type"][etype]
        print(f"  {etype:<6} {t['P']:>6.4f} {t['R']:>6.4f} "
              f"{t['F1']:>6.4f} {t['n_gold']:>7}")

    print(f"\n【按实体词数召回率】")
    lr = metrics["length_recall"]
    for b in ["1", "2", "3+"]:
        val = lr.get(b)
        print(f"  {b:>3} 词: {val if val is not None else 'N/A (无样本)'}")

    if "seqeval" in metrics and metrics["seqeval"]:
        seq_f1 = metrics["seqeval"].get("_micro_f1_seqeval", "N/A")
        print(f"\n【seqeval 交叉核对】micro-F1 = {seq_f1}")

    print(sep)


# ---------------------------------------------------------------------------
# 分歧样本抽取（对应 plan.md 6.2，供 B 使用）
# ---------------------------------------------------------------------------

def extract_disagreements(
    gold_samples: List[Sample],
    crf_preds: List[List[str]],
    bert_preds: List[List[str]],
    seed: int = 42,
    n: int = 20,
) -> List[Dict]:
    """
    从 CRF 与 BERT 预测有分歧的测试句中抽取 n 句。
    返回记录列表，供 error_cases.csv 写入。
    """
    import random

    candidates = []
    for gold, crf, bert in zip(gold_samples, crf_preds, bert_preds):
        crf_spans = set(bio_to_spans(fix_bio_sequence(crf)))
        bert_spans = set(bio_to_spans(fix_bio_sequence(bert)))
        if crf_spans != bert_spans:
            candidates.append({
                "id": gold.id,
                "tokens": gold.tokens,
                "gold_tags": gold.bio_tags,
                "crf_tags": crf,
                "bert_tags": bert,
            })

    if len(candidates) <= n:
        selected = candidates
    else:
        rng = random.Random(seed)
        selected = rng.sample(candidates, n)

    # 为每条记录补充 spans
    for rec in selected:
        rec["gold_entities"] = bio_to_spans(rec["gold_tags"])
        rec["crf_entities"] = bio_to_spans(fix_bio_sequence(rec["crf_tags"]))
        rec["bert_entities"] = bio_to_spans(fix_bio_sequence(rec["bert_tags"]))

    return selected


def save_error_cases(cases: List[Dict], out_file: Optional[Path] = None) -> None:
    """将分歧案例写入 results/error_cases.csv。"""
    out_file = out_file or (RESULTS_DIR / "error_cases.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not cases:
        print("[evaluate] 无分歧样本，跳过写入。")
        return

    rows = []
    for c in cases:
        rows.append({
            "id": c["id"],
            "sentence": " ".join(c["tokens"]),
            "gold_entities": str(c["gold_entities"]),
            "crf_entities": str(c["crf_entities"]),
            "bert_entities": str(c["bert_entities"]),
        })

    fieldnames = ["id", "sentence", "gold_entities", "crf_entities", "bert_entities"]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[evaluate] 错误案例已保存至 {out_file}（共 {len(cases)} 条）")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="统一评估工具（A 负责）")
    sub = parser.add_subparsers(dest="cmd")

    # 评估单个预测文件
    p_eval = sub.add_parser("eval", help="评估单个预测文件")
    p_eval.add_argument("pred_file", type=Path, help="预测文件路径（JSON）")
    p_eval.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p_eval.add_argument("--name", default="experiment", help="实验名称（用于 metrics.csv）")
    p_eval.add_argument("--save", action="store_true", help="是否追加写入 metrics.csv")
    p_eval.add_argument("--no-seqeval", action="store_true", help="跳过 seqeval 核对")

    # 对比两个预测文件
    p_diff = sub.add_parser("diff", help="抽取两个模型的分歧样本")
    p_diff.add_argument("crf_pred", type=Path)
    p_diff.add_argument("bert_pred", type=Path)
    p_diff.add_argument("--n", type=int, default=20)
    p_diff.add_argument("--seed", type=int, default=42)

    # 打印 gold 统计
    sub.add_parser("gold-stats", help="打印测试集 gold 实体统计")

    args = parser.parse_args()

    if args.cmd == "eval":
        results = evaluate_prediction_file(
            args.pred_file,
            split=args.split,
            run_seqeval=not args.no_seqeval,
        )
        print_summary(args.name, results)
        if args.save:
            save_metrics_to_csv(args.name, results)

    elif args.cmd == "diff":
        gold = load_split("test")
        crf_map = load_predictions(args.crf_pred)
        bert_map = load_predictions(args.bert_pred)

        crf_list = [crf_map.get(s.id, ["O"] * len(s.tokens)) for s in gold]
        bert_list = [bert_map.get(s.id, ["O"] * len(s.tokens)) for s in gold]

        cases = extract_disagreements(gold, crf_list, bert_list,
                                      seed=args.seed, n=args.n)
        save_error_cases(cases)
        print(f"[evaluate] 抽取了 {len(cases)} 条分歧样本。")

    elif args.cmd == "gold-stats":
        gold = load_split("test")
        from collections import Counter
        type_cnt: Counter = Counter()
        len_cnt: Counter = Counter()
        for s in gold:
            for sp in bio_to_spans(s.bio_tags):
                type_cnt[sp[2]] += 1
                len_cnt[sp[1] - sp[0]] += 1
        print(f"测试集句子数: {len(gold)}")
        print(f"实体总数: {sum(type_cnt.values())}")
        print("\n按类型:")
        for t, n in type_cnt.most_common():
            print(f"  {t}: {n}")
        print("\n按词数:")
        for l in sorted(len_cnt.keys()):
            print(f"  {l} 词: {len_cnt[l]}")

    else:
        parser.print_help()
