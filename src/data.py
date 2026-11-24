"""
src/data.py
数据读取、标签映射与 10% 固定抽样。
负责成员：A
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
# 10% 子集 ID 文件候选路径（优先共享版 → 正式版 → 暂定版）
SUBSET_IDS_CANDIDATES = [
    PROJECT_ROOT / "data" / "train_10pct_ids.shared.json",    # B/C 已确认的共享版
    PROJECT_ROOT / "data" / "train_10pct_ids.json",            # A 生成的正式版
    PROJECT_ROOT / "data" / "train_10pct_ids.proposed.json",   # B 的暂定版
]
SUBSET_IDS_FILE = PROJECT_ROOT / "data" / "train_10pct_ids.json"  # A 默认写入路径

# 候选路径列表（优先 data/raw/，再回退到 data/，与 B 的 train_crf.py 保持兼容）
_PARQUET_CANDIDATE_DIRS = [DATA_RAW_DIR, PROJECT_ROOT / "data"]


def _resolve_parquet(split: str) -> Path:
    """在多个候选目录中查找 parquet 文件，返回第一个存在的路径。"""
    filename = f"conll2003-{split}.parquet"
    for d in _PARQUET_CANDIDATE_DIRS:
        p = d / filename
        if p.exists():
            return p
    # 均不存在则返回首选路径（抛出 FileNotFoundError 时信息更清晰）
    return DATA_RAW_DIR / filename


PARQUET_FILES = {
    split: _resolve_parquet(split)
    for split in ["train", "validation", "test"]
}

# ---------------------------------------------------------------------------
# 标签映射（CoNLL-2003 IOB2，9 个标签）
# ---------------------------------------------------------------------------
LABEL_LIST: List[str] = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]

ID2LABEL: Dict[int, str] = {i: lbl for i, lbl in enumerate(LABEL_LIST)}
LABEL2ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_LIST)}

ENTITY_TYPES: List[str] = ["PER", "ORG", "LOC", "MISC"]


# ---------------------------------------------------------------------------
# 工具函数：标签格式转换
# ---------------------------------------------------------------------------

def ids_to_bio(tag_ids: List[int]) -> List[str]:
    """将整数标签 ID 列表转为 BIO 字符串列表。"""
    return [ID2LABEL[t] for t in tag_ids]


def bio_to_ids(bio_tags: List[str]) -> List[int]:
    """将 BIO 字符串列表转为整数标签 ID 列表。"""
    return [LABEL2ID[t] for t in bio_tags]


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class Sample:
    """单条样本的封装。"""

    def __init__(self, sample_id: str, tokens: List[str],
                 ner_tags: List[int], bio_tags: List[str]):
        self.id = sample_id
        self.tokens = tokens          # 原始词序列
        self.ner_tags = ner_tags      # 整数标签序列（与 HuggingFace 对齐）
        self.bio_tags = bio_tags      # BIO 字符串序列（与组内接口对齐）

    def __repr__(self):
        return (f"Sample(id={self.id!r}, n_tokens={len(self.tokens)}, "
                f"n_entities={self.n_entities()})")

    def n_tokens(self) -> int:
        return len(self.tokens)

    def n_entities(self) -> int:
        """统计标注实体数（以 B- 开头计数）。"""
        return sum(1 for t in self.bio_tags if t.startswith("B-"))


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _read_parquet(split: str) -> pd.DataFrame:
    """
    读取指定划分的 Parquet 文件。
    支持字段名：id（或 sample_id）、tokens、ner_tags。
    """
    path = PARQUET_FILES.get(split)
    if path is None:
        raise ValueError(f"未知划分: {split!r}，可选: {list(PARQUET_FILES.keys())}")
    if not path.exists():
        raise FileNotFoundError(
            f"找不到数据文件: {path}\n"
            f"请先将 Parquet 文件放入 data/raw/ 目录，参见 README.md 数据获取步骤。"
        )
    return pd.read_parquet(path)


def load_split(split: str, subset_ids: Optional[set] = None) -> List[Sample]:
    """
    加载指定划分的所有样本。

    Parameters
    ----------
    split : str
        划分名称，'train' / 'validation' / 'test'。
    subset_ids : set, optional
        若提供，只返回 ID 在此集合中的样本（用于加载 10% 子集）。

    Returns
    -------
    List[Sample]
        样本列表，顺序与 Parquet 文件一致（保留原始 ID）。
    """
    df = _read_parquet(split)

    # 兼容不同字段命名
    id_col = "id" if "id" in df.columns else "sample_id"
    samples: List[Sample] = []

    for _, row in df.iterrows():
        sid = str(row[id_col])
        if subset_ids is not None and sid not in subset_ids:
            continue

        tokens = list(row["tokens"])
        ner_tags = list(row["ner_tags"])
        bio_tags = ids_to_bio(ner_tags)

        assert len(tokens) == len(bio_tags), (
            f"样本 {sid} tokens 与标签长度不一致: "
            f"len(tokens)={len(tokens)}, len(tags)={len(bio_tags)}"
        )
        samples.append(Sample(sid, tokens, ner_tags, bio_tags))

    return samples


# ---------------------------------------------------------------------------
# 10% 固定抽样
# ---------------------------------------------------------------------------

def create_10pct_subset(
    seed: int = 42,
    frac: float = 0.10,
    save_path: Optional[Path] = None,
) -> List[str]:
    """
    从训练集按句无放回抽样，生成 10% 子集的 ID 清单并保存为 JSON。

    Parameters
    ----------
    seed : int
        随机种子（组内约定 42）。
    frac : float
        抽样比例（默认 0.10）。
    save_path : Path, optional
        保存路径，默认 data/train_10pct_ids.json。

    Returns
    -------
    List[str]
        被抽中样本的 ID 列表（顺序与抽样一致）。
    """
    df = _read_parquet("train")
    id_col = "id" if "id" in df.columns else "sample_id"
    all_ids = df[id_col].astype(str).tolist()

    n_sample = int(len(all_ids) * frac)
    rng = random.Random(seed)
    subset_ids = rng.sample(all_ids, n_sample)

    save_path = save_path or SUBSET_IDS_FILE
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": seed,
        "frac": frac,
        "total_train": len(all_ids),
        "n_sampled": n_sample,
        "ids": subset_ids,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[data] 10% 子集已生成并保存至 {save_path}")
    print(f"       总训练句: {len(all_ids)}，抽样句数: {n_sample}，seed={seed}")
    return subset_ids


def _read_subset_ids_file(path: Path) -> List[str]:
    """
    读取子集 ID 文件，同时兼容两种格式：
    1. 对象格式（A 的格式）：{"seed":42, "ids":[...]}
    2. 纯列表格式（B/C 的格式）：["6", "9", ...]
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        content = json.load(f)
    if isinstance(content, list):
        return [str(i) for i in content]
    elif isinstance(content, dict) and "ids" in content:
        return [str(i) for i in content["ids"]]
    else:
        raise ValueError(
            f"无法解析子集 ID 文件 {path}："
            f"期望对象格式 {{'ids':[...]}} 或纯列表格式 [...]"
        )


def load_10pct_subset(ids_file: Optional[Path] = None) -> Dict:
    """
    加载已保存的 10% 子集 ID 清单。

    优先加载顺序：
    1. 指定路径 ids_file
    2. data/train_10pct_ids.shared.json（B/C 共享版）
    3. data/train_10pct_ids.json（A 的正式版）
    4. data/train_10pct_ids.proposed.json（B 的暂定版）

    若均不存在则自动从训练集抽样生成。

    Returns
    -------
    dict
        包含 'ids'（ID 列表）及其他元数据字段的字典。
    """
    # 确定加载路径
    if ids_file is not None:
        target = Path(ids_file)
    else:
        target = next((p for p in SUBSET_IDS_CANDIDATES if p.exists()), None)

    if target is None:
        print("[data] 未找到子集 ID 文件，正在自动生成（seed=42, frac=0.10）…")
        create_10pct_subset()
        target = SUBSET_IDS_FILE
    else:
        print(f"[data] 使用子集 ID 文件: {target.name}")

    ids = _read_subset_ids_file(target)

    # 如果文件本身是对象格式，返回完整内容；否则包装为统一字典
    with open(target, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw["ids"] = [str(i) for i in raw["ids"]]  # 统一为字符串
        return raw
    return {
        "source_file": str(target.name),
        "n_sampled": len(ids),
        "ids": ids,
    }


def load_train_subset(seed: int = 42) -> List[Sample]:
    """加载 10% 训练子集的样本对象列表。"""
    subset = load_10pct_subset()
    return load_split("train", subset_ids=set(subset["ids"]))


# ---------------------------------------------------------------------------
# 数据质量检查（对应规划 4.2 检查清单）
# ---------------------------------------------------------------------------

def run_data_checks() -> Dict:
    """
    执行数据质量检查，返回检查结果字典。
    对应 plan.md 4.2 节的检查清单。
    """
    report = {}

    # 1. 样本数量
    for split in ["train", "validation", "test"]:
        df = _read_parquet(split)
        report[f"n_{split}"] = len(df)

    # 2. tokens 与 ner_tags 等长检查
    length_errors: List[str] = []
    for split in ["train", "validation", "test"]:
        for s in load_split(split):
            if len(s.tokens) != len(s.bio_tags):
                length_errors.append(f"{split}/{s.id}")
    report["length_errors"] = length_errors

    # 3. 10% 子集检查
    subset = load_10pct_subset()
    report["subset_n_sampled"] = subset["n_sampled"]
    report["subset_seed"] = subset["seed"]

    # 4. 四类实体数量对比（全量 vs 子集）
    def count_types(samples: List[Sample]) -> Dict[str, int]:
        cnt = {t: 0 for t in ENTITY_TYPES}
        for s in samples:
            for t in s.bio_tags:
                if t.startswith("B-"):
                    etype = t[2:]
                    if etype in cnt:
                        cnt[etype] += 1
        return cnt

    all_train = load_split("train")
    sub_train = load_train_subset()
    report["entity_counts_full"] = count_types(all_train)
    report["entity_counts_subset"] = count_types(sub_train)

    return report


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据准备工具（A 负责）")
    subparsers = parser.add_subparsers(dest="cmd")

    # 生成子集
    p_subset = subparsers.add_parser("subset", help="生成 10% 训练子集 ID 清单")
    p_subset.add_argument("--seed", type=int, default=42)
    p_subset.add_argument("--frac", type=float, default=0.10)

    # 数据检查
    subparsers.add_parser("check", help="执行数据质量检查")

    # 预览样本
    p_show = subparsers.add_parser("show", help="预览指定划分的前 N 条样本")
    p_show.add_argument("split", choices=["train", "validation", "test"])
    p_show.add_argument("-n", type=int, default=3)

    args = parser.parse_args()

    if args.cmd == "subset":
        create_10pct_subset(seed=args.seed, frac=args.frac)

    elif args.cmd == "check":
        import pprint
        result = run_data_checks()
        pprint.pprint(result)

    elif args.cmd == "show":
        samples = load_split(args.split)
        for s in samples[:args.n]:
            print(f"\n--- {s} ---")
            for tok, tag in zip(s.tokens, s.bio_tags):
                print(f"  {tok:20s} {tag}")

    else:
        parser.print_help()
