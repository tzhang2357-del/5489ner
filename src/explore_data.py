"""
src/explore_data.py
数据集探索与可视化：实体类型分布、句子长度分布、实体长度分布。
负责成员：A

用法:
    python src/explore_data.py

输出:
    results/data_exploration/entity_type_distribution.png
    results/data_exploration/sentence_length_distribution.png
    results/data_exploration/stats_by_split.csv
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import ENTITY_TYPES, load_split  # noqa: E402
from spans import bio_to_spans  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "data_exploration"
SPLITS = ["train", "validation", "test"]
COLORS = {"train": "#4878cf", "validation": "#6acc65", "test": "#d65f5f"}


def collect():
    """遍历三个划分，统计实体类型 / 句长 / 实体长度。"""
    per_type, sent_lens, ent_lens, rows = {}, {}, {}, []
    for split in SPLITS:
        samples = load_split(split)
        type_counts = {t: 0 for t in ENTITY_TYPES}
        lengths, ent_lengths = [], []
        for s in samples:
            lengths.append(s.n_tokens())
            for _start, end, etype in bio_to_spans(s.bio_tags):
                type_counts[etype] += 1
                ent_lengths.append(end - _start)

        per_type[split] = type_counts
        sent_lens[split] = lengths
        ent_lens[split] = ent_lengths
        rows.append({
            "split": split,
            "sentences": len(samples),
            "tokens": int(sum(lengths)),
            "entities": int(sum(type_counts.values())),
            **{f"n_{t}": type_counts[t] for t in ENTITY_TYPES},
            "mean_sent_len": round(float(np.mean(lengths)), 2),
            "max_sent_len": int(max(lengths)),
            "mean_ent_len": round(float(np.mean(ent_lengths)), 2),
            "pct_ent_ge3": round(
                100.0 * sum(1 for x in ent_lengths if x >= 3) / len(ent_lengths), 2
            ),
        })
        print(f"[explore] {split}: {len(samples)} sentences, "
              f"{sum(type_counts.values())} entities")
    return per_type, sent_lens, ent_lens, pd.DataFrame(rows)


def plot_type_distribution(per_type):
    """Figure: entity counts by type and split (grouped bars)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(ENTITY_TYPES))
    width = 0.26
    for i, split in enumerate(SPLITS):
        counts = [per_type[split][t] for t in ENTITY_TYPES]
        bars = ax.bar(x + (i - 1) * width, counts, width,
                      label=split, color=COLORS[split])
        for b in bars:
            ax.annotate(f"{int(b.get_height()):,}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(ENTITY_TYPES)
    ax.set_ylabel("Gold entity count")
    ax.set_title("Entity type distribution by split (CoNLL-2003)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "entity_type_distribution.png", dpi=200)
    plt.close(fig)


def plot_length_distributions(sent_lens, ent_lens):
    """Figure: sentence length histogram + entity length distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    ax = axes[0]
    bins = np.arange(0, 121, 5)
    for split in SPLITS:
        ax.hist(np.clip(sent_lens[split], 0, 120), bins=bins, alpha=0.55,
                label=split, color=COLORS[split])
    ax.set_xlabel("Tokens per sentence (clipped at 120)")
    ax.set_ylabel("Sentence count")
    ax.set_title("Sentence length distribution")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    max_len = 6
    x = np.arange(1, max_len + 1)
    width = 0.26
    for i, split in enumerate(SPLITS):
        counts = [sum(1 for v in ent_lens[split] if min(v, max_len) == k)
                  for k in x]
        ax.bar(x + (i - 1) * width, counts, width,
               label=split, color=COLORS[split])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in range(1, max_len)] + [f"{max_len}+"])
    ax.set_xlabel("Entity length (words)")
    ax.set_ylabel("Entity count (log scale)")
    ax.set_title("Gold entity length distribution")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sentence_length_distribution.png", dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_type, sent_lens, ent_lens, stats = collect()
    plot_type_distribution(per_type)
    plot_length_distributions(sent_lens, ent_lens)
    stats.to_csv(OUT_DIR / "stats_by_split.csv", index=False)
    print(f"[explore] figures + stats written to {OUT_DIR}")


if __name__ == "__main__":
    main()