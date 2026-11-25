"""Plot four verified experiment scores from finalize_results.py output."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    data = json.loads(args.scores.read_text())
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for offset, model, keys, color in [
        (-0.18, "CRF", ("E1", "E3"), "#17637a"),
        (0.18, "BERT", ("E2", "E4"), "#bc713a"),
    ]:
        scores = [100 * data[k]["strict_iob2"]["micro avg"]["f1-score"] for k in keys]
        bars = ax.bar([offset, 1 + offset], scores, 0.34, label=model, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xticks([0, 1], ["Full training set (14,041)", "Shared 10% (1,404)"])
    ax.set_ylim(0, 103)
    ax.set_ylabel("Test entity micro-F1 (%)")
    ax.set_title("CoNLL-2003 English named entity recognition")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.28), frameon=False)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
