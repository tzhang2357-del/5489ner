import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description="Plot test-set F1 from two CRF experiment summaries.")
    parser.add_argument("full_metrics", type=Path)
    parser.add_argument("subset_metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    full_data = json.loads(args.full_metrics.read_text(encoding="utf-8"))
    subset_data = json.loads(args.subset_metrics.read_text(encoding="utf-8"))

    groups = ("micro avg", "PER", "ORG", "LOC", "MISC")
    full = [100 * full_data["test"][group]["f1-score"] for group in groups]
    subset = [100 * subset_data["test"][group]["f1-score"] for group in groups]
    subset_label = "10% training set"
    if subset_data["subset_status"] == "provisional":
        subset_label += " (provisional)"

    figure, axes = plt.subplots(figsize=(8, 4.65), layout="constrained")
    full_bars = axes.bar([index - 0.18 for index in range(len(groups))], full, 0.36, color="#17637A", label="Full training set")
    subset_bars = axes.bar([index + 0.18 for index in range(len(groups))], subset, 0.36, color="#BF7743", label=subset_label)
    axes.bar_label(full_bars, fmt="%.2f%%", padding=3, fontsize=8)
    axes.bar_label(subset_bars, fmt="%.2f%%", padding=3, fontsize=8)
    axes.set_xticks(range(len(groups)), ("Overall (micro)", "PER", "ORG", "LOC", "MISC"))
    axes.set_ylim(0, 100)
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes.set_ylabel("Entity-level F1")
    axes.set_title("CRF performance on the CoNLL-2003 test set", loc="left", fontweight="bold")
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    axes.set_axisbelow(True)
    axes.grid(axis="y", alpha=0.25)
    axes.spines[["top", "right"]].set_visible(False)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
