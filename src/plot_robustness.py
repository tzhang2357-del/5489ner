"""Render CRF sampling variability and the document-bootstrap estimates."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def records(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    folder = p.parse_args().results
    draws = records(folder/"subset_scores.csv")
    intervals = records(folder/"paired_document_intervals.csv")
    seeds = [int(r["seed"]) for r in draws]
    scores = [100*float(r["micro"]) for r in draws]
    fig,(left,right) = plt.subplots(1,2, figsize=(9.6,3.8), layout="constrained")
    left.scatter(seeds,scores,color="#17637a")
    left.axhline(np.mean(scores),linestyle="--",linewidth=1,color="#555555",label="Ten-draw mean")
    left.set(xlabel="Training subset seed",ylabel="Validation micro-F1 (%)",title="CRF trained on 1,404 sentences")
    left.set_xticks(seeds)
    left.legend(frameon=False,fontsize=9)
    right.axvline(0,color="#555555",linestyle="--",linewidth=1)
    for i,r in enumerate(intervals):
        value,lo,hi = (float(r[key]) for key in ("delta_points","lower","upper"))
        right.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt="o",capsize=5,color="#17637a")
    right.set_yticks([0,1],["Overall","ORG"])
    right.set(xlabel="Validation F1 change (points)",title="Adding word clusters",ylim=(-.6,1.6))
    right.text(.5,.03,"95% paired document bootstrap intervals",transform=right.transAxes,ha="center",fontsize=9)
    for ax in (left,right):
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(folder/"robustness.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
