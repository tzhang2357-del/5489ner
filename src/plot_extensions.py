"""Render grouped-CV and ablation results, and describe training entity lengths."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ner_common import ENTITY_TYPES, entity_spans, load_split


def plot_results(folder):
    with (folder / "cv.csv").open() as handle:
        cv = list(csv.DictReader(handle))
    with (folder / "ablation.csv").open() as handle:
        ab = list(csv.DictReader(handle))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for i,row in enumerate(cv):
        values = [100*float(row[f"fold{j}"]) for j in (1,2,3)]
        ax.scatter(np.array([-.09,0,.09])+i, values, color=["#567abd","#519283","#ca8851"], s=45)
        ax.plot([i-.2,i+.2],[np.mean(values)]*2,color="#182730")
    ax.set_xticks(range(3), [f"({r['c1']}, {r['c2']})" for r in cv], fontsize=9)
    ax.set(xlabel="Regularization (c1, c2)", ylabel="Held-out micro-F1 (%)", title="Document-grouped cross-validation")
    ax.text(0.01, 0.02, "Dots: folds    Bar: mean", transform=ax.transAxes, fontsize=9)
    for name,label,color in [("micro_f1","Overall","#17637a"),("ORG","Organizations","#bc713a")]:
        vals = [100*float(r[name]) for r in ab]
        bx.plot(range(4),vals,marker="o",color=color,label=label)
        for i,v in enumerate(vals):
            bx.annotate(f"{v:.1f}",(i,v),xytext=(0,7),textcoords="offset points",ha="center",fontsize=9)
    bx.margins(y=.22)
    bx.legend(loc="lower right",fontsize=8)
    bx.set_xticks(range(4),["Word","+ shape","+ context","+ cluster"])
    bx.set(title="Official validation set",ylabel="Entity F1 (%)")
    for a in (ax,bx):
        a.spines[["top","right"]].set_visible(False)
        a.grid(axis="y",alpha=.16)
    fig.savefig(folder/"cv_ablation.png",dpi=180)
    plt.close(fig)


def plot_data(train_file, folder):
    counts = np.zeros((4,3), dtype=int)
    for row in load_split(train_file):
        for start,end,kind in entity_spans(row["gold_tags"]):
            counts[ENTITY_TYPES.index(kind),min(end-start,3)-1] += 1
    with (folder/"training_entity_lengths.csv").open("w", newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["type","one_word","two_words","three_plus"])
        writer.writerows([[kind,*row] for kind,row in zip(ENTITY_TYPES,counts)])
    fig, ax=plt.subplots(figsize=(6.5,3.4),layout="constrained")
    pos=np.arange(4)
    left=np.zeros(4)
    for i,(label,color) in enumerate([("1 word","#17637a"),("2 words","#64a99d"),("3+ words","#d99655")]):
        ax.barh(pos,counts[:,i],left=left,label=label,color=color)
        left+=counts[:,i]
    ax.set_yticks(pos,ENTITY_TYPES)
    ax.set(title="Gold entity mentions in the training split",xlabel="Number of mentions")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(ncol=3,loc="lower center",bbox_to_anchor=(.5,-.4),frameon=False)
    fig.savefig(folder/"training_entity_lengths.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("training_file",type=Path)
    p.add_argument("folder",type=Path)
    args=p.parse_args()
    plot_data(args.training_file,args.folder)
    plot_results(args.folder)
