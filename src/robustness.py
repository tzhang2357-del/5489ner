"""Sampling sensitivity for the CRF, and paired checks on its cluster feature."""
import argparse
import csv
import json
from pathlib import Path
from random import Random
from statistics import mean, stdev
from tempfile import TemporaryDirectory

import numpy as np

from extension_experiments import cluster_words, fit, save_table, transform
from ner_common import entity_spans, load_split, metrics, read_predictions, write_json
from train_crf import sentence_features, sha256


def score(rows, preds):
    report = metrics([r["gold_tags"] for r in rows], preds)
    return {k: report[name]["f1-score"] for k, name in
            [("micro", "micro avg"), ("ORG", "ORG"), ("PER", "PER"), ("LOC", "LOC"), ("MISC", "MISC")]}


def sample_runs(train, dev, x, dx, shared, settings, out):
    seeds = range(42, 52)
    scores, ids = [], {}
    for seed in seeds:
        indices = sorted(Random(seed).sample(range(len(train)), len(shared)))
        sample_ids = [train[i]["id"] for i in indices]
        if seed == 42 and sample_ids != shared:
            raise ValueError("Seed 42 differs from team subset")
        ids[str(seed)] = sample_ids
        model = fit([x[i] for i in indices], [train[i]["gold_tags"] for i in indices], settings)
        scores.append(dict(seed=seed, **score(dev, model.predict(dx))))
        print("10% draw", seed, scores[-1]["micro"], flush=True)
    save_table(out / "subset_scores.csv", scores)
    write_json(out / "subset_ids.json", ids)
    values = [r["micro"] for r in scores]
    return dict(draws=len(values), mean=mean(values), sd=stdev(values), low=min(values), high=max(values))


def document_bootstrap(dev, doc_ids, ext, out):
    # Counts per document keep all mentions and all sentences of a sampled article together.
    models = [read_predictions(ext / f"validation_{v}.jsonl", dev) for v in "CD"]
    docs = sorted(set(doc_ids.values()))
    by_doc = {doc: np.zeros((2, 2, 3), dtype=int) for doc in docs}
    if set(doc_ids) != {r["id"] for r in dev}:
        raise ValueError("Validation document IDs do not match validation sentences")
    for row in dev:
        sid = row["id"]
        gold = set(entity_spans(row["gold_tags"]))
        for mi, model in enumerate(models):
            pred = set(entity_spans(model[sid]))
            for ti, types in enumerate((None, "ORG")):
                g = gold if types is None else {s for s in gold if s[-1] == types}
                p = pred if types is None else {s for s in pred if s[-1] == types}
                by_doc[doc_ids[sid]][mi, ti] += (len(g & p), len(p), len(g))
    counts = np.stack([by_doc[d] for d in docs])
    # Common resamples for models C and D. Output units are percentage points.
    def delta(total):
        f1 = 200 * total[..., 0] / (total[..., 1] + total[..., 2])
        return f1[..., 1, :] - f1[..., 0, :]
    observed = delta(counts.sum(axis=0))
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(100):
        batch = rng.integers(len(docs), size=(100, len(docs)))
        samples.append(delta(counts[batch].sum(axis=1)))
    limits = np.quantile(np.concatenate(samples), [.025, .975], axis=0)
    result = [dict(metric=name, delta_points=observed[i], lower=limits[0,i], upper=limits[1,i],
                   documents=len(docs), replicates=10000) for i,name in enumerate(("micro","ORG"))]
    save_table(out / "paired_document_intervals.csv", result)


def fold_comparison(train, x, ext, settings, out):
    with (ext / "folds.csv").open() as file:
        fold_list = list(csv.DictReader(file))
    folds = {row["id"]: int(row["held_out_fold"]) for row in fold_list}
    docs = {row["id"]: row["document"] for row in fold_list}
    results = []
    for fold in (1, 2, 3):
        ti = [i for i,r in enumerate(train) if folds[r["id"]] != fold]
        vi = [i for i,r in enumerate(train) if folds[r["id"]] == fold]
        fit_rows = [train[i] for i in ti]
        eval_rows = [train[i] for i in vi]
        assert {docs[r["id"]] for r in fit_rows}.isdisjoint({docs[r["id"]] for r in eval_rows})
        with TemporaryDirectory() as tmp:
            lookup, _ = cluster_words(fit_rows, Path(tmp))
        record = dict(fold=fold, training_sentences=len(ti), held_out_sentences=len(vi), cluster_vocab=len(lookup))
        for variant in "CD":
            tx = transform(fit_rows, [x[i] for i in ti], variant, lookup)
            model = fit(tx, [r["gold_tags"] for r in fit_rows], settings)
            vx = transform(eval_rows, [x[i] for i in vi], variant, lookup)
            result = score(eval_rows, model.predict(vx))
            for metric in ("micro", "ORG"):
                record[f"{variant}_{metric}"] = result[metric]
            del tx, vx
        for metric in ("micro", "ORG"):
            record[f"delta_{metric}_points"] = 100*(record[f"D_{metric}"] - record[f"C_{metric}"])
        results.append(record)
        print("Paired fold", fold, record["delta_micro_points"], flush=True)
    save_table(out / "paired_folds.csv", results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "validation", "extension_folder", "shared_ids", "validation_docs", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.extension_folder / "protocol.json").read_text())
    assert sha256(args.train) == protocol["train_sha256"] and sha256(args.validation) == protocol["validation_sha256"]
    settings = (protocol["c1"], protocol["c2"])
    train, dev = load_split(args.train), load_split(args.validation)
    x, dx = [[sentence_features(r["tokens"]) for r in rows] for rows in (train, dev)]
    document_bootstrap(dev, json.loads(args.validation_docs.read_text()), args.extension_folder, out)
    summary = sample_runs(train, dev, x, dx, json.loads(args.shared_ids.read_text()), settings, out)
    write_json(out / "subset_summary.json", summary)
    fold_comparison(train, x, args.extension_folder, settings, out)


if __name__ == "__main__":
    main()
