"""Recompute the four experiments with the team's pinned evaluator."""
import argparse
import csv
import sys
from pathlib import Path

from ner_common import load_split, metrics, read_predictions, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve() / "src"))
    import data
    import evaluate

    args.out.mkdir(parents=True, exist_ok=True)
    data.PARQUET_FILES["test"] = args.test
    gold = load_split(args.test)
    team_gold = data.load_split("test")
    assert [(r["id"], r["tokens"], r["gold_tags"]) for r in gold] == [(s.id, s.tokens, s.bio_tags) for s in team_gold]
    with (args.repo / "results/metrics.csv").open() as handle:
        original = list(csv.DictReader(handle))
    table = []
    full = {}
    for key, model, size in [("E1", "CRF", "full"), ("E2", "BERT", "full"), ("E3", "CRF", "10pct"), ("E4", "BERT", "10pct")]:
        path = args.repo / f"results/predictions/{key.lower()}_{model.lower()}_{size}_test.jsonl"
        predictions = read_predictions(path, gold)
        sequences = [predictions[row["id"]] for row in gold]
        result = evaluate.evaluate_all(team_gold, sequences)
        seq = metrics([row["gold_tags"] for row in gold], sequences)
        scores = result["overall"]
        assert scores["F1"] == round(seq["micro avg"]["f1-score"], 4)
        record = dict(experiment=key, model=model, train_data="100%" if size == "full" else "10%", n_sentences=len(gold), **scores)
        for kind, values in result["per_type"].items():
            for stat in ("P", "R", "F1"):
                record[f"{stat}_{kind}"] = values[stat]
        for length, field in [("1", "1word"), ("2", "2word"), ("3+", "3plus")]:
            record[f"recall_{field}"] = result["length_recall"][length]
        table.append(record)
        full[key] = dict(team=result, strict_iob2=seq)
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(args.out / "four_scores.json", full)
    for row in table:
        old = next(r for r in original if r["experiment"] == row["experiment"])
        print(row["experiment"], {k: (old[k], row[k]) for k in row if k not in ("experiment", "model", "train_data") and (not old[k] or float(old[k]) != row[k])})


if __name__ == "__main__":
    main()
