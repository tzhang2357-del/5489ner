import argparse
import csv
from collections import Counter
from pathlib import Path
from random import Random

from ner_common import entity_spans, load_split, metrics, read_predictions, write_json


ERROR_TYPES = ("boundary", "type", "boundary_and_type", "missed", "spurious")
LENGTH_GROUPS = ("1", "2", "3+")


def classify_errors(gold, predicted):
    unmatched_gold = gold - predicted
    unmatched_predicted = predicted - gold
    pairs = []
    for reference in unmatched_gold:
        for guess in unmatched_predicted:
            start, end, kind = reference
            guess_start, guess_end, guess_kind = guess
            overlap = min(end, guess_end) - max(start, guess_start)
            if overlap <= 0:
                continue
            if (start, end) == (guess_start, guess_end):
                priority, category = 0, "type"
            elif kind == guess_kind:
                priority, category = 1, "boundary"
            else:
                priority, category = 2, "boundary_and_type"
            union = max(end, guess_end) - min(start, guess_start)
            pairs.append((priority, -overlap / union, reference, guess, category))

    errors = []
    for _, _, reference, guess, category in sorted(pairs):
        if reference in unmatched_gold and guess in unmatched_predicted:
            errors.append((category, reference, guess))
            unmatched_gold.remove(reference)
            unmatched_predicted.remove(guess)
    errors.extend(("missed", reference, None) for reference in sorted(unmatched_gold))
    errors.extend(("spurious", None, guess) for guess in sorted(unmatched_predicted))
    return errors


def render_span(tokens, span):
    if span is None:
        return "—"
    start, end, kind = span
    return f"{' '.join(tokens[start:end])} [{kind}]"


def render_spans(tokens, spans):
    return " | ".join(render_span(tokens, span) for span in sorted(spans)) or "—"


def render_errors(tokens, errors):
    return " | ".join(
        f"{category}: {render_span(tokens, gold)} → {render_span(tokens, guess)}"
        for category, gold, guess in errors
    ) or "—"


def analyze(rows, predictions, seed, case_count):
    gold_tags = [row["gold_tags"] for row in rows]
    error_counts = {name: Counter() for name in predictions}
    length_hits = {name: Counter() for name in predictions}
    length_totals = Counter()
    eligible = []
    for row in rows:
        row_id, tokens = row["id"], row["tokens"]
        gold = set(entity_spans(row["gold_tags"]))
        by_model = {name: set(entity_spans(sequences[row_id])) for name, sequences in predictions.items()}
        case = {"id": row_id, "sentence_tokens": " ".join(tokens), "gold_entities": render_spans(tokens, gold)}
        for start, end, _ in gold:
            length_totals[str(end - start) if end - start < 3 else "3+"] += 1

        for name, predicted in by_model.items():
            errors = classify_errors(gold, predicted)
            error_counts[name].update(category for category, _, _ in errors)
            for start, end, _ in gold & predicted:
                length_hits[name][str(end - start) if end - start < 3 else "3+"] += 1
            case[f"{name}_entities"] = render_spans(tokens, predicted)
            case[f"{name}_errors"] = render_errors(tokens, errors)

        include = by_model["crf"] != by_model["bert"] if "bert" in by_model else by_model["crf"] != gold
        if include:
            eligible.append(case)

    cases = Random(seed).sample(eligible, min(case_count, len(eligible)))
    summary = {
        "test_sentences": len(rows),
        "sample_definition": "CRF and BERT entity disagreements" if "bert" in predictions else "CRF entity mismatches against gold",
        "sample_seed": seed,
        "eligible_sentences": len(eligible),
        "sampled_sentences": len(cases),
        "models": {},
    }
    for name, sequences in predictions.items():
        summary["models"][name] = {
            "test_metrics": metrics(gold_tags, [sequences[row["id"]] for row in rows]),
            "test_error_counts": {kind: error_counts[name][kind] for kind in ERROR_TYPES},
            "test_recall_by_gold_entity_length": {
                group: {
                    "matched": length_hits[name][group],
                    "support": length_totals[group],
                    "recall": length_hits[name][group] / length_totals[group] if length_totals[group] else None,
                }
                for group in LENGTH_GROUPS
            },
        }
    return summary, cases


def main():
    parser = argparse.ArgumentParser(description="Analyze exact-span errors for CRF and optional BERT predictions.")
    for required in ("test", "crf", "output-dir"):
        parser.add_argument(f"--{required}", type=Path, required=True)
    parser.add_argument("--bert", type=Path, help="BERT predictions in the same JSONL or JSON contract")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=20)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be a positive integer")

    rows = load_split(args.test)
    predictions = {"crf": read_predictions(args.crf, rows)}
    if args.bert:
        predictions["bert"] = read_predictions(args.bert, rows)
    summary, cases = analyze(rows, predictions, args.seed, args.cases)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "error_summary.json", summary)
    fields = ["id", "sentence_tokens", "gold_entities"]
    fields += [column for name in predictions for column in (f"{name}_entities", f"{name}_errors")]
    with (args.output_dir / "error_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cases)
    print(f"Sampled {len(cases)} of {summary['eligible_sentences']} eligible test sentences")


if __name__ == "__main__":
    main()
