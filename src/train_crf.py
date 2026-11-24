import argparse
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import sklearn_crfsuite

from ner_common import entity_spans, load_split, metrics, write_json, write_predictions


CANDIDATES = ((0.1, 0.1), (0.01, 0.1), (0.1, 1.0))
MAX_ITERATIONS = 80


def sentence_features(tokens):
    def word_features(token):
        lowered = token.lower()
        return {
            "lower": lowered,
            "is_upper": token.isupper(),
            "is_title": token.istitle(),
            "is_digit": token.isdigit(),
            "has_digit": any(char.isdigit() for char in token),
            "has_hyphen": "-" in token,
        }

    basic = [word_features(token) for token in tokens]
    features = []
    for index, token in enumerate(tokens):
        lower = token.lower()
        word = {
            **basic[index],
            "bias": 1.0,
            "prefix_2": lower[:2],
            "prefix_3": lower[:3],
            "suffix_2": lower[-2:],
            "suffix_3": lower[-3:],
        }
        if index == 0:
            word["BOS"] = True
        else:
            word.update({f"prev:{key}": value for key, value in basic[index - 1].items()})
        if index == len(tokens) - 1:
            word["EOS"] = True
        else:
            word.update({f"next:{key}": value for key, value in basic[index + 1].items()})
        features.append(word)
    return features


def predict_tags(tokens, model):
    """Predict one raw IOB2-style tag per input token using the saved CRF."""
    return model.predict_single(sentence_features(tokens)) if tokens else []


def select_subset(rows, path):
    content = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(content, dict):
        content = content["ids"]
    if not isinstance(content, list) or not content:
        raise ValueError("Subset file must contain a nonempty JSON array of sample IDs")
    ids = {str(item) for item in content}
    if len(ids) != len(content):
        raise ValueError("Subset IDs must be unique")
    subset = [row for row in rows if row["id"] in ids]
    if len(subset) != len(ids):
        missing = ids - {row["id"] for row in subset}
        raise ValueError(f"IDs missing from the training split: {sorted(missing)[:10]}")
    return subset


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(args, output):
    all_train_rows = load_split(args.train)
    train_rows = select_subset(all_train_rows, args.subset_ids) if args.subset_ids else all_train_rows
    validation_rows = load_split(args.validation)
    train_features = [sentence_features(row["tokens"]) for row in train_rows]
    train_tags = [row["gold_tags"] for row in train_rows]
    validation_features = [sentence_features(row["tokens"]) for row in validation_rows]
    validation_tags = [row["gold_tags"] for row in validation_rows]

    candidates = []
    best_model = None
    best_score = -1.0
    best_report = None
    best_settings = None
    for c1, c2 in CANDIDATES:
        settings = {
            "algorithm": "lbfgs",
            "c1": c1,
            "c2": c2,
            "max_iterations": MAX_ITERATIONS,
            "all_possible_transitions": True,
        }
        model = sklearn_crfsuite.CRF(**settings)
        start = perf_counter()
        model.fit(train_features, train_tags)
        elapsed = perf_counter() - start
        report = metrics(validation_tags, model.predict(validation_features))
        score = report["micro avg"]["f1-score"]
        candidates.append({"settings": settings, "validation_micro": report["micro avg"], "fit_seconds": elapsed})
        print(f"{len(train_rows):,} sentences | c1={c1}, c2={c2} | validation F1={score:.4f} | {elapsed:.1f}s", flush=True)
        if score > best_score:
            best_model = model
            best_score = score
            best_report = report
            best_settings = settings

    joblib.dump(best_model, output / "model.joblib", compress=3)
    selection = {
        "subset_status": "provisional" if args.provisional else "provided" if args.subset_ids else "full",
        "subset_ids_sha256": sha256(args.subset_ids) if args.subset_ids else None,
        "training_sentences": len(train_rows),
        "training_split_sentences": len(all_train_rows),
        "training_entities_by_type": dict(Counter(
            kind for tags in train_tags for _, _, kind in entity_spans(tags)
        )),
        "validation_sentences": len(validation_rows),
        "data_sha256": {"train": sha256(args.train), "validation": sha256(args.validation)},
        "library_versions": {name: version(name) for name in ("pyarrow", "scikit-learn", "sklearn-crfsuite", "python-crfsuite", "seqeval", "joblib")},
        "selected_settings": best_settings,
        "candidates_in_order": candidates,
        "validation": best_report,
    }
    test_rows = load_split(args.test)
    test_features = [sentence_features(row["tokens"]) for row in test_rows]
    test_predictions = best_model.predict(test_features)
    write_predictions(output / "test_predictions.jsonl", test_rows, test_predictions)
    selection["test_sentences"] = len(test_rows)
    selection["data_sha256"]["test"] = sha256(args.test)
    selection["test"] = metrics([row["gold_tags"] for row in test_rows], test_predictions)
    write_json(output / "metrics.json", selection)
    print(f"Test micro-F1={selection['test']['micro avg']['f1-score']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a CoNLL-2003 linear-chain CRF baseline.")
    for field in ("train", "validation", "test", "output-dir"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--subset-ids", type=Path, help="JSON array of original training sentence IDs")
    parser.add_argument("--provisional", action="store_true", help="Mark the subset as awaiting team confirmation")
    args = parser.parse_args()
    if args.provisional and args.subset_ids is None:
        parser.error("--provisional requires --subset-ids")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}; choose a new directory")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir.parent) as temporary:
        output = Path(temporary) / "run"
        output.mkdir()
        train(args, output)
        output.rename(args.output_dir)


if __name__ == "__main__":
    main()
