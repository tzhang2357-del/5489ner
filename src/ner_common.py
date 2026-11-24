import json
from pathlib import Path

import pyarrow.parquet as pq
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


ENTITY_TYPES = ("PER", "ORG", "LOC", "MISC")
BIO_TAGS = {"O"} | {f"{prefix}-{kind}" for prefix in ("B", "I") for kind in ENTITY_TYPES}


def load_split(path):
    table = pq.read_table(path, columns=["id", "tokens", "ner_tags"])
    metadata = json.loads(table.schema.metadata[b"huggingface"])
    labels = metadata["info"]["features"]["ner_tags"]["feature"]["names"]
    if len(labels) != len(BIO_TAGS) or set(labels) != BIO_TAGS:
        raise ValueError(f"Unexpected CoNLL-2003 label inventory in {path}")

    rows = []
    seen = set()
    for row in table.to_pylist():
        row_id = str(row["id"])
        tokens = row["tokens"]
        tag_ids = row["ner_tags"]
        if row_id in seen or len(tokens) != len(tag_ids) or not tokens:
            raise ValueError(f"Invalid or duplicate sample {row_id} in {path}")
        if any(not isinstance(index, int) or not 0 <= index < len(labels) for index in tag_ids):
            raise ValueError(f"Invalid tag ID in sample {row_id} in {path}")
        tags = [labels[index] for index in tag_ids]
        if normalize_tags(tags) != tags:
            raise ValueError(f"Invalid IOB2 gold sequence in sample {row_id} in {path}")
        rows.append({"id": row_id, "tokens": tokens, "gold_tags": tags})
        seen.add(row_id)
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def normalize_tags(tags):
    normalized = []
    active_type = None
    for tag in tags:
        if tag not in BIO_TAGS:
            raise ValueError(f"Unexpected BIO tag: {tag!r}")
        if tag == "O":
            active_type = None
        else:
            prefix, kind = tag.split("-", 1)
            if prefix == "I" and kind != active_type:
                tag = f"B-{kind}"
            active_type = kind
        normalized.append(tag)
    return normalized


def entity_spans(tags):
    """Return half-open token spans as (start, end, type)."""
    spans = []
    start = None
    kind = None
    for position, tag in enumerate([*normalize_tags(tags), "O"]):
        if tag.startswith("I-"):
            continue
        if start is not None:
            spans.append((start, position, kind))
        if tag.startswith("B-"):
            start = position
            kind = tag[2:]
        else:
            start = None
    return spans


def metrics(gold_tags, pred_tags):
    if len(gold_tags) != len(pred_tags) or any(
        len(gold) != len(pred) for gold, pred in zip(gold_tags, pred_tags)
    ):
        raise ValueError("Gold and prediction sequence lengths differ")
    report = classification_report(
        gold_tags,
        [normalize_tags(tags) for tags in pred_tags],
        output_dict=True,
        mode="strict",
        scheme=IOB2,
        zero_division=0,
    )
    return {
        name: {
            "precision": float(values["precision"]),
            "recall": float(values["recall"]),
            "f1-score": float(values["f1-score"]),
            "support": int(values["support"]),
        }
        for name, values in report.items()
    }


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_predictions(path, rows, tags):
    if len(rows) != len(tags) or any(len(row["tokens"]) != len(seq) for row, seq in zip(rows, tags)):
        raise ValueError("Prediction sequence lengths differ from original tokens")
    with Path(path).open("w", encoding="utf-8") as handle:
        for row, sequence in zip(rows, tags):
            record = {"id": row["id"], "tokens": row["tokens"], "pred_tags": sequence}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_predictions(path, gold_rows):
    """Accept JSONL or a JSON array, keyed by the original test sample ID."""
    contents = Path(path).read_text(encoding="utf-8-sig")
    if contents.lstrip().startswith("["):
        records = json.loads(contents)
    else:
        records = [json.loads(line) for line in contents.splitlines() if line.strip()]
    gold_by_id = {row["id"]: row for row in gold_rows}
    predictions = {}
    for record in records:
        row_id = str(record["id"])
        if row_id in predictions or row_id not in gold_by_id:
            raise ValueError(f"Duplicate or unknown sample ID {row_id} in {path}")
        tokens = gold_by_id[row_id]["tokens"]
        tags = record["pred_tags"]
        if record["tokens"] != tokens or not isinstance(tags, list) or len(tags) != len(tokens):
            raise ValueError(f"Tokens or tag count do not match the test row {row_id} in {path}")
        predictions[row_id] = normalize_tags(tags)
    missing = gold_by_id.keys() - predictions.keys()
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} test sample IDs")
    return predictions
