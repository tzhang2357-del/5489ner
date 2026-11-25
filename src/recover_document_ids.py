"""Align the DOCSTART-bearing English training text to the Parquet sentence IDs."""
import argparse
from pathlib import Path

from ner_common import load_split, write_json


def recover(raw_training, sentences):
    docs = {}
    doc_id = -1
    words = []
    for line in (raw_training.read_text(encoding="utf-8") + "\n\n").splitlines():
        if line.startswith("-DOCSTART- "):
            if words:
                raise ValueError("Document boundary inside an unfinished sentence")
            doc_id += 1
        elif line.strip():
            words.append(line.split()[0])
        elif words:
            row = sentences[len(docs)]
            if doc_id < 0 or words != row["tokens"]:
                raise ValueError(f"Source tokens differ at Parquet row {row['id']}")
            docs[row["id"]] = doc_id
            words = []
    if len(docs) != len(sentences):
        raise ValueError("Source and Parquet sentence counts differ")
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source_text", "train_parquet", "output_json"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    write_json(args.output_json, recover(args.source_text, load_split(args.train_parquet)))
