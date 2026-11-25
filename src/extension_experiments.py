"""Document-grouped CRF tuning and controlled validation-set feature experiments."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize
from sklearn_crfsuite import CRF
from threadpoolctl import threadpool_limits

from ner_common import load_split, metrics, write_json, write_predictions
from train_crf import CANDIDATES, MAX_ITERATIONS, sentence_features, sha256


def save_table(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def fit(x, y, settings):
    c1, c2 = settings
    model = CRF(algorithm="lbfgs", c1=c1, c2=c2, all_possible_transitions=True,
                max_iterations=MAX_ITERATIONS)
    return model.fit(x, y)


def tune(rows, features, labels, doc_ids, out):
    groups = np.array([doc_ids[r["id"]] for r in rows])
    scores = defaultdict(list)
    folds = []
    for number, (train, held) in enumerate(GroupKFold(n_splits=3).split(groups, groups=groups), 1):
        assert set(groups[train]).isdisjoint(groups[held])
        folds.extend({"id": rows[i]["id"], "document": groups[i], "held_out_fold": number} for i in held)
        x, y = [features[i] for i in train], [labels[i] for i in train]
        hx, hy = [features[i] for i in held], [labels[i] for i in held]
        for settings in CANDIDATES:
            pred = fit(x, y, settings).predict(hx)
            score = metrics(hy, pred)["micro avg"]["f1-score"]
            scores[settings].append(score)
            print(f"Fold {number}, params {settings}, F1 {score:.5f}", flush=True)
    save_table(out / "folds.csv", folds)
    save_table(out / "cv.csv", [
        dict(c1=c1, c2=c2, fold1=v[0], fold2=v[1], fold3=v[2], mean=mean(v), sd=stdev(v))
        for (c1, c2), v in scores.items()
    ])
    return max(CANDIDATES, key=lambda s: mean(scores[s]))


def cluster_words(rows, out):
    contexts = defaultdict(Counter)
    for row in rows:
        words = [w.lower() for w in row["tokens"]]
        for i, word in enumerate(words):
            contexts[word]["left=" + (words[i-1] if i else "<BOS>")] += 1
            contexts[word]["right=" + (words[i+1] if i+1 < len(words) else "<EOS>")] += 1
    vocab = sorted(w for w, counts in contexts.items() if sum(counts.values()) >= 6)
    matrix = DictVectorizer().fit_transform([contexts[w] for w in vocab])
    matrix = TfidfTransformer(sublinear_tf=True).fit_transform(matrix)
    with threadpool_limits(limits=1):
        svd = TruncatedSVD(n_components=32, random_state=42)
        vectors = normalize(svd.fit_transform(matrix))
        km = KMeans(n_clusters=32, n_init=10, random_state=42)
        assigned = km.fit_predict(vectors)
    if not np.isfinite(vectors).all() or not np.isfinite(km.cluster_centers_).all():
        raise ValueError("Non-finite clustering coordinates")
    distances = np.stack([np.sum((vectors-center)**2, axis=1) for center in km.cluster_centers_], axis=1)
    assigned_dist = distances[np.arange(len(assigned)), assigned]
    if not np.allclose(assigned_dist, distances.min(axis=1)) or not np.isclose(assigned_dist.sum(), km.inertia_):
        raise ValueError("Independent Euclidean distances disagree with k-means assignments")
    examples = []
    for k, center in enumerate(km.cluster_centers_):
        ix = np.where(assigned == k)[0]
        nearest = ix[np.argsort(np.linalg.norm(vectors[ix] - center, axis=1))[:10]]
        examples.append(dict(cluster=k, words=len(ix), nearest=", ".join(vocab[i] for i in nearest)))
    save_table(out / "cluster_examples.csv", examples)
    lookup = {w: str(k) for w, k in zip(vocab, assigned)}
    write_json(out / "cluster_lookup.json", lookup)
    return lookup, float(svd.explained_variance_ratio_.sum())


def transform(rows, base, variant, clusters):
    if variant == "C":
        return base
    transformed = []
    for row, sent in zip(rows, base):
        sequence = []
        for token, features in zip(row["tokens"], sent):
            if variant == "A":
                item = {"bias": 1.0, "lower": token.lower()}
            elif variant == "B":
                item = {k: v for k,v in features.items() if ":" not in k and k not in ("BOS", "EOS")}
            else:
                item = dict(features, cluster=clusters.get(token.lower(), "UNK"))
            sequence.append(item)
        transformed.append(sequence)
    return transformed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "validation", "documents", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    train, dev = load_split(args.train), load_split(args.validation)
    x, dx = [[sentence_features(r["tokens"]) for r in rows] for rows in (train, dev)]
    y, dy = [[r["gold_tags"] for r in rows] for rows in (train, dev)]
    docs = json.loads(args.documents.read_text())
    settings = tune(train, x, y, docs, out)
    clusters, variance = cluster_words(train, out)
    results = []
    for variant in "ABCD":
        model = fit(transform(train, x, variant, clusters), y, settings)
        predictions = model.predict(transform(dev, dx, variant, clusters))
        report = metrics(dy, predictions)
        results.append(dict(variant=variant, micro_f1=report["micro avg"]["f1-score"],
                            **{k: report[k]["f1-score"] for k in ("PER","ORG","LOC","MISC")}))
        write_predictions(out / f"validation_{variant}.jsonl", dev, predictions)
        if variant in ("C","D"):
            joblib.dump(model, out / f"full_{variant}.joblib", compress=3)
        print(results[-1], flush=True)
    save_table(out / "ablation.csv", results)
    words = [w.lower() for row in dev for w in row["tokens"]]
    write_json(out / "protocol.json", dict(c1=settings[0], c2=settings[1], train_sha256=sha256(args.train),
               validation_sha256=sha256(args.validation), documents=len(set(docs.values())),
               clustered_types=len(clusters), validation_token_coverage=sum(w in clusters for w in words)/len(words),
               svd_variance_fraction=variance))


if __name__ == "__main__":
    main()
