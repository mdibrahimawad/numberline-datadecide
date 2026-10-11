"""Diagnose a signal at step 0 (untrained model): is it a broken pipeline or a lexical shortcut?

    python -m tt.diag_step0 --model pythia-410m-deduped

Per topic, compares the L1 probe on hidden states (several layers, every extracted checkpoint)
with probes that only see the words (bag of words / bigrams). If the step-0 probe is no better
than a words-only probe, the step-0 signal is lexical, not a bug. Also prints T3b per topic.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .mst import T3


def cv_auc(X, y, scale=True):
    clf = make_pipeline(StandardScaler(with_mean=not hasattr(X, "tocsr")), LogisticRegression(max_iter=2000)) \
        if scale else LogisticRegression(max_iter=2000)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="decision_function")
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m-deduped")
    ap.add_argument("--input", default="P1")
    a = ap.parse_args()
    it = pd.read_csv(C.STATES / a.input / "items.csv")
    y, topics = it.label.to_numpy().astype(int), it.topic.to_numpy()
    sdir = C.STATES / a.model / a.input
    steps = sorted(int(p.stem[4:]) for p in sdir.glob("step*.npz") if ".tmp" not in p.name)
    H = {s: np.load(sdir / f"step{s}.npz")["H"] for s in steps}
    L = H[steps[0]].shape[0] - 1
    layers = sorted({0, 1, L // 4, L // 2, 3 * L // 4, L})
    rows = []
    for t in np.unique(topics):
        i = topics == t
        txt, yt = it.text[i].tolist(), y[i]
        r = {"topic": t, "n": int(i.sum())}
        r["words_uni"] = cv_auc(CountVectorizer(token_pattern=r"[^\s.']+").fit_transform(txt), yt, scale=False)
        r["words_uni+bi"] = cv_auc(CountVectorizer(token_pattern=r"[^\s.']+", ngram_range=(1, 2)).fit_transform(txt), yt, scale=False)
        for s in steps:
            for l in layers:
                r[f"s{s}_L{l}"] = cv_auc(H[s][l][i].astype(np.float32), yt)
        rows.append(r)
    df = pd.DataFrame(rows).set_index("topic").T
    pd.set_option("display.width", 200)
    print("\n== L1 AUC per topic: words-only probes vs hidden-state probes (s<step>_L<layer>) ==")
    print(df.round(3).to_string())
    print("\n== T3b within-topic truth homophily per topic (chance ~0.5) ==")
    for s in steps:
        for l in (L // 2,):
            out = []
            for t in np.unique(topics):
                i = np.where(topics == t)[0]
                t3 = T3(H[s][l][i].astype(np.float64), np.zeros(len(i), int))
                out.append(f"{t}={t3.values(y[i])['T3b_within_truth_homophily']:.3f}")
            print(f"step {s:>6} layer {l}: " + "  ".join(out))


if __name__ == "__main__":
    main()
