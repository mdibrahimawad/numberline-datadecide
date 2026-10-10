"""Build the input sets (P1 single statements, P2 truth-context sequences) and the confound check.

    python -m tt.data fetch
    python -m tt.data build --tokenizer EleutherAI/pythia-1.4b-deduped   # word count if omitted

Outputs WORK/states/<input>/items.csv (one row per item, rows in random order so that no
analysis can pick up structure from the row order) and results/confounds_<input>.json.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from . import config as C

RAW_FILES = [f"{t}.csv" for t in C.P1_TOPICS] + ["counterfact.json"]


def fetch():
    C.RAW.mkdir(parents=True, exist_ok=True)
    for f in RAW_FILES:
        dst = C.RAW / f
        if not dst.exists():
            print("download", f)
            urllib.request.urlretrieve(C.GOT_URL + f, dst)


# ---------------------------------------------------------------- lengths

def length_fn(tokenizer_name: str | None):
    if tokenizer_name is None:
        return (lambda s: len(s.split())), "words"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    return (lambda s: len(tok(s)["input_ids"])), "tokens"


def length_auc(lengths: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    """CV AUC of a logistic regression predicting truth from length (and length^2)."""
    x = np.c_[lengths, lengths ** 2].astype(float)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=seed)
    p = cross_val_predict(LogisticRegression(), x, labels, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(labels, p)
    return float(max(auc, 1 - auc))


# ---------------------------------------------------------------- P1

def _sample_topic(df: pd.DataFrame, n: int, rng, lengths=None) -> pd.DataFrame:
    """n true + n false. With ``lengths`` given, match the length distribution of the classes."""
    if lengths is None:
        parts = [g.sample(min(n, len(g)), random_state=rng.integers(1 << 31)) for _, g in df.groupby("label")]
        k = min(len(p) for p in parts)
        return pd.concat([p.iloc[:k] for p in parts])
    df = df.assign(_len=lengths)
    out = []
    by = {lab: g for lab, g in df.groupby("label")}
    common = sorted(set(by[0]._len) & set(by[1]._len))
    quota = {L: min((by[0]._len == L).sum(), (by[1]._len == L).sum()) for L in common}
    total = sum(quota.values())
    scale = min(1.0, n / max(total, 1))
    for L in common:
        k = int(round(quota[L] * scale))
        for lab in (0, 1):
            g = by[lab][by[lab]._len == L]
            out.append(g.sample(k, random_state=rng.integers(1 << 31)))
    return pd.concat(out).drop(columns="_len")


def _balance_cities(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only countries that appear as the object of both true and false statements."""
    both = df.groupby("country").label.nunique()
    return df[df.country.isin(both[both == 2].index)]


def build_p1(tokenizer: str | None):
    rng = np.random.default_rng(C.SEED_DATA)
    nlen, unit = length_fn(tokenizer)
    items, report = [], {"length_unit": unit, "topics": {}}
    for topic in C.P1_TOPICS:
        df = pd.read_csv(C.RAW / f"{topic}.csv")
        if topic == "cities":
            df = _balance_cities(df)
        s = _sample_topic(df, C.P1_PER_CLASS, rng)
        if topic == "cities":                       # sampling can leave a country in one class only
            while True:
                s = _balance_cities(s)
                k = s.label.value_counts().min()
                s2 = pd.concat([g.iloc[:k] for _, g in s.groupby("label")])
                if len(s2) == len(s) and s2.groupby("country").label.nunique().min() == 2:
                    break
                s = s2
        lens = s.statement.map(nlen).to_numpy()
        auc = length_auc(lens, s.label.to_numpy())
        rebalanced = False
        if auc > C.LENGTH_AUC_MAX:                  # gate failed: length-matched resample
            s = _sample_topic(df, C.P1_PER_CLASS, rng, lengths=df.statement.map(nlen).to_numpy())
            lens = s.statement.map(nlen).to_numpy()
            auc = length_auc(lens, s.label.to_numpy())
            rebalanced = True
        r = {"n_true": int((s.label == 1).sum()), "n_false": int((s.label == 0).sum()),
             "length_auc": auc, "rebalanced": rebalanced, "passes": auc <= C.LENGTH_AUC_MAX,
             "length_by_class": {str(k): {"mean": float(np.mean(lens[s.label.to_numpy() == k])),
                                          "hist": dict(pd.Series(lens[s.label.to_numpy() == k]).value_counts().sort_index().astype(int).astype(str))}
                                 for k in (0, 1)}}
        if topic == "cities":
            per = s.groupby(["country", "label"]).size().unstack(fill_value=0)
            r["countries"] = int(len(per))
            r["countries_one_class_only"] = int(((per[0] == 0) | (per[1] == 0)).sum())
        report["topics"][topic] = r
        items.append(pd.DataFrame({"text": s.statement.to_numpy(), "label": s.label.to_numpy(), "topic": topic,
                                   "src_row": s.index.to_numpy()}))
    items = pd.concat(items, ignore_index=True)
    items = items.sample(frac=1, random_state=C.SEED_DATA).reset_index(drop=True)
    items.insert(0, "item_id", np.arange(len(items)))
    report["passes"] = all(r["passes"] for r in report["topics"].values())
    return items, report


# ---------------------------------------------------------------- P2

def build_p2(tokenizer: str | None):
    """Sequences of 4 CounterFact statements; the first 3 all true or all false (the label),
    the 4th is a fresh true fact cut before its attribute. Each 4th fact appears in one
    true-context and one false-context sequence whose context facts are the same, only the
    attributes differ (matched pair)."""
    rng = np.random.default_rng(C.SEED_DATA + 1)
    cf = json.load(open(C.RAW / "counterfact.json"))
    rows = [{"rel": x["requested_rewrite"]["relation_id"],
             "prompt": x["requested_rewrite"]["prompt"],
             "subject": x["requested_rewrite"]["subject"],
             "true": x["requested_rewrite"]["target_true"]["str"]} for x in cf]
    df = pd.DataFrame(rows)
    df = df[df.prompt.str.endswith("{}") == False]               # need "...{} ..." with attribute at the end
    df = df[df.prompt.str.count(r"\{\}") == 1]
    rels = df.rel.value_counts().index[: C.P2_N_RELATIONS]
    df = df[df.rel.isin(rels)].reset_index(drop=True)
    attrs = df.groupby("rel")["true"].apply(lambda s: sorted(set(s))).to_dict()

    def stmt(r, attr):
        return f"{r.prompt.format(r.subject)} {attr}."

    def false_attr(r):
        pool = [a for a in attrs[r.rel] if a != r.true]
        return pool[rng.integers(len(pool))]

    idx = rng.permutation(len(df))
    need = C.P2_N_PAIRS * (C.P2_CONTEXT + 1)
    assert len(idx) >= need, "not enough CounterFact facts"
    items = []
    for p in range(C.P2_N_PAIRS):
        sel = idx[p * (C.P2_CONTEXT + 1):(p + 1) * (C.P2_CONTEXT + 1)]
        ctx, last = df.iloc[sel[:-1]], df.iloc[sel[-1]]
        falses = [false_attr(r) for r in ctx.itertuples()]
        prefix_last = last.prompt.format(last.subject)
        for label in (1, 0):
            c = [stmt(r, r.true if label else fa) for r, fa in zip(ctx.itertuples(), falses)]
            items.append({"text": " ".join(c + [prefix_last]), "label": label, "topic": "counterfact",
                          "pair_id": p, "rel_last": last.rel, "target": " " + last.true})
    items = pd.DataFrame(items).sample(frac=1, random_state=C.SEED_DATA).reset_index(drop=True)
    items.insert(0, "item_id", np.arange(len(items)))
    nlen, unit = length_fn(tokenizer)
    lens = items.text.map(nlen).to_numpy()
    auc = length_auc(lens, items.label.to_numpy())
    report = {"length_unit": unit, "relations": list(rels), "n": len(items), "length_auc": auc,
              "passes": auc <= C.LENGTH_AUC_MAX}
    return items, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "build"])
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer for token lengths (word count if omitted)")
    ap.add_argument("--inputs", nargs="+", default=["P1", "P2"])
    a = ap.parse_args()
    if a.cmd == "fetch":
        return fetch()
    fetch()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    for name in a.inputs:
        items, rep = (build_p1 if name == "P1" else build_p2)(a.tokenizer)
        out = C.STATES / name
        out.mkdir(parents=True, exist_ok=True)
        items.to_csv(out / "items.csv", index=False)
        json.dump(rep, open(C.RESULTS / f"confounds_{name}.json", "w"), indent=1)
        print(name, len(items), "items; gate", "PASS" if rep["passes"] else "FAIL", json.dumps(
            {k: v for k, v in rep.items() if k != "topics"} if name == "P2" else
            {t: (r["n_true"], r["n_false"], round(r["length_auc"], 3), r["rebalanced"]) for t, r in rep["topics"].items()}))


if __name__ == "__main__":
    main()
