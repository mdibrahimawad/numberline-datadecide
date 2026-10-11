"""Feature-set ladder, deltas, residual analysis, effect sizes and controls (brief sections 7-8).

Evaluation protocol (fixed before results):
  * one case-control set per (target, seed): all positives + 5x random negatives per split;
  * models are fit on the train split (steps 1000-3999) and scored on the out-of-time test
    split (steps 4000-5999) only;
  * subset "full" = fit and score on all rows; subset "hard" = fit and score on rows with
    mem_Sf = 0 (where B0 is constant, so the hard rungs 1-2 measure B1 alone);
  * AUROC is computed unweighted (it is invariant to uniform negative subsampling); AUPRC and
    recall at precision 0.9 weight each test negative by (negatives in the pool's test split
    and subset) / (negatives drawn), i.e. natural prevalence.
  * probe block: standardise + PCA(256) fitted on the train rows only, then joined to the
    other (standardised) columns; L2 logistic regression with C chosen by inner 5-fold CV.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from memtopo import stats, textstats
from memtopo.config import paths, record_gate
from memtopo.topology import FEATURE_NAMES

log = logging.getLogger("memtopo")

B0 = ["mem_Sf"]
B1 = ["match_len", "match_frac"]
B2 = ["ce_cont", "ce_prompt", "mink20", "ent_cont_mean", "ent_first", "logp_first"]
B3 = ["ce_pre", "ce_post", "exposure_drop", "post_slope"]
B4 = list(textstats.B4_NAMES)

# rung -> families; "B5" is the probe block, the others are column families
RUNGS = {
    "1": ["B0"],
    "2": ["B0", "B1"],
    "3": ["B0", "B1", "B2"],
    "4": ["B0", "B1", "B2", "B3", "B4"],
    "5": ["B0", "B1", "B2", "B3", "B4", "B5"],
    "6": ["B0", "B1", "B2", "B3", "B4", "B5", "B6"],
    "6+T0": ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "T0"],
    "7": ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "T1", "T2", "T3"],
    "8": ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "T1", "T2", "T3", "T0"],
    "9": ["T1", "T2", "T3"],
}
RUNG_LABEL = {"1": "B0", "2": "B0-B1", "3": "B0-B2", "4": "B0-B4", "5": "B0-B5 (strong)",
              "6": "B0-B6", "6+T0": "B0-B6+T0", "7": "full (B0-B6+T1-T3)", "8": "full+T0", "9": "T1-T3 only"}
COMPARISONS = [("7", "6", "H1: topology beyond strong baseline"),
               ("8", "6+T0", "H4: topology beyond step-0 topology"),
               ("8", "7", "control: adding T0 to full"),
               ("5", "4", "probe beyond non-representation baseline"),
               ("6", "5", "geometry beyond probe")]
H2_FEATURES = {"h1_n_bars": -1, "h1_pers_mean": +1, "h1_entropy": -1, "h0_death_mean": +1}


# ---------------------------------------------------------------------- data assembly
class FeatureTable:
    def __init__(self, cfg: dict, model: str, sf: int):
        f = paths(cfg).features(model, sf)
        self.tab = pd.read_parquet(f / "loss.parquet").set_index("index")
        self.tab = self.tab.join(pd.read_parquet(f / "data.parquet").set_index("index"))
        self.fam = {"B0": B0, "B1": B1, "B2": B2, "B3": B3, "B4": B4}
        for fam, name in (("B6", "geom"), ("T1", "topo_T1"), ("T2", "topo_T2"), ("T3", "topo_T3"), ("T0", "topo_T0")):
            t = pd.read_parquet(f / f"{name}.parquet").set_index("index")
            self.fam[fam] = list(t.columns)
            self.tab = self.tab.join(t, how="left")
        pz = np.load(f / "probe.npz")
        self.probe_pos = {int(i): k for k, i in enumerate(pz["index"])}
        self.probe = pz["states"].reshape(len(pz["index"]), -1)
        self.labels = pd.read_parquet(f / "labels.parquet").set_index("index")
        self.raw_t1 = (pd.read_parquet(f / "topo_T1raw.parquet").set_index("index")
                       if (f / "topo_T1raw.parquet").exists() else None)

    def columns(self, rung: str) -> tuple[list[str], bool]:
        fams = RUNGS[rung]
        return [c for fam in fams if fam != "B5" for c in self.fam[fam]], "B5" in fams

    def probe_rows(self, idx) -> np.ndarray:
        return self.probe[[self.probe_pos[int(i)] for i in idx]].astype(np.float32)


class Prepared:
    """Train-fitted preprocessing for one rung: impute + scale columns; scale + PCA probe."""

    def __init__(self, ft: FeatureTable, rung: str, train_idx, n_pca: int, seed: int):
        self.ft, (self.cols, self.use_probe) = ft, ft.columns(rung)
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True)
        self.sc = StandardScaler()
        self.sc.fit(self.imp.fit_transform(ft.tab.loc[train_idx, self.cols].to_numpy(np.float64)))
        if self.use_probe:
            P = ft.probe_rows(train_idx)
            self.psc = StandardScaler().fit(P)
            self.pca = PCA(min(n_pca, *P.shape), random_state=seed, svd_solver="randomized").fit(self.psc.transform(P))
            self.pc_sc = StandardScaler().fit(self.pca.transform(self.psc.transform(P)))

    def transform(self, idx) -> np.ndarray:
        parts = []
        if self.cols:
            parts.append(self.sc.transform(self.imp.transform(self.ft.tab.loc[idx, self.cols].to_numpy(np.float64))))
        if self.use_probe:
            P = self.ft.probe_rows(idx)
            parts.append(self.pc_sc.transform(self.pca.transform(self.psc.transform(P))))
        return np.nan_to_num(np.hstack(parts))


def fit_lr(X, y, Cs, folds, seed):
    gs = GridSearchCV(LogisticRegression(max_iter=3000), {"C": Cs}, scoring="roc_auc",
                      cv=StratifiedKFold(folds, shuffle=True, random_state=seed), n_jobs=-1)
    return gs.fit(X, y).best_estimator_


def fit_hgb(X, y, seed):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True,
                                          random_state=seed).fit(X, y)


def bal_acc128(X, y, draws: int, folds: int, seed: int) -> tuple[float, float]:
    """'Don't Forget' protocol: 64 + 64 balanced draw, ridge classifier, k-fold CV accuracy."""
    rng = np.random.default_rng(seed)
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    if len(pos) < 64 or len(neg) < 64:
        return float("nan"), float("nan")
    acc = []
    for _ in range(draws):
        i = np.r_[rng.choice(pos, 64, replace=False), rng.choice(neg, 64, replace=False)]
        acc.append(cross_val_score(RidgeClassifier(), X[i], y[i],
                                   cv=StratifiedKFold(folds, shuffle=True, random_state=0)).mean())
    return float(np.mean(acc)), float(np.std(acc))


# ---------------------------------------------------------------------- the ladder
def run_ladder(cfg: dict, model: str, sf: int, target: str = "mem_T", clfs=("lr", "hgb"),
               rungs=None) -> None:
    p, ac = paths(cfg), cfg["analysis"]
    out = p.results(model, sf)
    out.mkdir(parents=True, exist_ok=True)
    ft = FeatureTable(cfg, model, sf)
    cc = pd.read_parquet(p.data / f"cc_{model}.parquet")
    lab = ft.labels
    rungs = rungs or list(RUNGS)
    rows, drows, preds = [], [], []
    for seed in cfg["case_control"]["seeds"]:
        s = cc[(cc["target"] == target) & (cc["seed"] == seed)].set_index("index")
        s = s[s.index.isin(ft.tab.index) & s.index.isin(list(ft.probe_pos))]
        s["mem_Sf"] = ft.tab.loc[s.index, "mem_Sf"].to_numpy()
        for subset in ("hard", "full"):
            ss = s[s["mem_Sf"] == 0] if subset == "hard" else s
            tr, te = ss[ss["split"] == "train"], ss[ss["split"] == "test"]
            ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()
            # natural-prevalence weights for test negatives
            pool_te = lab[(lab["split"] == "test") & ((lab[f"mem_S{sf}"] == 0) if subset == "hard" else True)]
            n_neg_true = int((pool_te[target] == 0).sum())
            w = np.where(yte == 1, 1.0, n_neg_true / max((yte == 0).sum(), 1))
            scores = {}
            for rung in rungs:
                prep = Prepared(ft, rung, tr.index, ac["probe_pca"], ac["seed"])
                Xtr, Xte = prep.transform(tr.index), prep.transform(te.index)
                for clf in clfs:
                    m = fit_lr(Xtr, ytr, ac["lr_Cs"], ac["inner_cv_folds"], ac["seed"]) if clf == "lr" \
                        else fit_hgb(Xtr, ytr, ac["seed"])
                    sc = m.predict_proba(Xte)[:, 1]
                    scores[(rung, clf)] = sc
                    # shuffled-label control (control 1): permute train labels, same pipeline
                    yperm = np.random.default_rng(ac["seed"] + 7).permutation(ytr)
                    mp_ = fit_lr(Xtr, yperm, ac["lr_Cs"], ac["inner_cv_folds"], ac["seed"]) if clf == "lr" \
                        else fit_hgb(Xtr, yperm, ac["seed"])
                    sh = mp_.predict_proba(Xte)[:, 1]
                    b = stats.bootstrap_auroc(yte, {"m": sc, "sh": sh}, ac["bootstrap"], ac["seed"])
                    ba = bal_acc128(Xte, yte, ac["bal128_draws"], ac["bal128_folds"], ac["seed"]) \
                        if clf == "lr" else (float("nan"), float("nan"))
                    rows.append({"target": target, "seed": seed, "clf": clf, "subset": subset, "rung": rung,
                                 "rung_label": RUNG_LABEL[rung], "n_features": Xtr.shape[1],
                                 "n_train": len(ytr), "n_test_pos": int(yte.sum()), "n_test_neg": int((yte == 0).sum()),
                                 "AUROC": stats.auroc(yte, sc), "AUROC_lo": stats.ci(b["m"])[0],
                                 "AUROC_hi": stats.ci(b["m"])[1], "AUPRC": stats.auprc(yte, sc, w),
                                 "recall@P0.9": stats.recall_at_precision(yte, sc, w, ac["recall_at_precision"]),
                                 "bal_acc128_mean": ba[0], "bal_acc128_sd": ba[1],
                                 "shuffled_AUROC": stats.auroc(yte, sh), "shuffled_lo": stats.ci(b["sh"])[0],
                                 "shuffled_hi": stats.ci(b["sh"])[1]})
                    preds.append(pd.DataFrame({"index": te.index, "target": target, "seed": seed, "clf": clf,
                                               "subset": subset, "rung": rung, "score": sc, "label": yte, "weight": w}))
                    log.info("%s S%d %s seed%d %s %s rung %s: AUROC %.4f", model, sf, target, seed, subset, clf,
                             rung, rows[-1]["AUROC"])
            for clf in clfs:
                for a, bname, what in COMPARISONS:
                    if (a, clf) not in scores or (bname, clf) not in scores:
                        continue
                    bs = stats.bootstrap_auroc(yte, {"a": scores[(a, clf)], "b": scores[(bname, clf)]},
                                               ac["bootstrap"], ac["seed"])
                    d, pval = stats.delong(yte, scores[(a, clf)], scores[(bname, clf)])
                    lo, hi = stats.ci(bs["a"] - bs["b"])
                    drows.append({"target": target, "seed": seed, "clf": clf, "subset": subset,
                                  "comparison": f"{a} vs {bname}", "meaning": what, "dAUROC": d,
                                  "CI_lo": lo, "CI_hi": hi, "DeLong_p": pval,
                                  "CI_excludes_0": bool(lo > 0 or hi < 0)})
    suffix = "" if target == "mem_T" else f"_{target}"
    _merge_csv(out / f"ladder{suffix}.csv", pd.DataFrame(rows), ["target", "seed", "clf", "subset", "rung"])
    _merge_csv(out / f"deltas{suffix}.csv", pd.DataFrame(drows), ["target", "seed", "clf", "subset", "comparison"])
    pd.concat(preds).to_parquet(out / f"preds{suffix}.parquet")


def _merge_csv(f, new: pd.DataFrame, keys: list[str]) -> None:
    if f.exists() and len(new):
        old = pd.read_csv(f, dtype={"rung": str})
        new = pd.concat([old, new]).drop_duplicates(keys, keep="last")
    new.to_csv(f, index=False)


# ---------------------------------------------------------------------- residual, H2, H3
def residual_analysis(cfg: dict, model: str, sf: int, target: str = "mem_T") -> pd.DataFrame:
    """AUROC of the topology-only model (rung 9) within deciles of continuation CE at S_f."""
    p, ac = paths(cfg), cfg["analysis"]
    out = p.results(model, sf)
    suffix = "" if target == "mem_T" else f"_{target}"
    pr = pd.read_parquet(out / f"preds{suffix}.parquet")
    ft = pd.read_parquet(p.features(model, sf) / "loss.parquet").set_index("index")
    rows = []
    for (seed, clf), g in pr[(pr["rung"] == "9") & (pr["subset"] == "hard")].groupby(["seed", "clf"]):
        g = g.assign(ce=ft.loc[g["index"], "ce_cont"].to_numpy())
        g["decile"] = pd.qcut(g["ce"].rank(method="first"), 10, labels=False)
        for dcl, h in g.groupby("decile"):
            b = stats.bootstrap_auroc(h["label"].to_numpy(), {"m": h["score"].to_numpy()}, ac["bootstrap"], ac["seed"])
            lo, hi = stats.ci(b["m"])
            rows.append({"target": target, "seed": seed, "clf": clf, "decile": int(dcl), "ce_median": h["ce"].median(),
                         "n_pos": int(h["label"].sum()), "n": len(h),
                         "AUROC": stats.auroc(h["label"], h["score"]), "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)
    df.to_csv(out / f"residual{suffix}.csv", index=False)
    return df


def prune_correlated(df: pd.DataFrame, order: list[str], thr: float) -> list[str]:
    """Fay et al.: keep features in ``order``, dropping any with |corr| > thr to one already kept."""
    c = df[order].corr().abs()
    kept = []
    for f in order:
        if df[f].std() == 0 or any(c.loc[f, k] > thr for k in kept):
            continue
        kept.append(f)
    return kept


def _decile_matched(vals, y, ce, rng):
    """Positives vs an equal number of negatives drawn from each CE decile (loss matching)."""
    dec = pd.qcut(pd.Series(ce).rank(method="first"), 10, labels=False).to_numpy()
    a, b = [], []
    for d in range(10):
        pos, neg = np.nonzero((dec == d) & (y == 1))[0], np.nonzero((dec == d) & (y == 0))[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        a.append(vals[pos])
        b.append(vals[rng.choice(neg, len(pos), replace=len(neg) < len(pos))])
    return np.concatenate(a), np.concatenate(b)


def effects(cfg: dict, model: str, sf: int, target: str = "mem_T", seed: int = 0) -> pd.DataFrame:
    """H2 (Hedges' g of pruned interpretable T1 features) and H3 (Cliff's delta of T2 velocity),
    hard subset, both splits, loss-decile matched."""
    p, ac = paths(cfg), cfg["analysis"]
    out = p.results(model, sf)
    ft = FeatureTable(cfg, model, sf)
    cc = pd.read_parquet(p.data / f"cc_{model}.parquet")
    s = cc[(cc["target"] == target) & (cc["seed"] == seed)].set_index("index")
    s = s[s.index.isin(ft.tab.index)]
    s = s[ft.tab.loc[s.index, "mem_Sf"].to_numpy() == 0]
    y = s["label"].to_numpy()
    ce = ft.tab.loc[s.index, "ce_cont"].to_numpy()
    rng = np.random.default_rng(ac["seed"])
    layers = sorted({int(c.rsplit("_L", 1)[1]) for c in ft.fam["T1"]})
    first = list(H2_FEATURES) + [n for n in FEATURE_NAMES if n not in H2_FEATURES]
    rows = []
    for l in layers:
        cols = [f"t1_full_{n}_L{l}" for n in first]
        train = ft.tab.loc[s.index[s["split"] == "train"], cols]
        kept = prune_correlated(train, cols, ac["prune_corr"])
        for c in kept:
            a, b = _decile_matched(ft.tab.loc[s.index, c].to_numpy(), y, ce, rng)
            g, lo, hi = stats.bootstrap_two_sample(stats.hedges_g, a, b, ac["bootstrap"], ac["seed"])
            name = c[len("t1_full_"):].rsplit("_L", 1)[0]
            rows.append({"target": target, "family": "T1", "feature": name, "layer": l, "hedges_g": g, "lo": lo, "hi": hi,
                         "expected_sign": H2_FEATURES.get(name, 0), "n_pos": len(a), "n_neg": len(b)})
    vcols = [c for c in ft.fam["T2"] if "_h1_" in c]
    for c in vcols + ["v_mean_prepost_h1", "v_mean_postsf_h1"]:
        if c.startswith("v_mean_"):
            kind = c.split("_")[2]
            vals = ft.tab.loc[s.index, [v for v in vcols if v.startswith(f"v_{kind}_h1")]].mean(axis=1).to_numpy()
        else:
            vals = ft.tab.loc[s.index, c].to_numpy()
        ok = np.isfinite(vals)
        a, b = vals[ok & (y == 1)], vals[ok & (y == 0)]
        dlt, lo, hi = stats.bootstrap_two_sample(stats.cliffs_delta, a, b, ac["bootstrap"], ac["seed"])
        rows.append({"target": target, "family": "T2", "feature": c, "layer": None, "cliffs_delta": dlt,
                     "lo": lo, "hi": hi, "expected_sign": +1, "n_pos": len(a), "n_neg": len(b)})
    df = pd.DataFrame(rows)
    suffix = "" if target == "mem_T" else f"_{target}"
    df[df["family"] == "T1"].drop(columns=["cliffs_delta"], errors="ignore").to_csv(out / f"effects{suffix}.csv", index=False)
    df[df["family"] == "T2"].drop(columns=["hedges_g"], errors="ignore").to_csv(out / f"velocity{suffix}.csv", index=False)
    return df


def gaussian_floor(cfg: dict, model: str, sf: int, reps: int = 300) -> pd.DataFrame:
    """Control 3: H1 bar counts / entropy / max persistence of matched isotropic Gaussian clouds
    vs the real normalised clouds at S_f."""
    from memtopo.topology import gaussian_floor as gf

    p = paths(cfg)
    ft = FeatureTable(cfg, model, sf)
    d = ft.probe.shape[1] // len({int(c.rsplit("_L", 1)[1]) for c in ft.fam["T1"]})
    g = pd.DataFrame(gf(64, d, reps, seed=0), columns=FEATURE_NAMES)
    rows = []
    for l in sorted({int(c.rsplit("_L", 1)[1]) for c in ft.fam["T1"]}):
        for n in ("h1_n_bars", "h1_entropy", "h1_pers_max", "h1_total_pers"):
            real = ft.tab[f"t1_full_{n}_L{l}"].dropna()
            rows.append({"layer": l, "feature": n, "real_mean": real.mean(), "real_median": real.median(),
                         "gauss_mean": g[n].mean(), "gauss_p95": g[n].quantile(0.95),
                         "frac_real_above_gauss_p95": float((real > g[n].quantile(0.95)).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(p.results(model, sf) / "gaussian_floor.csv", index=False)
    record_gate(p, f"gaussian_floor_{model}_S{sf}", {"d": int(d), "reps": reps})
    return df
