"""Linear measures: L1 logistic probe, L2 mass-mean probe, L3 cross-topic transfer, B1 loglik AUC."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C.LR_C, max_iter=C.LR_MAX_ITER))


def _auc(y, s):
    return 0.5 if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def _folds(y, strata, seed):
    return list(StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), strata))


def l1_l2_cv(X, y, strata, seed=0):
    """Out-of-fold scores of the logistic (L1) and mass-mean (L2) probes -> (auc_l1, auc_l2)."""
    s1, s2 = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in _folds(y, strata * 2 + y, seed):
        if len(np.unique(y[tr])) < 2:
            continue
        s1[te] = _lr().fit(X[tr], y[tr]).decision_function(X[te])
        d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        s2[te] = X[te] @ d
    return _auc(y, s1), _auc(y, s2)


def l1_split(X, y, strata, seed=0):
    """Qian et al.: one random 4:1 split, test accuracy (and AUC)."""
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=strata * 2 + y)
    m = _lr().fit(X[tr], y[tr])
    return float(accuracy_score(y[te], m.predict(X[te]))), _auc(y[te], m.decision_function(X[te]))


def l1_within_topic(X, y, topics, seed=0):
    """L1 trained and tested within each topic (5-fold), averaged over topics."""
    out = []
    for t in np.unique(topics):
        i = np.where(topics == t)[0]
        out.append(l1_l2_cv(X[i], y[i], np.zeros(len(i), int), seed)[0])
    return float(np.mean(out))


def l3_cross_topic(X, y, topics):
    """Train L1 on one topic, test on each other topic; mean AUC over ordered pairs."""
    ts = np.unique(topics)
    if len(ts) < 2:
        return np.nan
    aucs = []
    for a in ts:
        m = _lr().fit(X[topics == a], y[topics == a])
        for b in ts:
            if b != a:
                aucs.append(_auc(y[topics == b], m.decision_function(X[topics == b])))
    return float(np.mean(aucs))


def linear_cell(X, y, topics, n_null, rng):
    """All linear measures of one cell, with shuffled-label nulls. Returns a list of metric dicts."""
    X = np.asarray(X, dtype=np.float32)
    strata = np.unique(topics, return_inverse=True)[1]
    a1, a2 = l1_l2_cv(X, y, strata)
    acc, sauc = l1_split(X, y, strata)
    wt = l1_within_topic(X, y, topics)
    l3 = l3_cross_topic(X, y, topics)
    null1, null2, nullwt, null3 = [], [], [], []
    for _ in range(n_null):
        ys = _shuffle_within(y, strata, rng)
        n1, n2 = l1_l2_cv(X, ys, strata)
        null1.append(n1); null2.append(n2)
        nullwt.append(l1_within_topic(X, ys, topics))
        null3.append(l3_cross_topic(X, ys, topics))
    return [
        _m("L1_auc", a1, null1), _m("L2_auc", a2, null2), _m("L1_within_auc", wt, nullwt),
        _m("L3_auc", l3, null3), _m("L1_split_acc", acc), _m("L1_split_auc", sauc),
    ]


def _shuffle_within(y, strata, rng):
    ys = y.copy()
    for s in np.unique(strata):
        i = np.where(strata == s)[0]
        ys[i] = rng.permutation(y[i])
    return ys


def _m(name, value, null=None, upper=True):
    """Metric record with a z-score and a one-sided p from a normal fit to the null."""
    from scipy.stats import norm
    r = {"metric": name, "value": value, "null_mean": np.nan, "null_sd": np.nan, "z": np.nan, "p": np.nan}
    if null is not None and len(null) and not np.isnan(value):
        nm, ns = float(np.nanmean(null)), float(np.nanstd(null, ddof=1)) if len(null) > 1 else np.nan
        r.update(null_mean=nm, null_sd=ns)
        if ns and ns > 0:
            r["z"] = (value - nm) / ns
            r["p"] = float(norm.sf(r["z"]))
    return r


def auc_within_topics(score, y, topics):
    return float(np.mean([_auc(y[topics == t], score[topics == t]) for t in np.unique(topics)]))


def b1(loglik, y, topics, n_perm, rng):
    """B1: AUC of the statement log-likelihood for true vs false, within topic, averaged."""
    strata = np.unique(topics, return_inverse=True)[1]
    v = auc_within_topics(loglik, y, topics)
    null = [auc_within_topics(loglik, _shuffle_within(y, strata, rng), topics) for _ in range(n_perm)]
    return _m("B1_loglik_auc", v, null)
