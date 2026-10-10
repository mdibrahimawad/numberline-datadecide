"""T3: which points the H0 merges join. The finite H0 death times of a Vietoris-Rips filtration
are the edge lengths of the Euclidean minimum spanning tree, so MST edge homophily says
whether the cloud's connectivity follows topic or truth.

  T3a  MST of the whole cloud: fraction of edges joining same-topic points
  T3b  MST of each topic separately: fraction of edges joining same-truth points (mean over topics)
  T3c  topic means removed, MST of the pooled cloud: among cross-topic edges, fraction same-truth

All three are invariant to translating and rescaling the cloud, so no normalisation is needed.
Nulls permute truth labels within topic (T3a: topic labels over the whole cloud); the MST
itself never changes, so a permutation only re-reads labels on fixed edges.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from .topology import distances


def mst_edges(X: np.ndarray) -> np.ndarray:
    """(n-1, 2) edge list of the Euclidean MST. Exact duplicate points (distance 0) are kept
    connected via a tiny positive weight, since csgraph treats 0 as 'no edge'."""
    D = distances(X)
    off = ~np.eye(len(D), dtype=bool)
    D[off & (D <= 0)] = 1e-12
    T = minimum_spanning_tree(D).tocoo()
    return np.c_[T.row, T.col]


def homophily(edges: np.ndarray, lab: np.ndarray) -> float:
    if len(edges) == 0:
        return np.nan
    return float(np.mean(lab[edges[:, 0]] == lab[edges[:, 1]]))


class T3:
    """Holds the MSTs of one cell and evaluates homophily for any labelling."""

    def __init__(self, X: np.ndarray, topics: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.topics = topics
        self.full = mst_edges(X)
        self.within = []
        for t in np.unique(topics):
            idx = np.where(topics == t)[0]
            self.within.append(idx[mst_edges(X[idx])])
        Xc = X.copy()
        for t in np.unique(topics):
            Xc[topics == t] -= Xc[topics == t].mean(0)
        e = mst_edges(Xc)
        self.cross = e[topics[e[:, 0]] != topics[e[:, 1]]]

    def values(self, y: np.ndarray, topics: np.ndarray | None = None) -> dict:
        topics = self.topics if topics is None else topics
        return {"T3a_topic_homophily": homophily(self.full, topics),
                "T3b_within_truth_homophily": float(np.nanmean([homophily(e, y) for e in self.within])),
                "T3c_cross_truth_homophily": homophily(self.cross, y),
                "T3c_n_cross_edges": float(len(self.cross))}
