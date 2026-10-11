"""Pythia's pre-shuffled training data (EleutherAI/pile-deduped-pythia-preshuffled).

Layout assumed (and checked at run time, see ``check_layout`` and the alignment check):
the ``*.bin`` shards, in name order, are consecutive byte slices of one flat uint16 array in
which training sequence i occupies tokens [i * 2049, (i + 1) * 2049), and sequence i is
trained in step i // 1024. This is what Pythia's utils/unshard_memmap.py (concatenate the
shards) and utils/batch_viewer.py (index the merged memmap) imply. We never merge the
shards: a virtual reader maps global offsets to (shard, offset). Local shards are read with
np.memmap; shards not on disk are read with HTTP range requests (epoch-boundary search).
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("memtopo")


@dataclass
class Shard:
    name: str
    size: int          # bytes
    start: int         # global byte offset


@dataclass
class TokenStore:
    shards: list[Shard]
    seq_len: int = 2049
    batch_size: int = 1024
    itemsize: int = 2
    local_dir: Path | None = None
    repo: str | None = None
    _mm: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ geometry
    @property
    def total_bytes(self) -> int:
        return self.shards[-1].start + self.shards[-1].size

    @property
    def seq_bytes(self) -> int:
        return self.seq_len * self.itemsize

    @property
    def n_sequences(self) -> int:
        return self.total_bytes // self.seq_bytes

    def shards_for(self, lo: int, hi: int) -> list[Shard]:
        """Shards overlapping global byte range [lo, hi)."""
        starts = [s.start for s in self.shards]
        i = max(bisect.bisect_right(starts, lo) - 1, 0)
        out = []
        while i < len(self.shards) and self.shards[i].start < hi:
            out.append(self.shards[i])
            i += 1
        return out

    def steps_byte_range(self, first_step: int, last_step: int) -> tuple[int, int]:
        b = self.batch_size * self.seq_bytes
        return first_step * b, (last_step + 1) * b

    # ------------------------------------------------------------------ reading
    def _local(self, shard: Shard) -> np.memmap | None:
        if self.local_dir is None:
            return None
        if shard.name not in self._mm:
            f = self.local_dir / shard.name
            if not f.exists() or f.stat().st_size != shard.size:
                return None
            self._mm[shard.name] = np.memmap(f, dtype=np.uint8, mode="r")
        return self._mm[shard.name]

    def read_bytes(self, lo: int, hi: int) -> bytes:
        parts = []
        for s in self.shards_for(lo, hi):
            a, b = max(lo, s.start) - s.start, min(hi, s.start + s.size) - s.start
            mm = self._local(s)
            parts.append(bytes(mm[a:b]) if mm is not None else _http_range(self.repo, s.name, a, b))
        return b"".join(parts)

    def read(self, seq_index: int, n_tokens: int | None = None) -> np.ndarray:
        n = self.seq_len if n_tokens is None else n_tokens
        lo = seq_index * self.seq_bytes
        return np.frombuffer(self.read_bytes(lo, lo + n * self.itemsize), dtype=np.uint16).copy()

    def read_many(self, seq_indices: np.ndarray, n_tokens: int) -> np.ndarray:
        """First n_tokens of many sequences (local shards only; vectorised per shard)."""
        idx = np.asarray(seq_indices, dtype=np.int64)
        out = np.empty((len(idx), n_tokens), dtype=np.uint16)
        lo = idx * self.seq_bytes
        for s in self.shards:
            sel = np.nonzero((lo >= s.start) & (lo + n_tokens * self.itemsize <= s.start + s.size))[0]
            if sel.size == 0:
                continue
            mm = self._local(s)
            if mm is None:
                raise FileNotFoundError(f"{s.name} is not downloaded")
            u16 = mm[: (s.size // 2) * 2].view(np.uint16) if (s.start % 2 == 0) else None
            if u16 is None:
                raise ValueError("shard starts at an odd byte offset; layout differs from the brief")
            tok0 = (lo[sel] - s.start) // 2
            out[sel] = u16[tok0[:, None] + np.arange(n_tokens)[None, :]]
        # sequences straddling a shard boundary: slow path
        straddle = [i for i, l in enumerate(lo)
                    if len(self.shards_for(int(l), int(l) + n_tokens * self.itemsize)) > 1]
        for i in straddle:
            out[i] = self.read(int(idx[i]), n_tokens)
        return out

    def read_step(self, step: int) -> np.ndarray:
        """All 1024 sequences of one step, (1024, 2049) uint16 (one contiguous read)."""
        lo, hi = self.steps_byte_range(step, step)
        return np.frombuffer(self.read_bytes(lo, hi), dtype=np.uint16).reshape(self.batch_size, self.seq_len)


# ---------------------------------------------------------------------- HF access
def _token() -> str | None:
    return os.environ.get("HF_TOKEN")


def list_shards(repo: str) -> list[Shard]:
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True, token=_token())
             if getattr(f, "size", None) and f.path.endswith(".bin")]
    files.sort(key=lambda f: f.path)
    out, start = [], 0
    for f in files:
        out.append(Shard(f.path, int(f.size), start))
        start += int(f.size)
    return out


def _http_range(repo: str, filename: str, a: int, b: int) -> bytes:
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    url = hf_hub_url(repo, filename, repo_type="dataset")
    headers = build_hf_headers(token=_token())
    headers["Range"] = f"bytes={a}-{b - 1}"
    for attempt in range(5):
        try:
            r = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
            if r.status_code == 206 and len(r.content) == b - a:
                return r.content
            raise IOError(f"range read {filename}[{a}:{b}] -> HTTP {r.status_code}, {len(r.content)} bytes")
        except Exception as e:  # noqa: BLE001 - network: retry with backoff
            if attempt == 4:
                raise
            log.warning("retrying range read (%s)", e)
            import time
            time.sleep(2 ** (attempt + 1))
    raise AssertionError


def download_shards(store: TokenStore, shards: list[Shard]) -> None:
    from huggingface_hub import hf_hub_download

    for s in shards:
        f = store.local_dir / s.name
        if f.exists() and f.stat().st_size == s.size:
            continue
        log.info("downloading %s (%.1f GB)", s.name, s.size / 1e9)
        hf_hub_download(store.repo, s.name, repo_type="dataset", local_dir=store.local_dir, token=_token())
        assert f.stat().st_size == s.size, f"{s.name}: size mismatch after download"


def check_layout(store: TokenStore) -> dict:
    """Facts about the layout, recorded in gates.json: shard count/sizes, whether the total
    is a whole number of 2049-token sequences, and how many steps it holds."""
    rem = store.total_bytes % store.seq_bytes
    return {
        "n_shards": len(store.shards),
        "shard_names": [s.name for s in store.shards[:3]] + ["..."],
        "total_bytes": store.total_bytes,
        "whole_sequences": rem == 0,
        "remainder_bytes": rem,
        "n_sequences": store.n_sequences,
        "n_steps": store.n_sequences / store.batch_size,
    }


# ---------------------------------------------------------------------- epoch boundary
def hash_rows(rows: np.ndarray) -> np.ndarray:
    """64-bit polynomial hash of each row of a 2-D token array (uint64 wraparound)."""
    rows = np.asarray(rows)
    h = np.zeros(len(rows), dtype=np.uint64)
    base = np.uint64(1_000_003)
    with np.errstate(over="ignore"):
        for j in range(rows.shape[1]):
            h = h * base + rows[:, j].astype(np.uint64) + np.uint64(1)
    return h


def shingle_hashes(tokens: np.ndarray, k: int) -> np.ndarray:
    """Hashes of every k-gram of every row, shape (n_rows, seq_len - k + 1)."""
    t = np.atleast_2d(np.asarray(tokens, dtype=np.uint16))
    win = np.lib.stride_tricks.sliding_window_view(t, k, axis=1)       # (n, w, k)
    return hash_rows(win.reshape(-1, k)).reshape(win.shape[0], win.shape[1])


def build_reference(store: TokenStore, steps: range, k: int, per_seq: int = 4) -> np.ndarray:
    """Sorted unique hashes of ``per_seq`` evenly spaced k-grams per sequence from ``steps``;
    hashes occurring twice already within the reference (boilerplate) are removed."""
    pos = np.linspace(0, store.seq_len - k, per_seq).astype(int)
    hs = []
    for st in steps:
        x = store.read_step(st)
        hs.append(hash_rows(np.stack([x[:, p:p + k] for p in pos], axis=1).reshape(-1, k)))
    u, c = np.unique(np.concatenate(hs), return_counts=True)
    return u[c == 1]


def _hits_per_seq(ref: np.ndarray, tokens: np.ndarray, k: int) -> np.ndarray:
    h = shingle_hashes(tokens, k)
    i = np.searchsorted(ref, h)
    i[i == len(ref)] = 0
    return (ref[i] == h).sum(axis=1)


def step_hits(store: TokenStore, ref: np.ndarray, step: int, k: int) -> int:
    return int(_hits_per_seq(ref, store.read_step(step), k).sum())


def find_epoch_boundary(store: TokenStore, ref: np.ndarray, k: int, start: int, stop: int,
                        every: int, baseline_steps: list[int]) -> dict:
    """First step containing re-seen (second-epoch) text: coarse scan, then bisection.
    Epoch-1 steps of the deduplicated Pile should hit the reference ~0 times; second-epoch
    steps hit it about (reference steps / epoch steps) x per_seq x 1024 times."""
    base = {s: step_hits(store, ref, s, k) for s in baseline_steps}
    thr = max(20, 5 * max(base.values()))
    coarse = {}
    first = None
    for s in range(start, min(stop, int(store.n_sequences // store.batch_size)), every):
        coarse[s] = step_hits(store, ref, s, k)
        log.info("epoch scan step %d: %d hits (threshold %d)", s, coarse[s], thr)
        if coarse[s] > thr:
            first = s
            break
    if first is None:
        return {"found": False, "baseline_hits": base, "coarse_hits": coarse, "threshold": thr}
    lo, hi = first - every, first                 # lo below threshold, hi above
    hits = dict(coarse)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        hits[mid] = step_hits(store, ref, mid, k)
        lo, hi = (lo, mid) if hits[mid] > thr else (mid, hi)
    after = {s: step_hits(store, ref, s, k) for s in (hi + 1, hi + 50, hi + 500, hi + 5000)
             if s < store.n_sequences // store.batch_size}
    # first sequence with any hit, looking one step back (the switch can fall mid-step)
    first_seq = None
    for st in (hi - 1, hi):
        per = _hits_per_seq(ref, store.read_step(st), k)
        nz = np.nonzero(per)[0]
        if nz.size:
            first_seq = int(st * store.batch_size + nz[0])
            break
    return {"found": True, "boundary_step": hi, "first_hit_sequence": first_seq,
            "baseline_hits": base, "threshold": thr, "hits": hits, "hits_after": after,
            "primary_target_T": (hi // 1000) * 1000}


def digest(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
