"""Stages 2-3 and feature extraction of the brief (section 10). Every stage is resumable:
it skips outputs that already exist. Facts and gates go to <out_root>/gates.json."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from memtopo import pile, scan, textstats, topology as topo
from memtopo.config import layers_for, model_spec, paths, read_gates, record_gate

log = logging.getLogger("memtopo")
VOCAB = 65536                     # uint16 range; Pythia's vocabulary (50,304 rows) fits inside


# ---------------------------------------------------------------------- shared helpers
def token_store(cfg: dict, remote_only: bool = False) -> pile.TokenStore:
    p = paths(cfg)
    d = cfg["data"]
    shards = pile.list_shards(d["repo"])
    return pile.TokenStore(shards, d["seq_len"], d["batch_size"], 2,
                           None if remote_only else p.shards, d["repo"])


def load_pool(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(paths(cfg).pool)


def tokens_map(pool: pd.DataFrame) -> dict:
    return dict(zip(pool["index"].to_numpy(), pool["tokens64"].to_numpy()))


def evals_indices(cfg: dict, spec: dict) -> pd.DataFrame:
    """Memorized sequences at step 143000 from EleutherAI/pythia-memorized-evals (cached)."""
    p = paths(cfg)
    f = p.data / f"evals_{spec['key']}.parquet"
    if not f.exists():
        from datasets import get_dataset_split_names, load_dataset

        splits = get_dataset_split_names(cfg["labels"]["evals_repo"])
        record_gate(p, "evals_splits", splits)
        if spec["evals_split"] not in splits:
            raise ValueError(f"split {spec['evals_split']} not in {splits}")
        ds = load_dataset(cfg["labels"]["evals_repo"], split=spec["evals_split"])
        df = ds.to_pandas()
        df["tokens"] = df["tokens"].map(lambda t: np.asarray(t, dtype=np.uint16))
        df.to_parquet(f)
    return pd.read_parquet(f)


def step_range_indices(cfg: dict, steps) -> tuple[int, int]:
    b = cfg["data"]["batch_size"]
    return steps[0] * b, (steps[1] + 1) * b


# ---------------------------------------------------------------------- stage: data
def stage_data(cfg: dict, skip_epoch: bool = False) -> None:
    p, d = paths(cfg), cfg["data"]
    store = token_store(cfg)
    layout = pile.check_layout(store)
    record_gate(p, "layout", layout)
    log.info("layout: %s", layout)
    if not layout["whole_sequences"]:
        log.warning("total size is not a whole number of %d-token sequences: check the layout", d["seq_len"])

    # shards covering every locally needed step (pool, unigram, epoch reference)
    first = min(d["pool_steps"][0], d["unigram_steps"][0], d["epoch_ref_steps"][0])
    last = max(d["pool_steps"][1], d["unigram_steps"][1], d["epoch_ref_steps"][1])
    need = store.shards_for(*store.steps_byte_range(first, last))
    log.info("need shards %s (%.1f GB)", [s.name for s in need], sum(s.size for s in need) / 1e9)
    pile.download_shards(store, need)

    # epoch boundary (section 4)
    gates = read_gates(p)
    if not skip_epoch and not gates.get("epoch_boundary", {}).get("found"):
        k = d["shingle_len"]
        log.info("building reference shingles from steps %s", d["epoch_ref_steps"])
        ref = pile.build_reference(store, range(d["epoch_ref_steps"][0], d["epoch_ref_steps"][1] + 1), k)
        es = d["epoch_scan"]
        assert min(es["baseline_steps"]) > d["epoch_ref_steps"][1], "baseline steps must lie outside the reference"
        res = pile.find_epoch_boundary(store, ref, k, es["start"], es["stop"], es["every"], es["baseline_steps"])
        if not res["found"]:
            log.warning("no boundary in [%d, %d): widening to the whole run", es["start"], es["stop"])
            res = pile.find_epoch_boundary(store, ref, k, 6000, 143000, 2000, es["baseline_steps"])
        res["reference_size"] = int(len(ref))
        record_gate(p, "epoch_boundary", res)
        log.info("epoch boundary: %s", res)

    # pool (section 5.2)
    if not p.pool.exists():
        lo, hi = step_range_indices(cfg, d["pool_steps"])
        rng = np.random.default_rng(d["pool_seed"])
        idx = np.sort(rng.choice(np.arange(lo, hi), size=d["pool_size"], replace=False))
        toks = store.read_many(idx, 64)
        steps = idx // d["batch_size"]
        split = np.where(steps <= d["train_steps"][1], "train", "test")
        pool = pd.DataFrame({"index": idx, "step": steps, "split": split, "tokens64": list(toks)})
        pool.to_parquet(p.pool)
        record_gate(p, "pool", {"n": len(pool), "train": int((split == "train").sum()),
                                "test": int((split == "test").sum()), "digest": pile.digest(toks)})

    # unigram frequencies and B4 (section 6.1)
    lf = p.data / "unigram_logfreq.npy"
    if not lf.exists():
        np.save(lf, textstats.unigram_logfreq(store, tuple(d["unigram_steps"]), VOCAB))
    b4 = p.data / "b4.parquet"
    if not b4.exists():
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg["models"]["1b"]["name"])
        logfreq = np.load(lf)
        pool = load_pool(cfg)
        rows = [textstats.b4_row(t, tok.decode(t.tolist()), logfreq, cfg["labels"]["prompt_len"])
                for t in pool["tokens64"]]
        pd.DataFrame(rows, columns=textstats.B4_NAMES).assign(index=pool["index"].to_numpy()) \
            .to_parquet(b4)


def stage_align(cfg: dict, model: str) -> dict:
    """Mandatory alignment check: first 64 tokens read from the shards == memorized-evals tokens."""
    p, d = paths(cfg), cfg["data"]
    spec = model_spec(cfg, model)
    store = token_store(cfg)
    ev = evals_indices(cfg, spec)
    lo, hi = step_range_indices(cfg, (0, d["pool_steps"][1]))
    cand = ev[(ev["index"] >= lo) & (ev["index"] < hi)]
    pick = cand.sample(n=min(d["alignment_n"], len(cand)), random_state=0)
    got = store.read_many(pick["index"].to_numpy(), 64)
    exp = np.stack([np.asarray(t, dtype=np.uint16)[:64] for t in pick["tokens"]])
    ok = (got == exp).all(1)
    # diagnose off-by-one / shifted indexing if it fails
    shift = {}
    if not ok.all():
        for off in (-1, 1, -1024, 1024):
            g2 = store.read_many(pick["index"].to_numpy() + off, 64)
            shift[off] = int((g2 == exp).all(1).sum())
    res = {"model": model, "n_checked": int(len(pick)), "n_match": int(ok.sum()),
           "n_evals_in_range": int(len(cand)), "n_evals_total": int(len(ev)), "shift_diagnosis": shift}
    record_gate(p, f"alignment_{model}", res)
    log.info("alignment: %s", res)
    if not ok.all():
        raise SystemExit("ALIGNMENT FAILED: fix the indexing before continuing (see gates.json)")
    return res


# ---------------------------------------------------------------------- stage: labels
def target_T(cfg: dict) -> int:
    eb = read_gates(paths(cfg)).get("epoch_boundary", {})
    if not eb.get("found"):
        raise SystemExit("epoch boundary not determined yet: run `memtopo.cli data` first")
    return int(eb["primary_target_T"])


def stage_labels(cfg: dict, model: str) -> None:
    p, lc = paths(cfg), cfg["labels"]
    spec = model_spec(cfg, model)
    pool = load_pool(cfg)
    tmap = tokens_map(pool)
    all_idx = pool["index"].to_numpy()
    T = target_T(cfg)
    run = lambda step, idx: scan.scan(spec["name"], step, tmap, idx, [], p.scans(model, step), [1],
                                      lc["prompt_len"], lc["batch"], lc["precision"])
    for step in [*spec["forecast_steps"], T]:
        run(step, all_idx)
        scan.free_revision(spec["name"], step)

    ev = evals_indices(cfg, spec)
    in_evals = set(ev["index"].tolist())
    lab = pd.DataFrame({"index": all_idx, "split": pool["split"].to_numpy(), "step": pool["step"].to_numpy()})
    lab["mem_final"] = lab["index"].isin(in_evals).astype(int)
    lt = scan.read_parts(p.scans(model, T), "loss").set_index("index")
    lab["mem_T"] = lt.loc[all_idx, "mem"].to_numpy().astype(int)
    for sf in spec["forecast_steps"]:
        ls = scan.read_parts(p.scans(model, sf), "loss").set_index("index")
        lab[f"mem_S{sf}"] = ls.loc[all_idx, "mem"].to_numpy().astype(int)
        lab[f"match_len_S{sf}"] = ls.loc[all_idx, "match_len"].to_numpy().astype(int)
    lab.to_parquet(p.data / f"labels_{model}.parquet")

    # label fragility (section 5.3): recompute at 143000
    rng = np.random.default_rng(1)
    rand = rng.choice(all_idx, size=min(lc["fragility_n"], len(all_idx)), replace=False)
    listed = lab.loc[lab["mem_final"] == 1, "index"].to_numpy()
    listed = rng.choice(listed, size=min(lc["fragility_n"], len(listed)), replace=False)
    fin = lc["secondary_target"]
    run(fin, np.r_[rand, listed])
    lf = scan.read_parts(p.scans(model, fin), "loss").set_index("index")
    sub = lab.set_index("index")
    a = lf.loc[rand, "mem"].to_numpy() == sub.loc[rand, "mem_final"].to_numpy()
    frag = {"random_n": int(len(rand)), "random_agreement": float(a.mean()),
            "random_listed": int(sub.loc[rand, "mem_final"].sum()),
            "listed_n": int(len(listed)), "listed_reverified": float(lf.loc[listed, "mem"].mean()) if len(listed) else None}
    record_gate(p, f"fragility_{model}", frag)
    log.info("fragility: %s", frag)
    scan.free_revision(spec["name"], fin)

    # teacher-forced extractability vs real greedy decoding
    sf0 = spec["forecast_steps"][0]
    pos = lab.loc[lab[f"mem_S{sf0}"] == 1, "index"].to_numpy()
    chk = np.r_[rng.choice(all_idx, lc["greedy_check_n"] // 2, replace=False),
                rng.choice(pos, min(len(pos), lc["greedy_check_n"] // 2), replace=False)]
    g = scan.greedy_check(spec["name"], sf0, np.stack([tmap[i] for i in chk]), lc["prompt_len"],
                          lc["cont_len"], lc["batch"], lc["precision"])
    record_gate(p, f"greedy_check_{model}_S{sf0}", g)
    log.info("greedy check: %s", g)

    # prevalence gate + Biderman B0 numbers (section 7, last bullet)
    gate = {"T": T, "prevalence_mem_T": float(lab["mem_T"].mean()),
            "prevalence_mem_final": float(lab["mem_final"].mean()), "per_Sf": {}}
    from memtopo.stats import precision_recall_binary
    for sf in spec["forecast_steps"]:
        hard_test_pos = int(((lab[f"mem_S{sf}"] == 0) & (lab["split"] == "test") & (lab["mem_T"] == 1)).sum())
        pT = precision_recall_binary(lab["mem_T"], lab[f"mem_S{sf}"])
        pF = precision_recall_binary(lab["mem_final"], lab[f"mem_S{sf}"])
        gate["per_Sf"][sf] = {"prevalence_mem_Sf": float(lab[f"mem_S{sf}"].mean()),
                              "hard_test_positives_mem_T": hard_test_pos,
                              "gate_ge_300": hard_test_pos >= 300,
                              "B0_vs_mem_T": {"precision": pT[0], "recall": pT[1]},
                              "B0_vs_mem_final": {"precision": pF[0], "recall": pF[1]}}
    record_gate(p, f"labels_{model}", gate)
    log.info("labels gate: %s", gate)


# ---------------------------------------------------------------------- stage: case-control
def stage_casecontrol(cfg: dict, model: str) -> pd.DataFrame:
    """For each target (mem_T, mem_final) and seed: all positives + neg_ratio x as many random
    negatives, stratified by split. Expensive features are computed for the union."""
    p, cc = paths(cfg), cfg["case_control"]
    f = p.data / f"cc_{model}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    lab = pd.read_parquet(p.data / f"labels_{model}.parquet")
    rows = []
    for target in ("mem_T", "mem_final"):
        for seed in cc["seeds"]:
            rng = np.random.default_rng(1000 + seed)
            for split in ("train", "test"):
                s = lab[lab["split"] == split]
                pos = s.loc[s[target] == 1, "index"].to_numpy()
                neg_all = s.loc[s[target] == 0, "index"].to_numpy()
                neg = rng.choice(neg_all, size=min(len(neg_all), cc["neg_ratio"] * len(pos)), replace=False)
                for idx, y in ((pos, 1), (neg, 0)):
                    rows.append(pd.DataFrame({"index": idx, "target": target, "seed": seed, "split": split,
                                              "label": y, "neg_pool": len(neg_all), "neg_drawn": len(neg)}))
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(f)
    record_gate(p, f"casecontrol_{model}", {"union": int(df["index"].nunique()),
                                            "by": df.groupby(["target", "seed", "split", "label"]).size()
                                            .rename("n").reset_index().to_dict("records")})
    return df


# ---------------------------------------------------------------------- stage: scans
def exposure_ckpts(step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Checkpoint just before a sequence's step (floor to 1000s) and the next 1000 after it.
    Checkpoint N has trained on steps 0..N-1, so pre has not seen the sequence, post has."""
    pre = (np.asarray(step) // 1000) * 1000
    return pre, pre + 1000


def stage_scans(cfg: dict, model: str, sf: int) -> None:
    p, lc, fc = paths(cfg), cfg["labels"], cfg["features"]
    spec = model_spec(cfg, model)
    pool = load_pool(cfg)
    tmap = tokens_map(pool)
    cc_idx = set(stage_casecontrol(cfg, model)["index"].tolist())
    from transformers import AutoConfig
    n_layers = AutoConfig.from_pretrained(spec["name"]).num_hidden_layers
    layers = layers_for(n_layers, fc["ref_layers_16"])
    record_gate(p, f"layers_{model}", {"num_hidden_layers": n_layers, "layers": layers})
    pre, post = exposure_ckpts(pool["step"].to_numpy())
    idx = pool["index"].to_numpy()
    common = dict(layers=layers, prompt_len=lc["prompt_len"], batch=lc["batch"], dtype=lc["precision"],
                  also_raw=fc["also_raw"], workers=fc["ph_workers"])
    for ck in sorted(set(pre) | set(post)):
        assert ck <= sf, "leakage: exposure checkpoint after S_f"
        sel = idx[(pre == ck) | (post == ck)]
        scan.scan(spec["name"], int(ck), tmap, sel, [i for i in sel if i in cc_idx], p.scans(model, int(ck)), **common)
        scan.free_revision(spec["name"], int(ck))
    scan.scan(spec["name"], 0, tmap, [], sorted(cc_idx), p.scans(model, 0), **common)          # T0
    scan.free_revision(spec["name"], 0)
    scan.scan(spec["name"], sf, tmap, idx, sorted(cc_idx), p.scans(model, sf), probe=True, **common)
    scan.free_revision(spec["name"], sf)


# ---------------------------------------------------------------------- stage: features
def _wide(df: pd.DataFrame, cols: list[str], prefix: str = "") -> pd.DataFrame:
    w = df.pivot(index="index", columns="layer", values=cols)
    w.columns = [f"{prefix}{c}_L{l}" for c, l in w.columns]
    return w


def _dgm(row, key):
    return np.asarray(row[key], dtype=np.float64).reshape(-1, 2)


def _velocity_job(args):
    rows_a, rows_b, wp = args
    return [(topo.wasserstein(_dgm(a, "h1"), _dgm(b, "h1"), wp), topo.wasserstein(_dgm(a, "h0"), _dgm(b, "h0"), wp))
            for a, b in zip(rows_a, rows_b)]


def _pair_distances(da: pd.DataFrame, db: pd.DataFrame, wp: float, workers: int) -> pd.DataFrame:
    import concurrent.futures as cf
    import multiprocessing as mp

    m = da.merge(db, on=["index", "layer"], suffixes=("_a", "_b"))
    a = m[["h0_a", "h1_a"]].rename(columns=lambda c: c[:-2]).to_dict("records")
    b = m[["h0_b", "h1_b"]].rename(columns=lambda c: c[:-2]).to_dict("records")
    n = max(1, len(a) // (workers * 4) + 1)
    jobs = [(a[i:i + n], b[i:i + n], wp) for i in range(0, len(a), n)]
    with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        res = [r for chunk in ex.map(_velocity_job, jobs) for r in chunk]
    m["w_h1"], m["w_h0"] = [r[0] for r in res], [r[1] for r in res]
    return m[["index", "layer", "w_h1", "w_h0"]]


def stage_features(cfg: dict, model: str, sf: int) -> None:
    p, fc = paths(cfg), cfg["features"]
    out = p.features(model, sf)
    out.mkdir(parents=True, exist_ok=True)
    pool = load_pool(cfg)
    idx = pool["index"].to_numpy()
    lab = pd.read_parquet(p.data / f"labels_{model}.parquet").set_index("index")
    pre, post = exposure_ckpts(pool["step"].to_numpy())
    used = sorted(set(pre) | set(post) | {0, sf})
    assert max(used) <= sf, f"leakage: features would use checkpoint {max(used)} > S_f={sf}"
    record_gate(p, f"feature_checkpoints_{model}_S{sf}", used)

    # loss family + B3 (all pooled sequences)
    if not (out / "loss.parquet").exists():
        ls = scan.read_parts(p.scans(model, sf), "loss").set_index("index").loc[idx]
        f = pd.DataFrame(index=pd.Index(idx, name="index"))
        f["mem_Sf"] = ls["mem"].to_numpy()
        f["match_len"] = ls["match_len"].to_numpy()
        f["match_frac"] = ls["match_frac"].to_numpy()
        for c in ("ce_cont", "ce_prompt", "mink20", "ent_cont_mean", "ent_first", "logp_first"):
            f[c] = ls[c].to_numpy()
        ce = {}
        for ck in sorted(set(pre) | set(post)):
            ce[ck] = scan.read_parts(p.scans(model, int(ck)), "loss").set_index("index")["ce_cont"]
        f["ce_pre"] = [ce[a].get(i, np.nan) for a, i in zip(pre, idx)]
        f["ce_post"] = [ce[b].get(i, np.nan) for b, i in zip(post, idx)]
        f["exposure_drop"] = f["ce_pre"] - f["ce_post"]
        f["post_slope"] = (f["ce_cont"] - f["ce_post"]) / (sf - post)
        f.reset_index().to_parquet(out / "loss.parquet")
    if not (out / "data.parquet").exists():
        pd.read_parquet(p.data / "b4.parquet").to_parquet(out / "data.parquet")

    h_sf = scan.read_parts(p.scans(model, sf), "hidden")
    t1_cols = [c for c in h_sf.columns if c.startswith("t1_full_") and not c.startswith("t1_full_raw_")] + \
              [c for c in h_sf.columns if c.startswith("t1_cont_")]
    raw_cols = [c for c in h_sf.columns if c.startswith("t1_full_raw_")]
    geom_cols = [c for c in h_sf.columns if c.startswith("geom_")]
    if not (out / "geom.parquet").exists():
        _wide(h_sf, geom_cols).reset_index().to_parquet(out / "geom.parquet")
    if not (out / "topo_T1.parquet").exists():
        _wide(h_sf, t1_cols).reset_index().to_parquet(out / "topo_T1.parquet")
        if raw_cols:
            _wide(h_sf, raw_cols).reset_index().to_parquet(out / "topo_T1raw.parquet")
    if not (out / "topo_T0.parquet").exists():
        h0 = scan.read_parts(p.scans(model, 0), "hidden")
        _wide(h0, t1_cols, prefix="t0_").reset_index().to_parquet(out / "topo_T0.parquet")
    if not (out / "probe.npz").exists():
        pi, ps = scan.read_probe(p.scans(model, sf))
        np.savez(out / "probe.npz", index=pi, states=ps)

    # T2 velocity (pre -> post, post -> S_f) and T3 cross-layer change (at S_f)
    if not (out / "topo_T2.parquet").exists():
        cc = set(h_sf["index"].tolist())
        sel = pool[pool["index"].isin(cc)]
        spre, spost = exposure_ckpts(sel["step"].to_numpy())
        dg = {ck: scan.read_parts(p.scans(model, int(ck)), "dgm") for ck in sorted(set(spre) | set(spost))}
        dsf = scan.read_parts(p.scans(model, sf), "dgm")
        a_parts, b_parts = [], []
        for ck in sorted(set(spre)):
            ii = sel["index"].to_numpy()[spre == ck]
            a_parts.append(dg[ck][dg[ck]["index"].isin(ii)])
            b_parts.append(dg[ck + 1000][dg[ck + 1000]["index"].isin(ii)])
        da, db = pd.concat(a_parts), pd.concat(b_parts)
        d1 = _pair_distances(da, db, fc["wasserstein_p"], fc["ph_workers"]).rename(
            columns={"w_h1": "v_prepost_h1", "w_h0": "v_prepost_h0"})
        d2 = _pair_distances(db, dsf[dsf["index"].isin(cc)], fc["wasserstein_p"], fc["ph_workers"]).rename(
            columns={"w_h1": "v_postsf_h1", "w_h0": "v_postsf_h0"})
        t2 = _wide(d1.merge(d2, on=["index", "layer"]), ["v_prepost_h1", "v_prepost_h0", "v_postsf_h1", "v_postsf_h0"])
        t2.reset_index().to_parquet(out / "topo_T2.parquet")
        # T3: adjacent selected layers at S_f
        layers = sorted(dsf["layer"].unique())
        pairs = []
        for la, lb in zip(layers[:-1], layers[1:]):
            x = dsf[dsf["layer"] == la].assign(layer=0)
            y = dsf[dsf["layer"] == lb].assign(layer=0)
            pairs.append(_pair_distances(x, y, fc["wasserstein_p"], fc["ph_workers"])
                         .assign(pair=f"{la}-{lb}"))
        t3 = pd.concat(pairs).groupby("index")["w_h1"].agg(["mean", "max"])
        t3.columns = ["t3_h1_mean", "t3_h1_max"]
        t3.reset_index().to_parquet(out / "topo_T3.parquet")
    lab.reset_index().to_parquet(out / "labels.parquet")
    log.info("features for %s S%d written to %s", model, sf, out)
