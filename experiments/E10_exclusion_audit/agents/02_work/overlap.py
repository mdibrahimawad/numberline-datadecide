"""Overlap (shared-document) similarity between the 25 training distributions and effective number of corpora.
S_AB = sum_p |A_p ∩ B_p| / sqrt(|A||B|)  (cosine between uniform densities over the two corpora).
Uses only composition (token counts), nothing model-side."""
import json, sys, numpy as np
rows = json.load(open(sys.argv[1]))
NODE = {("cc16",): 676.1, ("cc16", "v17"): 598.42, ("fal0",): 547.34, ("fal0", "v2"): 456.40,
        ("c4_16",): 174.42, ("c4_16", "v17"): 138.44, ("dclm",): 3857.12, ("fineweb",): 15000.0}
def rec_pools(r):
    s, f = r["recipe"], r["fine"]
    P = {}
    def add(pool, path, n): P[(pool, path)] = n
    if s == "dolma1_6plus":
        add("cc", ("cc16",), f["d16_cc"]); add("fal", ("fal0",), f["falcon_v0"]); add("c4", ("c4_16",), f["d16_c4"])
        for k, pool in [("d16_reddit", "reddit"), ("d16_pes2o", "pes2o"), ("d16_books", "books"), ("d16_wiki", "wiki"),
                        ("starcoder", "starcoder"), ("arxiv", "arxiv"), ("stackexchange", "se"),
                        ("algebraic_stack", "as"), ("openwebmath", "owm"), ("tulu_v0", "tulu0")]:
            add(pool, ("full",), f[k])
        return P
    m = {"web_rest": ("cc", ("cc16", "v17"), 598.42), "falcon": ("fal", ("fal0", "v2"), 456.40), "c4": ("c4", ("c4_16", "v17"), 138.44),
         "reddit": ("reddit", ("full",), 79.94), "pes20_stem_papers": ("pes2o", ("full",), 57.21), "gutenberg_books": ("books", ("full",), 5.256),
         "wikipedia_wikibooks": ("wiki", ("full",), 7.378), "starcoder": ("starcoder", ("full",), 263.78), "arxiv": ("arxiv", ("full",), 27.97),
         "stackexchange": ("se", ("full",), 19.63), "algebraic_stack": ("as", ("full",), 12.62), "openwebmath": ("owm", ("full",), 12.73),
         "tulu": ("tulu2", ("full",), 16.54), "cc_news": ("news", ("full",), 14.26), "megawika": ("mw", ("full",), 4.56),
         "dclm": ("dclm", ("dclm",), 3857.12)}
    mix = "dolma1.7" in s
    for k, n in f.items():
        if k in m:
            pool, path, full = m[k]
            if mix and n < full * 0.999:  # random file subset (sample_factor)
                path = path + (f"rand_{s}",)
            if pool == "wiki": n = n / 2  # 2x upsampled: unique docs
            add(pool, path, n)
    if s == "fineweb-edu": add("fw", ("fineweb", "edu"), f["fineweb-edu-dedup/v0"])
    if s == "fineweb-pro": add("fw", ("fineweb", "prox"), f["prox_fineweb_pro/v0"])
    qc = {"falcon-and-cc-qc-10p": ("eli5_20", "eli5_10"), "falcon-and-cc-qc-20p": ("eli5_20",),
          "falcon-and-cc-qc-orig-10p": ("orig10",), "falcon-and-cc-qc-tulu-10p": ("tulu10",)}
    if s in qc:
        for k, n in f.items():
            if k.startswith("olmo-mix/"): add("cc", ("cc16", "v17") + qc[s], n)
            else: add("fal", ("fal0", "v2") + qc[s], n)
    dq = {"dclm-baseline-qc-7p-fw2": ("ft7fw2",), "dclm-baseline-qc-7p-fw3": ("ft7fw2", "ft7fw3"),
          "dclm-baseline-qc-fw-3p": ("fw3p",), "dclm-baseline-qc-fw-10p": ("fw10p",),
          "dclm-baseline-qc-10p": ("pos20", "pos10"), "dclm-baseline-qc-20p": ("pos20",)}
    if s in dq: add("dclm", ("dclm",) + dq[s], list(f.values())[0])
    if s == "dclm-baseline": pass  # added via 'dclm' key? single-pool recipe stores mixture key
    for k, n in f.items():
        if k.startswith("fastdclm/") : add("dclm", ("dclm",), n)
        if k == "c4/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001-fix" and s == "c4": add("c4", ("c4_16", "v17"), n)
        if k.startswith("falcon-refinedweb/v2") and s in ("falcon", "falcon-and-cc"): add("fal", ("fal0", "v2"), n)
        if k.startswith("olmo-mix/v1_7") and s == "falcon-and-cc": add("cc", ("cc16", "v17"), n)
    return P

def node_size(path, pool, sizes):
    if path in NODE: return NODE[path]
    return sizes.get((pool, path), None)

def overlap(pa, na, pb, nb, pool, mode, nodes):
    if pa[:len(pb)] == pb or pb[:len(pa)] == pa: return min(na, nb)
    i = 0
    while i < min(len(pa), len(pb)) and pa[i] == pb[i]: i += 1
    anc = pa[:i]
    N = NODE.get(anc) or nodes.get((pool, anc))
    sib_a, sib_b = pa[i], pb[i]
    if mode == "upper" and not sib_a.startswith("rand") and not sib_b.startswith("rand") and pool != "fw":
        return min(na, nb)
    return min(na * nb / N, na, nb)

recs = [r["recipe"] for r in rows]
pools = [rec_pools(r) for r in rows]
nodes = {}
for P in pools:
    for (pool, path), n in P.items(): nodes.setdefault((pool, path), n)
tot = [sum(P.values()) for P in pools]
out = {}
for mode in ("lower", "upper"):
    S = np.zeros((25, 25))
    for i in range(25):
        for j in range(25):
            ov = 0.0
            for (pool, pa), na in pools[i].items():
                for (pool2, pb), nb in pools[j].items():
                    if pool2 == pool: ov += overlap(pa, na, pb, nb, pool, mode, nodes)
            S[i, j] = ov / np.sqrt(tot[i] * tot[j])
    neff = 25 ** 2 / (S ** 2).sum()
    ev = np.linalg.eigvalsh(S)[::-1]
    neff_pr = ev.sum() ** 2 / (ev ** 2).sum()
    out[mode] = dict(neff_sumsq=neff, neff_eig=neff_pr, top_eigs=ev[:8].round(3).tolist())
    np.savetxt(sys.argv[2] + f"_{mode}.csv", S, delimiter=",", fmt="%.3f", header=",".join(recs))
    print(mode, out[mode])
    # within-pool nearest neighbours
    for i, r in enumerate(recs):
        o = np.argsort(-S[i]); nn = [(recs[k], round(S[i, k], 2)) for k in o[1:4]]
        print(f"  {r:32s} {nn}")
# diagonal check
print([round(sum(P.values()) - t, 3) for P, t in zip(pools, tot)][:3])
print("pool totals vs composition totals:", [round(sum(P.values()) / r["corpus_B"], 3) for P, r in zip(pools, rows)])
