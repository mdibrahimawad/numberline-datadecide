"""Per-recipe composition table + overlap-based effective number of corpora.
Blind to beta: reads only mixture.json, data map, confounders 'recipes' (code/math/number_density), count_features (non-beta cols)."""
import json, csv, math, sys
import numpy as np
REPO = "/home/user/numberline-datadecide"
recipes = [r["recipe"] for r in csv.DictReader(open(f"{REPO}/results/datadecide/alpha_beta.csv"))]
dmap = json.load(open(f"{REPO}/configs/datadecide_data_map.json"))["recipes"]
# exact Dolma1.7 source sizes (tokens, incl. 2x wiki) from OLMo@DataDecide named_data_mixes.py:4995 SOURCES_SIZES
SZ = dict(gutenberg_books=5_255_792_107, pes20_stem_papers=57_209_752_742, wikipedia_wikibooks=7_378_409_050,
          megawika=4_562_102_218, stackexchange=19_629_583_356, arxiv=27_973_662_590, algebraic_stack=12_622_576_536,
          openwebmath=12_734_300_603, tulu=16_536_907_601, cc_news=14_261_416_567, starcoder=263_775_304_843,
          c4=138_441_178_003, reddit=79_944_013_060, falcon=456_404_959_027, web_rest=598_422_659_025)
NF = dict(gutenberg_books=3, pes20_stem_papers=26, wikipedia_wikibooks=4, megawika=61, stackexchange=26, arxiv=32,
          algebraic_stack=16, openwebmath=13, tulu=66, cc_news=33, starcoder=93, c4=171, reddit=78, falcon=188, web_rest=223)
# Dolma 1.6++ v1_6 sub-sizes from comments named_data_mixes.py:2559-3864 (GT)
D16 = dict(books=5.256, pes2o=57.21, wiki=3.689, c4=174.418, reddit=79.988, cc=192.264+189.606+294.252)

def fine_tokens(slug):
    """Return dict fine_source -> tokens (B)."""
    mix = json.load(open(f"{REPO}/results/corpus_alpha_datadecide/{slug}/mixture.json"))
    T = {k: v["tokens"] / 1e9 for k, v in mix.items()}
    fps = dict(dmap[slug]["files_per_source"])
    if "c4_debug" in fps: fps["c4"] = fps.get("c4", 0) + fps.pop("c4_debug")
    out = {}
    if slug == "dolma1_6plus":
        v16 = T["olmo-mix/v1_6-decontaminated"]; s = sum(D16.values())
        for k, v in D16.items(): out["d16_" + k] = v * v16 / s
        rp = T["redpajama/v1_decon_fix"]; out["arxiv"] = rp * 27.97 / 47.60; out["stackexchange"] = rp * 19.63 / 47.60
        pp = T["proof-pile-2/v0_decontaminated"]; out["algebraic_stack"] = pp * 12.62 / 25.36; out["openwebmath"] = pp * 12.73 / 25.36
        out["tulu_v0"] = T["tulu_flan/v0_all_train"]; out["starcoder"] = T["starcoder/v0_decontaminated_doc_only"]
        out["falcon_v0"] = T["falcon-refinedweb/v0-0.05-heldout-complement_decontaminated"]
        return out
    if any(k in fps for k in SZ):
        # dolma 1.7 components: exact per-group totals from mixture, split within groups by files*avg size
        grp = {"olmo-mix/v1_6-decontaminated": ["gutenberg_books", "pes20_stem_papers", "wikipedia_wikibooks"],
               "proof-pile-2/v0_decontaminated": ["algebraic_stack", "openwebmath"],
               "redpajama_v1_decon_fix/arxiv": ["arxiv"], "redpajama_v1_decon_fix/stackexchange": ["stackexchange"],
               "starcoder/v0_decontaminated_doc_only": ["starcoder"], "c4/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001-fix": ["c4"],
               "reddit/v5-dedupe-pii-nsfw-toxic-fuzzydd-length": ["reddit"], "cc-news/v3": ["cc_news"], "megawika/v1": ["megawika"],
               "tulu_flan/v2-decontaminated-60M-shots_all-upweight_1-dialog_false-sep_newline": ["tulu"],
               "falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii": ["falcon"],
               "olmo-mix/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001": ["web_rest"]}
        for g, members in grp.items():
            if g not in T: continue
            est = {m: fps.get(m, 0) * SZ[m] / NF[m] for m in members}
            tot = sum(est.values())
            for m in members:
                if est[m] > 0: out[m] = T[g] * est[m] / tot
        if "fastdclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train" in T:
            out["dclm"] = T["fastdclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train"]
        return out
    # single/dual-pool recipes
    for k, v in T.items(): out[k] = v
    return out

conf = {r["recipe"]: r for r in json.load(open(f"{REPO}/results/datadecide/confounders.json"))["recipes"]}
cf = {r["recipe"]: r for r in csv.DictReader(open(f"{REPO}/results/datadecide/count_features.csv"))}
rows = []
for s in recipes:
    ft = fine_tokens(s); tot = sum(ft.values())
    g = lambda *ks: sum(v for k, v in ft.items() if any(k == x for x in ks)) / tot
    row = dict(recipe=s, corpus_B=tot,
               epochs=100.016 / tot,
               code=g("starcoder"), code_adj=g("stackexchange", "algebraic_stack"),
               math_pp2=g("algebraic_stack", "openwebmath"),
               academic=g("pes20_stem_papers", "arxiv", "d16_pes2o"),
               stem_all=g("algebraic_stack", "openwebmath", "pes20_stem_papers", "arxiv", "d16_pes2o", "stackexchange"),
               books=g("gutenberg_books", "d16_books"), wiki=g("wikipedia_wikibooks", "megawika", "d16_wiki"),
               reddit=g("reddit", "d16_reddit"), flan=g("tulu", "tulu_v0"), news=g("cc_news"),
               nonweb=None,
               conf_code=conf[s]["code_share"], conf_math=conf[s]["math_share"], numdens=conf[s]["number_density"],
               F01=float(cf[s]["F01_total_numbers_B"]), F20=float(cf[s]["F20_year_share_of_d4"]), F22=float(cf[s]["F22_pow2_excess"]),
               fine=ft)
    row["nonweb"] = row["code"] + row["code_adj"] + row["math_pp2"] - g("algebraic_stack") + row["academic"] + row["books"] + row["wiki"] + row["flan"]
    rows.append(row)

hdr = "recipe corpus_B epochs code code_adj math_pp2 academic stem_all books wiki reddit flan news nonweb conf_code conf_math numdens F01 F20 F22".split()
print("\t".join(hdr))
for r in rows:
    print("\t".join(f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h]) for h in hdr))
json.dump([{k: v for k, v in r.items()} for r in rows], open(sys.argv[1], "w"), indent=1)
