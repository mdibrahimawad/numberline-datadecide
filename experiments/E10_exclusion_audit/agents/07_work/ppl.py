import csv,collections,json,sys,math
import numpy as np
src=sys.argv[1]
M={"dolma17": "dolma1_7","no_code": "dolma1_7-no-code","no_math_no_code": "dolma1_7-no-math-code","no_reddit": "dolma1_7-no-reddit","no_flan": "dolma1_7-no-flan","dolma-v1-6-and-sources-baseline": "dolma1_6plus","c4": "c4","prox_fineweb_pro": "fineweb-pro","fineweb_edu_dedup": "fineweb-edu","falcon": "falcon","falcon_and_cc": "falcon-and-cc","falcon_and_cc_eli5_oh_top10p": "falcon-and-cc-qc-10p","falcon_and_cc_eli5_oh_top20p": "falcon-and-cc-qc-20p","falcon_and_cc_og_eli5_oh_top10p": "falcon-and-cc-qc-orig-10p","falcon_and_cc_tulu_qc_top10": "falcon-and-cc-qc-tulu-10p","DCLM-baseline": "dclm-baseline","dolma17-75p-DCLM-baseline-25p": "dclm-baseline-25p-dolma1.7-75p","dolma17-50p-DCLM-baseline-50p": "dclm-baseline-50p-dolma1.7-50p","dolma17-25p-DCLM-baseline-75p": "dclm-baseline-75p-dolma1.7-25p","dclm_ft7percentile_fw2": "dclm-baseline-qc-7p-fw2","dclm_ft7percentile_fw3": "dclm-baseline-qc-7p-fw3","dclm_fw_top10": "dclm-baseline-qc-fw-10p","dclm_fw_top3": "dclm-baseline-qc-fw-3p","pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p": "dclm-baseline-qc-10p","pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": "dclm-baseline-qc-20p"}
rows=list(csv.DictReader(open(src)))
cols=[c for c in rows[0] if c.startswith('eval/')]
g=collections.defaultdict(dict)
for r in rows:
    try: g[r['group']][float(r['step'])]={c:float(r[c]) if r[c] else math.nan for c in cols}
    except ValueError: pass
out={}
for raw,rec in M.items():
    key='baseline' if raw=='dolma17' else raw
    # default seed run: -1B-5xC-2, or -1B-5xC (no suffix) for 4 recipes
    cand=[f'{key}-1B-5xC-2', f'{key}-1B-5xC']
    dname=[c for c in cand if c in g]
    if not dname: print('MISSING',rec); continue
    dname=dname[0]
    traj=g[dname]; steps=sorted(traj)
    fin=traj[steps[-1]]
    others=[f'{key}-1B-5xC-{s}' for s in (4,5)]
    if key=='baseline': others=[f'dolma17-1B-5xC-{s}' for s in (4,5)]
    ofin=[g[o][max(g[o])] for o in others if o in g and max(g[o])==69369.0]
    # z-score of log ppl of default vs other seeds, per eval set
    dev=[]
    for c in cols:
        vals=[o[c] for o in ofin]
        if len(vals)==2 and not math.isnan(fin[c]):
            dev.append((c, math.log(fin[c]) - np.mean(np.log(vals)), abs(math.log(vals[0])-math.log(vals[1]))))
    # monotonic over last part: max relative increase between consecutive evals after step 20000
    late=[s for s in steps if s>=20000]
    incs=[]
    for c in ['eval/c4_en-validation/Perplexity','eval/pile-validation/Perplexity']:
        v=[traj[s][c] for s in late]
        incs.append(max((v[i+1]-v[i])/v[i] for i in range(len(v)-1)))
    out[rec]=dict(run=dname, n_steps=len(steps), final_step=steps[-1],
        c4=fin['eval/c4_en-validation/Perplexity'], pile=fin['eval/pile-validation/Perplexity'],
        wikitext=fin['eval/wikitext_103-validation/Perplexity'], stack=fin['eval/dolma_stack-validation/Perplexity'],
        pes2o=fin['eval/dolma_pes2o-validation/Perplexity'],
        n_other_seeds=len(ofin), max_abs_logdev_vs_seeds=max(abs(d[1]) for d in dev) if dev else None,
        mean_logdev_vs_seeds=float(np.mean([d[1] for d in dev])) if dev else None,
        seed_pair_logdiff_max=max(d[2] for d in dev) if dev else None,
        max_late_rel_increase=max(incs))
json.dump(out,open(sys.argv[2],'w'),indent=1)
print(f"{'recipe':32s} run{'':38s} n  c4ppl  pile   wiki  stack  pes2o | nseeds meanlogdev maxabslogdev seedpairmax | lateinc")
for k,v in out.items():
    print(f"{k:32s} {v['run'][:40]:40s} {v['n_steps']:2d} {v['c4']:6.2f} {v['pile']:6.2f} {v['wikitext']:6.2f} {v['stack']:6.2f} {v['pes2o']:6.2f} | {v['n_other_seeds']} {v['mean_logdev_vs_seeds'] if v['mean_logdev_vs_seeds'] is None else round(v['mean_logdev_vs_seeds'],4)} {v['max_abs_logdev_vs_seeds'] and round(v['max_abs_logdev_vs_seeds'],4)} {v['seed_pair_logdiff_max'] and round(v['seed_pair_logdiff_max'],4)} | {v['max_late_rel_increase']:.4f}")
