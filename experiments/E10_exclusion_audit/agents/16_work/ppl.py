import pandas as pd, numpy as np, sys
df=pd.read_csv(sys.argv[1])
slug={"baseline":"dolma1_7","no_code":"dolma1_7-no-code","no_math_no_code":"dolma1_7-no-math-code","no_reddit":"dolma1_7-no-reddit","no_flan":"dolma1_7-no-flan","dolma-v1-6-and-sources-baseline":"dolma1_6plus","c4":"c4","prox_fineweb_pro":"fineweb-pro","fineweb_edu_dedup":"fineweb-edu","falcon":"falcon","falcon_and_cc":"falcon-and-cc","falcon_and_cc_eli5_oh_top10p":"falcon-and-cc-qc-10p","falcon_and_cc_eli5_oh_top20p":"falcon-and-cc-qc-20p","falcon_and_cc_og_eli5_oh_top10p":"falcon-and-cc-qc-orig-10p","falcon_and_cc_tulu_qc_top10":"falcon-and-cc-qc-tulu-10p","DCLM-baseline":"dclm-baseline","dolma17-75p-DCLM-baseline-25p":"dclm-baseline-25p-dolma1.7-75p","dolma17-50p-DCLM-baseline-50p":"dclm-baseline-50p-dolma1.7-50p","dolma17-25p-DCLM-baseline-75p":"dclm-baseline-75p-dolma1.7-25p","dclm_ft7percentile_fw2":"dclm-baseline-qc-7p-fw2","dclm_ft7percentile_fw3":"dclm-baseline-qc-7p-fw3","dclm_fw_top10":"dclm-baseline-qc-fw-10p","dclm_fw_top3":"dclm-baseline-qc-fw-3p","pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p":"dclm-baseline-qc-10p","pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p":"dclm-baseline-qc-20p"}
def parse(g):
    for k in sorted(slug,key=len,reverse=True):
        for suf in ['-1B-5xC-2','-1B-5xC','-1B-5xC-4','-1B-5xC-5']:
            if g==k+suf: return slug[k], ('default' if suf in ('-1B-5xC-2','-1B-5xC') else suf[-1])
    return None,None
df[['recipe','seed']]=df.group.apply(lambda g: pd.Series(parse(g)))
d=df[df.recipe.notna()]
cols=[c for c in d.columns if c.startswith('eval/')]
short={c:c.split('/')[1].replace('-validation','') for c in cols}
fin=d[d.step==69369]
dd=fin[fin.seed=='default'].set_index('recipe')[cols].rename(columns=short)
print('n default final:',len(dd))
pd.set_option('display.width',250)
print(dd[['c4_en','pile','wikitext_103','dolma_stack','dolma_pes2o','m2d2_s2orc','dolma_books','dolma_reddit']].round(2).sort_values('pile').to_string())
# seed spread
sp=fin.groupby('recipe')[cols].agg(['count']).iloc[:,0]
print('seeds at final per recipe:\n',sp.to_string())
# spike check on default 1B: max ratio ppl(t)/min ppl before t in pile
print('\nspike check (pile ppl): max over t of ppl(t)/min_{s<t} ppl(s), steps>5000')
for r,g in d[d.seed=='default'].groupby('recipe'):
    g=g.sort_values('step'); x=g['eval/pile-validation/Perplexity'].values; s=g.step.values
    m=np.minimum.accumulate(x); ratio=x[1:]/m[:-1]; ok=s[1:]>5000
    i=np.argmax(np.where(ok,ratio,0)); print(f'{r:35s} n={len(x):3d} maxratio={ratio[ok].max():.3f} at step {s[1:][i]:.0f}')
print('\nrebound on c4_en and wikitext_103 (steps>5000):')
for r,g in d[d.seed=='default'].groupby('recipe'):
    g=g.sort_values('step'); out=[]
    for c in ['eval/c4_en-validation/Perplexity','eval/wikitext_103-validation/Perplexity','eval/dolma_common-crawl-validation/Perplexity']:
        x=g[c].values; s=g.step.values; m=np.minimum.accumulate(x); ratio=x[1:]/m[:-1]; ok=s[1:]>5000
        out.append(f'{ratio[ok].max():.3f}')
    print(f'{r:35s}', ' '.join(out), 'steps', g.step.min(), g.step.max(), len(g))
