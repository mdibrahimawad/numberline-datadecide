import json, csv, numpy as np, pandas as pd, sys
R='/home/user/numberline-datadecide/results/'
recipes=[r['recipe'] for r in csv.DictReader(open(R+'datadecide/alpha_beta.csv'))]
fl=pd.read_csv(R+'beta_fine/fixed_layer_beta.csv',index_col=0)
fl=fl[[c for c in fl.columns if c.startswith('beta_cont_L')]]  # beta only
rows=[]; prof={}
for rc in recipes:
    d=json.load(open(R+f'datadecide/{rc}.json'))
    s=d['selection']; L=s['selected_layer']
    ls={int(k):v for k,v in s['layer_scores'].items()}
    layers=sorted(ls)
    qm=np.array([ls[l]['q_median'] for l in layers]); qs=np.array([ls[l]['q_per_seed'] for l in layers])  # (nl,3)
    order=np.argsort(-qm); best=layers[order[0]]; ru=layers[order[1]]
    assert best==L
    margin=qm[order[0]]-qm[order[1]]
    seed_arg=[layers[int(np.argmax(qs[:,j]))] for j in range(3)]
    # seed noise of q at selected layer
    iL=layers.index(L)
    # paired test: layers l that beat L in >=1 seed
    beat=[l for i,l in enumerate(layers) if l!=L and np.any(qs[i]>=qs[iL])]
    # per-layer beta selection seeds E01
    bsel={int(k):v['beta_mean'] for k,v in s['layers_direct'].items()}
    bsd={int(k):v['beta_std'] for k,v in s['layers_direct'].items()}
    ev=d['evaluation']; es=ev['summary']
    beval=es['beta_mean']; bevsd=es['beta_std']
    # eval per-seed q at L
    qev=[np.sqrt(r['explained_variance']*abs(r['rho'])) for r in ev['runs']]
    # peaks
    early=max(range(2,6),key=lambda l:ls[l]['q_median']); mid=max(range(6,11),key=lambda l:ls[l]['q_median'])
    qe,qmid=ls[early]['q_median'],ls[mid]['q_median']
    lo,hi=sorted([early,mid]); dip=min(ls[l]['q_median'] for l in range(lo,hi+1))
    rows.append(dict(recipe=rc,L=L,qL=qm[order[0]],runner=ru,margin=margin,rel=margin/qm[order[0]],
        seed_arg=seed_arg,beat=beat,qsdL=np.std(qs[iL],ddof=1),qseedL=qs[iL].tolist(),
        early=early,qe=qe,mid=mid,qmid=qmid,dip=dip,
        bselL=bsel[L],bselR=bsel[ru],bsdL=bsd[L],beval=beval,bevsd=bevsd,qev=qev,
        e06L=fl.loc[rc,f'beta_cont_L{L}'],e06R=fl.loc[rc,f'beta_cont_L{ru}'],
        bsel_all={l:bsel[l] for l in layers}, e06_all={l:fl.loc[rc,f'beta_cont_L{l}'] for l in layers}, qm={l:ls[l]['q_median'] for l in layers}, qs={l:ls[l]['q_per_seed'] for l in layers}))
import pickle; pickle.dump(rows,open(sys.argv[1],'wb'))
for r in rows:
    print(f"{r['recipe'][:30]:30s} L={r['L']:2d} q={r['qL']:.3f} ru={r['runner']:2d} m={r['margin']:.4f} ({100*r['rel']:.1f}%) seeds={r['seed_arg']} qsd={r['qsdL']:.3f} beat={r['beat']} E{r['early']}={r['qe']:.3f} M{r['mid']}={r['qmid']:.3f} dip={r['dip']:.3f} bL={r['bselL']:.2f}±{r['bsdL']:.2f} bR={r['bselR']:.2f} bev={r['beval']:.2f}±{r['bevsd']:.2f} e06L={r['e06L']:.2f} e06R={r['e06R']:.2f} qev={np.round(r['qev'],3).tolist()}")
