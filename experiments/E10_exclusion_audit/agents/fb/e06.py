import json, math, numpy as np, pandas as pd, sys
sys.path.insert(0,'/home/user/numberline-datadecide')
from scipy.stats import kstest, chisquare
R='/home/user/numberline-datadecide/'
D='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/fb/'
from src.beta_fine import band, fit_geometric, fit_continuous, COARSE, N_GROUPS
fv=pd.read_csv(R+'results/beta_fine/fine_vs_data.csv')[['recipe','layer','old_layer','beta_coarse','beta_cont','beta_coarse_oldlayer','beta_cont_oldlayer','beta_coarse_std','beta_cont_std']]
recipes=list(pd.read_csv(R+'results/datadecide/alpha_beta.csv').recipe)
BOUND={0:10,3:100,6:1000,9:10000}  # digit boundary inside band (bands 0,3,6,9 straddle a power of ten)
def strata_of(g, n, nsub=4):
    """stratum id: digit length x nsub equal sub-bins of the band"""
    b=band(g); lo,hi=b.start,b.stop-1
    q=np.minimum(((np.asarray(n)-lo)/(hi-lo+1)*nsub).astype(int),nsub-1)
    if g in BOUND: d=(np.asarray(n)>=BOUND[g]).astype(int)
    else: d=np.zeros_like(q)
    return d*nsub+q
def prop_share(g,nsub=4):
    ints=np.arange(band(g).start,band(g).stop)
    s=strata_of(g,ints,nsub); u,c=np.unique(s,return_counts=True); return dict(zip(u,c/len(ints)))
rows=[]; grp_rows=[]
for r in recipes:
    J=json.load(open(R+f'results/beta_fine/step69369-seed-default/{r}.json'))
    Z=np.load(R+f'results/beta_fine/step69369-seed-default/{r}.npz')
    f=fv[fv.recipe==r].iloc[0]
    row={'recipe':r}
    # pooled filter stats
    for g in range(N_GROUPS):
        tried=sum(J['filter'][s][str(g)]['tried'] for s in J['filter']); acc=sum(J['filter'][s][str(g)]['accepted'] for s in J['filter'])
        fail=sum(J['filter'][s][str(g)]['failed_slots'] for s in J['filter'])
        row[f'rej{g}']=1-acc/tried; row[f'fail{g}']=fail
        T=np.concatenate([Z[f'seed{s}_targets'][Z[f'seed{s}_groups']==g] for s in J['seeds']])
        assert len(T)==acc
        b=band(g); lo,hi=b.start,b.stop-1
        # KS vs discrete uniform (continuity-corrected by jitter-free approach: compare to uniform on [lo,hi+1))
        ks=kstest(T+0.5,'uniform',args=(lo,hi-lo+1)).pvalue
        cover=len(np.unique(T))/len(b)
        # expected distinct coverage under uniform with same n accepted
        expcov=1-(1-1/len(b))**len(T)
        share_above=np.mean(T>=BOUND[g]) if g in BOUND else np.nan
        prop_above=np.mean(np.arange(lo,hi+1)>=BOUND[g]) if g in BOUND else np.nan
        # stratum acceptance rates (digit boundary only)
        if g in BOUND:
            na=(T>=BOUND[g]).sum(); nb=(T<BOUND[g]).sum()
            a_hi=na/(tried*prop_above); a_lo=nb/(tried*(1-prop_above))
        else: a_hi=a_lo=np.nan
        grp_rows.append(dict(recipe=r,group=g,lo=lo,hi=hi,tried=tried,acc=acc,rej=1-acc/tried,ks_p=ks,cover=cover,expcov=expcov,
                             share_above=share_above,prop_above=prop_above,acc_rate_low=a_lo,acc_rate_high=a_hi,mean_logn=np.mean(np.log10(T)),
                             prop_mean_logn=np.mean(np.log10(np.arange(lo,hi+1)))))
    # IPW recomputation of beta at old layer and selected layer
    for tag,L in [('old',int(f.old_layer)),('sel',int(f.layer))]:
        res={'raw_c':[],'ipw_c':[],'raw_k':[],'ipw_k':[],'ipwdig_c':[]}
        for s in J['seeds']:
            t=Z[f'seed{s}_targets']; gg=Z[f'seed{s}_groups']; pc1=Z[f'seed{s}_pc_scores'][L,:,0].astype(np.float64)
            # global stratum weights pooled over seeds (acceptance estimated from all 3 seeds)
            w=np.ones_like(t); wd=np.ones_like(t)
            for g in range(N_GROUPS):
                m=gg==g
                for nsub,ww in [(4,w),(1,wd)]:
                    allT=np.concatenate([Z[f'seed{x}_targets'][Z[f'seed{x}_groups']==g] for x in J['seeds']])
                    sa=strata_of(g,allT,nsub); ps=prop_share(g,nsub)
                    u,c=np.unique(sa,return_counts=True); accsh=dict(zip(u,c/len(allT)))
                    st=strata_of(g,t[m],nsub)
                    ww[m]=[ps[x]/accsh[x] for x in st]
            def gm(weights):
                return [np.sum(weights[gg==g]*pc1[gg==g])/np.sum(weights[gg==g]) for g in range(N_GROUPS)]
            mr=gm(np.ones_like(t)); mi=gm(w); md=gm(wd)
            res['raw_c'].append(fit_geometric(np.diff([mr[g] for g in COARSE]))[0])
            res['ipw_c'].append(fit_geometric(np.diff([mi[g] for g in COARSE]))[0])
            res['ipwdig_c'].append(fit_geometric(np.diff([md[g] for g in COARSE]))[0])
            # continuous fit: weighted via replication is awkward; use weighted least squares by sqrt(w) scaling
            u=np.log10(t)-1
            res['raw_k'].append(fit_continuous(u,pc1)[0])
            # weighted continuous fit
            from scipy.optimize import minimize_scalar
            sw=np.sqrt(w)
            def sse(lb):
                z=np.exp(lb*u) if abs(lb)>1e-6 else u
                X=np.column_stack([np.ones_like(u),z])*sw[:,None]; y=pc1*sw
                coef,*_=np.linalg.lstsq(X,y,rcond=None); rr=y-X@coef; return float(rr@rr)
            res['ipw_k'].append(float(np.exp(minimize_scalar(sse,bounds=(-6,6),method='bounded').x)))
        for k,v in res.items(): row[f'{tag}_{k}']=np.mean(v); row[f'{tag}_{k}_sd']=np.std(v,ddof=1)
    row['rep_coarse_old']=f.beta_coarse_oldlayer; row['rep_cont_old']=f.beta_cont_oldlayer
    row['rep_coarse_sel']=f.beta_coarse; row['rep_cont_sel']=f.beta_cont; row['old_layer']=f.old_layer; row['sel_layer']=f.layer
    rows.append(row)
df=pd.DataFrame(rows); gdf=pd.DataFrame(grp_rows)
df.to_csv(D+'e06_ipw.csv',index=False); gdf.to_csv(D+'e06_groups.csv',index=False)
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print(df[['recipe']+[f'rej{g}' for g in range(10)]].round(3).to_string())
print('failed slots total', df[[f'fail{g}' for g in range(10)]].values.sum())
for tag in ['old','sel']:
    x=df[['recipe',f'rep_coarse_{tag}',f'{tag}_raw_c',f'{tag}_ipw_c',f'{tag}_ipwdig_c',f'{tag}_raw_c_sd',f'rep_cont_{tag}',f'{tag}_raw_k',f'{tag}_ipw_k',f'{tag}_raw_k_sd']].copy()
    x['d_coarse']=x[f'{tag}_ipw_c']-x[f'{tag}_raw_c']; x['d_coarse_dig']=x[f'{tag}_ipwdig_c']-x[f'{tag}_raw_c']; x['d_cont']=x[f'{tag}_ipw_k']-x[f'{tag}_raw_k']
    print(tag); print(x.round(4).to_string())
