import sys, json, random, math
sys.path.insert(0,'/home/user/numberline-datadecide')
import numpy as np, pandas as pd
from scipy import stats
from src.beta_fine import make_prompt, fit_geometric, N_GROUPS, COARSE
R='/home/user/numberline-datadecide/results/beta_fine/step69369-seed-default/'
fv=pd.read_csv('/home/user/numberline-datadecide/results/beta_fine/fine_vs_data.csv')
K=100
rows=[]
def ctx_of(prompt):
    parts=prompt.split(',')[:-1]
    return [int(p.split('=')[0]) for p in parts]
def betas(means):
    m=[means[g] for g in range(N_GROUPS)]
    bf=fit_geometric(np.diff(m))[0]**3
    bc=fit_geometric(np.diff([m[g] for g in COARSE]))[0]
    return bc,bf
for _,r in fv.iterrows():
    rec=r.recipe; z=np.load(R+rec+'.npz'); j=json.load(open(R+rec+'.json'))
    for layer_name,L in (('sel',int(r.layer)),('old',int(r.old_layer))):
      for seed in (45,46,47):
        rng=random.Random(seed)
        init=[]
        for g in range(N_GROUPS):
            for _ in range(K):
                p,n=make_prompt(g,rng); init.append((g,n,p))
        t=z[f'seed{seed}_targets']; gr=z[f'seed{seed}_groups']; pc=z[f'seed{seed}_pc_scores'][L,:,0].astype(float)
        assert len(t)==N_GROUPS*K, (rec,len(t))
        assert all(gr[i]==init[i][0] for i in range(len(t)))
        match=np.array([init[i][1]==t[i] for i in range(len(t))])
        ctx=np.array([ctx_of(init[i][2]) for i in range(len(t))],float)
        mlog=np.log10(ctx+1).mean(1)          # mean log magnitude of 3 examples
        lastlog=np.log10(ctx[:,2]+1)          # example adjacent to target
        n4=(ctx>=1000).sum(1)
        u=np.log10(t)
        # first-try rejected slots (definitely rejected): mismatched; their first-try context known
        f=j['filter'][str(seed)]
        nrej={g:f[str(g)]['tried']-f[str(g)]['accepted'] for g in range(N_GROUPS)}
        # within-group regression on matched slots: pc ~ groupFE + u + ctx
        m=match
        G=np.eye(N_GROUPS)[gr[m]]
        X0=np.column_stack([G,u[m]-np.array([u[m][gr[m]==g].mean() for g in gr[m]])])
        Xc=np.column_stack([X0,mlog[m],lastlog[m],n4[m]])
        y=pc[m]
        def rss(X):
            b,*_=np.linalg.lstsq(X,y,rcond=None); e=y-X@b; return (e@e),b
        r0,_=rss(X0); r1,b1=rss(Xc)
        pr2=1-r1/r0
        df1=3; df2=len(y)-Xc.shape[1]
        F=((r0-r1)/df1)/(r1/df2); pF=stats.f.sf(F,df1,df2)
        # simple slope on mean log ctx only
        Xm=np.column_stack([X0,mlog[m]]); _,bm=rss(Xm); slope_m=bm[-1]
        # balance: group means of mlog among kept-matched
        gm_ctx=np.array([mlog[m][gr[m]==g].mean() for g in range(N_GROUPS)])
        anova_p=stats.f_oneway(*[mlog[m][gr[m]==g] for g in range(N_GROUPS)]).pvalue
        # selection: first-try rejected (mismatch) vs accepted context, pooled within group (demeaned)
        acc_ctx=[]; rej_ctx=[]
        # raw vs adjusted group means (matched only); adjustment with all 3 ctx coefficients
        means_raw={g:pc[m][gr[m]==g].mean() for g in range(N_GROUPS)}
        bc_c=b1[-3:]
        C=np.column_stack([mlog,lastlog,n4])[m]; Cbar=C.mean(0)
        means_adj={g:means_raw[g]-(C[gr[m]==g].mean(0)-Cbar)@bc_c for g in range(N_GROUPS)}
        means_all={g:pc[gr==g].mean() for g in range(N_GROUPS)}
        bc_all,bf_all=betas(means_all); bc_raw,bf_raw=betas(means_raw); bc_adj,bf_adj=betas(means_adj)
        span=means_all[9]-means_all[0]
        rows.append(dict(recipe=rec,layer_kind=layer_name,layer=L,seed=seed,match_frac=m.mean(),
            n_rej=sum(nrej.values()),pr2_ctx=pr2,F_p=pF,slope_mlog_per_span=slope_m/span,
            coef_mlog=bc_c[0]/span,coef_last=bc_c[1]/span,coef_n4=bc_c[2]/span,
            ctx_bal_range=gm_ctx.max()-gm_ctx.min(),ctx_bal_p=anova_p,
            bc_all=bc_all,bc_raw=bc_raw,bc_adj=bc_adj,bf_all=bf_all,bf_raw=bf_raw,bf_adj=bf_adj))
df=pd.DataFrame(rows)
df.to_csv('/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/a17/ctx_rows.csv',index=False)
print(df.groupby('layer_kind')[['match_frac','pr2_ctx','slope_mlog_per_span','ctx_bal_range']].describe().T.to_string())
