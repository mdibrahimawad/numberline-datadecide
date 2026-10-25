import numpy as np, pandas as pd, sys, itertools
sys.path.insert(0,'/home/user/numberline-datadecide')
from src.beta_fine import fit_geometric, COARSE, band
R='/home/user/numberline-datadecide/'
D='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/fb/'
ab=pd.read_csv(R+'results/datadecide/alpha_beta.csv')[['recipe','selected_layer','beta_direct_mean','beta_direct_std']]
fv=pd.read_csv(R+'results/beta_fine/fine_vs_data.csv')[['recipe','old_layer','beta_coarse_std','beta_cont_std']]
print('median E06 beta_coarse_std',fv.beta_coarse_std.median(),'beta_cont_std',fv.beta_cont_std.median(),'E01 beta_direct_std median',ab.beta_direct_std.median())
ft=pd.read_csv(D+'e06_firsttry.csv'); e1=pd.read_csv(D+'e01_filter.csv')
sub=pd.read_csv(D+'e06_subbin.csv')
out=[]
for r in ab.recipe:
    L=int(fv[fv.recipe==r].old_layer.iloc[0]); Z=np.load(R+f'results/beta_fine/step69369-seed-default/{r}.npz')
    r06={g:1-ft[(ft.recipe==r)&(ft.group==g)].first_ok.mean() for g in COARSE}
    r01={k:e1[e1.recipe==r][f'rej{i+1}'].iloc[0] for i,k in enumerate(COARSE)}
    res={}
    for tag,rr in [('e06',r06),('e01',r01)]:
        worst=[]
        for s in (45,46,47):
            t=Z[f'seed{s}_targets']; gg=Z[f'seed{s}_groups']; pc=Z[f'seed{s}_pc_scores'][L,:,0].astype(float)
            m=np.array([pc[gg==g].mean() for g in COARSE]); sd=np.array([pc[gg==g].std() for g in COARSE])
            b0=fit_geometric(np.diff(m))[0]; w=0
            for sg in itertools.product([-1,1],repeat=4):
                mm=m+np.array([rr[g] for g in COARSE])*np.array(sg)*sd  # population mean if rejected items sit 1 SD away
                w=max(w,abs(fit_geometric(np.diff(mm))[0]-b0))
            worst.append(w)
        res[tag]=np.mean(worst)
    sb=sub[sub.recipe==r]
    out.append(dict(recipe=r,adv1sd_e06=res['e06'],adv1sd_e01=res['e01'],n_subbins_lt50=int((sb[sb.group>=3].min_bin_ratio<0.5).sum()),min_subbin_ratio=sb[sb.group>=3].min_bin_ratio.min()))
o=pd.DataFrame(out); o.to_csv(D+'sens.csv',index=False); print(o.round(4).to_string())
print(sub[sub.min_bin_ratio<0.5].to_string())
