import json, numpy as np, pandas as pd, sys
sys.path.insert(0,'/home/user/numberline-datadecide')
from src.beta_fine import fit_geometric, COARSE
R='/home/user/numberline-datadecide/'
D='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/fb/'
f=pd.read_csv(D+'e01_filter.csv'); rj=pd.read_csv(D+'e01_rejections.csv')
fv=pd.read_csv(R+'results/beta_fine/fine_vs_data.csv')[['recipe','old_layer','beta_coarse_oldlayer']]
bound={1:10,2:100,3:1000,4:10000}; pabove={1:30/39,2:.5,3:.5,4:.5}
E06G={1:0,2:3,3:6,4:9}
out=[]
for _,row in f.iterrows():
    r=row.recipe; o={'recipe':r}
    shares={}
    for g in range(1,5):
        T=row[f'tried{g}']; Rj=T-240
        s=rj[(rj.recipe==r)&(rj.group==g)]
        fa=(s.target>=bound[g]).mean() if len(s) else pabove[g]
        acc_above=T*pabove[g]-fa*Rj
        sh=min(max(acc_above/240,0),1); shares[g]=sh
        o[f'nlog{g}']=len(s); o[f'fa{g}']=fa; o[f'share_above{g}']=sh; o[f'shift_pp{g}']=100*(sh-pabove[g])
    # transfer to E06 coarse bands at E01's layer
    L=int(fv[fv.recipe==r].old_layer.iloc[0])
    Z=np.load(R+f'results/beta_fine/step69369-seed-default/{r}.npz')
    b_acc=[];b_prop=[]
    for sd in (45,46,47):
        t=Z[f'seed{sd}_targets']; gg=Z[f'seed{sd}_groups']; pc=Z[f'seed{sd}_pc_scores'][L,:,0].astype(float)
        def means(sh_):
            m=[]
            for g in range(1,5):
                k=gg==E06G[g]; ab=t[k]>=bound[g]
                m.append(sh_[g]*pc[k][ab].mean()+(1-sh_[g])*pc[k][~ab].mean())
            return m
        b_acc.append(fit_geometric(np.diff(means(shares)))[0]); b_prop.append(fit_geometric(np.diff(means(pabove)))[0])
    o['beta_E01skew']=np.mean(b_acc); o['beta_balanced']=np.mean(b_prop); o['dbeta_E01skew']=np.mean(np.array(b_acc)-np.array(b_prop))
    o['dbeta_sd_seeds']=np.std(np.array(b_acc)-np.array(b_prop),ddof=1)
    out.append(o)
df=pd.DataFrame(out); df.to_csv(D+'e01_skew_transfer.csv',index=False)
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print(df[['recipe','nlog1','nlog2','nlog3','nlog4','shift_pp1','shift_pp2','shift_pp3','shift_pp4','beta_balanced','beta_E01skew','dbeta_E01skew','dbeta_sd_seeds']].round(3).to_string())
