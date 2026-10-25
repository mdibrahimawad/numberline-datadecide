# Does E01's narrow-band placement (980-1019, 9980-10019, around round numbers) shrink gap 3?
# Using E06 prompts at E01's layer: narrow gap3 = mean PC1[9900<=n<=10100] - mean PC1[980<=n<=1019]; broad = E06 band gap.
import numpy as np, pandas as pd
R='/home/user/numberline-datadecide/results/'
P='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/gaps/'
ab=pd.read_csv(R+'datadecide/alpha_beta.csv'); e1=pd.read_csv(P+'gap_summary.csv').set_index('recipe')
rows=[]
for rec,L in zip(ab.recipe,ab.selected_layer):
    z=np.load(R+f'beta_fine/step69369-seed-default/{rec}.npz'); nn=[];br=[];cnt=[]
    for s in (45,46,47):
        t=z[f'seed{s}_targets']; g=z[f'seed{s}_groups']; y=z[f'seed{s}_pc_scores'][L][:,0].astype(float)
        if np.corrcoef(np.log10(t),y)[0,1]<0: y=-y
        a=(t>=980)&(t<=1019); b=(t>=9900)&(t<=10100)
        nn.append(y[b].mean()-y[a].mean()); br.append(y[g==9].mean()-y[g==6].mean()); cnt.append((a.sum(),b.sum()))
    rows.append(dict(recipe=rec,narrow_gap3=np.mean(nn),broad_gap3=np.mean(br),ratio=np.mean(nn)/np.mean(br),
                     n_per_seed=np.mean(cnt,0).round(1).tolist(),E01_d3=e1.loc[rec,'d3'],E01_over_broad=e1.loc[rec,'ratio_d3']))
df=pd.DataFrame(rows); pd.set_option('display.width',200)
print(df.round(2).to_string()); print('median narrow/broad',df.ratio.median().round(3))
df.to_csv(P+'narrow_gap3.csv',index=False)
