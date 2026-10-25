import numpy as np, pandas as pd
P='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/gaps/'
e1=pd.read_csv(P+'e01_gaps_per_seed.csv'); e6=pd.read_csv(P+'e06_coarse_gaps_oldlayer.csv'); st=pd.read_csv(P+'e06_digit_steps.csv')
ab=pd.read_csv('/home/user/numberline-datadecide/results/datadecide/alpha_beta.csv')[['recipe','selected_layer','beta_direct_mean','beta_direct_std','beta_log_mean']]
fv=pd.read_csv('/home/user/numberline-datadecide/results/beta_fine/fine_vs_data.csv')[['recipe','beta_coarse_oldlayer']]
rows=[]
for rec in ab.recipe:
    a=e1[e1.recipe==rec]; b=e6[e6.recipe==rec]
    D=a[['d1','d2','d3']].values; E=b[['a1','a2','a3']].values
    m=D.mean(0); sd=D.std(0,ddof=1); me=E.mean(0); sde=E.std(0,ddof=1)
    q12,q23=m[1]/m[0],m[2]/m[1]; c=np.log(m[1])-0.5*(np.log(m[0])+np.log(m[2]))
    qe12,qe23=me[1]/me[0],me[2]/me[1]; ce=np.log(me[1])-0.5*(np.log(me[0])+np.log(me[2]))
    # per-seed min relative gap and z
    minrel=(D.min(1)/D.max(1)).min(); minrel6=(E.min(1)/E.max(1)).min()
    z=m/sd
    # leave-one-gap-out pairwise beta estimates (E01 seed-mean)
    loo=dict(b12=q12,b23=q23,b13=np.sqrt(m[2]/m[0]))
    sides=set(np.sign(np.log([q12,q23,np.sqrt(m[2]/m[0])])))
    s=st[st.recipe==rec].groupby('boundary').step.mean()
    rows.append(dict(recipe=rec,L=int(ab.loc[ab.recipe==rec,'selected_layer'].iloc[0]),
        beta=ab.loc[ab.recipe==rec,'beta_direct_mean'].iloc[0],
        d1=m[0],d2=m[1],d3=m[2],cv1=sd[0]/m[0],cv2=sd[1]/m[1],cv3=sd[2]/m[2],
        q12=q12,q23=q23,curv=c,b13=loo['b13'],
        e1=me[0],e2=me[1],e3=me[2],qe12=qe12,qe23=qe23,curv6=ce,
        bcoarse6=fv.loc[fv.recipe==rec,'beta_coarse_oldlayer'].iloc[0],
        minrel01=minrel,minrel06=minrel6,zmin01=z.min(),
        ratio_d1=m[0]/me[0], ratio_d3=m[2]/me[2],
        step10=s[10],step100=s[100],step1000=s[1000],step10000=s[10000]))
df=pd.DataFrame(rows)
df.to_csv(P+'gap_summary.csv',index=False)
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print(df.round(2).to_string())
print(df[['cv1','cv2','cv3']].median())
print('E06 per-seed CV median:'); 
print(e6.groupby('recipe')[['a1','a2','a3']].agg(lambda x: x.std(ddof=1)/x.mean()).median())
