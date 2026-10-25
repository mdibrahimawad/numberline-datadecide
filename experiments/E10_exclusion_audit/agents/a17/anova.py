import json, numpy as np, pandas as pd
d=pd.read_csv('e01_seeds.csv').set_index('recipe')
fv=pd.read_csv('/home/user/numberline-datadecide/results/beta_fine/fine_vs_data.csv').set_index('recipe')
R='/home/user/numberline-datadecide/results/beta_fine/step69369-seed-default/'
def twoway(M,name):
    M=np.asarray(M,float); n,k=M.shape
    g=M.mean(); rm=M.mean(1); cm=M.mean(0)
    res=M-rm[:,None]-cm[None,:]+g
    sd_model=np.std(rm,ddof=1); seed_eff=cm-g
    within_raw=np.sqrt(np.mean(np.var(M,axis=1,ddof=1)))
    inter=np.sqrt((res**2).sum()/((n-1)*(k-1)))
    # reliability of model means (ICC consistency, k seeds averaged)
    ms_r=k*np.var(rm,ddof=1); ms_e=(res**2).sum()/((n-1)*(k-1))
    icc=(ms_r-ms_e)/ms_r
    per_model_inter=np.sqrt((res**2).sum(1)/(k-1)*(k/(k-1)))  # approx per-model residual SD
    print(f'{name}: between-model SD={sd_model:.4f}  seed effects={np.round(seed_eff,4)}  pooled raw within-model SD={within_raw:.4f}  interaction SD={inter:.4f}  ICC(C,k)={icc:.3f}  inter/between={inter/sd_model:.3f}')
    return res, per_model_inter, sd_model, inter
out={}
res,pm,sb,si=twoway(d[['e45','e46','e47']].values,'E01 beta_direct eval seeds 45-47')
out['e01_ev_inter']=pm
res,pm,sb,si=twoway(d[['s42','s43','s44','e45','e46','e47']].values,'E01 beta_direct six seeds 42-47 (sel layer)')
out['e01_6_inter']=pm
for key in ['beta_coarse','beta_fine','beta_cont']:
    M=[]
    for rec in d.index:
        j=json.load(open(R+rec+'.json')); L=int(fv.loc[rec,'layer'])
        M.append([ps['layers'][str(L)][key] for ps in j['per_seed']])
    M=np.array(M); out[key+'_raw']=M
    res,pm,sb,si=twoway(M,'E06 '+key+' (sel layer)'); out[key+'_inter']=pm
    out[key+'_sd']=M.std(1,ddof=1)
t=pd.DataFrame({'e01_ev_sd':d.ev_sd,'e01_6_sd':d.six_sd,'e01_ev_inter':out['e01_ev_inter'],'e01_6_inter':out['e01_6_inter'],
 'e06c_sd':out['beta_coarse_sd'],'e06c_inter':out['beta_coarse_inter'],'e06f_sd':out['beta_fine_sd'],'e06cont_sd':out['beta_cont_sd']},index=d.index)
for c in t.columns: t[c+'_x']=t[c]/t[c].median()
t.to_csv('seed_table.csv')
pd.set_option('display.width',250)
print(t.round(3).to_string())
print(t.median().round(4))
