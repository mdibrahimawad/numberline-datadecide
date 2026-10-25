import pandas as pd, numpy as np
from scipy.stats import pearsonr, spearmanr
d=pd.read_pickle('table.pkl')
R3=['dclm-baseline','dolma1_7','fineweb-edu','c4','fineweb-pro','dclm-baseline-qc-7p-fw2','falcon-and-cc-qc-20p','dclm-baseline-qc-20p','falcon-and-cc-qc-orig-10p','falcon-and-cc-qc-tulu-10p','dclm-baseline-qc-7p-fw3','dclm-baseline-qc-fw-10p']
R4=R3+['falcon','dclm-baseline-qc-7p-fw3','dclm-baseline-qc-10p']
layer_bad=['falcon-and-cc-qc-10p','dolma1_7','fineweb-edu']
meas_fail=['dolma1_7-no-reddit']
rules={
 'V0 all 25':[],
 'M1 drop beta replication failure (no-reddit)':meas_fail,
 'M2 drop ill-defined layer choice (3)':layer_bad,
 'M3 M1+M2 (all measurement-validity flags)':meas_fail+layer_bad,
 'D1 common alpha support (drop 4 code/math Dolma + fineweb-pro)':['dolma1_6plus','dolma1_7','dolma1_7-no-reddit','dolma1_7-no-flan','fineweb-pro'],
 'D2 low data overlap R4 (keep 15)':[r for r in d.recipe if r not in R4],
 'D3 low data overlap R3 (keep 12)':[r for r in d.recipe if r not in R3],
 'D4 >10% repeated tokens':['fineweb-pro','dclm-baseline-qc-10p'],
 'D5 different training seed':['dclm-baseline-qc-fw-3p','dclm-baseline-qc-fw-10p','dclm-baseline-qc-10p','dclm-baseline-qc-20p'],
 'D6 copy-filter rejection >25%':['fineweb-pro','dclm-baseline-qc-7p-fw3'],
 'D7 drop derived mixes (3)':[r for r in d.recipe if r.startswith('dclm-baseline-') and 'dolma' in r],
 'M3+D1':list(set(meas_fail+layer_bad+['dolma1_6plus','dolma1_7','dolma1_7-no-reddit','dolma1_7-no-flan','fineweb-pro'])),
 'M3+D4+D6':list(set(meas_fail+layer_bad+['fineweb-pro','dclm-baseline-qc-10p','dclm-baseline-qc-7p-fw3'])),
 'M3+D5':list(set(meas_fail+layer_bad+['dclm-baseline-qc-fw-3p','dclm-baseline-qc-fw-10p','dclm-baseline-qc-10p','dclm-baseline-qc-20p'])),
 'M3+D7':list(set(meas_fail+layer_bad+[r for r in d.recipe if r.startswith('dclm-baseline-') and 'dolma' in r])),
}
def cs(m,col):
    x,y=d.alpha_ols[m],d[col][m]; r,p=pearsonr(x,y); s,q=spearmanr(x,y); return r,p,s,q
out=[]
for k,drop in rules.items():
    m=~d.recipe.isin(drop)
    for col,lab in (('beta_eval','published'),('beta_6','6-seed')):
        r,p,s,q=cs(m,col); out.append(dict(rule=k,beta=lab,n=int(m.sum()),r=r,p=p,rho=s,p_rho=q))
o=pd.DataFrame(out); pd.set_option('display.width',250); print(o.round(3).to_string(index=False))
o.to_csv('rules_out.csv',index=False)
