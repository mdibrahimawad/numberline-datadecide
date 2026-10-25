import pandas as pd, numpy as np, sys
W=sys.argv[1]
d=pd.read_csv(f"{W}/geom.csv"); e=pd.read_csv(f"{W}/extra.csv")
d=d.merge(e[['recipe','which','off1','off2','off3','g1','g2','g3','ratio_smooth_null']],on=['recipe','which'])
d['dC1']=(d.beta_arc5-d.beta_pc1).abs(); d['dC4']=(d.beta_med-d.beta_pc1).abs()
d['ev1_rel']=d.ev1/d.groupby('which').ev1.transform('median')
def flag(r,k=1.0):
    f=[]
    if r.dC1>0.10*k: f.append('C1')
    if r.slope_ratio_within_digit<0.5/k: f.append('C2')
    if r.eta2<0.5/k or r.beta_pc1_sd>0.05*k or r.neg_coarse_seeds>=2: f.append('C3')
    if r.dC4>0.10*k: f.append('C4')
    if r.ev1_rel<0.5/k: f.append('C5')
    return ','.join(f)
for k,nm in ((1,'flags'),(0.8,'flags_strict20'),(1.25,'flags_lenient25')):
    d[nm]=d.apply(lambda r:flag(r,k),axis=1)
d.to_csv(f"{W}/geom_flags.csv",index=False)
pd.set_option('display.width',250)
print(d[['recipe','which','layer','beta_pc1','beta_arc5','dC1','slope_ratio_within_digit','dC4','beta_pc1_sd','eta2','ev1','ev1_rel','flags','flags_strict20','flags_lenient25']].round(3).to_string(index=False))
print(d.groupby('which').ev1.median())
for w in ('sel','L8'):
    s=d[d.which==w]; print(w,'flagged:',list(s[s['flags'].fillna('')!=''].recipe))
print(d.groupby('which')[['off1','off2','off3']].apply(lambda x:x.describe().loc[['min','50%','max']]))
d['o1']=d.off1/d.g1; d['o2']=d.off2/d.g2; d['o3']=d.off3/d.g3
print(d.groupby('which')[['o1','o2','o3','slope_ratio_within_digit','ratio_smooth_null','pc2_amp','offline_frac','rho_pc2','share_u_pc1','cos_line_pc1','eta2','out_frac_pc1','out_frac_mah5','digit_step_excess']].describe().loc[['min','50%','max']].round(3).T)
