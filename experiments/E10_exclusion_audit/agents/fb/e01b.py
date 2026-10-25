import pandas as pd, numpy as np, re
D='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/fb/'
rj=pd.read_csv(D+'e01_rejections.csv')
bound={1:10,2:100,3:1000,4:10000}
rj['above']=rj.apply(lambda x: x.target>=bound[x.group],axis=1)
def kind(x):
    t=str(int(x.target)); g=x.generated
    if pd.isna(g): return 'no_int'
    g=str(int(g))
    if x.response.startswith(t[:2]+',') and len(t)>=4 and not x.response.startswith(t+','): return 'comma_format'
    if g.startswith(t) and len(g)>len(t): return 'extra_digits(prefix)'
    if t.startswith(g): return 'truncated'
    if len(g)==len(t): return 'same_len_other'
    return 'other_len'
rj['kind']=rj.apply(kind,axis=1)
# group-level: fraction of rejected that are above boundary vs proposal fraction
prop={1:(39-9)/39, 2:20/40, 3:20/40, 4:20/40}
print('proposal share above boundary:',prop)
t=rj.groupby('group').agg(n=('above','size'),above=('above','mean'))
print(t)
print(pd.crosstab(rj.group,rj.kind))
for g in [3,4]:
    s=rj[rj.group==g]
    print('group',g,'targets counts top:',s.target.value_counts().head(12).to_dict())
    print(pd.crosstab([s.above],s.kind))
# per model, group 3 and 4
pm=rj.groupby(['recipe','group']).agg(n=('above','size'),above=('above','mean')).unstack('group')
print(pm.round(2).to_string())
print(rj[rj.group==4].sample(15,random_state=0)[['recipe','target','response']].to_string())
print(rj[rj.group==3].sample(15,random_state=0)[['recipe','target','response']].to_string())
