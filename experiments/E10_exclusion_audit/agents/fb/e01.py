import json, pandas as pd, numpy as np, collections
R='/home/user/numberline-datadecide/'
ab=pd.read_csv(R+'results/datadecide/alpha_beta.csv')
recipes=list(ab.recipe)
rows=[];rej=[]
for r in recipes:
    d=json.load(open(R+f'results/datadecide/{r}.json'))
    fs=d['filter_summary']
    row={'recipe':r,'layer':d['selection']['selected_layer']}
    for g in '1234':
        row[f'rej{g}']=fs[g]['rejection_rate']; row[f'tried{g}']=fs[g]['candidates_tried']
        row[f'fail{g}']=fs[g]['failed_slots']
    # per-run rates
    runs=[('sel',x['seed'],x['groups']) for x in d['selection']['filter_stats']]+[('eval',x['seed'],x['filter_stats']) for x in d['evaluation']['runs']]
    for g in '1234':
        rr=[st[g]['rejection_rate'] for _,_,st in runs]; row[f'max{g}']=max(rr)
        tot=sum(st[g]['candidates_tried'] for _,_,st in runs); acc=sum(st[g]['accepted'] for _,_,st in runs)
        assert tot==fs[g]['candidates_tried'] and acc==fs[g]['accepted'], (r,g)
        for _,s,st in runs:
            for ex in st[g]['example_rejections']:
                rej.append({'recipe':r,'group':int(g),'seed':s,'target':ex['target'],'generated':ex['generated'],'response':ex['response'],'n_rej_run':st[g]['rejected']})
    rows.append(row)
df=pd.DataFrame(rows); rj=pd.DataFrame(rej)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(df[['recipe','layer','rej1','rej2','rej3','rej4','max3','max4','tried1','tried2','tried3','tried4','fail1','fail2','fail3','fail4']].round(3).to_string())
print('total rejected per group (sum over models):',{g:int((df[f'tried{g}']-240).sum()) for g in '1234'})
print('examples logged:',len(rj), 'of', int(sum((df[f'tried{g}']-240).sum() for g in '1234')))
df.to_csv('/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/fb/e01_filter.csv',index=False)
rj.to_csv('/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/fb/e01_rejections.csv',index=False)
