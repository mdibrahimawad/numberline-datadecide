import json, csv, sys, numpy as np

def fit_spacing_direct(g):
    g=np.asarray(g,float); i=np.arange(len(g))
    def loss(lb):
        B=np.exp(lb*i); s=g@B/(B@B); return np.sum((g-s*B)**2)
    xs=np.linspace(-8,8,160001); ls=np.array([loss(x) for x in xs[::100]]); x0=xs[::100][ls.argmin()]
    xs=np.linspace(x0-0.02,x0+0.02,40001); ls=np.array([loss(x) for x in xs]); lb=xs[ls.argmin()]
    b=np.exp(lb); B=b**i; s=g@B/(B@B); p=s*B; r2=1-np.sum((g-p)**2)/np.sum((g-g.mean())**2)
    return b,s,r2
R='/home/user/numberline-datadecide/results/datadecide/'
recipes=[r['recipe'] for r in csv.DictReader(open(R+'alpha_beta.csv'))]

def recon(b,s,bl):
    # unknowns d0,d1,d2 (i=0,1,2). eqs: sum d_i b^i = s sum b^2i ; sum i d_i b^i = s sum i b^2i ; d2 = bl^2 d0
    i=np.arange(3); A=np.array([b**i, i*b**i, [bl**2,0,-1]]); y=np.array([s*np.sum(b**(2*i)), s*np.sum(i*b**(2*i)),0])
    return np.linalg.solve(A,y)
def r2log(g):
    x=np.arange(1,4); c=np.polyfit(x,np.log(g),1); p=np.exp(c[1])*np.exp(c[0])**x
    return 1-np.sum((g-p)**2)/np.sum((g-g.mean())**2)
rows=[]
for rec in recipes:
    d=json.load(open(R+rec+'.json'))
    ev=d['evaluation']; L=ev['target_layer']; S=ev['summary']
    gaps=[];chk=[]
    for run in ev['runs']:
        g=recon(run['beta'],run['beta_scale'],run['beta_log'])
        b2,s2,r2=fit_spacing_direct(np.abs(g))
        chk.append(max(abs(b2-run['beta']),abs(r2-run['r2_beta']), abs(r2log(np.abs(g))-run['r2_beta_log'])))
        gaps.append(g)
    G=np.array(gaps)  # 3 seeds x 3 gaps
    gm=G.mean(0); gsd=G.std(0,ddof=1)
    fs=d['filter_summary']
    rej={k:v['rejection_rate'] for k,v in fs.items()}
    failed=sum(v['failed_slots'] for v in fs.values())
    # eval-only rejection
    evrej={}
    for gname in ['1','2','3','4']:
        t=sum(r['filter_stats'][gname]['candidates_tried'] for r in ev['runs']); a=sum(r['filter_stats'][gname]['accepted'] for r in ev['runs'])
        evrej[gname]=(t-a)/t
    sel=d['selection']; ls=sel['layer_scores']
    qs=sorted(((v['q_median'],int(k)) for k,v in ls.items()),reverse=True)
    q1,l1=qs[0]; q2,l2=qs[1]
    LD=sel['layers_direct']
    selbeta=LD[str(L)]['beta_mean']; selbsd=LD[str(L)]['beta_std']
    nb=[LD[str(x)]['beta_mean'] for x in (L-1,L+1) if str(x) in LD]
    b2nd=LD[str(l2)]['beta_mean']
    td=d['tokenization_diagnostics']
    # rejected example targets
    ex={g:[] for g in '1234'}
    for run in ev['runs']:
        for g in '1234':
            ex[g]+= [e['target'] for e in run['filter_stats'][g]['example_rejections']]
    for e in sel['filter_stats']:
        for g in '1234':
            ex[g]+= [x['target'] for x in e['groups'][g]['example_rejections']]
    rows.append(dict(recipe=rec,L=L,ev=S['explained_variance_mean'],ev_sd=S['explained_variance_std'],rho=S['rho_mean'],
        bd=S['beta_mean'],bd_sd=S['beta_std'],bl=S['beta_log_mean'],bl_sd=S['beta_log_std'],r2=S['r2_beta_mean'],
        bd_runs=[r['beta'] for r in ev['runs']], bl_runs=[r['beta_log'] for r in ev['runs']],
        gaps=G.tolist(), gm=gm.tolist(), gsd=gsd.tolist(), minrel=float((G/G.mean(1,keepdims=True)).min()), neg=int((G<=0).sum()),
        chk=max(chk), rej=rej, evrej=evrej, failed=failed, q1=q1,q2=q2,l2=l2, margin=(q1-q2)/q1, selbeta=selbeta,selbsd=selbsd, nb=nb, b2nd=b2nd,
        tok=td['target_token_length_histogram'], ex=ex, rho_runs=[r['rho'] for r in ev['runs']], fte=d['final_token_is_equals']))
json.dump(rows,open('/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/work/e01_rows.json','w'),indent=1)
for r in rows:
    print(f"{r['recipe'][:30]:30s} L{r['L']:2d} ev{r['ev']:.2f} rho{r['rho']:.3f} bd{r['bd']:.3f}±{r['bd_sd']:.3f} bl{r['bl']:.3f}±{r['bl_sd']:.3f} chk{r['chk']:.1e} gm",
      np.round(r['gm'],2), 'cv',np.round(np.array(r['gsd'])/np.array(r['gm']),2), f"minrel{r['minrel']:.2f} neg{r['neg']}",
      'rej',[round(v,2) for v in r['rej'].values()], f"fail{r['failed']} marg{r['margin']:.3f} L2={r['l2']} b2={r['b2nd']:.2f} selb{r['selbeta']:.2f}±{r['selbsd']:.2f} nb",np.round(r['nb'],2),'tok',r['tok'], r['fte'])
