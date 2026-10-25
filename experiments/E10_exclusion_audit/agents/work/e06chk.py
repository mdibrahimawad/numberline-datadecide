import json,numpy as np,csv
W='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/work/'
rows=json.load(open(W+'e01_rows.json'))
B='/home/user/numberline-datadecide/results/beta_fine/step69369-seed-default/'
out={}
for r in rows:
    rec=r['recipe']; L=r['L']
    z=np.load(B+rec+'.npz'); j=json.load(open(B+rec+'.json'))
    res=[]
    for s in (45,46,47):
        t=z[f'seed{s}_targets'].astype(float); g=z[f'seed{s}_groups']; p=z[f'seed{s}_pc_scores'][L,:,0].astype(float)
        if np.corrcoef(np.log10(t),p)[0,1]<0: p=-p
        dec=[0,3,6,9]; m=np.array([p[g==k].mean() for k in dec]); sd=np.array([p[g==k].std(ddof=1) for k in dec])
        gaps=np.diff(m)
        # digit steps
        a6=p[(g==6)&(t<1000)].mean(); b6=p[(g==6)&(t>=1000)].mean()
        a9=p[(g==9)&(t<10000)].mean(); b9=p[(g==9)&(t>=10000)].mean()
        a3=p[(g==3)&(t<100)].mean(); b3=p[(g==3)&(t>=100)].mean()
        res.append(dict(gaps=gaps,sd=sd,step3=b3-a3,step6=b6-a6,step9=b9-a9))
    G=np.array([x['gaps'] for x in res]); SD=np.array([x['sd'] for x in res])
    st6=np.mean([x['step6'] for x in res]); st9=np.mean([x['step9'] for x in res]); st3=np.mean([x['step3'] for x in res])
    gm=G.mean(0); sdw=SD.mean(0)
    # t for E01-size n=40 per group: gap/(sqrt(sd_i^2/40+sd_j^2/40))
    t40=gm/np.sqrt(sdw[:-1]**2/40+sdw[1:]**2/40)
    old=j['report']['old_layer']
    out[rec]=dict(L=L,gaps=gm.tolist(),neg=int((G<=0).sum()),sdw=sdw.tolist(),t40=t40.tolist(),step6_rel=st6/gm[2],step9_rel=st9/gm[2],step3_rel=st3/gm[1],
                  bc=old['beta_coarse']['mean'],bc_sd=old['beta_coarse']['std'],err=old['err_coarse']['mean'],ev=old['ev']['mean'],rho=old['rho']['mean'])
    o=out[rec]
    print(f"{rec[:28]:28s} L{L} gaps{np.round(gm,1)} neg{o['neg']} wsd{np.round(sdw,1)} t40{np.round(t40,1)} step100/d2 {o['step3_rel']:.2f} step1000/d3 {o['step6_rel']:.2f} step10000/d3 {o['step9_rel']:.2f} | E06 bc {o['bc']:.3f}±{o['bc_sd']:.3f} vs E01 {r['bd']:.3f} diff {o['bc']-r['bd']:+.3f}")
json.dump(out,open(W+'e06_rows.json','w'),indent=1)
