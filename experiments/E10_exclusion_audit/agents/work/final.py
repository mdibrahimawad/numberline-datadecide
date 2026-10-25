import json,numpy as np
W='/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/work/'
exec(open(W+'e01.py').read().split("R='/home")[0])  # fit_spacing_direct
a=json.load(open(W+'e01_rows.json')); b=json.load(open(W+'e06_rows.json'))
st1000=np.median([b[r['recipe']]['step6_rel'] for r in a]); st10000=np.median([b[r['recipe']]['step9_rel'] for r in a])
print('median digit steps rel to d3: 1000:',round(st1000,3),'10000:',round(st10000,3))
d=np.array([b[r['recipe']]['bc']-r['bd'] for r in a]); med=np.median(d); mad=np.median(abs(d-med))*1.4826
tab=[]
for r,x in zip(a,d):
    rec=r['recipe']; G=np.array(r['gaps']); gm=G.mean(0)
    # filter composition bias in g3/g4
    def shift(g, minority_pred):
        ex=r['ex'][g]; rr=r['rej'][g]
        if rr==0 or not ex: return 0.0
        f=sum(minority_pred(t) for t in ex)/len(ex)
        pa=min(2*rr*f,0.99); pb=min(2*rr*(1-f),0.99)
        return 0.5-(1-pa)/((1-pa)+(1-pb))   # drop in share of the class that differs from the 4-digit core
    s3=shift('3',lambda t:t<1000); s4=shift('4',lambda t:t>=10000)
    bias_d3=-(s3*st1000+s4*st10000)*gm[2]
    g2=gm.copy(); g2[2]-=bias_d3  # corrected (remove bias)
    b0=fit_spacing_direct(gm)[0]; b1=fit_spacing_direct(g2)[0]
    tseed=gm/G.std(0,ddof=1)
    minrel=(G/G.mean(1,keepdims=True)).min()
    ls=r['margin']; sel_b=r['selbeta']; b2=r['b2nd']
    rep_z=(sel_b-r['bd'])/np.sqrt((r['selbsd']**2+r['bd_sd']**2)/2)  # ddof0 stds, 3 seeds each
    nbmax=max(abs(v-sel_b) for v in r['nb'])
    row=dict(recipe=rec,L=r['L'],ev=r['ev'],rho=r['rho'],bd=r['bd'],bdsd=r['bd_sd'],bl=r['bl'],blsd=r['bl_sd'],lnratio=np.log(r['bl']/r['bd']),
      gaps=gm,minrel=minrel,tmin=tseed.min(),targ=int(tseed.argmin())+1,rej=r['rej'],maxrej=max(r['rej'].values()),maxg=max(r['rej'],key=r['rej'].get),
      tok1=r['tok']['1'],paired=sum(h for h in [0]),margin=ls,L2=r['l2'],dsel=abs(b2-sel_b),nbmax=nbmax,selb=sel_b,repz=rep_z,
      e06=b[rec]['bc'],resid=x-med,zr=(x-med)/mad,s3=s3,s4=s4,dbeta_filter=b1-b0, t40=min(b[rec]['t40']))
    tab.append(row)
json.dump([{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in t.items()} for t in tab],open(W+'table.json','w'),indent=1,default=float)
for t in tab:
    print(f"{t['recipe'][:30]:30s} L{t['L']:2d} EV{t['ev']:.2f} rho{t['rho']:.3f} bd{t['bd']:.3f}({t['bdsd']:.3f}) bl{t['bl']:.3f}({t['blsd']:.3f}) ln{t['lnratio']:+.2f} gaps{np.round(t['gaps'],1)} minrel{t['minrel']:.2f} tmin{t['tmin']:.1f}@d{t['targ']} rej{t['maxrej']:.2f}g{t['maxg']} tok1={t['tok1']} marg{t['margin']:.3f} dsel{t['dsel']:.2f} nb{t['nbmax']:.2f} repz{t['repz']:+.1f} e06{t['e06']:.3f} zr{t['zr']:+.1f} s3{t['s3']:.2f} s4{t['s4']:.2f} dbF{t['dbeta_filter']:+.3f} t40min{t['t40']:.1f}")
bs=np.array([t['bd'] for t in tab]); print('beta between-model sd',bs.std(ddof=1),'median seed sd',np.median([t['bdsd'] for t in tab]))
