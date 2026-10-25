import pickle,numpy as np,sys
rows=pickle.load(open('/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/rows.pkl','rb'))
TAU=0.10
def tie(r,delta):
    L=r['L']; T=set()
    for l,q in r['qm'].items():
        if l==L or l<2: continue
        if r['qm'][L]-q<delta: T.add(l)
    T|=set(r['seed_arg'])-{L}
    return sorted(T)
for delta in [0.005,0.01,0.015,0.02]:
    fl=[]
    for r in rows:
        T=tie(r,delta)
        d1=max([abs(r['bsel_all'][l]-r['bsel_all'][r['L']]) for l in T],default=0)
        d6=max([abs(r['e06_all'][l]-r['e06_all'][r['L']]) for l in T],default=0)
        if d1>TAU and d6>TAU: fl.append((r['recipe'],T,round(d1,2),round(d6,2)))
    print(delta,len(fl),fl)
print()
out=[]
for r in rows:
    L=r['L'];T=tie(r,0.01)
    det=[(l, round(r['qm'][L]-r['qm'][l],4), round(r['bsel_all'][l]-r['bsel_all'][L],2), round(r['e06_all'][l]-r['e06_all'][L],2)) for l in T]
    inband=max(range(6,11),key=lambda l:r['qm'][l])
    print(r['recipe'],L,'T=',det,'| evq-selq=%.3f'%(np.median(r['qev'])-r['qL']),'| bev-bsel=%.2f'%(r['beval']-r['bselL']),'| inband',inband, 'gap %.4f'%(r['qL']-r['qm'][inband]), 'dB01 %.2f dB06 %.2f'%(r['bsel_all'][inband]-r['bselL'], r['e06_all'][inband]-r['e06_all'][L]))
