import pickle,numpy as np
rows=pickle.load(open('/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/agents/rows.pkl','rb'))
TAU=0.10
def tie(r,delta):
    L=r['L']; T={l for l,q in r['qm'].items() if l>=2 and r['qm'][L]-q<delta}
    return sorted(T|set(r['seed_arg']))
def rng(r,T,k): v=[r[k][l] for l in T]; return max(v)-min(v)
for delta in [0.005,0.0075,0.01,0.015,0.02]:
    s=[];b=[]
    for r in rows:
        T=tie(r,delta); a,c=rng(r,T,'bsel_all'),rng(r,T,'e06_all')
        if a>TAU and c>TAU: s.append(r['recipe'])
        elif a>TAU or c>TAU: b.append(r['recipe'])
    print('delta',delta,'FLAG',s,'BORDER',b)
for tau in [0.08,0.12,0.15]:
    s=[r['recipe'] for r in rows if rng(r,tie(r,0.01),'bsel_all')>tau and rng(r,tie(r,0.01),'e06_all')>tau]
    print('tau',tau,s)
print('|recipe|L*|q(L*)|runner-up|margin|margin %|argmax per seed 42/43/44|tie set T (δ=0.01)|β_E01sel over T|range E01|β_E06 over T|range E06|early pk (2-5)|mid pk (6-10)|two peaks?|in band 6-10|β_eval (45-47)|verdict|')
print('|'+'---|'*18)
for r in rows:
    L=r['L'];T=tie(r,0.01); a,c=rng(r,T,'bsel_all'),rng(r,T,'e06_all')
    qe,qm=r['qe'],r['qmid']; lo=min(qe,qm)
    two = (lo-r['dip']>=0.01)
    comp = two and abs(qe-qm)<0.01
    tp = ('yes, competitive' if comp else ('yes' if two else 'no'))
    v = 'FLAG' if (a>TAU and c>TAU) else ('borderline' if (a>TAU or c>TAU) else 'ok')
    bt=', '.join(f"L{l}:{r['bsel_all'][l]:.2f}" for l in T); b6=', '.join(f"L{l}:{r['e06_all'][l]:.2f}" for l in T)
    print(f"|{r['recipe']}|{L}|{r['qL']:.4f}|{r['runner']}|{r['margin']:.4f}|{100*r['rel']:.1f}|{'/'.join(map(str,r['seed_arg']))}|{{{', '.join(map(str,T))}}}|{bt}|{a:.2f}|{b6}|{c:.2f}|L{r['early']} {qe:.3f}|L{r['mid']} {qm:.3f}|{tp} (dip {r['dip']:.3f})|{'yes' if 6<=L<=10 else 'NO'}|{r['beval']:.3f}±{r['bevsd']:.3f}|{v}|")
