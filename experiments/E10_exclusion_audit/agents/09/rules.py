import json,math,sys
import numpy as np
D=json.load(open('overlap.json')); R=D['recipes']; S=D['size']; n=len(R)
ab={'dolma1_7':'d17','dolma1_7-no-code':'d17-nc','dolma1_7-no-math-code':'d17-nmc','dolma1_7-no-reddit':'d17-nr','dolma1_7-no-flan':'d17-nf','dolma1_6plus':'d16p','c4':'c4','fineweb-pro':'fw-pro','fineweb-edu':'fw-edu','falcon':'fal','falcon-and-cc':'f&cc','falcon-and-cc-qc-10p':'f&cc-q10','falcon-and-cc-qc-20p':'f&cc-q20','falcon-and-cc-qc-orig-10p':'f&cc-qO','falcon-and-cc-qc-tulu-10p':'f&cc-qT','dclm-baseline':'dclm','dclm-baseline-qc-7p-fw2':'dclm-7fw2','dclm-baseline-qc-7p-fw3':'dclm-7fw3','dclm-baseline-qc-fw-3p':'dclm-fw3','dclm-baseline-qc-fw-10p':'dclm-fw10','dclm-baseline-qc-10p':'dclm-q10','dclm-baseline-qc-20p':'dclm-q20','dclm-baseline-25p-dolma1.7-75p':'mix25','dclm-baseline-50p-dolma1.7-50p':'mix50','dclm-baseline-75p-dolma1.7-25p':'mix75'}
def get(a,b,mode='point'):
    k=f'{a}|{b}'
    if k in D['pairs'][mode]: return D['pairs'][mode][k]
    m=D['pairs'][mode][f'{b}|{a}']; return dict(m,C_ab=m['C_ba'],C_ba=m['C_ab'])
def mat(key,mode='point'):
    M=np.eye(n)
    for i,a in enumerate(R):
        for j,b in enumerate(R):
            if i!=j: M[i,j]=get(a,b,mode)[key]
    return M
mode=sys.argv[1] if len(sys.argv)>1 else 'point'
out=[]
P=lambda *x: out.append(' '.join(str(y) for y in x))
# pair table
P('| A | B | I (B tok) | C(A in B) | C(B in A) | 1-TV | cos | shared trained tokens |')
P('|---|---|---|---|---|---|---|---|')
rows=[]
for i in range(n):
    for j in range(i+1,n):
        m=get(R[i],R[j],mode)
        if m['I']>0.05: rows.append((m['cos'],R[i],R[j],m))
for c,a,b,m in sorted(rows,reverse=True):
    P(f"| {ab[a]} | {ab[b]} | {m['I']:.1f} | {m['C_ab']:.2f} | {m['C_ba']:.2f} | {m['ov']:.2f} | {m['cos']:.2f} | {m['real']*100:.1f}% |")
P(f'\npairs with any overlap: {len(rows)} of {n*(n-1)//2}')
# n_eff (Clifford-Richardson-Hemon / Dutilleul): 1 + n^2/tr(R R)
for key,lab in [('cos','cos (doc-iid model)'),('ov','1-TV')]:
    M=mat(key,mode); P(f'n_eff [{lab}] = {1+n*n/np.sum(M*M):.1f}')
C=mat('C_ab',mode); Cmax=np.maximum(C,C.T)
P(f'n_eff [max containment] = {1+n*n/np.sum(Cmax*Cmax):.1f}')
# components
def comps(adj):
    seen=set(); cs=[]
    for i in range(n):
        if i in seen: continue
        st=[i]; c=[]
        while st:
            k=st.pop()
            if k in seen: continue
            seen.add(k); c.append(k); st+= [j for j in range(n) if adj[k,j] and j not in seen]
        cs.append(sorted(ab[R[x]] for x in c))
    return cs
for t in [0.05,0.5]:
    P(f'components, max-containment > {t}:', comps(Cmax>t))
    nm=[i for i,r in enumerate(R) if not r.startswith('dclm-baseline-') or '-qc-' in r]
    adj=Cmax>t; 
    for i in range(n):
        if R[i].endswith('p') and 'dolma1.7' in R[i]: adj[i,:]=False; adj[:,i]=False
    P(f'components (mixes removed), max-containment > {t}:', [c for c in comps(adj) if not any(x.startswith('mix') for x in c)])
# greedy keep rules
ORIG=['dclm-baseline','dolma1_7','dolma1_6plus','falcon-and-cc','falcon','fineweb-edu','c4','fineweb-pro']
def greedy(key,X,priority,exclude=()):
    keep=[]; drop={}
    for r in priority:
        if r in exclude: drop[r]='excluded a priori'; continue
        hit=[(k,get(r,k,mode)) for k in keep]
        if key=='C': bad=[(k,m['C_ab']) for k,m in hit if m['C_ab']>X]
        else: bad=[(k,m[key]) for k,m in hit if m[key]>X]
        if bad: drop[r]=', '.join(f'{ab[k]}:{v:.2f}' for k,v in bad)
        else: keep.append(r)
    return keep,drop
MIX=[r for r in R if 'dolma1.7' in r]
order_orig_first=ORIG+sorted([r for r in R if r not in ORIG],key=lambda r:-S[r])
order_size=sorted(R,key=lambda r:-S[r])
for name,key,X,order,exc in [
  ('R2 containment>0.5, originals first, mixes excluded','C',0.5,order_orig_first,MIX),
  ('R2b containment>0.5, size order, mixes kept','C',0.5,order_size,()),
  ('R3 cos>0.5, originals first, mixes excluded','cos',0.5,order_orig_first,MIX),
  ('R3b cos>0.5, originals first, mixes kept','cos',0.5,order_orig_first,()),
  ('R3c cos>0.3, originals first, mixes excluded','cos',0.3,order_orig_first,MIX),
  ('R3d cos>0.7, originals first, mixes excluded','cos',0.7,order_orig_first,MIX),
  ('R4 1-TV>0.5, originals first, mixes excluded','ov',0.5,order_orig_first,MIX),
  ('R4b 1-TV>0.8, originals first, mixes excluded','ov',0.8,order_orig_first,MIX),
  ]:
    k,d=greedy(key,X,order,exc)
    P(f'\n{name}: KEEP {len(k)}:', ', '.join(ab[x] for x in k))
    for r,why in d.items(): P(f'   drop {ab[r]}  <- {why}')
print('\n'.join(out))
