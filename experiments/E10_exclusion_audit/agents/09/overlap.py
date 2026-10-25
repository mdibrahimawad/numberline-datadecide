"""Pairwise data-overlap estimates for the 25 DataDecide recipes.
Uses only configs/datadecide_data_map.json (file lists) and mixture.json (token sizes).
No alpha/beta values are read."""
import json, csv, itertools, math, sys
ROOT='/home/user/numberline-datadecide'
OUT=sys.argv[1]
recs=[r['recipe'] for r in csv.DictReader(open(f'{ROOT}/results/datadecide/alpha_beta.csv'))]
dm=json.load(open(f'{ROOT}/configs/datadecide_data_map.json'))['recipes']
mix={r:json.load(open(f'{ROOT}/results/corpus_alpha_datadecide/{r}/mixture.json')) for r in recs}
size={r:sum(v['tokens'] for v in mix[r].values())/1e9 for r in recs}

# ---- file-level pools (exact shared files); tokens per file = pool tokens / pool files
def pool_of(path):
    p=path.split('/')[1:]
    s='/'.join(p)
    if 'v1_6-decontaminated/pes2o' in s: return 'pes2o'
    if 'v1_6-decontaminated/books' in s: return 'books'
    if 'v1_6-decontaminated/wiki' in s: return 'wiki'
    if s.startswith('olmo-mix/v1_7'): return 'd17cc'
    if s.startswith('falcon-refinedweb/v2'): return 'falcon'
    if s.startswith('c4/'): return 'c4'
    if s.startswith('starcoder'): return 'star'
    if s.startswith('reddit'): return 'reddit'
    if 'arxiv' in s: return 'arxiv'
    if 'stackexchange' in s: return 'se'
    if 'algebraic-stack' in s: return 'algstack'
    if 'open-web-math' in s: return 'owm'
    if s.startswith('tulu_flan/v2'): return 'tulu2'
    if s.startswith('cc-news'): return 'ccnews'
    if s.startswith('megawika'): return 'megawika'
    if s.startswith('fastdclm'): return 'dclm'
    if s.startswith('prox_fineweb'): return 'fwpro'
    if s.startswith('fineweb-edu'): return 'fwedu'
    return None
# pool token sizes (unique), from mixture.json of the full recipe + OLMo size comments
POOLTOK={'d17cc':598.42,'falcon':456.40,'star':263.78,'c4':138.44,'reddit':79.94,
 'pes2o':57.21,'books':5.256,'wiki':3.689,'arxiv':27.97,'se':19.63,
 'algstack':12.623,'owm':12.734,'tulu2':16.54,'ccnews':14.26,'megawika':4.56,
 'dclm':3857.1,'fwpro':82.7,'fwedu':192.2}
FILEBASED=[r for r in recs if r!='dolma1_6plus' and '-qc-' not in r]
files={}
for r in FILEBASED:
    m={}
    for p in dm[r]['paths']:
        m[p]=m.get(p,0)+1
    files[r]=m
allfiles={}
for r in FILEBASED:
    for p in files[r]: allfiles.setdefault(pool_of(p),set()).add(p)
assert None not in allfiles, allfiles.get(None)
ftok={pl:POOLTOK[pl]/len(fs) for pl,fs in allfiles.items()}
def tok(p): return ftok[pool_of(p)]
# fraction of each pool covered by a file-based recipe
def phi(r,pl):
    fs=allfiles[pl]; return sum(1 for p in files[r] if pool_of(p)==pl)/len(fs)

# ---- doc-level non-file relations
# dolma1_6plus contains (doc-level) the docs of these Dolma-1.7 pools, plus 206.5B of its own
D16_CONTAINS={'d17cc','c4','falcon','star','reddit','pes2o','books','wiki','arxiv','se','algstack','owm'}
# filtered subsets: pool-part sizes (B tokens)
SUB={
 'dclm-baseline-qc-7p-fw2':{'dclm':507.5},'dclm-baseline-qc-7p-fw3':{'dclm':218.0},
 'dclm-baseline-qc-fw-3p':{'dclm':118.1},'dclm-baseline-qc-fw-10p':{'dclm':98.5},
 'dclm-baseline-qc-10p':{'dclm':82.1},'dclm-baseline-qc-20p':{'dclm':171.1},
 'falcon-and-cc-qc-10p':{'d17cc':102.6,'falcon':43.5},'falcon-and-cc-qc-20p':{'d17cc':166.5,'falcon':119.4},
 'falcon-and-cc-qc-orig-10p':{'d17cc':98.1,'falcon':49.7},'falcon-and-cc-qc-tulu-10p':{'d17cc':83.7,'falcon':45.2}}
NESTED={frozenset(['dclm-baseline-qc-7p-fw2','dclm-baseline-qc-7p-fw3']),
        frozenset(['dclm-baseline-qc-10p','dclm-baseline-qc-20p']),
        frozenset(['falcon-and-cc-qc-10p','falcon-and-cc-qc-20p'])}
FW_PAIR=frozenset(['dclm-baseline-qc-fw-3p','dclm-baseline-qc-fw-10p'])

def I(a,b,mode='point'):
    """unique-token intersection (B) of the document sets of a and b, with
    multiplicity-weighted versions for wiki handled via returned list of (tokens, mA, mB)."""
    parts=[]
    if a==b: raise
    A,B=a,b
    # both file based
    if A in files and B in files:
        for p in set(files[A])&set(files[B]):
            parts.append((tok(p),files[A][p],files[B][p]))
        if {A,B}=={'fineweb-pro','fineweb-edu'}:
            parts.append(({'low':0.0,'point':1.0,'high':10.0}[mode],1,1))
        return parts
    # dolma1_6plus vs file-based
    if 'dolma1_6plus' in (A,B) and not ('-qc-' in A or '-qc-' in B):
        X=B if A=='dolma1_6plus' else A
        for p,m in files[X].items():
            if pool_of(p) in D16_CONTAINS:
                parts.append((tok(p),1 if A=='dolma1_6plus' else m, m if A=='dolma1_6plus' else 1))
        return parts
    # subset vs something
    if A in SUB and B in SUB:
        pa,pb=SUB[A],SUB[B]
        for pl in set(pa)&set(pb):
            N=POOLTOK[pl]
            if frozenset([A,B]) in NESTED: v=min(pa[pl],pb[pl])
            elif frozenset([A,B])==FW_PAIR: v={'low':pa[pl]*pb[pl]/N,'point':0.255*118.1,'high':min(pa[pl],pb[pl])}[mode]
            else: v={'low':pa[pl]*pb[pl]/N,'point':pa[pl]*pb[pl]/N,'high':min(pa[pl],pb[pl])}[mode]
            parts.append((v,1,1))
        return parts
    S,X=(A,B) if A in SUB else (B,A)
    for pl,s in SUB[S].items():
        if X=='dolma1_6plus': f=1.0 if pl in D16_CONTAINS else 0.0
        else: f=phi(X,pl) if pl in allfiles and any(pool_of(p)==pl for p in files[X]) else 0.0
        if f>0: parts.append((s*f,1,1))
    return parts

def selfstats(r):
    if r in files:
        tot=sum(tok(p)*m for p,m in files[r].items()); sq=sum(tok(p)*m*m for p,m in files[r].items())
    elif r=='dolma1_6plus': tot=size[r]; sq=size[r]
    else: tot=sum(SUB[r].values()); sq=tot
    return tot,sq
ST={r:selfstats(r) for r in recs}
def metrics(a,b,mode='point'):
    parts=I(a,b,mode)
    (ta,qa),(tb,qb)=ST[a],ST[b]
    inter=sum(t for t,_,_ in parts)
    Cab=sum(t*ma for t,ma,mb in parts)/ta   # share of a's training stream whose docs are in b
    Cba=sum(t*mb for t,ma,mb in parts)/tb
    ov=sum(t*min(ma/ta,mb/tb) for t,ma,mb in parts)          # 1 - total variation
    cos=sum(t*ma*mb for t,ma,mb in parts)/(ta*tb)/math.sqrt((qa/ta**2)*(qb/tb**2))
    # expected identical trained tokens (share of a 100B run), indep. shuffles
    pa=min(1,100/ta); pb=min(1,100/tb)
    real=sum(t*min(1,ma*pa)*min(1,mb*pb) for t,ma,mb in parts)/100
    return dict(I=inter,C_ab=Cab,C_ba=Cba,ov=ov,cos=cos,real=real)
res={}
for mode in ['low','point','high']:
    res[mode]={f'{a}|{b}':metrics(a,b,mode) for a,b in itertools.combinations(recs,2)}
json.dump({'recipes':recs,'size':{r:ST[r][0] for r in recs},'pairs':res},open(OUT,'w'),indent=1)
print({r:round(ST[r][0],1) for r in recs})
