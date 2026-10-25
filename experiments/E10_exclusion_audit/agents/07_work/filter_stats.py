import json, csv, math, sys
import numpy as np
R='/home/user/numberline-datadecide/results'
recs=[r['recipe'] for r in csv.DictReader(open(f'{R}/datadecide/alpha_beta.csv'))]
CENTRES=[10**(1+g/3) for g in range(10)]
def band(g):
    c=CENTRES[g]; return np.arange(math.ceil(c*0.9), math.floor(c*1.1)+1)
out=[]
for rec in recs:
    d=json.load(open(f'{R}/datadecide/{rec}.json'))
    # E01: selection filter_summary (aggregate) + eval runs
    fs=d['filter_summary']
    e01_tried=sum(v['candidates_tried'] for v in fs.values()); e01_acc=sum(v['accepted'] for v in fs.values())
    e01_g={g:v['rejection_rate'] for g,v in fs.items()}
    e01_fail=sum(v['failed_slots'] for v in fs.values())
    # eval runs
    ev_t={g:0 for g in '1234'}; ev_a={g:0 for g in '1234'}; ev_fail=0
    for r in d['evaluation']['runs']:
        for g,v in r['filter_stats'].items():
            ev_t[g]+=v['candidates_tried']; ev_a[g]+=v['accepted']; ev_fail+=v['failed_slots']
    ev_g={g:1-ev_a[g]/ev_t[g] for g in ev_t}
    rho=np.mean([r['rho'] for r in d['evaluation']['runs']]); ev=np.mean([r['explained_variance'] for r in d['evaluation']['runs']])
    # E06
    f=json.load(open(f'{R}/beta_fine/step69369-seed-default/{rec}.json'))
    t6={str(g):0 for g in range(10)}; a6=dict(t6); fail6=0
    for seed,gs in f['filter'].items():
        for g,v in gs.items():
            t6[g]+=v['tried']; a6[g]+=v['accepted']; fail6+=v['failed_slots']
    e06_g={g:1-a6[g]/t6[g] for g in t6}
    e06_tot=1-sum(a6.values())/sum(t6.values())
    rep=f['report']['selected_layer']
    # selection-bias check: accepted target mean log10 vs band uniform
    z=np.load(f'{R}/beta_fine/step69369-seed-default/{rec}.npz')
    shifts=[]
    for s in (45,46,47):
        T=z[f'seed{s}_targets']; G=z[f'seed{s}_groups']
        for g in range(10):
            b=band(g); exp=np.mean(np.log10(b)); got=np.mean(np.log10(T[G==g]))
            # in units of the 1/3-decade group spacing; also sampling SE under uniform
            shifts.append((g,s,(got-exp)/(1/3), np.std(np.log10(b))/math.sqrt((G==g).sum())/(1/3)))
    sh=np.array([x[2] for x in shifts]); se=np.array([x[3] for x in shifts])
    zsc=sh/se
    out.append(dict(recipe=rec,
        e01_sel_rej=1-e01_acc/e01_tried, e01_sel_maxg=max(e01_g,key=e01_g.get), e01_sel_max=max(e01_g.values()),
        e01_eval_rej=1-sum(ev_a.values())/sum(ev_t.values()), e01_eval_max=max(ev_g.values()), e01_eval_maxg=max(ev_g,key=ev_g.get),
        e01_g3_eval=ev_g['3'], e01_g4_eval=ev_g['4'], e01_failed=e01_fail+ev_fail,
        e06_rej=e06_tot, e06_max=max(e06_g.values()), e06_maxg=int(max(e06_g,key=e06_g.get)), e06_failed=fail6,
        e06_g={g:round(v,3) for g,v in e06_g.items()},
        rho_e01=rho, ev_e01=ev, rho_e06=rep['rho']['mean'], ev_e06=rep['ev']['mean'], layer_e06=rep['layer'],
        shift_max_abs=float(np.max(np.abs(sh))), z_max_abs=float(np.max(np.abs(zsc))), n_z_gt3=int((np.abs(zsc)>3).sum())))
json.dump(out, open(sys.argv[1],'w'), indent=1)
for o in out:
    print(f"{o['recipe']:34s} E01sel {o['e01_sel_rej']:.3f} max {o['e01_sel_max']:.2f}(g{o['e01_sel_maxg']}) | E01ev {o['e01_eval_rej']:.3f} g3 {o['e01_g3_eval']:.2f} g4 {o['e01_g4_eval']:.2f} fail {o['e01_failed']} | E06 {o['e06_rej']:.3f} max {o['e06_max']:.2f}(g{o['e06_maxg']}) fail {o['e06_failed']} | rho {o['rho_e01']:.3f}/{o['rho_e06']:.3f} EV {o['ev_e01']:.2f}/{o['ev_e06']:.2f} | shift {o['shift_max_abs']:.3f} zmax {o['z_max_abs']:.1f} nz>3 {o['n_z_gt3']}")
