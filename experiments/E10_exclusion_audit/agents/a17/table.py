import pandas as pd,numpy as np
c=pd.read_csv('ctx_rows.csv'); c['dbc']=c.bc_adj-c.bc_raw
e=pd.read_csv('e01_seeds.csv').set_index('recipe'); s=pd.read_csv('seed_table.csv',index_col=0)
fv=pd.read_csv('/home/user/numberline-datadecide/results/beta_fine/fine_vs_data.csv').set_index('recipe')
ft=pd.read_csv('first_try.csv')
ft['mlog_c']=ft.mlog-ft.groupby(['recipe','seed','g']).mlog.transform('mean')
sel={r:(g[g.rej==1].mlog_c.mean()-g[g.rej==0].mlog_c.mean() if g.rej.sum()>=10 else np.nan) for r,g in ft.groupby('recipe')}
maxrej=ft.groupby(['recipe','g']).rej.mean().groupby('recipe').max()
cs=c[c.layer_kind=='sel'].groupby('recipe'); co=c[c.layer_kind=='old'].groupby('recipe')
T=pd.DataFrame({'E01_L':e.L,'E01_beta':e.beta_direct_mean,'E01_sd_raw':e.ev_sd,'E01_sd_model':s.e01_ev_inter,'E01_6seed_model':s.e01_6_inter,
 'E06_L':fv.layer,'E06_bcoarse':cs.bc_all.mean(),'E06_c_sd':s.e06c_sd,'E06_c_model':s.e06c_inter,'E06_f_sd':s.e06f_sd,
 'ctx_pR2':cs.pr2_ctx.mean(),'ctx_slope':cs.slope_mlog_per_span.mean(),'dB_adj':cs.dbc.mean(),'dB_adj_max':cs.dbc.apply(lambda x:np.abs(x).max()),
 'ctx_pR2_E01L':co.pr2_ctx.mean(),'dB_adj_E01L':co.dbc.mean(),'filt_dmlog':pd.Series(sel),'max_grp_rej':maxrej})
order=list(e.index); T=T.loc[order]
bsd={'e01':0.1865,'e06c':0.1504,'e06f':0.1462}
med=s.median()
def flags(r):
    f=[]
    if r.E01_sd_model>0.5*bsd['e01']: f.append('S1-E01')
    if r.E06_c_model>0.5*bsd['e06c']: f.append('S1-E06c')
    if r.E06_f_sd>0.5*bsd['e06f']: f.append('S1-E06f')
    o=[k for k,col in (('E01',"e01_ev_inter"),('E01x6',"e01_6_inter"),('E06c',"e06c_inter"),('E06f',"e06f_sd")) if s.loc[r.name,col]>3*med[col]]
    if o: f.append('S2('+','.join(o)+')')
    if abs(r.dB_adj)>med['e06c_inter'] or abs(r.dB_adj_E01L)>med['e06c_inter']: f.append('C2')
    if r.max_grp_rej>0.30: f.append('C4')
    return ';'.join(f) if f else '-'
T['flags']=T.apply(flags,axis=1)
T.to_csv('final_table.csv')
fmt=T.copy()
for col in fmt.columns:
    if fmt[col].dtype.kind=='f': fmt[col]=fmt[col].map(lambda v: '' if pd.isna(v) else f'{v:.3f}')
print('| recipe | '+' | '.join(fmt.columns)+' |'); print('|'+'---|'*(len(fmt.columns)+1))
for rec,row in fmt.iterrows(): print(f'| {rec} | '+' | '.join(str(v) for v in row.values)+' |')
print(med.round(4).to_dict())
print('corr pR2 vs E06 layer', np.corrcoef(T.ctx_pR2,T.E06_L)[0,1], 'corr pR2 vs dB', np.corrcoef(T.ctx_pR2,T.dB_adj)[0,1])
