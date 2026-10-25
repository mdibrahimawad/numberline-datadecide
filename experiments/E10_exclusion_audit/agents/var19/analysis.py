"""Variance structure of beta ONLY (no alpha). Auditor 19."""
import json, sys
import numpy as np, pandas as pd
from scipy import optimize, stats

R = '/home/user/numberline-datadecide'
OUT = sys.argv[1]
fam = pd.read_csv(f'{R}/results/beta_fine/fine_vs_data.csv')[['recipe', 'family', 'layer', 'old_layer']]
recipes = list(pd.read_csv(f'{R}/results/datadecide/alpha_beta.csv')['recipe'])
fam = fam.set_index('recipe').loc[recipes].reset_index()

rows = []
for r in recipes:
    d = json.load(open(f'{R}/results/datadecide/{r}.json'))
    e = json.load(open(f'{R}/results/beta_fine/step69369-seed-default/{r}.json'))
    lay01 = d['evaluation']['target_layer']
    b01 = [x['beta'] for x in d['evaluation']['runs']]
    # E01 selection-seed info: near-tie and beta at runner-up layer (selection seeds 42-44)
    ls = {int(k): v['q_median'] for k, v in d['selection']['layer_scores'].items()}
    order = sorted(ls, key=lambda k: -ls[k])
    top, ru = order[0], order[1]
    ld = d['selection']['layers_direct']
    lay06 = e['report']['selected_layer']['layer']
    old = e['report']['old_layer']['layer']
    assert old == lay01
    ps = e['per_seed']
    rec = dict(recipe=r, L01=lay01, L06=lay06,
               b01_s=b01,
               co_old=[p['layers'][str(old)]['beta_coarse'] for p in ps],
               ct_old=[p['layers'][str(old)]['beta_cont'] for p in ps],
               co_sel=[p['layers'][str(lay06)]['beta_coarse'] for p in ps],
               ct_sel=[p['layers'][str(lay06)]['beta_cont'] for p in ps],
               sel_top=top, sel_ru=ru, q_top=ls[top], q_ru=ls[ru],
               bsel_top=ld[str(top)]['beta_mean'], bsel_ru=ld[str(ru)]['beta_mean'],
               rej_max=max(v['rejection_rate'] for v in d['filter_summary'].values()))
    rows.append(rec)
D = pd.DataFrame(rows).merge(fam[['recipe', 'family']], on='recipe')
METRICS = {'E01 beta_direct': 'b01_s', 'E06 beta_coarse@E01 layer': 'co_old',
           'E06 beta_cont@E01 layer': 'ct_old', 'E06 beta_coarse@sel layer': 'co_sel',
           'E06 beta_cont@sel layer': 'ct_sel'}
for name, c in METRICS.items():
    D[c + '_m'] = D[c].apply(np.mean)
    D[c + '_v'] = D[c].apply(lambda x: np.var(x, ddof=1))

fams = D['family'].values
ufam = list(dict.fromkeys(fams))


def reml(y, fam_idx, mod_idx, n_f, n_m):
    """REML for y = mu + f + m + e (nested). returns (sf2, sm2, se2)."""
    N = len(y)
    Zf = np.zeros((N, n_f)); Zf[np.arange(N), fam_idx] = 1
    Zm = np.zeros((N, n_m)); Zm[np.arange(N), mod_idx] = 1
    Kf, Km = Zf @ Zf.T, Zm @ Zm.T
    X = np.ones((N, 1))

    def nll(t):
        sf, sm, se = np.exp(t)
        V = sf * Kf + sm * Km + se * np.eye(N)
        L = np.linalg.cholesky(V)
        Vi = np.linalg.inv(V)
        XtViX = X.T @ Vi @ X
        b = np.linalg.solve(XtViX, X.T @ Vi @ y)
        r = y - X @ b
        return 0.5 * (2 * np.log(np.diag(L)).sum() + np.log(np.linalg.det(XtViX)) + r @ Vi @ r)
    best = None
    for s0 in [(-4, -4, -6), (-2, -3, -5), (-3, -2, -7), (-6, -2, -6)]:
        o = optimize.minimize(nll, np.array(s0, float), method='Nelder-Mead',
                              options=dict(xatol=1e-6, fatol=1e-9, maxiter=20000))
        if best is None or o.fun < best.fun:
            best = o
    return np.exp(best.x)


def long(col):
    y, fi, mi = [], [], []
    for j, row in D.iterrows():
        for v in row[col]:
            y.append(v); fi.append(ufam.index(row['family'])); mi.append(j)
    return np.array(y), np.array(fi), np.array(mi)


def anova_means(m, se2):
    """fixed-effect ANOVA on model means: shares of observed var of means."""
    g = pd.Series(m).groupby(fams)
    grand = m.mean()
    ss_t = ((m - grand) ** 2).sum()
    ss_b = sum(len(v) * (v.mean() - grand) ** 2 for _, v in g)
    ss_w = ss_t - ss_b
    ss_noise = se2 * (len(m) - 1)  # expected SS of pure noise around grand mean
    ss_noise_w = se2 * (len(m) - len(ufam))  # expected noise SS inside families
    return ss_t, ss_b, ss_w, ss_noise, ss_noise_w


out = {}
rng = np.random.default_rng(19)
lines = []
P = lambda *a: lines.append(' '.join(str(x) for x in a))
P('# per-metric variance decomposition')
for name, c in METRICS.items():
    y, fi, mi = long(c)
    sf, sm, se = reml(y, fi, mi, len(ufam), len(D))
    # parametric bootstrap CI
    bs = []
    for b in range(300):
        f_eff = rng.normal(0, np.sqrt(sf), len(ufam))
        m_eff = rng.normal(0, np.sqrt(sm), len(D))
        yb = y.mean() + f_eff[fi] + m_eff[mi] + rng.normal(0, np.sqrt(se), len(y))
        bs.append(reml(yb, fi, mi, len(ufam), len(D)))
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    m = D[c + '_m'].values
    pooled = D[c + '_v'].mean()
    se2 = pooled / 3
    tot = sf + sm + se / 3
    ss_t, ss_b, ss_w, ss_n, ss_nw = anova_means(m, se2)
    icc_true = sf / (sf + sm)
    icc_obs = sf / tot
    nf = pd.Series(fams).value_counts()
    neff_true = sum(n / (1 + (n - 1) * icc_true) for n in nf)
    neff_obs = sum(n / (1 + (n - 1) * icc_obs) for n in nf)
    rel = 1 - se2 / m.var(ddof=1)
    G = np.sqrt(max(m.var(ddof=1) - se2, 0) / se2)
    H = (4 * G + 1) / 3
    out[name] = dict(sf=sf, sm=sm, se=se, sf_ci=(lo[0], hi[0]), sm_ci=(lo[1], hi[1]), se_ci=(lo[2], hi[2]),
                     share_f=sf / tot, share_m=sm / tot, share_e=se / 3 / tot,
                     pooled_seed_sd=np.sqrt(pooled), se_mean=np.sqrt(se2), sd_means=m.std(ddof=1),
                     eta2_between=ss_b / ss_t, within_ss_share=ss_w / ss_t, noise_ss_share=ss_n / ss_t,
                     noise_share_of_within=ss_nw / ss_w,
                     icc_true=icc_true, icc_obs=icc_obs, neff_true=neff_true, neff_obs=neff_obs,
                     reliability=rel, G=G, strata=H)
    P(f'\n## {name}')
    for k, v in out[name].items():
        P(f'{k}: {np.round(v, 4) if not isinstance(v, tuple) else tuple(np.round(v, 5))}')
    # family means
    P('family means/sd/n:', {f: (round(D.loc[D.family == f, c + '_m'].mean(), 3),
                                 round(D.loc[D.family == f, c + '_m'].std(ddof=1), 3) if (D.family == f).sum() > 1 else None,
                                 int((D.family == f).sum())) for f in ufam})

# seed-variance heterogeneity (E01): chi2 with 2 df, Bonferroni 25
P('\n# seed variance heterogeneity')
for name, c in METRICS.items():
    pooled = D[c + '_v'].mean()
    stat = 2 * D[c + '_v'] / pooled
    p_lo = stats.chi2.cdf(stat, 2); p_hi = stats.chi2.sf(stat, 2)
    p2 = 2 * np.minimum(p_lo, p_hi)
    bart = stats.bartlett(*[np.array(x) for x in D[c]])
    D[c + '_p_seedvar'] = p2
    P(name, 'Bartlett', round(bart.statistic, 2), 'p', f'{bart.pvalue:.2g}', 'flags(p<0.002):',
      list(D.loc[p2 < 0.05 / 25, 'recipe']), 'min p', D.loc[p2.idxmin(), 'recipe'] if hasattr(p2, 'idxmin') else '', round(p2.min(), 4))

# cross-protocol reproducibility: E01 beta_direct vs E06 beta_coarse at same layer
d = D['b01_s_m'] - D['co_old_m']
exp_var = D['b01_s_v'].mean() / 3 + D['co_old_v'].mean() / 3
P('\n# cross-protocol E01 direct - E06 coarse@E01layer')
P('mean diff', round(d.mean(), 4), 'sd diff', round(d.std(ddof=1), 4), 'expected sd from seeds', round(np.sqrt(exp_var), 4),
  'excess (model x protocol) var', round(d.var(ddof=1) - exp_var, 5), 'excess sd', round(np.sqrt(max(d.var(ddof=1) - exp_var, 0)), 4))
P('pearson r', round(stats.pearsonr(D['b01_s_m'], D['co_old_m'])[0], 3), 'spearman', round(stats.spearmanr(D['b01_s_m'], D['co_old_m'])[0], 3))
med = d.median(); mad = np.median(np.abs(d - med))
D['xprot_diff'] = d
D['xprot_modz'] = 0.6745 * (d - med) / mad
D['xprot_z_noise'] = (d - d.mean()) / np.sqrt(D['b01_s_v'].mean() / 3 + D['co_old_v'].mean() / 3)
P('flags |modz|>3.5:', list(D.loc[D.xprot_modz.abs() > 3.5, 'recipe']))
P('flags |z_noise|>3.09:', list(D.loc[D.xprot_z_noise.abs() > 3.09, 'recipe']))

# layer-choice component: E06 cont selected vs old layer
dl = D['ct_sel_m'] - D['ct_old_m']
P('\n# layer choice (E06 beta_cont sel - old)  n switched', int((D.L06 != D.L01).sum()),
  'var of diff over all 25', round(dl.var(ddof=1), 5), 'sd', round(dl.std(ddof=1), 4),
  'mean sq diff/2 (per-measurement layer noise var)', round((dl ** 2).mean() / 2, 5))
# E01 selection near-tie: runner-up within 5% and beta differs
D['tie_margin'] = 1 - D.q_ru / D.q_top
D['dbeta_ru'] = D.bsel_ru - D.bsel_top
se_e01 = np.sqrt(D['b01_s_v'].mean() / 3)
D['near_tie'] = D.tie_margin < 0.05
D['layer_unstable'] = D.near_tie & (D.dbeta_ru.abs() > 3.09 * np.sqrt(2) * se_e01)
P('E01 near-tie (<5%) models:', list(D.loc[D.near_tie, 'recipe']))
P('E01 layer-unstable (near tie & |dbeta|>3.09*sqrt2*SE):', list(D.loc[D.layer_unstable, 'recipe']))

# within-family outliers
P('\n# within-family outliers')
for name, c in METRICS.items():
    se_m = np.sqrt(D[c + '_v'].mean() / 3)
    sm = out[name]['sm']
    mz, zn, zp = [], [], []
    for j, row in D.iterrows():
        g = D[D.family == row.family]
        x = row[c + '_m']
        others = g.loc[g.recipe != row.recipe, c + '_m'].values
        vals = g[c + '_m'].values
        if len(vals) >= 5:
            md = np.median(vals); mad = np.median(np.abs(vals - md))
            mz.append(0.6745 * (x - md) / mad)
        else:
            mz.append(np.nan)
        if len(others) >= 1:
            k = len(others)
            # var of median of others under pure noise ~ (pi/2) se^2/k (k>=3), mean otherwise
            vmed = (np.pi / 2 if k >= 3 else 1) * se_m ** 2 / k
            ref = np.median(others)
            zn.append((x - ref) / np.sqrt(se_m ** 2 + vmed))
            # predictive LOO z incl. genuine within-family variance
            zp.append((x - others.mean()) / np.sqrt(sm + se_m ** 2 + (sm + se_m ** 2) / k))
        else:
            zn.append(np.nan); zp.append(np.nan)
    D[c + '_modz'] = mz; D[c + '_znoise'] = zn; D[c + '_zpred'] = zp
    P(name, 'modz>3.5:', list(D.loc[D[c + '_modz'].abs() > 3.5, 'recipe']),
      '| znoise>3.09:', list(D.loc[D[c + '_znoise'].abs() > 3.09, 'recipe']),
      '| zpred>3.09:', list(D.loc[D[c + '_zpred'].abs() > 3.09, 'recipe']))

# mix dose-response: linear interpolation between dclm-baseline (0%) and dolma1_7 (100%)
P('\n# mix dose-response (dolma share): endpoints dclm-baseline, dolma1_7')
for name, c in METRICS.items():
    se_m = np.sqrt(D[c + '_v'].mean() / 3)
    v = D.set_index('recipe')[c + '_m']
    pts = {0: v['dclm-baseline'], 25: v['dclm-baseline-75p-dolma1.7-25p'], 50: v['dclm-baseline-50p-dolma1.7-50p'],
           75: v['dclm-baseline-25p-dolma1.7-75p'], 100: v['dolma1_7']}
    s = []
    for p in (25, 50, 75):
        pred = pts[0] + (pts[100] - pts[0]) * p / 100
        w = p / 100
        z = (pts[p] - pred) / (se_m * np.sqrt(1 + w ** 2 + (1 - w) ** 2))
        s.append(f'{p}%: obs {pts[p]:.3f} pred {pred:.3f} z {z:+.2f}')
    P(name, '; '.join(s), '| monotone:', all(np.diff([pts[k] for k in (0, 25, 50, 75, 100)]) > 0))

P('\n' + '\n'.join(str(x) for x in []))
D.to_pickle(OUT + '/D.pkl')
open(OUT + '/out.txt', 'w').write('\n'.join(lines))
print('\n'.join(lines))
