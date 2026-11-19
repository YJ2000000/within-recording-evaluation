"""Summarise Exp 3 for every representation: per-person mean AUROC, bootstrap 95% CI, Wilcoxon vs pop,
and representation comparisons (FM vs HC, HC+FM vs HC) within each condition.
Usage: python exp3_stats.py RESULTS_DIR  -> RESULTS_DIR/exp3_stats_all.csv, exp3_rep_compare.csv
"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

D = sys.argv[1] if len(sys.argv) > 1 else '.'
REPS = ['HC', 'FM', 'HC_FM']


def per_person():
    out = []
    for ds in ['COGBCI', 'UNIVERSE']:
        for rep in REPS:
            r = pd.read_csv(f'{D}/exp3_{ds}_{rep}.csv')
            r['setting'] = np.where(r.source == r.target, 'same-session same-task', 'same-session cross-task')
            x = pd.read_csv(f'{D}/exp3_crosssession_{ds}_{rep}.csv'); x['setting'] = 'cross-session same-task'
            for t in (r, x):
                t['rep'] = rep
                out.append(t.groupby(['dataset', 'rep', 'setting', 'k', 'cond', 'pid']).auc.mean().reset_index())
    return pd.concat(out)


def boot(v, n=2000):
    rng = np.random.default_rng(0)
    m = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return f'[{np.percentile(m, 2.5):.2f},{np.percentile(m, 97.5):.2f}]'


def pval(a):
    a = np.asarray(a)
    return wilcoxon(a).pvalue if np.any(a != 0) else 1.0


if __name__ == '__main__':
    P = per_person()
    W = P.pivot_table(index=['dataset', 'rep', 'setting', 'k', 'pid'], columns='cond', values='auc').reset_index()
    rows = []
    for key, g in W.groupby(['dataset', 'rep', 'setting', 'k']):
        g = g.dropna(subset=['pop', 'personal', 'joint'])
        for c in ['pop', 'personal', 'joint']:
            rows.append(dict(zip(['dataset', 'rep', 'setting', 'k'], key), cond=c, n=len(g), auc=round(g[c].mean(), 3), ci=boot(g[c].values)))
        for c in ['personal', 'joint']:
            dlt = g[c] - g['pop']
            rows.append(dict(zip(['dataset', 'rep', 'setting', 'k'], key), cond=f'{c}-pop', n=len(g), auc=round(dlt.mean(), 3), ci=f'p={pval(dlt):.3g}'))
    S = pd.DataFrame(rows)
    S.to_csv(f'{D}/exp3_stats_all.csv', index=False)
    C = []
    R = P.pivot_table(index=['dataset', 'setting', 'k', 'cond', 'pid'], columns='rep', values='auc').reset_index()
    for key, g in R.groupby(['dataset', 'setting', 'k', 'cond']):
        for rep in ['FM', 'HC_FM']:
            h = g.dropna(subset=['HC', rep]); dlt = h[rep] - h['HC']
            C.append(dict(zip(['dataset', 'setting', 'k', 'cond'], key), compare=f'{rep}-HC', n=len(h), delta=round(dlt.mean(), 3), p=round(pval(dlt), 4)))
    C = pd.DataFrame(C)
    C.to_csv(f'{D}/exp3_rep_compare.csv', index=False)
    pd.set_option('display.width', 220)
    for k in [6, 24]:
        T = S[(S.k == k) & S.cond.isin(['pop', 'personal', 'joint'])].pivot_table(index=['dataset', 'setting'], columns=['rep', 'cond'], values='auc', aggfunc='first')
        print(f'--- k={k} ---\n', T.to_string())
    print(S[(S.k == 6) & S.cond.str.endswith('-pop')].pivot_table(index=['dataset', 'setting', 'cond'], columns='rep', values=['auc', 'ci'], aggfunc='first').to_string())
    print(C[C.k == 6].pivot_table(index=['dataset', 'setting', 'cond'], columns='compare', values=['delta', 'p'], aggfunc='first').to_string())
