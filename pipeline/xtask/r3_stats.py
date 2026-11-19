"""R3 statistics on the development datasets.

(1) W family with W1 recomputed under the specified rest normalisation (the original run used session z-scores);
    Holm over the same 12 tests. -> results/r3/W_confirm_r3.csv
(2) Robustness of W2/W3: two-stage person-level analysis (per-person OLS of block means on difficulty + start time
    [+ family], then a one-sample t-test of the coefficients across people) and a by-person random slope for
    difficulty in COG-BCI. -> results/r3/robustness.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E
import confirm as CF

OUT = os.path.join(E.OUT, 'r3')
w = pd.read_csv(os.path.join(E.OUT, 'confirm', 'W_confirm.csv'))
w1 = pd.read_csv(f'{OUT}/w1_rest_vs_unit.csv'); w1 = w1[w1.norm == 'rest'].set_index('dataset')
w = w.copy(); w['normalisation'] = 'rest'
for ds in ('COGBCI', 'UNIVERSE', 'MPHRC'):
    i = w[(w.test == 'W1') & (w.dataset == ds)].index[0]
    w.loc[i, ['estimate', 'ci_lo', 'ci_hi', 'p']] = w1.loc[ds, ['diff', 'lo', 'hi', 'p']].values
w['p_holm'] = CF.holm(w.p.values)
w['supported'] = np.where(w.term.str.contains('TOST'), w.p_holm < .05, w.p_holm < .05)
w.to_csv(f'{OUT}/W_confirm_r3.csv', index=False)
print(w[['test', 'dataset', 'feature', 'estimate', 'ci_lo', 'ci_hi', 'p', 'p_holm']].round(4).to_string(index=False))

rows = []
for ds, feats in (('COGBCI', ['hr_mean']), ('UNIVERSE', ['hr_mean', 'temp_mean', 'scl_mean'])):
    b, _ = CF.blocks(ds)
    for f in feats:
        coefs = []
        for pid, g in b.dropna(subset=[f, 'start']).groupby('pid'):
            fam = ' + C(family)' if g.family.nunique() > 1 else ''
            try:
                r = smf.ols(f'{f} ~ y + start' + fam, g).fit()
                coefs.append(dict(pid=pid, y=r.params.get('y', np.nan), start=r.params.get('start', np.nan)))
            except Exception:
                pass
        c = pd.DataFrame(coefs).dropna()
        for term in ('y', 'start'):
            t = stats.ttest_1samp(c[term], 0)
            rng = np.random.default_rng(0); bs = [rng.choice(c[term].values, len(c)).mean() for _ in range(4000)]
            rows.append(dict(dataset=ds, feature=f, term=term, method='two-stage (person OLS, then t-test)', estimate=c[term].mean(),
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p=t.pvalue, n=len(c), share_negative=(c[term] < 0).mean()))
b, _ = CF.blocks('COGBCI')
bb = b.dropna(subset=['hr_mean', 'start'])
for re_, label in (('1', 'random intercept'), ('1 + y', 'random intercept + slope for difficulty')):
    r = smf.mixedlm('hr_mean ~ y + start + C(family)', bb, groups=bb.pid, re_formula=re_).fit(reml=True)
    lo, hi = r.conf_int().loc['y']
    rows.append(dict(dataset='COGBCI', feature='hr_mean', term='y', method=f'mixed model, {label}', estimate=r.params['y'], lo=lo, hi=hi,
                     p=r.pvalues['y'], n=bb.pid.nunique(), converged=r.converged))
o = pd.DataFrame(rows); o.to_csv(f'{OUT}/robustness.csv', index=False)
print(o.round(4).to_string(index=False))
