"""Human-robot collaboration case (MultiPhysio-HRC, day 2). EXPLORATORY: this dataset was already analysed in Exp 1-4.

Day-2 protocol (Bussola et al., arXiv 2510.00703, verified): rest -> manual disassembly (battery models 1-5)
-> collaborative disassembly with a cobot (models 1-5) -> rest. Order is fixed for everyone.
Questions:
  Q1 engagement: can each work mode be told apart from rest (LOSO, within-person AUROC)?
  Q2 manual vs cobot: physiology, NASA-TLX, and how much of the difference session drift alone could produce.
  Q3 drift over the HRC session: start rest vs end rest.
Features are z-scored per person over all day-2 windows (no labels), cardiac features need ibi_valid >= 0.8.
Output: results/confirm/hrc_case.csv (+ printed summary)
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E
import mech as M
import confirm as C

OUT = os.path.join(E.OUT, 'confirm')


def day2():
    df, feats = M.load('MPHRC', 'none')
    d = df[df.session.astype(str) == 'D2'].copy()
    d['rep'] = d.task.str.extract(r'(\d+)$').astype(float)
    # global position in the fixed protocol: rest-1 = 0, manual 1..5 -> 1..5, cobot 1..5 -> 6..10, rest-2 = 11
    d['pos'] = np.select([d.task == 'rest-1', d.family == 'manual', d.family == 'cobot', d.task == 'rest-2'],
                         [0, d.rep, 5 + d.rep, 11], np.nan)
    g = d.groupby('pid')[feats]
    d[feats] = (d[feats] - g.transform('mean')) / g.transform('std').replace(0, np.nan)
    d['pid'] = d.pid.astype(str)
    return d, feats


def loso_auc(d, feats, pos, neg):
    x = d[d.family.isin([pos, neg]) | d.task.isin([pos, neg])].copy()
    x['y'] = (x.family.eq(pos) | x.task.eq(pos)).astype(int)
    out = {}
    for pid in x.pid.unique():
        tr, te = x[x.pid != pid], x[x.pid == pid]
        if te.y.nunique() < 2:
            continue
        m = E.model().fit(tr[feats], tr.y)
        out[pid] = roc_auc_score(te.y, m.decision_function(te[feats]))
    return pd.Series(out)


def main():
    d, feats = day2()
    rows = []
    # Q1 engagement
    for mode in ('manual', 'cobot'):
        a = loso_auc(d, feats, mode, 'rest-1')
        rows.append(dict(question='Q1', contrast=f'{mode} vs start rest', estimate=a.mean(), ci=str(M.boot_ci(a.values)), n=len(a)))
    a = loso_auc(d, feats, 'cobot', 'manual')
    rows.append(dict(question='Q2', contrast='cobot vs manual (LOSO AUROC)', estimate=a.mean(), ci=str(M.boot_ci(a.values)), n=len(a)))
    rows.append(dict(question='Q2', contrast='clock-only model, cobot vs manual', estimate=1.0, ci='by design (fixed order)', n=len(a)))
    # NASA-TLX (block subjective) manual vs cobot
    s = d[d.family.isin(['manual', 'cobot'])].groupby(['pid', 'family']).subjective.mean().unstack().dropna()
    rows.append(dict(question='Q2', contrast='NASA-TLX cobot - manual', estimate=(s.cobot - s.manual).mean(),
                     ci=str(M.boot_ci((s.cobot - s.manual).values)), p=wilcoxon(s.cobot - s.manual).pvalue, n=len(s)))
    # block-level physiology: condition difference vs what the within-condition repetition trend predicts
    b = d[d.family.isin(['manual', 'cobot'])].groupby(['pid', 'family', 'rep', 'pos'])[feats].mean().reset_index()
    b['cobot'] = (b.family == 'cobot').astype(int)
    for f in ['hr_mean', 'rmssd', 'scl_mean', 'resp_rate']:
        bb = b.dropna(subset=[f])
        r1 = C.fit(f'{f} ~ cobot', bb)
        diff, se, lo, hi = C.coef(r1, 'cobot')
        r2 = C.fit(f'{f} ~ cobot + rep', bb)  # rep = 1..5 inside each mode
        slope, se2, lo2, hi2 = C.coef(r2, 'rep')
        rows.append(dict(question='Q2', contrast=f'{f}: cobot - manual (person-z SD)', estimate=diff, ci=f'[{lo:.2f}, {hi:.2f}]', n=bb.pid.nunique()))
        rows.append(dict(question='Q2', contrast=f'{f}: repetition slope within mode (per model)', estimate=slope, ci=f'[{lo2:.2f}, {hi2:.2f}]', n=bb.pid.nunique()))
        rows.append(dict(question='Q2', contrast=f'{f}: drift expected from 5 positions at that slope', estimate=5 * slope,
                         ci=f'[{5 * lo2:.2f}, {5 * hi2:.2f}]', n=bb.pid.nunique()))
    # Q3 start rest vs end rest
    r = d[d.task.isin(['rest-1', 'rest-2'])].groupby(['pid', 'task'])[feats].mean()
    for f in ['hr_mean', 'rmssd', 'scl_mean', 'resp_rate']:
        w = r[f].unstack().dropna()
        if len(w) > 5:
            dd = w['rest-2'] - w['rest-1']
            rows.append(dict(question='Q3', contrast=f'{f}: end rest - start rest (person-z SD)', estimate=dd.mean(), ci=str(M.boot_ci(dd.values)),
                             p=wilcoxon(dd).pvalue, n=len(w)))
    t = pd.DataFrame(rows); t.to_csv(f'{OUT}/hrc_case.csv', index=False)
    pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 70)
    print(t.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
