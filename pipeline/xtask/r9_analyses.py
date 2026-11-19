"""R9: additional checks (MAUS position-balanced contrasts, missingness-only model).

mauscheck  Is the MAUS heart-rate difficulty effect load or position? The palindrome (0-2-3-2-3-0) puts both 0-back blocks
           at the ends, so a linear position term cannot remove a U-shaped time course. Two position-balanced contrasts:
           (i) 3- vs 2-back within the middle blocks 2-5 (+ linear position); (ii) n-back minus the adjacent 0-back block,
           averaged over both ends ((pos2 - pos1) + (pos5 - pos6)) / 2, which cancels any linear drift.
missing    Can missingness alone separate the conditions? A model whose only feature is the share of valid beat intervals in
           each window (no physiology), under the same-session personal protocol (k = 6) and LOSO.
Outputs: results/r3/r9_maus_check.csv, r9_missingness.csv (+ _long)
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E
import mech as M
import prereg_r2 as P
import confirm as C
import clock_and_degrade as CD
from r7_analyses import score, auc, anchors

OUT = os.path.join(E.OUT, 'r3')


def stage_mauscheck():
    df, _ = P.load('MAUS_ECG', 'none')
    raw = pd.read_parquet(f'{P.W}/MAUS_ECG_windows.parquet')
    df = df.merge(raw[['pid', 'task', 't0', 'ibi_valid']].rename(columns={'ibi_valid': 'iv'}), on=['pid', 'task', 't0'], how='left')
    df.loc[df.iv < 0.8, 'hr_mean'] = np.nan
    b = df[df.family == 'nback'].groupby(['pid', 'position', 'level']).hr_mean.mean().reset_index().dropna()
    b['pid'] = b.pid.astype(str)
    rows = []
    # reference: 3- vs 0-back with linear position (as in Figure 2)
    r = C.fit('hr_mean ~ y + position', b[b.level.isin([0, 3])].assign(y=lambda x: (x.level == 3).astype(int)))
    e, se, lo, hi = C.coef(r, 'y'); rows.append(dict(contrast='3- vs 0-back, linear position (Figure 2)', bpm=e, lo=lo, hi=hi, n=b.pid.nunique()))
    # (i) middle blocks, position-balanced 3 vs 2
    mid = b[b.position.between(2, 5)].assign(y=lambda x: (x.level == 3).astype(int))
    r = C.fit('hr_mean ~ y + position', mid)
    e, se, lo, hi = C.coef(r, 'y'); rows.append(dict(contrast='3- vs 2-back, middle blocks 2-5, linear position', bpm=e, lo=lo, hi=hi, n=mid.pid.nunique()))
    # (ii) n-back minus adjacent 0-back, averaged over both ends (cancels linear drift)
    w = b.pivot_table(index='pid', columns='position', values='hr_mean')
    d = (((w[2] - w[1]) + (w[5] - w[6])) / 2).dropna()
    ci = M.boot_ci(d.values)
    rows.append(dict(contrast='n-back minus adjacent 0-back, both ends (linear drift cancels)', bpm=d.mean(), lo=ci[0], hi=ci[1], n=len(d),
                     p=stats.wilcoxon(d).pvalue))
    # the two ends separately: a load effect gives two positive differences, a linear drift gives opposite signs
    for a, z, lab in ((2, 1, 'start: 2-back (block 2) minus 0-back (block 1)'), (5, 6, 'end: 3-back (block 5) minus 0-back (block 6)')):
        dd = (w[a] - w[z]).dropna(); ci = M.boot_ci(dd.values)
        rows.append(dict(contrast=lab, bpm=dd.mean(), lo=ci[0], hi=ci[1], n=len(dd), p=stats.wilcoxon(dd).pvalue))
    o = pd.DataFrame(rows); o.to_csv(f'{OUT}/r9_maus_check.csv', index=False)
    return o


def stage_missing():
    sets = []
    d, feats, fams = CD.with_clock('UNIVERSE')
    raw = pd.read_parquet(f'{P.W}/UNIVERSE_windows.parquet')[['pid', 'session', 'task', 't0', 'ibi_valid']].rename(columns={'ibi_valid': 'iv'})
    d = d.merge(raw, on=['pid', 'session', 'task', 't0'], how='left')
    sets.append(('UNIVERSE wrist E4', 'hard vs. easy', d, ['unit', 'family']))
    for name, lab, sel, ylab in (('WESAD_WRIST', 'WESAD wrist E4', ('rest', 'stress'), 'stress vs. baseline'),
                                 ('COGWEAR_E4', 'CogWear wrist E4', ('rest', 'stroop'), 'Stroop vs. baseline'),
                                 ('MAUS_WRIST', 'MAUS wrist PPG', None, '3- vs. 0-back')):
        x = pd.read_parquet(f'{P.W}/{name}_windows.parquet').rename(columns={'ibi_valid': 'iv'})
        if sel:
            x = x[x.family.isin(sel)].assign(y=lambda q: (q.family == sel[1]).astype(int))
            if name == 'COGWEAR_E4':
                x = x[x.session == 'pre']
        else:
            x = x[x.task.isin(['trial1', 'trial3'])].assign(y=lambda q: (q.level == 3).astype(int))   # first 0- and 3-back blocks
        x['unit'] = x.pid.astype(str) + '|' + x.session.astype(str); x['family'] = 'one'
        sets.append((lab, ylab, x, ['unit', 'family']))
    rows = []
    for lab, ylab, d, keys in sets:
        d = d.copy(); d['pid'] = d.pid.astype(str)
        for (u, fam), own in d.groupby(keys):
            own = own.sort_values('t0')
            if own.y.nunique() < 2:
                continue
            pid = own.pid.iloc[0]
            a, t = anchors(own, 6)
            pop = d[(d.pid != pid) & (d.family == fam)]
            for prot, (Xtr, ytr, te) in {'same-session personal': (a[['iv']], a.y, t), 'population (LOSO)': (pop[['iv']], pop.y, own)}.items():
                if te.y.nunique() < 2:
                    continue
                try:
                    rows.append(dict(data=lab, contrast=ylab, protocol=prot, pid=pid, auc=auc(te.y, score(Xtr, ytr, te[['iv']]))))
                except ValueError:
                    pass
        print(lab, 'done', flush=True)
    r = pd.DataFrame(rows).dropna(subset=['auc']); r.to_csv(f'{OUT}/r9_missingness_long.csv', index=False)
    s = []
    for (lab, ylab, prot), g in r.groupby(['data', 'contrast', 'protocol'], sort=False):
        pp = g.groupby('pid').auc.mean(); s.append(dict(data=lab, contrast=ylab, protocol=prot, auc=pp.mean(), ci=M.boot_ci(pp.values), n=len(pp)))
    s = pd.DataFrame(s); s.to_csv(f'{OUT}/r9_missingness.csv', index=False)
    return s


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    for st in sys.argv[1:]:
        print(globals()[f'stage_{st}']().round(3).to_string(index=False), flush=True)
