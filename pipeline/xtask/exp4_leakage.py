"""Experiment 4: how much of same-session personal calibration is block identity rather than load?

(a) Split-half null. Take any single task block (one family, one level, one person-session) and cut it in half:
    first half y=0, second half y=1. There is no load difference, so any AUROC above 0.5 comes from the
    signal drifting over time. Same protocol as Exp 3: anchor = first k windows of each half, test = windows of
    each half that start after its last anchor window ends. Conditions: personal (anchor only) and pop
    (other people's split-half labels for the same family; tests for a shared time-on-task trend).
(b) Real labels, binned by position in block. Same-session personal and pop models from Exp 3 (k=6),
    AUROC on test windows grouped by how far into the block they start.
Features: handcrafted, rest-normalised per person-session (as in Exp 1/3).
Output: results/exp4_null_<DS>.csv, results/exp4_position_<DS>.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E

BUDGETS = [3, 6, 12]
BINS = [0, 120, 240, 360, 1e9]  # seconds from block start
BIN_LABELS = ['0-2 min', '2-4 min', '4-6 min', '>6 min']


def prep(ds):
    df, feats = E.load(ds)
    df = E.normalise(df.reset_index(drop=True), feats, 'rest')
    df = df[df.level >= 0].dropna(subset=feats, how='all').reset_index(drop=True)
    return df, feats


def fit_score(Xtr, ytr, Xte):
    m = E.model(); m.fit(Xtr, ytr)
    return m.decision_function(Xte)


def split_half(df):
    """Label each block's first half 0 and second half 1. Returns rows with y and a 'block' key."""
    out = []
    for key, g in df.groupby(['unit', 'family', 'level']):
        g = g.sort_values('t0')
        mid = g.t0.iloc[len(g) // 2]
        g = g.assign(y=(g.t0 >= mid).astype(int), block='|'.join(map(str, key)))
        out.append(g)
    return pd.concat(out)


def run_null(ds):
    df, feats = prep(ds)
    d = split_half(df)
    rows = []
    for (u, fam, lvl), g in d.groupby(['unit', 'family', 'level']):
        pid = g.pid.iloc[0]
        g = g.sort_values('t0')
        pop = d[(d.pid != pid) & (d.family == fam)]
        for k in BUDGETS:
            anchor = pd.concat([h.head(k) for _, h in g.groupby('y')])
            last = anchor.groupby('y').t1.max()
            test = g[(g.t0 >= g.y.map(last)) & ~g.index.isin(anchor.index)]
            if test.y.nunique() < 2 or (test.y == 0).sum() < 3 or (test.y == 1).sum() < 3:
                continue
            sp = fit_score(anchor[feats], anchor.y, test[feats])
            sq = fit_score(pop[feats], pop.y, test[feats])
            for cond, s in (('personal', sp), ('pop', sq)):
                rows.append(dict(dataset=ds, unit=u, pid=pid, family=fam, level=lvl, k=k, cond=cond,
                                 auc=roc_auc_score(test.y, s), n_test=len(test)))
    r = pd.DataFrame(rows)
    r.to_csv(f'{E.OUT}/exp4_null_{ds}.csv', index=False)
    return r


def run_position(ds, k=6):
    df, feats = prep(ds)
    d, fams = E.binary_task(df)
    rows = []
    for u, gu in d.groupby('unit'):
        pid = gu.pid.iloc[0]
        for A in fams:
            own = gu[gu.family == A].sort_values('t0')
            if own.y.nunique() < 2:
                continue
            anchor = pd.concat([h.head(k) for _, h in own.groupby('y')])
            last = anchor.groupby('y').t1.max()
            test = own[own.t0 >= own.y.map(last)].copy()
            if test.y.nunique() < 2:
                continue
            pop = d[(d.pid != pid) & (d.family == A)]
            test['sp'] = fit_score(anchor[feats], anchor.y, test[feats])
            test['sq'] = fit_score(pop[feats], pop.y, test[feats])
            test['pos'] = pd.cut(test.t0, BINS, right=False, labels=BIN_LABELS)
            for b, gb in test.groupby('pos', observed=True):
                if gb.y.nunique() < 2 or len(gb) < 6:
                    continue
                for cond, col in (('personal', 'sp'), ('pop', 'sq')):
                    rows.append(dict(dataset=ds, unit=u, pid=pid, family=A, pos=b, cond=cond,
                                     auc=roc_auc_score(gb.y, gb[col]), n=len(gb)))
    r = pd.DataFrame(rows)
    r.to_csv(f'{E.OUT}/exp4_position_{ds}.csv', index=False)
    return r


def summarise(r, by):
    p = r.groupby(by + ['pid']).auc.mean().reset_index()
    return p.groupby(by).auc.agg(['mean', 'count']).round(3)


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    for ds in sys.argv[1:] or ['COGBCI', 'UNIVERSE']:
        n = run_null(ds)
        print(f'== {ds} split-half null (no load difference) ==\n', summarise(n, ['k', 'cond']).unstack('cond').to_string(), flush=True)
        p = run_position(ds)
        print(f'== {ds} real labels by position in block (k=6) ==\n', summarise(p, ['pos', 'cond']).unstack('cond').to_string(), flush=True)
