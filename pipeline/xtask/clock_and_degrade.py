"""Two diagnostics that need no new data.

(1) Clock baseline: the same evaluation protocols as Exp 3, but the ONLY feature is time in the session
    (UNIVERSE: minutes since the first task block; COG-BCI: task position + fraction of the block elapsed).
    If a time-only model scores like the physiological model under a protocol, that protocol cannot tell load from time.
(2) Wrist-PPG degradation on COG-BCI ECG: drop windows the way the wrist PPG lost them in UNIVERSE
    (33% overall, more in the hard level, and optionally more in high-heart-rate windows) and refit the
    pre-registered W2 block model. Does a ~1 SD heart-rate effect survive?
Outputs: results/confirm/clock_baseline.csv, results/confirm/degradation.csv
"""
import os, sys, warnings, itertools
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E
import mech as M
import confirm as C

OUT = os.path.join(E.OUT, 'confirm'); os.makedirs(OUT, exist_ok=True)
K = 6


def with_clock(ds):
    df, feats = M.load(ds, 'rest')
    bt = M.block_times(ds)
    df = df.merge(bt, on=['pid', 'session', 'task'], how='left')
    if ds == 'UNIVERSE':
        df['clock'] = df.start + df.t0 / 60.0
    else:  # COG-BCI: position (0-7) + elapsed fraction of a ~5-10 min block
        df['clock'] = df.start + df.t0 / df.groupby('block').t1.transform('max')
    d, fams = M.leveled(df)
    return d.dropna(subset=['clock']).reset_index(drop=True), feats, fams


def fit_score(Xtr, ytr, Xte):
    m = E.model(); m.fit(Xtr, ytr)
    return m.decision_function(Xte)


def safe_auc(y, Xtr, ytr, Xte):
    try:
        return roc_auc_score(y, fit_score(Xtr, ytr, Xte))
    except ValueError:  # e.g. every anchor window lacks valid cardiac features
        return np.nan


def clock_protocols(ds):
    d, feats, fams = with_clock(ds)
    rows = []
    for rep, cols in (('clock', ['clock']), ('physiology', feats)):
        for pid in d.pid.astype(str).unique():
            mine = d[d.pid.astype(str) == pid]; others = d[d.pid.astype(str) != pid]
            units = list(mine.unit.unique())
            for A in fams:
                pop = others[others.family == A]
                for u in units:
                    own = mine[(mine.unit == u) & (mine.family == A)].sort_values('t0')
                    if own.y.nunique() < 2:
                        continue
                    anchor = pd.concat([g.head(K) for _, g in own.groupby('y')])
                    last = anchor.groupby('y').t1.max()
                    test = own[own.t0 >= own.y.map(last)]
                    if test.y.nunique() == 2:
                        rows.append(dict(dataset=ds, rep=rep, protocol='same-session personal', pid=pid, family=A,
                                         auc=safe_auc(test.y, anchor[cols], anchor.y, test[cols])))
                        rows.append(dict(dataset=ds, rep=rep, protocol='population (LOSO)', pid=pid, family=A,
                                         auc=safe_auc(test.y, pop[cols], pop.y, test[cols])))
                    for ut in units:
                        if ut == u:
                            continue
                        tst = mine[(mine.unit == ut) & (mine.family == A)]
                        if tst.y.nunique() == 2:
                            rows.append(dict(dataset=ds, rep=rep, protocol='cross-session personal', pid=pid, family=A,
                                             auc=safe_auc(tst.y, anchor[cols], anchor.y, tst[cols])))
    return pd.DataFrame(rows).dropna(subset=['auc'])


def degradation(nrep=200, seed=0):
    rng = np.random.default_rng(seed)
    df, feats = M.load('COGBCI', 'rest')
    d, fams = M.leveled(df)
    d = d.dropna(subset=['hr_mean']).merge(M.block_times('COGBCI'), on=['pid', 'session', 'task'], how='left')
    d['pid'] = d.pid.astype(str)
    zhr = (d.hr_mean - d.hr_mean.mean()) / d.hr_mean.std()
    scenarios = {  # dropout probability per window; calibrated to UNIVERSE wrist PPG (M8 and CP8)
        'none': lambda: np.zeros(len(d)),
        'random 33%': lambda: np.full(len(d), 0.33),
        'MNAR level (easy 31.5%, hard 34.5%)': lambda: np.where(d.y == 1, 0.345, 0.315),
        'Stroop-like (easy 45%, hard 58%)': lambda: np.where(d.y == 1, 0.58, 0.45),
        'Stroop-like + high-HR windows dropped more (OR 2/SD)': lambda: _calib(np.where(d.y == 1, 0.58, 0.45), zhr, 2.0),
    }
    rows = []
    for name, pfun in scenarios.items():
        for r in range(1 if name == 'none' else nrep):
            keep = rng.random(len(d)) >= pfun()
            b = d[keep].groupby(['pid', 'session', 'unit', 'family', 'task', 'y']).agg(hr_mean=('hr_mean', 'mean'), start=('start', 'first')).reset_index()
            res = C.fit('hr_mean ~ y + start + C(family)', b)
            est, se, lo, hi = C.coef(res, 'y')
            rows.append(dict(scenario=name, rep=r, kept=keep.mean(), estimate=est, ci_lo=lo, ci_hi=hi, p=C.p_greater(est, se)))
    return pd.DataFrame(rows)


def _calib(base, z, odds_ratio):
    """Scale dropout odds by odds_ratio per SD of heart rate, then rescale so each level keeps its mean dropout."""
    lo = np.log(base / (1 - base)) + np.log(odds_ratio) * z.values
    p = 1 / (1 + np.exp(-lo))
    for v in np.unique(base):
        m = base == v
        p[m] = np.clip(p[m] * v / p[m].mean(), 0, 0.98)
    return p


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    cb = pd.concat([clock_protocols(ds) for ds in ['UNIVERSE', 'COGBCI']])
    cb.to_csv(f'{OUT}/clock_baseline.csv', index=False)
    print(cb.groupby(['dataset', 'protocol', 'rep', 'pid']).auc.mean().groupby(['dataset', 'protocol', 'rep']).agg(['mean', 'count']).round(3).unstack('rep').to_string())
    dg = degradation()
    dg.to_csv(f'{OUT}/degradation.csv', index=False)
    print(dg.groupby('scenario').agg(kept=('kept', 'mean'), est=('estimate', 'mean'), est_p5=('estimate', lambda s: s.quantile(.05)),
                                       sig_rate=('p', lambda s: (s < 0.05).mean())).round(3).to_string())
