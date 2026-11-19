"""Experiment 1: cross-task transfer of workload models (leave-one-subject-out).

For every pair (source family A, target family B) within a dataset:
  train on family A windows of all other participants (easy=0 vs hard=1),
  test on the held-out participant's family B windows.
Primary metric: within-person AUROC (ranks hard above easy for that person).
Secondary (calibration preview): balanced accuracy on B using
  (pop)   the population decision threshold, vs
  (anch)  a person-specific threshold set from k anchor windows per level of the
          SAME person's family A (no target-task labels used).
Normalisation variants (unlabelled, per person-session):
  none    : raw features
  rest    : z-score w.r.t. that person-session's rest windows (deployable: needs a few minutes of rest)
Outputs: <windows>/../results/exp1_<DATASET>_<norm>.csv and summary tables.
"""
import os, sys, itertools, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
warnings.filterwarnings('ignore')

R = os.environ.get('DATA_ROOT', os.path.expanduser('/path/to/project/data/raw'))
W = os.path.join(os.path.dirname(R), 'windows'); OUT = os.path.join(os.path.dirname(R), 'results')
os.makedirs(OUT, exist_ok=True)
CARD = ['hr_mean', 'hr_std', 'rmssd', 'sdnn', 'pnn50']
EDA = ['scl_mean', 'scl_slope', 'phasic_std', 'scr_rate']
FEATS = {'COGBCI': CARD,
         'UNIVERSE': CARD + EDA + ['temp_mean', 'temp_slope', 'acc_std'],
         'MPHRC': CARD + EDA + ['resp_rate', 'resp_amp'],
         'COGLOAD': ['hr_mean', 'hr_std', 'rmssd', 'scl_mean', 'scl_slope', 'temp_mean', 'temp_slope', 'acc_std']}
K_ANCHOR = int(os.environ.get('K_ANCHOR', '6'))  # anchor windows per level (6 x 10-s step ~ 80 s of signal)
RNG = np.random.default_rng(0)


def load(ds):
    df = pd.read_parquet(f'{W}/{ds}_windows.parquet')
    feats = [f for f in FEATS[ds] if f in df.columns]
    if 'ibi_valid' in df.columns:  # cardiac features only from clean windows
        bad = df['ibi_valid'] < 0.8
        df.loc[bad, [c for c in CARD if c in df.columns]] = np.nan
    df['unit'] = df['pid'].astype(str) + '|' + df['session'].astype(str)
    return df, feats


def normalise(df, feats, how):
    if how == 'none':
        return df
    out = df.copy()
    for u, g in df.groupby('unit'):
        ref = g[g.family == 'rest'][feats]
        if len(ref) < 3:
            out.loc[g.index, feats] = np.nan
            continue
        mu, sd = ref.mean(), ref.std().replace(0, np.nan)
        sd = sd.fillna(ref.std().mean() if np.isfinite(ref.std().mean()) else 1.0)
        out.loc[g.index, feats] = (g[feats] - mu) / sd.clip(lower=1e-6)
    return out


def binary_task(df):
    """Keep families with >=2 difficulty levels; label lowest=0, highest=1 (middle level dropped)."""
    fam_lv = df[df.level >= 0].groupby('family')['level'].agg(['min', 'max', 'nunique'])
    fams = fam_lv[fam_lv['nunique'] >= 2].index.tolist()
    d = df[df.family.isin(fams)].copy()
    d = d[(d.level == d.family.map(fam_lv['min'])) | (d.level == d.family.map(fam_lv['max']))]
    d['y'] = (d.level == d.family.map(fam_lv['max'])).astype(int)
    return d, fams


def model():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def anchor_threshold(m, anchor):
    """Threshold on the decision score that best separates the person's anchor windows (midpoint of class means)."""
    s = m.decision_function(anchor[FEAT_COLS])
    s0, s1 = s[anchor.y.values == 0], s[anchor.y.values == 1]
    if len(s0) == 0 or len(s1) == 0:
        return 0.0
    return 0.5 * (np.median(s0) + np.median(s1))


def run(ds, norm):
    global FEAT_COLS
    df, feats = load(ds)
    df = normalise(df, feats, norm)
    d, fams = binary_task(df)
    FEAT_COLS = feats
    d = d.dropna(subset=feats, how='all')
    rows = []
    units = d.unit.unique()
    for A, B in itertools.product(fams, fams):
        for u in units:
            pid = u.split('|')[0]
            train = d[(d.family == A) & (d.pid.astype(str) != pid)]
            test = d[(d.family == B) & (d.unit == u)]
            if train.y.nunique() < 2 or test.y.nunique() < 2 or len(test) < 6:
                continue
            m = model().fit(train[feats], train.y)
            s = m.decision_function(test[feats])
            r = dict(dataset=ds, norm=norm, source=A, target=B, unit=u, pid=pid, n_test=len(test),
                     auc=roc_auc_score(test.y, s), bacc_pop=balanced_accuracy_score(test.y, s > 0))
            if A != B:  # anchor-calibrated threshold from the same person's source-task windows (first K per level)
                own = d[(d.family == A) & (d.unit == u)].sort_values('t0')
                parts = [g.head(K_ANCHOR) for _, g in own.groupby('y')]
                anchor = pd.concat(parts) if parts else own
                if len(anchor) and anchor.y.nunique() == 2:
                    thr = anchor_threshold(m, anchor)
                    r['bacc_anchor'] = balanced_accuracy_score(test.y, s > thr)
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(f'{OUT}/exp1_{ds}_{norm}.csv', index=False)
    return res


def boot_ci(x, n=2000):
    x = np.asarray(x); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    b = RNG.choice(x, (n, len(x)), replace=True).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def summarise(res):
    out = []
    for (ds, norm, A, B), g in res.groupby(['dataset', 'norm', 'source', 'target']):
        # average over sessions within person first, so each person counts once
        p = g.groupby('pid')[['auc', 'bacc_pop'] + (['bacc_anchor'] if 'bacc_anchor' in g else [])].mean()
        lo, hi = boot_ci(p.auc)
        row = dict(dataset=ds, norm=norm, source=A, target=B, n_people=len(p), auc=p.auc.mean(), auc_lo=lo, auc_hi=hi,
                   bacc_pop=p.bacc_pop.mean())
        if 'bacc_anchor' in p:
            row['bacc_anchor'] = p.bacc_anchor.mean()
        out.append(row)
    return pd.DataFrame(out)


if __name__ == '__main__':
    dss = sys.argv[1:] or ['COGBCI', 'UNIVERSE', 'COGLOAD']
    allres = []
    for ds in dss:
        for norm in ('none', 'rest'):
            r = run(ds, norm); allres.append(r)
            print(ds, norm, len(r), 'person-pairs', flush=True)
    S = summarise(pd.concat(allres))
    S.to_csv(f'{OUT}/exp1_summary.csv', index=False)
    pd.set_option('display.width', 200)
    print(S.round(3).to_string(index=False))
