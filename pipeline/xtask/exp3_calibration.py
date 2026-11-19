"""Experiment 3: how much does a short personal calibration help, and with which representation?

Representations
  HC     handcrafted window features (as in Exp 1)
  FM     NormWear embeddings: mean of the 6-s chunk [CLS] vectors inside each 30-s window, channels concatenated,
         PCA (fitted on the training people only) to 32 dims
  HC+FM  both
Every representation is z-scored per person-session with that session's rest windows (no labels).

Calibration conditions for held-out person p, anchor task family A, target family B:
  pop        population model trained on other people's A
  personal   trained only on p's first k windows per level of A
  joint      other people's A + p's anchor windows, the anchor carrying 50% of the total sample weight
Budgets k in {3, 6, 12, 24} windows per level (10-s step: k=6 is about 80 s of each level).
When A == B the test set is p's A windows that start after the last anchor window of that level.
Metrics on p's B: within-person AUROC and balanced accuracy at threshold 0.
Output: results/exp3_<DS>_<REP>.csv
"""
import os, sys, itertools, warnings
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E

SEG = os.path.join(os.path.dirname(E.R), 'segments')
EMB = os.environ.get('EMB_DIR', os.path.join(os.path.dirname(E.R), 'embeddings'))
BUDGETS = [3, 6, 12, 24]
CHUNK_S = 388 / 64


def fm_window_features(df, ds):
    """Attach window-level NormWear features (fm_0..fm_{768*C-1}) to the windows dataframe."""
    idx = pd.read_parquet(f'{SEG}/{ds}_index.parquet')
    e = np.load(f'{EMB}/emb_{ds}.npz')
    cls = e['cls'].astype(np.float32); N, C, D = cls.shape
    seg_of = pd.Series(e['seg_id']); t0s = e['chunk_t0']
    by_seg = {s: np.where(seg_of.values == s)[0] for s in np.unique(seg_of.values)}
    key = idx.set_index(['pid', 'session', 'task'])['seg_id']
    out = np.full((len(df), C * D), np.nan, np.float32)
    for i, (pid, ses, task, w0, w1) in enumerate(df[['pid', 'session', 'task', 't0', 't1']].itertuples(index=False)):
        try:
            s = key.loc[(str(pid), str(ses), task)]
        except KeyError:
            continue
        s = s.iloc[0] if isinstance(s, pd.Series) else s
        rows = by_seg.get(s)
        if rows is None:
            continue
        sel = rows[(t0s[rows] >= w0) & (t0s[rows] + CHUNK_S <= w1 + 1e-6)]
        if len(sel):
            out[i] = cls[sel].mean(0).reshape(-1)
    cols = [f'fm_{j}' for j in range(C * D)]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out, columns=cols)], axis=1), cols


def rest_norm(df, cols):
    out = df.copy()
    for u, g in df.groupby('unit'):
        ref = g.loc[g.family == 'rest', cols]
        if len(ref) < 3:
            out.loc[g.index, cols] = np.nan; continue
        mu, sd = ref.mean(), ref.std()
        sd = sd.where(sd > 1e-6, 1.0)
        out.loc[g.index, cols] = ((g[cols] - mu) / sd).values
    return out


def fit_predict(Xtr, ytr, wtr, Xte):
    m = E.model()
    m.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
    return m.decision_function(Xte)


def run(ds, rep):
    df, hc = E.load(ds)
    df = df.reset_index(drop=True)
    cols = []
    if 'HC' in rep:
        cols += hc
    if 'FM' in rep:
        df, fm = fm_window_features(df, ds)
        cols += fm
    df = rest_norm(df, cols)
    d, fams = E.binary_task(df)
    d = d.dropna(subset=cols, how='all').reset_index(drop=True)
    fm_cols = [c for c in cols if c.startswith('fm_')]
    rows = []
    for pid in d.pid.astype(str).unique():
        others = d[d.pid.astype(str) != pid]
        X_all = d[cols].copy()
        if fm_cols:  # PCA on other people's windows only (label-free), 32 dims
            pca = PCA(32, random_state=0).fit(others[fm_cols].fillna(0).values)
            Z = pca.transform(d[fm_cols].fillna(0).values)
            X_all = pd.concat([d[[c for c in cols if c not in fm_cols]], pd.DataFrame(Z, columns=[f'pc{i}' for i in range(32)])], axis=1)
        feat = list(X_all.columns)
        for u in d.loc[d.pid.astype(str) == pid, 'unit'].unique():
            for A, B in itertools.product(fams, fams):
                own_A = d[(d.unit == u) & (d.family == A)].sort_values('t0')
                pop_A = others[others.family == A]
                if own_A.y.nunique() < 2 or pop_A.y.nunique() < 2:
                    continue
                for k in BUDGETS:
                    anchor = pd.concat([g.head(k) for _, g in own_A.groupby('y')])
                    if A == B:
                        last = anchor.groupby('y').t1.max()
                        test = own_A[own_A.t0 >= own_A.y.map(last)]
                    else:
                        test = d[(d.unit == u) & (d.family == B)]
                    if test.y.nunique() < 2 or len(test) < 6:
                        continue
                    Xte = X_all.loc[test.index, feat]
                    res = {}
                    s = fit_predict(X_all.loc[pop_A.index, feat], pop_A.y, np.ones(len(pop_A)), Xte); res['pop'] = s
                    s = fit_predict(X_all.loc[anchor.index, feat], anchor.y, np.ones(len(anchor)), Xte); res['personal'] = s
                    w = np.r_[np.ones(len(pop_A)), np.full(len(anchor), len(pop_A) / len(anchor))]
                    Xj = pd.concat([X_all.loc[pop_A.index, feat], X_all.loc[anchor.index, feat]])
                    s = fit_predict(Xj, np.r_[pop_A.y.values, anchor.y.values], w, Xte); res['joint'] = s
                    for cond, s in res.items():
                        rows.append(dict(dataset=ds, rep=rep, unit=u, pid=pid, source=A, target=B, k=k, cond=cond,
                                         auc=roc_auc_score(test.y, s), bacc=balanced_accuracy_score(test.y, s > 0), n_test=len(test)))
    r = pd.DataFrame(rows)
    r.to_csv(f'{E.OUT}/exp3_{ds}_{rep.replace("+", "_")}.csv', index=False)
    return r


if __name__ == '__main__':
    reps = os.environ.get('REPS', 'HC').split(',')
    dss = sys.argv[1:] or ['COGBCI', 'UNIVERSE']
    allr = []
    for ds in dss:
        for rep in reps:
            r = run(ds, rep); allr.append(r); print(ds, rep, len(r), flush=True)
    R = pd.concat(allr)
    R['xtask'] = np.where(R.source == R.target, 'same', 'cross')
    S = (R.groupby(['dataset', 'rep', 'xtask', 'k', 'cond', 'pid'])[['auc', 'bacc']].mean()
           .groupby(['dataset', 'rep', 'xtask', 'k', 'cond']).mean().round(3).unstack('cond'))
    pd.set_option('display.width', 220)
    print(S.to_string())


def run_cross_session(ds, rep='HC'):
    """Control for block-identity leakage: anchor on one session's task A, test the SAME person's task A in another session."""
    df, hc = E.load(ds)
    df = df.reset_index(drop=True)
    cols = list(hc) if 'HC' in rep else []
    if 'FM' in rep:
        df, fm = fm_window_features(df, ds); cols += fm
    df = rest_norm(df, cols)
    d, fams = E.binary_task(df)
    d = d.dropna(subset=cols, how='all').reset_index(drop=True)
    fm_cols = [c for c in cols if c.startswith('fm_')]
    rows = []
    for pid in d.pid.astype(str).unique():
        mine_mask = d.pid.astype(str) == pid
        X_all = d[cols]
        if fm_cols:  # PCA fitted on other people's windows only
            pca = PCA(32, random_state=0).fit(d.loc[~mine_mask, fm_cols].fillna(0).values)
            Z = pca.transform(d[fm_cols].fillna(0).values)
            X_all = pd.concat([d[[c for c in cols if c not in fm_cols]], pd.DataFrame(Z, columns=[f'pc{i}' for i in range(32)])], axis=1)
        feat = list(X_all.columns)
        mine, others = d[mine_mask], d[~mine_mask]
        units = mine.unit.unique()
        if len(units) < 2:
            continue
        for ua, ut in itertools.permutations(units, 2):
            for A in fams:
                own_A = mine[(mine.unit == ua) & (mine.family == A)].sort_values('t0')
                test = mine[(mine.unit == ut) & (mine.family == A)]
                pop_A = others[others.family == A]
                if own_A.y.nunique() < 2 or test.y.nunique() < 2:
                    continue
                Xte = X_all.loc[test.index, feat]
                for k in BUDGETS:
                    anchor = pd.concat([g.head(k) for _, g in own_A.groupby('y')])
                    res = {'pop': fit_predict(X_all.loc[pop_A.index, feat], pop_A.y, np.ones(len(pop_A)), Xte),
                           'personal': fit_predict(X_all.loc[anchor.index, feat], anchor.y, np.ones(len(anchor)), Xte)}
                    w = np.r_[np.ones(len(pop_A)), np.full(len(anchor), len(pop_A) / len(anchor))]
                    Xj = pd.concat([X_all.loc[pop_A.index, feat], X_all.loc[anchor.index, feat]])
                    res['joint'] = fit_predict(Xj, np.r_[pop_A.y.values, anchor.y.values], w, Xte)
                    for cond, sc in res.items():
                        rows.append(dict(dataset=ds, rep=rep, pid=pid, anchor_unit=ua, test_unit=ut, family=A, k=k, cond=cond,
                                         auc=roc_auc_score(test.y, sc), bacc=balanced_accuracy_score(test.y, sc > 0)))
    r = pd.DataFrame(rows)
    r.to_csv(f'{E.OUT}/exp3_crosssession_{ds}_{rep.replace("+", "_")}.csv', index=False)
    return r
