"""Experiment 2 (diagnostics for interpreting Exp 1).

(a) Personal ceiling: can a model trained on the SAME person's data separate easy vs hard within a task?
    Time-split within each level (first 50% of windows -> train, last 50% -> test; a 30-s gap is
    removed between halves so overlapping windows never cross the split).
(b) Load-vs-rest cross-task transfer (LOSO): train 'task A vs rest' on other people, test 'task B vs rest'
    on the held-out person. Raw features (no rest normalisation, to avoid using test rest windows twice).
"""
import os, sys, itertools, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E

OUT = E.OUT


def personal_ceiling(ds):
    df, feats = E.load(ds)
    df = E.normalise(df, feats, 'rest')
    d, fams = E.binary_task(df)
    rows = []
    for (u, fam), g in d.groupby(['unit', 'family']):
        if g.y.nunique() < 2:
            continue
        tr, te = [], []
        for _, gl in g.groupby('y'):
            gl = gl.sort_values('t0'); n = len(gl)
            cut = gl.t0.iloc[n // 2] if n >= 6 else None
            if cut is None:
                continue
            tr.append(gl[gl.t1 <= cut]); te.append(gl[gl.t0 >= cut + 30])
        if len(tr) < 2:
            continue
        tr, te = pd.concat(tr).dropna(subset=feats), pd.concat(te).dropna(subset=feats)
        if tr.y.nunique() < 2 or te.y.nunique() < 2:
            continue
        m = E.model().fit(tr[feats], tr.y)
        rows.append(dict(dataset=ds, family=fam, unit=u, pid=u.split('|')[0], auc=roc_auc_score(te.y, m.decision_function(te[feats]))))
    return pd.DataFrame(rows)


def load_vs_rest(ds):
    df, feats = E.load(ds)
    task_fams = [f for f in df.family.unique() if not f.startswith('rest') and f != 'meditation']
    rows = []
    for A, B in itertools.product(task_fams, task_fams):
        for u in df.unit.unique():
            pid = u.split('|')[0]
            tr = df[(df.pid.astype(str) != pid) & (df.family.isin([A, 'rest']))]
            te = df[(df.unit == u) & (df.family.isin([B, 'rest']))]
            ytr, yte = (tr.family == A).astype(int), (te.family == B).astype(int)
            if ytr.nunique() < 2 or yte.nunique() < 2 or (yte == 0).sum() < 3:
                continue
            m = E.model().fit(tr[feats], ytr)
            rows.append(dict(dataset=ds, source=A, target=B, unit=u, pid=pid, auc=roc_auc_score(yte, m.decision_function(te[feats]))))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    dss = sys.argv[1:] or ['COGBCI', 'UNIVERSE', 'COGLOAD', 'MPHRC']
    pc, lr = [], []
    for ds in dss:
        if ds != 'MPHRC':
            pc.append(personal_ceiling(ds))
        lr.append(load_vs_rest(ds))
        print('done', ds, flush=True)
    pc = pd.concat(pc) if pc else pd.DataFrame(); lr = pd.concat(lr)
    pc.to_csv(f'{OUT}/exp2_personal_ceiling.csv', index=False); lr.to_csv(f'{OUT}/exp2_load_vs_rest.csv', index=False)
    pd.set_option('display.width', 200)
    if len(pc):
        print('\nPersonal ceiling (within-person AUROC, easy vs hard):')
        print(pc.groupby(['dataset', 'family']).auc.agg(['count', 'mean', 'median']).round(3))
    S = lr.groupby(['dataset', 'source', 'target', 'pid']).auc.mean().groupby(['dataset', 'source', 'target']).agg(['count', 'mean']).round(3)
    print('\nLoad-vs-rest transfer (LOSO, within-person AUROC):')
    print(S.to_string())


def personal_cross_task(ds):
    """Upper bound of task-anchored calibration: a model trained ONLY on the person's own task A (easy vs hard),
    tested on the same person's task B. Tells whether a person's 'difficulty direction' carries across tasks."""
    df, feats = E.load(ds)
    df = E.normalise(df, feats, 'rest')
    d, fams = E.binary_task(df)
    rows = []
    for u, g in d.groupby('unit'):
        for A, B in itertools.product(fams, fams):
            if A == B:
                continue
            tr, te = g[g.family == A].dropna(subset=feats), g[g.family == B].dropna(subset=feats)
            if tr.y.nunique() < 2 or te.y.nunique() < 2:
                continue
            m = E.model().fit(tr[feats], tr.y)
            rows.append(dict(dataset=ds, source=A, target=B, unit=u, pid=u.split('|')[0],
                             auc=roc_auc_score(te.y, m.decision_function(te[feats]))))
    return pd.DataFrame(rows)
