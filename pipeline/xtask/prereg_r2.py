"""Pre-registered analyses on the plan-B datasets (PREREG_R2.md, sha256 1a37c84f...; amendment 9afe86e3...).

  python prereg_r2.py WESAD DS007554 [SENSECOBOT] [COGWEAR] [MAUS]
Output: results/prereg_r2/<DATASET>_*.csv and a printed verdict per prediction.
"""
import os, sys, itertools, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E
import mech as M
import confirm as C

W = os.environ.get('WIN_DIR', '/path/to/project/data/windows')
OUT = os.path.join(E.OUT, 'prereg_r2'); os.makedirs(OUT, exist_ok=True)
CARD = ['hr_mean', 'hr_std', 'rmssd', 'sdnn', 'pnn50']
GROUPS = {'cardiac': CARD, 'eda': ['scl_mean', 'scl_slope', 'phasic_std', 'scr_rate'], 'temp': ['temp_mean', 'temp_slope'],
          'resp': ['resp_rate', 'resp_amp'], 'motion': ['acc_std']}
K = 6


def load(name, norm='rest'):
    df = pd.read_parquet(f'{W}/{name}_windows.parquet')
    if 'ibi_valid' in df:
        df.loc[df.ibi_valid < 0.8, [c for c in CARD if c in df]] = np.nan
    df['unit'] = df.pid.astype(str) + '|' + df.session.astype(str)
    df['block'] = df.unit + '|' + df.task.astype(str)
    feats = [f for g in GROUPS.values() for f in g if f in df.columns and df[f].notna().mean() > 0.3]
    if norm == 'rest':
        df = E.normalise(df, feats, 'rest')
    elif norm == 'unit':
        g = df.groupby('unit')[feats]
        df[feats] = (df[feats] - g.transform('mean')) / g.transform('std').replace(0, np.nan)
    df['pid'] = df.pid.astype(str)
    return df, feats


def fit_score(Xtr, ytr, Xte):
    m = E.model(); m.fit(Xtr, ytr)
    return m.decision_function(Xte)


def safe(fn, *a):
    try:
        return fn(*a)
    except ValueError:
        return np.nan


def split_half_null(df, feats, block_mask, k=K):
    """Personal model separating the first from the second half of ONE constant-condition block (exp4 protocol)."""
    rows = []
    for b, g in df[block_mask].groupby('block'):
        g = g.sort_values('t0'); mid = g.t0.iloc[len(g) // 2]
        g = g.assign(y=(g.t0 >= mid).astype(int))
        anchor = pd.concat([h.head(k) for _, h in g.groupby('y')])
        last = anchor.groupby('y').t1.max()
        test = g[(g.t0 >= g.y.map(last)) & ~g.index.isin(anchor.index)]
        if test.y.nunique() < 2 or test.y.value_counts().min() < 3:
            continue
        a = safe(lambda: roc_auc_score(test.y, fit_score(anchor[feats], anchor.y, test[feats])))
        rows.append(dict(block=b, pid=g.pid.iloc[0], auc=a))
    return pd.DataFrame(rows).dropna()


def loso_within_auc(d, feats):
    """LOSO population model, within-person AUROC (d has y)."""
    out = {}
    for pid in d.pid.unique():
        tr, te = d[d.pid != pid], d[d.pid == pid]
        if te.y.nunique() < 2 or tr.y.nunique() < 2:
            continue
        out[pid] = safe(lambda: roc_auc_score(te.y, fit_score(tr[feats], tr.y, te[feats])))
    return pd.Series(out).dropna()


def verdict_gt(ci, thr):
    return 'confirmed' if ci[0] > thr else ('refuted' if ci[1] < thr else 'inconclusive')


def verdict_lt(ci, thr):
    return 'confirmed' if ci[1] < thr else ('refuted' if ci[0] > thr else 'inconclusive')


# ------------------------------------------------------------------ WESAD (P2)
def wesad():
    rows = []
    for dev in ('CHEST', 'WRIST'):
        df, feats = load(f'WESAD_{dev}', 'none')
        # P2a split-half null inside the 20-min baseline, per feature group
        for gname, fs in GROUPS.items():
            fs = [f for f in fs if f in feats]
            if not fs:
                continue
            n = split_half_null(df, fs, df.family == 'rest')
            if len(n) < 5:
                continue
            p = n.groupby('pid').auc.mean(); ci = M.boot_ci(p.values)
            pred = {('WRIST', 'temp'): ('>', .6), ('WRIST', 'eda'): ('>', .6), ('CHEST', 'cardiac'): ('<', .6), ('CHEST', 'temp'): ('>', .6)}.get((dev, gname))
            v = (verdict_gt(ci, pred[1]) if pred[0] == '>' else verdict_lt(ci, pred[1])) if pred else 'report only'
            rows.append(dict(pred='P2a', device=dev, features=gname, estimate=p.mean(), ci=ci, n=len(p), prediction=f'{pred[0]} {pred[1]}' if pred else '', verdict=v))
    # P2b clock baseline: stress vs baseline, time is the only feature
    df, feats = load('WESAD_CHEST', 'none')
    d = df[df.family.isin(['rest', 'stress'])].copy(); d['y'] = (d.family == 'stress').astype(int)
    d['clock'] = d.start + d.t0 / 60.0
    raw = d.groupby('pid').apply(lambda g: roc_auc_score(g.y, g.clock))
    pop_clock = loso_within_auc(d, ['clock'])
    stress_after_base = d.groupby('pid').apply(lambda g: g[g.y == 1].start.min() > g[g.y == 0].start.min()).mean()
    rows.append(dict(pred='P2b', device='-', features='clock only (raw time, within-person AUROC)', estimate=raw.mean(), ci=M.boot_ci(raw.values), n=len(raw),
                     prediction='> 0.9', verdict=verdict_gt(M.boot_ci(raw.values), .9), note=f'stress after baseline in {stress_after_base:.0%} of people'))
    rows.append(dict(pred='P2b', device='-', features='clock only (LOSO model)', estimate=pop_clock.mean(), ci=M.boot_ci(pop_clock.values), n=len(pop_clock),
                     prediction='report', verdict='report only'))
    # P2c engagement-type contrast vs affect contrast (LOSO, all features), both devices
    for dev in ('CHEST', 'WRIST'):
        df, feats = load(f'WESAD_{dev}', 'rest')
        a = df[df.family.isin(['rest', 'stress'])].assign(y=lambda x: (x.family == 'stress').astype(int))
        b = df[df.family.isin(['amusement', 'stress'])].assign(y=lambda x: (x.family == 'stress').astype(int))
        sa, sb = loso_within_auc(a, feats), loso_within_auc(b, feats)
        j = pd.concat([sa.rename('a'), sb.rename('b')], axis=1).dropna()
        p = wilcoxon(j.a - j.b, alternative='greater').pvalue
        rows.append(dict(pred='P2c', device=dev, features='all', estimate=(j.a - j.b).mean(), ci=M.boot_ci((j.a - j.b).values), n=len(j),
                         prediction='stress-vs-baseline > stress-vs-amusement', verdict='confirmed' if p < .05 else 'not confirmed', p=p,
                         note=f'stress vs baseline {j.a.mean():.3f}; stress vs amusement {j.b.mean():.3f}'))
        # exploratory: same-session personal model for stress vs baseline (anchor/test as Exp 3) and clock personal model
        for rep, cols in (('physiology', feats), ('clock', ['clock'])):
            aa = a.assign(clock=a.start + a.t0 / 60.0); res = []
            for pid, g in aa.groupby('pid'):
                g = g.sort_values('t0'); anchor = pd.concat([h.sort_values('t0').head(K) for _, h in g.groupby('y')])
                last = anchor.groupby('y').t1.max(); test = g[g.t0 >= g.y.map(last)]
                if test.y.nunique() == 2:
                    res.append(safe(lambda: roc_auc_score(test.y, fit_score(anchor[cols], anchor.y, test[cols]))))
            res = np.array([r for r in res if np.isfinite(r)])
            rows.append(dict(pred='explor.', device=dev, features=f'same-session personal, {rep}', estimate=res.mean(), ci=M.boot_ci(res), n=len(res),
                             prediction='', verdict='exploratory'))
    # exploratory device comparison: block-level heart-rate effect, stress vs baseline, chest ECG vs wrist BVP
    eff = {}
    for dev in ('CHEST', 'WRIST'):
        df, feats = load(f'WESAD_{dev}', 'rest')
        b = df[df.family.isin(['rest', 'stress'])].groupby(['pid', 'family']).hr_mean.mean().unstack()
        eff[dev] = (b.stress - b.rest)
        u = df[df.family.isin(['rest', 'stress'])].groupby('family').ibi_valid.apply(lambda s: (s >= .8).mean())
        rows.append(dict(pred='explor.', device=dev, features='usable cardiac windows (rest / stress)', estimate=u.get('stress', np.nan),
                         ci=[round(float(u.get('rest', np.nan)), 3), round(float(u.get('stress', np.nan)), 3)], n=df.pid.nunique(), verdict='exploratory'))
    j = pd.concat(eff, axis=1).dropna()
    rows.append(dict(pred='explor.', device='WRIST/CHEST', features='HR effect ratio (stress - baseline, rest-SD)', estimate=j.WRIST.mean() / j.CHEST.mean(),
                     ci=M.boot_ci((j.WRIST / j.CHEST.mean()).values), n=len(j), verdict='exploratory',
                     note=f'chest {j.CHEST.mean():.2f} SD, wrist {j.WRIST.mean():.2f} SD'))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ ds007554 (P3)
def ds007554():
    rows, prim = [], []
    df, feats = load('DS007554', 'none')
    feats = [f for f in CARD if f in feats]
    b = df.groupby(['pid', 'session', 'task']).agg(hr=('hr_mean', 'mean'), subj=('subjective', 'first')).reset_index().dropna(subset=['hr'])
    r = C.fit("hr ~ C(task, Treatment('activemotor')) + C(session)", b)
    est, se, lo, hi = C.coef(r, "C(task, Treatment('activemotor'))[T.full]")
    prim.append(dict(pred='P3a', test='HR full - active motor (bpm)', estimate=est, ci=[round(lo, 2), round(hi, 2)], p=C.p_greater(est, se), n=b.pid.nunique()))
    r2 = C.fit("hr ~ C(task, Treatment('passivemotor')) + C(session)", b)
    est, se, lo, hi = C.coef(r2, "C(task, Treatment('passivemotor'))[T.activemotor]")
    prim.append(dict(pred='P3b', test='HR active - passive motor (bpm)', estimate=est, ci=[round(lo, 2), round(hi, 2)], p=C.p_greater(est, se), n=b.pid.nunique()))
    s = b.groupby(['pid', 'task']).subj.mean().unstack()
    dd = (s['full'] - s['activemotor']).dropna()
    prim.append(dict(pred='P3c', test='effort rating full - active motor (1-9)', estimate=dd.mean(), ci=M.boot_ci(dd.values),
                     p=wilcoxon(dd, alternative='greater').pvalue, n=len(dd)))
    # P3d block identity: full vs active motor; anchor = first K windows of each ses-01 block
    res = []
    for pid, g in df[df.task.isin(['full', 'activemotor'])].assign(y=lambda x: (x.task == 'full').astype(int)).groupby('pid'):
        s1 = g[g.session == 'ses-01'].sort_values('t0')
        if s1.y.nunique() < 2:
            continue
        anchor = pd.concat([h.head(K) for _, h in s1.groupby('y')])
        last = anchor.groupby('y').t1.max()
        within = s1[(s1.t0 >= s1.y.map(last)) & ~s1.index.isin(anchor.index)]
        across = g[g.session != 'ses-01']
        if within.y.nunique() < 2 or across.y.nunique() < 2:
            continue
        aw = safe(lambda: roc_auc_score(within.y, fit_score(anchor[feats], anchor.y, within[feats])))
        ac = safe(lambda: roc_auc_score(across.y, fit_score(anchor[feats], anchor.y, across[feats])))
        res.append(dict(pid=pid, within=aw, across=ac))
    res = pd.DataFrame(res).dropna()
    prim.append(dict(pred='P3d', test=f'personal AUROC within-block {res.within.mean():.3f} vs across repetitions {res.across.mean():.3f}',
                     estimate=(res.within - res.across).mean(), ci=M.boot_ci((res.within - res.across).values),
                     p=wilcoxon(res.within - res.across, alternative='greater').pvalue, n=len(res)))
    P = pd.DataFrame(prim); P['p_holm'] = C.holm(P.p.values); P['verdict'] = np.where(P.p_holm < .05, 'confirmed', 'not confirmed')
    # exploratory: all task pairs for P3d, and LOSO task-vs-task for engagement-type contrasts
    allp = []
    tasks = sorted(df.task.unique())
    for A, B in itertools.combinations(tasks, 2):
        for pid, g in df[df.task.isin([A, B])].assign(y=lambda x: (x.task == B).astype(int)).groupby('pid'):
            s1 = g[g.session == 'ses-01'].sort_values('t0')
            if s1.y.nunique() < 2:
                continue
            anchor = pd.concat([h.head(K) for _, h in s1.groupby('y')]); last = anchor.groupby('y').t1.max()
            within = s1[(s1.t0 >= s1.y.map(last)) & ~s1.index.isin(anchor.index)]; across = g[g.session != 'ses-01']
            if within.y.nunique() == 2 and across.y.nunique() == 2:
                allp.append(dict(pair=f'{A}|{B}', pid=pid, within=safe(lambda: roc_auc_score(within.y, fit_score(anchor[feats], anchor.y, within[feats]))),
                                 across=safe(lambda: roc_auc_score(across.y, fit_score(anchor[feats], anchor.y, across[feats])))))
    allp = pd.DataFrame(allp).dropna(); allp.to_csv(f'{OUT}/DS007554_P3d_allpairs.csv', index=False)
    pp = allp.groupby('pid')[['within', 'across']].mean()
    rows.append(dict(pred='explor.', test=f'P3d over all 21 task pairs: within {pp.within.mean():.3f} vs across {pp.across.mean():.3f}',
                     estimate=(pp.within - pp.across).mean(), ci=M.boot_ci((pp.within - pp.across).values), p=wilcoxon(pp.within - pp.across).pvalue, n=len(pp)))
    hrt = b.groupby('task').hr.mean().round(1).to_dict(); rt = b.groupby('task').subj.mean().round(2).to_dict()
    rows.append(dict(pred='explor.', test='mean HR (bpm) by task', estimate=np.nan, ci=str(hrt), n=b.pid.nunique()))
    rows.append(dict(pred='explor.', test='mean effort rating by task', estimate=np.nan, ci=str(rt), n=b.pid.nunique()))
    return pd.concat([P, pd.DataFrame(rows)], ignore_index=True)


# ------------------------------------------------------------------ SenseCobot (P5)
def sensecobot():
    rows = []
    # P5a engagement vs between-task, both sensor sets
    for dev in ('LAB', 'WRIST'):
        df, feats = load(f'SENSECOBOT_{dev}', 'rest')
        a = df[df.family.isin(['rest', 'task'])].assign(y=lambda x: (x.family == 'task').astype(int))
        b = df[df.task.isin(['task1', 'task5'])].assign(y=lambda x: (x.task == 'task5').astype(int))
        sa, sb = loso_within_auc(a, feats), loso_within_auc(b, feats)
        j = pd.concat([sa.rename('a'), sb.rename('b')], axis=1).dropna()
        p = wilcoxon(j.a - j.b, alternative='greater').pvalue
        rows.append(dict(pred='P5a', device=dev, features=','.join(feats), estimate=(j.a - j.b).mean(), ci=M.boot_ci((j.a - j.b).values), n=len(j),
                         prediction='task-vs-rest > task1-vs-task5', verdict='confirmed' if p < .05 else 'not confirmed', p=p,
                         note=f'task vs rest {j.a.mean():.3f}; task1 vs task5 {j.b.mean():.3f}'))
    # P5b device: heart-rate effect (task - rest, rest-SD) chest ECG vs E4 IBI; usable windows
    eff, use = {}, {}
    for dev in ('LAB', 'WRIST'):
        df, feats = load(f'SENSECOBOT_{dev}', 'rest')
        if 'hr_mean' not in df:
            continue
        bm = df.groupby(['pid', 'family']).hr_mean.mean().unstack()
        eff[dev] = bm.task - bm.rest
        raw = pd.read_parquet(f'{W}/SENSECOBOT_{dev}_windows.parquet')
        use[dev] = raw[raw.family == 'task'].ibi_valid.ge(.8).mean() if 'ibi_valid' in raw else np.nan
        if dev == 'WRIST':
            mv = raw[raw.family == 'task'].groupby('movement').ibi_valid.apply(lambda s: s.ge(.8).mean()).round(3).to_dict()
    j = pd.concat(eff, axis=1).dropna()
    ratio = j.WRIST.mean() / j.LAB.mean() if len(j) else np.nan
    ci = M.boot_ci((j.WRIST / j.LAB.mean()).values) if len(j) > 3 else [np.nan, np.nan]
    rows.append(dict(pred='P5b', device='WRIST/LAB', features='HR effect ratio (task - rest)', estimate=ratio, ci=ci, n=len(j),
                     prediction='CI lower > 0.5', verdict=verdict_gt(ci, .5) if np.isfinite(ci[0]) else 'not testable',
                     note=f'chest {j.LAB.mean():.2f} SD, wrist {j.WRIST.mean():.2f} SD; usable task windows chest {use.get("LAB", np.nan):.2f}, wrist {use.get("WRIST", np.nan):.2f}; wrist usable by movement (0 jog,1 freedrive,2 mix) {mv}'))
    # P5c split-half null inside the long programming tasks (task4, task5)
    for dev, groups in (('WRIST', ('temp', 'eda', 'cardiac', 'motion')), ('LAB', ('cardiac', 'eda'))):
        df, feats = load(f'SENSECOBOT_{dev}', 'none')
        for gname in groups:
            fs = [f for f in GROUPS[gname] if f in feats]
            if not fs:
                continue
            n = split_half_null(df, fs, df.task.isin(['task4', 'task5']))
            if len(n) < 5:
                continue
            pmean = n.groupby('pid').auc.mean(); ci = M.boot_ci(pmean.values)
            pred = {('WRIST', 'temp'): ('>', .6), ('WRIST', 'eda'): ('>', .6), ('LAB', 'cardiac'): ('<', .6)}.get((dev, gname))
            v = (verdict_gt(ci, pred[1]) if pred[0] == '>' else verdict_lt(ci, pred[1])) if pred else 'report only'
            label = 'finger GSR' if (dev, gname) == ('LAB', 'eda') else gname
            rows.append(dict(pred='P5c', device=dev, features=label, estimate=pmean.mean(), ci=ci, n=len(pmean),
                             prediction=f'{pred[0]} {pred[1]}' if pred else '', verdict=v))
    # P5d order and ratings
    raw = pd.read_parquet(f'{W}/SENSECOBOT_LAB_windows.parquet')
    r = raw[raw.family == 'task'].groupby(['pid', 'level'])[['subjective', 'physical']].first().groupby('level').mean().round(2)
    rows.append(dict(pred='P5d', device='-', features='effort (B) by task 1-5', estimate=np.nan, ci=str(r.subjective.to_dict()), n=raw.pid.nunique(),
                     verdict='descriptive', note='task order from file timestamps: baseline first for all; tasks mostly 1->5 (see DECISION_LOG CP10)'))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ CogWear (P4)
def cogwear():
    rows = []
    for dev in ('E4', 'SAMSUNG'):
        df, feats = load(f'COGWEAR_{dev}', 'unit')
        d = df[df.family.isin(['rest', 'stroop'])].assign(y=lambda x: (x.family == 'stroop').astype(int))
        res = []
        for pid, g in d.groupby('pid'):
            pre, post = g[g.session == 'pre'].sort_values('t0'), g[g.session == 'post']
            if pre.y.nunique() < 2 or post.y.nunique() < 2:
                continue
            anchor = pd.concat([h.head(K) for _, h in pre.groupby('y')]); last = anchor.groupby('y').t1.max()
            within = pre[(pre.t0 >= pre.y.map(last)) & ~pre.index.isin(anchor.index)]
            others = d[d.pid != pid]
            r = dict(pid=pid,
                     same=safe(lambda: roc_auc_score(within.y, fit_score(anchor[feats], anchor.y, within[feats]))) if within.y.nunique() == 2 else np.nan,
                     cross=safe(lambda: roc_auc_score(post.y, fit_score(anchor[feats], anchor.y, post[feats]))),
                     pop=safe(lambda: roc_auc_score(post.y, fit_score(others[feats], others.y, post[feats]))))
            res.append(r)
        res = pd.DataFrame(res)
        a = res.dropna(subset=['same', 'cross'])
        p = wilcoxon(a.same - a.cross, alternative='greater').pvalue if len(a) > 5 else np.nan
        rows.append(dict(pred='P4a', device=dev, features=','.join(feats), estimate=(a.same - a.cross).mean(), ci=M.boot_ci((a.same - a.cross).values), n=len(a),
                         prediction='same-session personal > cross-session personal', verdict='confirmed' if p < .05 else 'not confirmed', p=p,
                         note=f'same {a.same.mean():.3f}; cross {a.cross.mean():.3f}'))
        b = res.dropna(subset=['cross', 'pop'])
        rows.append(dict(pred='P4b', device=dev, features='', estimate=(b.cross - b['pop']).mean(), ci=M.boot_ci((b.cross - b['pop']).values), n=len(b),
                         prediction='report (personal - population, cross-session)', verdict='report only', note=f'pop {b["pop"].mean():.3f}'))
        raw = pd.read_parquet(f'{W}/COGWEAR_{dev}_windows.parquet')
        u = raw.groupby('family').ibi_valid.apply(lambda s: s.ge(.8).mean()).round(3).to_dict()
        rows.append(dict(pred='explor.', device=dev, features='usable cardiac windows by condition', estimate=np.nan, ci=str(u), n=raw.pid.nunique(), verdict='exploratory'))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ MAUS (P1): device vs manipulation
def maus():
    rows = []
    E_, Wr, Fi = load('MAUS_ECG', 'rest')[0], load('MAUS_WRIST', 'rest')[0], load('MAUS_FINGER', 'rest')[0]
    raw = {n: pd.read_parquet(f'{W}/MAUS_{n}_windows.parquet') for n in ('ECG', 'WRIST', 'FINGER')}
    # P1a torso-ECG heart rate, 3-back vs 0-back blocks, block position as covariate (rest-SD units; bpm reported too)
    def blocks(df):
        d = df[df.level.isin([0, 3])]
        return d.groupby(['pid', 'task', 'level', 'position']).hr_mean.mean().reset_index().assign(y=lambda x: (x.level == 3).astype(int)).dropna()
    bz = blocks(E_)
    r = C.fit('hr_mean ~ y + position', bz); est, se, lo, hi = C.coef(r, 'y')
    rb = C.fit('hr_mean ~ y + position', blocks(raw['ECG'][raw['ECG'].ibi_valid >= .8].assign(pid=lambda x: x.pid.astype(str))))
    eb, _, lob, hib = C.coef(rb, 'y')
    pa = C.p_greater(est, se)
    rows.append(dict(pred='P1a', device='ECG', features='hr_mean', estimate=est, ci=[round(lo, 3), round(hi, 3)], n=bz.pid.nunique(), p=pa,
                     prediction='3-back > 0-back (one-sided)', verdict='confirmed' if pa < .05 else 'not confirmed', note=f'{eb:.2f} bpm [{lob:.2f}, {hib:.2f}]'))
    # P1b wrist / ECG heart-rate effect ratio (bpm, same heart); fingertip PPG reported alongside
    def eff(df):
        d = df[df.ibi_valid >= .8]
        m = d[d.level.isin([0, 3])].groupby(['pid', 'level']).hr_mean.mean().unstack()
        return (m[3] - m[0]).rename(None)
    e_ecg, e_w, e_f = eff(raw['ECG']), eff(raw['WRIST']), eff(raw['FINGER'])
    rng = np.random.default_rng(0)
    for name, e2 in (('WRIST', e_w), ('FINGER', e_f)):
        j = pd.concat([e_ecg.rename('ecg'), e2.rename('dev')], axis=1).dropna()
        bs = []
        for _ in range(2000):
            s_ = j.sample(len(j), replace=True, random_state=int(rng.integers(1e9)))
            bs.append(s_.dev.mean() / s_.ecg.mean())
        ci = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
        v = 'device not the main limit' if ci[0] > .5 else ('device is the main limit' if ci[1] < .5 else 'inconclusive')
        rows.append(dict(pred='P1b' if name == 'WRIST' else 'P1b (finger, report)', device=f'{name}/ECG', features='HR effect ratio (3-back - 0-back, bpm)',
                         estimate=j.dev.mean() / j.ecg.mean(), ci=ci, n=len(j), prediction='CI lower > 0.5', verdict=v,
                         note=f'ECG {j.ecg.mean():.2f} bpm, {name.lower()} {j.dev.mean():.2f} bpm; r(person effects) = {np.corrcoef(j.ecg, j.dev)[0, 1]:.2f}'))
    # P1c usable windows: wrist < ECG, and wrist lower in hard than easy
    u = {n: raw[n][raw[n].level >= 0].groupby('pid').ibi_valid.apply(lambda s: s.ge(.8).mean()) for n in ('ECG', 'WRIST')}
    d1 = (u['WRIST'] - u['ECG']).dropna(); p1 = wilcoxon(d1, alternative='less').pvalue
    rows.append(dict(pred='P1c-1', device='WRIST vs ECG', features='usable cardiac windows', estimate=d1.mean(), ci=M.boot_ci(d1.values), n=len(d1), p=p1,
                     prediction='wrist < ECG', verdict='confirmed' if p1 < .05 else 'not confirmed', note=f'ECG {u["ECG"].mean():.3f}, wrist {u["WRIST"].mean():.3f}'))
    wl = raw['WRIST'][raw['WRIST'].level.isin([0, 3])].groupby(['pid', 'level']).ibi_valid.apply(lambda s: s.ge(.8).mean()).unstack()
    d2 = (wl[3] - wl[0]).dropna(); p2 = wilcoxon(d2, alternative='less').pvalue if (d2 != 0).any() else 1.0
    rows.append(dict(pred='P1c-2', device='WRIST', features='usable windows 3-back - 0-back', estimate=d2.mean(), ci=M.boot_ci(d2.values), n=len(d2), p=p2,
                     prediction='hard < easy', verdict='confirmed' if p2 < .05 else 'not confirmed'))
    # P1d same level, different block: personal model separating the two blocks of each level
    for name, df_, groups in (('ECG', load('MAUS_ECG', 'none')[0], ('cardiac', 'eda')), ('WRIST', load('MAUS_WRIST', 'none')[0], ('cardiac',))):
        for gname in groups:
            fs = [f for f in GROUPS[gname] if f in df_.columns]
            res = []
            for (pid, lvl), g in df_[df_.level >= 0].groupby(['pid', 'level']):
                bl = sorted(g.task.unique())
                if len(bl) != 2:
                    continue
                g = g.assign(y=(g.task == bl[1]).astype(int)).sort_values('t0')
                anchor = pd.concat([h.head(K) for _, h in g.groupby('y')]); last = anchor.groupby('y').t1.max()
                test = g[(g.t0 >= g.y.map(last)) & ~g.index.isin(anchor.index)]
                if test.y.nunique() == 2:
                    res.append(dict(pid=pid, auc=safe(lambda: roc_auc_score(test.y, fit_score(anchor[fs], anchor.y, test[fs])))))
            res = pd.DataFrame(res).dropna(); pm = res.groupby('pid').auc.mean(); ci = M.boot_ci(pm.values)
            pred = {('ECG', 'eda'): ('>', .6), ('ECG', 'cardiac'): ('<', .6)}.get((name, gname))
            v = (verdict_gt(ci, pred[1]) if pred[0] == '>' else verdict_lt(ci, pred[1])) if pred else 'report only'
            rows.append(dict(pred='P1d', device=name, features='finger GSR' if gname == 'eda' else gname, estimate=pm.mean(), ci=ci, n=len(pm),
                             prediction=f'{pred[0]} {pred[1]}' if pred else '', verdict=v, note='same level, different block (0-back 1 vs 6, 2-back 2 vs 4, 3-back 3 vs 5)'))
    # exploratory: NASA-TLX manipulation check and LOSO 3-back vs 0-back per device
    s = raw['ECG'][raw['ECG'].level >= 0].groupby(['pid', 'level']).subjective.first().unstack()
    dd = (s[3] - s[0]).dropna()
    rows.append(dict(pred='explor.', device='-', features='NASA-TLX 3-back - 0-back', estimate=dd.mean(), ci=M.boot_ci(dd.values), n=len(dd),
                     verdict='exploratory', note=f'felt harder in {(dd > 0).mean():.0%} of people'))
    for name, df_ in (('ECG', E_), ('FINGER', Fi), ('WRIST', Wr)):
        feats = [f for f in CARD if f in df_.columns]
        d = df_[df_.level.isin([0, 3])].assign(y=lambda x: (x.level == 3).astype(int))
        a = loso_within_auc(d, feats)
        rows.append(dict(pred='explor.', device=name, features='LOSO 3-back vs 0-back, cardiac', estimate=a.mean(), ci=M.boot_ci(a.values), n=len(a), verdict='exploratory'))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
    for name in sys.argv[1:]:
        t = {'WESAD': wesad, 'DS007554': ds007554, 'SENSECOBOT': sensecobot, 'COGWEAR': cogwear, 'MAUS': maus}[name]()
        t.to_csv(f'{OUT}/{name}_prereg.csv', index=False)
        print(f'===== {name}'); print(t.drop(columns=[c for c in ['p'] if c not in t]).round(3).to_string(index=False), flush=True)
