"""R3: additional analyses on the development and held-out datasets. All exploratory.

Stages (run: python r3_analyses.py <stage> ...; outputs in results/r3/):
  gapnull   split-half null with training-test gaps (embargo) and anchor budgets k, by feature group
  gapreal   real difficulty labels, same-session personal with gaps; random-window split; clock;
            physiology with each session's linear time trend removed (label-free)
  bracket   task vs START rest and task vs END rest (COG-BCI; MultiPhysio-HRC day 2), causal start-rest norm
  crossblk  MAUS palindrome: train on blocks 1-3, test on blocks 4-6 (same day, different blocks) vs same-session
  usable    usable-beat share by device/condition at thresholds 0.5-0.9, per-person medians
  agree     wrist (or watch) vs chest/reference heart rate: MAE, bias, share within 5 bpm, rest vs task
  units     effects in physical units (bpm, deg C/h, uS/h) and resting-reference sizes
  verdict   re-judge all 24 registered predictions with one CI rule
  rfnull    split-half null and same-session protocol with a random forest
"""
import os, sys, warnings, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E
import mech as M
import prereg_r2 as P
import clock_and_degrade as CD

OUT = os.path.join(E.OUT, 'r3'); os.makedirs(OUT, exist_ok=True)
WIN = os.environ.get('WIN_DIR', '/path/to/project/data/windows')
GAPS = ['orig', 0, 30, 60, 120, 300]
RNG = np.random.default_rng(20261103)


def lr_score(Xtr, ytr, Xte, model=None):
    m = (model or E.model)(); m.fit(Xtr, ytr)
    return m.decision_function(Xte) if hasattr(m, 'decision_function') else m.predict_proba(Xte)[:, 1]


def rf():
    return make_pipeline(SimpleImputer(strategy='median'),
                         RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=0, n_jobs=1))


def auc_or_nan(y, s):
    try:
        return roc_auc_score(y, s)
    except ValueError:
        return np.nan


def gap_ok(test, anchor, gap):
    """Keep test windows whose distance to every anchor window is >= gap seconds (negative distance = overlap)."""
    if gap == 'orig':
        return np.ones(len(test), bool)
    a0, a1 = anchor.t0.values[None, :], anchor.t1.values[None, :]
    w0, w1 = test.t0.values[:, None], test.t1.values[:, None]
    dist = np.maximum(a0 - w1, w0 - a1)
    return (dist >= gap).all(axis=1)


def same_block_split(g, k, gap):
    """g: windows of one or two blocks with y. Anchor = first k windows per class; test = later windows of the same
    blocks (original exp3/exp4 rule) further restricted by the gap rule."""
    anchor = pd.concat([h.head(k) for _, h in g.groupby('y')])
    last = anchor.groupby('y').t1.max()
    test = g[(g.t0 >= g.y.map(last)) & ~g.index.isin(anchor.index)]
    test = test[gap_ok(test, anchor, gap)]
    return anchor, test


# ------------------------------------------------------------------ datasets for the null
def null_sets():
    """(label, df, feats, block mask) for constant-condition blocks."""
    out = []
    for ds in ('COGBCI', 'UNIVERSE'):
        df, feats = E.load(ds)
        df = E.normalise(df.reset_index(drop=True), feats, 'rest')
        df = df[df.level >= 0].reset_index(drop=True)
        df['block'] = df.unit + '|' + df.task.astype(str)
        out.append((ds, df, feats, np.ones(len(df), bool)))
    for name, mask in (('WESAD_CHEST', lambda d: d.family == 'rest'), ('WESAD_WRIST', lambda d: d.family == 'rest'),
                       ('MAUS_FINGER', lambda d: d.family == 'nback'), ('MAUS_WRIST', lambda d: d.family == 'nback'),
                       ('SENSECOBOT_WRIST', lambda d: d.task.isin(['task4', 'task5']))):
        df, feats = P.load(name, 'none')
        df = df.reset_index(drop=True)
        out.append((name, df, feats, mask(df).values))
    return out


def stage_gapnull(model=None, tag=''):
    rows = []
    for name, df, feats, mask in null_sets():
        groups = {g: [f for f in P.GROUPS[g] if f in feats] for g in ('cardiac', 'eda', 'temp')}
        groups['all'] = feats
        groups = {g: f for g, f in groups.items() if f}
        d = df[mask]
        for b, g in d.groupby('block'):
            g = g.sort_values('t0'); mid = g.t0.iloc[len(g) // 2]
            g = g.assign(y=(g.t0 >= mid).astype(int))
            half_len = (g.t1.max() - g.t0.min()) / 2
            for k in ((3, 6, 12) if model is None else (6,)):
                for gap in (GAPS if model is None else ('orig', 60)):
                    anchor, test = same_block_split(g, k, gap)
                    if test.y.nunique() < 2 or test.y.value_counts().min() < 3:
                        continue
                    for gname, fs in groups.items():
                        if anchor[fs].notna().any(axis=1).sum() < 2 * k * 0.5:
                            continue
                        try:
                            s = lr_score(anchor[fs], anchor.y, test[fs], model)
                        except ValueError:
                            continue
                        rows.append(dict(dataset=name, block=b, pid=str(g.pid.iloc[0]), k=k, gap=str(gap), group=gname,
                                         half_min=half_len / 60, auc=auc_or_nan(test.y, s), n_test=len(test)))
        print(name, 'done', flush=True)
    r = pd.DataFrame(rows).dropna(subset=['auc'])
    r.to_csv(f'{OUT}/gapnull{tag}_long.csv', index=False)
    s = (r.groupby(['dataset', 'group', 'k', 'gap', 'pid']).auc.mean().reset_index()
         .groupby(['dataset', 'group', 'k', 'gap']).auc.agg(['mean', 'count']).reset_index())
    s['ci'] = [M.boot_ci(r[(r.dataset == a) & (r.group == b) & (r.k == c) & (r.gap == e)].groupby('pid').auc.mean().values)
               for a, b, c, e in zip(s.dataset, s.group, s.k, s.gap)]
    s.to_csv(f'{OUT}/gapnull{tag}_summary.csv', index=False)
    return s


# ------------------------------------------------------------------ real labels: gaps, random windows, clock, detrended
def detrend_unit(df, feats):
    """Remove each person-session's linear trend over session time from every feature (no labels used)."""
    out = df.copy()
    for u, g in df.groupby('unit'):
        t = g.clock.values
        for f in feats:
            x = g[f].values; ok = np.isfinite(x) & np.isfinite(t)
            if ok.sum() > 5 and np.ptp(t[ok]) > 0:
                b = np.polyfit(t[ok], x[ok], 1)
                out.loc[g.index, f] = x - np.polyval(b, t)
    return out


def stage_gapreal(model=None, tag=''):
    rows = []
    for ds in ('COGBCI', 'UNIVERSE'):
        d, feats, fams = CD.with_clock(ds)
        dd = detrend_unit(d, feats)
        reps = {'physiology': (d, feats), 'clock': (d, ['clock']), 'physiology detrended': (dd, feats)}
        for (u, A), own in d.groupby(['unit', 'family']):
            if own.y.nunique() < 2:
                continue
            own = own.sort_values('t0'); pid = str(own.pid.iloc[0])
            for gap in (GAPS[:-1] if model is None else ('orig', 60)):
                anchor, test = same_block_split(own, CD.K, gap)
                if test.y.nunique() < 2 or test.y.value_counts().min() < 3:
                    continue
                for rep, (src, cols) in reps.items():
                    try:
                        s = lr_score(src.loc[anchor.index, cols], anchor.y, src.loc[test.index, cols], model)
                    except ValueError:
                        continue
                    rows.append(dict(dataset=ds, protocol='same-session personal', gap=str(gap), rep=rep, pid=pid, unit=u,
                                     family=A, auc=auc_or_nan(test.y, s)))
            # random-window split (the most common within-person split in the audit): 50/50, stratified, 5 repeats
            for rep_i in range(5):
                tr = own.groupby('y', group_keys=False).apply(lambda h: h.sample(frac=0.5, random_state=int(RNG.integers(1e9))))
                te = own.drop(tr.index)
                for rep, (src, cols) in reps.items():
                    try:
                        s = lr_score(src.loc[tr.index, cols], tr.y, src.loc[te.index, cols], model)
                    except ValueError:
                        continue
                    rows.append(dict(dataset=ds, protocol='random windows', gap='none', rep=rep, pid=pid, unit=u, family=A,
                                     auc=auc_or_nan(te.y, s), rep_i=rep_i))
        print(ds, 'done', flush=True)
    r = pd.DataFrame(rows).dropna(subset=['auc'])
    r.to_csv(f'{OUT}/gapreal{tag}_long.csv', index=False)
    s = (r.groupby(['dataset', 'protocol', 'gap', 'rep', 'pid']).auc.mean().reset_index()
         .groupby(['dataset', 'protocol', 'gap', 'rep']).auc.agg(['mean', 'count']).reset_index())
    s['ci'] = [M.boot_ci(r[(r.dataset == a) & (r.protocol == b) & (r.gap == c) & (r.rep == e)].groupby('pid').auc.mean().values)
               for a, b, c, e in zip(s.dataset, s.protocol, s.gap, s.rep)]
    s.to_csv(f'{OUT}/gapreal{tag}_summary.csv', index=False)
    return s


# ------------------------------------------------------------------ bracketing rests
def start_rest_norm(df, feats, start_mask):
    out = df.copy()
    for u, g in df.groupby('unit'):
        ref = g[start_mask.loc[g.index]][feats]
        if len(ref) < 3:
            out.loc[g.index, feats] = np.nan; continue
        sd = ref.std().replace(0, np.nan); sd = sd.fillna(sd.mean() if np.isfinite(sd.mean()) else 1.0)
        out.loc[g.index, feats] = (g[feats] - ref.mean()) / sd.clip(lower=1e-6)
    return out


def stage_bracket():
    rows = []
    # COG-BCI: RS_Beg_EO at session start, RS_End_EO at session end; tasks in between
    df, feats = E.load('COGBCI'); df = df.reset_index(drop=True)
    sets = [('COGBCI', df, feats, df.task.eq('RS_Beg_EO'), df.task.eq('RS_End_EO'), df.level >= 0)]
    mp, mfeats = E.load('MPHRC'); mp = mp.reset_index(drop=True)
    mp = mp[mp.unit.isin(mp[mp.task == 'rest-2'].unit)].reset_index(drop=True)
    mp = mp[mp.task.isin(['rest-1', 'rest-2']) | mp.family.isin(['manual', 'cobot'])].reset_index(drop=True)
    sets.append(('MPHRC day 2', mp, mfeats, mp.task.eq('rest-1'), mp.task.eq('rest-2'), mp.family.isin(['manual', 'cobot'])))
    for name, d, fs, beg, end, task in sets:
        for normname, dn in (('start rest', start_rest_norm(d, fs, beg)),):
            for contrast, neg in (('task vs start rest', beg), ('task vs end rest', end), ('end rest vs start rest', beg)):
                pos = end if contrast.startswith('end') else task
                sub = dn[pos | neg].assign(y=pos[pos | neg].astype(int).values)
                feats_ok = [f for f in fs if sub[f].notna().mean() > 0.3]
                for rep, cols in (('physiology', feats_ok), ('physiology without temp/EDA level', [f for f in feats_ok if f not in ('temp_mean', 'scl_mean', 'temp_slope', 'scl_slope')])):
                    s = P.loso_within_auc(sub, cols)
                    rows.append(dict(dataset=name, norm=normname, contrast=contrast, rep=rep, auc=s.mean(), ci=M.boot_ci(s.values), n=len(s)))
            # clock is 1.0 vs start rest and 0.0 vs end rest by construction (rests bracket the tasks)
    r = pd.DataFrame(rows); r.to_csv(f'{OUT}/bracket.csv', index=False)
    return r


# ------------------------------------------------------------------ MAUS cross-block
def stage_crossblk():
    rows = []
    first, second = ['trial1', 'trial3'], ['trial5', 'trial6']   # 0-back / 3-back in each half (palindrome 0-2-3-2-3-0)
    for name in ('MAUS_ECG', 'MAUS_FINGER', 'MAUS_WRIST'):
        df, feats = P.load(name, 'rest')
        df = df[df.task.isin(first + second)].copy()
        df['y'] = (df.level == 3).astype(int)
        df['clock'] = df.position * 300 + df.t0
        for pid, g in df.groupby('pid'):
            for tr_tasks, te_tasks, lab in ((first, second, 'train blocks 1-3, test 4-6'), (second, first, 'train blocks 4-6, test 1-3')):
                tr, te = g[g.task.isin(tr_tasks)], g[g.task.isin(te_tasks)]
                for rep, cols in (('physiology', feats), ('clock', ['clock'])):
                    try:
                        rows.append(dict(dataset=name, protocol=lab, rep=rep, pid=pid, auc=auc_or_nan(te.y, lr_score(tr[cols], tr.y, te[cols]))))
                    except ValueError:
                        pass
            own = g[g.task.isin(first)].sort_values('t0')
            anchor, test = same_block_split(own, 6, 'orig')
            for rep, cols in (('physiology', feats), ('clock', ['clock'])):
                try:
                    rows.append(dict(dataset=name, protocol='same-session personal (blocks 1-3)', rep=rep, pid=pid,
                                     auc=auc_or_nan(test.y, lr_score(anchor[cols], anchor.y, test[cols]))))
                except ValueError:
                    pass
    r = pd.DataFrame(rows).dropna(subset=['auc']); r.to_csv(f'{OUT}/crossblk_long.csv', index=False)
    s = r.groupby(['dataset', 'protocol', 'rep']).auc.agg(['mean', 'count']).reset_index()
    s['ci'] = [M.boot_ci(r[(r.dataset == a) & (r.protocol == b) & (r.rep == c)].auc.values) for a, b, c in zip(s.dataset, s.protocol, s.rep)]
    s.to_csv(f'{OUT}/crossblk_summary.csv', index=False)
    return s


# ------------------------------------------------------------------ usability and agreement
USE_SETS = [  # (label, window file, condition column mapping)
    ('COG-BCI chest ECG', 'COGBCI'), ('MultiPhysio-HRC chest ECG', 'MPHRC'), ('ds007554 chest ECG', 'DS007554'),
    ('UNIVERSE wrist E4 (BVP)', 'UNIVERSE'), ('WESAD chest ECG', 'WESAD_CHEST'), ('WESAD wrist E4 (BVP)', 'WESAD_WRIST'),
    ('MAUS torso ECG', 'MAUS_ECG'), ('MAUS fingertip PPG', 'MAUS_FINGER'), ('MAUS wrist PPG', 'MAUS_WRIST'),
    ('SenseCobot chest ECG', 'SENSECOBOT_LAB'), ('SenseCobot wrist E4 (vendor IBI)', 'SENSECOBOT_WRIST'),
    ('CogWear wrist E4 (BVP)', 'COGWEAR_E4'), ('CogWear Galaxy Watch4 (PPG)', 'COGWEAR_SAMSUNG')]


def condition(name, d):
    if name.startswith('WESAD'):
        return d.family.map({'rest': 'rest', 'stress': 'stress (TSST)', 'amusement': 'amusement', 'meditation': 'meditation'})
    if name == 'MPHRC':
        return np.where(d.family.isin(['rest']), 'rest', np.where(d.family.isin(['manual', 'cobot']), 'collaborative work (day 2)', 'cognitive tasks (day 1)'))
    if name == 'DS007554':
        return d.family.map({'cognitive': 'cognitive', 'motor': 'motor', 'imagery': 'imagery'})
    return np.where(d.family.str.startswith('rest'), 'rest', 'task')


def stage_usable():
    rows = []
    for label, name in USE_SETS:
        d = pd.read_parquet(f'{WIN}/{name}_windows.parquet')
        if name == 'SENSECOBOT_LAB':
            d = d[d.ecg_qc_pass == 1]
        d = d[~d.family.isin(['rest_ec'])].copy()
        d['cond'] = condition(name, d)
        avail = d.has_ibi.astype(bool) if 'has_ibi' in d else pd.Series(True, index=d.index)
        for cond, g in d.groupby('cond'):
            for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
                u = g.ibi_valid.fillna(0).ge(thr)
                pp = u.groupby(g.pid).mean()
                rows.append(dict(device=label, file=name, cond=cond, thr=thr, pooled=u.mean(), person_median=pp.median(),
                                 person_q25=pp.quantile(.25), person_q75=pp.quantile(.75), n_people=g.pid.nunique(), n_windows=len(g),
                                 share_no_data=float((~avail[g.index]).mean()),
                                 pooled_given_data=float(u[avail[g.index]].mean()) if avail[g.index].any() else np.nan))
    r = pd.DataFrame(rows); r.to_csv(f'{OUT}/usable.csv', index=False)
    return r


def stage_agree():
    rows = []
    def add(label, m, ref, est, cond):
        m = m.dropna(subset=[ref, est])
        for c, g in m.groupby(cond):
            e = g[est] - g[ref]
            pp = e.abs().groupby(g.pid).median()
            rows.append(dict(pair=label, cond=c, n_windows=len(g), n_people=g.pid.nunique(), bias=e.median(),
                             mae=e.abs().median(), person_mae_median=pp.median(), within5=(e.abs() <= 5).mean(),
                             loa_lo=e.mean() - 1.96 * e.std(), loa_hi=e.mean() + 1.96 * e.std()))
    key = ['pid', 'session', 'task', 't0']
    # reference windows must have usable beats; the estimate is taken from every window where it exists
    for label, refn, estn, estcol, gate in (
            ('WESAD wrist E4 vs chest ECG', 'WESAD_CHEST', 'WESAD_WRIST', 'hr_mean', None),
            ('MAUS wrist PPG vs torso ECG', 'MAUS_ECG', 'MAUS_WRIST', 'hr_mean', None),
            ('MAUS fingertip PPG vs torso ECG', 'MAUS_ECG', 'MAUS_FINGER', 'hr_mean', None),
            ('CogWear Galaxy Watch4 vs wrist E4', 'COGWEAR_E4', 'COGWEAR_SAMSUNG', 'hr_mean', None),
            ('SenseCobot E4 HR column vs chest ECG', 'SENSECOBOT_LAB', 'SENSECOBOT_WRIST', 'hr_e4vendor', None),
            ('SenseCobot Bangle.js HR (conf >= 80) vs chest ECG', 'SENSECOBOT_LAB', 'SENSECOBOT_WRIST', 'hr_bangle_hiconf', None)):
        a = pd.read_parquet(f'{WIN}/{refn}_windows.parquet'); b = pd.read_parquet(f'{WIN}/{estn}_windows.parquet')
        a = a[a.ibi_valid >= 0.8]
        m = a[key + ['family', 'hr_mean']].rename(columns={'hr_mean': 'ref'}).merge(
            b[key + [estcol, 'ibi_valid']].rename(columns={estcol: 'est', 'ibi_valid': 'est_valid'}), on=key)
        m['cond'] = condition(refn, m)
        add(label + ' (all estimate windows)', m, 'ref', 'est', 'cond')
        if estcol == 'hr_mean':
            add(label + ' (estimate windows passing 80%)', m[m.est_valid >= 0.8], 'ref', 'est', 'cond')
    r = pd.DataFrame(rows); r.to_csv(f'{OUT}/agree.csv', index=False)
    return r


# ------------------------------------------------------------------ physical units
def add_raw(d, ds):
    raw, _ = M.load(ds, 'none')
    if 'ibi_valid' in raw:
        raw.loc[raw.ibi_valid < 0.8, 'hr_mean'] = np.nan
    cols = [c for c in ('hr_mean', 'temp_mean', 'scl_mean') if c in raw]
    r = raw[['pid', 'session', 'task', 't0'] + cols].rename(columns={c: c + '_raw' for c in cols})
    return d.merge(r, on=['pid', 'session', 'task', 't0'], how='left')


def stage_units():
    import statsmodels.formula.api as smf
    rows = []
    # difficulty effect in bpm (block model as in W2: block means of raw HR, y + position + family, random intercept)
    for ds in ('COGBCI', 'UNIVERSE'):
        d, feats, fams = CD.with_clock(ds)
        d = add_raw(d, ds)
        bm = d.groupby(['pid', 'unit', 'family', 'task', 'y']).agg(hr=('hr_mean_raw', 'mean'), start=('clock', 'min')).reset_index().dropna()
        f = smf.mixedlm('hr ~ y + start + C(family)', bm, groups=bm.pid).fit(reml=True)
        lo, hi = f.conf_int().loc['y']
        rows.append(dict(dataset=ds, quantity='difficulty effect on HR (bpm)', estimate=f.params['y'], lo=lo, hi=hi, n=bm.pid.nunique()))
    # drift in physical units per hour (UNIVERSE: start in minutes from first task block; per-session OLS of block means)
    d, feats, fams = CD.with_clock('UNIVERSE')
    d = add_raw(d, 'UNIVERSE')
    for v, unit in (('hr_mean_raw', 'bpm/h'), ('temp_mean_raw', 'deg C/h'), ('scl_mean_raw', 'uS/h')):
        sl = []
        for u, g in d.groupby('unit'):
            bm = g.groupby('task').agg(x=(v, 'mean'), t=('clock', 'min'), y=('y', 'first')).dropna()
            if len(bm) >= 4:
                X = np.c_[np.ones(len(bm)), bm.t, bm.y]
                sl.append(np.linalg.lstsq(X, bm.x.values, rcond=None)[0][1] * 60)
        sl = np.array(sl)
        rows.append(dict(dataset='UNIVERSE', quantity=f'within-session drift of task blocks ({unit})', estimate=np.median(sl),
                         lo=np.percentile(sl, 25), hi=np.percentile(sl, 75), n=len(sl), share_negative=(sl < 0).mean(), note='median, IQR across sessions'))
    # resting reference: windows per person-session and resting SD in physical units
    for ds in ('COGBCI', 'UNIVERSE', 'MPHRC', 'MAUS_ECG', 'WESAD_CHEST', 'SENSECOBOT_LAB', 'COGWEAR_E4'):
        dd = pd.read_parquet(f'{WIN}/{ds}_windows.parquet'); dd = dd[dd.family == 'rest']
        if 'ibi_valid' in dd:
            dd = dd.assign(hr_mean=dd.hr_mean.where(dd.ibi_valid >= .8))
        g = dd.groupby(['pid', 'session'])
        rows.append(dict(dataset=ds, quantity='rest windows per person-session (median)', estimate=g.size().median(), n=g.ngroups))
        rows.append(dict(dataset=ds, quantity='resting HR SD (bpm, median over person-sessions)', estimate=g.hr_mean.std().median(),
                         lo=g.hr_mean.std().quantile(.25), hi=g.hr_mean.std().quantile(.75), n=g.ngroups))
    r = pd.DataFrame(rows); r.to_csv(f'{OUT}/units.csv', index=False)
    return r


# ------------------------------------------------------------------ verdicts
def stage_verdict():
    """One rule for every registered prediction: compare the 95% CI of the registered quantity with its threshold.
    confirmed = CI entirely on the predicted side; refuted = CI entirely on the other side; otherwise inconclusive."""
    res = os.path.join(E.OUT, 'prereg_r2')
    out = []
    for f in ('MAUS', 'WESAD', 'DS007554', 'COGWEAR', 'SENSECOBOT'):
        t = pd.read_csv(f'{res}/{f}_prereg.csv'); t['dataset'] = f
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    t.to_csv(f'{OUT}/prereg_all_raw.csv', index=False)
    return t


# ------------------------------------------------------------------ W1 with the normalisation the plan specified
def stage_w1rest():
    import confirm as C
    rows, out = [], []
    for ds in ('COGBCI', 'UNIVERSE', 'MPHRC'):
        for norm in ('rest', 'unit'):
            df, feats = M.load(ds, norm)
            fl = df[df.level >= 0].groupby('family')['level'].agg(['min', 'max', 'nunique'])
            rest = df[df.family == 'rest']
            for fam in fl[fl['nunique'] >= 2].index:
                lo, hi = fl.loc[fam, 'min'], fl.loc[fam, 'max']
                E_ = df[(df.family == fam) & (df.level == lo)]; H_ = df[(df.family == fam) & (df.level == hi)]
                for name, (neg, pos) in {'easy_vs_rest': (rest, E_), 'hard_vs_easy': (E_, H_)}.items():
                    d = pd.concat([neg.assign(y=0), pos.assign(y=1)]).dropna(subset=feats, how='all')
                    sc = pd.Series(np.nan, index=d.index)
                    for pid in d.pid.unique():
                        tr, te = d[d.pid != pid], d[d.pid == pid]
                        if tr.y.nunique() == 2 and len(te):
                            sc.loc[te.index] = E.model().fit(tr[feats], tr.y).decision_function(te[feats])
                    for pid, v in M.within_auc(d, sc.values).items():
                        rows.append(dict(dataset=ds, norm=norm, family=fam, contrast=name, pid=str(pid), auc=v))
            t = pd.DataFrame(rows); d = t[(t.dataset == ds) & (t.norm == norm)]
            fam = ' + C(family)' if d.family.nunique() > 1 else ''
            r = C.fit(f"auc ~ C(contrast, Treatment('hard_vs_easy'))" + fam, d)
            b, se, lo_, hi_ = C.coef(r, "C(contrast, Treatment('hard_vs_easy'))[T.easy_vs_rest]")
            s = d.groupby(['contrast', 'pid']).auc.mean().groupby('contrast').mean()
            out.append(dict(dataset=ds, norm=norm, easy_vs_rest=s['easy_vs_rest'], hard_vs_easy=s['hard_vs_easy'],
                            diff=b, lo=lo_, hi=hi_, p=C.p_greater(b, se), n=d.pid.nunique()))
    pd.DataFrame(rows).to_csv(f'{OUT}/w1_long.csv', index=False)
    o = pd.DataFrame(out); o.to_csv(f'{OUT}/w1_rest_vs_unit.csv', index=False)
    return o


# ------------------------------------------------------------------ heart-rate effects in bpm (difficulty and drift), for Figure 3
def stage_bpm():
    import statsmodels.formula.api as smf
    import confirm as C
    rows = []
    span = {'COGBCI': 7.0, 'UNIVERSE': None}
    for ds in ('COGBCI', 'UNIVERSE', 'MPHRC'):
        d, feats, fams = CD.with_clock(ds) if ds != 'MPHRC' else (None, None, None)
        if ds == 'MPHRC':
            df, feats = M.load(ds, 'none'); d, fams = M.leveled(df); d['clock'] = np.nan
        else:
            d = add_raw(d, ds)
        hr = 'hr_mean_raw' if ds != 'MPHRC' else 'hr_mean'
        if ds == 'MPHRC':
            d.loc[d.ibi_valid < 0.8, 'hr_mean'] = np.nan
        bm = d.groupby(['pid', 'unit', 'family', 'task', 'y']).agg(hr=(hr, 'mean'), start=('clock', 'min')).reset_index()
        has_t = bm.start.notna().any()
        bm = bm.dropna(subset=['hr'] + (['start'] if has_t else []))
        fam = ' + C(family)' if bm.family.nunique() > 1 else ''
        r = C.fit('hr ~ y' + (' + start' if has_t else '') + fam, bm.assign(pid=bm.pid.astype(str)))
        b, se, lo, hi = C.coef(r, 'y')
        rows.append(dict(dataset=ds, kind='difficulty', bpm=b, lo=lo, hi=hi, n=bm.pid.nunique()))
        if has_t:
            sp = span[ds] if span[ds] else float(bm.groupby('unit').start.max().median())
            b, se, lo, hi = C.coef(r, 'start')
            rows.append(dict(dataset=ds, kind='drift over one session', bpm=b * sp, lo=lo * sp, hi=hi * sp, n=bm.pid.nunique(), span=sp))
    df, _ = P.load('MAUS_ECG', 'none')
    bz = df[df.level.isin([0, 3])].groupby(['pid', 'task', 'level', 'position']).hr_mean.mean().reset_index().assign(y=lambda x: (x.level == 3).astype(int)).dropna()
    r = C.fit('hr_mean ~ y + position', bz)
    b, se, lo, hi = C.coef(r, 'y'); rows.append(dict(dataset='MAUS', kind='difficulty', bpm=b, lo=lo, hi=hi, n=bz.pid.nunique()))
    b, se, lo, hi = C.coef(r, 'position'); rows.append(dict(dataset='MAUS', kind='drift over one session', bpm=b * 5, lo=lo * 5, hi=hi * 5, n=bz.pid.nunique(), span=5))
    o = pd.DataFrame(rows); o.to_csv(f'{OUT}/bpm.csv', index=False)
    return o


# ------------------------------------------------------------------ same-level, different-block control
def pair_auc(A, B, fs, k=6):
    """Personal model: label block A 0 and block B 1 (same condition). Anchors = first k windows of each block;
    test = later windows of each block. AUROC > 0.5 means the blocks are recognisable without any load difference."""
    g = pd.concat([A.sort_values('t0').assign(y=0), B.sort_values('t0').assign(y=1)])
    anchor = pd.concat([h.head(k) for _, h in g.groupby('y')])
    test = g.drop(anchor.index)
    if test.y.value_counts().min() < 3 or anchor[fs].notna().any(axis=1).sum() < k:
        return np.nan
    try:
        return auc_or_nan(test.y, lr_score(anchor[fs], anchor.y, test[fs]))
    except ValueError:
        return np.nan


def stage_samelevel():
    rows = []
    def run(name, df, feats, pairs, groups):
        df = df.reset_index(drop=True)
        gs = {g: [f for f in P.GROUPS[g] if f in feats] for g in groups}
        for pid, a_key, b_key, dist in pairs:
            A = df[(df.pid == pid) & a_key(df)]; B = df[(df.pid == pid) & b_key(df)]
            if len(A) < 10 or len(B) < 10:
                continue
            for gname, fs in gs.items():
                if fs:
                    rows.append(dict(dataset=name, pid=pid, group=gname, distance=dist, auc=pair_auc(A, B, fs)))
    # ds007554: the same task in two repetitions of one visit (chest ECG; raw bpm features, standardised inside the model)
    df, feats = P.load('DS007554', 'none'); df = df.reset_index(drop=True)
    pairs = []
    for pid, g in df.groupby('pid'):
        for task, h in g.groupby('task'):
            ss = sorted(h.session.unique())
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    pairs.append((pid, (lambda d, t=task, s_=ss[i]: (d.task == t) & (d.session == s_)),
                                  (lambda d, t=task, s_=ss[j]: (d.task == t) & (d.session == s_)), f'repetition {j - i} apart'))
    run('ds007554', df, feats, pairs, ['cardiac'])
    # MultiPhysio-HRC day 2: two blocks of the same work mode (different battery models)
    df, feats = P.load('MPHRC', 'none'); df = df.reset_index(drop=True)
    pairs = []
    for pid, g in df[df.family.isin(['manual', 'cobot'])].groupby('pid'):
        for mode in ('manual', 'cobot'):
            for i in range(1, 6):
                for j in range(i + 1, 6):
                    pairs.append((pid, (lambda d, t=f'{mode}-task-{i}': d.task == t), (lambda d, t=f'{mode}-task-{j}': d.task == t),
                                  'adjacent' if j - i == 1 else 'two or more apart'))
    run('MultiPhysio-HRC day 2', df, feats, pairs, ['cardiac', 'eda', 'resp'])
    # WESAD: meditation 1 vs meditation 2 (about 40 min apart)
    for name in ('WESAD_CHEST', 'WESAD_WRIST'):
        df, feats = P.load(name, 'none'); df = df.reset_index(drop=True)
        pairs = [(pid, (lambda d: d.task == 'meditation_1'), (lambda d: d.task == 'meditation_2'), 'about 40 min apart') for pid in df.pid.unique()]
        run(name, df, feats, pairs, ['cardiac', 'eda', 'temp'])
    r = pd.DataFrame(rows).dropna(subset=['auc']); r.to_csv(f'{OUT}/samelevel_long.csv', index=False)
    out = []
    for (ds, grp, dist), g in r.groupby(['dataset', 'group', 'distance']):
        pp = g.groupby('pid').auc.mean()
        out.append(dict(dataset=ds, group=grp, distance=dist, auc=pp.mean(), ci=M.boot_ci(pp.values), n=len(pp)))
    for (ds, grp), g in r.groupby(['dataset', 'group']):
        pp = g.groupby('pid').auc.mean()
        out.append(dict(dataset=ds, group=grp, distance='all pairs', auc=pp.mean(), ci=M.boot_ci(pp.values), n=len(pp)))
    o = pd.DataFrame(out); o.to_csv(f'{OUT}/samelevel_summary.csv', index=False)
    return o


# ------------------------------------------------------------------ offline decision layer (registered, exploratory)
def stage_decision():
    """Simulated adaptation policies on COG-BCI (graded load, chest ECG heart rate) and MultiPhysio-HRC day 2 (engagement).
    Load policy: trigger 'reduce demand' when the trailing 60 s mean of the heart-rate signal exceeds a threshold set on the
    other participants (LOSO, Youden point). Signals: (fixed) z vs the start-of-session rest; (refresh) the same signal minus
    the median of the person's preceding 10 min of task windows (label-free rolling baseline).
    False trigger = share of easiest-level windows flagged; missed trigger = share of hardest-level windows not flagged;
    reported for the first and second half of each session. Engagement policy (MultiPhysio-HRC): LOSO logistic model
    work vs start rest, threshold at the Youden point of the other people; false trigger = rest windows flagged as work (start rest, end rest)."""
    from sklearn.metrics import roc_curve
    rows = []
    df, feats = E.load('COGBCI'); df = df.reset_index(drop=True)
    df = start_rest_norm(df, ['hr_mean'], df.task.eq('RS_Beg_EO'))
    bt = M.block_times('COGBCI'); df = df.merge(bt, on=['pid', 'session', 'task'], how='left')
    d, fams = M.leveled(df[df.level >= 0])
    d = d.dropna(subset=['start']).copy()
    d['clock'] = d.start * 600 + d.t0                         # seconds; blocks ~5-10 min, position-major order
    d = d.sort_values(['unit', 'clock'])
    d['sig_fixed'] = d.groupby(['unit', 'task']).hr_mean.transform(lambda x: x.rolling(6, min_periods=3).mean())
    def rolling_ref(g):
        t = g.clock.values; x = g.sig_fixed.values; out = np.full(len(g), np.nan)
        for i in range(len(g)):
            m = (t < t[i] - 30) & (t >= t[i] - 600)
            if m.sum() >= 10:
                out[i] = x[i] - np.nanmedian(x[m])
        return pd.Series(out, index=g.index)
    d['sig_refresh'] = d.groupby('unit', group_keys=False).apply(rolling_ref)
    d['half'] = np.where(d.start <= d.groupby('unit').start.transform('median'), 'first half', 'second half')
    for sig in ('sig_fixed', 'sig_refresh'):
        for pid in d.pid.unique():
            tr = d[(d.pid != pid)].dropna(subset=[sig]); te = d[d.pid == pid].dropna(subset=[sig])
            if tr.y.nunique() < 2 or te.empty:
                continue
            fpr, tpr, thr = roc_curve(tr.y, tr[sig]); th = thr[np.argmax(tpr - fpr)]
            for half, g in te.groupby('half'):
                trig = g[sig] > th
                rows.append(dict(dataset='COG-BCI', policy={'sig_fixed': 'load, fixed rest baseline', 'sig_refresh': 'load, rolling 10-min baseline'}[sig],
                                 pid=pid, half=half, false_trigger=trig[g.y == 0].mean(), missed_trigger=1 - trig[g.y == 1].mean()))
    # engagement on the collaboration day
    mp, mf = E.load('MPHRC'); mp = mp.reset_index(drop=True)
    mp = mp[mp.unit.isin(mp[mp.task == 'rest-2'].unit)]
    mp = mp[mp.task.isin(['rest-1', 'rest-2']) | mp.family.isin(['manual', 'cobot'])].reset_index(drop=True)
    mp = start_rest_norm(mp, mf, mp.task.eq('rest-1'))
    fs = [f for f in mf if mp[f].notna().mean() > 0.3]
    mp['y'] = mp.family.isin(['manual', 'cobot']).astype(int)
    train_mask = mp.task.isin(['rest-1']) | (mp.y == 1)                 # the detector is trained on work vs start rest only
    for pid in mp.pid.unique():
        tr = mp[(mp.pid != pid) & train_mask]; te = mp[mp.pid == pid]
        m = E.model().fit(tr[fs], tr.y); ptr = m.predict_proba(tr[fs])[:, 1]
        fpr, tpr, thr = roc_curve(tr.y, ptr); th = thr[np.argmax(tpr - fpr)]     # threshold chosen on the other people
        pr = m.predict_proba(te[fs])[:, 1] > th
        te = te.assign(work=pr)
        rows.append(dict(dataset='MultiPhysio-HRC day 2', policy='engagement detector', pid=pid, half='start rest', false_trigger=te[te.task == 'rest-1'].work.mean(), missed_trigger=np.nan))
        rows.append(dict(dataset='MultiPhysio-HRC day 2', policy='engagement detector', pid=pid, half='end rest', false_trigger=te[te.task == 'rest-2'].work.mean(), missed_trigger=np.nan))
        rows.append(dict(dataset='MultiPhysio-HRC day 2', policy='engagement detector', pid=pid, half='work', false_trigger=np.nan, missed_trigger=1 - te[te.y == 1].work.mean()))
    r = pd.DataFrame(rows); r.to_csv(f'{OUT}/decision_long.csv', index=False)
    out = []
    for (ds, pol, half), g in r.groupby(['dataset', 'policy', 'half']):
        for mtr in ('false_trigger', 'missed_trigger'):
            v = g[mtr].dropna()
            if len(v):
                out.append(dict(dataset=ds, policy=pol, part=half, metric=mtr, mean=v.mean(), ci=M.boot_ci(v.values), n=len(v)))
    o = pd.DataFrame(out); o.to_csv(f'{OUT}/decision_summary.csv', index=False)
    return o


# ------------------------------------------------------------------ R5: drift vs autocorrelation; same-condition blocks with the paper's split rule
def detrend_block(g, fs):
    g = g.copy(); t = g.t0.values
    for f in fs:
        x = g[f].values.astype(float); ok = np.isfinite(x)
        if ok.sum() > 5:
            b = np.polyfit(t[ok], x[ok], 1); g.loc[:, f] = x - np.polyval(b, t)
    return g


def stage_r5null():
    """Split-half null (k = 6) with and without removing each block's linear time trend (label-free, whole block).
    If the null falls to 0.5 after detrending, monotone drift carried it; if it stays high, slower nonlinear structure or
    autocorrelation does."""
    rows = []
    for name, df, feats, mask in null_sets():
        if name.startswith('SENSECOBOT'):
            continue
        groups = {g: [f for f in P.GROUPS[g] if f in feats] for g in ('cardiac', 'eda', 'temp')}
        groups = {g: f for g, f in groups.items() if f}
        for b, g in df[mask].groupby('block'):
            g = g.sort_values('t0'); mid = g.t0.iloc[len(g) // 2]
            g = g.assign(y=(g.t0 >= mid).astype(int))
            for variant in ('raw', 'detrended'):
                gg = detrend_block(g, feats) if variant == 'detrended' else g
                for gap in ('orig', 60):
                    anchor, test = same_block_split(gg, 6, gap)
                    if test.y.nunique() < 2 or test.y.value_counts().min() < 3:
                        continue
                    for gname, fs in groups.items():
                        try:
                            s = lr_score(anchor[fs], anchor.y, test[fs])
                        except ValueError:
                            continue
                        rows.append(dict(dataset=name, pid=str(g.pid.iloc[0]), block=b, variant=variant, gap=str(gap), group=gname, auc=auc_or_nan(test.y, s)))
        print(name, 'done', flush=True)
    r = pd.DataFrame(rows).dropna(subset=['auc']); r.to_csv(f'{OUT}/r5_null_detrend_long.csv', index=False)
    out = []
    for key, g in r.groupby(['dataset', 'group', 'variant', 'gap']):
        pp = g.groupby('pid').auc.mean()
        out.append(dict(zip(['dataset', 'group', 'variant', 'gap'], key), auc=pp.mean(), ci=M.boot_ci(pp.values), n=len(pp)))
    o = pd.DataFrame(out); o.to_csv(f'{OUT}/r5_null_detrend_summary.csv', index=False)
    return o


def stage_r5samelevel():
    """Same task in two repetitions of one ds007554 visit (randomised task order), chest-ECG features. Same rule as the
    same-session personal protocol: anchors = first 6 windows of each block; test = windows of each block that start after
    its last anchor ends, optionally at least 60 s away from every anchor."""
    rows = []
    df, feats = P.load('DS007554', 'none'); df = df.reset_index(drop=True)
    fs = [f for f in P.GROUPS['cardiac'] if f in feats]
    for pid, g in df.groupby('pid'):
        for task, h in g.groupby('task'):
            ss = sorted(h.session.unique())
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    A = h[h.session == ss[i]].sort_values('t0').assign(y=0)
                    B = h[h.session == ss[j]].sort_values('t0').assign(y=1)
                    gg = pd.concat([A, B])
                    for gap in ('orig', 60):
                        anchor = pd.concat([x.head(6) for _, x in gg.groupby('y')])
                        last = anchor.groupby('y').t1.max()
                        test = gg[(gg.t0 >= gg.y.map(last)) & ~gg.index.isin(anchor.index)]
                        # anchors of the other block are in another repetition, so the gap applies within each block
                        if gap != 'orig':
                            test = test[[ (min(abs(r.t0 - anchor[anchor.y == r.y].t1.max()), 1e9) >= gap) for r in test.itertuples()]]
                        if test.y.value_counts().min() < 3 or anchor[fs].notna().any(axis=1).sum() < 6:
                            continue
                        try:
                            a = auc_or_nan(test.y, lr_score(anchor[fs], anchor.y, test[fs]))
                        except ValueError:
                            continue
                        rows.append(dict(pid=pid, task=task, apart=j - i, gap=str(gap), auc=a))
    r = pd.DataFrame(rows).dropna(subset=['auc']); r.to_csv(f'{OUT}/r5_samelevel_ds007554_long.csv', index=False)
    out = []
    for key, g in list(r.groupby(['apart', 'gap'])) + [((('all',) + (gp,)), g) for gp, g in r.groupby('gap')]:
        pp = g.groupby('pid').auc.mean()
        out.append(dict(apart=key[0], gap=key[1], auc=pp.mean(), ci=M.boot_ci(pp.values), n=len(pp)))
    o = pd.DataFrame(out); o.to_csv(f'{OUT}/r5_samelevel_ds007554_summary.csv', index=False)
    return o


def stage_r5engage():
    """UNIVERSE easy-task-vs-rest with all features, without skin temperature and SCL, and with cardiac features only
    (previously computed ad hoc; now part of the pipeline)."""
    df, feats = M.load('UNIVERSE', 'rest')
    mins = df[df.level >= 0].groupby('family')['level'].min(); rest = df[df.family == 'rest']; out = []
    for name, fs in (('all', feats), ('without temperature and SCL', [f for f in feats if f not in ('temp_mean', 'temp_slope', 'scl_mean', 'scl_slope')]),
                     ('cardiac only', [f for f in feats if f in P.CARD])):
        vals = [P.loso_within_auc(pd.concat([rest.assign(y=0), df[(df.family == fam) & (df.level == lv)].assign(y=1)]), fs) for fam, lv in mins.items()]
        pp = pd.concat(vals, axis=1).mean(axis=1)
        out.append(dict(features=name, auc=pp.mean(), ci=M.boot_ci(pp.values), n=len(pp)))
    o = pd.DataFrame(out); o.to_csv(f'{OUT}/r5_universe_engagement.csv', index=False)
    return o


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
    stages = {'gapnull': stage_gapnull, 'gapreal': stage_gapreal, 'bracket': stage_bracket, 'crossblk': stage_crossblk,
              'usable': stage_usable, 'agree': stage_agree, 'units': stage_units, 'verdict': stage_verdict, 'w1rest': stage_w1rest, 'bpm': stage_bpm, 'samelevel': stage_samelevel, 'r5null': stage_r5null, 'r5samelevel': stage_r5samelevel, 'r5engage': stage_r5engage, 'decision': stage_decision,
              'rfnull': lambda: (stage_gapnull(rf, '_rf'), stage_gapreal(rf, '_rf'))}
    for st in sys.argv[1:]:
        r = stages[st]()
        print(f'===== {st}'); print(r if not isinstance(r, pd.DataFrame) else r.round(3).to_string(index=False), flush=True)
