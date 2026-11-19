"""Mechanism stages M1-M6 (see PIPELINE.md). One command runs everything and writes the gate metrics.

  python mech.py all            # or: python mech.py M1 M3 ...
Outputs: results/mech/<stage>_*.csv and results/mech/gates.json (merged across runs).

Model of a window feature x used throughout:
  x = person + day + slow drift + task-vs-rest (engagement) + task family + difficulty + time-on-task + motion + noise
Each stage estimates one or more of these terms; none uses a new model class.
"""
import os, sys, json, glob, itertools, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E

OUT = os.path.join(E.OUT, 'mech'); os.makedirs(OUT, exist_ok=True)
GATES = os.path.join(OUT, 'gates.json')
DSS = ['COGBCI', 'UNIVERSE', 'MPHRC', 'COGLOAD']
GROUP = {'hr_mean': 'cardiac', 'hr_std': 'cardiac', 'rmssd': 'cardiac', 'sdnn': 'cardiac', 'pnn50': 'cardiac',
         'scl_mean': 'eda', 'scl_slope': 'eda', 'phasic_std': 'eda', 'scr_rate': 'eda',
         'temp_mean': 'temp', 'temp_slope': 'temp', 'acc_std': 'motion', 'resp_rate': 'resp', 'resp_amp': 'resp'}
RNG = np.random.default_rng(0)


def gate(stage, **kv):
    g = json.load(open(GATES)) if os.path.exists(GATES) else {}
    g.setdefault(stage, {}).update({k: (round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in kv.items()})
    json.dump(g, open(GATES, 'w'), indent=1, ensure_ascii=False)


def boot_ci(v, n=2000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return (np.nan, np.nan)
    m = [RNG.choice(v, len(v)).mean() for _ in range(n)]
    return [round(float(v), 3) for v in np.percentile(m, [2.5, 97.5])]


def load(ds, norm='none'):
    """Windows with block key. norm: none | unit (z over all windows of the person-session) | rest."""
    df, feats = E.load(ds)
    df = df.reset_index(drop=True)
    df['block'] = df['unit'] + '|' + df['task'].astype(str)
    df['is_task'] = (df.level >= 0).astype(int)
    if norm == 'rest':
        df = E.normalise(df, feats, 'rest')
    elif norm == 'unit':
        g = df.groupby('unit')[feats]
        df[feats] = (df[feats] - g.transform('mean')) / g.transform('std').replace(0, np.nan)
    return df, feats


def leveled(df):
    """Families with >=2 difficulty levels; adds y (1 = highest level, 0 = lowest), middle levels dropped."""
    d, fams = E.binary_task(df)
    return d, fams


def lr():
    return E.model()


def within_auc(d, score, label='y'):
    """Mean over people of the within-person AUROC (averaged over that person's sessions first)."""
    out = {}
    for pid, g in d.assign(_s=score).groupby('pid'):
        aucs = [roc_auc_score(h[label], h._s) for _, h in g.groupby('unit') if h[label].nunique() == 2]
        if aucs:
            out[pid] = np.mean(aucs)
    return pd.Series(out)


# ---------------------------------------------------------------- M1 signal anatomy
def M1():
    """Nested sum-of-squares decomposition per feature (raw values):
    person > session(day) > task-vs-rest > family > level > linear time-in-block > residual."""
    rows = []
    for ds in DSS:
        df, feats = load(ds)
        for f in feats:
            d = df[['pid', 'unit', 'is_task', 'family', 'level', 'block', 't0', f]].dropna()
            if len(d) < 100:
                continue
            x = d[f].values; tot = ((x - x.mean()) ** 2).sum()
            def ss(keys):
                m = d.groupby(keys)[f].transform('mean').values
                return ((m - x.mean()) ** 2).sum()
            s_p, s_u = ss(['pid']), ss(['unit'])
            s_e, s_f, s_l = ss(['unit', 'is_task']), ss(['unit', 'is_task', 'family']), ss(['unit', 'is_task', 'family', 'level'])
            s_b = ss(['block'])  # block = unit x task (repetitions split here)
            r = x - d.groupby('block')[f].transform('mean').values
            tc = d.t0 - d.groupby('block').t0.transform('mean')
            num = (r * tc).groupby(d.block).sum(); den = (tc ** 2).groupby(d.block).sum().replace(0, np.nan)
            slope = (num / den).fillna(0)
            fit = tc * d.block.map(slope)
            s_t = (fit ** 2).sum()
            res = ((r - fit) ** 2).sum()
            lv_rows = d.level >= 0
            # partial eta^2 of level among leveled families only
            dl = d[lv_rows & d.family.isin(d[lv_rows].groupby('family')['level'].nunique().loc[lambda s: s >= 2].index)]
            if len(dl):
                xl = dl[f].values
                ssl = ((dl.groupby(['unit', 'family', 'level'])[f].transform('mean') - dl.groupby(['unit', 'family'])[f].transform('mean')) ** 2).sum()
                ssw = ((xl - dl.groupby(['unit', 'family', 'level'])[f].transform('mean')) ** 2).sum()
                pe = ssl / (ssl + ssw)
            else:
                pe = np.nan
            rows.append(dict(dataset=ds, feature=f, group=GROUP.get(f, ''), person=s_p / tot, session=(s_u - s_p) / tot,
                             task_vs_rest=(s_e - s_u) / tot, family=(s_f - s_e) / tot, level=(s_l - s_f) / tot,
                             block_other=(s_b - s_l) / tot, time_in_block=s_t / tot, residual=res / tot, level_partial_eta2=pe))
    t = pd.DataFrame(rows)
    t.to_csv(f'{OUT}/M1_variance.csv', index=False)
    for ds, g in t.groupby('dataset'):
        gate('M1', **{f'{ds}_person_plus_session_median': (g.person + g.session).median(),
                      f'{ds}_level_partial_eta2_median': g.level_partial_eta2.median(),
                      f'{ds}_task_vs_rest_median': g.task_vs_rest.median(),
                      f'{ds}_time_in_block_median': g.time_in_block.median()})
        for grp, h in g.groupby('group'):
            gate('M1', **{f'{ds}_{grp}_level_eta2': h.level_partial_eta2.median(), f'{ds}_{grp}_time_share': h.time_in_block.median()})
    return t


# ---------------------------------------------------------------- M2 structure of the difficulty effect
def effect_sizes(ds):
    df, feats = load(ds, 'rest')
    d, fams = leveled(df)
    rows = []
    for (u, fam), g in d.groupby(['unit', 'family']):
        a, b = g[g.y == 0], g[g.y == 1]
        a = a.sort_values('t0'); h1, h2 = a.iloc[:len(a) // 2], a.iloc[len(a) // 2:]  # drift-only reference: halves of the easy block
        for f in feats:
            xa, xb = a[f].dropna(), b[f].dropna()
            if len(xa) < 3 or len(xb) < 3:
                continue
            sp = np.sqrt((xa.var() + xb.var()) / 2)
            y1, y2 = h1[f].dropna(), h2[f].dropna()
            s0 = np.sqrt((y1.var() + y2.var()) / 2) if len(y1) > 2 and len(y2) > 2 else np.nan
            rows.append(dict(dataset=ds, unit=u, pid=g.pid.iloc[0], family=fam, feature=f, d=(xb.mean() - xa.mean()) / sp if sp > 0 else np.nan,
                             d_null=(y2.mean() - y1.mean()) / s0 if s0 and s0 > 0 else np.nan))
    return pd.DataFrame(rows), feats, fams


def M2():
    allrows, cons = [], []
    for ds in DSS:
        es, feats, fams = effect_sizes(ds)
        if es.empty:
            continue
        allrows.append(es)
        p = es.groupby(['pid', 'family', 'feature'])[['d', 'd_null']].mean().reset_index()
        # group-level effect per family x feature.
        # consistency ratio CR = mean(d)^2 / mean(d^2): 1 = everyone shows the same effect, ~0 = sign is random across people
        # (what block-to-block drift produces when level order is randomised).
        for (fam, f), g in p.groupby(['family', 'feature']):
            v = g.d.dropna(); v0 = g.d_null.dropna()
            cons.append(dict(dataset=ds, family=fam, feature=f, group=GROUP.get(f, ''), mean_d=v.mean(), mean_abs_d=v.abs().mean(),
                             CR=v.mean() ** 2 / (v ** 2).mean() if len(v) else np.nan, mean_abs_d_null=v0.abs().mean(),
                             sign_pos=(v > 0).mean(), n=len(v), p=wilcoxon(v).pvalue if len(v) > 5 and (v != 0).any() else np.nan))
        cg = pd.DataFrame([c for c in cons if c['dataset'] == ds])
        for grp, h in cg.groupby('group'):
            gate('M2', **{f'{ds}_{grp}_CR_median': h.CR.median(), f'{ds}_{grp}_abs_d_median': h.mean_abs_d.median(),
                          f'{ds}_{grp}_abs_d_null_median': h.mean_abs_d_null.median()})
        # cross-family consistency of the difficulty direction
        M = p.pivot_table(index=['pid', 'feature'], columns='family', values='d')
        pairs = list(itertools.combinations([f for f in fams if f in M.columns], 2))
        grp_r, wp_r, sign_agree = [], [], []
        for A, B in pairs:
            gm = M.groupby(level='feature')[[A, B]].mean().dropna()
            if len(gm) >= 3:
                grp_r.append(np.corrcoef(gm[A], gm[B])[0, 1])
            for pid, h in M[[A, B]].dropna().groupby(level='pid'):
                if len(h) >= 3 and h[A].std() > 0 and h[B].std() > 0:
                    wp_r.append(np.corrcoef(h[A], h[B])[0, 1])
                sign_agree.append((np.sign(h[A]) == np.sign(h[B])).mean())
        gate('M2', **{f'{ds}_group_dir_r_mean': np.nanmean(grp_r) if grp_r else np.nan,
                      f'{ds}_within_person_dir_r_median': np.nanmedian(wp_r) if wp_r else np.nan,
                      f'{ds}_sign_agreement_mean': np.nanmean(sign_agree) if sign_agree else np.nan,
                      f'{ds}_abs_mean_d_median': pd.DataFrame(cons).query('dataset == @ds').mean_d.abs().median()})
    pd.concat(allrows).to_csv(f'{OUT}/M2_effects_long.csv', index=False)
    t = pd.DataFrame(cons); t.to_csv(f'{OUT}/M2_effects.csv', index=False)
    return t


def M2b(horizons=(30, 60, 120, 180)):
    """Time-scale: LOSO population model (rest-normalised), scores averaged over non-overlapping horizons within a block."""
    rows = []
    for ds in ['COGBCI', 'UNIVERSE', 'MPHRC', 'COGLOAD']:
        df, feats = load(ds, 'rest')
        d, fams = leveled(df)
        d = d.dropna(subset=feats, how='all')
        for pid in d.pid.unique():
            tr_all, te_all = d[d.pid != pid], d[d.pid == pid]
            for A, B in itertools.product(fams, fams):
                tr, te = tr_all[tr_all.family == A], te_all[te_all.family == B].copy()
                if tr.y.nunique() < 2 or te.y.nunique() < 2:
                    continue
                m = lr().fit(tr[feats], tr.y); te['s'] = m.decision_function(te[feats])
                for H in horizons:
                    agg = te.assign(h=(te.t0 // H).astype(int)).groupby(['unit', 'block', 'h']).agg(s=('s', 'mean'), y=('y', 'first')).reset_index()
                    aucs = [roc_auc_score(g.y, g.s) for _, g in agg.groupby('unit') if g.y.nunique() == 2]
                    if aucs:
                        rows.append(dict(dataset=ds, pid=pid, source=A, target=B, xtask='same' if A == B else 'cross', H=H, auc=np.mean(aucs)))
    t = pd.DataFrame(rows); t.to_csv(f'{OUT}/M2b_horizon.csv', index=False)
    s = t.groupby(['dataset', 'xtask', 'H', 'pid']).auc.mean().groupby(['dataset', 'xtask', 'H']).mean()
    for (ds, x, H), v in s.items():
        gate('M2b', **{f'{ds}_{x}_H{H}': v})
    return t


# ---------------------------------------------------------------- M3 engagement vs load
def M3():
    rows, geo = [], []
    for ds in DSS:
        for drop_motion in ([False, True] if ds in ('UNIVERSE', 'COGLOAD') else [False]):
            df, feats = load(ds, 'unit')
            if drop_motion:
                feats = [f for f in feats if f != 'acc_std']
            fl = df[df.level >= 0].groupby('family')['level'].agg(['min', 'max', 'nunique'])
            fams = fl[fl['nunique'] >= 2].index
            rest = df[df.family == 'rest']
            for fam in fams:
                lo, hi = fl.loc[fam, 'min'], fl.loc[fam, 'max']
                E_ = df[(df.family == fam) & (df.level == lo)]; H_ = df[(df.family == fam) & (df.level == hi)]
                contrasts = {'easy_vs_rest': (rest, E_), 'hard_vs_rest': (rest, H_), 'hard_vs_easy': (E_, H_)}
                for name, (neg, pos) in contrasts.items():
                    d = pd.concat([neg.assign(y=0), pos.assign(y=1)]).dropna(subset=feats, how='all')
                    sc = pd.Series(np.nan, index=d.index)
                    for pid in d.pid.unique():
                        tr, te = d[d.pid != pid], d[d.pid == pid]
                        if tr.y.nunique() == 2 and len(te):
                            sc.loc[te.index] = lr().fit(tr[feats], tr.y).decision_function(te[feats])
                    a = within_auc(d, sc.values)
                    for pid, v in a.items():
                        rows.append(dict(dataset=ds, motion_removed=drop_motion, family=fam, contrast=name, pid=pid, auc=v))
                if not drop_motion:  # geometry of the two effects in unit-z space, per person
                    for pid in df.pid.unique():
                        r_, e_, h_ = (x[x.pid == pid][feats].mean() for x in (rest, E_, H_))
                        vt, vl = (e_ - r_).values, (h_ - e_).values
                        ok = np.isfinite(vt) & np.isfinite(vl)
                        if ok.sum() >= 3 and np.linalg.norm(vt[ok]) > 0 and np.linalg.norm(vl[ok]) > 0:
                            geo.append(dict(dataset=ds, family=fam, pid=pid,
                                            cos=float(vt[ok] @ vl[ok] / np.linalg.norm(vt[ok]) / np.linalg.norm(vl[ok])),
                                            norm_ratio=float(np.linalg.norm(vl[ok]) / np.linalg.norm(vt[ok]))))
    t = pd.DataFrame(rows); t.to_csv(f'{OUT}/M3_contrasts.csv', index=False)
    G = pd.DataFrame(geo); G.to_csv(f'{OUT}/M3_geometry.csv', index=False)
    s = t.groupby(['dataset', 'motion_removed', 'contrast', 'pid']).auc.mean().groupby(['dataset', 'motion_removed', 'contrast']).mean()
    for ds in t.dataset.unique():
        er = (s[(ds, False, 'hard_vs_rest')] - s[(ds, False, 'easy_vs_rest')]) / max(s[(ds, False, 'easy_vs_rest')] - 0.5, 1e-3)
        kv = {f'{ds}_easy_vs_rest': s[(ds, False, 'easy_vs_rest')], f'{ds}_hard_vs_rest': s[(ds, False, 'hard_vs_rest')],
              f'{ds}_hard_vs_easy': s[(ds, False, 'hard_vs_easy')], f'{ds}_engagement_ratio': er}
        if (ds, True, 'easy_vs_rest') in s:
            kv[f'{ds}_easy_vs_rest_no_motion'] = s[(ds, True, 'easy_vs_rest')]
        if len(G) and ds in set(G.dataset):
            g = G[G.dataset == ds].groupby('pid')[['cos', 'norm_ratio']].mean()
            kv[f'{ds}_cos_task_load_median'] = g.cos.median(); kv[f'{ds}_load_over_task_norm_median'] = g.norm_ratio.median()
        gate('M3', **kv)
    return t


# ---------------------------------------------------------------- M4 what is the label?
def rm_corr(df, x, y):
    """Repeated-measures correlation: pooled Pearson r after centring x and y within person."""
    d = df[['pid', x, y]].dropna()
    d = d[d.groupby('pid')[y].transform('nunique') > 1]
    xc = d[x] - d.groupby('pid')[x].transform('mean'); yc = d[y] - d.groupby('pid')[y].transform('mean')
    return float((xc * yc).sum() / np.sqrt((xc ** 2).sum() * (yc ** 2).sum())), d.pid.nunique()


def boot_rm(df, x, y, n=1000):
    pids = df.pid.unique(); vals = []
    for _ in range(n):
        s = RNG.choice(pids, len(pids))
        b = pd.concat([df[df.pid == p].assign(pid=f'{p}_{i}') for i, p in enumerate(s)])
        vals.append(rm_corr(b, x, y)[0])
    return np.percentile(vals, [2.5, 97.5])


def M4():
    rows, agree_rows = [], []
    for ds in DSS:
        df, feats = load(ds, 'rest')
        task = df[df.level >= 0].dropna(subset=feats, how='all').copy()
        fl = task.groupby('family')['level'].agg(['min', 'max', 'nunique'])
        lev_fams = fl[fl['nunique'] >= 2].index
        tr_d = task[task.family.isin(lev_fams)].copy()
        tr_d = tr_d[(tr_d.level == tr_d.family.map(fl['min'])) | (tr_d.level == tr_d.family.map(fl['max']))]
        tr_d['y'] = (tr_d.level == tr_d.family.map(fl['max'])).astype(int)
        task['score'] = np.nan
        for pid in task.pid.unique():  # LOSO difficulty model (pooled leveled families) scored on every task window
            tr = tr_d[tr_d.pid != pid]
            if tr.y.nunique() == 2:
                task.loc[task.pid == pid, 'score'] = lr().fit(tr[feats], tr.y).decision_function(task.loc[task.pid == pid, feats])
        task['level_n'] = task.level / task.family.map(fl['max']).replace(0, np.nan)
        B = task.groupby(['pid', 'unit', 'block', 'family']).agg(score=('score', 'mean'), subj=('subjective', 'first'), level_n=('level_n', 'first'),
                                                                   **{f: (f, 'mean') for f in feats}).reset_index()
        B['subj_z'] = B.groupby('unit').subj.transform(lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s * np.nan)
        L = B[B.family.isin(lev_fams)]
        for tgt, src in (('subjective', B), ('subjective_leveled_only', L), ('level', L)):
            col = 'subj' if tgt.startswith('subjective') else 'level_n'
            r, n = rm_corr(src, 'score', col)
            rows.append(dict(dataset=ds, x='score', target=tgt, r=r, n_people=n))
            for f in feats:
                rf, nf = rm_corr(src, f, col)
                rows.append(dict(dataset=ds, x=f, target=tgt, r=rf, n_people=nf))
        # manipulation check and AUROC by agreement
        for (u, fam), g in L.groupby(['unit', 'family']):
            lo, hi = g[g.level_n == g.level_n.min()], g[g.level_n == g.level_n.max()]
            if len(lo) and len(hi) and lo.subj.notna().all() and hi.subj.notna().all() and lo.level_n.iloc[0] != hi.level_n.iloc[0]:
                w = tr_d[(tr_d.unit == u) & (tr_d.family == fam)]
                if w.y.nunique() == 2:
                    sc = task.loc[w.index, 'score']
                    agree_rows.append(dict(dataset=ds, unit=u, pid=g.pid.iloc[0], family=fam, subj_diff=hi.subj.mean() - lo.subj.mean(),
                                           auc=roc_auc_score(w.y, sc)))
        lc = L.dropna(subset=['score', 'subj', 'level_n'])
        if lc.pid.nunique() >= 5 and lc.subj.notna().any():
            rs, _ = rm_corr(lc, 'score', 'subj'); rl, _ = rm_corr(lc, 'score', 'level_n')
            ci_s, ci_l = boot_rm(lc, 'score', 'subj', 300), boot_rm(lc, 'score', 'level_n', 300)
            gate('M4', **{f'{ds}_r_score_subj': rs, f'{ds}_r_score_level': rl, f'{ds}_r_score_subj_ci': list(np.round(ci_s, 3)),
                          f'{ds}_r_score_level_ci': list(np.round(ci_l, 3))})
            ra, _ = rm_corr(B.dropna(subset=['score', 'subj']), 'score', 'subj')
            gate('M4', **{f'{ds}_r_score_subj_all_blocks': ra})
    t = pd.DataFrame(rows); t.to_csv(f'{OUT}/M4_label_corr.csv', index=False)
    A = pd.DataFrame(agree_rows); A.to_csv(f'{OUT}/M4_agreement.csv', index=False)
    for ds, g in A.groupby('dataset'):
        ag, dis = g[g.subj_diff > 0].auc, g[g.subj_diff <= 0].auc
        gate('M4', **{f'{ds}_manip_agree_rate': (g.subj_diff > 0).mean(), f'{ds}_auc_agree': ag.mean(), f'{ds}_auc_disagree': dis.mean(),
                      f'{ds}_auc_agree_minus_disagree': ag.mean() - dis.mean(), f'{ds}_n_agree_disagree': f'{len(ag)}/{len(dis)}'})
    return t, A


# ---------------------------------------------------------------- M5 time confounds
def M5():
    rows = []
    for ds in DSS:
        df, feats = load(ds, 'rest')
        task = df[df.level >= 0]
        for f in feats:  # within-block slope in rest-z units per minute
            d = task[['pid', 'block', 't0', f]].dropna()
            sl = d.groupby('block').apply(lambda g: np.polyfit(g.t0 / 60, g[f], 1)[0] if len(g) > 5 and g[f].std() > 0 else np.nan)
            pid_of = d.groupby('block').pid.first()
            pp = sl.groupby(pid_of).mean().dropna()
            rows.append(dict(dataset=ds, feature=f, slope_per_min=pp.mean(), sign_pos=(pp > 0).mean(), n=len(pp),
                             p=wilcoxon(pp).pvalue if len(pp) > 5 else np.nan))
        # detrending: remove each block's linear trend, keep its mean; compare LOSO difficulty AUROC
        d, fams = leveled(df)
        d = d.dropna(subset=feats, how='all')
        dt = d.copy()
        for f in feats:
            tc = dt.t0 - dt.groupby('block').t0.transform('mean')
            xc = dt[f] - dt.groupby('block')[f].transform('mean')
            sl = (xc * tc).groupby(dt.block).sum() / (tc ** 2).groupby(dt.block).sum().replace(0, np.nan)
            dt[f] = dt[f] - (tc * dt.block.map(sl)).fillna(0)
        res = {}
        for name, data in (('orig', d), ('detrended', dt)):
            for x in ('same', 'cross'):
                aucs = []
                for pid in data.pid.unique():
                    for A, Bf in itertools.product(fams, fams):
                        if (A == Bf) != (x == 'same'):
                            continue
                        tr = data[(data.pid != pid) & (data.family == A)]; te = data[(data.pid == pid) & (data.family == Bf)]
                        if tr.y.nunique() < 2 or te.y.nunique() < 2:
                            continue
                        a = within_auc(te, lr().fit(tr[feats], tr.y).decision_function(te[feats]))
                        aucs += list(a.values)
                res[f'{name}_{x}'] = np.mean(aucs) if aucs else np.nan
        gate('M5', **{f'{ds}_{k}': v for k, v in res.items()})
        # COG-BCI: session drift between beginning and end rest
        if ds == 'COGBCI':
            rs = df[df.family.isin(['rest', 'rest_ec'])].copy()
            rs['when'] = np.where(rs.task.str.contains('Beg'), 'beg', 'end')
            dd = []
            for (u, fam), g in rs.groupby(['unit', 'family']):
                a, b = g[g.when == 'beg'], g[g.when == 'end']
                for f in feats:
                    if a[f].notna().sum() > 2 and b[f].notna().sum() > 2:
                        sp = np.sqrt((a[f].var() + b[f].var()) / 2)
                        dd.append(dict(unit=u, pid=g.pid.iloc[0], feature=f, d=(b[f].mean() - a[f].mean()) / sp if sp > 0 else np.nan))
            dd = pd.DataFrame(dd); dd.to_csv(f'{OUT}/M5_cogbci_rest_drift.csv', index=False)
            m = dd.groupby(['feature', 'pid']).d.mean().groupby('feature').mean()
            gate('M5', **{f'COGBCI_rest_end_minus_beg_d_{f}': v for f, v in m.items()})
    t = pd.DataFrame(rows); t.to_csv(f'{OUT}/M5_time_slopes.csv', index=False)
    return t


# ---------------------------------------------------------------- M6 generality of block-identity leakage
def M6():
    import exp4_leakage as L
    rows, long_rows = [], []
    for ds in ['MPHRC', 'COGLOAD', 'UNIVERSE', 'COGBCI']:
        full = list(E.FEATS[ds])
        groups = {'all': full}
        for grp in ('cardiac', 'eda', 'temp', 'resp', 'motion'):
            fs = [f for f in full if GROUP.get(f) == grp]
            if fs:
                groups[grp] = fs
        for gname, fs in groups.items():
            E.FEATS[ds] = fs
            keep_out = E.OUT
            E.OUT = os.path.join(OUT, 'M6_tmp'); os.makedirs(E.OUT, exist_ok=True)  # never overwrite Exp 4 outputs
            try:
                n = L.run_null(ds)
            except Exception as e:
                print(ds, gname, 'null failed', e); continue
            finally:
                E.FEATS[ds] = full; E.OUT = keep_out
            if n.empty:
                continue
            long_rows.append(n.assign(features=gname))
            s = n.groupby(['k', 'cond', 'pid']).auc.mean().groupby(['k', 'cond']).mean()
            for (k, c), v in s.items():
                rows.append(dict(dataset=ds, features=gname, k=k, cond=c, auc=v, n_people=n.pid.nunique()))
    t = pd.DataFrame(rows); t.to_csv(f'{OUT}/M6_null_by_group.csv', index=False)
    pd.concat(long_rows).to_csv(f'{OUT}/M6_null_long.csv', index=False)
    for (ds, g), h in t[(t.k == 6) & (t.cond == 'personal')].groupby(['dataset', 'features']):
        gate('M6', **{f'{ds}_{g}_null_personal_k6': h.auc.iloc[0]})
    return t


# ---------------------------------------------------------------- M5o order: is difficulty confounded with time?
COG_ORDER = {1: 'PVT', 2: 'Flanker', 3: 'twoBACK', 4: 'oneBACK', 5: 'zeroBACK', 6: 'MATBeasy', 7: 'MATBmed', 8: 'MATBdiff'}


def block_times(ds):
    """(pid, session, task) -> block start in minutes from the first task block of that session, where the design allows it."""
    rows = []
    if ds == 'COGBCI':
        import scipy.io as sio
        nb = sio.loadmat(os.path.join(E.R, 'COGBCI', 'notebook.mat'), squeeze_me=True, struct_as_record=False)['notebook']
        for i in range(1, 30):
            for j in range(1, 4):
                order = getattr(getattr(nb, f'SBJ_{i}'), f'SESS_{j}').Order
                for pos, t in enumerate(order):  # tasks last 5-10 min; position used as the time axis
                    rows.append(dict(pid=f'sub-{i:02d}', session=f'ses-S{j}', task=COG_ORDER[int(t)], start=float(pos)))
    elif ds == 'UNIVERSE':
        import build_windows as BW
        for sd in sorted(glob.glob(os.path.join(E.R, 'UNIVERSE', 'UN_*', 'Lab[12]'))):
            logs = glob.glob(f'{sd}/Raw/Psychopy/*.log')
            if not logs:
                continue
            segs = [s for s in BW.universe_segments(logs[0]) if s[1] >= 0]
            if not segs:
                continue
            t0 = min(s[2] for s in segs)
            for fam, lvl, a, b in segs:
                rows.append(dict(pid=sd.split('/')[-2], session=sd.split('/')[-1], task=BW.U_TASKNAME.get((fam, lvl), fam), start=(a - t0) / 60))
    return pd.DataFrame(rows)


def M5o(nboot=1000):
    """Within each person-session-family: dx = x(hard) - x(easy), dt = start(hard) - start(easy).
    Across blocks: dx = level_effect + time_effect * dt. Order is randomised in both datasets, so the two separate.
    Person-cluster bootstrap for CIs."""
    rows = []
    for ds in ['COGBCI', 'UNIVERSE']:
        bt = block_times(ds)
        df, feats = load(ds, 'rest')
        d, fams = leveled(df)
        bm = d.groupby(['pid', 'session', 'unit', 'family', 'task', 'y'])[feats].mean().reset_index()
        bm = bm.merge(bt, on=['pid', 'session', 'task'], how='left')
        hard, easy = bm[bm.y == 1].set_index(['unit', 'family']), bm[bm.y == 0].set_index(['unit', 'family'])
        idx = hard.index.intersection(easy.index)
        D = (hard.loc[idx, feats + ['start']] - easy.loc[idx, feats + ['start']]).reset_index()
        D['pid'] = hard.loc[idx, 'pid'].values
        D = D.dropna(subset=['start'])
        gate('M5o', **{f'{ds}_hard_after_easy_rate': (D.start > 0).mean(), f'{ds}_n_pairs': len(D)})
        pids = D.pid.unique()
        for f in feats:
            x = D[['pid', 'start', f]].dropna()
            if len(x) < 10:
                continue
            def fit(z):
                A = np.c_[np.ones(len(z)), z.start.values]
                return np.linalg.lstsq(A, z[f].values, rcond=None)[0]
            b = fit(x)
            bs = []
            for _ in range(nboot):
                s = RNG.choice(pids, len(pids))
                z = pd.concat([x[x.pid == q] for q in s])
                if len(z) > 5:
                    bs.append(fit(z))
            bs = np.array(bs)
            unit = 'per position' if ds == 'COGBCI' else 'per minute'
            rows.append(dict(dataset=ds, feature=f, group=GROUP.get(f, ''), level_effect=b[0], level_ci=[round(float(v), 3) for v in np.percentile(bs[:, 0], [2.5, 97.5])],
                             time_effect=b[1], time_ci=[round(float(v), 3) for v in np.percentile(bs[:, 1], [2.5, 97.5])], time_unit=unit,
                             time_effect_over_session=b[1] * (7 if ds == 'COGBCI' else (D.start.abs().max())), n=len(x)))
    t = pd.DataFrame(rows); t.to_csv(f'{OUT}/M5o_order.csv', index=False)
    for (ds, grp), h in t.groupby(['dataset', 'group']):
        sig_l = sum((np.sign(lo) == np.sign(hi)) and lo != 0 for lo, hi in h.level_ci)
        sig_t = sum((np.sign(lo) == np.sign(hi)) and lo != 0 for lo, hi in h.time_ci)
        gate('M5o', **{f'{ds}_{grp}_n_level_sig': sig_l, f'{ds}_{grp}_n_time_sig': sig_t, f'{ds}_{grp}_n_feats': len(h)})
    return t


# ---------------------------------------------------------------- M8 cardiac data quality (wrist PPG vs ECG)
def M8():
    """Share of windows with usable beat detection and window-to-window HR jitter at rest, per dataset/sensor."""
    rows = []
    for ds in DSS:
        df = pd.read_parquet(f'{E.W}/{ds}_windows.parquet')
        if 'ibi_valid' not in df or 'hr_mean' not in df:
            continue
        df = df.sort_values(['pid', 'session', 'task', 't0'])
        ok = df.ibi_valid >= 0.8
        r = df[ok & (df.family == 'rest')]
        # successive non-overlapping windows (30 s apart) within the same rest block
        r = r[(r.t0 % 30) == 0]
        jit = r.groupby(['pid', 'session', 'task']).hr_mean.apply(lambda s: s.diff().abs().median()).median()
        tsk = df[ok & (df.level >= 0)]
        rows.append(dict(dataset=ds, usable_share_all=ok.mean(), usable_share_task=(df[df.level >= 0].ibi_valid >= 0.8).mean(),
                         rest_hr_jitter_bpm=jit, task_hr_sd_within_block_bpm=tsk.groupby(['pid', 'session', 'task']).hr_mean.std().median()))
    t = pd.DataFrame(rows); t.to_csv(f'{OUT}/M8_cardiac_quality.csv', index=False)
    for _, x in t.iterrows():
        gate('M8', **{f"{x.dataset}_usable_share_task": x.usable_share_task, f"{x.dataset}_rest_hr_jitter_bpm": x.rest_hr_jitter_bpm})
    return t


STAGES = {'M1': M1, 'M2': M2, 'M2b': M2b, 'M3': M3, 'M4': M4, 'M5': M5, 'M5o': M5o, 'M6': M6, 'M8': M8}

if __name__ == '__main__':
    todo = sys.argv[1:] or ['all']
    if todo == ['all']:
        todo = list(STAGES)
    for s in todo:
        print(f'===== {s}', flush=True)
        STAGES[s]()
        print(json.dumps(json.load(open(GATES)).get(s, {}), indent=1, ensure_ascii=False), flush=True)
