"""R7: additional analyses (evaluation protocols, MAUS inflation, power, bootstrap intervals, CogWear).

protocols  hard vs easy (COG-BCI, UNIVERSE) under random windows, same-session personal (1-min gap), population (LOSO)
           and cross-session personal, with all features and with cardiac features only (no EDA, temperature or motion);
           sensitivity of the personal protocols to the anchor budget k (3/6/12/24 per class) and to the logistic C (0.1/1/10).
           Also: does the physiological decision score track time within a class? Spearman correlation of the score with
           time inside each test block, oriented so that positive = towards the class recorded later (clock = +1).
maus       MAUS palindrome (0-2-3-2-3-0): (i) inflation with identical training windows (k = 6 anchors from blocks 1-3):
           same-session test windows vs. the later 0-/3-back blocks; (ii) leave-one-block-pair-out over the four
           (0-back, 3-back) block pairs, split into folds whose time order matches training and folds where it is reversed.
power      R6 shared-direction test: power for smaller and mixed drifts.
Outputs: results/r3/r7_*.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E
import mech as M
import prereg_r2 as P
import clock_and_degrade as CD
from r3_analyses import gap_ok

OUT = os.path.join(E.OUT, 'r3')
RNG = np.random.default_rng(20261104)


def model(C=1.0):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def score(Xtr, ytr, Xte, C=1.0):
    if Xtr.notna().any(axis=1).sum() < 4 or ytr.nunique() < 2:
        raise ValueError
    m = model(C); m.fit(Xtr, ytr)
    return m.decision_function(Xte)


def auc(y, s):
    try:
        return roc_auc_score(y, s)
    except ValueError:
        return np.nan


def anchors(own, k):
    a = pd.concat([h.head(k) for _, h in own.groupby('y')])
    last = a.groupby('y').t1.max()
    t = own[(own.t0 >= own.y.map(last)) & ~own.index.isin(a.index)]
    return a, t


def score_time(test, s, own):
    """Within-class Spearman correlation of score with time, oriented towards the later class."""
    later = 1 if own[own.y == 1].clock.mean() > own[own.y == 0].clock.mean() else 0
    sign = 1.0 if later == 1 else -1.0
    rs = []
    for c in (0, 1):
        m = test.y.values == c
        if m.sum() >= 5 and np.ptp(s[m]) > 0:
            rs.append(stats.spearmanr(s[m], test.clock.values[m]).correlation)
    return sign * np.mean(rs) if rs else np.nan


def stage_protocols():
    rows, st = [], []
    for ds in ('COGBCI', 'UNIVERSE'):
        d, feats, fams = CD.with_clock(ds)
        d['pid'] = d.pid.astype(str)
        fsets = {'all features': feats, 'cardiac only': [f for f in P.CARD if f in feats]}
        if ds == 'COGBCI':
            fsets = {'all features': feats}          # COG-BCI features are cardiac only
        for pid, mine in d.groupby('pid'):
            others = d[d.pid != pid]
            for A in fams:
                pop = others[others.family == A]
                units = list(mine[mine.family == A].unit.unique())
                for u in units:
                    own = mine[(mine.unit == u) & (mine.family == A)].sort_values('t0')
                    if own.y.nunique() < 2:
                        continue
                    for fname, fs in fsets.items():
                        base = dict(dataset=ds, features=fname, pid=pid, unit=u, family=A)
                        configs = [(k, 1.0) for k in (3, 6, 12, 24)] + [(6, 0.1), (6, 10.0)]
                        for k, C in configs:
                            a, t = anchors(own, k)
                            t60 = t[gap_ok(t, a, 60)]
                            if t60.y.nunique() == 2 and t60.y.value_counts().min() >= 3:
                                try:
                                    s = score(a[fs], a.y, t60[fs], C)
                                    rows.append(dict(base, k=k, C=C, protocol='same-session personal (1-min gap)', auc=auc(t60.y, s)))
                                    if k == 6 and C == 1.0:
                                        st.append(dict(base, protocol='same-session personal (1-min gap)', r=score_time(t60, s, own)))
                                except ValueError:
                                    pass
                            for ut in units:
                                if ut == u:
                                    continue
                                tst = mine[(mine.unit == ut) & (mine.family == A)]
                                if tst.y.nunique() == 2:
                                    try:
                                        s = score(a[fs], a.y, tst[fs], C)
                                        rows.append(dict(base, k=k, C=C, protocol='cross-session personal', auc=auc(tst.y, s)))
                                        if k == 6 and C == 1.0:
                                            st.append(dict(base, protocol='cross-session personal', r=score_time(tst, s, tst)))
                                    except ValueError:
                                        pass
                        # k-independent protocols (C = 1)
                        a, t = anchors(own, 6)
                        if t.y.nunique() == 2:
                            try:
                                s = score(pop[fs], pop.y, t[fs])
                                rows.append(dict(base, k=np.nan, C=1.0, protocol='population (LOSO)', auc=auc(t.y, s)))
                            except ValueError:
                                pass
                        for i in range(5):
                            tr = own.groupby('y', group_keys=False).apply(lambda h: h.sample(frac=0.5, random_state=int(RNG.integers(1e9))))
                            te = own.drop(tr.index)
                            try:
                                rows.append(dict(base, k=np.nan, C=1.0, protocol='random windows', rep_i=i, auc=auc(te.y, score(tr[fs], tr.y, te[fs]))))
                            except ValueError:
                                pass
        print(ds, 'done', flush=True)
    r = pd.DataFrame(rows).dropna(subset=['auc']); r.to_csv(f'{OUT}/r7_protocols_long.csv', index=False)
    s = pd.DataFrame(st).dropna(subset=['r']); s.to_csv(f'{OUT}/r7_scoretime_long.csv', index=False)
    summ = []
    for key, g in r.groupby(['dataset', 'features', 'protocol', 'k', 'C'], dropna=False):
        pp = g.groupby('pid').auc.mean()
        summ.append(dict(zip(['dataset', 'features', 'protocol', 'k', 'C'], key), mean=pp.mean(), ci=M.boot_ci(pp.values), n=len(pp)))
    summ = pd.DataFrame(summ); summ.to_csv(f'{OUT}/r7_protocols_summary.csv', index=False)
    ss = []
    for key, g in s.groupby(['dataset', 'features', 'protocol']):
        pp = g.groupby('pid').r.mean()
        ss.append(dict(zip(['dataset', 'features', 'protocol'], key), mean_r=pp.mean(), ci=M.boot_ci(pp.values),
                       share_positive=(pp > 0).mean(), p=stats.wilcoxon(pp).pvalue, n=len(pp)))
    ss = pd.DataFrame(ss); ss.to_csv(f'{OUT}/r7_scoretime_summary.csv', index=False)
    return summ, ss


def stage_maus():
    blocks = {'0a': 'trial1', '3a': 'trial3', '3b': 'trial5', '0b': 'trial6'}   # palindrome 0-2-3-2-3-0
    # in every fold the order of the two conditions is reversed between training and test (palindrome), so the clock scores 0;
    # 'halves' = first pair vs second pair (the cross-block protocol), 'crossed' = training blocks 1st and 4th vs test 2nd and 3rd
    folds = [(('0a', '3a'), ('0b', '3b'), 'halves'), (('0b', '3b'), ('0a', '3a'), 'halves'),
             (('0a', '3b'), ('0b', '3a'), 'crossed'), (('0b', '3a'), ('0a', '3b'), 'crossed')]
    rows, inf = [], []
    for name in ('MAUS_ECG', 'MAUS_FINGER', 'MAUS_WRIST'):
        df, feats = P.load(name, 'rest')
        df = df[df.task.isin(blocks.values())].copy()
        df['y'] = (df.level == 3).astype(int)
        df['clock'] = df.position * 300 + df.t0
        for pid, g in df.groupby('pid'):
            for rep, cols in (('physiology', feats), ('clock', ['clock'])):
                for trb, teb, kind in folds:
                    tr = g[g.task.isin([blocks[b] for b in trb])]; te = g[g.task.isin([blocks[b] for b in teb])]
                    try:
                        rows.append(dict(dataset=name, rep=rep, pid=pid, train='+'.join(trb), test='+'.join(teb), order=kind,
                                         auc=auc(te.y, score(tr[cols], tr.y, te[cols]))))
                    except ValueError:
                        pass
                # identical training windows: k = 6 anchors per class from blocks 1-3
                own = g[g.task.isin([blocks['0a'], blocks['3a']])].sort_values('t0')
                a, t = anchors(own, 6); later = g[g.task.isin([blocks['0b'], blocks['3b']])]
                try:
                    inf.append(dict(dataset=name, rep=rep, pid=pid, same_session=auc(t.y, score(a[cols], a.y, t[cols])),
                                    cross_block=auc(later.y, score(a[cols], a.y, later[cols]))))
                except ValueError:
                    pass
        print(name, 'done', flush=True)
    r = pd.DataFrame(rows).dropna(subset=['auc']); r.to_csv(f'{OUT}/r7_maus_lobpo_long.csv', index=False)
    s = []
    for (ds, rep, kind), g in r.groupby(['dataset', 'rep', 'order']):
        pp = g.groupby('pid').auc.mean(); s.append(dict(dataset=ds, rep=rep, folds=kind, mean=pp.mean(), ci=M.boot_ci(pp.values), n=len(pp)))
    for (ds, rep), g in r.groupby(['dataset', 'rep']):
        pp = g.groupby('pid').auc.mean(); s.append(dict(dataset=ds, rep=rep, folds='all four', mean=pp.mean(), ci=M.boot_ci(pp.values), n=len(pp)))
    s = pd.DataFrame(s); s.to_csv(f'{OUT}/r7_maus_lobpo_summary.csv', index=False)
    i = pd.DataFrame(inf).dropna(); i['inflation'] = i.same_session - i.cross_block
    i.to_csv(f'{OUT}/r7_maus_inflation_long.csv', index=False)
    t = []
    for (ds, rep), g in i.groupby(['dataset', 'rep']):
        t.append(dict(dataset=ds, rep=rep, same_session=g.same_session.mean(), cross_block=g.cross_block.mean(), inflation=g.inflation.mean(),
                      ci=M.boot_ci(g.inflation.values), p=stats.wilcoxon(g.inflation).pvalue if rep == 'physiology' else np.nan, n=len(g)))
    t = pd.DataFrame(t); t.to_csv(f'{OUT}/r7_maus_inflation.csv', index=False)
    return s, t


def stage_power(n_rep=100, seed=7):
    import r6_drift_test as R
    rng = np.random.default_rng(seed); out = []

    def ds(kind, slope):
        rows = []
        for p in range(20):
            if kind == 'shared drift':
                sl = rng.normal(slope, slope / 3)
            elif kind == 'half the people drift':
                sl = rng.normal(slope, slope / 3) if p % 2 == 0 else 0.0
            for b in range(4):
                x = rng.normal(size=42)
                if kind == 'shared + person-specific':
                    sl = rng.normal(slope, 0.03)
                x = x + sl * np.arange(42)
                w = np.convolve(x, np.ones(3) / 3, mode='valid')
                for i, v in enumerate(w):
                    rows.append(dict(pid=f'p{p:02d}', block=f'p{p:02d}|b{b}', family='sim', t0=10.0 * i, t1=10.0 * i + 30, f1=v))
        return pd.DataFrame(rows)

    for kind in ('shared drift', 'half the people drift', 'shared + person-specific'):
        for slope in (0.005, 0.01, 0.02, 0.03):
            for rep in range(n_rep):
                d = ds(kind, slope)
                r, dist = R.nulls(d, {'f1': ['f1']}, n_perm=500, seed=int(rng.integers(1 << 31)), personal=False)
                _, p_t = R.sign_test(r, 'f1')
                out.append(dict(kind=kind, slope=slope, rep=rep, population=dist['f1'][0], p_population=R.perm_p(dist['f1']), p_t=p_t))
            print(kind, slope, flush=True)
    s = pd.DataFrame(out); s.to_csv(f'{OUT}/r7_power_long.csv', index=False)
    t = s.groupby(['kind', 'slope'], sort=False).agg(population_null=('population', 'mean'),
                                                     power_permutation=('p_population', lambda p: (p < .05).mean()),
                                                     power_t=('p_t', lambda p: (p < .05).mean()), reps=('rep', 'count')).reset_index()
    t['drift_over_block_sd'] = t.slope * 41
    t.to_csv(f'{OUT}/r7_power.csv', index=False)
    return t



def stage_bootci(n_boot=1000, seed=11):
    """Person-level cluster bootstrap for the mixed-model heart-rate effects of Figure 2 (resample people, refit)."""
    import confirm as C
    from r3_analyses import add_raw
    rng = np.random.default_rng(seed)
    data = []
    for ds in ('COGBCI', 'UNIVERSE', 'MPHRC'):
        if ds == 'MPHRC':
            df, feats = M.load(ds, 'none'); d, fams = M.leveled(df); d['clock'] = np.nan
            d.loc[d.ibi_valid < 0.8, 'hr_mean'] = np.nan; hr = 'hr_mean'
        else:
            d, feats, fams = CD.with_clock(ds); d = add_raw(d, ds); hr = 'hr_mean_raw'
        bm = d.groupby(['pid', 'unit', 'family', 'task', 'y']).agg(hr=(hr, 'mean'), start=('clock', 'min')).reset_index()
        has_t = bm.start.notna().any()
        bm = bm.dropna(subset=['hr'] + (['start'] if has_t else [])).assign(pid=lambda x: x.pid.astype(str))
        f = 'hr ~ y' + (' + start' if has_t else '') + (' + C(family)' if bm.family.nunique() > 1 else '')
        sp = {'COGBCI': 7.0, 'UNIVERSE': float(bm.groupby('unit').start.max().median()) if has_t else None, 'MPHRC': None}[ds]
        terms = [('difficulty', 'y', 1.0)] + ([('drift over one session', 'start', sp)] if has_t else [])
        data.append((ds, bm.rename(columns={'hr': 'v'}), f.replace('hr ~', 'v ~'), terms))
    df, _ = P.load('MAUS_ECG', 'none')
    bz = (df[df.level.isin([0, 3])].groupby(['pid', 'task', 'level', 'position']).hr_mean.mean().reset_index()
          .assign(y=lambda x: (x.level == 3).astype(int)).dropna().rename(columns={'hr_mean': 'v'}))
    bz['pid'] = bz.pid.astype(str)
    data.append(('MAUS', bz, 'v ~ y + position', [('difficulty', 'y', 1.0), ('drift over one session', 'position', 5.0)]))
    rows = []
    for ds, bm, f, terms in data:
        r0 = C.fit(f, bm)
        pids = bm.pid.unique(); boots = {t[0]: [] for t in terms}
        for b in range(n_boot):
            pick = rng.choice(pids, len(pids))
            bb = pd.concat([bm[bm.pid == p].assign(pid=f'{p}_{i}') for i, p in enumerate(pick)])
            try:
                rb = C.fit(f, bb)
            except Exception:
                continue
            if rb is None:
                continue
            for name, term, sc in terms:
                boots[name].append(float(rb.params[term]) * sc)
        for name, term, sc in terms:
            b, se, lo, hi = C.coef(r0, term); v = np.array(boots[name])
            rows.append(dict(dataset=ds, kind=name, bpm=b * sc, wald_lo=lo * sc, wald_hi=hi * sc,
                             boot_lo=np.percentile(v, 2.5), boot_hi=np.percentile(v, 97.5), n_boot=len(v), n=len(pids)))
        print(ds, 'done', flush=True)
    o = pd.DataFrame(rows); o.to_csv(f'{OUT}/r7_bpm_bootci.csv', index=False)
    return o


def stage_cogwear():
    """CogWear E4 Stroop vs baseline (P4a design) with per-person values for intervals; all features and cardiac only."""
    df, feats = P.load('COGWEAR_E4', 'unit')
    d = df[df.family.isin(['rest', 'stroop'])].assign(y=lambda x: (x.family == 'stroop').astype(int))
    rows = []
    for fname, fs in (('all features', feats), ('cardiac only', [f for f in P.CARD if f in feats])):
        for pid, g in d.groupby('pid'):
            pre, post = g[g.session == 'pre'].sort_values('t0'), g[g.session == 'post']
            if pre.y.nunique() < 2 or post.y.nunique() < 2:
                continue
            a, t = anchors(pre, 6); others = d[d.pid != pid]
            r = dict(features=fname, pid=pid)
            for name, (Xtr, ytr, te) in {'same-session personal': (a[fs], a.y, t), 'cross-session personal': (a[fs], a.y, post),
                                         'population, other day': (others[fs], others.y, post)}.items():
                try:
                    r[name] = auc(te.y, score(Xtr, ytr, te[fs]))
                except ValueError:
                    r[name] = np.nan
            rows.append(r)
    r = pd.DataFrame(rows); r.to_csv(f'{OUT}/r7_cogwear_long.csv', index=False)
    out = []
    for fname, g in r.groupby('features'):
        g = g.dropna(subset=['same-session personal', 'cross-session personal'])
        for c in ('same-session personal', 'cross-session personal', 'population, other day'):
            out.append(dict(features=fname, protocol=c, mean=g[c].mean(), ci=M.boot_ci(g[c].dropna().values), n=g[c].notna().sum()))
    o = pd.DataFrame(out); o.to_csv(f'{OUT}/r7_cogwear_summary.csv', index=False)
    return o


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
    for st in sys.argv[1:]:
        res = globals()[f'stage_{st}']()
        for x in (res if isinstance(res, tuple) else (res,)):
            print(x.round(3).to_string(index=False), flush=True)
