"""R6: a valid test of whether the split-half null reflects a change SHARED across people (systematic drift).

The split-half null (personal model separating the two halves of one constant block) exceeds 0.5 for systematic drift,
for person-specific trends and for stationary but slowly autocorrelated signals alike. Two quantities isolate the part
that is shared across people:

  population null   direction learned from OTHER people doing the same task: the mean change from the first to the second
                    half of their blocks (features z-scored within each block, no labels beyond time). This person's test
                    windows (the same windows as the personal null) are scored along that direction; AUROC per block.
                    Inference: person-level sign-flip permutation (each person's halves swapped as a whole with p = 0.5),
                    which is exact under the null that each person's direction of change is symmetric about zero
                    (true for any stationary, time-reversible process, and for trends whose sign varies across people).
                    The permutation also absorbs the negative bias of leave-one-person-out estimation.
  sign test         per person, the mean half difference (second minus first half, within-block z units) of one primary
                    feature per signal; one-sample t-test across people.

Validity is checked by simulation with the same window structure (10 s samples, 30 s windows, 10 s step), protocol
(k = 6 anchors) and code: white noise, stationary AR(1), shared linear drift, AR(1) + shared drift, and person-specific
trends of random sign.
Outputs (results/r3/): r6_sim_long.csv, r6_sim.csv, r6_drift_long.csv, r6_drift_summary.csv, r6_sign_tests.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E
import mech as M
import prereg_r2 as P

OUT = os.path.join(E.OUT, 'r3')
K = 6


def split(g):
    g = g.sort_values('t0'); mid = g.t0.iloc[len(g) // 2]
    return g.assign(y=(g.t0 >= mid).astype(int))


def anchors_tests(g):
    anchor = pd.concat([h.head(K) for _, h in g.groupby('y')])
    last = anchor.groupby('y').t1.max()
    test = g[(g.t0 >= g.y.map(last)) & ~g.index.isin(anchor.index)]
    return anchor, test


def blockz(g, fs):
    z = g[fs].astype(float)
    sd = z.std().replace(0, np.nan)
    return ((z - z.mean()) / sd).fillna(0.0)


def auc_cols(scores, y):
    """AUROC of every column of scores (n x P) against binary y, average ranks for ties."""
    r = stats.rankdata(scores, axis=0)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def personal_auc(anchor, test, fs):
    if anchor[fs].notna().any(axis=1).sum() < K:
        return np.nan
    m = E.model(); m.fit(anchor[fs], anchor.y)
    try:
        return roc_auc_score(test.y, m.decision_function(test[fs]))
    except ValueError:
        return np.nan


def nulls(df, groups, n_perm=2000, seed=0, personal=True):
    """df: windows with pid, block, family, t0, t1, features (one constant condition per block).
    Returns (block-level rows, person-level sign-flip distribution per group)."""
    rng = np.random.default_rng(seed)
    allf = sorted({f for fs in groups.values() for f in fs})
    blocks = {}
    for b, g in df.groupby('block'):
        g = split(g)
        if g.y.nunique() < 2:
            continue
        z = blockz(g, allf)
        d = z[g.y.values == 1].mean() - z[g.y.values == 0].mean()
        anchor, test = anchors_tests(g)
        ok = test.y.nunique() == 2 and test.y.value_counts().min() >= 3
        blocks[b] = dict(pid=str(g.pid.iloc[0]), fam=g.family.iloc[0], g=g, z=z, d=d.fillna(0.0), anchor=anchor, test=test, ok=ok)
    pids = sorted({v['pid'] for v in blocks.values()})
    pix = {p: i for i, p in enumerate(pids)}
    S = rng.choice([-1.0, 1.0], size=(n_perm + 1, len(pids))); S[0] = 1.0
    # person x family direction = mean half difference over that person's blocks of the family
    D = {}
    for v in blocks.values():
        D.setdefault((v['pid'], v['fam']), []).append(v['d'])
    D = {k: pd.concat(v, axis=1).mean(axis=1) for k, v in D.items()}
    rows, dist = [], {}
    for gname, fs in groups.items():
        per_person = {}
        for b, v in blocks.items():
            if not v['ok']:
                continue
            others = [(p, f) for (p, f) in D if f == v['fam'] and p != v['pid']]
            if len(others) < 3:
                continue
            Dm = np.array([D[o][fs].values for o in others])                     # n_others x nf
            W = (S[:, [pix[p] for p, _ in others]] @ Dm) / len(others)            # (P+1) x nf
            W = W * S[:, [pix[v['pid']]]]                                          # flipping this person's own halves
            tz = v['z'].loc[v['test'].index, fs].values
            if not np.isfinite(v['test'][fs].values).any():
                continue
            a = auc_cols(tz @ W.T, v['test'].y.values)
            per_person.setdefault(v['pid'], []).append(a)
            rows.append(dict(block=b, pid=v['pid'], family=v['fam'], group=gname, population=a[0],
                             personal=personal_auc(v['anchor'], v['test'], fs) if personal else np.nan,
                             **{f'diff_{f}': v['d'][f] for f in fs}))
        if per_person:
            pp = np.array([np.mean(v, axis=0) for v in per_person.values()])   # people x (P+1)
            dist[gname] = pp.mean(axis=0)
    return pd.DataFrame(rows), dist


def perm_p(dist):
    return (1 + (dist[1:] >= dist[0]).sum()) / len(dist)


def sign_test(r, f):
    pp = r.groupby('pid')[f'diff_{f}'].mean().dropna()
    return pp, stats.ttest_1samp(pp, 0).pvalue


# ------------------------------------------------------------------ simulation (validity check)
def sim_dataset(kind, rng, n_people=20, n_blocks=4, n_samples=42, phi=0.98, slope=0.03, slope_sd=0.01):
    rows = []
    for p in range(n_people):
        sl = rng.normal(slope, slope_sd) if kind != 'person-specific trend' else 0.0
        for b in range(n_blocks):
            e = rng.normal(size=n_samples)
            if kind in ('stationary AR(1)', 'AR(1) + shared drift'):
                x = np.empty(n_samples); x[0] = e[0]
                for t in range(1, n_samples):
                    x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * e[t]   # unit marginal variance, stationary start
            else:
                x = e
            if kind in ('shared drift', 'AR(1) + shared drift'):
                x = x + sl * np.arange(n_samples)
            if kind == 'person-specific trend':
                x = x + rng.normal(0, slope) * np.arange(n_samples)          # sign varies across people and blocks
            w = np.convolve(x, np.ones(3) / 3, mode='valid')                  # 30 s windows, 10 s step
            for i, v in enumerate(w):
                rows.append(dict(pid=f'p{p:02d}', block=f'p{p:02d}|b{b}', family='sim', t0=10.0 * i, t1=10.0 * i + 30, f1=v))
    return pd.DataFrame(rows)


KINDS = ('white noise', 'stationary AR(1)', 'person-specific trend', 'shared drift', 'AR(1) + shared drift')


def simulate(n_rep=200, n_personal=25, seed=1):
    rng = np.random.default_rng(seed); out = []
    for kind in KINDS:
        for rep in range(n_rep):
            d = sim_dataset(kind, rng)
            r, dist = nulls(d, {'f1': ['f1']}, n_perm=500, seed=int(rng.integers(1 << 31)), personal=rep < n_personal)
            pp, p_sign = sign_test(r, 'f1')
            out.append(dict(kind=kind, rep=rep, personal=r.groupby('pid').personal.mean().mean(),
                            population=dist['f1'][0], p_population=perm_p(dist['f1']), p_sign=p_sign))
        print(kind, 'done', flush=True)
    s = pd.DataFrame(out); s.to_csv(f'{OUT}/r6_sim_long.csv', index=False)
    t = s.groupby('kind', sort=False).agg(personal_null=('personal', 'mean'), population_null=('population', 'mean'),
                                          reject_population=('p_population', lambda p: (p < .05).mean()),
                                          reject_sign=('p_sign', lambda p: (p < .05).mean()), reps=('rep', 'count')).reset_index()
    t.to_csv(f'{OUT}/r6_sim.csv', index=False)
    return t


# ------------------------------------------------------------------ data
def data_sets():
    out = []
    for ds in ('COGBCI', 'UNIVERSE', 'MPHRC'):
        df, feats = E.load(ds)
        df = E.normalise(df.reset_index(drop=True), feats, 'rest')
        df = df[df.level >= 0]
        df = df.reset_index(drop=True)
        df['block'] = df.unit + '|' + df.task.astype(str)
        out.append((ds, df, feats))
    for name, fam in (('WESAD_CHEST', 'rest'), ('WESAD_WRIST', 'rest'), ('MAUS_WRIST', 'nback')):
        df, feats = P.load(name, 'none')
        df = df[df.family == fam].reset_index(drop=True)
        out.append((name, df, feats))
    return out


PRIMARY = {'cardiac': 'hr_mean', 'eda': 'scl_mean', 'temp': 'temp_mean'}


def run_data():
    longs, summ, signs = [], [], []
    for name, df, feats in data_sets():
        groups = {g: [f for f in P.GROUPS[g] if f in feats] for g in ('cardiac', 'eda', 'temp')}
        groups = {g: f for g, f in groups.items() if f}
        r, dist = nulls(df, groups, n_perm=2000, seed=6)
        r['dataset'] = name; longs.append(r)
        for gname in groups:
            g = r[r.group == gname]
            pp = g.groupby('pid')[['personal', 'population']].mean()
            row = dict(dataset=name, group=gname, n=len(pp), personal=pp.personal.mean(), personal_ci=M.boot_ci(pp.personal.values),
                       population=dist[gname][0], population_ci=M.boot_ci(pp.population.values), p_population=perm_p(dist[gname]),
                       perm_null_95=float(np.percentile(dist[gname][1:], 95)))
            f = PRIMARY[gname]
            if f in groups[gname]:
                d, p = sign_test(g, f)
                row.update(feature=f, mean_diff_sd=d.mean(), diff_ci=M.boot_ci(d.values), share_increase=(d > 0).mean(), p_sign=p)
            summ.append(row)
        print(name, 'done', flush=True)
    pd.concat(longs).to_csv(f'{OUT}/r6_drift_long.csv', index=False)
    s = pd.DataFrame(summ); s.to_csv(f'{OUT}/r6_drift_summary.csv', index=False)
    return s


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    what = sys.argv[1:] or ['sim', 'data']
    if 'data' in what:
        print(run_data().round(3).to_string(index=False), flush=True)
    if 'sim' in what:
        print(simulate().round(3).to_string(index=False))
