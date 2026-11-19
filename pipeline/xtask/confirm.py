"""W1 confirmatory statistics for the main claims (pre-registered in PIPELINE.md section 8).

  python confirm.py        -> results/confirm/W_confirm.csv (+ W_secondary.csv)
Linear mixed models (statsmodels MixedLM, REML), random intercept per person. Holm over the 12 primary tests.
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import norm
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import exp1_transfer as E
import mech as M

OUT = os.path.join(E.OUT, 'confirm'); os.makedirs(OUT, exist_ok=True)
SESOI = 0.3  # smallest effect of interest, rest-SD units (TOST bounds)


def fit(formula, data, group='pid'):
    m = smf.mixedlm(formula, data, groups=data[group])
    ysd = float(np.nanstd(data[formula.split('~')[0].strip()]))
    for method in ('lbfgs', 'powell', 'nm'):
        try:
            r = m.fit(reml=True, method=method)
            fe = r.bse[r.fe_params.index]
            if np.isfinite(r.params).all() and np.isfinite(fe).all() and (fe < 10 * ysd + 1e-9).all():
                return r  # reject degenerate fits (singular random-effect covariance -> exploding SEs)
        except Exception:
            continue
    raise RuntimeError(f'MixedLM failed: {formula}')


def coef(r, name):
    b, se = float(r.params[name]), float(r.bse[name])
    return b, se, b - 1.96 * se, b + 1.96 * se


def p_greater(b, se):
    return float(1 - norm.cdf(b / se))


def p_tost(b, se, bound=SESOI):
    return float(max(1 - norm.cdf((b + bound) / se), 1 - norm.cdf((bound - b) / se)))


def blocks(ds):
    df, feats = M.load(ds, 'rest')
    d, fams = M.leveled(df)
    keys = ['pid', 'session', 'unit', 'family', 'task', 'y']
    b = d.groupby(keys)[feats].mean().reset_index()
    if ds in ('COGBCI', 'UNIVERSE'):
        b = b.merge(M.block_times(ds), on=['pid', 'session', 'task'], how='left')
    b['pid'] = b.pid.astype(str)
    return b, feats


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(run, 1.0)
    return adj


def main():
    prim, sec = [], []
    # ---------------- W1 engagement > difficulty
    m3 = pd.read_csv(f'{M.OUT}/M3_contrasts.csv')
    m3 = m3[(~m3.motion_removed) & m3.contrast.isin(['easy_vs_rest', 'hard_vs_easy'])].copy()
    m3['pid'] = m3.pid.astype(str)
    for ds in ['COGBCI', 'UNIVERSE', 'MPHRC']:
        d = m3[m3.dataset == ds]
        fam = ' + C(family)' if d.family.nunique() > 1 else ''
        r = fit(f"auc ~ C(contrast, Treatment('hard_vs_easy'))" + fam, d)
        b, se, lo, hi = coef(r, "C(contrast, Treatment('hard_vs_easy'))[T.easy_vs_rest]")
        prim.append(dict(test='W1', dataset=ds, feature='AUROC', term='easy_vs_rest - hard_vs_easy', estimate=b, se=se, ci_lo=lo, ci_hi=hi,
                         p=p_greater(b, se), kind='one-sided >0', n_people=d.pid.nunique()))
    # ---------------- W2 / W3 block-level models
    for ds in ['COGBCI', 'UNIVERSE', 'MPHRC']:
        b, feats = blocks(ds)
        has_t = 'start' in b and b.start.notna().any()
        fam = ' + C(family)' if b.family.nunique() > 1 else ''
        for f in feats:
            dd = b.dropna(subset=[f] + (['start'] if has_t else []))
            if dd.pid.nunique() < 8:
                continue
            r = fit(f'{f} ~ y' + (' + start' if has_t else '') + fam, dd)
            by, sey, loy, hiy = coef(r, 'y')
            row = dict(dataset=ds, feature=f, estimate=by, se=sey, ci_lo=loy, ci_hi=hiy, n_people=dd.pid.nunique(), n_blocks=len(dd))
            primary_w2 = f == 'hr_mean'
            if ds == 'COGBCI':
                rec = dict(row, test='W2', term='hard (level max vs min)', p=p_greater(by, sey), kind='one-sided >0')
            else:
                rec = dict(row, test='W2', term=f'hard, TOST |b|<{SESOI}', p=p_tost(by, sey), kind='TOST equivalence')
            (prim if primary_w2 else sec).append(rec)
            if has_t:
                bt, set_, lot, hit = coef(r, 'start')
                span = 7.0 if ds == 'COGBCI' else float(dd.groupby('unit').start.max().median())
                rec3 = dict(dataset=ds, feature=f, test='W3', term='time (per position)' if ds == 'COGBCI' else 'time (per minute)',
                            estimate=bt, se=set_, ci_lo=lot, ci_hi=hit, p=float(2 * (1 - norm.cdf(abs(bt / set_)))), kind='two-sided',
                            n_people=dd.pid.nunique(), n_blocks=len(dd), session_span=span, drift_over_session=bt * span, hard_effect=by,
                            drift_exceeds_hard=abs(bt * span) > abs(by))
                primary_w3 = (f == 'hr_mean') or (ds == 'UNIVERSE' and f in ('temp_mean', 'scl_mean'))
                (prim if primary_w3 else sec).append(rec3)
    # MPHRC: does the hard effect depend on whether the person felt the manipulation? (secondary)
    b, feats = blocks('MPHRC')
    ag = pd.read_csv(f'{M.OUT}/M4_agreement.csv'); ag = ag[ag.dataset == 'MPHRC']
    b = b.merge(ag[['unit', 'family', 'subj_diff']], on=['unit', 'family'], how='left')
    b['agree'] = (b.subj_diff > 0).astype(int)
    for f in ['hr_mean', 'rmssd', 'scl_mean', 'resp_rate']:
        dd = b.dropna(subset=[f, 'subj_diff'])
        r = fit(f'{f} ~ y * agree', dd)
        bi, sei, loi, hii = coef(r, 'y:agree')
        sec.append(dict(test='W2b', dataset='MPHRC', feature=f, term='hard x felt-harder', estimate=bi, se=sei, ci_lo=loi, ci_hi=hii,
                        p=float(2 * (1 - norm.cdf(abs(bi / sei)))), kind='two-sided', n_people=dd.pid.nunique()))
    # ---------------- W4a split-half null
    for ds in ['UNIVERSE', 'COGBCI']:
        n = pd.read_csv(f'{E.OUT}/exp4_null_{ds}.csv')
        n = n[(n.k == 6) & (n.cond == 'personal')].copy(); n['excess'] = n.auc - 0.5; n['pid'] = n.pid.astype(str)
        r = fit('excess ~ 1', n)
        b0, se0, lo0, hi0 = coef(r, 'Intercept')
        prim.append(dict(test='W4a', dataset=ds, feature='all HC', term='null AUROC - 0.5 (personal, k=6)', estimate=b0, se=se0, ci_lo=lo0, ci_hi=hi0,
                         p=p_greater(b0, se0), kind='one-sided >0', n_people=n.pid.nunique(), n_blocks=len(n)))
    # ---------------- W4b same-session vs cross-session (exp3, handcrafted, k=6)
    for ds in ['COGBCI', 'UNIVERSE']:
        a = pd.read_csv(f'{E.OUT}/exp3_{ds}_HC.csv'); a = a[(a.source == a.target) & (a.k == 6)].assign(setting='same')
        c = pd.read_csv(f'{E.OUT}/exp3_crosssession_{ds}_HC.csv'); c = c[c.k == 6].assign(setting='cross')
        d = pd.concat([a[['pid', 'cond', 'auc', 'setting']], c[['pid', 'cond', 'auc', 'setting']]])
        d = d[d.cond.isin(['pop', 'personal'])].copy(); d['pid'] = d.pid.astype(str)
        r = fit("auc ~ C(setting, Treatment('same')) * C(cond, Treatment('pop'))", d)
        nm = "C(setting, Treatment('same'))[T.cross]:C(cond, Treatment('pop'))[T.personal]"
        bi, sei, loi, hii = coef(r, nm)
        sec.append(dict(test='W4b', dataset=ds, feature='HC', term='(personal - pop) drop from same- to cross-session', estimate=bi, se=sei,
                        ci_lo=loi, ci_hi=hii, p=float(norm.cdf(bi / sei)), kind='one-sided <0', n_people=d.pid.nunique()))
    P = pd.DataFrame(prim)
    P['p_holm'] = holm(P.p.values)
    P['supported'] = P.p_holm < 0.05
    P.to_csv(f'{OUT}/W_confirm.csv', index=False)
    S = pd.DataFrame(sec); S.to_csv(f'{OUT}/W_secondary.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['test', 'dataset', 'feature', 'term', 'estimate', 'ci_lo', 'ci_hi', 'p', 'p_holm', 'supported']
    print(P[cols + [c for c in ['drift_over_session', 'hard_effect', 'drift_exceeds_hard'] if c in P]].round(4).to_string(index=False))
    print(S[['test', 'dataset', 'feature', 'term', 'estimate', 'ci_lo', 'ci_hi', 'p']].round(4).to_string(index=False))
    M.gate('W1', **{f'{r.test}_{r.dataset}_{r.feature}_supported': str(bool(r.supported)) for r in P.itertuples()})


if __name__ == '__main__':
    main()
