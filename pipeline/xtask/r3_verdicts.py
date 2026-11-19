"""R3: re-judge every registered claim with ONE rule.

Rule: the 95% CI of the registered quantity is compared with its registered threshold (0 for a directional
difference). confirmed = whole CI on the predicted side; refuted = whole CI on the other side; else inconclusive.
Counting: each registered threshold claim counts once. A claim evaluated on two sensor sets counts once and is
confirmed only if confirmed on every sensor set that could be evaluated. P2b holds by construction (baseline always
first) and is listed as a design check, not counted. Claims that follow from known device or design properties are
flagged low_risk. Output: results/r3/prereg_verdicts.csv
"""
import os, ast
import numpy as np, pandas as pd
import exp1_transfer as E

RES = os.path.join(E.OUT, 'prereg_r2'); OUT = os.path.join(E.OUT, 'r3'); os.makedirs(OUT, exist_ok=True)


def tab(name):
    t = pd.read_csv(f'{RES}/{name}_prereg.csv')
    t['ci'] = t.ci.apply(lambda s: [float(x) for x in str(s).strip('[]()').split(',')] if str(s)[:1] in '[(' and 'nan' not in str(s) else [np.nan, np.nan])
    return t


def judge(ci, side, thr):
    lo, hi = ci
    if not np.isfinite(lo):
        return 'not testable'
    if side == '>':
        return 'confirmed' if lo > thr else ('refuted' if hi < thr else 'inconclusive')
    return 'confirmed' if hi < thr else ('refuted' if lo > thr else 'inconclusive')


ma, we, ds, cw, sc = tab('MAUS'), tab('WESAD'), tab('DS007554'), tab('COGWEAR'), tab('SENSECOBOT')
row = lambda t, i: (t.loc[i, 'estimate'], t.loc[i, 'ci'], t.loc[i, 'n'])
claims = []
def add(cid, text, data, parts, side, thr, low_risk=False, design=False, note=''):
    """parts: list of (label, estimate, ci, n) for each sensor set."""
    vs = [judge(ci, side, thr) for _, _, ci, _ in parts]
    testable = [v for v in vs if v != 'not testable']
    if design:
        v = 'design check'
    elif not testable:
        v = 'not testable'
    elif all(x == 'confirmed' for x in testable):
        v = 'confirmed'
    elif all(x == 'refuted' for x in testable):
        v = 'refuted'
    else:
        v = 'inconclusive'
    res = '; '.join(f'{lab}{" " if lab else ""}{e:.4g} ({ci[0]:.4g}, {ci[1]:.4g}), n = {n:.0f}' if np.isfinite(ci[0]) else f'{lab} not testable (n = {n:.0f})'
                    for lab, e, ci, n in parts)
    claims.append(dict(id=cid, claim=text, data=data, result=res, threshold=f'{side} {thr}', verdict=v,
                       per_sensor='; '.join(vs), low_risk=low_risk, note=note))

add('P1a', 'Torso-ECG heart rate higher in 3-back than 0-back (rest SD)', 'MAUS', [('', *row(ma, 0))], '>', 0, note='+3.39 bpm (2.13, 4.66)')
add('P1b', 'Wrist/ECG heart-rate effect ratio, lower CI > 0.5', 'MAUS', [('', *row(ma, 1))], '>', 0.5)
add('P1c-i', 'Fewer usable wrist windows than ECG (wrist minus ECG)', 'MAUS', [('', *row(ma, 3))], '<', 0, low_risk=True)
add('P1c-ii', 'Wrist loses more windows in 3-back (hard minus easy)', 'MAUS', [('', *row(ma, 4))], '<', 0)
add('P1d-i', 'Two blocks of one level separable by finger EDA (> 0.6)', 'MAUS', [('', *row(ma, 6))], '>', 0.6)
add('P1d-ii', 'Two blocks of one level not separable by ECG (< 0.6)', 'MAUS', [('', *row(ma, 5))], '<', 0.6)
add('P2a-i', 'Split-half null, chest cardiac < 0.6', 'WESAD', [('', *row(we, 0))], '<', 0.6)
add('P2a-ii', 'Split-half null, chest temperature > 0.6', 'WESAD', [('', *row(we, 2))], '>', 0.6)
add('P2a-iii', 'Split-half null, wrist EDA > 0.6', 'WESAD', [('', *row(we, 6))], '>', 0.6)
add('P2a-iv', 'Split-half null, wrist temperature > 0.6', 'WESAD', [('', *row(we, 7))], '>', 0.6)
add('P2b', 'Clock-only model separates stress from baseline (> 0.9)', 'WESAD', [('', *row(we, 9))], '>', 0.9, design=True,
    note='baseline preceded stress for every participant, so 1.00 follows from the design')
add('P2c', 'Stress-vs-baseline AUROC above stress-vs-amusement', 'WESAD', [('chest', *row(we, 11)), ('wrist', *row(we, 14))], '>', 0)
add('P3a', 'Heart rate: cognitive-motor above motor-only (bpm)', 'ds007554', [('', *row(ds, 0))], '>', 0,
    note='registered model included order position; fitted with repetition only (deviation)')
add('P3b', 'Heart rate: active above passive motor (bpm)', 'ds007554', [('', *row(ds, 1))], '>', 0)
add('P3c', 'Effort rating: cognitive-motor above motor-only (1-9)', 'ds007554', [('', *row(ds, 2))], '>', 0, low_risk=True)
add('P3d', 'Personal AUROC within block above across repetitions', 'ds007554', [('', *row(ds, 3))], '>', 0)
add('P4a', 'Personal AUROC same-session above cross-session', 'CogWear', [('E4', *row(cw, 0)), ('watch', *row(cw, 3))], '>', 0,
    note='Galaxy Watch arm had no person with usable data in both sessions')
add('P5a', 'Programming-vs-rest AUROC above task 1-vs-task 5', 'SenseCobot', [('chest/finger', *row(sc, 0)), ('wrist', *row(sc, 1))], '>', 0)
add('P5b-i', 'Wrist/ECG heart-rate effect ratio, lower CI > 0.5', 'SenseCobot', [('', *row(sc, 2))], '>', 0.5,
    note='registered on E4 BVP; released BVP is 1 Hz, so the E4 beat-interval export was used (deviation); no usable wrist task windows')
usew = pd.read_csv(os.path.join(OUT, 'usable.csv'))
u = usew[(usew.thr == 0.8) & (usew.cond == 'task') & usew.file.isin(['SENSECOBOT_WRIST', 'SENSECOBOT_LAB'])].set_index('file')
wr = pd.read_parquet(os.path.join(os.path.dirname(E.R), 'windows', 'SENSECOBOT_WRIST_windows.parquet'))
lab = pd.read_parquet(os.path.join(os.path.dirname(E.R), 'windows', 'SENSECOBOT_LAB_windows.parquet'))
pw = wr[wr.family == 'task'].ibi_valid.fillna(0).ge(.8).groupby(wr.pid).mean()
pl = lab[(lab.family == 'task') & (lab.ecg_qc_pass == 1)].ibi_valid.ge(.8).groupby(lab.pid).mean()
j = pd.concat([pw.rename('w'), pl.rename('l')], axis=1).dropna(); dlt = (j.w - j.l).values
rng = np.random.default_rng(0); bs = [rng.choice(dlt, len(dlt)).mean() for _ in range(2000)]
add('P5b-ii', 'Fewer usable E4 windows than ECG during programming (wrist minus ECG)', 'SenseCobot',
    [('', dlt.mean(), [np.percentile(bs, 2.5), np.percentile(bs, 97.5)], len(dlt))], '<', 0, low_risk=True,
    note=f'E4 {u.loc["SENSECOBOT_WRIST", "pooled"]:.3f} vs chest {u.loc["SENSECOBOT_LAB", "pooled"]:.3f} (chest: 9 participants passing QC)')
add('P5c-i', 'Split-half null, wrist temperature > 0.6', 'SenseCobot', [('', *row(sc, 3))], '>', 0.6)
add('P5c-ii', 'Split-half null, wrist EDA > 0.6', 'SenseCobot', [('', *row(sc, 4))], '>', 0.6)
add('P5c-iii', 'Split-half null, chest cardiac < 0.6', 'SenseCobot', [('', *row(sc, 6))], '<', 0.6)

T = pd.DataFrame(claims)
T.to_csv(f'{OUT}/prereg_verdicts.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
print(T[['id', 'data', 'result', 'threshold', 'verdict', 'low_risk']].to_string(index=False))
print(T.verdict.value_counts().to_dict(), 'low-risk confirmed:', int((T.low_risk & (T.verdict == 'confirmed')).sum()))
