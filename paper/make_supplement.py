"""Supplementary Materials (Tables S1-S7) built from result files; markdown -> docx with pandoc."""
import os, subprocess, numpy as np, pandas as pd
R = '/path/to/project/data/results'; R3 = f'{R}/r3'; W = '/path/to/project/data/windows'; OUT = '/path/to/project/paper'


def md(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == 'f':
            df[c] = df[c].map(lambda v: '' if pd.isna(v) else f'{v:.{floatfmt}f}')
    return df.to_markdown(index=False)


parts = ['---\ntitle: "Supplementary Materials"\nsubtitle: "Within-Recording Evaluation Inflates Personalised Wearable Detection of Workload and Stress: Block Identity, Drift and Simple Controls"\n---\n']

parts.append('## Table S1. Deviations from the registered plan\n')
dev = [
    ('Verdict rule', 'All', 'Comparative claims: one-sided tests (Wilcoxon or mixed model), p < 0.05; threshold claims: CI rule.',
     'Adopted after the results were known: one CI rule for every claim (Section 3.4). Under the registered tests: P1c-ii not supported (p = 0.086), P3b not supported (Holm p = 0.31), P3d not supported (p = 0.39), P2c supported on both sensor sets (p = 0.004 chest, 0.025 wrist). Under the CI rule these four are inconclusive (P2c because the wrist interval includes 0).'),
    ('Counting', 'WESAD, SenseCobot', 'P2c and P5a were each counted once per sensor set; P5b-ii (fewer usable E4 windows) was not counted.',
     'Each registered claim counts once; multi-sensor claims are confirmed only if confirmed on every evaluable set. P5b-ii is counted (low risk).'),
    ('Holm correction within dataset', 'All', 'Registered for each dataset; applied only to ds007554.', 'Superseded by the CI rule; ds007554 Holm-adjusted p-values are reported for reference.'),
    ('P3a model', 'ds007554', 'Block model with repetition (1-3) and order position.', 'Fitted with repetition only; order position was not modelled.'),
    ('Normalisation', 'ds007554, CogWear', 'Rest z-scores for all features.', 'ds007554 has no rest recording, so heart rate is in bpm; CogWear (P4) used session z-scores because rest followed the Stroop block.'),
    ('P4a, watch arm', 'CogWear', 'Same-session vs. cross-session personal AUROC for both devices.', 'No participant had usable Galaxy Watch4 data in both sessions; the claim was evaluated on the E4 only.'),
    ('P5b sensor', 'SenseCobot', 'E4 BVP heart rate vs. chest-ECG heart rate.', 'BVP is released at 1 Hz; the E4 beat-interval export was used. It had no usable task windows, so the ratio (P5b-i) was not testable.'),
    ('P5c block', 'SenseCobot', 'Baseline or the longest single practice block.', 'Tasks 4 and 5 (the two longest blocks); the baseline consists of three 60 s segments.'),
    ('Chest-ECG timing and quality check', 'SenseCobot', 'Not foreseen.', 'After review, each file\'s sampling rate was derived from its timestamps and a chest-vs-wrist heart-rate check excluded 11 of 20 participants\' chest cardiac features (Table S3). P5a, P5c-ii and P5c-iii were recomputed; P5c-ii changed from inconclusive to confirmed.'),
    ('MAUS position-balanced contrasts, missingness-only model', 'MAUS, UNIVERSE, WESAD, CogWear', 'Not registered.',
     'Exploratory, specified after the other analyses following an internal review (Table S11).'),
    ('Hard vs. easy without slow signals, calibration budget, score–time test, MAUS block-pair validation, bootstrap intervals', 'Development, MAUS, CogWear', 'Not registered.',
     'Exploratory, specified after the other analyses following an internal review (Tables S6 and S10).'),
    ('Shared-direction test of the split-half null', 'Development and held-out', 'Not registered.',
     'Exploratory, specified after the other analyses: population null and t-test of the half difference (Table S9).'),
    ('Offline decision layer', 'MultiPhysio-HRC, SenseCobot', 'Exploratory, in outline: false- and missed-trigger rates of policies based on engagement, graded load and periodically refreshed baselines.', 'Not reported: a version specified after the other analyses was run but reused test windows as reference windows, so it is not included.'),
]
parts.append(pd.DataFrame(dev, columns=['Item', 'Data', 'Registered', 'Done']).to_markdown(index=False) + '\n')
parts.append('\nTimeline of the claims file: it was written on 2 November 2026 while WESAD, ds007554 and CogWear were downloading; the CogWear archive and the ds007554 metadata and rating files were already on disk when it was last saved, and MAUS and WESAD arrived afterwards. The file is fingerprinted (SHA-256, back matter of the paper) but not externally time-stamped.\n')


v = pd.read_csv(f'{R3}/prereg_verdicts.csv')
orig = {'P1c-ii': 'not confirmed (p = 0.086)', 'P3b': 'not confirmed (Holm p = 0.31)', 'P3d': 'not confirmed (p = 0.39)', 'P2c': 'confirmed (p = 0.004 chest, 0.025 wrist)',
        'P2b': 'confirmed', 'P5b-ii': 'not counted', 'P5c-ii': 'inconclusive before the timing correction'}
v['verdict_registered_tests'] = [orig.get(i, r) for i, r in zip(v.id, v.verdict)]
v['in_paper'] = ['no (SenseCobot excluded, see text)' if i.startswith('P5') else 'yes' for i in v.id]
parts.append('\nAll 23 claims written down before the held-out analyses (15 registered statements), with the verdict under the single CI rule used in the paper and under the originally specified tests.\n\n' + v[['id', 'claim', 'data', 'result', 'verdict', 'verdict_registered_tests', 'low_risk', 'in_paper']].to_markdown(index=False) + '\n')
parts.append('\nDatasets analysed during the project but not used in the paper: CogLoad (1 Hz beat-interval samples rather than a waveform; used during development only as an example of a fixed condition order) and SenseCobot (registered by amendment; the released chest-ECG files of 11 of 20 participants hold 546–585 samples per second of timestamp span instead of the documented 512 Hz, and their heart rate remained 0.55–0.71 times that of two wrist devices after the rate was corrected, see Table S3; the wrist waveform is released at 1 Hz). Their registered results are listed above for completeness.\n')
parts.append('\n## Table S2. Random-forest robustness check (100 trees, minimum leaf size 2)\n')
gn = pd.read_csv(f'{R3}/gapnull_rf_summary.csv', dtype={'gap': str})
gn = gn.pivot_table(index=['dataset', 'group'], columns='gap', values='mean').reset_index().rename(columns={'orig': 'null, overlap allowed', '60': 'null, 1-min gap'})
parts.append('Split-half null (k = 6), mean within-person AUROC:\n\n' + md(gn) + '\n')
if os.path.exists(f'{R3}/gapreal_rf_summary.csv'):
    gr = pd.read_csv(f'{R3}/gapreal_rf_summary.csv', dtype={'gap': str})
    lr = pd.read_csv(f'{R3}/gapreal_summary.csv', dtype={'gap': str})
    m = gr.merge(lr, on=['dataset', 'protocol', 'gap', 'rep'], suffixes=('_rf', '_logistic'))
    m = m[m.gap.isin(['none', 'orig', '60'])][['dataset', 'protocol', 'gap', 'rep', 'mean_rf', 'mean_logistic', 'count_rf']]
    parts.append('\nHard vs. easy, mean within-person AUROC (random forest vs. logistic regression):\n\n' + md(m) + '\n')

parts.append('\n## Table S3. SenseCobot chest ECG: effective sampling rate and quality check by participant\n')
L = pd.read_parquet(f'{W}/SENSECOBOT_LAB_windows.parquet')
qc = pd.read_csv(f'{W}/SENSECOBOT_ecg_qc.csv', dtype={'pid': str}).set_index('pid')
t = L.groupby('pid').agg(fs_median=('fs_ecg', 'median'))
L['usable_raw'] = L.ibi_valid_raw.ge(0.8)
u = L.groupby(['pid', 'family']).usable_raw.mean().unstack()
t = t.join(qc).join(u.rename(columns={'rest': 'usable_rest', 'task': 'usable_task'})).reset_index()
t['pass'] = t['pass'].map({True: 'yes', False: 'no'})
parts.append('Ratio = median chest heart rate / median wrist reference (E4 heart-rate column, Bangle.js with confidence >= 80), at rest where available. Usable shares before exclusion.\n\n' + md(t, 2) + '\n')

parts.append('\n## Table S4. Share of windows with usable beats at thresholds from 50% to 90% valid intervals\n')
us = pd.read_csv(f'{R3}/usable.csv')
p = us.pivot_table(index=['device', 'cond'], columns='thr', values='pooled').reset_index()
p.columns = [str(c) for c in p.columns]
n = us[us.thr == 0.8].set_index(['device', 'cond'])[['n_people', 'n_windows', 'share_no_data']]
p = p.merge(n.reset_index(), on=['device', 'cond'])
parts.append(md(p, 3) + '\n')

parts.append('\n## Table S5. Wrist heart-rate agreement with a simultaneous reference\n')
ag = pd.read_csv(f'{R3}/agree.csv')
parts.append('Bias and errors in bpm; within5 = share of windows within 5 bpm; loa = 95% limits of agreement.\n\n' + md(ag, 2) + '\n')

parts.append('\n## Table S6. Formal tests: first run with session-wise normalisation of the engagement contrasts\n')
w = pd.read_csv(f'{R}/confirm/W_confirm.csv')
parts.append(md(w[['test', 'dataset', 'feature', 'term', 'estimate', 'ci_lo', 'ci_hi', 'p', 'p_holm']], 4) + '\n')
w3 = pd.read_csv(f'{R3}/W_confirm_r3.csv')
tost = w3[w3.term.astype(str).str.contains('TOST')]
parts.append('\nThe two equivalence tests among the 12 formal tests of Table 3 (rest-SD units; bounds ±0.3). With the standard errors obtained, equivalence could not have been established; they are not interpreted.\n\n' + md(tost[[c for c in ['test', 'dataset', 'feature', 'term', 'estimate', 'ci_lo', 'ci_hi', 'p', 'p_holm'] if c in tost.columns]], 4) + '\n')
bc = pd.read_csv(f'{R3}/r7_bpm_bootci.csv')
parts.append('\nHeart-rate effects of Figure 2 in bpm: Wald intervals from the mixed models and person-level cluster-bootstrap intervals (people resampled with replacement, model refitted; 1000 resamples).\n\n' + md(bc, 2) + '\n')
rb = pd.read_csv(f'{R3}/robustness.csv')
parts.append('\nRobustness of the difficulty and drift terms:\n\n' + md(rb.drop(columns=[c for c in ['converged'] if c in rb]), 4) + '\n')

parts.append('\n## Table S7. Split-half null by anchor budget k and training-test gap (logistic regression)\n')
gs = pd.read_csv(f'{R3}/gapnull_summary.csv', dtype={'gap': str})
gs['cell'] = gs['mean'].map(lambda v: f'{v:.3f}') + ' (' + gs['count'].astype(int).astype(str) + ')'
g = gs.pivot_table(index=['dataset', 'group', 'k'], columns='gap', values='cell', aggfunc='first').reset_index()
g = g[[c for c in ['dataset', 'group', 'k', 'orig', '0', '30', '60', '120', '300'] if c in g.columns]]
parts.append('Mean within-person AUROC (number of participants). "orig" allows overlap between training and test windows; numbers are gaps in seconds.\n\n' + g.fillna('').to_markdown(index=False) + '\n')

parts.append('\n## Table S8. Same-condition two-block control\n')
sl = pd.read_csv(f'{R3}/r5_samelevel_ds007554_summary.csv')
parts.append('ds007554, chest-ECG features: the same task in two repetitions of one visit (apart = 1 for adjacent, 2 for first vs. third); gap: "orig" = same rule as the same-session protocol, 60 = 1-min gap.\n\n' + md(sl, 3) + '\n')
ap = pd.read_csv(f'{R3}/samelevel_summary.csv'); ap = ap[ap.dataset != 'ds007554']
parts.append('\nApproximate same-condition pairs (training windows may be adjacent to test windows): MultiPhysio-HRC day 2, two blocks of the same work mode (different battery models); WESAD, the two meditation blocks (after different preceding conditions).\n\n' + md(ap, 3) + '\n')

parts.append('\n## Table S9. Is the slow structure behind the split-half null shared across people?\n')
parts.append('Removing a linear trend within each block before the split-half null is not a valid check (it gives below-chance AUROC for uncorrelated noise), so we asked instead whether the change between the two halves of a block has a direction shared across people. '
             'Population null: the test windows of the personal null (k = 6) are scored along the mean first-to-second-half change of other people\'s blocks of the same task (features z-scored within each block); AUROC per block, averaged per person. '
             'Inference by person-level sign-flip permutation (each person\'s two halves swapped as a whole with probability 0.5; 2000 permutations; one-sided). '
             't-test: one-sample t-test of the per-person mean half difference (second minus first half, within-block SD units) of one primary feature per signal.\n')
sim = pd.read_csv(f'{R3}/r6_sim.csv')
sim.columns = ['Simulated signal', 'Personal null', 'Population null', 'Rejection rate, permutation test', 'Rejection rate, t-test', 'Replications']
parts.append('\n(a) Simulation with the same windows (10 s samples averaged into 30 s windows with a 10 s step), protocol (k = 6) and code: 20 people, 4 blocks of 7 min each. '
             'AR(1): coefficient 0.98 per 10 s, unit variance, stationary start. Shared drift: 0.03 SD per 10 s (about 1.2 SD over a block), slope N(0.03, 0.01) per person. '
             'Person-specific trend: slope N(0, 0.03) per block. Personal null from the first 25 replications; 500 permutations per replication; nominal level 0.05.\n\n' + md(sim, 3) + '\n')
dd = pd.read_csv(f'{R3}/r6_drift_summary.csv')
lg = pd.read_csv(f'{R3}/r6_drift_long.csv')
npers = lg.dropna(subset=['personal']).groupby(['dataset', 'group']).pid.nunique().rename('n_personal')
dd = dd.merge(npers.reset_index(), on=['dataset', 'group'], how='left')
lab = {'COGBCI': 'COG-BCI (chest)', 'UNIVERSE': 'UNIVERSE (wrist)', 'MPHRC': 'MultiPhysio-HRC (task blocks, both days)', 'WESAD_CHEST': 'WESAD chest (rest blocks)',
       'WESAD_WRIST': 'WESAD wrist (rest blocks)', 'MAUS_WRIST': 'MAUS wrist PPG (n-back blocks)'}
dd['dataset'] = dd.dataset.map(lab)
dd['n'] = dd.n.astype(int).astype(str) + np.where(dd.n_personal < dd.n, ' (' + dd.n_personal.fillna(0).astype(int).astype(str) + ')', '')
fmtci = lambda v, k=3: ' to '.join(f'{float(x):.{k}f}' for x in str(v).strip('[]').split(', '))
dd['personal'] = dd.personal.map('{:.3f}'.format) + ' (' + dd.personal_ci.map(fmtci) + ')'
dd['population'] = dd.population.map('{:.3f}'.format) + ' (' + dd.population_ci.map(fmtci) + ')'
dd['p_population'] = dd.p_population.map(lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}')
dd['mean_diff_sd'] = dd.mean_diff_sd.map('{:+.2f}'.format) + ' (' + dd.diff_ci.map(lambda v: fmtci(v, 2)) + ')'
dd['group'] = dd.group.map({'cardiac': 'cardiac', 'eda': 'EDA', 'temp': 'skin temperature'})
dd['feature'] = dd.feature.map({'hr_mean': 'heart rate', 'scl_mean': 'SCL', 'temp_mean': 'skin temperature'})
dd['share_increase'] = (100 * dd.share_increase).round(0).astype(int).astype(str) + '%'
dd['p_sign'] = dd.p_sign.map(lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}')
out = dd[['dataset', 'group', 'n', 'personal', 'population', 'p_population', 'perm_null_95', 'feature', 'mean_diff_sd', 'share_increase', 'p_sign']]
out.columns = ['Data', 'Signal', 'n (personal)', 'Personal null (95% CI)', 'Population null (95% CI)', 'Permutation p', 'Permutation 95th percentile',
               'Primary feature', 'Half difference, SD (95% CI)', 'Share increasing', 't-test p']
pw = f'{R3}/r7_power.csv'
if os.path.exists(pw):
    pt = pd.read_csv(pw)[['kind', 'slope', 'drift_over_block_sd', 'population_null', 'power_permutation', 'power_t', 'reps']]
    pt.columns = ['Simulated drift', 'Slope (SD per 10 s)', 'Drift over a 7-min block (SD)', 'Population null', 'Power, permutation test', 'Power, t-test', 'Replications']
    parts.append('\n(b) Power for smaller and mixed drifts (white noise plus a linear trend; 20 people, 4 blocks). Shared drift: slope N(s, s/3) per person. Half the people drift: the same for every second person, none for the others. Shared + person-specific: slope N(s, 0.03) per block, so the sign varies across people when s is small.\n\n' + md(pt, 3) + '\n')
parts.append('\n(c) Data. p-values are descriptive and not corrected for multiplicity. n = participants (in brackets, participants with enough non-missing training windows for the personal model). The 95th percentile of the permutation distribution shows the population-null value needed for p = 0.05.\n\n' + md(out, 3) + '\n')

parts.append('\n## Table S10. Hard versus easy: features, calibration budget, regularisation, score–time test and block-pair validation\n')
pr = pd.read_csv(f'{R3}/r7_protocols_summary.csv')
pr['k'] = pr.k.map(lambda v: '' if pd.isna(v) else str(int(v)))
pr['AUROC (95% CI)'] = pr['mean'].map('{:.3f}'.format) + ' (' + pr.ci.map(lambda v: ' to '.join(f'{float(x):.3f}' for x in str(v).strip('[]').split(', '))) + ')'
pr.loc[pr.dataset == 'COGBCI', 'features'] = 'cardiac (all available)'
pr = pr.rename(columns={'dataset': 'Data', 'features': 'Features', 'protocol': 'Protocol', 'n': 'n'})
parts.append('(a) Within-person AUROC, hard vs. easy (logistic regression; k = training windows per condition; C = inverse regularisation). Cross-session personal models use the same training windows as the same-session protocol. COG-BCI features are cardiac only. With k = 24, few COG-BCI blocks leave test windows after a 1-min gap.\n\n' + pr[['Data', 'Features', 'Protocol', 'k', 'C', 'AUROC (95% CI)', 'n']].to_markdown(index=False) + '\n')
st = pd.read_csv(f'{R3}/r7_scoretime_summary.csv')
st.loc[st.dataset == 'COGBCI', 'features'] = 'cardiac (all available)'
st.columns = ['Data', 'Features', 'Protocol', 'Mean r', '95% CI', 'Share positive', 'Wilcoxon p', 'n']
parts.append('\n(b) Score–time test: Spearman correlation of the physiological decision score with time inside each test block, within each condition, signed so that positive values point towards the condition recorded later (k = 6, C = 1). A time-only model gives +1.\n\n' + md(st, 3) + '\n')
mi = pd.read_csv(f'{R3}/r7_maus_inflation.csv')
cbw = pd.read_csv(f'{R3}/crossblk_summary.csv'); cbw = cbw[(cbw.dataset == 'MAUS_ECG') & (cbw.rep == 'physiology') & (cbw.protocol == 'train blocks 1-3, test 4-6')]
parts.append('\n(c) MAUS: same training windows (first 6 windows of the first 0- and 3-back blocks) tested on later windows of those blocks (same session) or on the second 0- and 3-back blocks (cross-block). Inflation = same-session minus cross-block, per person; Wilcoxon p two-sided. Sensitivity: training on all windows of the first blocks instead of k = 6 gave {:.3f} (95% CI {}) for the ECG model on the later blocks.\n\n'.format(cbw['mean'].iloc[0], cbw.ci.iloc[0]) + md(mi, 3) + '\n')
lb = pd.read_csv(f'{R3}/r7_maus_lobpo_summary.csv')
lb['folds'] = lb.folds.map({'halves': 'first pair vs. second pair', 'crossed': 'crossed pairs (training blocks 1st and 4th)', 'all four': 'all four folds'})
parts.append('\n(d) MAUS leave-one-block-pair-out over the four (0-back, 3-back) block pairs; all windows of the two training blocks. In every fold the order of the levels is reversed between training and test, so the clock scores 0.\n\n' + md(lb, 3) + '\n')
cw = pd.read_csv(f'{R3}/r7_cogwear_summary.csv')
parts.append('\n(e) CogWear (wrist E4), Stroop vs. baseline, with all features and cardiac features only.\n\n' + md(cw, 3) + '\n')

parts.append('\n## Table S11. Checks on the MAUS difficulty effect and on missingness\n')
mc = pd.read_csv(f'{R3}/r9_maus_check.csv')
mc['p'] = mc.p.map(lambda v: '' if pd.isna(v) else ('< 0.001' if v < 0.001 else f'{v:.3f}'))
parts.append('(a) MAUS torso-ECG heart rate (bpm), block means, ECG windows with at least 80% valid beat intervals. The palindrome (0-2-3-2-3-0) places both 0-back blocks at the ends of the session. A linear drift makes the two end differences opposite in sign; a load effect makes both positive. Mixed models with a random intercept per person (Wald intervals); differences with person-level bootstrap intervals and two-sided Wilcoxon tests.\n\n' + md(mc, 2) + '\n')
mg = pd.read_csv(f'{R3}/r9_missingness.csv')
parts.append('\n(b) Missingness-only model: logistic regression whose single feature is the share of valid beat intervals in each 30 s window (no physiological features). Same-session personal: first 6 windows per condition for training, later windows of the same blocks for testing; population: leave one subject out. MAUS: first 0- and 3-back blocks; CogWear: first session.\n\n' + md(mg, 3) + '\n')

open(f'{OUT}/Supplementary_Materials.md', 'w').write('\n'.join(parts))
subprocess.run(['pandoc', f'{OUT}/Supplementary_Materials.md', '-o', f'{OUT}/Supplementary_Materials.docx'], check=True)
print('written')
