"""Per-person numbers for the manuscript figures (R3). Exploratory re-use of the registered pipelines; no new tests.
Engagement contrasts use rest normalisation (as Section 3.2 specifies); the split-half null uses the R3 run (k = 6,
original anchor rule) where available."""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('DATA_ROOT', '/path/to/project/data/raw')
sys.path.insert(0, '/path/to/project/pipeline/xtask')
import prereg_r2 as P
OUT = '/path/to/project/paper/data'; R3 = '/path/to/project/data/results/r3'
rows = []
def add(ds, contrast, s):
    for pid, v in s.items():
        rows.append(dict(dataset=ds, contrast=contrast, pid=pid, auc=v))
w1 = pd.read_csv(f'{R3}/w1_long.csv'); w1 = w1[w1.norm == 'rest']
for (ds, c), g in w1.groupby(['dataset', 'contrast']):
    add(ds, c, g.groupby('pid').auc.mean())
df, feats = P.load('MAUS_ECG', 'rest')
rest = df[df.family == 'rest']; lo, hi = df[df.level == 0], df[df.level == 3]
for c, (neg, pos) in {'easy_vs_rest': (rest, lo), 'hard_vs_easy': (lo, hi)}.items():
    add('MAUS', c, P.loso_within_auc(pd.concat([neg.assign(y=0), pos.assign(y=1)]), feats))
df, feats = P.load('SENSECOBOT_LAB', 'rest')
add('SENSECOBOT', 'easy_vs_rest', P.loso_within_auc(df[df.family.isin(['rest', 'task'])].assign(y=lambda x: (x.family == 'task').astype(int)), feats))
add('SENSECOBOT', 'hard_vs_easy', P.loso_within_auc(df[df.task.isin(['task1', 'task5'])].assign(y=lambda x: (x.task == 'task5').astype(int)), feats))
pd.DataFrame(rows).to_csv(f'{OUT}/fig_engagement.csv', index=False)

nr = []
g6 = pd.read_csv(f'{R3}/gapnull_long.csv'); g6 = g6[(g6.k == 6) & (g6.gap == 'orig') & (g6.group != 'all')]
lab = {'COGBCI': 'COG-BCI', 'UNIVERSE': 'UNIVERSE', 'WESAD_CHEST': 'WESAD chest', 'WESAD_WRIST': 'WESAD wrist',
       'MAUS_FINGER': 'MAUS finger', 'MAUS_WRIST': 'MAUS wrist', 'SENSECOBOT_WRIST': 'SenseCobot wrist'}
for (ds, grp), g in g6.groupby(['dataset', 'group']):
    for pid, v in g.groupby('pid').auc.mean().items():
        nr.append(dict(dataset=lab[ds], group=grp, pid=pid, auc=v))
m6 = pd.read_csv('/path/to/project/data/results/mech/M6_null_long.csv')
m6 = m6[(m6.k == 6) & (m6.cond == 'personal') & (m6.dataset == 'MPHRC') & (m6.features != 'all')]
for grp, g in m6.groupby('features'):
    for pid, v in g.groupby('pid').auc.mean().items():
        nr.append(dict(dataset='MultiPhysio-HRC', group=grp, pid=pid, auc=v))
df, feats = P.load('SENSECOBOT_LAB', 'none')
for grp in ('cardiac', 'eda'):
    fs = [f for f in P.GROUPS[grp] if f in feats]
    n = P.split_half_null(df, fs, df.task.isin(['task4', 'task5']))
    for pid, v in n.groupby('pid').auc.mean().items():
        nr.append(dict(dataset='SenseCobot chest/finger', group=grp, pid=pid, auc=v))
pd.DataFrame(nr).to_csv(f'{OUT}/fig_null.csv', index=False)
print(pd.DataFrame(rows).groupby(['dataset', 'contrast']).auc.agg(['mean', 'count']).round(3).to_string())
print(pd.DataFrame(nr).groupby(['dataset', 'group']).auc.agg(['mean', 'count']).round(3).to_string())
