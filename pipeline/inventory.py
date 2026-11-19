"""Build a data inventory (per participant x task: signal seconds, labels present) for all downloaded datasets.
Outputs CSVs in /path/to/project/data/inventory/.  No pickle files are loaded.
"""
import os, glob, json, re
import numpy as np, pandas as pd
R = os.environ.get('DATA_ROOT', os.path.expanduser('/path/to/project/data/raw')); O = os.path.join(os.path.dirname(R), 'inventory')
os.makedirs(O, exist_ok=True)

def e4_seconds(path):
    """Empatica E4 csv: row0 = start unix time, row1 = sampling rate, then samples."""
    try:
        with open(path) as f:
            head = [f.readline() for _ in range(2)]
            n = sum(1 for _ in f)
        fs = float(head[1].split(',')[0])
        return n / fs if fs > 0 else np.nan, float(head[0].split(',')[0])
    except Exception:
        return np.nan, np.nan

# ---------------- UNIVERSE ----------------
rows = []
for p in sorted(glob.glob(f'{R}/UNIVERSE/UN_*')):
    pid = os.path.basename(p)
    for sess in ('Lab1', 'Lab2'):
        d = f'{p}/{sess}'
        if not os.path.isdir(d):
            rows.append(dict(dataset='UNIVERSE', pid=pid, session=sess, present=False)); continue
        r = dict(dataset='UNIVERSE', pid=pid, session=sess, present=True)
        for sig in ('EDA', 'BVP', 'TEMP', 'ACC'):
            r[f'{sig}_s'], _ = e4_seconds(f'{d}/Raw/Empatica/{sig}.csv')
        tl = f'{d}/Task_Labels.csv'
        if os.path.exists(tl):
            t = pd.read_csv(tl); r['n_tasks_labeled'] = len(t); r['tasks'] = ';'.join(t['Task'].astype(str))
        r['psychopy'] = os.path.exists(f'{d}/Raw/Psychopy')
        r['stretched_e4'] = os.path.exists(f'{d}/Raw/Stretched/stretched_empatica.csv')
        rows.append(r)
    w = f'{p}/Wild'
    if os.path.isdir(w):
        recs = sorted(glob.glob(f'{w}/Raw/*/Empatica/EDA.csv'))
        secs = [e4_seconds(x)[0] for x in recs]
        tl = f'{w}/Task_Labels.csv'
        nl = len(pd.read_csv(tl)) if os.path.exists(tl) else 0
        rows.append(dict(dataset='UNIVERSE', pid=pid, session='Wild', present=True, n_recordings=len(recs),
                         EDA_s=np.nansum(secs), n_tasks_labeled=nl))
    else:
        rows.append(dict(dataset='UNIVERSE', pid=pid, session='Wild', present=False))
pd.DataFrame(rows).to_csv(f'{O}/UNIVERSE_inventory.csv', index=False)

# ---------------- MultiPhysio-HRC ----------------
rows = []
for f in sorted(glob.glob(f'{R}/MPHRC/raw_parquet/*/*.parquet')):
    pid, task = f.split('/')[-2], os.path.basename(f)[:-8]
    df = pd.read_parquet(f, columns=['ECG', 'EDA'])
    rows.append(dict(dataset='MPHRC', pid=pid, task=task, seconds=len(df) / 256.0,
                     ecg_nan=float(df['ECG'].isna().mean()), eda_nan=float(df['EDA'].isna().mean())))
inv = pd.DataFrame(rows)
lab = pd.read_csv(f'{R}/MPHRC/features/labels.csv') if os.path.exists(f'{R}/MPHRC/features/labels.csv') else None
inv.to_csv(f'{O}/MPHRC_inventory.csv', index=False)
if lab is not None:
    lab.to_csv(f'{O}/MPHRC_labels.csv', index=False)

# ---------------- COG-BCI ----------------
rows = []
for f in sorted(glob.glob(f'{R}/COGBCI/proc/sub-*/ses-*/*.npz')):
    sub, ses, task = f.split('/')[-3], f.split('/')[-2], os.path.basename(f)[:-4]
    d = np.load(f, allow_pickle=False)
    rows.append(dict(dataset='COGBCI', pid=sub, session=ses, task=task, seconds=float(d['n_times'] / d['ecg_sfreq']),
                     ecg=bool(d['ecg'].shape[0]), n_eeg=int(d['eeg'].shape[0]), n_events=len(d['ev_desc'])))
errs = [json.load(open(e)) for e in glob.glob(f'{R}/COGBCI/proc/**/*.error.json', recursive=True)]
pd.DataFrame(rows).to_csv(f'{O}/COGBCI_inventory.csv', index=False)
json.dump(errs, open(f'{O}/COGBCI_errors.json', 'w'), indent=1)

# ---------------- CogLoad ----------------
rows = []
for f in sorted(glob.glob(f'{R}/CogLoad/ext/CogDatasets/CogLoad1/train/raw/*_sensors.csv')):
    if 'merged' in f:
        continue
    df = pd.read_csv(f, usecols=['user_id', 'level', 'task', 'hr', 'gsr', 'rr', 'TLX_mean'])
    for (task, level), g in df.groupby(['task', 'level']):
        rows.append(dict(dataset='CogLoad', pid=g['user_id'].iloc[0], task=task, level=level, seconds=len(g),
                         tlx=float(g['TLX_mean'].replace(-1, np.nan).max())))
pd.DataFrame(rows).to_csv(f'{O}/CogLoad_inventory.csv', index=False)
print('inventory written to', O)
