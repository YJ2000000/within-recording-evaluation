"""Segment every dataset into labelled tasks and compute window features.

Output: <DATA_ROOT>/../windows/<DATASET>_windows.parquet with one row per 30-s window:
  dataset, pid, session, family, level, task, nasa (task-level NASA-TLX if available), t0, t1, features...
family = task family (e.g. 'nback', 'matb', 'stroop'); level = ordinal difficulty within the family
(0 = easiest); rest segments have family 'rest' and level -1.
Usage: python build_windows.py [COGBCI MPHRC UNIVERSE COGLOAD]
"""
import os, sys, re, glob, datetime as dt
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import features as F

R = os.environ.get('DATA_ROOT', os.path.expanduser('/path/to/project/data/raw'))
OUT = os.path.join(os.path.dirname(R), 'windows')
os.makedirs(OUT, exist_ok=True)
NPROC = int(os.environ.get('NPROC', '2'))

# ---------------------------------------------------------------- COG-BCI
COG_TASKS = {'zeroBACK': ('nback', 0), 'oneBACK': ('nback', 1), 'twoBACK': ('nback', 2),
             'MATBeasy': ('matb', 0), 'MATBmed': ('matb', 1), 'MATBdiff': ('matb', 2),
             'PVT': ('pvt', 0), 'Flanker': ('flanker', 0),
             'RS_Beg_EO': ('rest', -1), 'RS_End_EO': ('rest', -1), 'RS_Beg_EC': ('rest_ec', -1), 'RS_End_Ec': ('rest_ec', -1)}
RSME_NAME = {'MATBeasy': 'MATB_easy', 'MATBmed': 'MATB_med', 'MATBdiff': 'MATB_diff', 'zeroBACK': 'ZeroBack',
             'oneBACK': 'OneBack', 'twoBACK': 'TwoBack', 'PVT': 'PVT', 'Flanker': 'Flanker'}

def _cog_one(f):
    sub, ses, task = f.split('/')[-3], f.split('/')[-2], os.path.basename(f)[:-4]
    if task not in COG_TASKS:
        return []
    d = np.load(f, allow_pickle=False)
    fs = float(d['ecg_sfreq']); dur = d['ecg'].shape[1] / fs
    try:
        beats = F.beats_from_ecg(d['ecg'][0], fs)
    except Exception:
        beats = np.array([])
    fam, lvl = COG_TASKS[task]
    rows = F.window_table(dur, beats=beats)
    for r in rows:
        r.update(dataset='COGBCI', pid=sub, session=ses, task=task, family=fam, level=lvl)
    return rows

def build_cogbci():
    files = sorted(glob.glob(f'{R}/COGBCI/proc/sub-*/ses-*/*.npz'))
    with Pool(NPROC) as p:
        rows = [r for rs in p.imap_unordered(_cog_one, files, chunksize=4) for r in rs]
    df = pd.DataFrame(rows)
    # subjective workload (RSME) per subject/session/task
    rs = pd.read_csv(f'{R}/COGBCI/RSME.txt')
    rs.columns = [c.strip() for c in rs.columns]
    rs['pid'] = rs['sbj'].map(lambda s: f'sub-{int(s):02d}'); rs['session'] = rs['Session'].map(lambda s: f'ses-S{int(s)}')
    inv = {v: k for k, v in RSME_NAME.items()}
    rs['task'] = rs['condition'].map(inv)
    df = df.merge(rs[['pid', 'session', 'task', 'Score']].rename(columns={'Score': 'subjective'}), on=['pid', 'session', 'task'], how='left')
    df.to_parquet(f'{OUT}/COGBCI_windows.parquet', index=False)
    return df

# ---------------------------------------------------------------- MultiPhysio-HRC
def _mphrc_map(stem):
    if stem == 'stroopeasy_0': return ('stroop', 0, 'stroopeasy', 1)
    if stem == 'stroophard_0': return ('stroop', 1, 'stroophard', 1)
    if stem == 'n-back_0': return ('nback', 0, 'N-Back', 1)
    if stem == 'mat_0': return ('arith', 0, 'MAT', 1)
    if stem == 'hanoi_0': return ('hanoi', 0, 'Hanoi', 1)
    if stem == 'vr-plank_0': return ('vrplank', 0, 'vr-plank', 1)
    if stem == 'meditation_0': return ('meditation', -1, 'meditation', 1)
    if stem == 'rest_0': return ('rest', -1, 'rest', 1)
    if stem == 'rest-1': return ('rest', -1, 'rest', 2)
    if stem == 'rest-2': return ('rest', -1, None, None)
    m = re.match(r'(manual|cobot)-task-(\d)', stem)
    if m: return (m.group(1), 0, f'{m.group(1)}-task', int(m.group(2)))
    return None

def _mphrc_one(f):
    pid, stem = f.split('/')[-2], os.path.basename(f)[:-8]
    mp = _mphrc_map(stem)
    if mp is None:
        return []
    x = pd.read_parquet(f, columns=['ECG', 'EDA', 'RESP'])
    fs = 256.0; dur = len(x) / fs
    if dur < F.WIN:
        return []
    try:
        beats = F.beats_from_ecg(x['ECG'].values, fs)
    except Exception:
        beats = np.array([])
    eda = F.eda_prepare(x['EDA'].values, fs)
    rows = F.window_table(dur, beats=beats, eda=eda, resp=(x['RESP'].values, fs))
    fam, lvl, cls, rep = mp
    day = 'D2' if stem.startswith(('manual', 'cobot', 'rest-')) else 'D1'
    for r in rows:
        r.update(dataset='MPHRC', pid=str(pid), session=day, task=stem, family=fam, level=lvl, label_class=cls, label_rep=rep)
    return rows

def build_mphrc():
    files = sorted(glob.glob(f'{R}/MPHRC/raw_parquet/*/*.parquet'))
    with Pool(NPROC) as p:
        rows = [r for rs in p.imap_unordered(_mphrc_one, files, chunksize=4) for r in rs]
    df = pd.DataFrame(rows)
    lab = pd.read_csv(f'{R}/MPHRC/features/labels.csv')
    lab['pid'] = lab['ID'].astype(str)
    lab = lab.rename(columns={'Class': 'label_class', 'Repetition': 'label_rep', 'NASA': 'subjective'})
    df = df.merge(lab[['pid', 'label_class', 'label_rep', 'subjective', 'STAI', 'Arousal', 'Valence', 'Age', 'Gender']],
                  on=['pid', 'label_class', 'label_rep'], how='left')
    df.to_parquet(f'{OUT}/MPHRC_windows.parquet', index=False)
    return df

# ---------------------------------------------------------------- UNIVERSE (lab sessions)
U_ROUTINES = [  # (regex on routine name, family, level)
    (r'^Arithmetix with NeededFiles/easy_tasks\.csv$', 'arith', 0), (r'^Arithmetix with NeededFiles/hard_tasks\.csv$', 'arith', 1),
    (r'^Sudoku \(True\)$', 'sudoku', 0), (r'^Sudoku \(False\)$', 'sudoku', 1),
    (r'^Stroop level EASY$', 'stroop', 0), (r'^Stroop level HARD$', 'stroop', 1),
    (r'^N-Back Trial \(True\)$', 'nback', 0), (r'^N-Back Trial \(False\)$', 'nback', 1),
    (r'^Relaxation-Video$', 'rest', -1), (r'^Eye Closing Routine$', 'rest_ec', -1)]
U_TASKNAME = {('arith', 0): 'arithmetix_easy', ('arith', 1): 'arithmetix_hard', ('sudoku', 0): 'sudoku_easy', ('sudoku', 1): 'sudoku_hard',
              ('stroop', 0): 'stroop_easy', ('stroop', 1): 'stroop_hard', ('nback', 0): 'n_back_easy', ('nback', 1): 'n_back_hard'}
LINE = re.compile(r'Routine (.+?) (started|finished) at (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)')
ANCHOR = re.compile(r'start_welcome_logging: ([\d.]+)')
TS = re.compile(r'started at (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)')

def universe_segments(logf):
    """Parse PsychoPy log -> list of (family, level, unix_start, unix_end). Local time -> unix via the welcome anchor."""
    txt = open(logf, errors='ignore').read().splitlines()
    anchor_unix, anchor_local = None, None
    starts, segs = {}, []
    for i, ln in enumerate(txt):
        m = ANCHOR.search(ln)
        if m and anchor_unix is None:
            anchor_unix = float(m.group(1))
            for ln2 in txt[i:i + 10]:
                m2 = TS.search(ln2)
                if m2:
                    anchor_local = dt.datetime.fromisoformat(m2.group(1)); break
        m = LINE.search(ln)
        if not m:
            continue
        name, what, ts = m.group(1), m.group(2), dt.datetime.fromisoformat(m.group(3))
        for rx, fam, lvl in U_ROUTINES:
            if re.match(rx, name):
                if what == 'started':
                    starts[(fam, lvl, name)] = ts
                elif (fam, lvl, name) in starts:
                    segs.append((fam, lvl, starts.pop((fam, lvl, name)), ts))
    if anchor_unix is None or anchor_local is None:
        return []
    offset = anchor_unix - anchor_local.replace(tzinfo=dt.timezone.utc).timestamp()
    offset = round(offset / 900) * 900  # timezone offset, rounded to 15 min
    return [(f, l, a.replace(tzinfo=dt.timezone.utc).timestamp() + offset, b.replace(tzinfo=dt.timezone.utc).timestamp() + offset) for f, l, a, b in segs]

def _read_e4(path):
    head = pd.read_csv(path, header=None, nrows=2)
    start, fs = float(head.iloc[0, 0]), float(head.iloc[1, 0])
    x = pd.read_csv(path, header=None, skiprows=2).values
    return start, fs, x

def _universe_one(sess_dir):
    pid, sess = sess_dir.split('/')[-2], sess_dir.split('/')[-1]
    logs = glob.glob(f'{sess_dir}/Raw/Psychopy/*.log')
    if not logs:
        return []
    segs = universe_segments(logs[0])
    try:
        b0, bfs, bvp = _read_e4(f'{sess_dir}/Raw/Empatica/BVP.csv')
        e0, efs, eda = _read_e4(f'{sess_dir}/Raw/Empatica/EDA.csv')
        t0_, tfs, tmp = _read_e4(f'{sess_dir}/Raw/Empatica/TEMP.csv')
        a0, afs, acc = _read_e4(f'{sess_dir}/Raw/Empatica/ACC.csv')
    except Exception:
        return []
    amag = np.linalg.norm(acc.astype(float), axis=1)
    tl = pd.read_csv(f'{sess_dir}/Task_Labels.csv') if os.path.exists(f'{sess_dir}/Task_Labels.csv') else None
    rows = []
    for fam, lvl, us, ue in segs:
        def cut(start, fs, arr):
            i0, i1 = int((us - start) * fs), int((ue - start) * fs)
            return arr[max(i0, 0):max(i1, 0)] if i1 > 0 else arr[:0]
        bv = cut(b0, bfs, bvp[:, 0]); ed = cut(e0, efs, eda[:, 0]); tp = cut(t0_, tfs, tmp[:, 0]); am = cut(a0, afs, amag)
        dur = min(len(bv) / bfs, len(ed) / efs) if len(bv) and len(ed) else 0
        if dur < F.WIN:
            continue
        try:
            beats = F.beats_from_ppg(bv, bfs)
        except Exception:
            beats = np.array([])
        ws = F.window_table(dur, beats=beats, eda=F.eda_prepare(ed, efs), temp=(tp, tfs), acc=(am, afs))
        task = U_TASKNAME.get((fam, lvl), fam)
        nasa = np.nan
        if tl is not None and (tl['Task'] == task).any():
            nasa = float(tl.loc[tl['Task'] == task, 'Weighted Nasa Score'].iloc[0])
        for r in ws:
            r.update(dataset='UNIVERSE', pid=pid, session=sess, task=task, family=fam, level=lvl, subjective=nasa)
        rows += ws
    return rows

def build_universe():
    dirs = sorted(glob.glob(f'{R}/UNIVERSE/UN_*/Lab[12]'))
    with Pool(NPROC) as p:
        rows = [r for rs in p.imap_unordered(_universe_one, dirs) for r in rs]
    df = pd.DataFrame(rows)
    df.to_parquet(f'{OUT}/UNIVERSE_windows.parquet', index=False)
    return df

# ---------------------------------------------------------------- CogLoad (1 Hz band data)
def build_cogload():
    rows = []
    for f in sorted(glob.glob(f'{R}/CogLoad/ext/CogDatasets/CogLoad1/train/raw/*_sensors.csv')):
        if 'merged' in f:
            continue
        d = pd.read_csv(f)
        d['amag'] = np.sqrt(d.band_ax ** 2 + d.band_ay ** 2 + d.band_az ** 2)
        # contiguous blocks of identical (task, level)
        blk = (d[['task', 'level']] != d[['task', 'level']].shift()).any(axis=1).cumsum()
        for _, g in d.groupby(blk):
            task, level = g['task'].iloc[0], str(g['level'].iloc[0])
            if task in ('quest', 'post'):
                continue
            if task.endswith('rest'):
                fam, lvl = 'rest', -1
            elif task in ('n2', 'n3'):
                fam, lvl = 'nback', 0 if task == 'n2' else 1
            else:
                fam, lvl = task.lower(), int(level)
            dur = float(len(g))
            if dur < F.WIN:
                continue
            hr = g['hr'].values.astype(float); rr = g['rr'].values.astype(float)
            for t0, t1 in F._windows(dur):
                s = slice(int(t0), int(t1))
                h = hr[s]; r_ = rr[s]; r_ = r_[(r_ > F.IBI_MIN) & (r_ < F.IBI_MAX)]
                row = dict(dataset='COGLOAD', pid=g['user_id'].iloc[0], session='S1', task=f'{task}_{level}', family=fam, level=lvl,
                           t0=t0, t1=t1, hr_mean=np.nanmean(h), hr_std=np.nanstd(h),
                           rmssd=1000 * np.sqrt(np.mean(np.diff(r_) ** 2)) if len(r_) > 5 else np.nan,
                           ibi_valid=len(r_) / max(1, len(h)),
                           subjective=float(g['TLX_mean'].replace(-1, np.nan).max()))
                row.update(F.simple_window(g['gsr'].values, 1.0, t0, t1, 'scl'))
                row.update(F.simple_window(g['temperature'].values, 1.0, t0, t1, 'temp'))
                row.update(F.acc_window(g['amag'].values, 1.0, t0, t1))
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(f'{OUT}/COGLOAD_windows.parquet', index=False)
    return df

if __name__ == '__main__':
    todo = sys.argv[1:] or ['COGLOAD', 'UNIVERSE', 'MPHRC', 'COGBCI']
    for name in todo:
        t = dt.datetime.now()
        df = {'COGBCI': build_cogbci, 'MPHRC': build_mphrc, 'UNIVERSE': build_universe, 'COGLOAD': build_cogload}[name]()
        print(name, df.shape, 'participants', df.pid.nunique(), 'families', sorted(df.family.unique()), f'{(dt.datetime.now()-t).seconds}s', flush=True)
