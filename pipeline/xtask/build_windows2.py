"""Window features for the plan-B datasets (same features and windows as build_windows.py).

WESAD     -> WESAD_CHEST (RespiBAN ECG, EDA, temp, resp, |ACC|) and WESAD_WRIST (E4 BVP, EDA, temp, |ACC|),
             same windows on the same time axis. Conditions from the synchronised label channel:
             1 baseline -> family 'rest', 2 stress (TSST), 3 amusement, 4 meditation (two runs).
             'start' = block start in minutes from the beginning of the first labelled block.
DS007554  -> chest ECG (Delsys, ~1.9 kHz) for 7 tasks x 3 repetitions (ses-01..03 within one visit). No rest recordings.
Usage: python build_windows2.py WESAD DS007554
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd
from scipy.signal import resample_poly
from multiprocessing import Pool
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import features as F

R2 = os.environ.get('DATA_ROOT2', '/path/to/project/data/raw2')
OUT = os.environ.get('WIN_DIR', '/path/to/project/data/windows')
NPROC = int(os.environ.get('NPROC', '2'))
WESAD_LAB = {1: 'rest', 2: 'stress', 3: 'amusement', 4: 'meditation'}


def _runs(lab, fs=4.0, min_s=60):
    """Contiguous runs of labels 1-4 -> list of (label, t_start_s, t_end_s)."""
    out, i, n = [], 0, len(lab)
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]:
            j += 1
        if lab[i] in WESAD_LAB and (j - i) / fs >= min_s:
            out.append((int(lab[i]), i / fs, j / fs))
        i = j
    return out


def beats_chunked(x, fs, fn, chunk_s=300.0, ov_s=10.0):
    """Beat detection in 5-min chunks (neurokit needs ~1 GB per 10 min of 700 Hz ECG; whole WESAD sessions run out of memory)."""
    n = len(x); out = []
    step = int(chunk_s * fs); ov = int(ov_s * fs)
    for s0 in range(0, n, step):
        a, b = max(0, s0 - ov), min(n, s0 + step + ov)
        if b - a < 10 * fs:
            continue
        try:
            bt = fn(x[a:b], fs) + a / fs
        except Exception:
            continue
        out.append(bt[(bt >= s0 / fs) & (bt < min(n, s0 + step) / fs)])
    return np.sort(np.concatenate(out)) if out else np.array([])


def _wesad_one(f):
    sid = os.path.basename(f).split('_')[0]
    z = np.load(f)
    runs = _runs(z['label4'])
    t_first = runs[0][1] if runs else 0.0
    rows = []
    cb = beats_chunked(z['c_ecg'], 700.0, F.beats_from_ecg)
    wb = beats_chunked(z['w_bvp'], 64.0, F.beats_from_ppg)
    c_eda = F.eda_prepare(z['c_eda'], 4.0); w_eda = F.eda_prepare(z['w_eda'], 4.0)
    med_count = 0
    for lab, a, b in runs:
        fam = WESAD_LAB[lab]
        if fam == 'meditation':
            med_count += 1
        task = f'{fam}_{med_count}' if fam == 'meditation' else fam
        dur = b - a
        def seg(x, fs):
            return np.asarray(x[int(a * fs):int(b * fs)])
        def shift(beats):
            bb = beats[(beats >= a) & (beats < b)] - a
            return bb
        for dev in ('CHEST', 'WRIST'):
            if dev == 'CHEST':
                ws = F.window_table(dur, beats=shift(cb), eda=_cut_eda(c_eda, a, b), temp=(seg(z['c_temp'], 4.0), 4.0),
                                    acc=(seg(z['c_acc'], 32.0), 32.0), resp=(seg(z['c_resp'], 70.0), 70.0))
            else:
                ws = F.window_table(dur, beats=shift(wb), eda=_cut_eda(w_eda, a, b), temp=(seg(z['w_temp'], 4.0), 4.0),
                                    acc=(seg(z['w_acc'], 32.0), 32.0))
            for r in ws:
                r.update(dataset=f'WESAD_{dev}', pid=sid, session='S1', task=task, family=fam,
                         level=-1 if fam == 'rest' else (1 if fam == 'stress' else 0), subjective=np.nan,
                         start=(a - t_first) / 60.0)
            rows += ws
    return rows


def _cut_eda(e, a, b):
    """eda_prepare returns a dict of 4 Hz arrays; cut every array to [a, b)."""
    if isinstance(e, dict):
        return {k: (v[int(a * 4):int(b * 4)] if hasattr(v, '__len__') and not np.isscalar(v) else v) for k, v in e.items()}
    if isinstance(e, tuple):
        return tuple(x[int(a * 4):int(b * 4)] if hasattr(x, '__len__') else x for x in e)
    return e


def build_wesad():
    files = sorted(glob.glob(f'{R2}/WESAD/WESAD/S*/S*_compact.npz'))
    with Pool(NPROC) as p:
        rows = [r for rs in p.imap_unordered(_wesad_one, files) for r in rs]
    df = pd.DataFrame(rows)
    for dev in ('CHEST', 'WRIST'):
        d = df[df.dataset == f'WESAD_{dev}'].dropna(axis=1, how='all')
        d.to_parquet(f'{OUT}/WESAD_{dev}_windows.parquet', index=False)
        print(f'WESAD_{dev}', d.shape, d.pid.nunique(), 'people', d.groupby('family').size().to_dict(), flush=True)


DS_TASKS = {'passivemotor': ('motor', 0), 'activemotor': ('motor', 1), 'full': ('motor', 2),  # motor family: passive < active < active+cognitive
            'nback': ('cognitive', 0), 'mentalarithmetic': ('cognitive', 1), 'nbackarithmetic': ('cognitive', 2), 'motorimagery': ('imagery', 0)}


def _ds_one(f):
    sub, ses = f.split('/')[-4], f.split('/')[-3]
    task = os.path.basename(f).split('_task-')[1].split('_')[0]
    js = pd.read_json(f.replace('.tsv.gz', '.json'), typ='series')
    fs = float(js['SamplingFrequency'])
    x = pd.read_csv(f, sep='\t', header=None).iloc[:, 0].values
    dur = len(x) / fs
    try:
        beats = F.beats_from_ecg(x, fs)
    except Exception:
        return []
    fam, lvl = DS_TASKS.get(task, (task, 0))
    ws = F.window_table(dur, beats=beats)
    for r in ws:
        r.update(dataset='DS007554', pid=sub, session=ses, task=task, family=fam, level=lvl, subjective=np.nan)
    return ws


def build_ds007554():
    files = sorted(glob.glob(f'{R2}/DS007554/sub-*/ses-*/beh/*recording-ECG_physio.tsv.gz'))
    with Pool(NPROC) as p:
        rows = [r for rs in p.imap_unordered(_ds_one, files) for r in rs]
    df = pd.DataFrame(rows)
    rat = pd.read_csv(f'{R2}/DS007554/phenotype/cognitive_load.tsv', sep='\t', na_values='n/a')
    name = {'nback': 'Nback', 'mentalarithmetic': 'Arithmetic', 'motorimagery': 'MotorImagery', 'passivemotor': 'PassiveMotor',
            'nbackarithmetic': 'NbackArithmetic', 'activemotor': 'ActiveMotor', 'full': 'Full'}
    long = rat.melt(id_vars='participant_id', var_name='k', value_name='subjective')
    long['session'] = 'ses-' + long.k.str.slice(5, 7); long['task'] = long.k.str.split('_').str[1].map({v: k for k, v in name.items()})
    df = df.drop(columns='subjective').merge(long.rename(columns={'participant_id': 'pid'})[['pid', 'session', 'task', 'subjective']],
                                             on=['pid', 'session', 'task'], how='left')
    df.to_parquet(f'{OUT}/DS007554_windows.parquet', index=False)
    print('DS007554', df.shape, df.pid.nunique(), 'people', df.groupby('task').size().to_dict(), flush=True)




# ------------------------------------------------------------------ SenseCobot (cobot programming; Shimmer chest ECG + finger GSR; E4 wrist at 1 Hz + E4 IBI)
SC = f'{R2}/SENSECOBOT/SenseCobot'


_MMSS = r'^(\d+):(\d+(?:\.\d+)?)$'


def _span_seconds(s):
    """Recording span (s) from a timestamp column. R3 fix: 'MM:SS.f' strings with MM < 24 were silently parsed by
    pandas as HH:MM:SS, giving spans of hours; detect that format first and unwrap hour roll-overs."""
    s = s.astype(str)
    if s.str.match(_MMSS).mean() > 0.99:
        mm = s.str.extract(_MMSS).astype(float)
        x = (mm[0] * 60 + mm[1]).values
        d = np.diff(x); d[d < -1800] += 3600
        return float(np.nansum(d))
    t = pd.to_datetime(s, format='mixed', errors='coerce')
    return float((t.max() - t.min()).total_seconds())


def _parse_times(s):
    """Timestamps come in two formats (full datetime, or Excel-mangled 'MM:SS.f'); return seconds (float) or NaN."""
    if s.astype(str).str.match(_MMSS).mean() > 0.99:   # R3: see _span_seconds
        mm = s.astype(str).str.extract(_MMSS).astype(float)
        x = (mm[0] * 60 + mm[1]).values
        d = np.r_[0, np.diff(x)]; d[d < -1800] += 3600
        return pd.Series(np.cumsum(d), index=s.index), pd.Series(pd.NaT, index=s.index)
    t = pd.to_datetime(s, format='mixed', errors='coerce')
    out = (t - t.min()).dt.total_seconds() if t.notna().any() else pd.Series(np.nan, index=s.index)
    bad = t.isna()
    if bad.any():
        mm = s[bad].astype(str).str.extract(r'^(\d+):(\d+(?:\.\d+)?)$').astype(float)
        out[bad] = mm[0] * 60 + mm[1]
    return out, t


def _sc_one(args):
    try:
        return _sc_one_inner(args)
    except pd.errors.EmptyDataError:
        return []


def _sc_one_inner(args):
    p, seg = args
    tag = 'Baseline' if seg == 'Baseline' else seg
    f_ecg = f'{SC}/ECG_Shimmer3_Signals/ECG_{tag}_P_{p}.csv'
    if not os.path.exists(f_ecg):
        return []
    e = pd.read_csv(f_ecg)
    # R3 fix: the Readme states 512 Hz, but P11-P21 files hold ~546-584 samples per second of
    # timestamp span (P14 task 1: ~463). Use the effective rate of each file; fall back to 512 Hz only if implausible.
    span = _span_seconds(e['Timestamp']) if 'Timestamp' in e else np.nan
    fs = (len(e) - 1) / span if np.isfinite(span) and span > 0 else np.nan
    fs_src = 'timestamps'
    if not (400 <= fs <= 700):
        fs, fs_src = 512.0, 'readme'
    dur = len(e) / fs
    beats = beats_chunked(e['ECG LL-RA CAL'].values, fs, F.beats_from_ecg)
    blocks = []  # (task, family, level, a, b) in seconds from segment start
    if seg == 'Baseline' and 'SourceStimuliName' in e:
        st = e['SourceStimuliName'].astype(str).values
        i = 0
        while i < len(st):
            j = i
            while j < len(st) and st[j] == st[i]:
                j += 1
            if st[i].startswith('60 Sec Video') and (j - i) / fs >= 30:
                blocks.append((f'rest_{len([b for b in blocks if b[1] == "rest"]) + 1}', 'rest', -1, i / fs, j / fs))
            i = j
    else:
        n = int(seg.split()[-1])
        blocks.append((f'task{n}', 'task', n, 0.0, dur))
    rows = []
    gsr = None
    f_g = f'{SC}/GSR_Shimmer3_Signals/GSR_{tag}_P_{p}.csv'
    if os.path.exists(f_g) and os.path.getsize(f_g) > 100:
        g = pd.read_csv(f_g, usecols=['GSR Conductance CAL'])['GSR Conductance CAL'].values
        gsr = F.eda_prepare(g, len(g) / dur if dur else 128.0)
    w = None
    f_w = f'{SC}/EDA_Empatica_Signals/EDA_Empatica_{tag}_P_{p}.csv'
    if os.path.exists(f_w) and os.path.getsize(f_w) > 100:
        w = pd.read_csv(f_w)
        wt, wabs = _parse_times(w.Timestamp)
        wfs = (len(w) - 1) / wt.max() if wt.max() > 0 else 1.0
        w_eda = F.eda_prepare(w.EDA.values, wfs)
        w_acc = np.linalg.norm(w[['ACC_X', 'ACC_Y', 'ACC_Z']].values.astype(float), axis=1)
    ibi_beats = None
    f_i = f'{SC}/IBI_Empatica_Signals/IBI_Empatica_{tag}_P_{p}.csv'
    if os.path.exists(f_i) and os.path.getsize(f_i) > 30 and w is not None:
        ib = pd.read_csv(f_i)
        it = pd.to_datetime(ib.Timestamp, format='mixed', errors='coerce')
        if it.notna().all() and wabs.notna().any():
            ibi_beats = ((it - wabs.min()).dt.total_seconds()).values
    # R3: continuous wrist heart-rate estimates for an agreement analysis against chest ECG:
    # the E4's own 1 Hz HR column and the Bangle.js HR with its confidence value.
    w_hr = None
    if w is not None and 'HR' in w:
        w_hr = (wt.values.astype(float), w.HR.values.astype(float))
    bangle = None
    f_b = f'{SC}/HR_Bangle_Signals/HR_Bangle_{tag}_P_{p}.csv'
    if os.path.exists(f_b) and os.path.getsize(f_b) > 30 and w is not None and wabs.notna().any():
        bg = pd.read_csv(f_b)
        bt = pd.to_datetime(bg.Timestamp, format='mixed', errors='coerce')
        if bt.notna().all():
            bangle = ((bt - wabs.min()).dt.total_seconds().values, bg.HR.values.astype(float), bg.Confidence.values.astype(float))

    def series_mean(ts, v, t0, t1, mask=None):
        m = (ts >= t0) & (ts < t1) & np.isfinite(v) & (v > 0)
        if mask is not None:
            m &= mask
        return (float(np.mean(v[m])), int(m.sum())) if m.sum() >= 5 else (np.nan, int(m.sum()))

    for task, fam, lvl, a, b in blocks:
        bd = b - a
        def cut_eda(x):
            return None if x is None else {k: (v[int(a * 4):int(b * 4)] if isinstance(v, np.ndarray) else v) for k, v in x.items()}
        lab = F.window_table(bd, beats=beats[(beats >= a) & (beats < b)] - a, eda=cut_eda(gsr))
        for r in lab:
            r.update(dataset='SENSECOBOT_LAB', pid=p, session='S1', task=task, family=fam, level=lvl, seg_start_s=a,
                     fs_ecg=fs, fs_src=fs_src)
        rows += lab
        if w is not None:
            ia, ibb = int(a * wfs), int(b * wfs)
            wr = F.window_table(bd, beats=(ibi_beats[(ibi_beats >= a) & (ibi_beats < b)] - a) if ibi_beats is not None else None,
                                eda=cut_eda(w_eda), temp=(w.TEMP.values[ia:ibb].astype(float), wfs), acc=(w_acc[ia:ibb], wfs))
            for r in wr:
                r.update(dataset='SENSECOBOT_WRIST', pid=p, session='S1', task=task, family=fam, level=lvl, seg_start_s=a,
                         has_ibi=ibi_beats is not None)
                if w_hr is not None:
                    r['hr_e4vendor'], _ = series_mean(w_hr[0], w_hr[1], a + r['t0'], a + r['t1'])
                if bangle is not None:
                    r['hr_bangle'], r['n_bangle'] = series_mean(bangle[0], bangle[1], a + r['t0'], a + r['t1'])
                    r['hr_bangle_hiconf'], _ = series_mean(bangle[0], bangle[1], a + r['t0'], a + r['t1'], bangle[2] >= 80)
            rows += wr
    return rows


def build_sensecobot():
    files = sorted(glob.glob(f'{SC}/ECG_Shimmer3_Signals/ECG_*_P_*.csv'))
    jobs = []
    for f in files:
        b = os.path.basename(f)[4:-4]  # e.g. 'Task 1_P_01' / 'Baseline_P_01'
        seg, p = b.rsplit('_P_', 1)
        jobs.append((p, seg))
    with Pool(NPROC) as pool:
        rows = [r for rs in pool.imap_unordered(_sc_one, jobs) for r in rs]
    df = pd.DataFrame(rows)
    tlx = pd.read_csv(f'{SC}/Additional_Information/NASA_TLX.csv', index_col=0); tlx.columns = [c.strip() for c in tlx.columns]
    mv = pd.read_csv(f'{SC}/Additional_Information/Movement_Type.csv', index_col=0); mv.columns = [c.strip() for c in mv.columns]
    def lookup(tab, col, pid):
        try:
            return float(tab.loc[f'P_{pid}', col])
        except Exception:
            return np.nan
    df['subjective'] = [lookup(tlx, f'B_TASK {l}', p) if l > 0 else np.nan for p, l in zip(df.pid, df.level)]
    df['physical'] = [lookup(tlx, f'A_TASK {l}', p) if l > 0 else np.nan for p, l in zip(df.pid, df.level)]
    df['movement'] = [lookup(mv, f'TASK {l}', p) if l > 0 else np.nan for p, l in zip(df.pid, df.level)]
    # R3 chest-ECG QC. After the rate fix, chest HR in P11-P21 is still 0.56-0.73 x the HR reported by two independent
    # wrist devices (E4 HR column, Bangle.js high-confidence HR), which agree with each other; beat periodicity in
    # those files implies a true rate near 900+ Hz with samples missing, which the released files do not let us repair.
    # Rule: keep a participant's chest cardiac features only if the median ratio chest HR / wrist HR over windows with
    # usable chest beats lies in [0.8, 1.2]; otherwise mask them (EDA from the finger is unaffected).
    lab, wr = df[df.dataset == 'SENSECOBOT_LAB'], df[df.dataset == 'SENSECOBOT_WRIST']
    m = lab[['pid', 'task', 'family', 't0', 'hr_mean', 'ibi_valid']].merge(
        wr[['pid', 'task', 't0', 'hr_e4vendor', 'hr_bangle_hiconf']], on=['pid', 'task', 't0'], how='left')
    m = m[m.ibi_valid >= 0.8]
    m['ref'] = m[['hr_e4vendor', 'hr_bangle_hiconf']].median(axis=1)
    m['ratio'] = m.hr_mean / m.ref
    def _ratio(g):   # seated rest preferred: the wrist references themselves are biased during the tasks
        r = g[g.family == 'rest'].ratio.dropna()
        return r.median() if len(r) >= 3 else g.ratio.median()
    qc = m.groupby('pid').apply(_ratio).reindex(sorted(lab.pid.unique()))
    qc_tab = pd.DataFrame({'ratio_chest_over_wrist': qc, 'pass': qc.between(0.8, 1.2)})
    qc_tab.to_csv(f'{OUT}/SENSECOBOT_ecg_qc.csv')
    print('SenseCobot chest QC pass:', int(qc_tab['pass'].sum()), 'of', len(qc_tab), flush=True)
    bad = (df.dataset == 'SENSECOBOT_LAB') & df.pid.map(qc_tab['pass']).eq(False)
    df['ecg_qc_pass'] = np.where(df.dataset == 'SENSECOBOT_LAB', ~bad, np.nan)
    df['ibi_valid_raw'] = df['ibi_valid']
    for c in ('hr_mean', 'hr_std', 'rmssd', 'sdnn', 'pnn50', 'ibi_valid'):
        df.loc[bad, c] = np.nan
    for dev in ('LAB', 'WRIST'):
        d = df[df.dataset == f'SENSECOBOT_{dev}'].dropna(axis=1, how='all')
        d.to_parquet(f'{OUT}/SENSECOBOT_{dev}_windows.parquet', index=False)
        print(f'SENSECOBOT_{dev}', d.shape, d.pid.nunique(), 'people', d.groupby('task').size().to_dict(), flush=True)





# ------------------------------------------------------------------ CogWear (E4 + Samsung Galaxy Watch4, Stroop vs baseline, two sessions ~2 weeks apart)
CW = glob.glob(f'{R2}/COGWEAR/cogwear*')[0] if glob.glob(f'{R2}/COGWEAR/cogwear*') else ''


def _cw_one(seg_dir):
    parts = seg_dir.split('/')
    if parts[-4] == 'survey_gamification':
        pid, sess, seg = parts[-3], parts[-2], parts[-1]
    else:
        pid, sess, seg = parts[-2], 'pilot', parts[-1]
    fam = {'baseline': 'rest', 'cognitive_load': 'stroop'}.get(seg)
    if fam is None:
        return []
    def rd(name):
        f = f'{seg_dir}/{name}.csv'
        return pd.read_csv(f) if os.path.exists(f) and os.path.getsize(f) > 50 else None
    bvp, eda, tmp, sam = rd('empatica_bvp'), rd('empatica_eda'), rd('empatica_temp'), rd('samsung_bvp')
    if bvp is None or len(bvp) < 64 * 40:
        return []
    t0 = bvp.time.iloc[0]; dur = bvp.time.iloc[-1] - t0
    rows = []
    try:
        b = F.beats_from_ppg(bvp.bvp.values, 64.0) + (bvp.time.iloc[0] - t0)
    except Exception:
        b = np.array([])
    e = F.eda_prepare(eda.eda.values, 4.0) if eda is not None else None
    ws = F.window_table(dur, beats=b, eda=e, temp=(tmp.temp.values, 4.0) if tmp is not None else None)
    for r in ws:
        r.update(dataset='COGWEAR_E4', pid=pid, session=sess, task=seg, family=fam, level=-1 if fam == 'rest' else 1, subjective=np.nan, abs_start=t0)
    rows += ws
    if sam is not None and 'PPG GREEN' in sam and len(sam) > 25 * 40:
        # R3 fix: the watch stream has out-of-order samples and gaps of up to tens of seconds.
        # Sort, drop repeated timestamps, put contiguous runs (no gap > 0.2 s) on a uniform 25 Hz grid, upsample
        # x4 for sub-sample peak timing, and detect beats run by run. Coverage is recorded per window.
        ts = sam.time.values.astype(float); xs = sam['PPG GREEN'].values.astype(float)
        o = np.argsort(ts, kind='stable'); ts, xs = ts[o], xs[o]
        keep = np.r_[True, np.diff(ts) > 0] & np.isfinite(xs); ts, xs = ts[keep], xs[keep]
        cut = np.where(np.diff(ts) > 0.2)[0]
        starts, ends = np.r_[0, cut + 1], np.r_[cut + 1, len(ts)]
        bs, cov_t = [], []
        for s0, s1 in zip(starts, ends):
            if ts[s1 - 1] - ts[s0] < 10:
                continue
            grid = np.arange(ts[s0], ts[s1 - 1], 1 / 25.0)
            xu = resample_poly(np.interp(grid, ts[s0:s1], xs[s0:s1]), 4, 1)
            cov_t.append((ts[s0] - t0, ts[s1 - 1] - t0))
            try:
                bs.append(F.beats_from_ppg(xu, 100.0) + (ts[s0] - t0))
            except Exception:
                pass
        bs = np.sort(np.concatenate(bs)) if bs else np.array([])
        ws = F.window_table(dur, beats=bs)
        for r in ws:
            cov = sum(max(0.0, min(r['t1'], c1) - max(r['t0'], c0)) for c0, c1 in cov_t) / (r['t1'] - r['t0'])
            r.update(dataset='COGWEAR_SAMSUNG', pid=pid, session=sess, task=seg, family=fam, level=-1 if fam == 'rest' else 1,
                     subjective=np.nan, abs_start=t0, coverage=cov)
        rows += ws
    return rows


def build_cogwear():
    dirs = sorted(glob.glob(f'{CW}/survey_gamification/*/*/*')) + sorted(glob.glob(f'{CW}/pilot/*/*'))
    dirs = [d for d in dirs if os.path.isdir(d)]
    with Pool(NPROC) as pool:
        rows = [r for rs in pool.imap_unordered(_cw_one, dirs) for r in rs]
    df = pd.DataFrame(rows)
    for dev in ('E4', 'SAMSUNG'):
        d = df[df.dataset == f'COGWEAR_{dev}'].dropna(axis=1, how='all')
        d.to_parquet(f'{OUT}/COGWEAR_{dev}_windows.parquet', index=False)
        print(f'COGWEAR_{dev}', d.shape, d.pid.nunique(), 'people', d.groupby(['session', 'family']).size().to_dict(), flush=True)





# ------------------------------------------------------------------ MAUS (torso ECG + fingertip PPG + finger GSR at 256 Hz; wrist PPG at 100 Hz)
def _maus_one(sub_dir):
    pid = os.path.basename(sub_dir)
    rd = lambda n: pd.read_csv(f'{sub_dir}/{n}')
    ecg, gsr, ppg, pix = rd('inf_ecg.csv'), rd('inf_gsr.csv'), rd('inf_ppg.csv'), rd('pixart.csv')
    rest, prest = rd('inf_resting.csv'), rd('pixart_resting.csv')
    blocks = [('rest', 'rest', -1, 0, rest['Resting_ECG'].values, rest['Resting_GSR'].values, rest['Resting_PPG'].values, prest.iloc[:, 0].values)]
    for i, col in enumerate(ecg.columns):
        n = int(col.split(':')[1].replace('back', ''))
        blocks.append((f'trial{i + 1}', 'nback', n, i + 1, ecg[col].values, gsr[gsr.columns[i]].values, ppg[ppg.columns[i]].values,
                       pix[pix.columns[i]].dropna().values))
    rows = []
    for task, fam, lvl, pos, e, g, p, w in blocks:
        dur = min(len(e) / 256.0, len(w) / 100.0)
        def safe_beats(fn, x, fs):
            try:
                return fn(x, fs)
            except Exception:
                return np.array([])
        sets = {'MAUS_ECG': dict(beats=safe_beats(F.beats_from_ecg, e, 256.0), eda=F.eda_prepare(g, 256.0)),
                'MAUS_FINGER': dict(beats=safe_beats(F.beats_from_ppg, p, 256.0)),
                'MAUS_WRIST': dict(beats=safe_beats(F.beats_from_ppg, w, 100.0))}
        for name, kw in sets.items():
            for r in F.window_table(dur, **kw):
                r.update(dataset=name, pid=pid, session='S1', task=task, family=fam, level=lvl, position=pos, subjective=np.nan)
                rows.append(r)
    return rows


def build_maus():
    root = f'{R2}/MAUS/MAUS'
    dirs = sorted(glob.glob(f'{root}/Data/Raw_data/*'))
    with Pool(NPROC) as pool:
        rows = [r for rs in pool.imap_unordered(_maus_one, dirs) for r in rs]
    df = pd.DataFrame(rows)
    tl = []
    for f in glob.glob(f'{root}/Subjective_rating/*/NASA_TLX.csv'):
        t = pd.read_csv(f).set_index('Scale Title')
        cols = [c for c in t.columns if c.startswith('Trial')]
        w = t['Weight'] if 'Weight' in t else None
        for i, c in enumerate(cols):
            v = pd.to_numeric(t[c], errors='coerce')
            score = float((v * w).sum() / w.sum()) if w is not None and w.sum() > 0 else float(v.mean())
            tl.append(dict(pid=f.split('/')[-2], task=f'trial{i + 1}', subjective=score))
    tl = pd.DataFrame(tl)
    df = df.drop(columns='subjective').merge(tl, on=['pid', 'task'], how='left')
    for name in ('MAUS_ECG', 'MAUS_FINGER', 'MAUS_WRIST'):
        d = df[df.dataset == name].dropna(axis=1, how='all')
        d.to_parquet(f'{OUT}/{name}_windows.parquet', index=False)
        print(name, d.shape, d.pid.nunique(), 'people', d.groupby('level').size().to_dict(), 'usable', round(d.ibi_valid.ge(.8).mean(), 3), flush=True)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        {'WESAD': build_wesad, 'DS007554': build_ds007554, 'SENSECOBOT': build_sensecobot, 'COGWEAR': build_cogwear, 'MAUS': build_maus}[name]()
