"""Export every labelled task segment as a 64 Hz multichannel array for foundation-model embedding.

Output (next to the windows folder):
  segments/<DS>/<seg_id>.npy        float32 [n_channels, L] at 64 Hz
  segments/<DS>_index.parquet       one row per segment: seg_id, dataset, pid, session, task, family, level, channels, duration_s
Channels: COGBCI = [ECG]; MPHRC = [ECG, EDA, RESP]; UNIVERSE = [BVP, EDA, ACC magnitude].
Uses the same segmentation as build_windows.py, so window t0/t1 (seconds from segment start) line up.
"""
import os, sys, glob
import numpy as np, pandas as pd
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(__file__))
import build_windows as B

R = B.R
OUT = os.path.join(os.path.dirname(R), 'segments')
FS = 64


def to64(x, fs):
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nanmedian(x) if np.isfinite(x).any() else 0.0)
    if fs == FS:
        return x.astype(np.float32)
    if fs > FS:
        from math import gcd
        f = int(round(fs)); g = gcd(FS, f)
        return resample_poly(x, FS // g, f // g).astype(np.float32)
    t_old = np.arange(len(x)) / fs
    t_new = np.arange(0, t_old[-1], 1.0 / FS)
    return np.interp(t_new, t_old, x).astype(np.float32)


def stack(chs):
    L = min(len(c) for c in chs)
    return np.stack([c[:L] for c in chs])


def save(ds, rows, sid, arr, meta):
    os.makedirs(f'{OUT}/{ds}', exist_ok=True)
    np.save(f'{OUT}/{ds}/{sid}.npy', arr)
    meta.update(seg_id=sid, dataset=ds, duration_s=arr.shape[1] / FS)
    rows.append(meta)


def cogbci():
    rows = []
    for i, f in enumerate(sorted(glob.glob(f'{R}/COGBCI/proc/sub-*/ses-*/*.npz'))):
        sub, ses, task = f.split('/')[-3], f.split('/')[-2], os.path.basename(f)[:-4]
        if task not in B.COG_TASKS:
            continue
        d = np.load(f, allow_pickle=False)
        fam, lvl = B.COG_TASKS[task]
        save('COGBCI', rows, i, stack([to64(d['ecg'][0], float(d['ecg_sfreq']))]),
             dict(pid=sub, session=ses, task=task, family=fam, level=lvl, channels='ECG'))
    return rows


def mphrc():
    rows = []
    for i, f in enumerate(sorted(glob.glob(f'{R}/MPHRC/raw_parquet/*/*.parquet'))):
        pid, stem = f.split('/')[-2], os.path.basename(f)[:-8]
        mp = B._mphrc_map(stem)
        if mp is None:
            continue
        x = pd.read_parquet(f, columns=['ECG', 'EDA', 'RESP'])
        if len(x) / 256.0 < 30:
            continue
        fam, lvl, cls, rep = mp
        day = 'D2' if stem.startswith(('manual', 'cobot', 'rest-')) else 'D1'
        save('MPHRC', rows, i, stack([to64(x[c].values, 256.0) for c in ('ECG', 'EDA', 'RESP')]),
             dict(pid=str(pid), session=day, task=stem, family=fam, level=lvl, channels='ECG,EDA,RESP'))
    return rows


def universe():
    rows, i = [], 0
    for sess_dir in sorted(glob.glob(f'{R}/UNIVERSE/UN_*/Lab[12]')):
        pid, sess = sess_dir.split('/')[-2], sess_dir.split('/')[-1]
        logs = glob.glob(f'{sess_dir}/Raw/Psychopy/*.log')
        if not logs:
            continue
        try:
            b0, bfs, bvp = B._read_e4(f'{sess_dir}/Raw/Empatica/BVP.csv')
            e0, efs, eda = B._read_e4(f'{sess_dir}/Raw/Empatica/EDA.csv')
            a0, afs, acc = B._read_e4(f'{sess_dir}/Raw/Empatica/ACC.csv')
        except Exception:
            continue
        amag = np.linalg.norm(acc.astype(float), axis=1)
        for fam, lvl, us, ue in B.universe_segments(logs[0]):
            def cut(start, fs, arr):
                i0, i1 = int((us - start) * fs), int((ue - start) * fs)
                return arr[max(i0, 0):max(i1, 0)] if i1 > 0 else arr[:0]
            bv, ed, am = cut(b0, bfs, bvp[:, 0]), cut(e0, efs, eda[:, 0]), cut(a0, afs, amag)
            if min(len(bv) / bfs, len(ed) / efs) < 30:
                continue
            task = B.U_TASKNAME.get((fam, lvl), fam)
            save('UNIVERSE', rows, i, stack([to64(bv, bfs), to64(ed, efs), to64(am, afs)]),
                 dict(pid=pid, session=sess, task=task, family=fam, level=lvl, channels='BVP,EDA,ACC'))
            i += 1
    return rows


if __name__ == '__main__':
    for ds, fn in [('COGBCI', cogbci), ('MPHRC', mphrc), ('UNIVERSE', universe)]:
        rows = fn()
        pd.DataFrame(rows).to_parquet(f'{OUT}/{ds}_index.parquet', index=False)
        print(ds, len(rows), 'segments', flush=True)
