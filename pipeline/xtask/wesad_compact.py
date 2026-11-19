"""WESAD: convert each S*.pkl (~0.9 GB, chest at 700 Hz float64) into a compact npz and delete the pkl.
Kept: chest ECG 700 Hz, chest EDA/Temp 4 Hz, chest Resp 70 Hz, chest |ACC| 32 Hz; wrist BVP 64 Hz, EDA/TEMP 4 Hz, |ACC| 32 Hz;
label at 4 Hz (chest label decimated); quest file parsed separately."""
import os, sys, glob, pickle, numpy as np
from scipy.signal import decimate, resample_poly
root = sys.argv[1]
for f in sorted(glob.glob(f'{root}/S*/S*.pkl')):
    out = f[:-4] + '_compact.npz'
    if os.path.exists(out):
        os.remove(f); continue
    d = pickle.load(open(f, 'rb'), encoding='latin1')
    c, w = d['signal']['chest'], d['signal']['wrist']
    lab = np.asarray(d['label'])
    def ds(x, up, down):
        return resample_poly(np.asarray(x, np.float64).ravel(), up, down).astype(np.float32)
    np.savez(out,
             c_ecg=np.asarray(c['ECG'], np.float32).ravel(),
             c_eda=ds(c['EDA'], 4, 700), c_temp=ds(c['Temp'], 4, 700), c_resp=ds(c['Resp'], 1, 10),
             c_acc=ds(np.linalg.norm(c['ACC'], axis=1), 32, 700),
             w_bvp=np.asarray(w['BVP'], np.float32).ravel(), w_eda=np.asarray(w['EDA'], np.float32).ravel(),
             w_temp=np.asarray(w['TEMP'], np.float32).ravel(), w_acc=np.linalg.norm(np.asarray(w['ACC'], np.float32), axis=1),
             label4=lab[::175].astype(np.int8))
    del d, c, w
    os.remove(f)
    print(os.path.basename(out), round(os.path.getsize(out) / 1e6, 1), 'MB', flush=True)
