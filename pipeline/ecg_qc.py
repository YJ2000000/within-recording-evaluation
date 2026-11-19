import numpy as np, pandas as pd, glob, os
from scipy.signal import butter, filtfilt, find_peaks
def hr_qc(x, fs):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < fs * 20: return np.nan, np.nan
    b, a = butter(3, [5, 20], btype='band', fs=fs); y = filtfilt(b, a, x)
    if abs(np.percentile(y, 1)) > abs(np.percentile(y, 99)): y = -y
    pk, _ = find_peaks(y, distance=int(0.33 * fs), height=np.percentile(y, 98) * 0.4)
    rr = np.diff(pk) / fs
    ok = (rr > 0.33) & (rr < 1.5)
    return (60 / np.median(rr[ok]) if ok.sum() > 10 else np.nan), ok.mean() if len(rr) else 0.0
R = os.environ.get('DATA_ROOT', os.path.expanduser('/path/to/project/data/raw'))
rows = []
for f in sorted(glob.glob(R + '/COGBCI/proc/sub-*/ses-*/*.npz')):
    d = np.load(f, allow_pickle=False)
    hr, frac = hr_qc(d['ecg'][0], float(d['ecg_sfreq']))
    rows.append(dict(dataset='COGBCI', pid=f.split('/')[-3], session=f.split('/')[-2], task=os.path.basename(f)[:-4], hr=hr, valid_rr_frac=frac))
for f in sorted(glob.glob(R + '/MPHRC/raw_parquet/*/*.parquet')):
    x = pd.read_parquet(f, columns=['ECG'])['ECG'].values
    hr, frac = hr_qc(x, 256.0)
    rows.append(dict(dataset='MPHRC', pid=f.split('/')[-2], session='', task=os.path.basename(f)[:-8], hr=hr, valid_rr_frac=frac))
q = pd.DataFrame(rows); q.to_csv(os.path.join(os.path.dirname(R), 'inventory', 'ECG_QC.csv'), index=False)
q['good'] = q.hr.between(45, 140) & (q.valid_rr_frac > 0.9)
print(q.groupby('dataset').agg(files=('good', 'size'), good=('good', 'mean'), hr_median=('hr', 'median')))
print(q[~q.good].groupby(['dataset', 'pid']).size().sort_values(ascending=False).head(12))
