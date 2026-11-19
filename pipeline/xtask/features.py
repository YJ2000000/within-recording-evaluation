"""Window-level physiological features shared by all datasets.

Every recording segment is processed once (filtering, beat detection), then cut into
fixed windows (default 30 s, 10 s step). Features are deliberately standard and
device-agnostic so that the same columns exist across datasets where the sensor exists.
"""
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, sosfiltfilt, find_peaks, resample_poly

WIN = 30.0
STEP = 10.0
IBI_MIN, IBI_MAX = 0.33, 1.5  # 40-180 bpm


def _windows(dur, win=WIN, step=STEP):
    t = 0.0
    while t + win <= dur + 1e-9:
        yield t, t + win
        t += step


def beats_from_ecg(x, fs):
    """R-peak times (s) from a raw ECG trace. Handles inverted polarity and downsamples to <=250 Hz."""
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nanmedian(x))
    if fs > 250:
        up, down = 250, int(round(fs))
        x = resample_poly(x, up, down); fs = 250.0
    try:
        x, _ = nk.ecg_invert(x, sampling_rate=int(fs))
    except Exception:
        pass
    clean = nk.ecg_clean(x, sampling_rate=int(fs), method='neurokit')
    _, info = nk.ecg_peaks(clean, sampling_rate=int(fs), method='neurokit', correct_artifacts=True)
    return np.asarray(info['ECG_R_Peaks'], float) / fs


def beats_from_ppg(x, fs):
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nanmedian(x))
    clean = nk.ppg_clean(x, sampling_rate=int(fs), method='elgendi')
    _, info = nk.ppg_peaks(clean, sampling_rate=int(fs), method='elgendi', correct_artifacts=True)
    return np.asarray(info['PPG_Peaks'], float) / fs


def cardiac_window(beats, t0, t1):
    b = beats[(beats >= t0) & (beats < t1)]
    out = dict(hr_mean=np.nan, hr_std=np.nan, rmssd=np.nan, sdnn=np.nan, pnn50=np.nan, ibi_valid=0.0)
    if len(b) < 4:
        return out
    ibi = np.diff(b)
    ok = (ibi > IBI_MIN) & (ibi < IBI_MAX)
    # successive-difference artifact rule (>20% change from local median)
    med = np.median(ibi[ok]) if ok.any() else np.nan
    ok &= np.abs(ibi - med) < 0.3 * med
    expected = (t1 - t0) / med if np.isfinite(med) else np.inf
    out['ibi_valid'] = float(ok.sum() / expected) if np.isfinite(expected) and expected > 0 else 0.0
    ibi = ibi[ok]
    if len(ibi) < 5:
        return out
    hr = 60.0 / ibi
    d = np.diff(ibi)
    out.update(hr_mean=hr.mean(), hr_std=hr.std(), rmssd=1000 * np.sqrt(np.mean(d ** 2)),
               sdnn=1000 * ibi.std(), pnn50=float(np.mean(np.abs(d) > 0.05)))
    return out


def _lowpass(x, fs, fc):
    sos = butter(2, fc, btype='low', fs=fs, output='sos')
    return sosfiltfilt(sos, x)


def eda_prepare(x, fs):
    """Resample EDA to 4 Hz, smooth (1 Hz low-pass), split tonic (0.05 Hz) and phasic."""
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nanmedian(x))
    if fs != 4:
        x = resample_poly(x, 4, int(round(fs))) if fs > 4 else np.interp(np.arange(0, len(x) / fs, 0.25), np.arange(len(x)) / fs, x)
    if len(x) < 40:
        return None
    x = _lowpass(x, 4.0, 1.0)
    tonic = _lowpass(x, 4.0, 0.05)
    return dict(x=x, tonic=tonic, phasic=x - tonic, fs=4.0)


def eda_window(e, t0, t1):
    out = dict(scl_mean=np.nan, scl_slope=np.nan, phasic_std=np.nan, scr_rate=np.nan)
    if e is None:
        return out
    i0, i1 = int(t0 * 4), int(t1 * 4)
    if i1 > len(e['x']) or i1 - i0 < 20:
        return out
    ton, ph = e['tonic'][i0:i1], e['phasic'][i0:i1]
    tt = np.arange(len(ton)) / 4.0
    amp = np.nanstd(e['x']) if np.nanstd(e['x']) > 0 else 1.0
    pk, _ = find_peaks(ph, prominence=0.05 * amp, distance=4)
    out.update(scl_mean=float(ton.mean()), scl_slope=float(np.polyfit(tt, ton, 1)[0] * 60),
               phasic_std=float(ph.std()), scr_rate=len(pk) / ((t1 - t0) / 60))
    return out


def simple_window(x, fs, t0, t1, prefix):
    """Mean and slope (per minute) of a slow signal such as skin temperature."""
    i0, i1 = int(t0 * fs), int(t1 * fs)
    seg = np.asarray(x[i0:i1], float)
    seg = seg[np.isfinite(seg)]
    if len(seg) < max(3, 0.5 * (i1 - i0)):
        return {f'{prefix}_mean': np.nan, f'{prefix}_slope': np.nan}
    tt = np.arange(len(seg)) / fs
    return {f'{prefix}_mean': float(seg.mean()), f'{prefix}_slope': float(np.polyfit(tt, seg, 1)[0] * 60)}


def acc_window(mag, fs, t0, t1):
    seg = np.asarray(mag[int(t0 * fs):int(t1 * fs)], float)
    seg = seg[np.isfinite(seg)]
    return {'acc_std': float(seg.std()) if len(seg) > 3 else np.nan}


def resp_window(r, fs, t0, t1):
    out = dict(resp_rate=np.nan, resp_amp=np.nan)
    seg = np.asarray(r[int(t0 * fs):int(t1 * fs)], float)
    if len(seg) < fs * 10 or not np.isfinite(seg).all():
        return out
    sos = butter(2, [0.1, 0.7], btype='band', fs=fs, output='sos')
    y = sosfiltfilt(sos, seg)
    pk, _ = find_peaks(y, distance=int(1.5 * fs), prominence=0.3 * y.std())
    out.update(resp_rate=len(pk) / ((t1 - t0) / 60), resp_amp=float(y.std()))
    return out


def window_table(dur, beats=None, eda=None, temp=None, acc=None, resp=None, win=WIN, step=STEP):
    """Return a list of dicts, one per window, with every available feature family."""
    rows = []
    for t0, t1 in _windows(dur, win, step):
        r = dict(t0=t0, t1=t1)
        if beats is not None:
            r.update(cardiac_window(beats, t0, t1))
        if eda is not None:
            r.update(eda_window(eda, t0, t1))
        if temp is not None:
            r.update(simple_window(temp[0], temp[1], t0, t1, 'temp'))
        if acc is not None:
            r.update(acc_window(acc[0], acc[1], t0, t1))
        if resp is not None:
            r.update(resp_window(resp[0], resp[1], t0, t1))
        rows.append(r)
    return rows
