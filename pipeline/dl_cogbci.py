"""COG-BCI (Zenodo 7413650): per subject, fetch each session's .set/.fdt via HTTP range,
keep ECG (500 Hz, float32), 19 standard 10-20 EEG channels resampled to 200 Hz (float16),
event annotations and behavioural .mat files; delete the raw EEGLAB files afterwards.
Output: /path/to/project/data/raw/COGBCI/proc/sub-XX/ses-SY/<task>.npz  (load with allow_pickle=False)
Resumable per task file.
"""
import os, sys, time, shutil, tempfile, json
import numpy as np
from remotezip import RemoteZip
import mne
mne.set_log_level('ERROR')

OUT = os.path.join(os.environ.get('DATA_ROOT', os.path.expanduser('/path/to/project/data/raw')), 'COGBCI', 'proc')
URL = 'https://zenodo.org/records/7413650/files/sub-{:02d}.zip?download=1'
EEG19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def process_subject(s):
    url = URL.format(s)
    for attempt in range(30):
        try:
            z = RemoteZip(url); names = [i.filename for i in z.infolist()]; break
        except Exception as e:
            log('retry open', s, e); time.sleep(65 if '429' in str(e) else 10)
    sets = [n for n in names if n.endswith('.set')]
    mats = [n for n in names if n.endswith('.mat') or n.endswith('.txt')]
    for n in mats:
        out = os.path.join(OUT, *n.split('/')[-4:])
        if not os.path.exists(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            for attempt in range(30):
                try:
                    open(out, 'wb').write(z.read(n)); break
                except Exception as e:
                    log('retry', n, e); time.sleep(65 if '429' in str(e) else 10)
    for n in sets:
        sub, ses, _, fn = n.split('/')[-4:]
        task = fn[:-4]
        out = os.path.join(OUT, sub, ses, task + '.npz')
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = tempfile.mkdtemp(dir=os.path.dirname(OUT))
        try:
            for m in (n, n[:-4] + '.fdt'):
                for attempt in range(30):
                    try:
                        open(os.path.join(tmp, os.path.basename(m)), 'wb').write(z.read(m)); break
                    except Exception as e:
                        log('retry', m, e); time.sleep(65 if '429' in str(e) else 10 * (attempt % 5 + 1))
                        try:
                            z = RemoteZip(url)
                        except Exception as e2:
                            log('reopen failed', e2)
            raw = mne.io.read_raw_eeglab(os.path.join(tmp, fn), preload=True)
            sf = raw.info['sfreq']
            ch = raw.ch_names
            ecg_name = [c for c in ch if c.upper().startswith('ECG')]
            ecg = raw.get_data(picks=ecg_name).astype(np.float32) if ecg_name else np.zeros((0, raw.n_times), np.float32)
            eeg_avail = [c for c in EEG19 if c in ch]
            eeg = raw.copy().pick(eeg_avail).resample(200.0).get_data() * 1e6  # microvolts
            ann = raw.annotations
            np.savez_compressed(out + '.part.npz',
                                ecg=ecg, ecg_sfreq=np.float32(sf), ecg_ch=np.array(ecg_name, dtype='U16'),
                                eeg=eeg.astype(np.float16), eeg_sfreq=np.float32(200.0), eeg_ch=np.array(eeg_avail, dtype='U8'),
                                ev_onset=np.asarray(ann.onset, np.float64), ev_dur=np.asarray(ann.duration, np.float64),
                                ev_desc=np.array([str(d) for d in ann.description], dtype='U32'),
                                n_times=np.int64(raw.n_times), all_ch=np.array(ch, dtype='U16'))
            os.replace(out + '.part.npz', out)
        except Exception as e:
            log('ERROR', n, repr(e))
            json.dump({'file': n, 'error': repr(e)}, open(out + '.error.json', 'w'))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    log('done subject', s)

if __name__ == '__main__':
    subs = [int(x) for x in sys.argv[1:]] or list(range(1, 30))
    for s in subs:
        process_subject(s)
    log('ALL DONE COGBCI')
