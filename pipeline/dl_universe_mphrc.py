"""Selective download of UNIVERSE (Zenodo 10371068) and MultiPhysio-HRC (Zenodo 18668043).
Uses HTTP range requests (remotezip) to pull only the members we need.
- UNIVERSE: Empatica raw CSVs (lab + wild), PsychoPy logs, stretched_empatica (lab), Task_Labels, notes.
  Muse EEG CSVs and all .pickle files are skipped (pickles are not loaded for security; EEG is optional).
- MultiPhysio-HRC: raw/ CSVs -> parquet float32 (original CSV deleted), plus features.zip.
Resumable: skips outputs that already exist.
"""
import os, sys, io, time, threading, queue, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from remotezip import RemoteZip
import pandas as pd, numpy as np

ROOT = os.environ.get('DATA_ROOT', os.path.expanduser('/path/to/project/data/raw'))
UNIV = ['https://zenodo.org/records/10371068/files/UNIVERSE_UN_101_to_UN_112.zip?download=1',
        'https://zenodo.org/records/10371068/files/UNIVERSE_UN_113_to_UN_124.zip?download=1']
MPHRC = 'https://zenodo.org/records/18668043/files/physiological_data.zip?download=1'
MPHRC_FEAT = 'https://zenodo.org/records/18668043/files/features.zip?download=1'
log_lock = threading.Lock()
def log(*a):
    with log_lock:
        print(time.strftime('%H:%M:%S'), *a, flush=True)

def want_universe(f):
    if f.endswith('/') or f.endswith('.pickle'):
        return False
    if '/Muse/' in f or 'stretched_muse' in f:
        return False
    return ('/Empatica/' in f or '/Psychopy/' in f or 'stretched_empatica' in f
            or f.endswith('Task_Labels.csv') or f.endswith('.pdf') or f.endswith('.txt'))


_rate_lock = threading.Lock(); _last = [0.0]
MIN_INTERVAL = float(os.environ.get('MIN_REQ_INTERVAL', '1.5'))  # seconds between requests in this process (Zenodo guest rate limit)
def throttle():
    with _rate_lock:
        wait = _last[0] + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()

def backoff(e, attempt):
    """Zenodo returns 429 when rate-limited: wait a full minute; other errors: short linear backoff."""
    time.sleep(65 if '429' in str(e) else 5 * (attempt + 1))

_local = threading.local()
def rz(url):
    d = getattr(_local, 'z', None)
    if d is None:
        d = _local.z = {}
    if url not in d:
        d[url] = RemoteZip(url)
    return d[url]

def fetch_member(url, name, out):
    if os.path.exists(out):
        return 'skip'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(30):
        try:
            throttle()
            data = rz(url).read(name)
            tmp = out + '.part'
            with open(tmp, 'wb') as fh:
                fh.write(data)
            os.replace(tmp, out)
            return len(data)
        except Exception as e:
            log('retry', name, e)
            _local.z = None
            backoff(e, attempt)
    raise RuntimeError('failed ' + name)

def mphrc_member(url, name):
    sub, fn = name.split('/')[-2], name.split('/')[-1]
    out = f'{ROOT}/MPHRC/raw_parquet/{sub}/{fn[:-4]}.parquet'
    if os.path.exists(out):
        return 'skip'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(30):
        try:
            throttle()
            data = rz(url).read(name)
            df = pd.read_csv(io.BytesIO(data))
            df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed') or c == ''])
            for c in df.columns:
                if c not in ('ID', 'Class', 'Repetition'):
                    df[c] = df[c].astype(np.float32)
            df.to_parquet(out + '.part', index=False)
            os.replace(out + '.part', out)
            return len(data)
        except Exception as e:
            log('retry', name, e)
            _local.z = None
            backoff(e, attempt)
    raise RuntimeError('failed ' + name)

def run_pool(jobs, fn, workers=int(os.environ.get('DL_WORKERS', '3')), label=''):
    done, tot_bytes, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fn, *j) for j in jobs]
        for f in as_completed(futs):
            r = f.result(); done += 1
            if isinstance(r, int):
                tot_bytes += r
            if done % 50 == 0 or done == len(jobs):
                log(f'{label} {done}/{len(jobs)} files, {tot_bytes/1e6:.0f} MB, {tot_bytes/1e6/max(1,time.time()-t0):.2f} MB/s')

if __name__ == '__main__':
    what = sys.argv[1:] or ['universe', 'mphrc']
    if 'universe' in what:
        for url in UNIV:
            for attempt in range(30):
                try:
                    with RemoteZip(url) as z:
                        names = [i.filename for i in z.infolist() if want_universe(i.filename)]
                    break
                except Exception as e:
                    log('retry listing', e); backoff(e, attempt)
            log('UNIVERSE', url.split('/')[-1][:30], len(names), 'members')
            run_pool([(url, n, f'{ROOT}/{n}') for n in names], fetch_member, label='UNIVERSE')
    if 'mphrc' in what:
        os.makedirs(f'{ROOT}/MPHRC', exist_ok=True)
        out = f'{ROOT}/MPHRC/features.zip'
        if not os.path.exists(out):
            urllib.request.urlretrieve(MPHRC_FEAT, out + '.part'); os.replace(out + '.part', out)
            log('MPHRC features.zip done')
        import zipfile
        with zipfile.ZipFile(out) as zf:  # labels + physiological features are needed downstream
            for member in ('features/labels.csv', 'features/bio_features_60s.csv'):
                if not os.path.exists(f'{ROOT}/MPHRC/{member}'):
                    zf.extract(member, f'{ROOT}/MPHRC')
        for extra in ('README.md', 'participants_task_overview.csv', 'features_table.pdf'):
            if not os.path.exists(f'{ROOT}/MPHRC/{extra}'):
                urllib.request.urlretrieve(f'https://zenodo.org/records/18668043/files/{extra}?download=1', f'{ROOT}/MPHRC/{extra}')
        for attempt in range(30):
            try:
                with RemoteZip(MPHRC) as z:
                    names = [i.filename for i in z.infolist() if i.filename.startswith('physiological_data/raw/') and i.filename.endswith('.csv')]
                break
            except Exception as e:
                log('retry listing', e); backoff(e, attempt)
        log('MPHRC raw members', len(names))
        run_pool([(MPHRC, n) for n in names], mphrc_member, label='MPHRC')
    log('ALL DONE', what)
