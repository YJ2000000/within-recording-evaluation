"""Plan-B downloads: ds007554 (ECG physio + metadata from OpenNeuro S3), CogWear (PhysioNet zip), WESAD (sciebo zip, keep only S*.pkl, quest, readme)."""
import os, re, sys, urllib.request, urllib.parse, zipfile, shutil, time
ROOT = '/path/to/project/data/raw2'
UA = {'User-Agent': 'xtask-data/0.1'}
def fetch(url, out, tries=5):
    if os.path.exists(out) and os.path.getsize(out) > 0: return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(out + '.part', 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(out + '.part', out); return
        except Exception as e:
            print('retry', url, e, flush=True); time.sleep(10 * (i + 1))
    raise RuntimeError(url)
def ds007554():
    keys, token = [], None
    while True:
        q = {'list-type': '2', 'prefix': 'ds007554/', 'max-keys': '1000'}
        if token: q['continuation-token'] = token
        x = urllib.request.urlopen(urllib.request.Request('https://s3.amazonaws.com/openneuro.org?' + urllib.parse.urlencode(q), headers=UA), timeout=60).read().decode()
        keys += list(zip(re.findall(r'<Key>(.*?)</Key>', x), map(int, re.findall(r'<Size>(\d+)</Size>', x))))
        m = re.search(r'<NextContinuationToken>(.*?)</NextContinuationToken>', x)
        if not m: break
        token = m.group(1)
    want = [(k, s) for k, s in keys if ('/beh/' in k and ('recording-ECG' in k or 'recording-biodex' in k or 'pushbutton' in k)) or k.endswith('scans.tsv')
            or '/phenotype/' in k or k.split('/')[-1] in ('participants.tsv', 'participants.json', 'README', 'dataset_description.json', 'CHANGES')]
    print('ds007554 keys', len(keys), 'selected', len(want), 'MB %.0f' % (sum(s for _, s in want) / 1e6), flush=True)
    for k, s in want:
        fetch('https://s3.amazonaws.com/openneuro.org/' + urllib.parse.quote(k), os.path.join(ROOT, 'DS007554', k[len('ds007554/'):]))
    print('ds007554 done', flush=True)
def cogwear():
    z = os.path.join(ROOT, 'COGWEAR', 'cogwear.zip')
    fetch('https://physionet.org/content/consumer-grade-wearables/get-zip/1.0.0/', z)
    zipfile.ZipFile(z).extractall(os.path.join(ROOT, 'COGWEAR')); os.remove(z)
    print('cogwear done', flush=True)
def wesad():
    z = os.path.join(ROOT, 'WESAD', 'WESAD.zip')
    fetch('https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download', z)
    with zipfile.ZipFile(z) as Z:
        for n in Z.namelist():
            b = n.split('/')[-1]
            if re.fullmatch(r'S\d+\.pkl', b) or b.endswith('_quest.csv') or b.endswith('_readme.txt') or 'readme' in b.lower() or b.endswith('.pdf'):
                Z.extract(n, os.path.join(ROOT, 'WESAD'))
    os.remove(z); print('wesad done', flush=True)
if __name__ == '__main__':
    for name in sys.argv[1:]:
        {'ds007554': ds007554, 'cogwear': cogwear, 'wesad': wesad}[name]()
