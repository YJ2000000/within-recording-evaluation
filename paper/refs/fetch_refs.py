"""Fetch and format references from Crossref (DOIs) and the arXiv API (preprints). Output refs/refs.json keyed by short key."""
import json, re, time, urllib.request, urllib.parse, urllib.error, html
UA = {'User-Agent': 'xtask-refs/0.1'}
REFS = {
 'young2015': '10.1080/00140139.2014.956151', 'charles2019': '10.1016/j.apergo.2018.08.028', 'fairclough2009': '10.1016/j.intcom.2008.10.011',
 'villani2018': '10.1016/j.mechatronics.2018.02.009', 'gervasi2020': '10.1007/s00170-020-05363-1', 'kreibig2010': '10.1016/j.biopsycho.2010.03.010',
 'boucsein2012': '10.1007/978-1-4614-1126-0', 'bent2020': '10.1038/s41746-020-0226-6', 'schmidt2019': '10.3390/s19194079',
 'persstress2024': '10.3390/s24103221', 'saeb2017': '10.1093/gigascience/gix019', 'li2021': '10.1109/TPAMI.2020.2973153',
 'schroeder2025': '10.3389/fnrgo.2025.1582724', 'kapoor2023': '10.1016/j.patter.2023.100804', 'kahraman2026': '10.21203/rs.3.rs-10535611/v1',
 'hinss2023': '10.1038/s41597-022-01898-y', 'universe2024': '10.1038/s41597-024-03738-7', 'mphrc2025': '10.3390/robotics14120184',
 'gjoreski2020': '10.3390/app10113843', 'wesad2018': '10.1145/3242969.3242985', 'ds007554_2026': '10.1038/s41597-026-08125-y',
 'sensecobot2024': '10.1145/3610977.3636440', 'cogwear2023': '10.13026/5f6t-b637', 'clues2026': '10.1145/3831656', 'mocas2024': '10.1109/taffc.2024.3414330',
 'makowski2021': '10.3758/s13428-020-01516-y', 'hart1988': '10.1016/S0166-4115(08)62386-9', 'lakens2017': '10.1177/1948550617697177',
 'nosek2018': '10.1073/pnas.1708274114', 'pedregosa2011': None, 'seabold2010': '10.25080/Majora-92bf1922-011', 'goldberger2000': '10.1161/01.CIR.101.23.e215',
 'hhiss2025': '10.1145/3749465',
 'menghini2019': '10.1111/psyp.13441', 'schuurmans2020': '10.1007/s10916-020-01648-w', 'lipponen2019': '10.1080/03091902.2019.1640306',
 'shaffer2017': '10.3389/fpubh.2017.00258', 'kulic2007': '10.1109/TRO.2007.904899', 'arai2010': '10.1016/j.cirp.2010.03.043',
 'byrne1996': '10.1016/0301-0511(95)05161-9', 'heard2018': '10.1109/THMS.2017.2782483', 'spr2012': '10.1111/j.1469-8986.2012.01384.x',
 'charlton2022': '10.1109/JPROC.2022.3149785', 'reiss2019': '10.3390/s19143079', 'hopko2022': '10.3389/frobt.2022.799522',
 'landi2018': '10.1016/j.mechatronics.2018.07.012', 'munoz2015': '10.1371/journal.pone.0138921', 'vandooren2012': '10.1016/j.physbeh.2012.01.020',
 'matuz2021': '10.1371/journal.pone.0238670', 'white2023': '10.3390/s23136077', 'christensen2012': '10.1016/j.neuroimage.2011.07.091', 'brouwer2015': '10.3389/fnins.2015.00136',
}
ARXIV = {'maus2021': '2111.02561', 'normwear2024': '2412.09758', 'wang2024': '2405.00213', 'shi2026': '2609.30721', 'cogadapt2026': '2605.22774',
         'safeeda2026': '2610.01692', 'clare2024': '2404.17098', 'mphrc_arxiv': '2510.00703', 'mocas_arxiv': '2210.03065'}
def name(a):
    fam = a.get('family', ''); giv = a.get('given', '')
    ini = ''.join(p[0] + '.' for p in re.split(r'[\s\-]+', giv) if p) if giv else ''
    return f'{fam}, {ini}'.strip(', ') if fam else a.get('name', '')
import os
out = json.load(open('/path/to/project/paper/refs/refs.json')) if os.path.exists('/path/to/project/paper/refs/refs.json') else {}
out = {k: v for k, v in out.items() if 'error' not in v}
def get(url):
    for i in range(7):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15 * (i + 1)); continue
            raise
        except Exception:
            time.sleep(5 * (i + 1)); continue
    raise RuntimeError('rate limited: ' + url)
for k, doi in REFS.items():
    if not doi or k in out:
        continue
    try:
        j = json.loads(get('https://api.crossref.org/works/' + urllib.parse.quote(doi)))['message']
    except Exception as e:
        try:  # DataCite (PhysioNet, Zenodo)
            d = json.loads(get('https://api.datacite.org/dois/' + urllib.parse.quote(doi)))['data']['attributes']
            out[k] = dict(doi=doi, authors=[(c.get('familyName', '') + ', ' + ''.join(x[0] + '.' for x in c.get('givenName', '').split())).strip(', ') or c.get('name', '') for c in d['creators']],
                          title=d['titles'][0]['title'], journal=d.get('publisher', ''), year=d.get('publicationYear'), version=d.get('version', ''), type='dataset')
            continue
        except Exception as e2:
            out[k] = dict(error=f'{e} / {e2}', doi=doi); print(k, 'ERROR', e, e2); continue
    au = [name(a) for a in j.get('author', [])]
    yr = (j.get('issued', {}).get('date-parts') or [[None]])[0][0]
    out[k] = dict(doi=doi, authors=au, title=html.unescape(re.sub('<[^>]+>', '', (j.get('title') or [''])[0])), journal=(j.get('short-container-title') or j.get('container-title') or [''])[0],
                  journal_full=(j.get('container-title') or [''])[0], year=yr, volume=j.get('volume', ''), issue=j.get('issue', ''), page=j.get('page', '') or j.get('article-number', ''),
                  type=j.get('type'), publisher=j.get('publisher'))
    json.dump(out, open('/path/to/project/paper/refs/refs.json', 'w'), indent=1, ensure_ascii=False)
    time.sleep(1.0)
for k, aid in ARXIV.items():
    if k in out:
        continue
    x = get(f'http://export.arxiv.org/api/query?id_list={aid}').decode()
    ent = re.search(r'<entry>(.*?)</entry>', x, re.S).group(1)
    t = re.sub(r'\s+', ' ', re.search(r'<title>(.*?)</title>', ent, re.S).group(1)).strip()
    au = [re.sub(r'\s+', ' ', a).strip() for a in re.findall(r'<name>(.*?)</name>', ent)]
    def nm(full):
        p = full.split(); return f'{p[-1]}, ' + ''.join(q[0] + '.' for q in p[:-1])
    yr = re.search(r'<published>(\d{4})', ent).group(1)
    out[k] = dict(arxiv=aid, authors=[nm(a) for a in au], title=t, year=int(yr), type='preprint')
    json.dump(out, open('/path/to/project/paper/refs/refs.json', 'w'), indent=1, ensure_ascii=False)
    time.sleep(4)
out['pedregosa2011'] = dict(authors=['Pedregosa, F.', 'Varoquaux, G.', 'Gramfort, A.', 'Michel, V.', 'Thirion, B.', 'Grisel, O.', 'Blondel, M.', 'Prettenhofer, P.', 'Weiss, R.', 'Dubourg, V.'],
                            et_al=True, title='Scikit-learn: Machine Learning in Python', journal='J. Mach. Learn. Res.', year=2011, volume='12', page='2825–2830', type='journal-article')
out['holm1979'] = dict(authors=['Holm, S.'], title='A Simple Sequentially Rejective Multiple Test Procedure', journal='Scand. J. Stat.', year=1979, volume='6', page='65–70', type='journal-article')
json.dump(out, open('/path/to/project/paper/refs/refs.json', 'w'), indent=1, ensure_ascii=False)
for k, v in out.items():
    print(k, '|', v.get('year'), '|', (v.get('authors') or ['?'])[0], '|', v.get('title', v.get('error'))[:95], '|', v.get('journal', v.get('arxiv', '')))
