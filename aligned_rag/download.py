"""Explicit large-data downloads, invoked on the experiment server only."""
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from exp11_ablation import save_json
from research_integrity import file_sha256

BEIR_URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip'
DPR_URL = 'https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz'


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f'[download] retaining existing {path}; remove explicitly to replace')
        return
    partial = Path(str(path) + '.partial')
    with urllib.request.urlopen(url) as response, partial.open('wb') as out:
        shutil.copyfileobj(response, out, length=1024*1024)
    partial.replace(path)


def download_beir(dataset, destination):
    if dataset not in ('hotpotqa', 'nq'):
        raise ValueError('phase-one downloader supports hotpotqa and nq')
    dest = Path(destination); dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f'{dataset}.zip'
    download(BEIR_URL.format(dataset=dataset), archive)
    wanted = ['corpus.jsonl', 'queries.jsonl', 'qrels/test.tsv']
    with zipfile.ZipFile(archive) as z:
        for name in wanted:
            member = f'{dataset}/{name}'
            info = z.getinfo(member)
            if info.is_dir(): raise ValueError('unexpected dataset archive layout')
            out = dest / dataset / name
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(str(out)+'.partial', 'wb') as target:
                shutil.copyfileobj(src, target)
            Path(str(out)+'.partial').replace(out)
    save_json(dest / dataset / 'download.json', {'url': BEIR_URL.format(dataset=dataset),
              'archive_sha256': file_sha256(archive), 'members': wanted})
    print(f'[out] {dest / dataset}; full BEIR corpus, not a 200K subset')


def download_dpr(destination):
    dest = Path(destination)
    download(DPR_URL, dest)
    save_json(str(dest)+'.source.json', {'url': DPR_URL, 'sha256': file_sha256(dest)})
