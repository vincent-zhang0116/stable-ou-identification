from pathlib import Path
import argparse,shutil,json,hashlib
ROOT=Path(__file__).parent
def prepare(destination):
    dest=Path(destination).resolve()
    assert not dest.exists() or not any(dest.iterdir()),'Destination must be new or empty'
    dest.mkdir(parents=True,exist_ok=True)
    for item in json.loads((ROOT/'Dependency_Provenance.json').read_text(encoding='utf8')):
        source=ROOT/item['copy'];assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
        target=dest/item['reproduction_path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    fresh=dest/'outputs'/ROOT.name;fresh.mkdir()
    for source in ROOT.iterdir():
        if source.suffix=='.py' or source.name.endswith('protocol.json') or source.name.endswith('freeze.json'):shutil.copy2(source,fresh/source.name)
    return fresh
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',required=True)
    print(prepare(p.parse_args().destination))
