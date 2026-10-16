from pathlib import Path
import hashlib,json,zipfile,datetime,ast
ROOT=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    assert json.loads((ROOT/'Verification.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'Reproduction_Smoke.json').read_text())['status']=='passed'
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    for item in json.loads((ROOT/'Dependency_Provenance.json').read_text()):
        assert sha(ROOT/item['copy'])==item['sha256']
    for name in ['Research_Report.md','Claim_Audit.md','Literature_and_Priority.md','Smooth_Noise_Identification.md','Cusp_Identification.md','Gaussian_Endpoint_Identified_Set.md','Nonuniform_Impossibility.md','Endpoint_Identified_Set.png','Cusp_Confirmation.png']:
        assert (ROOT/name).is_file(),name
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json' and '__pycache__' not in p.parts]
    sums={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    (ROOT/'SHA256SUMS.json').write_text(json.dumps(sums,indent=2),encoding='utf8');files.append(ROOT/'SHA256SUMS.json')
    archive=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in files:assert hashlib.sha256(z.read(ROOT.name+'/'+p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
    meta=dict(path=str(archive.resolve()),files=len(files),bytes=archive.stat().st_size,sha256=sha(archive),sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Completed exploratory proofs and numerical tests; originality and external review remain open. Frozen manuscript unchanged. Minimal dependencies included; full historical integrity audit requires earlier archives.')
    archive.with_suffix('.archive.json').write_text(json.dumps(meta,indent=2),encoding='utf8');print(json.dumps(meta,indent=2))
