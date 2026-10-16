from pathlib import Path
import json,hashlib
ROOT=Path(__file__).parent
if __name__=='__main__':
    checks=json.loads((ROOT/'SHA256SUMS.json').read_text(encoding='utf8'))
    for rel,digest in checks.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    print(json.dumps(dict(status='passed',files=len(checks),scope='File integrity only, not mathematical or originality certification.')))
