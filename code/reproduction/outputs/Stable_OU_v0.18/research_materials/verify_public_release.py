"""Verify this curated public inventory and reconcile archived baseline summaries.

This script does not modify the original data or rerun the complete simulation.
"""
from pathlib import Path
import csv, hashlib, json, sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'original_v013/source'))
import experiment as ex

def main():
    manifest=json.loads((ROOT/'SHA256SUMS.json').read_text(encoding='utf8'))
    errors=[]
    for rel,wanted in manifest.items():
        p=(ROOT/rel).resolve()
        if not p.is_relative_to(ROOT):raise ValueError('Unsafe manifest path')
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=wanted:errors.append(rel)
    with (ROOT/'original_v013/results/pathwise.csv').open(newline='') as fp:
        rows=list(csv.DictReader(fp))
    actual=ex.summarize(rows)
    expected=json.loads((ROOT/'original_v013/results/summary.json').read_text())
    assert len(rows)==36000
    assert actual==expected,'The current numerical environment changed a summary; inspect numerical tolerances.'
    assert not errors,errors
    print(json.dumps({'public_manifest_files':len(manifest),'manifest_mismatches':errors,
                      'baseline_pathwise_rows':len(rows),'baseline_summary_scenarios':len(actual),
                      'baseline_summary_recalculation':'exact numerical agreement in current environment'},indent=2))

if __name__=='__main__':main()
