"""Replay the frozen research algorithm in a new tree; never replace primary rows."""
from pathlib import Path
import argparse,shutil,subprocess,sys
def main():
    R=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();out=Path(a.out).resolve()
    if out.exists():raise SystemExit('Choose a new output directory; primary confirmation is immutable.')
    for f in ['numerical_research','research_materials/analysis','research_materials/original_v013/source']:(out/f).mkdir(parents=True,exist_ok=True)
    for name in ['benchmark.py','design.json','confirmation_freeze.json','PROTOCOL.md']:
        shutil.copy2(R/'numerical_research'/name,out/'numerical_research'/name)
    for name in ['analysis/new_experiments.py','original_v013/source/experiment.py']:
        shutil.copy2(R/'research_materials'/name,out/'research_materials'/name)
    subprocess.run([sys.executable,str(out/'numerical_research/benchmark.py'),'--stage','confirmation','--workers',str(a.workers)],check=True)
    print('Fresh confirmation outputs:',out/'numerical_research/confirmation')
if __name__=='__main__':main()
