"""Stage the exact historical execution source in a fresh output tree."""
from pathlib import Path
import argparse,shutil,subprocess,sys
def main():
    R=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();out=Path(a.out).resolve()
    if out.exists():raise SystemExit('Choose a new output directory; existing evidence is never overwritten.')
    for folder in ['analysis','original_v013/source','new_experiments']:(out/folder).mkdir(parents=True,exist_ok=True)
    shutil.copy2(R/'reproducibility/diagnostic_execution_source.py',out/'analysis/new_experiments.py');shutil.copy2(R/'research_materials/original_v013/source/experiment.py',out/'original_v013/source/experiment.py');shutil.copy2(R/'research_materials/new_experiments/config.json',out/'new_experiments/config.json')
    subprocess.run([sys.executable,str(out/'analysis/new_experiments.py'),'--workers',str(a.workers)],check=True)
    print('New diagnostic outputs:',out/'new_experiments')
if __name__=='__main__':main()
