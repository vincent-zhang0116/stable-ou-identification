"""Rebuild into a new tree. Input records are never replaced."""
from pathlib import Path
import sys,os,json,hashlib,subprocess,argparse,importlib.util
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--destination',required=True);p.add_argument('--fits',action='store_true');a=p.parse_args();d=Path(a.destination).resolve()
 assert not d.exists(),'Choose a new destination; existing files are protected.'
 for rel,h in json.loads((R/'SHA256SUMS.json').read_text()).items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==h,rel
 o=d/'outputs/Stable_OU_v0.20'
 for f in ['supplement','verification']:(o/f).mkdir(parents=True,exist_ok=True)
 spec=importlib.util.spec_from_file_location('transparency',R/'tools/restore_transparency.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.R=R;m.O=o;m.D=R/'outputs/Stable_OU_v0.18/numerical_research';m.audit()
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 if a.fits:
  subprocess.run([sys.executable,str(R/'outputs/Stable_OU_v0.18/numerical_research/replay_confirmation.py'),'--out',str(d/'confirmation'),'--workers','4'],check=True,env=env)
  st=R/'outputs/Stable_OU_Breakthrough_Exploration_2026-10-16'
  subprocess.run([sys.executable,str(st/'prepare_reproduction.py'),'--destination',str(d/'new_study')],check=True,env=env)
  fresh=d/'new_study/outputs'/st.name
  for f in ['simulation.py','cusp_pilot.py','cusp_confirmation.py']:subprocess.run([sys.executable,str(fresh/f)],check=True,env=env)
 print('Rebuilt outputs:',d)
if __name__=='__main__':main()
