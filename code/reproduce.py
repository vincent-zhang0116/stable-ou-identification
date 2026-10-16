"""Verify the archive and rebuild article displays into a new directory."""
from pathlib import Path
import sys,os,argparse,csv,json,hashlib,subprocess,math
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
def rows(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def write(path,data):
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def displays(source,destination):
    import numpy as np
    old=source/'outputs/Stable_OU_v0.18'
    base=rows(old/'research_materials/original_v013/results/pathwise.csv')
    table1=[]
    for noise in ['clean','cauchy','strong_gaussian','uniform_zero']:
        for n in [2000,8000,32000,128000]:
            selected=[x for x in base if x['group']=='central' and x['design']=='adaptive' and x['noise']==noise and int(x['n'])==n]
            N=len(selected);assert N==500
            hac=sum(int(x['hac_covered']) for x in selected);batch=sum(int(x['batch_covered']) for x in selected)
            assert all(int(x['hac_issued'])==int(x['batch_issued'])==1 for x in selected)
            z=1.959963984540054;p=batch/N;den=1+z*z/N;mid=(p+z*z/(2*N))/den;rad=z*math.sqrt(p*(1-p)/N+z*z/(4*N*N))/den
            table1.append(dict(noise=noise,n=n,N=N,HAC_covered=hac,batch_covered=batch,batch_Wilson_low=mid-rad,batch_Wilson_high=mid+rad))
    confirmation=rows(old/'numerical_research/confirmation/central_pathwise.csv')
    table2=[]
    for noise in ['clean','cauchy','strong_gaussian','uniform_zero']:
        for method in ['three_shared','nine_md','thirtysix_md','three_native']:
            selected=[x for x in confirmation if x['group']=='central' and x['noise']==noise and x['n']=='8000' and x['method']==method];assert len(selected)==500
            returned=[x for x in selected if x['success']=='1'];assert returned
            record=dict(noise=noise,method=method,N=500,returned=len(returned),boundary=sum(x['boundary'] in ['1','True','true'] for x in selected),loss=float(np.mean([float(x['joint_loss']) for x in selected])))
            for par in ['k','a','c']:record['RMSE_'+par]=float(np.sqrt(np.mean([float(x['error_'+par])**2 for x in returned])))
            table2.append(record)
    write(destination/'Table1.csv',table1);write(destination/'Table2.csv',table2)
    return dict(table1_rows=len(table1),table2_rows=len(table2))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',required=True);p.add_argument('--fits',action='store_true',help='rerun all 6050 recorded latent paths as well as reconstructing displays');a=p.parse_args()
    dest=Path(a.destination).resolve();assert not dest.exists(),'Choose a new destination; existing data are never overwritten.'
    for name,digest in json.loads((ROOT/'SHA256SUMS.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    dest.mkdir();material=ROOT/'reproduction'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,str(material/'rebuild.py'),'--destination',str(dest/'reconstructed')],env=env,check=True)
    display=dest/'article_displays';display.mkdir();report=displays(material,display)
    import shutil
    derived=dest/'reconstructed/outputs/Stable_OU_v0.20/supplement'
    for frozen in (ROOT/'tables').glob('*.csv'):
        assert frozen.read_bytes()==(derived/frozen.name).read_bytes(),frozen.name
    for name in ['Dual_Baseline_Figure_Data.csv','Figure5_Dual_Baseline.tex']:shutil.copy2(derived/name,display/('Figure2.csv' if name.endswith('csv') else 'Figure2.tex'))
    if a.fits:
        old=material/'outputs/Stable_OU_v0.18'
        subprocess.run([sys.executable,str(material/'rebuild.py'),'--destination',str(dest/'later_rerun'),'--fits'],env=env,check=True)
        subprocess.run([sys.executable,str(old/'research_materials/original_v013/source/experiment.py'),'--config',str(old/'research_materials/original_v013/config.json'),'--out',str(dest/'baseline_rerun'),'--workers','4'],env=env,check=True)
        subprocess.run([sys.executable,str(old/'reproducibility/replay_diagnostics.py'),'--out',str(dest/'diagnostic_rerun'),'--workers','2'],env=env,check=True)
    (dest/'Rebuild_Report.json').write_text(json.dumps(dict(**report,all_fits_rerun=a.fits),indent=2),encoding='utf8')
    print('Rebuilt:',dest)
if __name__=='__main__':main()
