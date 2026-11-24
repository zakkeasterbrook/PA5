"""Reproduce the ACB tests without changing any archived research evidence."""
import argparse,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refit',action='store_true',help='also rerun the numerical multistage LP at the one-point loss tolerance')
    args=parser.parse_args()
    import numpy as np
    import scipy
    if (np.__version__,scipy.__version__)!=('2.3.5','1.17.0'):
        raise RuntimeError('Install the versions in requirements.txt for recorded reproduction')
    env=dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    with tempfile.TemporaryDirectory(prefix='acb-reproduce-') as tmp:
        scratch=Path(tmp)
        for name in ('barrier.py','exact_policy.py','robust.py','verify_archived.py','metr_points.csv','results.json'):
            shutil.copyfile(ROOT/name,scratch/name)
        def run(script,*opts):
            return subprocess.run([sys.executable,str(scratch/script),*opts],cwd=scratch,env=env,check=True,text=True,capture_output=True).stdout
        print('Running exact moment, witness and policy checks ...',flush=True)
        run('barrier.py')
        recorded=json.loads((ROOT/'barrier_results.json').read_text());fresh=json.loads((scratch/'barrier_results.json').read_text())
        assert recorded['exact']==fresh['exact'] and recorded['data']==fresh['data']
        for k,v in fresh['independent'].items():
            if k.endswith('_checks'):assert v==recorded['independent'][k]
        assert fresh['independent']['max_joint_LP_absolute_error']<1e-8
        for a,b in zip(recorded['examples'],fresh['examples']):
            assert {k:v for k,v in a.items() if k!='numerical_optimum'}=={k:v for k,v in b.items() if k!='numerical_optimum'}
            assert abs(a['numerical_optimum']-b['numerical_optimum'])<1e-8
        print('Running exact optimizer and independently formulated policy LP checks ...',flush=True)
        run('exact_policy.py')
        recorded2=json.loads((ROOT/'exact_policy_results.json').read_text());fresh2=json.loads((scratch/'exact_policy_results.json').read_text())
        assert recorded2['counts']==fresh2['counts'] and recorded2['examples']==fresh2['examples']
        assert fresh2['max_independent_LP_error']<1e-8
        print('Checking archived multistage certificates in exact arithmetic ...',flush=True)
        verified=json.loads(run('verify_archived.py'))
        assert verified==json.loads((ROOT/'archive_verification.json').read_text())
        if args.refit:
            print('Refitting the multistage candidate LPs (solver-dependent) ...',flush=True)
            run('robust.py','--quick')
            new=json.loads(run('verify_archived.py'))
            assert len(new['archived_multistage_certificates'])==2
        print(json.dumps({'status':'pass','exact_checks':sum(fresh['exact'].values()),'independently_formulated_LP_comparisons':sum(v for k,v in fresh['independent'].items() if k.endswith('_checks'))+fresh2['counts']['independent_exact_optimizer_comparisons'],'checkpoint_pairs':sum(r['pairs'] for r in fresh['data']),'archived_certificates_verified':len(verified['archived_multistage_certificates']),'refit_completed':args.refit},indent=2))

if __name__=='__main__':main()
