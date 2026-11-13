#!/usr/bin/env python3
"""Run bounded finite-volume fluid experiments and optional MPI replay, recording provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--executable',type=Path,required=True);p.add_argument('--mpi-executable',type=Path);p.add_argument('--mpirun',default=os.environ.get('MPIRUN','mpirun'));p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);exe=args.executable.resolve();mpi=args.mpi_executable.resolve() if args.mpi_executable else None
    source=Path(__file__).resolve().parent;files=list(source.parent.glob('*.h'))+[source/'analyze.py',source/'run_validation.py',source.parent/'example/Main.cpp',exe]+([mpi] if mpi else [])
    (root/'sha256.json').write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='2',OMP_WAIT_POLICY='passive',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(root/'mplconfig'));records=[]
    def run(name,cmd):
        print('Running '+name,flush=True);cmd=list(map(str,cmd));start=time.monotonic()
        with (root/(name+'.log')).open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        records.append({'name':name,'argv':cmd,'exit_code':result.returncode,'seconds':time.monotonic()-start,'threads':2});(root/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
        if result.returncode:raise SystemExit(name+' failed; see log')
    for mode,ns,end in [('advect1d',(64,128,256),.2),('advect2d',(24,48),.2),('advect3d',(16,32),.15),('sound',(64,128),.15),('sod',(128,256,512),.2),('lax',(256,),.13),('strong',(256,),.012),('rarefaction',(256,),.1),('vacuum',(256,),.03),('blast2d',(64,),.04),('blast3d',(16,),.02)]:
        for n in ns:run(f'{mode}-{n}',[exe,root/f'{mode}-{n}',mode,n,end])
    if mpi:
        launcher=shlex.split(args.mpirun)
        for ranks in (1,2,4):run(f'mpi{ranks}-advect2d-24',launcher+['-np',ranks,mpi,root/f'mpi{ranks}-advect2d-24','advect2d',24,.2])
        for mode,n,end in [('sod',128,.2),('blast3d',16,.02)]:run(f'mpi2-{mode}-{n}',launcher+['-np',2,mpi,root/f'mpi2-{mode}-{n}',mode,n,end])
    run('analysis',[sys.executable,source/'analyze.py','--input',root,'--output',root/'figures'])
    print('Completed; figures: '+str(root/'figures'))

if __name__=='__main__':main()
