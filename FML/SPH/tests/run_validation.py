#!/usr/bin/env python3
"""Run bounded sound/Sod/blast validation and optional MPI/thread replay in a fresh directory."""
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
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--mpi-executable',type=Path)
    parser.add_argument('--mpirun',default=os.environ.get('MPIRUN','mpirun'))
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.threads<1:parser.error('threads must be positive')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    exe=args.executable.resolve();mpi=args.mpi_executable.resolve() if args.mpi_executable else None
    source=Path(__file__).resolve().parent;repo=source.parents[2]
    files=list((repo/'FML/SPH').glob('*.h'))+[repo/'FML/Hydrodynamics/EquationOfState.h',source/'test_sph.cpp',source/'analyze.py',source/'run_validation.py',source.parent/'example/Main.cpp']
    files+=[exe]+([mpi] if mpi else [])
    (root/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    records=[];env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OMP_WAIT_POLICY='passive',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(root/'mplconfig'))
    def run(name,command):
        command=list(map(str,command));print(f'Running {name}',flush=True);start=time.monotonic()
        with (root/(name+'.log')).open('w') as log:p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        records.append({'name':name,'argv':command,'exit_code':p.returncode,'seconds':time.monotonic()-start,'threads':args.threads})
        (root/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
        if p.returncode:raise SystemExit(f'{name} failed; see log')
    for mode,ns,end,dt in [('sound1d',(64,128,256),.25,.01),('sound3d',(8,16),.15,.005),('sod',(225,450,900),.1,.005),('blast3d',(16,),.1,.002)]:
        for n in ns:
            name=f'{mode}-{n}';run(name,[exe,root/name,mode,n,end,dt])
    if mpi:
        launcher=shlex.split(args.mpirun)
        for n in (1,2,4):
            name=f'mpi{n}-sound3d-8';run(name,launcher+['-np',n,mpi,root/name,'sound3d',8,.15,.005])
    run('analysis',[sys.executable,source/'analyze.py','--input',root,'--output',root/'figures'])
    print(f'Completed; plots: {root / "figures"}')


if __name__=='__main__':main()
