#!/usr/bin/env python3
"""Independent NumPy fits and held-out checks for the matched ZA/2LPT/PM workflow."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import numpy as np


def load(prefix):
    return (np.genfromtxt(str(prefix)+'-rows.csv', names=True, delimiter=',', dtype=None, encoding='utf8'),
            np.genfromtxt(str(prefix)+'-results.csv', names=True, delimiter=',', dtype=None, encoding='utf8'))


def check(prefix, tolerance=5e-10):
    rows, actual = load(prefix)
    nbin = 3
    def block(role, name):
        a=rows[rows['role']==role]
        assert len(a)%nbin==0 and np.array_equal(a['bin'],np.tile(np.arange(nbin),len(a)//nbin))
        return a[name].reshape(-1,nbin)
    keys={role:rows[(rows['role']==role)&(rows['bin']==0)]['seed'] for role in ['train','mean','eval']}
    allkeys=np.concatenate(list(keys.values()))
    assert len(np.unique(allkeys))==len(allkeys)
    assert np.all(rows['modes']>0)
    y=block('train','target');ye=block('eval','target')
    raw=np.cov(ye,rowvar=False,ddof=1);metrics=[];references={}
    for control in ['za','lpt2']:
        c=block('train',control);cm=block('mean',control);ce=block('eval',control)
        joint=np.cov(np.column_stack([y,c]),rowvar=False,ddof=1)
        cc=joint[nbin:,nbin:];yc=joint[:nbin,nbin:];scale=np.sqrt(np.diag(cc))
        correlation=cc/scale[:,None]/scale[None,:]
        condition=np.linalg.cond(correlation)
        for mode in ['diagonal','full']:
            model=('ZA' if control=='za' else '2LPT')+'-'+mode
            # Match the declared default ridge in standardized-control units.
            beta=(np.diag(np.diag(yc)/np.diag(cc)/(1+1e-8)) if mode=='diagonal' else
                  np.linalg.solve(correlation+1e-8*np.eye(nbin),(yc/scale).T).T/scale)
            z=ye-(ce-cm.mean(axis=0)) @ beta.T
            covariance=np.cov(z,rowvar=False,ddof=1)
            floor=beta @ (np.cov(cm,rowvar=False,ddof=1)/len(cm)) @ beta.T
            reference=dict(beta=beta,mean=z.mean(axis=0),covariance=covariance,
                           covariance_of_mean=covariance/len(ye)+floor,mean_floor=floor,
                           raw_mean=ye.mean(axis=0),raw_covariance=raw)
            max_error=0
            for quantity,value in reference.items():
                estimate=actual[(actual['model']==model)&(actual['quantity']==quantity)]['value'].reshape(value.shape)
                error=np.max(np.abs(estimate-value))/max(1.,np.max(np.abs(value)))
                assert error<tolerance,(model,quantity,error,tolerance)
                max_error=max(max_error,error);references[model+'-'+quantity]=value
            metrics.append(dict(model=model,ntrain=len(y),nmean=len(cm),neval=len(ye),
                numpy_max_scaled_error=float(max_error),standardized_control_condition=float(condition),
                residual_variance_reduction=(np.diag(raw)/np.diag(covariance)).tolist(),
                mean_variance_reduction=(np.diag(raw)/len(ye)/np.diag(reference['covariance_of_mean'])).tolist(),
                mean_uncertainty_floor_fraction=(np.diag(floor)/np.diag(reference['covariance_of_mean'])).tolist(),
                training_power_correlation=(np.diag(yc)/np.sqrt(np.diag(joint[:nbin,:nbin])*np.diag(cc))).tolist()))
    np.savez(str(prefix)+'-numpy-reference.npz',**references)
    times=np.genfromtxt(str(prefix)+'-timing.csv',names=True,delimiter=',',dtype=None,encoding='utf8')
    timing={role:dict(count=int(np.count_nonzero(times['role']==role)),
                     sum_seconds=float(times[times['role']==role]['total'].sum()),
                     median_seconds=float(np.median(times[times['role']==role]['total']))) for role in ['train','mean','eval']}
    return dict(models=metrics,timing=timing)


def compare(first,second):
    a,_=load(first);b,_=load(second)
    errors={}
    for name in ['target','za','lpt2','cross_za','cross_lpt2','r_za','r_lpt2']:
        x=a[name];y=b[name];assert np.array_equal(np.isnan(x),np.isnan(y))
        finite=np.isfinite(x);errors[name]=float(np.max(np.abs(x[finite]-y[finite]))/max(1.,np.max(np.abs(x[finite]))))
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mpi-binary',type=Path);parser.add_argument('--launcher',default='mpirun')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    commands=[];summary={}
    cases=[('fiducial',16,20,.5,64,1024,256),('late',16,20,1.,64,1024,256),
           ('step40',16,40,.5,12,32,16),('step80',16,80,.5,12,32,16),('mesh32',32,40,.5,12,32,16)]
    if args.smoke:cases=[('smoke',16,20,.5,12,32,16)]
    def run(name,options,binary=args.binary,launcher=()):
        prefix=args.output/name
        cmd=[*launcher,str(binary.resolve()),str(prefix.resolve()),*map(str,options), '12340','.001']
        start=time.monotonic();p=subprocess.run(cmd,capture_output=True,text=True,timeout=240,env=dict(os.environ,OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS','1')))
        (args.output/(name+'.log')).write_text(p.stdout+p.stderr)
        commands.append(dict(command=cmd,returncode=p.returncode,seconds=time.monotonic()-start))
        (args.output/'commands.json').write_text(json.dumps(commands,indent=2))
        if p.returncode:raise RuntimeError(p.stdout[-2000:]+p.stderr[-2000:])
        summary[name]=check(prefix)
        print(name,[(v['model'],v['mean_variance_reduction']) for v in summary[name]['models']],flush=True)
        return prefix
    for name,*options in cases:run(name,options)
    if not args.smoke:
        # Same first 12 training phases, physical input support and output bins.
        # Mesh refinement changes particles, force and IC meshes together.
        def targets(name):
            rows,_=load(args.output/name)
            return rows[(rows['role']=='train')&(rows['id']<12)]['target'].reshape(-1,3)
        coarse=targets('fiducial');mid=targets('step40');fine=targets('step80')
        norm=np.max(np.abs(fine));first=float(np.max(np.abs(coarse-mid))/norm);second=float(np.max(np.abs(mid-fine))/norm)
        assert second<first and second<.02,(first,second)
        summary['timestep_refinement']=dict(step20_to40=first,step40_to80=second)
        summary['mesh_refinement']=dict(mesh16_to32=float(np.max(np.abs(targets('mesh32')-mid))/np.max(np.abs(mid))))
    if args.mpi_binary:
        baseline=run('mpi-serial',[16,20,.5,12,32,16])
        for n in [2,4]:
            prefix=run('mpi-np'+str(n),[16,20,.5,12,32,16],args.mpi_binary,[*shlex.split(args.launcher),'-np',str(n)])
            error=compare(baseline,prefix);assert max(error.values())<1e-9,error
            summary['mpi-np'+str(n)]['serial_scaled_differences']=error
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'provenance.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        mpi_binary_sha256=hashlib.sha256(args.mpi_binary.read_bytes()).hexdigest() if args.mpi_binary else None,
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.csv')}),indent=2))
    print('Matched ensemble reference: PASS',flush=True)


if __name__=='__main__':main()
