#!/usr/bin/env python3
"""Independent high-precision decorated-mean reference; optional unmodified Halotools kernel."""
import argparse
import ast
import csv
from decimal import Decimal,localcontext
import hashlib
import json
from pathlib import Path
import subprocess
import shlex
import tempfile
import types


def reference(p,mu,split,ac,ass,q):
    low,high=split,1-split
    def shift(value,strength,upper=None):
        if strength>=0:
            bound=value*low/high
            if upper is not None:bound=min(upper-value,bound)
            return strength*bound
        bound=-value
        if upper is not None:bound=max(bound,(value-upper)*low/high)
        return abs(strength)*bound
    factor=Decimal(1) if q>split else -high/low
    return p+factor*shift(p,ac,Decimal(1)),mu+factor*shift(mu,ass)


def halotools_kernel(path,rows):
    # Execute only the original scalar/vector perturbation kernel, without importing
    # Halotools or substituting our implementation for its arithmetic.
    import numpy as np
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HeavisideAssembias')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_galprop_perturbation')
    method.decorator_list=[];namespace={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),namespace)
    data={name:np.array([float(row[name]) for row in rows]) for name in rows[0]}
    outputs=[]
    for quantity,strength,upper in [('p','ac',1.),('mu','as',float('inf'))]:
        stub=types.SimpleNamespace(_method_name_to_decorate='mean_occupation',gal_type='galaxies',
            lower_bound_mean_occupation_galaxies=0.,upper_bound_mean_occupation_galaxies=upper,
            assembias_strength=lambda masses: data[strength])
        perturbation=namespace['_galprop_perturbation'](stub,baseline_result=data[quantity],
            prim_haloprop=np.ones(len(rows)),splitting_result=data['split'])
        high=data['q']>data['split'];perturbation[~high]*=-(1-data['split'][~high])/data['split'][~high]
        outputs.append(data[quantity]+perturbation)
    error=max(float(np.max(np.abs(outputs[i]-data[name])/np.maximum(1.,np.abs(data[name])))) for i,name in enumerate(['central','satellites']))
    assert error<5e-14,error
    return dict(max_scaled_error=error,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                method='_galprop_perturbation',scope='Isolated original pointwise perturbation kernel; no percentile estimation or mock-population comparison')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--halotools-source',type=Path);p.add_argument('--launcher',default='');args=p.parse_args()
    temporary=tempfile.TemporaryDirectory(prefix='fml-hod-reference-') if args.output is None else None
    args.output=Path(temporary.name) if temporary else args.output;args.output.mkdir(parents=True,exist_ok=True)
    table=args.output/'means.csv';cmd=[*([*shlex.split(args.launcher),'-np','1'] if args.launcher else []),str(args.exe.resolve()),'--export',str(table.resolve())]
    run=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(args.output/'export.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError(run.stdout+run.stderr)
    rows=list(csv.DictReader(table.open()));worst=Decimal(0)
    with localcontext() as context:
        context.prec=80
        for row in rows:
            r={k:Decimal(v) for k,v in row.items()};expected=reference(*(r[k] for k in ['p','mu','split','ac','as','q']))
            for value,name in zip(expected,['central','satellites']):
                error=abs(value-r[name])/max(Decimal(1),abs(value));worst=max(worst,error)
                assert error<Decimal('5e-14'),(row,name,value,error)
    summary=dict(rows=len(rows),decimal_precision=80,max_scaled_error=float(worst))
    if args.halotools_source:summary['halotools']=halotools_kernel(args.halotools_source,rows)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'provenance.json').write_text(json.dumps(dict(command=cmd,returncode=run.returncode,
        binary_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        means_sha256=hashlib.sha256(table.read_bytes()).hexdigest()),indent=2))
    print('HOD assembly-bias reference: PASS',json.dumps(summary),flush=True)
    if temporary:temporary.cleanup()


if __name__=='__main__':main()
