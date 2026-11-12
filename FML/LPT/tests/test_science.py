#!/usr/bin/env python3
"""Nonzero sixth-order oracle, symmetries, equations, support and truncation diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import numpy as np
import eds_reference as ref
from test_eds_reference import error, norm


def fixture():
    positive={(1,0,0):.12+.03j,(0,1,1):-.08+.04j,(1,-1,1):.06-.02j}
    return {**positive,**{tuple(-x for x in m):v.conjugate() for m,v in positive.items()}}


def full(field):
    result=[]
    for component in field:
        values=dict(component)
        for m,v in component.items():
            if m[2]>0:values[tuple(-x for x in m)]=v.conjugate()
        result.append(values)
    return result


def crop(field,mesh):
    return [{m:v for m,v in c.items() if m[2]>=0 and all(2*abs(x)<mesh for x in m)} for c in field]


def jacobian(orders,growth,mesh):
    frequency=2*np.pi*np.fft.fftfreq(mesh,d=1/mesh)
    waves=np.meshgrid(frequency,frequency,frequency,indexing='ij')
    partial=np.zeros((3,mesh,mesh,mesh),dtype=complex);rows=[]
    for n,field in orders.items():
        array=np.zeros_like(partial)
        for i,c in enumerate(full(field)):
            for m,v in c.items():array[(i,)+tuple(x%mesh for x in m)]=v
        partial+=growth**n*array
        gradient=np.empty((mesh,mesh,mesh,3,3))
        for i in range(3):
            for j in range(3):gradient[...,i,j]=(np.fft.ifftn(1j*waves[j]*partial[i])*mesh**3).real
        determinant=np.linalg.det(gradient+np.eye(3))
        # Full Fourier coefficients, with normalized forward transforms.
        rows.append([np.sqrt(np.sum(np.abs(array)**2)),np.sqrt(np.sum(np.abs(partial)**2)),
                     determinant.min(),np.count_nonzero(determinant<=0)])
    return np.asarray(rows)


def equation_errors(delta,orders):
    # Project away roundoff-only forbidden modes; verify all exported coefficients separately.
    reachable={0:{ref.ZERO}}
    for n in orders:reachable[n]={ref.vector_sum(a,b) for a in reachable[n-1] for b in delta}
    physical={n:[{m:v for m,v in c.items() if m in reachable[n]} for c in full(f)] for n,f in orders.items()}
    matrix=ref.deformation(physical);worst_scalar=0;worst_cauchy=0
    for n in orders:
        scale=norm(physical[n])*2*np.pi*n*n
        residual=ref.add((1,ref.acceleration_divergence(matrix,n)),(-1.5,ref.determinant_coefficient(matrix,n)))
        worst_scalar=max(worst_scalar,norm([residual])/max(scale,1e-300))
        for j,k in [(0,1),(1,2),(2,0)]:
            terms=[]
            for i in range(3):
                dk={s:ref.add((s,f)) for s,f in matrix[i][k].items() if s}
                dj={s:ref.add((s,f)) for s,f in matrix[i][j].items() if s}
                terms.extend([(1,ref.coefficient([matrix[i][j],dk],n)),(-1,ref.coefficient([matrix[i][k],dj],n))])
            worst_cauchy=max(worst_cauchy,norm([ref.add(*terms)])/max(scale,1e-300))
    return dict(scalar=worst_scalar,cauchy=worst_cauchy)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True);parser.add_argument('--example-exe',type=Path)
    parser.add_argument('--precision',choices=['double','float'],default='double')
    parser.add_argument('--launcher',default='');parser.add_argument('--np-list',default='1')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();ranks=list(map(int,args.np_list.split()))
    if not ranks or any(n not in (1,2,4) for n in ranks):raise ValueError('Use ranks 1, 2 and/or 4')
    if not args.launcher and ranks!=[1]:raise ValueError('Multiple ranks require --launcher')
    temporary=tempfile.TemporaryDirectory(prefix='fml-eds-science-') if args.output is None else None
    root=Path(temporary.name) if temporary else args.output;root.mkdir(parents=True,exist_ok=True)
    tol=3e-5 if args.precision=='float' else 3e-12
    summary={};commands=[];checks=0
    def launch(cmd,label,threads=1):
        begin=time.monotonic();p=subprocess.run(cmd,capture_output=True,text=True,timeout=120,
            env=dict(os.environ,OMP_NUM_THREADS=str(threads),OMP_WAIT_POLICY='passive'))
        (root/(label+'.log')).write_text(p.stdout+p.stderr)
        commands.append(dict(command=cmd,threads=threads,returncode=p.returncode,seconds=time.monotonic()-begin))
        (root/'commands.json').write_text(json.dumps(commands,indent=2))
        if p.returncode:raise RuntimeError(f'{label}: {p.stdout[-1500:]}{p.stderr[-1500:]}')
    def run(label,delta,mesh=16,cutoff=1,count=ranks[0],threads=1,policy='cache'):
        nonlocal checks
        source=root/(label+'.input');prefix=root/label
        source.write_text(''.join(f'{m[0]} {m[1]} {m[2]} {v.real:.17g} {v.imag:.17g}\n' for m,v in sorted(delta.items())))
        launcher=[*shlex.split(args.launcher),'-np',str(count)] if args.launcher else []
        launch([*launcher,str(args.exe.resolve()),'--export',str(source.resolve()),str(prefix.resolve()),str(mesh),str(cutoff),'6',policy],label,threads)
        metadata=json.loads(Path(str(prefix)+'.report.json').read_text())
        assert metadata['scalar_bytes']==(4 if args.precision=='float' else 8),metadata
        actual={n:[{},{},{}] for n in range(1,7)}
        for rank in range(count):
            rows=np.loadtxt(str(prefix)+'.'+str(rank),ndmin=2)
            for n,x,y,z,axis,re,im in rows:
                c=actual[int(n)][int(axis)];m=(int(x),int(y),int(z))
                assert m not in c,'Overlapping output owners';c[m]=complex(re,im)
        assert sum(len(c) for f in actual.values() for c in f)==mesh*mesh*(mesh//2+1)*18
        checks+=1;return actual
    delta=fixture();oracle=ref.solve(delta,6)
    assert norm(oracle[6])>1e-9,'Sixth-order fixture must be nonzero'
    def compare(actual,expected,label):
        worst=0
        for n in actual:
            scale=norm(expected[n]);difference=error(actual[n],expected[n])
            assert difference<=tol*scale+1e-18,(label,n,difference,scale,tol)
            worst=max(worst,difference/max(scale,1e-300))
        summary[label]=dict(max_relative_coefficient_error=worst);print(label,worst,flush=True)
    baseline=None
    for count in ranks:
        for threads in [1,2]:
            for policy in ['cache','recompute']:
                label=f'order6-r{count}-t{threads}-{policy}'
                actual=run(label,delta,count=count,threads=threads,policy=policy)
                compare(actual,{n:crop(f,16) for n,f in oracle.items()},label)
                if baseline is None:baseline=actual
                else:compare(actual,baseline,label+'-baseline')
    equations=equation_errors(delta,baseline)
    assert max(equations.values())<tol,equations
    summary['equation_residuals']=equations
    shift=(.173,-.219,.087)
    translated=run('translation',{m:v*np.exp(-2j*np.pi*np.dot(m,shift)) for m,v in delta.items()})
    compare(translated,{n:[{m:v*np.exp(-2j*np.pi*np.dot(m,shift)) for m,v in c.items()} for c in f] for n,f in baseline.items()},'translation')
    reflected=run('reflection',{(-m[0],m[1],m[2]):v for m,v in delta.items()})
    compare(reflected,{n:[{(-m[0],m[1],m[2]):(-v if i==0 else v) for m,v in c.items()} for i,c in enumerate(f)] for n,f in baseline.items()},'reflection')
    permutation=(2,0,1)
    permuted=run('permutation',{tuple(m[j] for j in permutation):v for m,v in delta.items()})
    expected={n:crop([{tuple(m[j] for j in permutation):v for m,v in full(f)[i].items()} for i in permutation],16) for n,f in baseline.items()}
    compare(permuted,expected,'permutation')
    scaled=run('negative-amplitude',{m:-.7*v for m,v in delta.items()})
    compare(scaled,{n:[{m:(-.7)**n*v for m,v in c.items()} for c in f] for n,f in baseline.items()},'negative-amplitude')
    refined=run('mesh24',delta,mesh=24,count=ranks[-1],threads=2,policy='recompute')
    compare({n:crop(f,16) for n,f in refined.items()},baseline,'mesh16-to24')
    plane={(1,1,1):.1+.04j,(-1,-1,-1):.1-.04j}
    planar=run('plane-parallel',plane)
    leakage=max(norm(planar[n])/(max(abs(v) for v in plane.values())**n/(2*np.pi)) for n in range(2,7))
    assert leakage<tol,leakage;summary['plane_parallel']=dict(scaled_nonlinear_leakage=leakage)
    oblique={(1,2,-1):.1+.04j,(-1,-2,1):.1-.04j}
    oblique_fields=run('oblique-plane',oblique,cutoff=2)
    oblique_leakage=max(norm(oblique_fields[n])/(max(abs(v) for v in oblique.values())**n/(2*np.pi)) for n in range(2,7))
    assert oblique_leakage<tol,oblique_leakage
    summary['oblique_plane']=dict(scaled_nonlinear_leakage=oblique_leakage)
    zero=run('zero-seed',{})
    assert all(norm(f)==0 for f in zero.values())
    # Controlled input-support sensitivity: the added seed modes have descending amplitudes.
    support={1:delta,2:{**delta,(2,1,0):.02+.005j,(-2,-1,0):.02-.005j}}
    support[3]={**support[2],(3,1,1):.002-.001j,(-3,-1,-1):.002+.001j}
    fields={1:baseline,2:run('support2',support[2],cutoff=2),3:run('support3',support[3],cutoff=3)}
    def low_sum(orders):
        result=[{} for _ in range(3)]
        for n,f in orders.items():
            for i,c in enumerate(f):
                for m,v in c.items():
                    if max(map(abs,m))<=1:result[i][m]=result[i].get(m,0)+.5**n*v
        return result
    sums={k:low_sum(v) for k,v in fields.items()};support_errors=[error(sums[k],sums[3])/norm(sums[3]) for k in [1,2]]
    assert 0<support_errors[1]<support_errors[0],support_errors
    summary['input_support']=dict(cut1_vs3=support_errors[0],cut2_vs3=support_errors[1])
    if args.example_exe:
        for seed,growth in [('coupled',.5),('coupled',1.),('coupled',4.),('plane',.5),('plane',6.)]:
            label=f'example-{seed}-g{growth:g}';prefix=root/label
            launcher=[*shlex.split(args.launcher),'-np',str(ranks[-1])] if args.launcher else []
            launch([*launcher,str(args.example_exe.resolve()),str(prefix.resolve()),'6',str(growth),'16',seed],label,2)
            rows=np.genfromtxt(str(prefix)+'.csv',delimiter=',',names=True)
            expected=jacobian(baseline if seed=='coupled' else planar,growth,16)
            actual=np.column_stack([rows['order_rms'],rows['sum_rms'],rows['min_jacobian']])
            denominator=np.maximum(np.max(np.abs(expected[:,:3]),axis=0),1e-20)
            difference=np.max(np.abs(actual-expected[:,:3])/denominator)
            assert difference<tol,(label,difference)
            assert np.array_equal(rows['nonpositive_samples'],expected[:,3]),label
            assert np.allclose(rows['weighted_order_rms'],growth**np.arange(1,7)*rows['order_rms'],rtol=tol)
            summary[label]=dict(max_scaled_diagnostic_error=float(difference),min_jacobian=rows['min_jacobian'].tolist(),weighted_order_rms=rows['weighted_order_rms'].tolist())
    summary['export_cases']=checks
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    (root/'provenance.json').write_text(json.dumps(dict(numpy=np.__version__,precision=args.precision,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle_sha256=hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),
        inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.input')}),indent=2))
    print('Scientific EdS LPT checks: PASS',json.dumps(summary,sort_keys=True),flush=True)
    if temporary:temporary.cleanup()


if __name__=='__main__':main()
