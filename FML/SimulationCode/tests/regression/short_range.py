#!/usr/bin/env python3
"""Run common-timestep short-range SimulationCode regressions with saved full-precision states.

Build tests/test_short_range_simulation first. Uses only the Python standard library.
The MPI option additionally requires the specified launcher and a matching MPI binary.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import os
import shlex
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--mpi',action='store_true')
parser.add_argument('--launcher',default='mpirun')
args=parser.parse_args()
binary=args.binary.resolve()
if not binary.is_file():parser.error('Build the diagnostic driver and provide its path with --binary')
r=args.output.resolve();r.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[4]
mpi=args.mpi;name='mpi' if mpi else 'serial';cases=r/(name+'-cases');cases.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('simulation_regression_cases',Path(__file__).with_name('cases.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
base=dict(module.BASE);base.update(all_parameters_must_be_in_file=False,simulation_name='ShortRange',simulation_boxsize=64.,simulation_use_cola=False,particle_Npart_1D=8,ic_nmesh=16,force_nmesh=16,ic_initial_redshift=9.,output_redshifts=[0.],output_particles=False,pofk=False,ic_sigma8_normalization=True,ic_sigma8=.1,force_density_assignment_method='PCS',force_greens_function_kernel='continuous',force_gradient_kernel='continuous',force_short_range=True,force_short_range_split_scale=1.5,force_short_range_softening_support=.15,force_short_range_cutoff_tolerance=1e-8,force_short_range_backend='cell',force_short_range_table_points=16384)
launch=shlex.split(args.launcher)
def run(label,overrides,n=1,expected=0):
 d=cases/label;d.mkdir(exist_ok=True);p=dict(base);p.update(overrides);p['output_folder']=str(d/'output')
 single=p.pop('_single_kick',False)
 (d/'parameters.lua').write_text('\n'.join(k+' = '+module.lua_value(v) for k,v in p.items())+'\n')
 cmd=(launch+['-np',str(n)] if mpi else [])+[str(binary),str(d/'parameters.lua'),str(d/'state')]+(['single-kick'] if single else [])
 with (d/'run.log').open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1')).returncode
 assert (rc==0 if expected==0 else rc!=0),(label,rc)
 print(name,label,'PASS' if expected==0 else 'rejected',flush=True)
 if expected:
  assert 'force_short_range' in (d/'run.log').read_text(),label
  return None
 rows={}
 for f in d.glob('state.*.txt'):
  for line in f.read_text().splitlines():
   fields=line.split();key=int(fields[0]);assert key not in rows;rows[key]=list(map(float,fields[1:]))
 assert len(rows)==(7 if single else 512),(label,len(rows));return rows
if mpi:
 ref=run('cell-1',dict(timestep_nsteps=[8]),1)
 for n in [2,4]:
  x=run(f'cell-{n}',dict(timestep_nsteps=[8]),n);diff=max(abs(a-b) for key in ref for a,b in zip(ref[key],x[key]));assert diff<2e-9,diff;print('RANK_DIFFERENCE',n,diff,flush=True)
 for n in [1,2,4]:run(f'interlaced-{n}',dict(timestep_nsteps=[8],force_interlacing_order=2,force_deconvolve_window=True),n)
else:
 states={n:run(f'cell-{n}',dict(timestep_nsteps=[n])) for n in [4,8,16,32,64]}
 def rms(a,b):return math.sqrt(sum(sum((x-y-round(x-y))**2 for x,y in zip(a[k][:3],b[k][:3])) for k in a)/(3*len(a)))
 errors={n:rms(states[n],states[64]) for n in [4,8,16,32]};print('TIMESTEP_ERRORS',errors,flush=True)
 assert all(errors[b]<errors[a]*.6 for a,b in [(4,8),(8,16),(16,32)]),errors
 assert errors[16]<.35*errors[8] and errors[32]<.35*errors[16],errors
 (r/'timestep-errors.json').write_text(json.dumps(errors,indent=2)+'\n')
 tree=run('tree-exact',dict(timestep_nsteps=[8],force_short_range_backend='tree',force_short_range_opening_angle=0.))
 diff=max(abs(a-b) for k in tree for a,b in zip(tree[k],states[8][k]));assert diff<1e-11,diff;print('TREE_CELL_DIFFERENCE',diff)
 for order in [2,4]:run(f'interlaced-{order}',dict(timestep_nsteps=[8],force_interlacing_order=order,force_deconvolve_window=True))
 pm=run('pm-only',dict(timestep_nsteps=[8],force_short_range=False))
 tiny=run('tiny-split',dict(timestep_nsteps=[8],force_short_range_split_scale=1e-6,force_short_range_softening_support=1e-8))
 diff=max(abs(a-b) for k in pm for a,b in zip(pm[k],tiny[k]));assert diff<1e-9,diff;print('TINY_SPLIT_PM_DIFFERENCE',diff)
 for key,value in [('simulation_use_cola',True),('force_gradient_kernel','discrete_2pt'),('force_density_method','sheet'),('timestep_method','FastPM'),('gravity_model','mu'),('force_use_finite_difference_force',True)]:
  invalid={key:value,'timestep_nsteps':[1]}
  if key=='gravity_model':invalid.update(gravity_model_mu_parametrization='DE',gravity_model_mu_mu0=.5)
  run('reject-'+key,invalid,expected=1)

# Isolated calls of the actual SimulationCode::step, with a non-unit velocity scale,
# compare the complete kick to independent Ewald forces.
for order in [1,2]:
 run(f'single-kick-{order}',dict(timestep_nsteps=[1],force_nmesh=64,force_interlacing_order=order,force_deconvolve_window=True,_single_kick=True),4 if mpi else 1)

for order in [1,2]:
 run(f'lightcone-{order}',dict(timestep_nsteps=[4],ic_initial_redshift=.005,lightcone=True,plc_z_init=.005,plc_z_finish=0.,plc_pos_observer=[.5,.5,.5],plc_use_replicas=True,plc_ndim_rep=3,plc_output_gadgetfile=False,plc_output_asciifile=True,plc_relativistic_effects=True,force_interlacing_order=order),2 if mpi else 1)
 assert 'LightconePotential' in (cases/f'lightcone-{order}'/'run.log').read_text()

print('All common-timestep short-range integration cases passed',flush=True)
