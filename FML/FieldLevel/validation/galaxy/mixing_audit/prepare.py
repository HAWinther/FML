#!/usr/bin/env python3
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description='Freeze dataset-3 geometry and transport inputs');p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--exe',type=Path,required=True);a=p.parse_args();source=a.source.resolve();out=a.output.resolve()
 if out.exists() and any(out.iterdir()):raise ValueError('New audit needs empty output')
 out.mkdir(parents=True,exist_ok=True);plan=json.loads((HERE/'plan.json').read_text());(out/'plan.json').write_bytes((HERE/'plan.json').read_bytes())
 target=json.loads((source/'ensemble0/target.json').read_text());source_hashes={str(source/'ensemble0/target.json'):sha(source/'ensemble0/target.json'),str(source/'calibration-summary.json'):sha(source/'calibration-summary.json')}
 lines=[f"FML_GEOMETRY_DATA_1 {len(target['data'])}",*[' '.join(format(x,'.17g') for x in z) for z in target['data']]];(out/'data.txt').write_text('\n'.join(lines)+'\n')
 for draw in plan['endpoint_draws']:
  records={}
  for e in range(4):
   path=source/f'ensemble{e}/draws/draw-{draw}.rank-0.json';source_hashes[str(path)]=sha(path);records[e]=json.loads(path.read_text())
  first=records[plan['source_ensemble']];w0=np.array(first['white']);p0=np.array(first['parameters']);rows=[]
  for e in plan['destination_ensembles']:
   w1=np.array(records[e]['white']);p1=np.array(records[e]['parameters']);r0=np.linalg.norm(w0);r1=np.linalg.norm(w1);u0=w0/r0;u1=w1/r1;theta=np.arccos(np.clip(u0@u1,-1,1))
   if not 1e-6<theta<np.pi-1e-6:raise ValueError('Degenerate spherical route')
   for kind in ['white','physical','arc']:
    for j,t in enumerate(np.linspace(0,1,plan['points_per_path'])):
     params=(1-t)*p0+t*p1
     if kind=='white':white=(1-t)*w0+t*w1
     elif kind=='physical':white=((1-t)*p0[0]*w0+t*p1[0]*w1)/params[0]
     else:white=((1-t)*r0+t*r1)*(np.sin((1-t)*theta)/np.sin(theta)*u0+np.sin(t*theta)/np.sin(theta)*u1)
     if j==0:white=w0.copy();params=p0.copy()
     if j==100:white=w1.copy();params=p1.copy()
     rows.append([f'{kind}-to{e}',j,t,*params,*white])
  for j,t in enumerate(np.linspace(0,1,101)):
   params=p0.copy();params[0]=.6+.8*t;white=w0*p0[0]/params[0];rows.append(['rescale-control',j,t,*params,*white])
  with (out/f'points{draw}.csv').open('w') as f:
   writer=csv.writer(f);writer.writerow(['path','index','t','alpha','b_delta','b_delta2','b_K2','b_laplacian',*[f'w{j}' for j in range(64)]]);writer.writerows(rows)
 # Hash all transport records before examining their energy/flow summaries.
 for e in range(4):
  paths=[source/f'ensemble{e}/summary.json',source/f'calibration.chain{e}.csv',*[source/f'ensemble{e}/draws/draw-{j}.rank-0.json' for j in range(2048)]]
  for path in paths:source_hashes[str(path)]=sha(path)
 impl=[HERE/'prepare.py',HERE/'analyze.py',HERE.parent/'geometry/Probe.cpp']
 identity=dict(source=str(source),binary_sha256=sha(a.exe.resolve()),plan_sha256=sha(out/'plan.json'),source_hashes=source_hashes,implementation={str(p):sha(p) for p in impl},inputs={p.name:sha(p) for p in [out/'data.txt',out/'points1024.csv',out/'points2047.csv']})
 (out/'FROZEN.json').write_text(json.dumps(identity,indent=2)+'\n');print('Frozen two 1010-point geometry audits and all dataset-3 transport records')
if __name__=='__main__':main()
