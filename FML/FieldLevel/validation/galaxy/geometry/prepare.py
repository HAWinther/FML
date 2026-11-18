#!/usr/bin/env python3
"""Build declared deterministic paths from existing primary chain records."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('campaign',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args();root=args.campaign.resolve();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise RuntimeError('New audit requires an empty output directory')
    out.mkdir(parents=True,exist_ok=True);planpath=Path(__file__).resolve().parent/'plan.json';plan=json.loads(planpath.read_text());(out/'plan.json').write_bytes(planpath.read_bytes())
    def load(c):
        p=root/f'chain{c}/draws/draw-{plan["endpoint_draw"]}.rank-0.json';b=p.read_bytes();return json.loads(b),hashlib.sha256(b).hexdigest()
    first,first_hash=load(plan['source_chain']);w0=np.array(first['white']);p0=np.array(first['parameters']);n=w0.size;assert n==64
    rows=[];metadata=[]
    for c in plan['destination_chains']:
        last,h=load(c);w1=np.array(last['white']);p1=np.array(last['parameters']);metadata.append(dict(source_chain=plan['source_chain'],destination_chain=c,draw=plan['endpoint_draw'],source_sha256=first_hash,destination_sha256=h))
        r0=np.linalg.norm(w0);r1=np.linalg.norm(w1);u0=w0/r0;u1=w1/r1;theta=np.arccos(np.clip(u0@u1,-1,1));assert 1e-6<theta<np.pi-1e-6
        for kind in ['white','physical','arc']:
            for j,t in enumerate(np.linspace(0,1,plan['points_per_path'])):
                p=(1-t)*p0+t*p1
                if kind=='white':w=(1-t)*w0+t*w1
                elif kind=='physical':w=((1-t)*p0[0]*w0+t*p1[0]*w1)/p[0]
                else:w=((1-t)*r0+t*r1)*(np.sin((1-t)*theta)/np.sin(theta)*u0+np.sin(t*theta)/np.sin(theta)*u1)
                if j==0:w=w0.copy();p=p0.copy()
                if j==100:w=w1.copy();p=p1.copy()
                rows.append([f'{kind}-to{c}',j,t,*p,*w])
    # Pure rescaling control of a fixed physical density; not a Monte Carlo draw.
    for j,t in enumerate(np.linspace(0,1,101)):
        p=p0.copy();p[0]=.6+.8*t;w=w0*p0[0]/p[0];rows.append(['rescale-control',j,t,*p,*w])
    with (out/'points.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['path','index','t','alpha','b_delta','b_delta2','b_K2','b_laplacian',*[f'w{j}' for j in range(n)]]);writer.writerows(rows)
    (out/'endpoints.json').write_text(json.dumps(metadata,indent=2)+'\n');(out/'identity.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256(planpath.read_bytes()).hexdigest(),points_sha256=hashlib.sha256((out/'points.csv').read_bytes()).hexdigest(),point_count=len(rows)),indent=2)+'\n')
    print('Prepared',len(rows),'points on nine diagnostic paths and a rescaling control')
if __name__=='__main__':main()
