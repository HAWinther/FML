"""Predeclared physical initial-mode diagnostics; reuse the fixed native diagnostic engine."""
from pathlib import Path
import csv,subprocess
import numpy as np

def evaluate(root,source,diagnostics,gates):
 out=root/'physical-monitors';out.mkdir(exist_ok=True);results={}
 for case in ['baseline','candidate']:
  directory=source if case=='baseline' else root/'candidate/mock203'
  for e in range(4):
   x=np.loadtxt(directory/f'calibration.chain{e}.csv',delimiter=',',skiprows=1)
   if x.shape!=(512,13) or not np.isfinite(x).all():raise ValueError('Unexpected monitor layout')
   w=x[:,6:12];p=x[:,1,None]*w
   with (out/f'{case}.chain{e}.csv').open('w') as stream:
    writer=csv.writer(stream);writer.writerow(['draw',*[f'w{j}' for j in range(6)],*gates['names']]);writer.writerows([[j,*w[j],*p[j]] for j in range(512)])
  prefix=out/case
  with (out/(case+'.log')).open('w') as log:subprocess.run([str(diagnostics),str(prefix),'512'],stdout=log,stderr=subprocess.STDOUT,check=True)
  rows=list(csv.DictReader((out/(case+'.diagnostics.csv')).open()));assert len(rows)==12
  values=[]
  for j,row in enumerate(rows[6:]):
   d=dict(parameter=gates['names'][j],status=row['status'],rhat=float(row['rhat']),bulk_ess=float(row['bulk_ess']),tail_ess=float(row['tail_ess']))
   d['passed']=bool(d['status']=='ok' and np.isfinite([d['rhat'],d['bulk_ess'],d['tail_ess']]).all() and d['rhat']<gates['max_rhat'] and d['bulk_ess']>=gates['min_bulk_ess'] and d['tail_ess']>=gates['min_tail_ess'])
   for key in ['rhat','bulk_ess','tail_ess']:
    if not np.isfinite(d[key]):d[key]=None
   values.append(d)
  results[case]=values
 return results
