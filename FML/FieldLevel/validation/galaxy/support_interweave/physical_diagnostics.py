#!/usr/bin/env python3
"""Exploratory physical/white field monitors; never reclassify frozen study gates."""
from pathlib import Path
import argparse,csv,hashlib,json,subprocess
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('comparison',type=Path);p.add_argument('--diagnostics',type=Path,required=True);a=p.parse_args();root=a.comparison.resolve();exe=a.diagnostics.resolve()
 frozen=json.loads((root/'FROZEN.json').read_text());report=json.loads((root/'comparison-summary.json').read_text())
 if not report['complete'] or sha(exe)!=frozen['diagnostics_sha256']:raise ValueError('Complete fixed study and original diagnostics binary required')
 out=root/'physical-diagnostics'
 if out.exists():raise ValueError('Preserve existing supplementary diagnostics; use a new namespace')
 out.mkdir();names=['cos100','sin100','cos110','sin110','cos001','sin001'];inputs={str(root/'comparison-summary.json'):sha(root/'comparison-summary.json')};results={}
 for case in ['baseline','candidate']:
  source=Path(frozen['original_directory']) if case=='baseline' else root/'candidate/mock203';prefix=out/case
  for e in range(4):
   path=source/f'calibration.chain{e}.csv';inputs[str(path)]=sha(path);x=np.loadtxt(path,delimiter=',',skiprows=1)
   if x.shape!=(512,13) or not np.isfinite(x).all() or not np.array_equal(x[:,0],np.arange(512)):raise ValueError('Unexpected complete monitor layout')
   white=x[:,6:12];physical=x[:,1,None]*white
   with (out/f'{case}.chain{e}.csv').open('w') as stream:
    w=csv.writer(stream);w.writerow(['draw',*['white_'+n for n in names],*['alpha_white_'+n for n in names]]);w.writerows([[j,*white[j],*physical[j]] for j in range(512)])
  with (out/(case+'.log')).open('w') as log:subprocess.run([str(exe),str(prefix),'512'],stdout=log,stderr=subprocess.STDOUT,check=True)
  values=[]
  for j,row in enumerate(csv.DictReader((out/(case+'.diagnostics.csv')).open())):
   assert int(row['projection'])==j and j<12
   def finite(k):
    value=float(row[k]);return value if np.isfinite(value) else None
   values.append(dict(coordinate=('white_' if j<6 else 'alpha_white_')+names[j%6],status=row['status'],rhat=finite('rhat'),bulk_ess=finite('bulk_ess'),tail_ess=finite('tail_ess')))
  assert len(values)==12;results[case]=values
 r=dict(scope='Post-hoc exploratory field diagnostics on completed fixed chains; no new draws or altered gates',outcomes=results,definition='alpha times the six in-band white Fourier projections; proportional to physical initial modes at fixed reference spectrum',all_original_gates_unchanged=True,scientific_original_verdict={case:report['outcomes'][case]['passed'] for case in results},calibration_claim=False,source_sha256=sha(Path(__file__)),input_sha256=inputs,diagnostics_sha256=sha(exe),limits=['Requested after seeing the frozen outcomes, so exploratory rather than predeclared gates','The support-aware rescaling preserves alpha times each in-band white projection; these expose physical changes from the other kernels','Six physical modes are still not the complete field','Same fixed budgets and cached baseline; no extending, pooling or passing-subset claims'])
 (root/'physical-diagnostics.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
