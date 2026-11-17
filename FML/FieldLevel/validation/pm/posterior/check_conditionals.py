"""Compare independent 1D quadrature to the bounded conditional slice kernel."""
from pathlib import Path
import argparse,subprocess,json,numpy as np
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
x=subprocess.check_output([str(a.binary.resolve())],text=True);(out/'conditional_moments.csv').write_text(x);rows=np.genfromtxt(out/'conditional_moments.csv',delimiter=',',names=True);records=[]
for i in range(3):
 v=np.linspace(-4 if i==0 else -16,4 if i==0 else -4,400001);cut=0 if i==0 else -9.2
 u=.5*120*np.exp(-v)+61*v+2*v*v if i==0 else (.5*256*np.exp(-(v-np.log(1e-4)))+128*(v-np.log(1e-4))+2*(v-np.log(1e-4))**2 if i==1 else .5*((v+15.95)/.08)**2)
 w=np.exp(-(u-u.min()));norm=np.trapz(w,v);values=[np.trapz(v*w,v)/norm,np.trapz(v*v*w,v)/norm,np.trapz(w[v<=cut],v[v<=cut])/norm];sample=[float(rows[i][k]) for k in ['mean','second','cdf']];errors=np.abs(np.array(values)-sample);assert np.all(errors<np.array([.003,.02,.006])),(i,errors)
 records.append(dict(case=i,quadrature=values,sample=sample,errors=errors.tolist()))
(out/'conditional_checks.json').write_text(json.dumps(dict(passed=True,draws=200000,cases=records),indent=2)+'\n');print('conditional quadrature passed')
