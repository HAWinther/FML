"""Cross-check a native tiny momentum kick with independent CIC/FFT forces."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
python=np.atleast_1d(np.genfromtxt(a.input/'linear_force.csv',delimiter=',',names=True));native=np.atleast_1d(np.genfromtxt(a.input/'native.csv',delimiter=',',names=True));checks=[];comparison=[]
for row in native:
    match=python[(python['n']==row['n'])&(python['m']==row['m'])&(python['amplitude']==row['amplitude'])]
    assert len(match)==1
    err=max(abs(row[k]-match[k][0]) for k in ['longitudinal_response','relative_force_error'])
    checks.append(dict(check=f"native/NumPy N={int(row['n'])} M={int(row['m'])} da={row['delta_a']}",passed=bool(err<2e-5)))
    comparison.append(dict(n=int(row['n']),m=int(row['m']),delta_a=float(row['delta_a']),max_error=float(err)))
for n,m in sorted(set(zip(native['n'],native['m']))):
    rows=native[(native['n']==n)&(native['m']==m)];checks.append(dict(check=f'native shrinking kick {int(n)}/{int(m)}',passed=bool(np.ptp(rows['longitudinal_response'])<2e-5)))
fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
for amplitude,style in [(1e-3,'--'),(1e-4,'-')]:
    rows=python[(python['n']==8)&(python['amplitude']==amplitude)];rows=np.sort(rows,order='m')
    axes[0].plot(rows['m'],rows['longitudinal_response'],'o'+style,label=f'NumPy ε={amplitude:g}')
rows=native[(native['n']==8)&(native['delta_a']==1e-9)]
axes[0].plot(rows['m'],rows['longitudinal_response'],'kx',ms=8,label='Native tiny kick')
rows=python[(python['m']==32)&(python['amplitude']==1e-4)];rows=np.sort(rows,order='n')
axes[1].plot(rows['n'],rows['longitudinal_response'],'o-',label='NumPy (includes N>M)')
rows=native[(native['m']==32)&(native['delta_a']==1e-9)];rows=np.sort(rows,order='n')
axes[1].plot(rows['n'],rows['longitudinal_response'],'kx',ms=8,label='Native tiny kick (includes N>M)')
rows=python[(python['n']==64)&(python['amplitude']==1e-4)];rows=np.sort(rows,order='m')
axes[2].plot(rows['m'],rows['longitudinal_response'],'o-',label='NumPy, N=64')
rows=native[(native['n']==64)&(native['delta_a']==1e-9)];rows=np.sort(rows,order='m')
axes[2].plot(rows['m'],rows['longitudinal_response'],'kx',ms=8,label='Native tiny kick')
for ax in axes:
    ax.axhline(1,color='gray',ls=':',label='Continuum linear response');ax.set(ylabel=r'$\langle F,\psi\rangle / (\epsilon\langle\psi,\psi\rangle)$');ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[0].set(title='Fixed particle lattice N=8',xlabel='force mesh M');axes[1].set(title='Fixed force mesh M=32',xlabel='particle lattice N')
axes[2].set(title='Dense particles N=64',xlabel='force mesh M')
fig.suptitle('Independent force/assignment diagnostic; no inference or accumulated time evolution')
fig.savefig(a.output/'force_response.png',dpi=170);fig.savefig(a.output/'force_response.pdf');plt.close(fig)
summary=dict(code_consistency_passed=all(c['passed'] for c in checks),checks=checks,comparisons=comparison,max_native_numpy_difference=max(c['max_error'] for c in comparison),
    native_rows=[{k:float(r[k]) for k in native.dtype.names} for r in native],numpy_rows=[{k:float(r[k]) for k in python.dtype.names} for r in python],
    limitation='Sparse particle loads on fine force grids show a strong response error. Native and NumPy now include N>M; dense sampling does not remove finite force-window errors. This is one linear field, not a general accuracy guarantee.')
(a.output/'force_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(code_consistency_passed=summary['code_consistency_passed'],max_native_numpy_difference=summary['max_native_numpy_difference']),indent=2))
if not summary['code_consistency_passed']:raise SystemExit(1)
