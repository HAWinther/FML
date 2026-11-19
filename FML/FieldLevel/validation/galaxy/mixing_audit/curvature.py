#!/usr/bin/env python3
"""Interpret the bounded local mass-scaled Hessians; no global harmonic assumption."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);a=p.parse_args();root=a.audit.resolve();plan=json.loads((root/'curvature-plan.json').read_text());x=np.loadtxt(root/'curvature.csv',delimiter=',',skiprows=1)
 if x.shape!=(12*2*64*64,6) or not np.isfinite(x).all():raise ValueError('Incomplete/nonfinite matrix data')
 rows=[];allpass=True
 for e in range(4):
  for draw in [0,1024,2047]:
   matrices=[];antisym=[]
   for h in plan['finite_difference_steps']:
    r=x[(x[:,0]==e)&(x[:,1]==draw)&(x[:,2]==h)]
    if r.shape!=(4096,6) or len({(int(q[3]),int(q[4])) for q in r})!=4096:raise ValueError('Missing matrix entries')
    m=np.zeros((64,64));m[r[:,3].astype(int),r[:,4].astype(int)]=r[:,5]
    antisym.append(float(np.linalg.norm(m-m.T)/max(1,np.linalg.norm(m))));matrices.append(m)
   error=float(np.linalg.norm(matrices[0]-matrices[1])/max(1,np.linalg.norm(matrices[1])))
   passed=max(antisym)<=plan['maximum_relative_antisymmetry'] and error<=plan['maximum_relative_step_difference'];allpass &= passed
   eigen=np.linalg.eigvalsh((matrices[1]+matrices[1].T)/2);angles=[]
   for name,eps,leaps in [('baseline',.15,8),('half_step',.075,16)]:
    stable=eigen[(eigen>0)&(eps*eps*eigen<4)];phase=leaps*np.arccos(1-.5*eps*eps*stable)
    distance=np.min(np.abs(phase[:,None]-2*np.pi*np.arange(1,9)),axis=1)
    near=stable[distance<plan['near_full_period_radians']]
    angles.append(dict(case=name,locally_quadratic_unstable_eigenvalues=int(np.sum(eps*eps*eigen>=4)),nearest_nonzero_full_period_distance=float(distance.min()),near_full_period_eigenvalues=near.tolist()))
   rows.append(dict(ensemble=e,draw=draw,controls_passed=bool(passed),relative_antisymmetry=antisym,relative_step_difference=error,eigenvalues=eigen.tolist(),negative_eigenvalues=int(np.sum(eigen<-1e-5)),local_harmonic_angles=angles))
 report=dict(scope=plan['scope'],all_controls_passed=bool(allpass),states=rows,limits=['Only local conditional Hessians at selected correlated states, not the joint field/scalar Hessian','The nonlinear trajectory need not follow a fixed harmonic system; a near-period angle is not proof of resonance','Dense 64x64 matrix is diagnostic only; no production dense metric proposed','No new chains, discarded samples, changed priors or altered original verdicts'],matrix_sha256=hashlib.sha256((root/'curvature.csv').read_bytes()).hexdigest(),analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (root/'curvature-summary.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(all_controls_passed=bool(allpass),maximum_relative_antisymmetry=max(max(r['relative_antisymmetry']) for r in rows),maximum_step_difference=max(r['relative_step_difference'] for r in rows),states=[dict(ensemble=r['ensemble'],draw=r['draw'],eigen_min=min(r['eigenvalues']),eigen_max=max(r['eigenvalues']),negative=r['negative_eigenvalues'],angles=r['local_harmonic_angles']) for r in rows]),indent=2))
if __name__=='__main__':main()
