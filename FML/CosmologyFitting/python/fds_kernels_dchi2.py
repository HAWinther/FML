#!/usr/bin/env python3
"""The effect on the DESI DR1 full-shape chi^2 of exact-time two-fluid instead of EdS loop kernels for the Fading Dark
Sector (README, "Fading Dark Sector"). From CosmologyFitting/, with the outputs of tests/check_fds_kernels:
  tests/check_fds_kernels -0.85 0.053 | grep -v '^#' > fds.txt;  tests/check_fds_kernels 0 0 | grep -v '^#' > lcdm.txt
  python3 python/fds_kernels_dchi2.py fds.txt lcdm.txt
The relative changes of the real-space loops, eps_dd = dP_dd/P_L and eps_tt = dP_tt/(f^2 P_L) (eps_dt their mean), are
mapped onto P_0, P_2, P_4 with the Kaiser weights of the linear bias b = b1p/sigma_8 of each sample, convolved with the
window and compared with DESI's covariance: chi^2 of the change, and after refitting b1, alpha0, alpha2 and sn0
(linear templates). The linear P_cb, sigma_8, f: tests/reference/desi_fs_rept_inputs.txt (Planck 2018).
"""
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
D='external/desi_dr1_fs/'
F=np.loadtxt(sys.argv[1]); L=np.loadtxt(sys.argv[2])
zg=np.array([0.295,0.51,0.92,1.49]); kg=np.array(sorted(set(F[:,1])))
def grid(A,col_e,col_x,tt):
    g=np.zeros((len(zg),len(kg)))
    for r in A:
        i=np.argmin(abs(zg-r[0])); j=np.argmin(abs(kg-r[1]))
        g[i,j]=(r[col_e]-r[col_x])/(r[2]*(r[7]**2 if tt else 1))
    return RegularGridInterpolator((zg,kg),g,bounds_error=False,fill_value=None)
eF_dd,eF_tt=grid(F,3,4,False),grid(F,5,6,True); eL_dd,eL_tt=grid(L,3,4,False),grid(L,5,6,True)
inp='tests/reference/desi_fs_rept_inputs.txt'
meta=[l.split() for l in open(inp) if l.startswith('# z')]
zs=[float(m[2]) for m in meta]; s8=[float(m[8]) for m in meta]; f=[float(m[10])/float(m[8]) for m in meta]
A=np.loadtxt(inp); kin=A[:,0]
names=['BGS_z0','LRG_z0','LRG_z1','LRG_z2','ELG_z1','QSO_z0']
b1p=[1.06,1.09,1.19,1.02,0.52,0.79]
for label,(edd,ett) in [('total (exact - EdS)',(eF_dd,eF_tt)),('FDS-specific (minus the LCDM difference)',(None,None))]:
    tot=0; tota=0
    for i,n in enumerate(names):
        kth=np.loadtxt(D+n+'_ktheory.txt'); W=np.loadtxt(D+n+'_window.txt'); C=np.loadtxt(D+n+'_cov.txt'); Ci=np.linalg.inv(C)
        PL=np.exp(np.interp(np.log(kth),np.log(kin),np.log(A[:,1+3*i])))
        b=b1p[i]/s8[i]; ff=f[i]
        pts=np.column_stack([np.full(kth.size,zs[i]),np.clip(kth,0.02,0.3)])
        if edd is None:
            dd=eF_dd(pts)-eL_dd(pts); tt=eF_tt(pts)-eL_tt(pts)
        else:
            dd=edd(pts); tt=ett(pts)
        dt=0.5*(dd+tt)
        P0=(b*b*dd+2/3*b*ff*dt+ff*ff/5*tt)*PL; P2=(4/3*b*ff*dt+4/7*ff*ff*tt)*PL; P4=(8/35*ff*ff*tt)*PL
        d=W.dot(np.concatenate([P0,P2,P4]))
        chi=d@Ci@d
        # absorb with b1 (Kaiser derivative), alpha0, alpha2 (k^2 P_L, k^2 mu^2 P_L) and sn0
        T=[W.dot(np.concatenate([(2*b+2/3*ff)*PL,(4/3*ff)*PL,0*PL])), W.dot(np.concatenate([kth**2*PL,0*PL,0*PL])),
           W.dot(np.concatenate([kth**2*PL/3,2/3*kth**2*PL,0*PL])), W.dot(np.concatenate([np.ones_like(PL),0*PL,0*PL]))]
        T=np.array(T).T; M=T.T@Ci@T; x=np.linalg.solve(M,T.T@Ci@d); r=d-T@x; chia=r@Ci@r
        tot+=chi; tota+=chia
        print(f"  {n}: max |dP0|/P0_lin {np.max(abs(P0/((b*b+2/3*b*ff+ff*ff/5)*PL))[kth<0.2]):.1e}  dchi2 {chi:.3f}  after b1, alpha0, alpha2, sn0 {chia:.4f}")
    print(f"{label}: sum dchi2 = {tot:.3f}, after absorbing = {tota:.4f}\n")
