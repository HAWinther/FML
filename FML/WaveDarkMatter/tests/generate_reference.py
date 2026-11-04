"""Optional SciPy/NumPy reference regeneration; not a build/runtime dependency.
Independent adaptive method-of-lines Schrödinger evolution and radial BVP soliton.
"""
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp,solve_bvp
p=Path(__file__).resolve().parent
n=4;L=2*np.pi;eta=.7;G=.4
x=np.arange(n)*L/n
X,Y,Z=np.meshgrid(x,x,x,indexing='ij')
initial=(1+.1*np.cos(X)+.05*np.cos(2*Y))*np.exp(1j*.08*np.sin(Z))
k=2*np.pi*np.fft.fftfreq(n,d=L/n)
K=np.meshgrid(k,k,k,indexing='ij');k2=sum(q*q for q in K);nonzero=k2!=0
kernel=np.zeros_like(k2);kernel[nonzero]=-4*np.pi*G/k2[nonzero]
def rhs(t,z):
 psi=z.reshape((n,n,n));rho=np.abs(psi)**2
 U=np.fft.ifftn(np.fft.fftn(rho)*kernel).real
 return (-1j*(-.5*eta*np.fft.ifftn(-k2*np.fft.fftn(psi))+U/eta*psi)).ravel()
solution=solve_ivp(rhs,(0,.07),initial.ravel(),method='DOP853',rtol=2e-13,atol=2e-14)
assert solution.success
with (p/'EvolutionReference.h').open('w') as out:
 out.write('// SciPy DOP853 independent method-of-lines reference; see generate_reference.py.\n#pragma once\n#include <array>\nnamespace wave_reference {\ninline constexpr std::array<std::array<double,2>,64> evolved{{\n')
 for z in solution.y[:,-1]:out.write('{{{:.17g},{:.17g}}},\n'.format(z.real,z.imag))
 out.write('}};\n}\n')
# Spherical isolated ground state with f(0)=1, no nodes, and asymptotic U=-M/r.
# solve_bvp's singular matrix handles 2 f'/r and 2 U'/r at r=0.
r=np.linspace(0,12,1000)
f=np.exp(-r*r/2)
y=np.array([f,-r*f,-4/np.sqrt(1+r*r),4*r/(1+r*r)**1.5])
def ode(r,y,mu):return np.array([y[1],2*(y[2]-mu[0])*y[0],y[3],4*np.pi*y[0]**2])
def bc(lo,hi,mu):return np.array([lo[0]-1,lo[1],lo[3],hi[0],hi[2]+12*hi[3]])
bvp=solve_bvp(ode,bc,r,y,p=[-2.4],S=np.diag([0,-2,0,-2]),tol=2e-10,max_nodes=30000)
assert bvp.success,bvp.message
r=np.arange(601)*.02;f=bvp.sol(r)[0]
assert f.min()>-1e-10 and np.all(np.diff(f)<1e-10)
mu=bvp.p[0];mass=12**2*bvp.sol(12)[3]
with (p/'SolitonReference.h').open('w') as out:
 out.write('// Independent radial SciPy BVP; eta=G=1, f(0)=1, U(infinity)=0.\n#pragma once\n#include <array>\nnamespace wave_reference {\n')
 out.write('inline constexpr double soliton_mu={:.17g},soliton_mass={:.17g},dr=.02;\n'.format(mu,mass))
 out.write('inline constexpr std::array<double,601> soliton{{\n')
 for q in f:out.write('{:.17g},\n'.format(q))
 out.write('''}};
inline double profile(double r){
 if(r>=12)return 0;
 const int i=int(r/dr);const double t=r/dr-i;
 if(i>=598)return (1-t)*soliton[i]+t*soliton[i+1];
 double result=0;
 for(int j=-2;j<=3;++j){double weight=1;for(int k=-2;k<=3;++k)if(k!=j)weight*=(t-k)/double(j-k);const int index=i+j;result+=weight*soliton[index<0?-index:index];}
 return result;
}
}
''')
print('DOP853 evaluations',solution.nfev,'BVP nodes',len(bvp.x),'mu',mu,'mass',mass)
