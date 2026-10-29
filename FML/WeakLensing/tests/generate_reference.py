"""Regenerate Reference.h using independent SciPy QUADPACK integration.
Requires scipy==1.12.0; not needed for native tests. Flat EdS background,
analytic source planes and a prescribed scale-dependent galaxy bias/power law.
These are numerical/unit references, not cosmological calibration.
"""
from pathlib import Path
import math
import scipy
from scipy.integrate import quad
assert scipy.__version__ == '1.12.0'
dh=2997.92458
prefactor=1.5/dh**2

def z(chi): return (1-chi/(2*dh))**-2-1

def matter(k, redshift): return 2000*(k/.1)**.8*math.exp(-(k/.8)**2)/(1+redshift)**2

def kernel(chi, source): return prefactor*(1+z(chi))*chi*max(0,1-chi/source)

def foreground(chi):
    if 200 <= chi <= 800: return (chi-200)/600/400
    if 800 < chi <= 1000: return (1000-chi)/200/400
    return 0.

def spectrum(ell, which):
    def f(chi):
        if chi == 0: return 0.
        k=(ell+.5)/chi; redshift=z(chi)
        if which == 'auto': a,b=kernel(chi,1200),kernel(chi,1200); p=matter(k,redshift)
        elif which == 'cross': a,b=kernel(chi,1200),kernel(chi,2200); p=matter(k,redshift)
        else: a,b=foreground(chi),kernel(chi,2200);p=(1.4+.2*redshift+.1*k)*matter(k,redshift)
        return a*b*p/chi**2
    value,error=quad(f,0,2200,points=[200,800,1000,1200],epsabs=1e-20,epsrel=2e-12,limit=1000)
    assert error < 1e-10*abs(value), (value,error)
    return value

text='// Generated with SciPy 1.12.0 by generate_reference.py; independent adaptive quadrature.\n'
text+='#ifndef FML_WEAK_LENSING_REFERENCE_HEADER\n#define FML_WEAK_LENSING_REFERENCE_HEADER\n'
text+='inline const double lensing_reference[][4] = {\n'
for ell in [20.,100.,500.,1500.]:
    values=[ell]+[spectrum(ell,kind) for kind in ['auto','cross','galaxy']]
    text+='{' + ','.join(format(x,'.17g') for x in values) + '},\n'
text+='};\n#endif\n'
Path(__file__).with_name('Reference.h').write_text(text)
print(text)
