#!/usr/bin/env python3
"""Generate fixtures through the official Python interface (requires CAMB, NumPy).

OMP_NUM_THREADS=1 python3 generate_reference.py --upstream /path/to/ACTio-ReACTio --backend /path/to/libfml_react_backend.so
The C++ tests compare the independent FML array/ABI adapter with this interface, not with simulations.
"""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import camb

ap = argparse.ArgumentParser()
ap.add_argument('--upstream', type=Path, required=True)
ap.add_argument('--backend', type=Path, required=True)
args = ap.parse_args()
spec = importlib.util.spec_from_file_location('official_pyreact', args.upstream / 'pyreact/react.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
np.int = int  # Compatibility for the upstream wrapper's legacy NumPy alias; no engine changes.
module.ReACT.libname = str(args.backend.resolve())
engine = module.ReACT()
out = Path(__file__).resolve().parent / 'reference'
out.mkdir(exist_ok=True)
z = np.array([0., .5, 1., 2.])
k = np.geomspace(1e-5, 100., 128)
h, om, ob, As, ns = .67, .317, .05, 2.1e-9, .965

def camb_spectra(mnu=0., w=-1., wa=0.):
    p = camb.CAMBparams()
    p.set_cosmology(H0=100*h, ombh2=ob*h*h, omch2=(om-ob)*h*h, mnu=mnu, nnu=3.046)
    p.InitPower.set_params(As=As, ns=ns)
    p.set_dark_energy(w=w, wa=wa, dark_energy_model='ppf')
    p.WantCls = False
    p.set_matter_power(redshifts=z[::-1], kmax=100.)
    r = camb.get_results(p)
    total = r.get_matter_power_interpolator(nonlinear=False)
    cold = r.get_matter_power_interpolator(nonlinear=False, var1='delta_nonu', var2='delta_nonu')
    return (np.array([total.P(zz, k) for zz in z]), np.array([cold.P(zz, k) for zz in z]),
            r.get_Omega('nu', 0), r.get_sigma8()[-1])

base, basecb, _, sigma8 = camb_spectra()
for name in ['gr', 'fr', 'dgp', 'cpl_nu']:
    model = {'gr': 'gr', 'fr': 'f(r)', 'dgp': 'dgp', 'cpl_nu': 'cpl'}[name]
    pars = np.zeros(20)
    onu = 0.
    total, cold, ref = base, basecb, basecb
    if name in ['fr', 'dgp']:
        pars[0] = 1e-5 if name == 'fr' else .1
        # Generate self-consistent target linear spectra from the same LCDM seed via upstream growth.
        _, total, _, _ = engine.compute_reaction_ext(h, ns, om, ob, sigma8, z, k, base[0],
            model, pars, compute_pseudo=True, mass_loop=10, verbose=False)
        cold = total.copy()
    elif name == 'cpl_nu':
        pars[:2] = [-.9, .1]
        total, cold, onu, _ = camb_spectra(.3, -.9, .1)
        _, ref, _, _ = camb_spectra(.3)
        np.savetxt(out / 'cpl_nu_cosmology.txt', [onu], header='Omega_nu at z=0 from CAMB')
    reaction, linear, sigma, pseudo = engine.compute_reaction_nu_ext(h, ns, om+onu, ob, onu, As,
        z, k, total.flatten(), cold.flatten(), k, ref.flatten(), .05, model, pars,
        compute_pseudo=True, mass_loop=10, verbose=False)
    header = ('ACTio-ReACTio a36dda02db6b537a4ddcc7bf044c446c10a58373; '
              f'CAMB {camb.__version__}; model={model}; mass_loop=10; z=0,0.5,1,2; '
              'h=0.67 Om_cb=0.317 Ob=0.05 As=2.1e-9 ns=0.965 kpivot=0.05/Mpc')
    np.savetxt(out / (name+'_linear.txt'), np.column_stack([k] +
        [a[iz] for iz in range(4) for a in [total, cold, ref]]), fmt='%.16e',
        header=header+'\nk[h/Mpc], for each z: Ptotal Pcb PLCDMcb [(Mpc/h)^3]')
    np.savetxt(out / (name+'.txt'), np.column_stack([k] +
        [a[iz] for iz in range(4) for a in [reaction, linear, pseudo]]), fmt='%.16e',
        header=header+'\nk[h/Mpc], for each z: reaction Plinear Ppseudo [(Mpc/h)^3]')
    print(name, 'written; sigma8=', sigma, flush=True)
