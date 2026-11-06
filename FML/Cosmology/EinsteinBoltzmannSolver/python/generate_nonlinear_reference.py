#!/usr/bin/env python3
"""Regenerate the identical-linear-input HMcode-2020 (CAMB) and Bird (CLASS) fixtures.

Run from EinsteinBoltzmannSolver with OMP_NUM_THREADS=1. Requires camb, classy, numpy.
The C++ regression tests do not require these packages. Linear spectra and nonlinear references
come from the same external run, isolating nonlinear modelling from Boltzmann-solver errors.
"""
from pathlib import Path
import argparse
import importlib.metadata
import numpy as np
import camb
from classy import Class

out = Path(__file__).resolve().parents[1] / "tests/reference/nonlinear2020"
out.mkdir(parents=True, exist_ok=True)
args = argparse.ArgumentParser()
args.add_argument('--bird-only', action='store_true')
args = args.parse_args()
h, ob, oc, ns, As = .67, .05, .267, .965, 2.1e-9
zs = [2., 1., 0.]
kh = np.geomspace(1e-5, 1e7, 1001)
kr = np.geomspace(.01, 10., 31)
for name, mnu, w, wa, ok in [("lcdm", 0., -1., 0., 0.),
                              ("nu", .3, -1., 0., 0.),
                              ("w0wa", 0., -.8, .2, .02)]:
    if args.bird_only:
        continue
    p = camb.CAMBparams()
    p.set_cosmology(H0=100*h, ombh2=ob*h*h, omch2=oc*h*h, mnu=mnu,
                    nnu=3.046, omk=ok, TCMB=2.7255)
    p.InitPower.set_params(As=As, ns=ns)
    p.set_dark_energy(w=w, wa=wa, dark_energy_model='ppf')
    p.set_matter_power(redshifts=zs, kmax=100., k_per_logint=20)
    p.WantCls = False
    p.NonLinear = camb.model.NonLinear_pk
    p.NonLinearModel.set_params(halofit_version='mead2020')
    r = camb.get_results(p)
    pm = r.get_matter_power_interpolator(nonlinear=False, extrap_kmax=1e7)
    pcb = r.get_matter_power_interpolator(nonlinear=False, var1='delta_nonu', var2='delta_nonu',
                                         extrap_kmax=1e7)
    pnl = r.get_matter_power_interpolator(nonlinear=True)
    # Metadata: reference package/version and cosmology; all k in h/Mpc, all P in (Mpc/h)^3.
    header = f"CAMB {camb.__version__}, mead2020; h={h} Ob={ob} Oc={oc} mnu={mnu} w={w} wa={wa} Ok={ok}"
    np.savetxt(out / f"{name}_linear.txt", np.column_stack([kh] +
               [v for z in zs for v in (pm.P(z, kh), pcb.P(z, kh))]),
               fmt='%.10e', header=header + "\nk Pm_z2 Pcb_z2 Pm_z1 Pcb_z1 Pm_z0 Pcb_z0")
    np.savetxt(out / f"{name}_hmcode.txt", np.column_stack([kr] + [pnl.P(z, kr) for z in zs]),
               fmt='%.10e', header=header + "\nk Pnl_z2 Pnl_z1 Pnl_z0")
    print(name, 'HMcode reference written', flush=True)

cl = Class()
cl.set(dict(h=h, omega_b=ob*h*h, omega_cdm=oc*h*h, A_s=As, n_s=ns,
            N_ncdm=1, m_ncdm=.3, N_ur=2.0328, T_ncdm=.71611,
            output='mPk', non_linear='halofit', **{'P_k_max_1/Mpc': 1100., 'z_max_pk': 2.}))
cl.compute()
k = np.exp(np.arange(np.log(1e-4), np.log(1e3)+1e-9, .02))
np.savetxt(out / 'bird_linear.txt', np.column_stack([k] +
           [np.array([cl.pk_lin(v,z) for v in k]) for z in zs]), fmt='%.10e',
           header='classy '+importlib.metadata.version('classy')+'; same h Ob Oc as CAMB fixtures, mnu=0.3 eV\nk[1/Mpc] P_L_z2 P_L_z1 P_L_z0 [Mpc^3]')
np.savetxt(out / 'bird_halofit.txt', np.column_stack([kr] +
           [np.array([cl.pk(v*h,z)*h**3 for v in kr]) for z in zs]), fmt='%.10e',
           header='CLASS Takahashi+Bird\nk[h/Mpc] Pnl_z2 Pnl_z1 Pnl_z0 [(Mpc/h)^3]')
cl.struct_cleanup()
cl.empty()
print('Bird reference written')
