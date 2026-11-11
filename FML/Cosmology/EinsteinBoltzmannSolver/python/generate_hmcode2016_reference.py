#!/usr/bin/env python3
"""Regenerate the identical-linear-input HMcode-2016 fixtures (CAMB halofit_version = mead2016) used by
tests/test_hmcode2016.cpp (the HMcode-2016 prescription of FML/HaloModel).

Run from EinsteinBoltzmannSolver with OMP_NUM_THREADS=1. Requires camb and numpy (the fixtures were made with
CAMB 2.0.5). Same cosmologies as generate_nonlinear_reference.py; the linear spectra and the nonlinear reference
come from the same CAMB run.
"""
from pathlib import Path
import numpy as np
import camb

out = Path(__file__).resolve().parents[1] / "tests/reference/nonlinear2016"
out.mkdir(parents=True, exist_ok=True)
h, ob, oc, ns, As = .67, .05, .267, .965, 2.1e-9
zs = [2., 1., 0.]
kh = np.geomspace(1e-5, 1e7, 1001)
kr = np.geomspace(.01, 10., 31)
for name, mnu, w, wa, ok in [("lcdm", 0., -1., 0., 0.),
                              ("nu", .3, -1., 0., 0.),
                              ("w0wa", 0., -.8, .2, .02)]:
    p = camb.CAMBparams()
    p.set_cosmology(H0=100*h, ombh2=ob*h*h, omch2=oc*h*h, mnu=mnu,
                    nnu=3.046, omk=ok, TCMB=2.7255)
    p.InitPower.set_params(As=As, ns=ns)
    p.set_dark_energy(w=w, wa=wa, dark_energy_model='ppf')
    p.set_matter_power(redshifts=zs, kmax=100., k_per_logint=20)
    p.WantCls = False
    p.NonLinear = camb.model.NonLinear_pk
    p.NonLinearModel.set_params(halofit_version='mead2016')
    r = camb.get_results(p)
    pm = r.get_matter_power_interpolator(nonlinear=False, extrap_kmax=1e7)
    pcb = r.get_matter_power_interpolator(nonlinear=False, var1='delta_nonu', var2='delta_nonu',
                                         extrap_kmax=1e7)
    pnl = r.get_matter_power_interpolator(nonlinear=True)
    # Metadata: reference package/version and cosmology; all k in h/Mpc, all P in (Mpc/h)^3.
    header = f"CAMB {camb.__version__}, mead2016; h={h} Ob={ob} Oc={oc} mnu={mnu} w={w} wa={wa} Ok={ok}"
    np.savetxt(out / f"{name}_linear.txt", np.column_stack([kh] +
               [v for z in zs for v in (pm.P(z, kh), pcb.P(z, kh))]),
               fmt='%.10e', header=header + "\nk Pm_z2 Pcb_z2 Pm_z1 Pcb_z1 Pm_z0 Pcb_z0")
    np.savetxt(out / f"{name}_hmcode2016.txt", np.column_stack([kr] + [pnl.P(z, kr) for z in zs]),
               fmt='%.10e', header=header + "\nk Pnl_z2 Pnl_z1 Pnl_z0")
    print(name, 'HMcode-2016 reference written', flush=True)
