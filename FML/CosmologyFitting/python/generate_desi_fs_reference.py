#!/usr/bin/env python3
"""Reference ln L of the DESI DR1 full-shape likelihood for tests/test_fitting.cpp, computed with DESI's own code path
(desi-kp-cosmological-likelihoods dr1/cobaya: reptvelocileptors.calculate / get_pkpoles and desi_fs_bao_all.logp with
observable_name = spectrum-poles-rotated, solve = marg) for fixed REPT inputs (tests/reference/desi_fs_rept_inputs.txt:
P_cb, its no-wiggle part, P_thetatheta, q_par, q_per, sigma_8, sigma_8,thetatheta at the six effective redshifts).

Usage (from CosmologyFitting/): python3 python/generate_desi_fs_reference.py /path/to/data/likelihood
Needs numpy, scipy, h5py and velocileptors 3.1 (pip install velocileptors). Prints the six ln L.
"""
import sys

import h5py
import numpy as np
from scipy import interpolate
from velocileptors.EPT.ept_fullresum_varyDz_nu_fftw import REPT

SETS = [("BGS_BRIGHT-21.5", (0.1, 0.4)), ("LRG", (0.4, 0.6)), ("LRG", (0.6, 0.8)), ("LRG", (0.8, 1.1)),
        ("ELG_LOPnotqso", (1.1, 1.6)), ("QSO", (0.8, 2.1))]
# biases (b1p, b2p, bsp) near the maximum of each sample for the input cosmology
BIAS = [[1.0627, 0.5317, -0.2066], [1.0879, -0.5214, -0.4183], [1.1893, -0.6044, 0.0337], [1.0164, -0.2913, 0.0727],
        [0.5153, -0.0597, 0.2002], [0.7917, 0.0974, 0.1014]]
STOCH = {"BGS": (0.15, 150 * 10**(1 / 3) * 1.2**0.5 / 70.), "LRG": (0.15, 150 * 10**(1 / 3) * 1.8**0.5 / 70.),
         "ELG": (0.10, 150 * 2.1**0.5 / 70.), "QSO": (0.03, 150 * 10**(0.7 / 3) * 2.4**0.5 / 70.)}


def main():
    ddir = sys.argv[1]
    inp = "tests/reference/desi_fs_rept_inputs.txt"
    meta = [l.split() for l in open(inp) if l.startswith("# z")]
    qpar, qper, s8, s8t = (np.array([float(m[i]) for m in meta]) for i in (4, 6, 8, 10))
    A = np.loadtxt(inp)
    kin, nz = A[:, 0], len(meta)
    pk_dd, pknow_dd, pk_tt = (np.stack([A[:, c + 3 * i] for i in range(nz)], axis=-1) for c in (1, 2, 3))
    flat, win, prec, sns = [], [], [], []
    for tr, (z1, z2) in SETS:
        with h5py.File(f"{ddir}/likelihood_spectrum-poles-rotated_syst-rotation-hod-photo_{tr}_GCcomb_z{z1:.1f}-{z2:.1f}_thetacut0.05.h5", "r") as f:
            flat.append(np.concatenate([f["observable/spectrum/0/value"][()], f["observable/spectrum/2/value"][()]]))
            win.append(f["window/value"][()])
            prec.append(np.linalg.inv(f["covariance/value"][()]))
            sns.append(np.mean(f["observable/spectrum/0/num_shotnoise"][()] / f["observable/spectrum/0/norm"][()]))
            k = f["window/theory/spectrum/0/k"][()]
    # reptvelocileptors.calculate
    opts = dict(rbao=110, sbao=None, beyond_gauss=True, one_loop=True, shear=True, cutoff=20, jn=5, N=4000, threads=1,
                extrap_min=-4, extrap_max=3, import_wisdom=False)
    pt = REPT(kin, pk_dd[..., 0], pnw=pknow_dd[..., 0], kmin=k[0], kmax=k[-1], nk=200, **opts)
    pcb, pcb_nw, ptt = [10**interpolate.interp1d(np.log10(kin), np.log10(pk), kind="cubic", fill_value="extrapolate", axis=0,
                                                 assume_sorted=True)(np.log10(np.append(pt.kv, 1.))) for pk in [pk_dd, pknow_dd, pk_tt]]
    tables = {ell: [] for ell in (0, 2, 4)}
    for iz in range(nz):
        Dz = np.sqrt(pcb[-1, iz] / pcb[-1, 0])
        fk = np.sqrt(ptt[:-1, iz] / pcb[:-1, iz])
        pks = pt.compute_redshift_space_power_multipoles_tables(fk, apar=qpar[iz], aperp=qper[iz], ngauss=4,
                                                               pcb=pcb[:-1, iz], pcb_nw=pcb_nw[:-1, iz], Dz=Dz)[1:]
        for ill, ell in enumerate(tables):
            tables[ell].append(pks[ill])
    tables = np.array([np.stack(tables[ell], axis=-1) for ell in (0, 2, 4)])
    tables = interpolate.interp1d(pt.kv, tables, kind="cubic", fill_value="extrapolate", axis=1, assume_sorted=True)(k)
    # get_pkpoles (physical basis) and desi_fs_bao_all.logp
    all_marg = ["alpha0p", "alpha2p", "alpha4p", "alpha6p", "sn0p", "sn2p", "sn4p"]
    scales = np.array([12.5] * 4 + [2.] + [5.] * 2)
    gi = np.array([all_marg.index(p) for p in ["alpha0p", "alpha2p", "sn0p", "sn2p"]])
    prior_h = -np.diag(scales[gi]**-2.)
    out = []
    for i, (tr, _) in enumerate(SETS):
        fsat, sigv = STOCH[tr[:3]]
        sigma8, f = s8[i], s8t[i] / s8[i]
        b1L, b2L, bsL = BIAS[i][0] / sigma8 - 1., BIAS[i][1] / sigma8**2, BIAS[i][2] / sigma8**2
        b1, b2, bs, b3 = 1. + b1L, 8. / 21. * b1L + b2L, bsL - 2 / 7 * b1L, b1L
        grad = np.zeros((7, 7))
        grad[0, 0] = (1 + b1L)**2
        grad[1, 0] = grad[1, 1] = f * (1 + b1L)
        grad[2, 1] = f**2
        grad[2, 2] = f * (1 + b1L)
        grad[3, 2] = f**2
        for ii, ell in enumerate([0, 2, 4]):
            grad[ii - 3, ii - 3] = sns[i] * (fsat if ell > 0 else 1.) * sigv**ell
        tab = tables[..., i]
        mono = np.concatenate([[1, b1, b1**2, b2, b1 * b2, b2**2, bs, b1 * bs, b2 * bs, bs**2, b3, b1 * b3], np.zeros(7)])
        theory = win[i].dot(np.sum(tab * mono, axis=-1).ravel())
        g = win[i].dot(tab[..., -7:].dot(grad)[..., gi].reshape(-1, len(gi)))
        diff = theory - flat[i]
        ll = -0.5 * diff.dot(prec[i]).dot(diff)
        pg = prec[i].dot(g)
        lg, lh = -pg.T.dot(diff), -g.T.dot(pg)
        ph = prior_h + lh
        dx = -np.linalg.solve(ph, lg)
        ll += 0.5 * dx.dot(lh).dot(dx) + lg.dot(dx) - 0.5 * np.linalg.slogdet(-ph)[1] + 0.5 * dx.dot(prior_h).dot(dx)
        out.append(ll)
    print(" ".join(f"{v:.12f}" for v in out))


if __name__ == "__main__":
    main()
