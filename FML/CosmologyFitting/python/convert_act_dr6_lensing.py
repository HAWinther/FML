#!/usr/bin/env python3
"""Convert the ACT DR6 lensing likelihood data (ACT_dr6_likelihood_v1.2.tgz) for FML's act_dr6_lensing likelihoods.

Usage (from CosmologyFitting/):
  python3 python/convert_act_dr6_lensing.py [ACT_dr6_likelihood_v1.2.tgz or its extracted v1.2 folder]
Without an argument the tarball is downloaded (360 MB) from NASA LAMBDA:
  https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/ACT_dr6_likelihood_v1.2.tgz
Needs numpy. Writes external/act_dr6_lensing/<variant>_*.txt for the variants act_baseline, act_extended,
actplanck_baseline and actplanck_extended.

The likelihood is that of act_dr6_lenslike 1.2.1 (Madhavacheril et al. 2023, arXiv:2304.05203; Qu et al. 2023,
arXiv:2304.05202; with Planck PR4 lensing, Carron, Mirmelstein & Lewis 2022, arXiv:2206.07773) as used by its Cobaya
likelihood (like_corrections on, Hartlap correction with 792 ACT and 400 Planck simulations, trim_lmax = 2998).
Its theory vector is linear in the spectra: the binned C_L^kk plus the corrections for the dependence of the
normalization and of N1 on the CMB and lensing spectra,

  model = B_kk C^kk + sum_X M_X (C^X - C^X_fid) + M_N1kk (C^kk - C^kk_fid),   X = TT, EE, BB, TE,

with M_X = B diag(C^kk_fid / A_fid) (-2 dA/dC^X) + B dN1/dC^X for the ACT and the Planck parts. These matrices are
computed here (rows = bandpowers, columns = L = 0..Lmax), so the C++ side only applies them:

  <variant>_data.txt     the bandpowers (ACT, then Planck)
  <variant>_cinv.txt     the inverse covariance times the Hartlap factor
  <variant>_kk.txt       the matrix for C_L^kk = [L(L+1)]^2 C_L^phiphi / 4 (binning + N1 correction)
  <variant>_tt.txt, _ee.txt, _bb.txt, _te.txt   the matrices for the lensed CMB C_ell [muK^2] (no ell factors)
  <variant>_offset.txt   the constant: -sum_X M_X C^X_fid - M_N1kk C^kk_fid
"""
import io
import os
import sys
import tarfile
import urllib.request

import numpy as np

_cache = {}


def load(path, **kw):
    """np.loadtxt / np.load with a cache (the variants share the large correction matrices)"""
    key = (path, tuple(sorted(kw.items())))
    if key not in _cache:
        _cache[key] = np.load(path) if path.endswith(".npy") else np.loadtxt(path, **kw)
    return _cache[key]

URL = "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/ACT_dr6_likelihood_v1.2.tgz"
OUT = os.path.join("external", "act_dr6_lensing")
TRIM_LMAX = 2998
NLEN = TRIM_LMAX + 2  # theory arrays L = 0..2999, as act_dr6_lenslike's standardize
NSIMS_ACT, NSIMS_PLANCK = 792.0, 400.0  # the Cobaya likelihood's values


def standardize(ls, cls, extra_dims="y"):
    cstart = int(ls[0])
    assert np.all(np.isclose(np.diff(ls), 1.0)) and cstart <= 2
    cend = NLEN - cstart
    if extra_dims == "xyy":
        out = np.zeros((cls.shape[0], NLEN, NLEN))
        out[:, cstart:, cstart:] = cls[:, :cend, :cend]
    elif extra_dims == "yy":
        out = np.zeros((NLEN, NLEN))
        out[cstart:, cstart:] = cls[:cend, :cend]
    elif extra_dims == "xy":
        out = np.zeros((cls.shape[0], NLEN))
        out[:, cstart:] = cls[:, :cend]
    else:
        out = np.zeros(NLEN)
        out[cstart:] = cls[:cend]
    return out


def find_data_dir(arg):
    if arg is not None and os.path.isdir(arg):
        for d in (arg, os.path.join(arg, "v1.2")):
            if os.path.exists(os.path.join(d, "clkk_bandpowers_act.txt")):
                return d
        sys.exit(f"{arg}: no clkk_bandpowers_act.txt (expected the extracted v1.2 folder)")
    tmp = os.path.join(OUT, "raw")
    if not os.path.exists(os.path.join(tmp, "v1.2", "clkk_bandpowers_act.txt")):
        if arg is None:
            print(f"downloading {URL} (360 MB) ...", flush=True)
            data = urllib.request.urlopen(URL).read()
            tf = tarfile.open(fileobj=io.BytesIO(data))
        else:
            tf = tarfile.open(arg)
        os.makedirs(tmp, exist_ok=True)
        print(f"extracting into {tmp} ...", flush=True)
        tf.extractall(tmp)
    for root, _, files in os.walk(tmp):
        if "clkk_bandpowers_act.txt" in files:
            return root
    sys.exit("clkk_bandpowers_act.txt not found in the tarball")


def convert(ddir, variant):
    baseline = "_extended" not in variant
    include_planck = variant.startswith("actplanck")
    start, end = (2, -6) if baseline else (2, -3)

    # fiducial spectra (C_ell, no ell factors)
    f_ls, f_tt, f_ee, f_bb, f_te = np.loadtxt(f"{ddir}/like_corrs/cosmo2017_10K_acc3_lensedCls.dat", unpack=True)
    fac = 2.0 * np.pi / (f_ls * (f_ls + 1.0))
    fid = {s: standardize(f_ls, c * fac) for s, c in zip(["tt", "ee", "bb", "te"], [f_tt, f_ee, f_bb, f_te])}
    fd_ls, f_dd = np.loadtxt(f"{ddir}/like_corrs/cosmo2017_10K_acc3_lenspotentialCls.dat", unpack=True, usecols=[0, 5])
    fid["kk"] = standardize(fd_ls, f_dd * 2.0 * np.pi / 4.0)

    # bandpowers, binning and covariance
    y = np.loadtxt(f"{ddir}/clkk_bandpowers_act.txt")
    nact_tot = y.size
    data = y[start:end].copy()
    binmat = np.loadtxt(f"{ddir}/binning_matrix_act.txt")
    B_act = standardize(np.arange(binmat.shape[1]), binmat[start:end, :], "xy")
    fcov = np.loadtxt(f"{ddir}/covmat_actplanck.txt" if include_planck else f"{ddir}/covmat_act.txt")
    sel = np.s_[nact_tot + end : nact_tot]
    cov = np.delete(np.delete(fcov, sel, 0), sel, 1)
    sel = np.s_[:start]
    cov = np.delete(np.delete(cov, sel, 0), sel, 1)
    parts = [("", B_act)]
    if include_planck:
        data = np.append(data, np.loadtxt(f"{ddir}/clkk_bandpowers_planck.txt"))
        binmat = np.loadtxt(f"{ddir}/binning_matrix_planck.txt")
        parts.append(("_planck", standardize(np.arange(binmat.shape[1]), binmat, "xy")))
    nbins = data.size
    nsims = min(NSIMS_ACT, NSIMS_PLANCK) if include_planck else NSIMS_ACT
    cinv = np.linalg.inv(cov) * (nsims - nbins - 2.0) / (nsims - 1.0)

    # the linear corrections (act_dr6_lenslike.get_corrected_clkk with like_corrections). act_dr6_lenslike
    # standardizes all N1 derivatives with the multipoles of the last normalization file read (Planck's if included)
    n1_ls = np.loadtxt(f"{ddir}/like_corrs/n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax4000.txt")[0]
    if include_planck:
        n1_ls = np.loadtxt(f"{ddir}/like_corrs/PLANCK_n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax3000.txt")[0]
    rows = {s: [] for s in ["kk", "tt", "ee", "bb", "te"]}
    for suff, B in parts:
        if suff == "":
            cmat = load(f"{ddir}/like_corrs/norm_correction_matrix_Lmin0_Lmax4000.npy")
            fAL_ls, fAL = np.loadtxt(f"{ddir}/like_corrs/n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax4000.txt")
            n1name = "N1der_{}_lmin600_lmax3000_full.txt"
        else:
            cmat = load(f"{ddir}/like_corrs/P18_norm_correction_matrix_Lmin0_Lmax3000.npy")
            fAL_ls, fAL = np.loadtxt(f"{ddir}/like_corrs/PLANCK_n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax3000.txt")
            n1name = "N1_planck_der_{}_lmin100_lmax2048.txt"
        dNorm = standardize(np.arange(cmat.shape[1]), cmat, "xyy")
        fAL = standardize(fAL_ls, fAL)
        # norm_corr = sum_i c_i with c_i = -2 dNorm[i] @ cldiff, divided by fAL for L >= 2; clkk += norm_corr * clkk_fid
        ls = np.arange(NLEN)
        scale = np.where(ls >= 2, fid["kk"] / np.where(ls >= 2, fAL, 1.0), fid["kk"])
        dN1 = {
            s: standardize(n1_ls, load(f"{ddir}/like_corrs/" + n1name.format(s.upper())), "yy")
            for s in ["kk", "tt", "ee", "bb", "te"]
        }
        rows["kk"].append(B + B @ dN1["kk"])
        for i, s in enumerate(["tt", "ee", "bb", "te"]):
            rows[s].append(B @ (scale[:, None] * (-2.0 * dNorm[i])) + B @ dN1[s])
    M = {s: np.vstack(v) for s, v in rows.items()}
    # model = M_kk C^kk + sum_X M_X C^X + offset; the binning part of M_kk has no fiducial subtraction
    B_all = np.vstack([B for _, B in parts])
    offset = -(M["kk"] - B_all) @ fid["kk"] - sum(M[s] @ fid[s] for s in ["tt", "ee", "bb", "te"])

    os.makedirs(OUT, exist_ok=True)
    lmax_used = 0
    for s in M:
        nz = np.nonzero(np.any(M[s] != 0.0, axis=0))[0]
        lmax_used = max(lmax_used, int(nz[-1]) if nz.size else 0)
    head = f"{variant}: {nbins} bandpowers; columns L = 0..{lmax_used}"
    np.savetxt(f"{OUT}/{variant}_data.txt", data, header=head)
    np.savetxt(f"{OUT}/{variant}_cinv.txt", cinv, header=head)
    np.savetxt(f"{OUT}/{variant}_offset.txt", offset, header=head)
    for s in M:
        np.savetxt(f"{OUT}/{variant}_{s}.txt", M[s][:, : lmax_used + 1], header=head)
    print(f"  {variant}: {nbins} bandpowers, theory needed up to L = {lmax_used}")
    return fid


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    ddir = find_data_dir(arg)
    print(f"data in {ddir}")
    for v in ["act_baseline", "act_extended", "actplanck_baseline", "actplanck_extended"]:
        fid = convert(ddir, v)
    # the fiducial spectra, for the tests (L, C^TT, C^EE, C^BB, C^TE [muK^2], C^kk)
    L = np.arange(NLEN)
    np.savetxt(f"{OUT}/fiducial_cls.txt", np.column_stack([L, fid["tt"], fid["ee"], fid["bb"], fid["te"], fid["kk"]]),
               header="L, C_TT, C_EE, C_BB, C_TE [muK^2, no ell factors], C_kk")


if __name__ == "__main__":
    main()
