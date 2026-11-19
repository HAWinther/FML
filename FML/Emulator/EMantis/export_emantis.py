#!/usr/bin/env python3
"""Export the trained e-MANTIS emulators to the plain-text files read by EMantis.h.

e-MANTIS (Saez-Casares et al., https://gitlab.obspm.fr/e-mantis/e-mantis, GPL-3.0-or-later) ships the fitted
hyperparameters of its Gaussian processes together with the training data, and scikit-learn solves K alpha = y when
the emulator is loaded. This script loads every shipped emulator with the Python package and writes out what the
prediction needs, with the GP weights alpha = K^-1 y already computed, so the C++ side never sees the training
targets:

  - the parameter names and ranges, and the training parameters after the StandardScaler,
  - per node (scale factor): the bins (binned data) or B-spline knots and degree, the PCA (if any),
  - per Gaussian process: the kernel (Matern nu or RBF), its amplitude and length scales, the white noise level,
    the normalize_y mean and std, alpha, the regularisation added to the diagonal of K (for the predicted std)
    and the gp_std_factor of the output (if any).

Usage:  python3 export_emantis.py OUTPUT_DIR        (needs: pip install emantis==1.4.0)

The C++ code reproduces the Python predictions to ~1e-10 relative, the floating point reproducibility of the GP sums
(see tests/test_emantis.cpp, whose reference values are written by tests/make_emantis_reference.py).
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, Product, Sum, WhiteKernel

import emantis
from emantis._gp_emulation import GaussianProcessEmulator1D, GaussianProcessEmulator1Dx1D
from emantis.halo_mass_function import HMFEmulator
from emantis.matter_power_spectrum import NonLinearBoostEmulator, NonLinearMGBoostEmulator, Sigma8Emulator

FORMAT_VERSION = 1


def fmt(values):
    return " ".join(f"{float(v):.17g}" for v in np.atleast_1d(values))


def kernel_values(kernel):
    """(nu, amplitude, length_scales, noise_level) of ConstantKernel * (Matern|RBF) [+ WhiteKernel]."""
    noise = 0.0
    if isinstance(kernel, Sum):
        if not isinstance(kernel.k2, WhiteKernel):
            raise ValueError(f"Unsupported kernel {kernel}")
        noise = kernel.k2.noise_level
        kernel = kernel.k1
    if not (isinstance(kernel, Product) and isinstance(kernel.k1, ConstantKernel)):
        raise ValueError(f"Unsupported kernel {kernel}")
    k2 = kernel.k2
    if isinstance(k2, Matern):
        nu = float(k2.nu)
        if nu not in (0.5, 1.5, 2.5, np.inf):
            raise ValueError(f"Unsupported Matern nu = {nu}")
        if nu == np.inf:
            nu = -1.0
    elif isinstance(k2, RBF):
        nu = -1.0  # RBF
    else:
        raise ValueError(f"Unsupported kernel {kernel}")
    return nu, kernel.k1.constant_value, np.atleast_1d(k2.length_scale), noise


def write_emulator(path, emu, kind, extra):
    """emu: GaussianProcessEmulator1Dx1D (nodes in aexp) or GaussianProcessEmulator1D (one node, no node variable)."""
    if isinstance(emu, GaussianProcessEmulator1Dx1D):
        nodes = list(emu._nodes)
        node_emus = [emu._node_emulators[n] for n in nodes]
        config_node = emu._config_emu.node
        node_line = f"nodes {config_node.variable_name} {config_node.x_interp_type} {int(config_node.constant_bin_range)} {len(nodes)}"
    elif isinstance(emu, GaussianProcessEmulator1D):
        nodes = [0.0]
        node_emus = [emu]
        node_line = "nodes none lin 1 1"
    else:
        raise TypeError(type(emu))

    first = node_emus[0]
    config_data = first._config_emu.data
    params_range = emu.params_range
    names = list(params_range)
    scaler = first._params_scaler
    X = first._gp_regressor_list[0].X_train_
    for e in node_emus:
        if not np.array_equal(e._params_scaler.mean_, scaler.mean_) or not np.array_equal(e._params_scaler.scale_, scaler.scale_):
            raise ValueError("The parameter scaler differs between nodes")
        for gp in e._gp_regressor_list:
            if not np.array_equal(gp.X_train_, X):
                raise ValueError("The training parameters differ between GPs")

    lines = [f"FML_EMANTIS {FORMAT_VERSION}"]
    lines.append(f"# Exported from e-MANTIS {emantis.__version__} by export_emantis.py. e-MANTIS: I. Saez-Casares et al.,")
    lines.append("# https://gitlab.obspm.fr/e-mantis/e-mantis (GPL-3.0-or-later). Cite the e-MANTIS papers when using it.")
    lines.append(f"name {emu._name}")
    lines.append(f"kind {kind}")
    for key, value in extra.items():
        lines.append(f"{key} {value}")
    if config_data.data_type == "binned":
        lines.append(f"data binned {config_data.variable_name} {config_data.x_interp_type} {config_data.y_transform}")
    else:
        lines.append(f"data bspline {config_data.variable_name} {config_data.x_scale} {config_data.y_scale}")
    lines.append(f"params {len(names)}")
    for name in names:
        lines.append(f"{name} {fmt([params_range[name]['min_value'], params_range[name]['max_value']])}")
    lines.append(f"scaler_mean {fmt(scaler.mean_)}")
    lines.append(f"scaler_scale {fmt(scaler.scale_)}")
    lines.append(f"train {X.shape[0]}")
    for row in X:
        lines.append(fmt(row))
    lines.append(node_line)
    for node, e in zip(nodes, node_emus):
        lines.append(f"node {node:.17g}")
        if config_data.data_type == "binned":
            bins = e._data_bins
            lines.append(f"bins {0 if bins is None else len(bins)} {fmt(bins) if bins is not None else ''}".rstrip())
        else:
            lines.append(f"bspline {int(e._bspline_degree)} {len(e._data_bins)} {fmt(e._data_bins)}")
        if e._pca_enabled:
            pca = e._pca_transformer
            ncomp, nfeat = pca.components.shape
            lines.append(f"pca {ncomp} {nfeat}")
            lines.append(f"pca_scaler_mean {fmt(pca.scaler_mean)}")
            lines.append(f"pca_scaler_scale {fmt(pca.scaler_scale)}")
            lines.append(f"pca_mean {fmt(pca.pca_mean)}")
            for row in pca.components:
                lines.append(fmt(row))
        else:
            lines.append("pca 0 0")
        gps = e._gp_regressor_list
        factor = e._gp_std_factor
        lines.append(f"gps {len(gps)} {int(factor is not None)}")
        if factor is not None:
            lines.append(f"gp_std_factor {fmt(factor)}")
        for gp in gps:
            nu, amp, ls, noise = kernel_values(gp.kernel_)
            if len(ls) == 1:
                ls = np.full(X.shape[1], ls[0])
            if not gp.normalize_y:
                ymean, ystd = 0.0, 1.0
            else:
                ymean, ystd = float(np.squeeze(gp._y_train_mean)), float(np.squeeze(gp._y_train_std))
            reg = np.broadcast_to(np.asarray(gp.alpha, dtype=float), (X.shape[0],))
            lines.append(f"gp {fmt([nu, amp, noise, ymean, ystd])} {fmt(ls)}")
            lines.append(f"alpha {fmt(np.ravel(gp.alpha_))}")
            if np.all(reg == reg[0]):
                lines.append(f"reg 1 {fmt(reg[0])}")
            else:
                lines.append(f"reg {len(reg)} {fmt(reg)}")
    lines.append("end")
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"wrote {path} ({Path(path).stat().st_size / 1e6:.2f} MB)")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for model in ("LCDM", "wCDM"):
        e = Sigma8Emulator(model, verbose=False)
        write_emulator(out / f"sigma8_{model}.txt", e._emulator, "sigma8", {"model": model})
    e = NonLinearMGBoostEmulator("fR", verbose=False)
    write_emulator(out / "pk_nl_mg_boost_fR.txt", e._emulator, "pk_nl_mg_boost", {"model": "fR"})
    e = NonLinearBoostEmulator("wCDM", verbose=False)
    write_emulator(out / "pk_nl_boost_wCDM.txt", e._emulator, "pk_nl_boost", {"model": "wCDM"})
    for model in ("fR", "wCDM"):
        for mass_def in ("200c", "500c", "1000c", "b0.2"):
            for res_corr in (True, False):
                e = HMFEmulator(model, mass_def, use_resolution_correction=res_corr, verbose=False)
                write_emulator(out / f"hmf_{model}_{mass_def}_res_corr_{int(res_corr)}.txt", e._emulator, "hmf",
                               {"model": model, "mass_def": mass_def, "res_corr": int(res_corr)})


if __name__ == "__main__":
    main()
