# ReACT reference tools

FML computes the halo-model reaction of ReACT natively (`HaloModel/HaloModelReaction.h`,
`SphericalCollapse/ScreenedSphericalCollapse.h`, `PerturbationTheory/MGOneLoop.h`; solver: `nonlinear = react`).
The reference values in `..` were made with the original ACTio-ReACTio code (github.com/nebblu/ACTio-ReACTio,
pinned to a36dda02). These tools regenerate them; they are not part of FML and nothing in FML links to ReACT.

| File | |
|---|---|
| `build_backend.py`, `CMakeLists.txt` | Build ACTio-ReACTio and SUNDIALS 4.1.0 privately into `.backend/` (needs GSL, CMake, git) |
| `generate_reference.py` | `../<name>_linear.txt`, `../cpl_nu_cosmology.txt`: CAMB linear spectra (GR, CPL with 0.3 eV neutrinos) and ReACT's modified growth for f(R) (\|fR0\| = 1e-5) and nDGP (Omega_rc = 0.1); z = 0, 0.5, 1, 2 |
| `halomodel_oracle.cpp` | `../halomodel_<name>.txt`: ReACT's per-redshift halo-model quantities (delta_c, sigma, one-halo terms, E, k*, the one-loop P(k0), R) for those spectra |
| `scol_oracle.cpp` | The spherical-collapse values in `tests/test_screened_collapse.cpp` |

```sh
python3 build_backend.py --gsl-prefix "$HOME/local"
OMP_NUM_THREADS=1 python3 generate_reference.py --upstream .build/ACTio-ReACTio --backend .backend/lib/libfml_react_backend.so
# then build the oracles as described at the top of each file
```

Known differences of ReACT (a36dda02) from FML, both kept out of the references' use in the tests:
- its f(R) third-order screening term gamma3 (BeyondLCDM.cpp) is 1/3 of the correct one (Koyama, Taruya & Hiramatsu
  2009 eq. 3.6; Bose & Koyama 2016 eqs. 2.14, 2.34); FML's `mg_hu_sawicki(..., react_gamma3 = true)` and the solver's
  `react_compat_gamma3 = true` reproduce it;
- its spherical collapse for modified gravity brackets delta_ini in [1.1, 4.4] zeta_i / a (SCOL.cpp), which pins
  delta_c at the bracket edge at high redshift (5% low at z = 2.3 for sigma_8 = 0.8).
Its numerics are reproducible only to ~1e-3 in R (spherical collapse with CVODE and bisection; macOS and Linux
builds differ by up to 1.3e-3), and its angular quadrature of the one-loop P misses ~4% of the loop at k0 = 0.06.
