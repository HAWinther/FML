# Planck 2018 plik_lite data

Data files of the Planck 2018 high-ell plik_lite likelihood and of the two low-ell TT bins, as distributed with
planck-lite-py (https://github.com/heatherprince/planck-lite-py, commit 2c0d0f6). The data belong to the Planck
Collaboration (Planck Legacy Archive, https://pla.esac.esa.int/#cosmology).

| File | Content |
|---|---|
| `cl_cmb_plik_v22.dat` | bin centre, binned C_ell [muK^2], error: TT (215 bins, 30-2508), TE (199, 30-1996), EE (199, 30-1996) |
| `c_matrix_plik_v22.dat` | covariance of the 613 bins (Fortran unformatted record, lower triangle used) |
| `blmin.dat`, `blmax.dat`, `bweight.dat` | bin edges (ell - 30) and weights of each multipole |
| `CTT_bin_low_ell_2018.dat`, `blmin_low_ell.dat`, `blmax_low_ell.dat`, `bweight_low_ell.dat` | two Gaussian bins compressing the Commander TT likelihood for 2 <= ell <= 29 |

References: Planck Collaboration 2018 V (arXiv:1907.12875) for plik_lite; Prince & Dunkley 2019
(arXiv:1909.05869) for the low-ell bins. Used by `PlanckLiteLikelihood` (src/mcmc/Likelihoods.h);
`tests/reference/Dl_planck2015fit.dat` (CLASS spectra from planck-lite-py) and its ln L values check the
implementation.
