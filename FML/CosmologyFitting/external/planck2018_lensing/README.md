# Planck 2018 CMB lensing likelihood data

Planck 2018 lensing-potential power spectrum from the minimum-variance (T+P) SMICA reconstruction, conservative
multipole range 8 ≤ L ≤ 400 in 9 bandpowers (Planck 2018 VIII, arXiv:1807.06210): the `consext8` CMBlikes
dataset in plain text, as distributed with CosmoMC and Cobaya
(https://github.com/CobayaSampler/planck_supp_data_and_covmats, release v2.1, `lensing/2018/`).

| File | Content |
|---|---|
| `..._consext8_bandpowers.dat` | bin, L_min, L_max, L_av, PP = [L(L+1)]² C_L^φφ / 2π, error |
| `..._consext8_cov.dat` | 9 × 9 bandpower covariance |
| `..._consext8_window/window<b>.dat` | binning of [L(L+1)]² C_L^φφ / 2π into bandpower b |
| `..._consext8_lens_delta_window/window<b>.dat` | linear response of bandpower b to TT, EE, TE (D_ℓ in μK²) and PP: the dependence of the N0 and N1 bias corrections on the true CMB and lensing spectra |
| `..._consext8_lensing_fiducial_correction.dat` | the same response evaluated for the fiducial model (subtracted) |
| `..._consext8_CMBmarged*` | variant marginalized over the CMB spectra (PP response only, larger covariance) |
| `*.dataset` | the CMBlikes descriptions (for reference) |

Based on observations obtained with Planck (http://www.esa.int/Planck), an ESA science mission with instruments
and contributions directly funded by ESA Member States, NASA, and Canada.
