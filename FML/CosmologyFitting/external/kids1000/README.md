# KiDS-1000 cosmic shear (COSEBIs)

Data of the `kids1000_cosebis` likelihood, from the KiDS-1000 cosmic-shear data release
(<https://kids.strw.leidenuniv.nl/DR4/KiDS-1000_cosmicshear.php>, `KiDS1000_cosmic_shear_data_release.tgz`;
Asgari et al. 2021, arXiv:2007.15633; Hildebrandt et al. 2021 for the redshift distributions), converted with
`python/convert_kids1000.py <release folder>`:

| File | Content |
|---|---|
| `cosebis_En.txt` | bin1 bin2 n E_n: COSEBIs of the 15 pairs of the 5 tomographic bins, n = 1..5 (the fiducial scale cut), theta = 0.5' - 300' (from `data_fits/cosebis_KIDS1000_BlindC_with_m_bias_..._SOM_Fid.fits`, extension `En`) |
| `cosebis_cov.txt` | their 75 x 75 covariance (extension `COVMAT`, rows of n <= 5; includes the shear-calibration uncertainty) |
| `nz_source.txt` | z_low z_mid z_high n_1 .. n_5: the SOM redshift distributions (extension `NZ_SOURCE`, histograms with Delta z = 0.05) |
| `dz_prior.txt` | mean and covariance of the photo-z shifts delta_z_i (n_i(z) -> n_i(z - delta_z_i)) |

The release does not include the covariance file of the shifts used by its pipeline (`SOM_cov_multiplied.asc`).
Its chain samples the shifts as bias = L u with uncorrelated u ~ N(mu_u, 1) (`correlated_dz_priors`, `priors.ini`),
so `convert_kids1000.py` recovers L (exactly: the residual of the linear fit is 1e-16) and writes C = L L^T, which
has the round values of the file (1.1201e-4, ...), and the mean L mu_u = (0, -0.0020, -0.0130, -0.0110, 0.0060)
(Hildebrandt et al.: 0, -0.002, -0.013, -0.011, 0.006).
