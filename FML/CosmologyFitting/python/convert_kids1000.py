#!/usr/bin/env python3
"""Convert the KiDS-1000 cosmic-shear data release (Asgari et al. 2021, arXiv:2007.15633;
https://kids.strw.leidenuniv.nl/DR4/KiDS-1000_cosmicshear.php) into the text files of external/kids1000/ read by the
kids1000_cosebis likelihood.

    python3 python/convert_kids1000.py <path>/KiDS1000_cosmis_shear_data_release

Needs astropy and numpy. Writes
  cosebis_En.txt     bin1 bin2 n E_n: the COSEBIs of the 15 bin pairs, n = 1..5 (the fiducial scale cut
                     keep_ang_En = 0.5 5.5 of the release's pipeline.ini), theta = 0.5' - 300'
  cosebis_cov.txt    their 75 x 75 covariance (includes the uncertainty of the shear calibration m)
  nz_source.txt      z_low z_mid z_high n_1..n_5: the SOM redshift distributions (histograms, Delta z = 0.05)
  dz_prior.txt       mean (row 1) and covariance (rows 2-6) of the shifts delta_z_i, with n_i(z) -> n_i(z - delta_z_i).
                     The release does not contain the covariance file of its pipeline (SOM_cov_multiplied.asc): it is
                     recovered here from the chain, where the sampled shifts are the linear map bias = L u of the
                     uncorrelated parameters u (L the Cholesky factor of the covariance; exact to 1e-16).
"""
import sys
from pathlib import Path
import numpy as np
import astropy.io.fits as fits

release = Path(sys.argv[1])
out = Path(__file__).resolve().parents[1] / "external/kids1000"
out.mkdir(parents=True, exist_ok=True)
tag = "KIDS1000_BlindC_with_m_bias_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_goldclasses_Flag_SOM_Fid"
F = fits.open(release / "data_fits" / f"cosebis_{tag}.fits")
d = F["En"].data
cov = F["COVMAT"].data
keep = np.where(d["ANGBIN"] <= 5)[0]
assert len(keep) == 75
src = "# KiDS-1000 COSEBIs (Asgari et al. 2021), from the data release file cosebis_" + tag + ".fits\n"
with open(out / "cosebis_En.txt", "w") as f:
    f.write(src + "# theta_min = 0.5 arcmin, theta_max = 300 arcmin; bin1 bin2 n E_n\n")
    for i in keep:
        f.write(f"{d['BIN1'][i]} {d['BIN2'][i]} {d['ANGBIN'][i]} {d['VALUE'][i]:.10e}\n")
np.savetxt(out / "cosebis_cov.txt", cov[np.ix_(keep, keep)], fmt="%.10e",
           header=src.strip("#\n ") + "\ncovariance of cosebis_En.txt (75 x 75)")
nz = F["NZ_SOURCE"].data
np.savetxt(out / "nz_source.txt", np.column_stack([nz["Z_LOW"], nz["Z_MID"], nz["Z_HIGH"]] +
                                                   [nz[f"BIN{i}"] for i in range(1, 6)]), fmt="%.10e",
           header=src.strip("#\n ") + "\nz_low z_mid z_high n_1 n_2 n_3 n_4 n_5 (histograms)")

# delta_z prior from the chain (columns of output_multinest_C.txt: uncorr_bias_1..5 = 7..11, bias_1..5 = 19..23)
chain = np.loadtxt(release / "chains_and_config_files/main_chains_iterative_covariance/cosebis/chain/output_multinest_C.txt")
u, b = chain[:, 7:12], chain[:, 19:24]
L = np.linalg.lstsq(u, b, rcond=None)[0].T
assert np.abs(u @ L.T - b).max() < 1e-12
L = np.tril(L)
mu_u = np.array([0.000, -0.181, -1.110, -1.395, 1.265])  # priors.ini: uncorr_bias_i = gaussian mu_u 1
C = L @ L.T
np.savetxt(out / "dz_prior.txt", np.vstack([L @ mu_u, C]), fmt="%.10e",
           header="KiDS-1000 SOM photo-z shifts delta_z_i (n_i(z) -> n_i(z - delta_z_i)), recovered from the release's "
                  "COSEBIs chain\nrow 1: mean; rows 2-6: covariance")
print("written to", out)
print("delta_z mean", L @ mu_u)
print("sigma", np.sqrt(np.diag(C)))
