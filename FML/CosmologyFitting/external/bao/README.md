# BAO data

Gaussian BAO (and BAO + RSD) likelihoods (mean vector and covariance) of DESI and SDSS, as distributed by the Cobaya
sampler
(https://github.com/CobayaSampler/bao_data):

| File | Data | Reference |
|---|---|---|
| `desi_dr2_mean.txt`, `desi_dr2_cov.txt` | DESI DR2: BGS, LRG1-3, ELG, QSO, Lyman-alpha (13 values) | DESI Collaboration, arXiv:2503.14738 |
| `desi_dr1_mean.txt`, `desi_dr1_cov.txt` | DESI DR1 (12 values) | DESI Collaboration, arXiv:2404.03002 |
| `sdss_DR12Consensus_final.dat`, `final_consensus_covtot_dM_Hz_fsig.txt` | BOSS DR12 consensus BAO + full shape: D_M, H, fσ_8 at z = 0.38, 0.51, 0.61 (`boss_dr12`) | Alam et al. 2017, arXiv:1607.03155 |
| `sdss_DR12Consensus_bao.dat`, `BAO_consensus_covtot_dM_Hz.txt` | BOSS DR12 consensus BAO: D_M, H (`boss_dr12_bao`) | Alam et al. 2017 |
| `sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat`, `..._covtot.txt` | SDSS DR16 LRG BAO + RSD (BOSS DR12 z = 0.38, 0.51 and eBOSS z = 0.698; `eboss_dr16_lrg`) | eBOSS Collaboration, arXiv:2007.08991 |
| `sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8.dat`, `..._covtot.txt` | SDSS DR16 quasars at z = 1.48 (`eboss_dr16_qso`) | eBOSS Collaboration, arXiv:2007.08991 |
| `desi_dr1_shapefit_mean.txt`, `desi_dr1_shapefit_cov.txt` | DESI DR1 full shape, ShapeFit + BAO: q_iso, q_ap, df, dm of BGS, LRG1-3, ELG2, QSO and q_par, q_per of Lyman-alpha (26 values, block-diagonal covariance; `desi_dr1_shapefit`) | DESI Collaboration, arXiv:2411.12021 |

The ShapeFit files are the means and covariances of DESI's likelihood files
`likelihood_shapefit_spectrum-poles-rotated+bao-recon_syst-rotation-hod-photo_<tracer>_GCcomb_z<range>_thetacut0.05.h5`
and `likelihood_bao_syst_Lya_GCcomb_z1.8-4.2.h5` (data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0/
data/likelihood/, read by github.com/cosmodesi/desi-kp-cosmological-likelihoods), one line `z value quantity sample`
per value.

Format of the mean files: `z value quantity`, with quantity one of `DV_over_rs`, `DM_over_rs`, `DH_over_rs`
(distances divided by the sound horizon at the drag epoch), `bao_Hz_rs` (H(z) r_d/r_d,fid in km/s/Mpc) or `f_sigma8`;
the covariance matrix is in the same order. For BOSS DR12 the distances are multiplied by r_d,fid = 147.78 Mpc (in
Mpc) and H by r_d/r_d,fid (as in Cobaya, `rs_fid`).
