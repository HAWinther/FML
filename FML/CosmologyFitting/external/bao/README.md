# BAO data

Gaussian BAO likelihoods (mean vector and covariance) of DESI, as distributed by the Cobaya sampler
(https://github.com/CobayaSampler/bao_data):

| File | Data | Reference |
|---|---|---|
| `desi_dr2_mean.txt`, `desi_dr2_cov.txt` | DESI DR2: BGS, LRG1-3, ELG, QSO, Lyman-alpha (13 values) | DESI Collaboration, arXiv:2503.14738 |
| `desi_dr1_mean.txt`, `desi_dr1_cov.txt` | DESI DR1 (12 values) | DESI Collaboration, arXiv:2404.03002 |

Format of the mean files: `z value quantity`, with quantity one of `DV_over_rs`, `DM_over_rs`, `DH_over_rs`
(distances divided by the sound horizon at the drag epoch); the covariance matrix is in the same order.
