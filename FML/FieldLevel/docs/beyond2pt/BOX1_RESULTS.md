# Real-catalog Box-1 preparation

Completed 19 November 2026. Sampler experiments are frozen while the project moves to a concrete fixed-IC application. The first public challenge catalog has now been downloaded, verified and processed with the existing FML observation adapter. This is observed-data preparation; no amplitude likelihood or posterior has been computed.

The catalog has 3,591,859 unweighted galaxies in a 2000 Mpc/h periodic box at redshift 1. The HDF5 file has a one-dimensional `galaxies` compound dataset with float32 x/y/z fields. All coordinates are finite and inside the primary periodic box. Measured number density: 0.000448982375 (h/Mpc)^3. Poisson power: 2227.2589208 (Mpc/h)^3. The raw file is pinned by commit, Git blob identity, size and SHA256 in [the source receipt](box1-source.json).

| Cubic kmax [h/Mpc] | Stored half-complex modes | Real mode degrees |
|---|---:|---:|
| 0.08 | 67625 | 132650 |
| 0.10 | 127007 | 250046 |
| 0.12 | 231230 | 456532 |

The 96 and 128 observation meshes extract the same modes. Eight independent direct Fourier sums use all 3,591,859 galaxies, including low-frequency, cutoff-edge and corner modes. Maximum direct-sum discrepancy: 1.049e-08 Poisson mode standard deviations. Maximum difference between observation meshes: 3.274e-08 in the same units. The squared noise-normalized distance over the entire largest cubic data vector is 2.562e-11. These measure observation-transform agreement only; they do not certify forward-model error or parameter bias. The native serial extractions took about 19 and 21 seconds on this machine; this is not a dedicated benchmark.

## What blocks the actual fixed-IC result

The matching Box-1 initial field and a verified reference spectrum/background assignment have not been located. The paper documents a separate organizer-to-LEFTfield release of the true phases; the inspected public repository contains no such file. The data audit already recorded this dependency. Random or reconstructed phases would define a different analysis. Further sampler campaigns cannot resolve it.

The [application](../../example/beyond2pt/README.md) emits the required inputs explicitly and records `fixed_ic_fit_ready=false`. Once matching inputs are available, connect their documented normalization to the existing Eulerian fixed-IC likelihood, compute amplitude likelihood/posterior curves and compare model resolutions and EFT cutoffs. The saved 0.08/0.10/0.12 masks are observed-data preparation, not completed posterior cutoff comparisons.

## Saved artifacts

[Complete summary](box1-summary.json) retains the direct sums, coordinate bounds, metadata and mode counts. The reusable entry point is [prepare.py](../../example/beyond2pt/prepare.py). Raw catalog, native source hashes/binary, both mode vectors, direct-sum report, descriptive power CSV and `catalog-overview.png` stay outside the library at:

`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/beyond2pt-catalog-application`.

No additional sampler chains were run. Existing failed studies remain unchanged. No real-catalog posterior or published-result reproduction is claimed.
