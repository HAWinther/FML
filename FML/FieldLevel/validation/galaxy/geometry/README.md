# Deterministic posterior geometry audit

This is an exploratory diagnostic of the separated chain in the
[interweaving pilot](../interweave/README.md). It uses existing primary draw 512
from chain index 1 and the same draw index from chains 0, 2 and 3. The observed
chain separation motivated the choices: this is not a blind validation campaign.
No Monte Carlo draws, priors, likelihoods, chain weights or retained budgets change.
The [declared diagnostic scope](plan.json) includes all paths and controls.

Each of the three endpoint pairs has 101 points along three paths:

- Linear interpolation in the 64 whitened coordinates.
- Linear interpolation in alpha*white, divided by the interpolated alpha.
- Spherical interpolation of white-field directions, with linearly interpolated
  endpoint norms. This avoids a purely geometric drop in field norm from a chord.

Amplitude and physical biases interpolate linearly along each path. A separate
101-point rescaling control keeps alpha*white fixed while alpha varies from 0.6
through 1.4. All endpoints are real primary records, except the explicitly
constructed rescaling control. The generated point file and original endpoint
records are fingerprinted. Endpoints are not selected by the best posterior value.

At each fixed field/amplitude the Fourier likelihood is exactly quadratic in the
four bias coefficients, with the same bounded uniform priors as the sampler.
The profiler enumerates all 3^4 lower/free/upper active sets, solves the free
linear equations in prior-scaled coordinates, and selects the lowest feasible
objective. No proposal ridge or Gaussian prior approximation is added. A scaled
Karush-Kuhn-Tucker residual certifies the global convex minimum, including flat
faces. Independently known clipped, coupled interior and constant likelihoods
check the profiler. Every cosmology profile value is compared with the full
existing posterior; failure aborts the audit.

Outputs retain the original interpolated point potential, profiled-bias potential,
white prior, data quadratic, optimized biases and numerical certificates. Shared
likelihood constants are omitted from displayed relative potentials. Scalar
priors are uniform. Profiling is not marginalization and supplies no determinant,
configuration weight or evidence estimate. Field/amplitude coordinate changes also
have volume effects: point energies are not coordinate-invariant marginal densities.
A low point-energy path need not be an easy typical-set transition, and a barrier
on these paths need not exist along every possible curved route.

Build and prepare points from the repository root, supplying normal compiler/FFTW
settings and absolute output paths:

```sh
make -C FML/FieldLevel/validation/galaxy/geometry geometry_probe USE_MPI=false
python3 FML/FieldLevel/validation/galaxy/geometry/prepare.py /absolute/interweave/campaign /absolute/new/audit
FML/FieldLevel/validation/galaxy/geometry/geometry_probe /absolute/new/audit/points.csv /absolute/new/audit/results.csv
python3 FML/FieldLevel/validation/galaxy/geometry/analyze.py /absolute/new/audit
```

The probe is serial only and evaluates the exact N4/N8/N16 synthetic fixture used
by the inference pilots. It does not establish production MPI profiling support,
coverage, physical convergence or a real-catalog analysis. Raw point files, native
builds, logs and plots stay outside the library. The durable report is stored under
docs/validation/studies/geometry_audit/.

A follow-up checks the actual endpoint adjoints against three directional field
and five scalar finite differences per state (32 comparisons), explicitly failing
nonfinite comparisons. Those fixed endpoints have central perturbations inside
support. Replaying profiles after adding the checks must preserve the CSV bytes.
A separate exploratory analysis reuses profile values to evaluate likelihood-only
beta scaling at 1, 1/2, 1/4, 1/8, 1/16 and 0. With uniform bias priors the positive-
beta minimizer is unchanged. No tempered chain has been sampled; see
[the proposed replica-exchange test](NEXT_STEPS.md).
