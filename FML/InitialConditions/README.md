# Linear transfers and IC preparation

This module prepares linear inputs independently of COLASolver. The existing
particle/LPT assignment routines remain in their owning modules.

- `TransferProvider.h`: species, density/velocity gauges, units, capabilities and k/a coverage.
- `CAMBTransferProvider.h`: an owning adapter around `LinearTransferData`, retaining its exact
  transfer representation, derivative prescription and legacy power multiplication order.
- `LinearInput.h`: preparation of a two-column P(k), a single transfer, or provider-based input,
  with caller-supplied growth rescaling and primordial power.
- `SpeciesIC.h`: CDM/baryon transfer ratios, relative-mode velocities and mass contrasts.
- `InternalTransferProvider.h`: owning FML Einstein–Boltzmann solver adapter and explicit density-gauge views.

## Conventions

Queries use `k_hmpc` in h/Mpc and scale factor a. `density_transfer` returns
`delta_s/R`, dimensionless for unit primordial adiabatic curvature in the CAMB sign
convention. `density_over_k_mpc2` returns the compatibility representation in Mpc²;
it divides by **physical** k² in 1/Mpc², not k_hmpc². Primordial and matter power
spectra are dimensional, in (Mpc/h)³. This matches the existing CAMB reader and is
checked against independent CAMB density and velocity values.

`velocity_divergence_transfer` is theta/(a H R), with theta = div(u), u = a dx/dt.
The direct velocities supplied by both adapters are Newtonian-gauge quantities.
They are separate from `density_growth_rate = (1/delta) ddelta/dln(a)`; a density
rate is not silently substituted when direct velocity columns are absent.

CAMB density columns are CDM-synchronous, whereas its named Newtonian velocity
columns have different gauge/normalization conventions. See the [CAMB variable
reference](https://camb.readthedocs.io/en/latest/transfer_variables.html).
The table adapter assumes that input files follow these conventions, including
CLASS output with `format=camb`; it cannot infer the gauge of arbitrary user files.

The internal solver stores Newtonian density and v = -theta/(c k). For uncoupled
pressureless CDM/baryons, its CDM-synchronous view applies the **same CDM time shift**
to both species:

```
delta_s_sync = delta_s_Newtonian - 3 (a H)/(c k) v_cdm
```

It does not transform each species into its own rest frame. Native Newtonian
densities remain available through `with_density_gauge(Gauge::Newtonian)`, and
changing the density view leaves the explicitly labelled Newtonian velocities
unchanged. The gauge transformations follow [Ma & Bertschinger](https://arxiv.org/abs/astro-ph/9506072).
IC preparation rejects a provider whose density has not been converted to the
required CDM-synchronous convention.

This is the standard density/backscaling IC prescription used by the existing
CAMB-table path. It is **not** a new fully relativistic N-body-gauge evolution
scheme. COLASolver continues to assign velocities using its growth/LPT prescription,
including relative transfer-ratio derivatives for two species. Direct Newtonian
velocities are exposed for inspection/future consumers, not inserted into a
particle coordinate system with a different velocity convention.

## Ranges, ownership and normalization

The internal provider rejects extrapolation and owns the background, recombination
and perturbation solutions for its lifetime. Its first release supports flat GR
LCDM, unit adiabatic initial perturbations, CDM and baryons, and massless radiation.
Massive neutrinos, curved backgrounds, isocurvature mixtures and modified gravity
are rejected before solving. Independent gauge views share the solved state.

The CAMB provider owns a shared reference to its table data. Its default
`SourceExtrapolation` policy retains the existing reader's behavior, including
high-k power-law extrapolation and its existing time/derivative boundary handling.
Choose `RangePolicy::Reject` for strict coverage checks. Unavailable species and
velocity columns are reported explicitly. Adapters do not launch independent
random fields for different species: IC generation still uses one shared
adiabatic realization and species-specific ratios.

Primordial amplitude is distinct from unit-curvature transfer evolution. The
internal provider supports explicit primordial-power rescaling (used by
COLASolver's sigma8 normalization); this changes P, not unit-curvature transfers.
The legacy IC table retains its historical normalization behavior for compatibility.

## COLASolver

All existing `powerspectrum`, `transferfunction` and `transferinfofile` choices
remain supported. A simple P(k,z=0) input requires neither species transfers nor a
Boltzmann solve. Ordinary builds do not compile/link the internal solver.

To enable the optional internal source, rebuild with

```
make clean
make USE_INTERNAL_COSMOLOGY=true
```

then select

```lua
ic_type_of_input = "internal_transfer"
ic_input_filename = "solver.ini"
ic_input_redshift = 0.0  -- backscale from today; use zini for transfers evaluated at zini
```

`solver.ini` is an FML Einstein–Boltzmann parameter file. Its h, OmegaB, OmegaCDM,
TCMB, Neff, As, ns and pivot must match COLASolver, and `alpha_s` must be 0 (COLASolver's primordial
spectrum has no running). It must cover the IC mesh's
fundamental through its conservative corner wavenumber, the input/initial epochs,
and today. Cosmology and coverage mismatches fail before the expensive solve.
For example, matching COLASolver's standard LCDM parameters:

```ini
model = lcdm
h = .671
OmegaB = .049
OmegaCDM = .2685
OmegaK = 0
m_nu = 0
Neff = 3.046
TCMB = 2.7255
A_s = 2.215e-9
n_s = .966
k_pivot = .05
pert_k_min = .00001
pert_k_max = 2.0
pert_n_k = 200
```

`pert_k_min/max` are in 1/Mpc. Choose their extent and resolution for the actual
box, mesh and desired precision; these numbers are an example, not an accuracy
promise. Native nonlinear evolution, COLA and the existing LPT orders are unchanged.
The internal scalar solve runs on each MPI rank; distributed/shared transfer
initialization is not implemented here.

The existing sigma8 routine integrates over [1e-5,100] h/Mpc. With strict internal
transfers, sigma8 normalization requires that full range; otherwise it fails
before solving and the sigma8 diagnostic is skipped. Use As normalization for
ordinary finite-range internal runs, or supply sufficiently wide transfers.
Existing P(k)/table normalization behavior is unchanged.

Tests: `FML/tests/test_transfers.cpp` covers the lightweight/table paths;
`Cosmology/EinsteinBoltzmannSolver/tests/test_transfer_provider.cpp` checks the
converted densities/velocities against stored CAMB 1.5.4 values and tests the
internal adapter. `COLASolver/tests/check_internal_transfers.py` exercises the
optional executable, two-species ICs, backscaling, normalization coverage and simple P(k).
