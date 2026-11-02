# Linear transfers and IC preparation

This module prepares linear inputs independently of SimulationCode. The existing
particle/LPT assignment routines remain in their owning modules.

- `TransferProvider.h`: species, density/velocity gauges, units, capabilities and k/a coverage.
- `LinearInput.h`: preparation of a two-column P(k), a single transfer, or provider-based input,
  with caller-supplied growth rescaling and primordial power.
- `SpeciesIC.h`: CDM/baryon transfer ratios, relative-mode velocities and mass contrasts.
- `InternalTransferProvider.h`: owning FML Einstein–Boltzmann solver adapter and explicit density-gauge views.

The providers for CAMB and CLASS files, and `make_transfer_provider` (CAMB, CLASS or the internal solver, with the
gauge of the input declared), are in [TransferProviders](../TransferProviders/README.md).

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

The internal solver stores Newtonian density and v = -theta/(c k). Its CDM-synchronous
view applies the **same CDM time shift** to every species (CDM, baryons and massive
neutrinos, with w_s = p_s/rho_s; 0 for CDM and baryons):

```
delta_s_sync = delta_s_Newtonian - 3 (1 + w_s) (a H)/(c k) v_cdm
```

It does not transform each species into its own rest frame. Native Newtonian
densities remain available through `with_density_gauge(Gauge::Newtonian)`, and
changing the density view leaves the explicitly labelled Newtonian velocities
unchanged. The gauge transformations follow [Ma & Bertschinger](https://arxiv.org/abs/astro-ph/9506072).
IC preparation rejects a provider whose density has not been converted to the
required CDM-synchronous convention.

This is the standard density/backscaling IC prescription used by the existing
CAMB-table path. It is **not** a new fully relativistic N-body-gauge evolution
scheme. SimulationCode continues to assign velocities using its growth/LPT prescription,
including relative transfer-ratio derivatives for two species. Direct Newtonian
velocities are exposed for inspection/future consumers, not inserted into a
particle coordinate system with a different velocity convention.

## Ranges, ownership and normalization

The internal provider rejects extrapolation and owns the background, recombination
and perturbation solutions for its lifetime. It supports any model of the solver
(`settings.cosmo.model`, curvature allowed) with unit adiabatic initial perturbations
and its massive neutrinos (`deg_ncdm` degenerate species of mass `m_nu`, or
`m_nu_list`): `MassiveNeutrino` (all of them) is then available and `TotalMatter`
includes it. Combinations are weighted by the background densities rho_i(a) of the
model. Isocurvature modes and running (`alpha_s`) are rejected before solving.
Independent gauge views share the solved state.

The table providers of TransferProviders own their tables. Their `SourceExtrapolation`
policy extrapolates above k_max as a power law and uses the end values elsewhere.
Choose `RangePolicy::Reject` for strict coverage checks. Unavailable species and
velocity columns are reported explicitly. Adapters do not launch independent
random fields for different species: IC generation still uses one shared
adiabatic realization and species-specific ratios.

Primordial amplitude is distinct from unit-curvature transfer evolution. The
internal provider supports explicit primordial-power rescaling (used by
SimulationCode's sigma8 normalization); this changes P, not unit-curvature transfers.
The legacy IC table retains its historical normalization behavior for compatibility.

## SimulationCode

SimulationCode reads its transfer functions with TransferProviders (`camb`, `class` or
`internal`; a simple P(k,z=0) input needs neither species transfers nor a Boltzmann solve).
With `ic_type_of_input = "internal"` it runs FML's Einstein-Boltzmann solver when the
simulation starts, with the cosmology of the simulation (background, massive neutrinos,
`cosmology_model` and the primordial spectrum), so no input file is needed:

```lua
ic_type_of_input = "internal"
ic_input_filename = ""   -- optional: a solver .ini with precision/recombination settings
ic_input_redshift = ic_initial_redshift
```

The solver runs on task 0 and its tables are sent to the other tasks; they are
extrapolated outside their k-range as the CAMB and CLASS tables, so
`ic_sigma8_normalization` works as with the other inputs. See the SimulationCode
README for the k-grid, the cost and the accuracy tests.

Tests: `FML/tests/test_transfers.cpp` covers the lightweight/table paths (`LinearInput`, `SpeciesIC` and the
conventions of a table provider); `FML/tests/test_camb.cpp` the CAMB reader against CAMB;
`Cosmology/EinsteinBoltzmannSolver/tests/test_transfer_provider.cpp` checks the
converted densities/velocities against stored CAMB 1.5.4 values and tests the
internal adapter. `SimulationCode/tests/check_internal_transfers.py` runs `nbody` with
`ic_type_of_input = internal`: two-species ICs, backscaling, sigma8 normalization, a solver
.ini, neutrino particles, w0waCDM with curvature and simple P(k).
