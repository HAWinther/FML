# Transfer providers: CAMB, CLASS and the internal solver, with gauge conversion

`make_transfer_provider` (`MakeTransferProvider.h`) makes a `TransferProvider` (see
[`InitialConditions/TransferProvider.h`](../InitialConditions/TransferProvider.h)) from one of three sources. The
user declares the gauge of the input, and the provider returns densities in the gauge that is asked for:

| `source` | Input | Input gauge | Conversion |
| --- | --- | --- | --- |
| `camb` | A CAMB transfer info file (`CAMBFileTransferProvider`) | Always CDM-synchronous (CAMB's density columns) | None needed |
| `class` | Native CLASS output (`format = class`, `output = mTk[, vTk]`, `write_background = yes`) | Declared: `gauge = synchronous` (the default in CLASS) or `newtonian` | Newtonian → CDM-synchronous, at the tabulated redshifts |
| `internal` | FML's Einstein–Boltzmann solver (`USE_INTERNAL_COSMOLOGY`, through `InternalTransferProvider`): flat ΛCDM, at most one massive neutrino | Computes both gauges itself | By the solver |

This folder replaces the old CAMB reader (`CAMBUtils/CAMBReader.h`, `LinearTransferData`) and
`InitialConditions/CAMBTransferProvider.h`, which have been removed. SimulationCode reads its transfer functions with it
(`SimulationCode/src/TransferInput.cpp`: `ic_type_of_input = transferinfofile`, `class` or `internal_transfer`).

```cpp
#include <FML/TransferProviders/MakeTransferProvider.h>
using namespace FML::INITIALCONDITIONS;

TransferSourceSettings s;
s.source = "class";
s.class_root = "output/run_00_";        // CLASS's root: run_00_z1_tk.dat, ..., run_00_background.dat
s.input_gauge = Gauge::Newtonian;        // what CLASS was run with
s.output_gauge = Gauge::SynchronousCDM;  // what the particle initial conditions need (the default)
s.A_s = 2.1e-9;
s.n_s = 0.965;                           // and k_pivot_mpc = 0.05 by default
auto provider = make_transfer_provider(s);
double T = provider->density_transfer(Species::CDMBaryon, 0.1, 1.0 / 50.0);
double f = provider->density_growth_rate(Species::CDMBaryon, 0.1, 1.0 / 50.0);
```

## Files

- `TabulatedTransferProvider.h`: what the CAMB and CLASS readers share. A reader fills `TransferTables` (every
  species it has, including CDMBaryon and TotalMatter, at each redshift of its files, in the FML convention) and the
  base class does the interpolation, growth rates, extrapolation and primordial spectrum.
- `CAMBFileTransferProvider.h`: reads a CAMB transfer info file (`<folder> <n>`, then `<file> <z>` per line; files not
  found in `<folder>` are looked for next to the info file) and the files' columns by name from the header line
  (`k/h CDM baryon photon nu mass_nu total no_nu total_de Weyl v_CDM v_b v_b-v_c`), or by position without one:
  13 columns (CAMB), 9 (AxionCAMB) or 7 (old CAMB, or CLASS with `format = camb`). CDMBaryon is weighted by
  `Omegab`/`OmegaCDM`, TotalMatter is CAMB's `total` column. The CDM and baryon velocities (`v_CDM`, `v_b`) are
  Newtonian; there is no neutrino velocity in CAMB's files.
- `CLASSTransferProvider.h`: reads CLASS transfer files by column name (the `#  1:k (h/Mpc)  2:d_g ...` header) and
  the redshift from `at redshift z=` in the header, plus the background file (`CLASSBackground`: H(a), ρ and p of
  CDM, baryons and each massive neutrino). `find_class_output_files(root)` lists a run's files.
- `GaugeView.h`: the gauge conversion as a wrapper around any provider that has the Newtonian CDM velocity, for
  sources that cannot convert at their own tabulated redshifts. Also `GaugeBackground`, the H(a) and (ρ + p)/ρ it needs.
- `MakeTransferProvider.h`: `TransferSourceSettings` and `make_transfer_provider`.

## source = internal

The solver runs when the provider is made (a few minutes with the defaults: k from 10⁻⁵ to 10 h/Mpc, 400 modes) and
the provider owns the solution. The cosmology comes from the same fields as for the other sources (`h`, `Omegab`,
`OmegaCDM`, `OmegaMNu`, `A_s`, `n_s`, `k_pivot_mpc`); everything else (precision, recombination, `Yp`, `TCMB`, ...)
from `internal_settings` (`internal_cosmology_from_settings = false` uses `internal_settings.cosmo` as it is).

- **Massive neutrinos**: `OmegaMNu > 0` is one massive species whose mass gives Ω_ncdm = `OmegaMNu` today, with
  N_eff = 3.046 in total and T_ncdm/T_CMB = 0.71611 (as CLASS with `N_ur = 2.0328`, `N_ncdm = 1`). Then
  `MassiveNeutrino` is available and `TotalMatter` includes it. Several species are not supported.
- **Both gauges from one solve**: `make_transfer_provider` returns the `output_gauge` view; the other is
  `std::dynamic_pointer_cast<InternalTransferProvider>(p)->with_density_gauge(...)`, which shares the solution.
- **The table runs past a = 1** (`internal_amax = 1.1`): with the table ending at a = 1 the spline end distorted the
  growth rate at z = 0 by 2.6×10⁻³. Growth rates are a centred difference (±10⁻³ in log a) of the solver's splines.
- **Neutrino velocities near the fluid switch**: the solver uses CLASS's neutrino fluid approximation (switch at
  kτ = 31). Around the switch θ_ν differs from CLASS (6×10⁻³ at k = 0.01, z = 49; 40% at k = 0.03, z = 49; 10⁻³ or
  better at z ≤ 2), δ_ν agrees to 10⁻³. For neutrino particles at high redshift, consider raising
  `internal_settings.pert.ncdm_fluid_trigger` (not compared with CLASS without the approximation here).
- `scripts/make_class_transfer_tables.py`: makes the CLASS input (below).
- `tests/`: the test and the script that makes its reference data (below).

## Making the input

Both scripts take the same cosmology arguments and the same redshift sampling (60 redshifts uniform in log(1+z) up
to z = 200 by default), so CAMB and CLASS can be run on the same input:

```sh
# CAMB (pip install camb): source = "camb", camb_infofile = <outdir>/transfer_infofile_<prefix>.txt
python3 SimulationCode/scripts/make_camb_transfer_tables.py --OmegaMNu 0.0024 --prefix lcdm_nu0.1 --outdir input
# CLASS (pip install classy): source = "class", class_root = <outdir>/class_data_<prefix>/<prefix>_
python3 TransferProviders/scripts/make_class_transfer_tables.py --OmegaMNu 0.0024 --prefix lcdm_nu0.1 --outdir input
python3 TransferProviders/scripts/make_class_transfer_tables.py --OmegaMNu 0.0024 --prefix lcdm_nu0.1_newt \
    --gauge newtonian --outdir input      # Newtonian gauge: input_gauge = Gauge::Newtonian
```

The CLASS script writes CLASS's native format with classy (the same columns and header lines as the CLASS
executable; checked through the provider to agree with the executable's files to 2×10⁻⁴), prints the
`make_transfer_provider` settings, and switches CLASS's neutrino fluid approximation off unless
`--neutrino-fluid-approximation` is given. The CAMB script uses CAMB `AccuracyBoost = 3` for the same reason.

## Conventions

The same for all sources (and as `InternalTransferProvider`): k in h/Mpc, `density_transfer` = δ/ℛ in the CAMB
sign convention, `velocity_divergence_transfer` = θ/(aH ℛ). CLASS normalizes to unit initial curvature with the
opposite sign, so the CLASS provider returns −`d_i` and −`t_i`/(aH), with θ in CLASS's 1/Mpc. CAMB's columns are
T = δ/k² in Mpc² and −θ/(aH k²): the CAMB provider returns T k² and −T_v k² (k in 1/Mpc).

- **Gauge declaration is checked** as far as the columns allow. A synchronous run never has `t_cdm` (CDM is at rest);
  a Newtonian run with `vTk` has it; with `extra_metric_transfer_functions = yes` a synchronous run has `h` and `eta`.
  A mismatch is an error.
- **Newtonian → CDM-synchronous** uses the time shift that brings CDM to rest (Ma & Bertschinger 1995):
  δ_s^sync = δ_s^N + 3(1 + w_s)(ℋ/ck)² θ_cdm^N/ℋ, with ℋ/(ck) = a E(a)/(2997.92458 k[h/Mpc]). It needs `vTk`.
  The velocities are passed through unchanged, so a converted provider has CDM-synchronous densities and Newtonian
  velocities (as `InternalTransferProvider`).
- **Synchronous → Newtonian is not supported**: it needs the metric combination (h′ + 6η′)/2k², not only velocities.
  No SimulationCode input needs it.
- **TotalMatter** is the ρ-weighted CDM + baryon + massive-neutrino density in the provider's gauge. CLASS's `d_m`
  column is not used: it is the gauge-invariant (comoving) matter density, not the density in the run's gauge.
- **Several massive neutrino species** are combined weighted by ρ_i (densities) and ρ_i + p_i (velocities).
- **Interpolation** is bicubic in (log a, log k) with 4 or more redshifts (bilinear with 2–3), for both readers.
- **Outside the tables**: `RangePolicy::Reject` makes it an error. `RangePolicy::SourceExtrapolation` (the default of
  `TransferSourceSettings`) uses a power law in k above k_max
  through the last 0.1 in log k (as the old `LinearTransferData` did), and the value at the end below k_min and outside the
  redshift range (growth rates: the value at the end of the redshift range).
- **Growth rates** come from the derivative of a 4-point Lagrange polynomial in log a through the tables around a,
  for both readers, so CAMB and CLASS give the same growth rates from the same redshift sampling. The tables must be finely sampled and extend past the redshifts where growth rates are used
  (e.g. the initial redshift). With 7 redshifts between z = 49 and 0, f_cb was off by up to 29% at the top end of
  the table and several per cent elsewhere, for CLASS and CAMB alike; with 40 redshifts up to z = 60 they agree with
  each other to 2×10⁻⁴. The scripts' default 60 redshifts up to z = 200 are fine.

## Accuracy (tests/test_transfer_providers.cpp)

Reference: CLASS 3.4.1 in both gauges and CAMB 2.0.5, one 0.1 eV massive neutrino, 40 redshifts up to z = 60 (and
CLASS without massive neutrinos for the internal solver).

| Check | Result |
| --- | --- |
| Newtonian run converted vs synchronous run, CDM / baryons / CB / total, 10⁻³ ≤ k ≤ 1 h/Mpc | 7×10⁻⁴ |
| The same at k = 10⁻⁴ h/Mpc (the conversion cancels terms 10⁵ times the result) | 3×10⁻³ (unconverted: 5×10²) |
| Growth rate f_cb of the converted densities, 10⁻³ ≤ k ≤ 0.1 / k = 10⁻⁴ | 2×10⁻⁴ / 10⁻² |
| CLASS vs CAMB, CDM / baryons / CB / total, 10⁻³ ≤ k ≤ 1 | 5×10⁻⁴ |
| CLASS vs CAMB, P_cb(z = 0) / growth rate f_cb / Newtonian CDM velocity | 8×10⁻⁴ / 2×10⁻⁴ / 1.5×10⁻⁴ |
| Internal vs CLASS (massless ν), CDM / baryons / CB / total, both gauges, 10⁻³ ≤ k ≤ 1 (10⁻⁴ ≤ k ≤ 10⁻³) | 7×10⁻⁴ (6×10⁻⁵) |
| Internal vs CLASS (0.1 eV ν, fluid approximation in both), the same | 9×10⁻⁴ (2×10⁻⁴) |
| Internal vs CLASS, δ_ν at k ≤ 0.01 / at z = 0, k = 0.1 and 1 / θ_ν at k ≤ 3×10⁻³ | 1.6×10⁻³ / 7×10⁻⁴ / 2×10⁻⁴ |
| Internal vs CLASS, growth rate f_cb / Newtonian CDM velocity | 8×10⁻⁴ / 6×10⁻⁴ |
| Internal: GaugeView on its Newtonian view vs its own CDM-synchronous view | 10⁻⁵ |
| CAMB reader vs the old `LinearTransferData` on the same files (checked once before it was removed), CDM / b / CB / total, growth rates, velocities, extrapolation to k = 30 | 2×10⁻⁵ (interpolation in log a instead of z) |
| The same, massive neutrinos: δ_ν / f_ν | 7.5×10⁻⁴ / 4×10⁻³ |

The internal f_cb agrees with CLASS to 3×10⁻⁵–2.5×10⁻⁴ inside the tables; the 8×10⁻⁴ is at z = 0, the last CLASS
table, where the CLASS provider's Lagrange derivative is one-sided.

**Massive neutrinos below the free-streaming scale** depend on the codes' approximations. The test data use CLASS
with its default neutrino fluid approximation and CAMB at default accuracy, which differ in δ_ν by tens of per cent
at k ≥ 0.1 h/Mpc (with the fluid approximation off in CLASS: 4–26%). With the scripts' defaults (CLASS without the
fluid approximation, CAMB `AccuracyBoost = 3`, 3 degenerate neutrinos with Σm_ν ≈ 0.1 eV) they agree to 1.4% at
z = 0 for k ≤ 1 h/Mpc; at z = 49 they differ by 5% at k = 0.03 and 26% at k = 1, where δ_ν is tiny. For neutrino
particles use the scripts' defaults.

## Tests

The reference data come from CLASS and CAMB, which are not part of FML, so they are not in the repository (about
13 MB):

```sh
cd FML/TransferProviders/tests
CLASS=/path/to/class ./make_test_data.sh     # needs the CLASS executable and pip install camb; writes data/
make run                                     # without data/ it prints SKIP and exits 0
```

With `USE_INTERNAL_COSMOLOGY=true` (the default) the test also runs the internal solver twice (with and without the
massive neutrino) and compares it with CLASS; this takes about 5 minutes on 2 cores. `USE_INTERNAL_COSMOLOGY=false`
skips it. `InternalTransferProvider` itself is also tested in
[`Cosmology/EinsteinBoltzmannSolver/tests/test_transfer_provider.cpp`](../Cosmology/EinsteinBoltzmannSolver/tests/test_transfer_provider.cpp).
