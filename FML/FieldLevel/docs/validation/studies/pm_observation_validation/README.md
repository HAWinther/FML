> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Fixed PM observation response (#52)

`PeriodicPMModelT` now accepts a final optional `PMObservationSettings`, after its
existing domain argument. The default leaves existing outputs/targets unchanged.

```cpp
FF::PMObservationSettings observed;
observed.reference_cic_mesh = 32;
FF::PeriodicPMModel model(config, spectrum, dynamics, noise, {8}, domain, observed);
```

The built-in response is `W_reference(k)/W_force(k)` with the CIC window
`W_M=product_i sinc(pi*n_i/M)^2`. It smooths final predictions to a fixed coarser
assignment window. The reference must be even, >=4, <= the force mesh, and support
the strict observed band below its Nyquist. It is a **window convention**, not
spatial regridding or a different mode/layout allocation. It does not remove
aliasing or produce exact particle Fourier modes.

`observed.response.multiplier` also accepts fixed real, finite values in local
`model.modes()` order, reusing `FourierResponse`. They may be negative or zero,
must preserve cross-rank Hermitian symmetry, and compose multiplicatively with
the CIC reference. If any rank supplies custom values, every rank must supply
its owned-mode count; ranks with zero modes supply an empty vector. Input values
are copied into immutable cached coefficients. No parameter-dependent response
or derivative of response coefficients is supplied.

Prediction applies the fixed response to the signal; the pullback applies its
real transpose. **Gaussian noise is added after the response**, with the existing
P_noise/V normalization and noise inference. No determinant or data/noise rescaling
is silently introduced. Data must already be expressed in the declared observation
space. Reconstruction fields remain the raw dynamical/CIC fields.

Explicit responses use PM target tag 3, including force convention, reference mesh
and cached per-mode coefficients. Default raw-observation tag-1 and tag-2 byte
encodings remain unchanged. Same-layout restart is still required. The generic
Cartesian survey factory remains disabled; an observation window is not survey
pixelization, selection or a replacement for N/M layout handling.

## Bounded experiment

Use the same strong91 initial field, 122 active directions, 256 observed real
degrees, P=.002, box 1, Omega_m=.3, a=.2→1 and frozen noise seed 92/power 1e-4.
The original M32 mock is tagged `observation_cic,32` without changing its numerical
data. Both M32 and M64 now predict in that actual observation space. Untagged
legacy data reject in the new reference-window CLI mode; they remain valid for
legacy raw mode. Hashes document the source and tagged data.

For each force convention, run forward N128/M32 and N128/M64 with 32 steps;
full posterior gradients use N64 with the same M32/M64 pair. The compensated
case also checks N96→128 at M64 and 32→64 steps at N128/M64. A field and mixed
field/scalar finite-difference sweep uses compensated N64/M64/four steps and the
nontrivial M32 observation response.

Retain field <5%, projected gradient <15%, and discrepancy <.1 noise SD per real
observed degree. Check matched physical initial coefficients within 2e-12, finite
supported outputs, exact data/response metadata and conservation. FD uses eight
h values: best last-four scaled error <2e-5 and at least two <1e-4. Keep failures.
This is one weakly nonlinear realization and finite references, not posterior
coverage, a continuum solution or a convergence proof in arbitrary regimes.

## Reproduce

```sh
make -C FML/FieldLevel/pm_nonlinear_validation pm_nonlinear USE_MPI=true USE_PENCIL_GRID=true
python3 FML/FieldLevel/pm_observation_validation/run.py --bin-dir /path/to/binaries --output /path/to/fresh/OUT --np 4 --mpirun "mpirun --oversubscribe"
```

Use suitable compiler/library/launcher overrides and clean/separate builds when
changing flags. Phases `cases` and `analyze` are separate. The runner records PIDs
and exact arguments, bounds each subprocess to 300 seconds, and cleans up on
timeout/keyboard interruption. NumPy/Matplotlib are required. This explicit study
is outside routine CI; reports are saved before analysis returns nonzero for failures.
The native driver's final optional `CIC_REFERENCE` argument selects the window,
and `.observation.csv` records it. See [RESULTS.md](RESULTS.md) for actual evidence.
