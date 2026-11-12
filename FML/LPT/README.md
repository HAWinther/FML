# Arbitrary-order growing-mode EdS fields

`ArbitraryOrder.h` computes a finite user-selected order with longitudinal and
transverse contributions, conservative full-bandwidth products, collected/streamed
outputs, an owned-memory preflight and cache/recompute workspace choices. Fields
are Fourier-space vectors in unit-box coordinates, with no factorial in
`Psi(g)=sum g^n Psi_n`; the evaluator also returns `dPsi/dln(a)=sum n g^n Psi_n`.
General cosmology and automatic SimulationCode/COLA wiring are separate work.

```cpp
#include <FML/LPT/ArbitraryOrder.h>
using namespace FML::COSMOLOGY::LPT;
EdSLPTOptions options;
options.max_order = 6;
options.input_mode_limit = {1,1,1}; // actual strictly sub-Nyquist seed support
options.max_owned_bytes_per_rank = 512ULL * 1024 * 1024;
auto plan = plan_eds_lpt(delta_ref, options, EdSLPTOutput::RetainOrders);
auto fields = compute_eds_lpt(delta_ref, options);
auto sums = fields.evaluate(.5, options.max_owned_bytes_per_rank);
```

The small band-limited reference range reaches order 6 in float/double; this is
not a universal accuracy/convergence guarantee. Work widths exceed `2*P*K`, so
full-support high-order requests can be extremely expensive and should be planned
first. Higher orders are accepted subject to resource/representation limits and
marked beyond the validated range. Empty MPI slabs remain unsupported. See the
[API contract](ARBITRARY_ORDER_API.md), [derivation](ARBITRARY_ORDER_DERIVATION.md),
[implementation plan](ARBITRARY_ORDER_PLAN.md) and [scientific results](SCIENCE_RESULTS.md).

The runnable example uses three non-coplanar seed waves or one plane wave:

```sh
make -C FML/LPT/example USE_MPI=false
# prefix order growth mesh [coupled|plane]
FML/LPT/example/eds_lpt /tmp/eds 6 .5 16 coupled
FML/LPT/example/eds_lpt /tmp/plane 6 6 16 plane
make -C FML/LPT/tests run USE_MPI=false
# Clean before changing precision/compiler options:
make -C FML/LPT/tests clean
make -C FML/LPT/tests run USE_MPI=false LPT_PRECISION=float FFTW_LINK=-lfftw3f
```

`prefix.csv` reports per-order norms, growth-weighted norms, truncated-sum norms,
minimum sampled `det(I+grad Psi)` and nonpositive sample counts. The example limits
mesh widths to 4..64 and requires `2*order<mesh`, so its whole input-generated
output support fits on the diagnostic mesh. Its extra derivative copies and
partial sums are outside the solver's owned-memory report; provide process-memory
headroom. The reference-epoch density, unit-box normalization and EdS g scaling
are explicit in the source. Physical peculiar velocity additionally needs `a H`
and a consistent length conversion.

A negative sampled Jacobian identifies an orientation reversal. Positive samples
alone neither exclude crossings between cells nor establish perturbative convergence.
Decreasing Taylor terms can coexist with a crossed map. The plane-wave case has
exact pre-crossing ZA dynamics; continuing its formal fields after crossing is a
diagnostic demonstration, not a valid single-stream evolution.

`test_science.py` requires NumPy and is part of the normal module `run` target.
It supports `--output /path` to retain commands, sparse inputs, every exported
coefficient, summaries and hashes; otherwise temporary data are removed. The MPI
module runner validates 1/2/4 ranks and both workspace policies. Single precision
is tested in serial/OpenMP; float MPI requires matching FFTW float-MPI libraries
and has not been validated here. Clean between configurations and use the
[shared build guide](../BUILDING.md) for compiler/library paths.

## Low-order LPT de-aliasing

`DisplacementFields.h` provides an optional final `Dealiasing` argument for
`compute_2LPT_potential_fourier`, `compute_3LPT_potential_fourier`, and
`compute_3LPT_displacement_field`. Existing calls default to `Dealiasing::None`
and retain their original numerical calculation.

```cpp
using namespace FML::COSMOLOGY::LPT;
compute_2LPT_potential_fourier(delta, phi2, Dealiasing::Padding);
compute_3LPT_potential_fourier<3>(delta, phi1, phi2, phi3a, phi3b, A,
                                false, Dealiasing::Padding);
compute_3LPT_displacement_field<3>(delta, psi, velocity, dlogDdt,
                                  false, Dealiasing::Padding);
```

Real-space nonlinear products otherwise fold high-frequency convolution modes
back into resolved modes. `Padding` embeds the normalized Fourier coefficients
in a larger grid, constructs the potentials there, and crops to the original
mesh. The input is not modified. All returned grids use the original mesh and
boundary allocation; boundary values still require normal halo communication.

Standalone 2LPT uses at least 3/2 times the original width in every dimension.
3LPT uses at least twice the width to accommodate cubic terms. The intermediate
2LPT potential retains its full bandwidth on the work mesh when constructing
3LPT-b and the transverse potential: prematurely cropping it loses physical
mode couplings even if each individual product is de-aliased. Work widths are
rounded up to an even number divisible by the MPI task count.

Only wavevectors satisfying `2*abs(n_i) < Nmesh` in **every** dimension are kept
when embedding or cropping. Thus even-mesh input and output Nyquist planes
are zeroed explicitly, including in the returned 1LPT potential. Odd meshes
retain every represented signed mode. This avoids choosing a sign for a
self-conjugate Nyquist mode when taking mixed derivatives. There is no radial
cutoff, no smoothing of strictly sub-Nyquist input modes, and no volume factor
on the copied coefficients (`FFTWGrid` uses normalized forward transforms).

Without padding (`Dealiasing::None`) the even-mesh Nyquist modes are kept, but
their ambiguous derivatives are zero: a first derivative `i k_i` (displacements,
curl) or a mixed derivative `k_i k_j` (`i != j`, the Hessians in the 2LPT and
3LPT sources) is set to zero when `k_i` (or `k_j`) is at the Nyquist frequency,
while `k_i^2` is kept. The results are then covariant under reflections
`x_i -> -x_i`, as in `TidalTensor/Hessian.h`.

This is an accuracy option with substantial cost. In 3D, each work grid is
approximately 3.375 times larger for 2LPT and 8 times larger for 3LPT; the
original input and cropped outputs add to this. Runtime also increases with
the larger FFTs. MPI spectral remapping exchanges retained coefficients using
`FML::communicator`, with transient local packing buffers and chunked messages;
it never replicates the global grid on every rank. The existing streamed
Hessian memory optimization is also used on the padded mesh.

This option currently covers the library LPT entry points above. SimulationCode,
ALPT, primordial non-Gaussian field generation, and scale-dependent higher-order
kernels do not automatically enable it.

For the physical motivation and quadratic padding rule, see
[Michaux et al. (2020), section 2.3](https://arxiv.org/abs/2008.09588).
Tests use independent sparse integer-wavevector convolutions without wrapping,
including high-frequency cubic couplings and the transverse term.

## Shared growth interface

`GrowthHistory.h` exposes D1, signed D2 and their growth rates independently of SimulationCode.
The growth factors themselves (1LPT to 3LPT, scale dependent or not) are solved by
`FML/Cosmology/Growth/LPTGrowthFactors.h` (whose `history()` gives a `GrowthHistory`);
`compute_LPT_growth_factors_LCDM` / `_GeffGLCDM` here are wrappers of it.
Use `growth_history_from_log_a_splines(D1, D2)` to adapt this module's growth-solver output
to the shared [NBody timestep policies](../NBody/TimeStepping.md). The adapter owns copies
of the splines and preserves their normalization and time domain.
