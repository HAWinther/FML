# LPT de-aliasing

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

This is an accuracy option with substantial cost. In 3D, each work grid is
approximately 3.375 times larger for 2LPT and 8 times larger for 3LPT; the
original input and cropped outputs add to this. Runtime also increases with
the larger FFTs. MPI spectral remapping exchanges retained coefficients using
`FML::communicator`, with transient local packing buffers and chunked messages;
it never replicates the global grid on every rank. The existing streamed
Hessian memory optimization is also used on the padded mesh.

This option currently covers the library LPT entry points above. COLASolver,
ALPT, primordial non-Gaussian field generation, and scale-dependent higher-order
kernels do not automatically enable it.

For the physical motivation and quadratic padding rule, see
[Michaux et al. (2020), section 2.3](https://arxiv.org/abs/2008.09588).
Tests use independent sparse integer-wavevector convolutions without wrapping,
including high-frequency cubic couplings and the transverse term.

## Shared growth interface

`GrowthHistory.h` exposes D1, signed D2 and their growth rates independently of COLASolver.
Use `growth_history_from_log_a_splines(D1, D2)` to adapt this module's growth-solver output
to the shared [NBody timestep policies](../NBody/TimeStepping.md). The adapter owns copies
of the splines and preserves their normalization and time domain.
