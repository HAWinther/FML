# Fourier random-field backends

The existing four-argument Gaussian API is unchanged. It retains FML's legacy
line-seeded realization and RNG state updates, including the known repeated
31-bit line seeds inherited from the N-GenIC/2LPTic scheme. The explicit
`RandomFieldOptions::backend = RandomFieldBackend::NGenIC` option calls that
same implementation. Its seed comes from the supplied stateful RNG; the options
seed/stream fields apply only to Philox. In 3D, the existing GSL versus MT19937 choice is preserved.

## Philox mode counters (2D and 3D)

```cpp
using namespace FML::RANDOM::GAUSSIAN;
RandomFieldOptions options;
options.backend = RandomFieldBackend::Philox;
options.seed = 1234; // uint64_t
options.stream = 0;  // uint32_t: an independent realization/channel

generate_gaussian_random_field_fourier<3>(grid, nullptr, Pofk, false, options);
```

The implementation uses Philox4x32-10, with integer known-answer tests from the
[authors' Random123 repository](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors).
It is a self-contained implementation and adds no external library dependency.

The key is the 64-bit seed. The 128-bit counter consists of three injectively
encoded signed wavevector coordinates and the stream. The first nonzero signed
coordinate is made positive to choose one representative of {k,-k}; its partner
uses the conjugate. Each distinct representative has its own counter. This
removes the line-seed collision mechanism without constructing a seed table.
DC and even-mesh Nyquist planes are zero. Odd meshes retain all represented
nonzero modes. Midpoint uniforms lie strictly within (0,1); Gaussian amplitudes
use sqrt(-log(U)) and fixed amplitudes retain the same phases.

All MPI ranks must receive the same seed, stream and power function.
Power is evaluated directly at each mode to avoid mesh-dependent interpolation
when matching common modes across resolutions. With OpenMP it must be callable
concurrently. Nonfinite/negative spectra are rejected. The supplied stateful RNG
pointer is ignored (and may be null) for Philox. Repeated calls with the same
seed/stream reproduce the same field; change one to obtain another realization.
The optional real-space and non-Gaussian APIs accept the same final options
argument, including the cosmology non-Gaussian helper.

Integer Philox outputs are portable and checked against three published vectors.
Fourier coefficients are bit-identical across MPI decompositions and thread counts
on the tested compiler/math library. Transcendental functions may differ in their
last bits across platforms or compiler math modes; no cross-platform bitwise
promise is made for log, sin, cos or user power functions.

## COLASolver

```lua
ic_random_field_backend = "ngenic" -- default; old configurations retain their IC
ic_random_generator = "GSL"        -- or MT19937, as before
ic_random_seed = 1234

-- Alternative, distinct realization:
-- ic_random_field_backend = "philox"
```

`ic_random_generator` selects the stateful generator for the legacy backend and
is ignored by Philox. The new backend selector is always optional, including
with `all_parameters_must_be_in_file = true`. The existing integer seed setting
is interpreted as the same unsigned 32-bit seed value; the library Philox API
also supports full 64-bit seeds and separate streams. Gaussian and non-Gaussian
COLASolver IC paths honor the choice. Fixed amplitudes and phase reversal remain
available. Selecting Philox changes phases relative to N-GenIC; it does not claim
to reproduce another backend's realization.

## Active non-Gaussian kernels

The nonlocal generator only allocates inverse-power fields needed by the scaled
kernel coefficients, and only Fourier-transforms products that contribute to the
output. The dependencies are preserved: term 2 still needs the inverse-third-power
field when term 1 is zero; term 3 needs both inverse-third and inverse-two-thirds
fields even when their own output terms are inactive.

For the usual equilateral and orthogonal shapes with `u = 0`, this removes the
inverse-first-power grid and its two transforms: five simultaneous grids and six
transforms instead of six grids and eight transforms. These counts include the
output and local-source work, and exclude Gaussian generation (which directly
samples Fourier modes).

For generic kernels, excluding the local term from the table below:

| Active nonlocal outputs | Filtered input grids | Transforms, including local-source work |
|---|---:|---:|
| term 1 only | 1 | 4 |
| term 2 only | 2 | 5 |
| term 3 only | 3 | 6 |
| terms 1 and 2 | 2 | 6 |
| all three | 3 | 8 |

The local-source construction, mean/DC handling, aliasing convention and RNG
sampling are unchanged. Local-only and zero-fNL paths retain their existing
behavior. Finished source buffers are moved into the output instead of copied,
preserving the output grid's halo layout and Fourier status. Kernel parameters
must agree on every MPI rank because the transforms are collective.
