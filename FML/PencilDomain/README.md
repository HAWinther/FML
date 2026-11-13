# Pencil decomposition and grid

This is a standalone 1D/2D/3D MPI grid module. It does not change or
replace `FFTWGrid`, and it is not connected to `SimulationCode`.

The decomposition is a periodic 2D Cartesian process grid `P0 x P1`:

- Real cells are distributed over `x` (`P0`) and `y` (`P1`), with complete `z`.
- Hermitian Fourier modes are distributed over `ky` (`P0`) and nonnegative
  `kz` (`P1`), with complete `kx`.
- `PencilDomain` (in `PencilGrid.h`) exposes rank/coordinate mappings, cell and
  mode ownership, physical bounds, neighbors, and row/column communicators. The
  ownership of positions is that of `particle_domain()`, an `FML::Domain`
  (`Global/Domain.h`) with the edges of each rank's cells: a Slab in `x` when
  `P1 = 1` (every 1D and 2D grid), else a Pencil. Give it to `MPIParticles` so
  particles live on the rank that holds their cells.
- `PencilGrid` stores real and Fourier data in one FFTW-aligned allocation and
  transposes with `MPI_Alltoallw`. Ghost layers are periodic in `x` and `y`.

Set `ndim` to 1, 2, or 3 when constructing the decomposition. Supply only the
active mesh sizes; inactive axes are set to one. In 2D the process grid is
`P0 x 1` and the FFT performs the `y` and `x` transforms with a trivial `z`
axis. In 1D real `x` slabs are distributed, then gathered for a full-complex
FFT; the resulting spectrum is replicated on each rank. `owner_of_mode()`
returns rank 0 as the canonical owner for those replicated 1D Fourier modes.

## Basic use

Build the example from this directory with `make`; FFTW and MPI are required.
The example runs a normalized real-to-Fourier-to-real round trip:

```cpp
auto domain = std::make_shared<FML::GRID::PencilDomain>(
    std::array<int, 3>{128, 128, 128}, 500.0);
FML::GRID::PencilGrid grid(domain, /*nghost=*/2);
grid.fill_real_grid([](std::array<double, 3>& x) { return field(x); });
grid.fftw_r2c();
// Access local modes using grid.get_fourier_range() and grid[fourier_cell].
grid.fftw_c2r();
grid.communicate_boundaries();
```

For lower dimensions, use `PencilDomain({128, 1, 1}, 500.0, 1)` or
`PencilDomain({128, 128, 1}, 500.0, 2)`. Ghost layers are allocated
only along active axes.

`get_real_range()` and `get_fourier_range()` provide the same index-and-offset
iteration style as `FFTWGrid`. `get_real_range_with_ghosts()` includes ghost
cells. FFT transforms change `status()`; accessors check status and bounds.
Fourier wavevectors use physical units `2 pi n / boxsize`. The inverse transform
is normalized, so the round trip is the identity.

Choose the process-grid dimensions explicitly with the constructor's third
argument `{P0, P1}`, or pass `{0, 0}` for `MPI_Dims_create`. Each real and
Fourier pencil must contain at least one plane. Ghost width cannot exceed the
smallest real pencil width. A decomposition borrows the supplied communicator;
the caller must keep it alive until after the decomposition is destroyed.

## Implementation references

The transposed distributed FFT is based on the standard pencil decomposition
used by FFTW-MPI and the supplied in-tree prototype `PencilGrid.h/.cpp` and
`Domain.h/.cpp`. The implementation uses FFTW's guru interface for local
transforms and MPI derived datatypes for the all-to-all layouts.

## Validation

Run `make -C tests run` to build the checks and execute NDIM=1, 2, and 3 with
1, 2, and 4 MPI ranks. The test covers ownership/range mappings, FFT round trips,
real and Fourier row slices, periodic ghost copies including corners, and
accumulation of ghost contributions back onto their owners. Override the rank
list with `NP_LIST="1 2 4 8"` where the MPI runtime has enough resources.
