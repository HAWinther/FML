# PencilGrid: a grid on a pencil decomposition

`PencilGrid<N>` (N = 2, 3; `PencilGrid.h`) is a distributed grid with in-place real-to-complex FFTs, like
`FFTWGrid<N>` but on a `PencilDomain` (`Global/Domain.h`) of `P0 x P1` tasks instead of slabs. It has the same
interface and conventions as `FFTWGrid`, so code written for one reads the same for the other. The grid code of FML is
written against this common interface (`FFTWGrid/GridInterface.h`), so the library, the N-body code and the snapshot
analyzer run on slabs or on pencils (see [Using pencils in FML](#using-pencils-in-fml)). Needs `USE_MPI` and
`USE_FFTW`.

**When to use pencils.** A slab decomposition can use at most `Nmesh` tasks (one x-slice each) and FFTW-MPI needs
`NTasks` to divide `Nmesh`. Pencils split two axes, so up to `Nmesh^2` tasks, and any `Nmesh` works on the default
pencils. On few tasks slabs are a little faster (FFTW has no transposes to do there): use pencils when the number of
tasks approaches `Nmesh`, or when `Nmesh` is not divisible by it.

## Layout

- **3D**: the task with process coordinates `(p0, p1)` (rank `p0 * P1 + p1`) has the cells with x in its x-range, y
  in its y-range and all z. In Fourier space it has all kx, its part of ky in `[0, Nmesh)` (split over `P0`) and its
  part of kz in `[0, Nmesh/2]` (split over `P1`).
- **2D**: the domain has `P1 = 1` (slabs in x). Task `p0` has its x-range and all y; in Fourier space all kx and its
  part of ky in `[0, Nmesh/2]`.
- The x- and y-ranges are those of the domain, whose edges must lie on cells (`edge * Nmesh` an integer; with
  `PencilDomain::uniform` `P0` and `P1` must divide `Nmesh`). Without a domain: the process grid of
  `FML::pencil_domain()` (`MPI_Dims_create`) in 3D, `NTasks x 1` in 2D, with the edges on cells as even as possible
  (so any `Nmesh` works, unlike `FFTWGrid` which needs `NTasks` to divide it). `get_domain().particle_domain()` is the domain for `MPIParticles` (a Slab when `P1 = 1`), so that
  particles live on the task of their cells.
- The transform: r2c along the last axis, then (3D) a transpose in the row (`p0` fixed) and an FFT along y, then a
  transpose in the column (`p1` fixed) and an FFT along x (FFTW's guru interface). A transpose packs the block for
  each task (cache-blocked copies), exchanges them with `MPI_Alltoallv` and unpacks; with one task in the row or column
  the block is copied directly. The grid and one work array of about the same size take turns as source and
  destination. The copies of a grid share the communicators, the work array and the plans (so they must not be
  transformed at the same time from different threads).
- Speed: `tests/benchmark_pencil_grid` times a forward and an inverse FFT against `FFTWGrid` (`mpirun -np P
  ./benchmark_pencil_grid [Nmesh ...]`). On 2 tasks (2 cores) at Nmesh = 64 - 256: 0.9 - 1.3 times FFTWGrid's time;
  on 1 task 1 - 1.9 times (FFTW avoids the two transposes there). A whole N-body run on 2 tasks (96^3 particles and
  mesh, PM and COLA with the in-run power spectra) takes 1.05 - 1.2 times the slab time.

## Interface (as FFTWGrid)

```cpp
FML::GRID::PencilGrid<3> grid(Nmesh, n_extra_left, n_extra_right);        // or (..., PencilDomain domain)
grid.fill_real_grid(func);                                               // func(std::array<double,3> & x), x in [0,1)
for (auto && index : grid.get_real_range()) {                            // or get_real_range_of_rows(irow, irow + 1)
    auto coord = grid.get_coord_from_index(index);                        // local (ix, iy, iz)
    auto x = grid.get_real_position(coord);
    grid.set_real_from_index(index, value);
}
grid.fftw_r2c();                                                         // normalized by 1/Nmesh^N
for (auto && index : grid.get_fourier_range()) {
    std::array<double, 3> kvec;
    double kmag;
    grid.get_fourier_wavevector_and_norm_by_index(index, kvec, kmag);    // 2 pi n (multiply by 1/Boxsize)
    grid.set_fourier_from_index(index, grid.get_fourier_from_index(index) * f(kmag));
}
grid.fftw_c2r();
grid.communicate_boundaries();
```

The methods have the names and meaning of `FFTWGrid`: `get_index_real`, `get_coord_from_index`, `get_real`,
`set_real`, `add_real` (and `_from_index`), the Fourier versions, `get_real_position`, `get_fourier_wavevector`
(`_from_index`, `get_fourier_integer_wavevector_from_index`), `get_fourier_wavevector_and_norm(2)_by_index`,
`fill_real_grid`, `fill_fourier_grid`, `get_local_nrows`, `get_real_range_of_rows`, `get_local_nx`,
`get_local_x_start`, `get_nmesh`, `get_ntot_real`, `get_ntot_fourier`, `get_grid_status_real`, `nan_in_grids`,
`info`, `free`, and the free functions `fftw_r2c(in, out)`, `fftw_c2r(in, out)`. The differences:

- Real coordinates are local in x and y (`get_local_y_start`, `get_local_ny`), Fourier coordinates `(kx, iky, ikz)`
  are global in kx and local in ky and kz (`get_local_ky_start`, `get_local_nky`, `get_local_kz_start`,
  `get_local_nkz`).
- The Fourier rows are lines along kx: loop over `get_local_nrows_fourier()` rows with `get_fourier_range_of_rows`.
- The extra cells (`n_extra_left`, `n_extra_right`) are in x and, in 3D, in y (corners included). They are not kept
  through a Fourier transform (the Fourier data uses the same memory): call `communicate_boundaries` after.
  `add_contribution_from_extra_cells` adds them to the cells of the neighbors they belong to (after assigning
  particles) and sets them to zero.
- `dump_to_file` / `load_from_file` write and read a raw copy (the header, the domain and the data of each task:
  read it back with the same number of tasks). `storage_bytes(Nmesh, domain)` gives the memory a grid would take
  without allocating it.
- No 1D, no wisdom files. The plans use FML's planner flag (`FML::GRID::set_fftw_planner_flag`, as `FFTWGrid`;
  `FFTW_ESTIMATE` by default), made when the grid is constructed.

## Using pencils in FML

The particles live on the task of their cells, so a run on pencils sets the global domain once, at the start, and
makes every grid on it:

```cpp
FML::set_domain(FML::pencil_domain());          // or FML::pencil_domain(px, py), FML::Domain::uniform_pencil(px, py)
auto grid = FML::GRID::make_grid<FML::GRID::PencilGrid<3>>(Nmesh, n_extra_left, n_extra_right);  // on FML::domain
FML::PARTICLE::MPIParticles<Particle> part;     // takes FML::domain when created
```

| Code | How to get pencils |
|---|---|
| `SimulationCode` (N-body) | `make USE_PENCIL_GRID=true`; the pencils are `FML_PENCILS="px py"` from the environment, or as square as possible. Every model, COLA, lightcones, neutrinos, ICs and output work on both |
| `SnapshotAnalyzer` | `make USE_PENCIL_GRID=true` (the same `FML_PENCILS`); every analysis works on both |
| `Analysis` | `-DFML_ANALYSIS_PENCIL_GRID` (`AnalysisGrid<N>`) |
| `Survey` | `-DFML_SURVEY_PENCIL_MESH` (the survey meshes on the pencils of `FML::domain`) |
| Library functions on grids | Templated on the grid: pass a `PencilGrid` (power spectra, polyspectra, random fields, LPT, smoothing, Hessian, NUFFT, multigrid, voids, morphology, cosmic web, hybrid bias, sheet interpolation, glass ICs, ...). Some take the grid type as a template argument, e.g. `RSDReconstructionFourierMethod<N, T, FML::GRID::PencilGrid>` |
| EdS arbitrary-order LPT | `compute_eds_lpt(input, options)` with a `PencilGrid` input: computed natively on pencils |
| Particle methods | On the domain of the particles: FoF (`FoFDomain.h`), Delaunay / DTFE (`MPIPeriodicDelaunay(domain)`, `DTFE(domain)`), SPH, the GADGET / HDF5 / RAMSES readers |
| Wave dark matter | `FML::WAVEDM::SolverT<FML::GRID::CPencilGrid<3>>` |
| Flat-sky lensing | `shear_from_convergence` on a `PencilGrid<2>` |

FieldLevel: periodic linear and 1LPT/2LPT inference support
`DistributedLinearModelT<PencilGrid<3>>` / `DistributedLPTModelT<PencilGrid<3>>`,
including adjoints and physical reconstruction; see [FieldLevel/FRAMEWORK.md](../FieldLevel/docs/FRAMEWORK.md).
Masked Gaussian/Poisson observations and same-layout checkpoint/restart are also
supported. Distributed examples and the survey calibration driver select pencils
with `USE_MPI=true USE_PENCIL_GRID=true` (`FML_PENCILS="px py"` optionally sets the
process grid). Checkpoint migration and the original serial BAO backend remain
separate extensions; full scientific/scaling calibration is not claimed by the port.

## Writing code for slabs and pencils

Write the function once, templated on the grid, with a fast path for `FFTWGrid` where it matters:

```cpp
template <int N, template <int> class Grid = FML::GRID::FFTWGrid>
void my_filter(Grid<N> & grid) {
    for (FML::IndexIntType irow = 0; irow < grid.get_local_nrows_fourier(); irow++)
        for (auto && index : grid.get_fourier_range_of_rows(irow, irow + 1)) {
            auto k = grid.get_fourier_integer_wavevector_from_index(index);   // never decode the index yourself
            ...
        }
    if constexpr (FML::GRID::is_fftwgrid_v<Grid<N>>) { /* slab-only fast path */ }
}
```

The rules (`FFTWGrid/GridInterface.h` has the full list):

- Make grids with `make_grid<Grid>(Nmesh, left, right, domain)` (or `make_grid_like(grid, ...)`) on the domain of
  the particles, never with a constructor: the grid must be decomposed as the particles.
- Ask the grid what it has: `get_local_range(axis)` = `{first, count}` in global cells (all of an axis that is not
  split), `has_dc_mode(grid)` for k = 0, `CellOwners` for the task of any cell.
- Never mix real-space indices of grids with different extra cells: a `PencilGrid` real index depends on them (the
  y-extras lie inside each x-slice), an `FFTWGrid` one does not. Go through coordinates. Fourier indices do not depend
  on the extra cells for either grid.
- Particles that move must be communicated on every split axis (y too).
- `GhostedCells` (`FFTWGrid/GhostedCells.h`) holds per-cell data of any type with ghost layers from the neighbouring
  tasks, on slabs and pencils alike.

## Example and tests

`make` builds the example (an FFT round trip); `make -C tests run` runs the tests on 1, 2 and 4 tasks (`NP_LIST`):
the layout, indices and coordinates, the Fourier transform against `FFTWGrid`, the round trip, the extra cells (with
the corners) and their contribution back, copies, and the particle domain, for 2D and 3D, every process grid, uneven
pencils and odd meshes. The modules on pencils are tested by `tests/test_*_pencil` and `tests/test_*_domain` in the
top `tests` folder (each compares with the slabs, every process grid of the tasks), and by the N-body regression run
on pencils.

## CPencilGrid: complex fields (complex-to-complex FFTs)

`CPencilGrid<N>` (`CPencilGrid.h`, compiled in `PencilGrid.cpp`) is the complex counterpart of `PencilGrid`, as
`CFFTWGrid<N>` (`FFTWGrid/CFFTWGrid.h`) is of `FFTWGrid`: the same pencils, extra cells, interface and transposes,
but the values in position space are complex (`get_real` returns a `ComplexType`; no row padding) and the Fourier grid
has all `Nmesh^N` modes (3D: all kx, ky and kz both split over `[0, Nmesh)`; every mode has multiplicity 1 and the
integer wave-vector is in `(-Nmesh/2, Nmesh/2]`). The transforms are `fftw_forward()` (normalized by `1/Nmesh^N`) and
`fftw_backward()`: an FFT along the last axis on the pencils, then the same row / column transposes and FFTs as
`PencilGrid`. The wave dark matter solver runs on either (`WaveDarkMatter`: `SolverT<CPencilGrid<3>>`).
`tests/test_complex_grids` checks both against the exact coefficients of plane waves and against each other.
