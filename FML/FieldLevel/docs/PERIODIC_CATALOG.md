# Periodic catalog observations

`PeriodicCatalog.h` provides `PeriodicCatalogT<Grid>` for streaming a periodic
unweighted or nonnegative-weighted catalog to normalized Fourier density modes.
It supports FFTWGrid slabs and PencilGrid pencils. Positions can arrive on any
rank; the adapter wraps them and routes each batch to the fine-grid owners.
All ingestion, finish and mode-export calls are collective, including on ranks
with no input rows. The read-only `fourier_grid()` accessor is local.

For nonzero, non-Nyquist modes, the convention is

```
delta_n = sum_i w_i exp[-2 pi i n.(x_i-origin)/L] / sum_i w_i.
C_n = sum_i w_i^2 / (sum_i w_i)^2,
P_Poisson = L^3 C_n,
N_effective = 1/C_n.
```

The zero mode and every Nyquist plane are set to exactly zero. No shot noise is
subtracted from the observed field. `poisson_power` is an independent-object
Poisson **reference**, not an estimate of the actual stochastic covariance of
clustered galaxies. For unit weights, C=1/N and P=L^3/N. Zero weights are allowed;
`rows` still counts those entries, while effective count/variance use their weights.
Negative/nonfinite weights and nonfinite coordinates are rejected collectively.
At least one positive weight is required globally. Physical length units of
positions, origin and boxsize must agree; no cosmology/redshift conversion is
performed. Fourier density itself is dimensionless; power has length^3 units.

```cpp
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
auto prototype = FML::GRID::make_grid<Grid>(model.configuration().mesh, 0, 0,
                                           model.domain());
FF::PeriodicCatalogSettings options;
options.boxsize = model.configuration().boxsize;
options.origin = {0,0,0};
options.epsilon = 1e-8;
FF::PeriodicCatalogT<Grid> observation(prototype, options);
// Any number of collectively synchronized batches; non-reader ranks may pass {}.
observation.add_batch(local_entries); // CatalogEntry {physical_position, weight}
const auto statistics = observation.finish();
auto data = observation.modes(model.modes());
auto posterior = FF::fixed_noise_posterior(model, std::move(data),
                                          statistics.poisson_power);
```

The output prototype must share the model's mesh and local Fourier layout for
`modes(model.modes())`; wavevectors/indices are validated, so a foreign layout is
not silently interpreted. This helper preserves the model's spherical/cubic
mode selection, including cubic corners. Use `fourier_grid()` and an explicit
normalized mode transfer if a different observation mesh is needed. This first
adapter does not itself choose the initial, matter or likelihood cutoffs.

The spreading uses FML's PSWF NUFFT kernel. Batches accumulate into an owned real
buffer on the oversampled grid; only `finish()` performs the FFT/deconvolution.
The kernel, FFT grid/plans, ownership table and inverse kernel factors are reused.
No full catalog, full field or Fourier coefficients are all-gathered. Storage is
one oversampled grid (with ghost cells), one local real accumulator and one output
grid, plus bounded routing buffers. `max_input_rows` defaults to 262144 per rank;
`max_received_rows` defaults to 1048576 per rank. A skewed batch exceeding the
receive cap is rejected before payload transfer; reduce input batch sizes or
raise the explicit cap. MPI payloads use checked whole-record byte chunks,
without int-sized aggregate displacements. The message cap defaults to 8 MiB.

The actual fine mesh is rounded to a compatible even size with sufficient
kernel ghost support, with a search limit of 64 times the output mesh. Pencils
do not inherit the slab-only requirement that fine mesh size divide the total
rank count. The requested tolerance must be at least 32 times the grid scalar's
machine epsilon and require no more than the NUFFT kernel's supported width 16.
`fine_mesh()`/`kernel_width()` expose the chosen geometry. A requested epsilon is
the underlying kernel accuracy control, not a universal rigorous error bound;
validate it on the modes/catalog accuracy relevant to the inference.

Weights are internally scaled by a running **global** maximum. A larger weight
in a later batch rescales the existing accumulator/moments, avoiding overflow in
raw sums or squared weights. Fourier modes and Poisson reference are invariant
under a common positive weight scaling. Statistics include the scale and scaled
weight moments, so raw absolute sums need not be represented in a double.
Counts are checked 64-bit integers; weight moments are accumulated/reduced in
long double. Kernel/grid arithmetic uses the configured FML grid precision.

`finish()` is idempotent. A finished or failed stream must be `reset()` before
further ingestion; reset reuses allocations/plans. Objects are noncopyable and
nonmovable. Do not change their communicator/domain while alive. Positions are
fixed observations here: this adapter does not provide a position adjoint or
replace LPT matter assignment. The separate [PSWF matter option](NUFFT_MATTER.md)
provides evolved-particle deposition and its position adjoint. Catalog-specific
modeling and free-IC scientific validation remain open.
[Eulerian checkpoint/restart](EULERIAN_RESTART.md) is now available.
[Compact latent/product meshes](COMPACT_MESHES.md) retain the compact observation
layout used by this adapter.

## File conversion examples

The `catalog_to_modes` example reads three numeric physical coordinates plus an
optional weight per row, as unquoted CSV or whitespace. An exact `x,y,z` or
`x,y,z,weight` header and `#` comment lines are accepted. Input `-` reads standard
input on rank zero. MPI stream writers should send a final
`#FML_CATALOG_END` line rather than depend on launcher-specific pipe-EOF
forwarding; the HDF5 wrapper sends it automatically and verifies the native row
count. Plain input files end at EOF. Readers stream bounded batches; no temporary full catalog is
written. Outputs are one owned-mode CSV per rank plus a small JSON metadata file.
Output paths should be outside the library source tree. Use a fresh prefix when
changing rank counts, or read only the rank files declared by metadata `ranks`;
older files from a previous larger run are not removed automatically.

```sh
make -C FML/FieldLevel/example catalog_to_modes
/path/to/catalog_to_modes catalog.csv /external/catalog 96 2000 .1 cube
# MPI slab or pencil build (choose the appropriate compiler/FFTW variables):
make -C FML/FieldLevel/example catalog_to_modes USE_MPI=true USE_PENCIL_GRID=true
mpirun -np 4 /path/to/catalog_to_modes catalog.csv /external/catalog 96 2000 .1 cube
```

Use a clean or separate build when changing grid/MPI build flags. The arguments
are `INPUT PREFIX NMESH BOXSIZE KMAX cube|sphere [BATCH_ROWS] [EPSILON]`.
Defaults are 100000 rows per batch and epsilon=1e-8. The example uses origin zero;
use the header API for another origin.

`example/catalog_hdf5.py` optionally reads NumPy/h5py HDF5 input and pipes it to
the native example. It supports either a one-dimensional compound `galaxies`
dataset or a group of equal-length scalar columns. Coordinates default to x/y/z;
a weight field is optional. No native C++ HDF5 dependency is added.

```sh
python3 FML/FieldLevel/example/catalog_hdf5.py \
  --input catalog.h5 --exe /path/to/catalog_to_modes \
  --output-prefix /external/catalog --mesh 96 --boxsize 2000 --cutoff .1
# Add --weight FIELD for weights, --group NAME/--columns X Y Z for other schemas,
# and --np 4 --mpirun 'mpirun ...' for an MPI executable.
```

## Verification

The native suite compares **all** usable modes against independent direct sums,
including periodic images, boundaries, nonzero origins, increasing weight scales
across batches, root-only/distributed input, empty local ranks, one-record MPI
chunks, weight-scale invariance by 1e+/-150, Poisson normalization, reset/error
states and input/configuration/capacity failures. It plugs the exported data and
noise into an actual Eulerian inference target. Serial and slab/pencil 1/2/4-rank
checks are included. Native precision and end-to-end CSV/HDF5 results are in
[the adapter validation report](validation/studies/periodic_catalog/RESULTS.md).

```sh
make -C FML/FieldLevel/tests periodic-catalog USE_MPI=false
make -C FML/FieldLevel/tests periodic-catalog USE_MPI=true NP_LIST='1 2 4'
python3 FML/FieldLevel/tests/check_catalog_io.py --serial /path/to/catalog_to_modes
```

The IO check optionally accepts `--mpi-slab`, `--mpi-pencil` and `--mpirun`.
Its HDF5 tests are explicitly skipped if h5py is absent. These are small controlled
catalogs, not the public challenge data or a new sigma8 posterior.
