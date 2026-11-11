# CosmicWeb output storage

T-web, V-web and direct tensor classification keep the existing `FFTWGrid`
classification output by default. For lower memory use, opt in to byte labels:

```cpp
using namespace FML::COSMICWEB;
TWebOptions options;
options.classification_storage = ClassificationStorage::Bytes;
CosmicWeb result;
TWeb(delta, options, result);

// Coordinates are LOCAL to this rank's slab.
int type = result.get_classification({ix, iy, iz});
auto global_ix = result.get_local_x_start() + ix;
const auto & labels = result.classification_bytes.get_data();
```

`get_classification`, `get_nmesh`, `get_local_nx` and `get_local_x_start`
work with either storage mode. In byte mode `result.classification` remains
unallocated. `classification_bytes` contains exactly `local_nx * Nmesh * Nmesh`
bytes, ordered as `(ix * Nmesh + iy) * Nmesh + iz`. There are no ghost slices
or FFT padding. Do not use FFTW's padded linear indices with this array.

The four values remain void=0, sheet=1, filament=2 and knot=3. Label payload
drops from eight to one byte per physical cell in double-precision builds
(four to one in single precision), and byte storage also omits FFT padding.
For a 1024^3 mesh this is 1 GiB globally instead of 8 GiB of double-valued
physical labels, before padding. Tensor/eigenvalue grids and other workspace
are separate; the reduction applies to label storage, not total process RSS.

Other entry points accept an explicit storage argument:

```cpp
classify_tensor(tensor, threshold, result, ClassificationStorage::Bytes,
                /*store_eigenvalues=*/true, &density);
VWeb(shear, divergence, normalization, threshold, result,
     ClassificationStorage::Bytes, /*store_eigenvalues=*/false);
```

The original overloads, their default arguments and function-pointer signatures
remain available. `TWebOptions::classification_storage` is appended after the
existing options so existing aggregate initialization keeps its meaning.
Requesting eigenvalues and computing volume/mass fractions work in either mode.
Reusing a result with a different mode releases the previous label allocation.

## Method references

- T-web tidal-tensor classification: Hahn et al. (2007), [arXiv:astro-ph/0610280](https://arxiv.org/abs/astro-ph/0610280), and Forero-Romero et al. (2009), [arXiv:0809.4135](https://arxiv.org/abs/0809.4135).
- V-web velocity-shear classification: Hoffman et al. (2012), [arXiv:1201.3367](https://arxiv.org/abs/1201.3367).
- ORIGAMI shell-crossing morphology: Falck, Neyrinck & Szalay (2012), [arXiv:1201.2353](https://arxiv.org/abs/1201.2353). FML has its own axis sampling, parallelization and halo definitions; this is method attribution, not a claim of exact catalogue equivalence.

For code needing an FFTW grid, convert explicitly on all MPI ranks using the
same communicator/layout as classification:

```cpp
auto classification_grid = result.classification_as_grid();
// Or: result.classification_bytes.to_fftw_grid() for a byte grid alone.
```

Conversion allocates a separate floating-point grid and leaves the original
result unchanged. In default grid mode, `classification_as_grid()` returns a
copy of the existing grid. No conversion happens implicitly during queries.
