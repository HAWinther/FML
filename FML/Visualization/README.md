# Visualization

Pictures of fields and particles straight from C++, with no Python, matplotlib or image library: a projection of a
distributed grid or of the particles, a colour map, and a PNG file. Useful for a quick look at a simulation while it
runs, or for movies made from the snapshots.

```cpp
#include <FML/Visualization/Visualization.h>
using namespace FML::VISUALIZATION;

// A slab of 10% of the box through a density grid (FFTWGrid or PencilGrid), seen along z: the mean over the slab
Image image = project_grid(density, 2, 0.45, 0.1);
// The particles in the same slab, cloud in cell on 512^2 pixels, in units of the mean
Image image2 = project_particles(part.get_particles_ptr(), part.get_npart(), 512, 2, 0.45, 0.1);

if (FML::ThisTask == 0) {
    write_png("density.png", image, "magma", ColorScale::automatic_log());        // log scale, 1 - 99.9 percentiles
    write_png("delta.png", image_delta, "coolwarm", ColorScale::automatic_symmetric()); // signed field, about 0
    write_png("slab.png", image2, "viridis", ColorScale::logarithmic(0.1, 100.0)); // fixed limits
    write_npy("density.npy", image); // the values for numpy: plt.imshow(np.load(f), origin="lower")
}
```

## Files

| File | |
|---|---|
| `Projection.h` | `project_grid` (FFTWGrid / PencilGrid, 2D or 3D, any axis, a periodic slab, sum or mean) and `project_particles` (NGP or CIC, mass weighted if the particles have `get_mass`). Collective: the same image on every task |
| `Image.h` | `Image` (values, `image(ix, iy)` with iy = 0 the bottom row), `ColorScale` (linear, log, symmetric; fixed limits or percentiles), `to_rgb`, `write_png`, `write_npy`, `resize` |
| `Colormaps.h` | `magma`, `inferno`, `viridis`, `plasma`, `cividis` (matplotlib, CC0), `coolwarm` (diverging, Moreland 2009), `grey`, `cubehelix` (Green 2011); add `_r` to reverse |
| `PNG.h` | A PNG writer with its own deflate (filters per row, LZ77, fixed Huffman codes; stored blocks for noise). RGB or grey, 8 bits |
| `Visualization.h` | Includes all of the above |

Projecting along an axis, the image axes are the two others in order (along z: x to the right and y up; along y: x
and z; along x: y and z). A slab `[start, start + thickness)` in box units may wrap past 1; the whole box is the
default. For a grid, the cells whose positions are in the slab are used (FML's cells are at i / Nmesh, as in the
assignment and the Fourier transforms), and pixel (ix, iy) of the image is the cell at (ix, iy) / Nmesh.

The writers are not collective: call them on one task. The PNGs are a few times larger than with zlib at its best
setting; colour-mapped fields compress well (a 512^2 density map is about 0.2-0.5 MB).

## In SimulationCode and the SnapshotAnalyzer

`images = true` draws fields at every output (`image_<field>_z<z>.png`): the density, the velocity divergence, the
neutrinos, baryons - CDM and the fifth force (SimulationCode), with the settings `images_*` of FML/Analysis/Images.h.
See SimulationCode/ANALYSIS.md.

## Example and tests

`example/` makes pictures of a toy clustered box (ExampleData): the particles in a slab, the density grid in four
colour maps, a smoothed density contrast with a diverging map, and an .npy file. `make && mpirun -np 2 ./visualization`.

`tests/` checks the projections against direct sums (slabs and pencils, every axis, periodic slabs, 1, 2 and 4 tasks),
the particle weights and mass conservation, the colour maps and scales, and reads the PNG and .npy files back with
Python's standard library (zlib) to compare every pixel (`make run`).
