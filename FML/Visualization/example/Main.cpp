// Pictures of a clustered particle box (ExampleData) without Python: a projected density from the particles, the
// same slab from a density grid (FFTWGrid) in a few colour maps, a smoothed density contrast with a diverging map,
// and the values as .npy. Run: mpirun -np 2 ./visualization   (writes the PNGs in this folder)
#include <FML/ExampleData/ExampleData.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <FML/Visualization/Visualization.h>

#include <iostream>
#include <vector>

using namespace FML::VISUALIZATION;

struct Particle {
    double pos[3];
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};

int main() {
    // The same toy box on every task (positions in [0,1)); each task keeps the particles in its x-domain
    const auto box = FML::EXAMPLEDATA::clustered_box(400000, 1.0);
    std::vector<Particle> part;
    for (auto & p : box.particles)
        if (p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain)
            part.push_back({{p.pos[0], p.pos[1], p.pos[2]}});
    const size_t npart_total = box.particles.size();

    // 1. The particles in a slab of 10% of the box, seen along z: cloud in cell on 384^2 pixels
    const double slab_start = 0.45, slab_thickness = 0.1;
    const auto particle_image = project_particles(part.data(), part.size(), 384, 2, slab_start, slab_thickness);
    if (FML::ThisTask == 0) {
        write_png("particles_slab.png", particle_image, "magma", ColorScale::automatic_log(5.0, 99.9));
        std::cout << "particles_slab.png: " << particle_image.nx << "^2 pixels, max/mean density "
                  << particle_image.max() << "\n";
    }

    // 2. The density grid: 1 + delta from CIC, the same slab projected, in four colour maps
    const int nmesh = 256;
    const auto nextra = FML::INTERPOLATION::get_extra_slices("CIC");
    FML::GRID::FFTWGrid<3> density(nmesh, nextra.first, nextra.second);
    FML::INTERPOLATION::particles_to_grid(part.data(), part.size(), npart_total, density, "CIC");
    for (auto && index : density.get_real_range())
        density.set_real_from_index(index, 1.0 + density.get_real_from_index(index));
    const auto grid_image = project_grid(density, 2, slab_start, slab_thickness);
    if (FML::ThisTask == 0)
        for (auto name : {"magma", "viridis", "cubehelix", "grey_r"})
            write_png(std::string("density_") + name + ".png", grid_image, name, ColorScale::automatic_log(5.0, 99.9));

    // 3. The density contrast smoothed with a Gaussian of 1% of the box, a thin slice along x, with a diverging map
    //    symmetric about 0 (blue: underdense, red: overdense), enlarged to 512^2 pixels
    for (auto && index : density.get_real_range())
        density.set_real_from_index(index, density.get_real_from_index(index) - 1.0);
    density.fftw_r2c();
    FML::GRID::smoothing_filter_fourier_space(density, 0.01, "gaussian");
    density.fftw_c2r();
    const auto smooth_image = resize(project_grid(density, 0, 0.5, 2.0 / nmesh), 512, 512);
    if (FML::ThisTask == 0) {
        write_png("delta_smoothed.png", smooth_image, "coolwarm", ColorScale::automatic_symmetric(99.0));
        // The values, for numpy / matplotlib: plt.imshow(np.load("delta_smoothed.npy"), origin="lower")
        write_npy("delta_smoothed.npy", smooth_image);
        std::cout << "Wrote particles_slab.png, density_{magma,viridis,cubehelix,grey_r}.png, delta_smoothed.png and "
                     "delta_smoothed.npy\n";
    }
}
