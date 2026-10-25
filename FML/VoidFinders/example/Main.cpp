#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Triangulation/DTFE.h>
#include <FML/VoidFinders/SphericalVoidFinder.h>
#include <FML/VoidFinders/VoxelVoidFinder.h>
#include <FML/VoidFinders/ZobovVoidFinder.h>
#include <random>

//=======================================================================================
// Example of the void finders (as in Revolver, github.com/seshnadathur/Revolver):
//
// ZobovVoidFinder: watershed on the Voronoi densities of the particles (needs CGAL)
// VoxelVoidFinder: watershed on the CIC density field smoothed with a Gaussian (needs FFTW)
// VoxelWatershed : the watershed on any density grid, here the DTFE density field
// SphericalVoidFinder: spheres with integrated density contrast Delta = -0.8 (needs FFTW for the candidates)
//
// The watershed voids are the zones (no merging). Positions are in [0,1), the catalogues
// are written with distances in units of the box size given.
//=======================================================================================

const int NDIM = 3;
struct Particle {
    double Pos[NDIM];
    Particle() = default;
    constexpr int get_ndim() const { return NDIM; }
    double * get_pos() { return Pos; }
    // If the particles have a get_mass method the masses are used
};

int main() {

    const double boxsize = 1000.0; // Only used for the output
    const size_t npart = 100000;

    // Some clustered particles (the same on all tasks)
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 0.04);
    std::vector<std::array<double, NDIM>> centers(50);
    for (auto & c : centers)
        for (auto & x : c)
            x = uniform(rng);
    std::vector<Particle> partvec(npart);
    for (size_t i = 0; i < npart; i++)
        for (int idim = 0; idim < NDIM; idim++) {
            double x = i % 3 ? centers[i % 50][idim] + gauss(rng) : uniform(rng);
            x -= std::floor(x);
            partvec[i].Pos[idim] = x >= 1.0 ? 0.0 : x;
        }

    // Distribute them across tasks (each task keeps the particles in its x-domain)
    FML::PARTICLE::MPIParticles<Particle> part;
    part.create(partvec.data(), partvec.size(), partvec.size(), FML::xmin_domain, FML::xmax_domain, true);

    //=======================================================================================
    // ZOBOV voids and clusters. Default cuts as in Revolver: minimum density below the mean
    // (maximum above for clusters) and at least 5 particles
    //=======================================================================================
    FML::VOIDFINDER::ZobovOptions zobov_options;
    auto zobov_voids = FML::VOIDFINDER::ZobovVoidFinder(part.get_particles_ptr(), part.get_npart(), zobov_options);
    FML::VOIDFINDER::write_catalogue("zobov_voids.txt", zobov_voids, boxsize);

    zobov_options.find_clusters = true;
    auto zobov_clusters = FML::VOIDFINDER::ZobovVoidFinder(part.get_particles_ptr(), part.get_npart(), zobov_options);
    FML::VOIDFINDER::write_catalogue("zobov_clusters.txt", zobov_clusters, boxsize);

    //=======================================================================================
    // Voxel voids. Grid size and smoothing as in Revolver unless set in the options
    // (the default grid size is rounded down to a multiple of the number of tasks, set
    // Nmesh to get the same result for any number of tasks)
    //=======================================================================================
    FML::VOIDFINDER::VoxelOptions voxel_options;
    auto voxel_voids = FML::VOIDFINDER::VoxelVoidFinder(part.get_particles_ptr(), part.get_npart(), voxel_options);
    FML::VOIDFINDER::write_catalogue("voxel_voids.txt", voxel_voids, boxsize);

    //=======================================================================================
    // The watershed on any density field (in units of the mean), here from the DTFE
    //=======================================================================================
    FML::TRIANGULATION::DTFE<Particle> dtfe;
    dtfe.create(part.get_particles_ptr(), part.get_npart());
    FML::GRID::FFTWGrid<NDIM> density;
    dtfe.density_to_grid(64, density);
    auto dtfe_voids = FML::VOIDFINDER::VoxelWatershed(density);
    FML::VOIDFINDER::write_catalogue("dtfe_voxel_voids.txt", dtfe_voids, boxsize);

    //=======================================================================================
    // Spherical voids: grow spheres around low density grid cells until the density inside is
    // (1 + Delta) times the mean, keep the largest non-overlapping ones (R >= 2 mean separations)
    //=======================================================================================
    FML::VOIDFINDER::SphericalOptions spherical_options;
    spherical_options.delta_threshold = -0.8;
    spherical_options.recentering_steps = 0; // > 0 to move the centers to get larger spheres (slower)
    auto spherical_voids =
        FML::VOIDFINDER::SphericalVoidFinder(part.get_particles_ptr(), part.get_npart(), spherical_options);
    FML::VOIDFINDER::write_catalogue("spherical_voids.txt", spherical_voids, boxsize);

    // The catalogues are the same on all tasks
    if (FML::ThisTask == 0 and not zobov_voids.empty()) {
        auto & v = zobov_voids[0];
        std::cout << "The deepest ZOBOV void: center (" << v.pos[0] * boxsize << ", " << v.pos[1] * boxsize << ", "
                  << v.pos[2] * boxsize << ") R = " << v.radius * boxsize << " delta_min = " << v.delta_min() << "\n";
    }
}
