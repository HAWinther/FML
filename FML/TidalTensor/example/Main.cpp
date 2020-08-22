#include <vector>

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/TidalTensor/Hessian.h>

#include <FML/FileUtils/FileUtils.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h>

const int NDIM = 3;

template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;

//=======================================================
// A simple particle class compatible with MPIParticles
//=======================================================
struct Particle {
    double x[NDIM], v[NDIM];
    Particle() {}
    Particle(double * _x) { std::memcpy(x, _x, NDIM * sizeof(double)); }
    double * get_pos() { return x; }
    double * get_vel() { return nullptr; }
    int get_particle_byte_size() { return NDIM * sizeof(double); }
    void append_to_buffer(char * data) { std::memcpy(data, x, NDIM * sizeof(double)); }
    void assign_from_buffer(char * data) { std::memcpy(x, data, NDIM * sizeof(double)); }
};

void ReadParticlesFromFile(FML::PARTICLE::MPIParticles<Particle> & p) {
    // Read ascii file with [x,y,z]
    const double box = 1024.0;
    const std::string filename = "../../../TestData/particles_B1024.txt";
    const int ncols = 3;
    const int nskip_header = 0;
    const std::vector<int> cols_to_keep{0, 1, 2};
    auto data = FML::FILEUTILS::read_regular_ascii(filename, ncols, cols_to_keep, nskip_header);

    // Create particles and scale to [0,1)
    std::vector<Particle> part;
    for (auto & pos : data) {
        for (auto & x : pos)
            x /= box;
        part.push_back(Particle(pos.data()));
    }

    // Create MPI particles by letting each task keep only the particles that falls in its domain
    const bool all_tasks_have_the_same_particles = true;
    const int nalloc_per_task = part.size() / FML::NTasks * 2;
    p.create(part.data(),
             part.size(),
             nalloc_per_task,
             FML::xmin_domain,
             FML::xmax_domain,
             all_tasks_have_the_same_particles);
}

int main() {
    const int Nmesh = 64;

    // Set up a grid
    FFTWGrid<NDIM> f_real(Nmesh, 0, 1);

    // Read particle from file
    FML::PARTICLE::MPIParticles<Particle> p;
    ReadParticlesFromFile(p);

    std::string density_assignment_method = "CIC";
    FML::INTERPOLATION::particles_to_grid<NDIM, Particle>(
        p.get_particles().data(), p.get_npart(), p.get_npart_total(), f_real, density_assignment_method);

    // Compute the Hessian of the potential of f, i.e. D^-2 f_(ij)
    const double norm = 1.0;
    const bool hessian_of_potential_of_f = true;
    std::vector<FFTWGrid<NDIM>> hessian_real;
    FML::HESSIAN::ComputeHessianWithFT(f_real, hessian_real, norm, hessian_of_potential_of_f);

    // Compute eigenvalues of the Hessian above at each point in the grid
    std::vector<FFTWGrid<NDIM>> eigenvalues;
    std::vector<FFTWGrid<NDIM>> eigenvectors;
    const bool compute_eigenvectors = false;
    FML::HESSIAN::SymmetricTensorEigensystem(hessian_real, eigenvalues, eigenvectors, compute_eigenvectors);

    // Output the points in the grid where all eigenvalues are positive
    for (auto real_index : eigenvalues[0].get_real_range()) {
        bool all_positive = true;
        double e[NDIM];
        for (int idim = 0; idim < NDIM; idim++) {
            e[idim] = eigenvalues[idim].get_real_from_index(real_index);
            if (e[idim] < 0.0)
                all_positive = false;
        }
        if (all_positive) {
            auto coord = eigenvalues[0].get_coord_from_index(real_index);
            auto pos = eigenvalues[0].get_real_position(coord);
            for (int idim = 0; idim < NDIM; idim++)
                std::cout << pos[idim] << " ";
            for (int idim = 0; idim < NDIM; idim++) {
                std::cout << e[idim] << " ";
            }
            std::cout << "\n";
        }
    }
}
