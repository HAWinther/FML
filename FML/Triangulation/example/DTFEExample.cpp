#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Triangulation/DTFE.h>
#include <fstream>
#include <random>

//=======================================================================================
// Example of the DTFE: interpolate the density and the velocity field (and its
// divergence, vorticity and shear) of a set of particles to a grid.
//
// We make some clustered particles with a velocity field v = (sin 2pi x, 0, 0) so that
// the divergence is 2pi cos(2pi x) and write a slice of the grids to file.
// Positions must be in [0,1), the gradients are with respect to these positions.
//=======================================================================================

const int NDIM = 3;
struct Particle {
    double Pos[NDIM];
    double Vel[NDIM];
    Particle() = default;
    constexpr int get_ndim() const { return NDIM; }
    double * get_pos() { return Pos; }
    double * get_vel() { return Vel; }
    // If the particles have a get_mass method this mass is used
};

int main() {

    const size_t npart = 200000;
    const int Nmesh = 64; // Must be divisible by the number of tasks

    // Make the same particles on all tasks: half uniform, half in clusters
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 0.03);
    std::vector<Particle> partvec(npart);
    for (size_t i = 0; i < npart; i++) {
        auto & p = partvec[i];
        const double center[3] = {0.25 + 0.5 * (i % 2), 0.5, 0.5};
        for (int idim = 0; idim < NDIM; idim++) {
            double x = i % 3 == 0 ? center[idim] + gauss(rng) : uniform(rng);
            x -= std::floor(x);
            p.Pos[idim] = x >= 1.0 ? 0.0 : x;
            p.Vel[idim] = 0.0;
        }
        p.Vel[0] = std::sin(2.0 * M_PI * p.Pos[0]);
    }

    // Distribute them across tasks (each task keeps the particles in its x-domain)
    FML::PARTICLE::MPIParticles<Particle> part;
    part.create(partvec.data(), partvec.size(), partvec.size(), FML::xmin_domain, FML::xmax_domain, true);

    // Make the tessellation and compute the density at the particles. The buffer is how large part of the
    // neighbor domains we include (we warn if its too small)
    FML::TRIANGULATION::DTFE<Particle> dtfe;
    const double buffer_fraction = 0.5;
    dtfe.create(part.get_particles_ptr(), part.get_npart(), buffer_fraction);

    // The density (in units of the mean) averaged over the grid cells (2^3 samples per cell)
    FML::GRID::FFTWGrid<NDIM> density;
    dtfe.density_to_grid(Nmesh, density, FML::TRIANGULATION::DTFESampling::CellAverage, 2);

    // The velocity, its divergence, vorticity and shear (volume weighted cell averages)
    FML::TRIANGULATION::DTFEVelocityGrids vel;
    dtfe.velocity_to_grid(part.get_particles_ptr(),
                          Nmesh,
                          vel,
                          FML::TRIANGULATION::DTFESampling::CellAverage,
                          FML::TRIANGULATION::DTFEWeighting::Volume,
                          2);

    // Any other quantity: values[nfields * i + f] is field f of particle i. Here the speed squared
    std::vector<double> values;
    for (auto & p : part)
        values.push_back(p.Vel[0] * p.Vel[0]);
    std::vector<FML::GRID::FFTWGrid<NDIM>> fields, gradients;
    dtfe.fields_to_grid(Nmesh, values, 1, fields, &gradients);

    // Write the slice iz = Nmesh/2 (each task writes its own part)
    std::ofstream fp("dtfe_slice_" + std::to_string(FML::ThisTask) + ".txt");
    fp << "# x  y  density  v_x  divergence  exact_divergence  v^2\n";
    const int iz = Nmesh / 2;
    for (int ix = 0; ix < density.get_local_nx(); ix++) {
        const double x = (density.get_local_x_start() + ix) / double(Nmesh);
        for (int iy = 0; iy < Nmesh; iy++) {
            const std::array<int, NDIM> coord{ix, iy, iz};
            fp << x << " " << iy / double(Nmesh) << " " << density.get_real(coord) << " "
               << vel.velocity[0].get_real(coord) << " " << vel.divergence.get_real(coord) << " "
               << 2.0 * M_PI * std::cos(2.0 * M_PI * x) << " " << fields[0].get_real(coord) << "\n";
        }
    }
}
