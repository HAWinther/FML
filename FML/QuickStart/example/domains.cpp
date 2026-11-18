// Quick start 3: the same calculation on slabs, on pencils and with the particles on a Peano-Hilbert curve.
//
// A Domain says which MPI task owns which part of the box. The particles are distributed according to it, and the
// grid is made for the domain of the particles: FFTWGrid on slabs, PencilGrid on pencils. Particles on a
// Peano-Hilbert domain (load-balanced ranges of a space-filling curve) are assigned to a slab grid without moving
// them. The power spectrum is the same in all three cases.
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/Interpolation/DistributedDeposit.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/RandomGenerator/RandomGenerator.h>

#include <cmath>
#include <iostream>

struct Particle {
    double x[3];
    double * get_pos() { return x; }
    constexpr int get_ndim() const { return 3; }
};
using Binning = FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3>;
const int Nmesh = 64;

// P(k) of the particles with CIC on a grid of type Grid, made for the domain of the particles
template <template <int> class Grid>
Binning spectrum(FML::PARTICLE::MPIParticles<Particle> & part) {
    Binning pofk(Nmesh / 2);
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum<3, Particle, Grid>(
        Nmesh, part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), pofk, "CIC", false, part.get_domain());
    return pofk;
}

int main() {
    // A clustered catalogue (random points displaced by two plane waves, in x and in y), the same on every task
    std::vector<Particle> catalogue(200000);
    FML::RANDOM::RandomGenerator rng(1234);
    for (auto & p : catalogue) {
        for (auto & x : p.x)
            x = rng.generate_uniform();
        p.x[0] = std::fmod(p.x[0] + 0.03 * std::sin(2 * M_PI * p.x[0]) + 1.0, 1.0);
        p.x[1] = std::fmod(p.x[1] + 0.02 * std::sin(4 * M_PI * p.x[1]) + 1.0, 1.0);
    }
    const size_t nalloc = 2 * catalogue.size() / FML::NTasks + 1000;

    // 1. Slabs (the default domain)
    FML::PARTICLE::MPIParticles<Particle> slabs;
    slabs.create(catalogue, nalloc, FML::PARTICLE::Input::Replicated);
    auto p_slab = spectrum<FML::GRID::FFTWGrid>(slabs);

    // 2. Pencils: x and y split over a 2D grid of tasks
    FML::PARTICLE::MPIParticles<Particle> pencils(FML::pencil_domain());
    pencils.create(catalogue, nalloc, FML::PARTICLE::Input::Replicated);
    auto p_pencil = spectrum<FML::GRID::PencilGrid>(pencils);

    // 3. Peano-Hilbert: the particles stay on their tasks; the density is assigned to a slab grid across domains
    FML::PARTICLE::MPIParticles<Particle> hilbert(FML::Domain::uniform_peano_hilbert(FML::NTasks));
    hilbert.create(catalogue, nalloc, FML::PARTICLE::Input::Replicated);
    FML::GRID::FFTWGrid<3> density(Nmesh);
    FML::INTERPOLATION::particles_to_grid_any_domain<3, 2>(
        hilbert.get_particles_ptr(), hilbert.get_npart(), hilbert.get_npart_total(), density);
    density.fftw_r2c();
    FML::INTERPOLATION::deconvolve_window_function_fourier<3>(density, "CIC");
    Binning p_hilbert(Nmesh / 2);
    FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum<3>(density, p_hilbert);
    for (int i = 0; i < p_hilbert.n; i++) // subtract the shot noise as compute_power_spectrum does
        p_hilbert.pofk[i] -= 1.0 / double(catalogue.size());
    if (FML::ThisTask == 0)
        for (int i = 0; i < 4; i++)
            std::cout << "k = " << i + 1 << " x 2pi/box: P(k) on slabs " << p_slab.pofk[i] << ", pencils " << p_pencil.pofk[i]
                      << ", Peano-Hilbert " << p_hilbert.pofk[i] << "\n";

    double max_diff = 0.0;
    for (int i = 0; i < p_slab.n; i++)
        max_diff = std::max({max_diff, std::abs(p_pencil.pofk[i] - p_slab.pofk[i]) / std::abs(p_slab.pofk[i]),
                             std::abs(p_hilbert.pofk[i] - p_slab.pofk[i]) / std::abs(p_slab.pofk[i])});
    if (FML::ThisTask == 0)
        std::cout << "Largest relative difference between the three: " << max_diff << "\n";
    return max_diff < 1e-8 ? 0 : 1;
}
