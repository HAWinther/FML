// Quick start 1: the power spectrum of a particle catalogue.
//
// Your own particle type, distributed over the MPI tasks, and P(k) with cloud-in-cell assignment and interlacing. The
// catalogue is uniformly random, so P(k) is the shot noise 1/n: we check that.
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/RandomGenerator/RandomGenerator.h>

#include <cmath>
#include <iostream>

// Any type with get_pos() (positions in the box [0,1)) works as a particle: no base class needed
struct Particle {
    double x[3];
    double * get_pos() { return x; }
    constexpr int get_ndim() const { return 3; }
};

int main() {
    // The same random catalogue on every task (box units)
    const size_t Npart = 200000;
    std::vector<Particle> catalogue(Npart);
    FML::RANDOM::RandomGenerator rng(1234);
    for (auto & p : catalogue)
        for (auto & x : p.x)
            x = rng.generate_uniform();

    // Distribute it: every task keeps the particles in its part of the box
    FML::PARTICLE::MPIParticles<Particle> part;
    part.create(catalogue, 2 * Npart / FML::NTasks + 1000, FML::PARTICLE::Input::Replicated);

    // P(k) on a 64^3 grid, CIC with interlacing, without subtracting the shot noise
    const int Nmesh = 64;
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> pofk(Nmesh / 2);
    pofk.subtract_shotnoise = false;
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum<3>(
        Nmesh, part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), pofk, "CIC", true);

    // In a box of 1000 Mpc/h: k in h/Mpc, P in (Mpc/h)^3. The shot noise is V/N
    const double box = 1000.0, shotnoise = box * box * box / double(Npart);
    pofk.scale(box);
    double mean = 0.0;
    for (int i = 0; i < pofk.n; i++)
        mean += pofk.pofk[i] / shotnoise / pofk.n;
    if (FML::ThisTask == 0) {
        std::cout << "# k [h/Mpc]   P(k) [(Mpc/h)^3]   P / shot noise\n";
        for (int i = 0; i < pofk.n; i++)
            std::cout << pofk.kbin[i] << " " << pofk.pofk[i] << " " << pofk.pofk[i] / shotnoise << "\n";
        std::cout << "Mean P / shot noise = " << mean << " (expected 1)\n";
    }
    return std::abs(mean - 1.0) < 0.05 ? 0 : 1;
}
