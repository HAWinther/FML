// Quick start 2: a particle-mesh N-body simulation in about 60 lines.
//
// A scale-free simulation in an Einstein-de Sitter universe (Omega_m = 1, P(k) ~ k^-2), where linear theory is
// simple: the density grows as the scale factor a and H/H0 = a^-3/2. 2LPT initial conditions at a = 0.02, a leapfrog
// with time steps in ln a, and the power spectrum today against linear theory, P(k, a) = P(k, a_ini) (a / a_ini)^2:
// equal on the largest scales, above it on smaller scales where the growth is nonlinear. All lengths are in units of
// the box, the time variable is dtau = H0 dt / a^2 and the velocities are v = dx/dtau.
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>

#include <algorithm>
#include <cmath>
#include <iostream>

// The particle type: what the algorithms need (position and velocity), nothing more
struct Particle {
    double pos[3]{}, vel[3]{};
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return 3; }
};

int main() {
    const int Npart_1D = 128, Nmesh = 128, nsteps = 30;
    const double aini = 0.02, aend = 1.0;

    // Initial conditions: P(k)/V at aini as a function of k in units of 1/box, 2LPT, fixed amplitudes
    const double A = 1e-6;
    auto Pofk_over_volume = [&](double kBox) { return kBox > 0 ? A * std::pow(kBox / (2 * M_PI), -2) : 0.0; };
    auto one = [](double) { return 1.0; };
    const double vnorm = aini * aini * std::pow(aini, -1.5); // v = a^2 (H/H0) f_n Psi_n with f_1 = 1, f_2 = 2
    FML::RANDOM::RandomGenerator rng(1234);
    FML::PARTICLE::MPIParticles<Particle> part;
    FML::NBODY::NBodyInitialConditions<3, Particle>(part, Npart_1D, 1.5, Nmesh, true, &rng, one, Pofk_over_volume, 2,
                                                    "gaussian", 0.0, 1.0, 1.0 / aini - 1.0, {vnorm, 2 * vnorm});

    // The power spectrum of the initial conditions (CIC with interlacing; no shot noise on the initial lattice)
    auto power_spectrum = [&]() {
        FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> pofk(Nmesh / 2);
        pofk.subtract_shotnoise = false;
        FML::CORRELATIONFUNCTIONS::compute_power_spectrum<3>(
            Nmesh, part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), pofk, "CIC", true);
        return pofk;
    };
    const auto pofk_ini = power_spectrum();

    // Leapfrog with steps uniform in ln a: positions at a_i, velocities half a step ahead. The force is computed at
    // a_i, with the Poisson equation nabla^2 Phi = 1.5 Omega_m a delta; the time steps are the integrals of
    // dtau = a^-3/2 da (drift) and of (a / a_i) a^-3/2 da (kick)
    auto a_of = [&](double i) { return aini * std::pow(aend / aini, std::clamp(i, 0.0, double(nsteps)) / nsteps); };
    for (int i = 0; i <= nsteps; i++) {
        const double a = a_of(i), avel_old = a_of(i - 0.5), avel_new = a_of(i + 0.5), anew = a_of(i + 1);
        auto nleftright = FML::INTERPOLATION::get_extra_slices("CIC");
        FML::GRID::FFTWGrid<3> density(Nmesh, nleftright.first, nleftright.second);
        FML::INTERPOLATION::particles_to_grid(
            part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), density, "CIC");
        density.fftw_r2c();
        std::array<FML::GRID::FFTWGrid<3>, 3> force;
        FML::NBODY::compute_force_from_density_fourier<3>(density, force, "CIC", 1.5 * a);
        FML::NBODY::KickParticles<3>(force, part, 2.0 / a * (std::sqrt(avel_new) - std::sqrt(avel_old)), "CIC");
        FML::NBODY::DriftParticles<3, Particle>(part, 2.0 * (1.0 / std::sqrt(a) - 1.0 / std::sqrt(anew)));
    }

    // The power spectrum today against linear theory
    const auto pofk = power_spectrum();
    double max_dev = 0.0;
    if (FML::ThisTask == 0)
        std::cout << "# k [2pi/box]   P(k) / P_linear(k)\n";
    for (int i = 0; i < pofk.n; i++) {
        const double ratio = pofk.pofk[i] / (pofk_ini.pofk[i] * std::pow(aend / aini, 2));
        if (FML::ThisTask == 0)
            std::cout << pofk.kbin[i] / (2 * M_PI) << " " << ratio << "\n";
        if (pofk.kbin[i] < 2 * M_PI * 2.5)
            max_dev = std::max(max_dev, std::abs(ratio - 1.0));
    }
    if (FML::ThisTask == 0)
        std::cout << "Largest scales (k <= 2 x 2pi/box): |P/P_linear - 1| <= " << max_dev << "\n";
    return max_dev < 0.05 ? 0 : 1;
}
