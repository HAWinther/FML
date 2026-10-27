//==========================================================
// Tests for FML::NBODY (NBody/NBody.h) and FML::PGD (PGD/PotentialGradientDecent.h):
//   - the discrete 4-point Green's function is minus one over the square of the 4-point gradient (it had the wrong
//     sign of the sin^2 term) and the Fourier force on a plane wave with the 4-point kernels is 1/D4(k)
//   - the finite difference force uses the chosen Green's function and deconvolution (it always used -1/k^2 and
//     never deconvolved): with stencil 2 (4) it equals the Fourier force with the 2-point (4-point) gradient
//   - DriftParticles with tasks without particles (deadlocked: the empty tasks returned before the collectives)
//   - the periodic wrap after the drift never gives 1.0 and handles displacements larger than the box
//   - PGD with tasks without particles (aborted) and the size of the PGD displacement
//   - NBody.h and PotentialGradientDecent.h can be included in several translation units (test_nbody_odr.cpp)
// Bad input that must stop with a clear message:
//   ./test_nbody fdslices : potential with fewer than stencil_order/2 extra slices (read out of bounds before)
//   ./test_nbody stencil  : stencil order that is not implemented (threw inside an OpenMP region: std::terminate)
// Run with 1, 2 and 4 MPI tasks (OMP_NUM_THREADS=2)
//==========================================================
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/PGD/PotentialGradientDecent.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace FML;
using FFTWGrid = FML::GRID::FFTWGrid<3>;
using FloatType = FML::GRID::FloatType;
template <class T>
using MPIParticles = FML::PARTICLE::MPIParticles<T>;

int nbody_odr_check(); // defined in test_nbody_odr.cpp

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os.precision(6);
    os << x;
    return os.str();
}

struct Particle {
    double pos[3];
    double vel[3];
    double expected[3]; // Where the particle should end up
    double *get_pos() { return pos; }
    double *get_vel() { return vel; }
    constexpr int get_ndim() const { return 3; }
};

// Deterministic white noise as a function of the (global) cell (the same for any number of tasks)
double noise(std::array<double, 3> & x, int Nmesh) {
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 3; i++) {
        h ^= uint64_t(std::lround(x[i] * Nmesh)) + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        h ^= h >> 31;
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 29;
    }
    return double(h >> 11) / double(1ULL << 53) - 0.5;
}

// Deterministic uniform numbers in [0,1)
double uniform(uint64_t i) {
    uint64_t z = i * 0x9E3779B97F4A7C15ULL + 12345;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return double(z >> 11) / double(1ULL << 53);
}

void set_kernels(int laplace, int gradient, bool deconvolve) {
    NBODY::FIDUCIAL_LAPLACE_KERNEL = laplace;
    NBODY::FIDUCIAL_GRADIENT_KERNEL = gradient;
    NBODY::FIDUCIAL_DECONVOLVE_FORCE = deconvolve;
}

// max |a - b| / max |b| over all components and tasks
double rel_diff(std::array<FFTWGrid, 3> & a, std::array<FFTWGrid, 3> & b) {
    double maxdiff = 0.0, maxval = 0.0;
    for (int idim = 0; idim < 3; idim++)
        for (auto && i : b[idim].get_real_range()) {
            maxdiff = std::max(maxdiff, double(std::fabs(a[idim].get_real_from_index(i) - b[idim].get_real_from_index(i))));
            maxval = std::max(maxval, double(std::fabs(b[idim].get_real_from_index(i))));
        }
    FML::MaxOverTasks(&maxdiff);
    FML::MaxOverTasks(&maxval);
    return maxval > 0.0 ? maxdiff / maxval : 1e100;
}

int main(int argc, char ** argv) {
    std::string mode = argc > 1 ? argv[1] : "all";
    const int Nmesh = 16;
    const double dx = 1.0 / Nmesh;

    // A density field (white noise, i.e. all modes) in Fourier space
    FFTWGrid density(Nmesh, 1, 1);
    density.fill_real_grid([&](std::array<double, 3> & x) -> FloatType { return noise(x, Nmesh); });
    density.fftw_r2c();

    //=====================================================
    // Bad input: fewer extra slices than the stencil needs
    //=====================================================
    if (mode == "fdslices") {
        FFTWGrid potential;
        std::array<FFTWGrid, 3> force;
        NBODY::compute_potential_real_from_density_fourier<3>(density, potential, 1.0, 1);
        NBODY::compute_force_from_potential_real<3>(potential, force, "CIC", 4);
        if (FML::ThisTask == 0)
            std::cout << "[FAIL] fdslices: stencil_order = 4 with 1 extra slice did not give an error\n";
        return 1;
    }

    //=====================================================
    // Bad input: a stencil order that is not implemented
    //=====================================================
    if (mode == "stencil") {
        FFTWGrid potential;
        std::array<FFTWGrid, 3> force;
        NBODY::compute_potential_real_from_density_fourier<3>(density, potential, 1.0, 3);
        NBODY::compute_force_from_potential_real<3>(potential, force, "CIC", 3);
        if (FML::ThisTask == 0)
            std::cout << "[FAIL] stencil: stencil_order = 3 did not give an error\n";
        return 1;
    }

    if (mode != "all") {
        if (FML::ThisTask == 0)
            std::cout << "Unknown mode " << mode << " (all, fdslices, stencil)\n";
        return 1;
    }

    check(nbody_odr_check() == 1, "NBody.h and PotentialGradientDecent.h can be included in several translation units");

    //=====================================================
    // 1. The 4-point Green's function is -1/D4^2
    //=====================================================
    {
        double maxerr = 0.0;
        const std::vector<std::array<int, 3>> modes{{1, 0, 0}, {3, 2, 1}, {5, -7, 2}, {8, 8, -3}, {-6, 1, 4}, {7, 0, -3}};
        for (auto & n : modes) {
            std::array<double, 3> kvec;
            double kmag2 = 0.0, sumD2 = 0.0;
            for (int i = 0; i < 3; i++) {
                kvec[i] = 2.0 * M_PI * n[i];
                kmag2 += kvec[i] * kvec[i];
                const double D = NBODY::gradient_kernel_fourier(kvec[i], Nmesh, NBODY::DISCRETE_FOUR_POINT_GRADIENT);
                sumD2 += D * D;
            }
            const double G = NBODY::greens_function_laplace_operator_fourier<3>(
                kmag2, kvec, Nmesh, NBODY::DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT);
            maxerr = std::max(maxerr, std::fabs(G * sumD2 + 1.0));
        }
        check(maxerr < 1e-12, "discrete_4pt Green's function = -1 / (4-point gradient)^2 (max error " + num(maxerr) + ")");
        // The 4-point gradient vanishes when all components are 0 or the Nyquist frequency: no (infinite) potential
        std::array<double, 3> knyq{M_PI * Nmesh, 0.0, -M_PI * Nmesh};
        const double Gnyq = NBODY::greens_function_laplace_operator_fourier<3>(
            2.0 * M_PI * M_PI * Nmesh * Nmesh, knyq, Nmesh, NBODY::DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT);
        check(Gnyq == 0.0, "discrete_4pt Green's function is 0 for the Nyquist modes where D4 = 0 (got " + num(Gnyq) +
                               ")");
    }

    //=====================================================
    // 2. Fourier force on a plane wave with the 4-point kernels: F = sin(kx) / D4(k)
    //=====================================================
    {
        set_kernels(NBODY::DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT, NBODY::DISCRETE_FOUR_POINT_GRADIENT, false);
        const int n = 3;
        const double k = 2.0 * M_PI * n;
        FFTWGrid wave(Nmesh, 1, 1);
        wave.fill_real_grid([&](std::array<double, 3> & x) -> FloatType { return std::cos(k * x[1]); });
        wave.fftw_r2c();
        std::array<FFTWGrid, 3> force;
        NBODY::compute_force_from_density_fourier<3>(wave, force, "CIC", 1.0);
        const double D4 = (8.0 * std::sin(k * dx) - std::sin(2.0 * k * dx)) / (6.0 * dx);
        double maxerr = 0.0;
        for (auto && i : force[1].get_real_range()) {
            auto x = force[1].get_real_position(force[1].get_coord_from_index(i));
            maxerr = std::max(maxerr, std::fabs(force[1].get_real_from_index(i) - std::sin(k * x[1]) / D4) * D4);
            maxerr = std::max(maxerr, double(std::fabs(force[0].get_real_from_index(i))) * D4);
        }
        FML::MaxOverTasks(&maxerr);
        check(maxerr < 1e-10, "Fourier force on a plane wave with the 4-point kernels is sin(kx)/D4(k) (rel. error " +
                                  num(maxerr) + ")");
    }

    //=====================================================
    // 3. The finite difference force uses the chosen Green's function and deconvolution
    // A stencil of order 2 (4) is the same as the Fourier 2-point (4-point) gradient for all modes
    //=====================================================
    {
        struct Case {
            int laplace;
            int stencil;
            int gradient;
            bool deconvolve;
            std::string name;
        };
        const std::vector<Case> cases{
            {NBODY::CONTINUOUS_GREENS_FUNCTION, 2, NBODY::DISCRETE_TWO_POINT_GRADIENT, false, "continuous, stencil 2"},
            {NBODY::DISCRETE_GREENS_FUNCTION_FROM_TWO_POINT, 2, NBODY::DISCRETE_TWO_POINT_GRADIENT, false,
             "discrete_2pt, stencil 2"},
            {NBODY::DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT, 4, NBODY::DISCRETE_FOUR_POINT_GRADIENT, false,
             "discrete_4pt, stencil 4"},
            {NBODY::CONTINUOUS_GREENS_FUNCTION, 4, NBODY::DISCRETE_FOUR_POINT_GRADIENT, true,
             "continuous, stencil 4, deconvolved"},
            {NBODY::DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT, 4, NBODY::DISCRETE_FOUR_POINT_GRADIENT, true,
             "discrete_4pt, stencil 4, deconvolved"},
        };
        for (auto & c : cases) {
            set_kernels(c.laplace, c.gradient, c.deconvolve);
            std::array<FFTWGrid, 3> force_fourier;
            NBODY::compute_force_from_density_fourier<3>(density, force_fourier, "CIC", 1.5);
            FFTWGrid potential;
            std::array<FFTWGrid, 3> force_fd;
            NBODY::compute_potential_real_from_density_fourier<3>(density, potential, 1.5, c.stencil / 2);
            NBODY::compute_force_from_potential_real<3>(potential, force_fd, "CIC", c.stencil);
            const double err = rel_diff(force_fd, force_fourier);
            check(err < 1e-10, "finite difference force = Fourier force (" + c.name + ", rel. diff " + num(err) + ")");
        }
        set_kernels(NBODY::CONTINUOUS_GREENS_FUNCTION, NBODY::CONTINUOUS_GRADIENT, false);
    }

    //=====================================================
    // 4. Periodic wrap of the drift (pointer version, collective)
    //=====================================================
    {
        // pos, displacement
        const std::vector<std::array<double, 2>> cases{
            {0.0, -1e-18}, {0.3, 2.5}, {0.5, -1.7}, {0.9, 0.05}, {0.05, -0.1}, {0.25, 3.0}, {0.7, -2.7}, {0.4, 0.5}};
        std::vector<Particle> p(cases.size());
        for (size_t i = 0; i < cases.size(); i++)
            for (int idim = 0; idim < 3; idim++) {
                p[i].pos[idim] = cases[i][0];
                p[i].vel[idim] = cases[i][1];
            }
        NBODY::DriftParticles<3, Particle>(p.data(), p.size(), 1.0, true);
        bool in_box = true, correct = true, same_as_before = true;
        for (size_t i = 0; i < cases.size(); i++) {
            double x = cases[i][0] + cases[i][1];
            const double expected = x - std::floor(x) >= 1.0 ? 0.0 : x - std::floor(x);
            // The old code (one wrap) for positions in [-1, 2) that do not round to 1.0
            double xold = x;
            if (xold >= 1.0)
                xold -= 1.0;
            if (xold < 0.0)
                xold += 1.0;
            for (int idim = 0; idim < 3; idim++) {
                in_box = in_box and p[i].pos[idim] >= 0.0 and p[i].pos[idim] < 1.0;
                correct = correct and std::fabs(p[i].pos[idim] - expected) < 1e-14;
                if (x >= -1.0 and x < 2.0 and xold < 1.0)
                    same_as_before = same_as_before and p[i].pos[idim] == xold;
            }
        }
        check(in_box, "drift: positions are in [0,1) (never exactly 1.0)");
        check(correct, "drift: displacements larger than the box are wrapped");
        check(same_as_before, "drift: the usual positions are wrapped exactly as before (bit-identical)");
    }

    //=====================================================
    // 5. DriftParticles with tasks without particles: the particles are all on task 0
    //=====================================================
    {
        const int npart = 64;
        std::vector<Particle> part(npart);
        for (int i = 0; i < npart; i++) {
            for (int idim = 0; idim < 3; idim++) {
                part[i].pos[idim] = uniform(6 * i + idim);
                part[i].vel[idim] = (uniform(6 * i + 3 + idim) - 0.5) * 0.9; // less than half a box
            }
            part[i].pos[0] *= 0.1; // In the domain of task 0 for up to 10 tasks
            for (int idim = 0; idim < 3; idim++) {
                const double x = part[i].pos[idim] + part[i].vel[idim];
                part[i].expected[idim] = x - std::floor(x);
            }
        }
        MPIParticles<Particle> mp;
        mp.create(part.data(), npart, 2 * npart, FML::xmin_domain, FML::xmax_domain, true);
        NBODY::DriftParticles<3, Particle>(mp, 1.0, true);
        double maxerr = 0.0;
        int outside = 0;
        for (auto & p : mp) {
            if (p.pos[0] < FML::xmin_domain or p.pos[0] >= FML::xmax_domain)
                outside++;
            for (int idim = 0; idim < 3; idim++) {
                double d = std::fabs(p.pos[idim] - p.expected[idim]);
                maxerr = std::max(maxerr, std::min(d, 1.0 - d));
                if (p.pos[idim] < 0.0 or p.pos[idim] >= 1.0)
                    outside++;
            }
        }
        FML::MaxOverTasks(&maxerr);
        FML::SumOverTasks(&outside);
        check(mp.get_npart_total() == size_t(npart) and outside == 0 and maxerr < 1e-12,
              "DriftParticles with tasks without particles (npart_total = " + std::to_string(mp.get_npart_total()) +
                  ", outside the domain " + std::to_string(outside) + ", max error " + num(maxerr) + ")");
    }

    //=====================================================
    // 6. PGD with tasks without particles
    //=====================================================
    {
        const int npart = 200;
        std::vector<Particle> part(npart);
        for (int i = 0; i < npart; i++) {
            for (int idim = 0; idim < 3; idim++) {
                part[i].pos[idim] = uniform(3 * i + idim + 1000);
                part[i].vel[idim] = 0.0;
            }
            part[i].pos[0] *= 0.1; // All on task 0 (a big overdensity)
        }
        MPIParticles<Particle> mp;
        mp.create(part.data(), npart, 2 * npart, FML::xmin_domain, FML::xmax_domain, true);
        PGD::potential_gradient_decent_method<3, Particle>(
            mp.get_particles_ptr(), mp.get_npart(), 2.0, 100.0, 0.01, 1.5, Nmesh, "CIC");
        mp.communicate_particles();
        double xsum = 0.0;
        int bad = 0;
        for (auto & p : mp)
            for (int idim = 0; idim < 3; idim++) {
                if (not std::isfinite(p.pos[idim]) or p.pos[idim] < 0.0 or p.pos[idim] >= 1.0)
                    bad++;
                xsum += p.pos[idim];
            }
        FML::SumOverTasks(&bad);
        FML::SumOverTasks(&xsum);
        check(mp.get_npart_total() == size_t(npart) and bad == 0 and std::isfinite(xsum),
              "PGD with tasks without particles (npart_total = " + std::to_string(mp.get_npart_total()) + ")");
    }

    //=====================================================
    // 7. PGD displacement S = -alpha * norm * grad O grad^-2 delta for a lattice displaced by a sine wave
    //=====================================================
    {
        const int Np = 16, n = 2;
        const double A = 0.01, k = 2.0 * M_PI * n, alpha = 0.01, norm = 1.5, kl = 1.0, ks = 1000.0;
        MPIParticles<Particle> mp;
        mp.create_particle_grid(Np, 2.0, FML::xmin_domain, FML::xmax_domain);
        for (auto & p : mp) {
            for (int idim = 0; idim < 3; idim++)
                p.vel[idim] = 0.0;
            const double q = p.pos[0];
            p.pos[0] = q - A / k * std::sin(k * q); // delta = A cos(kq) to linear order
            if (p.pos[0] < 0.0)
                p.pos[0] += 1.0;
            p.expected[0] = p.pos[0];
        }
        mp.communicate_particles();
        PGD::potential_gradient_decent_method<3, Particle>(
            mp.get_particles_ptr(), mp.get_npart(), kl, ks, alpha, norm, Np, "CIC");
        const double filter = std::exp(-std::pow(k * k / (ks * ks), 2) - kl * kl / (k * k));
        const double W = std::pow(std::sin(k / (2.0 * Np)) / (k / (2.0 * Np)), 4); // CIC assignment + interpolation
        double num_ = 0.0, den = 0.0;
        for (auto & p : mp) {
            double s = p.pos[0] - p.expected[0];
            s -= std::round(s);
            const double e = -alpha * norm * A / k * std::sin(k * p.expected[0]) * filter * W;
            num_ += s * e;
            den += e * e;
        }
        FML::SumOverTasks(&num_);
        FML::SumOverTasks(&den);
        const double ratio = num_ / den;
        check(std::fabs(ratio - 1.0) < 0.02,
              "PGD displacement is -alpha norm grad grad^-2 delta (towards the overdensity), measured/expected = " +
                  num(ratio));
        mp.communicate_particles();
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All NBody tests passed\n" : std::to_string(fails) + " NBody test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
