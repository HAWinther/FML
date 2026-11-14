// The Fourier-space power-spectrum functions written against the grid interface
// (FFTWGrid/GridInterface.h) give the same results for FFTWGrid and PencilGrid (slabs and pencils, 2D and 3D).
// With an argument (a file name) it writes FFTWGrid's results with all digits (to compare with the original code)
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/TestHelpers.h>

#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

using FML::TESTS::check;
using namespace FML::GRID;
using FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning;

static_assert(has_fourier_grid_interface<FFTWGrid<3>>::value, "FFTWGrid<3> has the interface");
static_assert(has_fourier_grid_interface<FFTWGrid<2>>::value, "FFTWGrid<2> has the interface");
static_assert(has_fourier_grid_interface<PencilGrid<3>>::value, "PencilGrid<3> has the interface");
static_assert(has_fourier_grid_interface<PencilGrid<2>>::value, "PencilGrid<2> has the interface");

namespace {

    template <int N>
    FloatType field(const std::array<double, N> & x, int which) {
        double v = std::sin(2 * M_PI * (x[0] + which * 0.1)) + 0.5 * std::cos(2 * M_PI * (3 * x[1] - x[0]));
        v += 0.2 * std::sin(2 * M_PI * 5 * x[N - 1] + 0.3) + 0.1 * std::cos(2 * M_PI * (2 * x[0] + 7 * x[1]));
        if constexpr (N == 3)
            v += 0.3 * std::cos(2 * M_PI * (x[0] - 2 * x[1] + 4 * x[2]) + which);
        return FloatType(v);
    }

    // All the Fourier-space spectra of the grids (Fourier space)
    struct Spectra {
        std::vector<double> pofk, cross, deconvolved, p0, p2, p4;
    };
    template <int N, template <int> class Grid>
    Spectra spectra(const Grid<N> & a, const Grid<N> & b, int Nmesh) {
        Spectra s;
        PowerSpectrumBinning<N> bins(0.0, 2 * M_PI * Nmesh / 2, Nmesh / 2, PowerSpectrumBinning<N>::LINEAR_SPACING);
        auto p = bins;
        FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum<N>(a, p);
        s.pofk = p.pofk;
        p = bins;
        FML::CORRELATIONFUNCTIONS::bin_up_cross_power_spectrum<N>(a, b, p);
        s.cross = p.pofk;
        p = bins;
        FML::CORRELATIONFUNCTIONS::bin_up_deconvolved_power_spectrum<N>(a, p, "CIC");
        s.deconvolved = p.pofk;
        std::vector<PowerSpectrumBinning<N>> Pell(5, bins);
        std::vector<double> los(N, 0.0);
        los[N - 1] = 1.0;
        los[0] = 0.5;
        FML::CORRELATIONFUNCTIONS::compute_power_spectrum_multipoles_fourier<N>(a, Pell, los);
        s.p0 = Pell[0].pofk;
        s.p2 = Pell[2].pofk;
        s.p4 = Pell[4].pofk;
        return s;
    }

    double max_rel_diff(const Spectra & x, const Spectra & y) {
        double d = 0.0;
        auto cmp = [&](const std::vector<double> & u, const std::vector<double> & v) {
            if (u.size() != v.size()) {
                d = 1e30;
                return;
            }
            for (size_t i = 0; i < u.size(); i++)
                d = std::max(d, std::abs(u[i] - v[i]) / std::max(1e-12, std::abs(v[i])));
        };
        cmp(x.pofk, y.pofk);
        cmp(x.cross, y.cross);
        cmp(x.deconvolved, y.deconvolved);
        cmp(x.p0, y.p0);
        cmp(x.p2, y.p2);
        cmp(x.p4, y.p4);
        return d;
    }

    template <class Grid, int N>
    void fill(Grid & g, int which) {
        g.fill_real_grid(std::function<FloatType(std::array<double, N> &)>(
            [which](std::array<double, N> & x) { return field<N>(x, which); }));
        g.fftw_r2c();
    }

    template <int N>
    void run(int Nmesh, FILE * out) {
        // The reference: FFTWGrid
        FFTWGrid<N> fa(Nmesh), fb(Nmesh);
        fill<FFTWGrid<N>, N>(fa, 0);
        fill<FFTWGrid<N>, N>(fb, 1);
        const auto ref = spectra<N>(fa, fb, Nmesh);
        if (out and FML::ThisTask == 0) {
            for (auto * v : {&ref.pofk, &ref.cross, &ref.deconvolved, &ref.p0, &ref.p2, &ref.p4})
                for (double x : *v)
                    std::fprintf(out, "%d %d %.17g\n", N, Nmesh, x);
        }

        // PencilGrid on slabs and (3D) on the default pencils, and every process grid
        std::vector<FML::PencilDomain> domains{FML::PencilDomain::uniform(FML::NTasks, 1)};
        if constexpr (N == 3)
            for (int p0 = 1; p0 <= FML::NTasks; p0++)
                if (FML::NTasks % p0 == 0 and p0 != FML::NTasks and Nmesh % p0 == 0 and Nmesh % (FML::NTasks / p0) == 0)
                    domains.push_back(FML::PencilDomain::uniform(p0, FML::NTasks / p0));
        for (auto & domain : domains) {
            PencilGrid<N> pa(Nmesh, 0, 0, domain), pb(Nmesh, 0, 0, domain);
            fill<PencilGrid<N>, N>(pa, 0);
            fill<PencilGrid<N>, N>(pb, 1);
            const double d = max_rel_diff(spectra<N>(pa, pb, Nmesh), ref);
            const auto P = domain.process_grid();
            check(d < 1e-10,
                  "N = " + std::to_string(N) + " Nmesh = " + std::to_string(Nmesh) + " PencilGrid " +
                      std::to_string(P[0]) + "x" + std::to_string(P[1]) +
                      ": P(k), cross, deconvolved and multipoles = FFTWGrid's (max rel. diff " + FML::TESTS::num(d, 3) +
                      ")");
        }
    }


    //==================================================================================================================
    // Particles to grid and back: a PM force calculation written once for any grid
    //==================================================================================================================
    template <int N>
    struct Part {
        double pos[N];
        long long id;
        double * get_pos() { return pos; }
        constexpr int get_ndim() const { return N; }
    };

    // The same particles on every task: uniform ones and a clump (also across the periodic boundary)
    template <int N>
    std::vector<Part<N>> all_particles(int n) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::normal_distribution<double> G(0.0, 0.04);
        std::vector<Part<N>> part(n);
        for (int i = 0; i < n; i++) {
            part[i].id = i;
            for (int d = 0; d < N; d++) {
                double x = i % 3 == 0 ? 0.97 + G(gen) : U(gen);
                x -= std::floor(x);
                part[i].pos[d] = x < 1.0 ? x : 0.0;
            }
        }
        return part;
    }

    // delta(x) and the force F = -grad phi with grad^2 phi = delta at the particles, keyed by id (on every task)
    template <int N, template <int> class Grid>
    std::map<long long, std::array<double, N + 1>>
    pm_force(Grid<N> density, FML::PARTICLE::MPIParticles<Part<N>> & particles, const std::string & method) {
        using FML::INTERPOLATION::interpolate_grid_to_particle_positions;
        using FML::INTERPOLATION::particles_to_grid;
        std::vector<std::vector<FloatType>> values(N + 1);

        // The density at the particles
        particles_to_grid<N>(particles.get_particles_ptr(),
                             particles.get_npart(),
                             particles.get_npart_total(),
                             density,
                             method);
        density.communicate_boundaries();
        interpolate_grid_to_particle_positions<N>(density,
                                                  particles.get_particles_ptr(),
                                                  particles.get_npart(),
                                                  values[0],
                                                  method);

        // The force in Fourier space: F_d(k) = -i k_d phi(k), phi(k) = -delta(k) / k^2
        density.fftw_r2c();
        for (int d = 0; d < N; d++) {
            Grid<N> force = density;
            for (FML::IndexIntType irow = 0; irow < FML::IndexIntType(force.get_local_nrows_fourier()); irow++)
                for (auto && index : force.get_fourier_range_of_rows(irow, irow + 1)) {
                    std::array<double, N> kvec;
                    double k2;
                    force.get_fourier_wavevector_and_norm2_by_index(index, kvec, k2);
                    const ComplexType phi = k2 > 0.0 ? -force.get_fourier_from_index(index) / FloatType(k2) : 0.0;
                    force.set_fourier_from_index(index, ComplexType(0, -1) * FloatType(kvec[d]) * phi);
                }
            force.fftw_c2r();
            force.communicate_boundaries();
            interpolate_grid_to_particle_positions<N>(force,
                                                      particles.get_particles_ptr(),
                                                      particles.get_npart(),
                                                      values[d + 1],
                                                      method);
        }

        // Gather (id, delta, F) on every task
        std::vector<double> local;
        for (size_t i = 0; i < particles.get_npart(); i++) {
            local.push_back(double(particles[i].id));
            for (int c = 0; c <= N; c++)
                local.push_back(values[c][i]);
        }
        int nlocal = int(local.size());
        std::vector<int> counts(FML::NTasks), displacements(FML::NTasks, 0);
        MPI_Allgather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
        for (int i = 1; i < FML::NTasks; i++)
            displacements[i] = displacements[i - 1] + counts[i - 1];
        std::vector<double> all(displacements.back() + counts.back());
        MPI_Allgatherv(local.data(),
                       nlocal,
                       MPI_DOUBLE,
                       all.data(),
                       counts.data(),
                       displacements.data(),
                       MPI_DOUBLE,
                       FML::communicator);
        std::map<long long, std::array<double, N + 1>> result;
        for (size_t i = 0; i < all.size(); i += N + 2) {
            std::array<double, N + 1> v;
            for (int c = 0; c <= N; c++)
                v[c] = all[i + 1 + c];
            result[(long long)all[i]] = v;
        }
        return result;
    }

    // The largest difference of each component relative to its largest value
    template <int N>
    double max_rel_diff(const std::map<long long, std::array<double, N + 1>> & a,
                        const std::map<long long, std::array<double, N + 1>> & b) {
        if (a.size() != b.size())
            return 1e30;
        std::array<double, N + 1> scale{}, diff{};
        for (auto & [id, v] : a) {
            auto it = b.find(id);
            if (it == b.end())
                return 1e30;
            for (int c = 0; c <= N; c++) {
                scale[c] = std::max(scale[c], std::abs(v[c]));
                diff[c] = std::max(diff[c], std::abs(v[c] - it->second[c]));
            }
        }
        double d = 0.0;
        for (int c = 0; c <= N; c++)
            d = std::max(d, diff[c] / std::max(scale[c], 1e-300));
        return d;
    }

    template <int N>
    void run_pm(int Nmesh, int npart) {
        const auto all = all_particles<N>(npart);
        std::vector<Part<N>> mine;
        for (int i = FML::ThisTask; i < npart; i += FML::NTasks)
            mine.push_back(all[i]);

        for (std::string method : {"NGP", "CIC", "TSC", "PCS"}) {
            const auto extra = FML::INTERPOLATION::get_extra_slices(method);

            // The reference: FFTWGrid with the particles on the global slabs
            FML::PARTICLE::MPIParticles<Part<N>> slab_particles;
            slab_particles.create(mine, 4 * mine.size() + 100, FML::PARTICLE::Input::Distributed);
            const auto ref = pm_force<N>(FFTWGrid<N>(Nmesh, extra.first, extra.second), slab_particles, method);

            std::vector<FML::PencilDomain> domains{FML::PencilDomain::uniform(FML::NTasks, 1)};
            if constexpr (N == 3)
                for (int p0 = 1; p0 < FML::NTasks; p0++)
                    if (FML::NTasks % p0 == 0 and Nmesh % p0 == 0 and Nmesh % (FML::NTasks / p0) == 0)
                        domains.push_back(FML::PencilDomain::uniform(p0, FML::NTasks / p0));
            for (auto & domain : domains) {
                // The particles on the pencils of the grid
                FML::PARTICLE::MPIParticles<Part<N>> particles(domain.particle_domain());
                particles.create(mine, 4 * mine.size() + 100, FML::PARTICLE::Input::Distributed);
                const auto result =
                    pm_force<N>(PencilGrid<N>(Nmesh, extra.first, extra.second, domain), particles, method);
                const double d = max_rel_diff<N>(result, ref);
                const auto P = domain.process_grid();
                check(d < 1e-10,
                      "N = " + std::to_string(N) + " " + method + " PencilGrid " + std::to_string(P[0]) + "x" +
                          std::to_string(P[1]) +
                          ": delta and the PM force at the particles = FFTWGrid's (max rel. "
                          "diff " +
                          FML::TESTS::num(d, 3) + ", " + std::to_string(result.size()) + " particles)");
            }
        }
    }

    //==================================================================================================================
    // N-body: kick-drift-kick steps with an FFTWGrid (the Nmesh version) and with a PencilGrid (the grid version)
    //==================================================================================================================
    template <int N>
    struct PartV {
        double pos[N];
        double vel[N];
        long long id;
        double * get_pos() { return pos; }
        double * get_vel() { return vel; }
        constexpr int get_ndim() const { return N; }
    };

    // (id, position, velocity) of all the particles, on every task
    template <int N>
    std::map<long long, std::array<double, 2 * N>> gather_state(FML::PARTICLE::MPIParticles<PartV<N>> & particles) {
        std::vector<double> local;
        for (auto & p : particles) {
            local.push_back(double(p.id));
            for (int d = 0; d < N; d++)
                local.push_back(p.pos[d]);
            for (int d = 0; d < N; d++)
                local.push_back(p.vel[d]);
        }
        int nlocal = int(local.size());
        std::vector<int> counts(FML::NTasks), displacements(FML::NTasks, 0);
        MPI_Allgather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
        for (int i = 1; i < FML::NTasks; i++)
            displacements[i] = displacements[i - 1] + counts[i - 1];
        std::vector<double> all(displacements.back() + counts.back());
        MPI_Allgatherv(local.data(),
                       nlocal,
                       MPI_DOUBLE,
                       all.data(),
                       counts.data(),
                       displacements.data(),
                       MPI_DOUBLE,
                       FML::communicator);
        std::map<long long, std::array<double, 2 * N>> state;
        for (size_t i = 0; i < all.size(); i += 2 * N + 1) {
            std::array<double, 2 * N> v;
            for (int c = 0; c < 2 * N; c++)
                v[c] = all[i + 1 + c];
            state[(long long)all[i]] = v;
        }
        return state;
    }

    // The largest difference of the positions (periodic) and the velocities, relative to the largest velocity
    template <int N>
    double state_diff(const std::map<long long, std::array<double, 2 * N>> & a,
                      const std::map<long long, std::array<double, 2 * N>> & b) {
        if (a.size() != b.size())
            return 1e30;
        double dpos = 0.0, dvel = 0.0, vmax = 0.0;
        for (auto & [id, v] : a) {
            auto it = b.find(id);
            if (it == b.end())
                return 1e30;
            for (int d = 0; d < N; d++) {
                double dx = std::abs(v[d] - it->second[d]);
                dpos = std::max(dpos, std::min(dx, 1.0 - dx));
                dvel = std::max(dvel, std::abs(v[N + d] - it->second[N + d]));
                vmax = std::max(vmax, std::abs(v[N + d]));
            }
        }
        return std::max(dpos, dvel / std::max(vmax, 1e-300));
    }

    template <int N>
    void run_nbody(int Nmesh, int npart, int nsteps) {
        const auto all = all_particles<N>(npart);
        std::vector<PartV<N>> mine;
        for (int i = FML::ThisTask; i < npart; i += FML::NTasks) {
            PartV<N> p{};
            for (int d = 0; d < N; d++)
                p.pos[d] = all[i].pos[d];
            p.id = all[i].id;
            mine.push_back(p);
        }
        const double dt = 0.02, norm = 1.5;
        for (std::string method : {"CIC", "TSC"}) {
            const auto extra = FML::INTERPOLATION::get_extra_slices(method);
            FML::PARTICLE::MPIParticles<PartV<N>> slab_particles;
            slab_particles.create(mine, 4 * mine.size() + 100, FML::PARTICLE::Input::Distributed);
            for (int step = 0; step < nsteps; step++)
                FML::NBODY::KickDriftKickNBodyStep<N>(Nmesh, slab_particles, dt, method, norm);
            const auto ref = gather_state<N>(slab_particles);

            std::vector<FML::PencilDomain> domains{FML::PencilDomain::uniform(FML::NTasks, 1)};
            if constexpr (N == 3)
                for (int p0 = 1; p0 < FML::NTasks; p0++)
                    if (FML::NTasks % p0 == 0 and Nmesh % p0 == 0 and Nmesh % (FML::NTasks / p0) == 0)
                        domains.push_back(FML::PencilDomain::uniform(p0, FML::NTasks / p0));
            for (auto & domain : domains) {
                FML::PARTICLE::MPIParticles<PartV<N>> particles(domain.particle_domain());
                particles.create(mine, 4 * mine.size() + 100, FML::PARTICLE::Input::Distributed);
                const PencilGrid<N> grid(Nmesh, extra.first, extra.second, domain);
                for (int step = 0; step < nsteps; step++)
                    FML::NBODY::KickDriftKickNBodyStep<N>(grid, particles, dt, method, norm);
                const double d = state_diff<N>(gather_state<N>(particles), ref);
                const auto P = domain.process_grid();
                check(d < 1e-9,
                      "N = " + std::to_string(N) + " " + method + " PencilGrid " + std::to_string(P[0]) + "x" +
                          std::to_string(P[1]) + ": " + std::to_string(nsteps) +
                          " KickDriftKick steps = FFTWGrid's (max diff " + FML::TESTS::num(d, 3) + ")");
            }
        }
    }

} // namespace

int main(int argc, char ** argv) {
    FILE * out = argc > 1 and FML::ThisTask == 0 ? std::fopen(argv[1], "w") : nullptr;
    for (int Nmesh : {12, 24}) {
        if (Nmesh % FML::NTasks != 0)
            continue;
        run<2>(Nmesh, out);
        run<3>(Nmesh, out);
    }
    if (out)
        std::fclose(out);
    if (24 % FML::NTasks == 0) {
        run_pm<2>(24, 2000);
        run_pm<3>(24, 3000);
        run_nbody<2>(24, 2000, 3);
        run_nbody<3>(24, 3000, 3);
    }
    return FML::TESTS::summary("generic grid");
}
