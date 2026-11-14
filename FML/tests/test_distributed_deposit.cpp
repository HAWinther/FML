// Density assignment from particles on any domain (Interpolation/DistributedDeposit.h) against particles_to_grid with
// the particles in the slabs of the grid: particles on a Peano-Hilbert domain, scattered randomly over the tasks or all
// on one task; NGP-PQS, with and without masses, assignment offsets (interlacing); several passes over the patch and
// tiny messages (blocks split over messages); PencilGrids (every process grid that fits) and a hand-made box
// partition; MPIParticles on a PeanoHilbertDomain; OpenMP determinism.
#include <FML/Global/Global.h>
#include <FML/Interpolation/DistributedDeposit.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/GridTestHelpers.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <FML/tests/TestHelpers.h>

#ifdef USE_OMP
#include <omp.h>
#endif

using FML::TESTS::check;
namespace I = FML::INTERPOLATION;
namespace PH = FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT;
using namespace FML::TESTS::GRIDS;

template <int N>
struct Particle {
    double pos[N];
    double * get_pos() { return pos; }
    const double * get_pos() const { return pos; }
    constexpr int get_ndim() const { return N; }
};
template <int N>
struct MassParticle : Particle<N> {
    double mass = 1.0;
    double get_mass() const { return mass; }
};

/// The same global particles on every task: clumps, a uniform part and particles on cell and slab edges
template <class P, int N>
std::vector<P> global_particles(size_t n, int Nmesh) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> gauss(0, 0.03);
    std::vector<P> p(n);
    std::array<std::array<double, N>, 3> centers;
    for (auto & c : centers)
        for (int d = 0; d < N; d++)
            c[d] = uniform(rng);
    for (size_t i = 0; i < n; i++) {
        for (int d = 0; d < N; d++) {
            double x;
            if (i % 3 == 0)
                x = uniform(rng);
            else
                x = centers[i % 3][d] + gauss(rng);
            if (i % 11 == 0)
                x = std::floor(uniform(rng) * Nmesh) / Nmesh; // on a cell edge
            if (i % 13 == 0)
                x = std::nextafter(1.0, 0.0);
            x -= std::floor(x);
            if (x >= 1.0)
                x = std::nextafter(1.0, 0.0);
            p[i].pos[d] = x;
        }
        if constexpr (std::is_same_v<P, MassParticle<N>>)
            p[i].mass = 0.5 + uniform(rng);
    }
    return p;
}

enum class Distribution { PeanoHilbert, Random, OneTask };

/// A load-balanced Peano-Hilbert domain of the particles
template <class P>
FML::Domain peano_hilbert_domain(const std::vector<P> & all) {
    const unsigned bits = 10;
    std::vector<PH::Key> keys;
    for (size_t i = FML::ThisTask; i < all.size(); i += FML::NTasks)
        keys.push_back(PH::key_from_position({all[i].pos[0], all[i].pos[1], all[i].pos[2]}, bits));
    return FML::Domain::peano_hilbert(FML::PeanoHilbertDomain::build(keys, bits, 12));
}

/// The particles of this task in a distribution
template <class P, int N>
std::vector<P> distribute(const std::vector<P> & all, Distribution how) {
    std::vector<P> mine;
    if (how == Distribution::PeanoHilbert) {
        if constexpr (N == 3) {
            const auto domain = peano_hilbert_domain(all);
            for (auto & p : all)
                if (domain.owner(p.pos) == FML::ThisTask)
                    mine.push_back(p);
            return mine;
        }
        how = Distribution::Random;
    }
    for (size_t i = 0; i < all.size(); i++) {
        const int owner = how == Distribution::OneTask ? FML::NTasks - 1 : int((i * 2654435761u) % FML::NTasks);
        if (owner == FML::ThisTask)
            mine.push_back(all[i]);
    }
    return mine;
}

/// The particles of this task's slab of the grid (by cell, as particles_to_grid expects)
template <class P, int N>
std::vector<P> slab_particles(const std::vector<P> & all, const FML::GRID::FFTWGrid<N> & grid) {
    std::vector<P> mine;
    const int Nmesh = grid.get_nmesh();
    const long long x0 = grid.get_local_x_start(), nx = grid.get_local_nx();
    for (auto & p : all) {
        const long long ix = std::min<long long>(Nmesh - 1, (long long)(p.pos[0] * Nmesh));
        if (ix >= x0 and ix < x0 + nx)
            mine.push_back(p);
    }
    return mine;
}

/// The largest difference in the local main slices
template <int N>
double max_difference(const FML::GRID::FFTWGrid<N> & a, const FML::GRID::FFTWGrid<N> & b) {
    double err = 0;
    for (auto i : a.get_real_range())
        err = std::max(err, double(std::fabs(a.get_real_from_index(i) - b.get_real_from_index(i))));
    FML::MaxOverTasks(&err);
    return err;
}

template <int N, int ORDER, class P>
void compare_with_slab_deposit(const std::vector<P> & all,
                               int Nmesh,
                               Distribution how,
                               const std::string & label,
                               const std::array<double, N> & offset = {},
                               I::DepositOptions options = {},
                               int * passes = nullptr) {
    // Reference: particles_to_grid with the particles in the slabs (and the extra slices it needs)
    auto extra = I::get_extra_slices_needed_by_order<ORDER>();
    if (offset[0] != 0.0)
        extra.second += 1;
    FML::GRID::FFTWGrid<N> reference(Nmesh, extra.first, extra.second);
    const auto slab = slab_particles<P, N>(all, reference);
    I::particles_to_grid<N, ORDER, P>(slab.data(), slab.size(), all.size(), reference, offset);

    FML::GRID::FFTWGrid<N> density(Nmesh);
    const auto mine = distribute<P, N>(all, how);
    const auto info = I::particles_to_grid_any_domain<N, ORDER, P>(mine.data(), mine.size(), all.size(), density, offset,
                                                                   options);
    if (passes) {
        *passes = info.passes;
        FML::MaxOverTasks(passes);
    }
    double scale = 0;
    for (auto i : reference.get_real_range())
        scale = std::max(scale, double(std::fabs(reference.get_real_from_index(i))));
    FML::MaxOverTasks(&scale);
    const double err = max_difference(density, reference) / scale;
    char text[64];
    std::snprintf(text, sizeof text, " (relative difference %.1e)", err);
    check(err < 1e-13, label + text);
}

/// PencilGrids (every P0 x P1 process grid that divides the mesh): the field equals the slab one, from particles on a
/// Peano-Hilbert domain in MPIParticles; and deposit_to_partitioned_mesh with a hand-made box partition into a plain
/// array
void test_pencils() {
    constexpr int N = 3, ORDER = 3;
    const int Nmesh = 12;
    using P = MassParticle<N>;
    const auto all = global_particles<P, N>(4000, Nmesh);

    // MPIParticles on the Peano-Hilbert domain
    auto copy = all;
    FML::PARTICLE::MPIParticles<P> mp(peano_hilbert_domain(all));
    mp.create(copy, 2 * all.size(), FML::PARTICLE::Input::Replicated);

    // Reference: the slab result
    FML::GRID::FFTWGrid<N> slab(Nmesh);
    I::particles_to_grid_any_domain<N>(mp, slab, "TSC");
    const auto full_slab = global_real(slab);
    double scale = 0;
    for (double v : full_slab)
        scale = std::max(scale, std::fabs(v));

    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        const int p1 = FML::NTasks / p0;
        if (FML::NTasks % p0 or Nmesh % p0 or Nmesh % p1)
            continue;
        FML::GRID::PencilGrid<N> pencils(Nmesh, 0, 0, FML::PencilDomain::uniform(p0, p1));
        I::particles_to_grid_any_domain<N>(mp, pencils, "TSC");
        const auto full = global_real(pencils);
        double err = 0;
        for (size_t i = 0; i < full.size(); i++)
            err = std::max(err, std::fabs(full[i] - full_slab[i]));
        char text[128];
        std::snprintf(text, sizeof text, "PencilGrid %dx%d = slabs, MPIParticles on Peano-Hilbert (relative difference %.1e)",
                      p0, p1, err / scale);
        check(err / scale < 1e-13, text);
    }

    // deposit_to_partitioned_mesh with boxes split in x and y into a dense local array starting at -1
    int P0 = 1;
    for (int p = 1; p * p <= FML::NTasks; p++)
        if (FML::NTasks % p == 0)
            P0 = p;
    const int P1 = FML::NTasks / P0;
    auto split = [&](int parts, int part) {
        const int base = Nmesh / parts, rest = Nmesh % parts;
        return std::make_pair(part * base + std::min(part, rest), base + (part < rest ? 1 : 0));
    };
    I::MeshPartition<N> partition;
    partition.mesh.fill(Nmesh);
    partition.box.resize(FML::NTasks);
    for (int r = 0; r < FML::NTasks; r++) {
        // (a permuted rank order)
        const int px = (FML::NTasks - 1 - r) / P1, py = (FML::NTasks - 1 - r) % P1;
        partition.box[r].start.fill(0);
        partition.box[r].count.fill(Nmesh);
        std::tie(partition.box[r].start[0], partition.box[r].count[0]) = split(P0, px);
        std::tie(partition.box[r].start[1], partition.box[r].count[1]) = split(P1, py);
    }
    const auto & box = partition.box[FML::ThisTask];
    std::vector<double> local(box.volume(), -1.0);
    double mean_mass = 0;
    for (auto & p : all)
        mean_mass += p.mass;
    mean_mass /= all.size();
    const double factor = std::pow(Nmesh, 3) / double(all.size()) / mean_mass;
    auto add_local = [&](const std::array<int, N> & cell, double value) {
        const size_t i = (size_t(cell[0] - box.start[0]) * box.count[1] + (cell[1] - box.start[1])) * box.count[2] +
                         (cell[2] - box.start[2]);
        local[i] += value;
    };
    I::deposit_to_partitioned_mesh<N, ORDER, P>(mp.get_particles_ptr(), mp.get_npart(), partition, factor, add_local);
    double err = 0;
    for (int ix = 0; ix < box.count[0]; ix++)
        for (int iy = 0; iy < box.count[1]; iy++)
            for (int iz = 0; iz < box.count[2]; iz++) {
                const size_t g = (size_t(ix + box.start[0]) * Nmesh + iy + box.start[1]) * Nmesh + iz + box.start[2];
                err = std::max(err, std::fabs(local[(size_t(ix) * box.count[1] + iy) * box.count[2] + iz] - full_slab[g]));
            }
    FML::MaxOverTasks(&err);
    char text[128];
    std::snprintf(text, sizeof text, "%dx%d boxes into a plain array = slabs (relative difference %.1e)", P0, P1,
                  err / scale);
    check(err / scale < 1e-13, text);
}

void test_threads() {
#ifdef USE_OMP
    constexpr int N = 3, ORDER = 4;
    const int Nmesh = 48;
    using P = MassParticle<N>;
    const auto all = global_particles<P, N>(20000, Nmesh);
    const auto mine = distribute<P, N>(all, Distribution::PeanoHilbert);
    std::array<FML::GRID::FFTWGrid<N>, 3> grids{FML::GRID::FFTWGrid<N>(Nmesh), FML::GRID::FFTWGrid<N>(Nmesh),
                                                FML::GRID::FFTWGrid<N>(Nmesh)};
    const int threads[3] = {1, 2, 3};
    const int saved = omp_get_max_threads();
    for (int t = 0; t < 3; t++) {
        omp_set_num_threads(threads[t]);
        I::particles_to_grid_any_domain<N, ORDER, P>(mine.data(), mine.size(), all.size(), grids[t]);
    }
    omp_set_num_threads(saved);
    check(max_difference(grids[1], grids[2]) == 0.0, "OpenMP: 2 and 3 threads give identical grids");
    double scale = 0;
    for (auto i : grids[0].get_real_range())
        scale = std::max(scale, double(std::fabs(grids[0].get_real_from_index(i))));
    FML::MaxOverTasks(&scale);
    const double err = max_difference(grids[0], grids[1]) / scale;
    char text[128];
    std::snprintf(text, sizeof text, "OpenMP: 1 and 2 threads agree to round-off (relative difference %.1e)", err);
    check(err < 1e-13, text);
#endif
}

int main() {
    const int Nmesh = 24;
    {
        constexpr int N = 3;
        using P = Particle<N>;
        using PM = MassParticle<N>;
        const auto all = global_particles<P, N>(5000, Nmesh);
        const auto allm = global_particles<PM, N>(5000, Nmesh);
        compare_with_slab_deposit<N, 1, P>(all, Nmesh, Distribution::PeanoHilbert, "NGP, Peano-Hilbert domain");
        compare_with_slab_deposit<N, 2, P>(all, Nmesh, Distribution::PeanoHilbert, "CIC, Peano-Hilbert domain");
        compare_with_slab_deposit<N, 3, PM>(allm, Nmesh, Distribution::PeanoHilbert, "TSC with masses, Peano-Hilbert");
        compare_with_slab_deposit<N, 4, PM>(allm, Nmesh, Distribution::Random, "PCS with masses, random tasks");
        compare_with_slab_deposit<N, 5, P>(all, Nmesh, Distribution::OneTask, "PQS, all particles on one task");
        const double h = 0.5 / Nmesh;
        compare_with_slab_deposit<N, 2, PM>(allm, Nmesh, Distribution::PeanoHilbert,
                                            "CIC, interlacing offset (h, h, h)", {h, h, h});
        compare_with_slab_deposit<N, 3, P>(all, Nmesh, Distribution::Random, "TSC, offset (0, h, 0.9/Nmesh)",
                                           {0.0, h, 0.9 / Nmesh});
        // One slice per pass and messages of 3 values: blocks split over many messages
        I::DepositOptions tiny;
        tiny.max_patch_cells = 1;
        tiny.max_message_bytes = 3 * sizeof(double);
        int passes = 0;
        compare_with_slab_deposit<N, 4, PM>(allm, Nmesh, Distribution::Random, "PCS, one slice per pass, tiny messages",
                                            {}, tiny, &passes);
        check(passes > 1, "the small patch budget gives several passes (" + std::to_string(passes) + ")");
        I::DepositOptions nozero;
        nozero.skip_zero_blocks = false;
        compare_with_slab_deposit<N, 2, P>(all, Nmesh, Distribution::PeanoHilbert, "CIC, zero blocks sent too", {},
                                           nozero);
        // A compact domain has a small patch: with Peano-Hilbert domains the bounding box of a task is not the box
        {
            const auto mine = distribute<P, N>(all, Distribution::PeanoHilbert);
            FML::GRID::FFTWGrid<N> g(Nmesh);
            const auto info = I::particles_to_grid_any_domain<N, 2, P>(mine.data(), mine.size(), all.size(), g);
            double fraction = double(info.patch_count[0]) * info.patch_count[1] * info.patch_count[2] / std::pow(Nmesh, 3);
            FML::MaxOverTasks(&fraction);
            if (FML::ThisTask == 0)
                std::cout << "       largest patch / mesh with Peano-Hilbert domains: " << fraction << "\n";
        }
    }
    {
        constexpr int N = 2;
        using P = Particle<N>;
        const auto all = global_particles<P, N>(3000, Nmesh);
        compare_with_slab_deposit<N, 3, P>(all, Nmesh, Distribution::Random, "2D TSC, random tasks");
        compare_with_slab_deposit<N, 2, P>(all, Nmesh, Distribution::OneTask, "2D CIC, one task, offset",
                                           {0.5 / Nmesh, 0.5 / Nmesh});
    }
    test_pencils();
    test_threads();
    return FML::TESTS::summary("distributed deposit");
}
