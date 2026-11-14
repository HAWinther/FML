#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>

#include <cstdlib>
#include <map>
#include <random>
#include <set>

//==========================================================================================
// Tests for HilbertCurve and AdaptiveMPIGrid
// Run with different numbers of tasks: the printed global hashes must be identical.
//==========================================================================================

using namespace FML::GRID;

static int nfail = 0;
#define CHECK(expr, msg)                                                                                               \
    do {                                                                                                               \
        if (not(expr)) {                                                                                               \
            if (FML::ThisTask == 0)                                                                                    \
                std::cout << "  FAIL: " << msg << "\n";                                                                \
            nfail++;                                                                                                   \
        }                                                                                                              \
    } while (0)

//==========================================================================================
// Hilbert curve: bijection, adjacency of consecutive keys, hierarchy
//==========================================================================================
template <int NDIM>
void test_hilbert(int bits) {
    using HC = HilbertCurve<NDIM>;
    const uint64_t N = uint64_t(1) << (NDIM * bits);
    bool ok_bij = true, ok_adj = true, ok_hier = true;
    std::vector<char> seen(N, 0);
    typename HC::Coord prev{};
    for (uint64_t k = 0; k < N; k++) {
        auto x = HC::coord_from_key(k, bits);
        if (HC::key_from_coord(x, bits) != k)
            ok_bij = false;
        uint64_t lin = 0;
        for (int d = 0; d < NDIM; d++)
            lin = (lin << bits) | x[d];
        if (seen[lin]++)
            ok_bij = false;
        if (k > 0) {
            int dist = 0;
            for (int d = 0; d < NDIM; d++)
                dist += std::abs(int(x[d]) - int(prev[d]));
            if (dist != 1)
                ok_adj = false;
        }
        // Hierarchy: the prefix at every level identifies the aligned cell
        for (int l = 1; l < bits; l++) {
            int shift = NDIM * (bits - l);
            auto xc = HC::coord_from_key((k >> shift) << shift, bits);
            for (int d = 0; d < NDIM; d++)
                if ((xc[d] >> (bits - l)) != (x[d] >> (bits - l)))
                    ok_hier = false;
        }
        prev = x;
    }
    CHECK(ok_bij, "hilbert bijection NDIM=" << NDIM << " bits=" << bits);
    CHECK(ok_adj, "hilbert adjacency NDIM=" << NDIM << " bits=" << bits);
    CHECK(ok_hier, "hilbert hierarchy NDIM=" << NDIM << " bits=" << bits);

    // Random round trips at full depth
    std::mt19937_64 rng(1234);
    bool ok_rt = true;
    for (int i = 0; i < 100000; i++) {
        typename HC::Coord x;
        for (int d = 0; d < NDIM; d++)
            x[d] = uint32_t(rng() & ((uint64_t(1) << HC::LMAX) - 1));
        if (HC::coord_from_key(HC::key_from_coord(x)) != x)
            ok_rt = false;
    }
    CHECK(ok_rt, "hilbert round trip at LMAX NDIM=" << NDIM);
}

//==========================================================================================
// Grid tests
//==========================================================================================
struct TestData {
    uint64_t key;
    int level;
    double value;
};

// Deterministic per-cell random number so flags are independent of the number of tasks
inline double cell_random(uint64_t key, int level, int step) {
    uint64_t h = FML::GRID::AMRDETAIL::splitmix64(key ^ FML::GRID::AMRDETAIL::splitmix64(level * 1000 + step));
    return (h >> 11) * (1.0 / 9007199254740992.0);
}

template <int NDIM>
bool check_ghost_values(AdaptiveMPIGrid<NDIM, TestData> & grid) {
    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
        auto & c = grid.get_cell(i);
        grid.get_data(i) = TestData{c.key, c.level, double(c.key % 1000003)};
    }
    grid.exchange_ghosts();
    bool ok = true;
    for (IndexInt i = 0; i < grid.get_n_ghost(); i++) {
        auto & c = grid.get_ghost_cell(i);
        auto & v = grid.get_ghost_data(i);
        if (v.key != c.key or v.level != c.level or v.value != double(c.key % 1000003))
            ok = false;
    }
    return not FML::GRID::AMRDETAIL::allreduce_or(not ok);
}

template <int NDIM>
bool check_face_neighbors(const AdaptiveMPIGrid<NDIM, TestData> & grid) {
    using FN = typename AdaptiveMPIGrid<NDIM, TestData>::FaceNeighbors;
    bool ok = true;
    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
        auto & c = grid.get_cell(i);
        for (int dim = 0; dim < NDIM; dim++) {
            for (int sign = -1; sign <= 1; sign += 2) {
                auto nb = grid.get_face_neighbors(i, dim, sign);
                const uint64_t h = AdaptiveMPIGrid<NDIM, TestData>::size_int(c.level);
                const bool at_edge = sign > 0 ? (uint64_t(c.anchor[dim]) + h == (uint64_t(1) << AdaptiveMPIGrid<NDIM, TestData>::LMAX))
                                              : (c.anchor[dim] == 0);
                if (nb.relation == FN::BOUNDARY) {
                    if (not(at_edge and not grid.get_periodic()[dim]))
                        ok = false;
                    continue;
                }
                // Covered face area (in units of the finest face) must equal the face of the cell
                double area = 0.0;
                for (int j = 0; j < nb.n; j++) {
                    if (not nb.cells[j]) {
                        ok = false;
                        continue;
                    }
                    int l = grid.get_cell(nb.cells[j]).level;
                    int expect = c.level + (nb.relation == FN::FINER ? 1 : (nb.relation == FN::COARSER ? -1 : 0));
                    if (l != expect)
                        ok = false;
                    area += std::pow(std::ldexp(1.0, -std::max(l, int(c.level))), NDIM - 1);
                }
                if (std::abs(area - std::pow(std::ldexp(1.0, -c.level), NDIM - 1)) > 1e-14)
                    ok = false;
            }
        }
    }
    return not FML::GRID::AMRDETAIL::allreduce_or(not ok);
}

// Stencil (sum over face neighbours) computed with a blocking exchange vs. overlapped with the
// exchange (interior first, then boundary). Also two exchanges in flight at once, finished in
// reverse order, one of them via test().
template <int NDIM>
bool check_overlap(AdaptiveMPIGrid<NDIM, TestData> & grid) {
    const IndexInt n = grid.get_n_local();
    auto stencil = [&](IndexInt i) {
        double s = 0.0;
        for (int dim = 0; dim < NDIM; dim++)
            for (int sign = -1; sign <= 1; sign += 2) {
                auto nb = grid.get_face_neighbors(i, dim, sign);
                for (int j = 0; j < nb.n; j++)
                    s += grid.get_data(nb.cells[j]).value * (j + 1) * (dim + 2) * (sign + 3);
            }
        return s;
    };
    auto set_values = [&](double offset) {
        for (IndexInt i = 0; i < n; i++) {
            auto & c = grid.get_cell(i);
            grid.get_data(i) = TestData{c.key, c.level, offset + double(c.key % 99991) + c.level};
        }
    };

    bool ok = true;

    // Ghosts first hold stale values (offset 1). Then new values (offset 2) are sent while the
    // interior is computed: if an interior cell used a ghost, its result would be wrong.
    set_values(1.0);
    grid.exchange_ghosts();
    set_values(2.0);
    std::vector<double> res(n);
    {
        auto h = grid.begin_exchange_ghosts();
        for (auto i : grid.get_interior_cells())
            res[i] = stencil(i);
        h.wait();
        for (auto i : grid.get_boundary_cells())
            res[i] = stencil(i);
    }
    grid.exchange_ghosts(); // reference: blocking exchange of the same values
    for (IndexInt i = 0; i < n; i++)
        if (res[i] != stencil(i))
            ok = false;

    // Two exchanges in flight at once, finished in reverse order
    std::vector<int64_t> a(n), ga;
    std::vector<int8_t> b(n), gb;
    for (IndexInt i = 0; i < n; i++) {
        a[i] = int64_t(grid.get_cell(i).key);
        b[i] = int8_t(grid.get_cell(i).level);
    }
    auto ha = grid.begin_communicate_ghosts(a, ga);
    auto hb = grid.begin_communicate_ghosts(b, gb);
    std::fill(a.begin(), a.end(), -1); // allowed: values were packed at begin
    while (not hb.test()) {
    }
    ha.wait();
    for (IndexInt i = 0; i < grid.get_n_ghost(); i++)
        if (ga[i] != int64_t(grid.get_ghost_cell(i).key) or gb[i] != grid.get_ghost_cell(i).level)
            ok = false;
    return not FML::GRID::AMRDETAIL::allreduce_or(not ok);
}

// Truncated grids (every leaf finer than L replaced by its ancestor, uniform below levelmin) and the transfer
// between a grid and its truncation: restriction conserves the volume integral, gather gives every leaf the
// value of the coarse leaf containing it. Returns a hash of all the truncated leaf sets.
template <int NDIM>
uint64_t check_truncation(const AdaptiveMPIGrid<NDIM, TestData> & grid, bool & ok_out) {
    using Coarse = AdaptiveMPIGrid<NDIM, double>;
    const int lmin_trunc = 2;
    bool ok = true;
    uint64_t hash = 0;
    const IndexInt n = grid.get_n_local();
    std::vector<double> f(n);
    double integral = 0.0;
    for (IndexInt i = 0; i < n; i++) {
        auto & c = grid.get_cell(i);
        f[i] = double(c.key % 1000003) + 0.25 * c.level;
        integral += std::ldexp(f[i], -NDIM * c.level);
    }
    FML::SumOverTasks(&integral);
    for (int L = grid.get_levelmax(); L >= lmin_trunc; L--) {
        Coarse coarse = grid.template truncated<double>(L);
        ok = ok and coarse.check_invariants();
        hash = hash * 31 + coarse.global_hash();
        for (IndexInt j = 0; j < coarse.get_n_local(); j++)
            ok = ok and coarse.get_cell(j).level <= L and coarse.get_cell(j).level >= std::min(L, grid.get_levelmin());

        AdaptiveMPIGridTransfer<NDIM> transfer(grid, coarse);
        std::vector<double> fc;
        transfer.restrict_average(f, fc);
        double integral_c = 0.0;
        for (IndexInt j = 0; j < coarse.get_n_local(); j++)
            integral_c += std::ldexp(fc[j], -NDIM * coarse.get_cell(j).level);
        FML::SumOverTasks(&integral_c);
        ok = ok and std::abs(integral_c - integral) < 1e-9 * std::abs(integral);

        std::vector<uint64_t> ckey(coarse.get_n_local()), fkey;
        for (IndexInt j = 0; j < coarse.get_n_local(); j++)
            ckey[j] = coarse.get_cell(j).key;
        transfer.gather_from_coarse(ckey, fkey);
        for (IndexInt i = 0; i < n; i++) {
            auto & c = grid.get_cell(i);
            const uint64_t expect = c.level > L ? (c.key & ~(Coarse::key_range(L) - 1)) : c.key;
            ok = ok and fkey[i] == expect;
        }
    }
    ok_out = not FML::GRID::AMRDETAIL::allreduce_or(not ok);
    return hash;
}

template <int NDIM>
void test_grid(int levelmin, int levelmax, std::array<bool, NDIM> periodic, int nsteps, bool random_flags) {
    if (FML::ThisTask == 0) {
        std::cout << "Grid test NDIM=" << NDIM << " levels " << levelmin << "-" << levelmax << " periodic=(";
        for (auto p : periodic)
            std::cout << p;
        std::cout << ") " << (random_flags ? "random flags" : "moving shell") << "\n";
    }

    AdaptiveMPIGrid<NDIM, TestData> grid(levelmin, levelmax, periodic);
    CHECK(grid.check_invariants(), "invariants initial");
    CHECK(check_ghost_values(grid), "ghost values initial");

    for (int step = 0; step < nsteps; step++) {
        // Shell around a sphere moving diagonally (crosses periodic boundaries)
        std::array<double, NDIM> center;
        for (int d = 0; d < NDIM; d++)
            center[d] = std::fmod(0.1 + 0.13 * step * (d + 1), 1.0);
        const double R = 0.3;

        for (IndexInt i = 0; i < grid.get_n_local(); i++) {
            auto & c = grid.get_cell(i);
            int8_t flag;
            if (random_flags) {
                double u = cell_random(c.key, c.level, step);
                flag = u < 0.25 ? 1 : (u < 0.85 ? -1 : 0);
            } else {
                auto x = grid.get_cell_center(c);
                double r2 = 0.0;
                for (int d = 0; d < NDIM; d++) {
                    double dx = x[d] - center[d];
                    if (periodic[d]) {
                        if (dx > 0.5)
                            dx -= 1.0;
                        if (dx < -0.5)
                            dx += 1.0;
                    }
                    r2 += dx * dx;
                }
                double h = grid.get_cell_size(c.level);
                flag = std::abs(std::sqrt(r2) - R) < 1.5 * h ? 1 : -1;
            }
            grid.set_flag(i, flag);
        }
        grid.adapt();
        CHECK(grid.check_invariants(), "invariants after adapt step " << step);
        CHECK(check_ghost_values(grid), "ghost values after adapt step " << step);

        // Load balance with a weight that depends on level (deeper cells cost more)
        grid.load_balance([&](IndexInt i) { return 1.0 + 0.5 * grid.get_cell(i).level; });
        CHECK(grid.check_invariants(), "invariants after load balance step " << step);
        CHECK(check_ghost_values(grid), "ghost values after load balance step " << step);
        CHECK(check_face_neighbors(grid), "face neighbours step " << step);
        CHECK(check_overlap(grid), "overlapped ghost exchange step " << step);
        bool ok_trunc = false;
        const auto hash_trunc = check_truncation(grid, ok_trunc);
        CHECK(ok_trunc, "truncated grids and transfer step " << step);

        auto hash = grid.global_hash();
        auto ntot = grid.get_n_global();
        if (FML::ThisTask == 0)
            std::cout << "  step " << step << " ncells " << ntot << " hash " << std::hex << hash << " truncated "
                      << hash_trunc << std::dec << "\n";
    }
    grid.info();
}

//==========================================================================================
// The grid's decomposition as an FML::Domain, and particles that follow it through load balancing
//==========================================================================================
struct FollowParticle {
    double pos[3];
    long long id;
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};

void test_domain_and_particles() {
    if (FML::ThisTask == 0)
        std::cout << "Domain test: get_domain and MPIParticles::set_domain through load balancing\n";
    using Grid = AdaptiveMPIGrid<3, TestData>;
    Grid grid(2, 5, true);

    // Particles: each task makes its own (different) random ones, then they follow the grid
    const long long npart_per_task = 2000;
    std::mt19937 gen(1234 + FML::ThisTask);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<FollowParticle> part;
    for (long long i = 0; i < npart_per_task; i++) {
        // Clustered around (0.3, 0.6, 0.2) half of the time, so the load balance moves the boundaries a lot
        FollowParticle p{{U(gen), U(gen), U(gen)}, FML::ThisTask * npart_per_task + i};
        if (i % 2 == 0)
            for (int d = 0; d < 3; d++)
                p.pos[d] = std::fmod(std::array<double, 3>{0.3, 0.6, 0.2}[d] + 0.05 * U(gen) + 1.0, 1.0);
        part.push_back(p);
    }
    const long long ntot = npart_per_task * FML::NTasks;
    const long long idsum0 = ntot * (ntot - 1) / 2;
    FML::PARTICLE::MPIParticles<FollowParticle> particles;
    particles.create(part, 4 * npart_per_task, FML::PARTICLE::Input::Distributed);

    for (int step = 0; step < 3; step++) {
        // Refine where the particles are (the cells holding particles cost more in the load balance)
        for (IndexInt i = 0; i < grid.get_n_local(); i++) {
            const auto x = grid.get_cell_center(grid.get_cell(i));
            double r2 = 0.0;
            for (int d = 0; d < 3; d++) {
                double dx = x[d] - std::array<double, 3>{0.325, 0.625, 0.225}[d];
                r2 += dx * dx;
            }
            grid.set_flag(i, r2 < 0.04 ? 1 : 0);
        }
        grid.adapt();
        grid.load_balance([&](IndexInt i) { return 1.0 + std::pow(2.0, grid.get_cell(i).level); });

        // The domain owns the cells of the tasks: the corner and the center of every local leaf
        const FML::Domain domain = grid.get_domain();
        bool ok = domain.holds<FML::PeanoHilbertDomain>() and domain.ntasks() == FML::NTasks;
        for (IndexInt i = 0; i < grid.get_n_local(); i++) {
            const auto & c = grid.get_cell(i);
            ok = ok and domain.owner(grid.get_cell_center(c).data()) == FML::ThisTask and
                 domain.owner(grid.get_cell_corner(c).data()) == FML::ThisTask;
        }
        // ... and the same owner as the grid for any point
        std::mt19937 gen_points(99 + step);
        for (int i = 0; i < 1000; i++) {
            std::array<double, 3> x{U(gen_points), U(gen_points), U(gen_points)};
            Grid::Coord coord;
            for (int d = 0; d < 3; d++)
                coord[d] = Grid::coord_type(std::ldexp(x[d], Grid::LMAX));
            ok = ok and domain.owner(x.data()) == grid.owner_of_key(Grid::HC::key_from_coord(coord));
        }
        CHECK(not FML::GRID::AMRDETAIL::allreduce_or(not ok),
              "get_domain: owns the local leaves and agrees with owner_of_key, step " << step);

        // The particles follow: every particle is in a local leaf, none lost
        particles.set_domain(domain);
        ok = particles.get_domain() == domain;
        long long n = 0, idsum = 0;
        for (auto & p : particles) {
            std::array<double, 3> x{p.pos[0], p.pos[1], p.pos[2]};
            ok = ok and grid.find_leaf(x).is_local();
            n++;
            idsum += p.id;
        }
        FML::SumOverTasks(&n);
        FML::SumOverTasks(&idsum);
        CHECK(not FML::GRID::AMRDETAIL::allreduce_or(not ok) and n == ntot and idsum == idsum0,
              "MPIParticles::set_domain: every particle in a local leaf, none lost, step " << step);
        const auto ncells = grid.get_n_global();
        if (FML::ThisTask == 0)
            std::cout << "  step " << step << " ncells " << ncells << "\n";
    }

    // A grid made on a given domain: the load-balanced Peano-Hilbert domain of the particles (at levelmin bits, so
    // the boundaries fall on cells at levelmin)
    const int levelmin = 3;
    std::vector<FML::PeanoHilbertDomain::Key> keys;
    for (auto & p : particles)
        keys.push_back(FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT::key_from_position({p.pos[0], p.pos[1], p.pos[2]},
                                                                                     levelmin));
    const auto domain = FML::Domain::peano_hilbert(FML::PeanoHilbertDomain::build(keys, levelmin, 3 * levelmin));
    Grid grid2(levelmin, 5, true, domain);
    bool ok = grid2.check_invariants();
    for (IndexInt i = 0; i < grid2.get_n_local(); i++)
        ok = ok and domain.owner(grid2.get_cell_center(grid2.get_cell(i)).data()) == FML::ThisTask;
    std::mt19937 gen_points(7);
    for (int i = 0; i < 1000; i++) {
        std::array<double, 3> x{U(gen_points), U(gen_points), U(gen_points)};
        ok = ok and grid2.get_domain().owner(x.data()) == domain.owner(x.data());
    }
    particles.set_domain(domain);
    long long n = 0;
    for (auto & p : particles) {
        ok = ok and grid2.find_leaf(std::array<double, 3>{p.pos[0], p.pos[1], p.pos[2]}).is_local();
        n++;
    }
    FML::SumOverTasks(&n);
    CHECK(not FML::GRID::AMRDETAIL::allreduce_or(not ok) and n == ntot,
          "a grid on a given domain (the particles' load-balanced Peano-Hilbert domain): ownership and particles");
    const auto ncells2 = grid2.get_n_global();
    IndexInt nmin = grid2.get_n_local(), nmax = grid2.get_n_local();
    FML::MinOverTasks(&nmin);
    FML::MaxOverTasks(&nmax);
    if (FML::ThisTask == 0)
        std::cout << "  grid on the particles' domain: ncells " << ncells2 << ", per task " << nmin << " - " << nmax
                  << "\n";
}

int main(int argc, char ** argv) {
    // "./amrtest nbx" forces the NBX sparse exchange, "./amrtest dense" the Alltoall variant
    const std::string mode = argc > 1 ? argv[1] : "default";
    if (mode == "nbx") {
        AdaptiveMPIGrid<2, TestData>::nbx_min_tasks = 0;
        AdaptiveMPIGrid<3, TestData>::nbx_min_tasks = 0;
    } else if (mode == "dense") {
        AdaptiveMPIGrid<2, TestData>::nbx_min_tasks = 1 << 30;
        AdaptiveMPIGrid<3, TestData>::nbx_min_tasks = 1 << 30;
    }
    if (FML::ThisTask == 0)
        std::cout << "Running on " << FML::NTasks << " tasks (exchange mode: " << mode << ")\n";

    for (int bits = 1; bits <= 5; bits++)
        test_hilbert<2>(bits);
    for (int bits = 1; bits <= 4; bits++)
        test_hilbert<3>(bits);

    test_grid<2>(3, 8, {true, true}, 5, false);
    test_grid<2>(2, 7, {false, false}, 5, false);
    test_grid<2>(3, 7, {true, false}, 6, true);
    test_grid<3>(2, 5, {true, true, true}, 4, false);
    test_grid<3>(2, 5, {false, true, false}, 4, true);
    test_domain_and_particles();

    if (FML::ThisTask == 0)
        std::cout << (nfail == 0 ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    return nfail == 0 ? 0 : 1;
}
