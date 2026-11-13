#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/Global/Global.h>

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

        auto hash = grid.global_hash();
        auto ntot = grid.get_n_global();
        if (FML::ThisTask == 0)
            std::cout << "  step " << step << " ncells " << ntot << " hash " << std::hex << hash << std::dec << "\n";
    }
    grid.info();
}

int main() {
    if (FML::ThisTask == 0)
        std::cout << "Running on " << FML::NTasks << " tasks\n";

    for (int bits = 1; bits <= 5; bits++)
        test_hilbert<2>(bits);
    for (int bits = 1; bits <= 4; bits++)
        test_hilbert<3>(bits);

    test_grid<2>(3, 8, {true, true}, 5, false);
    test_grid<2>(2, 7, {false, false}, 5, false);
    test_grid<2>(3, 7, {true, false}, 6, true);
    test_grid<3>(2, 5, {true, true, true}, 4, false);
    test_grid<3>(2, 5, {false, true, false}, 4, true);

    if (FML::ThisTask == 0)
        std::cout << (nfail == 0 ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    return nfail == 0 ? 0 : 1;
}
