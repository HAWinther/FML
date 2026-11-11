#include <FML/PeanoHilbertDecomposition/DistributedGrid3D.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT;

namespace {
    int rank = 0;
    int failures = 0;
    size_t checks = 0;

    void check(bool condition, const std::string & message) {
        ++checks;
        if (!condition) {
            ++failures;
            if (failures <= 20)
                std::cerr << "rank " << rank << " failed: " << message << "\n";
        }
    }

    double field_value(const std::array<uint32_t, 3> & c, double offset = 0.0) {
        return offset + double(c[0] + 10 * c[1] + 100 * c[2]);
    }

    template <class Grid>
    std::vector<Key> reference_halo(const Grid & grid, uint32_t width, HaloConnectivity connectivity,
                                   bool periodic) {
        const int64_t n = grid.cells_per_dimension();
        std::vector<Key> result;
        for (size_t local = 0; local < grid.local_size(); ++local) {
            const auto center = grid.local_coordinate(local);
            for (int64_t dx = -int64_t(width); dx <= int64_t(width); ++dx)
                for (int64_t dy = -int64_t(width); dy <= int64_t(width); ++dy)
                    for (int64_t dz = -int64_t(width); dz <= int64_t(width); ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0)
                            continue;
                        const int nchanged = int(dx != 0) + int(dy != 0) + int(dz != 0);
                        if (connectivity == HaloConnectivity::Faces && nchanged != 1)
                            continue;
                        int64_t c[3] = {int64_t(center[0]) + dx, int64_t(center[1]) + dy,
                                        int64_t(center[2]) + dz};
                        bool inside = true;
                        for (int axis = 0; axis < 3; ++axis) {
                            if (periodic) {
                                c[axis] %= n;
                                if (c[axis] < 0)
                                    c[axis] += n;
                            } else if (c[axis] < 0 || c[axis] >= n) {
                                inside = false;
                            }
                        }
                        if (!inside)
                            continue;
                        const Key key = key_from_grid_coordinate(
                            {uint32_t(c[0]), uint32_t(c[1]), uint32_t(c[2])}, grid.decomposition().bits());
                        if (grid.owner(uint32_t(c[0]), uint32_t(c[1]), uint32_t(c[2])) != grid.rank())
                            result.push_back(key);
                    }
        }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }

    template <class Grid>
    void check_ghost_values(const Grid & grid, double offset, const std::string & label) {
        for (const Key key : grid.halo_keys()) {
            const auto c = grid_coordinate_from_key(key, grid.decomposition().bits());
            check(std::abs(grid.at(c[0], c[1], c[2]) - field_value(c, offset)) < 1.0e-12,
                  label + " value for ghost key " + std::to_string(key));
        }
    }
} // namespace

int main(int argc, char ** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
#else
    (void)argc;
    (void)argv;
    const int comm = 0;
    const int nranks = 1;
#endif

    // Exhaustive small-grid checks: bijection, inverse mapping and curve adjacency.
    for (unsigned bits = 1; bits <= 4; ++bits) {
        const uint32_t n = uint32_t(1) << bits;
        const size_t nkeys = size_t(1) << (3 * bits);
        std::vector<bool> seen(nkeys, false);
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < n; ++j)
                for (uint32_t k = 0; k < n; ++k) {
                    const Key key = key_from_grid_coordinate({i, j, k}, bits);
                    if (key < nkeys) {
                        check(!seen[size_t(key)], "key encoder is a bijection");
                        seen[size_t(key)] = true;
                    } else {
                        check(false, "key encoder stays inside the configured key range");
                    }
                    check(grid_coordinate_from_key(key, bits) == std::array<uint32_t, 3>{i, j, k},
                          "coordinate/key round trip");
                    const Position center{(double(i) + 0.5) / n, (double(j) + 0.5) / n,
                                           (double(k) + 0.5) / n};
                    check(key_from_position(center, bits) == key, "position encoder agrees with grid encoder");
                }
        for (size_t key = 0; key < nkeys; ++key) {
            const auto c = grid_coordinate_from_key(Key(key), bits);
            check(seen[key], "every key is represented");
            if (key > 0) {
                const auto previous = grid_coordinate_from_key(Key(key - 1), bits);
                const int distance = std::abs(int(c[0]) - int(previous[0])) +
                                     std::abs(int(c[1]) - int(previous[1])) +
                                     std::abs(int(c[2]) - int(previous[2]));
                check(distance == 1, "consecutive keys " + std::to_string(key - 1) + " and " +
                                          std::to_string(key) + " are not face-adjacent: (" +
                                          std::to_string(previous[0]) + "," + std::to_string(previous[1]) + "," +
                                          std::to_string(previous[2]) + ") -> (" + std::to_string(c[0]) + "," +
                                          std::to_string(c[1]) + "," + std::to_string(c[2]) + ")");
            }
        }
    }

    // Every key appears once globally, split across ranks, yielding known cuts.
    constexpr unsigned bits = 4;
    constexpr uint32_t n = 1U << bits;
    constexpr Key key_limit = Key(1) << (3 * bits);
    std::vector<Key> distributed_keys;
    for (Key key = Key(rank); key < key_limit; key += Key(nranks))
        distributed_keys.push_back(key);
    const auto decomposition = Decomposition::build(distributed_keys, comm, bits, 3 * bits);

    std::vector<Key> starts(static_cast<size_t>(nranks));
    std::vector<Key> ends(static_cast<size_t>(nranks));
#ifdef USE_MPI
    const Key local_start = decomposition.key_begin(rank);
    const Key local_end = decomposition.key_end(rank);
    MPI_Allgather(&local_start, 1, MPI_UINT64_T, starts.data(), 1, MPI_UINT64_T, comm);
    MPI_Allgather(&local_end, 1, MPI_UINT64_T, ends.data(), 1, MPI_UINT64_T, comm);
#else
    starts[0] = decomposition.key_begin(0);
    ends[0] = decomposition.key_end(0);
#endif
    check(starts.front() == 0 && ends.back() == key_limit, "rank intervals cover full key domain");
    for (int p = 0; p < nranks; ++p) {
        check(starts[size_t(p)] <= ends[size_t(p)], "rank key interval is nonnegative");
        if (p > 0)
            check(ends[size_t(p - 1)] == starts[size_t(p)], "rank intervals are contiguous and nonoverlapping");
    }
    for (Key key = 0; key < key_limit; ++key) {
        const int owner = decomposition.owner_of_key(key);
        check(owner >= 0 && owner < nranks && starts[size_t(owner)] <= key && key < ends[size_t(owner)],
              "key owner agrees with rank interval");
    }

    DistributedGrid3D<double> grid(n, decomposition, 0.0);
    grid.initialize([](const Position & center) {
        const uint32_t i = uint32_t(center[0] * n);
        const uint32_t j = uint32_t(center[1] * n);
        const uint32_t k = uint32_t(center[2] * n);
        return field_value({i, j, k});
    });
    for (size_t local = 0; local < grid.local_size(); ++local) {
        const auto c = grid.local_coordinate(local);
        check(grid.local_key(local) == key_from_grid_coordinate(c, bits), "local key matches local coordinate");
        check(grid.at(c[0], c[1], c[2]) == field_value(c), "local grid initialization and access");
    }

    // Arbitrary remote reads, with an empty query list on any rank that owns all keys.
    std::vector<DistributedGrid3D<double>::Coordinate> queries;
    bool found_remote = false;
    for (uint32_t i = 0; i < n && !found_remote; ++i)
        for (uint32_t j = 0; j < n && !found_remote; ++j)
            for (uint32_t k = 0; k < n && !found_remote; ++k)
                if (grid.owner(i, j, k) != rank) {
                    queries.push_back({i, j, k});
                    found_remote = true;
                }
    if (grid.local_size() > 0) {
        const auto c = grid.local_coordinate(0);
        queries.push_back(c);
    }
    const auto fetched = grid.fetch_remote_cells(queries, comm);
    check(fetched.size() == queries.size(), "remote fetch preserves query count");
    for (size_t q = 0; q < queries.size(); ++q) {
        check(fetched[q] == field_value(queries[q]), "remote fetch returns requested cell value");
        check(grid.at(queries[q][0], queries[q][1], queries[q][2]) == fetched[q],
              "remote fetch populates at() cache");
    }

    const std::array<std::tuple<uint32_t, HaloConnectivity, bool>, 4> halo_cases = {
        std::make_tuple(1U, HaloConnectivity::Faces, true),
        std::make_tuple(1U, HaloConnectivity::Full, true),
        std::make_tuple(2U, HaloConnectivity::Full, true),
        std::make_tuple(2U, HaloConnectivity::Full, false)};
    for (const auto & test_case : halo_cases) {
        const uint32_t width = std::get<0>(test_case);
        const HaloConnectivity connectivity = std::get<1>(test_case);
        const bool periodic = std::get<2>(test_case);
        for (size_t local = 0; local < grid.local_size(); ++local)
            grid[local] = field_value(grid.local_coordinate(local));
        const auto expected = reference_halo(grid, width, connectivity, periodic);
        grid.build_halo(width, connectivity, periodic);
        check(grid.halo_keys() == expected, "halo key list matches reference stencil geometry");
        grid.exchange_halo(comm);
        check_ghost_values(grid, 0.0, "initial halo exchange");
        for (size_t local = 0; local < grid.local_size(); ++local)
            grid[local] += 1000.0;
        grid.exchange_halo(comm);
        check_ghost_values(grid, 1000.0, "refreshed halo exchange");
    }

    // Concentrated keys create empty rank intervals and zero-cell ranks. Exercise
    // zero-count all-to-all paths and a collective remote read from those ranks.
    std::vector<Key> concentrated_keys;
    if (rank == 0)
        concentrated_keys.assign(64, 0);
    const auto concentrated = Decomposition::build(concentrated_keys, comm, bits, 3 * bits);
    DistributedGrid3D<double> sparse_grid(n, concentrated, 0.0);
    sparse_grid.initialize([](const Position & center) {
        const uint32_t i = uint32_t(center[0] * n);
        const uint32_t j = uint32_t(center[1] * n);
        const uint32_t k = uint32_t(center[2] * n);
        return field_value({i, j, k});
    });
    uint64_t local_cells = sparse_grid.local_size();
    uint64_t global_cells = local_cells;
    int local_empty = local_cells == 0 ? 1 : 0;
    int empty_ranks = local_empty;
#ifdef USE_MPI
    MPI_Allreduce(&local_cells, &global_cells, 1, MPI_UINT64_T, MPI_SUM, comm);
    MPI_Allreduce(&local_empty, &empty_ranks, 1, MPI_INT, MPI_SUM, comm);
#endif
    check(global_cells == sparse_grid.global_size(), "concentrated decomposition still covers every grid cell");
    check(nranks == 1 || empty_ranks > 0, "concentrated distribution creates empty ranks");

    std::vector<DistributedGrid3D<double>::Coordinate> sparse_queries;
    if (sparse_grid.owner(0, 0, 0) != rank)
        sparse_queries.push_back({0, 0, 0});
    const auto sparse_fetched = sparse_grid.fetch_remote_cells(sparse_queries, comm);
    check(sparse_fetched.size() == sparse_queries.size(), "empty-query ranks join remote fetch collective");
    for (const double value : sparse_fetched)
        check(value == field_value({0, 0, 0}), "empty-rank remote fetch returns correct value");
    sparse_grid.build_halo(1, HaloConnectivity::Full, true);
    sparse_grid.exchange_halo(comm);
    check_ghost_values(sparse_grid, 0.0, "halo exchange with empty ranks");

    int global_failures = failures;
    size_t global_checks = checks;
#ifdef USE_MPI
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, comm);
    const unsigned long long local_checks = checks;
    unsigned long long summed_checks = 0;
    MPI_Allreduce(&local_checks, &summed_checks, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    global_checks = size_t(summed_checks);
#endif
    if (rank == 0)
        std::cout << "PeanoHilbertDecomposition tests: " << global_checks << " checks, " << global_failures
                  << " failures on " << nranks << " rank(s)\n";

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return global_failures == 0 ? 0 : 1;
}
