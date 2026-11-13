#include <FML/Global/Global.h>
#include <FML/PeanoHilbertDomain/DistributedGrid3D.h>

#include <iostream>
#include <cmath>
#include <string>

// MPI is initialized and finalized by FML (Global.cpp, FMLSetup); everything is collective over FML::communicator
int main(int argc, char ** argv) {
    using namespace FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT;
    const int rank = FML::ThisTask;

    const bool concentrated = argc > 1 && std::string(argv[1]) == "concentrated";
    // A deterministic sample distribution; replace with keys from local particles.
    std::vector<Key> local_keys;
    if (!concentrated || rank == 0)
        for (int i = 0; i < 10000; ++i) {
            const double x = concentrated ? 0.01 : std::fmod((double(i) * 0.6180339887498949 + 0.17 * rank), 1.0);
            const double y = concentrated ? 0.01 : std::fmod((double(i) * 0.4142135623730950 + 0.31 * rank), 1.0);
            const double z = concentrated ? 0.01 : std::fmod((double(i) * 0.7320508075688772 + 0.13 * rank), 1.0);
            local_keys.push_back(key_from_position({x, y, z}, 5));
        }

    const auto decomposition = PeanoHilbertDomain::build(local_keys, 5, 12);
    DistributedGrid3D<double> grid(32, decomposition, 0.0);
    grid.initialize([](const Position & center) { return center[0] + center[1] + center[2]; });
    grid.build_halo(1, HaloConnectivity::Full, true);
    grid.exchange_halo();
    const std::vector<DistributedGrid3D<double>::Coordinate> remote_query{{0, 0, 0}};
    const auto queried_value = grid.fetch_remote_cells(remote_query);

    int local_errors = 0;
    auto expected_value = [](const DistributedGrid3D<double>::Coordinate & c) {
        const double inv_n = 1.0 / 32.0;
        return ((double(c[0]) + 0.5) + (double(c[1]) + 0.5) + (double(c[2]) + 0.5)) * inv_n;
    };
    if (std::abs(queried_value[0] - expected_value({0, 0, 0})) > 1.0e-14)
        ++local_errors;
    for (const Key key : grid.halo_keys()) {
        const auto coordinate = grid_coordinate_from_key(key, decomposition.bits());
        const Key roundtrip_key = key_from_grid_coordinate(coordinate, decomposition.bits());
        if (roundtrip_key != key) {
            ++local_errors;
            continue;
        }
        try {
            if (std::abs(grid.at(coordinate[0], coordinate[1], coordinate[2]) - expected_value(coordinate)) > 1.0e-14)
                ++local_errors;
        } catch (const std::exception & e) {
            ++local_errors;
        }
    }
    int global_errors = local_errors;
    FML::SumOverTasks(&global_errors);
    if (rank == 0)
        std::cout << "runtime validation errors: " << global_errors << "\n";
    std::cout << "rank " << rank << " owns Hilbert keys [" << decomposition.key_begin(rank) << ", "
              << decomposition.key_end(rank) << "), " << grid.local_size() << " of " << grid.global_size()
              << " grid cells, and " << decomposition.total_particles()
              << " particles; cell (0,0,0)=" << queried_value.front() << ", ghosts=" << grid.halo_keys().size() << "\n";

    return global_errors == 0 ? 0 : 1;
}
