#ifndef FML_FIELD_LEVEL_FRONTEND_H
#define FML_FIELD_LEVEL_FRONTEND_H
#include <FML/FieldLevel/DistributedLPT.h>
#include <FML/Global/MPICounts.h>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#ifdef FML_FIELDLEVEL_PENCIL_GRID
#include <FML/PencilDomain/PencilGrid.h>
#endif
namespace FML::FIELDLEVEL::FRONTEND {
#ifdef FML_FIELDLEVEL_PENCIL_GRID
    using Grid = FML::GRID::PencilGrid<3>;
    inline constexpr int pencil_backend = 1;
#else
    using Grid = FML::GRID::FFTWGrid<3>;
    inline constexpr int pencil_backend = 0;
#endif
    using LinearModel = FRAMEWORK::DistributedLinearModelT<Grid>;
    using LPTModel = FRAMEWORK::DistributedLPTModelT<Grid>;
    using LinearPosterior = FRAMEWORK::DistributedLinearPosteriorT<Grid>;
    using LPTPosterior = FRAMEWORK::DistributedLPTPosteriorT<Grid>;
    inline std::array<int, 2> process_grid() {
#ifdef FML_FIELDLEVEL_PENCIL_GRID
        return FML::domain.get<FML::PencilDomain>().process_grid();
#else
        return {FML::NTasks, 1};
#endif
    }
    // Called collectively after FML setup, before model creation. Use shared cell-aligned edges.
    inline void initialize(int mesh, int density_mesh = 0) {
#ifdef FML_FIELDLEVEL_PENCIL_GRID
        int px = 0, py = 0;
        FML::COMMUNICATION::collective_try(
            [&] {
                if (const char * value = std::getenv("FML_PENCILS")) {
                    std::istringstream input(value);
                    std::string trailing;
                    if (!(input >> px >> py) || (input >> trailing) || px < 1 || py < 1 ||
                        static_cast<long long>(px) * py != FML::NTasks)
                        throw std::invalid_argument(
                            "[Field frontend] FML_PENCILS requires positive px py with px*py=NTasks");
                }
            },
            "Field frontend pencil controls");
        int dims[2] = {px, py}, low[2], high[2];
        MPI_Allreduce(dims, low, 2, MPI_INT, MPI_MIN, FML::communicator);
        MPI_Allreduce(dims, high, 2, MPI_INT, MPI_MAX, FML::communicator);
        FML::COMMUNICATION::require_count(low[0] == high[0] && low[1] == high[1],
                                          "[Field frontend] Inconsistent FML_PENCILS across ranks");
        if (px == 0)
            MPI_Dims_create(FML::NTasks, 2, dims);
        const int common = std::gcd(mesh, density_mesh > 0 ? density_mesh : mesh);
        FML::COMMUNICATION::require_count(mesh >= 4 && common >= dims[0] && common >= dims[1] && dims[0] <= mesh &&
                                              dims[1] <= mesh / 2 + 1,
                                          "[Field frontend] Pencil process grid must fit both meshes");
        auto edges = [common](int parts) {
            std::vector<double> values(parts + 1);
            for (int p = 0; p <= parts; ++p)
                values[p] = double(static_cast<long long>(p) * common / parts) / common;
            return values;
        };
        FML::set_domain(FML::Domain::pencil(edges(dims[0]), edges(dims[1])));
        if (FML::ThisTask == 0)
            std::cout << "FieldLevel backend: PencilGrid " << dims[0] << " x " << dims[1] << '\n';
#else
        (void)mesh;
        (void)density_mesh;
        if (FML::ThisTask == 0)
            std::cout << "FieldLevel backend: FFTWGrid slabs\n";
#endif
    }
} // namespace FML::FIELDLEVEL::FRONTEND
#endif
