#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/Global/Global.h>

//==========================================================================================
// Example: refine a 2D periodic grid around a ring, load balance it along the Hilbert curve,
// exchange ghosts and evaluate a Laplacian with face neighbours that can be coarser/finer.
//==========================================================================================

using namespace FML::GRID;
constexpr int NDIM = 2;
using Grid = AdaptiveMPIGrid<NDIM, double>;

double ring(const std::array<double, NDIM> & x) {
    double r2 = 0.0;
    for (int d = 0; d < NDIM; d++)
        r2 += (x[d] - 0.6) * (x[d] - 0.6);
    const double dr = std::sqrt(r2) - 0.3;
    return std::exp(-dr * dr / (2 * 0.02 * 0.02));
}

int main() {
    const int levelmin = 4;
    const int levelmax = 9;
    Grid grid(levelmin, levelmax, true);

    // Refine where the field varies over a cell, one level per pass, until nothing changes
    for (int pass = 0; pass < levelmax - levelmin + 1; pass++) {
        for (IndexInt i = 0; i < grid.get_n_local(); i++) {
            auto & c = grid.get_cell(i);
            auto x = grid.get_cell_center(c);
            grid.get_data(i) = ring(x);
            const double h = grid.get_cell_size(c.level);
            double r2 = 0.0;
            for (int d = 0; d < NDIM; d++)
                r2 += (x[d] - 0.6) * (x[d] - 0.6);
            const bool interesting = std::abs(std::sqrt(r2) - 0.3) < 0.01 + 2 * h;
            grid.set_flag(i, interesting ? Grid::REFINE : Grid::COARSEN);
        }
        grid.adapt([](const double & parent, const Grid::Cell &, const Grid::Cell & child) {
            (void)parent;
            return ring(Grid::get_cell_center(child));
        });
        grid.load_balance([&](IndexInt i) { return 1.0 + grid.get_cell(i).level - levelmin; });
        grid.info();
    }

    // Laplacian using face neighbours (coarser: one cell, same: one cell, finer: 2^(NDIM-1) cells)
    std::vector<double> laplacian(grid.get_n_local());
    IndexInt n_coarse_fine = 0;
    auto compute_laplacian = [&](IndexInt i) {
        auto & c = grid.get_cell(i);
        const double h = grid.get_cell_size(c.level);
        double lap = 0.0;
        for (int dim = 0; dim < NDIM; dim++) {
            for (int sign = -1; sign <= 1; sign += 2) {
                auto nb = grid.get_face_neighbors(i, dim, sign);
                if (nb.relation != Grid::FaceNeighbors::SAME)
                    n_coarse_fine++;
                double avg = 0.0, hn = 0.0;
                for (int j = 0; j < nb.n; j++) {
                    avg += grid.get_data(nb.cells[j]) / nb.n;
                    hn = grid.get_cell_size(grid.get_cell(nb.cells[j]).level);
                }
                lap += (avg - grid.get_data(i)) / (0.5 * (h + hn)) / h;
            }
        }
        laplacian[i] = lap;
    };

    // Overlap the ghost exchange with the interior: start sending, compute cells that only have
    // local neighbours, wait, then compute the cells that need ghosts.
    // (adapt/load_balance already fill the ghosts; this is what a time step would look like.)
    auto exchange = grid.begin_exchange_ghosts();
    for (auto i : grid.get_interior_cells())
        compute_laplacian(i);
    exchange.wait();
    for (auto i : grid.get_boundary_cells())
        compute_laplacian(i);

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &n_coarse_fine, 1, MPI_LONG_LONG, MPI_SUM, FML::communicator);
#endif
    if (grid.check_invariants() and FML::ThisTask == 0)
        std::cout << "Grid is valid. Coarse/fine faces: " << n_coarse_fine << "\n";
    grid.write_ascii("cells");
}
