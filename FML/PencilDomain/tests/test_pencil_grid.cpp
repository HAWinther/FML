#include <random>
#include <FML/PencilDomain/PencilGrid.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {
    int rank = 0;
    int failures = 0;
    size_t checks = 0;

    void check(bool condition, const std::string & message) {
        ++checks;
        if (!condition) {
            ++failures;
            if (failures <= 20) std::cerr << "rank " << rank << ": " << message << '\n';
        }
    }

    double cell_value(const std::array<int,3> & c) {
        return 1.0 + c[0] + 100.0*c[1] + 10000.0*c[2];
    }
}

int main() {
    {
        using namespace FML::GRID;
        MPI_Comm_rank(FML::communicator, &rank);
        int nranks = 1;
        MPI_Comm_size(FML::communicator, &nranks);
        constexpr int n = 16;
        for (int ndim = 1; ndim <= 3; ++ndim) {
        const std::array<int,3> mesh{n, ndim >= 2 ? n : 1, ndim >= 3 ? n : 1};
        auto domain = std::make_shared<PencilDomain>(mesh, 1.0, std::array<int,2>{0,0}, FML::communicator, ndim);

        // Partition ranges exactly cover their axes, and rank/coordinate maps invert.
        for (int axis = 0; axis < 2; ++axis) {
            const int parts = domain->process_grid()[axis];
            int next = 0;
            for (int p = 0; p < parts; ++p) {
                auto r = axis == 0 ? domain->x_range_of(p) : domain->y_range_of(p);
                check(r.start == next && r.count > 0, "real partition ranges are contiguous and nonempty");
                next = r.end();
            }
            check(next == (axis == 0 ? mesh[0] : mesh[1]), "real partition ranges cover the axis");
        }
        for (int r = 0; r < nranks; ++r) {
            auto coords = domain->coords_of(r);
            check(domain->rank_of(coords[0],coords[1]) == r, "rank and process coordinates are inverse");
        }

        uint64_t local_owned = 0;
        for (int i = 0; i < n; ++i)
        for (int j = 0; j < mesh[1]; ++j)
                for (int k = 0; k < mesh[2]; ++k) {
                    int owner = domain->owner_of_cell(i,j,k);
                    check(owner >= 0 && owner < nranks, "cell owner is a valid communicator rank");
                    check(domain->owner_of_position((i+0.5)/mesh[0],(j+0.5)/mesh[1],(k+0.5)/mesh[2]) == owner,
                          "cell and cell-center position ownership agree");
                    if (owner == rank) ++local_owned;
                }
        uint64_t total_owned = 0;
        MPI_Allreduce(&local_owned,&total_owned,1,MPI_UINT64_T,MPI_SUM,domain->communicator());
        check(total_owned == uint64_t(mesh[0])*mesh[1]*mesh[2], "every global cell has exactly one owner");

        // The particle domain (Global/Domain.h): a Slab when P1 = 1, else a Pencil, owning the positions of the cells
        const auto & pd = domain->particle_domain();
        check(pd.ntasks() == nranks and pd.is_slab() == (domain->process_grid()[1] == 1),
              "the particle domain is made for the ranks (a Slab when P1 = 1)");
        {
            std::mt19937 rng(42);
            std::uniform_real_distribution<double> u(-0.5, 1.5);
            bool agree = true;
            for (int q = 0; q < 20000; ++q) {
                const double box = domain->boxsize();
                const double x = u(rng) * box, y = u(rng) * box, z = u(rng) * box;
                const int owner = domain->owner_of_cell(domain->cell_of_position(x, 0), domain->cell_of_position(y, 1),
                                                        domain->cell_of_position(z, 2));
                agree = agree and domain->owner_of_position(x, y, z) == owner;
            }
            check(agree, "the particle domain owns the positions of the rank's cells (random periodic positions)");
        }

        PencilGrid grid(domain, 1);
        grid.fill_real_grid([&](std::array<double,3>& x) {
            return std::sin(2.0*M_PI*x[0]) + (ndim >= 2 ? 0.5*std::cos(4.0*M_PI*x[1]) : 0.0) +
                   (ndim >= 3 ? 0.25*std::sin(2.0*M_PI*x[2]) : 0.0);
        });
        grid.fftw_r2c();
        // Test a nontrivial slice of flattened Fourier rows.
        const size_t fourier_rows = size_t(grid.get_local_nky()) * grid.get_local_nkz();
        const size_t f0 = fourier_rows / 3, f1 = (2*fourier_rows) / 3;
        size_t fcount = 0;
        for (auto f : grid.get_fourier_range_of_rows(f0,f1)) {
            check(f.index == (size_t(f.a)*grid.get_local_nkz()+f.b)*n+f.ix,
                  "Fourier row range reports correct packed offsets");
            ++fcount;
        }
        check(fcount == (f1-f0)*n, "Fourier row range has requested number of modes");
        grid.fftw_c2r();
        double local_error = 0.0;
        for (auto c : grid.get_real_range()) {
            auto x = grid.get_real_position({c.i,c.j,c.k});
            double expected = std::sin(2.0*M_PI*x[0]) + (ndim >= 2 ? 0.5*std::cos(4.0*M_PI*x[1]) : 0.0) +
                              (ndim >= 3 ? 0.25*std::sin(2.0*M_PI*x[2]) : 0.0);
            local_error = std::max(local_error,std::abs(double(grid[c])-expected));
        }
        double error = 0.0;
        MPI_Allreduce(&local_error,&error,1,MPI_DOUBLE,MPI_MAX,domain->communicator());
        check(error < 500.0*std::numeric_limits<FloatType>::epsilon(), "distributed FFT round trip is normalized and accurate");

        // Flattened real rows can begin/end inside an x slab.
        const size_t real_rows = size_t(grid.get_local_nx()) * grid.get_local_ny();
        const size_t r0 = real_rows / 3, r1 = (2*real_rows) / 3;
        size_t rcount = 0;
        for (auto c : grid.get_real_range_of_rows(r0,r1)) {
            check(c.index == (size_t(c.i+grid.nghost())*(grid.get_local_ny()+2*grid.nghost(1)) +
                              size_t(c.j+grid.nghost(1))) * (2*(mesh[2]/2+1)) + c.k,
                  "real row range reports correct packed offsets");
            ++rcount;
        }
        check(rcount == (r1-r0)*mesh[2], "real row range has requested number of cells");

        // Populate real cells, exchange periodic faces/corners, and verify every ghost.
        grid.fill_real_grid(FloatType(0));
        for (auto c : grid.get_real_range()) grid[c] = FloatType(cell_value(grid.global_cell(c)));
        grid.communicate_boundaries();
        for (auto c : grid.get_real_range_with_ghosts())
            check(std::abs(double(grid[c])-cell_value(grid.global_cell(c))) < 0.25,
                  "periodic ghost exchange copies the owning cell (including corners)");

        // Each ghost contributes one unit to its periodic owner. Construct an
        // independent global expected count by summing all ranks' ghost cells.
        grid.fill_real_grid(FloatType(0));
        for (auto c : grid.get_real_range_with_ghosts())
            if (c.i < 0 || c.i >= grid.get_local_nx() || c.j < 0 || c.j >= grid.get_local_ny())
                grid[c] = FloatType(1);
        size_t global_size=size_t(mesh[0])*mesh[1]*mesh[2];
        std::vector<uint64_t> expected(global_size,0), global_expected(expected.size(),0);
        for (auto c : grid.get_real_range_with_ghosts())
            if (c.i < 0 || c.i >= grid.get_local_nx() || c.j < 0 || c.j >= grid.get_local_ny()) {
                auto g = grid.global_cell(c);
                ++expected[(size_t(g[0])*mesh[1]+g[1])*mesh[2]+g[2]];
            }
        MPI_Allreduce(expected.data(),global_expected.data(),int(expected.size()),MPI_UINT64_T,MPI_SUM,domain->communicator());
        grid.add_ghosts_to_owners();
        for (auto c : grid.get_real_range()) {
            auto g = grid.global_cell(c);
            uint64_t want = global_expected[(size_t(g[0])*mesh[1]+g[1])*mesh[2]+g[2]];
            check(std::abs(double(grid[c])-double(want)) < 0.25, "ghost accumulation adds every periodic ghost exactly once");
        }

        int total_failures = 0;
        MPI_Allreduce(&failures,&total_failures,1,MPI_INT,MPI_SUM,domain->communicator());
        uint64_t total_checks = 0, local_checks = checks;
        MPI_Allreduce(&local_checks,&total_checks,1,MPI_UINT64_T,MPI_SUM,domain->communicator());
        if (rank == 0) {
            if (total_failures == 0) std::cout << "PencilGrid NDIM=" << ndim << " checks passed: " << total_checks << " across " << nranks << " rank(s)\n";
            else std::cerr << "PencilGrid NDIM=" << ndim << " checks failed: " << total_failures << " of " << total_checks << '\n';
        }
        if (total_failures) MPI_Abort(domain->communicator(),1);
        }
    }
    FML::finalize_mpi();
    return 0;
}
