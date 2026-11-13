// Tests of PencilGrid<N> (N = 2, 3): the layout, indices and coordinates, the Fourier transform against FFTWGrid,
// the extra cells and the matching particle domain. Run on 1 - 6 tasks
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/TestHelpers.h>

#include <array>
#include <cmath>
#include <map>
#include <string>
#include <vector>

using FML::TESTS::check;
using namespace FML::GRID;

namespace {

    // A smooth periodic test field with all kinds of modes (not symmetric)
    template <int N>
    FloatType field(const std::array<double, N> & x) {
        double v = 1.0 + std::sin(2 * M_PI * x[0]) + 0.5 * std::cos(2 * M_PI * (2 * x[1] + 0.3));
        v += 0.25 * std::sin(2 * M_PI * (3 * x[0] - x[1]) + 0.1);
        if constexpr (N == 3)
            v += 0.3 * std::cos(2 * M_PI * (x[0] + 2 * x[1] - 3 * x[2]) + 0.7) + 0.2 * std::sin(2 * M_PI * 5 * x[2]);
        return FloatType(v);
    }

    // A value that identifies a global cell
    template <int N>
    double cell_value(const std::array<int, N> & global) {
        double v = 1.0 + global[0] + 100.0 * global[1];
        if constexpr (N == 3)
            v += 10000.0 * global[2];
        return v;
    }

    int wrap(int i, int n) { return ((i % n) + n) % n; }

    template <int N, class Coord>
    Coord global_cell(const PencilGrid<N> & grid, Coord coord) {
        const int n = grid.get_nmesh();
        coord[0] = wrap(coord[0] + int(grid.get_local_x_start()), n);
        coord[1] = wrap(coord[1] + int(grid.get_local_y_start()), n);
        return coord;
    }

    template <int N>
    std::string label(const PencilGrid<N> & grid, const std::string & what) {
        const auto P = grid.get_domain().process_grid();
        return "N = " + std::to_string(N) + " Nmesh = " + std::to_string(grid.get_nmesh()) +
               " P = " + std::to_string(P[0]) + "x" + std::to_string(P[1]) + " extra (" +
               std::to_string(grid.get_n_extra_slices_left()) + ", " + std::to_string(grid.get_n_extra_slices_right()) +
               "): " + what;
    }

    //======================================================================
    // Layout, indices, coordinates and positions
    //======================================================================
    template <int N>
    void test_layout(PencilGrid<N> & grid) {
        const int n = grid.get_nmesh();

        // The cells and modes of the tasks add up to the whole grid
        double ncells = double(grid.get_local_nx()) * double(grid.get_local_ny()) * (N == 3 ? n : 1);
        double nmodes = double(grid.get_ntot_fourier());
        FML::SumOverTasks(&ncells);
        FML::SumOverTasks(&nmodes);
        check(ncells == std::pow(double(n), N), label(grid, "the real cells of the tasks cover the grid"));
        check(nmodes == std::pow(double(n), N - 1) * (n / 2 + 1), label(grid, "the Fourier cells cover the modes"));

        // The real range visits the cells once, the index and the coordinate are inverse, the positions are those
        // of the global cells
        bool ok = true;
        size_t count = 0;
        std::map<FML::IndexIntType, int> seen;
        for (auto && index : grid.get_real_range()) {
            const auto coord = grid.get_coord_from_index(index);
            ok = ok and grid.get_index_real(coord) == index and seen[index]++ == 0;
            ok = ok and coord[0] >= 0 and coord[0] < grid.get_local_nx() and coord[1] >= 0 and
                 coord[1] < grid.get_local_ny() and coord[N - 1] >= 0 and coord[N - 1] < n;
            const auto pos = grid.get_real_position(coord);
            const auto g = global_cell(grid, coord);
            for (int d = 0; d < N; d++)
                ok = ok and std::fabs(pos[d] - g[d] / double(n)) < 1e-14;
            count++;
        }
        ok = ok and count == size_t(grid.get_local_nrows()) * n;
        check(ok, label(grid, "real range, get_index_real / get_coord_from_index and get_real_position"));

        // The same for the extra cells
        ok = true;
        const int nl = grid.get_n_extra_slices_left(), nr = grid.get_n_extra_slices_right();
        for (int ix = -nl; ix < grid.get_local_nx() + nr; ix++)
            for (int iy = (N == 3 ? -nl : 0); iy < grid.get_local_ny() + (N == 3 ? nr : 0); iy++) {
                std::array<int, N> coord{};
                coord[0] = ix;
                coord[1] = iy;
                ok = ok and grid.get_coord_from_index(grid.get_index_real(coord)) == coord;
            }
        check(ok, label(grid, "get_index_real / get_coord_from_index of the extra cells"));

        // The Fourier range and coordinates
        ok = true;
        count = 0;
        for (auto && index : grid.get_fourier_range()) {
            const auto coord = grid.get_fourier_coord_from_index(index);
            ok = ok and grid.get_index_fourier(coord) == index;
            const auto k = grid.get_fourier_integer_wavevector_from_index(index);
            const auto kvec = grid.get_fourier_wavevector(coord);
            std::array<double, N> kvec2;
            double kmag;
            grid.get_fourier_wavevector_and_norm_by_index(index, kvec2, kmag);
            double k2 = 0.0;
            for (int d = 0; d < N; d++) {
                ok = ok and k[d] >= -n / 2 and k[d] <= n / 2 and kvec[d] == 2 * M_PI * k[d] and kvec2[d] == kvec[d];
                k2 += kvec[d] * kvec[d];
            }
            ok = ok and k[N - 1] >= 0 and std::fabs(kmag - std::sqrt(k2)) < 1e-12;
            count++;
        }
        ok = ok and count == size_t(grid.get_ntot_fourier());
        check(ok, label(grid, "Fourier range, get_index_fourier / get_fourier_coord_from_index and wave-vectors"));
    }

    //======================================================================
    // The Fourier transform: the same as FFTWGrid, and back
    //======================================================================
    template <int N>
    void test_fft(PencilGrid<N> & grid) {
        const int n = grid.get_nmesh();
        grid.fill_real_grid(
            std::function<FloatType(std::array<double, N> &)>([](std::array<double, N> & x) { return field<N>(x); }));
        PencilGrid<N> original = grid;
        grid.fftw_r2c();
        check(not grid.get_grid_status_real() and not grid.nan_in_grids(), label(grid, "r2c: status and finite"));

        // FFTWGrid of the same field (needs NTasks to divide Nmesh): all of its modes on every task
        if (n % FML::NTasks == 0) {
            FFTWGrid<N> reference(n);
            reference.fill_real_grid(std::function<FloatType(std::array<double, N> &)>(
                [](std::array<double, N> & x) { return field<N>(x); }));
            reference.fftw_r2c();
            std::vector<double> local;
            for (auto && index : reference.get_fourier_range()) {
                const auto k = reference.get_fourier_integer_wavevector_from_index(index);
                const auto z = reference.get_fourier_from_index(index);
                for (int d = 0; d < N; d++)
                    local.push_back(k[d]);
                local.push_back(z.real());
                local.push_back(z.imag());
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
            std::map<std::array<int, N>, std::complex<double>> modes;
            for (size_t i = 0; i < all.size(); i += N + 2) {
                std::array<int, N> k;
                for (int d = 0; d < N; d++)
                    k[d] = int(all[i + d]);
                modes[k] = {all[i + N], all[i + N + 1]};
            }

            double maxdiff = 0.0;
            bool found = true;
            for (auto && index : grid.get_fourier_range()) {
                const auto k = grid.get_fourier_integer_wavevector_from_index(index);
                auto it = modes.find(k);
                found = found and it != modes.end();
                if (it != modes.end())
                    maxdiff = std::max(maxdiff,
                                       std::abs(std::complex<double>(grid.get_fourier_from_index(index)) - it->second));
            }
            check(found and maxdiff < 1e-12, label(grid, "r2c = FFTWGrid's r2c (normalized by 1/Nmesh^N)"));
        }

        // The modes of the test field: the mean is 1
        double mean = 0.0;
        for (auto && index : grid.get_fourier_range()) {
            const auto k = grid.get_fourier_integer_wavevector_from_index(index);
            if (k == std::array<int, N>{})
                mean += grid.get_fourier_from_index(index).real();
        }
        FML::SumOverTasks(&mean);
        check(std::fabs(mean - 1.0) < 1e-12, label(grid, "the k = 0 mode is the mean"));

        grid.fftw_c2r();
        double maxerr = 0.0;
        for (auto && index : grid.get_real_range())
            maxerr = std::max(maxerr, std::fabs(grid.get_real_from_index(index) - original.get_real_from_index(index)));
        check(grid.get_grid_status_real() and maxerr < 1e-12, label(grid, "r2c then c2r gives back the grid"));

        // The free functions
        PencilGrid<N> fourier, real;
        fftw_r2c(original, fourier);
        fftw_c2r(fourier, real);
        maxerr = 0.0;
        for (auto && index : real.get_real_range())
            maxerr = std::max(maxerr, std::fabs(real.get_real_from_index(index) - original.get_real_from_index(index)));
        check(maxerr < 1e-12 and original.get_grid_status_real(), label(grid, "fftw_r2c / fftw_c2r(in, out)"));

        // fill_fourier_grid
        grid.fill_fourier_grid(std::function<ComplexType(std::array<double, N> &)>(
            [](std::array<double, N> & k) { return ComplexType(k[0], k[N - 1]); }));
        bool ok = not grid.get_grid_status_real();
        for (auto && index : grid.get_fourier_range()) {
            const auto kvec = grid.get_fourier_wavevector_from_index(index);
            ok = ok and grid.get_fourier_from_index(index) == ComplexType(kvec[0], kvec[N - 1]);
        }
        check(ok, label(grid, "fill_fourier_grid"));
    }

    //======================================================================
    // The extra cells
    //======================================================================
    template <int N>
    void test_extra_cells(PencilGrid<N> & grid) {
        const int nl = grid.get_n_extra_slices_left(), nr = grid.get_n_extra_slices_right();
        if (nl + nr == 0)
            return;
        const int nx = int(grid.get_local_nx()), ny = int(grid.get_local_ny()), n = grid.get_nmesh();
        const int nyl = N == 3 ? nl : 0, nyr = N == 3 ? nr : 0;
        auto all_cells = [&](auto && f) {
            for (int ix = -nl; ix < nx + nr; ix++)
                for (int iy = -nyl; iy < ny + nyr; iy++)
                    for (int iz = 0; iz < (N == 3 ? n : 1); iz++) {
                        std::array<int, N> coord{};
                        coord[0] = ix;
                        coord[1] = iy;
                        if constexpr (N == 3)
                            coord[2] = iz;
                        f(coord);
                    }
        };

        // communicate_boundaries: every extra cell (also the corners) gets the value of its periodic global cell
        grid.fill_real_grid(-1.0);
        for (auto && index : grid.get_real_range())
            grid.set_real_from_index(index,
                                     FloatType(cell_value<N>(global_cell(grid, grid.get_coord_from_index(index)))));
        grid.communicate_boundaries();
        bool ok = true;
        all_cells([&](const std::array<int, N> & c) {
            ok = ok and grid.get_real(c) == FloatType(cell_value<N>(global_cell(grid, c)));
        });
        check(ok, label(grid, "communicate_boundaries fills the extra cells (and the corners)"));

        // add_contribution_from_extra_cells: every cell gets the values of all its copies, the extra cells zero
        all_cells(
            [&](const std::array<int, N> & c) { grid.set_real(c, FloatType(cell_value<N>(global_cell(grid, c)))); });
        grid.add_contribution_from_extra_cells();
        auto copies = [&](int i, int nlocal) {
            // The local cell i is also an extra cell of the task before (i < nr) and of the task after (i >= n - nl)
            return 1 + (i < nr ? 1 : 0) + (i >= nlocal - nl ? 1 : 0);
        };
        ok = true;
        all_cells([&](const std::array<int, N> & c) {
            const bool main = c[0] >= 0 and c[0] < nx and c[1] >= 0 and c[1] < ny;
            double expected = 0.0;
            if (main)
                expected = cell_value<N>(global_cell(grid, c)) * copies(c[0], nx) * (N == 3 ? copies(c[1], ny) : 1);
            ok = ok and grid.get_real(c) == FloatType(expected);
        });
        check(ok, label(grid, "add_contribution_from_extra_cells adds the extra cells to their owners"));
    }

    //======================================================================
    // Copies, the particle domain
    //======================================================================
    template <int N>
    void test_misc(PencilGrid<N> & grid) {
        grid.fill_real_grid(
            std::function<FloatType(std::array<double, N> &)>([](std::array<double, N> & x) { return field<N>(x); }));
        PencilGrid<N> copy(grid);
        PencilGrid<N> assigned;
        assigned = grid;
        PencilGrid<N> moved(std::move(copy));
        bool ok = not copy and moved and assigned and moved.get_fourier_grid() != grid.get_fourier_grid();
        for (auto && index : grid.get_real_range())
            ok = ok and moved.get_real_from_index(index) == grid.get_real_from_index(index) and
                 assigned.get_real_from_index(index) == grid.get_real_from_index(index);
        moved.fftw_r2c();
        grid.fftw_r2c();
        for (auto && index : grid.get_fourier_range())
            ok = ok and moved.get_fourier_from_index(index) == grid.get_fourier_from_index(index);
        moved.free();
        check(ok and not moved, label(grid, "copy, assignment, move (sharing the transforms) and free"));

        // The particle domain of the pencils: the owner of the position of every local cell is this task
        const auto particle_domain = grid.get_domain().particle_domain();
        check(particle_domain.is_slab() == (grid.get_domain().process_grid()[1] == 1),
              label(grid, "the particle domain is a Slab when P1 = 1"));
        ok = true;
        grid.set_grid_status_real(true);
        for (auto && index : grid.get_real_range()) {
            const auto x = grid.get_real_position(grid.get_coord_from_index(index));
            ok = ok and particle_domain.owner(x.data()) == FML::ThisTask;
        }
        check(ok, label(grid, "the particle domain owns the positions of the local cells"));
    }

    template <int N>
    void run(PencilGrid<N> grid) {
        test_layout(grid);
        test_fft(grid);
        test_extra_cells(grid);
        test_misc(grid);
    }

    // P0 x P1 with P0 * P1 = NTasks
    std::vector<std::array<int, 2>> process_grids() {
        std::vector<std::array<int, 2>> grids;
        for (int p0 = 1; p0 <= FML::NTasks; p0++)
            if (FML::NTasks % p0 == 0)
                grids.push_back({p0, FML::NTasks / p0});
        return grids;
    }

    // Uneven edges on cells: part i gets base + (i % 2) cells (the rest to the last part)
    std::vector<double> uneven_edges(int parts, int n) {
        std::vector<double> e(parts + 1, 0.0);
        int cell = 0;
        for (int i = 0; i < parts; i++) {
            e[i] = cell / double(n);
            cell += std::max(1, n / parts - 1 + 2 * (i % 2));
        }
        e[parts] = 1.0;
        return e;
    }

} // namespace

int main() {
    // The default pencils of a 3D grid
    const auto P3 = FML::pencil_domain().get<FML::PencilDomain>().process_grid();
    for (int n : {12, 9}) {
        for (auto extra : {std::array<int, 2>{0, 0}, std::array<int, 2>{1, 1}, std::array<int, 2>{2, 1}}) {
            // 2D: slabs in x (the default pencils: edges on cells as even as possible)
            if (n >= 2 * FML::NTasks)
                run(PencilGrid<2>(n, extra[0], extra[1]));
            // 3D: the default pencils and every uniform process grid
            if (n >= 2 * P3[0] and n >= 2 * P3[1])
                run(PencilGrid<3>(n, extra[0], extra[1]));
            for (auto P : process_grids())
                if (n % P[0] == 0 and n % P[1] == 0 and n >= 2 * P[0] and n >= 2 * P[1])
                    run(PencilGrid<3>(n, extra[0], extra[1], FML::PencilDomain::uniform(P[0], P[1])));
        }
    }

    // Uneven pencils (edges on cells)
    for (auto P : process_grids()) {
        const int n = 16;
        auto domain = FML::PencilDomain::from_edges(uneven_edges(P[0], n), uneven_edges(P[1], n));
        run(PencilGrid<3>(n, 1, 1, domain));
        if (P[1] == 1)
            run(PencilGrid<2>(n, 1, 1, domain));
    }

    return FML::TESTS::summary("PencilGrid");
}
