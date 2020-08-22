#include <FML/FFTWGrid/FFTWGrid.h>
#include <numeric>

//===================================================
// Test that the methods are working as they should
//===================================================

template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;

template <int N>
void RunTests();

int main() {

    // Run some unit tests
    RunTests<2>();
    RunTests<3>();
    RunTests<4>();
    return 0;
}

template <int N>
void RunTests() {

    if (FML::ThisTask == 0)
        std::cout << "Running tests N = " << N << "\n";

    const int _nleft = 2;
    const int _nright = 2;
    const int _Nmesh = FML::NTasks < 10 ? 4 * FML::NTasks : 2 * FML::NTasks;
    FFTWGrid<N> grid(_Nmesh, _nleft, _nright);

    auto Nmesh = grid.get_nmesh();
    assert(Nmesh == _Nmesh);

    auto Ndim = grid.get_ndim();
    assert(Ndim == N);

    auto nleft = grid.get_n_extra_slices_left();
    assert(nleft == _nleft);

    auto nright = grid.get_n_extra_slices_right();
    assert(nright == _nright);

    auto local_nx = grid.get_local_nx();
    std::vector<int> n_per_task(FML::NTasks + 1, 0);
    n_per_task[FML::ThisTask] = local_nx;
    n_per_task[FML::NTasks] = local_nx;
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, n_per_task.data(), FML::NTasks + 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    assert(n_per_task[FML::NTasks] == Nmesh);

    int nbefore = 0;
    for (int i = 0; i < FML::ThisTask; i++)
        nbefore += n_per_task[i];
    assert(grid.get_local_x_start() == nbefore);

    auto ncellperslice = grid.get_ntot_real_slice_alloc();
    assert(ncellperslice >= FML::power(Nmesh, Ndim - 1));
    assert(grid.get_ntot_real() >= local_nx * FML::power(Nmesh, Ndim - 1));

    for (auto real_index : grid.get_real_range()) {
        auto coord = grid.get_coord_from_index(real_index);
        auto index = grid.get_index_real(coord);
        assert(index == real_index);
        auto value = 0.5 * FML::uniform_random() - 1.0;
        grid.set_real_from_index(real_index, value);
        assert(grid.get_real_from_index(real_index) == value);
        assert(grid.get_real(coord) == value);
    }

    auto grid2 = grid;

    grid2.fftw_r2c();

    for (auto fourier_index : grid2.get_fourier_range()) {
        auto coord = grid2.get_fourier_coord_from_index(fourier_index);

        auto index = grid2.get_index_fourier(coord);
        assert(index == fourier_index);

        auto value1 = grid2.get_fourier_from_index(fourier_index);
        auto value2 = grid2.get_fourier(coord);
        assert(value1 == value2);

        auto k1_vec = grid2.get_fourier_wavevector(coord);
        auto k2_vec = grid2.get_fourier_wavevector_from_index(fourier_index);
        for (int idim = 0; idim < Ndim; idim++) {
            assert(std::fabs(k1_vec[idim] - k2_vec[idim]) < 1e-10);
        }

        auto value = grid2.get_fourier(coord);
        value *= 2;
        grid2.set_fourier(coord, value);
        assert(std::abs(grid2.get_fourier(coord) - value) < 1e-10);
        value /= 2;
        grid2.set_fourier_from_index(fourier_index, value);

        double kmag, kmag2;
        grid2.get_fourier_wavevector_and_norm_by_index(fourier_index, k1_vec, kmag);
        grid2.get_fourier_wavevector_and_norm2_by_index(fourier_index, k2_vec, kmag2);
        assert(std::fabs(kmag - std::sqrt(kmag2)) < 1e-10);
        for (int idim = 0; idim < Ndim; idim++) {
            assert(std::fabs(k1_vec[idim] - k2_vec[idim]) < 1e-10);
        }
    }

    grid2.fftw_c2r();

    for (auto real_index : grid.get_real_range()) {
        auto value1 = grid.get_real_from_index(real_index);
        auto value2 = grid2.get_real_from_index(real_index);
        assert(std::fabs(value1 - value2) < 1e-10);
    }

    // Test dumping to file and reading it back in
    grid2.fill_real_grid(0.0);
    grid2.dump_to_file("output_grid");
    grid.load_from_file("output_grid");
    for (auto real_index : grid.get_real_range()) {
        auto value1 = grid.get_real_from_index(real_index);
        auto value2 = grid2.get_real_from_index(real_index);
        assert(std::fabs(value1 - value2) < 1e-10);
    }

    // Test fourier transforms
    grid.fill_real_grid(0.0);

    const int n = 2;
    auto solution = [&](std::array<double, N> & pos) -> double {
        auto value = 0.0;
        for (auto & x : pos)
            value += sin(n * 2 * M_PI * x);
        return value;
    };
    auto source = [&](std::array<double, N> & pos) -> double {
        return -(n * 2 * M_PI) * (n * 2 * M_PI) * solution(pos);
    };

    for (auto && index : grid.get_real_range()) {
        auto coord = grid.get_coord_from_index(index);
        auto pos = grid.get_real_position(coord);
        auto value = source(pos);
        grid.set_real_from_index(index, value);
    }

    grid.fftw_r2c();

    for (auto && index : grid.get_fourier_range()) {
        auto kvec = grid.get_fourier_wavevector_from_index(index);
        auto knorm2 = std::inner_product(kvec.begin(), kvec.end(), kvec.begin(), 0.0);
        auto value = grid.get_fourier_from_index(index);
        if (std::abs(value) > 1e-5) {
            assert(std::fabs(std::sqrt(knorm2) / (2.0 * M_PI) - n) < 1e-5);
        }
        if (knorm2 > 0.0) {
            grid.set_fourier_from_index(index, -value / knorm2);
        } else {
            grid.set_fourier_from_index(index, 0.0);
        }
    }

    grid.fftw_c2r();

    for (auto && index : grid.get_real_range()) {
        auto coord = grid.get_coord_from_index(index);
        auto pos = grid.get_real_position(coord);
        auto value = grid.get_real_from_index(index);
        auto value_analytic = solution(pos);
        assert(std::abs(value - value_analytic) < 1e-10);
    }

    if (FML::ThisTask == 0)
        std::cout << "Done\n" << std::flush;
}
