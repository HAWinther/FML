// The complex-to-complex grids CFFTWGrid (FFTW-MPI slabs) and CPencilGrid (pencils): the transforms against
// the exact coefficients of plane waves (both signs, the Nyquist modes), round trips, slabs = pencils, the extra cells,
// checkpoints and the grid interface (2D and 3D)
#include <FML/FFTWGrid/CFFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/PencilDomain/CPencilGrid.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
using FML::GRID::ComplexType;
using FML::GRID::CFFTWGrid;
using FML::GRID::CPencilGrid;

namespace {
    // A few plane waves: amplitude a_j at the integer wave-vector k_j (the even-mesh Nyquist mode +n/2 included)
    template <int N>
    struct Waves {
        std::vector<std::array<int, N>> k;
        std::vector<ComplexType> a;
        explicit Waves(int n) {
            std::array<int, N> k0{}, k1{}, k2{}, k3{};
            k0[0] = 1;
            k1[0] = -2, k1[1] = 1;
            k2[N - 1] = -1, k2[0] = 3;
            for (auto & v : k3)
                v = n / 2;
            k = {k0, k1, k2, k3};
            a = {{.5, .25}, {-.3, .1}, {.2, -.6}, {.05, .07}};
        }
        ComplexType operator()(const std::array<double, N> & x) const {
            std::complex<double> v = 0;
            for (size_t j = 0; j < k.size(); j++) {
                double phase = 0;
                for (int d = 0; d < N; d++)
                    phase += 2 * M_PI * k[j][d] * x[d];
                v += std::complex<double>(a[j].real(), a[j].imag()) * std::polar(1.0, phase);
            }
            return ComplexType(v.real(), v.imag());
        }
        // The coefficient of a mode (wave-vectors modulo n)
        ComplexType coefficient(const std::array<int, N> & m, int n) const {
            ComplexType c = 0;
            for (size_t j = 0; j < k.size(); j++) {
                bool same = true;
                for (int d = 0; d < N; d++)
                    same = same and ((m[d] - k[j][d]) % n + n) % n == 0;
                if (same)
                    c += a[j];
            }
            return c;
        }
    };

    // All modes on every task, by integer wave-vector modulo n (row major)
    template <class Grid>
    std::vector<ComplexType> all_modes(const Grid & g) {
        const int n = g.get_nmesh(), N = g.get_ndim();
        size_t ntot = 1;
        for (int d = 0; d < N; d++)
            ntot *= size_t(n);
        std::vector<double> re(ntot, 0.0), im(ntot, 0.0);
        for (auto && index : g.get_fourier_range()) {
            const auto m = g.get_fourier_integer_wavevector_from_index(index);
            size_t k = 0;
            for (int d = 0; d < N; d++)
                k = k * n + size_t((m[d] + n) % n);
            re[k] = g.get_fourier_from_index(index).real();
            im[k] = g.get_fourier_from_index(index).imag();
        }
        FML::SumArrayOverTasks(re.data(), re.size());
        FML::SumArrayOverTasks(im.data(), im.size());
        std::vector<ComplexType> all(ntot);
        for (size_t k = 0; k < ntot; k++)
            all[k] = ComplexType(re[k], im[k]);
        return all;
    }

    // The global position of a local cell coordinate (the extra cells: periodically wrapped)
    template <class Grid, int N>
    std::array<double, N> wrapped_position(const Grid & g, std::array<int, N> c) {
        const int n = g.get_nmesh();
        std::array<double, N> x;
        for (int d = 0; d < N; d++)
            x[d] = double(((c[d] + int(g.get_local_range(d)[0])) % n + n) % n) / n;
        return x;
    }

    template <class Grid, int N>
    void test_grid(Grid g, const std::string & label) {
        const int n = g.get_nmesh();
        const Waves<N> waves(n);
        const double tol = sizeof(FML::GRID::FloatType) == 4 ? 1e-5 : 1e-13;

        // Fill, forward: the coefficients are exactly the amplitudes
        g.fill_real_grid([&](std::array<double, N> & x) { return waves(x); });
        check(g.get_grid_status_real(), label + "fill_real_grid sets position-space status");
        const auto original = [&] {
            std::vector<ComplexType> v;
            for (auto && index : g.get_real_range())
                v.push_back(g.get_real_from_index(index));
            return v;
        }();
        g.fftw_forward();
        double error = 0;
        for (auto && index : g.get_fourier_range()) {
            const auto m = g.get_fourier_integer_wavevector_from_index(index);
            error = std::max(error, double(std::abs(g.get_fourier_from_index(index) - waves.coefficient(m, n))));
            for (int d = 0; d < N; d++)
                error = std::max(error, (m[d] > n / 2 or m[d] <= -(n + 1) / 2) ? 1.0 : 0.0);
        }
        FML::MaxOverTasks(&error);
        check(error < tol and not g.get_grid_status_real(),
              label + "forward = plane-wave amplitudes, all modes in (-n/2, n/2] (" + num(error, 2) + ")");

        // The number of modes: n^N over the tasks
        long long nmodes = g.get_ntot_fourier(), ncells = g.get_ntot_real();
        FML::SumOverTasks(&nmodes);
        FML::SumOverTasks(&ncells);
        check(nmodes == std::llround(std::pow(n, N)) and ncells == nmodes, label + "n^N cells and modes in total");

        // Round trip
        g.fftw_backward();
        error = 0;
        size_t i = 0;
        for (auto && index : g.get_real_range())
            error = std::max(error, double(std::abs(g.get_real_from_index(index) - original[i++])));
        FML::MaxOverTasks(&error);
        check(error < tol, label + "backward(forward) = identity (" + num(error, 2) + ")");

        // The extra cells: the periodic neighbours (pencils: also in y, with the corners)
        g.communicate_boundaries();
        error = 0;
        const int nl = g.get_n_extra_slices_left(), nr = g.get_n_extra_slices_right();
        const bool pencils = not std::is_same_v<Grid, CFFTWGrid<N>>;
        const int ylo = pencils and N == 3 ? -nl : 0;
        const int yhi = pencils and N == 3 ? int(g.get_local_range(1)[1]) + nr : (N == 3 ? n : n);
        for (int ix = -nl; ix < int(g.get_local_range(0)[1]) + nr; ix++)
            for (int iy = ylo; iy < (N == 3 ? yhi : n); iy++) {
                std::array<int, N> c{};
                c[0] = ix;
                c[1] = iy;
                for (int iz = 0; iz < (N == 3 ? n : 1); iz++) {
                    if constexpr (N == 3)
                        c[2] = iz;
                    error = std::max(error, double(std::abs(g.get_real(c) - waves(wrapped_position<Grid, N>(g, c)))));
                }
            }
        FML::MaxOverTasks(&error);
        check(error < tol, label + "communicate_boundaries fills the extra cells (" + num(error, 2) + ")");

        // Checkpoint round trip
        const std::string prefix = "test_cgrid_" + std::to_string(N) + (pencils ? "p" : "s");
        g.dump_to_file(prefix);
        Grid h;
        h.load_from_file(prefix);
        std::remove((prefix + "." + std::to_string(FML::ThisTask)).c_str());
        bool same = h.get_nmesh() == n and h.get_ntot_real() == g.get_ntot_real();
        for (auto && index : g.get_real_range())
            same = same and h.get_real_from_index(index) == g.get_real_from_index(index);
        int ok = same ? 1 : 0;
        FML::MinOverTasks(&ok);
        check(ok == 1, label + "dump_to_file / load_from_file round trip");

        // Copy, clone_empty, move, fill_fourier_grid
        auto copy = g;
        auto empty = g.clone_empty();
        double zero = 0;
        for (auto && index : empty.get_real_range())
            zero = std::max(zero, double(std::abs(empty.get_real_from_index(index))));
        auto moved = std::move(copy);
        check(zero == 0 and bool(moved) and not bool(copy) and moved.get_nmesh() == n,
              label + "copy, clone_empty and move");
        moved.fill_fourier_grid([](std::array<double, N> & k) { return ComplexType(k[0], -k[N - 1]); });
        error = 0;
        for (auto && index : moved.get_fourier_range()) {
            const auto k = moved.get_fourier_wavevector_from_index(index);
            error = std::max(error, double(std::abs(moved.get_fourier_from_index(index) - ComplexType(k[0], -k[N - 1]))));
        }
        FML::MaxOverTasks(&error);
        check(error == 0, label + "fill_fourier_grid by wave-vector");
    }

    // Pencils: add_contribution_from_extra_cells moves the extra cells to their owners (the sum is kept)
    template <int N>
    void test_add_contribution(CPencilGrid<N> g, const std::string & label) {
        g.fill_real_grid(ComplexType(0));
        const int nl = g.get_n_extra_slices_left(), nr = g.get_n_extra_slices_right();
        const int nx = int(g.get_local_range(0)[1]), ny = int(g.get_local_range(1)[1]);
        double added = 0;
        for (int ix = -nl; ix < nx + nr; ix++)
            for (int iy = (N == 3 ? -nl : 0); iy < (N == 3 ? ny + nr : 1); iy++) {
                std::array<int, N> c{};
                c[0] = ix;
                c[1] = iy;
                const bool extra = ix < 0 or ix >= nx or (N == 3 and (iy < 0 or iy >= ny));
                if (extra) {
                    g.set_real(c, ComplexType(1, 2));
                    added += 1;
                }
            }
        FML::SumOverTasks(&added);
        g.add_contribution_from_extra_cells();
        std::complex<double> sum = 0;
        for (auto && index : g.get_real_range())
            sum += std::complex<double>(g.get_real_from_index(index).real(), g.get_real_from_index(index).imag());
        double re = sum.real(), im = sum.imag();
        FML::SumOverTasks(&re);
        FML::SumOverTasks(&im);
        check(re == added and im == 2 * added, label + "add_contribution_from_extra_cells keeps the sum");
    }

    template <int N>
    void compare_slabs_pencils(int n, int p0, int p1) {
        const std::string label =
            std::to_string(N) + "D n=" + std::to_string(n) + " pencils " + std::to_string(p0) + "x" + std::to_string(p1) + ": ";
        const Waves<N> waves(n);
        CFFTWGrid<N> s(n);
        CPencilGrid<N> p(n, 0, 0, FML::PencilDomain::uniform(p0, p1));
        // Random-ish field: the plane waves times a non-band-limited factor
        auto f = [&](std::array<double, N> & x) { return waves(x) * ComplexType(1 + .3 * std::sin(7 * x[0] + x[N - 1]), .1); };
        s.fill_real_grid(f);
        p.fill_real_grid(f);
        s.fftw_forward();
        p.fftw_forward();
        const auto ms = all_modes(s), mp = all_modes(p);
        double d = 0;
        for (size_t i = 0; i < ms.size(); i++)
            d = std::max(d, double(std::abs(ms[i] - mp[i])));
        check(d < (sizeof(FML::GRID::FloatType) == 4 ? 1e-5 : 1e-14), label + "pencil modes = slab modes (" + num(d, 2) + ")");
    }
} // namespace

int main() {
    const int n = FML::NTasks % 3 == 0 ? 12 : 8;
    // Slabs (2 extra slices left, 1 right) and pencils on every process grid with the edges on cells
    test_grid<CFFTWGrid<3>, 3>(CFFTWGrid<3>(n, 2, 1), "CFFTWGrid 3D: ");
    test_grid<CFFTWGrid<2>, 2>(CFFTWGrid<2>(n, 1, 2), "CFFTWGrid 2D: ");
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        const int p1 = FML::NTasks / p0;
        if (FML::NTasks % p0 or n % p0 or n % p1)
            continue;
        const std::string tag = std::to_string(p0) + "x" + std::to_string(p1) + ": ";
        const auto domain = FML::PencilDomain::uniform(p0, p1);
        test_grid<CPencilGrid<3>, 3>(CPencilGrid<3>(n, 2, 1, domain), "CPencilGrid 3D " + tag);
        test_add_contribution<3>(CPencilGrid<3>(n, 1, 2, domain), "CPencilGrid 3D " + tag);
        compare_slabs_pencils<3>(n, p0, p1);
        compare_slabs_pencils<3>(n + 2 * FML::NTasks, p0, p1);
    }
    const auto slabs2d = FML::PencilDomain::uniform(FML::NTasks, 1);
    test_grid<CPencilGrid<2>, 2>(CPencilGrid<2>(n, 1, 2, slabs2d), "CPencilGrid 2D: ");
    test_add_contribution<2>(CPencilGrid<2>(n, 2, 1, slabs2d), "CPencilGrid 2D: ");
    compare_slabs_pencils<2>(n, FML::NTasks, 1);

    // Any mesh on the default pencils (edges on cells), the grid interface (make_grid on a slab domain)
    {
        CPencilGrid<3> odd(9);
        test_grid<CPencilGrid<3>, 3>(odd, "CPencilGrid 3D odd mesh 9 (default pencils): ");
        auto g = FML::GRID::make_grid<CPencilGrid<3>>(n, 0, 0, FML::Domain::uniform_slab(FML::NTasks));
        auto s = FML::GRID::make_grid<CFFTWGrid<3>>(n, 0, 0, FML::Domain::uniform_slab(FML::NTasks));
        check(g.get_local_range(0) == s.get_local_range(0) and g.get_local_range(1)[1] == n,
              "make_grid: CPencilGrid on a slab domain = the slabs of CFFTWGrid");
    }
    return FML::TESTS::summary("Complex grids (CFFTWGrid, CPencilGrid)");
}
