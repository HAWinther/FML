//==========================================================
// Tests for FFTWGrid and MPIGrid / MPIMultiGrid: FFTs with extra slices (FFTW_MEASURE and the
// 2D case where FFTW needs more memory than the local slices), move semantics, dump/load,
// ghost slice communication (1 task, thin slabs, non-periodic), index arithmetic, operators
//
// Run with 1, 2 or 4 MPI tasks: mpirun -np 4 ./test_grids
// A single test can be run with ./test_grids <name> (see the list in main)
//
// Bad input modes (must abort with a clear message):
//   ./test_grids truncated      : load_from_file of a truncated file
//                                 "[FFTWGrid::load_from_file] File is truncated or unreadable"
//   ./test_grids decomposition  : load_from_file of a file written with another domain decomposition
//                                 "[FFTWGrid::load_from_file] The grid in the file does not match the current domain decomposition"
//   ./test_grids dumpfail       : dump_to_file to a directory that does not exist
//                                 "[FFTWGrid::dump_to_file] Failed to open file for writing"
//==========================================================
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/MPIGrid/MPIGrid.h>
#include <FML/MPIGrid/MPIMultiGrid.h>

#include <climits>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <unistd.h>

template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;
template <int N, class T>
using MPIGrid = FML::GRID::MPIGrid<N, T>;
using FloatType = FML::GRID::FloatType;
using IndexInt = FML::GRID::IndexInt;

int fails = 0;
std::string sci(double x) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(1) << x;
    return os.str();
}
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
// True only if true on all tasks
bool all_tasks(bool ok) {
    int bad = ok ? 0 : 1;
    FML::SumOverTasks(&bad);
    return bad == 0;
}

// A file prefix in the temp directory that is the same on all tasks
std::string tmp_prefix(std::string label) {
    int pid = int(getpid());
#ifdef USE_MPI
    MPI_Bcast(&pid, 1, MPI_INT, 0, FML::communicator);
#endif
    return (std::filesystem::temp_directory_path() / ("fml_test_grids_" + label + "_" + std::to_string(pid))).string();
}

//===========================================================
// Fill main grid with random numbers, the extra slices with a sentinel value (slice dependent),
// transform r2c and c2r and check the extra slices are untouched and that we get back the field
//===========================================================
template <int N>
void fft_roundtrip_with_extra_slices(int Nmesh, int nleft, int nright, std::string label) {
    FFTWGrid<N> g(Nmesh, nleft, nright);
    const auto slice = g.get_ntot_real_slice_alloc();
    const auto Local_nx = g.get_local_nx();
    std::mt19937 gen(1234 + FML::ThisTask);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int s = -nleft; s < Local_nx + nright; s++) {
        FloatType * p = g.get_real_grid_by_slice(s);
        for (ptrdiff_t i = 0; i < slice; i++)
            p[i] = FloatType(1000 + s + 0.001 * (i % 97));
    }
    std::vector<FloatType> orig;
    for (auto && real_index : g.get_real_range()) {
        FloatType v = FloatType(dist(gen));
        g.set_real_from_index(real_index, v);
        orig.push_back(v);
    }
    auto sentinels_ok = [&]() {
        bool ok = true;
        for (int s = -nleft; s < 0; s++) {
            FloatType * p = g.get_real_grid_by_slice(s);
            for (ptrdiff_t i = 0; i < slice; i++)
                ok = ok and p[i] == FloatType(1000 + s + 0.001 * (i % 97));
        }
        for (int s = int(Local_nx); s < Local_nx + nright; s++) {
            FloatType * p = g.get_real_grid_by_slice(s);
            for (ptrdiff_t i = 0; i < slice; i++)
                ok = ok and p[i] == FloatType(1000 + s + 0.001 * (i % 97));
        }
        return ok;
    };

    FloatType * p0 = g.get_real_grid();
    g.fftw_r2c();
    const bool same_buffer = (p0 == g.get_real_grid());
    const bool ok_r2c = sentinels_ok();
    g.fftw_c2r();
    const bool ok_c2r = sentinels_ok();
    double err = 0.0;
    size_t k = 0;
    for (auto && real_index : g.get_real_range())
        err = std::max(err, std::fabs(double(g.get_real_from_index(real_index) - orig[k++])));
    FML::MaxOverTasks(&err);
    const double tol = sizeof(FloatType) == sizeof(float) ? 1e-5 : 1e-12;
    check(all_tasks(same_buffer), label + ": the grid buffer is not replaced by the transform");
    check(all_tasks(ok_r2c and ok_c2r), label + ": extra slices are preserved by r2c and c2r");
    check(err < tol, label + ": r2c + c2r gives back the field (err " + sci(err) + ")");
}

void test_measure() {
    // A size not used anywhere else so that the plan is made (with MEASURE) inside fftw_r2c / fftw_c2r
    FML::GRID::set_fftw_planner_flag(FFTW_MEASURE);
    fft_roundtrip_with_extra_slices<3>(4 * 7, 1, 2, "3D FFTW_MEASURE planner (Nmesh 28, extra slices 1+2)");
    fft_roundtrip_with_extra_slices<2>(4 * 9, 2, 1, "2D FFTW_MEASURE planner (Nmesh 36, extra slices 2+1)");
    FML::GRID::set_fftw_planner_flag(FFTW_ESTIMATE);
}

void test_2d_extra() {
    // In 2D the transposed dimension Nmesh/2+1 = 9 is not divisible by 2 or 4 tasks so FFTW
    // needs (and uses) more memory than our local slices: the excess overlaps the right extra slices
    fft_roundtrip_with_extra_slices<2>(16, 1, 2, "2D (Nmesh 16, extra slices 1+2)");
    fft_roundtrip_with_extra_slices<2>(16, 0, 1, "2D (Nmesh 16, extra slices 0+1)");
    fft_roundtrip_with_extra_slices<3>(16, 1, 1, "3D (Nmesh 16, extra slices 1+1)");
    fft_roundtrip_with_extra_slices<4>(8, 1, 1, "4D (Nmesh 8, extra slices 1+1)");
}

void test_index() {
    // MPIGrid index arithmetic must be done in 64 bit. We use a small grid and an x-coordinate that is far
    // outside of it so that the index exceeds INT_MAX (pure arithmetic, no memory access)
    MPIGrid<3, char> g(16, true);
    const int x = 9000000;
    std::array<int, 3> coord{x, 3, 5};
    const IndexInt expected = (IndexInt(x) * 16 + 3) * 16 + 5;
    std::array<int, 3> gcoord{x + g.get_xStartLocal(), 3, 5};
    const bool ok = g.index_from_coord(coord) == expected and g.index_from_globalcoord(gcoord) == expected;
    check(all_tasks(ok and expected > IndexInt(INT_MAX)),
          "MPIGrid index_from_coord / index_from_globalcoord in 64 bit (index " + std::to_string(expected) + ")");

    // And the normal round trips
    bool okrt = true;
    MPIGrid<3, double> h(8, true, 1, 1);
    for (IndexInt i = 0; i < h.get_NtotLocal(); i++) {
        okrt = okrt and h.index_from_coord(h.coord_from_index(i)) == i;
        okrt = okrt and h.index_from_globalcoord(h.globalcoord_from_index(i)) == i;
    }
    FFTWGrid<4> f(8);
    for (auto && index : f.get_fourier_range())
        okrt = okrt and f.get_index_fourier(f.get_fourier_coord_from_index(index)) == index;
    check(all_tasks(okrt), "MPIGrid and FFTWGrid<4> index <-> coord round trips");
}

void test_mpigrid_ops() {
    MPIGrid<2, double> a(8, true), b(8, true);
    for (IndexInt i = 0; i < a.get_NtotLocal(); i++) {
        a[i] = 1.0 + i;
        b[i] = 2.0;
    }
    MPIGrid<2, double> c = a + b;
    MPIGrid<2, double> d = a * b;
    MPIGrid<2, double> e = a - b;
    MPIGrid<2, double> f = a / b;
    bool ok = c.get_NtotLocal() == a.get_NtotLocal();
    for (IndexInt i = 0; ok and i < a.get_NtotLocal(); i++)
        ok = c[i] == 3.0 + i and d[i] == 2.0 * (1.0 + i) and e[i] == i - 1.0 and f[i] == 0.5 * (1.0 + i);
    // a and b are unchanged
    for (IndexInt i = 0; ok and i < a.get_NtotLocal(); i++)
        ok = a[i] == 1.0 + i and b[i] == 2.0;
    check(all_tasks(ok), "MPIGrid a+b, a*b, a-b, a/b (grid op grid)");
}

void test_move_and_free() {
    FFTWGrid<3> a(16, 1, 1);
    for (auto && real_index : a.get_real_range())
        a.set_real_from_index(real_index, FloatType(real_index));
    FloatType * pa = a.get_real_grid();

    FFTWGrid<3> b(std::move(a));
    bool ok = not bool(a) and a.get_nmesh() == 0 and a.get_local_nx() == 0 and a.get_ntot_fourier_alloc() == 0 and
              a.get_n_extra_slices_left() == 0 and b.get_real_grid() == pa and b.get_nmesh() == 16 and
              b.get_n_extra_slices_right() == 1;
    for (auto && real_index : b.get_real_range())
        ok = ok and b.get_real_from_index(real_index) == FloatType(real_index);
    check(all_tasks(ok), "FFTWGrid move constructor takes over the buffer and leaves the source empty");

    FFTWGrid<3> c(8);
    c = std::move(b);
    ok = not bool(b) and b.get_nmesh() == 0 and c.get_real_grid() == pa and c.get_nmesh() == 16;
    // The moved-from grid can be reused
    b = FFTWGrid<3>(8, 0, 1);
    ok = ok and bool(b) and b.get_nmesh() == 8 and b.get_real_grid() != c.get_real_grid();
    check(all_tasks(ok), "FFTWGrid move assignment takes over the buffer and leaves the source empty");

    // Copies are deep
    FFTWGrid<3> d = c;
    d.set_real_from_index(0, -1.0);
    check(all_tasks(d.get_real_grid() != c.get_real_grid() and c.get_real_from_index(0) == 0.0),
          "FFTWGrid copy is a deep copy");

    c.free();
    check(all_tasks(not bool(c) and c.get_nmesh() == 0 and c.get_local_nx() == 0 and c.get_ntot_fourier_alloc() == 0),
          "FFTWGrid::free releases the memory and resets the grid");

    MPIGrid<3, double> m(8, true, 1, 1);
    m.free();
    check(all_tasks(m.get_N() == 0 and m.get_NtotLocal() == 0 and m.get_n_extra_slices_left() == 0),
          "MPIGrid::free resets the grid");
    FML::GRID::MPIMultiGrid<3, double> mg(8, 2, true, 1, 1);
    mg.free();
    check(all_tasks(mg.get_Nlevel() == 0), "MPIMultiGrid::free resets the grid");
}

void test_status_and_nan() {
    FFTWGrid<3> g(8, 1, 1);
    g.fill_fourier_grid(0.0);
    const bool fourier = not g.get_grid_status_real();
    g.fill_real_grid(1.0);
    const bool real = g.get_grid_status_real();
    check(all_tasks(fourier and real), "fill_fourier_grid / fill_real_grid set the grid status");

    bool ok = not g.nan_in_grids();
    auto set_last = [&](FloatType v) {
        // The last cell of the right extra slice
        g.get_real_grid_by_slice(int(g.get_local_nx()))[g.get_ntot_real_slice_alloc() - 1] = v;
    };
    set_last(std::numeric_limits<FloatType>::infinity());
    ok = ok and g.nan_in_grids();
    set_last(-std::numeric_limits<FloatType>::infinity());
    ok = ok and g.nan_in_grids();
    set_last(std::numeric_limits<FloatType>::quiet_NaN());
    ok = ok and g.nan_in_grids();
    set_last(1.0);
    ok = ok and not g.nan_in_grids();
    check(all_tasks(ok), "nan_in_grids detects NaN and +-Inf (also in the extra slices)");
}

void test_dump_load() {
    const std::string prefix = tmp_prefix("dump");
    FFTWGrid<3> g(16, 1, 2);
    for (auto && real_index : g.get_real_range())
        g.set_real_from_index(real_index, FloatType(1.0 + real_index + 1000 * FML::ThisTask));
    g.get_real_grid_by_slice(-1)[3] = 42.0;
    g.dump_to_file(prefix);
    FFTWGrid<3> h;
    h.load_from_file(prefix);
    std::remove((prefix + "." + std::to_string(FML::ThisTask)).c_str());

    bool ok = h.get_nmesh() == 16 and h.get_local_nx() == g.get_local_nx() and
              h.get_local_x_start() == g.get_local_x_start() and h.get_n_extra_slices_right() == 2 and
              h.get_real_grid_by_slice(-1)[3] == FloatType(42.0);
    IndexInt ncells = 0;
    for (auto && real_index : h.get_real_range()) {
        ok = ok and h.get_real_from_index(real_index) == g.get_real_from_index(real_index);
        ncells++;
    }
    ok = ok and ncells == g.get_local_nx() * 16 * 16;
    check(all_tasks(ok), "dump_to_file + load_from_file into an empty grid gives the same grid");
}

// The bad input modes: these must abort
void bad_input(std::string mode) {
    const std::string prefix = tmp_prefix(mode);
    const std::string filename = prefix + "." + std::to_string(FML::ThisTask);
    FFTWGrid<3> g(16, 1, 1);
    if (mode == "dumpfail") {
        g.dump_to_file("/nonexistent_dir_fml_test_grids/grid");
        return;
    }
    g.dump_to_file(prefix);
    std::string s;
    {
        std::ifstream in(filename, std::ios::binary);
        s = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }
    if (mode == "truncated") {
        s = s.substr(0, s.size() - 100);
    } else if (mode == "decomposition") {
        // Header: 4 ints then Local_nx, Local_x_start (ptrdiff_t). Shift the start
        ptrdiff_t Local_x_start;
        std::memcpy(&Local_x_start, &s[4 * sizeof(int) + sizeof(ptrdiff_t)], sizeof(ptrdiff_t));
        Local_x_start += 1;
        std::memcpy(&s[4 * sizeof(int) + sizeof(ptrdiff_t)], &Local_x_start, sizeof(ptrdiff_t));
    }
    {
        std::ofstream out(filename, std::ios::binary);
        out.write(s.data(), s.size());
    }
    FFTWGrid<3> h;
    h.load_from_file(prefix);
    std::remove(filename.c_str());
}

void test_mpigrid_ghosts() {
    // Periodic: ghosts are the periodic neighbors (also with one task)
    for (int nextra : {1, 3}) {
        // nextra = 3 is more than the local slices with 2 and 4 tasks (thin slabs)
        const int N = 4;
        MPIGrid<2, double> g(N, true, nextra, nextra);
        for (IndexInt i = 0; i < g.get_NtotLocal(); i++) {
            auto gc = g.globalcoord_from_index(i);
            g[i] = 100 * gc[0] + gc[1];
        }
        g.communicate_boundaries();
        bool ok = true;
        const IndexInt nperslice = N;
        for (int s = -nextra; s < g.get_NLocal() + nextra; s++) {
            int gx = ((g.get_xStartLocal() + s) % N + N) % N;
            for (int iy = 0; iy < N; iy++)
                ok = ok and g[s * nperslice + iy] == 100 * gx + iy;
        }
        check(all_tasks(ok), "MPIGrid periodic ghosts are the periodic neighbors (" + std::to_string(nextra) +
                                 " extra slices, NLocal = " + std::to_string(g.get_NLocal()) + ")");
    }

    // Non-periodic: ghosts outside the box are not touched, the others are the neighbors
    {
        const int N = 8;
        MPIGrid<2, double> g(N, false, 1, 1);
        for (IndexInt i = -N; i < g.get_NtotLocal() + N; i++)
            g[i] = -7.0;
        for (IndexInt i = 0; i < g.get_NtotLocal(); i++)
            g[i] = 100 * g.globalcoord_from_index(i)[0];
        g.communicate_boundaries();
        const int xl = g.get_xStartLocal() - 1, xr = g.get_xStartLocal() + g.get_NLocal();
        const double expl = xl < 0 ? -7.0 : 100.0 * xl;
        const double expr = xr >= N ? -7.0 : 100.0 * xr;
        bool ok = true;
        for (int iy = 0; iy < N; iy++)
            ok = ok and g[-N + iy] == expl and g[g.get_NtotLocal() + iy] == expr;
        check(all_tasks(ok), "MPIGrid non-periodic ghosts: neighbors inside the box, untouched outside");
    }

    // Non-periodic with fewer slices than tasks (the last tasks have no cells): the right edge of the box is the
    // last task with cells, which must not get the wrapped slice from the first task
    if (FML::NTasks > 2) {
        const int N = FML::NTasks - 1;
        MPIGrid<2, double> g(N, false, 1, 1);
        bool ok = true;
        if (g.get_NLocal() > 0) {
            for (IndexInt i = -N; i < g.get_NtotLocal() + N; i++)
                g[i] = -7.0;
            for (IndexInt i = 0; i < g.get_NtotLocal(); i++)
                g[i] = 100 * g.globalcoord_from_index(i)[0];
        }
        g.communicate_boundaries();
        if (g.get_NLocal() > 0) {
            const int xl = g.get_xStartLocal() - 1, xr = g.get_xStartLocal() + g.get_NLocal();
            const double expl = xl < 0 ? -7.0 : 100.0 * xl;
            const double expr = xr >= N ? -7.0 : 100.0 * xr;
            for (int iy = 0; iy < N; iy++)
                ok = ok and g[-N + iy] == expl and g[g.get_NtotLocal() + iy] == expr;
        }
        check(all_tasks(ok), "MPIGrid non-periodic ghosts with tasks without cells (N = NTasks - 1)");
    }

    // send_slice gives one slice (and with one task the slice itself)
    {
        MPIGrid<3, double> g(8, true, 1, 1);
        for (IndexInt i = 0; i < g.get_NtotLocal(); i++)
            g[i] = g.globalcoord_from_index(i)[0];
        std::vector<double> recv;
        g.send_slice_right(0, recv);
        const int from = (g.get_xStartLocal() - g.get_NLocal() + 8) % 8;
        bool ok = recv.size() == size_t(64);
        for (auto v : recv)
            ok = ok and v == from;
        check(all_tasks(ok), "MPIGrid send_slice_right receives one slice of the left task");
    }
}

void test_fftwgrid_thin_slabs() {
    // Nmesh 4: with 2 and 4 tasks there are fewer local slices than extra slices
    for (int nextra : {1, 3, 5}) {
        const int Nmesh = 4;
        FFTWGrid<3> g(Nmesh, nextra, nextra);
        for (auto && real_index : g.get_real_range()) {
            auto coord = g.get_coord_from_index(real_index);
            g.set_real_from_index(real_index, FloatType(100 * (coord[0] + g.get_local_x_start()) + coord[1]));
        }
        g.communicate_boundaries();
        bool ok = true;
        for (int s = -nextra; s < g.get_local_nx() + nextra; s++) {
            const int gx = int(((g.get_local_x_start() + s) % Nmesh + Nmesh) % Nmesh);
            for (int iy = 0; iy < Nmesh; iy++)
                for (int iz = 0; iz < Nmesh; iz++)
                    ok = ok and g.get_real({s, iy, iz}) == FloatType(100 * gx + iy);
        }
        check(all_tasks(ok), "FFTWGrid ghosts are the periodic neighbors (" + std::to_string(nextra) +
                                 " extra slices, Local_nx = " + std::to_string(g.get_local_nx()) + ")");
    }
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";

    if (mode == "truncated" or mode == "decomposition" or mode == "dumpfail") {
        bad_input(mode);
        if (FML::ThisTask == 0)
            std::cout << "[FAIL] bad input mode " << mode << " did not abort\n";
        return 1;
    }

    const std::vector<std::pair<std::string, void (*)()>> tests = {
        {"measure", test_measure},
        {"extra2d", test_2d_extra},
        {"index", test_index},
        {"ops", test_mpigrid_ops},
        {"move", test_move_and_free},
        {"status", test_status_and_nan},
        {"dump", test_dump_load},
        {"ghosts", test_mpigrid_ghosts},
        {"thin", test_fftwgrid_thin_slabs},
    };
    bool found = false;
    for (auto & t : tests) {
        if (mode == "all" or mode == t.first) {
            t.second();
            found = true;
        }
    }
    if (not found) {
        if (FML::ThisTask == 0)
            std::cout << "Unknown mode " << mode << "\n";
        return 1;
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All grid tests passed\n" : std::to_string(fails) + " grid test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
