//==========================================================
// Tests for grid shifting, interlacing, NUFFT and spectral sheet interpolation
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Interpolation/SheetInterpolation.h>
#include <FML/NUFFT/NUFFT.h>
#include <FML/Smoothing/SmoothingFourier.h>

#include <complex>
#include <climits>
#include <limits>
#include <iomanip>
#include <sstream>
#include <random>

template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;
using FloatType = FML::GRID::FloatType;
using cplx = std::complex<double>;

template <int N>
struct Particle {
    double pos[N];
    double q[N];
    double * get_pos() { return pos; }
    double * get_q() { return q; }
    constexpr int get_ndim() const { return N; }
};

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

// The same random particles on all tasks; keep the ones in the local domain
template <int N>
std::vector<Particle<N>> random_particles(size_t n, unsigned int seed, std::vector<Particle<N>> * all = nullptr) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<Particle<N>> local;
    for (size_t i = 0; i < n; i++) {
        Particle<N> p;
        for (int idim = 0; idim < N; idim++)
            p.pos[idim] = p.q[idim] = dist(gen);
        if (all)
            all->push_back(p);
        if (p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain)
            local.push_back(p);
    }
    return local;
}

// Global integer wavevector -> index in a full (not half) complex array
template <int N>
size_t full_index(const std::array<int, N> & m, int nmesh) {
    size_t index = 0;
    for (int idim = 0; idim < N; idim++)
        index = index * nmesh + ((m[idim] % nmesh) + nmesh) % nmesh;
    return index;
}

// Direct sum F(m) = (1/n) sum_p exp(-i k x_p) for all modes of an nmesh^N grid
template <int N>
std::vector<cplx> direct_type1(const std::vector<Particle<N>> & all, int nmesh) {
    std::vector<cplx> F(FML::power(nmesh, N), 0.0);
    for (size_t index = 0; index < F.size(); index++) {
        std::array<int, N> m;
        size_t rest = index;
        for (int idim = N - 1; idim >= 0; idim--) {
            int mi = rest % nmesh;
            rest /= nmesh;
            m[idim] = mi <= nmesh / 2 ? mi : mi - nmesh;
        }
        cplx sum = 0.0;
        for (auto & p : all) {
            double kx = 0.0;
            for (int idim = 0; idim < N; idim++)
                kx += 2.0 * M_PI * m[idim] * p.pos[idim];
            sum += std::polar(1.0, -kx);
        }
        F[index] = sum / double(all.size());
    }
    return F;
}

template <int N>
void test_nufft(int nmesh, size_t npart, double epsilon, double sigma = 2.0) {
    std::vector<Particle<N>> all;
    auto part = random_particles<N>(npart, 1234, &all);
    const double norm = 1.0 / double(npart);

    // Type 1 against the direct sum
    FML::NUFFT::ESKernel kernel(epsilon, sigma);
    FFTWGrid<N> grid(nmesh);
    std::vector<double> values(part.size(), 1.0);
    FML::NUFFT::nufft_type1<N>(part.data(), part.size(), values, norm, grid, kernel);
    auto exact = direct_type1<N>(all, nmesh);
    double maxerr = 0.0, maxval = 0.0;
    for (auto && fourier_index : grid.get_fourier_range()) {
        auto m = grid.get_fourier_integer_wavevector_from_index(fourier_index);
        bool nyquist = false;
        for (int idim = 0; idim < N; idim++)
            nyquist = nyquist or std::abs(m[idim]) == nmesh / 2;
        cplx value = grid.get_fourier_from_index(fourier_index);
        cplx ref = nyquist ? 0.0 : exact[full_index<N>(m, nmesh)];
        maxerr = std::max(maxerr, std::abs(value - ref));
        maxval = std::max(maxval, std::abs(ref));
    }
    FML::MaxOverTasks(&maxerr);
    FML::MaxOverTasks(&maxval);
    check(maxerr < 10 * epsilon * maxval,
          "NUFFT type 1 (" + std::to_string(N) + "D, eps = " + sci(epsilon) + ", sigma = " + sci(sigma) +
              ") matches direct sum: max rel err " + sci(maxerr / maxval));

    // Type 2: evaluate the field from the grid above at the particle positions and compare to the direct sum
    // f(x) = sum_m F(m) e^{ikx} (each task sums its own modes for all particles, then reduce)
    std::vector<double> fdirect(all.size(), 0.0);
    for (auto && fourier_index : grid.get_fourier_range()) {
        auto m = grid.get_fourier_integer_wavevector_from_index(fourier_index);
        cplx value = grid.get_fourier_from_index(fourier_index);
        const double weight = (m[N - 1] == 0 or m[N - 1] == nmesh / 2) ? 1.0 : 2.0;
        for (size_t ip = 0; ip < all.size(); ip++) {
            double kx = 0.0;
            for (int idim = 0; idim < N; idim++)
                kx += 2.0 * M_PI * m[idim] * all[ip].pos[idim];
            fdirect[ip] += weight * (value * std::polar(1.0, kx)).real();
        }
    }
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, fdirect.data(), int(fdirect.size()), MPI_DOUBLE, MPI_SUM, FML::communicator);
#endif
    std::vector<FloatType> fnufft;
    FML::NUFFT::nufft_type2<N>(grid, part.data(), part.size(), fnufft, kernel);
    maxerr = 0.0;
    maxval = 0.0;
    size_t ilocal = 0;
    for (size_t ip = 0; ip < all.size(); ip++) {
        if (not(all[ip].pos[0] >= FML::xmin_domain and all[ip].pos[0] < FML::xmax_domain))
            continue;
        maxerr = std::max(maxerr, std::abs(fnufft[ilocal++] - fdirect[ip]));
        maxval = std::max(maxval, std::abs(fdirect[ip]));
    }
    FML::MaxOverTasks(&maxerr);
    FML::MaxOverTasks(&maxval);
    check(maxerr < 10 * epsilon * maxval,
          "NUFFT type 2 (" + std::to_string(N) + "D, eps = " + sci(epsilon) + ", sigma = " + sci(sigma) +
              ") matches direct sum: max rel err " + sci(maxerr / maxval));
}

void test_chunked_exchange() {
    struct Record {
        unsigned long long source, dest, index;
    };
    using namespace FML::COMMUNICATION;
    const auto capacity = detail::records_per_chunk<Record>(std::numeric_limits<size_t>::max());
    check(capacity == size_t(INT_MAX) / sizeof(Record) && capacity * sizeof(Record) <= size_t(INT_MAX) &&
              detail::records_per_chunk<Record>(sizeof(Record) - 1) == 0 &&
              detail::chunk_count(capacity + 1, capacity) == 2 &&
              detail::chunk_count(std::numeric_limits<unsigned long long>::max(), 1) ==
                  std::numeric_limits<unsigned long long>::max(),
          "record chunk arithmetic handles byte counts beyond INT_MAX without large allocations");
    for (int pattern = 0; pattern < 3; ++pattern)
        for (size_t bytes : {sizeof(Record), 3 * sizeof(Record) + 7, size_t(INT_MAX) + 1}) {
            auto count = [pattern](int source, int dest) -> size_t {
                if (pattern == 0)
                    return 0;
                if (pattern == 1)
                    return source == 0 && dest == FML::NTasks - 1 ? 17 : 0;
                return (source + dest) % 3 == 0 ? 0 : size_t(7 + 3 * source + dest);
            };
            std::vector<std::vector<Record>> send(FML::NTasks);
            std::vector<std::vector<char>> seen(FML::NTasks);
            for (int task = 0; task < FML::NTasks; ++task) {
                for (size_t i = 0; i < count(FML::ThisTask, task); ++i)
                    send[task].push_back({static_cast<unsigned long long>(FML::ThisTask),
                                          static_cast<unsigned long long>(task), i});
                seen[task].resize(count(task, FML::ThisTask), 0);
            }
            bool ok = true;
            exchange_records_chunked(send, [&](const Record * records, size_t n) {
                for (size_t i = 0; i < n; ++i) {
                    const auto & r = records[i];
                    if (r.dest != static_cast<unsigned long long>(FML::ThisTask) ||
                        r.source >= seen.size() || r.index >= seen[r.source].size()) {
                        ok = false;
                        continue;
                    }
                    if (seen[r.source][r.index]++)
                        ok = false;
                }
            }, bytes + size_t(FML::ThisTask) * sizeof(Record));
            for (const auto & source : seen)
                for (char n : source)
                    ok = ok && n == 1;
            for (const auto & peer : send)
                ok = ok && peer.empty() && peer.capacity() == 0;
            int bad = !ok;
            FML::MaxOverTasks(&bad);
            check(!bad, "chunked exchange delivers every record exactly once, pattern " + std::to_string(pattern) +
                            ", byte limit " + std::to_string(bytes));
        }
}

template <int N>
void test_mode_redistribution(int source_mesh, int dest_mesh) {
    FFTWGrid<N> source(source_mesh), dest(dest_mesh);
    auto value = [](const std::array<int, N> & m) {
        double real = 1, imag = 3;
        for (int d = 0; d < N; ++d) {
            real += (d + 1) * m[d];
            imag += (d + 2) * m[d] * m[d];
        }
        return FML::GRID::ComplexType(real, imag);
    };
    std::function<double(const std::array<int, N> &)> factor = [](const std::array<int, N> & m) {
        double scale = 1;
        for (auto k : m)
            scale += 0.25 * k * k;
        return scale;
    };
    source.set_grid_status_real(false);
    for (auto index : source.get_fourier_range())
        source.set_fourier_from_index(index, value(source.get_fourier_integer_wavevector_from_index(index)));
    dest.fill_fourier_grid(FML::GRID::ComplexType(123, 456));
    FML::NUFFT::copy_fourier_modes<N>(source, dest, factor);
    const int cutoff = std::min(source_mesh, dest_mesh) / 2;
    int bad = dest.get_grid_status_real();
    for (auto index : dest.get_fourier_range()) {
        const auto m = dest.get_fourier_integer_wavevector_from_index(index);
        bool keep = true;
        for (auto k : m)
            keep = keep && std::abs(k) < cutoff;
        const auto expected = keep ? value(m) * FloatType(factor(m)) : FML::GRID::ComplexType(0);
        bad = bad || dest.get_fourier_from_index(index) != expected;
    }
    for (auto index : source.get_fourier_range())
        bad = bad || source.get_fourier_from_index(index) != value(source.get_fourier_integer_wavevector_from_index(index));
    FML::MaxOverTasks(&bad);
    check(!bad, "NUFFT mode redistribution preserves exact coefficients, factor and cutoff: " + std::to_string(N) +
                    "D " + std::to_string(source_mesh) + " -> " + std::to_string(dest_mesh));
}

void test_redistribution_suite() {
    test_chunked_exchange();
    for (const auto meshes : {std::array<int, 2>{8, 16}, {16, 8}, {12, 8}, {8, 8}, {7, 11}, {2, 2}}) {
        // FFTWGrid intentionally requires divisibility by the MPI task count.
        if (meshes[0] % FML::NTasks || meshes[1] % FML::NTasks)
            continue;
        test_mode_redistribution<2>(meshes[0], meshes[1]);
        test_mode_redistribution<3>(meshes[0], meshes[1]);
    }
}

int main(int argc, char ** argv) {
#ifdef USE_MPI
    std::vector<std::vector<int>> outgoing(FML::NTasks);
    for (int t = 0; t < FML::NTasks; ++t)
        if ((FML::ThisTask + t) % 2)
            outgoing[t].push_back(100 * FML::ThisTask + t);
    outgoing[FML::ThisTask].push_back(-1);
    std::vector<int> expected;
    for (int t = 0; t < FML::NTasks; ++t) {
        if ((t + FML::ThisTask) % 2)
            expected.push_back(100 * t + FML::ThisTask);
        if (t == FML::ThisTask)
            expected.push_back(-1);
    }
    check(FML::INTERPOLATION::detail::alltoallv_elements(outgoing) == expected,
          "checked sheet exchange preserves peer order and handles empty edges");
    bool rejected_layout = false;
    std::vector<std::vector<int>> invalid(FML::ThisTask == 0 ? 0 : FML::NTasks);
    try {
        FML::INTERPOLATION::detail::alltoallv_elements(invalid);
    } catch (const std::overflow_error &) {
        rejected_layout = true;
    }
    check(rejected_layout, "one-rank invalid sheet layout fails collectively before indexing");
#endif

    if (argc > 1 && std::string(argv[1]) == "chunk_limit") {
        std::vector<std::vector<int>> send(FML::NTasks);
        FML::COMMUNICATION::exchange_records_chunked(send, [](const int *, size_t) {},
                                                    FML::ThisTask == 0 ? sizeof(int) - 1 : sizeof(int));
        return 1; // The invalid limit must terminate collectively before reaching here.
    }
    test_redistribution_suite();
    if (argc > 1 && std::string(argv[1]) == "redistribution")
        return fails == 0 ? 0 : 1;

    //===========================================================
    // 1. shift_grid_fourier: exact for band-limited fields, cos(k s) at Nyquist
    //===========================================================
    {
        const int n = 16;
        const std::array<double, 3> s = {0.0123, -0.0456, 0.31};
        auto f = [](std::array<double, 3> x) {
            return std::cos(2 * M_PI * (3 * x[0] - 2 * x[1] + x[2]) + 0.3) + 0.5 * std::sin(2 * M_PI * 7 * x[1]) +
                   0.2 * std::cos(2 * M_PI * (8 * x[0] + 5 * x[2])); // last term: Nyquist in x
        };
        auto fshift = [&](std::array<double, 3> x) {
            auto y = x;
            for (int i = 0; i < 3; i++)
                y[i] += s[i];
            // The Nyquist term becomes cos(pi n x) cos(pi n s_x) ... * shifted z-part
            return std::cos(2 * M_PI * (3 * y[0] - 2 * y[1] + y[2]) + 0.3) + 0.5 * std::sin(2 * M_PI * 7 * y[1]) +
                   0.2 * std::cos(2 * M_PI * 8 * x[0]) * std::cos(2 * M_PI * 8 * s[0]) * std::cos(2 * M_PI * 5 * y[2]);
        };
        FFTWGrid<3> grid(n);
        grid.fill_real_grid([&](std::array<double, 3> & x) { return f(x); });
        FML::GRID::shift_grid_real<3>(grid, s);
        double maxerr = 0.0;
        for (auto && real_index : grid.get_real_range()) {
            auto x = grid.get_real_position(grid.get_coord_from_index(real_index));
            maxerr = std::max(maxerr, std::abs(grid.get_real_from_index(real_index) - fshift(x)));
        }
        FML::MaxOverTasks(&maxerr);
        check(maxerr < 1e-10, "shift_grid_fourier is exact for a band-limited field (err " + sci(maxerr) + ")");
    }

    //===========================================================
    // 2. shift_particles
    //===========================================================
    {
        std::vector<Particle<3>> p(1);
        p[0].pos[0] = 0.9;
        p[0].pos[1] = 0.95;
        p[0].pos[2] = 0.05;
        FML::PARTICLE::shift_particles<Particle<3>, 3>(p.data(), 1, {0.2, 0.1, -0.1}, +1.0, 2);
        bool ok = std::abs(p[0].pos[0] - 1.1) < 1e-14 and std::abs(p[0].pos[1] - 0.05) < 1e-14 and
                  std::abs(p[0].pos[2] - 0.95) < 1e-14;
        FML::PARTICLE::shift_particles<Particle<3>, 3>(p.data(), 1, {0.2, 0.1, -0.1}, -1.0, 1);
        ok = ok and std::abs(p[0].pos[0] - 0.9) < 1e-14 and std::abs(p[0].pos[1] - 0.95) < 1e-14 and
             std::abs(p[0].pos[2] - 0.05) < 1e-14;
        check(ok, "shift_particles with and without wrapping in x");
    }

    //===========================================================
    // 3. Interlacing shifts: count and which aliases they cancel
    //===========================================================
    {
        // |sum_j exp(i 2pi n.s_j/dx)| / M for the alias families n = (1,0,0), (1,1,0), (1,1,1)
        auto residual = [](const std::vector<std::array<double, 3>> & shifts, std::array<int, 3> n) {
            cplx sum = 0.0;
            for (auto & s : shifts)
                sum += std::polar(1.0, 2 * M_PI * (n[0] * s[0] + n[1] * s[1] + n[2] * s[2]));
            return std::abs(sum) / shifts.size();
        };
        bool ok = true;
        // order: expected cancellation of (100), (110), (111)
        std::vector<std::tuple<int, bool, bool, bool>> cases = {{2, true, false, true},
                                                                {4, true, true, false},
                                                                {8, true, true, true},
                                                                {16, true, true, true}};
        for (auto & [order, c100, c110, c111] : cases) {
            auto shifts = FML::INTERPOLATION::generate_interlacing_shifts<3>(order, 1.0);
            ok = ok and int(shifts.size()) == order;
            ok = ok and ((residual(shifts, {1, 0, 0}) < 1e-12) == c100);
            ok = ok and ((residual(shifts, {1, 1, 0}) < 1e-12) == c110);
            ok = ok and ((residual(shifts, {1, 1, 1}) < 1e-12) == c111);
            for (auto & s : shifts)
                for (auto x : s)
                    ok = ok and x >= 0.0 and x < 1.0;
        }
        bool threw = false;
        try {
            FML::INTERPOLATION::generate_interlacing_shifts<3>(3, 1.0);
        } catch (std::runtime_error &) {
            threw = true;
        }
        check(ok and threw, "generate_interlacing_shifts (orders 2, 4, 8, 16 in 3D; 3 is rejected)");
    }

    //===========================================================
    // 4. NUFFT type 1 and 2 against direct sums
    //===========================================================
    test_nufft<3>(16, 300, 1e-6);
    test_nufft<3>(16, 300, 1e-10);
    test_nufft<2>(32, 300, 1e-6);
    test_nufft<3>(16, 300, 1e-6, 1.5);

    //===========================================================
    // 5. Interlaced assignment converges to the exact (NUFFT) density with increasing order
    //===========================================================
    {
        const int n = 32;
        auto part = random_particles<3>(20000, 99);
        size_t NumPartTot = part.size();
        FML::SumOverTasks(&NumPartTot);
        FFTWGrid<3> exact(n);
        FML::NUFFT::particles_to_fourier_grid_nufft<3>(part.data(), part.size(), NumPartTot, exact, 1e-8);
        std::vector<double> err;
        for (int order : {1, 2, 4, 8, 64}) {
            FFTWGrid<3> grid(n);
            FML::INTERPOLATION::particles_to_fourier_grid_interlaced<3>(part.data(),
                                                                        part.size(),
                                                                        NumPartTot,
                                                                        grid,
                                                                        "PCS",
                                                                        order);
            FML::INTERPOLATION::deconvolve_window_function_fourier<3>(grid, "PCS");
            // rms relative error for k < k_Ny / 2
            double sum_err = 0.0, sum_ref = 0.0;
            for (auto && fourier_index : grid.get_fourier_range()) {
                auto m = grid.get_fourier_integer_wavevector_from_index(fourier_index);
                double m2 = 0.0;
                for (int idim = 0; idim < 3; idim++)
                    m2 += m[idim] * m[idim];
                if (m2 == 0.0 or m2 >= (n / 4) * (n / 4))
                    continue;
                sum_err +=
                    std::norm(grid.get_fourier_from_index(fourier_index) - exact.get_fourier_from_index(fourier_index));
                sum_ref += std::norm(exact.get_fourier_from_index(fourier_index));
            }
            FML::SumOverTasks(&sum_err);
            FML::SumOverTasks(&sum_ref);
            err.push_back(std::sqrt(sum_err / sum_ref));
        }
        if (FML::ThisTask == 0)
            std::cout << "       PCS, k < kNy/2: rms error vs exact for interlacing order 1,2,4,8,64: " << err[0] << " "
                      << err[1] << " " << err[2] << " " << err[3] << " " << err[4] << "\n";
        // Half-cell shifts cancel the odd alias images (the largest, n = (1,0,0)). After that the error is
        // dominated by the even images (n = (2,0,0), ...) which only quarter-cell shifts (order 4^3 = 64) cancel
        check(err[1] < 0.1 * err[0] and err[2] < 1.01 * err[1] and err[3] < 1.01 * err[1] and err[4] < 0.1 * err[3],
              "Interlacing of order 1, 2, 4, 8, 64 reduces aliasing as expected");
    }

    //===========================================================
    // 6. NUFFT force against a direct mode sum
    //===========================================================
    {
        const int n = 16;
        std::vector<Particle<3>> all;
        auto part = random_particles<3>(200, 7, &all);
        const double norm = 1.5;
        std::array<std::vector<FloatType>, 3> force;
        FML::NUFFT::compute_force_nufft<3>(part.data(), part.size(), all.size(), n, norm, force, 1e-8);
        auto delta = direct_type1<3>(all, n);
        double maxerr = 0.0, maxval = 0.0;
        size_t ilocal = 0;
        for (auto & p : all) {
            if (not(p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain))
                continue;
            std::array<double, 3> f{};
            for (int i0 = -n / 2 + 1; i0 < n / 2; i0++)
                for (int i1 = -n / 2 + 1; i1 < n / 2; i1++)
                    for (int i2 = -n / 2 + 1; i2 < n / 2; i2++) {
                        if (i0 == 0 and i1 == 0 and i2 == 0)
                            continue;
                        std::array<int, 3> m = {i0, i1, i2};
                        double k2 = 0.0, kx = 0.0;
                        for (int idim = 0; idim < 3; idim++) {
                            k2 += std::pow(2 * M_PI * m[idim], 2);
                            kx += 2 * M_PI * m[idim] * p.pos[idim];
                        }
                        cplx phi = -norm * delta[full_index<3>(m, n)] / k2 * std::polar(1.0, kx);
                        for (int idim = 0; idim < 3; idim++)
                            f[idim] += (cplx(0, 2 * M_PI * m[idim]) * phi).real();
                    }
            for (int idim = 0; idim < 3; idim++) {
                maxerr = std::max(maxerr, std::abs(force[idim][ilocal] - f[idim]));
                maxval = std::max(maxval, std::abs(f[idim]));
            }
            ilocal++;
        }
        FML::MaxOverTasks(&maxerr);
        FML::MaxOverTasks(&maxval);
        check(maxerr < 1e-6 * maxval, "NUFFT force matches the direct mode sum: max rel err " + sci(maxerr / maxval));
    }

    //===========================================================
    // 7. Spectral sheet interpolation of a band-limited displacement field
    //===========================================================
    {
        const int Np = 16;
        const std::array<double, 3> offset = {0.5, 0.5, 0.5};
        // Psi(q): a few plane waves with |m| < Np/2
        auto psi = [](const std::array<double, 3> & q, int idim) {
            const double a = 0.01;
            if (idim == 0)
                return a * std::sin(2 * M_PI * (q[0] + 2 * q[1]) + 0.1);
            if (idim == 1)
                return a * std::cos(2 * M_PI * (3 * q[1] - q[2]));
            return a * std::sin(2 * M_PI * (5 * q[2] + q[0]) + 1.0);
        };
        // Particles on the offset lattice, displaced by Psi
        std::vector<Particle<3>> part;
        for (int i = 0; i < Np * Np * Np; i++) {
            Particle<3> p;
            std::array<double, 3> q;
            for (int idim = 0, n = 1; idim < 3; idim++, n *= Np)
                q[idim] = ((i / n) % Np + offset[idim]) / Np;
            for (int idim = 0; idim < 3; idim++) {
                p.q[idim] = q[idim];
                p.pos[idim] = q[idim] + psi(q, idim);
                p.pos[idim] -= std::floor(p.pos[idim]);
            }
            if (p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain)
                part.push_back(p);
        }

        // Psi from the particles
        std::array<FFTWGrid<3>, 3> Psi;
        std::array<double, 3> lattice_offset;
        FML::INTERPOLATION::displacement_field_from_particles<3>(part.data(), part.size(), Np, Psi, lattice_offset);
        bool ok = true;
        for (int idim = 0; idim < 3; idim++)
            ok = ok and std::abs(lattice_offset[idim] - offset[idim]) < 1e-10;
        check(ok, "displacement_field_from_particles finds the lattice offset");

        // Tracers for an arbitrary shift must be at q + s + Psi(q + s)
        for (int idim = 0; idim < 3; idim++)
            Psi[idim].fftw_r2c();
        const std::array<double, 3> s = {0.3 / Np, 0.7 / Np, 0.5 / Np};
        std::vector<FML::INTERPOLATION::SheetTracer<3>> tracers;
        FML::INTERPOLATION::sheet_tracers<3>(Psi, lattice_offset, s, tracers);
        size_t ntracers = tracers.size();
        FML::SumOverTasks(&ntracers);
        // For each tracer find the closest expected position (search the nearby lattice nodes)
        double maxerr = 0.0;
        int outside = 0;
        for (auto & t : tracers) {
            if (not(t.pos[0] >= FML::xmin_domain and t.pos[0] < FML::xmax_domain))
                outside = 1;
            double best = 1e10;
            std::array<int, 3> i0;
            for (int idim = 0; idim < 3; idim++)
                i0[idim] = int(std::floor(t.pos[idim] * Np - offset[idim] - s[idim] * Np));
            for (int j = 0; j < 27; j++) {
                std::array<double, 3> q, x;
                for (int idim = 0, n = 1; idim < 3; idim++, n *= 3)
                    q[idim] = (i0[idim] + (j / n) % 3 - 1 + offset[idim]) / Np + s[idim];
                double d2 = 0.0;
                for (int idim = 0; idim < 3; idim++) {
                    x[idim] = q[idim] + psi(q, idim);
                    double d = t.pos[idim] - x[idim];
                    d -= std::round(d);
                    d2 += d * d;
                }
                best = std::min(best, d2);
            }
            maxerr = std::max(maxerr, std::sqrt(best));
        }
        FML::MaxOverTasks(&maxerr);
        FML::MaxOverTasks(&outside);
        check(ntracers == size_t(Np * Np * Np) and outside == 0 and maxerr < 1e-10,
              "sheet_tracers are at q + s + Psi(q + s) (max err " + sci(maxerr) + ")");

        // Sheet density with R = 2 on a 32^3 grid equals the density of the analytic tracers
        const int R = 2, nmesh = 32;
        FFTWGrid<3> density(nmesh);
        FML::INTERPOLATION::particles_to_fourier_grid_sheet<3>(part.data(), part.size(), Np, R, density, "CIC");
        std::vector<Particle<3>> analytic;
        for (int i = 0; i < FML::power(R * Np, 3); i++) {
            Particle<3> p;
            std::array<double, 3> q;
            for (int idim = 0, n = 1; idim < 3; idim++, n *= R * Np)
                q[idim] = ((i / n) % (R * Np) + R * offset[idim]) / double(R * Np);
            for (int idim = 0; idim < 3; idim++) {
                p.pos[idim] = q[idim] + psi(q, idim);
                p.pos[idim] -= std::floor(p.pos[idim]);
            }
            if (p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain)
                analytic.push_back(p);
        }
        FFTWGrid<3> density_ref(nmesh, 1, 2);
        FML::INTERPOLATION::particles_to_grid<3>(analytic.data(),
                                                 analytic.size(),
                                                 FML::power(R * Np, 3),
                                                 density_ref,
                                                 "CIC");
        density_ref.fftw_r2c();
        double maxdiff = 0.0, maxref = 0.0;
        for (auto && fourier_index : density.get_fourier_range()) {
            maxdiff = std::max(maxdiff,
                               std::abs(density.get_fourier_from_index(fourier_index) -
                                        density_ref.get_fourier_from_index(fourier_index)));
            maxref = std::max(maxref, std::abs(density_ref.get_fourier_from_index(fourier_index)));
        }
        FML::MaxOverTasks(&maxdiff);
        FML::MaxOverTasks(&maxref);
        check(maxdiff < 1e-8 * maxref, "particles_to_fourier_grid_sheet (R = 2) equals the analytic sheet density");

        // The same with the exact Fourier modes of the tracers (NUFFT)
        FFTWGrid<3> density_nufft(nmesh);
        FML::INTERPOLATION::particles_to_fourier_grid_sheet<3>(part.data(),
                                                               part.size(),
                                                               Np,
                                                               R,
                                                               density_nufft,
                                                               "NUFFT",
                                                               1e-8);
        FFTWGrid<3> density_nufft_ref(nmesh);
        FML::NUFFT::particles_to_fourier_grid_nufft<3>(analytic.data(),
                                                       analytic.size(),
                                                       FML::power(R * Np, 3),
                                                       density_nufft_ref,
                                                       1e-8);
        maxdiff = maxref = 0.0;
        for (auto && fourier_index : density_nufft.get_fourier_range()) {
            maxdiff = std::max(maxdiff,
                               std::abs(density_nufft.get_fourier_from_index(fourier_index) -
                                        density_nufft_ref.get_fourier_from_index(fourier_index)));
            maxref = std::max(maxref, std::abs(density_nufft_ref.get_fourier_from_index(fourier_index)));
        }
        FML::MaxOverTasks(&maxdiff);
        FML::MaxOverTasks(&maxref);
        check(maxdiff < 1e-6 * maxref,
              "particles_to_fourier_grid_sheet with NUFFT equals the NUFFT of the analytic "
              "sheet tracers (rel err " +
                  sci(maxdiff / maxref) + ")");
    }

    // sort_particles_by_position: the same order as a stable sort by block (x-major), particles moved whole, and the
    // density field unchanged up to round-off
    {
        const int nb = 16;
        auto part = random_particles<3>(100000, 13);
        for (auto & p : part)
            p.q[0] = &p - part.data(); // the original index
        auto key = [&](const Particle<3> & p) {
            long long k = 0;
            for (int idim = 0; idim < 3; idim++)
                k = k * nb + std::max(0, std::min(int(p.pos[idim] * nb), nb - 1));
            return k;
        };
        auto expected = part;
        std::stable_sort(expected.begin(), expected.end(), [&](const Particle<3> & a, const Particle<3> & b) {
            return key(a) < key(b);
        });
        auto sorted = part;
        FML::PARTICLE::sort_particles_by_position(sorted.data(), sorted.size(), nb);
        bool ok = sorted.size() == expected.size();
        for (size_t i = 0; ok and i < sorted.size(); i++)
            for (int idim = 0; idim < 3; idim++)
                ok = ok and sorted[i].pos[idim] == expected[i].pos[idim] and sorted[i].q[idim] == expected[i].q[idim];
        int nok = ok ? 1 : 0;
        FML::MinOverTasks(&nok);

        FFTWGrid<3> g1(64, 1, 2), g2(64, 1, 2);
        FML::INTERPOLATION::particles_to_grid<3, Particle<3>>(part.data(), part.size(), 100000, g1, "CIC");
        FML::INTERPOLATION::particles_to_grid<3, Particle<3>>(sorted.data(), sorted.size(), 100000, g2, "CIC");
        double maxdiff = 0.0;
        for (auto && index : g1.get_real_range())
            maxdiff =
                std::max(maxdiff, std::fabs(double(g1.get_real_from_index(index)) - g2.get_real_from_index(index)));
        FML::MaxOverTasks(&maxdiff);
        check(nok == 1 and maxdiff < 1e-12,
              "sort_particles_by_position = stable sort by block, density unchanged (" + sci(maxdiff) + ")");
    }

#ifdef USE_OMP
    // particles_to_grid with several threads (chunks of slices, no atomic adds): equal to the one-thread result up
    // to round-off, and identical for 2 and 3 threads when the chunked path is used (deterministic)
    {
        const int nmesh = 96;
        auto part = random_particles<3>(200000, 11);
        const int nthreads_max = omp_get_max_threads();
        bool ok = true;
        std::string info;
        for (std::string method : {"NGP", "CIC", "TSC", "PCS"}) {
            const int order = FML::INTERPOLATION::interpolation_order_from_name(method);
            auto nlr = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(method);
            std::vector<FFTWGrid<3>> grids;
            for (int nthreads : {1, 2, 3}) {
                omp_set_num_threads(nthreads);
                FFTWGrid<3> g(nmesh, nlr.first, nlr.second);
                FML::INTERPOLATION::particles_to_grid<3, Particle<3>>(part.data(), part.size(), 200000, g, method);
                grids.push_back(g);
            }
            omp_set_num_threads(nthreads_max);
            int nchunks = grids[0].get_local_nx() / (2 * order + 2);
            if (FML::NTasks == 1 and nchunks % 2 == 1)
                nchunks--;
            const bool chunked = nchunks >= 4;
            double maxdiff = 0.0, maxdiff23 = 0.0;
            for (auto && index : grids[0].get_real_range()) {
                maxdiff = std::max(
                    maxdiff,
                    std::fabs(double(grids[1].get_real_from_index(index)) - grids[0].get_real_from_index(index)));
                maxdiff23 = std::max(
                    maxdiff23,
                    std::fabs(double(grids[1].get_real_from_index(index)) - grids[2].get_real_from_index(index)));
            }
            FML::MaxOverTasks(&maxdiff);
            FML::MaxOverTasks(&maxdiff23);
            ok = ok and maxdiff < 1e-12 and (not chunked or maxdiff23 == 0.0);
            info += " " + method + ": " + sci(maxdiff) + (chunked ? " (2 = 3 threads: " + sci(maxdiff23) + ")" : "");
        }
        check(ok, "particles_to_grid with threads equals one thread, deterministic:" + info);
    }
#endif

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All tests passed\n" : std::to_string(fails) + " test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
