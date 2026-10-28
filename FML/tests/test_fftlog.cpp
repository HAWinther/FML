//==========================================================
// Tests for FML::SOLVERS::FFTLog (FFTLog/FFTLog.h) against analytic Hankel transforms:
//   xi_l^m(r) = int dk/(2 pi^2) k^m j_l(kr) P(k) with P(k) = exp(-k^2) and m = l + 2:
//   xi_l^{l+2}(r) = sqrt(pi) r^l exp(-r^2/4) / (2^(l+2) 2 pi^2)
//   - ell = 0, 2, 4 (with the bias q = -1 for ell = 2, 4) on the returned r-grid (the r-grid was one log-step
//     off and the bias q != 0 gave wrong results)
//   - many points (N = 8192 over 8 decades: the log-gamma of the coefficients must not overflow)
//   - P(k) -> xi(r) -> P(k) round trip, DiscreteHankelTransform with precomputed coefficients
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/FFTLog/FFTLog.h>
#include <FML/Global/Global.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

using namespace FML::SOLVERS::FFTLog;

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os.precision(3);
    os << x;
    return os.str();
}

DVector logspace(double xmin, double xmax, int n) {
    DVector x(n);
    for (int i = 0; i < n; i++)
        x[i] = xmin * std::pow(xmax / xmin, i / double(n - 1));
    return x;
}

double exact(int ell, double r) {
    return std::sqrt(M_PI) * std::pow(r, ell) * std::exp(-r * r / 4) / std::pow(2.0, ell + 2) / (2 * M_PI * M_PI);
}

// Max |xi - exact| / max|exact| for r in [rmin, rmax]
double error(int ell, const std::pair<DVector, DVector> & res, double rmin, double rmax) {
    double maxerr = 0.0, maxval = 0.0;
    for (size_t i = 0; i < res.first.size(); i++) {
        const double r = res.first[i];
        if (r < rmin or r > rmax)
            continue;
        maxerr = std::max(maxerr, std::fabs(res.second[i] - exact(ell, r)));
        maxval = std::max(maxval, std::fabs(exact(ell, r)));
    }
    return maxerr / maxval;
}

// Independent convolution reference with caller-supplied coefficients.
std::pair<DVector, CVector> reference(const DVector & r, const CVector & a, double mu, double q, double kr) {
    const int n = int(r.size());
    const double L = std::log(r.back() / r.front()) * n / (n - 1.0);
    auto u = ComputeCoefficients(n, mu, q, L, kr);
    auto input = a;
    CVector b(n);
    auto forward = fftw_plan_dft_1d(n, reinterpret_cast<fftw_complex *>(input.data()),
                                   reinterpret_cast<fftw_complex *>(b.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    auto reverse = fftw_plan_dft_1d(n, reinterpret_cast<fftw_complex *>(b.data()),
                                   reinterpret_cast<fftw_complex *>(b.data()), FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(forward);
    for (int i = 0; i < n; ++i) b[i] *= u[i] / double(n);
    fftw_execute(reverse);
    fftw_destroy_plan(forward);
    fftw_destroy_plan(reverse);
    std::reverse(b.begin(), b.end());
    DVector k(n);
    k[0] = kr * std::exp(-L + L / n) / r[0];
    for (int i = 1; i < n; ++i) k[i] = k[0] * std::exp(i * L / n);
    return {k, b};
}

template<class T> bool close_vectors(const std::vector<T> & a, const std::vector<T> & b) {
    if (a.size() != b.size()) return false;
    double scale = 0, diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(std::abs(a[i])) || !std::isfinite(std::abs(b[i]))) return false;
        scale = std::max(scale, std::abs(b[i]));
        diff = std::max(diff, std::abs(a[i] - b[i]));
    }
    return diff <= 2e-12 * std::max(scale, 1e-100);
}

template<class T> bool close_results(const std::pair<DVector, std::vector<T>> & a,
                                    const std::pair<DVector, std::vector<T>> & b) {
    return close_vectors(a.first, b.first) && close_vectors(a.second, b.second);
}

void test_workspace() {
    static_assert(!std::is_copy_constructible<Workspace>::value, "Workspace owns FFTW plans");
    Workspace workspace;
    check(workspace.size() == 0 && workspace.buffer_bytes() == 0, "workspace starts empty");
    for (int n : {128, 129}) {
        const auto r = logspace(1e-3, 1e3, n);
        CVector a(n);
        for (int i = 0; i < n; ++i) a[i] = CDouble(std::sin(i * .2), std::cos(i * .3));
        const auto original = a;
        for (double q : {0.0, -.4}) {
            auto result = DiscreteHankelTransform(r, a, workspace, 2.5, q, 1.2, false);
            check(close_results(result, reference(r, a, 2.5, q, 1.2)),
                  "workspace complex convolution matches independent reference, N=" + std::to_string(n) + ", q=" + num(q));
        }
        check(a == original, "transform preserves complex input");
    }
    check(workspace.statistics().plan_builds == 4 && workspace.statistics().coefficient_builds == 4,
          "odd/even lengths rebuild plans; bias changes only rebuild coefficients");
    auto r = logspace(1e-3, 1e3, 129);
    CVector a(r.size(), CDouble(1, .3));
    const auto before = workspace.statistics();
    auto retained = DiscreteHankelTransform(r, a, workspace, 2.5, -.4, 1.2, false);
    const auto snapshot = retained;
    a[3] *= 2;
    auto changed = DiscreteHankelTransform(r, a, workspace, 2.5, -.4, 1.2, false);
    check(close_results(changed, reference(r, a, 2.5, -.4, 1.2)) && retained == snapshot &&
              workspace.statistics().plan_builds == before.plan_builds &&
              workspace.statistics().coefficient_builds == before.coefficient_builds,
          "new amplitudes reuse plans/coefficients and returned results remain owned");
    // Each part of the coefficient key changes independently.
    size_t builds = workspace.statistics().coefficient_builds;
    auto compare_fresh = [&](double mu, double q, double kr, bool noring) {
        auto result = DiscreteHankelTransform(r, a, workspace, mu, q, kr, noring);
        return close_results(result, DiscreteHankelTransform(r, a, mu, q, kr, noring));
    };
    bool keys = compare_fresh(3.5, -.4, 1.2, false);
    keys &= compare_fresh(3.5, -.2, 1.2, false);
    keys &= compare_fresh(3.5, -.2, 1.4, false);
    keys &= compare_fresh(3.5, -.2, 1.4, true);
    r = logspace(1e-3, 1e4, 129);
    keys &= compare_fresh(3.5, -.2, 1.4, true);
    check(keys && workspace.statistics().coefficient_builds == builds + 5 &&
              workspace.statistics().plan_builds == 4, "all geometry keys invalidate coefficients without rebuilding plans");
    builds = workspace.statistics().coefficient_builds;
    auto unscaled = DiscreteHankelTransform(r, a, workspace, 3.5, -.2, 1.4, true);
    for (auto & x : r) x *= 2;
    auto scaled = DiscreteHankelTransform(r, a, workspace, 3.5, -.2, 1.4, true);
    bool scaling = close_vectors(unscaled.second, scaled.second);
    for (size_t i = 0; i < r.size(); ++i) scaling &= std::abs(2 * scaled.first[i] / unscaled.first[i] - 1) < 1e-13;
    check(scaling && workspace.statistics().coefficient_builds == builds, "absolute grid rescaling reuses coefficients and rescales dual grid");
    const double L = std::log(r.back() / r.front()) * r.size() / (r.size() - 1.0);
    auto u = ComputeCoefficients(int(r.size()), 1.5, .1, L, 1.3);
    auto external = DiscreteHankelTransform(r, a, workspace, 1.5, .1, 1.3, true, &u);
    check(close_results(external, reference(r, a, 1.5, .1, 1.3)), "external coefficients bypass no-ringing adjustment");
    for (auto & x : u) x *= 2;
    auto external2 = DiscreteHankelTransform(r, a, workspace, 1.5, .1, 1.3, true, &u);
    for (auto & x : external.second) x *= 2;
    check(close_results(external, external2) && workspace.statistics().coefficient_builds == builds,
          "modified external coefficients are read live and not cached");
    auto internal = DiscreteHankelTransform(r, a, workspace, 3.5, -.2, 1.4, true);
    check(close_results(internal, scaled) && workspace.statistics().coefficient_builds == builds,
          "external coefficients preserve the internal cache");
    Workspace moved(std::move(workspace));
    workspace.prepare(16); // move-assign over existing plans, releasing the old ownership
    workspace = std::move(moved);
    check(moved.size() == 0 && close_results(internal, DiscreteHankelTransform(r, a, workspace, 3.5, -.2, 1.4, true)),
          "workspace ownership survives move construction and assignment");
    moved.prepare(8);
    check(moved.size() == 8, "moved-from workspace can be prepared again");
    workspace.clear();
    check(workspace.size() == 0 && workspace.statistics().plan_builds == 0, "clear releases buffers and resets statistics");
    Workspace measured(FFTW_MEASURE);
    const auto saved_r = r;
    const auto saved_a = a;
    check(close_results(reference(r, a, 1.5, .1, 1.3),
                        DiscreteHankelTransform(r, a, measured, 1.5, .1, 1.3, false)) && r == saved_r && a == saved_a,
          "FFTW_MEASURE plans on owned scratch and preserves caller input");
    const auto small_r = logspace(.01, 100, 32);
    const CVector small_a(32, CDouble(.5, -.2));
    check(close_results(reference(small_r, small_a, .5, 0, 1),
                        DiscreteHankelTransform(small_r, small_a, measured, .5, 0, 1, false)) &&
              small_a == CVector(32, CDouble(.5, -.2)) && measured.statistics().plan_builds == 4 &&
              measured.buffer_bytes() == 64 * sizeof(fftw_complex),
          "FFTW_MEASURE resizing preserves input and updates owned buffer size");
    auto rejects = [](auto operation) {
        try { operation(); } catch (const std::invalid_argument &) { return true; }
        return false;
    };
    bool invalid = rejects([&] { workspace.prepare(1); });
    invalid &= rejects([&] { workspace.prepare(size_t(std::numeric_limits<int>::max()) + 1); });
    invalid &= rejects([&] { DiscreteHankelTransform({}, {}, workspace, .5); });
    invalid &= rejects([&] { DiscreteHankelTransform({1}, {CDouble(1)}, workspace, .5); });
    invalid &= rejects([&] { DiscreteHankelTransform(r, {}, workspace, .5); });
    CVector short_u(2);
    invalid &= rejects([&] { DiscreteHankelTransform(r, a, workspace, .5, 0, 1, true, &short_u); });
    invalid &= rejects([&] { DiscreteHankelTransform({0, 1}, CVector(2), workspace, .5); });
    invalid &= rejects([&] { DiscreteHankelTransform(r, a, workspace, .5, 0, -1); });
    invalid &= rejects([&] { ComputeXiLM(0, 2, r, {}, workspace); });
    check(invalid && close_results(reference(r, a, 1.5, .1, 1.3),
                                   DiscreteHankelTransform(r, a, workspace, 1.5, .1, 1.3, false)),
          "invalid shapes, lengths and geometry throw; workspace remains usable");

    auto k = logspace(1e-4, 1e4, 1024);
    DVector pk(k.size());
    for (size_t i = 0; i < k.size(); ++i) pk[i] = std::exp(-k[i] * k[i]);
    for (int ell : {0, 2, 4})
        check(error(ell, ComputeXiLM(ell, ell + 2, k, pk, workspace, ell ? -1.0 : 0.0), ell ? .01 : .05, 10) < 1e-6,
              "retained workspace analytic multipole ell=" + std::to_string(ell));
    auto xi = ComputeCorrelationFunction(k, pk, workspace);
    auto back = ComputePowerSpectrum(xi.first, xi.second, workspace);
    double maxerr = 0;
    for (size_t i = 0; i < k.size(); ++i)
        if (back.first[i] > .01 && back.first[i] < 3)
            maxerr = std::max(maxerr, std::abs(back.second[i] - std::exp(-back.first[i] * back.first[i])));
    check(maxerr < 1e-6, "retained workspace spectrum/correlation round trip");
#ifdef USE_OMP
    std::vector<Workspace> workers(4);
    std::vector<std::pair<DVector, DVector>> results(4);
    for (auto & worker : workers) ComputeCorrelationFunction(k, pk, worker); // serial planning/warm-up
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < 4; ++i) results[i] = ComputeCorrelationFunction(k, pk, workers[i]);
    bool concurrent = true;
    for (int i = 0; i < 4; ++i)
        concurrent &= close_results(results[i], xi) && workers[i].statistics().plan_builds == 2;
    check(concurrent, "separately prepared workspaces execute concurrently");
#endif
}

int main() {
    test_workspace();
    // ell = 0, 2, 4 with N = 1024 (q = 0 for ell = 0, q = -1 otherwise)
    {
        auto k = logspace(1e-4, 1e4, 1024);
        DVector pk(k.size());
        for (size_t i = 0; i < k.size(); i++)
            pk[i] = std::exp(-k[i] * k[i]);
        for (int ell : {0, 2, 4}) {
            const double q = ell == 0 ? 0.0 : -1.0;
            auto res = ComputeXiLM(ell, ell + 2, k, pk, q);
            // (with q = 0 the transform rings at the smallest r: use r >= 0.05 for ell = 0)
            const double err = error(ell, res, ell == 0 ? 0.05 : 0.01, 10.0);
            check(err < 1e-6,
                  "xi_" + std::to_string(ell) + " of exp(-k^2), N = 1024 (q = " + num(q) + "): max error " + num(err) +
                      " (the output r-grid was one log-step off; q != 0 had the wrong phase)");
        }
        // xi from the default (q = 0) correlation function
        auto res0 = ComputeCorrelationFunction(k, pk);
        check(error(0, res0, 0.05, 10.0) < 1e-6, "ComputeCorrelationFunction = analytic");

        // Round trip P(k) -> xi(r) -> P(k)
        auto back = ComputePowerSpectrum(res0.first, res0.second);
        double maxerr = 0.0;
        for (size_t i = 0; i < back.first.size(); i++) {
            const double kk = back.first[i];
            if (kk < 0.01 or kk > 3.0)
                continue;
            maxerr = std::max(maxerr, std::fabs(back.second[i] - std::exp(-kk * kk)));
        }
        check(maxerr < 1e-6, "P(k) -> xi(r) -> P(k) (max error " + num(maxerr) + ")");

        // DiscreteHankelTransform with precomputed coefficients (the declared overload, did not link before)
        const int N = int(k.size());
        const double L = std::log(k[N - 1] / k[0]) * N / (N - 1.);
        CVector a(N);
        for (int i = 0; i < N; i++)
            a[i] = std::pow(k[i], 1.5) * pk[i];
        auto u = ComputeCoefficients(N, 0.5, 0.0, L, 1.0);
        auto dht = DiscreteHankelTransform(k, a, 0.5, 0.0, 1.0, false, &u);
        auto dht2 = DiscreteHankelTransform(k, a, 0.5, 0.0, 1.0, false, nullptr);
        double maxdiff = 0.0;
        for (int i = 0; i < N; i++)
            maxdiff = std::max(maxdiff, std::abs(dht.second[i] - dht2.second[i]));
        check(maxdiff < 1e-12 and dht.first == dht2.first, "DiscreteHankelTransform with precomputed coefficients");
    }

    // Many points over few decades: |Im z| of the log-gamma up to ~700
    {
        auto k = logspace(1e-4, 1e4, 8192);
        DVector pk(k.size());
        for (size_t i = 0; i < k.size(); i++)
            pk[i] = std::exp(-k[i] * k[i]);
        auto res = ComputeXiLM(2, 4, k, pk, -1.0);
        bool finite = true;
        for (auto x : res.second)
            finite = finite and std::isfinite(x);
        const double err = error(2, res, 0.01, 10.0);
        check(finite and err < 1e-6, "N = 8192 over 8 decades, ell = 2: finite, max error " + num(err));
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All FFTLog tests passed\n" : std::to_string(fails) + " FFTLog test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
