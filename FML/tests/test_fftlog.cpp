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

int main() {
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
