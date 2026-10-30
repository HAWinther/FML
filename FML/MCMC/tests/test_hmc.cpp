// Tests of the HMC sampler (HMC.h) on targets with known moments.
#include "TestHelpers.h"
#include <FML/MCMC/HMC.h>
#include <cstdlib>
#include <numeric>

using namespace FML::MCMC;
using Vector = std::vector<double>;

static void moments(const std::vector<double> & x, double & mean, double & var) {
    mean = std::accumulate(x.begin(), x.end(), 0.) / double(x.size());
    var = 0;
    for (double v : x)
        var += (v - mean) * (v - mean);
    var /= double(x.size() - 1);
}

// Check a sampled mean and variance against the truth, with tolerances from the effective sample size
static void check_moments(const std::string & what, const std::vector<double> & x, double mean_true, double var_true) {
    double mean, var;
    moments(x, mean, var);
    const double ess = double(x.size()) / integrated_autocorrelation_time(x);
    Test::check_close(what + " mean (5 sigma)", (mean - mean_true) / std::sqrt(var_true / ess), 0, 5);
    // var of a sample variance ~ 2 var^2 / ess for a Gaussian
    Test::check_rel(what + " variance", var, var_true, 5 * std::sqrt(2 / ess) + .02);
}

// 1. Independent Gaussians with variances spanning 1e-4 .. 1e2: diagonal mass adaptation must find them
static void test_diagonal_adaptation() {
    std::cout << "Diagonal Gaussian, variances 1e-4 .. 1e2, adapted diagonal mass\n";
    const int d = 50;
    Vector var(d), mu(d);
    for (int i = 0; i < d; ++i) {
        var[i] = std::pow(10., -4 + 6. * i / (d - 1));
        mu[i] = .3 * i;
    }
    auto U = [&](const Vector & x, Vector & g) {
        double u = 0;
        for (int i = 0; i < d; ++i) {
            g[i] = (x[i] - mu[i]) / var[i];
            u += .5 * (x[i] - mu[i]) * (x[i] - mu[i]) / var[i];
        }
        return u;
    };
    HMCOptions opt;
    opt.warmup = 1000;
    opt.samples = 3000;
    opt.n_leapfrog = 10;
    opt.seed = 4;
    const auto r = run_hmc(U, Vector(d, 0.), HMCMass{}, opt);
    std::cout << "  acceptance " << r.acceptance << ", step " << r.step_size << ", divergences " << r.divergences
              << "\n";
    // Dual averaging returns the averaged (smaller) step, so the realised acceptance lies somewhat above the target
    Test::check_close("acceptance in [0.7, 0.97] (target 0.8)",
                      (r.acceptance > .7 && r.acceptance < .97) ? 0. : 1.,
                      0,
                      0);
    for (int i : {0, d / 2, d - 1}) {
        check_moments("x" + std::to_string(i), recorded_column(r, i), mu[i], var[i]);
        Test::check_rel("adapted mass of x" + std::to_string(i) + " ~ 1/var", r.diagonal[i] * var[i], 1., .35);
    }
}

// 2. A correlated Gaussian "field" (custom block with exact mass = precision via its Cholesky factor) jointly with
//    two parameters with a diagonal mass; the field is coupled to the parameters
static void test_custom_block() {
    std::cout << "Correlated 40-dim block with operator mass + 2 diagonal parameters\n";
    const int nf = 40, np = 2, n = nf + np;
    // Precision of the full Gaussian: tridiagonal field precision A plus coupling to the parameters
    // U = 1/2 f^T A f + 1/2 (a^2 / sa^2) + 1/2 (b - c * sum f / nf)^2 / sb^2
    // Field block of A: 2 on the diagonal, -0.9 off the diagonal (strongly correlated)
    auto apply_A = [&](const Vector & f, Vector & y) {
        y.assign(nf, 0);
        for (int i = 0; i < nf; ++i) {
            y[i] = 2 * f[i];
            if (i > 0)
                y[i] -= .9 * f[i - 1];
            if (i + 1 < nf)
                y[i] -= .9 * f[i + 1];
        }
    };
    const double sa = 3, sb = .5, c = 2;
    auto U = [&](const Vector & x, Vector & g) {
        Vector f(x.begin(), x.begin() + nf), Af;
        apply_A(f, Af);
        double u = 0, mean_f = 0;
        for (int i = 0; i < nf; ++i) {
            u += .5 * f[i] * Af[i];
            mean_f += f[i] / nf;
        }
        const double a = x[nf], b = x[nf + 1], r = b - c * mean_f;
        u += .5 * a * a / (sa * sa) + .5 * r * r / (sb * sb);
        for (int i = 0; i < nf; ++i)
            g[i] = Af[i] - r / (sb * sb) * c / nf;
        g[nf] = a / (sa * sa);
        g[nf + 1] = r / (sb * sb);
        return u;
    };
    // Cholesky factor of the field-block precision A = L L^T (tridiagonal: L is lower bidiagonal)
    Vector Ld(nf), Lo(nf, 0);
    Ld[0] = std::sqrt(2.);
    for (int i = 1; i < nf; ++i) {
        Lo[i] = -.9 / Ld[i - 1];
        Ld[i] = std::sqrt(2 - Lo[i] * Lo[i]);
    }
    HMCMass mass;
    mass.n_custom = nf;
    mass.apply_sqrt = [&](Vector & z) { // L z has covariance A
        for (int i = nf - 1; i >= 0; --i)
            z[i] = Ld[i] * z[i] + (i > 0 ? Lo[i] * z[i - 1] : 0);
    };
    mass.apply_inverse = [&](Vector & p) { // solve L L^T v = p
        for (int i = 0; i < nf; ++i)
            p[i] = (p[i] - (i > 0 ? Lo[i] * p[i - 1] : 0)) / Ld[i];
        for (int i = nf - 1; i >= 0; --i)
            p[i] = (p[i] - (i + 1 < nf ? Lo[i + 1] * p[i + 1] : 0)) / Ld[i];
    };
    // Exact marginal variances: a ~ N(0, sa^2); b = c mean_f + N(0, sb^2), var(mean_f) = 1^T A^{-1} 1 / nf^2
    Vector ones(nf, 1.), w(ones);
    mass.apply_inverse(w); // A^{-1} 1
    const double var_mean_f = std::accumulate(w.begin(), w.end(), 0.) / (nf * nf);
    // f's marginal covariance is not A^{-1} exactly (b couples back) but the coupling is integrated out exactly:
    // marginalising b leaves f ~ N(0, A^{-1}), so var(f_0) = (A^{-1})_00
    Vector e0(nf, 0.);
    e0[0] = 1;
    mass.apply_inverse(e0);
    HMCOptions opt;
    opt.warmup = 600;
    opt.samples = 3000;
    opt.n_leapfrog = 8;
    opt.seed = 7;
    const auto r =
        run_hmc(U, Vector(n, 0.), mass, opt, [&](const Vector & x) { return Vector{x[0], x[nf], x[nf + 1]}; });
    std::cout << "  acceptance " << r.acceptance << ", step " << r.step_size << "\n";
    check_moments("field f0", recorded_column(r, 0), 0, e0[0]);
    check_moments("parameter a", recorded_column(r, 1), 0, sa * sa);
    check_moments("parameter b", recorded_column(r, 2), 0, c * c * var_mean_f + sb * sb);
}

// 3. Bounds by reflection: a half-normal (x >= 0) and a flat density on [-1, 2]
static void test_bounds() {
    std::cout << "Reflection at bounds: half-normal and uniform\n";
    auto U = [](const Vector & x, Vector & g) {
        g = {x[0], 0.};
        return .5 * x[0] * x[0];
    };
    HMCOptions opt;
    opt.warmup = 500;
    opt.samples = 6000;
    opt.n_leapfrog = 6;
    opt.lower = {0., -1.};
    opt.upper = {std::numeric_limits<double>::infinity(), 2.};
    opt.seed = 3;
    const auto r = run_hmc(U, Vector{.5, 0.}, HMCMass{}, opt);
    check_moments("half-normal", recorded_column(r, 0), std::sqrt(2 / M_PI), 1 - 2 / M_PI);
    check_moments("uniform", recorded_column(r, 1), .5, 9. / 12);
    double lo = 1e9, hi = -1e9;
    for (const auto & row : r.recorded) {
        lo = std::min(lo, std::min(row[0], row[1] + 1));
        hi = std::max(hi, row[1]);
    }
    Test::check_close("all samples inside the bounds", (lo >= 0 && hi <= 2) ? 0. : 1., 0, 0);
}

// 4. Gelman-Rubin from independent chains of the same target is ~0, and detects chains stuck in different modes
static void test_gelman_rubin() {
    std::cout << "Gelman-Rubin\n";
    auto U = [](const Vector & x, Vector & g) {
        g = {x[0]};
        return .5 * x[0] * x[0];
    };
    std::vector<std::vector<double>> chains;
    for (std::uint64_t s = 1; s <= 4; ++s) {
        HMCOptions opt;
        opt.warmup = 200;
        opt.samples = 2000;
        opt.seed = s;
        chains.push_back(recorded_column(run_hmc(U, Vector{double(s)}, HMCMass{}, opt), 0));
    }
    const double R = gelman_rubin(chains);
    std::cout << "  R - 1 = " << R << "\n";
    Test::check_close("R - 1 small for converged chains", R, 0, .02);
    for (std::size_t j = 0; j < chains.size(); ++j)
        for (auto & v : chains[j])
            v += 3. * double(j);
    Test::check_close("R - 1 large for separated chains", gelman_rubin(chains) > .5 ? 0. : 1., 0, 0);
}

int main() {
    test_diagonal_adaptation();
    test_custom_block();
    test_bounds();
    test_gelman_rubin();
    return Test::summary();
}
