#include <FML/ConnectedFourPoint/ConnectedFourPoint.h>
#include <iostream>
#include <random>
using namespace FML::FOURPOINT;
namespace {
    int checks = 0, failures = 0;
    void check(bool ok, const char * label) {
        int bad = !ok;
        FML::MaxOverTasks(&bad);
        ++checks;
        failures += bad;
        if (!FML::ThisTask)
            std::cout << (bad ? "FAIL " : "PASS ") << label << '\n';
    }
    template <class F>
    void rejects(F f, const char * label) {
        bool caught = false;
        try {
            f();
        } catch (const std::exception &) {
            caught = true;
        }
        check(caught, label);
    }
    bool close(long double a, long double b, long double tol = 1e-12L) {
        return std::abs(a - b) < tol * (1 + std::abs(b));
    }
} // namespace
int main() {
    // Exact Rademacher ensemble, disjoint rows; ranks >=2 include empty local partitions.
    const size_t n = FML::ThisTask < 2 ? 1 : 0;
    auto sign = [](size_t) {
        const double x = FML::ThisTask == 0 ? -1 : 1;
        return std::array<double, 4>{x, 2 * x, -3 * x, 4 * x};
    };
    auto row = [&](size_t i) {
        if (FML::NTasks == 1) {
            const double x = i ? 1 : -1;
            return std::array<double, 4>{x, 2 * x, -3 * x, 4 * x};
        }
        return sign(i);
    };
    const size_t rows = FML::NTasks == 1 ? 2 : n;
    const auto r = zero_mean_four_point(rows, row);
    check(r.samples == 2 && close(r.means[0], 0) && close(r.fourth, -24) && close(r.disconnected, -72) &&
              close(r.connected, 48),
          "analytic mixed Rademacher cumulant, empty ranks and replicated results");
    std::array<double, 16> cov{};
    const double a[4] = {1, 2, -3, 4};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            cov[4 * i + j] = a[i] * a[j];
    check(close(zero_mean_four_point(rows, row, cov).connected, 48), "singular PSD exact covariance reference");
    // Product quadrature for four independent Gaussians reproduces all moments through degree four.
    // Six equiprobable nodes {0,0,0,0,+sqrt(3),-sqrt(3)} for each variable.
    const double mixing[4][4] = {{1, 0, 0, 0}, {.2, 1, 0, 0}, {-.3, .5, 1, 0}, {.7, .2, -.4, 1}};
    std::vector<std::array<double, 4>> samples;
    for (int t = FML::ThisTask; t < 1296; t += FML::NTasks) {
        int code = t;
        double g[4];
        for (int i = 0; i < 4; ++i) {
            const int digit = code % 6;
            code /= 6;
            g[i] = digit < 4 ? 0 : (digit == 4 ? 1 : -1) * std::sqrt(3.);
        }
        std::array<double, 4> x{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                x[i] += mixing[i][j] * g[j];
        samples.push_back(x);
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            cov[4 * i + j] = 0;
            for (int k = 0; k < 4; ++k)
                cov[4 * i + j] += mixing[i][k] * mixing[j][k];
        }
    const auto gaussian = zero_mean_four_point(samples.size(), [&](size_t i) { return samples[i]; }, cov);
    check(gaussian.samples == 1296 && close(gaussian.connected, 0),
          "correlated Gaussian exact fourth-degree quadrature");
    rejects([&] { zero_mean_four_point(0, row); }, "collective empty input rejection");
    rejects(
        [&] {
            zero_mean_four_point(1, [](size_t) {
                throw std::runtime_error("callback");
                return std::array<double, 4>{};
            });
        },
        "callback errors coordinated");
    rejects([&] { zero_mean_four_point(1, [](size_t) { return std::array<double, 4>{NAN, 1, 1, 1}; }); },
            "nonfinite sample rejection");
    auto invalid = cov;
    invalid[1] += .1;
    rejects([&] { zero_mean_four_point(rows, row, invalid); }, "asymmetric reference rejection");
    invalid.fill(1);
    invalid[0] = .1;
    rejects([&] { zero_mean_four_point(rows, row, invalid); }, "indefinite reference rejection");
#ifdef USE_MPI
    if (FML::NTasks > 1)
        rejects(
            [&] {
                auto v = cov;
                if (FML::ThisTask == FML::NTasks - 1)
                    v[0] += 1;
                zero_mean_four_point(rows, row, v);
            },
            "rank-dependent reference rejected");
#endif
    auto g = gaussian_covariance_disjoint_shells({20, 90}, {4, 6});
    check(g.size() == 4 && close(g[0], 2.5) && close(g[3], 5) && g[1] == 0,
          "Gaussian full-mode sum P-squared normalization");
    FML::ENSEMBLE::RunningCovariance local(2);
    for (int i = FML::ThisTask; i < 3; i += FML::NTasks)
        local.add({double(i), 2. * i});
    auto summary = local.global().summary();
    auto c = connected_power_covariance(summary, {.25, 0, 0, .5}, 10);
    check(c.realizations == 3 && close(c.total[0], 1) && close(c.connected[0], .75) &&
              close(c.collapsed_trispectrum[1], 2000) && close(c.collapsed_trispectrum[3], 3500),
          "streaming N-1 ensemble covariance and volume-scaled collapsed trispectrum");
    auto scaled = connected_power_covariance(summary, {.25, 0, 0, .5}, 20);
    check(close(scaled.collapsed_trispectrum[0] / c.collapsed_trispectrum[0], 8), "collapsed volume scaling");
    auto negative = connected_power_covariance(summary, {2, 0, 0, 5}, 1);
    check(negative.connected[0] < 0 && negative.connected[3] < 0, "negative connected estimates retained");
    rejects([&] { gaussian_covariance_disjoint_shells({1}, {0}); }, "zero mode count rejected");
    rejects([&] { connected_power_covariance(summary, {1, 0, 1, 1}, 1); }, "asymmetric Gaussian covariance rejected");
    auto bad = summary;
    bad.covariance[1] += 1;
    rejects([&] { connected_power_covariance(bad, {1, 0, 0, 1}, 1); }, "asymmetric total covariance rejected");
    rejects([&] { connected_power_covariance(summary, {1, 0, 0, 1}, 0); }, "invalid box rejected");
    // Independent periodic-box analogue: two independent complex Gaussian modes with P=1.
    // Their exponential powers have variance 1 and Gaussian covariance 2P^2/Nfull=1.
    FML::ENSEMBLE::RunningCovariance mc(2);
    std::mt19937_64 rng(20261112 + FML::ThisTask);
    std::normal_distribution<double> normal;
    for (int i = FML::ThisTask; i < 100000; i += FML::NTasks) {
        double x = normal(rng), y = normal(rng), u = normal(rng), v = normal(rng);
        mc.add({.5 * (x * x + y * y), .5 * (u * u + v * v)});
    }
    const auto null = connected_power_covariance(mc.global().summary(), {1, 0, 0, 1}, 1);
    check(std::abs(null.connected[0]) < .06 && std::abs(null.connected[1]) < .03 && std::abs(null.connected[3]) < .06,
          "independent Gaussian-mode covariance null ensemble");
    if (!FML::ThisTask)
        std::cout << "FourPoint core: " << checks << " checks, " << failures << " failures\n";
    return failures ? 1 : 0;
}
