#include <FML/ConnectedFourPoint/Fourier.h>
#include <iostream>
#include <map>
using namespace FML::FOURPOINT;
namespace {
    using K = std::array<int, 3>;
    using C = std::complex<long double>;
    int checks = 0, failures = 0;
    constexpr long double tau = 6.283185307179586476925286766559L;
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
    bool close(long double a, long double b) {
        return std::abs(a - b) <
               std::max(2e-9L, 32.L * std::numeric_limits<FML::GRID::FloatType>::epsilon()) * (1 + std::abs(b));
    }
    long double real_value(long double x, long double y, long double z) {
        return 7 + std::cos(tau * x) + .4L * std::sin(tau * y) + .3L * std::cos(tau * (x + y)) +
               .2L * std::cos(3 * tau * x) + .15L * std::sin(4 * tau * y) + .11L * std::cos(tau * (x + y + z));
    }
    struct Mode {
        K k;
        C d;
    };
    using Modes = std::vector<Mode>;
    Modes full_modes(int n, double low, double high) {
        Modes modes;
        for (int x = -n / 2; x <= (n - 1) / 2; ++x)
            for (int y = -n / 2; y <= (n - 1) / 2; ++y)
                for (int z = -n / 2; z <= (n - 1) / 2; ++z) {
                    const int k2 = x * x + y * y + z * z;
                    if (!k2 || (n % 2 == 0 && (std::abs(x) == n / 2 || std::abs(y) == n / 2 || std::abs(z) == n / 2)))
                        continue;
                    const double radius = std::sqrt(double(k2));
                    if (radius < low || radius >= high)
                        continue;
                    C d{};
                    for (int i = 0; i < n; ++i)
                        for (int j = 0; j < n; ++j)
                            for (int l = 0; l < n; ++l) {
                                const long double phase = -tau * (x * i + y * j + z * l) / n;
                                d += real_value((long double)i / n, (long double)j / n, (long double)l / n) *
                                     C(std::cos(phase), std::sin(phase));
                            }
                    d /= static_cast<long double>(n) * n * n;
                    modes.push_back({{x, y, z}, d});
                }
        return modes;
    }
    K add(K a, K b) {
        for (int j = 0; j < 3; ++j)
            a[j] += b[j];
        return a;
    }
    K neg(K a) {
        for (auto & v : a)
            v = -v;
        return a;
    }
    struct Pair {
        long long count{};
        C sum{};
    };
    std::map<K, Pair> pairs(const Modes & a, const Modes & b) {
        std::map<K, Pair> out;
        for (auto x : a)
            for (auto y : b) {
                auto & p = out[add(x.k, y.k)];
                ++p.count;
                p.sum += x.d * y.d;
            }
        return out;
    }
    struct Oracle {
        long double raw{}, wick{};
        long long quartets{};
    };
    Oracle direct(const std::array<Modes, 4> & m) {
        const auto ab = pairs(m[0], m[1]), cd = pairs(m[2], m[3]);
        Oracle o;
        for (const auto & entry : ab) {
            const auto it = cd.find(neg(entry.first));
            if (it != cd.end()) {
                o.quartets += entry.second.count * it->second.count;
                o.raw += (entry.second.sum * it->second.sum).real();
            }
        }
        const auto s = [&](int i, int j) {
            const auto p = pairs(m[i], m[j]);
            const auto it = p.find({0, 0, 0});
            return it == p.end() ? 0.L : it->second.sum.real();
        };
        o.wick = s(0, 1) * s(2, 3) + s(0, 2) * s(1, 3) + s(0, 3) * s(1, 2);
        return o;
    }
} // namespace
int main() {
    for (int n : {8, (9 % FML::NTasks == 0 ? 9 : 12)}) {
        // Odd grids have unambiguous high modes; even-grid Nyquist planes must be excluded.
        FourierGrid input(n);
        input.fill_real_grid(
            [](std::array<double, 3> & x) { return FML::GRID::FloatType(real_value(x[0], x[1], x[2])); });
        input.fftw_r2c();
        std::vector<FML::GRID::ComplexType> saved;
        for (auto i : input.get_fourier_range())
            saved.push_back(input.get_fourier_from_index(i));
        const std::vector<double> edges{0, 1.1, 2.1, 4.1, 8.1, 9.1};
        ShellOptions options;
        options.boxsize = 10;
        for (auto e : edges)
            options.k_edges.push_back(e * double(tau) / 10);
        const std::vector<Configuration> configs{{0, 0, 0, 0},
                                                 {0, 0, 1, 1},
                                                 {0, 1, 1, 2},
                                                 {2, 2, 2, 2},
                                                 {3, 3, 3, 3},
                                                 {4, 4, 4, 4}};
        const auto result = connected_shell_four_point(input, options, configs);
        for (size_t c = 0; c < configs.size(); ++c) {
            std::array<Modes, 4> m;
            for (int i = 0; i < 4; ++i)
                m[i] = full_modes(n, edges[configs[c][i]], edges[configs[c][i] + 1]);
            const auto o = direct(m);
            const auto & r = result[c];
            bool ok = close(r.closing_quartets, o.quartets) && close(r.moment.fourth, o.raw) &&
                      close(r.moment.disconnected, o.wick) && close(r.moment.connected, o.raw - o.wick);
            for (int i = 0; i < 4; ++i)
                ok = ok && close(r.mode_counts[i], m[i].size()) && close(r.moment.means[i], 0);
            if (o.quartets) {
                const long double factor = 1e9L / o.quartets;
                ok = ok && r.valid && close(r.raw / factor, o.raw) && close(r.connected / factor, o.raw - o.wick) &&
                     close(r.connected, factor * r.moment.connected);
            } else
                ok = ok && !r.valid && std::isnan(r.connected);
            check(ok,
                  n % 2 ? "odd-mesh DFT / explicit closing-quartet oracle" :
                          "even-mesh DFT / explicit closing-quartet oracle");
        }
        ShellOptions broad{10, {0, 4.1 * double(tau) / 10}};
        const auto broad_result = connected_shell_four_point(input, broad, {{0, 0, 0, 0}})[0];
        const auto broad_modes = full_modes(n, 0, 4.1);
        const auto broad_oracle = direct({broad_modes, broad_modes, broad_modes, broad_modes});
        check(close(broad_result.closing_quartets, broad_oracle.quartets) &&
                  close(broad_result.moment.fourth, broad_oracle.raw),
              "broad-shell true closure excludes modulo-mesh aliases");
        auto low = connected_shell_four_point(input, options, {{0, 0, 0, 0}})[0];
        check(low.work_mesh == n && close(low.connected, result[0].connected),
              "padded and unpadded low-mode estimates agree");
        auto doubled = options;
        doubled.boxsize *= 2;
        for (auto & k : doubled.k_edges)
            k /= 2;
        auto scale = connected_shell_four_point(input, doubled, {{0, 0, 0, 0}})[0];
        check(close(scale.connected / low.connected, 512), "shell trispectrum length-to-ninth scaling");
        std::array<double, 16> external{};
        for (int i = 0; i < 4; ++i)
            external[4 * i + i] = 1;
        auto ref = connected_shell_four_point(input, options, {{0, 0, 0, 0}}, {external})[0];
        check(ref.moment.reference_covariance_used && close(ref.moment.disconnected, 0) &&
                  close(ref.moment.connected, ref.moment.fourth),
              "external covariance controls all three Wick products");
        size_t j = 0;
        bool unchanged = true;
        for (auto i : input.get_fourier_range())
            unchanged &= saved[j++] == input.get_fourier_from_index(i);
        check(unchanged && !input.get_grid_status_real(), "caller Fourier coefficients and space flag preserved");
        check(result[0].work_mesh > n, "high-mode quartet products use padding");
        rejects(
            [&] {
                auto o = options;
                o.k_edges[1] = o.k_edges[0];
                connected_shell_four_point(input, o, configs);
            },
            "invalid edges rejected");
        rejects([&] { connected_shell_four_point(input, options, {{-1, 0, 0, 0}}); }, "negative shell rejected");
        rejects([&] { connected_shell_four_point(input, options, {{9, 0, 0, 0}}); }, "out-of-range shell rejected");
        rejects([&] { connected_shell_four_point(input, options, configs, {external}); },
                "reference length mismatch rejected");
        if (FML::NTasks > 1) {
            rejects(
                [&] {
                    auto o = options;
                    if (FML::ThisTask == FML::NTasks - 1)
                        o.boxsize *= 2;
                    connected_shell_four_point(input, o, configs);
                },
                "rank-dependent options rejected before FFTs");
            rejects(
                [&] {
                    auto cfg = configs;
                    if (FML::ThisTask == FML::NTasks - 1)
                        cfg.pop_back();
                    connected_shell_four_point(input, options, cfg);
                },
                "rank-dependent configuration count rejected");
        }
        // Failure on one rank must be reported everywhere before any transform.
        if (FML::ThisTask == 0)
            for (auto i : input.get_fourier_range()) {
                input.set_fourier_from_index(i, {NAN, 0});
                break;
            }
        rejects([&] { connected_shell_four_point(input, options, configs); },
                "nonfinite Fourier data coordinated across ranks");
    }
    // Exact two-variable Gaussian quadrature: one random Hermitian mode pair.
    // A single field's plug-in Wick term has finite-mode bias; exact ensemble covariance does not.
    FourierGrid toy(8);
    ShellOptions toy_options{1, {0, 1.1 * double(tau)}};
    std::array<double, 16> exact;
    exact.fill(1);
    long double gaussian_connected = 0, plugin_connected = 0;
    for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b) {
            const double x = a < 4 ? 0 : (a == 4 ? 1 : -1) * std::sqrt(3.);
            const double y = b < 4 ? 0 : (b == 4 ? 1 : -1) * std::sqrt(3.);
            toy.fill_real_grid([&](std::array<double, 3> & p) {
                return x * std::cos(double(tau) * p[0]) + y * std::sin(double(tau) * p[0]);
            });
            toy.fftw_r2c();
            const auto reference = connected_shell_four_point(toy, toy_options, {{0, 0, 0, 0}}, {exact})[0];
            const auto plugin = connected_shell_four_point(toy, toy_options, {{0, 0, 0, 0}})[0];
            gaussian_connected += reference.moment.connected / 36;
            plugin_connected += plugin.moment.connected / 36;
        }
    check(close(gaussian_connected, 0), "exact Gaussian Fourier ensemble null with external Wick covariance");
    check(close(plugin_connected, -3), "single-mode plug-in Wick bias matches analytic value");
    if (!FML::ThisTask)
        std::cout << "FourPoint Fourier: " << checks << " checks, " << failures << " failures\n";
    return failures ? 1 : 0;
}
