#include <FML/Survey/SurveyReconstruction.h>
#include <FML/Survey/SurveyWindow.h>
#include <iostream>
#include <map>
using namespace FML::SURVEY;
namespace survey_detail = FML::SURVEY::detail;
int failures = 0;
const double tolerance = sizeof(Real) == 4 ? 3e-4 : 2e-9;
void check(bool good, const std::string& label) {
    int bad = !good;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (FML::ThisTask == 0)
        std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
}
bool near(double a, double b, double tol = tolerance) {
    return std::abs(a - b) <= tol * std::max({1.0, std::abs(a), std::abs(b)});
}
std::vector<CatalogPoint> split(const std::vector<CatalogPoint>& points, bool root_only = false) {
    std::vector<CatalogPoint> result;
    for (size_t i = 0; i < points.size(); ++i)
        if ((root_only ? 0 : int(i % FML::NTasks)) == FML::ThisTask)
            result.push_back(points[i]);
    return result;
}
double pol(int ell, double mu) {
    if (ell == 0)
        return 1;
    double p0 = 1, p1 = mu;
    for (int l = 2; l <= ell; ++l) {
        const double next = ((2 * l - 1) * mu * p1 - (l - 1) * p0) / l;
        p0 = p1;
        p1 = next;
    }
    return p1;
}
Vector line_of_sight(const Geometry& g, const Vector& position) {
    Vector n = g.fixed_los;
    if (g.local_los)
        for (int d = 0; d < 3; ++d)
            n[d] = position[d] - g.observer[d];
    double norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (norm)
        for (auto& x : n)
            x /= norm;
    return n;
}
using Cell = std::array<int, 3>;
using Cloud = std::vector<std::pair<Cell, double>>;
Cloud cloud(const CatalogPoint& point, const Geometry& g) {
    Cloud result;
    Cell c;
    Vector t;
    for (int d = 0; d < 3; ++d) {
        double u = (point.position[d] - g.minimum[d]) / g.cellsize();
        c[d] = int(std::floor(u));
        t[d] = u - c[d];
    }
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
                result.push_back({{c[0] + x, c[1] + y, c[2] + z},
                                  (x ? t[0] : 1 - t[0]) * (y ? t[1] : 1 - t[1]) * (z ? t[2] : 1 - t[2])});
    return result;
}
Vector physical(const Cell& c, const Geometry& g) {
    Vector x;
    for (int d = 0; d < 3; ++d)
        x[d] = g.minimum[d] + c[d] * g.cellsize();
    return x;
}
std::vector<CatalogPoint> catalogue(bool data) {
    std::vector<CatalogPoint> result;
    for (int i = 0; i < (data ? 9 : 15); ++i)
        result.push_back({{6.1 + .14 * (i % 5), 6.2 + .18 * ((i * 3) % 5), 6.3 + .13 * ((i * 2) % 7)},
                          .5 + .25 * (i % 4),
                          .02 + .005 * (i % 3)});
    return result;
}
PowerSpectrum direct_power(const std::vector<CatalogPoint>& data, const std::vector<CatalogPoint>& randoms,
                           const Geometry& g, const PowerOptions& options) {
    PowerSpectrum out;
    out.ells = options.ells;
    out.edges = options.edges;
    double wd = 0, wr = 0, i2 = 0;
    for (const auto& p : data)
        wd += p.weight;
    for (const auto& p : randoms) {
        wr += p.weight;
        i2 += p.nbar * p.weight * p.weight;
    }
    out.alpha = wd / wr;
    out.i2 = options.i2 > 0 ? options.i2 : out.alpha * i2;
    const size_t nb = options.edges.size() - 1;
    out.k.assign(nb, 0);
    out.nmodes.assign(nb, 0);
    out.raw.assign(options.ells.size(), std::vector<double>(nb));
    out.shotnoise = out.raw;
    std::map<Cell, double> field;
    for (const auto& p : data)
        for (const auto& cw : cloud(p, g))
            field[cw.first] += p.weight * cw.second;
    for (const auto& p : randoms)
        for (const auto& cw : cloud(p, g))
            field[cw.first] -= out.alpha * p.weight * cw.second;
    for (int ix = 0; ix < g.nmesh; ++ix)
        for (int iy = 0; iy < g.nmesh; ++iy)
            for (int iz = 0; iz < g.nmesh; ++iz) {
                Cell m{ix, iy, iz};
                Vector k;
                for (int d = 0; d < 3; ++d) {
                    if (m[d] > g.nmesh / 2)
                        m[d] -= g.nmesh;
                    k[d] = 2 * M_PI * m[d] / g.boxsize;
                }
                const double kmag = std::sqrt(dot(k, k));
                if (!kmag)
                    continue;
                int b = -1;
                for (size_t j = 0; j < nb; ++j)
                    if (kmag >= options.edges[j] && kmag < options.edges[j + 1])
                        b = int(j);
                if (b < 0)
                    continue;
                Vector khat = k;
                for (auto& x : khat)
                    x /= kmag;
                ++out.nmodes[b];
                out.k[b] += kmag;
                std::vector<std::complex<double>> a(options.ells.size());
                std::complex<double> a0 = 0;
                for (const auto& cv : field) {
                    const auto n = line_of_sight(g, physical(cv.first, g));
                    const double phase =
                        -2 * M_PI * (m[0] * cv.first[0] + m[1] * cv.first[1] + m[2] * cv.first[2]) / g.nmesh;
                    const auto wave = cv.second * std::polar(1.0, phase);
                    a0 += wave;
                    for (size_t l = 0; l < a.size(); ++l)
                        a[l] += wave * pol(options.ells[l], dot(khat, n));
                }
                double window = 1;
                if (options.compensate_cic)
                    for (auto kd : k) {
                        double x = .5 * kd * g.cellsize();
                        window *= x ? std::pow(std::sin(x) / x, 2) : 1;
                    }
                for (size_t l = 0; l < a.size(); ++l) {
                    const int ell = options.ells[l];
                    double noise = 0;
                    for (const auto& p : data)
                        noise += p.weight * p.weight * pol(ell, dot(khat, line_of_sight(g, p.position)));
                    for (const auto& p : randoms)
                        noise += out.alpha * out.alpha * p.weight * p.weight *
                                 pol(ell, dot(khat, line_of_sight(g, p.position)));
                    out.raw[l][b] += (2 * ell + 1) * (a0 * std::conj(a[l])).real() / (out.i2 * window * window);
                    out.shotnoise[l][b] += (2 * ell + 1) * noise / out.i2;
                }
            }
    for (size_t b = 0; b < nb; ++b)
        if (out.nmodes[b]) {
            out.k[b] /= out.nmodes[b];
            for (size_t l = 0; l < out.ells.size(); ++l) {
                out.raw[l][b] /= out.nmodes[b];
                out.shotnoise[l][b] /= out.nmodes[b];
            }
        }
    return out;
}
void test_power() {
    Geometry g;
    g.nmesh = 16;
    g.boxsize = 16;
    g.observer = {-10, -20, -30};
    const auto all_data = catalogue(true), all_randoms = catalogue(false);
    auto data = split(all_data), randoms = split(all_randoms, true);
    auto painted = paint_catalog(data, g);
    double total = 0;
    for (auto i : painted.get_real_range())
        total += painted.get_real_from_index(i);
    FML::SumOverTasks(&total);
    check(near(total, weight_sum(data)), "CIC conserves physical catalogue weights across arbitrary source ranks");
    PowerOptions options;
    options.edges = {0, .7, 1.3, 2, 2.7};
    for (bool local : {false, true})
        for (bool compensate : {false, true}) {
            g.local_los = local;
            g.fixed_los = {1, 2, 3};
            options.compensate_cic = compensate;
            const auto expected = direct_power(all_data, all_randoms, g, options);
            const auto actual = estimate_power(data, randoms, g, options);
            bool same = near(actual.alpha, expected.alpha) && near(actual.i2, expected.i2);
            for (size_t b = 0; b < actual.k.size(); ++b) {
                same &= actual.nmodes[b] == expected.nmodes[b] && near(actual.k[b], expected.k[b]);
                for (size_t l = 0; l < actual.ells.size(); ++l)
                    same &= near(actual.raw[l][b], expected.raw[l][b]) &&
                            near(actual.shotnoise[l][b], expected.shotnoise[l][b]) &&
                            near(actual.power[l][b], expected.raw[l][b] - expected.shotnoise[l][b]);
            }
            check(same, std::string("FFT monopole/quadrupole/hexadecapole and weighted shot terms match direct DFT: ") +
                            (local ? "local LOS" : "fixed LOS") + (compensate ? ", compensated" : ""));
        }
    g.local_los = true;
    auto original = estimate_power(data, randoms, g, options);
    for (auto& p : data)
        p.weight *= 7;
    for (auto& p : randoms)
        p.weight *= 7;
    auto scaled = estimate_power(data, randoms, g, options);
    bool invariant = true;
    for (size_t l = 0; l < 3; ++l)
        for (size_t b = 0; b < original.k.size(); ++b)
            invariant &= near(original.power[l][b], scaled.power[l][b]);
    check(invariant, "common catalogue-weight rescaling preserves normalized power and noise");
    check(near(fkp_weight(.001, 1000), .5), "FKP helper uses physical nbar times P0");
    if (FML::NTasks == 1) {
        g.nmesh = 15;
        options.compensate_cic = false;
        auto ref = direct_power(all_data, all_randoms, g, options);
        auto got = estimate_power(split(all_data), split(all_randoms), g, options);
        bool odd = true;
        for (size_t l = 0; l < 3; ++l)
            for (size_t b = 0; b < got.k.size(); ++b)
                odd &= near(got.raw[l][b], ref.raw[l][b]);
        check(odd, "odd mesh Hermitian multiplicities match full direct DFT");
    }
}
void test_window() {
    Geometry g;
    g.nmesh = 16;
    g.boxsize = 16;
    g.observer = {-10, -20, -30};
    const auto all = catalogue(false);
    auto randoms = split(all);
    const double alpha = .7, i2 = 2.3;
    std::vector<double> edges{0, .9, 1.5, 2.5, 3.5, 5, 7.9};
    for (bool local : {false, true}) {
        g.local_los = local;
        g.fixed_los = {1, 2, 3};
        auto got = estimate_window(randoms, g, edges, alpha, i2, 8, true);
        std::vector<std::vector<double>> expected(5, std::vector<double>(edges.size() - 1));
        for (size_t a = 0; a < all.size(); ++a)
            for (size_t b = 0; b < all.size(); ++b)
                if (a != b)
                    for (const auto& ca : cloud(all[a], g))
                        for (const auto& cb : cloud(all[b], g)) {
                            Vector s;
                            for (int d = 0; d < 3; ++d)
                                s[d] = (cb.first[d] - ca.first[d]) * g.cellsize();
                            const double radius = std::sqrt(dot(s, s));
                            int bin = -1;
                            for (size_t j = 0; j + 1 < edges.size(); ++j)
                                if (radius >= edges[j] && radius < edges[j + 1])
                                    bin = int(j);
                            if (bin < 0)
                                continue;
                            const auto n = line_of_sight(g, physical(ca.first, g));
                            const double mu = radius ? dot(s, n) / radius : 0;
                            for (int l = 0; l <= 4; ++l)
                                if (radius > 0 || l == 0)
                                    expected[l][bin] += (4 * l + 1) * pol(2 * l, mu) * alpha * alpha * all[a].weight *
                                                        all[b].weight * ca.second * cb.second /
                                                        (std::pow(g.cellsize(), 3) * i2);
                        }
        bool same = true;
        for (size_t l = 0; l < 5; ++l)
            for (size_t b = 0; b < got.s.size(); ++b)
                same &= near(got.q[l][b], got.nvectors[b] ? expected[l][b] / got.nvectors[b] : 0,
                             sizeof(Real) == 4 ? 1e-3 : 1e-8);
        check(same, std::string("FFT window ell=0..8 and exact CIC self-pair subtraction match direct random pairs: ") +
                        (local ? "local LOS" : "fixed LOS"));
    }
    check(near(survey_detail::coupling(0, 2, 2), .2) && near(survey_detail::coupling(2, 0, 2), 1) &&
              near(survey_detail::coupling(4, 0, 2), 0),
          "window Legendre couplings include correct mixing and selection rules");
    WindowMultipoles w;
    w.ells = {0, 2, 4, 6, 8};
    const int ns = 400;
    w.q.assign(5, std::vector<double>(ns));
    for (int s = 0; s < ns; ++s) {
        const double lo = s * .1, hi = (s + 1) * .1;
        w.s.push_back(.5 * (lo + hi));
        w.shell_volume.push_back(4 * M_PI / 3 * (hi * hi * hi - lo * lo * lo));
        w.q[0][s] = std::exp(-w.s[s] * w.s[s] / 36);
    }
    std::vector<double> kin;
    for (int i = 0; i <= 160; ++i)
        kin.push_back(6.0 * i / 160);
    std::vector<double> model;
    for (size_t i = 0; i + 1 < kin.size(); ++i)
        model.push_back(std::exp(-2 * std::pow(.5 * (kin[i] + kin[i + 1]), 2)));
    std::vector<double> kout{0, .3, .8, 1.5};
    const auto matrix = make_window_matrix(w, kin, kout, {0}, {0}, false);
    auto convolved = matrix.apply(model);
    const double effective = 18.0 / 11;
    bool gaussian = true;
    for (size_t i = 0; i < kout.size(); ++i)
        gaussian &= near(convolved[i], std::pow(effective / 2, 1.5) * std::exp(-effective * kout[i] * kout[i]), 8e-4);
    check(gaussian, "window matrix reproduces analytic Gaussian-times-Gaussian convolution");
    std::vector<double> quadrupole_model;
    for (size_t i = 0; i + 1 < kin.size(); ++i) {
        const double k = .5 * (kin[i] + kin[i + 1]);
        quadrupole_model.push_back(k * k * std::exp(-2 * k * k));
    }
    const auto quadrupole = make_window_matrix(w, kin, kout, {2}, {2}, false).apply(quadrupole_model);
    bool phase = true;
    for (size_t i = 0; i < kout.size(); ++i)
        phase &=
            near(quadrupole[i],
                 std::pow(effective / 2, 3.5) * kout[i] * kout[i] * std::exp(-effective * kout[i] * kout[i]), 8e-4);
    check(phase, "quadrupole window convolution has correct Bessel order and Fourier phase");
    const std::vector<double> output_edges{0, .2, .6, 1.2};
    const auto binned = make_binned_window_matrix(w, kin, output_edges, {0}, {0}, false).apply(model);
    auto primitive = [&](double k) {
        return std::sqrt(M_PI) * std::erf(std::sqrt(effective) * k) / (4 * std::pow(effective, 1.5)) -
               k * std::exp(-effective * k * k) / (2 * effective);
    };
    bool averaged = true;
    for (size_t b = 0; b < binned.size(); ++b) {
        const double a = output_edges[b], z = output_edges[b + 1];
        averaged &=
            near(binned[b],
                 std::pow(effective / 2, 1.5) * (primitive(z) - primitive(a)) / ((z * z * z - a * a * a) / 3), 8e-4);
    }
    check(averaged, "binned window matrix integrates the model over output shells");
    auto ic = make_window_matrix(w, kin, kout, {0}, {0}, true).apply(model);
    check(std::abs(ic[0]) < 1e-10, "global integral-constraint projection nulls convolved monopole at k=0");
    auto no_mix = make_window_matrix(w, {0, .2, .4}, {.1, .3}, {2}, {0}, false).apply({1, 2});
    check(std::abs(no_mix[0]) + std::abs(no_mix[1]) < 1e-12,
          "isotropic window does not mix input quadrupole into monopole");
    bool local_matrix = true;
    if (FML::ThisTask == 0) {
        const auto only_root = make_window_matrix(w, {0, .2, .4}, {.1}, {0}, {0}, false);
        local_matrix = std::isfinite(only_root.apply({1, 2})[0]);
    }
    check(local_matrix, "window matrix construction/application may run on rank zero alone");
}
void test_reconstruction() {
    Geometry g;
    g.nmesh = 16;
    g.boxsize = 16;
    g.observer = {-20, -3, -10};
    ReconstructionOptions options;
    options.bias = 2;
    options.growth_rate = .8;
    options.relative_tolerance = sizeof(Real) == 4 ? 2e-5 : 1e-9;
    const int n = g.nmesh;
    auto index = [&](Cell c) {
        for (auto& v : c)
            v = (v + n) % n;
        return (size_t(c[0]) * n + c[1]) * n + c[2];
    };
    std::vector<double> potential(size_t(n) * n * n);
    for (int x = 0; x < n; ++x)
        for (int y = 0; y < n; ++y)
            for (int z = 0; z < n; ++z)
                potential[index({x, y, z})] =
                    std::cos(2 * M_PI * x / n) + .4 * std::sin(4 * M_PI * y / n) + .3 * std::sin(2 * M_PI * z / n);
    for (bool local : {false, true}) {
        g.local_los = local;
        g.fixed_los = {1, 2, 3};
        std::vector<Vector> flux(potential.size());
        for (int x = 0; x < n; ++x)
            for (int y = 0; y < n; ++y)
                for (int z = 0; z < n; ++z) {
                    Cell c{x, y, z};
                    Vector grad;
                    for (int d = 0; d < 3; ++d) {
                        auto next = c;
                        ++next[d];
                        grad[d] = (potential[index(next)] - potential[index(c)]) / g.cellsize();
                    }
                    const auto los = line_of_sight(g, physical(c, g));
                    const double radial = dot(grad, los);
                    for (int d = 0; d < 3; ++d)
                        flux[index(c)][d] = grad[d] + options.growth_rate / options.bias * radial * los[d];
                }
        auto source = survey_detail::empty_mesh(g);
        for (auto i : source.get_real_range()) {
            auto c = source.get_coord_from_index(i);
            c[0] += source.get_local_x_start();
            double rhs = 0;
            for (int d = 0; d < 3; ++d) {
                auto prev = c;
                --prev[d];
                rhs -= (flux[index(c)][d] - flux[index(prev)][d]) / g.cellsize();
            }
            source.set_real_from_index(i, Real(rhs));
        }
        auto solution = solve_displacement(source, g, options);
        bool same = solution.report.converged;
        for (auto i : source.get_real_range()) {
            auto c = source.get_coord_from_index(i);
            c[0] += source.get_local_x_start();
            for (int d = 0; d < 3; ++d) {
                auto prev = c, next = c;
                --prev[d];
                ++next[d];
                const double expected = (potential[index(next)] - potential[index(prev)]) / (2 * g.cellsize());
                same &= near(solution.field[d].get_real_from_index(i), expected, sizeof(Real) == 4 ? 2e-4 : 1e-7);
            }
        }
        check(same, std::string("anisotropic displacement solve recovers manufactured field: ") +
                        (local ? "local LOS" : "fixed LOS"));
        ReconstructionOptions limited = options;
        limited.max_iterations = 1;
        limited.relative_tolerance = 1e-12;
        check(!solve_displacement(source, g, limited).report.converged,
              "iteration exhaustion is reported as nonconvergence");
    }
    auto constant = survey_detail::empty_mesh(g);
    for (auto i : constant.get_real_range())
        constant.set_real_from_index(i, 1);
    auto zero = solve_displacement(constant, g, options);
    check(zero.report.converged && zero.report.iterations == 0 && near(zero.report.removed_mean, 1),
          "constant source is projected out and zero source converges without iterations");
    std::array<Mesh, 3> fields{survey_detail::empty_mesh(g, 1), survey_detail::empty_mesh(g, 1),
                               survey_detail::empty_mesh(g, 1)};
    for (int d = 0; d < 3; ++d) {
        for (auto i : fields[d].get_real_range())
            fields[d].set_real_from_index(i, Real(d + .5));
        fields[d].communicate_boundaries();
    }
    std::vector<CatalogPoint> all;
    for (int i = 2; i < 14; ++i)
        all.push_back({{i + .1, 5.2, 7.3}, 1, .02});
    auto points = split(all);
    auto values = sample_displacement(points, fields, g);
    bool samples = values.size() == points.size();
    for (const auto& v : values)
        for (int d = 0; d < 3; ++d)
            samples &= near(v[d], d + .5);
    check(samples, "CIC displacement queries across MPI slabs return in original local catalogue order");

    auto all_randoms = catalogue(false), all_data = all_randoms;
    for (size_t i = 0; i < all_data.size(); ++i)
        all_data[i].weight *= 1 + .1 * std::sin(double(i));
    auto data = split(all_data), randoms = split(all_randoms, true);
    options.smoothing_radius = .8;
    options.minimum_random_fraction = .02;
    options.convention = ReconstructionConvention::RecSym;
    auto sym = reconstruct(data, randoms, g, options);
    options.convention = ReconstructionConvention::RecIso;
    auto iso = reconstruct(data, randoms, g, options);
    bool conventions = sym.solver.converged && iso.solver.converged;
    for (size_t i = 0; i < data.size(); ++i) {
        conventions &= sym.data[i].weight == data[i].weight && sym.data[i].nbar == data[i].nbar &&
                       sym.data[i].position == iso.data[i].position;
        const auto los = line_of_sight(g, data[i].position);
        const double radial = options.growth_rate * dot(sym.data_displacement[i], los);
        for (int d = 0; d < 3; ++d)
            conventions &=
                near(sym.data[i].position[d], data[i].position[d] - sym.data_displacement[i][d] - radial * los[d]);
    }
    for (size_t i = 0; i < randoms.size(); ++i) {
        conventions &= sym.randoms[i].weight == randoms[i].weight && sym.randoms[i].nbar == randoms[i].nbar;
        const auto los = line_of_sight(g, randoms[i].position);
        const double radial = options.growth_rate * dot(sym.random_displacement[i], los);
        for (int d = 0; d < 3; ++d) {
            conventions &= near(iso.randoms[i].position[d], randoms[i].position[d] - sym.random_displacement[i][d]);
            conventions &= near(sym.randoms[i].position[d], iso.randoms[i].position[d] - radial * los[d]);
        }
    }
    check(conventions, "end-to-end RecIso/RecSym preserve weights/order and apply distinct random RSD shifts");
    PowerOptions p;
    p.edges = {0, .7, 1.3, 2};
    auto reconstructed = estimate_power(sym.data, randoms, g, p, &sym.randoms);
    bool finite = true;
    for (const auto& row : reconstructed.power)
        for (double value : row)
            finite &= std::isfinite(value);
    check(finite, "shifted data/randoms feed reconstructed power while original randoms define the selection");
    options.mask = [](const Vector&) { return false; };
    auto masked = reconstruct(data, randoms, g, options);
    bool unchanged = true;
    for (size_t i = 0; i < data.size(); ++i)
        unchanged &= data[i].position == masked.data[i].position;
    for (size_t i = 0; i < randoms.size(); ++i)
        unchanged &= randoms[i].position == masked.randoms[i].position;
    check(unchanged && masked.solver.iterations == 0, "selection mask suppresses excluded density cells");
}
void test_validation() {
    Geometry g;
    g.nmesh = 16;
    g.boxsize = 16;
    g.observer = {-10, -20, -30};
    auto data = split(catalogue(true)), randoms = split(catalogue(false));
    auto rejects = [](auto operation) {
        try {
            operation();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    PowerOptions p;
    p.edges = {0, 1, 2};
    auto bad = data;
    if (FML::ThisTask == 0)
        bad.push_back({{.1, 6, 6}, 1, 1});
    check(rejects([&] { estimate_power(bad, randoms, g, p); }), "invalid padding on one rank fails collectively");
    auto no_nbar = randoms;
    for (auto& x : no_nbar)
        x.nbar = 0;
    check(rejects([&] { estimate_power(data, no_nbar, g, p); }), "missing selection normalization is rejected");
    p.ells = {1};
    check(rejects([&] { estimate_power(data, randoms, g, p); }), "unsupported multipoles are rejected");
    p.ells = {0, 2, 4};
    p.edges = {0, 1, 1};
    check(rejects([&] { estimate_power(data, randoms, g, p); }), "invalid bin edges are rejected");
    WindowMultipoles w;
    w.ells = {0};
    w.q = {{1, 1}};
    w.s = {1, 2};
    w.shell_volume = {1, 2};
    check(rejects([&] { make_window_matrix(w, {0, 1}, {.1}); }), "window matrix rejects missing angular multipoles");
    w.shell_volume.clear();
    check(rejects([&] {
              make_window_matrix(w, {0, 1}, {.1}, {0}, {0});
          }),
          "malformed window dimensions throw before indexing arrays");
    ReconstructionOptions r;
    r.bias = 0;
    auto source = survey_detail::empty_mesh(g);
    check(rejects([&] { solve_displacement(source, g, r); }), "nonpositive reconstruction bias is rejected");
}
int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    if (mode == "all" || mode == "power")
        test_power();
    if (mode == "all" || mode == "window")
        test_window();
    if (mode == "all" || mode == "reconstruction")
        test_reconstruction();
    if (mode == "all")
        test_validation();
    if (FML::ThisTask == 0)
        std::cout << (failures ? "Survey workflow tests FAILED\n" : "All survey workflow tests passed\n");
    return failures ? 1 : 0;
}
