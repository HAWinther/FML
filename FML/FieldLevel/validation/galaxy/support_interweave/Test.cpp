#include "Support.h"
#include <FML/tests/TestHelpers.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <iomanip>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using Field = FML::FIELDLEVEL::Field;
using FML::TESTS::check;
static double norm(const Field & v) {
    double r = 0;
    for (double x : v)
        r += x * x;
    return r;
}
static double dot(const Field & a, const Field & b) {
    double r = 0;
    for (size_t j = 0; j < a.size(); ++j)
        r += a[j] * b[j];
    return r;
}
static double error(const Field & a, const Field & b) {
    double r = 0;
    for (size_t j = 0; j < a.size(); ++j)
        r = std::max(r, std::abs(a[j] - b[j]));
    return r;
}
struct ToySupport {
    template <class Function>
    void local(Function && f) const {
        f();
    }
    uint64_t rank() const { return 1; }
    void controls(const std::vector<double> & p, size_t i, double a) const {
        if (i >= p.size() || !(p[i] > 0) || !std::isfinite(p[i]) || !(a > 0) || !std::isfinite(a))
            throw std::invalid_argument("Invalid toy scalars");
    }
    void validate(const Field & x) const {
        if (x.empty())
            throw std::invalid_argument("Empty toy");
        for (double v : x)
            if (!std::isfinite(v))
                throw std::invalid_argument("Invalid toy");
    }
    Field project(const Field & x) {
        validate(x);
        Field r(x.size(), 0);
        r[0] = x[0];
        return r;
    }
};
void toy(int n, bool gaussian, bool observed, bool wrong_jacobian = false) {
    const double sigma2 = .49, y = 1.2;
    ToySupport support;
    auto prior = [&](double a) { return gaussian ? .5 * std::pow((a - 1) / .25, 2) : 0.; };
    auto value = [&](const Field & s, double a) {
        return .5 * norm(s) + prior(a) + (observed ? .5 * std::pow(y - a * s[0], 2) / sigma2 : 0.);
    };
    double z = 0, expected_mean = 0, expected_cdf = 0;
    for (int k = 0; k < 20000; ++k) {
        const double a = .5 + (k + .5) / 20000, v = a * a + sigma2,
                     w = std::exp(-prior(a)) * (observed ? std::exp(-.5 * y * y / v) / std::sqrt(v) : 1.);
        z += w;
        expected_mean += w * a;
        expected_cdf += w * (a <= 1);
    }
    expected_mean /= z;
    expected_cdf /= z;
    double mean = 0, cdf = 0, unused = 0;
    size_t samples = 0;
    bool null_exact = true;
    for (unsigned chain = 0; chain < 4; ++chain) {
        std::mt19937_64 rng(941501 + 100 * n + 10 * gaussian + observed + 1000 * chain);
        std::normal_distribution<double> normal;
        std::uniform_real_distribution<double> uniform;
        Field s(n);
        std::vector<double> p = {.6 + .25 * chain};
        for (int it = 0; it < 1024 + 32768; ++it) {
            for (double & x : s)
                x = normal(rng);
            if (observed)
                s[0] = p[0] * y / (p[0] * p[0] + sigma2) + std::sqrt(sigma2 / (p[0] * p[0] + sigma2)) * s[0];
            double u = value(s, p[0]);
            for (int move = 0; move < 4; ++move) {
                const double a = RESCALING::reflected(p[0], .5, 1.5, .2 * normal(rng));
                auto trial = ACTIVE_SUPPORT::propose(support, s, p, 0, a);
                for (int j = 1; j < n; ++j)
                    null_exact &= trial.white[j] == s[j];
                if (wrong_jacobian)
                    trial.log_jacobian = n * (std::log(p[0]) - std::log(a));
                const double next = value(trial.white, a);
                double r = 0;
                while (r == 0)
                    r = uniform(rng);
                if (std::log(r) < RESCALING::log_acceptance(u, next, trial.log_jacobian)) {
                    s = std::move(trial.white);
                    p = std::move(trial.parameters);
                    u = next;
                }
            }
            if (it >= 1024) {
                mean += p[0];
                cdf += p[0] <= 1;
                if (n > 1)
                    unused += s[1] * s[1];
                ++samples;
            }
        }
    }
    mean /= samples;
    cdf /= samples;
    unused /= samples;
    if (wrong_jacobian)
        FML::TESTS::check_local(std::abs(mean - expected_mean) > .05 || std::abs(cdf - expected_cdf) > .05,
                                "negative control detects full-dimension Jacobian on active-only move");
    else {
        FML::TESTS::check_local(std::abs(mean - expected_mean) < .008 && std::abs(cdf - expected_cdf) < .015,
                                "active/null analytic Gaussian marginal and prior quadrature");
        FML::TESTS::check_local(null_exact && (n == 1 || std::abs(unused - 1) < .025),
                                "prior-null coordinates unchanged by proposals and retain unit variance");
    }
    std::cout << "MARGINAL " << n << ' ' << gaussian << ' ' << observed << ' ' << wrong_jacobian << ' ' << mean << ' '
              << expected_mean << ' ' << cdf << ' ' << expected_cdf << '\n';
}
template <class Grid>
void exercise(const FML::Domain & domain, bool compact, FF::Evolution evolution, bool narrow) {
    FF::EulerianLPTConfiguration c;
    c.mesh = compact ? 4 : 8;
    c.product_mesh = compact ? 8 : 0;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.evolution = evolution;
    const double kf = 2 * M_PI / 100;
    c.initial_cutoff = 1.7 * kf;
    c.matter_band = {1.8 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
    c.likelihood_band =
        narrow ? FML::FIELDLEVEL::EULERIAN::SharpKBand{1.1 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Sphere} :
                 FML::FIELDLEVEL::EULERIAN::SharpKBand{1.6 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
    c.matter_assignment = FF::MatterAssignment::PSWF;
    std::array<FF::Parameter, 4> bias = {{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                          {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                          {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                          {"b_laplacian", 7, 0, 1, -100, 100, FF::Prior::Uniform}}};
    using Model = FF::DistributedEulerianLPTModelT<Grid>;
    Model model(c, FF::sigma8_spectrum([](double k) { return 20000 * std::exp(-5 * k); }), bias, domain);
    auto work = model.workspace();
    ACTIVE_SUPPORT::InitialSupport<Model> support(model);
    const auto w = model.white_noise(4103), v = model.white_noise(5103), p = model.initial_parameters();
    const auto pw = support.project(w), pv = support.project(v);
    auto null = w;
    for (size_t j = 0; j < w.size(); ++j)
        null[j] -= pw[j];
    check(support.rank() == 18,
          "global real initial-support rank independent of owned cells, product mesh and likelihood band");
    check(model.maximum(error(pw, support.project(pw))) < 2e-12, "initial-support projector idempotent");
    check(std::abs(model.sum(dot(pw, null))) < 2e-10 && std::abs(model.sum(dot(pw, v) - dot(w, pv))) < 2e-10,
          "initial-support projector orthogonal and self-adjoint");
    auto tone = [&](std::array<int, 3> k) {
        Field x(model.cells());
        for (size_t q = 0; q < x.size(); ++q) {
            const auto n = model.layout().global_coordinate(q);
            x[q] = std::cos(2 * M_PI * (k[0] * n[0] + k[1] * n[1] + k[2] * n[2]) / c.mesh);
        }
        return x;
    };
    const auto inside = tone({1, 1, 0}), outside = tone({1, 1, 1}), zero = tone({0, 0, 0});
    check(model.maximum(error(inside, support.project(inside))) < 2e-12 &&
              model.sum(norm(support.project(outside))) < 1e-20 && model.sum(norm(support.project(zero))) < 1e-20,
          "independent analytic Fourier tones verify exact initial mask and zero exclusion");
    const auto data = model.mock(model.white_noise(4101), p, 1000, 4102, work);
    auto target = FF::fixed_noise_posterior(model, data, 1000.);
    const auto pred = model.predict(w, p, work);
    const double u = target.value(w, p, work);
    double pe = 0, gap = 0, reverse = 0;
    for (double alpha : {.7, 1.3}) {
        const auto trial = ACTIVE_SUPPORT::propose(support, w, p, 0, alpha),
                   back = ACTIVE_SUPPORT::propose(support, trial.white, trial.parameters, 0, p[0]);
        const auto next = model.predict(trial.white, trial.parameters, work);
        for (size_t j = 0; j < next.size(); ++j)
            pe = std::max(pe, std::abs(next[j] - pred[j]));
        const double next_u = target.value(trial.white, trial.parameters, work),
                     expected = .5 * model.sum(norm(trial.white) - norm(w));
        gap = std::max(gap, std::abs(next_u - u - expected) / std::max(1., std::abs(expected)));
        reverse = std::max(reverse, model.maximum(error(w, back.white)));
        auto nextnull = trial.white;
        const auto proj = support.project(trial.white);
        for (size_t j = 0; j < w.size(); ++j)
            nextnull[j] -= proj[j];
        check(model.maximum(error(null, nextnull)) < 2e-12, "prior-null component unchanged by active rescaling");
        check(std::abs(trial.log_jacobian - 18 * std::log(p[0] / alpha)) < 1e-13 &&
                  std::abs(trial.log_jacobian + back.log_jacobian) < 1e-13,
              "GLOBAL active-rank signed Jacobian and inverse");
        check(std::abs(RESCALING::log_acceptance(u, next_u, trial.log_jacobian) +
                       RESCALING::log_acceptance(next_u, u, back.log_jacobian)) < 1e-13,
              "forward/reverse full-target MH reciprocity");
    }
    check(model.maximum(pe) < 5e-10 && model.maximum(gap) < 1e-9 && reverse < 2e-12,
          "shared/compact 1LPT/2LPT predictions, actual prior gap and inverse preserved");
    check(FML::TESTS::throws([&] {
              support.local([&] {
                  if (FML::ThisTask == 0)
                      throw std::bad_alloc();
              });
          }),
          "rank-local proposal allocation failure propagates collectively");
    auto malformed = w;
    if (FML::ThisTask == 0)
        malformed.pop_back();
    check(FML::TESTS::throws([&] { support.project(malformed); }), "rank-local invalid input rejected collectively");
    if (FML::ThisTask == 0)
        std::cout << std::setprecision(17) << "SUPPORT_MODEL " << compact << ' ' << int(evolution) << ' ' << narrow
                  << ' ' << pe << ' ' << gap << ' ' << reverse << '\n';
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (FML::ThisTask == 0) {
            for (int n : {1, 64})
                for (bool gaussian : {false, true})
                    for (bool observed : {false, true})
                        toy(n, gaussian, observed);
            toy(64, false, true, true);
        }
        for (bool compact : {false, true})
            for (auto evolution : {FF::Evolution::Zeldovich, FF::Evolution::LPT2})
                for (bool narrow : {false, true}) {
                    exercise<FML::GRID::FFTWGrid<3>>(FML::Domain::uniform_slab(FML::NTasks),
                                                     compact,
                                                     evolution,
                                                     narrow);
#ifdef USE_MPI
                    const int px = FML::NTasks % 2 ? 1 : 2, py = FML::NTasks / px;
                    std::vector<double> ex(px + 1), ey(py + 1);
                    for (int j = 0; j <= px; ++j)
                        ex[j] = double(j) / px;
                    for (int j = 0; j <= py; ++j)
                        ey[j] = double(j) / py;
                    exercise<FML::GRID::PencilGrid<3>>(FML::Domain::pencil(ex, ey), compact, evolution, narrow);
#endif
                }
        return FML::TESTS::summary("exact initial-support rescaling");
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
