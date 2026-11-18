#include <FML/FieldLevel/EulerianCheckpoint.h>
#include <FML/tests/TestHelpers.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <memory>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Complex;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
static auto priors() {
    return std::array<FF::Parameter, 4>{{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                         {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                         {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                         {"b_laplacian", 7., 0, 1, -100, 100, FF::Prior::Uniform}}};
}
static double dot(const Field & a, const Field & b) {
    double x = 0;
    for (size_t j = 0; j < a.size(); ++j)
        x += a[j] * b[j];
    return x;
}
static double error(const Field & a, const Field & b) {
    double e = 0;
    for (size_t j = 0; j < a.size(); ++j)
        e = std::max(e, std::abs(a[j] - b[j]));
    return e;
}
template <class Model>
Field tone(const Model & m, std::array<int, 3> wave) {
    Field x(m.cells());
    for (size_t q = 0; q < x.size(); ++q) {
        const auto n = m.layout().global_coordinate(q);
        double phase = 0;
        for (int d = 0; d < 3; ++d)
            phase += 2 * M_PI * wave[d] * n[d] / m.configuration().mesh;
        x[q] = std::cos(phase);
    }
    return x;
}
template <class Grid>
void exercise(const FML::Domain & domain, const std::string & label, bool compact) {
    FF::EulerianLPTConfiguration c;
    c.mesh = compact ? 8 : 12;
    c.product_mesh = compact ? 12 : 0;
    c.density_mesh = 24;
    c.boxsize = 100;
    const double kf = 2 * M_PI / c.boxsize;
    c.initial_cutoff = 2.7 * kf;
    c.matter_band = {2.8 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
    c.likelihood_band = {2.5 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
    c.matter_assignment = FF::MatterAssignment::PSWF;
    c.nufft_epsilon = 1e-10;
    auto calls = std::make_shared<size_t>(0);
    auto bad = std::make_shared<bool>(false);
    auto spectrum = FF::sigma8_spectrum([calls, bad](double k) {
        ++*calls;
        if (*bad)
            throw std::runtime_error("Injected spectrum failure");
        return 500 * std::exp(-5 * k);
    });
    using Model = FF::DistributedEulerianLPTModelT<Grid>;
    Model m(c, spectrum, priors(), domain);
    auto w = m.workspace();
    auto p = m.initial_parameters();
    p[0] = 1.2;
    const FF::Modes data(m.modes().size(), 0);
    auto post = FF::fixed_noise_posterior(m, data, 1000.);
    const auto x = tone(m, {1, 1, 0});
    const double k = std::sqrt(2.) * kf;
    double worst = 0;
    for (double noise : {1., 1000., 1e6}) {
        auto target = FF::fixed_noise_posterior(m, data, noise);
        const double response = p[1] - p[4] * k * k;
        const double precision = 1 + response * response * p[0] * p[0] * 500 * std::exp(-5 * k) / noise;
        for (double exponent : {-1., -.5, 0., .5, 1.}) {
            const auto out = target.apply_field_metric(x, p, w, exponent);
            Field expected = x;
            for (auto & v : expected)
                v *= std::pow(precision, exponent);
            const double e = m.maximum(error(out, expected));
            worst = std::max(worst, e);
            check(e < 2e-10, label + " analytic noise/bias/Laplacian/power exponent=" + std::to_string(exponent));
        }
    }
    FF::GaussianFieldLikelihood radial(m.modes(), 1.1 * kf, m.volume(), data, [&](const auto & d) {
        m.validate_modes(d);
    });
    FF::BasicPosterior<Model> masked(m, std::move(radial), FF::GaussianNoise::fixed_power(1000));
    check(m.maximum(error(x, masked.apply_field_metric(x, p, w, 1))) < 2e-12,
          label + " actual likelihood cutoff mask respected");
    FF::GaussianFieldLikelihood twice_volume(m.modes(), HUGE_VAL, 2 * m.volume(), data, [&](const auto & d) {
        m.validate_modes(d);
    });
    FF::BasicPosterior<Model> rescaled(m, std::move(twice_volume), FF::GaussianNoise::fixed_power(1000));
    Field expected_volume = x;
    const double R = p[1] - p[4] * k * k;
    for (auto & v : expected_volume)
        v *= 1 + 2 * R * R * p[0] * p[0] * 500 * std::exp(-5 * k) / 1000;
    check(m.maximum(error(expected_volume, rescaled.apply_field_metric(x, p, w, 1))) < 2e-12,
          label + " actual likelihood covariance volume respected");
    FF::BasicPosterior<Model> indexed(m, data, FF::GaussianNoise::parameter(0));
    Field expected_index = x;
    for (auto & v : expected_index)
        v *= 1 + R * R * p[0] * p[0] * 500 * std::exp(-5 * k) / std::exp(p[0]);
    check(m.maximum(error(expected_index, indexed.apply_field_metric(x, p, w, 1))) < 2e-10,
          label + " explicit sampled-noise policy is evaluated at supplied snapshot");
    const auto zero = post.apply_field_metric(x, p, w, 0);
    check(zero == x, label + " zero power is exact identity");
    auto cached = post.apply_field_metric(x, p, w, 1);
    const auto count = *calls;
    post.apply_field_metric(x, p, w, .5);
    post.apply_field_metric(x, p, w, -1);
    check(*calls == count, label + " frozen spectral precision reused across metric powers");
    check(m.maximum(error(x, masked.apply_field_metric(x, p, w, 1))) < 2e-12 &&
              m.maximum(error(cached, post.apply_field_metric(x, p, w, 1))) < 2e-12,
          label + " same-noise mask switches invalidate cache collectively on empty-mode ranks");
    auto nonlinear = p;
    nonlinear[2] += 1;
    nonlinear[3] += 1;
    check(m.maximum(error(cached, post.apply_field_metric(x, nonlinear, w, 1))) < 2e-12,
          label + " quadratic operators absent from zero-field linear response");
    auto cancel = p;
    cancel[1] = cancel[4] * k * k;
    check(m.maximum(error(x, post.apply_field_metric(x, cancel, w, 1))) < 2e-12,
          label + " response cancellation gives identity");
    for (const auto & wave : std::vector<std::array<int, 3>>{{0, 0, 0}, {c.mesh / 2, 0, 0}, {3, 0, 0}}) {
        const auto null = tone(m, wave);
        check(m.maximum(error(null, post.apply_field_metric(null, p, w, -1))) < 2e-12,
              label + " DC/Nyquist/out-of-band identity");
    }
    auto a = m.white_noise(701), b = m.white_noise(702);
    const auto Ma = post.apply_field_metric(a, p, w, 1), Mb = post.apply_field_metric(b, p, w, 1);
    check(FML::TESTS::near(m.sum(dot(a, Mb)), m.sum(dot(Ma, b)), 2e-12) && m.sum(dot(a, Ma)) > 0,
          label + " symmetry and positive definiteness");
    const auto inverse = post.apply_field_metric(Ma, p, w, -1);
    check(m.maximum(error(inverse, a)) < 2e-11, label + " inverse composes on full sampled space");
    // Independent likelihood Hessian in a cosine direction at zero field. Residual and
    // quadratic-operator linear response vanish here; no metric formula is reused.
    Field plus = x, minus = x, gs1, gs2;
    std::vector<double> gp;
    const double h = 1e-5;
    for (auto & v : plus)
        v *= h;
    for (auto & v : minus)
        v *= -h;
    post.evaluate(plus, p, w, gs1, gp);
    post.evaluate(minus, p, w, gs2, gp);
    for (size_t q = 0; q < gs1.size(); ++q)
        gs1[q] = (gs1[q] - gs2[q]) / (2 * h);
    const auto exact = post.apply_field_metric(x, p, w, 1);
    const double hessian =
        std::abs(m.sum(dot(x, gs1)) - m.sum(dot(x, exact))) / std::max(1., std::abs(m.sum(dot(x, exact))));
    check(hessian < 2e-6, label + " matches independently differentiated zero-field posterior Hessian");
    // Applying a frozen metric must not replace the forward's current scalar snapshot/tape.
    auto current = p;
    current[0] = 1.1;
    const auto pred = m.predict(a, current, w);
    Field before, after;
    std::vector<double> bp, ap;
    m.pullback(current, pred, w, before, bp);
    post.apply_field_metric(b, p, w, .5);
    m.pullback(current, pred, w, after, ap);
    check(m.maximum(error(before, after)) < 2e-12 && error(bp, ap) < 2e-12,
          label + " frozen metric preserves current forward/reverse tape");
    auto malformed = a;
    if (FML::ThisTask == FML::NTasks - 1)
        malformed.pop_back();
    check(FML::TESTS::throws([&] { post.apply_field_metric(malformed, p, w, 1); }),
          label + " rank-local bad field size rejected");
    auto nonfinite = a;
    if (FML::ThisTask == FML::NTasks - 1)
        nonfinite[0] = NAN;
    check(FML::TESTS::throws([&] { post.apply_field_metric(nonfinite, p, w, 1); }),
          label + " nonfinite field rejected collectively");
    check(FML::TESTS::throws([&] { m.apply_gaussian_field_metric(a, p, w, 1, post.likelihood(), NAN); }),
          label + " nonfinite noise rejected");
#ifdef USE_MPI
    if (FML::NTasks > 1)
        check(FML::TESTS::throws(
                  [&] { m.apply_gaussian_field_metric(a, p, w, 1, post.likelihood(), FML::ThisTask ? 1. : 2.); }),
              label + " replicated metric controls required");
#endif
    auto other = FF::fixed_noise_posterior(m, data, 2300.);
    int callback_owner = -1;
    for (const auto & mode : m.modes())
        if (mode.k < c.initial_cutoff &&
            c.matter_band.contains({mode.n[0], mode.n[1], mode.n[2]}, c.boxsize, m.product_mesh()))
            callback_owner = FML::ThisTask;
    FML::MaxOverTasks(&callback_owner);
    if (FML::ThisTask == callback_owner)
        *bad = true;
    check(FML::TESTS::throws([&] { other.apply_field_metric(a, p, w, 1); }),
          label + " rank-local metric callback failure is collective");
    *bad = false;
    const auto recovered = other.apply_field_metric(a, p, w, 1);
    check(std::isfinite(m.sum(dot(a, recovered))), label + " failed cache rebuild can retry collectively");
    auto asymmetric_modes = m.modes();
    for (auto & mode : asymmetric_modes)
        if (mode.n[0] == 1 && mode.n[1] == 0 && mode.n[2] == 0)
            mode.weight = 0;
    FF::GaussianFieldLikelihood asymmetric(asymmetric_modes, HUGE_VAL, m.volume(), data, [&](const auto & d) {
        m.validate_modes(d);
    });
    FF::BasicPosterior<Model> invalid_mask(m, std::move(asymmetric), FF::GaussianNoise::fixed_power(1000));
    check(FML::TESTS::throws([&] { invalid_mask.apply_field_metric(a, p, w, .5); }),
          label + " asymmetric Hermitian-plane mask rejected before inconsistent metric powers");
    const double initial_potential = post.value(a, p, w);
    const auto fit = FF::map(post, a, p, false, 1);
    check(std::isfinite(fit.potential) && fit.potential < initial_potential,
          label + " MAP uses posterior-aware metric and lowers the target");
    auto selected = c;
    selected.likelihood_band = {1.6 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Sphere};
    Model restricted(selected, spectrum, priors(), domain);
    auto work = restricted.workspace();
    auto q = restricted.initial_parameters();
    auto target = FF::fixed_noise_posterior(restricted, FF::Modes(restricted.modes().size(), 0), 1000.);
    const auto excluded = tone(restricted, {2, 0, 0});
    check(restricted.maximum(error(excluded, target.apply_field_metric(excluded, q, work, 1))) < 2e-12,
          label + " likelihood band respected");
    selected = c;
    selected.matter_band = {1.6 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Sphere};
    selected.likelihood_band = {1.6 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
    Model filtered(selected, spectrum, priors(), domain);
    auto fw = filtered.workspace();
    auto fp = filtered.initial_parameters();
    auto filtered_target = FF::fixed_noise_posterior(filtered, FF::Modes(filtered.modes().size(), 0), 1000.);
    const auto corner = tone(filtered, {1, 1, 1});
    check(filtered.maximum(error(corner, filtered_target.apply_field_metric(corner, fp, fw, 1))) < 2e-12,
          label + " matter band respected for cubic corners");
    if (FML::ThisTask == 0)
        std::cout << "METRIC_CHECK " << label << ' ' << worst << ' ' << hessian << '\n';
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    for (bool compact : {false, true}) {
        const std::string label = compact ? "compact" : "shared";
        exercise<FML::GRID::FFTWGrid<3>>(FML::Domain::uniform_slab(FML::NTasks), "slab-" + label, compact);
#ifdef USE_MPI
        const int px = FML::NTasks % 2 ? 1 : 2, py = FML::NTasks / px;
        std::vector<double> ex(px + 1), ey(py + 1);
        for (int j = 0; j <= px; ++j)
            ex[j] = double(j) / px;
        for (int j = 0; j <= py; ++j)
            ey[j] = double(j) / py;
        exercise<FML::GRID::PencilGrid<3>>(FML::Domain::pencil(ex, ey), "pencil-" + label, compact);
#endif
    }
    return FML::TESTS::summary("Eulerian posterior-aware metric");
}
