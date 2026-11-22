// The field metric (HMC mass / MAP preconditioner) must use the noise the posterior
// actually uses, and the options MAP must dispatch through the posterior.
#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <FML/FieldLevel/MAP.h>
#include <FML/tests/TestHelpers.h>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace E = FML::FIELDLEVEL::EULERIAN;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;

static double max_difference(const Field & a, const Field & b) {
    double d = a.size() == b.size() ? 0 : HUGE_VAL;
    for (size_t q = 0; q < a.size() && q < b.size(); ++q)
        d = std::max(d, std::abs(a[q] - b[q]));
    FML::MaxOverTasks(&d);
    return d;
}

template <class Model>
void exercise(const Model & model, const std::string & label) {
    auto w = model.workspace();
    auto p = model.initial_parameters();
    const auto s = model.white_noise(11), x = model.white_noise(12);
    const auto data = model.mock(s, p, 13, w);
    // Default policy: unchanged historical call, bitwise.
    FF::BasicPosterior<Model> standard(model, data);
    check(max_difference(standard.apply_field_metric(x, p, w, -1), model.apply_field_metric(x, p, w, -1)) == 0,
          label + " default noise policy keeps the historical metric bitwise");
    // Fixed noise far from the (unused) last scalar.
    const double fixed = 1e-2 * std::exp(p.back());
    FF::BasicPosterior<Model> fixed_posterior(model, data, FF::GaussianNoise::fixed_power(fixed));
    auto q = p;
    q.back() = std::log(fixed);
    const auto expected = model.apply_field_metric(x, q, w, -1);
    const auto actual = fixed_posterior.apply_field_metric(x, p, w, -1);
    check(max_difference(actual, expected) <= 1e-14 * (1 + max_difference(expected, Field(expected.size(), 0.))),
          label + " fixed-noise posterior metric uses the fixed noise");
    check(max_difference(actual, model.apply_field_metric(x, p, w, -1)) > 1e-6,
          label + " fixed-noise metric differs from the unused last-scalar metric");
    // Options MAP runs and descends with the fixed-noise posterior.
    FF::MAPOptions options;
    options.vary_parameters = false;
    options.max_iterations = 3;
    const auto fit = FF::map(fixed_posterior, s, p, options);
    check(fit.trace.size() >= 2 && fit.fit.potential < fit.trace.front().potential,
          label + " options MAP descends with the posterior metric");
}

int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    FF::Configuration c;
    c.mesh = 16;
    c.density_mesh = 32;
    c.boxsize = 100;
    c.initial_cutoff = .2;
    c.bias_cutoff = .2;
    c.likelihood_cutoff = .2;
    const auto spectrum = FF::Spectrum::fixed([](double k) { return 3e3 * std::exp(-4 * k); });
    {
        auto linear = c;
        linear.evolution = FF::Evolution::Linear;
        exercise(FF::DistributedLinearModel(linear, spectrum), "linear");
        if (FML::NTasks == 1)
            exercise(FF::Model(linear, spectrum), "serial linear");
    }
    if (FML::NTasks == 1 || 16 % FML::NTasks == 0)
        exercise(FF::DistributedLPTModel(c, spectrum), "2lpt");
    if (FML::NTasks == 1)
        exercise(FF::Model(c, spectrum), "serial 2lpt");

    // The Eulerian galaxy model has only a model-specific metric: the options MAP must compile and run.
    FF::EulerianLPTConfiguration e;
    e.mesh = 12;
    e.density_mesh = 24;
    e.boxsize = 100;
    const double kf = 2 * M_PI / e.boxsize;
    e.initial_cutoff = 1.6 * kf;
    e.matter_band = {2.5 * kf, E::BandGeometry::Cube};
    e.likelihood_band = {1.6 * kf, E::BandGeometry::Cube};
    std::array<FF::Parameter, 4> priors{{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                         {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                         {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                         {"b_laplacian", 7., 0, 1, -100, 100, FF::Prior::Uniform}}};
    if (FML::NTasks == 1 || 12 % FML::NTasks == 0) {
        FF::DistributedEulerianLPTModel model(e, FF::sigma8_spectrum([](double k) { return 500 * std::exp(-5 * k); }),
                                              priors);
        auto w = model.workspace();
        const auto p = model.initial_parameters();
        const auto s = model.white_noise(21);
        auto posterior = FF::fixed_noise_posterior(model, model.mock(s, p, 1000, 22, w), 1000.);
        FF::MAPOptions options;
        options.max_iterations = 3;
        const auto fit = FF::map(posterior, s, p, options);
        check(fit.trace.size() >= 2 && fit.fit.potential < fit.trace.front().potential,
              "eulerian options MAP compiles and descends");
    }
    return FML::TESTS::summary("metric noise-policy");
}
