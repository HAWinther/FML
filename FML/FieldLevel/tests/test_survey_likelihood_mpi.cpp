#include <FML/FieldLevel/SurveyLikelihood.h>
#include <FML/tests/TestHelpers.h>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Complex;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check_local;
static double rel(double a, double b) { return std::abs(a - b) / std::max({1., std::abs(a), std::abs(b)}); }
static std::array<int, 3> coordinate(size_t id, int N) { return {int(id / (N * N)), int(id / N % N), int(id % N)}; }
static double delta(std::array<int, 3> q, int N) {
    return 2 * std::real(Complex(.03, .02) * std::polar(1., 2 * M_PI * q[0] / N) +
                         Complex(.04, -.01) * std::polar(1., 2 * M_PI * q[2] / N));
}
static double probe(std::array<int, 3> q, int N) {
    return .1 * std::cos(2 * M_PI * q[0] / N) + .04 * std::sin(2 * M_PI * (q[1] + q[2]) / N);
}
template <class Model, class Function>
static FF::Modes direct_adjoint(const Model & m, Function value) {
    const int N = m.configuration().mesh;
    FF::Modes result(m.modes().size());
    for (size_t i = 0; i < result.size(); ++i) {
        const auto & k = m.modes()[i];
        if (k.k >= m.configuration().likelihood_cutoff)
            continue;
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y)
                for (int z = 0; z < N; ++z)
                    result[i] += value(std::array<int, 3>{x, y, z}) *
                                 std::polar(1., -2 * M_PI * (k.n[0] * x + k.n[1] * y + k.n[2] * z) / N);
    }
    return result;
}
template <class Model>
static size_t global(const Model & m, size_t q) {
    const int N = m.configuration().mesh;
    return size_t(m.layout().local_start) * N * N + q;
}
static double mode_error(const FF::Modes & a, const FF::Modes & b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}
static void reference_tests() {
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.evolution = FF::Evolution::Linear;
    FF::DistributedLinearModel m(c, FF::Spectrum::fixed([](double) { return 200.; }));
    const int N = c.mesh;
    auto window = FF::make_survey_window(m, {-50, -50, -50}, [](auto p) {
        return FF::SurveyVoxel{p[0] < 25 && p[2] != -50, .6 + .002 * p[1]};
    });
    FF::Modes prediction(m.modes().size());
    for (size_t i = 0; i < prediction.size(); ++i) {
        const auto & k = m.modes()[i];
        if (k.n[0] == 1 && k.n[1] == 0 && k.n[2] == 0)
            prediction[i] = {.03, .02};
        if (k.n[0] == -1 && k.n[1] == 0 && k.n[2] == 0)
            prediction[i] = {.03, -.02};
        if (k.n[0] == 0 && k.n[1] == 0 && k.n[2] == 1)
            prediction[i] = {.04, -.01};
    }
    FF::CartesianSurveyObservation obs(m, window);
    const auto image = obs.synthesize(prediction);
    Field test(m.cells());
    double e = 0, left = 0, right = 0;
    for (size_t q = 0; q < image.size(); ++q) {
        auto xyz = coordinate(global(m, q), N);
        e = std::max(e, std::abs(image[q] - delta(xyz, N)));
        test[q] = probe(xyz, N);
        left += test[q] * image[q];
    }
    const auto bar = obs.adjoint(test), expected = direct_adjoint(m, [&](auto q) { return probe(q, N); });
    for (size_t i = 0; i < bar.size(); ++i)
        right += m.modes()[i].weight * std::real(std::conj(bar[i]) * prediction[i]);
    check_local(m.maximum(e) < 1e-12, "survey synthesis matches analytic real Fourier waves");
    check_local(m.maximum(mode_error(bar, expected)) < 2e-11,
                "survey reverse matches independent direct DFT with N^3 normalization");
    check_local(rel(m.sum(left), m.sum(right)) < 1e-12, "survey forward/reverse full-Hermitian dot product");
    auto large = prediction;
    for (auto & z : large)
        z *= 1e12;
    const auto large_image = obs.synthesize(large);
    double large_error = 0;
    for (size_t q = 0; q < large_image.size(); ++q)
        large_error = std::max(large_error, std::abs(large_image[q] / 1e12 - delta(coordinate(global(m, q), N), N)));
    check_local(m.maximum(large_error) < 1e-12, "large valid Hermitian input tolerates FFT roundoff in zero modes");
    check_local(FML::TESTS::throws([&] {
                    auto malformed = large;
                    for (size_t i = 0; i < malformed.size(); ++i) {
                        const auto & k = m.modes()[i];
                        if (k.n[0] == 1 && k.n[1] == 0 && k.n[2] == 0)
                            malformed[i] += Complex(1e9, 0);
                    }
                    obs.synthesize(malformed);
                }),
                "large malformed Hermitian input still fails collectively");
    auto active = [&](auto xyz) { return xyz[0] < 6 && xyz[2] != 0; };
    auto selection = [&](auto xyz) { return .6 + .002 * (-50 + 12.5 * xyz[1]); };
    auto data_value = [&](auto xyz) { return .02 * std::cos(2 * M_PI * xyz[1] / N); };
    auto variance = [&](auto xyz) {
        const size_t id = (size_t(xyz[0]) * N + xyz[1]) * N + xyz[2];
        return 1. + .02 * (id % 7);
    };
    Field data(m.cells()), var(m.cells()), exposure(m.cells());
    std::vector<std::uint64_t> counts(m.cells());
    for (size_t q = 0; q < data.size(); ++q) {
        auto xyz = coordinate(global(m, q), N);
        data[q] = active(xyz) ? data_value(xyz) : NAN;
        var[q] = active(xyz) ? variance(xyz) : NAN;
        exposure[q] = 2 + .1 * (global(m, q) % 5);
        counts[q] = global(m, q) % 4;
    }
    const double log_noise = std::log(.15 * obs.cell_volume());
    FF::SurveyGaussianLikelihood gaussian(m, window, data, var);
    FF::Modes g;
    double gn;
    const double U = m.sum(gaussian.evaluate(prediction, log_noise, g, gn));
    double exactU = 0, exactNoise = 0;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                std::array<int, 3> q{x, y, z};
                if (!active(q))
                    continue;
                const double v = .15 * variance(q), r = selection(q) * delta(q, N) - data_value(q), chi = r * r / v;
                exactU += .5 * (chi + std::log(2 * M_PI * v));
                exactNoise += .5 * (1 - chi);
            }
    const auto gg = direct_adjoint(m, [&](auto q) {
        return active(q) ? selection(q) * (selection(q) * delta(q, N) - data_value(q)) / (.15 * variance(q)) : 0.;
    });
    check_local(rel(U, exactU) < 1e-12 && rel(m.sum(gn), exactNoise) < 1e-12,
                "masked heteroscedastic Gaussian normalization and log-noise derivative");
    check_local(m.maximum(mode_error(g, gg)) < 1e-10,
                "Gaussian selection occurs once in mean and once in its chain rule");
    const double h = 1e-5;
    FF::Modes scratch;
    double scratchNoise;
    const double fd = m.sum(gaussian.evaluate(prediction, log_noise + h, scratch, scratchNoise) -
                            gaussian.evaluate(prediction, log_noise - h, scratch, scratchNoise)) /
                      (2 * h);
    check_local(rel(fd, m.sum(gn)) < 1e-8, "Gaussian sampled noise normalization finite difference");
    FF::SurveyPoissonLikelihood poisson(m, window, counts, exposure, .1);
    const auto rate = poisson.expected_counts(prediction);
    double rateError = 0;
    const double V = m.sum(poisson.evaluate(prediction, 123, g, gn));
    exactU = 0;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                std::array<int, 3> q{x, y, z};
                if (!active(q))
                    continue;
                const size_t id = (size_t(x) * N + y) * N + z;
                const double n = id % 4, lr = std::log(2 + .1 * (id % 5)) + std::log(selection(q)) + delta(q, N) - .1;
                exactU += std::exp(lr) - n * lr + std::lgamma(n + 1);
            }
    const auto pg = direct_adjoint(m, [&](auto q) {
        if (!active(q))
            return 0.;
        const size_t id = (size_t(q[0]) * N + q[1]) * N + q[2];
        return (2 + .1 * (id % 5)) * selection(q) * std::exp(delta(q, N) - .1) - double(id % 4);
    });
    for (size_t q = 0; q < rate.size(); ++q) {
        auto xyz = coordinate(global(m, q), N);
        const double r = active(xyz) ? exposure[q] * selection(xyz) * std::exp(delta(xyz, N) - .1) : 0.;
        rateError = std::max(rateError, std::abs(rate[q] - r));
    }
    check_local(rel(V, exactU) < 1e-12 && gn == 0 && m.maximum(rateError) < 1e-12,
                "positive Poisson link, exposure, selection and full count normalization");
    check_local(m.maximum(mode_error(g, pg)) < 1e-10, "Poisson gradient matches independent voxel formula/direct DFT");
    check_local(FML::TESTS::throws([&] {
                    auto bad = var;
                    if (FML::ThisTask == 0)
                        bad[1] = -1;
                    FF::SurveyGaussianLikelihood invalid(m, window, data, bad);
                }),
                "one-rank bad variance fails collectively");
    check_local(FML::TESTS::throws([&] {
                    auto bad = window;
                    if (FML::ThisTask == 0)
                        bad.selection[0] = 1.1;
                    FF::CartesianSurveyObservation invalid(m, bad);
                }),
                "one-rank invalid completeness fails collectively");
    check_local(FML::TESTS::throws([&] {
                    auto bad = counts;
                    if (FML::ThisTask == 0)
                        bad.pop_back();
                    FF::SurveyPoissonLikelihood invalid(m, window, bad, exposure);
                }),
                "one-rank count shape fails collectively");
    check_local(FML::TESTS::throws([&] {
                    FF::make_survey_window(m, {0, 0, 0}, [&](auto) {
                        if (FML::ThisTask == 0)
                            throw std::runtime_error("callback");
                        return FF::SurveyVoxel{};
                    });
                }),
                "one-rank geometry callback failure propagates collectively");
    check_local(FML::TESTS::throws([&] {
                    auto bad = prediction;
                    for (size_t i = 0; i < bad.size(); ++i) {
                        const auto & k = m.modes()[i];
                        if (k.n[0] == 1 && k.n[1] == 0 && k.n[2] == 0)
                            bad[i] += Complex(0, .2);
                    }
                    obs.synthesize(bad);
                }),
                "one-owner non-Hermitian prediction fails collectively");
    check_local(FML::TESTS::throws([&] {
                    auto bad = test;
                    if (FML::ThisTask == 0)
                        bad.pop_back();
                    obs.adjoint(bad);
                }),
                "one-rank real cotangent shape fails collectively before FFT");
    check_local(FML::TESTS::throws([&] {
                    auto bad = window;
                    if (FML::ThisTask == 0)
                        bad.origin[0] = NAN;
                    FF::CartesianSurveyObservation invalid(m, bad);
                }),
                "nonfinite survey origin rejected in serial and collectively");
    check_local(FML::TESTS::throws([&] {
                    FF::SurveyPoissonLikelihood invalid(m, window, counts, exposure, FML::ThisTask == 0 ? NAN : 0.);
                }),
                "nonfinite count-link offset rejected in serial and collectively");
    auto empty = window;
    std::fill(empty.observed.begin(), empty.observed.end(), 0);
    FF::SurveyGaussianLikelihood missing(m, empty, Field(m.cells(), NAN), Field(m.cells(), NAN));
    const double zero = m.sum(missing.evaluate(prediction, log_noise, g, gn));
    check_local(zero == 0 && gn == 0 && m.maximum(mode_error(g, FF::Modes(g.size(), 0))) == 0,
                "empty observed footprint contributes zero likelihood/gradient");
    auto zeroWindow = FF::full_survey_window(m);
    Field zeros(m.cells(), 0);
    FF::SurveyPoissonLikelihood zeroExposure(m, zeroWindow, std::vector<std::uint64_t>(m.cells()), zeros);
    check_local(m.sum(zeroExposure.evaluate(prediction, 0, g, gn)) == 0,
                "zero exposure and zero count have exact zero contribution");
    check_local(FML::TESTS::throws([&] {
                    auto n = std::vector<std::uint64_t>(m.cells());
                    if (FML::ThisTask == 0)
                        n[0] = 1;
                    FF::SurveyPoissonLikelihood invalid(m, zeroWindow, n, zeros);
                }),
                "positive count at zero exposure fails collectively");
    FF::SurveyPoissonLikelihood underflow(m,
                                          zeroWindow,
                                          std::vector<std::uint64_t>(m.cells(), 1),
                                          Field(m.cells(), 1),
                                          1000);
    check_local(std::isfinite(m.sum(underflow.evaluate(prediction, 0, g, gn))),
                "Poisson underflow uses finite log rate without clipping");
    FF::SurveyPoissonLikelihood overflow(m,
                                         zeroWindow,
                                         std::vector<std::uint64_t>(m.cells()),
                                         Field(m.cells(), 1),
                                         -1000);
    check_local(FML::TESTS::throws<FF::UnrepresentableState>([&] { overflow.evaluate(prediction, 0, g, gn); }),
                "Poisson overflow rejects collectively as a numerical proposal failure");
}
template <class Likelihood>
static void posterior_tests(FF::DistributedLPTModel & m,
                            Likelihood likelihood,
                            const Field & s,
                            const std::vector<double> & p,
                            const std::string & label) {
    FF::BasicPosterior<FF::DistributedLPTModel, Likelihood> post(m, std::move(likelihood));
    auto w = m.workspace();
    Field gs, scratch;
    std::vector<double> gp, sp;
    const double U = post.evaluate(s, p, w, gs, gp), h = 1e-5;
    auto direction = m.white_noise(818), a = s, b = s;
    double analytic = 0;
    for (size_t q = 0; q < s.size(); ++q) {
        a[q] += h * direction[q];
        b[q] -= h * direction[q];
        analytic += gs[q] * direction[q];
    }
    const double fd = (post.evaluate(a, p, w, scratch, sp) - post.evaluate(b, p, w, scratch, sp)) / (2 * h);
    check_local(rel(m.sum(analytic), fd) < 2e-6, label + " full nonlinear white-field chain rule");
    for (size_t j = 0; j < p.size(); ++j) {
        auto x = p, y = p;
        const double step = h * std::max(1., std::abs(p[j]));
        x[j] += step;
        y[j] -= step;
        const double num = (post.evaluate(s, x, w, scratch, sp) - post.evaluate(s, y, w, scratch, sp)) / (2 * step);
        check_local(rel(num, gp[j]) < 2e-6, label + " scalar gradient " + m.parameters()[j].name);
    }
    const auto fit = FF::map(post, s, p, false, 8, 1e-7);
    check_local(std::isfinite(fit.potential) && fit.potential < U, label + " MAP decreases masked posterior");
    FML::MCMC::HMCOptions opt;
    opt.warmup = 4;
    opt.samples = 8;
    opt.n_leapfrog = 2;
    opt.step_size = .01;
    opt.adapt_step_size = false;
    opt.adapt_mass = false;
    opt.seed = 18;
    const auto chain = FF::sample(post, s, p, opt, false);
    check_local(chain.divergences == 0 && chain.acceptance > .7 && chain.last != s,
                label + " shared HMC integration smoke");
    if (FML::ThisTask == 0)
        std::cout << "SURVEY_POSTERIOR " << label << " U=" << U << " gradient0=" << gp[0] << '\n';
}
static void nonlinear_tests(FF::Evolution evolution) {
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.growth_rate = .6;
    c.evolution = evolution;
    FF::DistributedLPTModel m(c, FF::Spectrum::fixed([](double) { return 200.; }));
    auto p = m.initial_parameters();
    p[0] = 1.4;
    p[1] = .2;
    p[2] = -.1;
    p[3] = 4;
    const auto s = m.white_noise(31);
    auto window = FF::make_survey_window(m, {-50, -50, -50}, [](auto r) {
        return FF::SurveyVoxel{r[0] + r[1] < 20, .7 + .002 * r[2]};
    });
    Field data(m.cells()), var(m.cells(), 1), exposure(m.cells(), 2);
    std::vector<std::uint64_t> counts(m.cells());
    for (size_t q = 0; q < data.size(); ++q) {
        data[q] = .01 * std::cos(double(global(m, q)));
        counts[q] = global(m, q) % 3;
    }
    const std::string name = evolution == FF::Evolution::LPT2 ? "2LPT" : "ZA";
    posterior_tests(m, FF::SurveyGaussianLikelihood(m, window, data, var), s, p, name + " Gaussian");
    posterior_tests(m, FF::SurveyPoissonLikelihood(m, window, counts, exposure), s, p, name + " Poisson");
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    std::cout << std::setprecision(15) << std::unitbuf;
    reference_tests();
    nonlinear_tests(FF::Evolution::Zeldovich);
    nonlinear_tests(FF::Evolution::LPT2);
    return FML::TESTS::summary("survey observation and likelihood MPI");
}
