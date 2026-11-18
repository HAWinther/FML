#include "Rescaling.h"
#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <FML/tests/TestHelpers.h>
#include <random>
#include <iomanip>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
static double norm2(const Field & s) {
    double sum = 0;
    for (double x : s)
        sum += x * x;
    return sum;
}
static void toy(int n, bool gaussian, bool observed) {
    const double sigma2 = .49, y = 1.2;
    auto prior = [&](double a) { return gaussian ? .5 * std::pow((a - 1.) / .25, 2) : 0.; };
    auto value = [&](const Field & s, const std::vector<double> & p) {
        return .5 * norm2(s) + prior(p[0]) + (observed ? .5 * std::pow(y - p[0] * s[0], 2) / sigma2 : 0.);
    };
    const int points = 20000;
    double z = 0, mean = 0, cdf = 0;
    for (int k = 0; k < points; ++k) {
        const double a = .5 + (k + .5) / points;
        const double v = a * a + sigma2;
        const double weight = std::exp(-prior(a)) * (observed ? std::exp(-.5 * y * y / v) / std::sqrt(v) : 1.);
        z += weight;
        mean += a * weight;
        cdf += (a <= 1.) * weight;
    }
    mean /= z;
    cdf /= z;
    double sampled_mean = 0, sampled_cdf = 0, unused = 0;
    int count = 0;
    for (int chain = 0; chain < 4; ++chain) {
        std::mt19937_64 rng(61000 + 100 * n + 10 * gaussian + observed + chain * 1000);
        std::normal_distribution<double> normal;
        std::uniform_real_distribution<double> uniform;
        std::vector<double> p = {.6 + .25 * chain};
        Field s(n);
        for (int it = 0; it < 1024 + 32768; ++it) {
            for (double & x : s)
                x = normal(rng);
            if (observed)
                s[0] = p[0] * y / (p[0] * p[0] + sigma2) + std::sqrt(sigma2 / (p[0] * p[0] + sigma2)) * s[0];
            double u = value(s, p);
            for (int move = 0; move < 4; ++move) {
                const double a = RESCALING::reflected(p[0], .5, 1.5, .2 * normal(rng));
                auto trial = RESCALING::propose(s, p, 0, a);
                const double next = value(trial.white, trial.parameters);
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
                sampled_mean += p[0];
                sampled_cdf += p[0] <= 1.;
                if (n > 1)
                    unused += s[1] * s[1];
                ++count;
            }
        }
    }
    sampled_mean /= count;
    sampled_cdf /= count;
    unused /= count;
    check(std::abs(sampled_mean - mean) < .008 && std::abs(sampled_cdf - cdf) < .015,
          "rescaling kernel matches independent analytic Gaussian marginal/quadrature");
    if (n > 1)
        check(std::abs(unused - 1) < .025, "unused real coordinates retain unit Gaussian variance with full Jacobian");
    std::cout << "MARGINAL_CHECK " << n << ' ' << gaussian << ' ' << observed << ' ' << sampled_mean << ' ' << mean
              << ' ' << sampled_cdf << ' ' << cdf << ' ' << unused << '\n';
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (FML::NTasks != 1)
            throw std::invalid_argument("Serial interweaving validation required");
        const Field s = {.2, -.3, .8};
        const std::vector<double> p = {1., 7.};
        const auto a = RESCALING::propose(s, p, 0, 1.3);
        const auto b = RESCALING::propose(a.white, a.parameters, 0, 1.);
        check(std::abs(a.log_jacobian - 3 * std::log(1 / 1.3)) < 1e-14 &&
                  std::abs(a.log_jacobian + b.log_jacobian) < 1e-14,
              "forward/inverse use correct signed real-coordinate Jacobian");
        double error = 0;
        for (size_t j = 0; j < s.size(); ++j)
            error = std::max(error, std::abs(b.white[j] - s[j]));
        check(error < 1e-15 && a.parameters[1] == 7. && b.parameters == p,
              "reciprocal field transform preserves other scalars");
        const double forward = RESCALING::log_acceptance(.5 * norm2(s), .5 * norm2(a.white), a.log_jacobian);
        const double reverse = RESCALING::log_acceptance(.5 * norm2(a.white), .5 * norm2(s), b.log_jacobian);
        check(std::abs(forward + reverse) < 1e-14,
              "Metropolis log ratio is reciprocal including prior-only directions");
        check(RESCALING::reflected(1., .5, 1.5, 1.) == 1. && RESCALING::reflected(1., .5, 1.5, -1.) == 1.,
              "reflection handles both boundary crossings symmetrically");
        check(FML::TESTS::throws([&] { RESCALING::propose(s, p, 0, 0.); }) &&
                  FML::TESTS::throws([&] { RESCALING::propose(s, p, 2, 1.); }),
              "invalid amplitude and scalar index rejected");
        check(RESCALING::log_acceptance(1., HUGE_VAL, 0.) == -HUGE_VAL, "nonfinite proposal target rejects");
        for (int n : {1, 3})
            for (bool gaussian : {false, true})
                for (bool observed : {false, true})
                    toy(n, gaussian, observed);
        double prediction_error = 0, gap_error = 0;
        for (bool compact : {false, true})
            for (auto evolution : {FF::Evolution::Zeldovich, FF::Evolution::LPT2}) {
                FF::EulerianLPTConfiguration c;
                c.mesh = compact ? 4 : 8;
                c.product_mesh = compact ? 8 : 0;
                c.density_mesh = 16;
                c.boxsize = 100;
                c.evolution = evolution;
                const double kf = 2 * M_PI / 100;
                c.initial_cutoff = 1.7 * kf;
                c.matter_band = {1.8 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
                c.likelihood_band = {1.6 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
                c.matter_assignment = FF::MatterAssignment::PSWF;
                std::array<FF::Parameter, 4> biases = {{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                                        {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                                        {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                                        {"b_laplacian", 7, 0, 1, -100, 100, FF::Prior::Uniform}}};
                FF::DistributedEulerianLPTModel model(
                    c,
                    FF::sigma8_spectrum([](double k) { return 20000 * std::exp(-5 * k); }),
                    biases);
                auto work = model.workspace();
                const auto white = model.white_noise(4103), params = model.initial_parameters();
                const auto data = model.mock(model.white_noise(4101), params, 1000, 4102, work);
                const auto target = FF::fixed_noise_posterior(model, data, 1000.);
                const auto prediction = model.predict(white, params, work);
                const double u = target.value(white, params, work);
                for (double alpha : {.7, 1.3}) {
                    const auto trial = RESCALING::propose(white, params, 0, alpha);
                    const auto next = model.predict(trial.white, trial.parameters, work);
                    double e = 0;
                    for (size_t j = 0; j < next.size(); ++j)
                        e = std::max(e, std::abs(next[j] - prediction[j]) / std::max(1., std::abs(prediction[j])));
                    const double delta = target.value(trial.white, trial.parameters, work) - u;
                    const double expected = .5 * (norm2(trial.white) - norm2(white));
                    const double gap = std::abs(delta - expected) / std::max(1., std::abs(expected));
                    prediction_error = std::max(prediction_error, e);
                    gap_error = std::max(gap_error, gap);
                    check(e < 5e-10 && gap < 1e-9,
                          "shared/compact 1LPT/2LPT rescaling preserves prediction and target differs only by actual "
                          "white prior");
                }
            }
        std::cout << std::setprecision(17) << "RESCALING_MODEL_CHECK " << prediction_error << ' ' << gap_error << '\n';
        std::cout << "Interweaving validation complete\n";
        return FML::TESTS::summary("amplitude-field interweaving");
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
