#include <FML/FieldLevel/FixedIC.h>
#include <FML/tests/TestHelpers.h>
#include <iomanip>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::TESTS::check;
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    // Partition an explicit Fourier likelihood across tasks; ranks with no modes are valid.
    std::array<FF::Modes, 2> columns;
    FF::Modes data;
    std::vector<double> weights;
    for (int k = 0; k < 7; ++k)
        if (k % FML::NTasks == FML::ThisTask) {
            columns[0].push_back({1 + .1 * k, .03 * k});
            columns[1].push_back({.1 - .04 * k, -.07 * k});
            data.push_back({.2 * k, -.1 * k});
            weights.push_back(k ? 2 : 1);
        }
    FF::FixedNoiseLinearBiasLikelihood<2> cache(columns, data, weights, .3, {.8, -.4});
    for (auto b : {std::array<double, 2>{.8, -.4}, {1.7, .2}, {-.5, 1.3}}) {
        double u = 0;
        std::array<double, 2> g{};
        for (size_t k = 0; k < data.size(); ++k) {
            auto r = b[0] * columns[0][k] + b[1] * columns[1][k] - data[k];
            u += .5 * weights[k] * (std::norm(r) / .3 + std::log(.3));
            for (size_t a = 0; a < 2; ++a)
                g[a] += weights[k] * std::real(std::conj(columns[a][k]) * r) / .3;
        }
        FML::SumOverTasks(&u);
        FML::SumArrayOverTasks(g.data(), g.size());
        auto cg = cache.gradient(b);
        check(FML::TESTS::near(u, cache.value(b), 2e-13) && FML::TESTS::near(g[0], cg[0], 2e-13) &&
                  FML::TESTS::near(g[1], cg[1], 2e-13),
              "cached bias value/gradient equals direct distributed residual sum");
    }
    auto bad = columns;
    if (FML::ThisTask == 0)
        bad[1].push_back(0.);
    check(FML::TESTS::throws([&] {
              FF::FixedNoiseLinearBiasLikelihood<2> invalid(bad, data, weights, .3, {.8, -.4});
          }),
          "malformed local basis rejected collectively");
    std::array<FF::Parameter, 2> gaussian_priors = {
        {{"a", 0, 1.4, .3, -3, 3, FF::Prior::Gaussian}, {"b", 0, -.2, .7, -3, 3, FF::Prior::Gaussian}}};
    auto conditional = cache.gaussian_proposal(gaussian_priors);
    std::array<double, 6> stats{};
    for (size_t k = 0; k < data.size(); ++k) {
        const double w = weights[k] / .3;
        stats[0] += w * std::norm(columns[0][k]);
        stats[1] += w * std::norm(columns[1][k]);
        stats[2] += w * std::real(std::conj(columns[0][k]) * columns[1][k]);
        stats[3] += w * std::real(std::conj(columns[0][k]) * data[k]);
        stats[4] += w * std::real(std::conj(columns[1][k]) * data[k]);
    }
    FML::SumArrayOverTasks(stats.data(), stats.size());
    double h00 = stats[0] + 1 / (.3 * .3), h11 = stats[1] + 1 / (.7 * .7), h01 = stats[2];
    double rhs0 = stats[3] + 1.4 / (.3 * .3), rhs1 = stats[4] - .2 / (.7 * .7), determinant = h00 * h11 - h01 * h01;
    std::array<double, 2> mu = {(h11 * rhs0 - h01 * rhs1) / determinant, (h00 * rhs1 - h01 * rhs0) / determinant};
    const double x0 = .1 - mu[0], x1 = .2 - mu[1];
    const double log_q =
        .5 * std::log(determinant) - std::log(2 * M_PI) - .5 * (h00 * x0 * x0 + 2 * h01 * x0 * x1 + h11 * x1 * x1);
    check(FML::TESTS::near(conditional.mean()[0], mu[0], 2e-13) &&
              FML::TESTS::near(conditional.mean()[1], mu[1], 2e-13) &&
              FML::TESTS::near(conditional.log_density({.1, .2}), log_q, 2e-13),
          "Gaussian-prior scaled proposal equals independently solved conditional mean/precision/determinant");
    // Independent 2D quadrature reference. The reparameterized target must include alpha^-1.
    struct ToyCache {
        double alpha;
        double value(const std::array<double, 1> & b) const { return .5 * std::pow((alpha * b[0] - 1.2) / .3, 2); }
    };
    FF::Parameter amplitude{"alpha", 1, 1, 1, .5, 1.5, FF::Prior::Uniform};
    std::array<FF::Parameter, 1> biases = {{{"b", 1, 1, 1, .1, 3, FF::Prior::Uniform}}};
    const int n = 800;
    double norm = 0, amean = 0, bmean = 0, acdf = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double a = .5 + (i + .5) / n, b = .1 + 2.9 * (j + .5) / n;
            double w = std::exp(-ToyCache{a}.value({b}));
            norm += w;
            amean += w * a;
            bmean += w * b;
            acdf += w * (a <= 1.);
        }
    amean /= norm;
    bmean /= norm;
    acdf /= norm;
    double sum_a = 0, sum_b = 0, cdf = 0, count = 0, acceptance = 0;
    for (int chain = 0; chain < 4; ++chain) {
        FF::FixedICOptions options;
        options.warmup = 512;
        options.samples = 8192;
        options.bias_sweeps = 2;
        options.amplitude_step = .25;
        options.seed = 12000 + chain;
        auto result = FF::sample_fixed_ic_amplitude<1>([](double a) { return ToyCache{a}; },
                                                       amplitude,
                                                       biases,
                                                       {1.},
                                                       {.6 + .25 * chain, 1.2},
                                                       options);
        check(result.draws.size() == size_t(options.samples) && result.potential.size() == result.draws.size(),
              "scalar sampler retains every draw including rejected amplitudes");
        for (auto d : result.draws) {
            sum_a += d[0];
            sum_b += d[1];
            cdf += d[0] <= 1.;
            ++count;
        }
        acceptance += result.sampling_acceptance;
    }
    check(std::abs(sum_a / count - amean) < .012 && std::abs(sum_b / count - bmean) < .02 &&
              std::abs(cdf / count - acdf) < .02,
          "amplitude/bias joint posterior matches independent quadrature including Jacobian");
    double lo = sum_a, hi = sum_a;
    FML::MinOverTasks(&lo);
    FML::MaxOverTasks(&hi);
    check(lo == hi && acceptance / 4 > .4, "replicated sampler RNG decisions agree across MPI ranks");
    FF::FixedICOptions short_run;
    short_run.samples = 8;
    short_run.warmup = 2;
    auto provider = [](double a) { return ToyCache{a}; };
    auto first = FF::sample_fixed_ic_amplitude<1>(provider, amplitude, biases, {1}, {1, 1}, short_run);
    auto replay = FF::sample_fixed_ic_amplitude<1>(provider, amplitude, biases, {1}, {1, 1}, short_run);
    check(first.draws == replay.draws && first.potential == replay.potential, "seeded scalar chains replay exactly");
    auto restricted = [](double alpha) {
        if (alpha > 1.1)
            throw FF::UnrepresentableState("test numerical domain");
        return ToyCache{alpha};
    };
    auto rejected = FF::sample_fixed_ic_amplitude<1>(restricted, amplitude, biases, {1}, {1, 1}, short_run);
    bool inside = true;
    for (auto d : rejected.draws)
        inside &= d[0] <= 1.1;
    check(inside, "unrepresentable amplitude proposals are rejected without losing the current cache");
    short_run.samples = 0;
    check(FML::TESTS::throws([&] {
              FF::sample_fixed_ic_amplitude<1>(provider, amplitude, biases, {1}, {1, 1}, short_run);
          }),
          "invalid sampling budget rejected collectively");
    if (FML::ThisTask == 0)
        std::cout << std::setprecision(17) << "QUADRATURE " << sum_a / count << ' ' << amean << ' ' << sum_b / count
                  << ' ' << bmean << ' ' << cdf / count << ' ' << acdf << '\n';
    // The normalized q must include its determinant and physical-coordinate scaling.
    const auto gaussian = FF::FixedICGaussianProposal<2>::from_precision({.3, -.4}, {4, 1, 1, 2});
    std::array<double, 2> point = {.8, .2};
    const double dx = point[0] - .3, dy = point[1] + .4;
    const double log_reference =
        .5 * std::log(7.) - std::log(2 * M_PI) - .5 * (4 * dx * dx + 2 * dx * dy + 2 * dy * dy);
    check(FML::TESTS::near(gaussian.log_density(point), log_reference, 1e-14),
          "Gaussian q includes normalized determinant and full correlated quadratic");
    std::mt19937_64 proposal_rng(4501);
    std::normal_distribution<double> normal;
    std::array<double, 2> means{};
    std::array<double, 4> second{};
    for (int draw = 0; draw < 100000; ++draw) {
        auto x = gaussian.draw(proposal_rng, normal);
        for (size_t i = 0; i < 2; ++i) {
            means[i] += x[i];
            for (size_t j = 0; j < 2; ++j)
                second[i * 2 + j] += x[i] * x[j];
        }
    }
    for (double & x : means)
        x /= 100000;
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            second[i * 2 + j] = second[i * 2 + j] / 100000 - means[i] * means[j];
    check(std::abs(means[0] - .3) < .008 && std::abs(means[1] + .4) < .008 && std::abs(second[0] - 2. / 7) < .009 &&
              std::abs(second[1] + 1. / 7) < .009 && std::abs(second[3] - 4. / 7) < .012,
          "Gaussian proposal draws have independent-reference mean and inverse-precision covariance");
    check(FML::TESTS::throws([&] {
              FF::FixedICGaussianProposal<2>::from_precision({0, 0}, {1, 3, 0, 1});
          }),
          "asymmetric Gaussian proposal precision is rejected");
    // A deliberately unidentified second coefficient must retain its UNIFORM prior, even though
    // proposal regularization approximates it with a Gaussian. No Gibbs refresh masks a wrong MH q ratio.
    std::array<FF::Parameter, 2> joint_priors = {biases[0],
                                                 FF::Parameter{"unidentified", 1, 0, 1, -2, 4, FF::Prior::Uniform}};
    auto joint_provider = [&](double a) {
        std::array<FF::Modes, 2> c;
        FF::Modes d;
        std::vector<double> w;
        if (FML::ThisTask == 0) {
            c[0] = {a};
            c[1] = {0.};
            d = {1.2};
            w = {1.};
        }
        return FF::FixedNoiseLinearBiasLikelihood<2>(c, d, w, .09, {1, 1});
    };
    double ja = 0, jb = 0, jc = 0, unidentified = 0, unidentified2 = 0, jcount = 0;
    int support_rejections = 0;
    for (int chain = 0; chain < 4; ++chain) {
        FF::FixedICOptions opt;
        opt.warmup = 512;
        opt.samples = 8192;
        opt.bias_sweeps = 0;
        opt.amplitude_step = .25;
        opt.seed = 22000 + chain;
        auto result = FF::sample_fixed_ic_joint_amplitude<2>(joint_provider,
                                                             amplitude,
                                                             joint_priors,
                                                             {.6 + .25 * chain, 1.2, 1},
                                                             opt);
        support_rejections += result.support_rejections;
        for (auto x : result.draws) {
            ja += x[0];
            jb += x[1];
            jc += x[0] <= 1.;
            unidentified += x[2];
            unidentified2 += x[2] * x[2];
            ++jcount;
        }
    }
    check(std::abs(ja / jcount - amean) < .014 && std::abs(jb / jcount - bmean) < .023 &&
              std::abs(jc / jcount - acdf) < .024,
          "joint Gaussian kernel matches bounded posterior quadrature with changing q determinant");
    check(std::abs(unidentified / jcount - 1.) < .045 &&
              std::abs(unidentified2 / jcount - std::pow(unidentified / jcount, 2) - 3.) < .14 &&
              support_rejections > 0,
          "rank-deficient likelihood keeps uniform-prior mean/variance through one-shot support rejection");
    FF::FixedICOptions replay_options;
    replay_options.warmup = 3;
    replay_options.samples = 8;
    replay_options.bias_sweeps = 0;
    auto j1 =
        FF::sample_fixed_ic_joint_amplitude<2>(joint_provider, amplitude, joint_priors, {1, 1, 1}, replay_options);
    auto j2 =
        FF::sample_fixed_ic_joint_amplitude<2>(joint_provider, amplitude, joint_priors, {1, 1, 1}, replay_options);
    check(j1.draws == j2.draws && j1.potential == j2.potential, "joint sampler seeded replay is exact");
    if (FML::ThisTask == 0)
        std::cout << "JOINT_QUADRATURE " << ja / jcount << ' ' << amean << ' ' << jb / jcount << ' ' << bmean << ' '
                  << unidentified / jcount << ' ' << unidentified2 / jcount - std::pow(unidentified / jcount, 2)
                  << '\n';
    return FML::TESTS::summary("fixed IC likelihood and amplitude sampler");
}
