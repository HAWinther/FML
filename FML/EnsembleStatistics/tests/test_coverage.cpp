#include <FML/EnsembleStatistics/ControlVariates.h>
#include <FML/RandomGenerator/CounterBasedRandom.h>
#include <FML/tests/TestHelpers.h>
#include <iomanip>
#include <iostream>
using namespace FML::ENSEMBLE;
int main() {
    // Independent repeated experiments with a known Gaussian target population mean.
    // Each whole experiment belongs to one rank; its phases never mix with another experiment.
    constexpr int trials = 2048, ntrain = 32, nmean = 256, neval = 64;
    Metadata tags{"Gaussian coverage v1", "one scalar", "5+1.7c+0.6epsilon", "unit Gaussian c"};
    RunningCovariance errors(3);
    Count covered = 0, naive_covered = 0;
    double predicted = 0, naive_predicted = 0;
    for (int trial = 0; trial < trials; ++trial)
        if (trial % FML::NTasks == FML::ThisTask) {
            auto key = [&](int row) {
                return PhaseKey{std::uint64_t(row), std::uint64_t(970000 + trial * 1000 + row)};
            };
            auto sample = [&](int row) {
                FML::RANDOM::COUNTER_BASED::Stream random(key(row).seed, 0);
                double c = random.normal();
                return std::pair<double, double>{5 + 1.7 * c + .6 * random.normal(), c};
            };
            PairedTraining training(1, 1, tags);
            for (int i = 0; i < ntrain; ++i) {
                auto s = sample(i);
                training.add({s.first}, {s.second}, key(i), key(i));
            }
            ControlMean mean(1, tags);
            for (int i = ntrain; i < ntrain + nmean; ++i)
                mean.add({sample(i).second}, key(i));
            auto model = training.fit(mean.estimate());
            ControlEvaluation evaluation(model);
            RunningCovariance raw(1);
            for (int i = ntrain + nmean; i < ntrain + nmean + neval; ++i) {
                auto s = sample(i);
                raw.add({s.first});
                evaluation.add({s.first}, {s.second}, key(i), key(i));
            }
            const auto result = evaluation.estimate();
            const double error = result.mean[0] - 5;
            const double variance = result.covariance_of_mean[0], naive = result.sample_covariance[0] / neval;
            covered += std::abs(error) <= 1.959963984540054 * std::sqrt(variance);
            naive_covered += std::abs(error) <= 1.959963984540054 * std::sqrt(naive);
            predicted += variance;
            naive_predicted += naive;
            errors.add({error, variance, raw.summary().mean[0] - 5});
        }
    const auto s = errors.global().summary();
    FML::SumOverTasks(&covered);
    FML::SumOverTasks(&naive_covered);
    FML::SumOverTasks(&predicted);
    FML::SumOverTasks(&naive_predicted);
    const double empirical = s.covariance[0], ratio = predicted / trials / empirical,
                 coverage = double(covered) / trials;
    FML::TESTS::check(s.count == trials && std::abs(s.mean[0]) < 4 * std::sqrt(empirical / trials),
                      "repeated independent-mean experiments retain the known Gaussian target mean");
    FML::TESTS::check(ratio > .9 && ratio < 1.1, "reported covariance of the mean matches repeated-experiment scatter");
    FML::TESTS::check(coverage > .925 && coverage < .975,
                      "nominal 95 percent Gaussian intervals cover across independent experiments");
    FML::TESTS::check(naive_covered < covered && double(naive_covered) / trials < .85,
                      "omitting the shared cheap-mean uncertainty produces detectable undercoverage");
    if (!FML::ThisTask)
        std::cout << std::setprecision(17) << "trials=" << trials << " bias=" << s.mean[0]
                  << " empirical_variance=" << empirical << " reported_variance=" << predicted / trials
                  << " naive_variance=" << naive_predicted / trials << " coverage=" << coverage
                  << " naive_coverage=" << double(naive_covered) / trials << '\n';
    return FML::TESTS::summary("ensemble coverage");
}
