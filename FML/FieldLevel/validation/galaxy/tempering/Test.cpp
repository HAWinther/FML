#include "Exchange.h"
#include <FML/tests/TestHelpers.h>
#include <iomanip>
using FML::TESTS::check;
static double gaussian_layout(const std::vector<double> & beta, unsigned passes) {
    EXCHANGE::layout(beta, passes);
    const size_t n = beta.size();
    std::vector<EXCHANGE::PhysicalState> states(n);
    std::vector<std::array<double, 5>> moments(n);
    for (size_t j = 0; j < n; ++j)
        states[j].label = j;
    std::mt19937_64 rng(817);
    std::normal_distribution<double> normal;
    const double y = 2.3, sigma2 = .3;
    const int warmup = 1024, samples = 100000;
    for (int it = 0; it < warmup + samples; ++it) {
        for (size_t j = 0; j < n; ++j) {
            const double den = sigma2 + 2 * beta[j];
            const double sum = std::sqrt(2.) * beta[j] * y / den + std::sqrt(sigma2 / den) * normal(rng),
                         difference = normal(rng);
            const double x = (sum + difference) / std::sqrt(2.), v = (sum - difference) / std::sqrt(2.);
            states[j].white = {x};
            states[j].parameters = {v};
            states[j].quadratic = .5 * std::pow(x + v - y, 2) / sigma2;
        }
        for (unsigned pass = 0; pass < passes; ++pass)
            EXCHANGE::neighbors(states, beta, unsigned((it + pass) % 2), rng);
        if (it >= warmup)
            for (size_t j = 0; j < n; ++j) {
                const double x = states[j].white[0], v = states[j].parameters[0];
                moments[j][0] += x;
                moments[j][1] += v;
                moments[j][2] += x * x;
                moments[j][3] += v * v;
                moments[j][4] += x * v;
            }
    }
    double worst = 0;
    for (size_t j = 0; j < n; ++j) {
        for (double & v : moments[j])
            v /= samples;
        const double mean = beta[j] * y / (sigma2 + 2 * beta[j]), variance = 1 - beta[j] / (sigma2 + 2 * beta[j]),
                     cov = -beta[j] / (sigma2 + 2 * beta[j]);
        const double error = std::max({std::abs(moments[j][0] - mean),
                                       std::abs(moments[j][1] - mean),
                                       std::abs(moments[j][2] - moments[j][0] * moments[j][0] - variance),
                                       std::abs(moments[j][3] - moments[j][1] * moments[j][1] - variance),
                                       std::abs(moments[j][4] - moments[j][0] * moments[j][1] - cov)});
        worst = std::max(worst, error);
        check(error < .02, "eleven-level repeated exchanges preserve analytic correlated Gaussian marginal");
    }
    return worst;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (FML::NTasks != 1)
            throw std::invalid_argument("Serial replica validation required");
        check(EXCHANGE::log_ratio(1, .25, 2, 10) == -6 && EXCHANGE::log_ratio(1, .25, 10, 2) == 6,
              "swap sign and reciprocal ratio agree with joint tempered-density change");
        check(FML::TESTS::throws([] {
                  EXCHANGE::ladder({1, .5, .5, 0});
              }) &&
                  FML::TESTS::throws([] {
                      EXCHANGE::ladder({1, .5});
                  }),
              "malformed temperature ladders reject");
        check(FML::TESTS::throws([] { EXCHANGE::labels({}); }), "empty ensembles reject before endpoint access");
        std::vector<double> beta = {1, .5, .25, .125, .0625, 0};
        std::vector<EXCHANGE::PhysicalState> states(6);
        for (size_t j = 0; j < 6; ++j) {
            states[j].label = j;
            states[j].white = {double(j)};
            states[j].parameters = {10 + double(j)};
        }
        auto bad = states;
        bad[1].quadratic = HUGE_VAL;
        std::mt19937_64 rng(717);
        check(FML::TESTS::throws([&] { EXCHANGE::neighbors(bad, beta, 0, rng); }) && bad[0].label == 0,
              "invalid replica likelihood rejected before exchange mutation");
        std::vector<unsigned> phase(6);
        std::vector<uint64_t> trips(6);
        EXCHANGE::observe(states, phase, trips);
        std::swap(states.front(), states.back());
        EXCHANGE::observe(states, phase, trips);
        std::swap(states.front(), states.back());
        EXCHANGE::observe(states, phase, trips);
        check(trips[0] == 1 && phase[0] == 1, "transport counts cold-prior-cold visits by physical walker label");
        // Independent correlated-Gaussian reference: L=(x+a-y)^2/(2 sigma2), both priors N(0,1).
        // Exact local draws then exchanges test the complete joint invariant law at every beta.
        const double y = 2.3, sigma2 = .3;
        std::normal_distribution<double> normal;
        std::array<std::array<double, 5>, 6> moments{};
        for (size_t j = 0; j < 6; ++j)
            states[j].label = j;
        const int warmup = 1024, samples = 100000;
        for (int it = 0; it < warmup + samples; ++it) {
            for (size_t j = 0; j < 6; ++j) {
                const double denominator = sigma2 + 2 * beta[j];
                const double sum =
                    std::sqrt(2.) * beta[j] * y / denominator + std::sqrt(sigma2 / denominator) * normal(rng);
                const double difference = normal(rng);
                const double x = (sum + difference) / std::sqrt(2.), a = (sum - difference) / std::sqrt(2.);
                states[j].white = {x};
                states[j].parameters = {a};
                states[j].quadratic = .5 * std::pow(x + a - y, 2) / sigma2;
            }
            const auto outcomes = EXCHANGE::neighbors(states, beta, unsigned(it % 2), rng);
            if (it >= warmup)
                for (size_t j = 0; j < 6; ++j) {
                    const double x = states[j].white[0], a = states[j].parameters[0];
                    moments[j][0] += x;
                    moments[j][1] += a;
                    moments[j][2] += x * x;
                    moments[j][3] += a * a;
                    moments[j][4] += x * a;
                }
        }
        double worst = 0;
        for (size_t j = 0; j < 6; ++j) {
            for (double & m : moments[j])
                m /= samples;
            const double mean = beta[j] * y / (sigma2 + 2 * beta[j]), variance = 1 - beta[j] / (sigma2 + 2 * beta[j]),
                         covariance = -beta[j] / (sigma2 + 2 * beta[j]);
            const double error = std::max({std::abs(moments[j][0] - mean),
                                           std::abs(moments[j][1] - mean),
                                           std::abs(moments[j][2] - moments[j][0] * moments[j][0] - variance),
                                           std::abs(moments[j][3] - moments[j][1] * moments[j][1] - variance),
                                           std::abs(moments[j][4] - moments[j][0] * moments[j][1] - covariance)});
            worst = std::max(worst, error);
            check(error < .02, "complete-state exchange preserves analytic correlated Gaussian marginal at this beta");
        }
        std::cout << std::setprecision(17) << "EXCHANGE_GAUSSIAN_CHECK " << worst << '\n';
        check(EXCHANGE::expected_attempts(0, 5, 1) == 3 && EXCHANGE::expected_attempts(1, 5, 1) == 2 &&
                  EXCHANGE::expected_attempts(0, 5, 4) == 10 && EXCHANGE::expected_attempts(1, 5, 4) == 10,
              "repeat-pass edge counters preserve parity and legacy counts");
        std::vector<double> too_many(65);
        for (size_t j = 0; j < 65; ++j)
            too_many[j] = 1 - double(j) / 64;
        check(FML::TESTS::throws([&] { EXCHANGE::layout(too_many, 1); }) &&
                  FML::TESTS::throws([&] { EXCHANGE::layout(beta, 0); }) &&
                  FML::TESTS::throws([&] { EXCHANGE::layout(beta, 9); }),
              "unsupported replica/pass counts reject");
        std::vector<EXCHANGE::PhysicalState> transport(6);
        for (size_t j = 0; j < 6; ++j)
            transport[j].label = j;
        std::vector<unsigned> qualified(6);
        std::vector<uint64_t> completed(6);
        EXCHANGE::observe(transport, qualified, completed, false);
        std::swap(transport.front(), transport.back());
        EXCHANGE::observe(transport, qualified, completed, false);
        std::swap(transport.front(), transport.back());
        EXCHANGE::observe(transport, qualified, completed, false);
        const bool no_bounce = completed[0] == 0;
        std::swap(transport.front(), transport.back());
        EXCHANGE::prior_refresh_visit(transport.back().label, qualified);
        std::swap(transport.front(), transport.back());
        EXCHANGE::observe(transport, qualified, completed, false);
        check(no_bounce && completed[0] == 1, "prior bounces do not count without an actual refresh");
        const std::vector<double> bridge =
            {1, .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625, .001953125, 0};
        const double bridge_error = gaussian_layout(bridge, 4);
        std::cout << "BRIDGE_GAUSSIAN_CHECK " << bridge_error << '\n';
        return FML::TESTS::summary("replica-exchange kernel");
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
