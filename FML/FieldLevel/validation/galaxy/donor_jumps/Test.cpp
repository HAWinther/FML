#include "Donors.h"
#include <FML/tests/TestHelpers.h>
#include <map>
#include <iomanip>
using FML::TESTS::check;
double log_target(const DONORS::Field & w, const std::vector<double> & p) {
    if (p[0] < -5 || p[0] > 5)
        return -HUGE_VAL;
    const double a = std::log(.3) - .5 * std::pow((w[0] + 3) / .3, 2),
                 b = std::log(.7) - .5 * std::pow((w[0] - 3) / .3, 2), m = std::max(a, b);
    double out = m + std::log(std::exp(a - m) + std::exp(b - m)) - .5 * std::pow((p[0] - .5 * w[0]) / .4, 2);
    for (size_t j = 1; j < w.size(); ++j)
        out -= .5 * w[j] * w[j];
    return out;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (FML::NTasks != 1)
            throw std::invalid_argument("Serial fixed-donor controls");
        std::vector<DONORS::State> states = {{{-3, 0, 0, 0}, {-1.5}, 0, 0},
                                             {{3, 0, 0, 0}, {1.5}, 1, 0},
                                             {{-2.8, 0, 0, 0}, {-1.4}, 0, 1},
                                             {{3.2, 0, 0, 0}, {1.6}, 1, 1}};
        DONORS::Pool pool(states);
        std::map<std::vector<double>, int> counts;
        double reverse_error = 0, balance_error = 0;
        for (size_t i = 0; i < pool.size(); ++i)
            for (size_t j = 0; j < pool.size(); ++j)
                if (i != j)
                    for (double scale : {.35, 1.}) {
                        const DONORS::Field x = {-.2, .7, -.5, .3};
                        const std::vector<double> p = {.1};
                        auto y = pool.translate(x, p, i, j, scale),
                             back = pool.translate(y.white, y.parameters, j, i, scale);
                        for (size_t k = 0; k < x.size(); ++k)
                            reverse_error = std::max(reverse_error, std::abs(back.white[k] - x[k]));
                        reverse_error = std::max(reverse_error, std::abs(back.parameters[0] - p[0]));
                        std::vector<double> delta;
                        for (size_t k = 0; k < x.size(); ++k)
                            delta.push_back(y.white[k] - x[k]);
                        delta.push_back(y.parameters[0] - p[0]);
                        // Count exact donor deltas, avoiding subtraction of a nonzero floating origin.
                        auto zero = pool.translate(DONORS::Field(4, 0), std::vector<double>(1, 0), i, j, scale);
                        auto key = zero.white;
                        key.push_back(zero.parameters[0]);
                        ++counts[key];
                        const auto lx = log_target(x, p), ly = log_target(y.white, y.parameters);
                        if (std::isfinite(ly))
                            balance_error =
                                std::max(balance_error,
                                         std::abs((lx + std::min(0., ly - lx)) - (ly + std::min(0., lx - ly))));
                        check(y.white[1] == x[1] && y.white[2] == x[2] && y.white[3] == x[3],
                              "donor jumps preserve prior-null coordinates");
                    }
        bool symmetric = true;
        for (const auto & kv : counts) {
            auto negative = kv.first;
            for (auto & x : negative)
                x = -x;
            symmetric &= counts[negative] == kv.second;
        }
        check(symmetric && reverse_error < 2e-14,
              "uniform ordered donor pairs and scale mixture give reciprocal translations");
        check(balance_error < 2e-12, "independent MH pair detailed-balance identity including unequal mode weights");
        check(FML::TESTS::throws([&] {
                  pool.translate({0, 0, 0, 0}, {0}, 0, 0, 1);
              }) &&
                  FML::TESTS::throws([&] { pool.translate({0}, {0}, 0, 1, 1); }),
              "invalid pair and field dimensions rejected");
        double positive = 0, mean = 0, pmean = 0, unused = 0;
        size_t n = 0;
        for (unsigned chain = 0; chain < 4; ++chain) {
            std::mt19937_64 rng(942501 + chain);
            std::normal_distribution<double> normal;
            std::uniform_real_distribution<double> uniform;
            DONORS::Field w = {chain % 2 ? 3. : -3., 0, 0, 0};
            std::vector<double> p = {.5 * w[0]};
            double current = log_target(w, p);
            for (int it = 0; it < 2048 + 65536; ++it) {
                // A symmetric continuous local kernel plus one donor kernel; both use full target MH.
                auto local = w;
                auto lp = p;
                local[0] += .12 * normal(rng);
                lp[0] += .12 * normal(rng);
                double next = log_target(local, lp);
                if (std::log(std::max(uniform(rng), 1e-300)) < next - current) {
                    w = local;
                    p = lp;
                    current = next;
                }
                auto trial = pool.draw(w, p, rng);
                next = log_target(trial.white, trial.parameters);
                if (std::log(std::max(uniform(rng), 1e-300)) < next - current) {
                    w = std::move(trial.white);
                    p = std::move(trial.parameters);
                    current = next;
                }
                for (size_t j = 1; j < w.size(); ++j)
                    w[j] = normal(rng);
                current = log_target(w, p);
                if (it >= 2048) {
                    positive += w[0] > 0;
                    mean += w[0];
                    pmean += p[0];
                    unused += w[1] * w[1];
                    ++n;
                }
            }
        }
        positive /= n;
        mean /= n;
        pmean /= n;
        unused /= n;
        check(std::abs(positive - .7) < .018 && std::abs(mean - 1.2) < .1 && std::abs(pmean - .6) < .065,
              "correlated multimodal Gaussian-mixture weights and means match independent analytic reference");
        check(std::abs(unused - 1) < .02, "prior-null Gaussian variance retained by complete toy sampler");
        // The same local kernel without donor jumps should not be mistaken for mode mixing.
        DONORS::Field stuck = {-3, 0, 0, 0};
        std::vector<double> sp = {-1.5};
        std::mt19937_64 rng(943101);
        std::normal_distribution<double> normal;
        std::uniform_real_distribution<double> uniform;
        double current = log_target(stuck, sp);
        unsigned visits = 0;
        for (unsigned it = 0; it < 16384; ++it) {
            auto t = stuck;
            auto p = sp;
            t[0] += .12 * normal(rng);
            p[0] += .12 * normal(rng);
            const double next = log_target(t, p);
            if (std::log(std::max(uniform(rng), 1e-300)) < next - current) {
                stuck = t;
                sp = p;
                current = next;
            }
            visits += stuck[0] > 0;
        }
        check(visits < 10, "negative control detects local-only trapping across separated modes");
        std::cout << std::setprecision(17) << "MIXTURE_REFERENCE " << positive << ' ' << mean << ' ' << pmean << ' '
                  << unused << '\n';
        return FML::TESTS::summary("fixed-donor joint jumps");
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
