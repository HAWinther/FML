#ifndef FML_VALIDATION_REPLICA_EXCHANGE_H
#define FML_VALIDATION_REPLICA_EXCHANGE_H
// Replica exchange method: Earl & Deem (2005), https://arxiv.org/abs/physics/0508111.
// Native likelihood-only tempering prototype; beta=0 prior refresh and bounded validation are described
// in this directory's README.md. This header is not a production distributed-replica sampler.
#include <FML/FieldLevel/Framework.h>
#include <random>
namespace EXCHANGE {
    struct PhysicalState {
        FML::FIELDLEVEL::Field white;
        std::vector<double> parameters;
        double quadratic = 0;
        size_t label = 0;
    };
    struct Attempt {
        size_t edge;
        bool accepted;
        double log_ratio;
    };
    inline void ladder(const std::vector<double> & beta) {
        if (beta.size() < 2 || beta.front() != 1 || beta.back() != 0)
            throw std::invalid_argument("Replica ladder needs cold=1 and prior=0");
        for (size_t j = 0; j < beta.size(); ++j)
            if (!std::isfinite(beta[j]) || beta[j] < 0 || beta[j] > 1 || (j && !(beta[j] < beta[j - 1])))
                throw std::invalid_argument("Invalid replica ladder");
    }
    inline void layout(const std::vector<double> & beta, unsigned passes) {
        ladder(beta);
        if (beta.size() > 64 || passes < 1 || passes > 8)
            throw std::invalid_argument("Replica layout exceeds supported count/pass bounds");
    }
    inline uint64_t expected_attempts(size_t edge, uint64_t measured, unsigned passes) {
        if (passes < 1 || passes > 8)
            throw std::invalid_argument("Invalid exchange pass count");
        return (passes / 2) * measured + (passes % 2) * (edge % 2 ? measured / 2 : (measured + 1) / 2);
    }
    inline void prior_refresh_visit(size_t label, std::vector<unsigned> & phase) {
        if (label >= phase.size() || phase[label] > 2)
            throw std::invalid_argument("Invalid prior-refresh transport label");
        if (phase[label] == 1)
            phase[label] = 2;
    }
    inline void labels(const std::vector<PhysicalState> & states) {
        if (states.size() < 2)
            throw std::invalid_argument("Replica ensemble needs at least two states");
        std::vector<bool> seen(states.size(), false);
        for (const auto & s : states) {
            if (s.label >= states.size() || seen[s.label])
                throw std::invalid_argument("Replica labels must be a permutation");
            seen[s.label] = true;
        }
    }
    inline double log_ratio(double bi, double bj, double li, double lj) {
        if (!std::isfinite(bi) || !std::isfinite(bj) || bi < 0 || bi > 1 || bj < 0 || bj > 1 || !std::isfinite(li) ||
            !std::isfinite(lj) || li < 0 || lj < 0)
            throw std::invalid_argument("Invalid replica exchange controls");
        const double result = (bi - bj) * (li - lj);
        if (!std::isfinite(result))
            throw std::invalid_argument("Unrepresentable replica log ratio");
        return result;
    }
    inline std::vector<Attempt> neighbors(std::vector<PhysicalState> & states,
                                          const std::vector<double> & beta,
                                          unsigned parity,
                                          std::mt19937_64 & rng) {
        ladder(beta);
        labels(states);
        if (states.size() != beta.size() || parity > 1)
            throw std::invalid_argument("Replica schedule/layout mismatch");
        for (const auto & s : states)
            if (!std::isfinite(s.quadratic) || s.quadratic < 0)
                throw std::invalid_argument("Invalid replica likelihood");
        std::uniform_real_distribution<double> uniform;
        std::vector<Attempt> result;
        for (size_t j = parity; j + 1 < states.size(); j += 2) {
            double u = 0;
            while (u == 0)
                u = uniform(rng);
            const double loga = log_ratio(beta[j], beta[j + 1], states[j].quadratic, states[j + 1].quadratic);
            const bool accepted = std::log(u) < loga;
            if (accepted)
                std::swap(states[j], states[j + 1]);
            result.push_back({j, accepted, loga});
        }
        return result;
    }
    inline void observe(const std::vector<PhysicalState> & states,
                        std::vector<unsigned> & phase,
                        std::vector<uint64_t> & trips,
                        bool mark_hot_visit = true) {
        labels(states);
        if (phase.size() != states.size() || trips.size() != states.size())
            throw std::invalid_argument("Replica transport layout mismatch");
        for (unsigned p : phase)
            if (p > 2)
                throw std::invalid_argument("Invalid replica transport phase");
        const size_t cold = states.front().label, hot = states.back().label;
        if (phase[cold] == 2)
            ++trips[cold];
        phase[cold] = 1;
        if (mark_hot_visit && phase[hot] == 1)
            phase[hot] = 2;
    }
} // namespace EXCHANGE
#endif
