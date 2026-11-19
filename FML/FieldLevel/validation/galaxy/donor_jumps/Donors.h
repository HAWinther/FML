#ifndef FML_VALIDATION_FIXED_DONOR_JUMPS_H
#define FML_VALIDATION_FIXED_DONOR_JUMPS_H
#include <FML/FieldLevel/Framework.h>
namespace DONORS {
    using Field = FML::FIELDLEVEL::Field;
    struct State {
        Field white;
        std::vector<double> parameters;
        unsigned ensemble = 0, draw = 0;
    };
    struct Proposal {
        Field white;
        std::vector<double> parameters;
        size_t first, second;
        double scale;
    };
    // Immutable donor differences, uniformly ordered pairs. Reverse (j,i) has the
    // same probability as (i,j); scales are chosen independently of the current state.
    // This is a translation in STORED WHITE and physical scalar coordinates: Jacobian 1.
    class Pool {
        std::vector<State> states_;

      public:
        explicit Pool(std::vector<State> states) : states_(std::move(states)) {
            if (states_.size() < 2 || states_.size() > 64 || states_[0].white.empty() || states_[0].parameters.empty())
                throw std::invalid_argument("Invalid bounded donor pool");
            for (const auto & s : states_) {
                if (s.white.size() != states_[0].white.size() || s.parameters.size() != states_[0].parameters.size())
                    throw std::invalid_argument("Donor dimensions differ");
                for (const auto * v : {&s.white, &s.parameters})
                    for (double x : *v)
                        if (!std::isfinite(x))
                            throw std::invalid_argument("Nonfinite donor");
            }
        }
        size_t size() const { return states_.size(); }
        const State & state(size_t i) const { return states_.at(i); }
        Proposal translate(const Field & w, const std::vector<double> & p, size_t i, size_t j, double scale) const {
            if (w.size() != states_[0].white.size() || p.size() != states_[0].parameters.size() || i >= size() ||
                j >= size() || i == j || !std::isfinite(scale) || !(scale > 0))
                throw std::invalid_argument("Invalid donor translation");
            Proposal r{w, p, i, j, scale};
            for (size_t k = 0; k < w.size(); ++k) {
                r.white[k] += scale * (states_[i].white[k] - states_[j].white[k]);
                if (!std::isfinite(r.white[k]))
                    throw std::invalid_argument("Nonfinite donor proposal");
            }
            for (size_t k = 0; k < p.size(); ++k) {
                r.parameters[k] += scale * (states_[i].parameters[k] - states_[j].parameters[k]);
                if (!std::isfinite(r.parameters[k]))
                    throw std::invalid_argument("Nonfinite donor scalar");
            }
            return r;
        }
        template <class RNG>
        Proposal draw(const Field & w, const std::vector<double> & p, RNG & rng) const {
            std::uniform_int_distribution<size_t> first(0, size() - 1), second(0, size() - 2);
            const auto i = first(rng);
            auto j = second(rng);
            if (j >= i)
                ++j;
            std::uniform_int_distribution<int> choose(0, 1);
            return translate(w, p, i, j, choose(rng) ? 1. : .35);
        }
    };
} // namespace DONORS
#endif
