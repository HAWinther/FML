#ifndef FML_VALIDATION_FIXED_DONOR_NATIVE_H
#define FML_VALIDATION_FIXED_DONOR_NATIVE_H
#include "Donors.h"
#include <FML/FieldLevel/validation/galaxy/support_interweave/Native.h>
#include <FML/FieldLevel/EulerianCheckpoint.h>
#include <memory>
namespace DONOR_DRIVER {
    inline bool enabled = false;
    inline std::vector<DONORS::State> raw;
    inline std::unique_ptr<DONORS::Pool> pool;
    inline std::string identifier;
    inline void load(const std::filesystem::path & path) {
        namespace CP = FML::FIELDLEVEL::FRAMEWORK::CHECKPOINT;
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input)
            throw std::invalid_argument("Missing donor pool");
        const auto end = input.tellg();
        if (end < 0 || std::uintmax_t(end) > 131072)
            throw std::invalid_argument("Oversized donor pool");
        CP::Codec::Bytes bytes(size_t(end), 0);
        input.seekg(0);
        input.read(reinterpret_cast<char *>(bytes.data()), std::streamsize(bytes.size()));
        if (!input)
            throw std::invalid_argument("Cannot read donor pool");
        std::istringstream in(std::string(bytes.begin(), bytes.end()));
        std::string tag, extra;
        unsigned flag, dim, np, count;
        if (!(in >> tag >> flag >> dim >> np >> count) || tag != "FML_FIXED_DONORS_1" || flag > 1 || dim != 64 ||
            np != 5 || count > 64 || (flag && count < 2) || (!flag && count))
            throw std::invalid_argument("Invalid bounded donor header");
        enabled = flag;
        raw.clear();
        for (unsigned i = 0; i < count; ++i) {
            DONORS::State s;
            s.white.resize(dim);
            s.parameters.resize(np);
            if (!(in >> s.ensemble >> s.draw) || s.ensemble > 3 || s.draw > 511)
                throw std::invalid_argument("Invalid donor identity");
            for (auto * v : {&s.white, &s.parameters})
                for (double & x : *v)
                    if (!(in >> x) || !std::isfinite(x))
                        throw std::invalid_argument("Invalid donor state");
            raw.push_back(std::move(s));
        }
        if (in >> extra)
            throw std::invalid_argument("Extra donor fields");
        identifier = "FML immutable donor translations 1:" + std::to_string(CP::Codec::checksum(bytes));
    }
    template <class Model>
    void prepare(ACTIVE_SUPPORT::InitialSupport<Model> & support) {
        if (enabled) {
            auto projected = raw;
            for (auto & s : projected)
                s.white = support.project(s.white);
            pool = std::make_unique<DONORS::Pool>(std::move(projected));
        }
    }
    inline const std::string & identity() {
        if (identifier.empty())
            throw std::invalid_argument("Missing donor identity");
        return identifier;
    }
    template <class Stats>
    bool valid_counters(const Stats & s, uint64_t total, uint64_t measured, bool positive) {
        if (!enabled || !positive)
            return s.donor_attempts == 0 && s.donor_accepted == 0 && s.donor_support_rejections == 0 &&
                   s.donor_values_total == 0 && s.donor_values_sampling == 0 && s.donor_large_attempts == 0 &&
                   s.donor_large_accepted == 0;
        return s.donor_attempts == measured && s.donor_accepted <= measured &&
               s.donor_support_rejections <= measured - s.donor_accepted &&
               s.donor_values_sampling + s.donor_support_rejections == measured &&
               s.donor_values_total >= s.donor_values_sampling && s.donor_values_total <= total &&
               s.donor_large_attempts <= measured && s.donor_large_accepted <= s.donor_large_attempts &&
               s.donor_large_accepted <= s.donor_accepted &&
               s.donor_accepted - s.donor_large_accepted <= measured - s.donor_large_attempts;
    }
    template <class Support, class Target, class State, class Stats, class Work>
    void step(Support &,
              const Target & target,
              State & s,
              Stats & stats,
              double & current,
              Work & work,
              uint64_t seed,
              bool measured) {
        if (!enabled)
            return;
        std::mt19937_64 rng(seed + 4);
        std::uniform_real_distribution<double> uniform;
        const auto proposal = pool->draw(s.white, s.parameters, rng);
        if (measured) {
            ++stats.donor_attempts;
            stats.donor_large_attempts += proposal.scale == 1.;
        }
        if (!target.model().in_support(proposal.parameters)) {
            if (measured)
                ++stats.donor_support_rejections;
            return;
        }
        ++stats.donor_values_total;
        if (measured)
            ++stats.donor_values_sampling;
        double next = HUGE_VAL;
        try {
            next = target.value(proposal.white, proposal.parameters, work);
        } catch (const FML::FIELDLEVEL::FRAMEWORK::UnrepresentableState &) {
        }
        double u = 0;
        while (u == 0)
            u = uniform(rng);
        if (std::isfinite(next) && std::log(u) < current - next) {
            s.white = proposal.white;
            s.parameters = proposal.parameters;
            current = next;
            if (measured) {
                ++stats.donor_accepted;
                stats.donor_large_accepted += proposal.scale == 1.;
            }
        }
    }
} // namespace DONOR_DRIVER
#endif
