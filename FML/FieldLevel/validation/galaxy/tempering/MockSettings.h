#ifndef FML_VALIDATION_MOCK_SETTINGS_H
#define FML_VALIDATION_MOCK_SETTINGS_H
#include <cstdint>
#include <initializer_list>
#include <istream>
#include <limits>
#include <stdexcept>
#include <string>
namespace MOCK {
    struct Seeds {
        uint64_t id = 0, truth = 4101, noise = 4102, initial = 4103, scalar_truth = 0;
        bool prior_predictive = false;
        void validate(unsigned ensembles) const {
            const auto limit = uint64_t(std::numeric_limits<uint32_t>::max());
            if (id < 1 || id > 65535 || ensembles < 1 || ensembles > 4 || truth < 1 || noise < 1 || initial < 1 ||
                truth > limit || noise > limit || initial > limit - ensembles + 1 || truth == noise ||
                (truth >= initial && truth < initial + ensembles) || (noise >= initial && noise < initial + ensembles))
                throw std::invalid_argument("Invalid independent-mock identity/data/initial seeds");
        }
        uint64_t generation_key(unsigned role, unsigned ensemble = 0) const {
            if (role > 3 || ensemble > 3 || (role != 3 && ensemble != 0))
                throw std::invalid_argument("Invalid prior-predictive generation role/ensemble");
            const uint64_t raw = role == 0 ? truth : role == 1 ? scalar_truth : role == 2 ? noise : initial + ensemble;
            if (!prior_predictive) {
                if (role == 1)
                    throw std::invalid_argument("Fixed-truth mode has no scalar generation seed");
                return raw;
            }
            if (id < 3 || id > 65535 || raw < 1 || raw > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument("Invalid prior-predictive generation seed");
            return UINT64_C(0x5400000000000000) | (uint64_t(role) << 48) | (id << 32) | raw;
        }
        void read(std::istream & in, unsigned ensembles) {
            std::string tag;
            if (!(in >> tag) || (tag != "FML_MOCK_1" && tag != "FML_MOCK_2"))
                throw std::invalid_argument("Missing independent-mock metadata");
            prior_predictive = tag == "FML_MOCK_2";
            scalar_truth = 0;
            std::initializer_list<uint64_t *> old_fields = {&id, &truth, &noise, &initial};
            std::initializer_list<uint64_t *> new_fields = {&id, &truth, &noise, &initial, &scalar_truth};
            for (auto p : prior_predictive ? new_fields : old_fields) {
                std::string token;
                size_t used;
                if (!(in >> token))
                    throw std::invalid_argument("Incomplete independent-mock metadata");
                *p = std::stoull(token, &used);
                if (used != token.size())
                    throw std::invalid_argument("Invalid mock seed token");
            }
            validate(ensembles);
            const auto limit = uint64_t(std::numeric_limits<uint32_t>::max());
            if (prior_predictive &&
                (id < 3 || scalar_truth < 1 || scalar_truth > limit || scalar_truth == truth || scalar_truth == noise ||
                 (scalar_truth >= initial && scalar_truth < initial + ensembles)))
                throw std::invalid_argument("Invalid prior-predictive scalar seed or reserved mock ID");
        }
    };
    inline uint64_t stream(uint64_t mock, unsigned ensemble, unsigned replica, unsigned sweep, unsigned purpose) {
        if (mock < 1 || mock > 65535 || ensemble > 3 || replica > 63 || sweep >= 1000000 || purpose > 3)
            throw std::invalid_argument("Mock RNG tuple outside supported bounds");
        return UINT64_C(0x5300000000000000) | (mock << 32) | (uint64_t(ensemble) << 30) | (uint64_t(replica) << 24) |
               (uint64_t(sweep) << 4) | purpose;
    }
} // namespace MOCK
#endif
