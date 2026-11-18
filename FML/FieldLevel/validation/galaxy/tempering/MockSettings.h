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
        uint64_t id = 0, truth = 4101, noise = 4102, initial = 4103;
        void validate(unsigned ensembles) const {
            const auto limit = uint64_t(std::numeric_limits<uint32_t>::max());
            if (id < 1 || id > 65535 || ensembles < 1 || ensembles > 4 || truth < 1 || noise < 1 || initial < 1 ||
                truth > limit || noise > limit || initial > limit - ensembles + 1 || truth == noise ||
                (truth >= initial && truth < initial + ensembles) || (noise >= initial && noise < initial + ensembles))
                throw std::invalid_argument("Invalid independent-mock identity/data/initial seeds");
        }
        void read(std::istream & in, unsigned ensembles) {
            std::string tag;
            if (!(in >> tag) || tag != "FML_MOCK_1")
                throw std::invalid_argument("Missing independent-mock metadata");
            for (auto p : {&id, &truth, &noise, &initial}) {
                std::string token;
                size_t used;
                if (!(in >> token))
                    throw std::invalid_argument("Incomplete independent-mock metadata");
                *p = std::stoull(token, &used);
                if (used != token.size())
                    throw std::invalid_argument("Invalid mock seed token");
            }
            validate(ensembles);
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
