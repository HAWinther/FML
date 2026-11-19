#ifndef FML_VALIDATION_SUPPORT_RESCALING_NATIVE_H
#define FML_VALIDATION_SUPPORT_RESCALING_NATIVE_H
#include "Support.h"
#include <FML/FieldLevel/EulerianCheckpoint.h>
namespace SUPPORT_DRIVER {
    inline bool legacy = false;
    inline std::string policy_identity;
    inline void load(const std::filesystem::path & path) {
        std::ifstream in(path);
        std::string tag, kind, extra;
        if (!(in >> tag >> kind) || tag != "FML_SUPPORT_RESCALING_1" || (kind != "active" && kind != "legacy") ||
            (in >> extra))
            throw std::invalid_argument("Missing/invalid support-rescaling policy");
        legacy = kind == "legacy";
        policy_identity = "FML exact initial support 1:" + kind;
    }
    template <class Model>
    std::string identity(const ACTIVE_SUPPORT::InitialSupport<Model> & support) {
        if (policy_identity.empty())
            throw std::invalid_argument("Missing support policy");
        return policy_identity + ":global_real_rank=" + std::to_string(support.rank());
    }
    template <class Model>
    auto propose(ACTIVE_SUPPORT::InitialSupport<Model> & support,
                 const FML::FIELDLEVEL::Field & w,
                 const std::vector<double> & p,
                 size_t index,
                 double alpha) {
        return legacy ? RESCALING::propose(w, p, index, alpha) : ACTIVE_SUPPORT::propose(support, w, p, index, alpha);
    }
} // namespace SUPPORT_DRIVER
#endif
