#ifndef FML_INTERNAL_TRANSFER_LOADER_H
#define FML_INTERNAL_TRANSFER_LOADER_H
#include <FML/InitialConditions/TransferProvider.h>
#include <memory>
namespace FML::INITIALCONDITIONS {
    struct InternalTransferRequest {
        std::string filename;
        double h, Omegab, OmegaCDM, TCMB, Neff, As, ns, kpivot_mpc;
        TransferRange required;
        bool require_sigma8_coverage{false};
    };
#ifdef USE_INTERNAL_COSMOLOGY
    std::shared_ptr<TransferProvider> load_internal_transfers(const InternalTransferRequest & request);
#else
    inline std::shared_ptr<TransferProvider> load_internal_transfers(const InternalTransferRequest &) {
        throw std::runtime_error("internal_transfer needs a COLASolver build with USE_INTERNAL_COSMOLOGY=true");
    }
#endif
} // namespace FML::INITIALCONDITIONS
#endif
