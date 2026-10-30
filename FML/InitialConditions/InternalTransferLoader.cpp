#include <FML/InitialConditions/InternalTransferLoader.h>
#ifdef USE_INTERNAL_COSMOLOGY
#include <FML/InitialConditions/InternalTransferProvider.h>
namespace FML::INITIALCONDITIONS {
    std::shared_ptr<TransferProvider> load_internal_transfers(const InternalTransferRequest & request) {
        auto settings = FML::COSMOLOGY::ParameterFile::read(request.filename);
        const auto & p = settings.cosmo;
        auto matches = [](double a, double b) {
            return std::abs(a - b) <= 1e-8 * std::max({std::abs(a), std::abs(b), 1e-30});
        };
        if (!matches(p.h, request.h) || !matches(p.OmegaB, request.Omegab) || !matches(p.OmegaCDM, request.OmegaCDM) ||
            !matches(p.TCMB, request.TCMB) || !matches(p.Neff, request.Neff) || !matches(p.A_s, request.As) ||
            !matches(p.n_s, request.ns) || !matches(p.k_pivot, request.kpivot_mpc))
            throw std::invalid_argument(
                "[internal_transfer] Solver and SimulationCode cosmological/primordial parameters differ");
        const double kmin = settings.pert.k_min * FML::COSMOLOGY::Constants::Mpc / p.h;
        const double kmax = settings.pert.k_max * FML::COSMOLOGY::Constants::Mpc / p.h;
        if (request.require_sigma8_coverage && (kmin > 1e-5 || kmax < 100.0))
            throw std::out_of_range("[internal_transfer] sigma8 normalization requires transfer coverage from 1e-5 to "
                                    "100 h/Mpc; use As normalization or supply wider-range inputs");
        if (kmin > request.required.kmin_hmpc || kmax < request.required.kmax_hmpc ||
            std::exp(settings.pert.x_start) > request.required.amin ||
            std::exp(settings.pert.x_end) < request.required.amax)
            throw std::out_of_range(
                "[internal_transfer] Solver k/a range does not cover the IC mesh, input epoch and today");
        return std::make_shared<InternalTransferProvider>(std::move(settings), Gauge::SynchronousCDM);
    }
} // namespace FML::INITIALCONDITIONS

#endif
