#ifndef FML_CAMB_TRANSFER_PROVIDER_H
#define FML_CAMB_TRANSFER_PROVIDER_H
#include <FML/CAMBUtils/CAMBReader.h>
#include <FML/InitialConditions/TransferProvider.h>
#include <memory>
namespace FML::INITIALCONDITIONS {
    class CAMBTransferProvider final : public TransferProvider {
        std::shared_ptr<const FML::FILEUTILS::LinearTransferData> data;
        RangePolicy policy;

      public:
        explicit CAMBTransferProvider(std::shared_ptr<const FML::FILEUTILS::LinearTransferData> data,
                                      RangePolicy policy = RangePolicy::SourceExtrapolation)
            : data(std::move(data)), policy(policy) {
            if (!this->data || !bool(*this->data) || !(this->data->get_h() > 0) || !std::isfinite(this->data->get_h()))
                throw std::invalid_argument("[CAMBTransferProvider] Initialized transfer data are required");
        }
        double h() const override { return data->get_h(); }
        TransferRange range() const override {
            return {data->get_kmin_hmpc_splines(),
                    data->get_kmax_hmpc_splines(),
                    1.0 / (1.0 + data->get_zmax_splines()),
                    1.0 / (1.0 + data->get_zmin_splines())};
        }
        Gauge density_gauge() const override { return Gauge::SynchronousCDM; }
        RangePolicy range_policy() const override { return policy; }
        bool has_density(Species species) const override {
            switch (species) {
                case Species::CDM:
                    return data->transfer_col_cdm >= 0;
                case Species::Baryon:
                    return data->transfer_col_baryon >= 0;
                case Species::CDMBaryon:
                    return data->transfer_col_cdm >= 0 && data->transfer_col_baryon >= 0;
                case Species::MassiveNeutrino:
                    return data->transfer_col_mnu >= 0;
                case Species::TotalMatter:
                    return data->transfer_col_cdm >= 0 && data->transfer_col_baryon >= 0 && data->transfer_col_mnu >= 0;
            }
            return false;
        }
        bool has_velocity(Species species) const override {
            return species == Species::CDM    ? data->transfer_col_vcdm >= 0 :
                   species == Species::Baryon ? data->transfer_col_vb >= 0 :
                                                species == Species::CDMBaryon && data->transfer_col_vcdm >= 0 &&
                                                    data->transfer_col_vb >= 0;
        }
        double density_over_k_mpc2(Species species, double k, double a) const override {
            check_query(species, k, a);
            switch (species) {
                case Species::CDM:
                    return data->get_cdm_transfer_function(k, a);
                case Species::Baryon:
                    return data->get_baryon_transfer_function(k, a);
                case Species::CDMBaryon:
                    return data->get_cdm_baryon_transfer_function(k, a);
                case Species::MassiveNeutrino:
                    return data->get_massive_neutrino_transfer_function(k, a);
                case Species::TotalMatter:
                    return data->get_total_transfer_function(k, a);
            }
            throw std::invalid_argument("[CAMBTransferProvider] Unknown species");
        }
        double density_transfer(Species species, double k, double a) const override {
            return density_over_k_mpc2(species, k, a) * (k * h()) * (k * h());
        }
        double density_growth_rate(Species species, double k, double a) const override {
            check_query(species, k, a);
            switch (species) {
                case Species::CDM:
                    return data->get_cdm_growth_rate(k, a);
                case Species::Baryon:
                    return data->get_baryon_growth_rate(k, a);
                case Species::CDMBaryon:
                    return data->get_cdm_baryon_growth_rate(k, a);
                case Species::MassiveNeutrino:
                    return data->get_massive_neutrino_growth_rate(k, a);
                case Species::TotalMatter:
                    return data->get_total_growth_rate(k, a);
            }
            throw std::invalid_argument("[CAMBTransferProvider] Unknown species");
        }
        double velocity_divergence_transfer(Species species, double k, double a) const override {
            check_query(species, k, a, true);
            const int column_species = species == Species::CDM ? 0 : species == Species::Baryon ? 1 : 2;
            return data->get_newtonian_velocity_divergence_transfer(column_species, k, a);
        }
        double primordial_power(double k) const override {
            if (!(k > 0) || !std::isfinite(k))
                throw std::invalid_argument("[TransferProvider] Invalid k_hmpc");
            return data->get_primordial_power_spectrum(k);
        }
        double power(Species species, double k, double a) const override {
            check_query(species, k, a);
            // Keep the legacy multiplication order, including cancellations in the neutrino weighting.
            if (species == Species::CDMBaryon)
                return data->get_cdm_baryon_power_spectrum(k, a);
            if (species == Species::MassiveNeutrino)
                return data->get_massive_neutrino_power_spectrum(k, a);
            if (species == Species::TotalMatter)
                return data->get_total_power_spectrum(k, a);
            return TransferProvider::power(species, k, a);
        }
    };
} // namespace FML::INITIALCONDITIONS
#endif
