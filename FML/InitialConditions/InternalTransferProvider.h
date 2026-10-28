#ifndef FML_INTERNAL_TRANSFER_PROVIDER_H
#define FML_INTERNAL_TRANSFER_PROVIDER_H
#include <FML/InitialConditions/TransferProvider.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <algorithm>
#include <memory>
namespace FML::INITIALCONDITIONS {
    /// Owning solver adapter. The converted density uses the CDM rest frame, not each
    /// species' own comoving frame. Velocity transfers remain explicitly Newtonian.
    class InternalTransferProvider final : public TransferProvider {
        struct State {
            FML::COSMOLOGY::RunSettings settings;
            FML::COSMOLOGY::BackgroundCosmology background;
            std::unique_ptr<FML::COSMOLOGY::RecombinationHistory> recombination;
            std::unique_ptr<FML::COSMOLOGY::Perturbations> perturbations;
            explicit State(FML::COSMOLOGY::RunSettings options)
                : settings(std::move(options)), background(settings.cosmo) {
                background.solve(settings.bg_x_start, settings.bg_x_end, settings.bg_n_points);
                recombination = std::make_unique<FML::COSMOLOGY::RecombinationHistory>(background, settings.rec);
                recombination->solve();
                perturbations =
                    std::make_unique<FML::COSMOLOGY::Perturbations>(background, *recombination, settings.pert);
                perturbations->solve();
            }
        };
        std::shared_ptr<const State> state;
        Gauge gauge;
        double primordial_scale{1.0};
        double density_one(Species species, double k_hmpc, double a) const {
            const double x = std::log(a), k_si = k_hmpc * h() / FML::COSMOLOGY::Constants::Mpc;
            const auto & pert = *state->perturbations;
            double delta = species == Species::CDM ? pert.get_delta_cdm(x, k_si) : pert.get_delta_b(x, k_si);
            if (gauge == Gauge::SynchronousCDM)
                delta -= 3.0 * state->background.Hp_of_x(x) / (FML::COSMOLOGY::Constants::c * k_si) *
                         pert.get_v_cdm(x, k_si);
            return delta;
        }

      public:
        explicit InternalTransferProvider(FML::COSMOLOGY::RunSettings settings, Gauge gauge = Gauge::SynchronousCDM)
            : gauge(gauge) {
            const auto & p = settings.cosmo;
            if (p.model != "lcdm" || p.OmegaK != 0.0 || p.m_nu != 0.0 || settings.pert.ic_adiabatic != 1.0 ||
                std::any_of(settings.pert.iso_amplitude.begin(), settings.pert.iso_amplitude.end(), [](double v) {
                    return v != 0;
                }))
                throw std::invalid_argument("[InternalTransferProvider] Initial release requires flat LCDM, no massive "
                                            "neutrinos and a unit adiabatic mode");
            if (!(p.h > 0) || !std::isfinite(p.h) || !(p.A_s > 0) || !std::isfinite(p.A_s) || !(p.k_pivot > 0) ||
                !std::isfinite(p.k_pivot) || !std::isfinite(p.n_s) || !(p.OmegaB > 0) || !(p.OmegaCDM > 0) ||
                (gauge != Gauge::Newtonian && gauge != Gauge::SynchronousCDM))
                throw std::invalid_argument("[InternalTransferProvider] Invalid species, h or gauge");
            state = std::make_shared<State>(std::move(settings));
        }
        /// Share the solved state; changing the density view never changes velocity gauge.
        InternalTransferProvider with_density_gauge(Gauge requested) const {
            if (requested != Gauge::Newtonian && requested != Gauge::SynchronousCDM)
                throw std::invalid_argument("[InternalTransferProvider] Unsupported density gauge");
            auto result = *this;
            result.gauge = requested;
            return result;
        }
        double h() const override { return state->settings.cosmo.h; }
        TransferRange range() const override {
            const auto & p = *state->perturbations;
            return {p.get_k_array().front() * FML::COSMOLOGY::Constants::Mpc / h(),
                    p.get_k_array().back() * FML::COSMOLOGY::Constants::Mpc / h(),
                    std::exp(p.get_x_array().front()),
                    std::exp(p.get_x_array().back())};
        }
        Gauge density_gauge() const override { return gauge; }
        bool has_density(Species s) const override {
            return s == Species::CDM || s == Species::Baryon || s == Species::CDMBaryon || s == Species::TotalMatter;
        }
        bool has_velocity(Species s) const override { return has_density(s); }
        double density_transfer(Species s, double k, double a) const override {
            check_query(s, k, a);
            if (s == Species::CDM || s == Species::Baryon)
                return density_one(s, k, a);
            const auto & p = state->settings.cosmo;
            return (p.OmegaCDM * density_one(Species::CDM, k, a) + p.OmegaB * density_one(Species::Baryon, k, a)) /
                   (p.OmegaCDM + p.OmegaB);
        }
        double density_growth_rate(Species s, double k, double a) const override {
            check_query(s, k, a);
            const double x = std::log(a), lo = std::max(x - 1e-3, std::log(range().amin)),
                         hi = std::min(x + 1e-3, std::log(range().amax));
            const double value = density_transfer(s, k, a);
            if (value == 0 || !(hi > lo))
                throw std::domain_error("[InternalTransferProvider] Undefined density logarithmic derivative");
            return (density_transfer(s, k, std::exp(hi)) - density_transfer(s, k, std::exp(lo))) / ((hi - lo) * value);
        }
        double velocity_divergence_transfer(Species s, double k, double a) const override {
            check_query(s, k, a, true);
            const double x = std::log(a), k_si = k * h() / FML::COSMOLOGY::Constants::Mpc;
            const auto & pert = *state->perturbations;
            const auto & p = state->settings.cosmo;
            const double velocity =
                s == Species::CDM ?
                    pert.get_v_cdm(x, k_si) :
                s == Species::Baryon ?
                    pert.get_v_b(x, k_si) :
                    (p.OmegaCDM * pert.get_v_cdm(x, k_si) + p.OmegaB * pert.get_v_b(x, k_si)) / (p.OmegaCDM + p.OmegaB);
            return -FML::COSMOLOGY::Constants::c * k_si / state->background.Hp_of_x(x) * velocity;
        }
        void rescale_primordial_power(double factor) override {
            if (!(factor > 0) || !std::isfinite(factor))
                throw std::invalid_argument("[InternalTransferProvider] Invalid primordial power rescaling");
            primordial_scale *= factor;
        }
        double primordial_power(double k) const override {
            if (!(k > 0) || !std::isfinite(k))
                throw std::invalid_argument("[InternalTransferProvider] Invalid k_hmpc");
            const auto & p = state->settings.cosmo;
            return primordial_scale * 2 * M_PI * M_PI / (k * k * k) * p.A_s * std::pow(k * h() / p.k_pivot, p.n_s - 1);
        }
    };
} // namespace FML::INITIALCONDITIONS
#endif
