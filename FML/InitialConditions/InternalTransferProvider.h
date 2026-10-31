#ifndef FML_INTERNAL_TRANSFER_PROVIDER_H
#define FML_INTERNAL_TRANSFER_PROVIDER_H
#include <FML/InitialConditions/TransferProvider.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <algorithm>
#include <memory>
#include <vector>
namespace FML::INITIALCONDITIONS {
    /// Owning solver adapter. The converted density uses the CDM rest frame, not each
    /// species' own comoving frame. Velocity transfers remain explicitly Newtonian.
    /// Flat LCDM with at most one massive neutrino species (the solver's m_nu): MassiveNeutrino is then available and
    /// TotalMatter includes it.
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
        bool massive_nu() const { return state->settings.cosmo.m_nu > 0.0; }
        // delta of one species in the requested gauge. Newtonian -> CDM-synchronous is the time shift that brings CDM
        // to rest: delta_s += 3 (1 + w_s) Hp theta_cdm / (c k)^2 = -3 (1 + w_s) Hp v_cdm / (c k)
        double density_one(Species species, double x, double k_si) const {
            const auto & pert = *state->perturbations;
            double delta = species == Species::CDM    ? pert.get_delta_cdm(x, k_si) :
                           species == Species::Baryon ? pert.get_delta_b(x, k_si) :
                                                        pert.get_delta_ncdm(x, k_si);
            if (gauge == Gauge::SynchronousCDM)
                delta -= 3.0 * (1.0 + w_of(species, x)) * state->background.Hp_of_x(x) /
                         (FML::COSMOLOGY::Constants::c * k_si) * pert.get_v_cdm(x, k_si);
            return delta;
        }
        double w_of(Species species, double x) const {
            if (species != Species::MassiveNeutrino)
                return 0.0;
            return state->background.pNcdm(x) / state->background.rhoNcdm(x);
        }
        /// rho_s a^3 in units of rho_c0 (the weights of the combinations; Omega for CDM and baryons)
        double rho_a3_of(Species species, double x) const {
            const auto & p = state->settings.cosmo;
            return species == Species::CDM    ? p.OmegaCDM :
                   species == Species::Baryon ? p.OmegaB :
                                                state->background.rhoNcdm(x) * std::exp(3.0 * x);
        }
        /// The species a combination is made of
        std::vector<Species> components(Species s) const {
            if (s == Species::CDMBaryon)
                return {Species::CDM, Species::Baryon};
            if (s == Species::TotalMatter) {
                if (massive_nu())
                    return {Species::CDM, Species::Baryon, Species::MassiveNeutrino};
                return {Species::CDM, Species::Baryon};
            }
            return {s};
        }
        static double k_si_of(double k_hmpc, double h) { return k_hmpc * h / FML::COSMOLOGY::Constants::Mpc; }

      public:
        explicit InternalTransferProvider(FML::COSMOLOGY::RunSettings settings, Gauge gauge = Gauge::SynchronousCDM)
            : gauge(gauge) {
            const auto & p = settings.cosmo;
            // alpha_s: primordial_power() and SimulationCode's primordial spectrum have no running
            if (p.model != "lcdm" || p.OmegaK != 0.0 || p.alpha_s != 0.0 ||
                settings.pert.ic_adiabatic != 1.0 ||
                std::any_of(settings.pert.iso_amplitude.begin(), settings.pert.iso_amplitude.end(), [](double v) {
                    return v != 0;
                }))
                throw std::invalid_argument("[InternalTransferProvider] Requires flat LCDM (with at most one massive "
                                            "neutrino species), no running alpha_s and a unit adiabatic mode");
            if (!(p.h > 0) || !std::isfinite(p.h) || !(p.A_s > 0) || !std::isfinite(p.A_s) || !(p.k_pivot > 0) ||
                !std::isfinite(p.k_pivot) || !std::isfinite(p.n_s) || !(p.OmegaB > 0) || !(p.OmegaCDM > 0) || !(p.m_nu >= 0) ||
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
            return s == Species::CDM || s == Species::Baryon || s == Species::CDMBaryon || s == Species::TotalMatter ||
                   (s == Species::MassiveNeutrino && massive_nu());
        }
        bool has_velocity(Species s) const override { return has_density(s); }
        /// Combinations are weighted by rho_i(a): CDM + baryons, and with massive neutrinos also them in TotalMatter
        double density_transfer(Species s, double k, double a) const override {
            check_query(s, k, a);
            const double x = std::log(a), k_si = k_si_of(k, h());
            double sum = 0.0, weight = 0.0;
            for (auto c : components(s)) {
                const double rho = rho_a3_of(c, x);
                sum += rho * density_one(c, x, k_si);
                weight += rho;
            }
            return sum / weight;
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
        /// Newtonian theta/(aH) = -c k v / Hp; combinations are weighted by rho_i + p_i
        double velocity_divergence_transfer(Species s, double k, double a) const override {
            check_query(s, k, a, true);
            const double x = std::log(a), k_si = k_si_of(k, h());
            const auto & pert = *state->perturbations;
            double sum = 0.0, weight = 0.0;
            for (auto c : components(s)) {
                const double v = c == Species::CDM    ? pert.get_v_cdm(x, k_si) :
                                 c == Species::Baryon ? pert.get_v_b(x, k_si) :
                                                        pert.get_v_ncdm(x, k_si);
                const double rho_plus_p = rho_a3_of(c, x) * (1.0 + w_of(c, x));
                sum += rho_plus_p * v;
                weight += rho_plus_p;
            }
            return -FML::COSMOLOGY::Constants::c * k_si / state->background.Hp_of_x(x) * sum / weight;
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
