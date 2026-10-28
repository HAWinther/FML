#ifndef FML_TRANSFER_PROVIDER_H
#define FML_TRANSFER_PROVIDER_H
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
namespace FML::INITIALCONDITIONS {
    enum class Species { CDM, Baryon, CDMBaryon, MassiveNeutrino, TotalMatter };
    enum class Gauge { Newtonian, SynchronousCDM };
    enum class RangePolicy { Reject, SourceExtrapolation };
    struct TransferRange {
        double kmin_hmpc, kmax_hmpc, amin, amax;
        bool contains(double k_hmpc, double a) const {
            const double eps = 16 * std::numeric_limits<double>::epsilon();
            return k_hmpc >= kmin_hmpc * (1 - eps) && k_hmpc <= kmax_hmpc * (1 + eps) && a >= amin * (1 - eps) &&
                   a <= amax * (1 + eps);
        }
    };
    /// All transfers refer to unit adiabatic primordial curvature in the CAMB sign convention.
    /// k is in h/Mpc; P_R and P_s are dimensional spectra in (Mpc/h)^3.
    /// Density is delta_s/R. Velocity is theta_s/(aH R), theta = div(u), u = a dx/dt.
    /// Density derivatives are NOT velocity transfers; the metric terms depend on gauge.
    class TransferProvider {
      public:
        virtual ~TransferProvider() = default;
        virtual double h() const = 0;
        virtual TransferRange range() const = 0;
        virtual Gauge density_gauge() const = 0;
        virtual Gauge velocity_gauge() const { return Gauge::Newtonian; }
        virtual bool has_density(Species species) const = 0;
        virtual bool has_velocity(Species) const { return false; }
        virtual RangePolicy range_policy() const { return RangePolicy::Reject; }
        virtual double density_transfer(Species species, double k_hmpc, double a) const = 0;
        /// Compatibility representation T_delta/k_Mpc^2, in Mpc^2. CAMB adapter avoids a round trip.
        virtual double density_over_k_mpc2(Species species, double k_hmpc, double a) const {
            const double k_mpc = k_hmpc * h();
            return density_transfer(species, k_hmpc, a) / (k_mpc * k_mpc);
        }
        virtual double density_growth_rate(Species, double, double) const {
            throw std::runtime_error("[TransferProvider] Density growth rate is unavailable");
        }
        virtual double velocity_divergence_transfer(Species, double, double) const {
            throw std::runtime_error("[TransferProvider] Direct velocity transfer is unavailable");
        }
        virtual double primordial_power(double k_hmpc) const = 0;
        virtual void rescale_primordial_power(double) {
            throw std::runtime_error("[TransferProvider] This source owns its primordial normalization externally");
        }
        virtual double power(Species species, double k_hmpc, double a) const {
            const double transfer = density_transfer(species, k_hmpc, a);
            return transfer * transfer * primordial_power(k_hmpc);
        }
        void check_query(Species species, double k_hmpc, double a, bool velocity = false) const {
            if (!(k_hmpc > 0) || !(a > 0) || !std::isfinite(k_hmpc) || !std::isfinite(a))
                throw std::invalid_argument("[TransferProvider] k_hmpc and a must be positive and finite");
            if (!(velocity ? has_velocity(species) : has_density(species)))
                throw std::invalid_argument("[TransferProvider] Requested species/quantity is unavailable");
            if (range_policy() == RangePolicy::Reject && !range().contains(k_hmpc, a))
                throw std::out_of_range("[TransferProvider] Query outside the supported k/a range");
        }
        void require_ic_density_gauge() const {
            if (density_gauge() != Gauge::SynchronousCDM)
                throw std::invalid_argument(
                    "[TransferProvider] Legacy particle ICs require CDM-synchronous density transfers");
        }
    };
} // namespace FML::INITIALCONDITIONS
#endif
