#ifndef FML_GAUGE_VIEW_H
#define FML_GAUGE_VIEW_H

#include <FML/InitialConditions/TransferProvider.h>
#include <FML/Units/Units.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>

//=====================================================================================================================
// A TransferProvider seen in another density gauge. Supported: Newtonian -> CDM-synchronous (the gauge the particle
// initial conditions need). The time shift that brings the CDM to rest (Ma & Bertschinger 1995) gives for each species
//
//   delta_s^sync = delta_s^N + 3 (1 + w_s) (calH / c k)^2 V_cdm^N,   V = theta / calH,   calH = a H
//
// with calH / (c k) = a E(a) / (c/(100 km/s) k_hmpc): the source must provide the Newtonian CDM velocity. Velocities
// are passed through unchanged (they stay in the velocity gauge of the source), as in InternalTransferProvider.
// Synchronous -> Newtonian is not supported: it needs the metric (h' + 6 eta') / 2 k^2, not only velocities.
//
// Accuracy: on scales k <~ aH/c the converted density is a small difference of two large terms, and the growth rate
// is a numerical derivative of that difference between the source's interpolated values. With a coarse redshift
// sampling in the source this is inaccurate there; a provider that converts at its own tabulated redshifts (as
// CLASSTransferProvider does) is better. Use a GaugeView for sources that cannot do that.
//=====================================================================================================================

namespace FML::INITIALCONDITIONS {

    /// The background functions a GaugeView needs
    struct GaugeBackground {
        /// H(a) / H0
        std::function<double(double)> E_of_a;
        /// (rho + p) / rho of a species. Only needed for MassiveNeutrino and TotalMatter (CDM and baryons: 1)
        std::function<double(Species, double)> one_plus_w;
    };

    class GaugeView final : public TransferProvider {
        std::shared_ptr<TransferProvider> source;
        Gauge target;
        GaugeBackground bg;

        bool converts() const { return source->density_gauge() != target; }

        double one_plus_w(Species s, double a) const {
            if (s == Species::CDM or s == Species::Baryon or s == Species::CDMBaryon)
                return 1.0;
            if (not bg.one_plus_w)
                throw std::invalid_argument("[GaugeView] Converting massive neutrino or total matter densities needs "
                                            "(rho + p)/rho from the background (GaugeBackground::one_plus_w)");
            return bg.one_plus_w(s, a);
        }

        double converted_density(Species s, double k, double a) const {
            const double x = a * bg.E_of_a(a) / (FML::UTILS::SI::c_over_H0_mpch * k);
            return source->density_transfer(s, k, a) +
                   3.0 * one_plus_w(s, a) * x * x * source->velocity_divergence_transfer(Species::CDM, k, a);
        }

      public:
        GaugeView(std::shared_ptr<TransferProvider> source, Gauge target, GaugeBackground background = {})
            : source(std::move(source)), target(target), bg(std::move(background)) {
            if (not this->source)
                throw std::invalid_argument("[GaugeView] No source provider");
            if (target != Gauge::SynchronousCDM and target != Gauge::Newtonian)
                throw std::invalid_argument("[GaugeView] Unknown target gauge");
            if (not converts())
                return;
            if (not(this->source->density_gauge() == Gauge::Newtonian and target == Gauge::SynchronousCDM))
                throw std::invalid_argument("[GaugeView] Only Newtonian -> CDM-synchronous is supported (the other way "
                                            "needs the metric perturbations, not only velocities)");
            if (not this->source->has_velocity(Species::CDM) or this->source->velocity_gauge() != Gauge::Newtonian)
                throw std::invalid_argument("[GaugeView] Converting Newtonian-gauge densities needs the Newtonian-gauge "
                                            "CDM velocity (theta_cdm), which the source does not have");
            if (not bg.E_of_a)
                throw std::invalid_argument("[GaugeView] Converting the gauge needs H(a)/H0 (GaugeBackground::E_of_a)");
        }

        const TransferProvider & underlying() const { return *source; }

        double h() const override { return source->h(); }
        TransferRange range() const override { return source->range(); }
        Gauge density_gauge() const override { return target; }
        Gauge velocity_gauge() const override { return source->velocity_gauge(); }
        RangePolicy range_policy() const override { return source->range_policy(); }
        bool has_density(Species s) const override {
            if (not converts())
                return source->has_density(s);
            if (not source->has_density(s))
                return false;
            // Neutrinos and total matter also need (1 + w) from the background
            return bg.one_plus_w or s == Species::CDM or s == Species::Baryon or s == Species::CDMBaryon;
        }
        bool has_velocity(Species s) const override { return source->has_velocity(s); }

        double density_transfer(Species s, double k, double a) const override {
            if (not converts())
                return source->density_transfer(s, k, a);
            check_query(s, k, a);
            return converted_density(s, k, a);
        }

        /// dlog(delta)/dlog(a) of the converted density: the gauge term changes it, so this is not the growth rate of
        /// the source. Central difference in log(a), one-sided at the ends of the range
        double density_growth_rate(Species s, double k, double a) const override {
            if (not converts())
                return source->density_growth_rate(s, k, a);
            check_query(s, k, a);
            const auto r = range();
            const double x = std::log(a), dx = 1e-3;
            const double lo = std::max(x - dx, std::log(r.amin)), hi = std::min(x + dx, std::log(r.amax));
            const double value = converted_density(s, k, a);
            if (value == 0.0 or not(hi > lo))
                throw std::domain_error("[GaugeView] Undefined density logarithmic derivative");
            return (converted_density(s, k, std::exp(hi)) - converted_density(s, k, std::exp(lo))) / ((hi - lo) * value);
        }

        double velocity_divergence_transfer(Species s, double k, double a) const override {
            return source->velocity_divergence_transfer(s, k, a);
        }
        double primordial_power(double k) const override { return source->primordial_power(k); }
        void rescale_primordial_power(double factor) override { source->rescale_primordial_power(factor); }
    };

} // namespace FML::INITIALCONDITIONS
#endif
