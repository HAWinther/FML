#ifndef FML_BARYONIFICATION_PROFILES_H
#define FML_BARYONIFICATION_PROFILES_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace FML::BARYONIFICATION {

    namespace detail {
        inline void require(bool condition, const char * message) {
            if (!condition)
                throw std::invalid_argument(message);
        }
        inline bool positive(double x) { return std::isfinite(x) && x > 0.0; }
        inline bool nonnegative(double x) { return std::isfinite(x) && x >= 0.0; }
        inline constexpr double pi = 3.1415926535897932384626433832795;
    } // namespace detail

    /// Monotone cumulative mass table. Linear interpolation in radius and mass is deliberately used:
    /// unlike an unconstrained cubic spline it cannot introduce negative densities or overshoot.
    /// Tables start at (0,0); mass is constant beyond the last radius. Units are caller-selected.
    class EnclosedMassProfile {
      public:
        EnclosedMassProfile(std::vector<double> radii, std::vector<double> masses)
            : r(std::move(radii)), m(std::move(masses)) {
            detail::require(r.size() >= 2 && r.size() == m.size(), "Mass profile: need matching tables of size >= 2");
            detail::require(r.front() == 0.0 && m.front() == 0.0, "Mass profile: must start at (0,0)");
            for (size_t i = 0; i < r.size(); ++i) {
                detail::require(detail::nonnegative(r[i]) && detail::nonnegative(m[i]),
                                "Mass profile: radii and masses must be finite and nonnegative");
                if (i > 0)
                    detail::require(r[i] > r[i - 1] && m[i] >= m[i - 1],
                                    "Mass profile: radii must increase and masses must not decrease");
            }
        }

        double mass(double radius) const {
            detail::require(detail::nonnegative(radius), "Mass profile: radius must be finite and nonnegative");
            if (radius >= r.back())
                return m.back();
            return interpolate(r, m, radius);
        }

        /// Leftmost radius enclosing the requested mass (zero maps to zero, including on a central plateau).
        double radius(double mass_value) const {
            detail::require(detail::nonnegative(mass_value) && mass_value <= m.back(),
                            "Mass profile: inverse mass is outside the table");
            if (mass_value == 0.0)
                return 0.0;
            return interpolate(m, r, mass_value);
        }

        double maximum_radius() const { return r.back(); }
        double total_mass() const { return m.back(); }
        const std::vector<double> & radii() const { return r; }
        const std::vector<double> & masses() const { return m; }

      private:
        std::vector<double> r, m;
        static double interpolate(const std::vector<double> & x, const std::vector<double> & y, double v) {
            const auto it = std::lower_bound(x.begin(), x.end(), v);
            const size_t i = size_t(it - x.begin());
            if (i == 0)
                return y.front();
            if (it == x.end())
                return y.back();
            if (*it == v)
                return y[i];
            return y[i - 1] + (y[i] - y[i - 1]) * ((v - x[i - 1]) / (x[i] - x[i - 1]));
        }
    };

    /// Mass-conserving isolated-halo transport: M_target(r_new) = M_source(r_old).
    /// Source and target must have the same outer radius and enclosed mass. Displacements are zero
    /// outside that radius; no taper is applied to the interior transport.
    class RadialMapping {
      public:
        RadialMapping(EnclosedMassProfile source, EnclosedMassProfile target)
            : source(std::move(source)), target(std::move(target)) {
            detail::require(this->source.maximum_radius() == this->target.maximum_radius(),
                            "Radial mapping: source and target must share an outer radius");
            const double total = this->source.total_mass();
            detail::require(detail::positive(total) && std::abs(total - this->target.total_mass()) <=
                                                           64.0 * std::numeric_limits<double>::epsilon() * total,
                            "Radial mapping: source and target must have equal positive total mass");
            // A plateau at the boundary would otherwise make the displacement discontinuous there.
            const auto & sm = this->source.masses();
            const auto & tm = this->target.masses();
            detail::require(sm.back() > sm[sm.size() - 2] && tm.back() > tm[tm.size() - 2],
                            "Radial mapping: profiles must increase in their last interval");
            identity = this->source.radii() == this->target.radii() && sm == tm;
        }

        double mapped_radius(double radius) const {
            detail::require(detail::nonnegative(radius), "Radial mapping: radius must be finite and nonnegative");
            if (identity || radius == 0.0 || radius >= maximum_radius())
                return radius;
            return target.radius(std::min(source.mass(radius), target.total_mass()));
        }
        double displacement(double radius) const { return mapped_radius(radius) - radius; }
        double maximum_radius() const { return source.maximum_radius(); }
        const EnclosedMassProfile & source_profile() const { return source; }
        const EnclosedMassProfile & target_profile() const { return target; }

      private:
        EnclosedMassProfile source, target;
        bool identity = false;
    };

    /// M200 and R200 must describe the same halo definition (normally 200 times critical density).
    /// All radii, particle positions and the box length use the SAME length unit. Mass units are arbitrary.
    struct HaloProperties {
        double mass200;
        double radius200;
        double concentration;
    };

    /// Schneider et al. (2019), arXiv:1810.08629, equations 6, 10--19.
    /// These are illustrative model parameters, not a calibration of FML/COLA to hydrodynamics.
    struct ModelParameters {
        double baryon_fraction = 0.157;
        double truncation_radius = 4.0;   // r_t / R200 for truncated NFW
        double gas_core_radius = 0.1;     // r_co / R200
        double gas_ejection_radius = 4.0; // r_ej / R200
        double gas_mass_scale = 1e14;     // same mass unit as M200
        double gas_mass_exponent = 0.3;   // beta = 3 - (M_c / M200)^mu
        double stellar_amplitude = 0.09;
        double stellar_mass_scale = 2.5e11;
        double stellar_total_exponent = 0.32;
        double stellar_central_exponent = 0.6;
        double stellar_radius = 0.015; // R_h / R200
        bool relax_collisionless = true;
        double relaxation_amplitude = 0.3; // r_f/r_i - 1 = A [(M_i/M_f)^n - 1]
        double relaxation_exponent = 2.0;

        void validate() const {
            using namespace detail;
            require(nonnegative(baryon_fraction) && baryon_fraction < 1.0,
                    "Baryonification: baryon_fraction must be in [0,1)");
            require(positive(truncation_radius) && positive(gas_core_radius) && gas_core_radius < 1.0 &&
                        positive(gas_ejection_radius) && gas_ejection_radius > 1.0 && positive(stellar_radius),
                    "Baryonification: invalid profile radii (need gas core < 1 and ejection > 1)");
            require(positive(gas_mass_scale) && nonnegative(gas_mass_exponent) && positive(stellar_mass_scale) &&
                        nonnegative(stellar_amplitude) && nonnegative(stellar_total_exponent) &&
                        nonnegative(stellar_central_exponent),
                    "Baryonification: invalid mass scale, stellar fraction or exponent");
            require(nonnegative(relaxation_amplitude) && relaxation_amplitude < 1.0 && positive(relaxation_exponent),
                    "Baryonification: relaxation needs 0 <= amplitude < 1 and exponent > 0");
        }
    };

    struct ProfileNumerics {
        size_t radial_bins = 2048;
        double minimum_radius = 1e-6; // first positive radius / R200
        double maximum_radius = 30.0; // outer support / R200; increase to test cutoff convergence
    };

    struct HaloProfiles {
        EnclosedMassProfile reference; // truncated NFW halo, WITHOUT background
        EnclosedMassProfile gas;
        EnclosedMassProfile central_stars;
        EnclosedMassProfile collisionless; // dark matter + satellite/diffuse stars
        EnclosedMassProfile source;        // reference + common background
        EnclosedMassProfile target;        // three components + common background
        double total_halo_mass;
        double gas_fraction;
        double central_stellar_fraction;
        double collisionless_fraction;
        RadialMapping mapping() const { return {source, target}; }
    };

    namespace detail {
        template <class F>
        std::vector<double> integrate_shells(const std::vector<double> & x, F shell) {
            std::vector<double> cumulative(x.size(), 0.0);
            for (size_t i = 1; i < x.size(); ++i) {
                const double lo = x[i - 1], hi = x[i];
                const double area = (hi - lo) / 6.0 * (shell(lo) + 4.0 * shell((lo + hi) / 2.0) + shell(hi));
                require(nonnegative(area), "Baryonification: non-finite or negative profile integral");
                cumulative[i] = cumulative[i - 1] + area;
                require(std::isfinite(cumulative[i]), "Baryonification: profile integral overflow");
            }
            require(positive(cumulative.back()), "Baryonification: profile has zero integral");
            return cumulative;
        }
    } // namespace detail

    /// Build the 2019 profile family with finite support and a shared UNIFORM background density.
    /// Densities are mass / length^3. Use comoving quantities together or physical quantities together.
    /// Differences from the published implementation: profiles are normalized at the finite outer radius,
    /// and the correlated two-halo term is replaced by the supplied uniform background. See README.md.
    inline HaloProfiles make_halo_profiles(const HaloProperties & halo,
                                           const ModelParameters & p,
                                           double background_density,
                                           const ProfileNumerics & numerics = {}) {
        using namespace detail;
        p.validate();
        require(positive(halo.mass200) && positive(halo.radius200) && positive(halo.concentration),
                "Baryonification: halo mass, radius and concentration must be finite and positive");
        require(nonnegative(background_density), "Baryonification: invalid background density");
        require(numerics.radial_bins >= 32 && positive(numerics.minimum_radius) && numerics.minimum_radius < 1.0 &&
                    positive(numerics.maximum_radius) && numerics.maximum_radius > 1.0,
                "Baryonification: need >= 32 radial bins, 0 < minimum_radius < 1 < maximum_radius");

        const double fstar =
            p.stellar_amplitude * std::pow(p.stellar_mass_scale / halo.mass200, p.stellar_total_exponent);
        const double fcga =
            p.stellar_amplitude * std::pow(p.stellar_mass_scale / halo.mass200, p.stellar_central_exponent);
        require(nonnegative(fstar) && nonnegative(fcga) && fcga <= fstar && fstar <= p.baryon_fraction,
                "Baryonification: stellar laws require 0 <= f_central <= f_stars <= baryon_fraction for this halo");
        const double fgas = p.baryon_fraction - fstar, fclm = 1.0 - fgas - fcga;
        const double beta = 3.0 - std::pow(p.gas_mass_scale / halo.mass200, p.gas_mass_exponent);
        require(std::isfinite(beta), "Baryonification: non-finite gas slope");

        std::vector<double> x{0.0};
        const double loglo = std::log(numerics.minimum_radius), loghi = std::log(numerics.maximum_radius);
        for (size_t i = 0; i < numerics.radial_bins; ++i)
            x.push_back(std::exp(loglo + (loghi - loglo) * double(i) / double(numerics.radial_bins - 1)));
        x.back() = numerics.maximum_radius;
        x.push_back(1.0); // normalize M(<R200) without interpolation across that radius
        std::sort(x.begin(), x.end());
        x.erase(std::unique(x.begin(), x.end()), x.end());
        auto nfw = integrate_shells(x, [&](double q) {
            const double c = halo.concentration, t = q / p.truncation_radius;
            return q / (c * std::pow(1.0 + c * q, 2) * std::pow(1.0 + t * t, 2));
        });
        const auto i200 = size_t(std::lower_bound(x.begin(), x.end(), 1.0) - x.begin());
        const double norm = halo.mass200 / nfw[i200];
        for (double & mass : nfw)
            mass *= norm;
        const double total = nfw.back();
        require(positive(total), "Baryonification: non-finite halo normalization");
        auto gas = integrate_shells(x, [&](double q) {
            if (q == 0.0)
                return 0.0;
            const double v = q / p.gas_ejection_radius;
            return std::exp(2.0 * std::log(q) - beta * std::log1p(q / p.gas_core_radius) -
                            0.5 * (7.0 - beta) * std::log1p(v * v));
        });
        const double gas_norm = fgas * total / gas.back();
        for (double & mass : gas)
            mass *= gas_norm;
        gas.back() = fgas * total;
        std::vector<double> radii(x.size()), stars(x.size()), clm(x.size());
        const double star_norm = fcga * total / std::erf(x.back() / (2.0 * p.stellar_radius));
        for (size_t i = 0; i < x.size(); ++i) {
            radii[i] = halo.radius200 * x[i];
            stars[i] = star_norm * std::erf(x[i] / (2.0 * p.stellar_radius));
            clm[i] = fclm * nfw[i];
        }
        stars.back() = fcga * total;
        EnclosedMassProfile reference(radii, nfw), gas_profile(radii, gas), stellar_profile(radii, stars);
        EnclosedMassProfile collisionless(radii, clm);
        if (p.relax_collisionless && p.relaxation_amplitude > 0.0 && fclm < 1.0) {
            std::vector<double> rf = radii;
            for (size_t i = 1; i + 1 < rf.size(); ++i) {
                const double ri = radii[i], mi = nfw[i];
                double lo = (1.0 - p.relaxation_amplitude) * ri, hi = radii.back();
                for (int iter = 0; iter < 100; ++iter) {
                    const double mid = lo + 0.5 * (hi - lo);
                    if (mid == lo || mid == hi)
                        break;
                    const double mf = fclm * mi + gas_profile.mass(mid) + stellar_profile.mass(mid);
                    const double residual =
                        mid / ri - 1.0 - p.relaxation_amplitude * (std::pow(mi / mf, p.relaxation_exponent) - 1.0);
                    if (residual > 0.0)
                        hi = mid;
                    else
                        lo = mid;
                }
                rf[i] = lo + 0.5 * (hi - lo);
                require(rf[i] > rf[i - 1], "Baryonification: relaxation produced shell crossing");
            }
            EnclosedMassProfile relaxed_shells(rf, clm);
            for (size_t i = 0; i < radii.size(); ++i)
                clm[i] = relaxed_shells.mass(radii[i]);
            collisionless = EnclosedMassProfile(radii, clm);
        }

        std::vector<double> source(radii.size()), target(radii.size());
        for (size_t i = 0; i < radii.size(); ++i) {
            const double background = (4.0 * pi / 3.0) * background_density * std::pow(radii[i], 3);
            source[i] = nfw[i] + background;
            target[i] = gas[i] + stars[i] + clm[i] + background;
        }
        target.back() = source.back(); // shared outer mass, exactly, avoiding a round-off displacement at the edge
        return {std::move(reference),
                std::move(gas_profile),
                std::move(stellar_profile),
                std::move(collisionless),
                EnclosedMassProfile(radii, source),
                EnclosedMassProfile(radii, target),
                total,
                fgas,
                fcga,
                fclm};
    }

} // namespace FML::BARYONIFICATION
#endif
