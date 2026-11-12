#ifndef FML_NBODY_TIME_STEPPING_H
#define FML_NBODY_TIME_STEPPING_H

#include <FML/Global/Global.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>

#include <FML/LPT/GrowthHistory.h>
#include <FML/NBody/TimeStepTypes.h>
#include <functional>
#include <stdexcept>

#include <cmath>
#include <utility>
#include <vector>

namespace FML::NBODY {

    //=============================================================================
    /// The time-stepping of the simulation: where the steps are in the scale factor
    /// and the size of the drift and kick steps. The methods:
    ///
    ///  Quinn    : standard leapfrog (Quinn et al. 1997, https://arxiv.org/abs/astro-ph/9710043)
    ///  Tassev   : modified leapfrog for COLA (Tassev et al. 2013, https://arxiv.org/abs/1301.0322)
    ///  FastPM   : leapfrog where the kick and drift factors are computed from the
    ///             linear growth factor so that the linear growing mode is followed
    ///             exactly for any step size (Feng et al. 2016, arXiv:1603.00476)
    ///  BullFrog : drift-kick-drift in growth-factor time with a kick that is chosen so
    ///             that the 2LPT growing mode is followed after every step
    ///             (Rampf, List & Hahn 2024, arXiv:2409.19049)
    ///
    /// The velocities are v = a^2 dx/dt / H0 = a^3 E(a) dx/da (x in units of the box).
    /// Each step is: velocities *= velocity_scale; kick (v += F * delta_time_kick with the
    /// force computed at apos); drift (x += v * delta_time_drift from apos to apos_new).
    ///
    /// Quinn, Tassev, FastPM (KDK): the first kick staggers the velocities before
    /// drifting the positions; the last entry synchronizes by only changing vel.
    /// BullFrog (DKD): the first step only drifts half a step, the forces are computed at
    /// the half-steps and the last step drifts half a step to synchronize pos and vel.
    ///
    /// Velocity friction (optional; none by default): with dv/dlna = -Gamma(a) v + force (e.g. dark matter -
    /// dark energy scattering) the caller gives the friction integral I(a) = int Gamma dlna (up to a constant).
    /// The steps are those of the friction-free velocity u = e^I v, converted back to v: the velocities are
    /// multiplied by velocity_scale = e^-(I(avel_new) - I(avel)) and the kick and drift factors are weighted by
    /// e^(I(a) - I(avel_new)) and e^(I(avel) - I(a)). Quinn and Tassev: in the quadratures (Tassev's analytic
    /// kick with the factor at the time of the force). FastPM: the kick that keeps the linear mode exact,
    /// (Gp(avel_new) - velocity_scale Gp(avel)) / Gf(apos), its drift is unchanged. BullFrog: unchanged (its
    /// coefficients only use D(a), E(a) and Gp(a), which include the friction through the growth history).
    /// The growth history must be that of the growth equations with the same friction.
    ///
    /// Several particle species with different frictions (e.g. dark scattering acting on the CDM but not on the
    /// baryons) that feel the same force: one TimeStepping per species with its friction integral, its own growth
    /// history (FastPM) and force_growth(a), the growth factor of the field that sources the force (default: the
    /// growth history's own). The steps of a species are restep(steps) of the reference steps: the same scale
    /// factors, the species' drift, kick and velocity_scale. restep of a TimeStepping's own steps gives them back
    /// exactly. Not for BullFrog (its coefficients are those of one fluid).
    //=============================================================================
    class TimeStepping {
      public:
        using ScaleFactors = std::pair<std::vector<double>, std::vector<double>>; // (a_pos, a_vel)
        using TimeSteps = std::pair<std::vector<double>, std::vector<double>>;    // (dt_drift, dt_kick)

        /// All we need to know to take one step
        struct Step {
            double apos;             // Scale factor of the positions at the start of the step (forces computed here)
            double apos_new;         // ...and at the end of the step
            double avel;             // Scale factor of the velocities at the start of the step
            double avel_new;         // ...and at the end of the step
            double delta_time_drift; // x += v * delta_time_drift
            double delta_time_kick;  // v += F * delta_time_kick
            double velocity_scale;   // v *= velocity_scale (before the kick)
        };

        /// force_prefactor(a) is the amplitude used by the caller's force grid,
        /// e.g. 3/2 OmegaM mu(a) a for the SimulationCode force convention.
        TimeStepping(TimestepOptions options,
                     std::function<double(double)> HoverH0,
                     std::function<double(double)> force_prefactor,
                     FML::COSMOLOGY::GrowthHistory growth = {},
                     std::function<double(double)> friction_integral = {},
                     std::function<double(double)> force_growth = {})
            : settings(options), HoverH0(std::move(HoverH0)), force_prefactor(std::move(force_prefactor)),
              growth(std::move(growth)), friction_integral(std::move(friction_integral)),
              force_growth(std::move(force_growth)) {
            if (!this->HoverH0 || !this->force_prefactor)
                throw std::invalid_argument("[TimeStepping] Expansion and force-prefactor functions are required");
            if (!std::isfinite(settings.cola_nLPT) ||
                (settings.method == TimestepMethod::Tassev && settings.cola_nLPT == 0.0) ||
                !std::isfinite(settings.spacing_power) ||
                (settings.spacing == ScaleFactorSpacing::Powerlaw && settings.spacing_power == 0.0))
                throw std::invalid_argument("[TimeStepping] Invalid timestep exponent");
        }

        /// The scale factor of step i (can be fractional) out of nsteps between amin and amax
        double scale_factor_of_step(double amin, double amax, int nsteps, double i) const {
            validate_interval(amin, amax, nsteps);
            if (nsteps == 0)
                return amin;
            if (!std::isfinite(i))
                throw std::invalid_argument("[TimeStepping] Step index must be finite");
            switch (settings.spacing) {
                case ScaleFactorSpacing::Linear:
                    return amin + (amax - amin) * i / double(nsteps);
                case ScaleFactorSpacing::Logarithmic:
                    return std::exp(std::log(amin) + std::log(amax / amin) * i / double(nsteps));
                case ScaleFactorSpacing::Powerlaw: {
                    // Steps uniformly spaced in a^n (n = 1 is linear, n -> 0 is logarithmic)
                    const double n = settings.spacing_power;
                    return amin * std::pow(1.0 + (std::pow(amax / amin, n) - 1.0) * i / double(nsteps), 1.0 / n);
                }
                case ScaleFactorSpacing::GrowthFactor: {
                    // Steps uniformly spaced in the linear growth factor D(a) (scale-independent part)
                    if (i <= 0.0)
                        return amin;
                    if (i >= nsteps)
                        return amax;
                    const double Dmin = D(amin), Dmax = D(amax);
                    const double Dtarget = Dmin + (Dmax - Dmin) * i / double(nsteps);
                    return scale_factor_of_D(Dtarget, amin, amax);
                }
            }
            throw std::runtime_error("TimeStepping: unknown scale factor spacing");
        }

        /// The scale factors the positions and velocities are at after each step
        ScaleFactors scalefactors_KDK(double amin, double amax, int nsteps) const {
            validate_interval(amin, amax, nsteps);
            std::vector<double> scalefactor_pos{amin};
            std::vector<double> scalefactor_vel{amin};
            for (int i = 0; i <= nsteps; i++) {
                double apos_new = (i == nsteps) ? amax : scale_factor_of_step(amin, amax, nsteps, i + 1.0);
                double avel_new = (i == 0) ? scale_factor_of_step(amin, amax, nsteps, 0.5) :
                                             (i == nsteps ? amax : scale_factor_of_step(amin, amax, nsteps, i + 0.5));
                scalefactor_pos.push_back(apos_new);
                scalefactor_vel.push_back(avel_new);
            }
            return {scalefactor_pos, scalefactor_vel};
        }

        /// Quinn/Tassev quadratures. Use steps() for FastPM/BullFrog coefficients.
        TimeSteps deltatime_KDK(double amin, double amax, int nsteps) const {
            validate_interval(amin, amax, nsteps);
            if (nsteps == 0)
                return {{}, {}};

            std::vector<double> delta_time_drift;
            std::vector<double> delta_time_kick;
            for (int i = 0; i <= nsteps; i++) {
                double apos_old = scale_factor_of_step(amin, amax, nsteps, i);
                double apos_new = (i == nsteps) ? amax : scale_factor_of_step(amin, amax, nsteps, i + 1.0);
                double avel_old = (i == 0) ? amin : scale_factor_of_step(amin, amax, nsteps, i - 0.5);
                double avel_new = (i == 0) ? scale_factor_of_step(amin, amax, nsteps, 0.5) :
                                             (i == nsteps ? amax : scale_factor_of_step(amin, amax, nsteps, i + 0.5));
                const auto dt = deltatime_of_step(apos_old, apos_new, avel_old, avel_new, i == 0, i == nsteps);
                delta_time_drift.push_back(dt.first);
                delta_time_kick.push_back(dt.second);
            }
            return {delta_time_drift, delta_time_kick};
        }

        /// The steps of this TimeStepping at the scale factors of other steps (from steps() of a TimeStepping with
        /// the same method and spacing, e.g. of another particle species): the same apos, apos_new, avel, avel_new
        /// and this one's drift, kick and velocity_scale. restep(steps(...)) == steps(...) exactly. Not for BullFrog
        std::vector<Step> restep(const std::vector<Step> & reference) const {
            if (settings.method == TimestepMethod::BullFrog)
                throw std::invalid_argument("[TimeStepping] restep is not implemented for BullFrog");
            std::vector<Step> result;
            const size_t n = reference.size();
            for (size_t i = 0; i < n; i++) {
                const auto & r = reference[i];
                const double velocity_scale = friction_factor(r.avel, r.avel_new);
                if (settings.method == TimestepMethod::FastPM) {
                    const double drift = (D(r.apos_new) - D(r.apos)) / Gp(r.avel_new);
                    const double kick = (Gp(r.avel_new) - velocity_scale * Gp(r.avel)) / Gf(r.apos);
                    result.push_back({r.apos, r.apos_new, r.avel, r.avel_new, drift, kick, velocity_scale});
                } else {
                    const auto dt = deltatime_of_step(r.apos, r.apos_new, r.avel, r.avel_new, i == 0, i + 1 == n);
                    result.push_back({r.apos, r.apos_new, r.avel, r.avel_new, dt.first, dt.second, velocity_scale});
                }
            }
            return result;
        }

        /// The steps we take to go from amin to amax (nsteps + 1 steps, the last one synchronizes)
        std::vector<Step> steps(double amin, double amax, int nsteps) const {
            validate_interval(amin, amax, nsteps);
            if (nsteps == 0)
                return {};
            if (settings.method == TimestepMethod::BullFrog)
                return steps_bullfrog(amin, amax, nsteps);

            // Leapfrog (KDK): Quinn, Tassev and FastPM
            const auto as = scalefactors_KDK(amin, amax, nsteps);
            std::vector<Step> result;
            if (settings.method == TimestepMethod::FastPM) {
                for (int i = 0; i <= nsteps; i++) {
                    const double apos = as.first[i], apos_new = as.first[i + 1];
                    const double avel = as.second[i], avel_new = as.second[i + 1];
                    const double velocity_scale = friction_factor(avel, avel_new);
                    const double drift = (D(apos_new) - D(apos)) / Gp(avel_new);
                    const double kick = (Gp(avel_new) - velocity_scale * Gp(avel)) / Gf(apos);
                    result.push_back({apos, apos_new, avel, avel_new, drift, kick, velocity_scale});
                }
                return result;
            }
            const auto dt = deltatime_KDK(amin, amax, nsteps);
            for (int i = 0; i <= nsteps; i++)
                result.push_back({as.first[i],
                                  as.first[i + 1],
                                  as.second[i],
                                  as.second[i + 1],
                                  dt.first[i],
                                  dt.second[i],
                                  friction_factor(as.second[i], as.second[i + 1])});
            return result;
        }

      private:
        TimestepOptions settings;
        std::function<double(double)> HoverH0, force_prefactor;
        FML::COSMOLOGY::GrowthHistory growth;
        std::function<double(double)> friction_integral;
        std::function<double(double)> force_growth;

        // Quinn/Tassev drift and kick of step i (first: i = 0, last: i = nsteps) of a KDK sequence
        std::pair<double, double> deltatime_of_step(
            double apos_old, double apos_new, double avel_old, double avel_new, bool first, bool last) const {
            // The force of the kick is computed at apos_old (with poisson_factor(apos_old)) and the Tassev drift
            // weights the drift with the time of the velocities avel_new. With linear spacing these are the
            // midpoints of [avel_old, avel_new] and [apos_old, apos_new] (the first and last kick: their ends),
            // and we use that expression (which differs from apos_old and avel_new by round-off) so that the
            // results are unchanged. With other spacings the midpoints are not these times, which gave the linear
            // growth wrong by -0.76% (Quinn, 20 logarithmic steps from a = 0.05) instead of -0.38%
            double akick = apos_old, adrift = avel_new;
            if (settings.spacing == ScaleFactorSpacing::Linear) {
                akick = first ? avel_old : (last ? avel_new : (avel_old + avel_new) / 2.);
                adrift = (apos_old + apos_new) / 2.;
            }
            return {drift_timestep(apos_old, apos_new, adrift), kick_timestep(avel_old, avel_new, akick)};
        }

        using DVector = FML::INTERPOLATION::SPLINE::DVector;
        using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
        using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;

        static void validate_interval(double amin, double amax, int nsteps) {
            if (!(amin > 0.0) || !std::isfinite(amin) || !std::isfinite(amax) || amax < amin || nsteps < 0 ||
                (nsteps > 0 && amax == amin))
                throw std::invalid_argument("[TimeStepping] Need 0 < amin < amax and nonnegative step count");
        }
        // Integrate dt/da from alow to ahigh
        static double integrate(ODEFunction deriv, double alow, double ahigh) {
            if (alow == ahigh)
                return 0.0;
            DVector tini{0.0};
            DVector avec{alow, ahigh};
            ODESolver ode;
            ode.solve(deriv, avec, tini);
            return ode.get_final_data()[0];
        }

        // The explicit time evolution of the prefactor to the force (1.5*OmegaM*a in GR)
        // (In general this will depend on scale, but we are not that crazy that we compute
        // a factor per scale. Probably perfectly fine to simply use the GR expression in general)
        double expansion(double a) const {
            const double value = HoverH0(a);
            if (!(value > 0.0) || !std::isfinite(value))
                throw std::domain_error("[TimeStepping] H/H0 must be positive and finite");
            return value;
        }
        double poisson_factor(double a) const {
            const double value = force_prefactor(a);
            if (value == 0.0 || !std::isfinite(value))
                throw std::domain_error("[TimeStepping] Force prefactor must be nonzero and finite");
            return value;
        }

        // e^-(I(ato) - I(afrom)): how much the friction reduces the velocity from afrom to ato (1 without friction)
        double friction_factor(double afrom, double ato) const {
            if (!friction_integral)
                return 1.0;
            const double value = std::exp(-(friction_integral(ato) - friction_integral(afrom)));
            if (!(value > 0.0) || !std::isfinite(value))
                throw std::domain_error("[TimeStepping] Friction factor must be positive and finite");
            return value;
        }

        // The drift from alow to ahigh with the velocities at avel (Tassev; and the time of the velocities with
        // friction: v(a) = v(avel) e^-(I(a) - I(avel)))
        double drift_timestep(double alow, double ahigh, double avel) const {
            if (settings.method == TimestepMethod::Tassev) {
                // Tassev et al.
                const double amid = avel;
                ODEFunction deriv = [&](double a, [[maybe_unused]] const double * t, double * dtda) {
                    dtda[0] = 1.0 / (a * a * a * expansion(a));
                    dtda[0] *= std::pow(a / amid, settings.cola_nLPT);
                    if (friction_integral)
                        dtda[0] *= friction_factor(avel, a);
                    return GSL_SUCCESS;
                };
                return integrate(deriv, alow, ahigh);
            }
            // Quinn et al.
            ODEFunction deriv = [&](double a, [[maybe_unused]] const double * t, double * dtda) {
                dtda[0] = 1.0 / (a * a * a * expansion(a));
                if (friction_integral)
                    dtda[0] *= friction_factor(avel, a);
                return GSL_SUCCESS;
            };
            return integrate(deriv, alow, ahigh);
        }

        // The kick from alow to ahigh with the force computed at apos
        double kick_timestep(double alow, double ahigh, double apos) const {
            const double amid = apos;

            if (settings.method == TimestepMethod::Tassev) {
                // Tassev et al.
                const double n = settings.cola_nLPT;
                const double da = (std::pow(ahigh, n) - std::pow(alow, n)) / (n * std::pow(amid, n - 1.0));
                const double kick = da / (amid * amid * amid * expansion(amid));
                return friction_integral ? kick * friction_factor(amid, ahigh) : kick;
            }
            // Quinn et al. (with friction the kick at a is reduced by e^-(I(ahigh) - I(a)) by the end of the kick)
            ODEFunction deriv = [&](double a, [[maybe_unused]] const double * t, double * dtda) {
                dtda[0] = 1.0 / (a * a * a * expansion(a)) * poisson_factor(a) / poisson_factor(amid);
                if (friction_integral)
                    dtda[0] *= friction_factor(a, ahigh);
                return GSL_SUCCESS;
            };
            return integrate(deriv, alow, ahigh);
        }

        //=============================================================================
        // Growth factors for FastPM and BullFrog. We use the scale-independent (large
        // scale) growth factors. With v = a^3 E dx/da the linear growing mode has
        // x = q + D Psi, v = Gp Psi, F = Gf Psi
        //=============================================================================

        // Linear growth factor
        double D(double a) const { return growth.linear(a); }
        // Second order growth factor (x = q + D Psi1 + E Psi2 with E ~ -3/7 D^2)
        double E(double a) const { return growth.second_order(a); }
        // Gp = a^3 E dD/da
        double Gp(double a) const {
            const double value = growth.linear_momentum(a, expansion(a));
            if (value == 0.0 || !std::isfinite(value))
                throw std::domain_error("[TimeStepping] Linear momentum growth must be nonzero and finite");
            return value;
        }
        // Gf = 1.5 OmegaM Geff/G a D: the force of the linear mode, F = Gf Psi (= a^3 E dGp/da from the growth
        // equation without friction). With force_growth (several species) D is that of the field of the force
        double Gf(double a) const {
            const double value = poisson_factor(a) * (force_growth ? force_growth(a) : D(a));
            if (value == 0.0 || !std::isfinite(value))
                throw std::domain_error("[TimeStepping] Force growth must be nonzero and finite");
            return value;
        }
        // dE/dD
        double dEdD(double a) const { return growth.second_order_slope(a); }

        // The scale factor where the growth factor is Dvalue (bisection in log(a))
        double scale_factor_of_D(double Dvalue, double alow, double ahigh) const {
            const double Dlow = D(alow), Dhigh = D(ahigh);
            if (!(Dhigh > Dlow) || Dvalue < Dlow || Dvalue > Dhigh)
                throw std::domain_error("[TimeStepping] Growth-time spacing requires increasing linear growth");
            double loglow = std::log(alow), loghigh = std::log(ahigh);
            for (int i = 0; i < 100 and loghigh - loglow > 1e-14; i++) {
                const double logmid = 0.5 * (loglow + loghigh);
                if (D(std::exp(logmid)) < Dvalue)
                    loglow = logmid;
                else
                    loghigh = logmid;
            }
            return std::exp(0.5 * (loglow + loghigh));
        }

        // BullFrog: drift-kick-drift in D-time. With w = dx/dD = v/Gp and g = F/Gf (normalized such
        // that div g = -delta/D) a step from D_n to D_n+1 is (eq. 2.1 and 4.12 in arXiv:2409.19049):
        //   x_{n+1/2} = x_n + dD/2 w_n
        //   w_{n+1}   = alpha w_n + (1 - alpha) g(x_{n+1/2}) at D_{n+1/2} = D_n + dD/2
        //   x_{n+1}   = x_{n+1/2} + dD/2 w_{n+1}
        // with alpha = (E'_{n+1} - F) / (E'_n - F) and F = (E_n + E'_n dD/2) / D_{n+1/2} - D_{n+1/2}
        // We combine the two half-drifts between kicks into one drift
        std::vector<Step> steps_bullfrog(double amin, double amax, int nsteps) const {
            std::vector<double> a(nsteps + 1), Dn(nsteps + 1), ahalf(nsteps), Dhalf(nsteps);
            for (int i = 0; i <= nsteps; i++) {
                a[i] = (i == nsteps) ? amax : scale_factor_of_step(amin, amax, nsteps, i);
                Dn[i] = D(a[i]);
            }
            for (int i = 0; i < nsteps; i++) {
                Dhalf[i] = 0.5 * (Dn[i] + Dn[i + 1]);
                ahalf[i] = scale_factor_of_D(Dhalf[i], a[i], a[i + 1]);
            }

            std::vector<Step> result;
            // First step: only drift half a step
            result.push_back({a[0], ahalf[0], a[0], a[0], (Dhalf[0] - Dn[0]) / Gp(a[0]), 0.0, 1.0});
            for (int i = 0; i < nsteps; i++) {
                // Kick from D_i to D_i+1 with the force at D_i+1/2
                const double dD = Dn[i + 1] - Dn[i];
                const double Fh = (E(a[i]) + dEdD(a[i]) * dD / 2.0) / Dhalf[i] - Dhalf[i];
                const double alpha = (dEdD(a[i + 1]) - Fh) / (dEdD(a[i]) - Fh);
                const double velocity_scale = alpha * Gp(a[i + 1]) / Gp(a[i]);
                const double kick = (1.0 - alpha) * Gp(a[i + 1]) / Gf(ahalf[i]);

                // Drift to the next half-step (or to the end)
                const double apos_new = (i + 1 == nsteps) ? a[nsteps] : ahalf[i + 1];
                const double Dpos_new = (i + 1 == nsteps) ? Dn[nsteps] : Dhalf[i + 1];
                const double drift = (Dpos_new - Dhalf[i]) / Gp(a[i + 1]);
                result.push_back({ahalf[i], apos_new, a[i], a[i + 1], drift, kick, velocity_scale});
            }
            return result;
        }
    };

} // namespace FML::NBODY

#endif
