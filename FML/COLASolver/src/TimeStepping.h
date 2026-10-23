#ifndef TIMESTEPPING_HEADER
#define TIMESTEPPING_HEADER

#include <FML/Global/Global.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"
#include "GravityModel.h"
#include "SimulationParameters.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace COLASOLVER {

    //=============================================================================
    /// The time-stepping of the simulation: where the steps are in the scale factor
    /// and the size of the drift and kick steps (Quinn et al. or Tassev et al.)
    ///
    /// We use leapfrog (KDK): the first step displaces pos and vel by half a step,
    /// the last step synchronizes them again by only moving vel
    //=============================================================================
    template <int NDIM>
    class TimeStepping {
      public:
        using ScaleFactors = std::pair<std::vector<double>, std::vector<double>>; // (a_pos, a_vel)
        using TimeSteps = std::pair<std::vector<double>, std::vector<double>>;    // (dt_drift, dt_kick)

        TimeStepping(const TimestepSettings & settings,
                     std::shared_ptr<Cosmology> cosmo,
                     std::shared_ptr<GravityModel<NDIM>> grav)
            : settings(settings), cosmo(cosmo), grav(grav) {}

        /// The scale factor of step i (can be fractional) out of nsteps between amin and amax
        double scale_factor_of_step(double amin, double amax, int nsteps, double i) const {
            if (nsteps == 0)
                return amin;
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
            }
            throw std::runtime_error("TimeStepping: unknown scale factor spacing");
        }

        /// The scale factors the positions and velocities are at after each step
        ScaleFactors scalefactors_KDK(double amin, double amax, int nsteps) const {
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

        /// The size of the drift and kick steps
        TimeSteps deltatime_KDK(double amin, double amax, int nsteps) const {
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
                delta_time_drift.push_back(drift_timestep(apos_old, apos_new));
                delta_time_kick.push_back(kick_timestep(avel_old, avel_new, i, nsteps));
            }
            return {delta_time_drift, delta_time_kick};
        }

        /// Split a total number of steps into the number of steps between each output
        std::vector<int>
        steps_between_outputs(double initial_redshift, const std::vector<double> & output_redshifts, int nsteps) const {
            auto as = scalefactors_KDK(1.0 / (1.0 + initial_redshift), 1.0 / (1.0 + output_redshifts.back()), nsteps);
            std::vector<int> nsteps_between_outputs;
            size_t icount = 0;
            for (auto zout : output_redshifts) {
                int i = 0;
                while (true) {
                    FML::assert_mpi(icount + i < as.first.size(),
                                    "[TimeStepping::steps_between_outputs] Ran past the last step. Are the "
                                    "output_redshifts decreasing?");
                    double z = 1.0 / as.first[icount + i] - 1.0;
                    if (z < zout + 1e-3)
                        break;
                    i++;
                }
                nsteps_between_outputs.push_back(i);
                icount += i;
            }
            return nsteps_between_outputs;
        }

        /// Print all the steps we will take
        void print_steps(double initial_redshift,
                         const std::vector<double> & output_redshifts,
                         const std::vector<int> & nsteps) const {
            if (FML::ThisTask != 0)
                return;
            for (size_t ioutput = 0; ioutput < output_redshifts.size(); ioutput++) {
                std::cout << "Next output z = " << output_redshifts[ioutput] << " nsteps: " << nsteps[ioutput] << "\n";
                double zmin = (ioutput == 0) ? initial_redshift : output_redshifts[ioutput - 1];
                double zmax = output_redshifts[ioutput];
                auto as = scalefactors_KDK(1.0 / (1.0 + zmin), 1.0 / (1.0 + zmax), nsteps[ioutput]);
                for (size_t i = 0; i < as.first.size() - 1; i++) {
                    std::cout << "    Step " << std::setw(4) << i << " / " << std::setw(4) << as.first.size() - 2
                              << " Pos: " << std::setw(15) << as.first[i] << " -> " << std::setw(15) << as.first[i + 1]
                              << " Vel: " << std::setw(15) << as.second[i] << " -> " << std::setw(15)
                              << as.second[i + 1] << "\n";
                }
            }
        }

      private:
        TimestepSettings settings;
        std::shared_ptr<Cosmology> cosmo;
        std::shared_ptr<GravityModel<NDIM>> grav;

        using DVector = FML::INTERPOLATION::SPLINE::DVector;
        using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
        using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;

        // Integrate dt/da from alow to ahigh
        static double integrate(ODEFunction deriv, double alow, double ahigh) {
            DVector tini{0.0};
            DVector avec{alow, ahigh};
            ODESolver ode;
            ode.solve(deriv, avec, tini);
            return ode.get_final_data()[0];
        }

        // The explicit time evolution of the prefactor to the force (1.5*OmegaM*a in GR)
        // (In general this will depend on scale, but we are not that crazy that we compute
        // a factor per scale. Probably perfectly fine to simply use the GR expression in general)
        double poisson_factor(double a) const { return 1.5 * cosmo->get_OmegaM() * grav->GeffOverG(a) * a; }

        double drift_timestep(double alow, double ahigh) const {
            if (settings.method == TimestepMethod::Tassev) {
                // Tassev et al.
                double amid = (ahigh + alow) / 2.;
                ODEFunction deriv = [&](double a, [[maybe_unused]] const double * t, double * dtda) {
                    dtda[0] = 1.0 / (a * a * a * cosmo->HoverH0_of_a(a));
                    dtda[0] *= std::pow(a / amid, settings.cola_nLPT);
                    return GSL_SUCCESS;
                };
                return integrate(deriv, alow, ahigh);
            }
            // Quinn et al.
            ODEFunction deriv = [&](double a, [[maybe_unused]] const double * t, double * dtda) {
                dtda[0] = 1.0 / (a * a * a * cosmo->HoverH0_of_a(a));
                return GSL_SUCCESS;
            };
            return integrate(deriv, alow, ahigh);
        }

        double kick_timestep(double alow, double ahigh, int istep, int nsteps) const {
            double amid = (ahigh + alow) / 2.;
            if (istep == 0)
                amid = alow;
            else if (istep == nsteps)
                amid = ahigh;

            if (settings.method == TimestepMethod::Tassev) {
                // Tassev et al.
                const double n = settings.cola_nLPT;
                const double da = (std::pow(ahigh, n) - std::pow(alow, n)) / (n * std::pow(amid, n - 1.0));
                return da / (amid * amid * amid * cosmo->HoverH0_of_a(amid));
            }
            // Quinn et al.
            ODEFunction deriv = [&](double a, [[maybe_unused]] const double * t, double * dtda) {
                dtda[0] = 1.0 / (a * a * a * cosmo->HoverH0_of_a(a)) * poisson_factor(a) / poisson_factor(amid);
                return GSL_SUCCESS;
            };
            return integrate(deriv, alow, ahigh);
        }
    };

} // namespace COLASOLVER

#endif
