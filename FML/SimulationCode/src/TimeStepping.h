#ifndef TIMESTEPPING_HEADER
#define TIMESTEPPING_HEADER
#include <FML/NBody/TimeStepping.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include "Cosmology.h"
#include "GravityModel.h"
#include "SimulationParameters.h"
namespace SIMCODE {
    /// Adapts SimulationCode's model ownership and settings to the shared coefficient generator.
    /// k=0 retains the existing large-scale approximation for scale-dependent growth.
    template <int NDIM>
    FML::COSMOLOGY::GrowthHistory large_scale_growth_history(std::shared_ptr<GravityModel<NDIM>> grav) {
        if (!grav)
            throw std::invalid_argument("[TimeStepping] Gravity model is required");
        return {[grav](double a) { return grav->get_D_1LPT(a); },
                [grav](double a) { return grav->get_f_1LPT(a); },
                [grav](double a) { return grav->get_D_2LPT(a); },
                [grav](double a) { return grav->get_f_2LPT(a); }};
    }
    template <int NDIM>
    class TimeStepping : public FML::NBODY::TimeStepping {
      public:
        TimeStepping(const TimestepSettings & settings,
                     std::shared_ptr<Cosmology> cosmo,
                     std::shared_ptr<GravityModel<NDIM>> grav)
            : FML::NBODY::TimeStepping(
                  {settings.method, settings.spacing, settings.cola_nLPT, settings.spacing_power},
                  [cosmo](double a) { return cosmo->HoverH0_of_a(a); },
                  [cosmo, grav](double a) { return 1.5 * cosmo->get_OmegaM() * grav->GeffOverG(a) * a; },
                  large_scale_growth_history(grav)) {
            if (!cosmo)
                throw std::invalid_argument("[TimeStepping] Cosmology is required");
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
                auto all_steps = steps(1.0 / (1.0 + zmin), 1.0 / (1.0 + zmax), nsteps[ioutput]);
                for (size_t i = 0; i < all_steps.size(); i++) {
                    const auto & s = all_steps[i];
                    std::cout << "    Step " << std::setw(4) << i << " / " << std::setw(4) << all_steps.size() - 1
                              << " Pos: " << std::setw(15) << s.apos << " -> " << std::setw(15) << s.apos_new
                              << " Vel: " << std::setw(15) << s.avel << " -> " << std::setw(15) << s.avel_new << "\n";
                }
            }
        }
    };
} // namespace SIMCODE
#endif
