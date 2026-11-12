#ifndef SPECIESTIMESTEPPING_HEADER
#define SPECIESTIMESTEPPING_HEADER

#include <FML/LPT/GrowthHistory.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"
#include "GravityModel.h"
#include "SimulationParameters.h"
#include "Species.h"
#include "TimeStepping.h"

#include <cmath>
#include <iostream>
#include <map>
#include <memory>

//=============================================================================
// The time stepping of particle species with different velocity frictions (GravityModel::velocity_friction): with
// CDM and baryons as separate particles (particle_baryons) and dark scattering the CDM feels the full drag
// (velocity_friction_cdm) and the baryons none, while both feel the force of the CDM+baryon density. With coupled
// quintessence the CDM has the velocity term +beta phi' v and also feels the fifth force of the CDM (G 2 beta^2,
// GravityModel::cdm_fifth_force_coupling, NBodySimulation::kick_cdm_fifth_force), and the CDM fraction f_c(a) of the CDM+baryon density varies. The steps
// (their scale factors) are those of the CDM+baryon fluid (SIMCODE::TimeStepping, the friction f_c Gamma of the
// one-fluid growth and of the COLA frame); each species gets its own drift, kick and velocity_scale at the same
// scale factors (FML::NBODY::TimeStepping::restep):
//  Quinn, Tassev: from the friction integral of the species.
//  FastPM: from the linear growth of the species in the two-fluid system (TwoSpeciesGrowth below; the kick with the
//          force of the CDM+baryon growth D_cb), so the linear growing mode of each species is followed exactly.
//  BullFrog: not supported (its coefficients are those of one fluid).
// With COLA the frame is the CDM+baryon one; a species whose friction differs from the frame's gets the change of
// its LPT velocity (velocity_scale_species - velocity_scale_frame) v_LPT(avel) in every kick (see step()).
//=============================================================================

namespace SIMCODE {

    /// The linear growth of CDM and baryons as two fluids with their own friction Gamma_c, Gamma_b, sourced by the
    /// CDM+baryon density (the 1LPT equation of the gravity model at k = 0, G_eff/G of gravity alone mu):
    ///   D_i'' + (2 + dlnE/dx + Gamma_i) D_i' = 3/2 Omega_m(a) mu s_1 (g_ic f_c D_c + g_ib f_b D_b),   x = ln a
    /// with f_c(a) = rho_c / rho_cb and the couplings g_cc = 1 + 2 beta^2 (coupled quintessence; 1 otherwise), g_cb =
    /// g_bc = g_bb = 1. From the growing mode of the one-fluid growth (the friction and the coupling are negligible at
    /// the start) and normalized to D_cb = f_c D_c + f_b D_b = 1 at a_normalize. With the one-fluid friction and
    /// couplings (all g = GeffOverG / GeffOverG_gravity of the gravity model) both are the one-fluid D1 (checked:
    /// max_one_fluid_deviation)
    template <int NDIM>
    class TwoSpeciesGrowth {
      public:
        using Spline = FML::INTERPOLATION::SPLINE::Spline;

        TwoSpeciesGrowth(const GravityModel<NDIM> & grav,
                         const Cosmology & cosmo,
                         std::function<double(double)> friction_cdm,
                         std::function<double(double)> friction_baryon,
                         double a_start,
                         double a_end,
                         double a_normalize) {
            cosmo_ptr = &cosmo;
            g_cc = 1.0 + grav.cdm_fifth_force_coupling(1.0);
            auto physical = [this](int i, int j, double) { return i == 0 and j == 0 ? g_cc : 1.0; };
            solve(grav, cosmo, friction_cdm, friction_baryon, physical, a_start, a_end, a_normalize, D_c, D_b);
            // The check: the one-fluid friction and coupling for both gives the one-fluid growth (as the gravity model
            // solves it)
            Spline Dc1, Db1;
            auto one_fluid = [&grav](double a) { return grav.velocity_friction(a); };
            auto one_fluid_coupling = [&grav](int, int, double a) { return grav.GeffOverG(a) / grav.GeffOverG_gravity(a); };
            solve(grav, cosmo, one_fluid, one_fluid, one_fluid_coupling, a_start, a_end, a_normalize, Dc1, Db1);
            for (double a = a_normalize; a <= std::min(a_end, 1.0); a *= 1.05) {
                const double ref = grav.get_D_1LPT(a) / grav.get_D_1LPT(a_normalize);
                max_one_fluid_deviation = std::max(max_one_fluid_deviation, std::abs(Dc1(std::log(a)) / ref - 1.0));
            }
        }

        double fraction_cdm(double a) const { return cosmo_ptr->cdm_fraction_cb(a); }
        double D_cdm(double a) const { return D_c(std::log(a)); }
        double D_baryon(double a) const { return D_b(std::log(a)); }
        double D_cb(double a) const {
            const double f_c = fraction_cdm(a);
            return f_c * D_cdm(a) + (1.0 - f_c) * D_baryon(a);
        }
        /// The coupling g_cc of the CDM to the CDM (1 + 2 beta^2; 1 without a coupling)
        double cdm_self_coupling() const { return g_cc; }
        /// mu_i = g_ic f_c + g_ib f_b: the force on species i (relative to gravity) for D_c = D_b
        double force_coupling(ParticleSpecies s, double a) const {
            if (g_cc == 1.0)
                return 1.0;
            const double f_c = fraction_cdm(a);
            return s == ParticleSpecies::CDM ? g_cc * f_c + (1.0 - f_c) : 1.0;
        }
        /// The growth of the field that sources the force on species i, (g_ic f_c D_c + g_ib f_b D_b) / mu_i (D_cb
        /// without a coupling)
        double D_force(ParticleSpecies s, double a) const {
            if (g_cc == 1.0 or s != ParticleSpecies::CDM)
                return D_cb(a);
            const double f_c = fraction_cdm(a);
            return (g_cc * f_c * D_cdm(a) + (1.0 - f_c) * D_baryon(a)) / force_coupling(s, a);
        }
        double f_cdm(double a) const { return D_c.deriv_x(std::log(a)) / D_c(std::log(a)); }
        double f_baryon(double a) const { return D_b.deriv_x(std::log(a)) / D_b(std::log(a)); }

        /// The growth history of a species for FastPM (D1 and its rate; the second order of the one fluid)
        FML::COSMOLOGY::GrowthHistory history(ParticleSpecies s, std::shared_ptr<GravityModel<NDIM>> grav) const {
            auto self = std::make_shared<const TwoSpeciesGrowth>(*this);
            const bool cdm = s == ParticleSpecies::CDM;
            return {[self, cdm](double a) { return cdm ? self->D_cdm(a) : self->D_baryon(a); },
                    [self, cdm](double a) { return cdm ? self->f_cdm(a) : self->f_baryon(a); },
                    [grav](double a) { return grav->get_D_2LPT(a); },
                    [grav](double a) { return grav->get_f_2LPT(a); }};
        }

        /// The largest relative difference of the two-fluid solution with the one-fluid friction for both species from
        /// the gravity model's D1 (a_normalize <= a <= 1): the accuracy of the ODE solutions
        double max_one_fluid_deviation{0.0};

      private:
        const Cosmology * cosmo_ptr{nullptr};
        double g_cc{1.0};
        Spline D_c, D_b;

        /// coupling(i, j, a): g_ij (i, j = 0: CDM, 1: baryons)
        void solve(const GravityModel<NDIM> & grav,
                   const Cosmology & cosmo,
                   const std::function<double(double)> & friction_cdm,
                   const std::function<double(double)> & friction_baryon,
                   const std::function<double(int, int, double)> & coupling,
                   double a_start,
                   double a_end,
                   double a_normalize,
                   Spline & Dc,
                   Spline & Db) const {
            const int n = 2000;
            std::vector<double> x(n);
            for (int i = 0; i < n; i++)
                x[i] = std::log(a_start) + std::log(a_end / a_start) * i / double(n - 1);
            FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double xx, const double * y, double * dydx) {
                const double a = std::exp(xx);
                const double dlogEdx = cosmo.dlogHdloga_of_a(a);
                const double f_c = cosmo.cdm_fraction_cb(a), fb = 1.0 - f_c;
                // G_eff/G of gravity alone (the gravity model's without the one-fluid coupling of the CDM)
                const double mu = grav.GeffOverG_gravity(a);
                const double norm = 1.5 * cosmo.Omega_m(a) * mu * grav.source_factor_1LPT(a);
                const double source_c = norm * (coupling(0, 0, a) * f_c * y[0] + coupling(0, 1, a) * fb * y[2]);
                const double source_b = norm * (coupling(1, 0, a) * f_c * y[0] + coupling(1, 1, a) * fb * y[2]);
                dydx[0] = y[1];
                dydx[1] = source_c - (2.0 + dlogEdx + friction_cdm(a)) * y[1];
                dydx[2] = y[3];
                dydx[3] = source_b - (2.0 + dlogEdx + friction_baryon(a)) * y[3];
                return GSL_SUCCESS;
            };
            // The growing mode of the one fluid at a_start
            const double D0 = grav.get_D_1LPT(a_start), f0 = grav.get_f_1LPT(a_start);
            std::vector<double> ini{D0, f0 * D0, D0, f0 * D0};
            FML::SOLVERS::ODESOLVER::ODESolver ode(1e-6, 1e-12, 1e-12);
            ode.solve(deriv, x, ini);
            auto dc = ode.get_data_by_component(0), db = ode.get_data_by_component(2);
            Spline tmpc(x, dc), tmpb(x, db);
            const double f_cn = cosmo.cdm_fraction_cb(a_normalize);
            const double norm = f_cn * tmpc(std::log(a_normalize)) + (1.0 - f_cn) * tmpb(std::log(a_normalize));
            for (auto & v : dc)
                v /= norm;
            for (auto & v : db)
                v /= norm;
            Dc = Spline(x, dc, "D_cdm(log(a))");
            Db = Spline(x, db, "D_baryon(log(a))");
        }
    };

    /// Do the species need their own time stepping (a friction or a coupling of the CDM, and CDM and baryons as
    /// separate particles)?
    template <int NDIM>
    bool species_have_own_time_stepping(const SimulationParameters & params, const GravityModel<NDIM> & grav) {
        return (grav.has_velocity_friction() or grav.has_cdm_fifth_force()) and params.particle.baryons;
    }

    /// The TimeStepping of each CDM+baryon species (CDM, Baryon) with its friction (and for FastPM its growth)
    template <int NDIM>
    std::map<ParticleSpecies, FML::NBODY::TimeStepping>
    make_species_time_stepping(const SimulationParameters & params,
                               std::shared_ptr<Cosmology> cosmo,
                               std::shared_ptr<GravityModel<NDIM>> grav) {
        const auto & ts = params.timestep;
        if (ts.method == FML::NBODY::TimestepMethod::BullFrog)
            throw std::runtime_error("timestep_method = BullFrog is not implemented for particle species with "
                                     "different velocity frictions or couplings (dark scattering or coupled "
                                     "quintessence with particle_baryons)");
        const double aini = params.ic.a_initial();
        auto friction_cdm = [grav](double a) { return grav->velocity_friction_cdm(a); };
        auto no_friction = [](double) { return 0.0; };
        auto growth = std::make_shared<TwoSpeciesGrowth<NDIM>>(*grav, *cosmo, friction_cdm, no_friction, aini / 50.0,
                                                               2.0, aini);
        if (FML::ThisTask == 0) {
            std::cout << "\n#=====================================================\n";
            std::cout << "# Time stepping of the species with their own friction (CDM: Gamma = "
                      << grav->velocity_friction_cdm(1.0) << " at a = 1, baryons: none) and coupling (CDM-CDM: "
                      << growth->cdm_self_coupling() << ")\n";
            std::cout << "# Two-fluid linear growth D(a = 1) / D(a_ini): CDM " << growth->D_cdm(1.0)
                      << ", baryons " << growth->D_baryon(1.0) << ", CDM+baryons " << growth->D_cb(1.0)
                      << " (one fluid with f_c Gamma: " << grav->get_D_1LPT(1.0) / grav->get_D_1LPT(aini) << ")\n";
            std::cout << "# Check of the two-fluid solver (one-fluid friction for both: deviation from the one-fluid "
                         "D1): "
                      << growth->max_one_fluid_deviation << "\n";
            std::cout << "#=====================================================\n";
        }
        if (growth->max_one_fluid_deviation > 1e-4)
            throw std::runtime_error("The two-fluid growth with the one-fluid friction differs from the one-fluid "
                                     "growth by " +
                                     std::to_string(growth->max_one_fluid_deviation));

        FML::NBODY::TimestepOptions options{ts.method, ts.spacing, ts.cola_nLPT, ts.spacing_power};
        auto HoverH0 = [cosmo](double a) { return cosmo->HoverH0_of_a(a); };
        // The force on a species for D_c = D_b (gravity: the PM force without the one-fluid coupling, times
        // force_coupling: the fifth force on the CDM) and the growth of the field that sources it
        auto force_prefactor = [cosmo, grav, growth](ParticleSpecies s) {
            return [cosmo, grav, growth, s](double a) {
                return 1.5 * cosmo->get_OmegaM() * cosmo->clustering_mass_factor(a) *
                       grav->GeffOverG_gravity(a) * growth->force_coupling(s, a) * a;
            };
        };
        auto force_growth = [growth](ParticleSpecies s) {
            return [growth, s](double a) { return growth->D_force(s, a); };
        };
        std::map<ParticleSpecies, FML::NBODY::TimeStepping> result;
        result.emplace(ParticleSpecies::CDM,
                       FML::NBODY::TimeStepping(options,
                                                HoverH0,
                                                force_prefactor(ParticleSpecies::CDM),
                                                growth->history(ParticleSpecies::CDM, grav),
                                                grav->velocity_friction_integral(1e-6, 100.0, true),
                                                force_growth(ParticleSpecies::CDM)));
        result.emplace(ParticleSpecies::Baryon,
                       FML::NBODY::TimeStepping(options,
                                                HoverH0,
                                                force_prefactor(ParticleSpecies::Baryon),
                                                growth->history(ParticleSpecies::Baryon, grav),
                                                {},
                                                force_growth(ParticleSpecies::Baryon)));
        return result;
    }
} // namespace SIMCODE

#endif
