#ifndef GRAVITYMODEL_COUPLEDQUINTESSENCE_HEADER
#define GRAVITYMODEL_COUPLEDQUINTESSENCE_HEADER

#include "GravityModel.h"

#include <iostream>
#include <memory>

//=============================================================================
/// Method reference: Amendola (2000), https://arxiv.org/abs/astro-ph/9908023.
/// Coupled quintessence (Wetterich 1995; Amendola 2000): the force and the velocity term of a scalar field coupled to
/// the CDM, whose particles have the mass m ~ exp(-beta phi). The background (phi(a), the CDM density with the varying
/// mass) is that of cosmology_model = CoupledQuintessence, which this model requires (it takes beta and phi(a) from
/// there, as JBD takes phi(a) from its cosmology). In the N-body equations, the sub-horizon limit of those of the
/// Einstein-Boltzmann solver (FML/Cosmology/Perturbations/SpeciesPerturbation_CoupledQuintessence.h):
///  - the CDM feels G (1 + 2 beta^2) from the CDM (the fifth force of the light field: unscreened and scale
///    independent on N-body scales) and G from the baryons; the baryons feel G (cdm_fifth_force_coupling = 2 beta^2)
///  - the CDM has the velocity term +beta phi' v (velocity_friction_cdm = -beta dphi/dlna: an acceleration for
///    beta phi' > 0); the baryons none
/// With the CDM and baryons as one set of particles they are a momentum-weighted fluid: G_eff/G = 1 + 2 beta^2 f_c^2
/// (GeffOverG) and the friction f_c (-beta phi'), with f_c(a) = rho_c(a) / rho_cb(a). With CDM and baryons as
/// separate particles (particle_baryons) the PM force is gravity alone (GeffOverG_gravity = 1) and the CDM gets the
/// fifth force from the CDM density in a second kick (NBodySimulation::kick_cdm_fifth_force) and its own steps
/// (SpeciesTimeStepping.h). The varying CDM mass in the Poisson source is Cosmology::clustering_mass_factor
//=============================================================================
template <int NDIM>
class GravityModelCoupledQuintessence final : public GravityModel<NDIM> {
  public:
    explicit GravityModelCoupledQuintessence(std::shared_ptr<Cosmology> cosmo)
        : GravityModel<NDIM>(cosmo, "CoupledQuintessence") {
        FML::assert_mpi(cosmo->get_name() == "CoupledQuintessence",
                        "The CoupledQuintessence gravity model requires cosmology_model = CoupledQuintessence (it "
                        "takes the coupling beta and phi(a) from the background)");
    }

    /// The CDM+baryon particles as one fluid: the momentum-weighted force f_c [(1 + 2 beta^2) f_c + f_b] + f_b
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0) const override {
        const double f_c = this->cosmo->cdm_fraction_cb(a);
        return 1.0 + cdm_fifth_force_coupling(a) * f_c * f_c;
    }
    double GeffOverG_gravity(double /*a*/, double /*koverH0*/ = 0) const override { return 1.0; }
    bool has_cdm_fifth_force() const override { return this->cosmo->cq_beta() != 0.0; }
    double cdm_fifth_force_coupling(double /*a*/) const override {
        const double beta = this->cosmo->cq_beta();
        return 2.0 * beta * beta;
    }

    bool has_velocity_friction() const override { return this->cosmo->cq_beta() != 0.0; }
    double velocity_friction(double a) const override {
        if (not has_velocity_friction())
            return 0.0;
        return this->cosmo->cdm_fraction_cb(a) * velocity_friction_cdm(a);
    }
    double velocity_friction_cdm(double a) const override {
        if (not has_velocity_friction())
            return 0.0;
        return -this->cosmo->cq_beta() * this->cosmo->cq_dphidlna(a);
    }

    // The fifth force has an infinite range on N-body scales (as gravity)
    double short_range_comoving_range(double /*a*/) const override { return 0.; }

    void info_model() const override {
        {
            const double beta = this->cosmo->cq_beta();
            std::cout << "# Coupled quintessence: beta = " << beta << ", CDM-CDM G_eff/G = 1 + 2 beta^2 = "
                      << 1.0 + 2.0 * beta * beta << ", CDM+baryons as one fluid G_eff/G(a = 1) = " << GeffOverG(1.0)
                      << "\n";
            std::cout << "# Velocity term of the CDM: Gamma = -beta dphi/dlna = " << velocity_friction_cdm(1.0)
                      << " (a = 1), " << velocity_friction_cdm(0.5) << " (a = 0.5) per ln a (one fluid: "
                      << velocity_friction(1.0) << ", " << velocity_friction(0.5) << ")\n";
        }
    }
};

#endif
