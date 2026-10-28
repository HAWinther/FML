#ifndef FML_COSMOLOGY_MODELS_CUBICGALILEON_H
#define FML_COSMOLOGY_MODELS_CUBICGALILEON_H

// =============================================================================
// Cubic Galileon (Deffayet, Esposito-Farese & Vikman 2009; Barreira et al. 2013) on its tracker solution,
// flat, without a cosmological constant:
//   S = int sqrt(-g) [ R/(2 kappa) + c2 X - g3 X box phi ] + S_m,   X = -(1/2)(d phi)^2,  c2 = -1
// (kinetic gravity braiding with K = c2 X, G = g3 X). On the tracker H dphi/dt = xi is constant (an attractor),
// so rho_phi = xi^2/(2 kappa H^2) and the expansion has no free parameter beyond those of LCDM:
//   E^2 = [rho + sqrt(rho^2 + 4 Omega_phi)] / 2,   Omega_phi = 1 - Omega_m - Omega_r
// (rho: all other species, units of rho_c0), with xi = sqrt(6 Omega_phi) H0^2 and g3 = 1/(3 xi) (kappa = 1).
// The scalar is described by a background species (CubicGalileonSpecies, rho_phi = E^2 - rho) and by the gravity
// (CubicGalileonGravity), which owns its perturbations because the scalar mixes with the metric.
//
// Perturbations (Newtonian gauge; derived in python/derive_cubic_galileon.py): variables Phi, dphi and
// u = dphi' - phi' Psi (conformal time in 1/H0), with
//   Psi    = -Phi - 3 shear/(c^2 k^2)                       (no gravitational slip from the cubic term)
//   Phi'   from the 00 Einstein equation,  u' from the scalar field equation, in which Phi_P'' and Psi' are
//          eliminated with the derivative of the 0i equation and total matter momentum conservation.
// The 0i equation itself is not used and serves as a test.
// =============================================================================

#include <FML/Cosmology/Models/GeneralRelativity.h>

#include <memory>
#include <vector>

namespace FML::COSMOLOGY {

    class CubicGalileonSpecies : public Species {
      public:
        // others: all other species (flat universe); Omega_phi from closure
        CubicGalileonSpecies(const std::vector<std::shared_ptr<const Species>> & others, double Omega_phi);
        std::string name() const override { return "cubic galileon"; }
        Role role() const override { return Role::DarkEnergy; }
        std::string info() const override;
        double rho(double x) const override;
        double p(double x) const override { return -rho(x) - drhodx(x) / 3.0; }
        double drhodx(double x) const override;
        double ddrhoddx(double x) const override;
        double E2(double x) const; // (H/H0)^2 on the tracker
        double dE2dx(double x) const;
        double Omega_phi() const { return Om; }

      private:
        std::vector<std::shared_ptr<const Species>> others;
        double Om;
        void sums(double x, double & r, double & dr, double & ddr) const;
    };

    class CubicGalileonGravity : public GeneralRelativity {
      public:
        explicit CubicGalileonGravity(double Omega_phi);
        std::string name() const override { return "cubic Galileon"; }
        std::string info() const override;
        void init_background(const BackgroundCosmology & bg) override;

        int n_vars() const override { return 3; }
        enum { PHIX, PHIXX, RPP, N_CACHE }; // phi_x, phi_xx, H0^2 a^2 (rho + p) of the matter
        int cache_size() const override { return N_CACHE; }
        void update_cache(const BackgroundCosmology & bg, const ModeAtX & b, double * cache) const override;
        void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override;
        MetricAtX metric(const ModeAtX & b,
                         const double * cache,
                         const double * y,
                         const Moments & total,
                         double * dydx) const override;
        std::vector<std::string> output_names() const override { return {"delta_phi_gal", "u_gal"}; }
        void outputs(const ModeAtX & b,
                     const double * cache,
                     const double * y,
                     const MetricAtX & m,
                     double * out) const override;

        double xi() const { return xi_; } // H dphi/dt on the tracker [H0^2, field in reduced Planck masses]
        double g3() const { return g3_; }
        double c2() const { return -1.0; }
        // phi_x = dphi/dx = xi / E^2
        double phi_x(double E2) const { return xi_ / E2; }

      private:
        double xi_, g3_;
    };

} // namespace FML::COSMOLOGY

#endif
