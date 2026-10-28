#ifndef FML_COSMOLOGY_MODELS_SCALARFIELD_H
#define FML_COSMOLOGY_MODELS_SCALARFIELD_H

// =============================================================================
// A minimally coupled scalar field phi (units of the reduced Planck mass) with a potential
//   V(phi) = 3 H0^2 M_pl^2 v(phi)
// given by a derived class (quintessence, early dark energy). The field starts at rest at phi_ini at x_ini
// (deep in the radiation era, where Hubble friction freezes it).
//
// Background (GR, x = ln a, phi_x = dphi/dx), with rho_o the other species, curvature and an optional
// cosmological constant Omega_L kept by the field (used when the field's own density enters the closure):
//   rho_phi = E^2 phi_x^2/6 + v,   p_phi = E^2 phi_x^2/6 - v     [units of rho_c0]
//   E^2 = (rho_o + Omega_L + v) / (1 - phi_x^2/6)
//   phi_xx = -(3 + dlnE/dx) phi_x - 3 v_phi / E^2,   dlnE/dx = (rho_o' - E^2 phi_x^2) / (2 E^2)
//
// Perturbations (Newtonian gauge; Mukhanov's Klein-Gordon equation): with dphi the field perturbation and
// w = dphi_x - phi_x Psi (which removes Psi' from the equation),
//   dphi_x = w + phi_x Psi
//   w_x    = -(2 + Hp'/Hp) w - (ck/Hp)^2 dphi - 3 v_phiphi dphi / E^2 - 3 phi_x Phi' - 3 v_phi Psi / E^2
// Moments (times H0^2 a^2 / rho_c0):
//   delta rho = Hp^2 phi_x w / 3 + H0^2 a^2 v_phi dphi,   delta p = Hp^2 phi_x w / 3 - H0^2 a^2 v_phi dphi
//   flux = (ck/Hp) Hp^2 phi_x dphi / 3,   no anisotropic stress
// Initial conditions dphi = w = 0 (the field is frozen; the adiabatic mode is approached quickly).
// =============================================================================

#include <FML/Cosmology/Background/Species.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <memory>
#include <vector>

namespace FML::COSMOLOGY {

    class ScalarFieldSpecies : public Species {
      public:
        ScalarFieldSpecies(const std::vector<std::shared_ptr<const Species>> & others,
                           double OmegaK,
                           double OmegaL = 0.0);
        Role role() const override { return Role::DarkEnergy; }

        double rho(double x) const override;
        double p(double x) const override;
        double drhodx(double x) const override;
        double ddrhoddx(double x) const override;
        std::unique_ptr<SpeciesPerturbations> perturbations(const Perturbations & pert) const override;

        // Potential (units of 3 H0^2 M_pl^2) and its derivatives
        virtual double v(double phi) const = 0;
        virtual double v_phi(double phi) const = 0;
        virtual double v_phiphi(double phi) const = 0;

        double phi(double x) const;
        double dphidx(double x) const;
        double ddphidx(double x) const { return ddphi(x, phi(x), dphidx(x)); }
        double E2(double x) const; // (H/H0)^2 of the model (as summed by GR)
        double OmegaL() const { return OmegaL_; }

      protected:
        std::vector<std::shared_ptr<const Species>> others;
        double OmegaK, OmegaL_;
        double x_ini = -25.0, x_fin = 10.0, phi_ini_ = 0.0;
        Spline1D phi_spline, dphi_spline;
        bool perturbed = true; // false: no perturbations (a cosmological constant)

        double rho_others(double x) const;
        double drho_others(double x) const;
        double ddrho_others(double x) const;
        double E2(double x, double phi, double dphi) const;
        double ddphi(double x, double phi, double dphi) const; // phi_xx from the Klein-Gordon equation
        // Integrate from rest at phi_ini on n points in [x_ini, x_end] and store the splines (x_fin = x_end)
        void solve(double phi_ini, int n, double x_end, double abserr = 1e-14, double relerr = 1e-12);
    };

} // namespace FML::COSMOLOGY

#endif
