#ifndef FML_COSMOLOGY_BACKGROUND_FIELDEQUATION_DGP_H
#define FML_COSMOLOGY_BACKGROUND_FIELDEQUATION_DGP_H

// =====================================================================================================================
// DGP braneworld (Dvali, Gabadadze & Porrati 2000), flat, crossover scale r_c, Omega_rc = c^2 / (4 r_c^2 H0^2):
//   H^2 - eps H c / r_c = 8 pi G rho / 3   ->   E = eps sqrt(Omega_rc) + sqrt(Omega_rc + rho / rho_c0)
//   eps = +1 self-accelerating branch (sDGP), eps = -1 normal branch (nDGP, needs dark energy)
// With lcdm_background the expansion is that of GR (nDGP with an LCDM background, Schmidt 2009).
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquations.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Background/Species.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::BACKGROUND {

    class DGPFieldEquations : public FieldEquations {
      public:
        DGPFieldEquations(double Omega_rc, int branch, bool lcdm_background = false)
            : Omega_rc(Omega_rc), sq(std::sqrt(Omega_rc)), branch(branch), lcdm_bg(lcdm_background) {
            if (not(Omega_rc > 0.0) or not std::isfinite(Omega_rc))
                throw std::invalid_argument("[DGP] Omega_rc must be positive");
            if (branch != 1 and branch != -1)
                throw std::invalid_argument("[DGP] branch must be +1 (self-accelerating) or -1 (normal)");
        }
        std::string name() const override { return branch > 0 ? "sdgp" : "ndgp"; }
        std::string info() const override {
            std::ostringstream s;
            s << (branch > 0 ? "self-accelerating" : "normal-branch") << " DGP, Omega_rc = " << Omega_rc
              << (lcdm_bg ? " (LCDM background)" : "");
            return s.str();
        }
        void init(const ComponentBackground & bg) override {
            if (bg.parameters().Omega_k != 0.0)
                throw std::invalid_argument("[DGP] only implemented for a flat universe");
        }
        double E2(double x, const ComponentBackground & bg) const override {
            if (lcdm_bg)
                return bg.rho_total(x);
            const double E = branch * sq + std::sqrt(Omega_rc + bg.rho_total(x));
            return E * E;
        }
        double dE2dx(double x, const ComponentBackground & bg) const override {
            if (lcdm_bg)
                return bg.drho_total(x);
            const double root = std::sqrt(Omega_rc + bg.rho_total(x));
            return (branch * sq + root) * bg.drho_total(x) / root;
        }
        /// Omega_Lambda that gives E(0) = 1, given the density of the other species today
        static double OmegaLambda(double Omega_rc, int branch, double rho_other0) {
            return 1.0 - 2.0 * branch * std::sqrt(Omega_rc) - rho_other0;
        }
        double get_Omega_rc() const { return Omega_rc; }
        int get_branch() const { return branch; }
        bool get_lcdm_background() const { return lcdm_bg; }

      private:
        double Omega_rc, sq;
        int branch;
        bool lcdm_bg;
    };

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
