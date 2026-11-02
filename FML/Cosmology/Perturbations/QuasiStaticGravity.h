#ifndef FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_H
#define FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_H

// =====================================================================================================================
// QuasiStaticGravity: the quasi-static limit (sub-horizon, time derivatives of the extra fields neglected) of the
// linear perturbations of a theory of gravity, on a Background. It is the one definition of the theory's mu, Sigma
// (and the second-order source) used by
//   the Einstein-Boltzmann solver   PerturbedFieldEquation_MuSigma and its subclasses (HuSawicky, DGP)
//   SimulationCode                  GravityModel_fofr, _DGP, _JBD, _mu (linear growth, LPT, the linear part of the
//                                   force; the N-body side - screening, multigrid - stays in SimulationCode)
// With the comoving density rho Delta of all species (Planck sign of the spatial potential, Phi_P):
//   k^2 Psi           = -4 pi G a^2 mu(k, a)    rho Delta
//   k^2 (Phi_P + Psi) = -8 pi G a^2 Sigma(k, a) rho Delta
// mu = G_eff/G for non-relativistic matter (growth, N-body force), Sigma for the lensing (Weyl) potential; GR is
// mu = Sigma = 1.
//
// Units: x = ln a and k in units of H0/c, koverH0 = c k / H0 (= k [h/Mpc] * 2997.92458; k [1/m] * c / H0_SI), as
// SimulationCode's GeffOverG(a, koverH0).
//
// second_order_source(x, koverH0): the D1^2 source of the 2LPT growth equation relative to the linear one,
//   D2'' + (2 + dlnH/dx) D2' = (3/2) Omega_m(a) mu (D2 - second_order_source D1^2)
// (1 in GR and in theories without a nonlinear term in their field equation; DGP's brane-bending term makes it
// smaller). It is the screening (Vainshtein) term, so a code without screening uses 1.
//
// mu_sigma(x, koverH0) gives mu, Sigma and d(2 Sigma - mu)/dx at fixed k (needed by the Einstein-Boltzmann solver);
// the default derivative is a central difference with step 1e-4 (the theories override it where it is exact).
//
// screening(): how the theory screens its fifth force in nonlinear structures (Screening below): the type (what local
// quantity decides it: the Newtonian potential for chameleon-like theories, the density for Vainshtein, the gradient
// of the potential for k-mouflage) and the fraction of the fifth force that survives as a function of that quantity.
// This is the physics of the screening; applying it to a density field on a grid (SimulationCode, NBody) is done by
// the code that has the grid.
//
// The object keeps a reference to the background: the background must outlive it.
// Theories: QuasiStaticGravity_MuSigma.h (Planck/DES), _HuSawicky.h (f(R)), _DGP.h, _BransDicke.h, _Symmetron.h,
// _MBeta.h (the (m(a), beta(a)) parametrisation of chameleon-like theories).
// =====================================================================================================================

#include <FML/Cosmology/Background/Background.h>

#include <functional>
#include <memory>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    /// What decides the screening: nothing (no screening), the Newtonian potential Phi_N (chameleon, symmetron,
    /// f(R)), the density rho/rho_mean (Vainshtein: DGP, Galileons) or |grad Phi_N| (k-mouflage)
    enum class ScreeningType { None, Potential, Density, Gradient };
    inline std::string screening_type_name(ScreeningType t) {
        switch (t) {
            case ScreeningType::Potential:
                return "potential";
            case ScreeningType::Density:
                return "density";
            case ScreeningType::Gradient:
                return "gradient";
            default:
                return "none";
        }
    }

    /// The screening of a theory's fifth force
    struct Screening {
        ScreeningType type = ScreeningType::None;
        /// The fraction (0..1) of the linear fifth force that survives at x = ln a where the local quantity of the
        /// type is value (Potential: Phi_N in units of c^2; Density: rho/rho_mean; Gradient: |grad Phi_N| in units of
        /// c H0). efficiency scales the strength of the screening approximation (1: standard; a numerical knob of the
        /// codes that use it, not a parameter of the theory)
        std::function<double(double x, double value, double efficiency)> factor;
        /// The threshold of the local quantity where screening sets in (Potential: Phi_crit(a)); empty if none
        std::function<double(double x)> threshold;
        /// A short description (the formula)
        std::string description;
        explicit operator bool() const { return type != ScreeningType::None; }
    };

    class QuasiStaticGravity {
      public:
        explicit QuasiStaticGravity(const BACKGROUND::Background & bg) : bg(bg) {}
        virtual ~QuasiStaticGravity() = default;

        virtual std::string name() const = 0;
        virtual std::string info() const { return name(); }

        /// G_eff/G for non-relativistic matter at x = ln a, k = koverH0 H0/c (koverH0 = 0: the large-scale limit)
        virtual double mu(double x, double koverH0) const = 0;
        /// The lensing potential's coupling (1: as in GR)
        virtual double Sigma(double /*x*/, double /*koverH0*/) const { return 1.0; }
        /// Whether mu or Sigma depends on k
        virtual bool scale_dependent() const { return false; }
        /// The D1^2 source of the 2LPT equation relative to GR (1: none; see the header)
        virtual double second_order_source(double /*x*/, double /*koverH0*/) const { return 1.0; }
        /// How the fifth force is screened (default: not at all)
        virtual Screening screening() const { return {}; }

        struct MuSigma {
            double mu = 1.0, Sigma = 1.0, dg = 0.0;
        };
        /// mu, Sigma and d(2 Sigma - mu)/dx at fixed k
        virtual MuSigma mu_sigma(double x, double koverH0) const {
            const double h = 1e-4;
            MuSigma m;
            m.mu = mu(x, koverH0);
            m.Sigma = Sigma(x, koverH0);
            const double gp = 2.0 * Sigma(x + h, koverH0) - mu(x + h, koverH0);
            const double gm = 2.0 * Sigma(x - h, koverH0) - mu(x - h, koverH0);
            m.dg = (gp - gm) / (2.0 * h);
            return m;
        }

        const BACKGROUND::Background & background() const { return bg; }

      protected:
        const BACKGROUND::Background & bg;
    };

    using QuasiStaticGravityPtr = std::shared_ptr<const QuasiStaticGravity>;

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
