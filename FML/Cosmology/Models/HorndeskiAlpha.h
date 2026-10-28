#ifndef FML_COSMOLOGY_MODELS_HORNDESKIALPHA_H
#define FML_COSMOLOGY_MODELS_HORNDESKIALPHA_H

// =============================================================================
// Horndeski gravity in the alpha parameterization of Bellini & Sawicki (2014), as in hi_class
// (Zumalacarregui et al. 2017), with alpha_T = 0 (gravitational waves at the speed of light, GW170817):
//   M2 = M*^2 (effective Planck mass^2 in units of 1/(8 pi G)),  alpha_M = dln M2 / dln a,
//   alpha_B (braiding), alpha_K (kineticity),  all functions of time given by an AlphaFunctions object.
// The background expansion is given separately: by default the sum of the species as in GR (LCDM, or w0-wa
// dark energy as the effective scalar-field fluid), as in hi_class; it can also be taken from another Gravity
// (used in the tests to rebuild Brans-Dicke from its alphas).
//
// Perturbations (Newtonian gauge; derived in python/derive_horndeski.py and python/derive_horndeski_alpha.py for
// the family L = F(phi) R/2 + G2(phi, X) - G3(phi, X) box phi, which realizes any alpha_K, alpha_B, alpha_M):
// variables Phi, v = -delta phi / phi' (conformal time) and u = v' + Psi (which removes Psi' from the scalar
// equation). In units H0 = c = kappa = 1, conformal time, H = a'/a, Phi_P = -Phi:
//   Psi    = Phi_P + alpha_M H v - Sig/(M2 k^2)                                           (traceless ij)
//   Phi_P' from the 00 equation,  u' from the scalar equation (with Phi_P'' and Psi' eliminated),
//   0i:    M2 (alpha_B H u - 2 H Psi - 2 Phi_P') + Q + v (alpha_B H^2 M2 - Rx) = 0      (not used; a test)
// with the matter background R = a^2 (rho + p) (all species but dark energy) and
// Rx = 2 M2 (H^2 - H') - R ( = (M2 - 1) R + M2 a^2 (rho + p)_DE for the GR-sum background).
// Q = a^2 (rho+p) theta/k^2 and Sig = (3/2) a^2 (rho+p) sigma of the matter.
// No-ghost and gradient stability:  D = alpha_K + 3 alpha_B^2 / 2 > 0,  c_s^2 > 0, with
//   D c_s^2 = (2 - alpha_B) [alpha_B/2 + alpha_M - Hdot/H^2] - (rho + p)/(H^2 M2) + alphadot_B / H   (cosmic time)
// =============================================================================

#include <FML/Cosmology/Background/Gravity.h>
#include <FML/Cosmology/Common.h>

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace FML::COSMOLOGY {

    // The alphas at one time; derivatives with respect to x = ln a. Rx (see above) is optional (NaN: computed from
    // the expansion, which loses precision when Rx << H^2, so a background that is not the GR sum should give it).
    // dM2 = M2 - 1 (separately, for precision when M2 is close to 1).
    struct Alphas {
        double M2 = 1.0, dM2 = 0.0, aK = 0.0, aB = 0.0, aM = 0.0, daKdx = 0.0, daBdx = 0.0;
        double Rx = std::numeric_limits<double>::quiet_NaN();
    };

    class AlphaFunctions {
      public:
        virtual ~AlphaFunctions() = default;
        virtual std::string info() const = 0;
        virtual void init(const BackgroundCosmology & /*bg*/) {}
        virtual Alphas operator()(double x, const BackgroundCosmology & bg) const = 0;
    };

    // hi_class parameterizations (alpha_X = c_X f(a)), with M2 from integrating alpha_M from M2_ini at early times:
    //   ProptoOmega:  f = Omega_DE(a) (fraction of the dark-energy species);  M2 = M2_ini exp(c_M int Omega_DE dx)
    //   ProptoScale:  f = a;                                                   M2 = M2_ini exp(c_M a)
    //   Constant:     f = 1;                                                   M2 = M2_ini exp(c_M (x - ln a_ini)),
    //   a_ini = 1e-14
    class StandardAlphas : public AlphaFunctions {
      public:
        enum class Type { ProptoOmega = 0, ProptoScale = 1, Constant = 2 };
        StandardAlphas(Type type, double cK, double cB, double cM, double M2_ini);
        std::string info() const override;
        void init(const BackgroundCosmology & bg) override;
        Alphas operator()(double x, const BackgroundCosmology & bg) const override;

      private:
        Type type;
        double cK, cB, cM, M2_ini;
        double x_lo = -25.0, x_hi = 3.0;
        Spline1D I_spline; // int_{-inf}^x Omega_DE dx (ProptoOmega)
        void omega_de(double x, const BackgroundCosmology & bg, double & Om, double & dOm) const;
    };

    // Alphas from a function (tests, models that provide their own)
    class FunctionAlphas : public AlphaFunctions {
      public:
        using Fn = std::function<Alphas(double x, const BackgroundCosmology & bg)>;
        FunctionAlphas(std::string name, Fn f) : name_(std::move(name)), f_(std::move(f)) {}
        std::string info() const override { return name_; }
        Alphas operator()(double x, const BackgroundCosmology & bg) const override { return f_(x, bg); }

      private:
        std::string name_;
        Fn f_;
    };

    class HorndeskiAlphaGravity : public Gravity {
      public:
        // background: nullptr for the GR sum of the species (flat)
        HorndeskiAlphaGravity(std::shared_ptr<AlphaFunctions> alphas,
                              std::shared_ptr<Gravity> background = nullptr,
                              bool check_stability = true);
        std::string name() const override { return "Horndeski (alphas)"; }
        std::string info() const override;
        void init_background(const BackgroundCosmology & bg) override;

        double E2(double x, const BackgroundCosmology & bg) const override;
        double dE2dx(double x, const BackgroundCosmology & bg) const override;
        double ddE2ddx(double x, const BackgroundCosmology & bg) const override;

        // Background quantities used by the perturbations (H0 = kappa = 1, conformal time)
        struct State {
            Alphas al;
            double a, H, Hd, R, Rd, Rx, aKd, aBd;
            double D() const { return al.aK + 1.5 * al.aB * al.aB; }
            double cs2_num() const; // D c_s^2
        };
        State state(double x, const BackgroundCosmology & bg) const;
        const AlphaFunctions & alphas() const { return *alpha; }

        int n_vars() const override { return 3; }
        enum { M2_, AK, AB, AM, AKD, ABD, R_, RD, RX, N_CACHE };
        int cache_size() const override { return N_CACHE; }
        void update_cache(const BackgroundCosmology & bg, const ModeAtX & b, double * cache) const override;
        void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override;
        MetricAtX metric(const ModeAtX & b,
                         const double * cache,
                         const double * y,
                         const Moments & total,
                         double * dydx) const override;
        // v = -delta phi / phi' [1/H0] and u = v' + Psi
        std::vector<std::string> output_names() const override { return {"v_horndeski", "u_horndeski"}; }
        // Tensors: M*^2 and alpha_M of the alphas (alpha_T = 0 in this class)
        TensorCoefficients tensor(double x, const BackgroundCosmology & bg) const override {
            const Alphas a = (*alpha)(x, bg);
            return {a.M2, a.aM, 0.0};
        }
        void outputs(const ModeAtX & b,
                     const double * cache,
                     const double * y,
                     const MetricAtX & m,
                     double * out) const override;

      private:
        std::shared_ptr<AlphaFunctions> alpha;
        std::shared_ptr<Gravity> background;
        bool check;
        bool trivial = false; // all alphas vanish: GR, the scalar perturbations are not evolved
        std::string stability_info;
    };

} // namespace FML::COSMOLOGY

#endif
