#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_HORNDESKIALPHA_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_HORNDESKIALPHA_H

// =====================================================================================================================
// Horndeski gravity in the alpha parameterization of Bellini & Sawicki (2014), as in hi_class (Zumalacarregui et al.
// 2017), with alpha_T = 0:
//   M2 = M*^2 (effective Planck mass^2 in units of 1/(8 pi G)),  alpha_M = dln M2 / dln a,
//   alpha_B (braiding), alpha_K (kineticity),  functions of time given by an AlphaFunctions object.
// The expansion is that of the background (a ComponentBackground: as hi_class the GR sum of the species, LCDM or w0wa
// dark energy as the effective scalar-field fluid; or any other, e.g. Brans-Dicke rebuilt from its alphas).
//
// Perturbations (Newtonian gauge; derived in Cosmology/EinsteinBoltzmannSolver/python/derive_horndeski.py and
// derive_horndeski_alpha.py for L = F(phi) R/2 + G2(phi, X) - G3(phi, X) box phi, which realizes any alpha_K,
// alpha_B, alpha_M): variables Phi, v = -delta phi / phi' (conformal time) and u = v' + Psi (which removes Psi' from
// the scalar equation). In units H0 = c = kappa = 1, conformal time, H = a'/a, Phi_P = -Phi:
//   Psi    = Phi_P + alpha_M H v - Sig/(M2 k^2)                                           (traceless ij)
//   Phi_P' from the 00 equation,  u' from the scalar equation (with Phi_P'' and Psi' eliminated),
//   0i:    M2 (alpha_B H u - 2 H Psi - 2 Phi_P') + Q + v (alpha_B H^2 M2 - Rx) = 0      (not used; a test)
// with the matter background R = a^2 (rho + p) (all species but dark energy) and Rx = 2 M2 (H^2 - H') - R
// (= (M2 - 1) R + M2 a^2 (rho + p)_DE for a GR background). Q = a^2 (rho+p) theta/k^2 and Sig = (3/2) a^2 (rho+p) sigma
// of the matter. No-ghost and gradient stability: D = alpha_K + 3 alpha_B^2 / 2 > 0, c_s^2 > 0, with
//   D c_s^2 = (2 - alpha_B) [alpha_B/2 + alpha_M - Hdot/H^2] - (rho + p)/(H^2 M2) + alphadot_B / H   (cosmic time)
//
// HorndeskiAlphaBackground: the alphas on a background (the per-x State, and the stability check of hi_class from
// deep in the radiation era to a = e: M2 > 0, D >= 0, c_s^2 >= 0 and c_s^2 <= 1e5; D = 0 with alpha_M != 0 is always
// rejected), made once per model. PerturbedFieldEquation_HorndeskiAlpha: the perturbations, three variables (Phi, v,
// u), initial conditions v = u = 0 (the alphas vanish early in the standard parameterizations), outputs v_horndeski,
// u_horndeski; tensors M*^2, alpha_M, alpha_T = 0.
// =====================================================================================================================

#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Perturbations/PerturbationTypes.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation.h>
#include <FML/Spline/Spline.h>

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    /// The alphas at one time; derivatives with respect to x = ln a. Rx (see the header) is optional (NaN: computed from
    /// the expansion, which loses precision when Rx << H^2, so a background that is not the GR sum should give it).
    /// dM2 = M2 - 1 (separately, for precision when M2 is close to 1).
    struct Alphas {
        double M2 = 1.0, dM2 = 0.0, aK = 0.0, aB = 0.0, aM = 0.0, daKdx = 0.0, daBdx = 0.0;
        double Rx = std::numeric_limits<double>::quiet_NaN();
    };

    class AlphaFunctions {
      public:
        virtual ~AlphaFunctions() = default;
        virtual std::string info() const = 0;
        /// Called once with the background before the alphas are evaluated
        virtual void init(const BACKGROUND::ComponentBackground & /*bg*/) {}
        virtual Alphas operator()(double x, const BACKGROUND::ComponentBackground & bg) const = 0;
    };

    /// hi_class parameterizations (alpha_X = c_X f(a)), with M2 from integrating alpha_M from M2_ini at early times:
    ///   ProptoOmega:  f = Omega_DE(a) (dark-energy component / GR sum);  M2 = M2_ini exp(c_M int Omega_DE dx)
    ///   ProptoScale:  f = a;                                               M2 = M2_ini exp(c_M a)
    ///   Constant:     f = 1;                                               M2 = M2_ini exp(c_M (x - ln a_ini)),
    ///   a_ini = 1e-14
    class StandardAlphas : public AlphaFunctions {
      public:
        enum class Type { ProptoOmega = 0, ProptoScale = 1, Constant = 2 };
        StandardAlphas(Type type, double cK, double cB, double cM, double M2_ini)
            : type(type), cK(cK), cB(cB), cM(cM), M2_ini(M2_ini) {
            if (not(M2_ini > 0.0))
                throw std::invalid_argument("horndeski: need M2_ini > 0");
        }
        std::string info() const override {
            static const char * names[] = {"propto_omega", "propto_scale", "constant_alphas"};
            std::ostringstream os;
            os << names[int(type)] << " (c_K = " << cK << ", c_B = " << cB << ", c_M = " << cM
               << ", M*^2_ini = " << M2_ini << ")";
            return os.str();
        }
        void init(const BACKGROUND::ComponentBackground & bg) override {
            if (type != Type::ProptoOmega)
                return;
            // cumulative Simpson integral of Omega_DE (negligible below x_lo)
            const int n = 28001;
            std::vector<double> xs(n), I(n, 0.0), f(n);
            for (int i = 0; i < n; i++)
                xs[i] = x_lo + (x_hi - x_lo) * i / double(n - 1);
            for (int i = 0; i < n; i++) {
                double d;
                omega_de(xs[i], bg, f[i], d);
            }
            const double h = xs[1] - xs[0];
            for (int i = 1; i < n; i++) {
                double fm, d;
                omega_de(0.5 * (xs[i - 1] + xs[i]), bg, fm, d);
                I[i] = I[i - 1] + h / 6.0 * (f[i - 1] + 4.0 * fm + f[i]);
            }
            I_spline.create(xs, I, "horndeski int Omega_DE");
        }
        Alphas operator()(double x, const BACKGROUND::ComponentBackground & bg) const override {
            Alphas al;
            double f = 1.0, df = 0.0, lnM = 0.0;
            switch (type) {
                case Type::ProptoOmega:
                    omega_de(x, bg, f, df);
                    lnM = cM * (x < x_lo ? 0.0 : I_spline(std::min(x, x_hi)));
                    break;
                case Type::ProptoScale:
                    f = df = std::exp(x);
                    lnM = cM * f;
                    break;
                case Type::Constant:
                    lnM = cM * (x + std::log(1e14));
                    break;
            }
            al.aK = cK * f;
            al.aB = cB * f;
            al.aM = cM * f;
            al.daKdx = cK * df;
            al.daBdx = cB * df;
            al.M2 = M2_ini * std::exp(lnM);
            al.dM2 = M2_ini * std::expm1(lnM) + (M2_ini - 1.0);
            return al;
        }

      private:
        Type type;
        double cK, cB, cM, M2_ini;
        double x_lo = -25.0, x_hi = 3.0;
        FML::INTERPOLATION::SPLINE::GSLSpline I_spline;

        /// Omega_DE = rho_DE / E^2 (GR sum) and its x-derivative
        static void omega_de(double x, const BACKGROUND::ComponentBackground & bg, double & Om, double & dOm) {
            using BACKGROUND::Component;
            const double r = bg.rho(Component::DarkEnergy, x), dr = bg.drhodx(Component::DarkEnergy, x);
            const double E2 = bg.rho_total(x) + bg.rho_k(x), dE2 = bg.drho_total(x) + bg.drho_k(x);
            Om = r / E2;
            dOm = (dr - Om * dE2) / E2;
        }
    };

    /// Alphas from a function (tests, models that provide their own)
    class FunctionAlphas : public AlphaFunctions {
      public:
        using Fn = std::function<Alphas(double x, const BACKGROUND::ComponentBackground & bg)>;
        FunctionAlphas(std::string name, Fn f) : label(std::move(name)), fn(std::move(f)) {}
        std::string info() const override { return label; }
        Alphas operator()(double x, const BACKGROUND::ComponentBackground & bg) const override { return fn(x, bg); }

      private:
        std::string label;
        Fn fn;
    };

    /// The alphas on a background: the State the perturbations need, and the stability check (see the header)
    class HorndeskiAlphaBackground {
      public:
        /// Background quantities used by the perturbations (H0 = kappa = 1, conformal time)
        struct State {
            Alphas al;
            double a, H, Hd, R, Rd, Rx, aKd, aBd;
            double D() const { return al.aK + 1.5 * al.aB * al.aB; }
            /// D c_s^2 = (2 - aB)[aB/2 + aM - Hdot/H^2] - (rho+p)/(H^2 M2) + alphadot_B/H, -Hdot/H^2 = (R + Rx)/(2 M2
            /// H^2): the matter terms combine to [(2 - aB) Rx - aB R]/(2 M2 H^2)
            double cs2_num() const {
                const double B = al.aB;
                return (2.0 - B) * (B / 2.0 + al.aM) + ((2.0 - B) * Rx - B * R) / (2.0 * al.M2 * H * H) + aBd / H;
            }
        };

        /// alphas: initialized here with the background (bg must outlive this object); check_stability: as hi_class
        HorndeskiAlphaBackground(std::shared_ptr<AlphaFunctions> alphas,
                                 const BACKGROUND::ComponentBackground & bg,
                                 bool check_stability = true)
            : alpha(std::move(alphas)), bg(bg), check(check_stability),
              gr_sum(bg.field_equations_as<BACKGROUND::GeneralRelativity>() != nullptr) {
            if (not alpha)
                throw std::invalid_argument(name() + ": no alpha functions");
            if (bg.parameters().Omega_k != 0.0)
                throw std::invalid_argument(name() + ": only implemented for a flat universe");
            for (const auto & s : bg.species())
                (s->component() == BACKGROUND::Component::DarkEnergy ? dark_energy : matter).push_back(s.get());
            alpha->init(bg);
            stability();
        }
        std::string name() const { return "Horndeski (alphas)"; }
        std::string info() const { return "Horndeski, alpha_T = 0: " + alpha->info() + stability_info; }
        const AlphaFunctions & alphas() const { return *alpha; }
        /// All alphas vanish on [-15, 1] (GR)
        bool trivial() const { return is_trivial; }

        State state(double x) const {
            State s;
            s.al = (*alpha)(x, bg);
            s.a = std::exp(x);
            const double E = bg.E(x), L = bg.dlogEdx(x);
            s.H = s.a * E;
            s.Hd = s.H * s.H * (1.0 + L);
            // matter (all species but dark energy) and dark energy: R = 3 a^2 (rho + p) = -a^2 rho_x and
            // R_x = -a^2 (2 rho_x + rho_xx) for conserved species (rho in units of rho_c0)
            double drho = 0.0, ddrho = 0.0, drho_de = 0.0;
            for (const auto * sp : dark_energy)
                drho_de += sp->drhodx(x);
            for (const auto * sp : matter) {
                drho += sp->drhodx(x);
                ddrho += sp->d2rhodx2(x);
            }
            const double a2 = s.a * s.a;
            s.R = -a2 * drho;
            s.Rd = -s.H * a2 * (2.0 * drho + ddrho);
            if (not std::isnan(s.al.Rx))
                s.Rx = s.al.Rx;
            else if (gr_sum)
                s.Rx = s.al.dM2 * s.R - s.al.M2 * a2 * drho_de;
            else
                s.Rx = -2.0 * s.al.M2 * s.H * s.H * L - s.R;
            s.aKd = s.H * s.al.daKdx;
            s.aBd = s.H * s.al.daBdx;
            return s;
        }

      private:
        std::shared_ptr<AlphaFunctions> alpha;
        const BACKGROUND::ComponentBackground & bg;
        bool check, gr_sum;
        std::vector<const BACKGROUND::Species *> matter, dark_energy;
        bool is_trivial = false;
        std::string stability_info;

        void stability() {
            constexpr double max_cs2 = 1e5;
            is_trivial = true;
            double minD = 1e300, mincs2 = 1e300, maxcs2 = 0.0, minM2 = 1e300, x_cs2 = 2.0, x_D = 2.0;
            double x_D0 = 2.0; // first x where D = 0 while alpha_M != 0
            for (double x = -15.0; x <= 1.0 + 1e-12; x += 0.005) {
                const auto s = state(x);
                if (s.al.aK != 0.0 or s.al.aB != 0.0 or s.al.aM != 0.0)
                    is_trivial = false;
                const double D = s.D();
                if (D == 0.0 and s.al.aM != 0.0)
                    x_D0 = std::min(x_D0, x);
                if (not std::isfinite(D) or not std::isfinite(s.cs2_num()) or not std::isfinite(s.al.M2) or
                    not std::isfinite(s.Rx))
                    throw std::runtime_error(name() + ": non-finite alphas or background at x = " + std::to_string(x));
                minM2 = std::min(minM2, s.al.M2);
                minD = std::min(minD, D);
                if (D < 0.0)
                    x_D = std::min(x_D, x);
                if (D > 0.0) {
                    const double cs2 = s.cs2_num() / D;
                    mincs2 = std::min(mincs2, cs2);
                    maxcs2 = std::max(maxcs2, cs2);
                    if (cs2 < 0.0)
                        x_cs2 = std::min(x_cs2, x);
                }
            }
            std::ostringstream os;
            if (is_trivial) {
                os << "; all alphas vanish (GR)";
            } else {
                // D = 0 with alpha_M != 0 (alpha_K = alpha_B = 0): the scalar has no kinetic term and the perturbation
                // equations drop it, which is not the limit of small alpha_K (the scalar is then quasi-static with
                // c_s^2 -> infinity). Not a stability question: rejected also without the stability check.
                if (x_D0 <= 1.0) {
                    std::ostringstream e;
                    e << name() << ": alpha_K = alpha_B = 0 with alpha_M != 0 (from a = " << std::exp(x_D0)
                      << "): the scalar perturbation is not dynamical (D = alpha_K + 3 alpha_B^2/2 = 0); use alpha_K > "
                         "0";
                    throw std::invalid_argument(e.str());
                }
                os << "; min D = " << minD << ", min c_s^2 = " << mincs2 << ", max c_s^2 = " << maxcs2;
                if (check) {
                    if (not(minM2 > 0.0))
                        throw std::invalid_argument(name() + ": M*^2 <= 0");
                    std::ostringstream e;
                    e << name() << ": ";
                    if (minD < 0.0) {
                        e << "ghost instability (D = alpha_K + 3 alpha_B^2/2 < 0 from a = " << std::exp(x_D) << ")";
                        throw std::invalid_argument(e.str());
                    }
                    if (mincs2 < 0.0) {
                        e << "gradient instability (c_s^2 < 0 from a = " << std::exp(x_cs2)
                          << ", min c_s^2 = " << mincs2 << ")";
                        throw std::invalid_argument(e.str());
                    }
                    // alpha_K << alpha_B^2 at early times: D ~ alpha_B^2 while D c_s^2 ~ alpha_B, so c_s^2 ~ 1/alpha_B
                    // diverges and the scalar equation is too stiff to integrate
                    if (maxcs2 > max_cs2) {
                        e << "c_s^2 up to " << maxcs2 << " > " << max_cs2
                          << ": the scalar perturbation is too stiff to evolve (alpha_K << alpha_B^2 at early times); "
                             "use a larger alpha_K";
                        throw std::invalid_argument(e.str());
                    }
                }
            }
            stability_info = os.str();
        }
    };

    class PerturbedFieldEquation_HorndeskiAlpha : public PerturbedFieldEquation {
      public:
        /// hd: the alphas on the background of ctx
        PerturbedFieldEquation_HorndeskiAlpha(std::shared_ptr<const HorndeskiAlphaBackground> hd,
                                              const PerturbationContext & ctx)
            : PerturbedFieldEquation(ctx), hd(std::move(hd)) {
            if (not this->hd)
                throw std::invalid_argument("[PerturbedFieldEquation_HorndeskiAlpha] No alphas");
        }
        std::string name() const override { return hd->name(); }
        std::string info() const override { return hd->info(); }
        const HorndeskiAlphaBackground & alpha_background() const { return *hd; }

        int n_vars() const override { return 3; }
        enum { M2_, AK, AB, AM, AKD, ABD, R_, RD, RX, N_CACHE };
        int cache_size() const override { return N_CACHE; }
        void update_cache(const ModeAtX & b, double * cache) const override {
            const auto s = hd->state(b.x);
            cache[M2_] = s.al.M2;
            cache[AK] = s.al.aK;
            cache[AB] = s.al.aB;
            cache[AM] = s.al.aM;
            cache[AKD] = s.aKd;
            cache[ABD] = s.aBd;
            cache[R_] = s.R;
            cache[RD] = s.Rd;
            cache[RX] = s.Rx;
        }
        void initial_conditions(const ModeAtX &, const InitialData & ic, double * y) const override {
            y[0] = ic.Phi;
            y[1] = 0.0; // the alphas (and the scalar's effect) vanish early for the standard parameterizations
            y[2] = 0.0;
        }

        /// Equations from python/derive_horndeski_alpha.py, units H0 = c = kappa = 1, conformal time: H = Hp/H0,
        /// k = ck/H0, P = Phi_P = -Phi, P1 = Phi_P', U = u = v' + Psi; matter drho = a^2 delta rho, dp = a^2 delta p,
        /// Sig = (3/2) a^2 (rho+p) sigma (3 times the Moments / H0^2)
        MetricAtX metric(const ModeAtX & b,
                         const double * cache,
                         const double * y,
                         const Moments & total,
                         double * dydx) const override {
            const double a = b.a, H0sq = b.H0a2 / (a * a), H0 = std::sqrt(H0sq);
            const double H = b.Hp / H0, k = b.ckH * H, k2 = k * k;
            const double drho = 3.0 * total.drho / H0sq, dp = 3.0 * total.dp / H0sq, Sig = 3.0 * total.shear / H0sq;
            const double M2 = cache[M2_], aK = cache[AK], aB = cache[AB], aM = cache[AM], aKd = cache[AKD],
                         aBd = cache[ABD];
            const double R = cache[R_], Rd = cache[RD], Rx = cache[RX];
            const double P = -y[0], v = y[1], U = y[2];
            const double Psi = P + aM * H * v - Sig / (M2 * k2);
            const double H2 = H * H, H3 = H2 * H, H4 = H2 * H2, M22 = M2 * M2, aB2 = aB * aB;

            // 00 Einstein equation
            const double P1 = (-H3 * M2 * aB * v - H3 * M2 * aK * v / 3.0 - H2 * M2 * Psi * aB + 2.0 * H2 * M2 * Psi -
                               H2 * M2 * U * aB - H2 * M2 * U * aK / 3.0 - H * M2 * aB * k2 * v / 3.0 + H * Rx * v -
                               0.5 * H * aB * v * (R + Rx) + 2.0 / 3.0 * M2 * P * k2 + drho / 3.0) /
                              (H * M2 * (aB - 2.0));
            // scalar field equation
            double U1 = 0.0;
            const double D = aK + 1.5 * aB2;
            if (D != 0.0) {
                const double cU = -6.0 * H3 * M22 * aB2 * aM - 24.0 * H3 * M22 * aB2 - 4.0 * H3 * M22 * aK * aM -
                                  16.0 * H3 * M22 * aK - 6.0 * H2 * M22 * aB * aBd - 4.0 * H2 * M22 * aKd +
                                  3.0 * H * M2 * R * aB2 + 4.0 * H * M2 * R * aK + 3.0 * H * M2 * Rx * aB2 +
                                  6.0 * H * M2 * Rx * aB + 4.0 * H * M2 * Rx * aK;
                const double cv =
                    -6.0 * H4 * M22 * aB2 * aM - 24.0 * H4 * M22 * aB2 - 4.0 * H4 * M22 * aK * aM -
                    16.0 * H4 * M22 * aK - 6.0 * H3 * M22 * aB * aBd - 4.0 * H3 * M22 * aKd + 6.0 * H2 * M2 * R * aB2 -
                    6.0 * H2 * M2 * R * aB + 6.0 * H2 * M2 * R * aK + 6.0 * H2 * M2 * Rx * aB2 +
                    6.0 * H2 * M2 * Rx * aB + 6.0 * H2 * M2 * Rx * aK - 6.0 * H * M2 * R * aBd - 6.0 * H * M2 * Rd * aB -
                    6.0 * H * M2 * Rx * aBd + 3.0 * R * R * aB + 6.0 * R * Rx * aB - 6.0 * R * Rx + 3.0 * Rx * Rx * aB -
                    6.0 * Rx * Rx +
                    k2 * (-4.0 * H2 * M22 * aB * aM - 4.0 * H2 * M22 * aB - 8.0 * H2 * M22 * aM - 4.0 * H * M22 * aBd +
                          2.0 * M2 * R * aB + 2.0 * M2 * Rx * aB - 4.0 * M2 * Rx);
                const double cPsi = 6.0 * H3 * M22 * aB2 + 4.0 * H3 * M22 * aK - 12.0 * H2 * M22 * aBd +
                                    4.0 * H * M22 * aB * k2 + 6.0 * H * M2 * R * aB + 6.0 * H * M2 * Rx * aB -
                                    12.0 * H * M2 * Rx;
                const double cP = 8.0 * H * M22 * aM * k2;
                const double cP1 = -12.0 * H * M22 * aBd + 6.0 * M2 * R * aB + 6.0 * M2 * Rx * aB - 12.0 * M2 * Rx;
                const double num = cU * U + cv * v + cPsi * Psi + cP * P + cP1 * P1 - 6.0 * H * M2 * aB * dp +
                                   4.0 * H * M2 * aB * Sig;
                U1 = num / (4.0 * H2 * M22 * D); // denominator 2 H^2 M2^2 (3 aB^2 + 2 aK)
            }

            MetricAtX m;
            m.Phi = y[0];
            m.Psi = Psi;
            dydx[0] = -P1 / H;                        // d/dx = (1/H) d/dtau
            dydx[1] = D != 0.0 ? (U - Psi) / H : 0.0; // D = 0: no scalar perturbations (all alphas vanish)
            dydx[2] = U1 / H;
            m.dPhi = dydx[0];
            return m;
        }
        /// v = -delta phi / phi' [1/H0] and u = v' + Psi
        std::vector<std::string> output_names() const override { return {"v_horndeski", "u_horndeski"}; }
        void outputs(const ModeAtX &, const double *, const double * y, const MetricAtX &, double * out) const override {
            out[0] = y[1];
            out[1] = y[2];
        }
        /// Tensors: M*^2 and alpha_M of the alphas (alpha_T = 0)
        TensorCoefficients tensor(double x) const override {
            const Alphas a = hd->alphas()(x, ctx.background);
            return {a.M2, a.aM, 0.0};
        }

      private:
        std::shared_ptr<const HorndeskiAlphaBackground> hd;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
