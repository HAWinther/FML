#include <FML/Cosmology/Models/HorndeskiAlpha.h>
#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    namespace {
        bool is_de(const Species & s) { return s.role() == Species::Role::DarkEnergy; }
    } // namespace

    // =============================================================================
    // hi_class parameterizations
    // =============================================================================

    StandardAlphas::StandardAlphas(Type t, double cK_, double cB_, double cM_, double M2_ini_)
        : type(t), cK(cK_), cB(cB_), cM(cM_), M2_ini(M2_ini_) {
        if (!(M2_ini > 0.0))
            throw std::invalid_argument("horndeski: need M2_ini > 0");
    }

    std::string StandardAlphas::info() const {
        static const char * names[] = {"propto_omega", "propto_scale", "constant_alphas"};
        std::ostringstream os;
        os << names[int(type)] << " (c_K = " << cK << ", c_B = " << cB << ", c_M = " << cM << ", M*^2_ini = " << M2_ini
           << ")";
        return os.str();
    }

    // Omega_DE = rho_DE / E^2 (GR sum) and its x-derivative
    void StandardAlphas::omega_de(double x, const BackgroundCosmology & bg, double & Om, double & dOm) const {
        double r = 0.0, dr = 0.0;
        for (const auto & s : bg.species())
            if (is_de(*s)) {
                r += s->rho(x);
                dr += s->drhodx(x);
            }
        const double E2 = bg.rho_species(x) + bg.rhoK(x), dE2 = bg.drho_species(x) + bg.drhoKdx(x);
        Om = r / E2;
        dOm = (dr - Om * dE2) / E2;
    }

    void StandardAlphas::init(const BackgroundCosmology & bg) {
        if (type != Type::ProptoOmega)
            return;
        // cumulative Simpson integral of Omega_DE (negligible below x_lo)
        const int n = 28001;
        const auto xs = Utils::linspace(x_lo, x_hi, n);
        std::vector<double> I(n, 0.0), f(n);
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

    Alphas StandardAlphas::operator()(double x, const BackgroundCosmology & bg) const {
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

    // =============================================================================
    // Gravity
    // =============================================================================

    HorndeskiAlphaGravity::HorndeskiAlphaGravity(std::shared_ptr<AlphaFunctions> a,
                                                 std::shared_ptr<Gravity> bgrav,
                                                 bool check_stability)
        : alpha(std::move(a)), background(std::move(bgrav)), check(check_stability) {}

    std::string HorndeskiAlphaGravity::info() const {
        std::string s = "Horndeski, alpha_T = 0: " + alpha->info();
        if (background)
            s += "; expansion from " + background->info();
        return s + stability_info;
    }

    double HorndeskiAlphaGravity::E2(double x, const BackgroundCosmology & bg) const {
        return background ? background->E2(x, bg) : bg.rho_species(x) + bg.rhoK(x);
    }
    double HorndeskiAlphaGravity::dE2dx(double x, const BackgroundCosmology & bg) const {
        return background ? background->dE2dx(x, bg) : bg.drho_species(x) + bg.drhoKdx(x);
    }
    double HorndeskiAlphaGravity::ddE2ddx(double x, const BackgroundCosmology & bg) const {
        return background ? background->ddE2ddx(x, bg) : bg.ddrho_species(x) + bg.ddrhoKddx(x);
    }

    // D c_s^2 = (2 - aB)[aB/2 + aM - Hdot/H^2] - (rho+p)/(H^2 M2) + alphadot_B/H with -Hdot/H^2 = (R + Rx)/(2 M2 H^2)
    // (cosmic time; H, R, Rx conformal): the matter terms combine to [(2 - aB) Rx - aB R]/(2 M2 H^2)
    double HorndeskiAlphaGravity::State::cs2_num() const {
        const double B = al.aB;
        return (2.0 - B) * (B / 2.0 + al.aM) + ((2.0 - B) * Rx - B * R) / (2.0 * al.M2 * H * H) + aBd / H;
    }

    HorndeskiAlphaGravity::State HorndeskiAlphaGravity::state(double x, const BackgroundCosmology & bg) const {
        State s;
        s.al = (*alpha)(x, bg);
        s.a = std::exp(x);
        const double e2 = E2(x, bg), L = 0.5 * dE2dx(x, bg) / e2; // dlnE/dx
        s.H = s.a * std::sqrt(e2);
        s.Hd = s.H * s.H * (1.0 + L);
        // matter (all species but dark energy) and dark energy, each conserved: rho + p = -rho_x/3, so
        // R = 3 a^2 (rho + p) = -a^2 rho_x and R_x = -a^2 (2 rho_x + rho_xx) (rho in units of rho_c0), no finite
        // differences
        double drho = 0.0, ddrho = 0.0, drho_de = 0.0;
        for (const auto & sp : bg.species()) {
            if (is_de(*sp)) {
                drho_de += sp->drhodx(x);
            } else {
                drho += sp->drhodx(x);
                ddrho += sp->ddrhoddx(x);
            }
        }
        const double a2 = s.a * s.a;
        s.R = -a2 * drho;
        s.Rd = -s.H * a2 * (2.0 * drho + ddrho);
        if (!std::isnan(s.al.Rx))
            s.Rx = s.al.Rx;
        else if (!background)
            s.Rx = s.al.dM2 * s.R - s.al.M2 * a2 * drho_de;
        else
            s.Rx = -2.0 * s.al.M2 * s.H * s.H * L - s.R;
        s.aKd = s.H * s.al.daKdx;
        s.aBd = s.H * s.al.daBdx;
        return s;
    }

    void HorndeskiAlphaGravity::init_background(const BackgroundCosmology & bg) {
        if (bg.get_params().OmegaK != 0.0)
            throw std::invalid_argument(name() + ": only implemented for a flat universe");
        if (background)
            background->init_background(bg);
        alpha->init(bg);
        // Stability (as hi_class): M2 > 0, no ghost D >= 0, no gradient instability c_s^2 >= 0, from deep in the
        // radiation era to well into the future
        trivial = true;
        double minD = 1e300, mincs2 = 1e300, minM2 = 1e300, x_cs2 = 2.0, x_D = 2.0; // x_*: first instability
        for (double x = -15.0; x <= 1.0 + 1e-12; x += 0.005) {
            const auto s = state(x, bg);
            if (s.al.aK != 0.0 || s.al.aB != 0.0 || s.al.aM != 0.0)
                trivial = false;
            const double D = s.D();
            if (!std::isfinite(D) || !std::isfinite(s.cs2_num()) || !std::isfinite(s.al.M2) || !std::isfinite(s.Rx))
                throw std::runtime_error(name() + ": non-finite alphas or background at x = " + std::to_string(x));
            minM2 = std::min(minM2, s.al.M2);
            minD = std::min(minD, D);
            if (D < 0.0)
                x_D = std::min(x_D, x);
            if (D > 0.0) {
                const double cs2 = s.cs2_num() / D;
                mincs2 = std::min(mincs2, cs2);
                if (cs2 < 0.0)
                    x_cs2 = std::min(x_cs2, x);
            }
        }
        std::ostringstream os;
        if (trivial) {
            os << "; all alphas vanish (GR)";
        } else {
            os << "; min D = " << minD << ", min c_s^2 = " << mincs2;
            if (check) {
                if (!(minM2 > 0.0))
                    throw std::invalid_argument(name() + ": M*^2 <= 0");
                std::ostringstream e;
                e << name() << ": ";
                if (minD < 0.0) {
                    e << "ghost instability (D = alpha_K + 3 alpha_B^2/2 < 0 from a = " << std::exp(x_D) << ")";
                    throw std::invalid_argument(e.str());
                }
                if (mincs2 < 0.0) {
                    e << "gradient instability (c_s^2 < 0 from a = " << std::exp(x_cs2) << ", min c_s^2 = " << mincs2
                      << ")";
                    throw std::invalid_argument(e.str());
                }
            }
        }
        stability_info = os.str();
    }

    void HorndeskiAlphaGravity::update_cache(const BackgroundCosmology & bg, const ModeAtX & b, double * cache) const {
        const auto s = state(b.x, bg);
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

    void HorndeskiAlphaGravity::initial_conditions(const ModeAtX &, const InitialData & ic, double * y) const {
        y[0] = ic.Phi;
        y[1] = 0.0; // the alphas (and the scalar's effect) vanish early for the standard parameterizations
        y[2] = 0.0;
    }

    // Equations from python/derive_horndeski_alpha.py, units H0 = c = kappa = 1, conformal time: H = Hp/H0,
    // k = ck/H0, P = Phi_P = -Phi, P1 = Phi_P', U = u = v' + Psi; matter drho = a^2 delta rho, dp = a^2 delta p,
    // Q = a^2 (rho+p) theta/k^2, Sig = (3/2) a^2 (rho+p) sigma (3 times the code's Moments / H0^2)
    MetricAtX HorndeskiAlphaGravity::metric(const ModeAtX & b,
                                            const double * cache,
                                            const double * y,
                                            const Moments & total,
                                            double * dydx) const {
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
            const double cv = -6.0 * H4 * M22 * aB2 * aM - 24.0 * H4 * M22 * aB2 - 4.0 * H4 * M22 * aK * aM -
                              16.0 * H4 * M22 * aK - 6.0 * H3 * M22 * aB * aBd - 4.0 * H3 * M22 * aKd +
                              6.0 * H2 * M2 * R * aB2 - 6.0 * H2 * M2 * R * aB + 6.0 * H2 * M2 * R * aK +
                              6.0 * H2 * M2 * Rx * aB2 + 6.0 * H2 * M2 * Rx * aB + 6.0 * H2 * M2 * Rx * aK -
                              6.0 * H * M2 * R * aBd - 6.0 * H * M2 * Rd * aB - 6.0 * H * M2 * Rx * aBd +
                              3.0 * R * R * aB + 6.0 * R * Rx * aB - 6.0 * R * Rx + 3.0 * Rx * Rx * aB - 6.0 * Rx * Rx +
                              k2 * (-4.0 * H2 * M22 * aB * aM - 4.0 * H2 * M22 * aB - 8.0 * H2 * M22 * aM -
                                    4.0 * H * M22 * aBd + 2.0 * M2 * R * aB + 2.0 * M2 * Rx * aB - 4.0 * M2 * Rx);
            const double cPsi = 6.0 * H3 * M22 * aB2 + 4.0 * H3 * M22 * aK - 12.0 * H2 * M22 * aBd +
                                4.0 * H * M22 * aB * k2 + 6.0 * H * M2 * R * aB + 6.0 * H * M2 * Rx * aB -
                                12.0 * H * M2 * Rx;
            const double cP = 8.0 * H * M22 * aM * k2;
            const double cP1 = -12.0 * H * M22 * aBd + 6.0 * M2 * R * aB + 6.0 * M2 * Rx * aB - 12.0 * M2 * Rx;
            const double num =
                cU * U + cv * v + cPsi * Psi + cP * P + cP1 * P1 - 6.0 * H * M2 * aB * dp + 4.0 * H * M2 * aB * Sig;
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

    void HorndeskiAlphaGravity::outputs(const ModeAtX &,
                                        const double *,
                                        const double * y,
                                        const MetricAtX &,
                                        double * out) const {
        out[0] = y[1];
        out[1] = y[2];
    }

} // namespace FML::COSMOLOGY
