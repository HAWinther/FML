#include <FML/Cosmology/Models/CubicGalileon.h>
#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    using Constants::c;

    // =============================================================================
    // Background: E^2 = [rho + s]/2, s = sqrt(rho^2 + 4 Om); rho_phi = E^2 - rho
    // =============================================================================

    CubicGalileonSpecies::CubicGalileonSpecies(const std::vector<std::shared_ptr<const Species>> & others_in,
                                               double Omega_phi)
        : others(others_in), Om(Omega_phi) {
        if (!(Om > 0.0))
            throw std::invalid_argument("cubic Galileon: need Omega_phi > 0");
    }

    void CubicGalileonSpecies::sums(double x, double & r, double & dr, double & ddr) const {
        r = dr = ddr = 0.0;
        for (const auto & s : others) {
            r += s->rho(x);
            dr += s->drhodx(x);
            ddr += s->ddrhoddx(x);
        }
    }

    double CubicGalileonSpecies::E2(double x) const {
        double r, dr, ddr;
        sums(x, r, dr, ddr);
        return 0.5 * (r + std::sqrt(r * r + 4.0 * Om));
    }
    // E2' = (r' + r r'/s)/2
    double CubicGalileonSpecies::dE2dx(double x) const {
        double r, dr, ddr;
        sums(x, r, dr, ddr);
        const double s = std::sqrt(r * r + 4.0 * Om);
        return 0.5 * (dr + r * dr / s);
    }
    // rho_phi = (s - r)/2 = 2 Om/(r + s) (the first form cancels catastrophically when r >> Om)
    double CubicGalileonSpecies::rho(double x) const {
        double r, dr, ddr;
        sums(x, r, dr, ddr);
        return 2.0 * Om / (r + std::sqrt(r * r + 4.0 * Om));
    }
    // rho_phi' = (r r'/s - r')/2 = -r' rho_phi / s
    double CubicGalileonSpecies::drhodx(double x) const {
        double r, dr, ddr;
        sums(x, r, dr, ddr);
        const double s = std::sqrt(r * r + 4.0 * Om), q = 2.0 * Om / (r + s);
        return -dr * q / s;
    }
    // rho_phi'' = -(r'' q + r' q')/s + r' q s'/s^2,  q' = -r' q/s,  s' = r r'/s
    double CubicGalileonSpecies::ddrhoddx(double x) const {
        double r, dr, ddr;
        sums(x, r, dr, ddr);
        const double s = std::sqrt(r * r + 4.0 * Om), q = 2.0 * Om / (r + s);
        const double dq = -dr * q / s, ds = r * dr / s;
        return -(ddr * q + dr * dq) / s + dr * q * ds / (s * s);
    }

    std::string CubicGalileonSpecies::info() const {
        std::ostringstream os;
        os << "cubic galileon (tracker; Omega = " << Om << ", w0 = " << p(0.0) / rho(0.0) << ")";
        return os.str();
    }

    // =============================================================================
    // Gravity: perturbations
    // =============================================================================

    CubicGalileonGravity::CubicGalileonGravity(double Omega_phi)
        : xi_(std::sqrt(6.0 * Omega_phi)), g3_(1.0 / (3.0 * xi_)) {}

    std::string CubicGalileonGravity::info() const {
        std::ostringstream os;
        os << "cubic Galileon (tracker, c2 = -1, xi = H dphi/dt = " << xi_ << " H0^2, g3 = " << g3_ << ")";
        return os.str();
    }

    void CubicGalileonGravity::init_background(const BackgroundCosmology & bg) {
        if (bg.get_params().OmegaK != 0.0)
            throw std::invalid_argument(name() + ": only implemented for a flat universe");
        int n = 0;
        for (const auto & s : bg.species())
            n += dynamic_cast<const CubicGalileonSpecies *>(s.get()) != nullptr;
        if (n != 1)
            throw std::invalid_argument(name() + ": the model needs exactly one CubicGalileonSpecies");
    }

    void CubicGalileonGravity::update_cache(const BackgroundCosmology & bg, const ModeAtX & b, double * cache) const {
        const double E2 = bg.E2_of_x(b.x), dE2 = bg.dE2dx_of_x(b.x);
        cache[PHIX] = phi_x(E2);
        cache[PHIXX] = -cache[PHIX] * dE2 / E2;
        double rpp = 0.0;
        for (const auto & s : bg.species())
            if (!dynamic_cast<const CubicGalileonSpecies *>(s.get()))
                rpp += s->rho(b.x) + s->p(b.x);
        cache[RPP] = b.H0a2 * rpp;
    }

    void CubicGalileonGravity::initial_conditions(const ModeAtX &, const InitialData & ic, double * y) const {
        y[0] = ic.Phi;
        y[1] = 0.0; // the field perturbation is negligible early (phi_x ~ 1/E^2)
        y[2] = 0.0;
    }

    // Equations from python/derive_cubic_galileon.py, in units H0 = c = kappa = 1 and conformal time:
    // a, H = Hp/H0, Hd = dH/dtau, F1 = phi', F2 = phi'', k = ck/H0, P = Phi_P = -Phi, D = dphi, U = u,
    // matter: drho = a^2 delta rho, dp = a^2 delta p, Q = a^2 (rho+p) theta/k^2, Sig = (3/2) a^2 (rho+p) sigma,
    // R = a^2 (rho+p) (kappa rho = 3 H0^2 rho/rho_c0, so these are 3 times the code's moments / H0^2)
    MetricAtX CubicGalileonGravity::metric(const ModeAtX & b,
                                           const double * cache,
                                           const double * y,
                                           const Moments & total,
                                           double * dydx) const {
        const double a = b.a, H0sq = b.H0a2 / (a * a), H0 = std::sqrt(H0sq);
        const double H = b.Hp / H0, Hd = H * H * b.dHpH;
        const double px = cache[PHIX], pxx = cache[PHIXX];
        const double F1 = H * px, F2 = H * H * (b.dHpH * px + pxx);
        const double k = b.ckH * H;
        const double drho = 3.0 * total.drho / H0sq, dp = 3.0 * total.dp / H0sq, Sig = 3.0 * total.shear / H0sq;
        const double Q = 3.0 * total.flux / (H0sq * k), R = 3.0 * cache[RPP] / H0sq;
        const double c2 = -1.0, g3 = g3_;
        const double P = -y[0], D = y[1], U = y[2];
        const double Psi = P - Sig / (k * k);
        const double a2 = a * a, a4 = a2 * a2, F12 = F1 * F1, F13 = F12 * F1, F14 = F12 * F12, k2 = k * k,
                     g32 = g3 * g3;

        // 00 Einstein equation
        const double P1 = (D * F12 * g3 * k2 - 3.0 * F13 * H * Psi * g3 + 9.0 * F12 * H * U * g3 + F1 * U * a2 * c2 +
                           6.0 * H * H * Psi * a2 + 2.0 * P * a2 * k2 + a2 * drho) /
                          (3.0 * (F13 * g3 - 2.0 * H * a2));
        // scalar field equation
        const double U1 =
            (-36.0 * D * F14 * H * H * g32 + 9.0 * D * F14 * Hd * g32 + 18.0 * D * F13 * F2 * H * g32 -
             6.0 * D * F13 * H * a2 * c2 * g3 + 3.0 * D * F12 * F2 * a2 * c2 * g3 - 4.0 * D * F1 * H * a2 * g3 * k2 -
             4.0 * D * F2 * a2 * g3 * k2 - 2.0 * D * a4 * c2 * k2 + 9.0 * F14 * F1 * H * Psi * g32 +
             21.0 * F14 * H * U * g32 + 3.0 * F14 * Psi * a2 * c2 * g3 - 6.0 * F13 * F2 * U * g32 +
             3.0 * F13 * U * a2 * c2 * g3 + 12.0 * F12 * H * H * Psi * a2 * g3 + 30.0 * F12 * H * P1 * a2 * g3 -
             12.0 * F12 * H * Q * a2 * g3 + 6.0 * F12 * Hd * Psi * a2 * g3 + 3.0 * F12 * Psi * R * a2 * g3 -
             2.0 * F12 * Psi * a2 * g3 * k2 - 2.0 * F12 * Sig * a2 * g3 + 3.0 * F12 * a2 * dp * g3 +
             24.0 * F1 * F2 * H * Psi * a2 * g3 + 12.0 * F1 * F2 * P1 * a2 * g3 + 4.0 * F1 * H * Psi * a4 * c2 -
             12.0 * F1 * Hd * U * a2 * g3 + 6.0 * F1 * P1 * a4 * c2 - 12.0 * F2 * H * U * a2 * g3 +
             2.0 * F2 * Psi * a4 * c2 - 4.0 * H * U * a4 * c2) /
            (3.0 * F14 * g32 + 12.0 * F1 * H * a2 * g3 + 2.0 * a4 * c2);

        MetricAtX m;
        m.Phi = y[0];
        m.Psi = Psi;
        dydx[0] = -P1 / H; // d/dx = (1/H) d/dtau
        dydx[1] = (U + F1 * Psi) / H;
        dydx[2] = U1 / H;
        m.dPhi = dydx[0];
        return m;
    }

    void CubicGalileonGravity::outputs(const ModeAtX &,
                                       const double *,
                                       const double * y,
                                       const MetricAtX &,
                                       double * out) const {
        out[0] = y[1];
        out[1] = y[2];
    }

} // namespace FML::COSMOLOGY
