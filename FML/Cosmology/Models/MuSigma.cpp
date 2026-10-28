#include <FML/Cosmology/Models/MuSigma.h>
#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    using Constants::c;

    std::string MuSigmaGravity::info() const {
        std::ostringstream os;
        os << "mu-Sigma (Planck/DES: mu0 = " << mu0 << ", Sigma0 = " << Sigma0 << ")";
        return os.str();
    }

    void MuSigmaGravity::init_background(const BackgroundCosmology & bg) {
        if (bg.get_params().OmegaK != 0.0)
            throw std::invalid_argument(name() + ": only implemented for a flat universe");
        if (bg.rhoDE(0.0) <= 0.0)
            throw std::invalid_argument(name() + ": needs dark energy (Omega_DE > 0)");
        // Omega_DE(a)/Omega_DE,0 is in [0, 1] in the past, so 2 Sigma - mu >= 0 iff 1 + 2 Sigma0 - mu0 >= 0
        if (1.0 + 2.0 * Sigma0 - mu0 < 0.0)
            throw std::invalid_argument(name() + ": need 1 + 2 Sigma0 - mu0 >= 0 (2 Sigma - mu >= 0)");
    }

    // Omega_DE(a)/Omega_DE,0 = (rho_DE / E^2) / rho_DE(0), and its x-derivative
    MuSigmaGravity::MuSigma MuSigmaGravity::mu_sigma(double x, double /*k*/, const BackgroundCosmology & bg) const {
        const double E2 = bg.E2_of_x(x), rde = bg.rhoDE(x);
        const double f = rde / (E2 * bg.rhoDE(0.0));
        const double df = f * (bg.drhoDEdx(x) / rde - bg.dE2dx_of_x(x) / E2);
        MuSigma m;
        m.mu = 1.0 + mu0 * f;
        m.Sigma = 1.0 + Sigma0 * f;
        m.dg = (2.0 * Sigma0 - mu0) * df;
        return m;
    }

    void MuSigmaGravity::update_cache(const BackgroundCosmology & bg, const ModeAtX & b, double * cache) const {
        const MuSigma m = mu_sigma(b.x, b.k, bg);
        cache[MU] = m.mu;
        cache[SIGMA] = m.Sigma;
        cache[DG] = m.dg;
        double rpp = 0.0;
        for (const auto & s : bg.species())
            rpp += s->rho(b.x) + s->p(b.x);
        cache[RPP] = b.H0a2 * rpp;
        cache[DHPH] = b.dHpH;
    }

    MetricAtX MuSigmaGravity::metric(const ModeAtX & b,
                                     const double * cache,
                                     const double * y,
                                     const Moments & total,
                                     double * dydx) const {
        const double P = y[0]; // Phi_GR
        const double mu = cache[MU], g = 2.0 * cache[SIGMA] - mu, dg = cache[DG], rpp = cache[RPP];
        const double c2k2 = c * c * b.k * b.k, A = 1.5 / c2k2;
        const double F = total.flux, ckH = b.ckH;
        MetricAtX m;
        m.Psi = -mu * P - 3.0 * total.shear / c2k2;
        m.Phi = g * P;
        const double num = -P + A * (-ckH * F + 3.0 * (cache[DHPH] - 1.0) * F / ckH - 2.0 * total.shear +
                                     3.0 * rpp * m.Psi - 3.0 * rpp * dg * P);
        const double den = 1.0 + 3.0 * A * rpp * g;
        if (!(den > 0.0))
            throw std::runtime_error(name() + ": singular Phi_GR equation (2 Sigma - mu < 0)");
        // GR: the same equation gives Psi_GR - G (G = 3 flux / (2 (ck/Hp) Hp^2)); replace it by the energy-constraint
        // form that the GR code evolves (they differ only by violations of the Poisson constraint, which it damps)
        const double Psi_GR = -P - 3.0 * total.shear / c2k2, G = 1.5 * F / (ckH * b.Hp * b.Hp);
        const double dP_GR = Psi_GR - ckH * ckH * P / 3.0 + total.drho / (2.0 * b.Hp * b.Hp);
        dydx[0] = dP_GR + (num / den - (Psi_GR - G));
        m.dPhi = dg * P + g * dydx[0];
        return m;
    }

} // namespace FML::COSMOLOGY
