#include <FML/Cosmology/Models/BransDicke.h>
#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    using Constants::c;

    BransDickeGravity::BransDickeGravity(double omega_in,
                                         const std::vector<std::shared_ptr<const Species>> & species_in)
        : omega(omega_in), species(species_in) {
        if (!(omega > -1.5))
            throw std::invalid_argument("Brans-Dicke: need omega > -3/2");
        double rho_o = 0.0;
        for (const auto & s : species)
            rho_o += s->rho(0.0);
        // Iterate: F_i so that F(0) = phi_0, Omega_Lambda so that E(0) = 1 (rho(0) = F + F_x - omega F_x^2/(6F) at x =
        // 0)
        const double F0 = phi_today();
        double F_i = F0;
        OmegaL = 1.0 - rho_o;
        for (int it = 0;; it++) {
            const double Fe = solve(F_i);
            const double dF = dF_spline(0.0);
            const double OmegaL_new = Fe + dF - omega * dF * dF / (6.0 * Fe) - rho_o;
            const bool done = std::fabs(Fe / F0 - 1.0) < 1e-14 && std::fabs(OmegaL_new - OmegaL) < 1e-14;
            F_i *= F0 / Fe;
            OmegaL = OmegaL_new;
            if (done)
                break;
            if (it > 200)
                throw std::runtime_error("Brans-Dicke: background iteration did not converge");
        }
        solve(F_i);
    }

    std::string BransDickeGravity::info() const {
        std::ostringstream os;
        os << "Brans-Dicke (omega = " << omega << ", phi_ini = " << phi(x_ini) << ", phi_0 = " << phi(0.0)
           << ", G_eff/G at early times = " << G_eff(x_ini) << ", Omega_Lambda = " << OmegaL << ")";
        return os.str();
    }

    void BransDickeGravity::init_background(const BackgroundCosmology & bg) {
        if (bg.get_params().OmegaK != 0.0)
            throw std::invalid_argument(name() + ": only implemented for a flat universe");
        if (std::fabs(E2(0.0, bg) - 1.0) > 1e-10)
            throw std::invalid_argument(name() + ": E(0) != 1 (inconsistent model)");
    }

    void BransDickeGravity::sums(double x, double & rho, double & drho, double & p) const {
        rho = OmegaL;
        drho = 0.0;
        p = -OmegaL;
        for (const auto & s : species) {
            rho += s->rho(x);
            drho += s->drhodx(x);
            p += s->p(x);
        }
    }

    // From E^2 = rho/D, D = F + F_x - omega F_x^2/(6F), and F_xx = -(3 + L) F_x + S/E^2 with S = 3(rho - 3p)/(3 + 2
    // omega):
    //   L [2 D - (1 - omega y/3) F_x] = (rho'/rho) D - F_x - (1 - omega y/3)(S/E^2 - 3 F_x) - omega F_x y^2/6,  y =
    //   F_x/F
    double BransDickeGravity::dlnE(double F, double dF, double rho, double drho, double p) const {
        const double y = dF / F, D = F + dF - omega * dF * y / 6.0, E2 = rho / D;
        const double S = 3.0 * (rho - 3.0 * p) / (3.0 + 2.0 * omega), q = 1.0 - omega * y / 3.0;
        return ((drho / rho) * D - dF - q * (S / E2 - 3.0 * dF) - omega * dF * y * y / 6.0) / (2.0 * D - q * dF);
    }

    double BransDickeGravity::solve(double F_i) {
        ODEFunction rhs = [&](double x, const double * Y, double * dY) {
            double rho, drho, p;
            sums(x, rho, drho, p);
            const double F = Y[0], dF = Y[1];
            const double D = F + dF - omega * dF * dF / (6.0 * F), E2 = rho / D;
            const double L = dlnE(F, dF, rho, drho, p);
            dY[0] = dF;
            dY[1] = -(3.0 + L) * dF + 3.0 * (rho - 3.0 * p) / ((3.0 + 2.0 * omega) * E2);
            return GSL_SUCCESS;
        };
        const auto xs = Utils::linspace(x_ini, x_fin, 14001);
        ODESolver ode(1e-4, 1e-15, 1e-12);
        ode.solve(rhs, xs, {F_i, 0.0}, gsl_odeiv2_step_rk8pd);
        const auto F = ode.get_data_by_component(0), dF = ode.get_data_by_component(1);
        std::vector<double> L(xs.size());
        for (size_t i = 0; i < xs.size(); i++) {
            double rho, drho, p;
            sums(xs[i], rho, drho, p);
            L[i] = dlnE(F[i], dF[i], rho, drho, p);
        }
        F_spline.create(xs, F, "BD phi");
        dF_spline.create(xs, dF, "BD dphidx");
        L_spline.create(xs, L, "BD dlnE");
        return F_spline(0.0);
    }

    double BransDickeGravity::phi(double x) const {
        if (x < x_ini)
            return F_spline(x_ini);
        if (x > x_fin)
            throw std::out_of_range("Brans-Dicke: x beyond the end of the background solution");
        return F_spline(x);
    }
    double BransDickeGravity::dphidx(double x) const {
        if (x < x_ini)
            return 0.0;
        if (x > x_fin)
            throw std::out_of_range("Brans-Dicke: x beyond the end of the background solution");
        return dF_spline(x);
    }

    double BransDickeGravity::E2(double x, const BackgroundCosmology & bg) const {
        const double F = phi(x), dF = dphidx(x);
        return bg.rho_species(x) / (F + dF - omega * dF * dF / (6.0 * F));
    }
    double BransDickeGravity::dE2dx(double x, const BackgroundCosmology & bg) const {
        double p = 0.0;
        for (const auto & s : bg.species())
            p += s->p(x);
        return 2.0 * dlnE(phi(x), dphidx(x), bg.rho_species(x), bg.drho_species(x), p) * E2(x, bg);
    }
    // E2'' = 2 E^2 (L' + 2 L^2), L' from the spline of L
    double BransDickeGravity::ddE2ddx(double x, const BackgroundCosmology & bg) const {
        double p = 0.0;
        for (const auto & s : bg.species())
            p += s->p(x);
        const double L = dlnE(phi(x), dphidx(x), bg.rho_species(x), bg.drho_species(x), p);
        const double dL = (x < x_ini) ? 0.0 : L_spline.deriv_x(x);
        return 2.0 * E2(x, bg) * (dL + 2.0 * L * L);
    }

    // B = F_xx + (2 + Hp'/Hp) F_x = 3 (rho - 3p) / ((3 + 2 omega) E^2)  (Hp'/Hp = 1 + dlnE/dx)
    void BransDickeGravity::update_cache(const BackgroundCosmology & bg, const ModeAtX & b, double * cache) const {
        double p = 0.0;
        for (const auto & s : bg.species())
            p += s->p(b.x);
        cache[F_] = phi(b.x);
        cache[FX] = dphidx(b.x);
        cache[B_] = 3.0 * (bg.rho_species(b.x) - 3.0 * p) / ((3.0 + 2.0 * omega) * bg.E2_of_x(b.x));
    }

    void BransDickeGravity::initial_conditions(const ModeAtX &, const InitialData & ic, double * y) const {
        y[0] = ic.Phi;
        y[1] = 0.0; // the field is frozen in the radiation era (its source rho - 3p vanishes)
        y[2] = 0.0;
    }

    MetricAtX BransDickeGravity::metric(const ModeAtX & b,
                                        const double * cache,
                                        const double * y,
                                        const Moments & total,
                                        double * dydx) const {
        const double F = cache[F_], Fx = cache[FX], B = cache[B_];
        const double Phi = y[0], dF = y[1], w = y[2];
        const double Hp2 = b.Hp * b.Hp, ckH2 = b.ckH * b.ckH;
        MetricAtX m;
        m.Phi = Phi;
        m.Psi = -Phi - dF / F - 3.0 * total.shear / (c * c * b.k * b.k * F);
        const double Psi = m.Psi;
        m.dPhi = (3.0 * total.drho / Hp2 + omega * Fx * w / F - omega * dF * Fx * Fx / (2.0 * F * F) +
                  3.0 * (2.0 * F * Psi - dF) + 3.0 * (Fx * Psi - w) - ckH2 * (2.0 * F * Phi + dF)) /
                 (6.0 * F + 3.0 * Fx);
        dydx[0] = m.dPhi;
        dydx[1] = w + Fx * Psi;
        dydx[2] = -(2.0 + b.dHpH) * w - ckH2 * dF + B * Psi - 3.0 * Fx * m.dPhi +
                  3.0 * (total.drho - 3.0 * total.dp) / ((3.0 + 2.0 * omega) * Hp2);
        return m;
    }

    void BransDickeGravity::outputs(const ModeAtX &,
                                    const double * cache,
                                    const double * y,
                                    const MetricAtX & m,
                                    double * out) const {
        out[0] = y[1] / cache[F_];
        out[1] = (y[2] + cache[FX] * m.Psi) / cache[F_]; // dF_x = w + F_x Psi
    }

} // namespace FML::COSMOLOGY
