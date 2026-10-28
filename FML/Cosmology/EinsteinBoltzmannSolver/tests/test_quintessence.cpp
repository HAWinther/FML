// =============================================================================
// Tests of the quintessence and coupled quintessence models (models/Quintessence.h,
// models/CoupledQuintessence.h). Run with `make test`.
//
//   * lambda = 0 is LCDM (background and perturbations)
//   * Background: closure, rho' and rho'' vs finite differences, and the field equation solved
//     independently here in cosmic time (variables phi, dphi/dt) for w0
//   * Perturbations: the Einstein constraints that are not used in the evolution hold with the field's
//     density and momentum included,
//       Poisson:   c^2 k^2 Phi = (3/2) H0^2 a^2 rho Delta
//       momentum:  Phi' - Psi = -(3/2) flux / ((ck/Hp) Hp^2)
//     (Phi is evolved with the energy constraint). They hold only if the Klein-Gordon equation conserves
//     energy and momentum together with the metric, so they check the coupling terms.
//   * Coupled quintessence: beta = 0 is the quintessence model; background derivatives and exchange; an
//     independent integration in cosmic time; the Einstein constraints with CDM and field (these hold only if
//     the energy and momentum exchanged between them cancel); the growth of CDM and baryons against the
//     quasi-static two-fluid growth equations with the fifth force (G_eff = G (1 + 2 beta^2)) and the drag
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Models/CoupledQuintessence.h>
#include <FML/Cosmology/Models/Quintessence.h>

#include <cmath>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

namespace {
    CosmologyParameters quint(double lambda) {
        CosmologyParameters p;
        p.model = "quintessence";
        p.model_params["quint_lambda"] = lambda;
        return p;
    }

    CosmologyParameters cq(double lambda, double beta) {
        CosmologyParameters p;
        p.model = "coupled_quintessence";
        p.model_params = {{"cq_lambda", lambda}, {"cq_beta", beta}};
        return p;
    }

    // Einstein constraints at late times with all species: returns the largest relative violation of the
    // Poisson and the momentum constraint. drho_x, flux_x: extra species (H0^2 a^2 units) at (x, k)
    template <class Run, class Extra>
    void
    constraints(const Run & r, double kmax, const Extra & extra, double & ePoisson, double & eMomentum, double & frac) {
        const auto & cs = r.cosmo;
        const auto & pt = r.pert;
        ePoisson = eMomentum = frac = 0.0;
        for (double k : pt.get_k_array()) {
            if (k * Mpc > kmax)
                continue;
            for (double x : {-1.0, -0.5, 0.0}) {
                const double H0a2 = std::pow(cs.get_H0() * std::exp(x), 2), Hp = cs.Hp_of_x(x), ckH = c * k / Hp;
                const double rc = cs.rhoCDM(x), rb = cs.rhoB(x), rg = cs.rhoGamma(x), rn = cs.rhoNu(x);
                const double drho_m = H0a2 * (rc * pt.get_delta_cdm(x, k) + rb * pt.get_delta_b(x, k) +
                                              4.0 * rg * pt.get_Theta(0, x, k) + 4.0 * rn * pt.get_Nu(0, x, k));
                const double flux_m = H0a2 * (-rc * pt.get_v_cdm(x, k) - rb * pt.get_v_b(x, k) +
                                              4.0 * rg * pt.get_Theta(1, x, k) + 4.0 * rn * pt.get_Nu(1, x, k));
                double drho_q, flux_q;
                extra(x, k, drho_q, flux_q);
                const double D = drho_m + drho_q + 3.0 * (flux_m + flux_q) / ckH;
                ePoisson = std::max(ePoisson, std::fabs(c * c * k * k * pt.get_Phi(x, k) / (1.5 * D) - 1.0));
                const double lhs = pt.get_dPhidx(x, k) - pt.get_Psi(x, k),
                             rhs = -1.5 * (flux_m + flux_q) / (ckH * Hp * Hp);
                eMomentum = std::max(eMomentum, std::fabs(lhs / rhs - 1.0));
                frac = std::max(frac, std::fabs((drho_q + 3.0 * flux_q / ckH) / D));
            }
        }
    }

    struct Run {
        BackgroundCosmology cosmo;
        RecombinationHistory rec;
        Perturbations pert;
        Run(const CosmologyParameters & p, const PerturbationOptions & o)
            : cosmo(p), rec((cosmo.solve(), cosmo)), pert(cosmo, rec, o) {
            rec.solve();
            pert.solve();
        }
    };

    const QuintessenceSpecies & field(const BackgroundCosmology & b) {
        for (const auto & s : b.species())
            if (auto q = dynamic_cast<const QuintessenceSpecies *>(s.get()))
                return *q;
        throw std::runtime_error("no quintessence");
    }
} // namespace

int main() {
    std::cout << "lambda = 0 is LCDM\n";
    PerturbationOptions o;
    o.n_k = 5;
    o.k_min = 0.0005 / Mpc;
    o.k_max = 0.1 / Mpc;
    {
        Run l(CosmologyParameters(), o), q(quint(0.0), o);
        double eb = 0.0, ep = 0.0;
        for (double x = -15.0; x <= 2.0; x += 0.5)
            eb = std::max(eb, std::fabs(q.cosmo.Hp_of_x(x) / l.cosmo.Hp_of_x(x) - 1.0));
        for (double k : l.pert.get_k_array())
            for (double x : {-7.0, -2.0, 0.0})
                ep = std::max(ep, std::fabs(q.pert.get_Phi(x, k) / l.pert.get_Phi(x, k) - 1.0));
        check_close("Hp(x) = LCDM", eb, 0.0, 1e-12);
        check_close("Phi = LCDM", ep, 0.0, 1e-9);
    }

    std::cout << "Background (lambda = 1)\n";
    BackgroundCosmology b(quint(1.0));
    b.solve();
    const auto & qs = field(b);
    {
        check_close("closure: E^2(0) = 1", b.E2_of_x(0.0), 1.0, 1e-12);
        check_rel("Omega_phi = 1 - Omega_others", qs.rho(0.0), CosmologyParameters().OmegaLambda(), 1e-10);
        double e1 = 0.0, e2 = 0.0;
        for (double x : {-8.0, -3.0, -1.0, 0.0, 1.0}) {
            const double h = 1e-4;
            e1 = std::max(e1, std::fabs((qs.rho(x + h) - qs.rho(x - h)) / (2 * h) - qs.drhodx(x)) / qs.rho(x));
            e2 = std::max(e2, std::fabs((qs.drhodx(x + h) - qs.drhodx(x - h)) / (2 * h) - qs.ddrhoddx(x)) / qs.rho(x));
        }
        check_close("drho/dx vs finite differences", e1, 0.0, 1e-7);
        check_close("d2rho/dx2 vs finite differences", e2, 0.0, 1e-6);
        check("thawing: w = -1 early, w0 > -1", std::fabs(qs.w(-5.0) + 1.0) < 1e-4 && qs.w(0.0) > -0.95);

        // Independent solution in cosmic time t (units 1/H0), y = (a, phi, dphi/dt), from the background of the
        // other species: H^2 = rho_o(a) + (dphi/dt)^2/6 + V0 e^{-phi}, phi'' = -3 H phi' + 3 V0 e^{-phi}
        // (field in reduced Planck masses, densities in units of rho_c0). Start in the radiation era at rest.
        auto rho_o = [&](double a) { return b.rho_species(std::log(a)) - qs.rho(std::log(a)); };
        const double V0 = qs.V0_value();
        double a = std::exp(-12.0), f = 0.0, df = 0.0, t = 0.0;
        auto H = [&](double a, double f, double df) { return std::sqrt(rho_o(a) + df * df / 6.0 + V0 * std::exp(-f)); };
        auto rk = [&](double hh) {
            auto der = [&](double a, double f, double df, double & da, double & dff, double & ddf) {
                const double Hh = H(a, f, df);
                da = a * Hh;
                dff = df;
                ddf = -3.0 * Hh * df + 3.0 * V0 * std::exp(-f);
            };
            double a1, f1, g1, a2, f2, g2, a3, f3, g3, a4, f4, g4;
            der(a, f, df, a1, f1, g1);
            der(a + 0.5 * hh * a1, f + 0.5 * hh * f1, df + 0.5 * hh * g1, a2, f2, g2);
            der(a + 0.5 * hh * a2, f + 0.5 * hh * f2, df + 0.5 * hh * g2, a3, f3, g3);
            der(a + hh * a3, f + hh * f3, df + hh * g3, a4, f4, g4);
            a += hh / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4);
            f += hh / 6.0 * (f1 + 2 * f2 + 2 * f3 + f4);
            df += hh / 6.0 * (g1 + 2 * g2 + 2 * g3 + g4);
            t += hh;
        };
        while (a < 1.0) {
            const double hh = std::min(1e-3, 1e-3 / H(a, f, df)); // dlna <= 1e-3
            const double a_old = a, f_old = f, df_old = df, t_old = t;
            rk(hh);
            if (a > 1.0) { // step back and hit a = 1 with a smaller step (linear in ln a)
                const double frac = std::log(1.0 / a_old) / std::log(a / a_old);
                a = a_old;
                f = f_old;
                df = df_old;
                t = t_old;
                rk(frac * hh);
            }
        }
        const double rho_phi = df * df / 6.0 + V0 * std::exp(-f), p_phi = df * df / 6.0 - V0 * std::exp(-f);
        std::cout << "  w0 = " << qs.w(0.0) << " (cosmic-time integration: " << p_phi / rho_phi
                  << "), phi0 = " << qs.phi(0.0) << " (" << f << ")\n";
        check_close("w0 = independent integration", qs.w(0.0), p_phi / rho_phi, 1e-6);
        check_close("phi0 = independent integration", qs.phi(0.0), f, 1e-6);
    }

    std::cout << "Perturbations (lambda = 1): Einstein constraints with the field\n";
    {
        Run r(quint(1.0), o);
        const auto & q = field(r.cosmo);
        const auto & cs = r.cosmo;
        const auto & pt = r.pert;
        double ePoisson = 0.0, eMomentum = 0.0, de_frac = 0.0;
        for (double k : pt.get_k_array()) {
            if (k * Mpc > 0.02)
                continue; // horizon scales, where the field perturbation matters
            for (double x : {-1.0, -0.5, 0.0}) {
                const double H0a2 = std::pow(cs.get_H0() * std::exp(x), 2), Hp = cs.Hp_of_x(x), ckH = c * k / Hp;
                const double rc = cs.rhoCDM(x), rb = cs.rhoB(x), rg = cs.rhoGamma(x), rn = cs.rhoNu(x);
                const double drho_m = H0a2 * (rc * pt.get_delta_cdm(x, k) + rb * pt.get_delta_b(x, k) +
                                              4.0 * rg * pt.get_Theta(0, x, k) + 4.0 * rn * pt.get_Nu(0, x, k));
                const double flux_m = H0a2 * (-rc * pt.get_v_cdm(x, k) - rb * pt.get_v_b(x, k) +
                                              4.0 * rg * pt.get_Theta(1, x, k) + 4.0 * rn * pt.get_Nu(1, x, k));
                const double drho_q = H0a2 * q.rho(x) * pt.get("delta_de", x, k);
                const double flux_q = ckH * Hp * Hp * q.dphidx(x) * pt.get("delta_phi", x, k) / 3.0;
                const double D = drho_m + drho_q + 3.0 * (flux_m + flux_q) / ckH;
                const double Phi = pt.get_Phi(x, k);
                ePoisson = std::max(ePoisson, std::fabs(c * c * k * k * Phi / (1.5 * D) - 1.0));
                const double lhs = pt.get_dPhidx(x, k) - pt.get_Psi(x, k);
                const double rhs = -1.5 * (flux_m + flux_q) / (ckH * Hp * Hp);
                eMomentum = std::max(eMomentum, std::fabs(lhs / rhs - 1.0));
                de_frac = std::max(de_frac, std::fabs((drho_q + 3.0 * flux_q / ckH) / D));
            }
        }
        std::cout << "  field contribution to rho Delta: up to " << de_frac << "\n";
        check("the field perturbation contributes > 1% of rho Delta", de_frac > 1e-2);
        check_close("Poisson constraint with the field", ePoisson, 0.0, 1e-6);
        check_close("momentum constraint with the field", eMomentum, 0.0, 1e-6);
    }

    std::cout << "Coupled quintessence: beta = 0 is quintessence\n";
    {
        PerturbationOptions o2 = o;
        CosmologyParameters q = quint(0.5);
        Run a(q, o2), b0(cq(0.5, 0.0), o2);
        double e = 0.0;
        for (double k : a.pert.get_k_array())
            for (double x : {-7.0, -2.0, 0.0}) {
                e = std::max(e, std::fabs(b0.pert.get_Phi(x, k) / a.pert.get_Phi(x, k) - 1.0));
                e = std::max(e, std::fabs(b0.pert.get_delta_cdm(x, k) / a.pert.get_delta_cdm(x, k) - 1.0));
            }
        check_close("beta = 0: Phi, delta_cdm = quintessence model", e, 0.0, 1e-8);
    }

    const double lam = 0.5, beta = 0.1;
    std::cout << "Coupled quintessence (lambda = " << lam << ", beta = " << beta << "): background\n";
    BackgroundCosmology cb(cq(lam, beta));
    cb.solve();
    const CoupledField * cf = nullptr;
    for (const auto & s : cb.species())
        if (auto f = dynamic_cast<const CoupledField *>(s.get()))
            cf = f;
    const auto & d = cf->sector();
    {
        const CosmologyParameters p0;
        check_close("E(0) = 1", cb.E2_of_x(0.0), 1.0, 1e-12);
        check_rel("Omega_c(0) = OmegaCDM", cb.rhoCDM(0.0), p0.OmegaCDM, 1e-12);
        check_rel("Omega_phi(0) from closure", cb.rhoDE(0.0), p0.OmegaLambda(), 1e-10);
        double e1 = 0.0, e2 = 0.0, eex = 0.0;
        for (double x : {-8.0, -3.0, -1.0, 0.0, 1.0}) {
            const double h = 1e-4;
            for (const Species * sp : {static_cast<const Species *>(cf), cb.species()[1].get()}) {
                e1 = std::max(e1, std::fabs((sp->rho(x + h) - sp->rho(x - h)) / (2 * h) - sp->drhodx(x)) / sp->rho(x));
                e2 = std::max(e2,
                              std::fabs((sp->drhodx(x + h) - sp->drhodx(x - h)) / (2 * h) - sp->ddrhoddx(x)) /
                                  sp->rho(x));
            }
            // exchange: rho_c' + rho_phi' = -3 rho_c - 3 (rho_phi + p_phi)
            eex = std::max(
                eex,
                std::fabs(cb.drhoCDMdx(x) + cf->drhodx(x) + 3.0 * cb.rhoCDM(x) + 3.0 * (cf->rho(x) + cf->p(x))) /
                    cb.rhoCDM(x));
        }
        check_close("rho', rho'' vs finite differences (CDM and field)", std::max(e1, e2), 0.0, 1e-6);
        check_close("energy exchange cancels in the total", eex, 0.0, 1e-12);
        check("CDM dilutes faster than a^-3 (energy flows to the field)",
              cb.rhoCDM(-1.0) * std::exp(-3.0) > cb.rhoCDM(0.0));

        // Independent integration in cosmic time (units 1/H0): rho_c = C a^-3 e^{-beta phi},
        //   H^2 = rho_o(a) + rho_c + phi'^2/6 + V0 e^{-lambda phi},  phi'' = -3 H phi' + 3 lambda V0 e^{-lambda phi} +
        //   3 beta rho_c
        auto rho_o = [&](double a) {
            const double x = std::log(a);
            return cb.rho_species(x) - cb.rhoCDM(x) - cf->rho(x);
        };
        const double V0 = d.V0, C = d.C;
        // Start at a = e^-12 from the code's field (the coupling has already moved it slightly in the radiation era)
        auto rc = [&](double a, double f) { return C * std::pow(a, -3.0) * std::exp(-beta * f); };
        auto H = [&](double a, double f, double df) {
            return std::sqrt(rho_o(a) + rc(a, f) + df * df / 6.0 + V0 * std::exp(-lam * f));
        };
        double a = std::exp(-12.0), f = d.phi(-12.0), df = d.dphi(-12.0) * std::sqrt(d.E2(-12.0)); // dphi/dt = H phi_x
        auto der = [&](double a, double f, double df, double & da, double & dff, double & ddf) {
            const double Hh = H(a, f, df);
            da = a * Hh;
            dff = df;
            ddf = -3.0 * Hh * df + 3.0 * lam * V0 * std::exp(-lam * f) + 3.0 * beta * rc(a, f);
        };
        auto rk = [&](double hh) {
            double a1, f1, g1, a2, f2, g2, a3, f3, g3, a4, f4, g4;
            der(a, f, df, a1, f1, g1);
            der(a + 0.5 * hh * a1, f + 0.5 * hh * f1, df + 0.5 * hh * g1, a2, f2, g2);
            der(a + 0.5 * hh * a2, f + 0.5 * hh * f2, df + 0.5 * hh * g2, a3, f3, g3);
            der(a + hh * a3, f + hh * f3, df + hh * g3, a4, f4, g4);
            a += hh / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4);
            f += hh / 6.0 * (f1 + 2 * f2 + 2 * f3 + f4);
            df += hh / 6.0 * (g1 + 2 * g2 + 2 * g3 + g4);
        };
        while (a < 1.0) {
            const double hh = 1e-3 / H(a, f, df);
            const double a0 = a, f0 = f, df0 = df;
            rk(hh);
            if (a > 1.0) {
                const double frac = std::log(1.0 / a0) / std::log(a / a0);
                a = a0;
                f = f0;
                df = df0;
                rk(frac * hh);
            }
        }
        const double rphi = df * df / 6.0 + V0 * std::exp(-lam * f), pphi = df * df / 6.0 - V0 * std::exp(-lam * f);
        std::cout << "  w0 = " << cf->p(0.0) / cf->rho(0.0) << " (cosmic time: " << pphi / rphi
                  << "), phi0 = " << d.phi(0.0) << " (" << f << "), Omega_c(0) " << rc(1.0, f) << "\n";
        check_close("w0 = independent integration", cf->p(0.0) / cf->rho(0.0), pphi / rphi, 1e-6);
        check_close("phi0 = independent integration", d.phi(0.0), f, 1e-6);
        check_rel("Omega_c(0) = independent integration", cb.rhoCDM(0.0), rc(1.0, f), 1e-6);
    }

    std::cout << "Coupled quintessence: perturbations\n";
    {
        PerturbationOptions o2;
        o2.n_k = 6;
        o2.k_min = 0.0005 / Mpc;
        o2.k_max = 0.2 / Mpc;
        Run r(cq(lam, beta), o2), r0(cq(lam, 0.0), o2);
        double eP, eM, frac;
        constraints(
            r,
            0.05,
            [&](double x, double k, double & drho, double & flux) {
                const double H0a2 = std::pow(r.cosmo.get_H0() * std::exp(x), 2), Hp = r.cosmo.Hp_of_x(x);
                drho = H0a2 * cf->rho(x) * r.pert.get("delta_de", x, k);
                flux = (c * k / Hp) * Hp * Hp * d.dphi(x) * r.pert.get("delta_phi", x, k) / 3.0;
            },
            eP,
            eM,
            frac);
        std::cout << "  field contribution to rho Delta: up to " << frac << "\n";
        check_close("Poisson constraint with coupled CDM and field", eP, 0.0, 1e-6);
        check_close("momentum constraint with coupled CDM and field", eM, 0.0, 1e-6);

        // Quasi-static two-fluid growth from x0 = -4, started with the code's values; ratio to beta = 0 (same lambda)
        auto growth = [&](const Run & run, double bt, double k) {
            const BackgroundCosmology & bc = run.cosmo;
            const CoupledField * f = nullptr;
            for (const auto & s : bc.species())
                if (auto ff = dynamic_cast<const CoupledField *>(s.get()))
                    f = ff;
            const auto & sec = f->sector();
            const double x0 = -4.0, h = 1e-3;
            double dc = run.pert.get_delta_cdm(x0, k), db = run.pert.get_delta_b(x0, k);
            double dcx = (run.pert.get_delta_cdm(x0 + h, k) - run.pert.get_delta_cdm(x0 - h, k)) / (2 * h);
            double dbx = (run.pert.get_delta_b(x0 + h, k) - run.pert.get_delta_b(x0 - h, k)) / (2 * h);
            auto acc = [&](double x, double dc, double db, double dcx, double dbx, double & ac, double & ab) {
                const double E2 = bc.E2_of_x(x), L = 0.5 * bc.dE2dx_of_x(x) / E2;
                const double src = 1.5 * (bc.rhoCDM(x) * dc + bc.rhoB(x) * db) / E2;
                ac = -(2.0 + L - bt * sec.dphi(x)) * dcx + src + 3.0 * bt * bt * bc.rhoCDM(x) * dc / E2;
                ab = -(2.0 + L) * dbx + src;
            };
            const int n = 40000;
            const double hx = -x0 / n;
            for (int i = 0; i < n; i++) { // RK4 on (dc, db, dcx, dbx)
                const double x = x0 + i * hx;
                double k1c, k1b, k2c, k2b, k3c, k3b, k4c, k4b;
                acc(x, dc, db, dcx, dbx, k1c, k1b);
                acc(x + hx / 2, dc + hx / 2 * dcx, db + hx / 2 * dbx, dcx + hx / 2 * k1c, dbx + hx / 2 * k1b, k2c, k2b);
                acc(x + hx / 2,
                    dc + hx / 2 * (dcx + hx / 2 * k1c),
                    db + hx / 2 * (dbx + hx / 2 * k1b),
                    dcx + hx / 2 * k2c,
                    dbx + hx / 2 * k2b,
                    k3c,
                    k3b);
                acc(x + hx,
                    dc + hx * (dcx + hx / 2 * k2c),
                    db + hx * (dbx + hx / 2 * k2b),
                    dcx + hx * k3c,
                    dbx + hx * k3b,
                    k4c,
                    k4b);
                dc += hx * (dcx + hx / 6 * (k1c + k2c + k3c));
                db += hx * (dbx + hx / 6 * (k1b + k2b + k3b));
                dcx += hx / 6 * (k1c + 2 * k2c + 2 * k3c + k4c);
                dbx += hx / 6 * (k1b + 2 * k2b + 2 * k3b + k4b);
            }
            return dc / run.pert.get_delta_cdm(x0, k);
        };
        const double k = r.pert.get_k_array().back();
        const double code = (r.pert.get_delta_cdm(0.0, k) / r.pert.get_delta_cdm(-4.0, k)) /
                            (r0.pert.get_delta_cdm(0.0, k) / r0.pert.get_delta_cdm(-4.0, k));
        const double ref = growth(r, beta, k) / growth(r0, 0.0, k);
        std::cout << "  growth of delta_cdm from z = 54 to 0, beta = " << beta << " vs 0: " << code
                  << ", two-fluid growth equations: " << ref << "\n";
        check_close("CDM growth with fifth force and drag = growth equations", code / ref, 1.0, 5e-4);
    }

    return Test::summary();
}
