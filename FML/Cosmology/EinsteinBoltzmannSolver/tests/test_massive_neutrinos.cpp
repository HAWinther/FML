// =============================================================================
// Tests for massive neutrinos: momentum integrals, background, perturbations.
// The comparison with CLASS is python/compare_class_neutrinos.py. Run with `make test`.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/MassiveNeutrinos.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>

#include <gsl/gsl_integration.h>

#include <algorithm>
#include <cmath>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

namespace {
    // Adaptive reference for int_0^inf q^a eps^b f0 dq
    double reference_integral(double M, int a, int b) {
        struct P {
            double M;
            int a, b;
        } par{M, a, b};
        gsl_function F;
        F.function = [](double q, void * pp) {
            const auto * p = static_cast<P *>(pp);
            return std::pow(q, p->a) * std::pow(std::sqrt(q * q + p->M * p->M), p->b) / (std::exp(q) + 1.0);
        };
        F.params = &par;
        gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
        double r, e;
        gsl_integration_qagiu(&F, 0.0, 0.0, 1e-12, 1000, w, &r, &e);
        gsl_integration_workspace_free(w);
        return r;
    }

    // Comoving-gauge density contrast of CDM + baryons (+ massive neutrinos) from the perturbations
    double Delta_m(const BackgroundCosmology & cosmo, const Perturbations & pert, double x, double k) {
        const double f = 3.0 * cosmo.Hp_of_x(x) / (c * k);
        double num = cosmo.rhoCDM(x) * (pert.get_delta_cdm(x, k) - f * pert.get_v_cdm(x, k)) +
                     cosmo.rhoB(x) * (pert.get_delta_b(x, k) - f * pert.get_v_b(x, k));
        double den = cosmo.rhoCDM(x) + cosmo.rhoB(x);
        if (pert.has_massive_nu()) {
            num += cosmo.rhoNcdm(x) * pert.get_delta_ncdm(x, k) -
                   f * (cosmo.rhoNcdm(x) + cosmo.pNcdm(x)) * pert.get_v_ncdm(x, k);
            den += cosmo.rhoNcdm(x);
        }
        return num / den;
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
} // namespace

int main() {
    std::cout << "Momentum integrals\n";
    {
        const double I0 = 7.0 * std::pow(M_PI, 4) / 120.0;
        check_rel("I_rho(0) = 7 pi^4/120", MassiveNeutrinos::I_rho(0.0), I0, 1e-9);
        check_rel("I_p(0) = I_rho(0)/3 (radiation)", MassiveNeutrinos::I_p(0.0), I0 / 3.0, 1e-9);
        check_rel("I_rho(M) -> (3/2) zeta(3) M for M >> 1",
                  MassiveNeutrinos::I_rho(1e4),
                  1.5 * 1.2020569031595942 * 1e4,
                  1e-7);
        double err = 0.0;
        for (double M : {0.1, 1.0, 3.0, 30.0, 300.0}) {
            err = std::max(err, std::fabs(MassiveNeutrinos::I_rho(M) / reference_integral(M, 2, 1) - 1.0));
            err = std::max(err, std::fabs(MassiveNeutrinos::I_p(M) / (reference_integral(M, 4, -1) / 3.0) - 1.0));
            err = std::max(err, std::fabs(MassiveNeutrinos::I_pp(M) / (reference_integral(M, 6, -3) / 3.0) - 1.0));
        }
        check_close("48-node quadrature vs adaptive (rho, p, pseudo-p)", err, 0.0, 3e-7);
        const double M = 2.7, h = 1e-5;
        check_rel("dI_p/dlnM = I_pp - I_p",
                  (MassiveNeutrinos::I_p(M * (1 + h)) - MassiveNeutrinos::I_p(M * (1 - h))) / (2 * h),
                  MassiveNeutrinos::I_pp(M) - MassiveNeutrinos::I_p(M),
                  1e-7);
        // Perturbation quadrature: 8 nodes integrate the massless density moment to 3e-5
        MassiveNeutrinos::Quadrature q8(8);
        double s = 0.0;
        for (int i = 0; i < q8.size(); i++)
            s += q8.w[i] * std::pow(q8.q[i], 3);
        check_rel("8-node quadrature: int q^3 f0", s, I0, 5e-5);
    }

    std::cout << "Background\n";
    {
        CosmologyParameters p;
        p.m_nu = 0.06;
        BackgroundCosmology cosmo(p);
        cosmo.solve();
        check_rel("Omega_ncdm h^2 = m_nu / 93.14 eV (T_ncdm = 0.71611)", p.OmegaNcdm() * p.h * p.h, 0.06 / 93.14, 1e-4);
        check_close("closure: E^2(x = 0) = 1", cosmo.E2_of_x(0.0), 1.0, 1e-13);
        check_rel("N_ncdm = (0.71611/(4/11)^(1/3))^4 = 1.0132", p.N_ncdm(), 1.01319, 2e-5);
        // Early times: the total radiation is the same as for Neff massless species
        CosmologyParameters p0;
        BackgroundCosmology c0(p0);
        const double x = -16.0;
        check_rel("relativistic: rho_nu + rho_ncdm = rho_nu(Neff massless)",
                  cosmo.rhoNu(x) + cosmo.rhoNcdm(x),
                  c0.rhoNu(x),
                  1e-6);
        check_close("w = 1/3 early", cosmo.pNcdm(x) / cosmo.rhoNcdm(x), 1.0 / 3.0, 1e-7);
        check_close("w ~ <q^2>/(3 M^2) ~ 0 today", cosmo.pNcdm(0.0) / cosmo.rhoNcdm(0.0), 0.0, 1e-4);
        double err = 0.0;
        for (double xx : {-12.0, -7.0, -4.0, -1.0}) {
            const double h = 1e-4;
            err = std::max(
                err,
                std::fabs((cosmo.rhoNcdm(xx + h) - cosmo.rhoNcdm(xx - h)) / (2 * h) / cosmo.drhoNcdmdx(xx) - 1));
            err = std::max(
                err,
                std::fabs((cosmo.drhoNcdmdx(xx + h) - cosmo.drhoNcdmdx(xx - h)) / (2 * h) / cosmo.ddrhoNcdmddx(xx) -
                          1));
        }
        check_close("drho/dx, d2rho/dx2 vs finite differences", err, 0.0, 1e-6);
        check_close("m_nu = 0 reproduces the massless model (Omega_Lambda)",
                    p0.OmegaLambda() - CosmologyParameters().OmegaLambda(),
                    0.0,
                    1e-15);
    }

    std::cout << "Perturbations\n";
    {
        PerturbationOptions o;
        o.n_k = 4;
        o.k_min = 0.001 / Mpc;
        o.k_max = 0.3 / Mpc;
        CosmologyParameters p0, p3, pt;
        p3.m_nu = 0.3;
        pt.m_nu = 1e-4; // nearly massless
        Run m0(p0, o), m3(p3, o);
        PerturbationOptions oh = o;
        oh.ncdm_fluid_trigger = 1e30; // full hierarchy throughout
        Run m3h(p3, oh);
        PerturbationOptions ot = oh; // same truncation as the massless hierarchy, more momenta
        ot.lmax_ncdm = o.lmax_nu;
        ot.n_q_ncdm = 16;
        Run mt(pt, ot);
        const auto & ks = m0.pert.get_k_array();
        check("m_nu > 0: massive neutrino hierarchy active", m3.pert.has_massive_nu() && !m0.pert.has_massive_nu());

        // Adiabatic initial conditions: delta_ncdm = delta_nu = delta_gamma
        const double xs = o.x_start;
        check_rel("IC: delta_ncdm = 4 Nu_0",
                  m3.pert.get_delta_ncdm(xs, ks[1]),
                  4.0 * m3.pert.get_Nu(0, xs, ks[1]),
                  1e-4);

        // A very light species behaves like a massless one until it becomes non-relativistic (z ~ 2 m/T)
        double err = 0.0;
        for (double k : ks)
            err = std::max(err, std::fabs(mt.pert.get_Phi(-7.0, k) / m0.pert.get_Phi(-7.0, k) - 1.0));
        check_close("m_nu = 1e-4 eV: Phi(z = 1100) as for massless", err, 0.0, 1e-5);

        // Fluid approximation vs the full hierarchy (CLASS's ncdmfa has the same ~0.2% error in P(k))
        err = 0.0;
        for (double k : ks)
            err = std::max(err, std::fabs(m3.pert.get_Phi(0.0, k) / m3h.pert.get_Phi(0.0, k) - 1.0));
        check_close("m_nu = 0.3 eV: fluid approximation vs hierarchy, Phi today", err, 0.0, 3e-3);

        // Free streaming: neutrinos cluster like CDM on large scales but not below the free-streaming scale
        const double kl = ks.front(), kh = ks.back();
        check_rel("k = 0.001/Mpc: delta_ncdm = delta_cdm today",
                  m3.pert.get_delta_ncdm(0.0, kl),
                  m3.pert.get_delta_cdm(0.0, kl),
                  2e-2);
        const double rfs = m3.pert.get_delta_ncdm(0.0, kh) / m3.pert.get_delta_cdm(0.0, kh);
        std::cout << "    delta_ncdm / delta_cdm today at k = 0.3/Mpc: " << rfs << "\n";
        check("k = 0.3/Mpc: delta_ncdm < delta_cdm / 5 today (free streaming)", std::fabs(rfs) < 0.2);

        // Small-scale suppression of the matter power spectrum, roughly -8 f_nu (Hu, Eisenstein & Tegmark 1998)
        const double f_nu = p3.OmegaNcdm() / (p3.OmegaB + p3.OmegaCDM + p3.OmegaNcdm());
        const double r = std::pow(Delta_m(m3.cosmo, m3.pert, 0.0, kh) / Delta_m(m0.cosmo, m0.pert, 0.0, kh), 2) - 1.0;
        std::cout << "    P_m(k = 0.3/Mpc) change: " << r << " = " << r / f_nu << " f_nu\n";
        check("P_m(k = 0.3/Mpc) suppressed by 5-9 f_nu", r < -5.0 * f_nu && r > -9.0 * f_nu);
        const double rl = std::pow(Delta_m(m3.cosmo, m3.pert, 0.0, kl) / Delta_m(m0.cosmo, m0.pert, 0.0, kl), 2) - 1.0;
        check("large scales much less affected than small scales", std::fabs(rl) < 0.3 * std::fabs(r));
    }
    return Test::summary();
}
