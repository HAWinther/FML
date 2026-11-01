//==========================================================
// Tests of Cosmology/Background/BackgroundModel_Quintessence.h and BackgroundModel_EDE.h (background only):
//   - exponential quintessence and early dark energy against the Einstein-Boltzmann solver's models before the solver
//     moved onto this module (same parameters, with massive neutrinos and curvature; its old model classes, stored
//     tables, see SolverReference.h; x in steps of 0.185): E, dlogE/dx, Omega_i, eta, t, chi
//   - all four quintessence potentials (with phi_ini, dphidlna_ini) and EDE against SimulationCode's former
//     CosmologyQuintessence / CosmologyEDE (its own scalar-field solver, massless neutrinos; values stored in
//     data/simcode_scalar_fields.txt)
//   - independent checks: lambda = 0 is LCDM, the Klein-Gordon drho/dx against a finite difference, -1 <= w <= 1,
//     the inverse-power-law tracker w = -2/(alpha + 2) in matter domination, EDE f_max = f_EDE at z_c, the switch to
//     the oscillating fluid
//   - the registry entries, errors
//==========================================================
#include <FML/Cosmology/Background/Presets.h>
#include <FML/Cosmology/Background/BackgroundModel_EDE.h>
#include <FML/Cosmology/Background/BackgroundModel_Quintessence.h>

#include "SolverReference.h"

#include <cmath>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::check_local;
using FML::TESTS::num;
using namespace FML::COSMOLOGY::BACKGROUND;

template <class F>
bool throws(F && f) {
    try {
        f();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

double rel(double a, double b) { return std::abs(a / b - 1.0); }

BackgroundParameters planck(double m_nu = 0.06, double Ok = 0.0) {
    BackgroundParameters p;
    p.h = 0.6736;
    p.Omega_b = 0.02237 / (p.h * p.h);
    p.Omega_cdm = 0.1200 / (p.h * p.h);
    p.Omega_k = Ok;
    if (m_nu > 0.0)
        p.m_nu_eV = {m_nu};
    return p;
}

// ---------------------------------------------------------------------------------------------------------------------
// Against the Einstein-Boltzmann solver
// ---------------------------------------------------------------------------------------------------------------------

// The solver's stored background (data/solver_scalar_fields.txt, the block of the label)
void compare_with_solver(const Background & bg, const std::string & label, double tol) {
    double errE = 0.0, errO = 0.0, errd = 0.0;
    for (const auto & r : solver_table("solver_scalar_fields.txt", label)) {
        const double x = r.x;
        errE = std::max({errE, rel(bg.E(x), r.E), std::abs(bg.dlogEdx(x) - r.dlogEdx)});
        errO = std::max({errO,
                         std::abs(bg.Omega(Component::Baryons, x) - r.Omega_b),
                         std::abs(bg.Omega(Component::CDM, x) - r.Omega_cdm),
                         std::abs(bg.Omega(Component::Photons, x) - r.Omega_gamma),
                         std::abs(bg.Omega(Component::MassiveNeutrinos, x) - r.Omega_ncdm),
                         std::abs(bg.Omega(Component::DarkEnergy, x) - r.Omega_de),
                         std::abs(bg.Omega(Component::Curvature, x) - r.Omega_k)});
        errd = std::max({errd, rel(bg.eta(x), r.eta), rel(bg.t_Gyr(x), r.t)});
        if (std::abs(x) > 0.05)
            errd = std::max(errd, rel(bg.chi(x), r.chi));
    }
    check_local(errE < tol, label + ": E, dlogE/dx = solver (max error " + num(errE, 2) + ")");
    check_local(errO < tol, label + ": Omega_i(x) incl. Omega_DE = solver (max error " + num(errO, 2) + ")");
    check_local(errd < tol, label + ": eta, t, chi = solver (max error " + num(errd, 2) + ")");
}

void test_vs_solver() {
    for (auto [lambda, mnu, Ok] :
         {std::tuple{0.5, 0.06, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.15, 0.0}, {1.4, 0.06, 0.02}, {0.8, 0.06, -0.02}}) {
        const auto p = planck(mnu, Ok);
        compare_with_solver(*make_background("quintessence", p, {{"quint_lambda", lambda}}),
                            "quintessence lambda = " + num(lambda, 2) + ", m_nu = " + num(mnu, 2) +
                                ", Omega_k = " + num(Ok, 2),
                            1e-6);
    }
    for (auto [fEDE, lzc, th, n, mnu] : {std::tuple{0.1, 3.5, 2.8, 3.0, 0.06},
                                         {0.05, 4.5, 2.0, 3.0, 0.0},
                                         {0.12, 3.3, 1.5, 4.0, 0.1},
                                         {0.08, 3.6, 2.8, 2.5, 0.06}}) {
        compare_with_solver(*make_ede(planck(mnu), fEDE, lzc, th, n),
                            "EDE f = " + num(fEDE, 2) + ", log10 z_c = " + num(lzc, 2) + ", theta_i = " + num(th, 2) +
                                ", n = " + num(n, 2) + ", m_nu = " + num(mnu, 2),
                            1e-6);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Against SimulationCode
// ---------------------------------------------------------------------------------------------------------------------

// SimulationCode's former dark energy (field + bare Lambda) and E (block `label` of data/simcode_scalar_fields.txt)
// against ours
void compare_with_simcode(const ComponentBackground & bg,
                          const std::string & block_label,
                          const std::string & label,
                          double tol) {
    const FML::TESTS::ReferenceData data(std::string(REFERENCE_DATA_DIR) + "/simcode_scalar_fields.txt");
    auto block = data.block(block_label);
    double errE = 0.0, errO = 0.0;
    while (not block.done()) {
        const auto r = block.numbers("x"); // x, E, Omega_DE
        errE = std::max(errE, rel(bg.E(r[0]), r[1]));
        errO = std::max(errO, std::abs(bg.Omega(Component::DarkEnergy, r[0]) - r[2]));
    }
    check_local(errE < tol and errO < tol,
                label + ": E, Omega_DE = SimulationCode (max error " + num(errE, 2) + ", " + num(errO, 2) + ")");
}

void test_vs_simcode() {
    const auto p = planck(0.0);
    struct Case {
        std::string potential;
        QuintessencePotential V;
        double phi_ini, dphi_ini;
    };
    const std::vector<Case> cases{
        {"exponential", QuintessencePotential::exponential(1.2), 0.3, 0.0},
        {"exponential", QuintessencePotential::exponential(0.7), 0.0, 0.5},
        {"inverse_power_law", QuintessencePotential::inverse_power_law(1.0), 0.05, 0.0},
        {"inverse_power_law", QuintessencePotential::inverse_power_law(2.0), 0.2, 0.0},
        {"pngb", QuintessencePotential::pngb(1.0), 1.0, 0.0},
        {"pngb", QuintessencePotential::pngb(0.7), 0.5, 0.0},
        {"hilltop", QuintessencePotential::hilltop(1.0, 2.0), 0.1, 0.0},
        {"hilltop", QuintessencePotential::hilltop(2.0, 4.0), -0.5, 0.0},
    };
    int icase = 0;
    for (const auto & c : cases) {
        auto bg = make_quintessence(p, c.V, c.phi_ini, c.dphi_ini);
        // SimulationCode started the field at a = 1e-10, we at x = -25: the same except for a kinetic start
        compare_with_simcode(*bg,
                             "quintessence " + std::to_string(icase++),
                             "quintessence " + c.V.info() + ", phi_ini = " + num(c.phi_ini, 2) +
                                 ", dphidlna_ini = " + num(c.dphi_ini, 2),
                             c.dphi_ini == 0.0 ? 2e-6 : 1e-4);
    }
    // (n = 1.5 with z_c after equality: allowed by SimulationCode and here, not by the Einstein-Boltzmann solver)
    int iede = 0;
    for (auto [fEDE, zc, th, n] :
         {std::tuple{0.1, 3500.0, 2.8, 3.0}, {0.05, 20000.0, 2.0, 3.5}, {0.03, 300.0, 2.0, 1.5}}) {
        auto bg = make_ede(p, fEDE, std::log10(zc), th, n);
        compare_with_simcode(*bg,
                             "ede " + std::to_string(iede++),
                             "EDE f = " + num(fEDE, 2) + ", z_c = " + num(zc, 5) + ", n = " + num(n, 2),
                             1e-5);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Independent checks
// ---------------------------------------------------------------------------------------------------------------------

const ScalarFieldSpecies * field_of(const ComponentBackground & bg) {
    for (const auto & s : bg.species())
        if (auto f = dynamic_cast<const ScalarFieldSpecies *>(s.get()))
            return f;
    return nullptr;
}

// drho/dx of the field against a finite difference of rho (to a = 1.6), and -1 <= w <= 1 (to a = e^2)
void check_field(const ComponentBackground & bg, const std::string & label) {
    const auto * f = field_of(bg);
    double err = 0.0, wmin = 1e100, wmax = -1e100;
    for (double x = -20.0; x <= 2.0; x += 0.0137) {
        if (std::abs(x - f->x_fluid()) < 0.01)
            continue;
        // (not for a > 1.6: a field oscillating fast in the future makes the finite difference of the interpolated
        // rho noisy at the 1e-4 level, while E itself is converged to ~3e-7 there)
        const double h = 1e-4;
        const double num_d = (f->rho(x + h) - f->rho(x - h)) / (2.0 * h);
        if (x <= 0.5)
            err = std::max(err, std::abs(num_d - f->drhodx(x)) / std::abs(bg.E(x) * bg.E(x)));
        wmin = std::min(wmin, f->w(x));
        wmax = std::max(wmax, f->w(x));
    }
    check_local(err < 1e-6 and wmin >= -1.0 - 1e-12 and wmax <= 1.0 + 1e-12,
                label + ": drho/dx = -E^2 phi_x^2 = d rho/dx (max error / E^2 " + num(err, 2) + "), w in [" +
                    num(wmin, 4) + ", " + num(wmax, 4) + "]");
}

void test_independent() {
    const auto p = planck(0.06, 0.01);
    // lambda = 0 is LCDM
    auto q0 = make_quintessence(p, QuintessencePotential::exponential(0.0));
    auto l = make_lcdm(p);
    double err = 0.0;
    for (double x = -22.0; x <= 2.0; x += 0.1)
        err = std::max({err,
                        rel(q0->E(x), l->E(x)),
                        std::abs(q0->Omega(Component::DarkEnergy, x) - l->Omega(Component::DarkEnergy, x)),
                        rel(q0->chi(x == 0.0 ? 0.5 : x), l->chi(x == 0.0 ? 0.5 : x))});
    check_local(err < 1e-12, "quintessence with lambda = 0 is LCDM (max error " + num(err, 2) + ")");

    for (const auto & [V, phi_ini] :
         std::vector<std::pair<QuintessencePotential, double>>{{QuintessencePotential::exponential(1.0), 0.0},
                                                               {QuintessencePotential::inverse_power_law(2.0), 0.1},
                                                               {QuintessencePotential::pngb(1.0), 0.8},
                                                               {QuintessencePotential::hilltop(1.0, 3.0), 0.2}}) {
        auto bg = make_quintessence(p, V, phi_ini);
        check_field(*bg, "quintessence " + V.info());
        check_local(std::abs(bg->E(0.0) - 1.0) < 1e-10 and
                        std::abs(bg->Omega_m0() + bg->Omega_r(0.0) + bg->Omega0(Component::DarkEnergy) + p.Omega_k -
                                 1.0) < 1e-10,
                    "quintessence " + V.info() + ": the budget closes");
    }

    // Tracker: w = (alpha w_B - 2) / (alpha + 2) = -2 / (alpha + 2) in matter domination, from very different phi_ini
    {
        const double alpha = 4.0;
        double maxdev = 0.0;
        for (double phi_ini : {0.01, 0.1}) {
            auto bg = make_quintessence(planck(0.0), QuintessencePotential::inverse_power_law(alpha), phi_ini);
            const auto * f = field_of(*bg);
            for (double z : {100.0, 30.0, 10.0})
                maxdev = std::max(maxdev, std::abs(f->w(-std::log1p(z)) + 2.0 / (alpha + 2.0)));
        }
        check_local(maxdev < 0.02,
                    "inverse power law (alpha = 4) on the tracker w = -1/3 in matter domination (max deviation " +
                        num(maxdev, 2) + ")");
    }

    // Early dark energy: the peak is f_EDE at z_c
    for (auto [fEDE, lzc, th, n] : {std::tuple{0.1, 3.5, 2.8, 3.0}, {0.04, 5.0, 1.0, 5.0}}) {
        auto bg = make_ede(planck(0.06), fEDE, lzc, th, n);
        const auto * e = dynamic_cast<const EarlyDarkEnergySpecies *>(field_of(*bg));
        double fm, xp;
        e->fraction_peak(fm, xp);
        // the fraction rho_EDE / rho_tot of the background itself
        const double xc = -std::log1p(std::pow(10.0, lzc));
        const double fbg = e->rho(xc) / (bg->E(xc) * bg->E(xc));
        check_local(std::abs(fm / fEDE - 1.0) < 1e-9 and std::abs(xp - xc) < 1e-8 and std::abs(fbg / fEDE - 1.0) < 1e-6,
                    "EDE: max rho_EDE / rho_tot = f_EDE at z_c (" + num(fm, 10) +
                        " at log10 z = " + num(std::log10(std::expm1(-xp)), 8) + ")");
        check_field(*bg, "EDE n = " + num(n, 2));
    }

    // The switch to the oscillating fluid: EDE with n = 2.2 thawing deep in the radiation era (m / H grows as
    // a^((5-n)/(n+1)) there) oscillates fast before equality
    {
        const double fEDE = 0.05, lzc = 6.0, n = 2.2, wosc = (n - 1.0) / (n + 1.0);
        auto bg = make_ede(planck(0.0), fEDE, lzc, 2.0, n);
        const auto * f = field_of(*bg);
        const double xf = f->x_fluid();
        const double r1 = f->rho(xf + 0.5) * std::exp(3.0 * (1.0 + wosc) * (xf + 0.5));
        const double r2 = f->rho(-1.0) * std::exp(3.0 * (1.0 + wosc) * -1.0);
        check_local(std::isfinite(xf) and xf < -3.0 and rel(r1, r2) < 1e-12 and f->phi(-1.0) == 0.0 and
                        std::abs(bg->E(0.0) - 1.0) < 1e-10,
                    "EDE n = 2.2, z_c = 1e6 oscillates: a fluid with w = (n-1)/(n+1) from a = " + num(std::exp(xf), 4));
        // the cycle-averaged w just before the switch, and the density at the switch on the trend of the oscillations
        double wsum = 0.0, lr = 0.0;
        int nw = 0;
        for (double x = xf - 0.1; x < xf; x += 0.0001, nw++) {
            wsum += f->w(x);
            lr += std::log(f->rho(x)) + 3.0 * (1.0 + wosc) * (x - xf);
        }
        const double jump = lr / nw - std::log(f->rho(xf + 1e-9));
        check_local(std::abs(wsum / nw - wosc) < 0.01 and std::abs(jump) < 0.01,
                    "EDE: before the switch <w> = " + num(wsum / nw, 4) + " (fluid " + num(wosc, 4) +
                        "), <ln rho> - ln rho_fluid = " + num(jump, 2));
        // (no comparison with the Einstein-Boltzmann solver here: it integrates the ~1e5 oscillations to a = e^10)
    }
}

void test_registry_and_errors() {
    const auto p = planck(0.06);
    auto a = make_background("quintessence", p, {{"quint_lambda", 1.0}});
    auto b = make_quintessence(p, QuintessencePotential::exponential(1.0));
    auto c = make_background("ede", p);
    auto d = make_ede(p, 0.1, 3.5, 2.8, 3.0);
    auto e = make_background("ede_zc", p, {{"ede_zc", std::pow(10.0, 3.5)}});
    check_local(a->E(-3.0) == b->E(-3.0) and c->E(-8.0) == d->E(-8.0) and
                    std::abs(e->E(-8.0) / d->E(-8.0) - 1.0) < 1e-9,
                "registry: quintessence, ede, ede_zc = the presets");
    for (std::string m : {"quintessence_inverse_power_law", "quintessence_pngb", "quintessence_hilltop"})
        check_local(std::abs(make_background(m, p)->E(0.0) - 1.0) < 1e-10, "registry: " + m + " with its defaults");
    check_local(throws([&] { make_quintessence(p, QuintessencePotential::inverse_power_law(1.0), 0.0); }),
                "inverse power law needs phi_ini > 0");
    check_local(throws([&] { make_quintessence(p, QuintessencePotential::exponential(1.0), 0.0, 3.0); }),
                "|dphidlna_ini| >= sqrt(6) is an error");
    check_local(throws([&] { make_quintessence(p, QuintessencePotential::pngb(0.5), 0.5); }),
                "PNGB f = 0.5 from phi_ini = 0.5 cannot give the dark energy today (SimulationCode fails too)");
    check_local(throws([&] { make_quintessence(p, QuintessencePotential::exponential(10.0)); }),
                "exponential with lambda = 10 cannot give the dark energy today");
    check_local(throws([&] { make_ede(p, 0.1, 3.5, 2.8, 1.0); }) and
                    throws([&] { make_ede(p, 1.2, 3.5, 2.8, 3.0); }) and
                    throws([&] { make_ede(p, 0.1, 3.5, 3.2, 3.0); }),
                "EDE parameter checks (n > 1, f_EDE < 1, theta_i < pi)");
    check_local(throws([&] { make_ede(p, 0.05, 5.0, 2.0, 1.5); }),
                "EDE n = 1.5 with z_c in the radiation era has no solution (the fraction peaks only after equality)");
    BackgroundRange r;
    r.x_min = std::log(1e-12);
    r.x_max = std::log(50.0);
    auto wide = make_quintessence(p, QuintessencePotential::exponential(1.0), 0.0, 0.0, r);
    check_local(wide->E(r.x_min) > 0.0 and wide->E(r.x_max) > 0.0 and field_of(*wide)->x_start() <= r.x_min and
                    field_of(*wide)->x_end() >= r.x_max,
                "the field solution covers a wider range");
}

int main() {
    test_vs_solver();
    test_vs_simcode();
    test_independent();
    test_registry_and_errors();
    if (FML::ThisTask == 0) {
        make_quintessence(planck(0.06), QuintessencePotential::exponential(1.0))->info();
        make_ede(planck(0.06), 0.1, 3.5, 2.8, 3.0)->info();
    }
    return FML::TESTS::summary("Background scalar fields");
}
