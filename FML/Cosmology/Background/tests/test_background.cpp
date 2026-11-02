//==========================================================
// Tests of Cosmology/Background:
//   - analytic backgrounds: Einstein-de Sitter (FunctionBackground) and LCDM without radiation (ComponentBackground),
//     open and closed: E, conformal and cosmic time, comoving and transverse distances, x(chi)
//   - the Fermi-Dirac integrals of massive neutrinos (I_rho, I_p, the pseudo-pressure I_pp) and their limits; the
//     massive neutrino species (prefactor, M(x), pseudo_p, dp/dx), p_total, ScaledSpecies, the Brans-Dicke accessors
//   - sin_K, cot_K (flat, open, closed, the series region)
//   - LCDM with massive neutrinos and curvature, w0wa and DGP against the Einstein-Boltzmann solver's background before
//     it moved onto this module (its old model classes; stored tables, see SolverReference.h)
//   - TabulatedBackground from a sampled ComponentBackground
//   - TabulatedBackground::from_class and make_lcdm against a CLASS run (if the TransferProviders test data exist:
//     ./test_background <dir>, made with TransferProviders/tests/make_test_data.sh; otherwise SKIP)
//   - errors: unknown models and parameters, an open budget, queries outside the range
//==========================================================
#include <FML/Cosmology/Background/FunctionBackground.h>
#include <FML/Cosmology/Background/Presets.h>
#include <FML/Cosmology/Background/TabulatedBackground.h>

#include "SolverReference.h"

#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::check_local;
using FML::TESTS::num;
using namespace FML::COSMOLOGY::BACKGROUND;
using FML::UTILS::SI::c_over_H0_mpch;

template <class F>
bool throws(F && f) {
    try {
        f();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

// int_a^b f with composite Simpson (n even)
double simpson(const std::function<double(double)> & f, double a, double b, int n = 200000) {
    const double h = (b - a) / n;
    double s = f(a) + f(b);
    for (int i = 1; i < n; i++)
        s += f(a + i * h) * (i % 2 ? 4.0 : 2.0);
    return s * h / 3.0;
}

double rel(double a, double b) { return std::abs(a / b - 1.0); }

BackgroundParameters no_radiation(double Ob, double Oc, double Ok = 0.0) {
    BackgroundParameters p;
    p.h = 0.7;
    p.Omega_b = Ob;
    p.Omega_cdm = Oc;
    p.Omega_k = Ok;
    p.T_CMB = 0.0;
    p.N_eff = 0.0;
    return p;
}

void test_einstein_de_sitter() {
    const auto p = no_radiation(0.05, 0.95);
    FunctionBackground eds("eds",
                           p,
                           [](double x) { return std::exp(-1.5 * x); },
                           nullptr,
                           nullptr,
                           {std::log(1e-8), 1.0, 200});
    double err = 0.0, err_inv = 0.0;
    for (double a : {1e-6, 1e-3, 0.1, 0.5, 0.99, 2.0}) {
        const double x = std::log(a);
        err = std::max({err,
                        rel(eds.eta(x), 2.0 * std::sqrt(a) * c_over_H0_mpch),
                        rel(eds.t_H0(x), 2.0 / 3.0 * std::pow(a, 1.5)),
                        rel(eds.chi(x), 2.0 * (1.0 - std::sqrt(a)) * c_over_H0_mpch),
                        rel(eds.dlogEdx(x), -1.5)});
        err_inv = std::max(err_inv, std::abs(eds.x_of_chi(eds.chi(x)) - x));
    }
    check_local(err < 1e-9, "Einstein-de Sitter: eta, t, chi, dlogE/dx analytic (max error " + num(err, 2) + ")");
    check_local(err_inv < 1e-9, "x_of_chi inverts chi (max error in x " + num(err_inv, 2) + ")");
    check_local(rel(eds.age_Gyr(), 2.0 / 3.0 / (0.7 * FML::UTILS::SI::H0_over_h) / FML::UTILS::SI::Gyr) < 1e-9,
                "age = 2/(3 H0) in Gyr");
    check_local(throws([&] { eds.Omega(Component::Photons, 0.0); }) and eds.Omega(Component::CDM, 0.0) == 0.95,
                "FunctionBackground: default Omega for matter and curvature, others not available");
}

void test_lcdm_no_radiation() {
    for (double Ok : {0.0, 0.08, -0.08}) {
        const auto p = no_radiation(0.05, 0.25, Ok);
        auto bg = make_lcdm(p);
        const double Om = 0.30, OL = 1.0 - Om - Ok;
        const auto E = [&](double a) { return std::sqrt(Om / (a * a * a) + Ok / (a * a) + OL); };
        double err = 0.0;
        for (double a : {1e-4, 0.1, 0.5, 1.0, 3.0}) {
            const double x = std::log(a);
            const double chi = simpson([&](double aa) { return 1.0 / (aa * aa * E(aa)); }, a, 1.0) * c_over_H0_mpch;
            const double t = simpson([&](double aa) { return 1.0 / (aa * E(aa)); }, 1e-12, a, 2000000);
            const double s = std::sqrt(std::abs(Ok)), d = chi / c_over_H0_mpch;
            const double DM = Ok == 0 ? chi : (Ok > 0 ? std::sinh(s * d) : std::sin(s * d)) / s * c_over_H0_mpch;
            err = std::max(
                {err, rel(bg->E(x), E(a)), rel(bg->t_H0(x), t), rel(bg->Omega_m(x), Om / (a * a * a) / (E(a) * E(a)))});
            if (a != 1.0)
                err = std::max(
                    {err, rel(bg->chi(x), chi), rel(bg->D_M(x), DM), rel(bg->D_A(x), a * DM), rel(bg->D_L(x), DM / a)});
        }
        if (Ok == 0.0 and OL > 0) {
            // t(a) = 2 / (3 sqrt(OL)) asinh(sqrt(OL/Om) a^1.5)
            const double a = 0.6;
            err = std::max(err,
                           rel(bg->t_H0(std::log(a)),
                               2.0 / (3.0 * std::sqrt(OL)) * std::asinh(std::sqrt(OL / Om) * std::pow(a, 1.5))));
        }
        check_local(err < 1e-8,
                    "LCDM without radiation, Omega_k = " + num(Ok, 2) +
                        ": E, Omega_m, t, chi, D_M, D_A, D_L (max error " + num(err, 2) + ")");
    }
}

void test_neutrinos() {
    const auto & fd = FermiDiracIntegrals::get();
    double err = 0.0;
    for (double M : {1e-5, 1e-3, 0.3, 3.0, 30.0, 300.0, 3e4, 1e8}) {
        const auto f0 = [](double q) { return 1.0 / (std::exp(q) + 1.0); };
        const double Ir =
            simpson([&](double q) { return q * q * std::sqrt(q * q + M * M) * f0(q); }, 0.0, 80.0, 400000);
        const double Ip = simpson([&](double q) { return q * q * q * q / (3.0 * std::sqrt(q * q + M * M)) * f0(q); },
                                  0.0,
                                  80.0,
                                  400000);
        err = std::max({err, rel(fd.I_rho(M), Ir), rel(fd.I_p(M), Ip)});
    }
    check_local(err < 1e-8, "Fermi-Dirac I_rho, I_p (max error " + num(err, 2) + ")");

    // The pseudo-pressure integral, in the table and in its small-M series and large-M asymptote (M < 1e-4, > 1e7),
    // also next to the edges of the table
    double err_pp = 0.0, err_lim = 0.0;
    for (double M :
         {1e-6, 1e-5, 0.99e-4, 1.01e-4, 1e-3, 0.03, 0.3, 1.0, 3.0, 30.0, 300.0, 3e4, 0.99e7, 1.01e7, 1e8, 1e10}) {
        const auto f0 = [](double q) { return 1.0 / (std::exp(q) + 1.0); };
        const double Ipp = simpson(
            [&](double q) {
                const double e = std::sqrt(q * q + M * M);
                return q * q * q * q * q * q / (3.0 * e * e * e) * f0(q);
            },
            0.0,
            80.0,
            400000);
        double & e = (M < 1e-4 or M > 1e7) ? err_lim : err_pp;
        e = std::max(e, rel(fd.I_pp(M), Ipp));
    }
    check_local(err_pp < 1e-8, "Fermi-Dirac I_pp in the table (max error " + num(err_pp, 2) + ")");
    check_local(
        err_lim < 1e-8,
        "Fermi-Dirac I_pp series I_rho_0/3 - pi^2 M^2/24 (M < 1e-4) and 945/4 zeta(7)/M^3 (M > 1e7) (max error " +
            num(err_lim, 2) + ")");

    BackgroundParameters p;
    p.h = 0.67;
    p.m_nu_eV = {0.1};
    auto bg = make_lcdm(p);
    const double Onu = bg->Omega0(Component::MassiveNeutrinos);
    check_local(rel(Onu * p.h * p.h, 0.1 / 93.14) < 2e-3,
                "Omega_nu h^2 = m / 93.14 eV (" + num(Onu * p.h * p.h * 93.14 / 0.1, 6) + ")");
    const double x = std::log(1e-9);
    const double ratio = bg->Omega(Component::MassiveNeutrinos, x) / bg->Omega(Component::Photons, x);
    check_local(rel(ratio, 7.0 / 8.0 * std::pow(4.0 / 11.0, 4.0 / 3.0) * p.N_ncdm_per_species()) < 1e-8,
                "relativistic massive neutrinos count as N_ncdm = " + num(p.N_ncdm_per_species(), 5));
    const double w0 = bg->w(Component::MassiveNeutrinos, 0.0), we = bg->w(Component::MassiveNeutrinos, x);
    check_local(w0 > 0 and w0 < 1e-3 and std::abs(we - 1.0 / 3.0) < 1e-8, "massive neutrino w: 1/3 early, ~0 today");
    check_local(std::abs(bg->Omega_m(0.0) + bg->Omega_r(0.0) + bg->Omega0(Component::DarkEnergy) - 1.0) < 1e-12,
                "the budget closes");
}

// MassiveNeutrinoSpecies: prefactor, M_of_x, pseudo_p, dp/dx; p_total; ScaledSpecies and the Brans-Dicke accessors
void test_species() {
    BackgroundParameters p;
    p.h = 0.67;
    p.m_nu_eV = {0.06};
    const MassiveNeutrinoSpecies nu(0.06, p);
    const auto & fd = FermiDiracIntegrals::get();
    double err_def = 0.0, err_fd = 0.0;
    for (double x : {-20.0, -12.0, -8.0, -5.0, -3.0, -1.0, 0.0, 1.0}) {
        const double M = nu.M_of_x(x), a4 = std::exp(-4.0 * x);
        err_def = std::max({err_def,
                            rel(M, nu.M_today() * std::exp(x)),
                            rel(nu.rho(x), nu.prefactor() * a4 * fd.I_rho(M)),
                            rel(nu.pseudo_p(x), nu.prefactor() * a4 * fd.I_pp(M))});
        const double h = 1e-4, dp = (nu.p(x + h) - nu.p(x - h)) / (2.0 * h);
        err_fd = std::max(err_fd, rel(nu.dpdx(x), dp));
    }
    check_local(err_def < 1e-14,
                "massive neutrinos: rho = prefactor a^-4 I_rho(M(x)), pseudo_p (max error " + num(err_def, 2) + ")");
    check_local(err_fd < 1e-7,
                "massive neutrinos: dp/dx = -(5p - pseudo_p) = finite difference of p (max error " + num(err_fd, 2) +
                    ")");
    check_local(
        std::abs(nu.prefactor() - p.Omega_gamma() * std::pow(p.T_ncdm_over_TCMB, 4) * 15.0 / std::pow(M_PI, 4)) < 1e-18,
        "massive neutrinos: prefactor = Omega_gamma (15/pi^4) (T_ncdm/T_CMB)^4");

    // <E/m> = rho / (prefactor a^-4 (3/2) zeta(3) M): 1 + (15/2) zeta(5)/zeta(3) / M^2 when non-relativistic,
    // (7 pi^4 / 180) / zeta(3) / M when relativistic
    const double zeta3 = 1.2020569031595942, zeta5 = 1.0369277551433699;
    double err_E = 0.0;
    for (double x : {-20.0, -8.0, -3.0, 0.0})
        err_E = std::max(err_E,
                         rel(nu.mean_energy_per_mass(x),
                             nu.rho(x) / (nu.prefactor() * std::exp(-4.0 * x) * 1.5 * zeta3 * nu.M_of_x(x))));
    const double Mnr = nu.M_of_x(3.0), Mr = nu.M_of_x(-20.0);
    check_local(err_E < 1e-14 and rel(nu.mean_energy_per_mass(3.0), 1.0 + 7.5 * zeta5 / zeta3 / (Mnr * Mnr)) < 1e-6 and
                    rel(nu.mean_energy_per_mass(-20.0), 7.0 * std::pow(M_PI, 4) / 180.0 / zeta3 / Mr) < 1e-6,
                "massive neutrinos: mean_energy_per_mass = rho / rho_nonrelativistic (and its limits)");

    // p_total
    auto bg = make_lcdm(p);
    double err_p = 0.0;
    for (double x : {-15.0, -5.0, 0.0}) {
        double sum = 0.0;
        for (auto c : all_components)
            if (c != Component::Curvature)
                sum += bg->p(c, x);
        err_p = std::max(err_p, rel(bg->p_total(x), sum));
    }
    check_local(err_p < 1e-14, "p_total = sum of the components' pressures");

    // ScaledSpecies (the Brans-Dicke Gtoday definition rescales the radiation and the massive neutrinos)
    auto jbd = make_jbd(p, 100.0, 1.1, BransDickeDensityDefinition::Gtoday);
    const auto * bd = jbd->field_equations_as<BransDickeFieldEquations>();
    const double phi_star = bd->phi_star(), phi0 = phi_star / 1.1;
    int n_scaled = 0;
    bool ok = true;
    for (const auto & s : jbd->species())
        if (const auto * sc = dynamic_cast<const ScaledSpecies *>(s.get())) {
            n_scaled++;
            ok = ok and sc->factor() == phi_star / phi0 and sc->species_ptr().get() == &sc->species() and
                 sc->rho(-3.0) == sc->factor() * sc->species().rho(-3.0) and
                 sc->species().component() == s->component();
            if (s->component() == Component::MassiveNeutrinos)
                ok = ok and dynamic_cast<const MassiveNeutrinoSpecies *>(&sc->species()) != nullptr;
        }
    check_local(ok and n_scaled == 3, "ScaledSpecies: species() and factor() (photons, massless and massive nu)");
    check_local(throws([] { ScaledSpecies(nullptr, 2.0); }), "ScaledSpecies of nothing throws");

    // Brans-Dicke: the start of the field solution
    check_local(bd->x_start() <= std::min(-25.0, jbd->x_min()) and bd->phi_initial() == bd->phi(bd->x_start()) and
                    bd->phi(bd->x_start() - 5.0) == bd->phi_initial() and bd->dphidx(bd->x_start() - 5.0) == 0.0,
                "Brans-Dicke: x_start, phi_initial (frozen before)");
}

// sin_K, cot_K: flat, open and closed, the series region and its edge, cot_K = d ln sin_K / d chi
void test_geometry() {
    double err = 0.0, err_d = 0.0, jump = 0.0;
    for (double Ok : {0.0, 0.05, -0.05, 1e-7, -1e-7}) {
        auto bg = make_lcdm(no_radiation(0.05, 0.25, Ok));
        const double K = -Ok / (c_over_H0_mpch * c_over_H0_mpch);
        check_local(rel(bg->K_hmpc2(), K) < 1e-15 or (K == 0.0 and bg->K_hmpc2() == 0.0),
                    "K = -Omega_k (H0/c)^2, Omega_k = " + num(Ok));
        for (double chi : {1.0, 100.0, 3000.0, 10000.0}) {
            const double u = K * chi * chi, s = std::sqrt(std::abs(u));
            const double S = u == 0.0 ? chi : (u > 0 ? std::sin(s) : std::sinh(s)) / s * chi;
            const double C = u == 0.0 ? 1.0 / chi : (u > 0 ? s / std::tan(s) : s / std::tanh(s)) / chi;
            err = std::max({err, rel(bg->sinK(chi), S), rel(bg->cotK(chi), C)});
            const double h = 1e-3 * chi;
            err_d =
                std::max(err_d,
                         rel(bg->cotK(chi), (std::log(bg->sinK(chi + h)) - std::log(bg->sinK(chi - h))) / (2.0 * h)));
        }
        if (K != 0.0) {
            // the edge of the series |K chi^2| = 1e-4 (sin_K/chi and chi cot_K vary by ~1e-17 across it)
            const double chi_edge = std::sqrt(1e-4 / std::abs(K)), lo = chi_edge * (1.0 - 1e-12),
                         hi = chi_edge * (1.0 + 1e-12);
            jump =
                std::max({jump, rel(bg->sinK(lo) / lo, bg->sinK(hi) / hi), rel(bg->cotK(lo) * lo, bg->cotK(hi) * hi)});
        }
    }
    check_local(err < 1e-12,
                "sin_K, cot_K = the closed forms (flat, open, closed, series region) (max error " + num(err, 2) + ")");
    check_local(err_d < 1e-6, "cot_K = d ln sin_K / d chi (finite difference, max error " + num(err_d, 2) + ")");
    check_local(jump < 1e-14, "sin_K, cot_K continuous at the edge of the series (max jump " + num(jump, 2) + ")");
}

// Against the solver's stored background (data/solver_background.txt, the block of the label)
void compare_with_solver(const Background & bg, const std::string & label, double tol) {
    double errE = 0.0, errO = 0.0, errd = 0.0;
    for (const auto & r : solver_table("solver_background.txt", label)) {
        const double x = r.x;
        errE = std::max({errE, rel(bg.E(x), r.E), std::abs(bg.dlogEdx(x) - r.dlogEdx)});
        errO = std::max({errO,
                         std::abs(bg.Omega(Component::Baryons, x) - r.Omega_b),
                         std::abs(bg.Omega(Component::CDM, x) - r.Omega_cdm),
                         std::abs(bg.Omega(Component::Photons, x) - r.Omega_gamma),
                         std::abs(bg.Omega(Component::MasslessNeutrinos, x) - r.Omega_nu),
                         std::abs(bg.Omega(Component::MassiveNeutrinos, x) - r.Omega_ncdm),
                         std::abs(bg.Omega(Component::DarkEnergy, x) - r.Omega_de),
                         std::abs(bg.Omega(Component::Curvature, x) - r.Omega_k)});
        errd = std::max({errd, rel(bg.eta(x), r.eta), rel(bg.t_Gyr(x), r.t)});
        if (std::abs(x) > 0.05)
            errd = std::max({errd, rel(bg.chi(x), r.chi), rel(bg.D_A(x), r.D_A)});
    }
    check_local(errE < tol, label + ": E, dlogE/dx = solver (max error " + num(errE, 2) + ")");
    check_local(errO < tol, label + ": Omega_i(x) = solver (max error " + num(errO, 2) + ")");
    check_local(errd < 10 * tol, label + ": eta, t, chi, D_A = solver (max error " + num(errd, 2) + ")");
}

void test_vs_solver() {
    BackgroundParameters p;
    p.h = 0.6736;
    p.Omega_b = 0.02237 / (p.h * p.h);
    p.Omega_cdm = 0.1200 / (p.h * p.h);
    p.m_nu_eV = {0.06};
    compare_with_solver(*make_lcdm(p), "LCDM, 0.06 eV", 1e-6);
    auto pk = p;
    pk.Omega_k = 0.03;
    pk.m_nu_eV = {};
    compare_with_solver(*make_lcdm(pk), "LCDM, Omega_k = 0.03", 1e-6);
    pk.Omega_k = -0.03;
    compare_with_solver(*make_lcdm(pk), "LCDM, Omega_k = -0.03", 1e-6);
    compare_with_solver(*make_background("w0wa", p, {{"w0", -0.9}, {"wa", 0.3}}), "w0wa (-0.9, 0.3)", 1e-6);
    for (int branch : {-1, 1}) {
        const double Orc = branch < 0 ? 0.25 : 0.0;
        compare_with_solver(*make_dgp(p, Orc, branch), branch < 0 ? "nDGP" : "sDGP", 1e-6);
    }
}

void test_tabulated() {
    BackgroundParameters p;
    p.m_nu_eV = {0.1, 0.05};
    p.Omega_k = 0.01;
    auto bg = make_w0wa(p, -0.95, -0.2);
    TabulatedBackground::Tables t;
    for (int i = 0; i <= 6000; i++)
        t.x.push_back(bg->x_min() + (bg->x_max() - bg->x_min()) * i / 6000.0);
    for (double x : t.x) {
        t.E.push_back(bg->E(x));
        for (auto c : all_components)
            t.Omega[c].push_back(bg->Omega(c, x));
    }
    TabulatedBackground tab("tabulated", p, t);
    double err = 0.0;
    for (double x = -20.0; x <= 2.0; x += 0.31) {
        err =
            std::max({err,
                      rel(tab.E(x), bg->E(x)),
                      std::abs(tab.dlogEdx(x) - bg->dlogEdx(x)),
                      rel(tab.t_Gyr(x), bg->t_Gyr(x)),
                      rel(tab.eta(x), bg->eta(x)),
                      std::abs(tab.Omega(Component::MassiveNeutrinos, x) - bg->Omega(Component::MassiveNeutrinos, x))});
        if (std::abs(x) > 0.05)
            err = std::max(err, rel(tab.D_M(x), bg->D_M(x)));
    }
    check_local(err < 1e-6, "TabulatedBackground of a sampled w0wa background (max error " + num(err, 2) + ")");
}

void test_class(const std::string & dir) {
    for (std::string file : {"class_synchronous_00_background.dat", "class_massless_synchronous_00_background.dat"}) {
        const std::string path = dir + "/" + file;
        if (not std::ifstream(path)) {
            std::cout << "[SKIP] CLASS comparison: no [" << path << "]\n";
            continue;
        }
        auto cls = TabulatedBackground::from_class(path);
        const auto & p = cls->parameters();
        const bool massive = not p.m_nu_eV.empty();
        check_local(rel(p.h, 0.67) < 1e-6 and rel(p.Omega_b, 0.0224 / (.67 * .67)) < 1e-4 and
                        rel(p.Omega_cdm, 0.119 / (.67 * .67)) < 1e-4 and rel(p.T_CMB, 2.7255) < 1e-4 and
                        rel(p.N_eff, 3.046) < 2e-3 and (massive ? rel(p.m_nu_eV[0], 0.1) < 1e-3 : true),
                    file + ": parameters inferred (h " + num(p.h, 6) + ", T_CMB " + num(p.T_CMB, 6) + ", N_eff " +
                        num(p.N_eff, 5) + (massive ? ", m_nu " + num(p.m_nu_eV[0], 5) : std::string()) + ")");
        // CLASS's own distances and times against those integrated from its H
        std::ifstream in(path);
        std::string line;
        double err_self = 0.0, err_model = 0.0;
        auto model = make_lcdm(p);
        while (std::getline(in, line)) {
            if (line.empty() or line[0] == '#')
                continue;
            std::istringstream s(line);
            double z, t, eta, H, chi;
            s >> z >> t >> eta >> H >> chi;
            if (z > 1e6 or z < 1e-3)
                continue;
            const double x = -std::log1p(z);
            err_self =
                std::max({err_self, rel(cls->t_Gyr(x), t), rel(cls->eta(x), eta * p.h), rel(cls->chi(x), chi * p.h)});
            err_model = std::max(
                {err_model, rel(model->E(x), cls->E(x)), rel(model->chi(x), chi * p.h), rel(model->t_Gyr(x), t)});
        }
        check_local(err_self < 1e-4,
                    file + ": t, eta, chi from CLASS's H(z) = CLASS (max error " + num(err_self, 2) + ")");
        check_local(err_model < 2e-4,
                    file + ": make_lcdm with these parameters = CLASS (max error " + num(err_model, 2) + ")");
    }
}

void test_errors() {
    BackgroundParameters p;
    check_local(throws([&] { make_background("nope", p); }), "unknown model is an error");
    check_local(throws([&] { make_background("w0wa", p, {{"w1", 0.0}}); }), "unknown model parameter is an error");
    check_local(
        throws([&] { ComponentBackground("open", p, standard_species(p), std::make_shared<GeneralRelativity>()); }),
        "a budget that does not close (E(0) != 1) is an error");
    auto bg = make_lcdm(p);
    check_local(throws([&] { bg->chi(5.0); }) and throws([&] { bg->eta(-30.0); }) and
                    throws([&] { bg->x_of_chi(1e9); }),
                "queries outside the range are errors");
    auto lcdm_reg = make_background("lcdm", p);
    check_local(lcdm_reg->E(-1.0) == bg->E(-1.0) and lcdm_reg->name() == "lcdm", "registry builds the presets");
    p.Omega_k = 0.01;
    check_local(throws([&] { make_dgp(p, 0.25, -1); }), "DGP only flat");
}

int main(int argc, char ** argv) {
    test_einstein_de_sitter();
    test_lcdm_no_radiation();
    test_neutrinos();
    test_species();
    test_geometry();
    test_vs_solver();
    test_tabulated();
    test_class(argc > 1 ? argv[1] : "../../../TransferProviders/tests/data");
    test_errors();
    if (FML::ThisTask == 0) {
        BackgroundParameters p;
        p.m_nu_eV = {0.06};
        make_lcdm(p)->info();
    }
    return FML::TESTS::summary("Background");
}
