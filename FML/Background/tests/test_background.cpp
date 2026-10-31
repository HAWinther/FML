//==========================================================
// Tests of FML/Background:
//   - analytic backgrounds: Einstein-de Sitter (FunctionBackground) and LCDM without radiation (ComponentBackground),
//     open and closed: E, conformal and cosmic time, comoving and transverse distances, x(chi)
//   - the Fermi-Dirac integrals of massive neutrinos and their limits
//   - LCDM with massive neutrinos and curvature, w0wa and DGP against the Einstein-Boltzmann solver's background
//   - TabulatedBackground from a sampled ComponentBackground
//   - TabulatedBackground::from_class and make_lcdm against a CLASS run (if the TransferProviders test data exist:
//     ./test_background <dir>, made with TransferProviders/tests/make_test_data.sh; otherwise SKIP)
//   - errors: unknown models and parameters, an open budget, queries outside the range
//==========================================================
#include <FML/Background/FunctionBackground.h>
#include <FML/Background/Presets.h>
#include <FML/Background/TabulatedBackground.h>

#include <FML/Cosmology/Background/BackgroundCosmology.h>

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
namespace EBS = FML::COSMOLOGY;
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
    FunctionBackground eds("eds", p, [](double x) { return std::exp(-1.5 * x); }, nullptr, nullptr, {std::log(1e-8), 1.0, 200});
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
            err = std::max({err, rel(bg->E(x), E(a)), rel(bg->t_H0(x), t), rel(bg->Omega_m(x), Om / (a * a * a) / (E(a) * E(a)))});
            if (a != 1.0)
                err = std::max({err, rel(bg->chi(x), chi), rel(bg->D_M(x), DM), rel(bg->D_A(x), a * DM),
                                rel(bg->D_L(x), DM / a)});
        }
        if (Ok == 0.0 and OL > 0) {
            // t(a) = 2 / (3 sqrt(OL)) asinh(sqrt(OL/Om) a^1.5)
            const double a = 0.6;
            err = std::max(err, rel(bg->t_H0(std::log(a)), 2.0 / (3.0 * std::sqrt(OL)) * std::asinh(std::sqrt(OL / Om) * std::pow(a, 1.5))));
        }
        check_local(err < 1e-8, "LCDM without radiation, Omega_k = " + num(Ok, 2) +
                                    ": E, Omega_m, t, chi, D_M, D_A, D_L (max error " + num(err, 2) + ")");
    }
}

void test_neutrinos() {
    const auto & fd = FermiDiracIntegrals::get();
    double err = 0.0;
    for (double M : {1e-5, 1e-3, 0.3, 3.0, 30.0, 300.0, 3e4, 1e8}) {
        const auto f0 = [](double q) { return 1.0 / (std::exp(q) + 1.0); };
        const double Ir = simpson([&](double q) { return q * q * std::sqrt(q * q + M * M) * f0(q); }, 0.0, 80.0, 400000);
        const double Ip = simpson([&](double q) { return q * q * q * q / (3.0 * std::sqrt(q * q + M * M)) * f0(q); }, 0.0, 80.0, 400000);
        err = std::max({err, rel(fd.I_rho(M), Ir), rel(fd.I_p(M), Ip)});
    }
    check_local(err < 1e-8, "Fermi-Dirac I_rho, I_p (max error " + num(err, 2) + ")");

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

// The same cosmology in the Einstein-Boltzmann solver
EBS::CosmologyParameters ebs_parameters(const BackgroundParameters & p) {
    EBS::CosmologyParameters e;
    e.h = p.h;
    e.OmegaB = p.Omega_b;
    e.OmegaCDM = p.Omega_cdm;
    e.OmegaK = p.Omega_k;
    e.TCMB = p.T_CMB;
    e.Neff = p.N_eff;
    e.m_nu = p.m_nu_eV.empty() ? 0.0 : p.m_nu_eV[0];
    e.T_ncdm_over_TCMB = p.T_ncdm_over_TCMB;
    return e;
}

void compare_with_ebs(const Background & bg, EBS::CosmologyParameters e, const std::string & label, double tol) {
    EBS::BackgroundCosmology ebs(e);
    ebs.solve(-24.0, 2.0, 20000);
    const double Mpch = FML::UTILS::SI::Mpc / e.h;
    double errE = 0.0, errO = 0.0, errd = 0.0;
    for (double x = -22.0; x <= 1.5; x += 0.37) {
        errE = std::max({errE, rel(bg.E(x), ebs.H_of_x(x) / ebs.get_H0()),
                         std::abs(bg.dlogEdx(x) - (ebs.dHpdx_of_x(x) / ebs.Hp_of_x(x) - 1.0))});
        errO = std::max({errO,
                         std::abs(bg.Omega(Component::Baryons, x) - ebs.OmegaB_of_x(x)),
                         std::abs(bg.Omega(Component::CDM, x) - ebs.OmegaCDM_of_x(x)),
                         std::abs(bg.Omega(Component::Photons, x) - ebs.OmegaGamma_of_x(x)),
                         std::abs(bg.Omega(Component::MasslessNeutrinos, x) - ebs.OmegaNu_of_x(x)),
                         std::abs(bg.Omega(Component::MassiveNeutrinos, x) - ebs.OmegaNcdm_of_x(x)),
                         std::abs(bg.Omega(Component::DarkEnergy, x) - ebs.OmegaDE_of_x(x)),
                         std::abs(bg.Omega(Component::Curvature, x) - ebs.OmegaK_of_x(x))});
        errd = std::max({errd, rel(bg.eta(x), ebs.eta_of_x(x) / Mpch),
                         rel(bg.t_Gyr(x), ebs.t_of_x(x) / FML::UTILS::SI::Gyr)});
        if (std::abs(x) > 0.05)
            errd = std::max({errd, rel(bg.chi(x), ebs.comoving_distance_of_x(x) / Mpch),
                             rel(bg.D_A(x), ebs.angular_distance_of_x(x) / Mpch)});
    }
    check_local(errE < tol, label + ": E, dlogE/dx = solver (max error " + num(errE, 2) + ")");
    check_local(errO < tol, label + ": Omega_i(x) = solver (max error " + num(errO, 2) + ")");
    check_local(errd < 10 * tol, label + ": eta, t, chi, D_A = solver (max error " + num(errd, 2) + ")");
}

void test_vs_ebs() {
    BackgroundParameters p;
    p.h = 0.6736;
    p.Omega_b = 0.02237 / (p.h * p.h);
    p.Omega_cdm = 0.1200 / (p.h * p.h);
    p.m_nu_eV = {0.06};
    compare_with_ebs(*make_lcdm(p), ebs_parameters(p), "LCDM, 0.06 eV", 1e-6);
    auto pk = p;
    pk.Omega_k = 0.03;
    pk.m_nu_eV = {};
    compare_with_ebs(*make_lcdm(pk), ebs_parameters(pk), "LCDM, Omega_k = 0.03", 1e-6);
    pk.Omega_k = -0.03;
    compare_with_ebs(*make_lcdm(pk), ebs_parameters(pk), "LCDM, Omega_k = -0.03", 1e-6);

    auto e = ebs_parameters(p);
    e.model = "w0wa";
    e.model_params = {{"w0", -0.9}, {"wa", 0.3}};
    compare_with_ebs(*make_background("w0wa", p, {{"w0", -0.9}, {"wa", 0.3}}), e, "w0wa (-0.9, 0.3)", 1e-6);

    for (int branch : {-1, 1}) {
        const double Orc = branch < 0 ? 0.25 : 0.0;
        auto d = make_dgp(p, Orc, branch);
        e.model = "dgp";
        e.model_params = {{"Omega_rc", Orc}, {"dgp_branch", double(branch)}};
        compare_with_ebs(*d, e, branch < 0 ? "nDGP" : "sDGP", 1e-6);
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
        err = std::max({err, rel(tab.E(x), bg->E(x)), std::abs(tab.dlogEdx(x) - bg->dlogEdx(x)),
                        rel(tab.t_Gyr(x), bg->t_Gyr(x)), rel(tab.eta(x), bg->eta(x)),
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
            err_self = std::max({err_self, rel(cls->t_Gyr(x), t), rel(cls->eta(x), eta * p.h), rel(cls->chi(x), chi * p.h)});
            err_model = std::max({err_model, rel(model->E(x), cls->E(x)), rel(model->chi(x), chi * p.h),
                                  rel(model->t_Gyr(x), t)});
        }
        check_local(err_self < 1e-4, file + ": t, eta, chi from CLASS's H(z) = CLASS (max error " + num(err_self, 2) + ")");
        check_local(err_model < 2e-4, file + ": make_lcdm with these parameters = CLASS (max error " + num(err_model, 2) + ")");
    }
}

void test_errors() {
    BackgroundParameters p;
    check_local(throws([&] { make_background("nope", p); }), "unknown model is an error");
    check_local(throws([&] { make_background("w0wa", p, {{"w1", 0.0}}); }), "unknown model parameter is an error");
    check_local(throws([&] {
                    ComponentBackground("open", p, standard_species(p), std::make_shared<GeneralRelativity>());
                }),
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
    test_vs_ebs();
    test_tabulated();
    test_class(argc > 1 ? argv[1] : "../../TransferProviders/tests/data");
    test_errors();
    if (FML::ThisTask == 0) {
        BackgroundParameters p;
        p.m_nu_eV = {0.06};
        make_lcdm(p)->info();
    }
    return FML::TESTS::summary("Background");
}
