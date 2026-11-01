//==========================================================
// Tests of the models of Cosmology/Background beyond LCDM, w0wa and DGP (BackgroundModel_BransDicke.h,
// _CoupledQuintessence.h, _CubicGalileon.h, _DarkScattering.h, _IDE.h and the registry entries of the models whose
// background is LCDM or w0wa):
//   - each against the Einstein-Boltzmann solver's model of the same name and parameters before the solver moved onto
//     this module (its old model classes; stored tables, see SolverReference.h): E, dlogE/dx, Omega_i, eta, t, chi,
//     D_A, with and without massive neutrinos
//   - Jordan-Brans-Dicke with GeffG_today != 1 and the density parameter definitions: against SimulationCode's
//     CosmologyJBD, and the physical equivalence of the Gtoday, Gnewton and Gbare definitions
//   - the registry entries (Presets.h registry()) and errors
//==========================================================
#include <FML/Cosmology/Background/BackgroundModel_BransDicke.h>
#include <FML/Cosmology/Background/BackgroundModel_CoupledQuintessence.h>
#include <FML/Cosmology/Background/BackgroundModel_CubicGalileon.h>
#include <FML/Cosmology/Background/BackgroundModel_DarkScattering.h>
#include <FML/Cosmology/Background/BackgroundModel_IDE.h>
#include <FML/Cosmology/Background/Presets.h>

#include "SolverReference.h"
#include <FML/SimulationCode/src/Cosmology_JBD.h>

#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
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
using ModelParameters = std::map<std::string, double>;

// ---------------------------------------------------------------------------------------------------------------------
// Against the Einstein-Boltzmann solver (its stored backgrounds, data/solver_models.txt)
// ---------------------------------------------------------------------------------------------------------------------
BackgroundParameters planck(double m_nu = 0.0, double Ok = 0.0) {
    BackgroundParameters p;
    p.h = 0.6736;
    p.Omega_b = 0.02237 / (p.h * p.h);
    p.Omega_cdm = 0.1200 / (p.h * p.h);
    p.Omega_k = Ok;
    if (m_nu > 0.0)
        p.m_nu_eV = {m_nu};
    return p;
}

// The model by name against the solver's, compared at x in [-22, 1.5]. eta_offset: eta and t are compared up to a
// constant (the solver started them with the GR radiation-era values, which do not hold for Brans-Dicke, where
// G/phi_ini != G)
void compare_with_solver(const std::string & model,
                         const BackgroundParameters & p,
                         const ModelParameters & mp,
                         bool eta_offset = false,
                         double tol = 1e-6) {
    // (the solver's model also had the parameters of the perturbations, which do not change the background)
    ModelParameters background_mp;
    for (const auto & q : find_model(model).parameters)
        if (mp.count(q.name))
            background_mp[q.name] = mp.at(q.name);
    auto bg = make_background(model, p, background_mp);
    std::ostringstream label;
    label << model;
    for (const auto & [k, v] : mp)
        label << " " << k << "=" << v;
    if (not p.m_nu_eV.empty())
        label << " m_nu=" << p.m_nu_eV[0];
    if (p.Omega_k != 0.0)
        label << " Omega_k=" << p.Omega_k;
    const auto table = solver_table("solver_models.txt", label.str());
    double errE = 0.0, errO = 0.0, errd = 0.0;
    // (the first row is x = -22)
    const double d_eta = eta_offset ? bg->eta(-22.0) - table.front().eta : 0.0;
    const double d_t = eta_offset ? bg->t_Gyr(-22.0) - table.front().t : 0.0;
    for (const auto & r : table) {
        const double x = r.x;
        errE = std::max({errE, rel(bg->E(x), r.E), std::abs(bg->dlogEdx(x) - r.dlogEdx)});
        errO = std::max({errO,
                         std::abs(bg->Omega(Component::Baryons, x) - r.Omega_b),
                         std::abs(bg->Omega(Component::CDM, x) - r.Omega_cdm),
                         std::abs(bg->Omega(Component::Photons, x) - r.Omega_gamma),
                         std::abs(bg->Omega(Component::MasslessNeutrinos, x) - r.Omega_nu),
                         std::abs(bg->Omega(Component::MassiveNeutrinos, x) - r.Omega_ncdm),
                         std::abs(bg->Omega(Component::DarkEnergy, x) - r.Omega_de),
                         std::abs(bg->Omega(Component::Curvature, x) - r.Omega_k),
                         std::abs(bg->Omega(Component::Other, x))});
        errd = std::max({errd, rel(bg->eta(x) - d_eta, r.eta), rel(bg->t_Gyr(x) - d_t, r.t)});
        if (std::abs(x) > 0.05)
            errd = std::max({errd, rel(bg->chi(x), r.chi), rel(bg->D_A(x), r.D_A)});
    }
    check_local(errE < tol and errO < tol and errd < tol,
                label.str() + ": = solver (max error E, dlogE/dx " + num(errE, 2) + ", Omega_i " + num(errO, 2) +
                    ", eta, t, chi, D_A " + num(errd, 2) +
                    (eta_offset ? "; eta, t up to the solver's GR initial values, offset in eta(x_min) " +
                                      num(d_eta / bg->eta(-22.0), 2) + " relative" :
                                  std::string()) +
                    ")");
}

void test_vs_solver() {
    compare_with_solver("jbd", planck(0.06), {{"bd_omega", 1000.0}}, true);
    compare_with_solver("jbd", planck(), {{"bd_omega", 50.0}}, true);
    compare_with_solver("jbd", planck(0.1), {{"bd_omega", 10.0}}, true);
    compare_with_solver("coupled_quintessence", planck(0.06), {{"cq_lambda", 0.5}, {"cq_beta", 0.05}});
    compare_with_solver("coupled_quintessence", planck(), {{"cq_lambda", 1.2}, {"cq_beta", 0.15}});
    compare_with_solver("coupled_quintessence", planck(0.0, 0.02), {{"cq_lambda", 0.8}, {"cq_beta", -0.1}});
    compare_with_solver("cubic_galileon", planck(), {});
    compare_with_solver("cubic_galileon", planck(0.06), {});
    compare_with_solver("dark_scattering", planck(0.06), {{"ds_w", -0.9}, {"ds_xi", 10.0}});
    compare_with_solver("dark_scattering", planck(0.0, -0.02), {{"ds_w", -1.1}, {"ds_xi", 0.0}});
    compare_with_solver("ide", planck(0.06), {{"ide_w", -0.95}, {"ide_xi", -0.1}});
    compare_with_solver("ide", planck(), {{"ide_w", -1.05}, {"ide_xi", 0.1}});
    compare_with_solver("ide", planck(0.0, 0.02), {{"ide_w", -0.8}, {"ide_xi", -0.3}});
    compare_with_solver("musigma", planck(0.06), {{"mu0", 0.3}, {"Sigma0", -0.1}});
    compare_with_solver("fr", planck(0.06), {{"fR0", 1e-5}, {"fR_n", 2.0}});
    compare_with_solver("horndeski", planck(), {{"hd_cB", 0.3}, {"hd_stability_check", 0.0}});
    compare_with_solver("horndeski",
                        planck(0.06),
                        {{"hd_w0", -0.9}, {"hd_wa", 0.2}, {"hd_cM", 0.1}, {"hd_stability_check", 0.0}});
}

// ---------------------------------------------------------------------------------------------------------------------
// Jordan-Brans-Dicke: GeffG_today and the density parameter definitions
// ---------------------------------------------------------------------------------------------------------------------
const BransDickeFieldEquations & bd_of(const ComponentBackground & bg) {
    return *bg.field_equations_as<BransDickeFieldEquations>();
}

// SimulationCode's CosmologyJBD (massless neutrinos: its massive ones are specified by OmegaMNu and have other
// conventions) against make_jbd. Returns the max error in E, dlogE/dx, phi, Omega_b, Omega_gamma, Omega_Lambda
double
compare_with_simcode(const BackgroundParameters & p, double omega, double GeffG, BransDickeDensityDefinition def) {
    static const char * names[] = {"hi-class", "Gtoday", "Gnewton", "Gbare"};
    FML::UTILS::ParameterMap pm;
    pm["cosmology_OmegaMNu"] = 0.0;
    pm["cosmology_Omegab"] = p.Omega_b;
    pm["cosmology_OmegaCDM"] = p.Omega_cdm;
    pm["cosmology_OmegaK"] = p.Omega_k;
    pm["cosmology_h"] = p.h;
    pm["cosmology_As"] = 2.1e-9;
    pm["cosmology_ns"] = 0.965;
    pm["cosmology_kpivot_mpc"] = 0.05;
    pm["cosmology_Neffective"] = p.N_eff;
    pm["cosmology_TCMB_kelvin"] = p.T_CMB;
    pm["cosmology_JBD_wBD"] = omega;
    pm["cosmology_JBD_GeffG_today"] = GeffG;
    pm["cosmology_JBD_density_parameter_definition"] = std::string(names[int(def)]);
    CosmologyJBD sc;
    sc.read_parameters(pm);
    {
        std::ostringstream quiet;
        auto * old = std::cout.rdbuf(quiet.rdbuf());
        sc.init();
        std::cout.rdbuf(old);
    }
    auto bg = make_jbd(p, omega, GeffG, def);
    const auto & bd = bd_of(*bg);
    double err = 0.0;
    for (double x = -20.0; x <= 2.0; x += 0.29) {
        const double a = std::exp(x);
        err = std::max({err,
                        rel(bg->E(x), sc.HoverH0_of_a(a)),
                        std::abs(bg->dlogEdx(x) - sc.dlogHdloga_of_a(a)),
                        rel(bd.phi(x), sc.phi_of_a(a)),
                        std::abs(bg->Omega(Component::Baryons, x) - sc.get_Omegab(a)),
                        std::abs(bg->Omega(Component::Photons, x) - sc.get_OmegaR(a)),
                        std::abs(bg->Omega(Component::DarkEnergy, x) - sc.get_OmegaLambda(a))});
    }
    return err;
}

void test_jbd_options() {
    using D = BransDickeDensityDefinition;
    // Today's values: phi(0) = phi_* / GeffG_today, E(0) = 1
    for (auto def : {D::HiClass, D::Gtoday, D::Gnewton, D::Gbare}) {
        auto bg = make_jbd(planck(0.06, 0.01), 100.0, 0.95, def);
        const auto & bd = bd_of(*bg);
        check_local(rel(bd.phi(0.0), bd.phi_star() / 0.95) < 1e-12 and std::abs(bg->E(0.0) - 1.0) < 1e-12,
                    "jbd definition " + std::to_string(int(def)) +
                        ", GeffG_today = 0.95, Omega_k = 0.01: phi(0) = "
                        "phi_*/GeffG_today, E(0) = 1");
    }

    // SimulationCode (its closed universes do not converge for Omega_k <~ -0.03)
    const auto p = planck();
    for (auto [def, GeffG, omega, Ok] :
         std::vector<std::tuple<D, double, double, double>>{{D::HiClass, 1.0, 1000.0, 0.0},
                                                            {D::HiClass, 0.95, 100.0, 0.0},
                                                            {D::Gbare, 1.05, 100.0, 0.0},
                                                            {D::HiClass, 0.95, 100.0, 0.03},
                                                            {D::Gbare, 1.0, 50.0, -0.01},
                                                            {D::Gtoday, 1.0, 1000.0, 0.0},
                                                            {D::Gtoday, 0.95, 100.0, 0.02},
                                                            {D::Gnewton, 0.95, 100.0, 0.0}}) {
        auto pk = p;
        pk.Omega_k = Ok;
        const double err = compare_with_simcode(pk, omega, GeffG, def);
        check_local(err < 1e-5,
                    "jbd = SimulationCode's CosmologyJBD (definition " + std::to_string(int(def)) +
                        ", omega = " + num(omega, 4) + ", GeffG_today = " + num(GeffG, 3) +
                        ", Omega_k = " + num(Ok, 2) + "; max error E, dlogE/dx, phi, Omega_i " + num(err, 2) + ")");
    }
    // Gtoday, Gnewton and Gbare are the same physics: Omega_i^bare = (G/G_*) Omega_i. E(x) and phi(x) agree, the
    // Omega_i(x) differ by G/G_*
    for (double GeffG : {1.0, 0.9}) {
        for (auto def : {D::Gtoday, D::Gnewton}) {
            const double omega = 20.0;
            auto pd = planck(0.1, 0.02);
            auto ref = make_jbd(pd, omega, GeffG, def);
            const double g = bd_of(*ref).G_over_Gstar();
            auto pb = pd;
            pb.Omega_b *= g;
            pb.Omega_cdm *= g;
            pb.Omega_k = pd.Omega_k; // curvature is not a density
            auto bare = make_jbd(pb, omega, GeffG, D::Gbare);
            double err = 0.0;
            for (double x = -22.0; x <= 2.0; x += 0.31)
                for (auto c : {Component::Baryons,
                               Component::CDM,
                               Component::Photons,
                               Component::MasslessNeutrinos,
                               Component::MassiveNeutrinos,
                               Component::DarkEnergy})
                    err = std::max({err,
                                    rel(bare->E(x), ref->E(x)),
                                    std::abs(bare->dlogEdx(x) - ref->dlogEdx(x)),
                                    rel(bd_of(*bare).phi(x), bd_of(*ref).phi(x)),
                                    std::abs(bare->Omega(c, x) - g * ref->Omega(c, x)),
                                    rel(bare->Omega(Component::Curvature, x), ref->Omega(Component::Curvature, x)),
                                    rel(bare->eta(x), ref->eta(x))});
            check_local(err < 1e-8,
                        "jbd definition " + std::to_string(int(def)) + " = Gbare with Omega_i x G/G_* = " + num(g, 6) +
                            " (GeffG_today = " + num(GeffG, 2) +
                            ", massive nu, Omega_k; max "
                            "error " +
                            num(err, 2) + ")");
        }
    }
    // Gtoday = Gnewton for GeffG_today = 1
    auto a = make_jbd(planck(0.06), 50.0, 1.0, D::Gtoday), b = make_jbd(planck(0.06), 50.0, 1.0, D::Gnewton);
    check_local(rel(a->E(-3.0), b->E(-3.0)) < 1e-12 and
                    rel(a->Omega0(Component::CDM), b->Omega0(Component::CDM)) < 1e-12,
                "jbd: Gtoday = Gnewton for GeffG_today = 1");
    // GR limit
    auto gr = make_lcdm(planck(0.06)), bd = make_jbd(planck(0.06), 1e8);
    double err = 0.0;
    for (double x = -22.0; x <= 2.0; x += 0.31)
        err = std::max(err, rel(bd->E(x), gr->E(x)));
    check_local(err < 1e-7, "jbd -> LCDM for omega = 1e8 (max error in E " + num(err, 2) + ")");
}

// ---------------------------------------------------------------------------------------------------------------------
// Limits, registry and errors
// ---------------------------------------------------------------------------------------------------------------------
void test_limits_and_errors() {
    const auto p = planck(0.06);
    auto lcdm = make_lcdm(p);
    const auto max_diff = [&](const Background & bg) {
        double err = 0.0;
        for (double x = -22.0; x <= 2.0; x += 0.31)
            err = std::max({err, rel(bg.E(x), lcdm->E(x)), std::abs(bg.dlogEdx(x) - lcdm->dlogEdx(x))});
        return err;
    };
    const double e_cq = max_diff(*make_coupled_quintessence(p, 0.0, 0.0));
    const double e_ide = max_diff(*make_ide(p, -1.0, 0.0)), e_ds = max_diff(*make_dark_scattering(p, -1.0));
    check_local(e_cq < 1e-8 and e_ide < 1e-12 and e_ds < 1e-12,
                "coupled quintessence (lambda = beta = 0), ide and dark scattering (w = -1, xi = 0) = LCDM (" +
                    num(e_cq, 2) + ", " + num(e_ide, 2) + ", " + num(e_ds, 2) + ")");
    // The coupled CDM and field: drho/dx against finite differences, the total exchange cancels
    auto cq = make_coupled_quintessence(p, 1.0, 0.2);
    double err = 0.0;
    for (double x = -15.0; x <= 1.0; x += 0.53)
        for (const auto & s : cq->species()) {
            const double h = 1e-4, fd = (s->rho(x + h) - s->rho(x - h)) / (2.0 * h);
            err = std::max(err, std::abs(s->drhodx(x) - fd) / std::max(std::abs(s->rho(x)), 1e-300));
        }
    check_local(err < 1e-6,
                "coupled quintessence: drho/dx of every species = finite difference (max " + num(err, 2) + ")");
    // Registry
    check_local(make_background("dark_scattering", p)->E(-1.0) == make_dark_scattering(p, -0.9)->E(-1.0) and
                    make_background("fr", p)->E(-1.0) == lcdm->E(-1.0) and
                    make_background("horndeski", p, {{"hd_wa", 0.1}})->E(-1.0) == make_w0wa(p, -1.0, 0.1)->E(-1.0),
                "registry entries build the presets (defaults)");
    check_local(throws([&] { make_background("jbd", p, {{"bd_density_parameter_definition", 4.0}}); }) and
                    throws([&] { make_background("jbd", p, {{"bd_omega", -2.0}}); }) and
                    throws([&] { make_background("ide", p, {{"ds_w", -0.9}}); }) and
                    throws([&] { make_background("dgp", p, {{"dgp_branch", 0.5}}); }),
                "jbd: invalid definition and omega are errors; dgp_branch != +-1 and unknown parameters are errors");
    // Every parameter of the registry is described; the parameters that only concern the perturbations (the
    // perturbation registry of Cosmology/Perturbations) are not those of the background
    bool described = true;
    for (const auto & m : registry())
        for (const auto & q : m.parameters)
            described = described and not q.description.empty();
    check_local(described and throws([&] { make_background("dark_scattering", p, {{"ds_xi", 1.0}}); }) and
                    throws([&] { make_background("musigma", p, {{"mu0", 0.1}}); }) and
                    throws([&] { make_background("horndeski", p, {{"hd_cK", 0.1}}); }),
                "registry: every parameter has a description; perturbation parameters (ds_xi, mu0, hd_cK) are not "
                "background parameters");
    check_local(throws([&] { make_cubic_galileon(planck(0.0, 0.01)); }), "cubic Galileon only flat");
    check_local(throws([&] { make_ide(p, -0.9, 3.0); }), "ide: rho_cdm < 0 before today is an error");
    // A custom range is covered by the field solutions
    BackgroundRange r{std::log(1e-12), std::log(30.0), 100};
    auto wide = make_jbd(p, 100.0, 1.0, BransDickeDensityDefinition::HiClass, r);
    auto wide_cq = make_coupled_quintessence(p, 0.5, 0.05, r);
    check_local(std::isfinite(wide->eta(r.x_max)) and std::isfinite(wide_cq->eta(r.x_max)) and
                    rel(wide->E(-2.0), make_jbd(p, 100.0)->E(-2.0)) < 1e-8,
                "jbd and coupled quintessence on a wider range a in [1e-12, 30]");
}

int main() {
    test_vs_solver();
    test_jbd_options();
    test_limits_and_errors();
    if (FML::ThisTask == 0) {
        make_jbd(planck(0.06), 1000.0)->info();
        make_coupled_quintessence(planck(0.06), 0.5, 0.05)->info();
    }
    return FML::TESTS::summary("Background models");
}
