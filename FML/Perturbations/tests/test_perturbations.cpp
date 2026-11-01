//==========================================================
// Tests of FML/Perturbations against the Einstein-Boltzmann solver's own models (Cosmology/Models: its Species and
// Gravity, as an explicit Model of BackgroundCosmology; the solver itself runs the new classes for lcdm and jbd),
// equation by equation. For LCDM (m_nu = 0, 0.06, 0.1 eV; Omega_k = 0, +0.02, -0.02) and
// Jordan-Brans-Dicke (bd_omega = 100, 1000) the solver's model and the new one are built with the same parameters,
// and the solver's species / gravity perturbations (from a Perturbations object) are compared with the new
// SpeciesPerturbation / PerturbedFieldEquation objects:
//   - names, n_vars, cache_size, prefers_explicit, output names
//   - for k from 1e-4 to 5 /Mpc and x from -17 to 0, each side with a ModeAtX filled from its own background:
//     the ModeAtX itself, switch points, caches, initial conditions (adiabatic and isocurvature), moments, rhs,
//     mode switch and outputs with fixed state vectors and metric; the field equations' G_eff_initial, initial
//     conditions, cache, metric (fixed variables and moments), outputs and tensor coefficients
// The tolerance is the level at which the two backgrounds agree (1e-7), except for the species quantities that depend
// on the massive-neutrino pressure and pseudo-pressure (cache, moments, rhs, outputs, switch_mode: 3e-7): the solver
// computes I_p and I_pp with a 48-node Gauss-Laguerre quadrature, which is off by up to 7e-8 near M ~ 0.3 (the new
// background's tabulated integrals agree with direct integration to 1e-10), and c_a^2 - w amplifies it.
// Also: the free-streaming density, neutrinos = false and Omega_nu = 0, the jbd density definitions (rescaled massive
// neutrinos), the registry and errors.
//==========================================================
#include <FML/Perturbations/Presets.h>

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::check_local;
using FML::TESTS::num;
namespace EBS = FML::COSMOLOGY;
namespace BG = FML::COSMOLOGY::BACKGROUND;
namespace PT = FML::COSMOLOGY::PERTURBATIONS;
using ModelParameters = std::map<std::string, double>;

const double Mpc = FML::UTILS::SI::Mpc, c = FML::UTILS::SI::c;

template <class F>
bool throws(F && f) {
    try {
        f();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------------------------------

/// 2 |a - b| / (|a| + |b|) (0 if both are 0, infinite if not finite)
double rel(double a, double b) {
    if (not std::isfinite(a) or not std::isfinite(b))
        return std::numeric_limits<double>::infinity();
    const double s = std::abs(a) + std::abs(b);
    return s > 0.0 ? 2.0 * std::abs(a - b) / s : 0.0;
}
/// max |a_i - b_i| / max |a_i| (relative to the largest element)
double vec_err(const std::vector<double> & a, const std::vector<double> & b) {
    if (a.size() != b.size())
        return std::numeric_limits<double>::infinity();
    double s = 0.0, d = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        if (not std::isfinite(a[i]) or not std::isfinite(b[i]))
            return std::numeric_limits<double>::infinity();
        s = std::max(s, std::abs(a[i]));
        d = std::max(d, std::abs(a[i] - b[i]));
    }
    return s > 0.0 ? d / s : d;
}

/// max_i rel(a_i, b_i) (each element relative to itself: for quantities of different units)
double elem_err(const std::vector<double> & a, const std::vector<double> & b) {
    if (a.size() != b.size())
        return std::numeric_limits<double>::infinity();
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        d = std::max(d, rel(a[i], b[i]));
    return d;
}

struct MaxErrors {
    std::map<std::string, double> e;
    std::vector<std::string> order;
    void add(const std::string & what, double v) {
        if (not e.count(what)) {
            e[what] = 0.0;
            order.push_back(what);
        }
        if (not(v >= 0.0) or not std::isfinite(v))
            e[what] = std::numeric_limits<double>::infinity();
        else
            e[what] = std::max(e[what], v);
    }
};

double moments_err(const EBS::Moments & a, const PT::Moments & b) {
    return std::max({rel(a.drho, b.drho), rel(a.dp, b.dp), rel(a.flux, b.flux), rel(a.shear, b.shear)});
}

// ---------------------------------------------------------------------------------------------------------------------
// The two sides
// ---------------------------------------------------------------------------------------------------------------------

BG::BackgroundParameters planck(double m_nu = 0.0, double Ok = 0.0) {
    BG::BackgroundParameters p;
    p.h = 0.6736;
    p.Omega_b = 0.02237 / (p.h * p.h);
    p.Omega_cdm = 0.1200 / (p.h * p.h);
    p.Omega_k = Ok;
    if (m_nu > 0.0)
        p.m_nu_eV = {m_nu};
    return p;
}

EBS::CosmologyParameters ebs_parameters(const BG::BackgroundParameters & p) {
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

/// The solver's model with its Perturbations object (no solve: only its species and gravity are used)
struct EBSSide {
    EBS::CosmologyParameters params;
    std::unique_ptr<EBS::BackgroundCosmology> cosmo;
    std::unique_ptr<EBS::RecombinationHistory> rec;
    std::unique_ptr<EBS::Perturbations> pert;
    EBSSide(const std::string & model,
            const BG::BackgroundParameters & p,
            const ModelParameters & mp,
            EBS::PerturbationOptions opt = {}) {
        params = ebs_parameters(p);
        params.model = model;
        params.model_params = mp;
        cosmo = std::make_unique<EBS::BackgroundCosmology>(params, EBS::Models::build(params)); // the solver's own model
        cosmo->solve(-24.0, 2.0, 20000);
        rec = std::make_unique<EBS::RecombinationHistory>(*cosmo);
        pert = std::make_unique<EBS::Perturbations>(*cosmo, *rec, opt);
    }
    /// Added to the solver's conformal time (see compare)
    double eta_offset = 0.0;
    /// The ModeAtX as Perturbations::update fills it (recombination quantities 0)
    EBS::ModeAtX mode(double x, double k) const {
        EBS::ModeAtX b;
        const double K = cosmo->get_K(), H0 = cosmo->get_H0();
        b.k = k;
        b.s2 = pert->s_ell(k, 2);
        b.curv = 1.0 - 3.0 * K / (k * k);
        b.x = x;
        b.a = std::exp(x);
        b.Hp = cosmo->Hp_of_x(x);
        b.dHpH = cosmo->dHpdx_of_x(x) / b.Hp;
        b.ckH = c * k / b.Hp;
        b.dtau = b.ddtau = b.R = 0.0;
        b.eta = cosmo->eta_of_x(x) + eta_offset;
        b.cotK_H = c * cosmo->cotK(b.eta) / b.Hp;
        b.H0a2 = H0 * H0 * b.a * b.a;
        b.H0a2_rho_b = b.H0a2 * cosmo->rhoB(x);
        b.H0a2_rho_gamma = b.H0a2 * cosmo->rhoGamma(x);
        return b;
    }
};

// (the solver's ModeAtX is the same type as PT::ModeAtX)
std::vector<double> mode_fields(const PT::ModeAtX & b) {
    return {b.s2, b.curv, b.a, b.Hp, b.dHpH, b.ckH, b.eta, b.cotK_H, b.H0a2, b.H0a2_rho_b, b.H0a2_rho_gamma};
}

/// A fixed state vector of n variables (O(1), no zeros)
std::vector<double> state(int n, double phase = 0.0) {
    std::vector<double> y(n);
    for (int i = 0; i < n; i++)
        y[i] = 0.5 + std::sin(1.3 * i + 0.7 + phase);
    return y;
}

// ---------------------------------------------------------------------------------------------------------------------
// The comparison of one model
// ---------------------------------------------------------------------------------------------------------------------

// eta_offset: the solver starts its conformal time with the GR radiation-era value, which does not hold for
// Brans-Dicke (G/phi_ini != G), so its eta is off by a constant there (as in Background/tests/test_models); the
// solver's ModeAtX then gets eta + (eta_new - eta_solver)(x = -22), and its switch points (computed by the solver from
// its own eta) are compared with the tolerance tol_switch
void compare(const std::string & model,
             const BG::BackgroundParameters & p,
             const ModelParameters & mp,
             double tol,
             bool eta_offset = false,
             double tol_switch = 1e-7) {
    std::ostringstream label;
    label << model;
    for (const auto & [key, v] : mp)
        label << " " << key << "=" << v;
    label << " m_nu=" << (p.m_nu_eV.empty() ? 0.0 : p.m_nu_eV[0]);
    if (p.Omega_k != 0.0)
        label << " Omega_k=" << p.Omega_k;
    const std::string L = label.str() + ": ";

    EBSSide ebs(model, p, mp);
    auto pm = PT::make_perturbation_model(model, p, mp);
    const PT::PerturbationContext ctx = pm->context();
    const auto species = pm->species_perturbations(ctx);
    const auto fe = pm->field_equation_perturbations(ctx);
    const auto & ebs_species = ebs.pert->get_species();
    const auto & gravity = ebs.cosmo->gravity();
    if (eta_offset)
        ebs.eta_offset = ctx.eta(-22.0) - ebs.cosmo->eta_of_x(-22.0);

    // ---- Structure ------------------------------------------------------------------------------------------------
    bool same = species.size() == ebs_species.size();
    std::string names;
    for (size_t i = 0; same and i < species.size(); i++) {
        const auto & a = *ebs_species[i];
        const auto & b = *species[i];
        names += (i ? ", " : "") + b.name();
        same = same and a.name() == b.name() and a.info() == b.info() and a.cache_size() == b.cache_size() and
               a.output_names() == b.output_names();
        for (int mode = 0; mode < 2; mode++)
            same = same and a.n_vars(mode) == b.n_vars(mode) and a.prefers_explicit(mode) == b.prefers_explicit(mode);
    }
    same = same and fe->n_vars() == gravity.n_vars() and fe->cache_size() == gravity.cache_size() and
           fe->output_names() == gravity.output_names() and fe->name() == gravity.name();
    check_local(same,
                L + "species (" + names + ") and field equations (" + fe->name() +
                    "): names, n_vars, cache_size, prefers_explicit, output names = solver");
    if (not same)
        return;

    const std::vector<double> ks{1e-4 / Mpc, 2e-3 / Mpc, 0.05 / Mpc, 0.5 / Mpc, 5.0 / Mpc};
    const std::vector<double> xs{-17.0, -12.0, -8.0, -5.0, -2.0, 0.0};
    MaxErrors err;

    // ---- Switch points ---------------------------------------------------------------------------------------------
    for (double k : ks)
        for (size_t i = 0; i < species.size(); i++) {
            const auto a = ebs_species[i]->switch_points(k), b = species[i]->switch_points(k);
            double e = a.size() == b.size() ? 0.0 : std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < a.size() and j < b.size(); j++)
                e = std::max(e, std::abs(a[j] - b[j]));
            err.add("switch points [abs in x]", e);
        }

    // ---- Free-streaming density (f_nu, OmNu of the initial conditions)
    // -----------------------------------------------
    const double x_ini = -18.0;
    double OmNu_ebs = 0.0, OmNu_new = 0.0;
    for (size_t i = 0; i < ebs.cosmo->species().size(); i++) {
        const auto & s = ebs.cosmo->species()[i];
        if (s->free_streaming() and s->perturbations(*ebs.pert))
            OmNu_ebs += s->rho(x_ini) * std::exp(4.0 * x_ini);
    }
    for (const auto & s : species)
        OmNu_new += s->free_streaming_density(x_ini) * std::exp(4.0 * x_ini);
    err.add("free-streaming density", rel(OmNu_ebs, OmNu_new));
    err.add("G_eff_initial", rel(gravity.G_eff_initial(), fe->G_eff_initial()));

    const double OmG = ebs.cosmo->rhoGamma(0.0);
    EBS::InitialData ic_ebs;
    ic_ebs.f_nu = OmNu_ebs > 0.0 ? OmNu_ebs / (OmG + OmNu_ebs) : 0.0;
    ic_ebs.OmNu = OmNu_ebs;
    ic_ebs.H0 = ebs.cosmo->get_H0();
    ic_ebs.G_eff = gravity.G_eff_initial();
    ic_ebs.Psi = -1.0 / (1.5 + 0.4 * ic_ebs.f_nu);
    ic_ebs.Phi = -(1.0 + 0.4 * ic_ebs.f_nu) * ic_ebs.Psi;
    PT::InitialData ic_new;
    ic_new.f_nu = OmNu_new > 0.0 ? OmNu_new / (OmG + OmNu_new) : 0.0;
    ic_new.OmNu = OmNu_new;
    ic_new.H0 = ctx.H0;
    ic_new.G_eff = fe->G_eff_initial();
    ic_new.Psi = -1.0 / (1.5 + 0.4 * ic_new.f_nu);
    ic_new.Phi = -(1.0 + 0.4 * ic_new.f_nu) * ic_new.Psi;
    EBS::IsoValues iso_ebs;
    iso_ebs.delta_c = 0.8;
    iso_ebs.v_c = -0.03;
    iso_ebs.N0 = 0.2;
    iso_ebs.N1 = -0.1;
    iso_ebs.N2 = 0.05;
    PT::IsoValues iso_new;
    iso_new.delta_c = 0.8;
    iso_new.v_c = -0.03;
    iso_new.N0 = 0.2;
    iso_new.N1 = -0.1;
    iso_new.N2 = 0.05;

    for (double k : ks) {
        for (double x : xs) {
            const EBS::ModeAtX be = ebs.mode(x, k);
            PT::ModeAtX bn = ctx.mode_at_x(x, k);
            bn.dtau = bn.ddtau = bn.R = 0.0;
            err.add("ModeAtX", elem_err(mode_fields(be), mode_fields(bn)));

            const EBS::MetricAtX me{0.3, -0.4, 0.02};
            const PT::MetricAtX mn{0.3, -0.4, 0.02};
            for (size_t i = 0; i < species.size(); i++) {
                const auto & a = *ebs_species[i];
                const auto & b = *species[i];
                std::vector<double> ca(a.cache_size()), cb(b.cache_size());
                a.update_cache(be, ca.data());
                b.update_cache(bn, cb.data());
                err.add("cache", elem_err(ca, cb));

                // Initial conditions (adiabatic and isocurvature)
                std::vector<double> ya(a.n_vars(0)), yb(b.n_vars(0));
                a.initial_conditions(be, ic_ebs, ya.data());
                b.initial_conditions(bn, ic_new, yb.data());
                err.add("initial conditions", vec_err(ya, yb));
                EBS::InitialData ie = ic_ebs;
                PT::InitialData in = ic_new;
                ie.iso = &iso_ebs;
                in.iso = &iso_new;
                a.initial_conditions(be, ie, ya.data());
                b.initial_conditions(bn, in, yb.data());
                err.add("initial conditions (isocurvature)", vec_err(ya, yb));

                const int n_modes = b.name() == "massive neutrinos" ? 2 : 1;
                for (int mode = 0; mode < n_modes; mode++) {
                    const auto y = state(b.n_vars(mode), 0.1 * mode);
                    err.add("moments",
                            moments_err(a.moments(be, ca.data(), mode, y.data()),
                                        b.moments(bn, cb.data(), mode, y.data())));
                    std::vector<double> da(y.size()), db(y.size());
                    a.rhs(be, ca.data(), mode, me, y.data(), da.data());
                    b.rhs(bn, cb.data(), mode, mn, y.data(), db.data());
                    err.add("rhs", vec_err(da, db));
                    std::vector<double> oa(a.output_names().size()), ob(oa.size());
                    a.outputs(be, ca.data(), mode, y.data(), oa.data());
                    b.outputs(bn, cb.data(), mode, y.data(), ob.data());
                    err.add("outputs", vec_err(oa, ob));
                    if (mode == 1) {
                        const auto y0 = state(b.n_vars(0));
                        std::vector<double> Fa(3), Fb(3);
                        a.switch_mode(be, ca.data(), 1, y0.data(), Fa.data());
                        b.switch_mode(bn, cb.data(), 1, y0.data(), Fb.data());
                        err.add("switch_mode", vec_err(Fa, Fb));
                    }
                }
            }

            // ---- Field equations ----------------------------------------------------------------------------------
            std::vector<double> ga(gravity.cache_size()), gb(fe->cache_size());
            gravity.update_cache(*ebs.cosmo, be, ga.data());
            fe->update_cache(bn, gb.data());
            err.add("field equations: cache", elem_err(ga, gb));
            const int nv = fe->n_vars();
            std::vector<double> ya(nv), yb(nv);
            gravity.initial_conditions(be, ic_ebs, ya.data());
            fe->initial_conditions(bn, ic_new, yb.data());
            err.add("field equations: initial conditions", vec_err(ya, yb));

            // Fixed variables and moments of realistic size
            std::vector<double> y{0.7, 1e-3, -2e-3};
            y.resize(nv);
            const double Hp2 = be.Hp * be.Hp, k2c2 = c * c * k * k;
            EBS::Moments te;
            te.drho = 0.3 * Hp2;
            te.dp = 0.05 * Hp2;
            te.flux = 0.1 * Hp2;
            te.shear = 0.01 * k2c2;
            PT::Moments tn;
            tn.drho = te.drho;
            tn.dp = te.dp;
            tn.flux = te.flux;
            tn.shear = te.shear;
            std::vector<double> da(nv), db(nv);
            const auto ma = gravity.metric(be, ga.data(), y.data(), te, da.data());
            const auto mb = fe->metric(bn, gb.data(), y.data(), tn, db.data());
            err.add("field equations: metric", vec_err({ma.Phi, ma.Psi, ma.dPhi}, {mb.Phi, mb.Psi, mb.dPhi}));
            err.add("field equations: derivatives", vec_err(da, db));
            std::vector<double> oa(gravity.output_names().size()), ob(oa.size());
            gravity.outputs(be, ga.data(), y.data(), ma, oa.data());
            fe->outputs(bn, gb.data(), y.data(), mb, ob.data());
            err.add("field equations: outputs", vec_err(oa, ob));
        }
    }
    for (double x : xs) {
        const auto a = gravity.tensor(x, *ebs.cosmo);
        const auto b = fe->tensor(x);
        err.add("field equations: tensor",
                std::max({rel(a.M2, b.M2), std::abs(a.alpha_M - b.alpha_M), std::abs(a.alpha_T - b.alpha_T)}));
    }

    const double tol_ncdm = 3.0 * tol;
    const std::set<std::string> ncdm_checks{"cache", "moments", "rhs", "outputs", "switch_mode"};
    for (const auto & what : err.order)
        check_local(err.e[what] < (what.rfind("switch points", 0) == 0 ? tol_switch :
                                   ncdm_checks.count(what)             ? tol_ncdm :
                                                                         tol),
                    L + what + " = solver (max error " + num(err.e[what], 2) + ")");
    if (eta_offset)
        check_local(true,
                    L + "(the solver's eta is offset by " + num(ebs.eta_offset / ctx.eta(-22.0), 2) +
                        " relative at x = -22: its GR initial value)");
}

// ---------------------------------------------------------------------------------------------------------------------
// Options, density definitions, registry and errors
// ---------------------------------------------------------------------------------------------------------------------

/// (the solver's Perturbations::get_species() is a PT::SpeciesPerturbationList)
std::vector<std::string> names_of(const PT::SpeciesPerturbationList & l) {
    std::vector<std::string> n;
    for (const auto & s : l)
        n.push_back(s->name());
    return n;
}

void test_options() {
    // neutrinos = false, and no massless neutrinos (N_eff = the massive species' share): the same species as the solver
    {
        auto p = planck(0.06);
        EBS::PerturbationOptions opt;
        opt.neutrinos = false;
        EBSSide ebs("lcdm", p, {}, opt);
        auto pm = PT::make_perturbation_model("lcdm", p);
        PT::PerturbationSettings s;
        s.neutrinos = false;
        const auto a = names_of(ebs.pert->get_species()), b = names_of(pm->species_perturbations(pm->context(s)));
        check_local(a == b and b.size() == 2, "lcdm neutrinos = false: species (cdm, massive) = solver");
    }
    {
        auto p = planck(0.06);
        p.N_eff = p.N_ncdm_per_species();
        EBSSide ebs("lcdm", p, {});
        auto pm = PT::make_perturbation_model("lcdm", p);
        const auto a = names_of(ebs.pert->get_species()), b = names_of(pm->species_perturbations(pm->context()));
        check_local(a == b and b.size() == 2, "lcdm Omega_nu = 0: species (cdm, massive) = solver");
    }
    // Settings reach the species
    {
        auto pm = PT::make_perturbation_model("lcdm", planck(0.06));
        PT::PerturbationSettings s;
        s.lmax_nu = 10;
        s.lmax_ncdm = 5;
        s.n_q_ncdm = 4;
        const auto l = pm->species_perturbations(pm->context(s));
        check_local(l.size() == 3 and l[1]->n_vars(0) == 11 and l[2]->n_vars(0) == 4 * 6,
                    "settings lmax_nu, lmax_ncdm, n_q_ncdm");
    }

    // jbd density definitions: Gtoday rescales the massive neutrinos (not a MassiveNeutrinoSpecies): rho = C I_rho(M)
    // holds in the cache, and the coupling g enters G_eff_initial and the tensors
    for (int d : {0, 1, 2, 3}) {
        auto pm = PT::make_perturbation_model("jbd",
                                              planck(0.06),
                                              {{"bd_omega", 100.0}, {"bd_density_parameter_definition", double(d)}});
        const auto ctx = pm->context();
        const auto l = pm->species_perturbations(ctx);
        const auto fe = pm->field_equation_perturbations(ctx);
        const auto * bdm = dynamic_cast<const PT::PerturbationModel_BransDicke *>(pm.get());
        const auto & bd = bdm->field_equations();
        double e = 0.0;
        for (double x : {-15.0, -5.0, 0.0}) {
            const auto b = ctx.mode_at_x(x, 0.1 / Mpc);
            std::vector<double> cc(l[2]->cache_size());
            l[2]->update_cache(b, cc.data());
            // cache: M, C, rho, ...: C I_rho(M) = rho
            e = std::max(e, rel(cc[1] * BG::FermiDiracIntegrals::get().I_rho(cc[0]), cc[2]));
        }
        const double g = bd.G_over_Gstar();
        const bool ok_g =
            rel(fe->G_eff_initial(), g / bd.phi_initial()) < 1e-14 and rel(fe->tensor(0.0).M2, bd.phi(0.0) / g) < 1e-14;
        check_local(l.size() == 3 and e < 1e-12 and ok_g,
                    "jbd bd_density_parameter_definition = " + std::to_string(d) +
                        ": massive neutrinos rho = C I_rho(M) (error " + num(e, 2) + "), g = " + num(g) +
                        " in G_eff_initial and M*^2");
    }
}

void test_registry_and_errors() {
    const auto p = planck(0.06);
    auto lcdm = PT::make_perturbation_model("lcdm", p);
    auto jbd = PT::make_perturbation_model("jbd", p, {{"bd_omega", 50.0}});
    check_local(lcdm->name() == "lcdm" and jbd->name() == "jbd" and lcdm->background().name() == "lcdm" and
                    jbd->background().name() == "jbd",
                "registry: lcdm, jbd with their backgrounds");
    check_local(throws([&] { PT::make_perturbation_model("dgp", p); }),
                "registry: a model without perturbations throws");
    check_local(throws([&] { PT::make_perturbation_model("jbd", p, {{"no_such", 1.0}}); }),
                "registry: unknown model parameter throws");
    check_local(throws([&] { PT::make_perturbation_model("jbd", planck(0.06, 0.02)); }),
                "jbd with curvature throws (flat only, as the solver)");
    check_local(throws([&] { PT::PerturbationModel_BransDicke(BG::make_lcdm(p)); }) and
                    throws([&] { PT::PerturbationModel_LCDM(BG::make_jbd(p, 100.0)); }),
                "models on the wrong background throw");
    check_local(throws([&] { lcdm->species_perturbations(jbd->context()); }), "a context of another background throws");
    PT::PerturbationSettings s;
    s.x_start = -30.0;
    check_local(throws([&] { lcdm->context(s); }), "x_start outside the background range throws");
    s = {};
    s.lmax_nu = 2;
    check_local(throws([&] { lcdm->species_perturbations(lcdm->context(s)); }), "lmax_nu < 3 throws");
    auto closed = PT::make_perturbation_model("lcdm", planck(0.0, -0.02));
    check_local(throws([&] { closed->context().mode_at_x(-5.0, 1e-5 / Mpc); }), "closed universe with s_2 = 0 throws");
    // The momentum quadrature integrates the Fermi-Dirac rho as the background
    const PT::MassiveNeutrinoQuadrature Q(48);
    double I = 0.0;
    for (int i = 0; i < Q.size(); i++)
        I += Q.w[i] * Q.q[i] * Q.q[i] * Q.q[i];
    check_local(rel(I, BG::FermiDiracIntegrals::I_rho_0) < 1e-10,
                "48-node quadrature: I_rho(0) = 7 pi^4/120 (error " + num(rel(I, BG::FermiDiracIntegrals::I_rho_0), 2) +
                    ")");
}

int main() {
    const double tol = 1e-7;
    compare("lcdm", planck(0.06), {}, tol);
    compare("lcdm", planck(0.1, 0.02), {}, tol);
    compare("lcdm", planck(0.06, -0.02), {}, tol);
    compare("lcdm", planck(0.0), {}, tol);
    compare("jbd", planck(0.06), {{"bd_omega", 100.0}}, tol, true, 1e-6);
    compare("jbd", planck(0.1), {{"bd_omega", 1000.0}}, tol, true, 1e-6);
    test_options();
    test_registry_and_errors();
    return FML::TESTS::summary("Perturbations");
}
