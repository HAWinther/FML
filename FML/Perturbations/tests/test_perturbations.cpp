//==========================================================
// Tests of FML/Perturbations against the Einstein-Boltzmann solver's old classes (Cosmology/Models: its Species and
// Gravity, as an explicit Model of BackgroundCosmology; the solver itself now runs the new classes for every model),
// equation by equation. For every model of the solver (lcdm, w0wa, musigma, dgp, fr, horndeski, quintessence, ede and
// ede_zc, jbd, coupled_quintessence, cubic_galileon, dark_scattering, ide), at two or more parameter points with massive
// neutrinos and, where the solver allows it, curvature, the solver's model and the new one are built with the same
// parameters, and the solver's species / gravity perturbations (from Species::perturbations with a Perturbations
// object of the explicit Model as the host, and its Gravity) are compared with the new SpeciesPerturbation /
// PerturbedFieldEquation objects:
//   - names, info, n_vars, cache_size, prefers_explicit, output names (a new species may add cache entries and outputs
//     after the solver's: the scalar field's phi_xx, phi - phi_min, c_s^2 and q_de)
//   - for k from 1e-4 to 5 /Mpc and x from -17 to 0, each side with a ModeAtX filled from its own background:
//     the ModeAtX itself, switch points, caches, initial conditions (adiabatic and isocurvature), moments, rhs,
//     mode switch and outputs with fixed state vectors and metric; the field equations' G_eff_initial, initial
//     conditions, cache, metric (fixed variables and moments), outputs and tensor coefficients
// The tolerance is the level at which the two backgrounds agree (1e-7), except for the species quantities that depend
// on the massive-neutrino pressure and pseudo-pressure (cache, moments, rhs, outputs, switch_mode: 3e-7): the solver
// computes I_p and I_pp with a 48-node Gauss-Laguerre quadrature, which is off by up to 7e-8 near M ~ 0.3 (the new
// background's tabulated integrals agree with direct integration to 1e-10), and c_a^2 - w amplifies it. The maximum
// errors are printed.
// Also: the free-streaming density, neutrinos = false and Omega_nu = 0, the jbd density definitions (rescaled massive
// neutrinos), the oscillating-fluid mode of the scalar field (which the solver does not have: frequency, sound speed,
// the switch), the models without a reference (the other quintessence potentials), the registry and errors.
//==========================================================
#include <FML/Perturbations/Presets.h>

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
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

/// The solver's old model (an explicit Model of BackgroundCosmology) with a Perturbations object as the host of its
/// species' perturbations (Species::perturbations; not solved) and its gravity
struct EBSSide {
    EBS::CosmologyParameters params;
    std::unique_ptr<EBS::BackgroundCosmology> cosmo;
    std::unique_ptr<EBS::RecombinationHistory> rec;
    std::unique_ptr<EBS::Perturbations> pert;
    /// The perturbations of the old species (those that have them) and the species
    std::vector<std::unique_ptr<EBS::SpeciesPerturbations>> species;
    std::vector<const EBS::Species *> owners;
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
        for (const auto & s : cosmo->species())
            if (auto q = s->perturbations(*pert)) {
                species.push_back(std::move(q));
                owners.push_back(s.get());
            }
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
// its own eta) are compared with the tolerance tol_switch. ebs_model, ebs_mp: the solver's model if it has another
// name or parameters (ede_zc). The species checks of the massive neutrinos are listed separately (tolerance 3 tol, see
// the header). Returns the largest error of all other checks and that of the massive neutrinos.
std::pair<double, double> compare(const std::string & model,
               const BG::BackgroundParameters & p,
               const ModelParameters & mp,
               double tol,
               bool eta_offset = false,
               double tol_switch = 1e-7,
               const std::string & ebs_model = "",
               const ModelParameters & ebs_mp = {}) {
    std::ostringstream label;
    label << model;
    for (const auto & [key, v] : mp)
        label << " " << key << "=" << v;
    label << " m_nu=" << (p.m_nu_eV.empty() ? 0.0 : p.m_nu_eV[0]);
    if (p.Omega_k != 0.0)
        label << " Omega_k=" << p.Omega_k;
    const std::string L = label.str() + ": ";

    EBSSide ebs(ebs_model.empty() ? model : ebs_model, p, ebs_model.empty() ? mp : ebs_mp);
    // on the range the solver uses (BackgroundCosmology), so that a scalar field is stored on the same nodes
    auto pm = PT::make_perturbation_model(model, p, mp, BG::BackgroundRange{-25.0, 5.0, 200});
    const PT::PerturbationContext ctx = pm->context();
    const auto species = pm->species_perturbations(ctx);
    const auto fe = pm->field_equation_perturbations(ctx);
    const auto & ebs_species = ebs.species;
    const auto & gravity = ebs.cosmo->gravity();
    if (eta_offset)
        ebs.eta_offset = ctx.eta(-22.0) - ebs.cosmo->eta_of_x(-22.0);

    // ---- Structure ------------------------------------------------------------------------------------------------
    // (a new species may have more cache entries and outputs, after the solver's)
    auto prefix = [](const std::vector<std::string> & a, const std::vector<std::string> & b) {
        return a.size() <= b.size() and std::equal(a.begin(), a.end(), b.begin());
    };
    bool same = species.size() == ebs_species.size();
    std::string names;
    for (size_t i = 0; same and i < species.size(); i++) {
        const auto & a = *ebs_species[i];
        const auto & b = *species[i];
        names += (i ? ", " : "") + b.name();
        same = same and a.name() == b.name() and a.info() == b.info() and a.cache_size() <= b.cache_size() and
               prefix(a.output_names(), b.output_names());
        for (int mode = 0; mode < 2; mode++)
            same = same and a.n_vars(mode) == b.n_vars(mode) and a.prefers_explicit(mode) == b.prefers_explicit(mode);
    }
    same = same and fe->n_vars() == gravity.n_vars() and fe->cache_size() == gravity.cache_size() and
           fe->output_names() == gravity.output_names() and fe->name() == gravity.name();
    check_local(same,
                L + "species (" + names + ") and field equations (" + fe->name() +
                    "): names, n_vars, cache_size, prefers_explicit, output names = solver");
    if (not same)
        return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};

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
    for (const auto * s : ebs.owners)
        if (s->free_streaming())
            OmNu_ebs += s->rho(x_ini) * std::exp(4.0 * x_ini);
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
            if (not std::isfinite(elem_err(mode_fields(be), mode_fields(bn)))) {
                std::cout << "  (non-finite ModeAtX at x = " << x << ", k = " << k * Mpc << "/Mpc:";
                for (double v : mode_fields(be))
                    std::cout << " " << v;
                std::cout << " |";
                for (double v : mode_fields(bn))
                    std::cout << " " << v;
                std::cout << ")\n";
            }

            const EBS::MetricAtX me{0.3, -0.4, 0.02};
            const PT::MetricAtX mn{0.3, -0.4, 0.02};
            for (size_t i = 0; i < species.size(); i++) {
                const auto & a = *ebs_species[i];
                const auto & b = *species[i];
                const std::string P = b.name() == "massive neutrinos" ? "massive neutrinos: " : "";
                std::vector<double> ca(a.cache_size()), cb(b.cache_size());
                a.update_cache(be, ca.data());
                b.update_cache(bn, cb.data());
                err.add(P + "cache", elem_err(ca, std::vector<double>(cb.begin(), cb.begin() + ca.size())));

                // Initial conditions (adiabatic and isocurvature)
                std::vector<double> ya(a.n_vars(0)), yb(b.n_vars(0));
                a.initial_conditions(be, ic_ebs, ya.data());
                b.initial_conditions(bn, ic_new, yb.data());
                err.add(P + "initial conditions", vec_err(ya, yb));
                EBS::InitialData ie = ic_ebs;
                PT::InitialData in = ic_new;
                ie.iso = &iso_ebs;
                in.iso = &iso_new;
                a.initial_conditions(be, ie, ya.data());
                b.initial_conditions(bn, in, yb.data());
                err.add(P + "initial conditions (isocurvature)", vec_err(ya, yb));

                const int n_modes = (b.name() == "massive neutrinos" or not b.switch_points(k).empty()) ? 2 : 1;
                const size_t n_out = a.output_names().size();
                for (int mode = 0; mode < n_modes; mode++) {
                    const auto y = state(b.n_vars(mode), 0.1 * mode);
                    err.add(P + "moments",
                            moments_err(a.moments(be, ca.data(), mode, y.data()),
                                        b.moments(bn, cb.data(), mode, y.data())));
                    std::vector<double> da(y.size()), db(y.size());
                    a.rhs(be, ca.data(), mode, me, y.data(), da.data());
                    b.rhs(bn, cb.data(), mode, mn, y.data(), db.data());
                    err.add(P + "rhs", vec_err(da, db));
                    std::vector<double> oa(n_out), ob(b.output_names().size());
                    a.outputs(be, ca.data(), mode, y.data(), oa.data());
                    b.outputs(bn, cb.data(), mode, y.data(), ob.data());
                    ob.resize(n_out);
                    err.add(P + "outputs", vec_err(oa, ob));
                    if (mode == 1) {
                        const auto y0 = state(b.n_vars(0));
                        std::vector<double> Fa(b.n_vars(1)), Fb(b.n_vars(1));
                        a.switch_mode(be, ca.data(), 1, y0.data(), Fa.data());
                        b.switch_mode(bn, cb.data(), 1, y0.data(), Fb.data());
                        err.add(P + "switch_mode", vec_err(Fa, Fb));
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
    double worst = 0.0, worst_ncdm = 0.0;
    for (const auto & what : err.order) {
        const bool sw = what.rfind("switch points", 0) == 0, ncdm = what.rfind("massive neutrinos:", 0) == 0;
        if (ncdm)
            worst_ncdm = std::max(worst_ncdm, err.e[what]);
        else if (not sw)
            worst = std::max(worst, err.e[what]);
        check_local(err.e[what] < (sw ? tol_switch : ncdm ? tol_ncdm : tol),
                    L + what + " = solver (max error " + num(err.e[what], 2) + ")");
    }
    if (eta_offset)
        check_local(true,
                    L + "(the solver's eta is offset by " + num(ebs.eta_offset / ctx.eta(-22.0), 2) +
                        " relative at x = -22: its GR initial value)");
    return {worst, worst_ncdm};
}

// ---------------------------------------------------------------------------------------------------------------------
// Options, density definitions, registry and errors
// ---------------------------------------------------------------------------------------------------------------------

template <class List>
std::vector<std::string> names_of(const List & l) {
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
        const auto a = names_of(ebs.species), b = names_of(pm->species_perturbations(pm->context(s)));
        check_local(a == b and b.size() == 2, "lcdm neutrinos = false: species (cdm, massive) = solver");
    }
    {
        auto p = planck(0.06);
        p.N_eff = p.N_ncdm_per_species();
        EBSSide ebs("lcdm", p, {});
        auto pm = PT::make_perturbation_model("lcdm", p);
        const auto a = names_of(ebs.species), b = names_of(pm->species_perturbations(pm->context()));
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
    check_local(throws([&] { PT::make_perturbation_model("no_such_model", p); }), "registry: an unknown model throws");
    // every background model has perturbations, with the parameters of the background registry
    bool all = true;
    std::string failed;
    for (const auto & m : BG::registry()) {
        try {
            auto pm = PT::make_perturbation_model(m.name, p);
            pm->species_perturbations(pm->context());
            pm->field_equation_perturbations(pm->context());
        } catch (const std::exception & e) {
            all = false;
            failed += " " + m.name + " (" + e.what() + ")";
        }
    }
    check_local(all, "registry: every model of the background registry has perturbations (default parameters)" + failed);
    // the checks of the solver
    check_local(throws([&] { PT::make_perturbation_model("ide", p, {{"ide_xi", 0.1}, {"ide_w", -0.95}}); }) and
                    not throws([&] {
                        PT::make_perturbation_model(
                            "ide", p, {{"ide_xi", 0.1}, {"ide_w", -0.95}, {"ide_stability_check", 0.0}});
                    }),
                "ide: doom factor > 0 rejected (unless ide_stability_check = 0)");
    check_local(throws([&] { PT::make_perturbation_model("musigma", p, {{"mu0", 2.0}, {"Sigma0", 0.0}}); }) and
                    throws([&] { PT::make_perturbation_model("musigma", planck(0.0, 0.02)); }) and
                    throws([&] { PT::make_perturbation_model("fr", p, {{"fR0", -1e-5}}); }) and
                    throws([&] { PT::make_perturbation_model("dark_scattering", p, {{"ds_xi", -1.0}}); }) and
                    throws([&] { PT::make_perturbation_model("dark_scattering", p, {{"ds_cs2", 0.0}}); }) and
                    throws([&] { PT::make_perturbation_model("horndeski", p, {{"hd_cK", -1.0}}); }) and
                    throws([&] { PT::make_perturbation_model("horndeski", p, {{"hd_cK", 0.0}, {"hd_cM", 0.3}}); }) and
                    throws([&] { PT::make_perturbation_model("horndeski", p, {{"hd_parametrization", 3.0}}); }) and
                    throws([&] { PT::make_perturbation_model("horndeski", planck(0.0, 0.02)); }),
                "musigma (1 + 2 Sigma0 - mu0 < 0, curvature), fr (fR0 < 0), dark_scattering (xi < 0, cs2 = 0), "
                "horndeski (ghost, D = 0 with alpha_M, parametrization, curvature) rejected");
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

// ---------------------------------------------------------------------------------------------------------------------
// The scalar field: the oscillating fluid, and the potentials without a solver reference
// ---------------------------------------------------------------------------------------------------------------------

/// The ScalarField species of a model (the last species)
const PT::SpeciesPerturbation_ScalarField & scalar_field(const PT::SpeciesPerturbationList & l) {
    const auto * f = dynamic_cast<const PT::SpeciesPerturbation_ScalarField *>(l.back().get());
    if (not f)
        throw std::runtime_error("no scalar field");
    return *f;
}

void test_scalar_field_fluid() {
    const auto p = planck(0.06);
    // EDE with n = 2.2, z_c = 1e6: the background becomes a fluid with w = 0.375 at z ~ 1e4
    auto pm = PT::make_perturbation_model(
        "ede", p, {{"ede_fEDE", 0.05}, {"ede_log10zc", 6.0}, {"ede_theta_i", 2.0}, {"ede_n", 2.2}});
    const auto ctx = pm->context();
    const auto l = pm->species_perturbations(ctx);
    const auto & f = scalar_field(l);
    const auto & e = dynamic_cast<const PT::PerturbationModel_EDE &>(*pm).early_dark_energy();
    const double xf = e.x_fluid(), n = 2.2, w = (n - 1.0) / (n + 1.0);
    const auto sp = f.switch_points(0.1 / Mpc);
    check_local(sp.size() == 1 and sp[0] == xf and xf > -10.0 and xf < -8.0 and f.n_vars(1) == 2,
                "EDE n = 2.2, z_c = 1e6: one switch point, at the background's switch to the fluid (x = " + num(xf, 5) +
                    ")");

    // The angular frequency: for a small amplitude the power law V ~ mu (theta^2/2)^n, varpi T = 2 pi with
    // T = 4 A / sqrt(6 rho) sqrt(pi) Gamma(1 + 1/(2n)) / Gamma(1/2 + 1/(2n)) (H0 = 1), A the amplitude in phi
    {
        const double fpl = e.f(), th = 1e-3, A = fpl * th, rho = e.v(A);
        const double T = 4.0 * A / std::sqrt(6.0 * rho) * std::sqrt(M_PI) * std::tgamma(1.0 + 0.5 / n) /
                         std::tgamma(0.5 + 0.5 / n);
        const double r = f.angular_frequency(rho) * T / (2.0 * M_PI);
        check_local(std::abs(r - 1.0) < 1e-5,
                    "EDE: oscillation frequency at small amplitude = power-law (Gamma-function) value (error " +
                        num(std::abs(r - 1.0), 2) + ")");
    }
    // The sound speed in the fluid: w for k << a varpi, 1 for k >> a varpi
    {
        const double x = xf + 0.5;
        std::vector<double> cc(f.cache_size());
        f.update_cache(ctx.mode_at_x(x, 1e-5 / Mpc), cc.data());
        const double cs_lo = cc[PT::SpeciesPerturbation_ScalarField::CS2];
        f.update_cache(ctx.mode_at_x(x, 1e4 / Mpc), cc.data());
        const double cs_hi = cc[PT::SpeciesPerturbation_ScalarField::CS2];
        check_local(std::abs(cs_lo / w - 1.0) < 1e-6 and std::abs(cs_hi - 1.0) < 1e-3,
                    "fluid c_s^2 = w_osc = " + num(w, 4) + " on large scales (" + num(cs_lo, 8) +
                        "), -> 1 below the Jeans scale (" + num(cs_hi, 6) + ")");
    }
    // The switch: u from the shift epsilon = dphi / phi_x near the crossing, delta keeps the comoving density
    {
        const double k = 0.05 / Mpc;
        const auto b = ctx.mode_at_x(xf, k);
        std::vector<double> cc(f.cache_size());
        f.update_cache(b, cc.data());
        using SF = PT::SpeciesPerturbation_ScalarField;
        const double y[2] = {1e-3, -2e-2};
        double yn[2];
        f.switch_mode(b, cc.data(), 1, y, yn);
        const auto m0 = f.moments(b, cc.data(), 0, y);
        std::vector<double> cf(f.cache_size());
        auto bf = ctx.mode_at_x(xf + 1e-9, k);
        f.update_cache(bf, cf.data());
        const auto m1 = f.moments(bf, cf.data(), 1, yn);
        const double D0 = m0.drho + 3.0 * m0.flux / b.ckH, D1 = m1.drho + 3.0 * m1.flux / bf.ckH;
        // a pure shift: dphi = eps phi_x, w = eps phi_xx (+ eta (D, n phi_x) of the amplitude, here 0)
        const double eps = 1e-4, ys[2] = {eps * cc[SF::DPHI], eps * cc[SF::DDPHI]};
        double ysn[2];
        f.switch_mode(b, cc.data(), 1, ys, ysn);
        check_local(rel(D0, D1) < 1e-7 and rel(ysn[1], b.ckH * eps) < 1e-12 and std::abs(cf[SF::DPHI]) == 0.0 and
                        rel(cf[SF::RHO], cc[SF::RHO]) < 1e-7,
                    "switch to the fluid: the comoving density is continuous (" + num(rel(D0, D1), 2) +
                        "), u = (ck/Hp) epsilon for a shift, rho continuous");
    }
    // The fluid from the start (n = 1.5, z_c = 10^3.5: the field oscillates fast from a ~ 0.013, before x_start = -3)
    {
        PT::PerturbationSettings set;
        set.x_start = -3.0;
        auto pf = PT::make_perturbation_model(
            "ede", p, {{"ede_fEDE", 0.05}, {"ede_log10zc", 3.5}, {"ede_theta_i", 2.0}, {"ede_n", 1.5}});
        const auto & ef = dynamic_cast<const PT::PerturbationModel_EDE &>(*pf).early_dark_energy();
        if (ef.x_fluid() < set.x_start) {
            const auto c2 = pf->context(set);
            const auto lf = pf->species_perturbations(c2);
            const auto & ff = scalar_field(lf);
            PT::InitialData ic;
            ic.Psi = -0.6;
            double y[2];
            const auto b = c2.mode_at_x(set.x_start, 0.01 / Mpc);
            ff.initial_conditions(b, ic, y);
            const double wf = ef.w_oscillating();
            check_local(ff.switch_points(0.01 / Mpc).empty() and ff.fluid(0) and rel(y[0], -1.5 * (1.0 + wf) * ic.Psi) < 1e-14 and
                            rel(y[1], 0.5 * b.ckH * ic.Psi) < 1e-14,
                        "a field that is a fluid at x_start (x_fluid = " + num(ef.x_fluid(), 4) +
                            "): one fluid mode with adiabatic initial conditions");
        } else
            check_local(false, "EDE n = 1.5, z_c = 10^3.5 should be a fluid at x = -3 (x_fluid = " + num(ef.x_fluid(), 4) + ")");
    }
}

void test_potentials_and_derivatives() {
    const auto p = planck(0.06);
    // v_phiphi of the potentials against a central difference of v_phi, and the scalar field of the potentials the
    // solver does not have
    double ev = 0.0;
    std::string names;
    for (const auto & [model, mp] : std::vector<std::pair<std::string, ModelParameters>>{
             {"quintessence", {{"quint_lambda", 0.7}}},
             {"quintessence_inverse_power_law", {{"quint_alpha", 2.0}, {"quint_phi_ini", 0.2}}},
             {"quintessence_pngb", {{"quint_f", 1.0}, {"quint_phi_ini", 1.0}}},
             {"quintessence_hilltop", {{"quint_mu", 2.0}, {"quint_p", 4.0}, {"quint_phi_ini", -0.5}}},
             {"ede", {}}}) {
        auto pm = PT::make_perturbation_model(model, p, mp);
        const auto ctx = pm->context();
        const auto l = pm->species_perturbations(ctx);
        const auto & sf = scalar_field(l);
        names += " " + sf.name();
        const BG::ScalarFieldSpecies * q = nullptr;
        for (const auto & s : pm->background().species())
            if (auto f = dynamic_cast<const BG::ScalarFieldSpecies *>(s.get()))
                q = f;
        // (the error of the difference is ~ (h / phi)^2 for EDE, whose v ~ phi^6 near the minimum)
        for (double x : {-10.0, -3.0, -1.0, 0.0}) {
            const double phi = q->phi(x), h = 1e-5 * std::max(1e-2, std::abs(phi));
            const double fd = (q->v_phi(phi + h) - q->v_phi(phi - h)) / (2.0 * h);
            ev = std::max(ev, std::abs(q->v_phiphi(phi) - fd) / (std::abs(fd) + std::abs(q->v_phiphi(phi)) * 1e-3 + 1e-300));
        }
    }
    check_local(ev < 1e-5, "v_phiphi of all potentials (exponential, inverse power law, PNGB, hilltop, EDE) = central "
                           "difference of v_phi (" + num(ev, 2) + "); scalar fields:" + names);

    // d2rhodx2 of the species of every model (analytic where the background gives it) against the 5-point difference
    // of drhodx (the default of Species)
    double ed = 0.0;
    std::string worst;
    for (const auto & m : BG::registry()) {
        auto bg = BG::make_background(m.name, p);
        for (const auto & s : bg->species())
            for (double x : {-12.0, -5.0, -2.0, 0.0}) {
                const double hh = 1e-5; // (an oscillating field varies on scales ~ 1e-2 in x)
                const double fd = (s->drhodx(x - 2 * hh) - 8 * s->drhodx(x - hh) + 8 * s->drhodx(x + hh) -
                                   s->drhodx(x + 2 * hh)) /
                                  (12 * hh);
                const double e = std::abs(s->d2rhodx2(x) - fd) / (std::abs(s->rho(x)) + 1e-300);
                if (e > ed) {
                    ed = e;
                    worst = m.name + " " + s->name();
                }
            }
    }
    check_local(ed < 1e-7, "d2rhodx2 of every species of every model = 5-point difference of drhodx (max " + num(ed, 2) +
                               " relative to rho, " + worst + ")");
}

int main() {
    // The tolerance (3 times that for the massive neutrinos, see the header). Early dark energy: 1e-6, as late in the
    // oscillating phase (x = -2, 0, after ~1e4 oscillations, where the field's density is < 1e-6 of the total) the
    // instantaneous field quantities (phi_x, v_phi, ... in the cache and rhs) of the two solutions differ by up to 7e-7
    // through the drift of the oscillation phase of two independent integrations (both store the exact ODE solution on
    // the same nodes; the shooting of mu, f agrees to 1e-10); up to x = -5 they agree to 1e-8 as the other models
    double tol = 1e-7;
    std::vector<std::pair<std::string, std::pair<double, double>>> worst;
    auto run = [&](const std::string & model,
                   const BG::BackgroundParameters & p,
                   const ModelParameters & mp,
                   bool eta_offset = false,
                   double tol_switch = 1e-7,
                   const std::string & ebs_model = "",
                   const ModelParameters & ebs_mp = {}) {
        const auto e = compare(model, p, mp, tol, eta_offset, tol_switch, ebs_model, ebs_mp);
        bool found = false;
        for (auto & [m, v] : worst)
            if (m == model) {
                v.first = std::max(v.first, e.first);
                v.second = std::max(v.second, e.second);
                found = true;
            }
        if (not found)
            worst.push_back({model, e});
    };
    run("lcdm", planck(0.06), {});
    run("lcdm", planck(0.1, 0.02), {});
    run("lcdm", planck(0.06, -0.02), {});
    run("lcdm", planck(0.0), {});
    run("jbd", planck(0.06), {{"bd_omega", 100.0}}, true, 1e-6);
    run("jbd", planck(0.1), {{"bd_omega", 1000.0}}, true, 1e-6);
    run("w0wa", planck(0.06, 0.02), {{"w0", -0.9}, {"wa", 0.1}});
    run("w0wa", planck(0.1, -0.02), {{"w0", -1.05}, {"wa", 0.2}});
    run("musigma", planck(0.06), {{"mu0", 0.3}, {"Sigma0", 0.1}});
    run("musigma", planck(0.1), {{"mu0", -0.2}, {"Sigma0", 0.3}});
    run("dgp", planck(0.06), {{"Omega_rc", 0.25}});
    run("dgp", planck(0.1), {{"Omega_rc", 0.0}, {"dgp_branch", 1.0}});
    run("dgp", planck(0.06), {{"Omega_rc", 0.1}, {"dgp_lcdm_background", 1.0}});
    run("fr", planck(0.06), {{"fR0", 1e-5}});
    run("fr", planck(0.1), {{"fR0", 1e-4}, {"fR_n", 2.0}});
    run("horndeski", planck(0.06), {{"hd_cK", 1.0}, {"hd_cB", 0.6}, {"hd_cM", 0.3}, {"hd_w0", -0.95}, {"hd_wa", 0.1}});
    run("horndeski", planck(0.1), {{"hd_parametrization", 1.0}, {"hd_cK", 0.5}, {"hd_cB", 0.2}, {"hd_cM", 0.1}});
    run("horndeski", planck(0.0), {{"hd_parametrization", 2.0}, {"hd_cK", 0.1}, {"hd_cB", 0.05}, {"hd_M2_ini", 1.2}});
    run("quintessence", planck(0.06, 0.02), {{"quint_lambda", 1.0}});
    run("quintessence", planck(0.1, -0.02), {{"quint_lambda", 0.5}});
    tol = 1e-6;
    run("ede", planck(0.06), {{"ede_fEDE", 0.1}, {"ede_log10zc", 3.5}, {"ede_theta_i", 2.8}, {"ede_n", 3.0}});
    run("ede", planck(0.1, 0.02), {{"ede_fEDE", 0.12}, {"ede_log10zc", 3.3}, {"ede_theta_i", 1.5}, {"ede_n", 4.0}});
    run("ede_zc",
        planck(0.06),
        {{"ede_fEDE", 0.08}, {"ede_zc", 4000.0}, {"ede_theta_i", 2.0}, {"ede_n", 2.5}},
        false,
        1e-7,
        "ede",
        {{"ede_fEDE", 0.08}, {"ede_log10zc", std::log10(4000.0)}, {"ede_theta_i", 2.0}, {"ede_n", 2.5}});
    tol = 1e-7;
    run("coupled_quintessence", planck(0.06, 0.02), {{"cq_lambda", 0.5}, {"cq_beta", 0.05}});
    run("coupled_quintessence", planck(0.1), {{"cq_lambda", 1.0}, {"cq_beta", -0.1}});
    run("cubic_galileon", planck(0.06), {});
    run("cubic_galileon", planck(0.1), {});
    run("dark_scattering", planck(0.06, 0.02), {{"ds_w", -0.9}, {"ds_xi", 10.0}});
    run("dark_scattering", planck(0.1, -0.02), {{"ds_w", -1.1}, {"ds_xi", 50.0}, {"ds_cs2", 0.5}});
    run("dark_scattering", planck(0.06), {{"ds_w", -0.8}, {"ds_xi", 0.0}});
    run("ide", planck(0.06, 0.02), {{"ide_xi", -0.1}, {"ide_w", -0.95}});
    run("ide", planck(0.1), {{"ide_xi", 0.1}, {"ide_w", -1.05}, {"ide_cs2", 0.5}});
    std::cout << "Largest error of the equation-by-equation comparison per model (all but the massive-neutrino species; "
                 "massive neutrinos):\n";
    for (const auto & [m, e] : worst)
        std::cout << "  " << std::left << std::setw(22) << m << std::setw(10) << num(e.first, 2) << num(e.second, 2)
                  << "\n";
    test_scalar_field_fluid();
    test_potentials_and_derivatives();
    test_options();
    test_registry_and_errors();
    return FML::TESTS::summary("Perturbations");
}
