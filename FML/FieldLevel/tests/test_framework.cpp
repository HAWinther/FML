#include <FML/FieldLevel/Framework.h>
#include <FML/tests/TestHelpers.h>
#include <iostream>
#include <iomanip>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Complex;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check_local;
static FF::Configuration config(FF::Evolution evolution) {
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.growth_rate = .6;
    c.evolution = evolution;
    return c;
}
static double relative(double a, double b) { return std::abs(a - b) / std::max({1., std::abs(a), std::abs(b)}); }
static double inner(const Field & a, const Field & b) {
    double x = 0;
    for (size_t i = 0; i < a.size(); ++i)
        x += a[i] * b[i];
    return x;
}
static double maxerr(const Field & a, const Field & b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}
static void gradients(FF::Evolution kind) {
    auto spectrum = FML::FIELDLEVEL::BAOSpectrum::eisenstein_hu(200.);
    FF::Model model(config(kind), FF::Spectrum::bao(spectrum));
    auto w = model.workspace();
    auto p = model.initial_parameters();
    p[0] = 1.03;
    p[1] = 1.4;
    if (kind != FF::Evolution::Linear) {
        p[2] = .2;
        p[3] = -.15;
        p[4] = 12.;
    }
    auto s = model.white_noise(12);
    auto data = model.mock(model.white_noise(11), p, 55, w);
    FF::Posterior posterior(model, data);
    Field gs;
    std::vector<double> gp;
    const double U = posterior.evaluate(s, p, w, gs, gp);
    auto direction = model.white_noise(22);
    const double analytic = inner(gs, direction), h = 1e-5;
    auto plus = s, minus = s;
    for (size_t i = 0; i < s.size(); ++i) {
        plus[i] += h * direction[i];
        minus[i] -= h * direction[i];
    }
    Field scratch;
    std::vector<double> sp;
    const double numeric =
        (posterior.evaluate(plus, p, w, scratch, sp) - posterior.evaluate(minus, p, w, scratch, sp)) / (2 * h);
    check_local(relative(analytic, numeric) < 2e-6, "whitened field gradient, evolution " + std::to_string(int(kind)));
    for (size_t j = 0; j < p.size(); ++j) {
        auto a = p, b = p;
        const double step = 1e-5 * std::max(1., std::abs(p[j]));
        a[j] += step;
        b[j] -= step;
        const double num =
            (posterior.evaluate(s, a, w, scratch, sp) - posterior.evaluate(s, b, w, scratch, sp)) / (2 * step);
        if (relative(gp[j], num) >= 2e-6)
            std::cout << "gradient " << model.parameters()[j].name << " analytic " << gp[j] << " numeric " << num
                      << '\n';
        check_local(relative(gp[j], num) < 2e-6,
                    "parameter gradient: " + model.parameters()[j].name + " evolution " + std::to_string(int(kind)));
    }
    const auto cotangent = model.mock(model.white_noise(32), p, 33, w);
    model.predict(s, p, w);
    Field pull;
    std::vector<double> pp;
    model.pullback(p, cotangent, w, pull, pp);
    const auto yp = model.predict(plus, p, w), ym = model.predict(minus, p, w);
    double dot = 0;
    for (size_t i = 0; i < yp.size(); ++i)
        dot += model.modes()[i].weight * std::real(std::conj(cotangent[i]) * (yp[i] - ym[i])) / (2 * h);
    check_local(relative(inner(pull, direction), dot) < 2e-6,
                "forward/reverse dot product, evolution " + std::to_string(int(kind)));
    auto invalid = p;
    invalid[0] = 2;
    check_local(std::isinf(posterior.evaluate(s, invalid, w, scratch, sp)), "parameter support is enforced");
    auto fresh = model.workspace();
    check_local(FML::TESTS::throws([&] { model.pullback(p, cotangent, fresh, pull, pp); }),
                "pullback requires matching forward cache");
    check_local(std::isfinite(U), "posterior is finite");
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    std::cout << std::setprecision(14);
    if (FML::NTasks != 1) {
        check_local(FML::TESTS::throws([] {
                        FF::Model m(config(FF::Evolution::Linear), FF::Spectrum::fixed([](double) { return 200.; }));
                    }),
                    "framework rejects unsupported multi-rank use");
        return FML::TESTS::summary("field framework MPI guard");
    }
    for (auto kind : {FF::Evolution::Linear, FF::Evolution::Zeldovich, FF::Evolution::LPT2})
        gradients(kind);
    auto c = config(FF::Evolution::Linear);
    c.growth_rate = 0;
    FF::Model linear(c, FF::Spectrum::fixed([](double) { return 200.; }));
    auto w = linear.workspace();
    auto p = linear.initial_parameters();
    auto truth = linear.white_noise(44);
    auto data = linear.mock(truth, p, 45, w);
    FF::Posterior posterior(linear, data);
    auto exact = linear.linear_conditional(data, p, w);
    auto result = FF::map(posterior, Field(linear.cells(), 0), p, false, 5, 1e-8);
    check_local(result.status == FF::MAPStatus::Converged && maxerr(result.white, exact.mean) < 1e-10,
                "linear MAP matches exact conditional mean");
    check_local(std::abs(exact.log_marginal - linear.linear_log_marginal(data, p)) < 1e-10,
                "exact conditional and marginal use the same probability normalization");
    bool informative = false;
    for (size_t i = 0; i < exact.covariance.size(); ++i)
        if (linear.modes()[i].k < c.likelihood_cutoff)
            informative |= exact.covariance[i] < .9;
    check_local(informative, "negative control: actual data reduce latent-field uncertainty");
    auto no_information = p;
    no_information.back() = 30;
    auto prior = linear.linear_conditional(data, no_information, w);
    double departure = 0;
    for (auto v : prior.covariance)
        departure = std::max(departure, std::abs(1 - v));
    check_local(departure < 1e-9, "high-noise limit returns the prior covariance");
    Field zero(linear.cells(), 0), gs;
    std::vector<double> gp;
    FF::Posterior zeros(linear, FF::Modes(linear.modes().size(), 0));
    zeros.evaluate(zero, p, w, gs, gp);
    double dof = 0;
    for (const auto & m : linear.modes())
        if (m.k < c.likelihood_cutoff)
            dof += m.weight;
    check_local(std::abs(gp.back() - .5 * dof) < 1e-10, "sampled noise includes log-determinant normalization");
    FML::MCMC::HMCOptions options;
    options.warmup = 100;
    options.samples = 400;
    options.n_leapfrog = 8;
    options.seed = 17;
    auto project = [&](const Field & field) {
        double v = 0;
        for (size_t i = 0; i < field.size(); ++i) {
            const int ix = int(i / (c.mesh * c.mesh));
            v += field[i] * std::cos(2 * M_PI * ix / c.mesh);
        }
        return 2 * v / field.size();
    };
    std::vector<double> modes;
    auto chain = FF::sample(posterior, exact.mean, p, options, false, [&](const auto & field, const auto &, size_t) {
        modes.push_back(project(field));
    });
    double mean = 0, variance = 0;
    for (auto v : modes)
        mean += v;
    mean /= modes.size();
    for (auto v : modes)
        variance += (v - mean) * (v - mean);
    variance /= modes.size();
    double covariance = 1;
    for (size_t i = 0; i < linear.modes().size(); ++i) {
        const auto & m = linear.modes()[i];
        if (m.n[0] == 1 && m.n[1] == 0 && m.n[2] == 0)
            covariance = exact.covariance[i];
    }
    const double expected_mean = project(exact.mean), expected_variance = 2 * covariance / linear.cells();
    check_local(std::abs(mean - expected_mean) < 5 * std::sqrt(expected_variance / modes.size()) &&
                    std::abs(variance / expected_variance - 1) < .3,
                "field HMC mean/covariance agree with exact linear posterior");
    check_local(chain.recorded.size() == size_t(options.samples) && chain.divergences == 0 && chain.acceptance > .5,
                "field posterior samples stream through callback");
    // Independent 2D quadrature of the exact field-marginalized linear nuisance posterior.
    double z = 0, mb = 0, mn = 0, vb = 0, vn = 0;
    const int nb = 100, nn = 160;
    auto logpost = [&](double b, double logn) {
        auto q = p;
        q[0] = b;
        q[1] = logn;
        double v = linear.linear_log_marginal(data, q);
        for (size_t j = 0; j < q.size(); ++j)
            v -= .5 * std::pow((q[j] - linear.parameters()[j].mean) / linear.parameters()[j].sigma, 2);
        return v;
    };
    double maximum = -HUGE_VAL;
    for (int i = 0; i < nb; ++i)
        for (int j = 0; j < nn; ++j)
            maximum = std::max(maximum, logpost(-3.5 + 10. * (i + .5) / nb, p.back() - 8 + 16. * (j + .5) / nn));
    for (int i = 0; i < nb; ++i)
        for (int j = 0; j < nn; ++j) {
            const double b = -3.5 + 10. * (i + .5) / nb, n = p.back() - 8 + 16. * (j + .5) / nn,
                         weight = std::exp(logpost(b, n) - maximum);
            z += weight;
            mb += weight * b;
            mn += weight * n;
            vb += weight * b * b;
            vn += weight * n * n;
        }
    const double exact_b = mb / z, exact_n = mn / z, var_b = vb / z - exact_b * exact_b,
                 var_n = vn / z - exact_n * exact_n;
    options.warmup = 600;
    options.samples = 2000;
    options.n_leapfrog = 24;
    options.target_acceptance = .95;
    options.seed = 19;
    std::vector<FML::MCMC::HMCResult> joint;
    joint.push_back(FF::sample(posterior, exact.mean, p, options, true));
    auto dispersed = p;
    dispersed[0] = .8;
    dispersed[1] += 1;
    const auto other_mean = linear.linear_conditional(data, dispersed, w).mean;
    options.seed = 20;
    joint.push_back(FF::sample(posterior, other_mean, dispersed, options, true));
    const auto db = FML::MCMC::diagnose_hmc_chains(joint, 0), dn = FML::MCMC::diagnose_hmc_chains(joint, 1);
    double hb = 0, hn = 0, vhb = 0, vhn = 0;
    size_t count = 0;
    for (const auto & run : joint)
        for (const auto & row : run.recorded) {
            hb += row[0];
            hn += row[1];
            ++count;
        }
    hb /= count;
    hn /= count;
    for (const auto & run : joint)
        for (const auto & row : run.recorded) {
            vhb += (row[0] - hb) * (row[0] - hb);
            vhn += (row[1] - hn) * (row[1] - hn);
        }
    vhb /= count;
    vhn /= count;
    std::cout << "JOINT_LINEAR b quadrature/HMC " << exact_b << ' ' << hb << " log_noise " << exact_n << ' ' << hn
              << " Rhat " << db.rhat << ' ' << dn.rhat << " ESS " << db.ess_bulk << ' ' << dn.ess_bulk << '\n';
    check_local(db.valid() && dn.valid() && db.rhat < 1.05 && dn.rhat < 1.05 && db.ess_bulk > 60 && dn.ess_bulk > 60 &&
                    db.ess_tail > 40 && dn.ess_tail > 40,
                "dispersed joint chains have usable rank-normalized Rhat and bulk/tail ESS");
    check_local(std::abs(hb - exact_b) < 5 * std::sqrt(var_b / db.ess_bulk) + .002 &&
                    std::abs(hn - exact_n) < 5 * std::sqrt(var_n / dn.ess_bulk) + .002,
                "joint field/bias/noise means agree with exact quadrature within Monte Carlo uncertainty");
    const double be = std::min(db.ess_bulk, db.ess_tail), ne = std::min(dn.ess_bulk, dn.ess_tail);
    check_local(std::abs(vhb / var_b - 1) < 5 * std::sqrt(2 / be) + .03 &&
                    std::abs(vhn / var_n - 1) < 5 * std::sqrt(2 / ne) + .03,
                "joint nuisance posterior widths agree with exact quadrature");
    std::cout << "JOINT_HEALTH " << joint[0].acceptance << ' ' << joint[1].acceptance << " divergences "
              << joint[0].divergences << ' ' << joint[1].divergences << '\n';
    check_local(joint[0].acceptance > .5 && joint[1].acceptance > .5 && joint[0].divergences < options.samples / 200 &&
                    joint[1].divergences < options.samples / 200,
                "joint wide-prior sampler has useful acceptance and a low divergent-proposal rate");
    // BAO wrapper predicts the same biased redshift-space field as the existing application.
    auto bc = config(FF::Evolution::LPT2);
    bc.initial_cutoff = bc.bias_cutoff = bc.likelihood_cutoff = .11;
    const auto spectrum = FML::FIELDLEVEL::BAOSpectrum::eisenstein_hu(200);
    FF::Model bao(bc, FF::Spectrum::bao(spectrum));
    auto bw = bao.workspace();
    auto bp = bao.initial_parameters();
    bp[0] = 1.02;
    bp[1] = 1.4;
    bp[2] = .2;
    bp[3] = -.1;
    FML::FIELDLEVEL::Settings st;
    st.nmesh = bc.mesh;
    st.nmesh_density = bc.density_mesh;
    st.boxsize = bc.boxsize;
    st.kmax = .11;
    st.bias = bp[1];
    st.b2 = bp[2] + 2. / 3 * bp[3];
    st.bG2 = bp[3];
    st.growth_rate = bc.growth_rate;
    st.centering = 0;
    st.model = FML::FIELDLEVEL::Model::LPT;
    FML::FIELDLEVEL::FieldLevelBAO legacy(st, spectrum);
    const auto field = bao.white_noise(77);
    const auto a = bao.predict(field, bp, bw), b = legacy.predict(field, bp[0]);
    double error = 0;
    for (size_t i = 0; i < a.size(); ++i)
        error = std::max(error, std::abs(a[i] - b[i]));
    check_local(error < 1e-11, "BAO compatibility adapter preserves existing 2LPT/bias/RSD predictions");
    auto lc = config(FF::Evolution::Zeldovich);
    FF::Model lpt(lc, FF::Spectrum::fixed([](double) { return 200.; }));
    auto lw = lpt.workspace();
    auto lp = lpt.initial_parameters();
    auto ld = lpt.mock(lpt.white_noise(71), lp, 72, lw);
    FF::Posterior lpost(lpt, ld);
    options.warmup = 80;
    options.samples = 100;
    options.n_leapfrog = 4;
    options.seed = 81;
    auto lchain = FF::sample(lpost, lpt.white_noise(73), lp, options, true);
    bool moved = false;
    for (const auto & row : lchain.recorded)
        moved |= row[1] != lp[1];
    check_local(lchain.recorded.size() == 100 && lchain.acceptance > .3 && moved,
                "joint nonlinear field/tracer/noise inference runs through the same sampler API");
    auto rw = lpt.workspace();
    auto reconstructed = lpt.reconstruct(truth, lp, rw, 30.);
    check_local(reconstructed.initial_density.size() == lpt.cells() &&
                    reconstructed.matter_density.size() == lpt.cells() &&
                    reconstructed.velocity[0].size() == lpt.cells(),
                "reconstruction returns initial/matter fields and Lagrangian velocity samples");
    auto changed = lp;
    changed[0] = 2.1;
    const auto another = lpt.reconstruct(truth, changed, rw, 30.);
    check_local(maxerr(reconstructed.matter_density, another.matter_density) < 1e-12 &&
                    maxerr(reconstructed.velocity[0], another.velocity[0]) < 1e-12,
                "matter reconstruction is independent of tracer nuisance coefficients");
    const double eps = 1e-10;
    bool correct_velocity = true;
    for (int d = 0; d < 3; ++d)
        for (size_t q = 0; q < lpt.cells(); ++q)
            correct_velocity &= std::abs(reconstructed.velocity[d][q] - 30 * reconstructed.displacement[d][q]) < eps;
    check_local(correct_velocity, "Zeldovich velocity factor matches displacement convention");
    check_local(FML::TESTS::throws([&] { linear.predict(Field(1), p, w); }), "framework rejects malformed white field");
    check_local(FML::TESTS::throws([&] { linear.predict(truth, std::vector<double>{}, w); }),
                "framework rejects malformed parameters");
    auto other = linear;
    auto wrong_workspace = other.workspace();
    check_local(FML::TESTS::throws([&] { linear.predict(truth, p, wrong_workspace); }),
                "workspaces belong to their immutable model");
    auto badc = c;
    badc.mesh = 7;
    check_local(FML::TESTS::throws([&] { FF::Model invalid(badc, FF::Spectrum::fixed([](double) { return 200.; })); }),
                "unsupported odd-grid backend is rejected explicitly");
    auto bad_response = FF::FourierResponse{std::vector<double>(linear.modes().size(), 1)};
    for (size_t i = 0; i < linear.modes().size(); ++i)
        if (linear.modes()[i].n[2] == 0) {
            bad_response.multiplier[i] = 2;
            break;
        }
    check_local(FML::TESTS::throws(
                    [&] { FF::Model invalid(c, FF::Spectrum::fixed([](double) { return 200.; }), bad_response); }),
                "response must preserve Hermitian plane symmetry");
    auto bad_data = data;
    for (size_t i = 0; i < linear.modes().size(); ++i)
        if (linear.modes()[i].n[2] == 0) {
            bad_data[i] += Complex(0, .1);
            break;
        }
    check_local(FML::TESTS::throws([&] { FF::Posterior invalid(linear, bad_data); }),
                "likelihood rejects non-Hermitian input data");
    FF::FourierResponse response;
    response.multiplier.resize(linear.modes().size());
    for (size_t i = 0; i < response.multiplier.size(); ++i)
        response.multiplier[i] = std::exp(-100 * linear.modes()[i].k * linear.modes()[i].k);
    FF::Model observed(c, FF::Spectrum::fixed([](double) { return 200.; }), response, {{"b1", 1.7, 1.6, .2, 0, 5}});
    auto ow = observed.workspace();
    auto op = observed.initial_parameters();
    auto od = observed.mock(truth, op, 87, ow);
    FF::Posterior obs_posterior(observed, od);
    auto omap = FF::map(obs_posterior, Field(observed.cells(), 0), op, false, 5, 1e-8);
    const auto conditional = observed.linear_conditional(od, op, ow);
    check_local(observed.parameters()[0].mean == 1.6 && maxerr(omap.white, conditional.mean) < 1e-10,
                "independent Fourier observation response and configurable scalar priors");
    return FML::TESTS::summary("field framework");
}
