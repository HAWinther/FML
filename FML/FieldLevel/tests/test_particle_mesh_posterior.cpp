#include <FML/FieldLevel/ParticleMeshPosterior.h>
#include <FML/tests/TestHelpers.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <fstream>
#include <iomanip>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using FML::FIELDLEVEL::Complex;
using FML::TESTS::check;
static FF::Configuration configuration(int n = 4, int force = 8) {
    FF::Configuration c; c.mesh = n; c.density_mesh = force; c.boxsize = 1;
    c.evolution = FF::Evolution::Linear;
    c.initial_cutoff = c.bias_cutoff = 10.; c.likelihood_cutoff = std::min(15., .9 * M_PI * force);
    return c;
}
static FF::Spectrum spectrum() {
    FF::Spectrum s;
    s.parameters = {{"log_amplitude", .1, 0., .6, -2., 2.}, {"tilt", -.1, 0., .5, -1.5, 1.5}};
    s.evaluate = [](double k, const auto & p) {
        const double l = std::log(k / (2 * M_PI));
        return FF::Power{.003 * std::exp(p[0] + p[1] * l), {1., l}};
    }; return s;
}
static FF::PMSettings settings() {
    FF::PMSettings s; s.background.omega_m = .3; s.scale_factors = {.2, .3, .4, .5, .6}; return s;
}
static FF::Parameter noise() { const double p = std::log(1e-4); return {"log_noise_power", p, p, 1., -15., 0.}; }
static double dot(const Field & a, const Field & b) { double out = 0; for (size_t i = 0; i < a.size(); ++i) out += a[i] * b[i]; return out; }
static double error(const Field & a, const Field & b) {
    if (a.size() != b.size()) return HUGE_VAL;
    double out = 0; for (size_t i = 0; i < a.size(); ++i) out = std::max(out, std::abs(a[i] - b[i])); return out;
}
static Field global(const FF::Model::Layout & l, const Field & local) {
    Field out(l.global_cells); for (size_t q = 0; q < local.size(); ++q) out[l.global_cell_id(q)] = local[q];
    FML::SumArrayOverTasks(out.data(), out.size()); return out;
}
template <class Model>
static Field global_modes(const Model & m, const FF::Modes & local) {
    const int n = m.configuration().density_mesh; Field out(size_t(n) * n * (n / 2 + 1) * 2);
    for (size_t i = 0; i < local.size(); ++i) {
        const auto k = m.modes()[i].n;
        const size_t q = ((size_t((k[0] + n) % n) * n + (k[1] + n) % n) * (n / 2 + 1) + k[2]) * 2;
        out[q] = local[i].real(); out[q + 1] = local[i].imag();
    }
    FML::SumArrayOverTasks(out.data(), out.size()); return out;
}
struct Summary { Field prediction, data, gradient, scalar, fit, reconstruction; double potential{}, fit_potential{}; int status{}; };
template <class Grid>
static Summary exercise(const FML::Domain & domain, const std::string & label, bool recovery,
                        int n = 4, int force = 8, std::ostream * table = nullptr, std::ostream * fields = nullptr, FF::PMForceWindow window = FF::PMForceWindow::None, FF::PMAssignment assignment = FF::PMAssignment::CIC) {
    using Model = FF::PeriodicPMModelT<Grid>;
    auto options = settings(); options.force_window = window; options.assignment = assignment;
    Model m(configuration(n, force), spectrum(), options, noise(), {3}, domain);
    auto w = m.workspace(); auto p = m.initial_parameters(); auto truth = m.white_noise(91);
    auto white = m.white_noise(93); for (auto & x : white) x *= .2;
    const auto mean = m.predict(truth, p, w); const auto true_initial = m.reconstruction(w).initial_density;
    const auto real = global(m.observation_layout(), m.reconstruction(w).matter_density);
    const auto data = m.mock(truth, p, 92, w);
    check(data == m.mock(truth, p, 92, w), label + " fixed-seed mock replay");
    bool high_modes = false; for (const auto & k : m.modes()) high_modes |= k.k > M_PI * n;
    check(n != 4 || m.maximum(high_modes ? 1 : 0) == 1, label + " nonlinear observation support can exceed IC Nyquist");
    double dft_error = 0, noise_error = 0;
    for (size_t i = 0; i < m.modes().size(); ++i) {
        Complex exact = 0, exact_noise = 0;
        const auto k = m.modes()[i].n;
        for (int q = 0; q < force * force * force; ++q) {
            const double phase = -2 * M_PI / force * (k[0] * (q / (force * force)) + k[1] * (q / force % force) + k[2] * (q % force));
            const auto basis = std::exp(Complex(0, phase)) / double(force * force * force);
            exact += basis * real[q];
            FML::RANDOM::COUNTER_BASED::Stream rng(92, q, 0x4649454cu, 0x504d4e53u);
            exact_noise += basis * rng.normal() * std::sqrt(std::exp(p.back()) * force * force * force / m.volume());
        }
        dft_error = std::max(dft_error, std::abs(exact - mean[i]));
        noise_error = std::max(noise_error, std::abs(exact_noise - (data[i] - mean[i])));
    }
    check(m.maximum(dft_error) < 2e-13, label + " force-mesh Fourier prediction matches independent DFT");
    check(m.maximum(noise_error) < 2e-13, label + " noise matches independent ID-stream DFT and P_noise/V normalization");
    FF::PeriodicPMPosteriorT<Grid> posterior(m, data);
    Field grad; std::vector<double> gp;
    Summary out; out.potential = posterior.evaluate(white, p, w, grad, gp);
    out.prediction = global_modes(m, m.predict(white, p, w)); out.data = global_modes(m, data);
    out.gradient = global(m.layout(), grad); out.scalar = gp;
    const auto prediction = m.predict(white, p, w);
    double local_likelihood = 0;
    for (size_t i = 0; i < prediction.size(); ++i)
        local_likelihood += .5 * m.modes()[i].weight * (std::norm(prediction[i] - data[i]) / std::exp(p.back()) * m.volume() + p.back() - std::log(m.volume()));
    double expected = m.sum(local_likelihood + .5 * dot(white, white));
    for (size_t j = 0; j < p.size(); ++j) expected += m.parameters()[j].penalty(p[j]);
    check(std::abs(expected - out.potential) < 2e-11, label + " independent normalized likelihood plus once-only priors");
    {
        auto c = configuration(n, force); c.boxsize = 2.;
        c.initial_cutoff *= .5; c.bias_cutoff *= .5; c.likelihood_cutoff *= .5;
        auto power = spectrum(); const auto base_power = power.evaluate;
        power.evaluate = [base_power](double k, const auto & p) { auto value = base_power(2 * k, p); value.value *= 8; return value; };
        auto np = noise(); np.initial += std::log(8.); np.mean += std::log(8.);
        np.lower += std::log(8.); np.upper += std::log(8.);
        Model scaled(c, power, options, np, {3}, domain); auto sw = scaled.workspace();
        const auto sp = scaled.initial_parameters(); const auto sd = scaled.mock(truth, sp, 92, sw);
        FF::PeriodicPMPosteriorT<Grid> scaled_posterior(scaled, sd);
        Field gs; std::vector<double> gsp;
        const double su = scaled_posterior.evaluate(white, sp, sw, gs, gsp);
        check(std::abs(su - out.potential) < 2e-9 && scaled.maximum(error(gs, grad)) < 2e-9 && error(gsp, gp) < 2e-9,
              label + " box/P/noise volume rescaling preserves posterior and gradients");
    }
    auto direction = m.white_noise(94); for (auto & x : direction) x *= .2;
    for (int component = -1; component < int(p.size()); ++component) {
        const double analytic = component < 0 ? m.sum(dot(grad, direction)) : gp[component];
        double best = HUGE_VAL;
        for (double h : {.002, .0004, .00008, .000016}) {
            auto sp = white, sm = white, pp = p, pm = p;
            if (component < 0) for (size_t q = 0; q < sp.size(); ++q) { sp[q] += h * direction[q]; sm[q] -= h * direction[q]; }
            else { pp[component] += h; pm[component] -= h; }
            Field unused; std::vector<double> scalar;
            const double high = posterior.evaluate(sp, pp, w, unused, scalar), low = posterior.evaluate(sm, pm, w, unused, scalar);
            const double numeric = (high - low) / (2 * h);
            const double relative = std::abs(numeric - analytic) / std::max({1e-8, std::abs(analytic), std::abs(numeric)});
            best = std::min(best, relative);
            if (table && FML::ThisTask == 0) *table << label << ',' << component << ',' << h << ',' << analytic << ',' << numeric << ',' << relative << '\n';
        }
        check(best < 1e-5, label + " posterior white/scalar gradient " + std::to_string(component));
        if (FML::ThisTask == 0) std::cout << "PM_POSTERIOR_ERROR " << label << ' ' << component << ' ' << std::setprecision(17) << best << '\n';
    }
    auto bad = p; if (FML::ThisTask == 0) bad[0] = -3.;
    check(std::isinf(posterior.evaluate(white, bad, w, grad, gp)) && m.maximum(std::abs(dot(grad, grad))) == 0,
          label + " out-of-support proposal rejected collectively");
    auto malformed = white; if (FML::ThisTask == 0) malformed.pop_back();
    check(FML::TESTS::throws([&] { posterior.evaluate(malformed, p, w, grad, gp); }), label + " one-rank malformed state rejected");
    auto invalid_data = data; if (FML::ThisTask == 0) invalid_data.push_back(0.);
    check(FML::TESTS::throws([&] { FF::PeriodicPMPosteriorT<Grid> invalid(m, invalid_data); }), label + " one-rank malformed data rejected");
    check(m.apply_field_metric(white, p, w, -1) == white, label + " explicit identity white-field metric");
    if (recovery) {
        // Separate conservative-band recovery from the broader support/gradient
        // experiment above. Short-wave CIC modes can place the MAP on a cell face.
        auto fit_configuration = configuration(n, force); fit_configuration.likelihood_cutoff = 10.;
        Model resolved(fit_configuration, spectrum(), settings(), noise(), {3}, domain);
        const auto & m = resolved; auto w = m.workspace(); auto p = m.initial_parameters();
        const auto data = m.mock(truth, p, 92, w);
        const auto true_initial = m.reconstruction(w).initial_density;
        FF::PeriodicPMPosteriorT<Grid> posterior(m, data);
        auto start = m.white_noise(95); for (auto & x : start) x *= .001;
        const double initial_potential = posterior.evaluate(start, p, w, grad, gp);
        const auto initial_density = m.reconstruction(w).initial_density;
        const auto fit = FF::map(posterior, start, p, false, 200, 2e-5);
        out.fit = global(m.layout(), fit.white); out.fit_potential = fit.potential; out.status = int(fit.status);
        m.predict(fit.white, fit.parameters, w); const auto fitted = m.reconstruction(w).initial_density;
        out.reconstruction = global(m.layout(), fitted);
        double before = 0, after = 0, cross = 0, a2 = 0, b2 = 0;
        for (size_t q = 0; q < truth.size(); ++q) {
            before += std::pow(initial_density[q] - true_initial[q], 2);
            after += std::pow(fitted[q] - true_initial[q], 2);
            cross += fitted[q] * true_initial[q]; a2 += fitted[q] * fitted[q]; b2 += true_initial[q] * true_initial[q];
        }
        before = m.sum(before); after = m.sum(after);
        const double correlation = m.sum(cross) / std::sqrt(m.sum(a2) * m.sum(b2));
        check(fit.potential < initial_potential - 1e-3, label + " MAP lowers posterior");
        check(after < .6 * before && correlation > .8, label + " MAP improves initial-density error and truth correlation");
        check(fit.status == FF::MAPStatus::Converged, label + " fixed-parameter MAP reaches gradient tolerance");
        const auto joint = FF::map(posterior, start, p, true, 8, 2e-5);
        check(joint.potential < initial_potential && m.in_support(joint.parameters) && error(joint.parameters, p) > 1e-6,
              label + " bounded joint MAP updates spectrum/noise with lower posterior");
        if (FML::ThisTask == 0) std::cout << "PM_MAP " << label << ' ' << initial_potential << ' ' << fit.potential << ' '
            << fit.gradient_norm << ' ' << fit.iterations << ' ' << int(fit.status) << ' ' << after / before << ' ' << correlation
            << " cutoff=" << m.configuration().likelihood_cutoff << '\n';
        const auto truth_field = global(m.layout(), true_initial), start_field = global(m.layout(), initial_density);
        if (fields && FML::ThisTask == 0) for (size_t q = 0; q < truth_field.size(); ++q)
            *fields << label << ',' << q << ',' << truth_field[q] << ',' << start_field[q] << ',' << out.reconstruction[q] << '\n';
    }
    return out;
}
[[maybe_unused]] static void compare(const Summary & a, const Summary & b, const std::string & name, bool map) {
    double e = std::max({std::abs(a.potential - b.potential), error(a.gradient, b.gradient), error(a.scalar, b.scalar),
                         error(a.prediction, b.prediction), error(a.data, b.data)});
    if (map) e = std::max({e, error(a.fit, b.fit), error(a.reconstruction, b.reconstruction), std::abs(a.fit_potential - b.fit_potential)});
    check(e < (map ? 2e-7 : 2e-11), name + " posterior/gradient/mock/MAP backend equality");
    if (FML::ThisTask == 0) std::cout << "PM_POSTERIOR_BACKEND_ERROR " << name << ' ' << e << '\n';
}
int main(int argc, char ** argv) {
    const auto slab = FML::Domain::uniform_slab(FML::NTasks); FML::set_domain(slab);
    std::ofstream table, fields;
    if (argc > 1 && FML::ThisTask == 0) {
        table.open(std::string(argv[1]) + ".gradients.csv"); fields.open(std::string(argv[1]) + ".fields.csv");
        table << std::setprecision(17) << "backend,component,h,analytic,numeric,relative_error\n";
        fields << std::setprecision(17) << "backend,cell,truth,start,map\n";
    }
    const auto baseline = exercise<FML::GRID::FFTWGrid<3>>(slab, "slab", true, 4, 8, &table, &fields);
    const auto dense = exercise<FML::GRID::FFTWGrid<3>>(slab, "dense-slab", false, 8, 4, &table);
    const auto compensated = exercise<FML::GRID::FFTWGrid<3>>(slab, "compensated-slab", false, 8, 4, &table, nullptr, FF::PMForceWindow::CICCompensated);
    const auto tsc = exercise<FML::GRID::FFTWGrid<3>>(slab, "TSC-compensated", false, 8, 4, &table, nullptr, FF::PMForceWindow::TSCCompensated, FF::PMAssignment::TSC);
    check(FML::TESTS::throws([&] { auto bad = noise(); bad.sigma = 0; FF::PeriodicPMModel invalid(configuration(), spectrum(), settings(), bad); }),
          "invalid noise prior rejected");
#ifdef USE_MPI
    const int px = FML::NTasks == 4 ? 2 : FML::NTasks, py = FML::NTasks / px;
    const auto pencil = FML::Domain::uniform_pencil(px, py); FML::set_domain(pencil);
    compare(baseline, exercise<FML::GRID::PencilGrid<3>>(pencil, "pencil", true, 4, 8, &table, &fields), "pencil", true);
    compare(dense, exercise<FML::GRID::PencilGrid<3>>(pencil, "dense-pencil", false, 8, 4, &table), "dense-pencil", false);
    compare(compensated, exercise<FML::GRID::PencilGrid<3>>(pencil, "compensated-pencil", false, 8, 4, &table, nullptr, FF::PMForceWindow::CICCompensated), "compensated-pencil", false);
    compare(tsc, exercise<FML::GRID::PencilGrid<3>>(pencil, "TSC-pencil", false, 8, 4, &table, nullptr, FF::PMForceWindow::TSCCompensated, FF::PMAssignment::TSC), "TSC-pencil", false);
    if (FML::NTasks == 4) {
        exercise<FML::GRID::PencilGrid<3>>(pencil, "unequal-N10", false, 10, 20, &table);
    }
    if (FML::NTasks > 1) {
        FML::set_domain(slab);
        check(FML::TESTS::throws([&] { auto bad = noise(); bad.mean += FML::ThisTask; FF::PeriodicPMModel invalid(configuration(), spectrum(), settings(), bad); }),
              "rank-dependent noise priors rejected collectively");
    }
#endif
    if (argc > 1 && FML::ThisTask == 0) {
        std::ofstream f(std::string(argv[1]) + ".snapshot"); f << std::setprecision(17) << baseline.potential << '\n' << baseline.fit_potential << '\n';
        for (const auto * x : {&baseline.prediction, &baseline.data, &baseline.gradient, &baseline.scalar, &baseline.fit, &baseline.reconstruction})
            for (double y : *x) f << y << '\n';
    }
    return FML::TESTS::summary("FieldLevel periodic PM posterior");
}
