#include <FML/FieldLevel/ParticleMeshAdjoint.h>
#include <FML/tests/TestHelpers.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <fstream>
#include <iomanip>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
static FF::Configuration configuration(int n = 8, int force = 16) {
    FF::Configuration c; c.mesh = n; c.density_mesh = force; c.boxsize = 1;
    c.evolution = FF::Evolution::Linear;
    c.initial_cutoff = c.bias_cutoff = c.likelihood_cutoff = 10.;
    return c;
}
static FF::Spectrum spectrum() {
    FF::Spectrum p;
    p.parameters = {{"log_amplitude", .1, 0., 1.}, {"tilt", -.2, 0., 1.}};
    p.evaluate = [](double k, const auto & theta) {
        const double logk = std::log(k / (2 * M_PI));
        return FF::Power{.004 * std::exp(theta[0] + theta[1] * logk), {1., logk}};
    };
    return p;
}
static FF::PMSettings settings(double om = .3, int steps = 8) {
    FF::PMSettings s; s.background.omega_m = om; s.scale_factors.clear();
    for (int i = 0; i <= steps; ++i) s.scale_factors.push_back(.2 + .4 * i / steps);
    return s;
}
static double dot(const Field & a, const Field & b) {
    double value = 0; if (!a.empty() && !b.empty()) for (size_t i = 0; i < a.size(); ++i) value += a[i] * b[i];
    return value;
}
template <class Result>
static double objective(const Result & r, const FF::PMCotangent & c) {
    double value = dot(r.initial_density, c.initial_density) + dot(r.matter_density, c.matter_density);
    for (int d = 0; d < 3; ++d) value += dot(r.displacement[d], c.displacement[d]) + dot(r.momentum[d], c.momentum[d]);
    FML::SumOverTasks(&value); return value;
}
static FF::PMCotangent cotangent(const FF::Model::Layout & ic, const FF::Model::Layout & mesh) {
    FF::PMCotangent c;
    c.initial_density.resize(ic.owned_cells); c.matter_density.resize(mesh.owned_cells);
    for (int d = 0; d < 3; ++d) { c.displacement[d].resize(ic.owned_cells); c.momentum[d].resize(ic.owned_cells); }
    for (size_t q = 0; q < ic.owned_cells; ++q) {
        const double id = ic.global_cell_id(q);
        c.initial_density[q] = .23 * std::cos(.37 * id) / ic.global_cells;
        for (int d = 0; d < 3; ++d) {
            c.displacement[d][q] = 2 * std::sin(.021 * id + d) / ic.global_cells;
            c.momentum[d][q] = 3 * std::cos(.031 * id + d) / ic.global_cells;
        }
    }
    for (size_t q = 0; q < mesh.owned_cells; ++q) {
        const double id = mesh.global_cell_id(q);
        c.matter_density[q] = (std::sin(.17 * id) + .4 * std::cos(.053 * id)) / mesh.global_cells;
    }
    return c;
}
static double error(const Field & a, const Field & b) {
    if (a.size() != b.size()) return HUGE_VAL;
    double e = 0; for (size_t j = 0; j < a.size(); ++j) e = std::max(e, std::abs(a[j] - b[j])); return e;
}
static Field global(const FF::Model::Layout & layout, const Field & local) {
    Field out(layout.global_cells);
    for (size_t q = 0; q < local.size(); ++q) out[layout.global_cell_id(q)] = local[q];
    FML::SumArrayOverTasks(out.data(), out.size()); return out;
}
struct Summary { Field gradient, spectrum; double objective{}, directional_error{}; size_t low_replay{}, high_replay{}; };
template <class Grid>
static Summary exercise(const FML::Domain & domain, const std::string & label,
                        int n = 8, int nf = 16, double om = .3, int steps = 8, bool detailed = true,
                        std::ostream * table = nullptr, FF::PMForceWindow window = FF::PMForceWindow::None, FF::PMAssignment assignment = FF::PMAssignment::CIC) {
    auto options = settings(om, steps); options.force_window = window; options.assignment = assignment;
    FF::ParticleMeshForwardT<Grid> m(configuration(n, nf), spectrum(), options, domain);
    FF::ParticleMeshAdjointT<Grid> adjoint(m);
    auto w = m.workspace(); auto s = m.white_noise(91); for (auto & x : s) x *= .7;
    auto theta = m.initial_parameters();
    check(FML::TESTS::throws([&] { adjoint.pullback({}, w); }), label + " adjoint rejects missing forward");
    const auto & r = m.evolve(s, theta, w);
    for (const auto & node : r.diagnostics) check(std::abs(node.mass_error) < 2e-12, label + " assignment conserves mass");
    auto c = cotangent(m.layout(), r.density_layout);
    const auto stored_density = r.matter_density, stored_initial = r.initial_density;
    const auto stored_position = r.displacement, stored_momentum = r.momentum;
    const auto grad = adjoint.pullback(c, w, {4});
    const auto low = adjoint.pullback(c, w, {0});
    const auto capped = adjoint.pullback(c, w, {1});
    double e = m.maximum(error(grad.white, low.white));
    e = std::max(e, error(grad.spectrum, low.spectrum));
    check(e < 2e-12, label + " bounded checkpoints and full-prefix recomputation give the same gradient");
    check(low.report.peak_checkpoint_states == 1 && grad.report.peak_checkpoint_states <= 5 &&
              low.report.replayed_steps > grad.report.replayed_steps &&
              low.report.checkpoint_payload_bytes == 6 * m.cells() * sizeof(double),
          label + " checkpoint limit bounds storage and reduces replay work");
    check(capped.report.peak_checkpoint_states <= 2 && m.maximum(error(capped.white, grad.white)) < 2e-12 &&
              error(capped.spectrum, grad.spectrum) < 2e-12,
          label + " single midpoint slot obeys budget with identical derivatives");
    {
        const auto saved_s = s, saved_theta = theta;
        for (auto & x : s) x += .1;
        theta[0] += .2;
        const auto cached = adjoint.pullback(c, w);
        check(m.maximum(error(cached.white, grad.white)) == 0 && cached.spectrum == grad.spectrum,
              label + " reverse uses cached inputs after caller mutates its input arrays");
        s = saved_s; theta = saved_theta;
    }
    bool unchanged = stored_density == w.result().matter_density && stored_initial == w.result().initial_density &&
                     stored_position == w.result().displacement && stored_momentum == w.result().momentum;
    check(unchanged, label + " reverse leaves public forward result unchanged");
    auto direction = m.white_noise(93); for (auto & x : direction) x *= .2;
    std::vector<double> dt{.2, -.3};
    auto finite_difference = [&](const FF::PMCotangent & covector, const FF::PMGradient & g,
                                 bool move_white, bool move_theta, const std::string & channel) {
        double analytic = move_white ? m.sum(dot(g.white, direction)) : 0;
        if (move_theta) analytic += dot(g.spectrum, dt);
        double best = HUGE_VAL;
        for (double h : {.01, .002, .0004, .00008}) {
            auto plus = s, minus = s, tp = theta, tm = theta;
            if (move_white) for (size_t q = 0; q < s.size(); ++q) { plus[q] += h * direction[q]; minus[q] -= h * direction[q]; }
            if (move_theta) for (size_t j = 0; j < theta.size(); ++j) { tp[j] += h * dt[j]; tm[j] -= h * dt[j]; }
            const double high = objective(m.evolve(plus, tp, w), covector);
            const double low_value = objective(m.evolve(minus, tm, w), covector);
            const double numeric = (high - low_value) / (2 * h);
            const double rel = std::abs(numeric - analytic) / std::max({1e-9, std::abs(numeric), std::abs(analytic)});
            best = std::min(best, rel);
            if (table && FML::ThisTask == 0) *table << label << ',' << channel << ',' << h << ',' << analytic << ',' << numeric << ',' << rel << '\n';
        }
        check(best < 8e-6, label + " " + channel + " directional FD / transpose identity");
        if (FML::ThisTask == 0) std::cout << std::setprecision(17) << "PM_ADJOINT_ERROR " << label << ' ' << channel << ' ' << best << '\n';
        return best;
    };
    Summary out;
    out.directional_error = finite_difference(c, grad, true, true, "joint");
    if (detailed) {
        finite_difference(c, grad, true, false, "white");
        finite_difference(c, grad, false, true, "spectrum");
        for (int channel = 0; channel < 4; ++channel) {
            FF::PMCotangent one;
            if (channel == 0) one.initial_density = c.initial_density;
            if (channel == 1) one.matter_density = c.matter_density;
            if (channel == 2) one.displacement = c.displacement;
            if (channel == 3) one.momentum = c.momentum;
            m.evolve(s, theta, w); const auto g = adjoint.pullback(one, w);
            finite_difference(one, g, true, true, "output-" + std::to_string(channel));
        }
        const auto original_direction = direction, original_dt = dt;
        for (size_t cell : {size_t(0), m.layout().global_cells / 3, m.layout().global_cells - 1}) {
            direction.assign(m.cells(), 0.);
            for (size_t q = 0; q < m.cells(); ++q) if (m.layout().global_cell_id(q) == cell) direction[q] = 1.;
            finite_difference(c, grad, true, false, "white-cell-" + std::to_string(cell));
        }
        direction = original_direction;
        for (size_t j = 0; j < dt.size(); ++j) {
            dt.assign(dt.size(), 0.); dt[j] = 1.;
            finite_difference(c, grad, false, true, "spectrum-component-" + std::to_string(j));
        }
        dt = original_dt;
    }
    m.evolve(s, theta, w);
    const auto zero = adjoint.pullback({}, w);
    double norm = 0; for (double x : zero.white) norm = std::max(norm, std::abs(x));
    for (double x : zero.spectrum) norm = std::max(norm, std::abs(x));
    check(m.maximum(norm) == 0, label + " empty cotangents give exact zero gradients");
    auto bad = c;
    if (FML::ThisTask == 0) bad.matter_density.push_back(1.);
    check(FML::TESTS::throws([&] { adjoint.pullback(bad, w); }), label + " one-rank wrong cotangent size rejected");
    bad = c; if (FML::ThisTask == 0) bad.displacement[0][0] = HUGE_VAL;
    check(FML::TESTS::throws([&] { adjoint.pullback(bad, w); }), label + " one-rank nonfinite cotangent rejected");
    check(FML::TESTS::throws([&] { adjoint.pullback(c, w, {33}); }), label + " invalid checkpoint budget rejected");
    // Empty local optional fields are legal independently of other ranks.
    bad = c; if (FML::ThisTask == 0) { bad.matter_density.clear(); bad.initial_density.clear(); }
    const auto partial = adjoint.pullback(bad, w);
    finite_difference(bad, partial, true, true, "rank-local-zeros");
    m.evolve(s, theta, w);
    out.objective = objective(w.result(), c);
    out.gradient = global(m.layout(), grad.white); out.spectrum = grad.spectrum;
    out.low_replay = low.report.replayed_steps; out.high_replay = grad.report.replayed_steps;
    if (FML::ThisTask == 0) std::cout << "PM_REPLAY " << label << ' ' << low.report.replayed_steps << ' '
        << grad.report.replayed_steps << ' ' << low.report.peak_checkpoint_states << ' ' << grad.report.peak_checkpoint_states << '\n';
    return out;
}
[[maybe_unused]] static void compare(const Summary & a, const Summary & b, const std::string & label) {
    const double e = std::max({error(a.gradient, b.gradient), error(a.spectrum, b.spectrum), std::abs(a.objective - b.objective)});
    check(e < 2e-12, label + " canonical gradient/spectrum/objective equality");
    if (FML::ThisTask == 0) std::cout << "PM_ADJOINT_BACKEND_ERROR " << label << ' ' << e << '\n';
}
int main(int argc, char ** argv) {
    const auto slab = FML::Domain::uniform_slab(FML::NTasks); FML::set_domain(slab);
    std::ofstream table;
    if (argc > 1 && FML::ThisTask == 0) {
        table.open(argv[1]); table << std::setprecision(17) << "backend,channel,h,analytic,numeric,relative_error\n";
    }
    for(int invalid:{0,4})check(FML::TESTS::throws([&]{auto s=settings();s.assignment=static_cast<FF::PMAssignment>(invalid);FF::ParticleMeshForward bad(configuration(),spectrum(),s);}),"invalid PM assignment rejected");
    check(FML::TESTS::throws([&]{auto s=settings();s.assignment=FF::PMAssignment::TSC;s.force_window=FF::PMForceWindow::CICCompensated;FF::ParticleMeshForward bad(configuration(),spectrum(),s);}),"mismatched force compensation rejected");
    auto baseline = exercise<FML::GRID::FFTWGrid<3>>(slab, "slab", 8, 16, .3, 8, true, &table);
    const auto dense = exercise<FML::GRID::FFTWGrid<3>>(slab, "dense-slab", 8, 4, .3, 4, true, &table);
    const auto compensated = exercise<FML::GRID::FFTWGrid<3>>(slab, "CIC-compensated", 8, 4, .3, 4, true, &table, FF::PMForceWindow::CICCompensated);
    const auto tsc = exercise<FML::GRID::FFTWGrid<3>>(slab, "TSC", 8, 4, .3, 4, true, &table, FF::PMForceWindow::None, FF::PMAssignment::TSC);
    const auto tsc_comp = exercise<FML::GRID::FFTWGrid<3>>(slab, "TSC-compensated", 8, 4, .3, 4, true, &table, FF::PMForceWindow::TSCCompensated, FF::PMAssignment::TSC);
    exercise<FML::GRID::FFTWGrid<3>>(slab, "EdS-small", 4, 4, 1., 4, false, &table);
#ifdef USE_MPI
    for (auto shape : {std::array<int, 2>{FML::NTasks, 1}, {1, FML::NTasks}}) {
        if (shape[0] > 8 || shape[1] > 5) continue;
        const auto domain = FML::Domain::uniform_pencil(shape[0], shape[1]); FML::set_domain(domain);
        auto value = exercise<FML::GRID::PencilGrid<3>>(domain, "pencil", 8, 16, .3, 8, false, &table);
        compare(baseline, value, "pencil");
        if (shape[1] <= 3)
            compare(compensated, exercise<FML::GRID::PencilGrid<3>>(domain, "compensated-pencil", 8, 4, .3, 4, false, &table, FF::PMForceWindow::CICCompensated), "compensated-pencil");
        if (shape[1] <= 3) {
            compare(tsc, exercise<FML::GRID::PencilGrid<3>>(domain, "TSC-pencil", 8, 4, .3, 4, false, &table, FF::PMForceWindow::None, FF::PMAssignment::TSC), "TSC-pencil");
            compare(tsc_comp, exercise<FML::GRID::PencilGrid<3>>(domain, "TSC-compensated-pencil", 8, 4, .3, 4, false, &table, FF::PMForceWindow::TSCCompensated, FF::PMAssignment::TSC), "TSC-compensated-pencil");
        }
        if (shape[1] <= 3)
            compare(dense, exercise<FML::GRID::PencilGrid<3>>(domain, "dense-pencil", 8, 4, .3, 4, false, &table), "dense-pencil");
    }
    if (FML::NTasks == 4) {
        const auto domain = FML::Domain::uniform_pencil(2, 2); FML::set_domain(domain);
        compare(baseline, exercise<FML::GRID::PencilGrid<3>>(domain, "2x2", 8, 16, .3, 8, false, &table), "2x2");
        compare(tsc_comp, exercise<FML::GRID::PencilGrid<3>>(domain, "TSC-2x2", 8, 4, .3, 4, false, &table, FF::PMForceWindow::TSCCompensated, FF::PMAssignment::TSC), "TSC-2x2");
        exercise<FML::GRID::PencilGrid<3>>(domain, "unequal-N10", 10, 20, .3, 4, false, &table);
    }
    if (FML::NTasks > 1) {
        FML::set_domain(slab);
        check(FML::TESTS::throws([&]{auto s=settings();s.assignment=FML::ThisTask?FF::PMAssignment::TSC:FF::PMAssignment::CIC;FF::ParticleMeshForward bad(configuration(),spectrum(),s);}),"rank-dependent assignment rejected collectively");
        FF::ParticleMeshForward m(configuration(), spectrum(), settings()); auto w = m.workspace();
        m.evolve(m.white_noise(1), m.initial_parameters(), w);
        FF::ParticleMeshAdjoint a(m);
        check(FML::TESTS::throws([&] { a.pullback({}, w, {size_t(FML::ThisTask == 0 ? 0 : 1)}); }),
              "rank-dependent checkpoint budgets rejected collectively");
    }
#endif
    if (argc > 1 && FML::ThisTask == 0) {
        std::ofstream f(std::string(argv[1]) + ".snapshot"); f << std::setprecision(17);
        f << baseline.objective << '\n'; for (double x : baseline.gradient) f << x << '\n';
        for (double x : baseline.spectrum) f << x << '\n';
    }
    return FML::TESTS::summary("FieldLevel PM adjoint");
}
