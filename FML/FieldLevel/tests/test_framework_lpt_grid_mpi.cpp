#include <FML/PencilDomain/PencilGrid.h>
#include <FML/FieldLevel/DistributedLPT.h>
#include <FML/tests/TestHelpers.h>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Complex;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
using Pencil = FML::GRID::PencilGrid<3>;
using PencilModel = FF::DistributedLPTModelT<Pencil>;
static double rel(double a, double b) { return FML::TESTS::relative_difference(a, b); }
static double dot(const Field & a, const Field & b) {
    double v = 0;
    for (size_t i = 0; i < a.size(); ++i)
        v += a[i] * b[i];
    return v;
}
static double error(const Field & a, const Field & b) {
    if (a.size() != b.size())
        return HUGE_VAL;
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, rel(a[i], b[i]));
    return e;
}
template <class Model>
Field global_field(const Model & m, const Field & x) {
    Field all(m.layout().global_cells, 0);
    for (size_t q = 0; q < x.size(); ++q)
        all[m.layout().global_cell_id(q)] = x[q];
    FML::SumArrayOverTasks(all.data(), all.size());
    return all;
}
template <class Model>
Field global_modes(const Model & m, const FF::Modes & x) {
    const int n = m.configuration().mesh;
    Field all(size_t(n) * n * (n / 2 + 1) * 2, 0);
    for (size_t i = 0; i < x.size(); ++i) {
        const auto & k = m.modes()[i].n;
        const size_t q = ((size_t((k[0] + n) % n) * n + (k[1] + n) % n) * (n / 2 + 1) + k[2]) * 2;
        all[q] = x[i].real();
        all[q + 1] = x[i].imag();
    }
    FML::SumArrayOverTasks(all.data(), all.size());
    return all;
}
static FF::Spectrum spectrum() {
    FF::Spectrum s;
    s.parameters = {{"log_amplitude", .1, 0, 1}, {"tilt", .2, 0, 1}};
    s.evaluate = [](double k, const auto & p) {
        return FF::Power{200 * std::exp(p[0]) * std::pow(k / .1, p[1]), {1., std::log(k / .1)}};
    };
    return s;
}
static std::vector<double> edges(int parts, int n) {
    std::vector<double> e(parts + 1);
    for (int j = 0; j <= parts; ++j)
        e[j] = double(j * n / parts) / n;
    return e;
}
static bool use_pswf = false;
static FF::Configuration config(int n = 8, int nd = 16) {
    FF::Configuration c;
    c.matter_assignment = use_pswf ? FF::MatterAssignment::PSWF : FF::MatterAssignment::Smooth;
    c.mesh = n;
    c.density_mesh = nd;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .18;
    c.growth_rate = .6;
    c.evolution = FF::Evolution::LPT2;
    return c;
}
// Independent analytic Fourier fields: no source/destination index mapping is reused.
template <class Grid>
void transfer_and_assignment(const FF::Configuration & c, const FML::Domain & domain, const std::string & label) {
    auto coarse = FML::GRID::make_grid<Grid>(c.mesh, 0, 0, domain);
    auto fine = make_grid_like(coarse, c.density_mesh, 0, 0);
    FF::DETAIL::LPTResampling<Grid> up(coarse, fine), down(fine, coarse);
    auto a = [](auto k) { return Complex(1 + .03 * k[0] - .02 * k[1], .04 * k[2] + .01 * k[0]); };
    auto b = [](auto k) { return Complex(.6 - .02 * k[2], .01 * k[1] - .03 * k[0]); };
    for (auto i : coarse.get_fourier_range())
        coarse.set_fourier_from_index(i, a(coarse.get_fourier_integer_wavevector_from_index(i)));
    coarse.set_grid_status_real(false);
    up.apply(coarse, fine);
    double e = 0, left = 0;
    for (auto i : fine.get_fourier_range()) {
        const auto k = fine.get_fourier_integer_wavevector_from_index(i);
        bool keep = true;
        for (int v : k)
            keep &= std::abs(v) < c.mesh / 2;
        const Complex expected = keep ? a(k) : Complex(0);
        e = std::max(e, std::abs(Complex(fine.get_fourier_from_index(i)) - expected));
        left += FML::GRID::fourier_mode_multiplicity(k[2], c.density_mesh) *
                std::real(std::conj(Complex(fine.get_fourier_from_index(i))) * b(k));
        fine.set_fourier_from_index(i, b(k));
    }
    fine.set_grid_status_real(false);
    down.apply(fine, coarse);
    double right = 0;
    for (auto i : coarse.get_fourier_range()) {
        const auto k = coarse.get_fourier_integer_wavevector_from_index(i);
        bool keep = true;
        for (int v : k)
            keep &= std::abs(v) < c.mesh / 2;
        e = std::max(e, std::abs(Complex(coarse.get_fourier_from_index(i)) - (keep ? b(k) : Complex(0))));
        right += FML::GRID::fourier_mode_multiplicity(k[2], c.mesh) *
                 std::real(std::conj(a(k)) * Complex(coarse.get_fourier_from_index(i)));
    }
    FML::MaxOverTasks(&e);
    FML::SumOverTasks(&left);
    FML::SumOverTasks(&right);
    check(e < 1e-13 && rel(left, right) < 2e-13,
          label + " normalized Fourier transfer and weighted transpose (DC/Nyquist included)");
    FF::DETAIL::DistributedAssignmentT<Grid> scatter(fine);
    const size_t nd = c.density_mesh, ntot = nd * nd * nd;
    auto bar = [](size_t id) { return std::cos(.017 * id); };
    double expected = 0;
    auto requests = [&](auto add) {
        if (FML::NTasks > 1 && FML::ThisTask == 0)
            return;
        for (size_t q = 0; q < 23; ++q) {
            const size_t id = size_t(FML::ThisTask) * 23 + q;
            add((id * 37 + 11) % ntot, 1 + std::sin(.1 * id));
            add((ntot - 1 - id % ntot), -.3);
        }
    };
    requests([&](size_t node, double value) { expected += value * bar(node); });
    check(FML::TESTS::throws([&] { scatter.reverse(fine); }), label + " reverse requires matching scatter");
    scatter.deposit(fine, requests);
    double actual = 0;
    for (auto i : fine.get_real_range()) {
        auto local = fine.get_coord_from_index(i);
        size_t id = 0;
        for (int d = 0; d < 3; ++d)
            id = id * nd + local[d] + fine.get_local_range(d)[0];
        actual += fine.get_real_from_index(i) * bar(id);
        fine.set_real_from_index(i, bar(id));
    }
    auto reverse = scatter.reverse(fine);
    e = 0;
    requests([&](size_t node, double) { e = std::max(e, std::abs(scatter.at(reverse, node) - bar(node))); });
    FML::MaxOverTasks(&e);
    FML::SumOverTasks(&actual);
    FML::SumOverTasks(&expected);
    check(rel(actual, expected) < 2e-13 && e < 1e-14,
          label + " sparse scatter/reverse dot product with collisions/distant nodes/empty requests");
}
struct Snapshot {
    Field prediction, mock, gradient, scalar, large_gradient, reconstruction, metric, chain;
    double potential{}, large_potential{};
};
template <class Model>
Snapshot exercise(const FF::Configuration & c, const FML::Domain & domain, const std::string & label) {
    auto lc = c;
    lc.evolution = FF::Evolution::Linear;
    lc.matter_assignment = FF::MatterAssignment::Smooth;
    typename Model::LinearModel metadata(lc, spectrum(), {}, {}, domain);
    FF::FourierResponse response;
    for (const auto & mode : metadata.modes())
        response.multiplier.push_back(.9 + .02 * mode.n[0] * mode.n[0]);
    Model m(c, spectrum(), response, {}, domain);
    auto w = m.workspace();
    auto p = m.initial_parameters();
    p[2] = 1.4;
    p[3] = .2;
    p[4] = -.15;
    p[5] = 12;
    auto s = m.white_noise(11), direction = m.white_noise(12);
    auto data = m.mock(m.white_noise(13), p, 14, w);
    FF::BasicPosterior<Model> posterior(m, data);
    Field gradient, tmp;
    std::vector<double> gp, pg;
    Snapshot out;
    out.potential = posterior.evaluate(s, p, w, gradient, gp);
    out.gradient = global_field(m, gradient);
    out.scalar = gp;
    out.mock = global_modes(m, data);
    const double h = 1e-5;
    auto plus = s, minus = s;
    for (size_t q = 0; q < s.size(); ++q) {
        plus[q] += h * direction[q];
        minus[q] -= h * direction[q];
    }
    double fd = (posterior.evaluate(plus, p, w, tmp, pg) - posterior.evaluate(minus, p, w, tmp, pg)) / (2 * h);
    const double field_error = rel(fd, m.sum(dot(gradient, direction)));
    check(field_error < 2e-6, label + " posterior field finite difference");
    double scalar_error = 0;
    for (size_t j = 0; j < p.size(); ++j) {
        auto a = p, b = p;
        double step = h * std::max(1., std::abs(p[j]));
        a[j] += step;
        b[j] -= step;
        fd = (posterior.evaluate(s, a, w, tmp, pg) - posterior.evaluate(s, b, w, tmp, pg)) / (2 * step);
        scalar_error = std::max(scalar_error, rel(fd, gp[j]));
    }
    check(scalar_error < 2e-6, label + " all spectrum/bias/noise scalar finite differences");
    auto prediction = m.predict(s, p, w);
    out.prediction = global_modes(m, prediction);
    m.pullback(p, data, w, tmp, pg);
    const double pull = m.sum(dot(tmp, direction));
    auto yp = m.predict(plus, p, w), ym = m.predict(minus, p, w);
    double mode_dot = 0;
    for (size_t i = 0; i < data.size(); ++i)
        mode_dot += m.modes()[i].weight * std::real(std::conj(data[i]) * (yp[i] - ym[i])) / (2 * h);
    const double adjoint_error = rel(pull, m.sum(mode_dot));
    check(adjoint_error < 2e-6, label + " forward/adjoint dot product through deposition/resampling/2LPT");
    auto large = s;
    for (auto & x : large)
        x *= 80;
    out.large_potential = posterior.evaluate(large, p, w, tmp, pg);
    out.large_gradient = global_field(m, tmp);
    const double analytic = m.sum(dot(tmp, direction));
    auto la = large, lb = large;
    for (size_t q = 0; q < s.size(); ++q) {
        la[q] += h * direction[q];
        lb[q] -= h * direction[q];
    }
    fd = (posterior.evaluate(la, p, w, tmp, pg) - posterior.evaluate(lb, p, w, tmp, pg)) / (2 * h);
    const double large_error = rel(fd, analytic);
    check(large_error < 2e-6, label + " large-displacement wrap/rank-crossing adjoint");
    auto rec = m.reconstruct(s, p, w, 7.);
    auto append = [&](const Field & f) {
        auto all = global_field(m, f);
        out.reconstruction.insert(out.reconstruction.end(), all.begin(), all.end());
    };
    append(rec.initial_density);
    append(rec.matter_density);
    for (int d = 0; d < 3; ++d) {
        append(rec.displacement[d]);
        append(rec.velocity[d]);
    }
    check(error(global_modes(m, rec.tracer), out.prediction) < 1e-12,
          label + " reconstruction retains observed tracer and physical fields");
    check(FML::TESTS::throws([&] { m.pullback(p, data, w, tmp, pg); }),
          label + " reconstruction invalidates forward cache");
    out.metric = global_field(m, m.apply_field_metric(s, p, w, -.5));
    auto fit = FF::map(posterior, s, p, false, 3, 1e-7);
    check(std::isfinite(fit.potential) && fit.potential < out.potential, label + " generic MAP lowers objective");
    FML::MCMC::HMCOptions o;
    o.warmup = 0;
    o.samples = 3;
    o.n_leapfrog = 2;
    o.jitter = false;
    o.adapt_step_size = false;
    o.adapt_mass = false;
    o.step_size = .001;
    o.seed = 811;
    auto chain = FF::sample(posterior, s, p, o, true);
    out.chain = global_field(m, Field(chain.last.begin(), chain.last.begin() + m.cells()));
    out.chain.insert(out.chain.end(), chain.recorded.back().begin(), chain.recorded.back().end());
    bool sync = true;
    for (const auto & row : chain.recorded)
        for (double x : row)
            sync &= m.maximum(x) == -m.maximum(-x);
    check(chain.divergences == 0 && sync, label + " joint HMC uses generic execution and replicated scalar decisions");
    check(FML::TESTS::throws([&] {
              auto bad = s;
              if (FML::ThisTask == 0)
                  bad.pop_back();
              m.predict(bad, p, w);
          }),
          label + " one-rank malformed state rejected collectively");
    // Model-owned domain is retained even if callers change the global domain between evaluations.
    auto saved = FML::domain;
    FML::set_domain(FML::Domain::uniform_slab(FML::NTasks));
    auto fresh = m.workspace();
    auto again = m.predict(s, p, fresh);
    check(error(global_modes(m, again), out.prediction) < 1e-12,
          label + " both workspace meshes preserve model domain");
    FML::set_domain(saved);
    if (FML::ThisTask == 0)
        std::cout << std::setprecision(17) << "LPT_GRADIENT " << label << ' ' << field_error << ' ' << scalar_error
                  << ' ' << adjoint_error << ' ' << large_error << '\n';
    return out;
}
static void compare(const Snapshot & a, const Snapshot & b, const std::string & label) {
    double e = std::max(rel(a.potential, b.potential), rel(a.large_potential, b.large_potential));
    for (auto pair : {std::pair{&a.prediction, &b.prediction},
                      {&a.mock, &b.mock},
                      {&a.gradient, &b.gradient},
                      {&a.scalar, &b.scalar},
                      {&a.large_gradient, &b.large_gradient},
                      {&a.reconstruction, &b.reconstruction},
                      {&a.metric, &b.metric},
                      {&a.chain, &b.chain}})
        e = std::max(e, error(*pair.first, *pair.second));
    check(e < 2e-9, label + " slab/pencil predictions/posteriors/gradients/physical reconstruction/metric/HMC agree");
    if (FML::ThisTask == 0)
        std::cout << std::setprecision(17) << "LPT_GRID_PARITY " << label << ' ' << e << '\n';
}
// Orthogonal u=(1,1,0), v=(0,0,1): S2=A*B*cos(u.q)*cos(v.q).
// Analytic real displacements and velocity have no assignment approximation.
template <class Model>
void manufactured(const FML::Domain & domain, const std::string & label) {
    auto c = config();
    c.initial_cutoff = .10;
    c.bias_cutoff = .07;
    c.likelihood_cutoff = .18;
    auto power = FF::Spectrum::fixed([](double) { return 200.; });
    Model second(c, power, {}, {}, domain);
    auto w = second.workspace();
    auto p = second.initial_parameters();
    p[0] = 1;
    const double A = .5, B = .4, T = std::sqrt(200. * 512 / second.volume());
    Field white(second.cells());
    for (size_t q = 0; q < white.size(); ++q) {
        auto pos = second.layout().global_coordinate(q);
        double u = 2 * M_PI * (pos[0] + pos[1]) / 8, z = 2 * M_PI * pos[2] / 8;
        white[q] = (A * std::cos(u) + B * std::cos(z)) / T;
    }
    auto rec = second.reconstruct(white, p, w, 7.);
    double e = 0;
    for (size_t q = 0; q < white.size(); ++q) {
        auto pos = second.layout().global_coordinate(q);
        double u = 2 * M_PI * (pos[0] + pos[1]) / 8, z = 2 * M_PI * pos[2] / 8;
        e = std::max(e, std::abs(rec.initial_density[q] - A * std::cos(u) - B * std::cos(z)));
        for (int d = 0; d < 3; ++d) {
            double first = d < 2 ? -A / (4 * M_PI) * std::sin(u) : -B / (2 * M_PI) * std::sin(z);
            double psi2 =
                -3. / 7 * A * B / (6 * M_PI) * (d < 2 ? std::sin(u) * std::cos(z) : std::cos(u) * std::sin(z));
            e = std::max({e,
                          std::abs(rec.displacement[d][q] - first - psi2),
                          std::abs(rec.velocity[d][q] - 7 * (first + 2 * psi2))});
        }
    }
    check(second.maximum(e) < 2e-12, label + " analytic 2LPT density/displacement/velocity coefficient and sign");
    auto prediction = rec.tracer;
    double pe = 0, signal = 0;
    for (size_t i = 0; i < prediction.size(); ++i) {
        const auto & m = second.modes()[i];
        Complex exact = 0, first_only = 0;
        for (int x = 0; x < 8; ++x)
            for (int y = 0; y < 8; ++y)
                for (int z = 0; z < 8; ++z) {
                    double u = 2 * M_PI * (x + y) / 8, zz = 2 * M_PI * z / 8;
                    double phase1 = 0, phase2 = 0;
                    int pos[3] = {x, y, z};
                    for (int d = 0; d < 3; ++d) {
                        double first = d < 2 ? -A / (4 * M_PI) * std::sin(u) : -B / (2 * M_PI) * std::sin(zz);
                        double psi2 = -3. / 7 * A * B / (6 * M_PI) *
                                      (d < 2 ? std::sin(u) * std::cos(zz) : std::cos(u) * std::sin(zz));
                        phase1 += m.n[d] * (pos[d] / 8. + first * (d == 2 ? 1 + c.growth_rate : 1));
                        phase2 += m.n[d] * psi2 * (d == 2 ? 1 + 2 * c.growth_rate : 1);
                    }
                    first_only += std::polar(1., -2 * M_PI * phase1) / 512.;
                    exact += std::polar(1., -2 * M_PI * (phase1 + phase2)) / 512.;
                }
        pe = std::max(pe, std::abs(prediction[i] - exact));
        signal = std::max(signal, std::abs(exact - first_only));
    }
    pe = second.maximum(pe);
    signal = second.maximum(signal);
    check(pe < 3e-5 && signal > 1e-3,
          label + " independent particle DFT confirms nonlinear 2LPT/RSD (finite assignment tolerance)");
    for (size_t q = 0; q < white.size(); ++q) {
        auto pos = second.layout().global_coordinate(q);
        white[q] = A * std::cos(2 * M_PI * (pos[0] + pos[1]) / 8) / T;
    }
    auto two = second.predict(white, p, w);
    c.evolution = FF::Evolution::Zeldovich;
    Model first(c, power, {}, {}, domain);
    auto fw = first.workspace();
    auto one = first.predict(white, p, fw);
    e = 0;
    for (size_t i = 0; i < one.size(); ++i)
        e = std::max(e, std::abs(one[i] - two[i]));
    check(second.maximum(e) < 1e-12, label + " diagonal single-wave second-order source cancels");
    if (FML::ThisTask == 0)
        std::cout << std::setprecision(17) << "LPT_MANUFACTURED " << label << ' ' << pe << ' ' << signal << '\n';
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    use_pswf = argc > 1 && std::string(argv[1]) == "--pswf";
    auto c = config();
    Snapshot slab[2];
    bool have_slab = c.mesh % FML::NTasks == 0;
    if (have_slab) {
        auto dom = FML::Domain::uniform_slab(FML::NTasks);
        transfer_and_assignment<FML::GRID::FFTWGrid<3>>(c, dom, "slabs");
        for (int j = 0; j < 2; ++j) {
            c.evolution = j ? FF::Evolution::LPT2 : FF::Evolution::Zeldovich;
            slab[j] = exercise<FF::DistributedLPTModel>(c, dom, j ? "slab-2LPT" : "slab-1LPT");
        }
        manufactured<FF::DistributedLPTModel>(dom, "slabs");
    }
    Snapshot reference[2];
    bool have = false;
    for (int px = 1; px <= FML::NTasks; ++px)
        if (FML::NTasks % px == 0) {
            int py = FML::NTasks / px;
            if (px > 8 || py > 5)
                continue;
            auto dom = FML::Domain::pencil(edges(px, 8), edges(py, 8));
            std::string label = std::to_string(px) + "x" + std::to_string(py);
            transfer_and_assignment<Pencil>(c, dom, label);
            for (int j = 0; j < 2; ++j) {
                c.evolution = j ? FF::Evolution::LPT2 : FF::Evolution::Zeldovich;
                auto result = exercise<PencilModel>(c, dom, label + (j ? "-2LPT" : "-1LPT"));
                if (have_slab)
                    compare(slab[j], result, label + (j ? "-2LPT" : "-1LPT"));
                else if (have)
                    compare(reference[j], result, label + (j ? "-2LPT" : "-1LPT"));
                else
                    reference[j] = result;
            }
            have = true;
            manufactured<PencilModel>(dom, label);
        }
    // Fine mesh need not be an integer multiple; a shared physical domain must align with both meshes.
    if (FML::NTasks <= 4) {
        int px = FML::NTasks % 2 == 0 ? 2 : 1, py = FML::NTasks / px;
        auto dom = FML::Domain::pencil(edges(px, 8), edges(py, 8));
        if (px == 2)
            dom = FML::Domain::pencil({0, .25, 1}, edges(py, 8));
        auto alt = config(8, 12);
        transfer_and_assignment<Pencil>(alt, dom, "N8-Nd12");
        exercise<PencilModel>(alt, dom, "N8-Nd12-2LPT");
        for (int width : {4, 8}) {
            alt.kernel_width = width;
            exercise<PencilModel>(alt, dom, "width-" + std::to_string(width));
        }
        if (px == 2) {
            auto bad = FML::Domain::pencil({0, .375, 1}, edges(py, 8));
            check(FML::TESTS::throws([&] { PencilModel invalid(alt, spectrum(), {}, {}, bad); }),
                  "density domain incompatibility rejected before workspace FFT");
        }
    }
    // Nondivisible N=10 on 4/6/8 ranks, cell-aligned unequal pencils on both meshes.
    if (FML::NTasks % 2 == 0 && FML::NTasks / 2 <= 5) {
        int px = 2, py = FML::NTasks / px;
        auto dom = FML::Domain::pencil({0, .3, 1}, edges(py, 10));
        auto alt = config(10, 20);
        transfer_and_assignment<Pencil>(alt, dom, "uneven-N10");
        exercise<PencilModel>(alt, dom, "uneven-N10-2LPT");
    }
    if (FML::NTasks == 8) {
        auto alt = config(6, 12);
        alt.initial_cutoff = .08;
        alt.bias_cutoff = .07;
        alt.likelihood_cutoff = .13;
        const auto domain = FML::Domain::pencil(edges(4, 6), edges(2, 6));
        transfer_and_assignment<Pencil>(alt, domain, "more-ranks-than-N");
        exercise<PencilModel>(alt, domain, "more-ranks-than-N-2LPT");
    }
    return FML::TESTS::summary("field framework slab/pencil LPT");
}
