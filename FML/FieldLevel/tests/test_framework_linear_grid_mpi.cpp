#include <FML/PencilDomain/PencilGrid.h>
#include <FML/FieldLevel/DistributedLinear.h>
#include <FML/tests/TestHelpers.h>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FF::Modes;
using FML::FIELDLEVEL::Complex;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
using PencilModel = FF::DistributedLinearModelT<FML::GRID::PencilGrid<3>>;
static double inner(const Field & a, const Field & b) {
    double x = 0;
    for (size_t q = 0; q < a.size(); ++q)
        x += a[q] * b[q];
    return x;
}
static double error(const Field & a, const Field & b) {
    if (a.size() != b.size())
        return HUGE_VAL;
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::max({1., std::abs(a[i]), std::abs(b[i])}));
    return e;
}
template <class Model>
Field global_field(const Model & m, const Field & local) {
    Field result(m.layout().global_cells, 0);
    for (size_t q = 0; q < local.size(); ++q)
        result[m.layout().global_cell_id(q)] = local[q];
    FML::SumArrayOverTasks(result.data(), result.size());
    return result;
}
template <class Model>
Field global_modes(const Model & m, const Modes & modes) {
    const int n = m.configuration().mesh;
    Field result(size_t(n) * n * (n / 2 + 1) * 2, 0);
    for (size_t i = 0; i < modes.size(); ++i) {
        const auto & k = m.modes()[i].n;
        const size_t q = ((size_t((k[0] + n) % n) * n + (k[1] + n) % n) * (n / 2 + 1) + k[2]) * 2;
        result[q] = modes[i].real();
        result[q + 1] = modes[i].imag();
    }
    FML::SumArrayOverTasks(result.data(), result.size());
    return result;
}
struct Snapshot {
    Field white, momentum, prediction, mock, gradient, scalar, mean, metric, reconstruction, chain;
    double potential{}, marginal{};
};
static FF::Spectrum spectrum() {
    FF::Spectrum s;
    s.parameters = {{"amplitude", .1, 0, 1}, {"tilt", .2, 0, 1}};
    s.evaluate = [](double k, const auto & p) {
        return FF::Power{200 * std::exp(p[0]) * std::pow(k / .1, p[1]), {1., std::log(k / .1)}};
    };
    return s;
}
template <class Model>
Snapshot exercise(Model & m, const std::string & label) {
    auto w = m.workspace();
    auto p = m.initial_parameters();
    auto s = m.white_noise(91);
    auto direction = m.white_noise(94);
    const auto layout = m.layout();
    const int n = m.configuration().mesh;
    bool ids = true;
    for (size_t q = 0; q < s.size(); ++q) {
        const auto c = layout.global_coordinate(q);
        ids &= c[0] >= layout.local_starts[0] && c[1] >= layout.local_starts[1] && c[2] >= layout.local_starts[2];
        FML::RANDOM::COUNTER_BASED::Stream rng(91, layout.global_cell_id(q), 0x4649454cu);
        ids &= s[q] == rng.normal();
    }
    check(ids && m.sum(double(m.cells())) == double(n) * n * n, label + " coordinates/seeded fields cover mesh");
    check(FML::TESTS::throws([&] { layout.global_cell_id(m.cells()); }), label + " invalid compact cell rejected");
    Snapshot out;
    out.white = global_field(m, s);
    Field momentum(s.size());
    FF::Execution<Model>::hmc(m).draw_custom(momentum, 73, 0x100000001ull);
    bool correct = true;
    for (size_t q = 0; q < s.size(); ++q) {
        FML::RANDOM::COUNTER_BASED::Stream rng(73, layout.global_cell_id(q), 0x484d434du);
        FML::RANDOM::COUNTER_BASED::Stream step(rng.next64(), 0x100000001ull, 0x53544550u);
        correct &= momentum[q] == step.normal();
    }
    check(correct, label + " HMC momentum uses global IDs and all iteration bits");
    out.momentum = global_field(m, momentum);
    auto data = m.mock(m.white_noise(92), p, 93, w);
    out.mock = global_modes(m, data);
    FF::BasicPosterior<Model> posterior(m, data);
    Field gs, tmp;
    std::vector<double> gp, pg;
    out.potential = posterior.evaluate(s, p, w, gs, gp);
    out.gradient = global_field(m, gs);
    out.scalar = gp;
    const auto prediction = m.predict(s, p, w);
    out.prediction = global_modes(m, prediction);
    double dft_error = 0;
    for (size_t i = 0; i < prediction.size(); ++i) {
        const auto & mode = m.modes()[i];
        Complex sk = 0;
        for (int x = 0; x < n; ++x)
            for (int y = 0; y < n; ++y)
                for (int z = 0; z < n; ++z) {
                    FML::RANDOM::COUNTER_BASED::Stream rng(91, (size_t(x) * n + y) * n + z, 0x4649454cu);
                    sk +=
                        rng.normal() * std::polar(1., -2 * M_PI * (mode.n[0] * x + mode.n[1] * y + mode.n[2] * z) / n);
                }
        sk /= double(layout.global_cells);
        const double n2 = double(mode.n[0]) * mode.n[0] + double(mode.n[1]) * mode.n[1] + double(mode.n[2]) * mode.n[2];
        const double b = p[2] + m.configuration().growth_rate * mode.n[2] * mode.n[2] / n2;
        const auto pow = spectrum().evaluate(mode.k, {p[0], p[1]});
        const Complex ref = mode.k < m.configuration().initial_cutoff ?
                                b * std::sqrt(pow.value * layout.global_cells / m.volume()) * sk :
                                Complex(0);
        dft_error = std::max(dft_error, std::abs(prediction[i] - ref));
    }
    check(m.maximum(dft_error) < 1e-12, label + " prediction agrees with independent DFT");
    const double h = 1e-5;
    auto plus = s, minus = s;
    for (size_t q = 0; q < s.size(); ++q) {
        plus[q] += h * direction[q];
        minus[q] -= h * direction[q];
    }
    double fd = (posterior.evaluate(plus, p, w, tmp, pg) - posterior.evaluate(minus, p, w, tmp, pg)) / (2 * h);
    check(FML::TESTS::relative_difference(fd, m.sum(inner(gs, direction))) < 3e-7,
          label + " field gradient finite difference");
    double pe = 0;
    for (size_t j = 0; j < p.size(); ++j) {
        auto a = p, b = p;
        a[j] += h;
        b[j] -= h;
        fd = (posterior.evaluate(s, a, w, tmp, pg) - posterior.evaluate(s, b, w, tmp, pg)) / (2 * h);
        pe = std::max(pe, FML::TESTS::relative_difference(fd, gp[j]));
    }
    check(pe < 3e-7, label + " all scalar gradients finite difference");
    m.predict(s, p, w);
    m.pullback(p, data, w, tmp, pg);
    const double dot = m.sum(inner(tmp, direction));
    auto a = m.predict(plus, p, w), b = m.predict(minus, p, w);
    double mode_dot = 0;
    for (size_t i = 0; i < a.size(); ++i)
        mode_dot += m.modes()[i].weight * std::real(std::conj(data[i]) * (a[i] - b[i])) / (2 * h);
    check(FML::TESTS::relative_difference(dot, m.sum(mode_dot)) < 2e-10,
          label + " Fourier-weighted adjoint dot product");
    FF::FourierResponse response;
    for (const auto & mode : m.modes())
        response.multiplier.push_back(.8 + .03 * mode.n[0] * mode.n[0]);
    Model responding(m.configuration(), spectrum(), response, {}, m.domain());
    auto rw = responding.workspace();
    auto rp = responding.predict(s, p, rw);
    double re = 0;
    for (size_t i = 0; i < rp.size(); ++i)
        re = std::max(re, std::abs(rp[i] - response.multiplier[i] * prediction[i]));
    check(m.maximum(re) < 1e-13, label + " nonidentity response follows physical modes");
    FF::BasicPosterior<Model> response_post(responding, data);
    response_post.evaluate(s, p, rw, tmp, pg);
    const double ra = m.sum(inner(tmp, direction));
    fd = (response_post.evaluate(plus, p, rw, tmp, pg) - response_post.evaluate(minus, p, rw, tmp, pg)) / (2 * h);
    check(FML::TESTS::relative_difference(ra, fd) < 3e-7, label + " nonidentity response adjoint finite difference");
    auto conditional = m.linear_conditional(data, p, w);
    out.mean = global_field(m, conditional.mean);
    out.marginal = conditional.log_marginal;
    auto fit = FF::map(posterior, s, p, false, 12, 1e-9);
    check(fit.status == FF::MAPStatus::Converged && m.maximum(error(fit.white, conditional.mean)) < 1e-9,
          label + " MAP equals exact conditional mean");
    out.metric = global_field(m, m.apply_field_metric(s, p, w, -.5));
    out.reconstruction = global_field(m, m.reconstruct(s, p, w, 2.).initial_density);
    FML::MCMC::HMCOptions options;
    options.warmup = 0;
    options.samples = 4;
    options.n_leapfrog = 3;
    options.jitter = false;
    options.step_size = .01;
    options.adapt_step_size = false;
    options.adapt_mass = false;
    options.seed = 141;
    auto chain = FF::sample(posterior, s, p, options, false);
    out.chain = global_field(m, chain.last);
    check(chain.divergences == 0 && chain.recorded.size() == 4, label + " generic Execution drives finite HMC chain");
    check(FML::TESTS::throws([&] {
              auto bad = s;
              if (FML::ThisTask == 0)
                  bad.pop_back();
              posterior.evaluate(bad, p, w, tmp, pg);
          }),
          label + " one-rank bad state rejected collectively");
    auto bad = data;
    for (size_t i = 0; i < bad.size(); ++i)
        if (m.modes()[i].n[0] == 1 && m.modes()[i].n[1] == 0 && m.modes()[i].n[2] == 0)
            bad[i] += Complex(0, 1);
    check(FML::TESTS::throws([&] { m.validate_modes(bad); }), label + " cross-owner Hermitian error rejected");
    return out;
}
static void compare(const Snapshot & a, const Snapshot & b, const std::string & label) {
    check(a.white == b.white && a.momentum == b.momentum, label + " identical white noise and HMC momenta");
    double e = 0;
    for (auto pair : {std::pair{&a.prediction, &b.prediction},
                      {&a.mock, &b.mock},
                      {&a.gradient, &b.gradient},
                      {&a.scalar, &b.scalar},
                      {&a.mean, &b.mean},
                      {&a.metric, &b.metric},
                      {&a.reconstruction, &b.reconstruction},
                      {&a.chain, &b.chain}})
        e = std::max(e, error(*pair.first, *pair.second));
    e = std::max({e,
                  FML::TESTS::relative_difference(a.potential, b.potential),
                  FML::TESTS::relative_difference(a.marginal, b.marginal)});
    check(e < 2e-10, label + " predictions/posterior/gradients/conditional/metric/reconstruction/HMC agree");
    if (FML::ThisTask == 0)
        std::cout << std::setprecision(17) << "GRID_PARITY " << label << ' ' << e << '\n';
}
static std::vector<double> edges(int parts, int n) {
    std::vector<double> result(parts + 1);
    for (int i = 0; i <= parts; ++i)
        result[i] = double(i * n / parts) / n;
    return result;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .15;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .11;
    c.growth_rate = .6;
    c.evolution = FF::Evolution::Linear;
    Snapshot reference;
    bool have = false;
    if (c.mesh % FML::NTasks == 0) {
        FF::DistributedLinearModel m(c, spectrum(), {}, {}, FML::Domain::uniform_slab(FML::NTasks));
        reference = exercise(m, "slabs");
        have = true;
    }
    for (int px = 1; px <= FML::NTasks; ++px)
        if (FML::NTasks % px == 0) {
            const int py = FML::NTasks / px;
            if (px > c.mesh || py > c.mesh / 2 + 1)
                continue;
            auto dom = FML::Domain::pencil(edges(px, c.mesh), edges(py, c.mesh));
            PencilModel m(c, spectrum(), {}, {}, dom);
            const auto label = std::to_string(px) + "x" + std::to_string(py);
            auto result = exercise(m, label);
            if (have)
                compare(reference, result, label);
            else {
                reference = result;
                have = true;
            }
            // Workspaces retain the model domain even after the global default changes.
            auto saved = FML::domain;
            FML::set_domain(FML::Domain::uniform_slab(FML::NTasks));
            auto w = m.workspace();
            check(error(global_modes(m, m.predict(m.white_noise(91), m.initial_parameters(), w)), result.prediction) <
                      1e-13,
                  label + " workspace preserves explicit domain");
            FML::set_domain(saved);
        }
    // Mesh not divisible by task count (for 4/6 ranks); unequal widths in both axes when possible.
    c.mesh = 10;
    c.density_mesh = 20;
    int px = FML::NTasks % 2 == 0 ? 2 : 1, py = FML::NTasks / px;
    if (py <= c.mesh) {
        auto xe = edges(px, c.mesh), ye = edges(py, c.mesh);
        if (px == 2)
            xe[1] = .3;
        if (py == 2)
            ye[1] = .4;
        PencilModel m(c, spectrum(), {}, {}, FML::Domain::pencil(xe, ye));
        exercise(m, "uneven-N10");
        if (px > 1 || py > 1)
            check(FML::TESTS::throws([&] {
                      auto x = xe, y = ye;
                      (px > 1 ? x : y)[1] = .123;
                      PencilModel bad(c, spectrum(), {}, {}, FML::Domain::pencil(x, y));
                  }),
                  "off-cell pencil edges rejected before FFT");
    }
    if (FML::NTasks > 4 && FML::NTasks % 2 == 0 && FML::NTasks / 2 <= 4) {
        auto small = c;
        small.mesh = 4;
        small.density_mesh = 8;
        small.initial_cutoff = .1;
        small.likelihood_cutoff = .09;
        PencilModel m(small, spectrum(), {}, {}, FML::Domain::pencil(edges(FML::NTasks / 2, 4), edges(2, 4)));
        exercise(m, "more-ranks-than-N");
    }
    if (FML::NTasks > 1) {
        check(FML::TESTS::throws([&] {
                  auto dom = FML::ThisTask == 0 ? FML::Domain::uniform_pencil(1, FML::NTasks) :
                                                  FML::Domain::uniform_slab(FML::NTasks);
                  PencilModel bad(c, spectrum(), {}, {}, dom);
              }),
              "rank-inconsistent domain types rejected collectively");
        check(FML::TESTS::throws([&] {
                  auto xe = edges(1, c.mesh), ye = edges(FML::NTasks, c.mesh);
                  if (FML::ThisTask == 0 && ye.size() > 2)
                      ye[1] += .001;
                  PencilModel bad(c, spectrum(), {}, {}, FML::Domain::pencil(xe, ye));
              }),
              "rank-inconsistent domain edges rejected collectively");
    }
    if (FML::NTasks >= 6) {
        auto small = c;
        small.mesh = 8;
        check(FML::TESTS::throws([&] {
                  PencilModel bad(small, spectrum(), {}, {}, FML::Domain::pencil(edges(1, 8), edges(FML::NTasks, 8)));
              }),
              "empty Fourier pencil partition rejected before FFT construction");
    }
    return FML::TESTS::summary("field framework slab/pencil linear");
}
