#include <FML/PencilDomain/PencilGrid.h>
#include "SurveyTestHelpers.h"
using FML::TESTS::check;
using Pencil = FML::GRID::PencilGrid<3>;
using PL = FF::DistributedLinearModelT<Pencil>;
using PN = FF::DistributedLPTModelT<Pencil>;
static std::vector<double> edges(int parts, int n) {
    std::vector<double> e(parts + 1);
    for (int j = 0; j <= parts; ++j)
        e[j] = double(j * n / parts) / n;
    return e;
}
static double dot(const Field & a, const Field & b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
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
Field global_field(const Model & m, const Field & local) {
    Field x(m.layout().global_cells, 0);
    for (size_t q = 0; q < local.size(); ++q)
        x[m.layout().global_cell_id(q)] = local[q];
    FML::SumArrayOverTasks(x.data(), x.size());
    return x;
}
struct Snapshot {
    Field mean, gradient, parameters, chain;
    double U{};
};
template <bool Poisson, class Model>
Snapshot evaluate(Model & m, const std::string & label) {
    auto w = m.workspace();
    auto p = m.initial_parameters();
    auto s = m.white_noise(11);
    if (m.configuration().evolution != FF::Evolution::Linear) {
        p[2] = 1.4;
        p[3] = .2;
        p[4] = -.15;
        p[5] = 4;
    }
    auto window = FF::make_survey_window(m, {-50, -50, -50}, [](auto r) {
        return FF::SurveyVoxel{r[0] + r[1] < 25 && r[2] > -50, .6 + .001 * r[1] + .002 * r[2]};
    });
    Field data(m.cells()), variance(m.cells()), exposure(m.cells());
    std::vector<std::uint64_t> counts(m.cells());
    bool coordinates = true;
    const double dx = m.configuration().boxsize / m.configuration().mesh;
    for (size_t q = 0; q < m.cells(); ++q) {
        auto pos = m.layout().global_coordinate(q);
        size_t id = m.layout().global_cell_id(q);
        double x = -50 + dx * pos[0], y = -50 + dx * pos[1], z = -50 + dx * pos[2];
        coordinates &= window.observed[q] == (x + y < 25 && z > -50) && window.selection[q] == .6 + .001 * y + .002 * z;
        data[q] = window.observed[q] ? .02 * std::cos(.1 * id) : NAN;
        variance[q] = window.observed[q] ? 1 + .01 * (id % 7) : NAN;
        exposure[q] = window.observed[q] ? 2 + .1 * (id % 5) : NAN;
        counts[q] = id % 4;
    }
    check(coordinates, label + " window callbacks receive physical global x/y/z");
    auto saved = FML::domain;
    FML::set_domain(FML::Domain::uniform_slab(FML::NTasks));
    auto likelihood = [&] {
        if constexpr (Poisson)
            return FF::SurveyPoissonLikelihood(m, window, counts, exposure, .1);
        else
            return FF::SurveyGaussianLikelihood(m, window, data, variance);
    }();
    FML::set_domain(saved);
    auto prediction = m.predict(s, p, w);
    Snapshot result;
    if constexpr (Poisson)
        result.mean = global_field(m, likelihood.expected_counts(prediction));
    else
        result.mean = global_field(m, likelihood.expected_field(prediction));
    FF::BasicPosterior<Model, decltype(likelihood)> post(m, std::move(likelihood));
    Field gs, tmp;
    std::vector<double> gp, g;
    result.U = post.evaluate(s, p, w, gs, gp);
    result.gradient = global_field(m, gs);
    result.parameters = gp;
    auto direction = m.white_noise(13), a = s, b = s;
    const double h = 1e-5;
    for (size_t q = 0; q < s.size(); ++q) {
        a[q] += h * direction[q];
        b[q] -= h * direction[q];
    }
    double fd = (post.evaluate(a, p, w, tmp, g) - post.evaluate(b, p, w, tmp, g)) / (2 * h);
    double field_error = rel(fd, m.sum(dot(gs, direction)));
    check(field_error < 2e-6, label + " full posterior field finite difference");
    double scalar_error = 0;
    for (size_t j = 0; j < p.size(); ++j) {
        auto x = p, y = p;
        double step = h * std::max(1., std::abs(p[j]));
        x[j] += step;
        y[j] -= step;
        fd = (post.evaluate(s, x, w, tmp, g) - post.evaluate(s, y, w, tmp, g)) / (2 * step);
        scalar_error = std::max(scalar_error, rel(fd, gp[j]));
    }
    check(scalar_error < 2e-6, label + " all spectrum/bias/noise scalar finite differences");
    auto fit = FF::map(post, s, p, false, 3, 1e-7);
    check(std::isfinite(fit.potential) && fit.potential < result.U, label + " MAP decreases masked posterior");
    FML::MCMC::HMCOptions o;
    o.warmup = 0;
    o.samples = 3;
    o.n_leapfrog = 2;
    o.jitter = false;
    o.step_size = .001;
    o.adapt_step_size = false;
    o.adapt_mass = false;
    o.seed = 71;
    auto chain = FF::sample(post, s, p, o, true);
    result.chain = global_field(m, Field(chain.last.begin(), chain.last.begin() + m.cells()));
    result.chain.insert(result.chain.end(), chain.recorded.back().begin(), chain.recorded.back().end());
    check(chain.divergences == 0, label + " joint HMC smoke");
    if (FML::ThisTask == 0)
        std::cout << std::setprecision(17) << "SURVEY_GRID_GRADIENT " << label << ' ' << field_error << ' '
                  << scalar_error << '\n';
    return result;
}
template <class Model>
std::vector<Snapshot> run(FF::Configuration c, const FML::Domain & domain, const std::string & label) {
    FF::Spectrum spectrum;
    spectrum.parameters = {{"log_amplitude", .1, 0, 1}, {"tilt", .2, 0, 1}};
    spectrum.evaluate = [](double k, const auto & p) {
        return FF::Power{200 * std::exp(p[0]) * std::pow(k / .1, p[1]), {1., std::log(k / .1)}};
    };
    Model m(c, spectrum, {}, {}, domain);
    return {evaluate<false>(m, label + "-Gaussian"), evaluate<true>(m, label + "-Poisson")};
}
static void compare(const std::vector<Snapshot> & a, const std::vector<Snapshot> & b, const std::string & label) {
    double e = 0;
    for (size_t j = 0; j < a.size(); ++j) {
        e = std::max({e,
                      rel(a[j].U, b[j].U),
                      error(a[j].mean, b[j].mean),
                      error(a[j].gradient, b[j].gradient),
                      error(a[j].parameters, b[j].parameters),
                      error(a[j].chain, b[j].chain)});
    }
    check(e < 2e-10, label + " slab/pencil observed means/posteriors/gradients/HMC agree");
    if (FML::ThisTask == 0)
        std::cout << std::setprecision(17) << "SURVEY_GRID_PARITY " << label << ' ' << e << '\n';
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.growth_rate = .6;
    std::vector<Snapshot> reference[3];
    bool have = c.mesh % FML::NTasks == 0;
    if (have)
        for (int j = 0; j < 3; ++j) {
            c.evolution = FF::Evolution(j);
            auto dom = FML::Domain::uniform_slab(FML::NTasks);
            reference[j] = j == 0 ? run<FF::DistributedLinearModel>(c, dom, "slab-linear") :
                                    run<FF::DistributedLPTModel>(c, dom, j == 1 ? "slab-ZA" : "slab-2LPT");
        }
    for (int px = 1; px <= FML::NTasks; ++px)
        if (FML::NTasks % px == 0) {
            int py = FML::NTasks / px;
            if (px > 8 || py > 5)
                continue;
            const auto dom = FML::Domain::pencil(edges(px, 8), edges(py, 8));
            std::string label = std::to_string(px) + "x" + std::to_string(py);
            reference_tests<PL>(dom);
            for (int j = 0; j < 3; ++j) {
                c.evolution = FF::Evolution(j);
                auto result =
                    j == 0 ? run<PL>(c, dom, label + "-linear") : run<PN>(c, dom, label + (j == 1 ? "-ZA" : "-2LPT"));
                if (have)
                    compare(reference[j], result, label + "-" + std::to_string(j));
                else
                    reference[j] = result;
            }
            have = true;
        }
    if (FML::NTasks % 2 == 0 && FML::NTasks / 2 <= 5) {
        c.mesh = 10;
        c.density_mesh = 20;
        c.evolution = FF::Evolution::LPT2;
        auto dom = FML::Domain::pencil({0, .3, 1}, edges(FML::NTasks / 2, 10));
        run<PN>(c, dom, "uneven-N10-2LPT");
    }
    return FML::TESTS::summary("slab/pencil survey observation and likelihood");
}
