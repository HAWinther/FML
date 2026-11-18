#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <FML/FieldLevel/PeriodicCatalog.h>
#include <FML/tests/TestHelpers.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <iomanip>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace E = FML::FIELDLEVEL::EULERIAN;
using FML::FIELDLEVEL::Complex;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
static double dot(const Field & a, const Field & b) {
    double r = 0;
    for (size_t i = 0; i < a.size(); ++i)
        r += a[i] * b[i];
    return r;
}
static double signal(const std::array<int, 3> & q, int n) {
    const double x = 2 * M_PI * q[0] / n, y = 2 * M_PI * q[1] / n, z = 2 * M_PI * q[2] / n;
    return .4 * std::cos(2 * x + y) + .3 * std::sin(y - z) + .2 * std::cos(2 * z);
}
static auto priors() {
    return std::array<FF::Parameter, 4>{{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                         {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                         {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                         {"b_laplacian", 7., 0, 1, -100, 100, FF::Prior::Uniform}}};
}
// Test-only global tables indexed by integer wavevector, independently of Fourier storage layout.
template <class Model>
Field mode_table(const Model & m, const FF::Modes & a) {
    Field all(9 * 9 * 9 * 2, 0);
    for (size_t j = 0; j < a.size(); ++j) {
        const auto & n = m.modes()[j].n;
        const size_t i = ((n[0] + 4) * 9 + n[1] + 4) * 9 + n[2] + 4;
        all[2 * i] = a[j].real();
        all[2 * i + 1] = a[j].imag();
    }
    FML::SumArrayOverTasks(all.data(), all.size());
    return all;
}
template <class Model>
FF::Modes from_table(const Model & m, const Field & all) {
    FF::Modes out(m.modes().size());
    for (size_t j = 0; j < out.size(); ++j) {
        const auto & n = m.modes()[j].n;
        const size_t i = ((n[0] + 4) * 9 + n[1] + 4) * 9 + n[2] + 4;
        out[j] = {all[2 * i], all[2 * i + 1]};
    }
    return out;
}
static double error(const Field & a, const Field & b) {
    double e = 0;
    for (size_t j = 0; j < a.size(); ++j)
        e = std::max(e, std::abs(a[j] - b[j]));
    return e;
}
template <class Model>
Field field_table(const Model & m, const Field & a) {
    Field all(m.layout().global_cells, 0);
    for (size_t q = 0; q < a.size(); ++q)
        all[m.layout().global_cell_id(q)] = a[q];
    FML::SumArrayOverTasks(all.data(), all.size());
    return all;
}
template <class Grid>
Field exercise(const FML::Domain & domain, const std::string & label, bool pswf) {
    using Model = FF::DistributedEulerianLPTModelT<Grid>;
    FF::EulerianLPTConfiguration c;
    c.mesh = 8;
    c.product_mesh = 12;
    c.density_mesh = 24;
    c.boxsize = 100;
    const double kf = 2 * M_PI / c.boxsize;
    c.initial_cutoff = 2.7 * kf; // Deliberately above the old half-Nyquist of the compact mesh.
    c.matter_band = {2.8 * kf, E::BandGeometry::Cube};
    c.likelihood_band = {2.5 * kf, E::BandGeometry::Cube};
    c.matter_assignment = pswf ? FF::MatterAssignment::PSWF : FF::MatterAssignment::Smooth;
    c.nufft_epsilon = 1e-10;
    auto spectrum = FF::sigma8_spectrum([](double k) { return 500 * std::exp(-5 * k); });
    Model m(c, spectrum, priors(), domain);
    auto full = c;
    full.mesh = c.product_mesh;
    full.product_mesh = 0;
    Model reference(full, spectrum, priors(), domain);
    auto w = m.workspace(), rw = reference.workspace();
    auto p = m.initial_parameters();
    p[0] = 1.08;
    Field s(m.cells()), padded(reference.cells());
    for (size_t q = 0; q < s.size(); ++q)
        s[q] = signal(m.layout().global_coordinate(q), c.mesh);
    const double gain = std::pow(double(c.mesh) / c.product_mesh, 1.5);
    for (size_t q = 0; q < padded.size(); ++q)
        padded[q] = gain * signal(reference.layout().global_coordinate(q), c.product_mesh);
    check(m.layout().global_cells == 512 && reference.layout().global_cells == 1728 && m.product_mesh() == 12,
          label + " compact state is independent of product/particle grids");
    const auto prediction = m.predict(s, p, w);
    const auto refprediction = reference.predict(padded, p, rw);
    const double prediction_error = error(mode_table(m, prediction), mode_table(reference, refprediction));
    check(prediction_error < 2e-12, label + " physical amplitudes match independently sampled padded reference");
    const auto columns = m.bias_modes(w), rcolumns = reference.bias_modes(rw);
    double column_error = 0;
    for (int a = 0; a < 4; ++a)
        column_error = std::max(column_error, error(mode_table(m, columns[a]), mode_table(reference, rcolumns[a])));
    check(column_error < 2e-12, label + " all Eulerian bias columns survive mode restriction");
    const double power = 1000;
    const auto data = m.mock(s, p, power, 801, w);
    const auto noise = field_table(m, m.white_noise(801));
    double noise_error = 0;
    for (size_t j = 0; j < data.size(); ++j) {
        Complex sum = 0;
        for (size_t q = 0; q < noise.size(); ++q) {
            const std::array<int, 3> x = {int(q / 64), int(q / 8 % 8), int(q % 8)};
            double phase = 0;
            for (int d = 0; d < 3; ++d)
                phase -= 2 * M_PI * m.modes()[j].n[d] * x[d] / 8.;
            sum += noise[q] * std::exp(Complex(0, phase)) / 512.;
        }
        const auto expected = std::sqrt(power * 512. / m.volume()) * sum;
        noise_error = std::max(noise_error, std::abs(data[j] - prediction[j] - expected));
    }
    check(m.maximum(noise_error) < 2e-12, label + " mock noise normalization uses compact cell count");
    const auto reference_data = from_table(reference, mode_table(m, data));
    auto post = FF::fixed_noise_posterior(m, data, power, FF::InitialFieldPrior::ConditionOnField);
    auto rpost = FF::fixed_noise_posterior(reference, reference_data, power, FF::InitialFieldPrior::ConditionOnField);
    Field gs, temp, rgs;
    std::vector<double> gp, tp, rgp;
    const double U = post.evaluate(s, p, w, gs, gp), rU = rpost.evaluate(padded, p, rw, rgs, rgp);
    check(FML::TESTS::near(U, rU, 2e-12) && error(gp, rgp) < 2e-11,
          label + " conditional target and scalar gradients match");
    const auto direction = m.white_noise(802);
    auto a = s, b = s;
    const double h = 1e-4;
    for (size_t q = 0; q < s.size(); ++q) {
        a[q] += h * direction[q];
        b[q] -= h * direction[q];
    }
    const double numeric = (post.value(a, p, w) - post.value(b, p, w)) / (2 * h);
    const double field_error = FML::TESTS::relative_difference(numeric, m.sum(dot(gs, direction)));
    check(field_error < 3e-6, label + " compact field gradient has correct Euclidean transfer transpose");
    double scalar_error = 0;
    for (size_t j = 0; j < p.size(); ++j) {
        auto pa = p, pb = p;
        const double step = h * std::max(1., std::abs(p[j]));
        pa[j] += step;
        pb[j] -= step;
        const double fd = (post.value(s, pa, w) - post.value(s, pb, w)) / (2 * step);
        scalar_error = std::max(scalar_error, FML::TESTS::relative_difference(fd, gp[j]));
    }
    check(scalar_error < 3e-6, label + " all scalar derivatives with separate meshes");
    auto free = FF::fixed_noise_posterior(m, data, power);
    Field fg;
    std::vector<double> fp;
    const double Uf = free.evaluate(s, p, w, fg, fp);
    double prior_error = 0;
    for (size_t q = 0; q < s.size(); ++q)
        prior_error = std::max(prior_error, std::abs(fg[q] - gs[q] - s[q]));
    check(FML::TESTS::near(Uf - U, .5 * m.sum(dot(s, s)), 2e-12) && m.maximum(prior_error) < 2e-12,
          label + " prior remains unit Gaussian on compact cells");
    auto metric_half = free.apply_field_metric(direction, p, w, .5);
    auto metric_twice = free.apply_field_metric(metric_half, p, w, .5);
    auto metric_full = free.apply_field_metric(direction, p, w, 1.);
    check(m.maximum(error(metric_twice, metric_full)) < 2e-11, label + " compact field metric powers compose");
    const auto metric_s = free.apply_field_metric(s, p, w, 1.);
    check(FML::TESTS::near(m.sum(dot(s, metric_full)), m.sum(dot(metric_s, direction)), 2e-12) &&
              m.sum(dot(direction, metric_full)) > 0,
          label + " compact metric is symmetric positive");
    Field null(s.size());
    for (size_t q = 0; q < null.size(); ++q)
        null[q] = (m.layout().global_coordinate(q)[0] % 2 ? 1. : -1.);
    const auto null_metric = free.apply_field_metric(null, p, w, -1.);
    check(m.maximum(error(null, null_metric)) < 2e-12, label + " removed Nyquist directions keep identity metric");
    m.validate_modes(data);
    check(true, label + " compact observation modes preserve Hermitian layout");
    auto prototype = FML::GRID::make_grid<Grid>(c.mesh, 0, 0, domain);
    FF::PeriodicCatalogSettings options;
    options.boxsize = c.boxsize;
    options.epsilon = 1e-8;
    FF::PeriodicCatalogT<Grid> catalog(prototype, options);
    std::vector<FF::CatalogEntry> entries;
    if (FML::ThisTask == 0)
        entries.push_back({{1., 2., 3.}, 1.});
    catalog.add_batch(entries);
    catalog.finish();
    check(catalog.modes(m.modes()).size() == m.modes().size(),
          label + " catalog adapter uses compact observation layout");
    FML::MCMC::HMCOptions o;
    o.warmup = 0;
    o.samples = 2;
    o.n_leapfrog = 2;
    o.step_size = .001;
    o.jitter = false;
    o.adapt_step_size = false;
    o.adapt_mass = false;
    o.seed = 803;
    const auto chain = FF::sample(free, s, p, o, true);
    check(chain.divergences == 0 && chain.last.size() == m.cells() + p.size(),
          label + " compact HMC field/scalar block sizes");
    // Exercise both workspace move and a second prediction after scratch-heavy operations.
    auto moved = std::move(w);
    const auto again = m.predict(s, p, moved);
    check(error(mode_table(m, again), mode_table(m, prediction)) < 2e-12,
          label + " moved workspace reuses transfer plans");
    check(FML::TESTS::throws([&] { m.predict(s, p, w); }), label + " moved-from workspace rejected");
    auto malformed = s;
    if (FML::ThisTask == FML::NTasks - 1)
        malformed.pop_back();
    check(FML::TESTS::throws([&] { m.predict(malformed, p, moved); }),
          label + " rank-local compact field size rejected");
    auto bad = c;
    bad.product_mesh = 6;
    check(FML::TESTS::throws([&] { Model invalid(bad, spectrum, priors(), domain); }),
          label + " undersized product grid rejected");
    bad = c;
    bad.initial_cutoff = M_PI * c.mesh / c.boxsize;
    check(FML::TESTS::throws([&] { Model invalid(bad, spectrum, priors(), domain); }),
          label + " unresolved compact initial band rejected");
    if (FML::NTasks > 1) {
        bad = c;
        if (FML::ThisTask == 0)
            bad.mesh = 12;
        check(FML::TESTS::throws([&] { Model invalid(bad, spectrum, priors(), domain); }),
              label + " mismatched compact configuration rejected");
    }
    if (FML::ThisTask == 0)
        std::cout << std::setprecision(17) << "COMPACT_CHECK " << label << ' ' << prediction_error << ' '
                  << column_error << ' ' << field_error << ' ' << scalar_error << '\n';
    auto snapshot = field_table(m, Field(chain.last.begin(), chain.last.begin() + m.cells()));
    snapshot.insert(snapshot.end(), chain.last.begin() + m.cells(), chain.last.end());
    return snapshot;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    for (bool pswf : {false, true}) {
        const std::string name = pswf ? "PSWF" : "smooth";
        const auto slab =
            exercise<FML::GRID::FFTWGrid<3>>(FML::Domain::uniform_slab(FML::NTasks), "slab-" + name, pswf);
#ifdef USE_MPI
        for (int px = 1; px <= FML::NTasks; ++px)
            if (FML::NTasks % px == 0) {
                const int py = FML::NTasks / px;
                std::vector<double> ex(px + 1), ey(py + 1);
                for (int j = 0; j <= px; ++j)
                    ex[j] = double(j) / px;
                for (int j = 0; j <= py; ++j)
                    ey[j] = double(j) / py;
                auto pencil =
                    exercise<FML::GRID::PencilGrid<3>>(FML::Domain::pencil(ex, ey),
                                                       std::to_string(px) + "x" + std::to_string(py) + "-" + name,
                                                       pswf);
                check(error(slab, pencil) < 2e-10, "slab/pencil compact HMC trajectory parity");
            }
        if (FML::NTasks == 2 || FML::NTasks == 4) {
            const int py = FML::NTasks / 2;
            std::vector<double> ey(py + 1);
            for (int j = 0; j <= py; ++j)
                ey[j] = double(j) / py;
            auto uneven =
                exercise<FML::GRID::PencilGrid<3>>(FML::Domain::pencil({0, .25, 1}, ey), "unequal-" + name, pswf);
            check(error(slab, uneven) < 2e-10, "unequal pencil compact HMC trajectory parity");
        }
#endif
    }
    return FML::TESTS::summary("compact Eulerian LPT");
}
