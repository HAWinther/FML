#include <FML/FieldLevel/PeriodicCatalog.h>
#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <FML/tests/TestHelpers.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <limits>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::TESTS::check;
using Complex = FML::FIELDLEVEL::Complex;
static std::vector<FF::CatalogEntry> catalog() {
    std::vector<FF::CatalogEntry> all;
    for (int i = 0; i < 59; ++i) {
        FF::CatalogEntry e;
        e.position = {100 * std::fmod(.013 + i * .38196601125, 1.),
                      100 * std::fmod(.71 + i * .17320508075, 1.),
                      100 * std::fmod(.29 + i * .27182818284, 1.)};
        e.weight = i % 7 == 0 ? 0 : .25 + i * i;
        if (i == 0)
            e.position = {0, 100, -100};
        if (i == 1)
            e.position = {50, 50, 50};
        if (i == 2)
            e.position = {100, 0, 100};
        if (i % 3 == 0)
            e.position[0] -= 200;
        if (i % 5 == 0)
            e.position[1] += 300;
        all.push_back(e);
    }
    return all;
}
static Complex direct(const std::vector<FF::CatalogEntry> & all, const std::array<int, 3> & n) {
    long double sum = 0;
    std::complex<long double> result = 0;
    for (const auto & e : all) {
        long double phase = 0;
        sum += e.weight;
        for (int d = 0; d < 3; ++d)
            phase -= 2 * (long double)M_PI * n[d] * e.position[d] / 100;
        result += (long double)e.weight * std::complex<long double>(std::cos(phase), std::sin(phase));
    }
    return Complex(result / sum);
}
template <class Grid>
void exercise(const FML::Domain & domain, const std::string & label) {
    auto prototype = FML::GRID::make_grid<Grid>(16, 0, 0, domain);
    FF::PeriodicCatalogSettings c;
    c.boxsize = 100;
    c.epsilon = 1e-8;
    c.message_bytes = 32; // One record per MPI chunk.
    FF::PeriodicCatalogT<Grid> stream(prototype, c);
    auto all = catalog();
    check(FML::TESTS::throws([&] { stream.fourier_grid(); }), label + " unfinished catalog cannot be read");
    for (size_t start = 0; start < all.size(); start += 13) {
        std::vector<FF::CatalogEntry> batch;
        for (size_t i = start; i < std::min(all.size(), start + 13); ++i)
            if (int((i * 3 + 1) % FML::NTasks) == FML::ThisTask)
                batch.push_back(all[i]);
        stream.add_batch(batch);
    }
    const auto stats = stream.finish();
    const auto & grid = stream.fourier_grid();
    double w = 0, w2 = 0;
    for (auto e : all) {
        w += e.weight;
        w2 += e.weight * e.weight;
    }
    check(stats.rows == all.size() && FML::TESTS::near(stats.mode_variance, w2 / w / w, 2e-14) &&
              FML::TESTS::near(stats.poisson_power, 1e6 * w2 / w / w, 2e-14),
          label + " exact global count and weighted Poisson normalization");
    double error = 0, excluded = 0;
    for (auto i : grid.get_fourier_range()) {
        auto n = grid.get_fourier_integer_wavevector_from_index(i);
        bool zero = n[0] == 0 && n[1] == 0 && n[2] == 0;
        for (int v : n)
            zero |= std::abs(v) == 8;
        Complex actual = grid.get_fourier_from_index(i);
        if (zero)
            excluded = std::max(excluded, std::abs(actual));
        else
            error = std::max(error, std::abs(actual - direct(all, n)));
    }
    FML::MaxOverTasks(&error);
    FML::MaxOverTasks(&excluded);
    check(error < 3e-8 && excluded == 0,
          label + " all non-Nyquist modes match independent direct sums; excluded modes exact zero");
    check(stream.finish().poisson_power == stats.poisson_power, label + " finish is idempotent");
    check(FML::TESTS::throws([&] { stream.add_batch({}); }), label + " adding after finish requires reset");
    // Exact integration with the inference model's local cubic mode list and fixed-noise target.
    FF::EulerianLPTConfiguration lc;
    lc.mesh = 16;
    lc.density_mesh = 32;
    lc.boxsize = 100;
    lc.initial_cutoff = .12;
    lc.matter_band = {.13, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
    lc.likelihood_band = {.11, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
    std::array<FF::Parameter, 4> priors = {
        {{"b_delta", 1, 0, 1}, {"b_delta2", 0, 0, 1}, {"b_K2", 0, 0, 1}, {"b_laplacian", 0, 0, 1}}};
    FF::DistributedEulerianLPTModelT<Grid> model(lc, FF::sigma8_spectrum([](double) { return 100.; }), priors, domain);
    auto data = stream.modes(model.modes());
    auto target = FF::fixed_noise_posterior(model, data, stats.poisson_power, FF::InitialFieldPrior::ConditionOnField);
    auto work = model.workspace();
    const auto p = model.initial_parameters();
    auto s = model.white_noise(192);
    check(std::isfinite(target.value(s, p, work)),
          label + " catalog modes/noise connect directly to Eulerian inference target");
    auto bad_modes = model.modes();
    if (!bad_modes.empty())
        bad_modes[0].n[0] += 1;
    check(FML::TESTS::throws([&] { stream.modes(bad_modes); }),
          label + " foreign/misindexed mode metadata rejected collectively");
    for (double scale : {1e150, 1e-150}) {
        stream.reset();
        for (size_t start = 0; start < all.size(); start += 17) {
            std::vector<FF::CatalogEntry> batch;
            if (FML::ThisTask == 0)
                for (size_t i = start; i < std::min(all.size(), start + 17); ++i) {
                    auto e = all[i];
                    e.weight *= scale;
                    batch.push_back(e);
                }
            stream.add_batch(batch);
        }
        auto scaled = stream.finish();
        auto modes = stream.modes(model.modes());
        double e = 0;
        for (size_t i = 0; i < modes.size(); ++i)
            e = std::max(e, std::abs(modes[i] - data[i]));
        FML::MaxOverTasks(&e);
        check(e < 2e-12 && FML::TESTS::near(scaled.mode_variance, stats.mode_variance, 2e-14),
              label + " root-only streaming and global weight rescaling invariance");
    }
    auto shifted_config = c;
    shifted_config.origin = {17, -41, 2};
    FF::PeriodicCatalogT<Grid> shifted(prototype, shifted_config);
    std::vector<FF::CatalogEntry> shifted_batch;
    if (FML::ThisTask == 0)
        for (auto e : all) {
            for (int d = 0; d < 3; ++d)
                e.position[d] += shifted_config.origin[d];
            shifted_batch.push_back(e);
        }
    shifted.add_batch(shifted_batch);
    shifted.finish();
    auto shifted_modes = shifted.modes(model.modes());
    double shift_error = 0;
    for (size_t i = 0; i < data.size(); ++i)
        shift_error = std::max(shift_error, std::abs(data[i] - shifted_modes[i]));
    FML::MaxOverTasks(&shift_error);
    check(shift_error < 2e-12, label + " explicit box origin preserves relative-coordinate phases");
    stream.reset();
    std::vector<FF::CatalogEntry> local;
    for (size_t i = 0; i < all.size(); ++i)
        if (int(i % FML::NTasks) == FML::ThisTask) {
            auto e = all[i];
            e.weight = 1;
            local.push_back(e);
        }
    stream.add_batch(local);
    auto unit = stream.finish();
    check(unit.rows == all.size() && FML::TESTS::near(unit.poisson_power, 1e6 / all.size(), 2e-14),
          label + " unweighted variance 1/N and power V/N");
    stream.reset();
    check(FML::TESTS::throws([&] { stream.finish(); }), label + " empty global catalog rejected");
    check(FML::TESTS::throws([&] { stream.add_batch({}); }), label + " failed stream requires reset");
    stream.reset();
    local.clear();
    if (FML::ThisTask == FML::NTasks - 1)
        local.push_back({{0, 0, 0}, -1});
    check(FML::TESTS::throws([&] { stream.add_batch(local); }),
          label + " rank-local negative weight rejected collectively");
    stream.reset();
    local.clear();
    if (FML::ThisTask == FML::NTasks - 1)
        local.push_back({{std::numeric_limits<double>::quiet_NaN(), 0, 0}, 1});
    check(FML::TESTS::throws([&] { stream.add_batch(local); }),
          label + " rank-local nonfinite coordinate rejected collectively");
    stream.reset();
    local.clear();
    if (FML::ThisTask == 0)
        local.push_back({{0, 0, 0}, 0});
    stream.add_batch(local);
    check(FML::TESTS::throws([&] { stream.finish(); }), label + " zero total catalog weight rejected");
    auto tiny = c;
    tiny.max_received_rows = 1;
    FF::PeriodicCatalogT<Grid> capped(prototype, tiny);
    local.clear();
    if (FML::ThisTask == 0) {
        local.push_back({{1, 1, 1}, 1});
        local.push_back({{1, 1, 1}, 1});
    }
    check(FML::TESTS::throws([&] { capped.add_batch(local); }),
          label + " bounded receive buffer rejects skew before payload exchange");
    auto invalid = c;
    invalid.epsilon = -1;
    check(FML::TESTS::throws([&] { FF::PeriodicCatalogT<Grid> bad(prototype, invalid); }),
          label + " invalid NUFFT controls rejected before kernel assertions");
#ifdef USE_MPI
    if (FML::NTasks > 1) {
        auto mismatch = c;
        if (FML::ThisTask)
            mismatch.origin[0] = 1;
        check(FML::TESTS::throws([&] { FF::PeriodicCatalogT<Grid> bad(prototype, mismatch); }),
              label + " inconsistent origin rejected collectively");
    }
#endif
    if (FML::ThisTask == 0)
        std::cout << "CATALOG_ERROR " << label << ' ' << error << " fine=" << stream.fine_mesh() << '\n';
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    exercise<FML::GRID::FFTWGrid<3>>(FML::Domain::uniform_slab(FML::NTasks), "slab");
#ifdef USE_MPI
    for (int px = 1; px <= FML::NTasks; ++px)
        if (FML::NTasks % px == 0) {
            int py = FML::NTasks / px;
            std::vector<double> x(px + 1), y(py + 1);
            for (int p = 0; p <= px; ++p)
                x[p] = double(p) / px;
            for (int p = 0; p <= py; ++p)
                y[p] = double(p) / py;
            exercise<FML::GRID::PencilGrid<3>>(FML::Domain::pencil(x, y),
                                               "pencil-" + std::to_string(px) + "x" + std::to_string(py));
        }
#endif
    return FML::TESTS::summary("periodic catalog NUFFT");
}
