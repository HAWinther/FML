#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <FML/FieldLevel/Checkpoint.h>
#include <FML/FieldLevel/FixedIC.h>
#include <FML/tests/TestHelpers.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace E = FML::FIELDLEVEL::EULERIAN;
using FML::FIELDLEVEL::Complex;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
static double dot(const Field & a, const Field & b) {
    double s = 0;
    for (size_t q = 0; q < a.size(); ++q)
        s += a[q] * b[q];
    return s;
}
static std::array<FF::Parameter, 4> priors() {
    return {{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
             {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
             {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
             {"b_laplacian", 7., 0, 1, -100, 100, FF::Prior::Uniform}}};
}
template <class Grid>
void exercise(const FML::Domain & domain, const std::string & label, bool pswf = false) {
    FF::EulerianLPTConfiguration c;
    c.matter_assignment = pswf ? FF::MatterAssignment::PSWF : FF::MatterAssignment::Smooth;
    c.nufft_epsilon = 1e-8;
    c.mesh = 12;
    c.density_mesh = 24;
    c.boxsize = 100;
    const double kf = 2 * M_PI / c.boxsize;
    c.initial_cutoff = 1.6 * kf;
    c.matter_band = {2.5 * kf, E::BandGeometry::Cube};
    c.likelihood_band = {1.6 * kf, E::BandGeometry::Cube};
    using Model = FF::DistributedEulerianLPTModelT<Grid>;
    Model model(c, FF::sigma8_spectrum([](double k) { return 500 * std::exp(-5 * k); }), priors(), domain);
    auto w = model.workspace();
    auto p = model.initial_parameters();
    p[0] = 1.08;
    auto s = model.white_noise(401), direction = model.white_noise(402);
    check(p.size() == 5 && model.parameters().back().name == "b_laplacian",
          label + " five physical scalars and no dummy noise");
    double count = 0, corners = 0;
    for (const auto & m : model.modes()) {
        count += m.weight;
        if (m.k > c.likelihood_band.cutoff)
            corners += m.weight;
    }
    check(model.sum(count) == 26 && model.sum(corners) == 8,
          label + " cubic likelihood enumerates all 26 modes including corners");
    const double power = 1000;
    auto truth = p;
    truth[0] = .97;
    truth[1] = 1.8;
    truth[2] = .2;
    const auto data = model.mock(s, truth, power, 403, w);
    auto posterior = FF::fixed_noise_posterior(model, data, power, FF::InitialFieldPrior::ConditionOnField);
    Field gs, temp;
    std::vector<double> gp, tp;
    const double U = posterior.evaluate(s, p, w, gs, gp);
    check(std::isfinite(U) && gp.size() == p.size(), label + " finite fixed-noise conditional target");
    const auto prediction = model.predict(s, p, w);
    const auto columns = model.bias_modes(w);
    std::vector<double> weights;
    for (const auto & mode : model.modes())
        weights.push_back(mode.weight);
    std::array<double, 4> bias{};
    std::copy(p.begin() + 1, p.end(), bias.begin());
    FF::FixedNoiseLinearBiasLikelihood<4> cached(columns, data, weights, power / model.volume(), bias);
    check(FML::TESTS::near(cached.value(bias), U, 2e-13), label + " cached Fourier columns reproduce full target");
    const auto cg = cached.gradient(bias);
    double cache_error = 0;
    for (size_t a = 0; a < 4; ++a)
        cache_error = std::max(cache_error, std::abs(cg[a] - gp[a + 1]));
    check(cache_error < 2e-12, label + " cached bias gradients reproduce full reverse path");
    FF::Modes cotangent(prediction.size());
    for (size_t j = 0; j < prediction.size(); ++j)
        cotangent[j] = (prediction[j] - data[j]) / (power / model.volume());
    Field tape_g;
    std::vector<double> tape_p;
    model.pullback(p, cotangent, w, tape_g, tape_p);
    double tape_error = 0;
    for (size_t q = 0; q < gs.size(); ++q)
        tape_error = std::max(tape_error, std::abs(tape_g[q] - gs[q]));
    for (size_t j = 0; j < gp.size(); ++j)
        tape_error = std::max(tape_error, std::abs(model.sum(tape_p[j]) - gp[j]));
    check(model.maximum(tape_error) < 2e-11, label + " Fourier bias export preserves the full reverse tape");


    double manual = 0;
    for (size_t j = 0; j < prediction.size(); ++j)
        manual += .5 * model.modes()[j].weight *
                  (std::norm(prediction[j] - data[j]) / (power / model.volume()) + std::log(power / model.volume()));
    check(FML::TESTS::near(U, model.sum(manual), 2e-13),
          label + " Fourier/Hermitian/volume normalization and fixed noise");
    check(FML::TESTS::near(U, posterior.value(s, p, w), 2e-13), label + " value and gradient paths agree");
    const double h = 1e-4;
    auto a = s, b = s;
    for (size_t q = 0; q < s.size(); ++q) {
        a[q] += h * direction[q];
        b[q] -= h * direction[q];
    }
    const double numeric = (posterior.value(a, p, w) - posterior.value(b, p, w)) / (2 * h);
    check(FML::TESTS::near(model.sum(dot(gs, direction)), numeric, 3e-6),
          label + " field derivative through 2LPT, particles and Eulerian operators");
    for (size_t j = 0; j < p.size(); ++j) {
        auto pa = p, pb = p;
        const double step = h * std::max(1., std::abs(p[j]));
        pa[j] += step;
        pb[j] -= step;
        const double fd = (posterior.value(s, pa, w) - posterior.value(s, pb, w)) / (2 * step);
        if (FML::ThisTask == 0)
            std::cout << "GRAD " << label << ' ' << j << ' ' << gp[j] << ' ' << fd << '\n';
        check(FML::TESTS::near(gp[j], fd, 3e-6), label + " scalar derivative " + model.parameters()[j].name);
    }
    auto free = FF::fixed_noise_posterior(model, data, power);
    Field freeg;
    std::vector<double> freep;
    const double Uf = free.evaluate(s, p, w, freeg, freep);
    double e = 0;
    for (size_t q = 0; q < s.size(); ++q)
        e = std::max(e, std::abs(freeg[q] - gs[q] - s[q]));
    for (size_t j = 0; j < p.size(); ++j)
        e = std::max(e, std::abs(freep[j] - gp[j]));
    check(FML::TESTS::near(Uf - U, .5 * model.sum(dot(s, s)), 2e-13) && model.maximum(e) < 2e-12,
          label + " free-field prior changes only field penalty/gradient");
    auto reference = posterior.value_reference(s, p, w);
    auto trial = p;
    trial[0] += 1e-4;
    trial.back() += 1e-3;
    const auto change = posterior.value_change(a, trial, w, s, reference);
    const double value = posterior.value(a, trial, w);
    check(FML::TESTS::near(change.value, value, 2e-13) && FML::TESTS::near(change.change, value - U, 2e-10),
          label + " stable target differences use fixed noise and conditional field policy");
    auto free_ref = free.value_reference(s, p, w);
    const auto free_change = free.value_change(a, trial, w, s, free_ref);
    check(FML::TESTS::near(free_change.change, free.value(a, trial, w) - Uf, 2e-10),
          label + " free-field stable difference");
    const auto result = model.predict(s, p, w);
    // Bias linearity must hold after evolved matter is fixed.
    auto doubled = p;
    for (size_t j = 1; j < doubled.size(); ++j)
        doubled[j] *= 2;
    const auto two = model.predict(s, doubled, w);
    e = 0;
    for (size_t j = 0; j < result.size(); ++j)
        e = std::max(e, std::abs(two[j] - 2. * result[j]));
    check(model.maximum(e) < 2e-13, label + " linear Eulerian coefficients after unbiased evolution");
    // Unit Eulerian density must coincide with the legacy unbiased real-space LPT matter prediction.
    auto unbiased = p;
    unbiased[1] = 1;
    for (size_t j = 2; j < p.size(); ++j)
        unbiased[j] = 0;
    const auto matter = model.predict(s, unbiased, w);
    FF::Configuration gc;
    gc.matter_assignment = c.matter_assignment;
    gc.nufft_epsilon = c.nufft_epsilon;
    gc.mesh = c.mesh;
    gc.density_mesh = c.density_mesh;
    gc.boxsize = c.boxsize;
    gc.initial_cutoff = gc.bias_cutoff = c.initial_cutoff;
    gc.likelihood_cutoff = std::nextafter(std::sqrt(3.) * c.matter_band.cutoff, HUGE_VAL);
    FF::DistributedLPTModelT<Grid> legacy(gc,
                                          FF::sigma8_spectrum([](double k) { return 500 * std::exp(-5 * k); }),
                                          {},
                                          {},
                                          domain);
    auto lw = legacy.workspace();
    auto lp = legacy.initial_parameters();
    lp[0] = p[0];
    lp[1] = 1;
    for (size_t j = 2; j < lp.size(); ++j)
        lp[j] = 0;
    const auto legacy_matter = legacy.predict(s, lp, lw);
    std::map<std::array<int, 3>, Complex> lookup;
    for (size_t j = 0; j < legacy_matter.size(); ++j)
        lookup[{legacy.modes()[j].n[0], legacy.modes()[j].n[1], legacy.modes()[j].n[2]}] = legacy_matter[j];
    e = 0;
    for (size_t j = 0; j < matter.size(); ++j)
        e = std::max(
            e,
            std::abs(matter[j] - lookup.at({model.modes()[j].n[0], model.modes()[j].n[1], model.modes()[j].n[2]})));
    check(model.maximum(e) < 2e-13, label + " unbiased matter agrees with existing LPT core");
    // Independent particle Fourier sum: no assignment stencil or deconvolution is reused.
    const auto reconstruction = legacy.reconstruct(s, lp, lw, 0.);
    std::array<double, 52> direct{};
    std::map<std::array<int, 3>, size_t> direct_index;
    size_t mode_id = 0;
    for (int nx = -1; nx <= 1; ++nx)
        for (int ny = -1; ny <= 1; ++ny)
            for (int nz = -1; nz <= 1; ++nz) {
                if (nx == 0 && ny == 0 && nz == 0)
                    continue;
                const std::array<int, 3> wave = {nx, ny, nz};
                direct_index[wave] = mode_id;
                for (size_t q = 0; q < model.cells(); ++q) {
                    auto x = model.layout().global_coordinate(q);
                    double phase = 0;
                    for (int d = 0; d < 3; ++d)
                        phase -= 2 * M_PI * wave[d] * (double(x[d]) / c.mesh + reconstruction.displacement[d][q]);
                    direct[2 * mode_id] += std::cos(phase) / model.layout().global_cells;
                    direct[2 * mode_id + 1] += std::sin(phase) / model.layout().global_cells;
                }
                ++mode_id;
            }
    FML::SumArrayOverTasks(direct.data(), direct.size());
    e = 0;
    for (size_t j = 0; j < matter.size(); ++j) {
        const auto & n = model.modes()[j].n;
        const auto id = direct_index.at({n[0], n[1], n[2]});
        e = std::max(e, std::abs(matter[j] - Complex(direct[2 * id], direct[2 * id + 1])));
    }
    e = model.maximum(e);
    if (FML::ThisTask == 0)
        std::cout << "DIRECT_SUM " << label << ' ' << e << '\n';
    check(e < (pswf ? 1e-8 : 1e-4), label + " mass assignment agrees with direct particle Fourier sum");

    auto bad = data;
    if (FML::ThisTask == FML::NTasks - 1) {
        if (bad.empty())
            bad.push_back(Complex(0));
        else
            bad.pop_back();
    }
    check(FML::TESTS::throws([&] {
              auto invalid = FF::fixed_noise_posterior(model, bad, power);
              (void)invalid;
          }),
          label + " malformed data rejected collectively");
    check(FML::TESTS::throws([&] {
              auto invalid = FF::fixed_noise_posterior(model, data, -1);
              (void)invalid;
          }),
          label + " invalid fixed covariance rejected collectively");
#ifdef USE_MPI
    if (FML::NTasks > 1)
        check(FML::TESTS::throws([&] {
                  auto invalid = FF::fixed_noise_posterior(model, data, FML::ThisTask ? power : power * 2);
                  (void)invalid;
              }),
              label + " inconsistent fixed covariance rejected collectively");
#endif
}
// Independent single-particle Fourier amplitude and position derivative; no LPT reverse code reused.
static void particle_derivative_check() {
    FML::NUFFT::PSWFKernel<3> kernel(1e-10);
    const int nf = 32;
    const std::array<int, 3> wave{5, -3, 6};
    auto mode = [&](std::array<double, 3> x, std::array<Complex, 3> * gradient) {
        std::array<Complex, 3> a{}, da{};
        for (int d = 0; d < 3; ++d) {
            x[d] -= std::floor(x[d]);
            const double u = nf * x[d], h = .5 * kernel.width;
            const int start = int(std::ceil(u - h));
            double w[16], dw[16];
            kernel.weights_and_derivatives(start - u + h, w, dw);
            const double ft = kernel.fourier(2 * M_PI * wave[d] / nf);
            for (int j = 0; j < kernel.width; ++j) {
                const auto phase = std::exp(Complex(0, -2 * M_PI * wave[d] * (start + j) / nf));
                a[d] += w[j] * phase / ft;
                da[d] -= nf * dw[j] * phase / ft;
            }
        }
        if (gradient)
            for (int d = 0; d < 3; ++d)
                (*gradient)[d] = da[d] * a[(d + 1) % 3] * a[(d + 2) % 3];
        return a[0] * a[1] * a[2];
    };
    double value_error = 0, direct_gradient_error = 0, finite_difference_error = 0;
    for (int q = 0; q < 19; ++q) {
        std::array<double, 3> x{q == 0 ? 0. : .017 + .071 * q, -.021 + .137 * q, 1. + .039 * q};
        std::array<Complex, 3> gradient;
        const auto value = mode(x, &gradient);
        double phase = 0;
        for (int d = 0; d < 3; ++d) phase -= 2 * M_PI * wave[d] * x[d];
        const auto exact = std::exp(Complex(0, phase));
        value_error = std::max(value_error, std::abs(value - exact));
        for (int d = 0; d < 3; ++d) {
            direct_gradient_error = std::max(direct_gradient_error,
                std::abs(gradient[d] - Complex(0, -2 * M_PI * wave[d]) * exact));
            auto a = x, b = x;
            a[d] += 1e-6; b[d] -= 1e-6;
            finite_difference_error = std::max(finite_difference_error,
                std::abs(gradient[d] - (mode(a, nullptr) - mode(b, nullptr)) / 2e-6));
        }
    }
    if (FML::ThisTask == 0)
        std::cout << "PARTICLE_DERIVATIVE " << value_error << ' ' << direct_gradient_error << ' '
                  << finite_difference_error << '\n';
    check(value_error < 2e-10 && direct_gradient_error < 2e-7 && finite_difference_error < 2e-8,
          "PSWF individual particle positions: direct Fourier derivative and finite differences");
}
static void noise_policy_regression() {
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.evolution = FF::Evolution::Linear;
    c.initial_cutoff = c.bias_cutoff = c.likelihood_cutoff = .1;
    FF::Spectrum spectrum;
    spectrum.parameters = {{"log_amplitude", .3, .3, 1, -2, 2, FF::Prior::Uniform}};
    spectrum.evaluate = [](double, const auto & p) { return FF::Power{500 * std::exp(p[0]), {1.}}; };
    FF::DistributedLinearModel model(c, spectrum);
    auto w = model.workspace();
    const auto p = model.initial_parameters();
    auto s = model.white_noise(901);
    auto data = model.mock(s, p, 902, w);
    FF::BasicPosterior<FF::DistributedLinearModel> old(model, data),
        indexed(model, data, FF::GaussianNoise::parameter(2));
    Field ga, gb;
    std::vector<double> pa, pb;
    const double ua = old.evaluate(s, p, w, ga, pa), ub = indexed.evaluate(s, p, w, gb, pb);
    check(ua == ub && ga == gb && pa == pb, "default and explicit last-noise layouts are byte-identical");
    // One scalar controls both spectrum and noise here: this independently checks adding the
    // covariance derivative at a non-last index, including its log determinant.
    FF::BasicPosterior<FF::DistributedLinearModel> first(model, data, FF::GaussianNoise::parameter(0));
    first.evaluate(s, p, w, ga, pa);
    for (size_t j = 0; j < p.size(); ++j) {
        auto a = p, b = p;
        a[j] += 1e-5;
        b[j] -= 1e-5;
        const double numeric = (first.value(s, a, w) - first.value(s, b, w)) / 2e-5;
        check(FML::TESTS::near(numeric, pa[j], 2e-7), "non-last sampled noise derivative " + std::to_string(j));
    }
    FF::BasicPosterior<FF::DistributedLinearModel> fixed(model, data, FF::GaussianNoise::fixed_power(1000));
    FF::BasicPosterior<FF::DistributedLinearModel> changed(model, data, FF::GaussianNoise::fixed_power(1100));
    FF::BasicPosterior<FF::DistributedLinearModel> conditional(model,
                                                               data,
                                                               FF::GaussianNoise::fixed_power(1000),
                                                               FF::InitialFieldPrior::ConditionOnField);
    const auto old_id = FF::CHECKPOINT::identity(old, "noise policy regression");
    check(old_id != FF::CHECKPOINT::identity(fixed, "noise policy regression") &&
              FF::CHECKPOINT::identity(fixed, "noise policy regression") !=
                  FF::CHECKPOINT::identity(changed, "noise policy regression") &&
              FF::CHECKPOINT::identity(fixed, "noise policy regression") !=
                  FF::CHECKPOINT::identity(conditional, "noise policy regression"),
          "checkpoint target identity includes noise and field-prior policy");
    check(FML::TESTS::throws([&] { FF::map(conditional, s, p); }), "joint MAP rejects a conditional field target");
    FML::MCMC::HMCOptions options;
    options.samples = 1;
    options.warmup = 0;
    check(FML::TESTS::throws([&] { FF::sample(conditional, s, p, options); }),
          "field HMC rejects a conditional field target");
    check(FML::TESTS::throws([&] {
              FF::BasicPosterior<FF::DistributedLinearModel> invalid(model, data, FF::GaussianNoise::parameter(20));
          }),
          "out-of-range noise index rejected");
    c.evolution = FF::Evolution::Zeldovich;
    FF::DistributedLPTModel smooth(c, spectrum);
    auto sw = smooth.workspace();
    const auto observations = smooth.predict(s, smooth.initial_parameters(), sw);
    FF::BasicPosterior<FF::DistributedLPTModel> smooth_target(smooth, observations);
    c.matter_assignment = FF::MatterAssignment::PSWF;
    FF::DistributedLPTModel pswf(c, spectrum);
    FF::BasicPosterior<FF::DistributedLPTModel> pswf_target(pswf, observations);
    c.nufft_epsilon = 1e-10;
    FF::DistributedLPTModel tighter(c, spectrum);
    FF::BasicPosterior<FF::DistributedLPTModel> tighter_target(tighter, observations);
    check(FF::CHECKPOINT::identity(smooth_target, "assignment") !=
              FF::CHECKPOINT::identity(pswf_target, "assignment") &&
          FF::CHECKPOINT::identity(pswf_target, "assignment") !=
              FF::CHECKPOINT::identity(tighter_target, "assignment"),
          "checkpoint fingerprint distinguishes PSWF and its accuracy");
    check(FML::TESTS::throws([&] { FF::Model unsupported(c, spectrum); }),
          "compatibility backend rejects unsupported PSWF selection");
    c.evolution = FF::Evolution::Linear;
    check(FML::TESTS::throws([&] { FF::DistributedLinearModel unsupported(c, spectrum); }),
          "linear backend rejects inactive PSWF selection");
    c.evolution = FF::Evolution::Zeldovich;
    c.nufft_epsilon = 0;
    check(FML::TESTS::throws([&] { FF::DistributedLPTModel invalid(c, spectrum); }),
          "invalid PSWF tolerance rejected before kernel assertions");
    c.nufft_epsilon = 1e-8;
    c.density_mesh = c.mesh;
    check(FML::TESTS::throws([&] { FF::DistributedLPTModel invalid(c, spectrum); }),
          "insufficient PSWF upsampling rejected");
    c.density_mesh = 16;
    if (FML::ThisTask == FML::NTasks - 1)
        c.matter_assignment = FF::MatterAssignment(77);
    check(FML::TESTS::throws([&] { FF::DistributedLPTModel invalid(c, spectrum); }),
          "rank-local invalid assignment rejected collectively");
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    noise_policy_regression();
    particle_derivative_check();
    for (double epsilon : {1e-4, 1e-8, 1e-12}) {
        FML::NUFFT::PSWFKernel<3> kernel(epsilon);
        double max_weight_error = 0, max_derivative_error = 0;
        for (double delta : {.001, .17, .5, .83, .999}) {
            double w[16], dw[16], original[16], plus[16], minus[16];
            kernel.weights_and_derivatives(delta, w, dw);
            kernel.weights(delta, original);
            kernel.weights(delta + 1e-5, plus);
            kernel.weights(delta - 1e-5, minus);
            for (int j = 0; j < kernel.width; ++j) {
                max_weight_error = std::max(max_weight_error, std::abs(w[j] - original[j]));
                max_derivative_error = std::max(max_derivative_error,
                    std::abs(dw[j] - (plus[j] - minus[j]) / 2e-5));
            }
        }
        check(max_weight_error < 1e-14 && max_derivative_error < 2e-9,
              "PSWF fitted-weight analytic derivative epsilon=" + std::to_string(epsilon));
    }
    auto sp = FF::sigma8_spectrum([](double) { return 7.; });
    auto power = sp.evaluate(.1, {1.2});
    check(FML::TESTS::near(power.value, 7 * 1.2 * 1.2, 1e-14) && FML::TESTS::near(power.dlog[0], 2 / 1.2, 1e-14),
          "sigma8 amplitude and analytic power derivative");
    exercise<FML::GRID::FFTWGrid<3>>(FML::Domain::uniform_slab(FML::NTasks), "slab");
    exercise<FML::GRID::FFTWGrid<3>>(FML::Domain::uniform_slab(FML::NTasks), "slab-PSWF", true);
#ifdef USE_MPI
    const int px = FML::NTasks % 2 ? 1 : 2, py = FML::NTasks / px;
    std::vector<double> ex(px + 1), ey(py + 1);
    for (int p = 0; p <= px; ++p)
        ex[p] = double(p) / px;
    for (int p = 0; p <= py; ++p)
        ey[p] = double(p) / py;
    exercise<FML::GRID::PencilGrid<3>>(FML::Domain::pencil(ex, ey), "pencil");
    exercise<FML::GRID::PencilGrid<3>>(FML::Domain::pencil(ex, ey), "pencil-PSWF", true);
#endif
    return FML::TESTS::summary("Eulerian LPT composition");
}
