#include <FML/FieldLevel/DistributedLPT.h>
#include <FML/tests/TestHelpers.h>
#include <iostream>
#include <iomanip>
#include <fstream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Complex;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check_local;
static double rel(double a, double b) { return std::abs(a - b) / std::max({1., std::abs(a), std::abs(b)}); }
static double dot(const Field & a, const Field & b) {
    double s = 0;
    for (size_t q = 0; q < a.size(); ++q)
        s += a[q] * b[q];
    return s;
}
static double maxerr(const Field & a, const Field & b) {
    double e = 0;
    for (size_t q = 0; q < a.size(); ++q)
        e = std::max(e, std::abs(a[q] - b[q]));
    return e;
}
static void run_case(int argc, char ** argv, FF::Evolution evolution) {
    const std::string label = evolution == FF::Evolution::LPT2 ? "2LPT" : "1LPT";
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.evolution = evolution;
    c.growth_rate = .6;
    FF::Spectrum spectrum;
    spectrum.parameters = {{"log_amplitude", .1, 0, 1}, {"tilt", .2, 0, 1}};
    spectrum.evaluate = [](double k, const auto & p) {
        return FF::Power{200 * std::exp(p[0]) * std::pow(k / .1, p[1]), {1., std::log(k / .1)}};
    };
    auto lc = c;
    lc.evolution = FF::Evolution::Linear;
    FF::DistributedLinearModel metadata(lc, spectrum);
    FF::FourierResponse response;
    for (const auto & mode : metadata.modes())
        response.multiplier.push_back(.9 + .02 * mode.n[0] * mode.n[0]);
    FF::DistributedLPTModel model(c, spectrum, response);
    auto w = model.workspace();
    auto p = model.initial_parameters();
    p[2] = 1.4;
    p[3] = .2;
    p[4] = -.15;
    p[5] = 12;
    auto s = model.white_noise(11), direction = model.white_noise(12);
    auto data = model.mock(model.white_noise(13), p, 14, w);
    FF::DistributedLPTPosterior posterior(model, data);
    Field gs, tmp;
    std::vector<double> gp, ptmp;
    const double U = posterior.evaluate(s, p, w, gs, gp), h = 1e-5;
    auto plus = s, minus = s;
    for (size_t q = 0; q < s.size(); ++q) {
        plus[q] += h * direction[q];
        minus[q] -= h * direction[q];
    }
    const double numeric =
        (posterior.evaluate(plus, p, w, tmp, ptmp) - posterior.evaluate(minus, p, w, tmp, ptmp)) / (2 * h);
    check_local(rel(model.sum(dot(gs, direction)), numeric) < 2e-6,
                label + " white-field gradient with distinct cutoffs/bias/RSD");
    for (size_t j = 0; j < p.size(); ++j) {
        auto a = p, b = p;
        const double step = h * std::max(1., std::abs(p[j]));
        a[j] += step;
        b[j] -= step;
        const double fd =
            (posterior.evaluate(s, a, w, tmp, ptmp) - posterior.evaluate(s, b, w, tmp, ptmp)) / (2 * step);
        if (rel(fd, gp[j]) >= 2e-6 && FML::ThisTask == 0)
            std::cout << "GRAD " << j << ' ' << gp[j] << ' ' << fd << '\n';
        check_local(rel(fd, gp[j]) < 2e-6, label + " scalar derivative " + model.parameters()[j].name);
    }
    model.predict(s, p, w);
    Field pull;
    std::vector<double> pullp;
    model.pullback(p, data, w, pull, pullp);
    const auto yp = model.predict(plus, p, w), ym = model.predict(minus, p, w);
    double left = 0;
    for (size_t i = 0; i < data.size(); ++i)
        left += model.modes()[i].weight * std::real(std::conj(data[i]) * (yp[i] - ym[i])) / (2 * h);
    check_local(rel(model.sum(left), model.sum(dot(pull, direction))) < 2e-6,
                label + " forward/reverse dot product through both MPI communication operators");
    const auto prediction = model.predict(s, p, w);
    if (FML::NTasks == 1) {
        FF::Model serial(c, spectrum, response);
        auto sw = serial.workspace();
        FF::Posterior sp(serial, data);
        Field sg;
        std::vector<double> spg;
        const auto ref = serial.predict(s, p, sw);
        double error = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            error = std::max(error, std::abs(ref[i] - prediction[i]));
        const double su = sp.evaluate(s, p, sw, sg, spg);
        check_local(error < 1e-12, label + " prediction matches original serial backend");
        double pgerr = 0;
        for (size_t j = 0; j < p.size(); ++j)
            pgerr = std::max(pgerr, std::abs(gp[j] - spg[j]));
        check_local(rel(U, su) < 1e-13 && maxerr(gs, sg) < 1e-11 && pgerr < 1e-9,
                    label + " serial posterior and all field/scalar gradients");
    }
    // Strong displacements traverse rank boundaries and periodic wraps; check the reverse path there too.
    auto large = s;
    for (auto & x : large)
        x *= 80;
    const double largeU = posterior.evaluate(large, p, w, tmp, ptmp), large_analytic = model.sum(dot(tmp, direction));
    auto la = large, lb = large;
    for (size_t q = 0; q < s.size(); ++q) {
        la[q] += h * direction[q];
        lb[q] -= h * direction[q];
    }
    const double large_fd =
        (posterior.evaluate(la, p, w, tmp, ptmp) - posterior.evaluate(lb, p, w, tmp, ptmp)) / (2 * h);
    check_local(rel(large_analytic, large_fd) < 2e-6, "large-displacement periodic/rank-crossing adjoint");
    // Independent scatter/gather dot test: analytic mesh cotangent, distant nodes,
    // colliding deposits, empty requests and no particle physics shared with the model.
    FML::GRID::FFTWGrid<3> grid(16);
    FF::DETAIL::DistributedAssignment scatter(grid);
    check_local(FML::TESTS::throws([&] { scatter.reverse(grid); }),
                "assignment reverse requires a successful matching scatter");
    auto meshbar = [](std::uint64_t node) { return std::cos(.017 * node); };
    double expected = 0;
    auto requests = [&](auto add) {
        if (FML::NTasks > 1 && FML::ThisTask == 0)
            return;
        for (size_t q = 0; q < model.cells(); ++q) {
            const size_t global = size_t(model.layout().local_start) * 64 + q;
            const auto a = (global * 37 + 11) % 4096, b = (4095 - global) % 4096;
            const double value = 1 + std::sin(.1 * global);
            add(a, value);
            add(b, -.3 * value);
        }
    };
    requests([&](auto node, double value) { expected += value * meshbar(node); });
    scatter.deposit(grid, requests);
    double actual = 0;
    size_t q = 0;
    for (auto i : grid.get_real_range()) {
        const size_t global = size_t(grid.get_local_x_start()) * 256 + q++;
        actual += grid.get_real_from_index(i) * meshbar(global);
        grid.set_real_from_index(i, meshbar(global));
    }
    const auto reverse = scatter.reverse(grid);
    double transport_error = 0;
    requests([&](auto node, double) {
        transport_error = std::max(transport_error, std::abs(scatter.at(reverse, node) - meshbar(node)));
    });
    check_local(rel(model.sum(actual), model.sum(expected)) < 1e-13 && model.maximum(transport_error) < 1e-14,
                "sparse MPI deposition/reverse exact dot product including empty request ranks");
    check_local(FML::TESTS::throws([&] {
                    scatter.deposit(grid, [&](auto add) {
                        if (FML::ThisTask == 0)
                            add(std::numeric_limits<std::uint64_t>::max(), 1.);
                    });
                }),
                "invalid routed node key fails collectively before MPI payload exchange");
    const auto fit = FF::map(posterior, s, p, false, 5, 1e-7);
    check_local(std::isfinite(fit.potential) && fit.potential < U,
                "distributed nonlinear MAP reduces the global objective");
    FML::MCMC::HMCOptions options;
    options.warmup = 4;
    options.samples = 8;
    options.n_leapfrog = 2;
    options.jitter = false;
    options.step_size = .005;
    options.adapt_step_size = false;
    options.adapt_mass = false;
    options.seed = 17;
    auto chain = FF::sample(posterior, s, p, options, true);
    bool synced = true;
    for (const auto & row : chain.recorded)
        for (double x : row)
            synced &= model.maximum(x) == -model.maximum(-x);
    check_local(synced && chain.acceptance > .5 && chain.divergences == 0 &&
                    chain.recorded.front() != chain.recorded.back(),
                label + " joint HMC moves with synchronized scalar states (execution smoke only)");
    check_local(FML::TESTS::throws([&] {
                    auto wrong = s;
                    if (FML::ThisTask == 0)
                        wrong.pop_back();
                    posterior.evaluate(wrong, p, w, tmp, ptmp);
                }),
                label + " malformed local state fails collectively");
    auto nonfinite = s;
    if (FML::ThisTask == 0)
        nonfinite[0] = HUGE_VAL;
    check_local(std::isinf(posterior.evaluate(nonfinite, p, w, tmp, ptmp)),
                "nonfinite particle proposal rejects collectively before coordinate conversion");
    check_local(FML::TESTS::throws([&] {
                    auto bad = c;
                    bad.evolution = FF::Evolution::Linear;
                    FF::DistributedLPTModel unsupported(bad, spectrum);
                }),
                "non-LPT evolution rejected explicitly");
    double norm = 0;
    for (size_t i = 0; i < prediction.size(); ++i)
        norm += model.modes()[i].weight * std::norm(prediction[i]);
    const double checksum = model.sum(dot(gs, direction)), prednorm = model.sum(norm);
    if (FML::ThisTask == 0) {
        std::cout << std::setprecision(17) << "LPT_MPI_RESULT " << label << ' ' << U << ' ' << checksum << ' '
                  << prednorm << ' ' << largeU << ' ' << large_analytic << ' ' << fit.potential << ' '
                  << chain.acceptance;
        for (double x : gp)
            std::cout << ' ' << x;
        for (double x : chain.recorded.back())
            std::cout << ' ' << x;
        std::cout << '\n';
    }
    if (argc == 2) {
        FML::COMMUNICATION::collective_try(
            [&] {
                std::ofstream out(std::string(argv[1]) + "-" + label + ".rank" + std::to_string(FML::ThisTask) +
                                  ".txt");
                if (!out)
                    throw std::runtime_error("Cannot open rank-comparison output");
                out << std::setprecision(17);
                for (size_t i = 0; i < prediction.size(); ++i) {
                    const auto & mode = model.modes()[i];
                    out << "M " << mode.n[0] << ' ' << mode.n[1] << ' ' << mode.n[2] << ' ' << prediction[i].real()
                        << ' ' << prediction[i].imag() << '\n';
                }
                const size_t offset = size_t(model.layout().local_start) * c.mesh * c.mesh;
                for (size_t q = 0; q < gs.size(); ++q)
                    out << "F " << offset + q << ' ' << gs[q] << '\n';
                out.close();
                if (!out)
                    throw std::runtime_error("Cannot write rank-comparison output");
            },
            "LPT comparison output");
    }
}
// Two orthogonal plane waves give S2=A B cos(kx) cos(kz) analytically.
// Compare observable modes to an independent direct sum over the exact particle positions.
static void manufactured() {
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.evolution = FF::Evolution::LPT2;
    c.growth_rate = .6;
    c.initial_cutoff = .08;
    c.bias_cutoff = .07;
    c.likelihood_cutoff = .10;
    auto power = FF::Spectrum::fixed([](double) { return 200.; });
    FF::DistributedLPTModel second(c, power);
    auto w = second.workspace();
    auto p = second.initial_parameters();
    p[0] = 1;
    const double A = .5, B = .4, T = std::sqrt(200. * 512 / second.volume());
    Field white(second.cells());
    for (size_t q = 0; q < white.size(); ++q) {
        const size_t global = size_t(second.layout().local_start) * 64 + q;
        const double x = 2 * M_PI * double(global / 64) / 8, z = 2 * M_PI * double(global % 8) / 8;
        white[q] = (A * std::cos(x) + B * std::cos(z)) / T;
    }
    const auto prediction = second.predict(white, p, w);
    double error = 0, second_signal = 0;
    for (size_t i = 0; i < prediction.size(); ++i) {
        const auto & m = second.modes()[i];
        Complex exact = 0, first_only = 0;
        for (int x = 0; x < 8; ++x)
            for (int y = 0; y < 8; ++y)
                for (int z = 0; z < 8; ++z) {
                    const double ax = 2 * M_PI * x / 8, az = 2 * M_PI * z / 8;
                    const double px1 = -A / (2 * M_PI) * std::sin(ax);
                    const double pz1 = -(1 + c.growth_rate) * B / (2 * M_PI) * std::sin(az);
                    const double px2 = -3. / 7 * A * B / (4 * M_PI) * std::sin(ax) * std::cos(az);
                    const double pz2 =
                        -(1 + 2 * c.growth_rate) * 3. / 7 * A * B / (4 * M_PI) * std::cos(ax) * std::sin(az);
                    const double phase1 =
                        2 * M_PI *
                        (m.n[0] * (double(x) / 8 + px1) + m.n[1] * double(y) / 8 + m.n[2] * (double(z) / 8 + pz1));
                    const double phase2 = phase1 + 2 * M_PI * (m.n[0] * px2 + m.n[2] * pz2);
                    first_only += std::polar(1., -phase1) / 512.;
                    exact += std::polar(1., -phase2) / 512.;
                }
        error = std::max(error, std::abs(prediction[i] - exact));
        second_signal = std::max(second_signal, std::abs(exact - first_only));
    }
    const double analytic_error = second.maximum(error), signal = second.maximum(second_signal);
    if (FML::ThisTask == 0)
        std::cout << "MANUFACTURED_2LPT " << std::setprecision(17) << analytic_error << ' ' << signal << '\n';
    check_local(analytic_error < 2e-5 && signal > .01,
                "2LPT EdS coefficient/sign and RSD match analytic two-wave particle DFT (finite assignment tolerance)");
    // A single axis wave has exactly zero second-order source.
    for (size_t q = 0; q < white.size(); ++q) {
        const size_t global = size_t(second.layout().local_start) * 64 + q;
        white[q] = A * std::cos(2 * M_PI * double(global / 64) / 8) / T;
    }
    const auto one_wave = second.predict(white, p, w);
    c.evolution = FF::Evolution::Zeldovich;
    FF::DistributedLPTModel first(c, power);
    auto fw = first.workspace();
    const auto first_wave = first.predict(white, p, fw);
    error = 0;
    for (size_t i = 0; i < one_wave.size(); ++i)
        error = std::max(error, std::abs(one_wave[i] - first_wave[i]));
    check_local(second.maximum(error) < 1e-12, "single-plane-wave 2LPT source cancels on distributed slabs");
    // The supported assignment widths also need to preserve the forward/adjoint contract.
    for (int width : {4, 8}) {
        auto alternate = c;
        alternate.evolution = FF::Evolution::LPT2;
        alternate.kernel_width = width;
        FF::DistributedLPTModel model(alternate, power);
        auto work = model.workspace();
        auto state = model.white_noise(51), direction = model.white_noise(52);
        auto data = model.mock(model.white_noise(53), p, 54, work);
        FF::DistributedLPTPosterior posterior(model, data);
        Field gradient, tmp;
        std::vector<double> gp, ptmp;
        posterior.evaluate(state, p, work, gradient, gp);
        auto plus = state, minus = state;
        const double h = 1e-5;
        for (size_t q = 0; q < state.size(); ++q) {
            plus[q] += h * direction[q];
            minus[q] -= h * direction[q];
        }
        const double fd =
            (posterior.evaluate(plus, p, work, tmp, ptmp) - posterior.evaluate(minus, p, work, tmp, ptmp)) / (2 * h);
        check_local(rel(model.sum(dot(gradient, direction)), fd) < 2e-6,
                    "2LPT field adjoint with assignment width " + std::to_string(width));
    }
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    run_case(argc, argv, FF::Evolution::Zeldovich);
    run_case(argc, argv, FF::Evolution::LPT2);
    manufactured();
    return FML::TESTS::summary("distributed 1LPT/2LPT field framework");
}
