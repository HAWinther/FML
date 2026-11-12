#include <FML/FieldLevel/DistributedLinear.h>
#include <FML/tests/TestHelpers.h>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FF::Modes;
using FML::FIELDLEVEL::Complex;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check_local;
static double relative(double a, double b) { return std::abs(a - b) / std::max({1., std::abs(a), std::abs(b)}); }
static double inner(const Field & a, const Field & b) {
    double x = 0;
    for (size_t q = 0; q < a.size(); ++q)
        x += a[q] * b[q];
    return x;
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
    FF::Spectrum spectrum;
    spectrum.parameters = {{"log_amplitude", .1, 0, 1}, {"tilt", .2, 0, 1}};
    spectrum.evaluate = [](double k, const auto & p) {
        return FF::Power{200 * std::exp(p[0]) * std::pow(k / .1, p[1]), {1., std::log(k / .1)}};
    };
    FF::DistributedLinearModel model(c, spectrum);
    auto w = model.workspace();
    auto p = model.initial_parameters();
    auto s = model.white_noise(91), truth = model.white_noise(92);
    auto data = model.mock(truth, p, 93, w);
    FF::DistributedLinearPosterior posterior(model, data);
    Field gs, tmp;
    std::vector<double> gp, ptmp;
    const double U = posterior.evaluate(s, p, w, gs, gp);
    check_local(model.sum(double(model.cells())) == 512 && model.cells() == size_t(model.layout().local_width) * 64,
                "field vectors contain only owned cells; ownership covers global mesh");
    // Independent O(N^3) DFT per retained mode, with no FFT conventions reused.
    const auto prediction = model.predict(s, p, w);
    double error = 0;
    for (size_t i = 0; i < prediction.size(); ++i) {
        const auto & m = model.modes()[i];
        Complex sk = 0;
        for (int x = 0; x < 8; ++x)
            for (int y = 0; y < 8; ++y)
                for (int z = 0; z < 8; ++z) {
                    const size_t q = (size_t(x) * 8 + y) * 8 + z;
                    FML::RANDOM::COUNTER_BASED::Stream rng(91, q, 0x4649454cu);
                    sk += rng.normal() * std::polar(1., -2 * M_PI * (m.n[0] * x + m.n[1] * y + m.n[2] * z) / 8.);
                }
        sk /= 512.;
        const double P = spectrum.evaluate(m.k, {p[0], p[1]}).value;
        const double B =
            p[2] + c.growth_rate * double(m.n[2] * m.n[2]) / (m.n[0] * m.n[0] + m.n[1] * m.n[1] + m.n[2] * m.n[2]);
        const Complex ref = m.k < c.initial_cutoff ? B * std::sqrt(P * 512 / model.volume()) * sk : Complex(0);
        error = std::max(error, std::abs(prediction[i] - ref));
    }
    check_local(model.maximum(error) < 1e-12,
                "distributed prediction matches independent direct DFT and global normalization");
    auto direction = model.white_noise(94), plus = s, minus = s;
    const double h = 1e-5;
    for (size_t q = 0; q < s.size(); ++q) {
        plus[q] += h * direction[q];
        minus[q] -= h * direction[q];
    }
    const double numeric =
        (posterior.evaluate(plus, p, w, tmp, ptmp) - posterior.evaluate(minus, p, w, tmp, ptmp)) / (2 * h);
    check_local(relative(model.sum(inner(gs, direction)), numeric) < 2e-7,
                "global posterior directional field gradient");
    for (size_t j = 0; j < p.size(); ++j) {
        auto a = p, b = p;
        a[j] += h;
        b[j] -= h;
        const double fd = (posterior.evaluate(s, a, w, tmp, ptmp) - posterior.evaluate(s, b, w, tmp, ptmp)) / (2 * h);
        check_local(relative(fd, gp[j]) < 2e-7,
                    "global scalar gradient including once-only scalar prior " + std::to_string(j));
    }
    // A separate adjoint dot product includes cross-rank Hermitian plane modes.
    model.predict(s, p, w);
    Field pull;
    std::vector<double> pp;
    model.pullback(p, data, w, pull, pp);
    const auto yp = model.predict(plus, p, w), ym = model.predict(minus, p, w);
    double dot = 0;
    for (size_t i = 0; i < data.size(); ++i)
        dot += model.modes()[i].weight * std::real(std::conj(data[i]) * (yp[i] - ym[i])) / (2 * h);
    check_local(relative(model.sum(inner(pull, direction)), model.sum(dot)) < 1e-10,
                "distributed forward/reverse dot product");
    const auto exact = model.linear_conditional(data, p, w);
    const auto fit = FF::map(posterior, s, p, false, 10, 1e-9);
    error = 0;
    for (size_t q = 0; q < s.size(); ++q)
        error = std::max(error, std::abs(fit.white[q] - exact.mean[q]));
    check_local(fit.status == FF::MAPStatus::Converged && model.maximum(error) < 1e-10,
                "distributed MAP equals exact linear conditional mean");
    if (FML::NTasks == 1) {
        FF::Model serial(c, spectrum);
        auto sw = serial.workspace();
        FF::Posterior sp(serial, data);
        Field ref;
        std::vector<double> rg;
        const double su = sp.evaluate(s, p, sw, ref, rg);
        error = 0;
        for (size_t q = 0; q < s.size(); ++q)
            error = std::max(error, std::abs(gs[q] - ref[q]));
        for (size_t j = 0; j < p.size(); ++j)
            error = std::max(error, std::abs(gp[j] - rg[j]));
        const auto se = serial.linear_conditional(data, p, sw);
        check_local(relative(su, U) < 1e-13 && error < 1e-11 && relative(se.log_marginal, exact.log_marginal) < 1e-13,
                    "distributed backend agrees with existing serial posterior, gradients and exact marginal");
    }
    FF::FourierResponse response;
    for (const auto & mode : model.modes())
        response.multiplier.push_back(.8 + .03 * mode.n[0] * mode.n[0]);
    FF::DistributedLinearModel responding(c, spectrum, response);
    auto rw = responding.workspace();
    const auto response_prediction = responding.predict(s, p, rw);
    error = 0;
    for (size_t i = 0; i < prediction.size(); ++i)
        error = std::max(error, std::abs(response_prediction[i] - response.multiplier[i] * prediction[i]));
    check_local(model.maximum(error) < 1e-13, "local Fourier response preserves cross-rank mode ownership");
    FF::DistributedLinearPosterior response_post(responding, data);
    response_post.evaluate(s, p, rw, tmp, ptmp);
    const double response_analytic = model.sum(inner(tmp, direction));
    const double response_fd =
        (response_post.evaluate(plus, p, rw, tmp, ptmp) - response_post.evaluate(minus, p, rw, tmp, ptmp)) / (2 * h);
    check_local(relative(response_analytic, response_fd) < 2e-7, "nonidentity Fourier response adjoint");
    // Projection on a unit-norm real cosine: its conditional variance is exactly 1/precision.
    Field basis(s.size());
    for (size_t q = 0; q < s.size(); ++q) {
        const size_t global = size_t(model.layout().local_start) * 64 + q;
        basis[q] = std::sqrt(2. / 512) * std::cos(2 * M_PI * double(global / 64) / 8);
    }
    const double expected = model.sum(inner(basis, exact.mean));
    const double P = spectrum.evaluate(2 * M_PI / 100, {p[0], p[1]}).value;
    const double variance = 1 / (1 + p[2] * p[2] * P / std::exp(p.back()));
    FML::MCMC::HMCOptions options;
    options.warmup = 200;
    options.samples = 1600;
    options.n_leapfrog = 6;
    options.step_size = .15;
    options.seed = 871;
    options.target_acceptance = .85;
    std::vector<double> projections;
    auto chain = FF::sample(posterior, s, p, options, false, [&](const auto & field, const auto &, size_t) {
        projections.push_back(model.sum(inner(basis, field)));
    });
    double mean = 0, var = 0;
    for (double x : projections)
        mean += x / projections.size();
    for (double x : projections)
        var += (x - mean) * (x - mean) / (projections.size() - 1);
    const double ess = projections.size() / FML::MCMC::integrated_autocorrelation_time(projections);
    check_local(ess > 80 && std::abs(mean - expected) < 5 * std::sqrt(variance / ess) &&
                    std::abs(var / variance - 1) < .2,
                "distributed field HMC projection agrees with exact conditional mean and covariance");
    check_local(chain.divergences == 0 && chain.acceptance > .65,
                "distributed field HMC acceptance and divergence regression");
    // Joint scalar trajectory must remain replicated while local fields evolve.
    options.warmup = 50;
    options.samples = 60;
    options.seed = 872;
    options.step_size = .01;
    const auto joint = FF::sample(posterior, s, p, options, true);
    bool synchronized = true;
    for (const auto & row : joint.recorded)
        for (double x : row)
            synchronized &= model.maximum(x) == -model.maximum(-x);
    check_local(synchronized && joint.recorded.size() == 60,
                "joint HMC scalar states and decisions remain synchronized");
    check_local(FML::TESTS::throws([&] {
                    auto bad = s;
                    if (FML::ThisTask == 0)
                        bad.pop_back();
                    posterior.evaluate(bad, p, w, tmp, ptmp);
                }),
                "one-rank state dimension error fails collectively before FFT");
    auto bad = data;
    if (FML::ThisTask == 0)
        for (size_t i = 0; i < bad.size(); ++i)
            if (model.modes()[i].n[2] == 0) {
                bad[i] += Complex(0, 1);
                break;
            }
    check_local(FML::TESTS::throws([&] { FF::DistributedLinearPosterior invalid(model, bad); }),
                "cross-rank Hermitian data violations rejected collectively");
    check_local(FML::TESTS::throws([&] {
                    auto broken = spectrum;
                    broken.evaluate = [](double, const auto &) -> FF::Power {
                        if (FML::ThisTask == 0)
                            throw std::runtime_error("test callback failure");
                        return {200, {0, 0}};
                    };
                    FF::DistributedLinearModel invalid(c, broken);
                    auto iw = invalid.workspace();
                    invalid.predict(s, p, iw);
                }),
                "one-rank spectrum exception propagates collectively");
    check_local(FML::TESTS::throws([&] {
                    auto broken = spectrum;
                    broken.evaluate = [](double, const auto &) { return FF::Power{200, {}}; };
                    FF::DistributedLinearModel invalid(c, broken);
                    auto iw = invalid.workspace();
                    FF::DistributedLinearPosterior ip(invalid, data);
                    ip.evaluate(s, p, iw, tmp, ptmp);
                }),
                "missing spectrum derivatives remain a visible contract error");
    auto extreme = p;
    extreme[0] = 1000;
    const double invalidU = posterior.evaluate(s, extreme, w, tmp, ptmp);
    check_local(std::isinf(invalidU) && inner(tmp, tmp) == 0 && inner(ptmp, ptmp) == 0,
                "unrepresentable spectrum proposal rejected collectively with cleared gradients");
    check_local(FML::TESTS::throws([&] {
                    auto wrong = c;
                    wrong.evolution = FF::Evolution::LPT2;
                    FF::DistributedLinearModel unsupported(wrong, spectrum);
                }),
                "nonlinear evolution rejected explicitly by distributed linear backend");
    if (FML::NTasks > 1) {
        check_local(FML::TESTS::throws([&] {
                        auto a = p;
                        if (FML::ThisTask == 0)
                            a[0] += .1;
                        posterior.evaluate(s, a, w, tmp, ptmp);
                    }),
                    "inconsistent replicated parameters rejected collectively");
        check_local(FML::TESTS::throws([&] {
                        auto o = options;
                        if (FML::ThisTask == 0)
                            ++o.samples;
                        FF::sample(posterior, s, p, o);
                    }),
                    "inconsistent HMC controls rejected collectively");
        check_local(FML::TESTS::throws([&] {
                        auto wrong = c;
                        if (FML::ThisTask == 0)
                            wrong.mesh = 10;
                        FF::DistributedLinearModel invalid(wrong, spectrum);
                    }),
                    "inconsistent mesh configuration rejected before FFT allocation");
    }
    const double checksum = model.sum(inner(gs, direction));
    const double fieldsum = model.sum(inner(chain.last, basis)); // field-only final state
    if (FML::ThisTask == 0) {
        std::cout << std::setprecision(17) << "MPI_RESULT " << U << ' ' << checksum << ' ' << exact.log_marginal << ' '
                  << fieldsum << ' ' << chain.acceptance << ' ' << chain.step_size;
        for (double x : gp)
            std::cout << ' ' << x;
        for (double x : joint.recorded.back())
            std::cout << ' ' << x;
        std::cout << "\nJOINT_HEALTH " << joint.acceptance << ' ' << joint.divergences;
        std::cout << "\nHMC_PROJECTION " << mean << ' ' << expected << ' ' << var << ' ' << variance << ' ' << ess
                  << '\n';
    }
    return FML::TESTS::summary("distributed linear field framework");
}
