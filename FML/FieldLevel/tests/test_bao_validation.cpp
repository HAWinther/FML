// Bounded scientific validation of the serial BAO prototype. Not production coverage.
#include <FML/FieldLevel/FieldLevelBAO.h>
#include <FML/tests/TestHelpers.h>
#include <fstream>
#include <iomanip>
#include <map>
using namespace FML::FIELDLEVEL;
using FML::TESTS::check_local;
// Mirror the public sample adapter, but record physical-white field summaries as
// well as beta. This is test-only; the default public beta recorder is unchanged.
static FML::MCMC::HMCResult run(const FieldLevelBAO & model, Field x, double beta, FML::MCMC::HMCOptions opt) {
    const size_t n = model.ncells();
    const auto m = model.mass_spectrum(beta);
    std::vector<double> root(m.size()), inverse(m.size());
    for (size_t j = 0; j < m.size(); ++j) {
        root[j] = std::sqrt(m[j]);
        inverse[j] = 1 / m[j];
    }
    FML::MCMC::HMCMass mass;
    mass.n_custom = n;
    mass.apply_sqrt = [&](auto & p) { p = model.filter(p, root); };
    mass.apply_inverse = [&](auto & p) { p = model.filter(p, inverse); };
    mass.diagonal = {2500};
    opt.lower = {model.settings().beta_min};
    opt.upper = {model.settings().beta_max};
    auto potential = [&](const auto & state, auto & gradient) {
        Field f(state.begin(), state.begin() + n), g;
        double gb;
        double U = model.potential_and_gradient(f, state[n], g, gb);
        std::copy(g.begin(), g.end(), gradient.begin());
        gradient[n] = gb;
        return U;
    };
    auto record = [&](const auto & state) {
        const Field f(state.begin(), state.begin() + n);
        const auto w = model.white_from_sampled(f, state[n]);
        double a = 0, b = 0, power = 0;
        const int N = model.settings().nmesh;
        for (size_t j = 0; j < n; ++j) {
            const double phase = 2 * M_PI * double(j / (N * N)) / N;
            a += w[j] * std::cos(phase);
            b += w[j] * std::sin(phase);
            power += w[j] * w[j];
        }
        return std::vector<double>{state[n], a * std::sqrt(2. / n), b * std::sqrt(2. / n), power / n};
    };
    x.push_back(beta);
    return FML::MCMC::run_hmc(potential, std::move(x), mass, opt, record);
}
static std::pair<double, double> moments(const std::vector<FML::MCMC::HMCResult> & runs, int col) {
    double sum = 0, sum2 = 0, n = 0;
    for (const auto & r : runs)
        for (const auto & row : r.recorded) {
            sum += row[col];
            sum2 += row[col] * row[col];
            ++n;
        }
    return {sum / n, std::sqrt((sum2 - sum * sum / n) / (n - 1))};
}
static bool informative(double sd, const Settings & s) {
    return std::isfinite(sd) && sd > 0 && sd < .75 * (s.beta_max - s.beta_min) / std::sqrt(12.);
}
static void inference(const std::string & output, int samples, int mocks, bool second_order) {
    Settings s;
    s.nmesh = 12;
    s.nmesh_density = 24;
    s.boxsize = 600;
    s.kmax = .06;
    s.model = Model::LPT1;
    s.kernel_width = 4;
    s.noise_power = 300;
    s.beta_min = .85;
    s.beta_max = 1.15;
    if (second_order) {
        s.model = Model::LPT;
        s.lpt_order = 2;
        s.b2 = .3;
        s.bG2 = -.2;
        s.growth_rate = .7;
    }
    auto spectrum = BAOSpectrum::eisenstein_hu(1500);
    // Deliberately stronger toy wiggles make a small, affordable information test.
    // These settings are not a forecast for the physical 8%-wiggle spectrum.
    spectrum.amplitude = .5;
    FieldLevelBAO model(s, spectrum);
    std::ofstream table(output);
    if (!table)
        throw std::runtime_error("Cannot write validation rows");
    table << std::setprecision(17) << "# mock chain draw beta_true beta white_cos white_sin white_power\n";
    int covered = 0;
    for (int mock = 0; mock < mocks; ++mock) {
        std::mt19937_64 rng(7001 + 97 * mock);
        const double truth_beta =
            s.beta_min + (s.beta_max - s.beta_min) * (double(rng() >> 11) + .5) / 9007199254740992.;
        const auto truth = model.white_noise(rng);
        model.set_data(model.make_mock(truth, truth_beta, rng));
        std::vector<FML::MCMC::HMCResult> chains;
        for (int c = 0; c < 2; ++c) {
            FML::MCMC::HMCOptions opt;
            opt.warmup = 250;
            opt.samples = samples;
            opt.n_leapfrog = 10;
            opt.target_acceptance = .8;
            opt.seed = 10001 + 137 * mock + c;
            chains.push_back(run(model, model.white_noise(rng), c ? 1.12 : .88, opt));
            for (size_t i = 0; i < chains.back().recorded.size(); ++i) {
                table << mock << ' ' << c << ' ' << i << ' ' << truth_beta;
                for (double z : chains.back().recorded[i])
                    table << ' ' << z;
                table << '\n';
            }
        }
        for (int col = 0; col < 4; ++col) {
            const auto d = FML::MCMC::diagnose_hmc_chains(chains, col);
            std::cout << "DIAGNOSTIC mock=" << mock << " col=" << col << " Rhat=" << d.rhat << " bulk=" << d.ess_bulk
                      << " tail=" << d.ess_tail << '\n';
            check_local(d.valid() && d.rhat < 1.05 && d.ess_bulk > 80 && d.ess_tail > 50,
                        "dispersed BAO chains mix for beta/physical-white summaries");
        }
        const auto stats = moments(chains, 0);
        check_local(informative(stats.second, s), "nonlinear BAO posterior is narrower than the prior");
        // Modest mock recovery diagnostic; not a repeated-mock coverage certification.
        const double z = (stats.first - truth_beta) / stats.second;
        covered += std::abs(z) < 2;
        std::cout << "RECOVERY mock=" << mock << " truth=" << truth_beta << " mean=" << stats.first
                  << " sd=" << stats.second << " z=" << z << '\n';
        check_local(std::abs(z) < 4, "informative nonlinear mock is compatible with truth");
        for (const auto & c : chains) {
            std::cout << "SAMPLER acceptance=" << c.acceptance << " divergences=" << c.divergences << '\n';
            check_local(c.acceptance > .6 && c.divergences == 0, "BAO chain acceptance and divergence checks");
        }
    }
    std::cout << "MOCK_SUMMARY inside_2sigma=" << covered << '/' << mocks
              << " (small prior-drawn ensemble; report binomial uncertainty)\n";
    // A no-wiggle likelihood has no beta information in white coordinates.
    spectrum.amplitude = 0;
    s.centering = 0;
    FieldLevelBAO flat(s, spectrum);
    std::mt19937_64 rng(941);
    const auto white = flat.white_noise(rng);
    flat.set_data(flat.make_mock(white, .97, rng));
    double u = flat.potential(white, .87);
    check_local(std::abs(flat.potential(white, 1.13) - u) < 1e-10,
                "no-wiggle negative control has flat beta likelihood");
    const double prior_sd = (s.beta_max - s.beta_min) / std::sqrt(12.);
    check_local(std::abs(1. - .97) < 3.5 * prior_sd, "prior-only control passes the historical weak recovery rule");
    check_local(!informative(prior_sd, s), "information criterion rejects the prior-only negative control");
}
using Key = std::array<int, 3>;
// Fixed physical initial modes, represented without random resampling at every resolution.
static Field plane_field(const FieldLevelBAO & m) {
    const int N = m.settings().nmesh;
    Field delta(m.ncells());
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                delta[(size_t(x) * N + y) * N + z] = .12 * std::cos(2 * M_PI * x / N) +
                                                     .09 * std::cos(2 * M_PI * z / N) +
                                                     .04 * std::sin(2 * M_PI * (x + y) / N);
    std::vector<double> inv(m.modes().size());
    for (size_t i = 0; i < inv.size(); ++i)
        inv[i] = 1 / m.effective_transfer(m.modes()[i], 1.);
    return m.filter(delta, inv);
}
static std::map<Key, Complex> modes(const FieldLevelBAO & m, const Field & x, bool direct) {
    const auto values = m.predict(x, 1.);
    const auto psi = m.displacement(x, 1.);
    std::map<Key, Complex> result;
    const int N = m.settings().nmesh;
    for (size_t i = 0; i < values.size(); ++i) {
        const auto & k = m.modes()[i];
        if (k.k > .041)
            continue;
        Complex d = values[i];
        if (direct) {
            d = 0;
            for (int a = 0; a < N; ++a)
                for (int b = 0; b < N; ++b)
                    for (int c = 0; c < N; ++c) {
                        const size_t p = (size_t(a) * N + b) * N + c;
                        const double phase =
                            -2 * M_PI *
                            (k.n[0] * (double(a) / N + psi[0][p]) + k.n[1] * (double(b) / N + psi[1][p]) +
                             k.n[2] * (double(c) / N + psi[2][p]));
                        d += Complex(std::cos(phase), std::sin(phase));
                    }
            d *= m.settings().bias / double(m.ncells());
        }
        result[{k.n[0], k.n[1], k.n[2]}] = d;
    }
    return result;
}
static double difference(const std::map<Key, Complex> & a, const std::map<Key, Complex> & b) {
    double err = 0, norm = 0;
    for (const auto & kv : a) {
        err += std::norm(kv.second - b.at(kv.first));
        norm += std::norm(b.at(kv.first));
    }
    return std::sqrt(err / norm);
}
static void adapter_parity() {
    Settings s;
    s.nmesh = 8;
    s.boxsize = 300;
    s.kmax = .06;
    s.kernel_width = 4;
    for (auto model : {Model::LPT1, Model::LPT}) {
        s.model = model;
        s.b2 = .2;
        s.bG2 = -.1;
        s.growth_rate = model == Model::LPT ? .5 : 0;
        FieldLevelBAO fl(s, BAOSpectrum::eisenstein_hu());
        std::mt19937_64 rng(818);
        const auto white = fl.white_noise(rng);
        fl.set_data(fl.make_mock(fl.white_noise(rng), 1.02, rng));
        FML::MCMC::HMCOptions opt;
        opt.warmup = 30;
        opt.samples = 20;
        opt.n_leapfrog = 4;
        opt.seed = 98;
        const auto public_run = sample(fl, white, 1., opt);
        const auto monitored = run(fl, white, 1., opt);
        bool same = public_run.last == monitored.last && public_run.potential == monitored.potential &&
                    public_run.diagonal == monitored.diagonal && public_run.step_size == monitored.step_size &&
                    public_run.divergences == monitored.divergences;
        for (size_t i = 0; i < public_run.recorded.size(); ++i)
            same &= public_run.recorded[i][0] == monitored.recorded[i][0];
        check_local(same, "field-summary recorder preserves the public BAO sampler trajectory exactly");
    }
}
static void convergence() {
    Settings s;
    s.nmesh = 8;
    s.boxsize = 300;
    s.kmax = .075;
    s.model = Model::LPT1;
    s.centering = 0;
    // Reference bypasses deposition/FFT, and uses four times as many particles per axis.
    auto refsettings = s;
    refsettings.nmesh = 32;
    FieldLevelBAO reference(refsettings, BAOSpectrum::eisenstein_hu());
    const auto ref = modes(reference, plane_field(reference), true);
    std::vector<double> error;
    for (int N : {8, 16}) {
        auto t = s;
        t.nmesh = N;
        t.kernel_width = 8;
        FieldLevelBAO m(t, BAOSpectrum::eisenstein_hu());
        const auto x = plane_field(m);
        const double e = difference(modes(m, x, false), ref);
        const double direct = difference(modes(m, x, false), modes(m, x, true));
        error.push_back(e);
        std::cout << "MESH n=" << N << " relative_to_direct_high_resolution=" << e << " deposition_error=" << direct
                  << '\n';
        check_local(direct < 2e-5, "smooth deposited modes agree with independent direct particle Fourier sum");
    }
    check_local(error[1] < 2e-6 && (error[0] < 1e-8 || error[1] < .5 * error[0]),
                "fixed physical field converges toward higher-resolution direct mock");
    auto a = s;
    a.nmesh = 16;
    a.kernel_width = 8;
    a.kmax = .055;
    auto b = a;
    b.kmax = .075;
    FieldLevelBAO low(a, BAOSpectrum::eisenstein_hu()), high(b, BAOSpectrum::eisenstein_hu());
    const double cutoff = difference(modes(low, plane_field(low), false), modes(high, plane_field(high), false));
    std::cout << "CUTOFF low_band_change=" << cutoff << '\n';
    check_local(cutoff < 1e-10, "adding empty initial high-k support leaves low-band prediction unchanged");
    // A resolved high-k pair produces a low-k beat. Omitting these initial modes
    // is a physical model mismatch, not a deposition or derivative error.
    auto richsettings = refsettings;
    richsettings.kmax = .115;
    FieldLevelBAO rich(richsettings, BAOSpectrum::eisenstein_hu());
    auto richwhite = plane_field(rich);
    Field added(rich.ncells());
    const int R = rich.settings().nmesh;
    for (size_t p = 0; p < added.size(); ++p) {
        const double phase = 2 * M_PI * double(p / (R * R)) / R;
        added[p] = .3 * std::cos(4 * phase) + .3 * std::cos(5 * phase);
    }
    std::vector<double> inv(rich.modes().size());
    for (size_t i = 0; i < inv.size(); ++i)
        inv[i] = 1 / rich.effective_transfer(rich.modes()[i], 1.);
    const auto extra = rich.filter(added, inv);
    for (size_t p = 0; p < richwhite.size(); ++p)
        richwhite[p] += extra[p];
    const auto richer = modes(rich, richwhite, true);
    const double mismatch = difference(ref, richer);
    std::cout << "MODEL_MISMATCH unresolved_high_k_relative_change=" << mismatch << '\n';
    check_local(mismatch > 100 * error.back() && mismatch > 1e-3,
                "higher-cutoff direct mock exposes missing-mode coupling above numerical error");
    for (int width : {4, 6, 8}) {
        auto t = a;
        t.kernel_width = width;
        FieldLevelBAO m(t, BAOSpectrum::eisenstein_hu());
        auto x = plane_field(m);
        const double e = difference(modes(m, x, false), ref);
        std::cout << "ASSIGNMENT width=" << width << " direct_high_resolution_error=" << e << '\n';
        check_local(e < (width == 4 ? 1e-2 :
                         width == 6 ? 1e-4 :
                                      2e-5),
                    "assignment-width convergence against independent higher-resolution mock");
    }
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    if (FML::NTasks != 1)
        return 1;
    std::cout << std::setprecision(14) << std::unitbuf;
    convergence();
    adapter_parity();
    if (argc > 1) {
        int samples = argc > 2 ? std::stoi(argv[2]) : 600, mocks = argc > 3 ? std::stoi(argv[3]) : 4;
        if (samples < 100 || mocks < 1)
            throw std::invalid_argument("Require >=100 samples and >=1 mock");
        const std::string model = argc > 4 ? argv[4] : "1lpt";
        if (model != "1lpt" && model != "2lpt")
            throw std::invalid_argument("Expected 1lpt or 2lpt");
        inference(argv[1], samples, mocks, model == "2lpt");
    }
    return FML::TESTS::summary("BAO validation");
}
