// Free-IC galaxy sampler comparison on the joint-pilot target: joint HMC over (field, sigma8 ratio, four
// biases), the same with (b_delta, sigma8 ratio) replaced by (b_delta * sigma8 ratio, sigma8 ratio), and
// HMC over (field, sigma8 ratio) with the biases marginalized analytically (exact conditional draws added).
// All three sample the same posterior: Gaussian bias priors with the means/widths of the pilot's uniform
// priors, uniform sigma8-ratio prior on [0.5, 1.5], fixed noise. Budget/starts/HMC options follow the pilot.
//
//   reparam_study run   OUT DATASET METHOD CHAIN    METHOD = joint | reparam | marginal
//   reparam_study analyze OUT                       diagnostics over all completed runs
#include <FML/FieldLevel/MarginalizedBias.h>
#include <FML/MCMC/Diagnostics.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace E = FML::FIELDLEVEL::EULERIAN;
namespace FS = std::filesystem;
using FML::FIELDLEVEL::Field;
using Grid = FML::GRID::FFTWGrid<3>;
using Model = FF::DistributedEulerianLPTModelT<Grid>;

static const std::array<FF::Parameter, 4> bias_priors{{{"b_delta", 2, 2, 4 / std::sqrt(12.)},
                                                       {"b_delta2", 0, 0, 8 / std::sqrt(12.)},
                                                       {"b_K2", 0, 0, 8 / std::sqrt(12.)},
                                                       {"b_laplacian", 0, 0, 200 / std::sqrt(12.)}}};
static const std::vector<std::vector<double>> starts = {
    {.8, 1.2, -.3, .2, -20}, {1.2, 2, .8, -.6, 30}, {.9, 2.4, -.8, .6, 50}, {1.1, .8, 1, -1, -40}};
static const int warmup = 256, samples = 1024;
static const std::vector<std::string> monitors = {"sigma8_ratio", "b1_sigma8", "b_delta", "b_delta2", "b_K2",
                                                  "b_laplacian",  "cos100",    "sin100",  "cos110",   "sin110",
                                                  "cos001",       "sin001"};

static Model make_model() {
    FF::EulerianLPTConfiguration c;
    c.mesh = 4;
    c.product_mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    const double kf = 2 * M_PI / c.boxsize;
    c.initial_cutoff = 1.7 * kf;
    c.matter_band = {1.8 * kf, E::BandGeometry::Cube};
    c.likelihood_band = {1.6 * kf, E::BandGeometry::Cube};
    c.matter_assignment = FF::MatterAssignment::PSWF;
    return Model(c, FF::sigma8_spectrum([](double k) { return 20000 * std::exp(-5 * k); }), bias_priors);
}
// Dataset 0: the joint pilot's truth and seeds. 1..3: truths drawn from the priors.
struct Dataset {
    std::vector<double> truth;
    std::uint64_t white_seed, noise_seed;
};
static Dataset dataset(int id) {
    if (id == 0)
        return {{1, 1.6, .3, -.2, 7}, 4101, 4102};
    FML::RANDOM::COUNTER_BASED::Stream rng(9100, std::uint64_t(id), 0x54525554u);
    std::vector<double> truth = {.5 + rng.uniform()};
    for (const auto & p : bias_priors)
        truth.push_back(p.mean + p.sigma * rng.normal());
    return {truth, 9100 + 10 * std::uint64_t(id), 9101 + 10 * std::uint64_t(id)};
}
static std::array<double, 6> projections(const Field & s) {
    std::array<double, 6> out{};
    const int waves[3][3] = {{1, 0, 0}, {1, 1, 0}, {0, 0, 1}};
    for (int q = 0; q < 64; ++q) {
        const int x = q / 16, y = q / 4 % 4, z = q % 4;
        for (int w = 0; w < 3; ++w) {
            const double phase = 2 * M_PI * (x * waves[w][0] + y * waves[w][1] + z * waves[w][2]) / 4;
            out[2 * w] += std::sqrt(2. / 64) * std::cos(phase) * s[q];
            out[2 * w + 1] += std::sqrt(2. / 64) * std::sin(phase) * s[q];
        }
    }
    return out;
}

static int run(const FS::path & out, int id, const std::string & method, int chain) {
    const auto model = make_model();
    auto w = model.workspace();
    const auto d = dataset(id);
    const double noise = 1000;
    const auto data = model.mock(model.white_noise(d.white_seed), d.truth, noise, d.noise_seed, w);
    const auto joint = FF::fixed_noise_posterior(model, data, noise);
    const FF::BiasMarginalizedPosterior<Grid> marginal(model, data, noise);
    FML::MCMC::HMCOptions o;
    o.warmup = warmup;
    o.samples = samples;
    o.n_leapfrog = 12;
    o.step_size = .05;
    o.jitter = true;
    o.adapt_step_size = true;
    o.target_acceptance = .9;
    o.adapt_mass = true;
    o.seed = 4110 + std::uint64_t(chain) + 100 * std::uint64_t(id);
    const auto white0 = model.white_noise(4103 + std::uint64_t(chain));
    const auto & start = starts[chain];
    const size_t n = model.cells();
    long gradients = 0;
    std::vector<std::array<double, 12>> rows;
    auto store = [&](const Field & s, double alpha, const std::array<double, 4> & b) {
        std::array<double, 12> r{};
        r[0] = alpha;
        r[1] = b[0] * alpha;
        for (int a = 0; a < 4; ++a)
            r[2 + a] = b[a];
        const auto proj = projections(s);
        std::copy(proj.begin(), proj.end(), r.begin() + 6);
        rows.push_back(r);
    };
    const auto t0 = std::chrono::steady_clock::now();
    FML::MCMC::HMCResult result;
    if (method == "marginal") {
        size_t draw = 0;
        auto sink = [&](const Field & s, const std::vector<double> & theta, size_t) {
            auto c = marginal.conditional_bias(s, theta, w);
            FML::RANDOM::COUNTER_BASED::Stream rng(o.seed, draw++, 0x42494153u);
            std::array<double, 4> z{}, b = c.mean;
            for (auto & v : z)
                v = rng.normal();
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j <= i; ++j)
                    b[i] += c.cholesky[i * 4 + j] * z[j];
            store(s, theta[0], b);
        };
        result = FF::sample_bias_marginalized(marginal, white0, {start[0]}, o, sink);
        gradients = long(marginal.evaluations());
    } else {
        const bool re = method == "reparam";
        if (!re && method != "joint")
            throw std::invalid_argument("METHOD must be joint, reparam or marginal");
        // Scalars: joint (alpha, b1, b2, bK2, blap); reparam (alpha, beta = b1 alpha, b2, bK2, blap).
        std::vector<double> x(white0);
        std::vector<double> y = start;
        if (re)
            y[1] = start[1] * start[0];
        x.insert(x.end(), y.begin(), y.end());
        Field field(n), gs;
        std::vector<double> p(5), gp;
        auto physical = [&](const std::vector<double> & state) {
            std::copy_n(state.begin(), n, field.begin());
            std::copy_n(state.begin() + n, 5, p.begin());
            if (re)
                p[1] = state[n + 1] / state[n];
        };
        auto potential = [&](const std::vector<double> & state, std::vector<double> & gradient) {
            ++gradients;
            physical(state);
            double U = joint.evaluate(field, p, w, gs, gp);
            std::copy(gs.begin(), gs.end(), gradient.begin());
            std::copy(gp.begin(), gp.end(), gradient.begin() + n);
            if (re && std::isfinite(U)) {
                const double alpha = state[n], beta = state[n + 1];
                // b1 = beta / alpha; density in (alpha, beta) is pi(alpha, b1) / alpha.
                gradient[n] = gp[0] - gp[1] * beta / (alpha * alpha) + 1 / alpha;
                gradient[n + 1] = gp[1] / alpha;
                U += std::log(alpha);
            }
            return U;
        };
        const auto frozen = p = start;
        FML::MCMC::HMCMass mass;
        mass.n_custom = n;
        mass.apply_sqrt = [&](auto & v) { v = joint.apply_field_metric(v, frozen, w, .5); };
        mass.apply_inverse = [&](auto & v) { v = joint.apply_field_metric(v, frozen, w, -1); };
        for (int j = 0; j < 5; ++j) {
            const auto & spec = model.parameters()[j];
            double scale = spec.scale();
            if (re && j == 1)
                scale *= start[0];
            mass.diagonal.push_back(1 / (scale * scale));
            o.lower.push_back(spec.lower);
            o.upper.push_back(spec.upper);
        }
        auto recorder = [&](const std::vector<double> & state) {
            physical(state);
            store(field, p[0], {p[1], p[2], p[3], p[4]});
            return std::vector<double>(state.begin() + n, state.end());
        };
        result = FML::MCMC::run_hmc(potential, x, mass, o, recorder, FF::Execution<Model>::hmc(model));
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    FS::create_directories(out);
    const auto stem = out / ("d" + std::to_string(id) + "-" + method + "-c" + std::to_string(chain));
    std::ofstream csv(stem.string() + ".csv");
    csv << std::setprecision(17) << "draw";
    for (const auto & m : monitors)
        csv << ',' << m;
    csv << '\n';
    for (size_t i = 0; i < rows.size(); ++i) {
        csv << i;
        for (double v : rows[i])
            csv << ',' << v;
        csv << '\n';
    }
    std::ofstream meta(stem.string() + ".json");
    meta << std::setprecision(17) << "{\"dataset\":" << id << ",\"method\":\"" << method << "\",\"chain\":" << chain
         << ",\"truth\":[" << d.truth[0];
    for (size_t j = 1; j < d.truth.size(); ++j)
        meta << ',' << d.truth[j];
    meta << "],\"potential_mean\":" << [&] {
        double m = 0;
        for (double u : result.potential)
            m += u;
        return result.potential.empty() ? 0. : m / double(result.potential.size());
    }() << ",\"potential_min\":" << (result.potential.empty() ? 0. : *std::min_element(result.potential.begin(), result.potential.end()))
         << ",\"samples\":" << result.completed_samples << ",\"acceptance\":" << result.acceptance
         << ",\"divergences\":" << result.divergences << ",\"step_size\":" << result.step_size
         << ",\"gradient_evaluations\":" << gradients << ",\"seconds\":" << seconds << "}\n";
    std::cout << stem.filename().string() << ": acceptance " << result.acceptance << ", divergences "
              << result.divergences << ", " << seconds << " s\n";
    return 0;
}

static std::vector<std::vector<double>> read_csv(const FS::path & file) {
    std::ifstream in(file);
    std::string line;
    std::getline(in, line);
    std::vector<std::vector<double>> columns(monitors.size());
    while (std::getline(in, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        double draw;
        row >> draw;
        for (auto & c : columns) {
            double v;
            row >> v;
            c.push_back(v);
        }
    }
    return columns;
}

static int analyze(const FS::path & out) {
    std::ofstream report(out / "diagnostics.csv");
    report << std::setprecision(8) << "dataset,method,coordinate,rhat,ess_bulk,ess_tail,mean,sd,q05,q50,q95\n";
    for (int id = 0; id < 4; ++id)
        for (const std::string method : {"joint", "reparam", "marginal"}) {
            std::vector<std::vector<std::vector<double>>> chains;
            for (int c = 0; c < 4; ++c) {
                const auto f = out / ("d" + std::to_string(id) + "-" + method + "-c" + std::to_string(c) + ".csv");
                if (FS::exists(f))
                    chains.push_back(read_csv(f));
            }
            if (chains.size() != 4)
                continue;
            for (size_t j = 0; j < monitors.size(); ++j) {
                std::vector<std::vector<double>> x;
                std::vector<double> pooled;
                for (auto & c : chains) {
                    x.push_back(c[j]);
                    pooled.insert(pooled.end(), c[j].begin(), c[j].end());
                }
                const auto d = FML::MCMC::diagnose_chains(x);
                std::sort(pooled.begin(), pooled.end());
                double mean = 0, var = 0;
                for (double v : pooled)
                    mean += v;
                mean /= pooled.size();
                for (double v : pooled)
                    var += (v - mean) * (v - mean);
                auto q = [&](double f) { return pooled[size_t(f * (pooled.size() - 1))]; };
                report << id << ',' << method << ',' << monitors[j] << ',' << d.rhat << ',' << d.ess_bulk << ','
                       << d.ess_tail << ',' << mean << ',' << std::sqrt(var / (pooled.size() - 1)) << ',' << q(.05)
                       << ',' << q(.5) << ',' << q(.95) << '\n';
            }
        }
    std::cout << "Wrote " << (out / "diagnostics.csv").string() << '\n';
    return 0;
}

int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (FML::NTasks != 1)
            throw std::invalid_argument("Serial study: one rank per chain");
        if (argc == 6 && std::string(argv[1]) == "run")
            return run(argv[2], std::stoi(argv[3]), argv[4], std::stoi(argv[5]));
        if (argc == 3 && std::string(argv[1]) == "analyze")
            return analyze(argv[2]);
        throw std::invalid_argument("Usage: reparam_study run OUT DATASET METHOD CHAIN | analyze OUT");
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
