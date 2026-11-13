// Bounded conditional survey-inference validation. All physics/nuisances fixed.
#include <FML/FieldLevel/SurveyCheckpoint.h>
#include <FML/tests/TestHelpers.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace CP = FF::CHECKPOINT;
namespace C = FML::MCMC::CHECKPOINT;
namespace FS = std::filesystem;
using FML::FIELDLEVEL::Complex;
using FML::FIELDLEVEL::Field;
using Key = std::array<int, 3>;
static void sum(Field & x) {
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, x.data(), int(x.size()), MPI_DOUBLE, MPI_SUM, FML::communicator);
#else
    (void)x;
#endif
}
struct RNG {
    using result_type = std::uint64_t;
    FML::RANDOM::COUNTER_BASED::Stream stream;
    RNG(std::uint64_t seed, std::uint64_t id, std::uint32_t purpose) : stream(seed, id, purpose) {}
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT64_MAX; }
    result_type operator()() { return stream.next64(); }
};
// Validation-only compact global coefficient table (13 independent complex modes
// for this fixed band). No global real field is gathered by the experiment.
struct Basis {
    int N, start;
    size_t cells;
    double L, P;
    std::vector<Key> keys;
    std::vector<Field> cosine, sine, probes;
    template <class Model>
    Basis(const Model & m, double power)
        : N(m.configuration().mesh), start(m.layout().local_start), cells(m.cells()), L(m.configuration().boxsize),
          P(power) {
        const double cut = std::max(m.configuration().initial_cutoff, m.configuration().likelihood_cutoff);
        for (int x = -N / 2 + 1; x < N / 2; ++x)
            for (int y = -N / 2 + 1; y < N / 2; ++y)
                for (int z = 0; z < N / 2; ++z) {
                    if (!(z > 0 || x > 0 || (x == 0 && y > 0)))
                        continue;
                    if (2 * M_PI / L * std::sqrt(double(x * x + y * y + z * z)) >= cut)
                        continue;
                    keys.push_back({x, y, z});
                }
        for (const auto & k : keys) {
            Field co(cells), si(cells);
            for (size_t q = 0; q < cells; ++q) {
                const double angle =
                    2 * M_PI * (k[0] * (start + int(q / (N * N))) + k[1] * int(q / N % N) + k[2] * int(q % N)) / N;
                co[q] = std::cos(angle);
                si[q] = std::sin(angle);
            }
            cosine.push_back(std::move(co));
            sine.push_back(std::move(si));
        }
    }
    void make_probes() {
        for (int i = 0; i < 16; ++i) {
            const size_t id = (19 + 37 * i) % (N * N * N);
            Field row(2 * keys.size());
            for (size_t k = 0; k < keys.size(); ++k) {
                const auto & v = keys[k];
                const double phase =
                    2 * M_PI * (v[0] * int(id / (N * N)) + v[1] * int(id / N % N) + v[2] * int(id % N)) / N;
                row[2 * k] = std::sqrt(2 * P / (L * L * L)) * std::cos(phase);
                row[2 * k + 1] = std::sqrt(2 * P / (L * L * L)) * std::sin(phase);
            }
            probes.push_back(std::move(row));
        }
    }
    Field project(const Field & white) const {
        Field result(2 * keys.size() + 1, 0);
        for (size_t k = 0; k < keys.size(); ++k)
            for (size_t q = 0; q < cells; ++q) {
                result[2 * k] += white[q] * cosine[k][q];
                result[2 * k + 1] += white[q] * sine[k][q];
            }
        for (double x : white)
            result.back() += x * x;
        sum(result);
        const double scale = std::sqrt(2. / (N * N * N));
        for (size_t k = 0; k < 2 * keys.size(); ++k)
            result[k] *= scale;
        result.back() /= double(N) * N * N;
        return result;
    }
    Field white(const Field & phi) const {
        Field result(cells, 0);
        const double scale = std::sqrt(2. / (N * N * N));
        for (size_t k = 0; k < keys.size(); ++k)
            for (size_t q = 0; q < cells; ++q)
                result[q] += scale * (phi[2 * k] * cosine[k][q] + phi[2 * k + 1] * sine[k][q]);
        return result;
    }
    Field initial(const Field & phi) const {
        Field field(cells, 0);
        const double scale = std::sqrt(2 * P / (L * L * L));
        for (size_t k = 0; k < keys.size(); ++k)
            for (size_t q = 0; q < cells; ++q)
                field[q] += scale * (phi[2 * k] * cosine[k][q] + phi[2 * k + 1] * sine[k][q]);
        return field;
    }
    template <class Model>
    Field coefficients(const Model & m, const FF::Modes & prediction) const {
        Field result(2 * keys.size(), 0);
        std::map<Key, size_t> lookup;
        for (size_t k = 0; k < keys.size(); ++k)
            lookup[keys[k]] = k;
        for (size_t i = 0; i < prediction.size(); ++i) {
            const auto & mode = m.modes()[i];
            Key key{mode.n[0], mode.n[1], mode.n[2]};
            auto found = lookup.find(key);
            if (found != lookup.end() && mode.k < m.configuration().likelihood_cutoff) {
                result[2 * found->second] = prediction[i].real();
                result[2 * found->second + 1] = prediction[i].imag();
            }
        }
        sum(result);
        return result;
    }
    Field synthesize(const Field & coefficients) const {
        Field field(cells, 0);
        for (size_t k = 0; k < keys.size(); ++k)
            for (size_t q = 0; q < cells; ++q)
                field[q] += 2 * (coefficients[2 * k] * cosine[k][q] - coefficients[2 * k + 1] * sine[k][q]);
        return field;
    }
};
// Validation-only dense reference metric. The active subspace is tiny here;
// this is not a dense-matrix strategy for production cosmological volumes.
struct FisherMetric {
    size_t n;
    Field lower;
    template <bool Poisson, class Model>
    FisherMetric(const Model & m,
                 const Basis & basis,
                 const Field & parameters,
                 const FF::SurveyWindow & window,
                 const Field & variance,
                 const Field & exposure,
                 std::integral_constant<bool, Poisson>)
        : n(2 * basis.keys.size()), lower(n * n, 0) {
        const double Vcell = m.volume() / m.layout().global_cells, scale = std::sqrt(2 * basis.P / m.volume());
        Field row(n), matrix(n * n, 0);
        for (size_t q = 0; q < m.cells(); ++q)
            if (window.observed[q]) {
                const double weight = Poisson ? exposure[q] * window.selection[q] :
                                                window.selection[q] * window.selection[q] /
                                                    (std::exp(parameters.back()) / Vcell * variance[q]);
                for (size_t k = 0; k < basis.keys.size(); ++k) {
                    const auto & key = basis.keys[k];
                    const double k2 = double(key[0] * key[0] + key[1] * key[1] + key[2] * key[2]);
                    const double factor = parameters[0] + m.configuration().growth_rate * key[2] * key[2] / k2;
                    row[2 * k] = scale * factor * basis.cosine[k][q];
                    row[2 * k + 1] = scale * factor * basis.sine[k][q];
                }
                for (size_t i = 0; i < n; ++i)
                    for (size_t j = 0; j < n; ++j)
                        matrix[i * n + j] += weight * row[i] * row[j];
            }
        sum(matrix);
        for (size_t i = 0; i < n; ++i)
            matrix[i * n + i] += 1;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j <= i; ++j) {
                double value = matrix[i * n + j];
                for (size_t k = 0; k < j; ++k)
                    value -= lower[i * n + k] * lower[j * n + k];
                if (i == j) {
                    if (!(value > 0))
                        throw std::runtime_error("Non-SPD reference metric");
                    lower[i * n + j] = std::sqrt(value);
                } else
                    lower[i * n + j] = value / lower[j * n + j];
            }
        double error = 0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                double value = 0;
                for (size_t k = 0; k < n; ++k)
                    value += lower[i * n + k] * lower[j * n + k];
                error = std::max(error, std::abs(value - matrix[i * n + j]));
            }
        if (error > 1e-10)
            throw std::runtime_error("Reference metric factorization failed");
    }
    void draw(Field & x) const {
        Field out(n, 0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j <= i; ++j)
                out[i] += lower[i * n + j] * x[j];
        x = std::move(out);
    }
    void inverse(Field & x) const {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < i; ++j)
                x[i] -= lower[i * n + j] * x[j];
            x[i] /= lower[i * n + i];
        }
        for (size_t i = n; i-- > 0;) {
            for (size_t j = i + 1; j < n; ++j)
                x[i] -= lower[j * n + i] * x[j];
            x[i] /= lower[i * n + i];
        }
    }
};
static FF::Configuration configuration(int N = 8, int width = 6, FF::Evolution evolution = FF::Evolution::LPT2) {
    FF::Configuration c;
    c.mesh = N;
    c.density_mesh = 2 * N;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .10;
    c.likelihood_cutoff = .11;
    c.growth_rate = .6;
    c.kernel_width = width;
    c.evolution = evolution;
    return c;
}
static FF::SurveyVoxel selection(const std::array<double, 3> & x) {
    const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    return {r2 < 38 * 38 && x[2] > -25, .85 * std::exp(-r2 / (2 * 35 * 35))};
}
template <class Model>
static Field model_parameters(const Model & m, double noise) {
    auto p = m.initial_parameters();
    for (size_t j = 0; j < p.size(); ++j) {
        const auto & name = m.parameters()[j].name;
        if (name == "b1")
            p[j] = 1.4;
        else if (name == "b2")
            p[j] = .3;
        else if (name == "b_s2")
            p[j] = -.2;
        else if (name == "b_laplacian")
            p[j] = 0;
    }
    p.back() = std::log(noise);
    return p;
}
static void write_bytes(std::ofstream & stream, const C::Encoder & e) {
    stream.write(reinterpret_cast<const char *>(e.bytes.data()), std::streamsize(e.bytes.size()));
    if (!stream)
        throw std::runtime_error("Snapshot write failed");
}
template <bool Poisson, class Model>
static bool
inference(const Model & m, const FS::path & directory, int mock, int nchain, int minimum_samples, int warmup) {
    const int N = m.configuration().mesh;
    const double V = m.volume(), Vcell = V / m.layout().global_cells, power = 2000, noise_actual = 50, nbar = .015;
    const int thin = 8, cap = 2400;
    Basis basis(m, power);
    basis.make_probes();
    auto window = FF::make_survey_window(m, {-50, -50, -50}, selection);
    Field variance(m.cells()), exposure(m.cells(), nbar * Vcell), radial(m.cells());
    double information = 0, radial_mean = 0, npixels = 0;
    for (size_t q = 0; q < m.cells(); ++q) {
        const double x = -50 + 100. * (basis.start + q / (N * N)) / N, y = -50 + 100. * (q / N % N) / N,
                     z = -50 + 100. * (q % N) / N;
        const double hetero = 1 + .4 * x * x / (50 * 50);
        variance[q] = hetero;
        radial[q] = (x * x + y * y + z * z) / (38 * 38);
        if (window.observed[q]) {
            information +=
                Poisson ? exposure[q] * window.selection[q] : window.selection[q] * window.selection[q] / hetero;
            radial_mean += radial[q];
            ++npixels;
        }
    }
    const double mean_information = m.sum(information) / m.layout().global_cells;
    const double noise_reference = Poisson ? Vcell / mean_information : noise_actual / mean_information;
    // A reference-noise convention gives a mask-averaged diagonal metric. Actual
    // Gaussian pixel variances remain noise_actual/Vcell*hetero, identically.
    if (!Poisson)
        for (auto & v : variance)
            v *= noise_actual / noise_reference;
    const auto p = model_parameters(m, noise_reference);
    auto work = m.workspace();
    const auto truth = m.white_noise(1000 + mock), truth_phi = basis.project(truth),
               initial_truth = basis.initial(truth_phi);
    const auto truth_modes = basis.coefficients(m, m.predict(truth, p, work)),
               tracer_truth = basis.synthesize(truth_modes);
    const auto original = m.reconstruct(truth, p, work, 1.).initial_density;
    double error = 0;
    for (size_t q = 0; q < m.cells(); ++q)
        error = std::max(error, std::abs(original[q] - initial_truth[q]));
    if (m.maximum(error) > 1e-10)
        throw std::runtime_error("Independent initial-density basis normalization disagrees");
    Field data(m.cells()), holdout(m.cells()), mean_truth(m.cells());
    std::vector<std::uint64_t> counts(m.cells());
    for (size_t q = 0; q < m.cells(); ++q) {
        const size_t id = size_t(basis.start) * N * N + q;
        RNG rng(2000 + 101 * mock + (Poisson ? 100000 : 0), id, 0x44415441u);
        mean_truth[q] = Poisson ? exposure[q] * window.selection[q] * std::exp(tracer_truth[q]) :
                                  window.selection[q] * tracer_truth[q];
        if (!window.observed[q]) {
            mean_truth[q] = 0;
            data[q] = 0;
            holdout[q] = 0;
            counts[q] = 0;
            continue;
        }
        RNG future(3000 + 101 * mock + (Poisson ? 100000 : 0), id, 0x484f4c44u);
        if constexpr (Poisson) {
            counts[q] = std::poisson_distribution<std::uint64_t>(mean_truth[q])(rng);
            data[q] = double(counts[q]);
            holdout[q] = double(std::poisson_distribution<std::uint64_t>(mean_truth[q])(future));
        } else {
            const double sigma = std::sqrt(std::exp(p.back()) / Vcell * variance[q]);
            data[q] = mean_truth[q] + sigma * rng.stream.normal();
            holdout[q] = mean_truth[q] + sigma * future.stream.normal();
        }
    }
    auto make_likelihood = [&]() {
        if constexpr (Poisson)
            return FF::SurveyPoissonLikelihood(m, window, counts, exposure);
        else
            return FF::SurveyGaussianLikelihood(m, window, data, variance);
    };
    auto likelihood = make_likelihood();
    FF::BasicPosterior<Model, decltype(likelihood)> posterior(m, std::move(likelihood));
    FML::COMMUNICATION::collective_try(
        [&] {
            if (FML::ThisTask == 0) {
                FS::create_directories(directory);
                FS::remove(directory / "complete.txt");
            }
        },
        "Validation directory");
    const std::string suffix = ".rank" + std::to_string(FML::ThisTask);
    std::ofstream snapshots((directory / ("snapshots" + suffix + ".bin")).string(), std::ios::binary);
    C::Encoder header;
    header.text("FML survey validation snapshots 1");
    header.integer(N);
    header.integer(basis.start);
    header.integer(m.layout().local_width);
    header.integer(thin);
    write_bytes(snapshots, header);
    std::ofstream cells((directory / ("cells" + suffix + ".tsv")).string());
    cells << std::setprecision(17)
          << "# id mask selection variance exposure data holdout initial_truth tracer_truth mean_truth\n";
    for (size_t q = 0; q < m.cells(); ++q)
        cells << size_t(basis.start) * N * N + q << ' ' << int(window.observed[q]) << ' ' << window.selection[q] << ' '
              << (Poisson ? 0 : std::exp(p.back()) / Vcell * variance[q]) << ' ' << exposure[q] << ' ' << data[q] << ' '
              << holdout[q] << ' ' << initial_truth[q] << ' ' << tracer_truth[q] << ' ' << mean_truth[q] << '\n';
    cells.close();
    if (!cells || !snapshots)
        throw std::runtime_error("Cannot create validation data");
    const double observed = m.sum(npixels);
    radial_mean = m.sum(radial_mean) / observed;
    double radial_norm = 0;
    for (size_t q = 0; q < m.cells(); ++q) {
        radial[q] -= radial_mean;
        if (window.observed[q])
            radial_norm += radial[q] * radial[q];
    }
    radial_norm = std::sqrt(m.sum(radial_norm));
    std::ofstream predictive;
    if (FML::ThisTask == 0) {
        predictive.open(directory / "predictive.tsv");
        predictive << std::setprecision(17)
                   << "# chain draw observed_D replicated_D observed_total replicated_total observed_radial "
                      "replicated_radial observed_tail replicated_tail holdout_D holdout_total holdout_radial "
                      "holdout_tail expected_total\n";
    }
    std::vector<FML::MCMC::HMCResult> diagnostic(nchain), predictive_diagnostic(nchain);
#ifdef ACTIVE_MODE_REFERENCE
    FisherMetric metric(m, basis, p, window, variance, exposure, std::integral_constant<bool, Poisson>{});
    std::vector<FML::MCMC::HMCState> states(nchain);
#endif
    std::vector<FF::CheckpointOptions> io(nchain);
    std::vector<int> draws(nchain, 0);
    FML::MCMC::HMCOptions options;
    options.warmup = warmup;
    options.samples = cap;
#ifdef ACTIVE_MODE_REFERENCE
    options.n_leapfrog = 6;
#else
    options.n_leapfrog = 24;
#endif
    options.target_acceptance = .9;
    bool converged = false, divergence_free = true;
    int round = 0;
    do {
        for (int chain = 0; chain < nchain; ++chain) {
            options.seed = 5000 + 101 * mock + chain;
            auto & checkpoint = io[chain];
            checkpoint.directory = (directory / ("checkpoint-" + std::to_string(chain))).string();
            checkpoint.problem_id = "survey-validation-v1:P2000:fixed:" + std::string(Poisson ? "poisson" : "gaussian");
            checkpoint.max_iterations = round == 0 ? warmup + minimum_samples : std::min(800, cap - draws[chain]);
            const size_t old = diagnostic[chain].recorded.size();
            auto sink = [&](const Field & white, const auto &, size_t draw) {
                auto phi = basis.project(white);
                auto row = phi;
                for (const auto & weights : basis.probes) {
                    double value = 0;
                    for (size_t k = 0; k < weights.size(); ++k)
                        value += weights[k] * phi[k];
                    row.push_back(value);
                }
                diagnostic[chain].recorded.push_back(std::move(row));
                if (draw % thin != 0)
                    return;
                const auto coefficients = basis.coefficients(m, m.predict(white, p, work));
                const auto tracer = basis.synthesize(coefficients);
                Field expected(m.cells());
                Field statistics(13, 0);
                for (size_t q = 0; q < m.cells(); ++q) {
                    if (!window.observed[q]) {
                        expected[q] = 0;
                        continue;
                    }
                    expected[q] = Poisson ? exposure[q] * window.selection[q] * std::exp(tracer[q]) :
                                            window.selection[q] * tracer[q];
                    const size_t id = size_t(basis.start) * N * N + q;
                    RNG rng(9000000 + (Poisson ? 1000000000ULL : 0ULL) + mock * 100000 + chain * 10000 + draw,
                            id,
                            0x50524544u);
                    double replicated, ro, rr, rh;
                    if constexpr (Poisson) {
                        replicated = double(std::poisson_distribution<std::uint64_t>(expected[q])(rng));
                        const double lr = std::log(exposure[q]) + std::log(window.selection[q]) + tracer[q];
                        auto residual = [&](double n) {
                            const double dev =
                                std::max(0., 2 * (expected[q] - n + (n > 0 ? n * (std::log(n) - lr) : 0.)));
                            return std::copysign(std::sqrt(dev), n - expected[q]);
                        };
                        ro = residual(data[q]);
                        rr = residual(replicated);
                        rh = residual(holdout[q]);
                    } else {
                        const double sigma = std::sqrt(std::exp(p.back()) / Vcell * variance[q]);
                        replicated = expected[q] + sigma * rng.stream.normal();
                        ro = (data[q] - expected[q]) / sigma;
                        rr = (replicated - expected[q]) / sigma;
                        rh = (holdout[q] - expected[q]) / sigma;
                    }
                    statistics[0] += ro * ro;
                    statistics[1] += rr * rr;
                    statistics[2] += data[q];
                    statistics[3] += replicated;
                    statistics[4] += ro * radial[q];
                    statistics[5] += rr * radial[q];
                    statistics[6] += std::abs(ro) > 2;
                    statistics[7] += std::abs(rr) > 2;
                    statistics[8] += rh * rh;
                    statistics[9] += holdout[q];
                    statistics[10] += rh * radial[q];
                    statistics[11] += std::abs(rh) > 2;
                    statistics[12] += expected[q];
                }
                sum(statistics);
                for (size_t j : {0, 1, 6, 7, 8, 11})
                    statistics[j] /= observed;
                for (size_t j : {4, 5, 10})
                    statistics[j] /= radial_norm;
                predictive_diagnostic[chain].recorded.push_back({statistics[0], statistics[12], statistics[4]});
                if (FML::ThisTask == 0) {
                    predictive << chain << ' ' << draw;
                    for (double x : statistics)
                        predictive << ' ' << x;
                    predictive << '\n';
                }
                C::Encoder record;
                record.integer(chain);
                record.integer(draw);
                record.vector(expected);
                write_bytes(snapshots, record);
            };
            auto start = m.white_noise(6000 + 101 * mock + chain);
            for (size_t q = 0; q < start.size(); ++q)
                start[q] *= chain % 2 ? 2. : .5;
#ifdef ACTIVE_MODE_REFERENCE
            auto initial_phi = basis.project(start);
            initial_phi.resize(metric.n);
            auto state_potential = [&](const Field & phi, Field & gradient) {
                const auto white = basis.white(phi);
                Field full_gradient;
                std::vector<double> gp;
                const double U = posterior.evaluate(white, p, work, full_gradient, gp);
                gradient = basis.project(full_gradient);
                gradient.resize(metric.n);
                return U;
            };
            FML::MCMC::HMCMass mass;
            mass.n_custom = metric.n;
            mass.apply_sqrt = [&](Field & x) { metric.draw(x); };
            mass.apply_inverse = [&](Field & x) { metric.inverse(x); };
            FML::MCMC::HMCRunControl control;
            control.state = &states[chain];
            control.max_iterations = checkpoint.max_iterations;
            const size_t draw_offset = draws[chain];
            size_t local_draw = 0;
            auto recorder = [&](const Field & phi) {
                const auto white = basis.white(phi);
                sink(white, p, draw_offset + local_draw++);
                return phi;
            };
            // Replicated active coordinates: U/gradient are globally reduced by
            // the posterior/projector; kinetic energy is counted exactly once.
            const auto result = FML::MCMC::run_hmc(state_potential,
                                                   checkpoint.resume ? states[chain].position : initial_phi,
                                                   std::move(mass),
                                                   options,
                                                   recorder,
                                                   {},
                                                   control);
            FML::COMMUNICATION::collective_try(
                [&] {
                    if (FML::ThisTask == 0)
                        C::save((directory / ("reference-state-" + std::to_string(chain) + ".bin")).string(),
                                states[chain]);
                },
                "Reference state snapshot");
#else
            const auto result = checkpoint.resume ?
                                    FF::sample_checkpointed(posterior, {}, {}, options, checkpoint, false, sink) :
                                    FF::sample_checkpointed(posterior, start, p, options, checkpoint, false, sink);
#endif
            for (size_t j = 0; j < result.potential.size(); ++j)
                diagnostic[chain].recorded[old + j].push_back(result.potential[j]);
            diagnostic[chain].step_size = result.step_size;
            diagnostic[chain].acceptance = result.acceptance;
            diagnostic[chain].divergences = result.divergences;
            draws[chain] = result.completed_samples;
            checkpoint.resume = true;
            divergence_free &= result.divergences == 0;
        }
        converged = true;
        double max_rhat = 0, min_bulk = HUGE_VAL, min_tail = HUGE_VAL;
        for (size_t j = 0; j < diagnostic[0].recorded[0].size(); ++j) {
            const auto d = FML::MCMC::diagnose_hmc_chains(diagnostic, j);
            converged &= d.valid() && d.rhat < 1.02 && d.ess_bulk > 150 && d.ess_tail > 100;
            max_rhat = std::max(max_rhat, d.rhat);
            min_bulk = std::min(min_bulk, d.ess_bulk);
            min_tail = std::min(min_tail, d.ess_tail);
        }
        for (size_t j = 0; j < 3; ++j) {
            const auto d = FML::MCMC::diagnose_hmc_chains(predictive_diagnostic, j);
            converged &= d.valid() && d.rhat < 1.05 && d.ess_bulk > 50 && d.ess_tail > 40;
        }
        if (FML::ThisTask == 0)
            std::cout << directory.filename() << " round=" << round << " draws=" << draws[0] << " Rhat=" << max_rhat
                      << " bulk=" << min_bulk << " tail=" << min_tail << " divergences=" << (!divergence_free) << '\n';
        for (const auto & chain : diagnostic)
            converged &= chain.acceptance > .65;
        ++round;
    } while (!converged && draws[0] < cap && divergence_free);
    snapshots.close();
    if (FML::ThisTask == 0)
        predictive.close();
    if (FML::ThisTask == 0) {
        std::ofstream trace(directory / "trace.tsv");
        trace << std::setprecision(17)
              << "# chain draw phi[cos,sin for each key] white_power initial_probes[16] potential\n";
        std::ofstream stats(directory / "diagnostics.tsv");
        stats << std::setprecision(17) << "# column rhat bulk tail\n";
        for (int chain = 0; chain < nchain; ++chain)
            for (size_t draw = 0; draw < diagnostic[chain].recorded.size(); ++draw) {
                trace << chain << ' ' << draw;
                for (double value : diagnostic[chain].recorded[draw])
                    trace << ' ' << value;
                trace << '\n';
            }
        for (size_t j = 0; j < diagnostic[0].recorded[0].size(); ++j) {
            const auto d = FML::MCMC::diagnose_hmc_chains(diagnostic, j);
            stats << j << ' ' << d.rhat << ' ' << d.ess_bulk << ' ' << d.ess_tail << '\n';
        }
        std::ofstream ppstats(directory / "predictive_diagnostics.tsv");
        ppstats << std::setprecision(17) << "# column rhat bulk tail\n";
        for (size_t j = 0; j < 3; ++j) {
            const auto d = FML::MCMC::diagnose_hmc_chains(predictive_diagnostic, j);
            ppstats << j << ' ' << d.rhat << ' ' << d.ess_bulk << ' ' << d.ess_tail << '\n';
        }
#ifdef ACTIVE_MODE_REFERENCE
        constexpr bool collapsed = true;
#else
        constexpr bool collapsed = false;
#endif
        std::ofstream meta(directory / "meta.tsv");
        meta << std::setprecision(17) << "mesh " << N << "\nboxsize " << basis.L << "\npower " << power << "\nmock "
             << mock << "\ncollapsed " << collapsed << "\nactive_coordinates " << 2 * basis.keys.size() << "\nchains "
             << nchain << "\ndraws " << draws[0] << "\nwarmup " << warmup << "\nthin " << thin << "\npoisson "
             << Poisson << "\nnoise_reference " << noise_reference << "\nnoise_actual " << (Poisson ? 0 : noise_actual)
             << "\nnbar " << nbar << "\nobserved_voxels " << observed << "\nconverged " << converged
             << "\ndivergence_free " << divergence_free << '\n';
        std::ofstream modes(directory / "modes.tsv");
        modes << std::setprecision(17) << "# nx ny nz truth_phi_cos truth_phi_sin\n";
        for (size_t k = 0; k < basis.keys.size(); ++k)
            modes << basis.keys[k][0] << ' ' << basis.keys[k][1] << ' ' << basis.keys[k][2] << ' ' << truth_phi[2 * k]
                  << ' ' << truth_phi[2 * k + 1] << '\n';
        std::ofstream chainstats(directory / "chains.tsv");
        chainstats << std::setprecision(17) << "# chain acceptance divergences step_size\n";
        for (int i = 0; i < nchain; ++i)
            chainstats << i << ' ' << diagnostic[i].acceptance << ' ' << diagnostic[i].divergences << ' '
                       << diagnostic[i].step_size << '\n';
        std::ofstream probes(directory / "probes.tsv");
        probes << "# id\n";
        for (int i = 0; i < 16; ++i)
            probes << (19 + 37 * i) % (N * N * N) << '\n';
        trace.close();
        stats.close();
        ppstats.close();
        meta.close();
        modes.close();
        chainstats.close();
        probes.close();
        if (!trace || !stats || !ppstats || !meta || !modes || !chainstats || !probes || !predictive || !snapshots)
            throw std::runtime_error("Incomplete experiment output");
        std::ofstream done(directory / "complete.txt");
        done << (converged && divergence_free ? "PASS" : "FAIL") << '\n';
    }
    return converged && divergence_free;
}
static int convergence(const FS::path & root) {
    if (FML::NTasks != 1)
        throw std::invalid_argument("Convergence fixture runs on one rank");
    FS::create_directories(root);
    std::ofstream table(root / "convergence.tsv");
    table << std::setprecision(17) << "# mesh width subdivisions volume selection_volume expected_counts\n";
    for (int N : {8, 12, 16, 24, 32, 48})
        for (int width : {4, 6, 8}) {
            auto c = configuration(N, width);
            FF::DistributedLPTModel m(c, FF::Spectrum::fixed([](double) { return 2000.; }));
            Basis basis(m, 2000);
            auto p = model_parameters(m, 500);
            auto w = m.workspace();
            Field white(m.cells());
            const double transfer = std::sqrt(2000. * m.layout().global_cells / m.volume());
            for (size_t q = 0; q < white.size(); ++q) {
                const double x = 2 * M_PI * double(q / (N * N)) / N, y = 2 * M_PI * double(q / N % N) / N,
                             z = 2 * M_PI * double(q % N) / N;
                white[q] = (.15 * std::cos(x) + .11 * std::cos(z) + .06 * std::sin(x + y)) / transfer;
            }
            const auto coeff = basis.coefficients(m, m.predict(white, p, w)), tracer = basis.synthesize(coeff);
            std::ofstream modes(root / ("coefficients-N" + std::to_string(N) + "-W" + std::to_string(width) + ".tsv"));
            modes << std::setprecision(17) << "# nx ny nz real imag\n";
            for (size_t k = 0; k < basis.keys.size(); ++k)
                modes << basis.keys[k][0] << ' ' << basis.keys[k][1] << ' ' << basis.keys[k][2] << ' ' << coeff[2 * k]
                      << ' ' << coeff[2 * k + 1] << '\n';
            for (int sub : {1, 2, 4, 8}) {
                double volume = 0, selected = 0, expected = 0;
                const double dx = 100. / N, dV = dx * dx * dx;
                for (size_t q = 0; q < white.size(); ++q) {
                    const double x = -50 + dx * (q / (N * N)), y = -50 + dx * (q / N % N), z = -50 + dx * (q % N);
                    double fraction = 0, response = 0;
                    for (int i = 0; i < sub; ++i)
                        for (int j = 0; j < sub; ++j)
                            for (int k = 0; k < sub; ++k) {
                                const auto cell = selection({x + dx * ((i + .5) / sub - .5),
                                                             y + dx * ((j + .5) / sub - .5),
                                                             z + dx * ((k + .5) / sub - .5)});
                                if (cell.observed) {
                                    fraction += 1;
                                    response += cell.selection;
                                }
                            }
                    const double norm = double(sub) * sub * sub;
                    volume += dV * fraction / norm;
                    selected += dV * response / norm;
                    expected += .015 * dV * response / norm * std::exp(tracer[q]);
                }
                table << N << ' ' << width << ' ' << sub << ' ' << volume << ' ' << selected << ' ' << expected << '\n';
            }
            std::cout << "Convergence N=" << N << " width=" << width << " complete\n";
        }
    return 0;
}
template <bool Poisson, class Model>
static void check_reference(const Model & m) {
    Basis basis(m, 2000);
    const auto p = model_parameters(m, 500);
    auto w = m.workspace();
    const auto full = m.white_noise(71), phi = basis.project(full), reduced = basis.white(phi);
    const auto a = m.predict(full, p, w);
    const auto b = m.predict(reduced, p, w);
    double error = 0;
    for (size_t i = 0; i < a.size(); ++i)
        error = std::max(error, std::abs(a[i] - b[i]));
    FML::TESTS::check_local(m.maximum(error) < 1e-11, "active-mode reduction preserves nonlinear/linear predictions");
    auto window = FF::make_survey_window(m, {-50, -50, -50}, selection);
    Field var(m.cells(), .1), exposure(m.cells(), 20), data(m.cells());
    std::vector<std::uint64_t> counts(m.cells(), 2);
    auto make = [&]() {
        if constexpr (Poisson)
            return FF::SurveyPoissonLikelihood(m, window, counts, exposure);
        else
            return FF::SurveyGaussianLikelihood(m, window, data, var);
    };
    auto like = make();
    FF::BasicPosterior<Model, decltype(like)> post(m, std::move(like));
    Field gs;
    std::vector<double> gp;
    const double U = post.evaluate(full, p, w, gs, gp), V = post.evaluate(reduced, p, w, gs, gp);
    double prior = 0;
    for (size_t q = 0; q < full.size(); ++q)
        prior += .5 * (full[q] * full[q] - reduced[q] * reduced[q]);
    FML::TESTS::check_local(std::abs(U - V - m.sum(prior)) < 1e-8,
                            "omitted coordinates contribute only the independent Gaussian prior");
    const auto gradient = basis.project(gs);
    bool good = true;
    for (size_t j : {size_t(0), 2 * basis.keys.size() - 1}) {
        auto x = phi, y = phi;
        x[j] += 1e-5;
        y[j] -= 1e-5;
        const double numeric =
            (post.evaluate(basis.white(x), p, w, gs, gp) - post.evaluate(basis.white(y), p, w, gs, gp)) / 2e-5;
        good &= std::abs(numeric - gradient[j]) < 2e-6 * std::max(1., std::abs(numeric));
    }
    FML::TESTS::check_local(good, "active-coordinate gradient includes the complete production posterior chain rule");
    FisherMetric metric(m, basis, p, window, var, exposure, std::integral_constant<bool, Poisson>{});
    Field v(metric.n);
    double norm = 0;
    for (size_t j = 0; j < v.size(); ++j) {
        v[j] = std::sin(.37 * (j + 1));
        norm += v[j] * v[j];
    }
    metric.draw(v);
    auto inverse = v;
    metric.inverse(inverse);
    double kinetic = 0;
    for (size_t j = 0; j < v.size(); ++j)
        kinetic += v[j] * inverse[j];
    FML::TESTS::check_local(std::abs(kinetic - norm) < 1e-10,
                            "dense reference momentum factor and inverse define the same kinetic energy");
}
static int reference_checks() {
    auto c = configuration();
    auto spectrum = FF::Spectrum::fixed([](double) {  return 2000.; });
    FF::DistributedLPTModel nonlinear(c, spectrum);
    check_reference<false>(nonlinear);
    check_reference<true>(nonlinear);
    c.evolution = FF::Evolution::Linear;
    FF::DistributedLinearModel linear(c, spectrum);
    check_reference<false>(linear);
    check_reference<true>(linear);
    return FML::TESTS::summary("survey active-mode reference");
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    std::cout << std::setprecision(10) << std::unitbuf;
    try {
        if (argc == 2 && std::string(argv[1]) == "--check-reference")
            return reference_checks();
        if (argc == 3 && std::string(argv[1]) == "--convergence")
            return convergence(argv[2]);
        if (argc < 8 || argc > 9)
            throw std::invalid_argument("Usage: survey_validate OUTPUT gaussian|poisson FIRST_MOCK NMOCKS NCHAINS "
                                        "SAMPLES WARMUP [2lpt|linear]; or --convergence OUTPUT");
        const FS::path root = argv[1];
        const std::string kind = argv[2], evolution = argc == 9 ? argv[8] : "2lpt";
        int first = std::stoi(argv[3]), mocks = std::stoi(argv[4]), chains = std::stoi(argv[5]),
            samples = std::stoi(argv[6]), warmup = std::stoi(argv[7]);
        if ((kind != "gaussian" && kind != "poisson") || (evolution != "2lpt" && evolution != "linear") || first < 0 ||
            mocks < 1 || mocks > 64 || chains < 2 || chains > 8 || samples < 100 || samples > 2400 || warmup < 20)
            throw std::invalid_argument("Invalid experiment controls");
        auto c = configuration(8, 6, evolution == "linear" ? FF::Evolution::Linear : FF::Evolution::LPT2);
        auto spectrum = FF::Spectrum::fixed([](double) { return 2000.; });
        bool good = true;
        auto execute = [&](const auto & model) {
            for (int i = first; i < first + mocks; ++i) {
                const FS::path dir = root / (evolution + "_" + kind) / ("mock" + std::to_string(i));
                const bool ok = kind == "gaussian" ? inference<false>(model, dir, i, chains, samples, warmup) :
                                                     inference<true>(model, dir, i, chains, samples, warmup);
                good &= ok;
            }
        };
        if (evolution == "linear") {
            FF::DistributedLinearModel model(c, spectrum);
            execute(model);
        } else {
            FF::DistributedLPTModel model(c, spectrum);
            execute(model);
        }
        return good ? 0 : 1;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
#ifdef USE_MPI
        MPI_Abort(FML::communicator, 1);
#endif
        return 1;
    }
}
