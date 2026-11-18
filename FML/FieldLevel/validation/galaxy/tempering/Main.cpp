#include "Exchange.h"
#include "MockSettings.h"
#include "PriorPredictive.h"
#include <FML/FieldLevel/EulerianCheckpoint.h>
#include <FML/FieldLevel/FixedIC.h>
#include <FML/FieldLevel/validation/galaxy/interweave/Rescaling.h>
#include <fstream>
#include <iomanip>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace CP = FF::CHECKPOINT;
namespace FS = std::filesystem;
using FML::FIELDLEVEL::Field;
using Model = FF::DistributedEulerianLPTModel;
using Posterior = FF::BasicPosterior<Model>;
const std::vector<double> legacy_betas = {1, .5, .25, .125, .0625, 0};
std::vector<double> betas = legacy_betas;
unsigned exchange_passes = 1;
MOCK::Seeds mock;
struct PassTrace {
    unsigned parity;
    std::vector<EXCHANGE::Attempt> events;
    std::vector<size_t> labels;
};
int Warmup = 64, Samples = 256, Total = 320, Ensembles = 2, SeedLayout = 1;
static uint64_t sweep_seed(int ensemble, size_t replica, int sweep, unsigned domain = 0) {
    if (SeedLayout == 1)
        return domain == 3 ? 7300000 + ensemble * 1000000 + sweep :
                             2300000 + ensemble * 1000000 + replica * 10000 + sweep * 10 + domain;
    if (SeedLayout == 3)
        return MOCK::stream(mock.id, unsigned(ensemble), unsigned(replica), unsigned(sweep), domain);
    // Injective tuple encoding: 2 ensemble, 6 replica, 20 sweep and 4 purpose bits.
    return UINT64_C(0x5200000000000000) | (uint64_t(ensemble) << 30) | (uint64_t(replica) << 24) |
           (uint64_t(sweep) << 4) | domain;
}
struct Stats {
    double accept_sum = 0;
    uint64_t divergences = 0, scalar_accepted = 0, scalar_attempts = 0, support_rejections = 0, rescale_accepted = 0,
             rescale_attempts = 0, cache_builds = 0, prior_refreshes = 0;
};
struct Ensemble {
    int next = 0;
    std::vector<EXCHANGE::PhysicalState> states = std::vector<EXCHANGE::PhysicalState>(betas.size());
    std::vector<Stats> stats = std::vector<Stats>(betas.size());
    std::vector<uint64_t> swap_attempts = std::vector<uint64_t>(betas.size() - 1),
                          swap_accepted = std::vector<uint64_t>(betas.size() - 1),
                          trips = std::vector<uint64_t>(betas.size());
    std::vector<unsigned> phase = std::vector<unsigned>(betas.size());
};
static std::vector<uint64_t> counters(const Stats & s) {
    return {s.divergences,
            s.scalar_accepted,
            s.scalar_attempts,
            s.support_rejections,
            s.rescale_accepted,
            s.rescale_attempts,
            s.cache_builds,
            s.prior_refreshes};
}
static void save(const FS::path & root, const std::string & identity, const Ensemble & s) {
    CP::Codec::Encoder e;
    e.text("FML serial replica ensemble 1");
    e.text(identity);
    e.integer(s.next);
    e.integer(betas.size());
    for (size_t j = 0; j < betas.size(); ++j) {
        const auto & x = s.states[j];
        e.vector(x.white);
        e.vector(x.parameters);
        e.real(x.quadratic);
        e.integer(x.label);
        e.real(s.stats[j].accept_sum);
        for (auto v : counters(s.stats[j]))
            e.integer(v);
        e.integer(s.phase[j]);
        e.integer(s.trips[j]);
    }
    for (size_t j = 0; j + 1 < betas.size(); ++j) {
        e.integer(s.swap_attempts[j]);
        e.integer(s.swap_accepted[j]);
    }
    CP::Codec::write_file((root / "CURRENT.tmp").string(), e.bytes);
    FS::rename(root / "CURRENT.tmp", root / "CURRENT");
}
static Ensemble load(const FS::path & root, const std::string & identity, const Model & model) {
    const size_t k = betas.size(), cells = model.cells(), limit = std::numeric_limits<size_t>::max();
    if (cells > (limit - 512) / 8)
        throw std::runtime_error("Replica checkpoint size overflow");
    const size_t per = 8 * cells + 512;
    if (k > (limit - 4096) / per)
        throw std::runtime_error("Replica checkpoint size overflow");
    const auto bytes = CP::Codec::read_file((root / "CURRENT").string(), 4096 + k * per);
    CP::Codec::Decoder d{bytes};
    if (d.text(64) != "FML serial replica ensemble 1" || d.text() != identity)
        throw std::runtime_error("Replica target/plan/ensemble changed");
    Ensemble s;
    s.next = int(d.size(Total));
    if (d.size(64) != betas.size())
        throw std::runtime_error("Replica count changed");
    const uint64_t draws = std::max(0, s.next - Warmup);
    for (size_t j = 0; j < betas.size(); ++j) {
        auto & x = s.states[j];
        auto & v = s.stats[j];
        x.white = d.vector(model.cells());
        x.parameters = d.vector(5);
        x.quadratic = d.real();
        x.label = d.size(betas.size() - 1);
        v.accept_sum = d.real();
        v.divergences = d.integer();
        v.scalar_accepted = d.integer();
        v.scalar_attempts = d.integer();
        v.support_rejections = d.integer();
        v.rescale_accepted = d.integer();
        v.rescale_attempts = d.integer();
        v.cache_builds = d.integer();
        v.prior_refreshes = d.integer();
        s.phase[j] = unsigned(d.size(2));
        s.trips[j] = d.size(draws * exchange_passes);
        bool valid = model.in_support(x.parameters) && std::isfinite(x.quadratic) && x.quadratic >= 0 &&
                     std::isfinite(v.accept_sum) && v.accept_sum >= 0 && v.accept_sum <= draws;
        for (double a : x.white)
            valid &= std::isfinite(a);
        if (j + 1 < betas.size())
            valid &= v.divergences <= draws && v.scalar_attempts == 4 * draws &&
                     v.scalar_accepted <= v.scalar_attempts && v.support_rejections <= v.scalar_attempts &&
                     v.rescale_attempts == 4 * draws && v.rescale_accepted <= v.rescale_attempts &&
                     v.cache_builds == 5 * uint64_t(s.next) && v.prior_refreshes == 0;
        else
            valid &= v.accept_sum == 0 && v.divergences == 0 && v.scalar_attempts == 0 && v.scalar_accepted == 0 &&
                     v.support_rejections == 0 && v.rescale_attempts == 0 && v.rescale_accepted == 0 &&
                     v.cache_builds == 0 && v.prior_refreshes == uint64_t(s.next);
        if (!valid)
            throw std::runtime_error("Invalid replica checkpoint state/counters");
    }
    for (size_t j = 0; j + 1 < betas.size(); ++j) {
        s.swap_attempts[j] = d.integer();
        s.swap_accepted[j] = d.integer();
        const uint64_t expected = EXCHANGE::expected_attempts(j, draws, exchange_passes);
        if (s.swap_attempts[j] != expected || s.swap_accepted[j] > expected)
            throw std::runtime_error("Invalid replica swap counters");
    }
    d.end();
    EXCHANGE::labels(s.states);
    return s;
}
static void array(std::ostream & out, const std::vector<double> & x) {
    out << '[';
    for (size_t j = 0; j < x.size(); ++j) {
        if (j)
            out << ',';
        out << x[j];
    }
    out << ']';
}
static double quadratic(const Model & model,
                        const FF::Modes & data,
                        const std::vector<double> & weights,
                        const EXCHANGE::PhysicalState & s,
                        Model::Workspace & work) {
    const auto & pred = model.predict(s.white, s.parameters, work);
    double result = 0;
    for (size_t j = 0; j < data.size(); ++j)
        result += .5 * weights[j] * std::norm(pred[j] - data[j]) * model.volume() / 1000.;
    if (!std::isfinite(result))
        throw std::runtime_error("Nonfinite base likelihood");
    return result;
}
static void prior_refresh(const Model & model, EXCHANGE::PhysicalState & s, uint64_t seed) {
    for (const auto & p : model.parameters())
        if (p.prior != FF::Prior::Uniform || !std::isfinite(p.lower) || !std::isfinite(p.upper) || !(p.lower < p.upper))
            throw std::invalid_argument("This prior refresh needs bounded uniform scalars");
    s.white = model.white_noise(seed);
    s.parameters.resize(model.parameters().size());
    std::mt19937_64 rng(seed + 1);
    std::uniform_real_distribution<double> uniform;
    for (size_t j = 0; j < s.parameters.size(); ++j) {
        const auto & p = model.parameters()[j];
        s.parameters[j] = p.lower + (p.upper - p.lower) * uniform(rng);
    }
}
static void positive_update(const Model & model,
                            const FF::Modes & data,
                            const Posterior & target,
                            EXCHANGE::PhysicalState & s,
                            Stats & stats,
                            const std::vector<double> & frozen,
                            uint64_t seed,
                            double beta,
                            bool measured,
                            Model::Workspace & work) {
    FML::MCMC::HMCOptions o;
    o.warmup = 0;
    o.samples = 1;
    o.n_leapfrog = 8;
    o.step_size = .15;
    o.jitter = false;
    o.adapt_mass = false;
    o.adapt_step_size = false;
    o.seed = seed;
    const auto field = FF::DETAIL::sample_run(target, s.white, s.parameters, o, false, {}, frozen);
    s.white = field.last;
    std::array<FF::Parameter, 4> priors;
    std::copy(model.parameters().begin() + 1, model.parameters().end(), priors.begin());
    auto provider = [&](double alpha) {
        auto p = s.parameters;
        p[0] = alpha;
        model.predict(s.white, p, work);
        return FF::FixedNoiseLinearBiasLikelihood<4>(model.bias_modes(work),
                                                     data,
                                                     target.likelihood().weights(),
                                                     1000. / beta / target.likelihood().volume(),
                                                     {0, 0, 0, 0});
    };
    FF::FixedICOptions scalar;
    scalar.warmup = 0;
    scalar.samples = 4;
    scalar.bias_sweeps = 8;
    scalar.amplitude_step = .07;
    scalar.seed = seed + 1;
    std::array<double, 5> p;
    std::copy(s.parameters.begin(), s.parameters.end(), p.begin());
    const auto result = FF::sample_fixed_ic_joint_amplitude<4>(provider, model.parameters()[0], priors, p, scalar);
    s.parameters.assign(result.draws.back().begin(), result.draws.back().end());
    stats.cache_builds += result.forward_evaluations;
    std::mt19937_64 rng(seed + 2);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> uniform;
    double current = target.value(s.white, s.parameters, work);
    if (!std::isfinite(current))
        throw std::runtime_error("Nonfinite replica rescaling start");
    for (int attempt = 0; attempt < 4; ++attempt) {
        auto proposal = RESCALING::propose(s.white,
                                           s.parameters,
                                           0,
                                           RESCALING::reflected(s.parameters[0], .5, 1.5, .1 * normal(rng)));
        double next = HUGE_VAL;
        try {
            next = target.value(proposal.white, proposal.parameters, work);
        } catch (const FF::UnrepresentableState &) {
        }
        double u = 0;
        while (u == 0)
            u = uniform(rng);
        const bool accepted = std::log(u) < RESCALING::log_acceptance(current, next, proposal.log_jacobian);
        if (accepted) {
            s.white = std::move(proposal.white);
            s.parameters = std::move(proposal.parameters);
            current = next;
        }
        if (measured) {
            ++stats.rescale_attempts;
            stats.rescale_accepted += accepted;
        }
    }
    if (measured) {
        stats.accept_sum += field.acceptance;
        stats.divergences += field.divergences;
        stats.scalar_attempts += result.amplitude_attempts;
        stats.scalar_accepted += result.amplitude_accepted;
        stats.support_rejections += result.support_rejections;
    }
}
static void record(const FS::path & root,
                   const Ensemble & s,
                   const FF::Modes & prediction,
                   const std::vector<EXCHANGE::Attempt> & events,
                   const std::vector<PassTrace> & trace,
                   size_t refreshed_label,
                   size_t draw) {
    const auto final = root / "draws" / ("draw-" + std::to_string(draw) + ".rank-0.json");
    const auto temporary = FS::path(final.string() + ".tmp");
    std::ofstream out(temporary);
    out << std::setprecision(17) << "{\"draw\":" << draw << ",\"rank\":0,\"white\":";
    array(out, s.states[0].white);
    out << ",\"parameters\":";
    array(out, s.states[0].parameters);
    out << ",\"prediction\":[";
    for (size_t j = 0; j < prediction.size(); ++j) {
        if (j)
            out << ',';
        out << '[' << prediction[j].real() << ',' << prediction[j].imag() << ']';
    }
    out << "],\"labels\":[";
    for (size_t j = 0; j < betas.size(); ++j) {
        if (j)
            out << ',';
        out << s.states[j].label;
    }
    out << "],\"quadratic\":[";
    for (size_t j = 0; j < betas.size(); ++j) {
        if (j)
            out << ',';
        out << s.states[j].quadratic;
    }
    out << "],\"replica_parameters\":[";
    for (size_t j = 0; j < betas.size(); ++j) {
        if (j)
            out << ',';
        array(out, s.states[j].parameters);
    }
    std::vector<double> attempts(betas.size() - 1), accepted(betas.size() - 1);
    for (const auto & e : events) {
        attempts[e.edge] += 1;
        accepted[e.edge] += e.accepted;
    }
    out << "],\"swap_attempts\":";
    array(out, attempts);
    out << ",\"swap_accepted\":";
    array(out, accepted);
    if (exchange_passes > 1) {
        out << ",\"refreshed_prior_label\":" << refreshed_label << ",\"exchange_trace\":[";
        for (size_t p = 0; p < trace.size(); ++p) {
            if (p)
                out << ',';
            out << "{\"parity\":" << trace[p].parity << ",\"attempts\":[";
            std::vector<double> a(betas.size() - 1), b(betas.size() - 1);
            for (const auto & e : trace[p].events) {
                a[e.edge] = 1;
                b[e.edge] = e.accepted;
            }
            for (size_t j = 0; j < a.size(); ++j) {
                if (j)
                    out << ',';
                out << a[j];
            }
            out << "],\"accepted\":";
            array(out, b);
            out << ",\"labels\":[";
            for (size_t j = 0; j < trace[p].labels.size(); ++j) {
                if (j)
                    out << ',';
                out << trace[p].labels[j];
            }
            out << "]}";
        }
        out << ']';
    }
    out << "}\n";
    out.close();
    if (!out)
        throw std::runtime_error("Cannot publish replica draw");
    FS::rename(temporary, final);
}
static void summary(const FS::path & root, const Ensemble & s) {
    const auto temporary = root / "summary.json.tmp";
    std::ofstream out(temporary);
    const int draws = std::max(0, s.next - Warmup);
    out << std::setprecision(17) << "{\"connectivity_only\":true,\"completed_iterations\":" << s.next
        << ",\"completed_samples\":" << draws << ",\"warmup\":" << Warmup << ",\"samples\":" << Samples
        << ",\"betas\":";
    array(out, betas);
    out << ",\"replicas\":[";
    for (size_t j = 0; j < betas.size(); ++j) {
        if (j)
            out << ',';
        const auto & v = s.stats[j];
        out << "{\"beta\":" << betas[j] << ",\"field_acceptance\":" << (draws ? v.accept_sum / draws : 0)
            << ",\"divergences\":" << v.divergences << ",\"scalar_accepted\":" << v.scalar_accepted
            << ",\"scalar_attempts\":" << v.scalar_attempts << ",\"support_rejections\":" << v.support_rejections
            << ",\"rescale_accepted\":" << v.rescale_accepted << ",\"rescale_attempts\":" << v.rescale_attempts
            << ",\"cache_builds\":" << v.cache_builds << ",\"prior_refreshes\":" << v.prior_refreshes << '}';
    }
    out << "],\"swap_attempts\":[";
    for (size_t j = 0; j + 1 < betas.size(); ++j) {
        if (j)
            out << ',';
        out << s.swap_attempts[j];
    }
    out << "],\"swap_accepted\":[";
    for (size_t j = 0; j + 1 < betas.size(); ++j) {
        if (j)
            out << ',';
        out << s.swap_accepted[j];
    }
    out << "],\"roundtrips\":[";
    for (size_t j = 0; j < betas.size(); ++j) {
        if (j)
            out << ',';
        out << s.trips[j];
    }
    out << "],\"transport_phase\":[";
    for (size_t j = 0; j < betas.size(); ++j) {
        if (j)
            out << ',';
        out << s.phase[j];
    }
    out << ']';
    if (exchange_passes > 1)
        out << ",\"exchange_passes\":" << exchange_passes;
    if (SeedLayout == 3)
        out << ",\"mock\":{\"id\":" << mock.id << ",\"truth_seed\":" << mock.truth << ",\"noise_seed\":" << mock.noise
            << ",\"initial_seed_base\":" << mock.initial << '}';
    if (mock.prior_predictive)
        out << ",\"truth_mode\":\"prior_predictive_v1\",\"scalar_truth_seed\":" << mock.scalar_truth;
    out << "}\n";
    out.close();
    if (!out)
        throw std::runtime_error("Cannot publish replica summary");
    FS::rename(temporary, root / "summary.json");
}
static void verify_targets(const Model & model,
                           const FF::Modes & data,
                           const Posterior & base,
                           const std::vector<Posterior> & targets,
                           Model::Workspace & work,
                           const FS::path & root) {
    double worst = 0;
    const auto weights = base.likelihood().weights();
    const auto initial = model.initial_parameters();
    EXCHANGE::PhysicalState a, b;
    a.white = model.white_noise(4103);
    a.parameters = initial;
    b.white = model.white_noise(4104);
    b.parameters = {.9, 1.2, .5, -.4, 30};
    a.quadratic = quadratic(model, data, weights, a, work);
    b.quadratic = quadratic(model, data, weights, b, work);
    auto prior = [](const Field & w) {
        double result = 0;
        for (double x : w)
            result += .5 * x * x;
        return result;
    };
    for (size_t j = 0; j + 1 < betas.size(); ++j) {
        const auto & t = targets[j];
        const double actual = t.value(b.white, b.parameters, work) - t.value(a.white, a.parameters, work);
        const double expected = prior(b.white) - prior(a.white) + betas[j] * (b.quadratic - a.quadratic);
        worst = std::max(worst, std::abs(actual - expected) / std::max(1., std::abs(expected)));
        model.predict(a.white, a.parameters, work);
        FF::FixedNoiseLinearBiasLikelihood<4> cache(model.bias_modes(work),
                                                    data,
                                                    weights,
                                                    1000. / betas[j] / model.volume(),
                                                    {0, 0, 0, 0});
        const std::array<double, 4> original = {initial[1], initial[2], initial[3], initial[4]};
        const double reference = t.value(a.white, initial, work) - cache.value(original);
        for (const auto & bias :
             std::vector<std::array<double, 4>>{{.2, -3, 2, 70}, {2, .5, -.3, 7}, {3, -1, -2, -60}}) {
            auto p = initial;
            std::copy(bias.begin(), bias.end(), p.begin() + 1);
            const double full = t.value(a.white, p, work), difference = (full - cache.value(bias)) - reference;
            worst = std::max(worst, std::abs(difference) / std::max(1., std::abs(full)));
        }
    }
    if (worst > 1e-10)
        throw std::runtime_error("Tempered target/cache/prior inconsistency");
    std::vector<double> mean(5), second(5);
    double wmean = 0, wsecond = 0;
    EXCHANGE::PhysicalState state;
    const int n = 8192;
    for (int k = 0; k < n; ++k) {
        prior_refresh(model, state, 8400000 + 2 * k);
        wmean += state.white[0];
        wsecond += state.white[0] * state.white[0];
        for (int j = 0; j < 5; ++j) {
            const auto & p = model.parameters()[j];
            const double x = (state.parameters[j] - .5 * (p.lower + p.upper)) / (p.upper - p.lower);
            mean[j] += x;
            second[j] += x * x;
        }
    }
    if (std::abs(wmean / n) > .04 || std::abs(wsecond / n - 1) > .06)
        throw std::runtime_error("beta=0 white prior refresh mismatch");
    for (int j = 0; j < 5; ++j)
        if (std::abs(mean[j] / n) > .05 || std::abs(second[j] / n - 1. / 12) > .06 / 12)
            throw std::runtime_error("beta=0 scalar prior refresh mismatch");
    std::ofstream out(root / "target-verification.json");
    out << std::setprecision(17) << "{\"passed\":true,\"positive_betas\":" << betas.size() - 1
        << ",\"cache_points\":" << 3 * (betas.size() - 1) << ",\"prior_draws\":8192,\"max_scaled_error\":" << worst
        << ",\"prior_white_mean\":" << wmean / n << ",\"prior_white_second\":" << wsecond / n << "}\n";
    out.close();
    if (!out)
        throw std::runtime_error("Cannot publish target verification");
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (argc < 4 || (argc > 8 && argc != 10) || FML::NTasks != 1)
            throw std::invalid_argument(
                "tempering_pilot DIRECTORY start|resume|extend ENSEMBLE [sweep_budget [ladder_file "
                "[exchange_passes [settings_file]]]]; extend additionally requires SOURCE OLD_SAMPLES and budget=0, "
                "serial only");
        const FS::path root = FS::absolute(argv[1]).lexically_normal();
        const std::string phase = argv[2];
        if (phase != "start" && phase != "resume" && phase != "extend")
            throw std::invalid_argument("Expected start, resume or extend");
        const bool resume = phase == "resume", extend = phase == "extend";
        if ((extend && argc != 10) || (!extend && argc > 8))
            throw std::invalid_argument("Invalid import argument count");
        size_t used;
        const int ensemble = std::stoi(argv[3], &used);
        if (used != std::string(argv[3]).size() || ensemble < 0 || ensemble > 3)
            throw std::invalid_argument("Invalid ensemble index");
        int budget = 16;
        if (argc >= 5) {
            budget = std::stoi(argv[4], &used);
            if (used != std::string(argv[4]).size() || budget < 0)
                throw std::invalid_argument("Invalid sweep budget");
        }
        if (argc >= 6) {
            std::ifstream input(argv[5]);
            if (!input)
                throw std::invalid_argument("Missing explicit ladder file");
            std::vector<double> parsed;
            std::string token;
            while (input >> token) {
                size_t consumed;
                const double value = std::stod(token, &consumed);
                if (consumed != token.size())
                    throw std::invalid_argument("Invalid ladder token");
                parsed.push_back(value);
                if (parsed.size() > 64)
                    throw std::invalid_argument("Too many replicas");
            }
            if (input.bad())
                throw std::runtime_error("Ladder read failed");
            betas = std::move(parsed);
        }
        if (argc >= 7) {
            size_t consumed;
            const int p = std::stoi(argv[6], &consumed);
            if (consumed != std::string(argv[6]).size() || p < 1 || p > 8)
                throw std::invalid_argument("Invalid exchange pass count");
            exchange_passes = unsigned(p);
        }
        if (argc >= 8) {
            std::ifstream input(argv[7]);
            std::string extra;
            if (!(input >> Warmup >> Samples >> Ensembles >> SeedLayout) || Warmup < 0 || Warmup % 2 || Samples < 1 ||
                Samples > 1000000 || Warmup > 1000000 - Samples || Ensembles < 1 || Ensembles > 4 ||
                (SeedLayout != 1 && SeedLayout != 2 && SeedLayout != 3))
                throw std::invalid_argument("Invalid schedule: even warmup, samples, ensembles, seed layout");
            if (SeedLayout == 3)
                mock.read(input, unsigned(Ensembles));
            if (input >> extra)
                throw std::invalid_argument("Extra settings fields");
            Total = Warmup + Samples;
        }
        if (ensemble >= Ensembles || (SeedLayout == 1 && (Warmup != 64 || Samples != 256 || Ensembles != 2)))
            throw std::invalid_argument("New schedules require tuple seed layout 2");
        int old_samples = 0;
        if (extend) {
            old_samples = std::stoi(argv[9], &used);
            if (used != std::string(argv[9]).size() || SeedLayout != 2 || budget != 0 || old_samples < 1 ||
                old_samples >= Samples)
                throw std::invalid_argument(
                    "Extension needs seed layout 2, zero budget and a smaller old sample count");
        }
        EXCHANGE::layout(betas, exchange_passes);
        if (!resume && FS::exists(root) && !FS::is_empty(root))
            throw std::invalid_argument("Start requires new or empty directory");
        FS::create_directories(root / "draws");
        FF::EulerianLPTConfiguration c;
        c.mesh = 4;
        c.product_mesh = 8;
        c.density_mesh = 16;
        c.boxsize = 100;
        const double kf = 2 * M_PI / 100;
        c.initial_cutoff = 1.7 * kf;
        c.matter_band = {1.8 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.likelihood_band = {1.6 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.matter_assignment = FF::MatterAssignment::PSWF;
        std::array<FF::Parameter, 4> bias = {{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                              {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                              {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                              {"b_laplacian", 7, 0, 1, -100, 100, FF::Prior::Uniform}}};
        Model model(c, FF::sigma8_spectrum([](double k) { return 20000 * std::exp(-5 * k); }), bias);
        auto work = model.workspace();
        const auto truth = mock.prior_predictive ? MOCK::prior_parameters(model.parameters(), mock.generation_key(1)) :
                                                   model.initial_parameters();
        const auto truth_white = model.white_noise(mock.generation_key(0));
        const auto data = model.mock(truth_white, truth, 1000, mock.generation_key(2), work);
        const auto base = FF::fixed_noise_posterior(model, data, 1000.);
        const auto weights = base.likelihood().weights();
        std::vector<Posterior> targets;
        targets.reserve(betas.size() - 1);
        for (size_t j = 0; j + 1 < betas.size(); ++j)
            targets.push_back(FF::fixed_noise_posterior(model, data, 1000. / betas[j]));
        const std::vector<std::vector<double>> starts = {{.8, 1.2, -.3, .2, -20},
                                                         {1.2, 2, .8, -.6, 30},
                                                         {.9, 2.4, -.8, .6, 50},
                                                         {1.1, .8, 1, -1, -40}};
        const auto frozen = starts[ensemble];
        auto checkpoint_identity = [&]() {
            std::string problem = "FML replica connectivity "
                                  "1:6de0c06e5a05246ffa04c700b7a5f5615a1bf19787c37879c957318998a5a731:ensemble=" +
                                  std::to_string(ensemble);
            if (betas != legacy_betas || exchange_passes != 1) {
                CP::Codec::Encoder config;
                config.text("FML explicit replica layout 1");
                config.vector(betas);
                config.integer(exchange_passes);
                problem += ":layout=" + std::to_string(CP::Codec::checksum(config.bytes));
            }
            if (SeedLayout != 1) {
                CP::Codec::Encoder config;
                config.text("FML replica schedule tuple seeds 2");
                for (int v : {Warmup, Samples, Ensembles, SeedLayout})
                    config.integer(v);
                config.vector(frozen);
                problem += ":schedule=" + std::to_string(CP::Codec::checksum(config.bytes));
            }
            if (SeedLayout == 3) {
                CP::Codec::Encoder config;
                config.text("FML independent mock seed metadata 1");
                for (auto v : {mock.id, mock.truth, mock.noise, mock.initial})
                    config.integer(v);
                problem += ":mock=" + std::to_string(CP::Codec::checksum(config.bytes));
            }
            if (mock.prior_predictive) {
                CP::Codec::Encoder config;
                config.text("FML prior predictive truth 1: uniform physical scalars, Gaussian white and noise");
                config.integer(mock.scalar_truth);
                config.vector(truth);
                for (const auto & p : model.parameters()) {
                    config.text(p.name);
                    config.real(p.lower);
                    config.real(p.upper);
                }
                problem += ":prior_predictive=" + std::to_string(CP::Codec::checksum(config.bytes));
            }
            return CP::identity(base, problem);
        };
        const auto identity = checkpoint_identity();
        Ensemble s;
        if (resume || extend) {
            if (extend) {
                const int new_samples = Samples;
                Samples = old_samples;
                Total = Warmup + Samples;
                s = load(FS::absolute(argv[8]).lexically_normal(), checkpoint_identity(), model);
                if (s.next != Total)
                    throw std::runtime_error("Extension requires a completed source schedule");
                Samples = new_samples;
                Total = Warmup + Samples;
            } else {
                s = load(root, identity, model);
            }
            for (const auto & state : s.states) {
                const double q = quadratic(model, data, weights, state, work);
                if (std::abs(q - state.quadratic) > 1e-11 * std::max(1., std::abs(q)))
                    throw std::runtime_error("Saved replica likelihood/target mismatch");
            }
        } else {
            for (size_t j = 0; j < betas.size(); ++j) {
                s.states[j].white = model.white_noise(mock.generation_key(3, unsigned(ensemble)));
                s.states[j].parameters = frozen;
                s.states[j].label = j;
                s.states[j].quadratic = quadratic(model, data, weights, s.states[j], work);
            }
            std::ofstream out(root / "target.json");
            out << std::setprecision(17) << "{\"truth_white\":";
            array(out, truth_white);
            out << ",\"data\":[";
            for (size_t j = 0; j < data.size(); ++j) {
                if (j)
                    out << ',';
                out << '[' << data[j].real() << ',' << data[j].imag() << ']';
            }
            out << "],\"k\":[";
            for (size_t j = 0; j < data.size(); ++j) {
                if (j)
                    out << ',';
                out << model.modes()[j].k;
            }
            out << "],\"weights\":[";
            for (size_t j = 0; j < weights.size(); ++j) {
                if (j)
                    out << ',';
                out << weights[j];
            }
            out << ']';
            if (SeedLayout == 3) {
                out << ",\"truth_parameters\":";
                array(out, truth);
                out << ",\"mock\":{\"id\":" << mock.id << ",\"truth_seed\":" << mock.truth
                    << ",\"noise_seed\":" << mock.noise << ",\"initial_seed_base\":" << mock.initial << '}';
            }
            if (mock.prior_predictive) {
                out << ",\"truth_mode\":\"prior_predictive_v1\",\"scalar_truth_seed\":" << mock.scalar_truth;
                out << ",\"scalar_priors\":[";
                for (size_t j = 0; j < model.parameters().size(); ++j) {
                    if (j)
                        out << ',';
                    const auto & p = model.parameters()[j];
                    out << "{\"name\":\"" << p.name << "\",\"kind\":\"uniform\",\"lower\":" << p.lower
                        << ",\"upper\":" << p.upper << '}';
                }
                out << "],\"generation_keys\":[";
                for (unsigned role = 0; role < 4; ++role) {
                    if (role)
                        out << ',';
                    out << '\"' << mock.generation_key(role) << '\"';
                }
                out << "],\"noise_white\":";
                array(out, model.white_noise(mock.generation_key(2)));
                out << ",\"truth_prediction\":[";
                const auto & prediction = model.predict(truth_white, truth, work);
                for (size_t j = 0; j < prediction.size(); ++j) {
                    if (j)
                        out << ',';
                    out << '[' << prediction[j].real() << ',' << prediction[j].imag() << ']';
                }
                out << "],\"mode_n\":[";
                for (size_t j = 0; j < model.modes().size(); ++j) {
                    if (j)
                        out << ',';
                    const auto & n = model.modes()[j].n;
                    out << '[' << n[0] << ',' << n[1] << ',' << n[2] << ']';
                }
                out << ']';
                EXCHANGE::PhysicalState generating;
                generating.white = truth_white;
                generating.parameters = truth;
                out << ",\"truth_quadratic\":" << quadratic(model, data, weights, generating, work);
            }
            out << "}\n";
            out.close();
            if (!out)
                throw std::runtime_error("Cannot publish target");
            if (budget == 0)
                verify_targets(model, data, base, targets, work, root);
        }
        const int stop = s.next + std::min(budget, Total - s.next);
        for (; s.next < stop;) {
            const bool measured = s.next >= Warmup;
            const size_t refreshed_label = s.states.back().label;
            for (size_t j = 0; j < betas.size(); ++j) {
                const uint64_t seed = sweep_seed(ensemble, j, s.next);
                if (j + 1 < betas.size())
                    positive_update(model,
                                    data,
                                    targets[j],
                                    s.states[j],
                                    s.stats[j],
                                    frozen,
                                    seed,
                                    betas[j],
                                    measured,
                                    work);
                else {
                    prior_refresh(model, s.states[j], seed);
                    if (measured && exchange_passes > 1)
                        EXCHANGE::prior_refresh_visit(s.states[j].label, s.phase);
                    ++s.stats[j].prior_refreshes;
                }
                s.states[j].quadratic = quadratic(model, data, weights, s.states[j], work);
            }
            std::mt19937_64 swap_rng(sweep_seed(ensemble, 63, s.next, 3));
            std::vector<EXCHANGE::Attempt> all_events;
            std::vector<PassTrace> traces;
            for (unsigned pass = 0; pass < exchange_passes; ++pass) {
                const unsigned parity = unsigned((s.next + pass) % 2);
                const auto events = EXCHANGE::neighbors(s.states, betas, parity, swap_rng);
                all_events.insert(all_events.end(), events.begin(), events.end());
                if (measured) {
                    for (const auto & e : events) {
                        ++s.swap_attempts[e.edge];
                        s.swap_accepted[e.edge] += e.accepted;
                    }
                    EXCHANGE::observe(s.states, s.phase, s.trips, exchange_passes == 1);
                    PassTrace trace;
                    trace.parity = parity;
                    trace.events = events;
                    for (const auto & x : s.states)
                        trace.labels.push_back(x.label);
                    traces.push_back(std::move(trace));
                }
            }
            if (measured)
                record(root,
                       s,
                       model.predict(s.states[0].white, s.states[0].parameters, work),
                       all_events,
                       traces,
                       refreshed_label,
                       size_t(s.next - Warmup));
            ++s.next;
            if (s.next % 16 == 0)
                save(root, identity, s);
        }
        save(root, identity, s);
        summary(root, s);
        std::cout << "Replica sweeps " << s.next << "/" << Total << "; cold records " << std::max(0, s.next - Warmup)
                  << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
