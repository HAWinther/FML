#include <FML/EnsembleStatistics/MatchedSpectra.h>
#include <FML/NBody/NBody.h>
#include <FML/ParticleTypes/SimpleParticle.h>
#include <FML/RandomFields/CounterBasedRandomField.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace FML::ENSEMBLE;
using Grid = FML::GRID::FFTWGrid<3>;
using Particles = FML::PARTICLE::MPIParticles<COLAParticle<3>>;
namespace {
    constexpr double box = 200., aini = .05;
    constexpr int bins = 3;
    void require_all(bool good, const char * message) {
        int ok = good;
        FML::MinOverTasks(&ok);
        detail::require(ok, message);
    }
    using Clock = std::chrono::steady_clock;
    double elapsed(Clock::time_point start) {
        double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        FML::MaxOverTasks(&seconds);
        return seconds;
    }
    Grid density(Particles & particles, int mesh) {
        Grid grid(mesh, 0, 1);
        FML::INTERPOLATION::particles_to_grid(particles.get_particles_ptr(),
                                              particles.get_npart(),
                                              particles.get_npart_total(),
                                              grid,
                                              "CIC");
        grid.fftw_r2c();
        return grid;
    }
    void lpt_positions(Particles & particles, double a, int order) {
        const double growth = a / aini;
        for (auto & p : particles)
            for (int d = 0; d < 3; ++d) {
                p.Pos[d] = FML::PARTICLE::wrap_coordinate_periodic(p.q[d] + growth * p.D_1LPT[d] +
                                                                   (order == 2 ? growth * growth * p.D_2LPT[d] : 0));
                p.Vel[d] = std::sqrt(aini) * (p.D_1LPT[d] + 2 * p.D_2LPT[d]);
            }
        particles.communicate_particles();
    }
    // EdS canonical momentum p=a^2 dx/d(H0 t). Force grid here is i k delta/k^2;
    // kick integral = integral (3/2) a^(-1/2) da, drift = integral a^(-3/2) da.
    void evolve(Particles & particles, int mesh, int steps, double afinal) {
        const double da = (afinal - aini) / steps;
        for (int step = 0; step <= steps; ++step) {
            const double a = aini + step * da;
            const double old_vel = step == 0 ? aini : a - .5 * da;
            const double new_vel = step == steps ? afinal : a + .5 * da;
            auto delta = density(particles, mesh);
            std::array<Grid, 3> force;
            FML::NBODY::compute_force_from_density_fourier<3>(delta, force, "CIC", 1.);
            FML::NBODY::KickParticles<3>(force, particles, 3 * (std::sqrt(new_vel) - std::sqrt(old_vel)), "CIC");
            if (step < steps)
                FML::NBODY::DriftParticles<3>(particles, 2 * (1 / std::sqrt(a) - 1 / std::sqrt(a + da)));
        }
    }
    MatchedSpectra spectra(const Grid & target, const Grid & control) {
        return matched_spectra(target, control, box, 2 * M_PI / box, 6 * M_PI / box, bins);
    }
} // namespace
int main(int argc, char ** argv) {
    int mesh = 0, steps = 0, ntrain = 0, nmean = 0, neval = 0;
    double afinal = 0, amplitude = 0;
    std::uint64_t seed = 0;
    bool valid = argc == 10;
    if (valid)
        try {
            // stox must consume the entire argument, rather than accepting a numeric prefix.
            auto integer = [](const char * text) {
                size_t n;
                int v = std::stoi(text, &n);
                detail::require(n == std::strlen(text), "Invalid integer");
                return v;
            };
            auto real = [](const char * text) {
                size_t n;
                double v = std::stod(text, &n);
                detail::require(n == std::strlen(text), "Invalid number");
                return v;
            };
            mesh = integer(argv[2]);
            steps = integer(argv[3]);
            afinal = real(argv[4]);
            ntrain = integer(argv[5]);
            nmean = integer(argv[6]);
            neval = integer(argv[7]);
            size_t n;
            seed = std::stoull(argv[8], &n);
            valid = argv[8][0] != '-' && n == std::strlen(argv[8]);
            amplitude = real(argv[9]);
            valid = valid && mesh >= 16 && mesh <= 256 && mesh % 2 == 0 && mesh >= FML::NTasks && steps >= 1 &&
                    steps <= 10000 && afinal > aini && afinal <= 1 && amplitude > 0 && amplitude <= .1 &&
                    std::isfinite(amplitude) && ntrain >= 2 && ntrain <= 1000000 && nmean >= 2 && nmean <= 1000000 &&
                    neval >= 2 && neval <= 1000000 && seed <= UINT64_MAX - 3000000;
        } catch (const std::exception &) {
            valid = false;
        }
    require_all(valid,
                "Usage: matched_pm prefix mesh steps afinal ntrain nmean neval seed amplitude (invalid options)");
    Count schema = 1469598103934665603ULL;
    detail::values(schema,
                   {double(mesh), double(steps), afinal, double(ntrain), double(nmean), double(neval), amplitude});
    detail::word(schema, seed);
    detail::same(schema);
    const std::string prefix = argv[1];
    std::ofstream rows, timing, results, identities, config;
    if (!FML::ThisTask) {
        rows.open(prefix + "-rows.csv");
        timing.open(prefix + "-timing.csv");
        results.open(prefix + "-results.csv");
        identities.open(prefix + "-models.csv");
        config.open(prefix + "-config.txt");
        for (auto * f : {&rows, &timing, &results, &config})
            *f << std::setprecision(17);
        rows << "role,id,seed,bin,k,modes,target,za,lpt2,cross_za,cross_lpt2,r_za,r_lpt2\n";
        timing << "role,id,ic,za,lpt2,pm,total\n";
        results << "model,quantity,row,col,value\n";
        identities << "model,metadata_hash,model_hash,ntrain,nmean,neval,ridge\n";
        config << "schema=matched-pm-v1\nbox=" << box << "\naini=" << aini << "\naf=" << afinal << "\nmesh=" << mesh
               << "\nsteps=" << steps << "\nseed=" << seed << "\namplitude=" << amplitude << "\n";
    }
    require_all(FML::ThisTask || (rows && timing && results && identities && config), "Cannot open validation outputs");
    std::ostringstream tag;
    tag << std::setprecision(17) << "EdS;box=" << box << ";mesh=" << mesh << ";steps=" << steps << ";aini=" << aini
        << ";af=" << afinal << ";P/V=" << amplitude << "*q/6*exp(-(q/20)^2);q<=12pi;v1";
    std::vector<Metadata> tags;
    std::vector<PairedTraining> training;
    std::vector<ControlMean> means;
    for (const auto & method : {"ZA", "2LPT"}) {
        tags.push_back(
            {tag.str(), "raw CIC P(k);centers=2pi,4pi,6pi / L;volume units;no noise subtraction;v1", "PM", method});
        training.emplace_back(bins, bins, tags.back());
        means.emplace_back(bins, tags.back());
    }
    std::vector<ControlModel> models;
    std::vector<ControlEvaluation> evaluation;
    RunningCovariance raw(bins);
    const int total = ntrain + nmean + neval;
    for (int id = 0; id < total; ++id) {
        if (id == ntrain + nmean)
            for (int method = 0; method < 2; ++method) {
                const auto t = training[method].global();
                const auto mu = means[method].global().estimate();
                for (auto mode : {FitMode::Diagonal, FitMode::Full}) {
                    FitOptions options;
                    options.mode = mode;
                    models.push_back(t.fit(mu, options));
                    evaluation.emplace_back(models.back());
                }
            }
        const std::string role = id < ntrain ? "train" : id < ntrain + nmean ? "mean" : "eval";
        const auto start = Clock::now();
        Grid delta(mesh, 0, 1);
        FML::RANDOM::COUNTER_BASED::generate_fourier<3>(
            delta,
            seed + id,
            0,
            [&](double q) {
                return q > 12 * M_PI ? 0. : aini * aini * amplitude * (q / 6) * std::exp(-std::pow(q / 20, 2));
            },
            false);
        Particles particles;
        std::vector<Grid> potentials;
        FML::NBODY::NBodyInitialConditions<3>(particles,
                                              mesh,
                                              2.,
                                              delta,
                                              potentials,
                                              2,
                                              box,
                                              1 / aini - 1,
                                              {std::sqrt(aini), 2 * std::sqrt(aini)});
        delta.free();
        const double ic = elapsed(start);
        auto begin = Clock::now();
        lpt_positions(particles, afinal, 1);
        auto za = density(particles, mesh);
        const double za_time = elapsed(begin);
        begin = Clock::now();
        lpt_positions(particles, afinal, 2);
        auto lpt = density(particles, mesh);
        const double lpt_time = elapsed(begin);
        begin = Clock::now();
        Grid target;
        if (role != "mean") {
            lpt_positions(particles, aini, 2);
            evolve(particles, mesh, steps, afinal);
            target = density(particles, mesh);
        }
        const double pm_time = elapsed(begin);
        const auto first = spectra(role == "mean" ? za : target, za),
                   second = spectra(role == "mean" ? lpt : target, lpt);
        const std::array<Vector, 2> controls{first.control, second.control};
        const PhaseKey key{std::uint64_t(id), seed + id};
        if (!FML::ThisTask) {
            if (role == "train")
                for (int method = 0; method < 2; ++method)
                    training[method].add(first.target, controls[method], key, key);
            if (role == "mean")
                for (int method = 0; method < 2; ++method)
                    means[method].add(controls[method], key);
            if (role == "eval") {
                raw.add(first.target);
                for (size_t model = 0; model < evaluation.size(); ++model)
                    evaluation[model].add(first.target, controls[model / 2], key, key);
            }
            for (int b = 0; b < bins; ++b)
                rows << role << ',' << id << ',' << seed + id << ',' << b << ',' << first.k[b] << ',' << first.modes[b]
                     << ',' << (role == "mean" ? detail::nan() : first.target[b]) << ',' << controls[0][b] << ','
                     << controls[1][b] << ',' << (role == "mean" ? detail::nan() : first.cross[b]) << ','
                     << (role == "mean" ? detail::nan() : second.cross[b]) << ','
                     << (role == "mean" ? detail::nan() : first.correlation[b]) << ','
                     << (role == "mean" ? detail::nan() : second.correlation[b]) << '\n';
        }
        const double all_time = elapsed(start);
        if (!FML::ThisTask)
            timing << role << ',' << id << ',' << ic << ',' << za_time << ',' << lpt_time << ',' << pm_time << ','
                   << all_time << '\n';
        require_all(FML::ThisTask || (rows && timing), "Cannot write validation row");
    }
    const auto baseline = raw.global().summary();
    for (size_t i = 0; i < models.size(); ++i) {
        const auto estimate = evaluation[i].global().estimate();
        if (!FML::ThisTask) {
            const std::string name = std::string(i / 2 ? "2LPT" : "ZA") + (i % 2 ? "-full" : "-diagonal");
            identities << name << ',' << models[i].metadata().fingerprint() << ',' << models[i].fingerprint() << ','
                       << ntrain << ',' << nmean << ',' << neval << ',' << models[i].fit_options().ridge << '\n';
            auto write = [&](const std::string & quantity, const Vector & v, int columns) {
                for (size_t j = 0; j < v.size(); ++j)
                    results << name << ',' << quantity << ',' << j / columns << ',' << j % columns << ',' << v[j]
                            << '\n';
            };
            write("beta", models[i].coefficients(), bins);
            write("mean", estimate.mean, 1);
            write("covariance", estimate.sample_covariance, bins);
            write("covariance_of_mean", estimate.covariance_of_mean, bins);
            write("mean_floor", models[i].mean_uncertainty(), bins);
            write("raw_mean", baseline.mean, 1);
            write("raw_covariance", baseline.covariance, bins);
        }
    }
    require_all(FML::ThisTask || (results && identities && config), "Cannot write validation results");
}
