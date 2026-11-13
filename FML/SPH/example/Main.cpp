#include <FML/SPH/SPH.h>
#include <FML/Global/MPICounts.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace FML::SPH;
namespace FS = std::filesystem;
namespace C = FML::COMMUNICATION;

template <int D>
int experiment(const FS::path & dir, const std::string & mode, int n, double end, double dtmax) {
    Options o;
    o.h_min = .3 / n;
    o.h_max = 1.7 / n;
    o.density_tolerance = 1e-10;
    if (mode == "sod") {
        o.gamma = 1.4;
        o.eta = 1.5;
        o.viscosity_alpha = 1;
        o.viscosity_beta = 2;
        o.conductivity_alpha = .2;
        o.balsara = false;
        o.h_min = .02 / n;
        o.h_max = 15. / n;
    } else if (mode == "blast") {
        o.viscosity_alpha = 1;
        o.conductivity_alpha = .1;
        o.balsara = false;
    } else {
        o.viscosity_alpha = o.viscosity_beta = o.conductivity_alpha = 0;
    }
    const double u0 = 1 / (o.gamma * (o.gamma - 1)), amplitude = 1e-4;
    std::vector<Particle<D>> particles;
    size_t count = 1;
    for (int d = 0; d < D; ++d)
        count *= n;
    if (mode == "sod")
        count = n;
    for (size_t id = 0; id < count; ++id) {
        if (int(id % FML::NTasks) != FML::ThisTask)
            continue;
        Particle<D> p;
        p.id = id;
        p.mass = 1. / count;
        p.energy = u0;
        size_t k = id;
        for (int d = 0; d < D; ++d) {
            p.position[d] = (k % n + .5) / n;
            k /= n;
        }
        if (mode == "sod") {
            const size_t left = 8 * count / 9, right = count - left;
            const bool isleft = id < left;
            p.position[0] = isleft ? .5 * (id + .5) / left : .5 + .5 * (id - left + .5) / right;
            p.mass = .5625 / count;
            p.energy = (isleft ? 1. : .1) / ((o.gamma - 1) * (isleft ? 1. : .125));
        }
        if (mode == "blast") {
            p.energy = 1e-5;
            double r2 = 0;
            for (auto x : p.position)
                r2 += (x - .5) * (x - .5);
            if (r2 < .12 * .12)
                p.energy += 1; // Normalize the finite injection region below.
        }
        particles.push_back(p);
    }
    double rho0 = 1;
    if (mode == "sound") {
        Solver<D> uniform(particles, o);
        auto state = uniform.evaluate();
        rho0 = 0;
        for (auto & g : state.gas)
            rho0 += g.density;
        FML::SumOverTasks(&rho0);
        rho0 /= count;
        for (auto & p : particles) {
            const double phase = 2 * M_PI * p.position[0];
            p.position[0] -= amplitude / (2 * M_PI) * std::sin(phase);
            p.velocity[0] = amplitude * std::cos(phase);
            p.energy = u0 * std::pow(1 + amplitude * std::cos(phase), o.gamma - 1);
        }
    }
    if (mode == "blast") {
        double injected = 0;
        for (const auto & p : particles)
            injected += p.mass * (p.energy - 1e-5);
        FML::SumOverTasks(&injected);
        if (injected <= 0)
            throw std::runtime_error("Blast injection unresolved");
        for (auto & p : particles)
            p.energy = 1e-5 + (p.energy - 1e-5) * .01 / injected;
    }
    Solver<D> gas(particles, o);
    C::collective_try(
        [&] {
            if (FML::ThisTask == 0) {
                if (FS::exists(dir))
                    throw std::invalid_argument("Use a fresh output directory");
                FS::create_directories(dir);
            }
        },
        "SPH experiment directory");
    std::ofstream history;
    C::collective_try(
        [&] {
            if (FML::ThisTask == 0) {
                history.open(dir / "history.tsv");
                history << std::setprecision(17) << "# time mass kinetic thermal total momentum[dimension]\n";
                if (!history)
                    throw std::runtime_error("Cannot create history");
            }
        },
        "SPH history");
    const auto initial = gas.diagnostics();
    auto record = [&] {
        auto d = gas.diagnostics();
        if (FML::ThisTask == 0) {
            history << gas.time() << ' ' << d.mass << ' ' << d.kinetic << ' ' << d.thermal << ' ' << d.gas_energy();
            for (auto p : d.momentum)
                history << ' ' << p;
            history << '\n';
        }
    };
    record();
    int steps = 0;
    while (gas.time() < end) {
        double remaining = end - gas.time();
        if (remaining <= 16 * std::numeric_limits<double>::epsilon() * std::max(1., end))
            break;
        gas.advance(std::min(dtmax, remaining));
        if (++steps % 10 == 0)
            record();
        if (steps > 100000)
            throw std::runtime_error("Experiment step limit");
    }
    record();
    auto fields = gas.evaluate();
    const auto final = gas.diagnostics();
    double velocity_error = 0, density_error = 0,
           energy_error = std::abs(final.gas_energy() / initial.gas_energy() - 1), momentum = 0;
    for (size_t i = 0; i < gas.particles().size(); ++i) {
        const auto & p = gas.particles()[i];
        if (mode == "sound") {
            const double phase = 2 * M_PI * (p.position[0] - gas.time());
            velocity_error += std::abs(p.velocity[0] - amplitude * std::cos(phase)) / count / amplitude;
            density_error +=
                std::abs(fields.gas[i].density / rho0 - 1 - amplitude * std::cos(phase)) / count / amplitude;
        }
    }
    FML::SumOverTasks(&velocity_error);
    FML::SumOverTasks(&density_error);
    for (auto p : final.momentum)
        momentum += p * p;
    momentum = std::sqrt(momentum);
    C::collective_try(
        [&] {
            std::ofstream file(dir / ("particles.rank" + std::to_string(FML::ThisTask) + ".tsv"));
            file << std::setprecision(17)
                 << "# id position[D] velocity[D] mass u h rho pressure sound_speed omega div curl limiter\n";
            for (size_t i = 0; i < gas.particles().size(); ++i) {
                const auto & p = gas.particles()[i];
                const auto & g = fields.gas[i];
                file << p.id;
                for (auto x : p.position)
                    file << ' ' << x;
                for (auto v : p.velocity)
                    file << ' ' << v;
                file << ' ' << p.mass << ' ' << p.energy << ' ' << p.h << ' ' << g.density << ' ' << g.pressure << ' '
                     << g.sound_speed << ' ' << g.omega << ' ' << g.divergence << ' ' << g.curl << ' ' << g.limiter
                     << '\n';
            }
            file.close();
            if (!file)
                throw std::runtime_error("Particle output failed");
            if (FML::ThisTask == 0) {
                history.close();
                if (!history)
                    throw std::runtime_error("History output failed");
                std::ofstream meta(dir / "meta.tsv");
                meta << std::setprecision(17) << "dimension " << D << "\nn " << n << "\nparticles " << count
                     << "\ntime " << gas.time() << "\nsteps " << steps << "\ngamma " << o.gamma << "\neta " << o.eta
                     << "\nrho0 " << rho0 << "\namplitude " << amplitude << "\nvelocity_l1_over_amplitude "
                     << velocity_error << "\ndensity_l1_over_amplitude " << density_error << "\nrelative_energy_error "
                     << energy_error << "\nfinal_momentum_norm " << momentum << '\n';
                meta.close();
                if (!meta)
                    throw std::runtime_error("Metadata output failed");
                std::cout << "EXPERIMENT " << mode << " D=" << D << " n=" << n << " steps=" << steps
                          << " velocity_L1=" << velocity_error << " density_L1=" << density_error
                          << " energy=" << energy_error << " momentum=" << momentum << '\n';
            }
        },
        "SPH experiment output");
    return 0;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    std::cout << std::setprecision(17);
    try {
        if (argc < 5 || argc > 6)
            throw std::invalid_argument("Usage: sph_example OUTPUT sound1d|sound3d|sod|blast3d N END_TIME [MAX_DT]");
        const std::string mode = argv[2];
        const int n = std::stoi(argv[3]);
        const double end = std::stod(argv[4]), dtmax = argc == 6 ? std::stod(argv[5]) : end;
        if (n < 8 || n > 1000000 || ((mode == "sound3d" || mode == "blast3d") && n > 128) || !std::isfinite(end) ||
            end <= 0 || !std::isfinite(dtmax) || dtmax <= 0)
            throw std::invalid_argument("Invalid experiment controls");
        if (mode == "sound1d")
            return experiment<1>(argv[1], "sound", n, end, dtmax);
        if (mode == "sound3d")
            return experiment<3>(argv[1], "sound", n, end, dtmax);
        if (mode == "sod")
            return experiment<1>(argv[1], "sod", n, end, dtmax);
        if (mode == "blast3d")
            return experiment<3>(argv[1], "blast", n, end, dtmax);
        throw std::invalid_argument("Unknown experiment");
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
#ifdef USE_MPI
        MPI_Abort(FML::communicator, 1);
#endif
        return 1;
    }
}
