// Actual SimulationCode evolution versus an independent periodic direct-force ODE.
#include "../src/ModelCompatibility.h"
#include "../src/Models.h"
#include "../src/ParameterSchema.h"
#include "../src/Particles.h"
#include "../src/Simulation.h"
#include "DirectDynamicsReference.h"
#include <FML/tests/TestHelpers.h>
#include <fstream>
#include <iomanip>
using P = ParticleWithMass<ParticleBasic<3>>;
namespace D = SIMCODE::DYNAMICS_TEST;
struct DynamicsSimulation : NBodySimulation<3, P> {
    using Base = NBodySimulation<3, P>;
    using Base::Base;
    using Base::step;
};
std::vector<D::Particle> gather(DynamicsSimulation & simulation, std::vector<int> & owner) {
    auto & container = simulation.get_particles();
    std::vector<double> values(8 * container.get_npart_total());
    auto * p = container.get_particles_ptr();
    int valid = 1;
    for (size_t i = 0; i < container.get_npart(); ++i)
        valid &= p[i].id >= 1 && size_t(p[i].id) <= container.get_npart_total();
    FML::MinOverTasks(&valid);
    FML::assert_mpi(valid, "Dynamics particle ID outside catalogue");
    for (size_t i = 0; i < container.get_npart(); ++i) {
        const size_t j = size_t(p[i].id - 1);
        for (int d = 0; d < 3; ++d) {
            values[8 * j + d] = p[i].pos[d];
            values[8 * j + 3 + d] = p[i].vel[d];
        }
        values[8 * j + 6] = FML::ThisTask + 1;
        values[8 * j + 7] = 1;
    }
    FML::SumArrayOverTasks(values.data(), values.size());
    std::vector<D::Particle> result(container.get_npart_total());
    owner.resize(result.size());
    for (size_t i = 0; i < result.size(); ++i) {
        for (int d = 0; d < 3; ++d) {
            result[i].x[d] = values[8 * i + d];
            result[i].v[d] = values[8 * i + 3 + d];
        }
        owner[i] = int(values[8 * i + 6]) - 1;
        FML::assert_mpi(values[8 * i + 7] == 1 && owner[i] >= 0 && owner[i] < FML::NTasks,
                        "Missing or duplicated dynamics particle ID");
    }
    return result;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    if (argc != 6) {
        if (FML::ThisTask == 0)
            std::cerr << "Usage: benchmark_short_range_dynamics parameters.lua output-prefix fixture "
                         "simulation|reference steps|tolerance\n";
        return 1;
    }
    try {
        const bool reference = std::string(argv[4]) == "reference";
        FML::assert_mpi(reference || std::string(argv[4]) == "simulation", "Unknown dynamics mode");
        FML::assert_mpi(!reference || FML::NTasks == 1, "Direct ODE reference is serial test-only work");
        SIMCODE::ParameterMap param;
        SIMCODE::read_parameterfile<3>(param, argv[1]);
        auto cosmo = SIMCODE::make_cosmology(param.get<std::string>("cosmology_model"));
        auto grav = SIMCODE::make_gravity_model<3>(param.get<std::string>("gravity_model"), cosmo);
        FML::TESTS::quietly([&] {
            cosmo->read_parameters(param);
            cosmo->init();
            grav->read_parameters(param);
            grav->init();
        });
        SIMCODE::SimulationParameters params;
        params.read(param, grav->is_growth_scaledependent());
        SIMCODE::check_parameters_against_models<3>(params, *cosmo, *grav, false);
        FML::assert_mpi(grav->get_name() == "GR" && !params.simulation.use_cola &&
                            params.timestep.method == FML::NBODY::TimestepMethod::Quinn,
                        "This dynamics benchmark requires GR Quinn N-body evolution");
        const auto fixture = D::fixture(argv[3], cosmo->get_OmegaM());
        const int outputs = 16;
        FML::assert_mpi(std::abs(params.ic.a_initial() - fixture.begin) < 1e-14 &&
                            (!params.force.short_range.enabled ||
                             std::abs(params.force.short_range.softening_support - fixture.softening) < 1e-14),
                        "Fixture epoch/softening differs from the simulation input");
        std::ofstream states, checks;
        if (FML::ThisTask == 0) {
            states.open(std::string(argv[2]) + "-states.csv");
            checks.open(std::string(argv[2]) + "-checks.csv");
        }
        int writable = FML::ThisTask != 0 || (states && checks);
        FML::MinOverTasks(&writable);
        FML::assert_mpi(writable, "Cannot write dynamics outputs");
        if (FML::ThisTask == 0) {
            states << std::setprecision(17) << "sample,a,id,mass,x,y,z,vx,vy,vz,owner\n";
            checks << std::setprecision(17) << "sample,a,spatial_reference_error,rhs_evaluations,migrations\n";
        }
        auto save = [&](int sample, double a, const std::vector<D::Particle> & p, const std::vector<int> & owner) {
            if (FML::ThisTask == 0)
                for (size_t i = 0; i < p.size(); ++i) {
                    states << sample << ',' << a << ',' << i + 1 << ',' << p[i].mass;
                    for (double x : p[i].x)
                        states << ',' << x;
                    for (double v : p[i].v)
                        states << ',' << v;
                    states << ',' << owner[i] << '\n';
                }
        };
        if (reference) {
            const double tolerance = std::stod(argv[5]);
            FML::assert_mpi(tolerance > 0 && tolerance <= 1e-6, "Invalid direct-reference tolerance");
            D::DirectEvolution evolution(
                fixture,
                cosmo->get_OmegaM(),
                params.simulation.boxsize,
                [&](double a) { return cosmo->HoverH0_of_a(a); },
                tolerance);
            for (int sample = 0; sample <= outputs; ++sample) {
                const double a = fixture.begin + (fixture.end - fixture.begin) * sample / outputs;
                const auto p = evolution.advance(a);
                save(sample, a, p, std::vector<int>(p.size(), -1));
                const double delta = evolution.force_refinement_error();
                FML::TESTS::check(delta < 1e-10, "direct spatial-force refinement at output " + std::to_string(sample));
                checks << sample << ',' << a << ',' << delta << ',' << evolution.evaluations() << ",0\n";
            }
        } else {
            const int nsteps = std::stoi(argv[5]);
            FML::assert_mpi(nsteps >= outputs && nsteps % outputs == 0, "Steps must be a positive multiple of 16");
            DynamicsSimulation simulation(cosmo, grav);
            FML::TESTS::quietly([&] {
                simulation.read_parameters(param, params);
                simulation.init();
            });
            auto & container = simulation.get_particles();
            std::vector<P> all(fixture.particles.size());
            for (size_t i = 0; i < all.size(); ++i) {
                all[i].id = i + 1;
                all[i].mass = fixture.particles[i].mass;
                for (int d = 0; d < 3; ++d) {
                    const double x = fixture.particles[i].x[d];
                    all[i].pos[d] = x - std::floor(x);
                    all[i].vel[d] = fixture.particles[i].v[d];
                }
            }
            container.create(all, 128, [](P & p) {
                return p.pos[0] >= FML::xmin_domain && p.pos[0] < FML::xmax_domain;
            });
            std::vector<int> owner;
            auto old = gather(simulation, owner);
            auto unwrapped = fixture.particles;
            int migrations = 0;
            save(0, fixture.begin, unwrapped, owner);
            if (FML::ThisTask == 0)
                checks << "0," << fixture.begin << ",0,0,0\n";
            for (int sample = 1; sample <= outputs; ++sample) {
                const double low = fixture.begin + (fixture.end - fixture.begin) * (sample - 1) / outputs,
                             high = fixture.begin + (fixture.end - fixture.begin) * sample / outputs;
                const auto steps = simulation.time_stepping().steps(low, high, nsteps / outputs);
                for (const auto & step : steps) {
                    FML::TESTS::quietly([&] { simulation.step(step); });
                    const auto previous_owner = owner;
                    auto current = gather(simulation, owner);
                    for (size_t i = 0; i < current.size(); ++i) {
                        migrations += owner[i] != previous_owner[i];
                        for (int d = 0; d < 3; ++d) {
                            FML::assert_mpi(std::abs(current[i].v[d] * step.delta_time_drift) < .25,
                                            "Dynamics step too long for unambiguous unwrapping");
                            double dx = current[i].x[d] - old[i].x[d];
                            dx -= std::round(dx);
                            unwrapped[i].x[d] += dx;
                            unwrapped[i].v[d] = current[i].v[d];
                        }
                    }
                    old = std::move(current);
                }
                FML::TESTS::check(std::abs(steps.back().avel_new - high) < 1e-14 &&
                                      std::abs(steps.back().apos_new - high) < 1e-14,
                                  "synchronized output " + std::to_string(sample));
                save(sample, high, unwrapped, owner);
                if (FML::ThisTask == 0)
                    checks << sample << ',' << high << ",0,0," << migrations << '\n';
            }
        }
        int written = FML::ThisTask != 0 || (states.good() && checks.good());
        FML::MinOverTasks(&written);
        FML::assert_mpi(written, "Failed to save dynamics diagnostics");
        return FML::TESTS::summary("short-range dynamics " + std::string(argv[3]));
    } catch (const std::exception & e) {
        std::cerr << "Short-range dynamics: " << e.what() << '\n';
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
}
