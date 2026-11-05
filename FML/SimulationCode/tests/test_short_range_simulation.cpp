// End-to-end diagnostic driver: run the real SimulationCode pipeline and save full-precision final states.
#include "../src/ModelCompatibility.h"
#include "../src/Models.h"
#include "../src/ParameterSchema.h"
#include "../src/Particles.h"
#include "../src/Simulation.h"
#include <fstream>
#include <iomanip>
#include "ShortRangeReference.h"
struct DiagnosticSimulation : NBodySimulation<3, ParticleBasic<3>> {
    using Base = NBodySimulation<3, ParticleBasic<3>>;
    using Base::Base;
    using Base::Step;
    using Base::step;
};
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    if (argc != 3 && argc != 4) {
        if (FML::ThisTask == 0)
            std::cerr << "Usage: test_short_range_simulation parameters.lua state-prefix\n";
        return 1;
    }
    try {
        SIMCODE::ParameterMap param;
        SIMCODE::read_parameterfile<3>(param, argv[1]);
        auto cosmo = SIMCODE::make_cosmology(param.get<std::string>("cosmology_model"));
        cosmo->read_parameters(param);
        cosmo->init();
        auto grav = SIMCODE::make_gravity_model<3>(param.get<std::string>("gravity_model"), cosmo);
        grav->read_parameters(param);
        grav->init();
        SIMCODE::SimulationParameters params;
        params.read(param, grav->is_growth_scaledependent());
        SIMCODE::check_parameters_against_models<3>(params, *cosmo, *grav, false);
        DiagnosticSimulation simulation(cosmo, grav);
        simulation.read_parameters(param, params);
        simulation.init();
        if (argc == 4) {
            // Isolate one real pipeline kick against a reference independent of the mesh and adapter.
            using Vector = SIMCODE::TEST_REFERENCE::Vector;
            std::vector<Vector> positions{{.013, .12, .11},
                                          {.042, .13, .12},
                                          {.989, .11, .12},
                                          {.512, .52, .53},
                                          {.529, .54, .55},
                                          {.251, .25, .25},
                                          {.731, .7, .7}};
            std::vector<ParticleBasic<3>> all(positions.size());
            for (size_t i = 0; i < all.size(); ++i) {
                all[i].id = i + 1;
                for (int d = 0; d < 3; ++d) {
                    all[i].pos[d] = positions[i][d];
                    all[i].vel[d] = .1 * (i + d + 1);
                }
            }
            auto & container = simulation.get_particles();
            container.create(all, 64, [](auto & p) {
                return p.pos[0] >= FML::xmin_domain && p.pos[0] < FML::xmax_domain;
            });
            DiagnosticSimulation::Step step{.5, .5, .5, .5, 0., .001, .7};
            simulation.step(step);
            double err2 = 0, ref2 = 0;
            for (size_t i = 0; i < container.get_npart(); ++i) {
                auto & p = container.get_particles_ptr()[i];
                const size_t id = size_t(p.id - 1);
                Vector reference{};
                for (size_t j = 0; j < positions.size(); ++j)
                    if (j != id) {
                        Vector dx;
                        for (int d = 0; d < 3; ++d) {
                            dx[d] = positions[id][d] - positions[j][d];
                            dx[d] -= std::round(dx[d]);
                        }
                        auto force = SIMCODE::TEST_REFERENCE::ewald(dx);
                        for (int d = 0; d < 3; ++d)
                            reference[d] += force[d];
                    }
                for (int d = 0; d < 3; ++d) {
                    reference[d] *= 1.5 * cosmo->get_OmegaM() * .5 / (4 * M_PI * positions.size());
                    const double actual = (p.vel[d] - .7 * all[id].vel[d]) / .001;
                    err2 += std::pow(actual - reference[d], 2);
                    ref2 += reference[d] * reference[d];
                    if (p.pos[d] != positions[id][d])
                        throw std::runtime_error("Isolated kick unexpectedly moved a position");
                }
            }
            FML::SumOverTasks(&err2);
            FML::SumOverTasks(&ref2);
            const double relative = std::sqrt(err2 / ref2);
            if (FML::ThisTask == 0)
                std::cout << "REAL_STEP_EWALD_ERROR " << relative << '\n';
            if (!(relative < .002))
                throw std::runtime_error("Real PM/short-range kick failed independent Ewald check");
        } else
            simulation.run();
        auto & particles = simulation.get_particles();
        auto * p = particles.get_particles_ptr();
        std::ofstream file(std::string(argv[2]) + "." + std::to_string(FML::ThisTask) + ".txt");
        file << std::setprecision(17);
        for (size_t i = 0; i < particles.get_npart(); ++i) {
            file << p[i].id;
            for (double x : p[i].pos)
                file << ' ' << x;
            for (double v : p[i].vel)
                file << ' ' << v;
            file << '\n';
        }
        if (!file)
            throw std::runtime_error("Cannot write diagnostic state");
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Short-range simulation: " << e.what() << '\n';
        return 1;
    }
}
