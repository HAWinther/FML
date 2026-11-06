#ifndef SIMCODE_DIRECT_DYNAMICS_REFERENCE_H
#define SIMCODE_DIRECT_DYNAMICS_REFERENCE_H
#include "DynamicsFixtures.h"
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>
#include <functional>
#include <memory>
namespace SIMCODE::DYNAMICS_TEST {
    // Test-only adaptive RKF45. No mesh, production kernel or timestep-coefficient calls.
    class DirectEvolution {
        struct Context {
            std::vector<double> masses;
            std::function<double(double)> expansion;
            double omega_m, h, mass{0};
            size_t evaluations{0};
            std::string error;
            std::vector<Vector> forces(const double * y, double alpha = 3.5, int modes = 8) const {
                std::vector<CLUSTER_REFERENCE::Source> sources(masses.size());
                for (size_t i = 0; i < masses.size(); ++i) {
                    sources[i].mass = masses[i];
                    for (int d = 0; d < 3; ++d) {
                        const double x = y[6 * i + d];
                        sources[i].x[d] = x - std::floor(x);
                    }
                }
                CLUSTER_REFERENCE::Ewald ewald(sources, alpha, modes);
                std::vector<Vector> force(masses.size());
                for (size_t i = 0; i < masses.size(); ++i)
                    force[i] = ewald.force(i, h);
                return force;
            }
            static int rhs(double a, const double * y, double * dy, void * data) {
                auto & c = *static_cast<Context *>(data);
                try {
                    const auto force = c.forces(y);
                    const double time = 1 / (a * a * a * c.expansion(a));
                    const double normalization = 1.5 * c.omega_m * a / (4 * std::acos(-1.) * c.mass);
                    for (size_t i = 0; i < c.masses.size(); ++i)
                        for (int d = 0; d < 3; ++d) {
                            dy[6 * i + d] = y[6 * i + 3 + d] * time;
                            dy[6 * i + 3 + d] = force[i][d] * normalization * time;
                        }
                    ++c.evaluations;
                    return GSL_SUCCESS;
                } catch (const std::exception & e) {
                    c.error = e.what();
                    return GSL_EBADFUNC;
                }
            }
        } context;
        gsl_odeiv2_system system;
        std::unique_ptr<gsl_odeiv2_driver, decltype(&gsl_odeiv2_driver_free)> driver{nullptr, &gsl_odeiv2_driver_free};
        std::vector<double> state;
        double epoch;

      public:
        DirectEvolution(const Fixture & f,
                        double omega_m,
                        double box,
                        std::function<double(double)> expansion,
                        double tolerance)
            : system{&Context::rhs, nullptr, 6 * f.particles.size(), &context}, epoch(f.begin) {
            context.expansion = std::move(expansion);
            context.omega_m = omega_m;
            context.h = f.softening / box;
            for (const auto & p : f.particles) {
                context.masses.push_back(p.mass);
                context.mass += p.mass;
                for (double x : p.x)
                    state.push_back(x);
                for (double v : p.v)
                    state.push_back(v);
            }
            gsl_set_error_handler_off();
            driver.reset(gsl_odeiv2_driver_alloc_y_new(&system, gsl_odeiv2_step_rkf45, 1e-5, tolerance, tolerance));
            if (!driver)
                throw std::runtime_error("Cannot allocate direct-reference integrator");
        }
        std::vector<Particle> advance(double a) {
            if (a > epoch) {
                const int status = gsl_odeiv2_driver_apply(driver.get(), &epoch, a, state.data());
                if (status != GSL_SUCCESS)
                    throw std::runtime_error("Direct-reference integration failed: " + context.error + " " +
                                             gsl_strerror(status));
            }
            std::vector<Particle> result(context.masses.size());
            for (size_t i = 0; i < result.size(); ++i) {
                result[i].mass = context.masses[i];
                for (int d = 0; d < 3; ++d) {
                    result[i].x[d] = state[6 * i + d];
                    result[i].v[d] = state[6 * i + 3 + d];
                }
            }
            return result;
        }
        double force_refinement_error() const {
            const auto first = context.forces(state.data()), second = context.forces(state.data(), 5, 10);
            double error = 0, scale = 0;
            for (size_t i = 0; i < first.size(); ++i)
                for (int d = 0; d < 3; ++d) {
                    error += std::pow(first[i][d] - second[i][d], 2);
                    scale += second[i][d] * second[i][d];
                }
            return std::sqrt(error / scale);
        }
        size_t evaluations() const { return context.evaluations; }
    };
} // namespace SIMCODE::DYNAMICS_TEST
#endif
