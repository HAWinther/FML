#ifndef SIMCODE_SHORT_RANGE_COLA_REFERENCE_H
#define SIMCODE_SHORT_RANGE_COLA_REFERENCE_H
// Included after DiagnosticSimulation: test-only manufactured nonzero COLA frames.
#include <FML/tests/TestHelpers.h>
inline void check_short_range_cola_frame(DiagnosticSimulation & simulation,
                                         std::shared_ptr<GravityModel<3>> grav,
                                         const SIMCODE::SimulationParameters & params) {
    using Vector = SIMCODE::TEST_REFERENCE::Vector;
    using Grid = FML::GRID::FFTWGrid<3>;
    const double pi = std::acos(-1.), box = params.simulation.boxsize, aini = params.ic.a_initial(), a = .5;
    auto cosmo = grav->get_cosmo();
    auto & frame = simulation.frame();
    const bool enabled = params.simulation.use_cola, scaled = enabled && params.simulation.use_scaledependent_cola;
    const int order = params.simulation.cola_LPT_order;
    std::vector<Vector> positions{{.013, .12, .11},
                                  {.042, .13, .12},
                                  {.989, .11, .12},
                                  {.512, .52, .53},
                                  {.529, .54, .55},
                                  {.251, .25, .25},
                                  {.731, .7, .7}};
    std::vector<DiagnosticParticle> all(positions.size());
    double mass = 0;
    for (size_t i = 0; i < all.size(); ++i) {
        auto & p = all[i];
        p.id = i + 1;
        p.mass = 1 + .3 * i;
        mass += p.mass;
        for (int d = 0; d < 3; ++d) {
            p.pos[d] = positions[i][d];
            p.vel[d] = .1 * (i + d + 1);
            p.q[d] = double(((d + 2) * int(i) + d + 1) % 16) / 16;
            p.Psi_1LPT[d] = d == 0 ? .002 * std::sin(2 * pi * p.q[0]) : 0;
            p.Psi_2LPT[d] = d == 1 ? .0007 * std::sin(4 * pi * p.q[1]) : 0;
            if (scaled) {
                p.Psi_1LPT[d] = .015 * (i + d + 1);
                p.Psi_2LPT[d] = -.011 * (i + d + 1);
            }
        }
    }
    frame.lattice_offset = 0;
    if (scaled) {
        auto potential = [&](int axis, double k, double amplitude) {
            Grid phi(16, 1, 1);
            for (auto i : phi.get_real_range()) {
                const auto x = phi.get_real_position(phi.get_coord_from_index(i));
                phi.set_real_from_index(i, -amplitude / k * std::cos(k * x[axis]));
            }
            phi.fftw_r2c();
            return phi;
        };
        frame.phi_1LPT_ini_fourier = potential(0, 2 * pi, .002);
        frame.phi_2LPT_ini_fourier = potential(1, 4 * pi, .0007);
    }
    auto psi = [&](size_t i, int n, int d) {
        if (!enabled || n > order)
            return 0.;
        if (!scaled)
            return double(n == 1 ? all[i].Psi_1LPT[d] : all[i].Psi_2LPT[d]);
        return n == 1 ? (d == 0 ? .002 * std::sin(2 * pi * all[i].q[0]) : 0.) :
                        (d == 1 ? .0007 * std::sin(4 * pi * all[i].q[1]) : 0.);
    };
    auto kH0 = [&](int n) { return scaled ? 2 * pi * n / (box * grav->H0_hmpc) : 0.; };
    auto growth = [&](int n, double epoch) {
        return n == 1 ? grav->get_D_1LPT(epoch, kH0(n)) : grav->get_D_2LPT(epoch, kH0(n));
    };
    auto velocity = [&](int n, double epoch) {
        const double f = n == 1 ? grav->get_f_1LPT(epoch, kH0(n)) : grav->get_f_2LPT(epoch, kH0(n));
        return epoch * epoch * cosmo->HoverH0_of_a(epoch) * f * growth(n, epoch) / growth(n, aini);
    };
    const bool eds = grav->get_name() == "GR" && std::abs(cosmo->get_OmegaM() - 1) < 1e-14;
    auto frame_force = [&](int n) {
        // Independent closed EdS solution. Other models use a numerical derivative
        // of their public frame velocity, not COLA's force/source-factor routines.
        if (eds)
            return n == 1 ? 1.5 * a * a / aini : 5 * a * a * a / (aini * aini);
        const double h = a * 1e-4;
        const double derivative =
            (-velocity(n, a + 2 * h) + 8 * velocity(n, a + h) - 8 * velocity(n, a - h) + velocity(n, a - 2 * h)) /
            (12 * h);
        return a * a * a * cosmo->HoverH0_of_a(a) * derivative;
    };
    auto drift_factor = [&](int n, double end) {
        if (eds)
            return n == 1 ? (end - a) / aini : (end * end - a * a) / (aini * aini);
        return (growth(n, end) - growth(n, a)) / growth(n, aini);
    };
    const auto model = grav->short_range_interaction(a);
    std::vector<Vector> acceleration(all.size());
    for (size_t i = 0; i < all.size(); ++i)
        for (size_t j = 0; j < all.size(); ++j)
            if (i != j) {
                Vector dx;
                for (int d = 0; d < 3; ++d) {
                    dx[d] = positions[i][d] - positions[j][d];
                    dx[d] -= std::round(dx[d]);
                }
                const double r = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
                FML::assert_mpi(r > params.force.short_range.softening_support / box,
                                "Frame fixture must lie outside spline support");
                auto force = SIMCODE::TEST_REFERENCE::ewald(dx);
                for (int d = 0; d < 3; ++d)
                    force[d] *= model.range_mpch == 0 ? model.muinf : model.mu0;
                if (model.range_mpch > 0) {
                    const double lambda = model.range_mpch / box;
                    Vector first{}, second{};
                    for (int x = -2; x <= 2; ++x)
                        for (int y = -2; y <= 2; ++y)
                            for (int z = -2; z <= 2; ++z) {
                                Vector image{dx[0] + x, dx[1] + y, dx[2] + z};
                                const double r =
                                    std::sqrt(image[0] * image[0] + image[1] * image[1] + image[2] * image[2]);
                                const double coefficient =
                                    -(model.muinf - model.mu0) * (1 + r / lambda) * std::exp(-r / lambda) / (r * r * r);
                                for (int d = 0; d < 3; ++d) {
                                    second[d] += coefficient * image[d];
                                    if (std::abs(x) <= 1 && std::abs(y) <= 1 && std::abs(z) <= 1)
                                        first[d] += coefficient * image[d];
                                }
                            }
                    double delta = 0;
                    for (int d = 0; d < 3; ++d) {
                        delta = std::max(delta, std::abs(first[d] - second[d]));
                        force[d] += second[d];
                    }
                    FML::assert_mpi(delta < 1e-9, "Yukawa image reference has not converged");
                }
                for (int d = 0; d < 3; ++d)
                    acceleration[i][d] += 1.5 * cosmo->get_OmegaM() * a / (4 * pi * mass) * all[j].mass * force[d];
            }
    for (int scenario = 0; scenario < 3; ++scenario) {
        const bool kick = scenario != 2;
        const double end = scenario == 1 ? a : .51, dtkick = kick ? .001 : 0., dtdrift = scenario == 1 ? 0. : .002;
        DiagnosticSimulation::Step step{a, end, .49, kick ? .505 : .49, dtdrift, dtkick, .7};
        auto & container = simulation.get_particles();
        container.create(all, 64, [](auto & p) { return p.pos[0] >= FML::xmin_domain && p.pos[0] < FML::xmax_domain; });
        std::vector<Vector> expected_velocity(all.size()), expected_position(all.size());
        for (size_t i = 0; i < all.size(); ++i)
            for (int d = 0; d < 3; ++d) {
                double residual = (kick ? .7 : 1.) * all[i].vel[d] + dtkick * acceleration[i][d], drift = 0;
                for (int n = 1; n <= order && enabled; ++n) {
                    residual -= dtkick * frame_force(n) * psi(i, n, d);
                    drift += drift_factor(n, end) * psi(i, n, d);
                }
                expected_velocity[i][d] = residual;
                expected_position[i][d] = positions[i][d] + drift + dtdrift * residual;
            }
        simulation.step(step);
        double error_v = 0, error_x = 0;
        int identities = 1;
        auto * p = container.get_particles_ptr();
        for (size_t j = 0; j < container.get_npart(); ++j) {
            const size_t i = size_t(p[j].id - 1);
            identities &= i < all.size();
            if (i >= all.size())
                continue;
            identities &= p[j].mass == all[i].mass;
            for (int d = 0; d < 3; ++d) {
                identities &= p[j].q[d] == all[i].q[d];
                error_v = std::max(error_v, std::abs(p[j].vel[d] - expected_velocity[i][d]));
                double dx = p[j].pos[d] - expected_position[i][d];
                dx -= std::round(dx);
                error_x = std::max(error_x, std::abs(dx));
            }
        }
        FML::MaxOverTasks(&error_v);
        FML::MaxOverTasks(&error_x);
        FML::TESTS::check(identities, "frame step preserves IDs, masses and Lagrangian positions");
        FML::TESTS::check(error_v < 5e-8 && error_x < 1e-8,
                          "combined body force and one frame correction: scenario " + std::to_string(scenario));
        {
            ScopedLPTVelocity<3, DiagnosticParticle> physical(frame, container, step.avel_new);
            p = container.get_particles_ptr();
            double error = 0;
            for (size_t j = 0; j < container.get_npart(); ++j) {
                const size_t i = size_t(p[j].id - 1);
                for (int d = 0; d < 3; ++d) {
                    double expected = expected_velocity[i][d];
                    for (int n = 1; n <= order && enabled; ++n)
                        expected += velocity(n, step.avel_new) * psi(i, n, d);
                    error = std::max(error, std::abs(p[j].vel[d] - expected));
                }
            }
            FML::MaxOverTasks(&error);
            FML::TESTS::check(error < 5e-8, "physical velocity restoration uses the same frame");
        }
        if (FML::ThisTask == 0)
            std::cout << "COLA_FRAME_ERROR " << scenario << ' ' << error_v << ' ' << error_x << '\n';
    }
    if (FML::TESTS::summary("short-range COLA frame") != 0)
        throw std::runtime_error("Short-range COLA frame reference failed");
}
#endif
