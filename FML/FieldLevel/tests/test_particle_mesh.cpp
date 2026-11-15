#include <FML/FieldLevel/ParticleMesh.h>
#include <FML/ParticleTypes/SimpleParticle.h>
#include <FML/tests/TestHelpers.h>
#ifdef USE_GSL
#include <FML/NBody/NBody.h>
#endif
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <fstream>
#include <iomanip>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
using Vec = std::array<double, 3>;
using Grid = FML::GRID::FFTWGrid<3>;
static FF::Configuration config(int n = 8, int force = 16) {
    FF::Configuration c;
    c.mesh = n; c.density_mesh = force; c.boxsize = 1;
    c.initial_cutoff = c.bias_cutoff = c.likelihood_cutoff = 8.;
    c.evolution = FF::Evolution::Linear;
    return c;
}
static FF::Spectrum spectrum() {
    FF::Spectrum s;
    s.evaluate = [](double, const auto &) { return FF::Power{1., {}}; };
    return s;
}
static FF::PMSettings settings(double om = 1., int steps = 8) {
    FF::PMSettings s;
    s.background.omega_m = om; s.scale_factors.clear();
    for (int j = 0; j <= steps; ++j) s.scale_factors.push_back(.1 + .7 * j / steps);
    return s;
}
template <class Model>
static Field initial(const Model & m, double amplitude = .05) {
    Field s(m.cells());
    // P=1,L=1 implies delta=white*sqrt(N^3). Keep the physical amplitude fixed.
    const double norm = std::sqrt(double(m.layout().global_cells));
    for (size_t q = 0; q < s.size(); ++q) {
        auto c = m.layout().global_coordinate(q);
        s[q] = amplitude * (std::cos(2 * M_PI * c[0] / m.configuration().mesh) +
                           .6 * std::sin(2 * M_PI * c[1] / m.configuration().mesh) +
                           .4 * std::cos(2 * M_PI * c[2] / m.configuration().mesh)) / norm;
    }
    return s;
}
static Field global(const FF::Model::Layout & l, const Field & local) {
    Field out(l.global_cells);
    for (size_t q = 0; q < local.size(); ++q) out[l.global_cell_id(q)] = local[q];
    FML::SumArrayOverTasks(out.data(), out.size());
    return out;
}
static double error(const Field & a, const Field & b) {
    if (a.size() != b.size()) return HUGE_VAL;
    double out = 0;
    for (size_t i = 0; i < a.size(); ++i) out = std::max(out, std::abs(a[i] - b[i]));
    return out;
}
struct Snapshot { Field density, delta; std::array<Field, 3> displacement, momentum; };
template <class Result>
static Snapshot snapshot(const Result & r) {
    Snapshot s;
    s.density = global(r.density_layout, r.matter_density);
    s.delta = global(r.particle_layout, r.initial_density);
    for (int d = 0; d < 3; ++d) {
        s.displacement[d] = global(r.particle_layout, r.displacement[d]);
        s.momentum[d] = global(r.particle_layout, r.momentum[d]);
    }
    return s;
}
static double error(const Snapshot & a, const Snapshot & b) {
    double e = std::max(error(a.density, b.density), error(a.delta, b.delta));
    for (int d = 0; d < 3; ++d)
        e = std::max({e, error(a.displacement[d], b.displacement[d]), error(a.momentum[d], b.momentum[d])});
    return e;
}
template <class Backend>
static Snapshot exercise(const FML::Domain & domain, const std::string & label, double om = .3, int n = 8, int nf = 16) {
    using Model = FF::ParticleMeshForwardT<Backend>;
    const auto c = config(n, nf);
    Model m(c, spectrum(), settings(om), domain);
    auto w = m.workspace();
    check(FML::TESTS::throws([&] { w.result(); }), label + " rejects uninitialized result");
    auto s = initial(m), p = m.initial_parameters();
    const auto & r = m.evolve(s, p, w);
    const auto baseline = snapshot(r);
    bool conservation = r.diagnostics.size() == m.settings().scale_factors.size();
    for (const auto & d : r.diagnostics) {
        conservation &= std::abs(d.mass_error) < 1e-12 && std::isfinite(d.density_rms);
        for (double momentum : d.mean_momentum) conservation &= std::abs(momentum) < 1e-12;
    }
    check(conservation, label + " mass and total momentum conserved at every node");
    check(std::abs(r.a - .8) < 1e-15 && r.particle_layout.global_cells == size_t(n) * n * n &&
              r.density_layout.global_cells == size_t(nf) * nf * nf,
          label + " explicit time and particle/force layouts");
    check(error(baseline, snapshot(m.evolve(s, p, w))) == 0, label + " repeated evolution resets workspace exactly");
    auto saved = FML::domain;
    FML::set_domain(FML::Domain::uniform_slab(FML::NTasks));
    auto fresh = m.workspace();
    check(error(baseline, snapshot(m.evolve(s, p, fresh))) < 2e-13, label + " copied domain survives global changes");
    FML::set_domain(saved);
    auto moved = std::move(fresh);
    check(error(baseline, snapshot(m.evolve(s, p, moved))) < 2e-13, label + " movable workspace preserves routing");
    auto bad = s; if (FML::ThisTask == 0) bad.pop_back();
    check(FML::TESTS::throws([&] { m.evolve(bad, p, w); }), label + " one-rank malformed IC rejected collectively");
    check(FML::TESTS::throws([&] { w.result(); }), label + " failed evolution invalidates result");
    bad = s; if (FML::ThisTask == 0 && !bad.empty()) bad[0] = HUGE_VAL;
    check(FML::TESTS::throws([&] { m.evolve(bad, p, w); }), label + " one-rank nonfinite IC rejected collectively");
    auto other = m;
    check(FML::TESTS::throws([&] { other.evolve(s, p, w); }), label + " foreign workspace rejected");
    return baseline;
}

// Independent dense CIC/DFT oracle. Test-only replicated particles, no production
// deposition, Poisson, FFT, gather or timestep quadrature is used here.
template <class Visit>
static void dense_stencil(Vec x, int n, Visit visit) {
    int cell[3]; double f[3];
    for (int d = 0; d < 3; ++d) {
        x[d] -= std::floor(x[d]); const double u = n * x[d];
        cell[d] = int(u); f[d] = u - cell[d];
    }
    for (int i = 0; i < 8; ++i) {
        size_t q = 0; double weight = 1;
        for (int d = 0; d < 3; ++d) {
            const int bit = (i >> d) & 1;
            q = n * q + (cell[d] + bit) % n;
            weight *= bit ? f[d] : 1 - f[d];
        }
        visit(q, weight);
    }
}
static std::vector<Vec> dense_force(const std::vector<Vec> & x, int n, Field & delta, bool compensated = false) {
    const int total = n * n * n;
    delta.assign(total, -1.);
    for (auto p : x) dense_stencil(p, n, [&](size_t q, double v) { delta[q] += v * total / x.size(); });
    std::vector<Vec> force(total);
    for (int k = 0; k < total; ++k) {
        const int coordinate[3] = {k / (n * n), k / n % n, k % n};
        int wave[3]; double norm = 0;
        for (int d = 0; d < 3; ++d) { wave[d] = coordinate[d] <= n / 2 ? coordinate[d] : coordinate[d] - n; norm += wave[d] * wave[d]; }
        if (norm == 0) continue;
        std::complex<double> value = 0;
        for (int j = 0; j < total; ++j) {
            const double phase = 2 * M_PI / n * (wave[0] * (j / (n * n)) + wave[1] * (j / n % n) + wave[2] * (j % n));
            value += delta[j] * std::exp(std::complex<double>(0, -phase)) / double(total);
        }
        if (compensated) {
            double response = 1.;
            for (int component : wave) if (component) {
                const double angle = M_PI * component / n;
                response *= std::pow(std::sin(angle) / angle, 4);
            }
            value /= response;
        }
        for (int j = 0; j < total; ++j) {
            const double phase = 2 * M_PI / n * (wave[0] * (j / (n * n)) + wave[1] * (j / n % n) + wave[2] * (j % n));
            const auto z = value * std::exp(std::complex<double>(0, phase));
            for (int d = 0; d < 3; ++d)
                if (std::abs(wave[d]) != n / 2) force[j][d] += (std::complex<double>(0, wave[d] / (2 * M_PI * norm)) * z).real();
        }
    }
    std::vector<Vec> out(x.size());
    for (size_t q = 0; q < x.size(); ++q)
        dense_stencil(x[q], n, [&](size_t j, double weight) { for (int d = 0; d < 3; ++d) out[q][d] += weight * force[j][d]; });
    return out;
}
static void direct_reference(int n = 4, int force = 4, bool compensated = false) {
    auto options = settings(1., 3);
    if (compensated) options.force_window = FF::PMForceWindow::CICCompensated;
    FF::ParticleMeshForward model(config(n, force), spectrum(), options);
    auto w = model.workspace(); auto s = initial(model, .2);
    const auto actual = snapshot(model.evolve(s, {}, w));
    std::vector<Vec> x(n * n * n), p(x.size()), shift(x.size());
    for (int q = 0; q < n * n * n; ++q) {
        const int c[3] = {q / (n * n), q / n % n, q % n};
        // Analytic displacement of the three independent initial fundamental modes.
        const Vec psi = {-.2 * std::sin(2 * M_PI * c[0] / n) / (2 * M_PI),
                         .12 * std::cos(2 * M_PI * c[1] / n) / (2 * M_PI),
                         -.08 * std::sin(2 * M_PI * c[2] / n) / (2 * M_PI)};
        for (int d = 0; d < 3; ++d) {
            shift[q][d] = .1 * psi[d]; x[q][d] = double(c[d]) / n + shift[q][d];
            p[q][d] = std::sqrt(.1) * shift[q][d];
        }
    }
    Field delta;
    auto f = dense_force(x, force, delta, compensated);
    const auto nodes = settings(1., 3).scale_factors;
    for (size_t j = 1; j < nodes.size(); ++j) {
        const double a = nodes[j - 1], b = nodes[j], mid = (a + b) / 2;
        for (size_t q = 0; q < x.size(); ++q) for (int d = 0; d < 3; ++d) {
            p[q][d] += 3 * (std::sqrt(mid) - std::sqrt(a)) * f[q][d];
            const double dx = 2 * (1 / std::sqrt(a) - 1 / std::sqrt(b)) * p[q][d];
            x[q][d] += dx; shift[q][d] += dx;
        }
        f = dense_force(x, force, delta, compensated);
        for (size_t q = 0; q < x.size(); ++q) for (int d = 0; d < 3; ++d)
            p[q][d] += 3 * (std::sqrt(b) - std::sqrt(mid)) * f[q][d];
    }
    double e = error(actual.density, delta);
    for (size_t q = 0; q < x.size(); ++q) for (int d = 0; d < 3; ++d)
        e = std::max({e, std::abs(actual.displacement[d][q] - shift[q][d]), std::abs(actual.momentum[d][q] - p[q][d])});
    check(e < 3e-13, "PM matches independent analytic IC + dense CIC/direct DFT multi-step EdS oracle");
    if (FML::ThisTask == 0) std::cout << "PM_DIRECT_ERROR " << std::setprecision(17) << e << '\n';
}
#ifdef USE_GSL
static void native_reference() {
    auto c = config(); auto controls = settings(.3, 8);
    FF::ParticleMeshForward m(c, spectrum(), controls);
    FF::DistributedLinearModel seed(c, spectrum()); auto iw = seed.workspace();
    auto s = initial(m, .3); auto ic = seed.reconstruct(s, seed.initial_parameters(), iw, 0.);
    std::vector<COLAParticle<3>> local(m.cells());
    const double a = controls.scale_factors.front(), D = controls.background.growth(a);
    for (size_t q = 0; q < local.size(); ++q) {
        auto coord = m.layout().global_coordinate(q); local[q].id = m.layout().global_cell_id(q);
        for (int d = 0; d < 3; ++d) {
            local[q].Pos[d] = double(coord[d]) / c.mesh + D * ic.displacement[d][q];
            local[q].Pos[d] -= std::floor(local[q].Pos[d]);
            local[q].Vel[d] = a * a * controls.background.E(a) * controls.background.growth_rate(a) * D * ic.displacement[d][q];
        }
    }
    FML::PARTICLE::MPIParticles<COLAParticle<3>> particles;
    particles.create(local, 2 * local.size() + 32, FML::PARTICLE::Input::Distributed);
    auto density = [&] {
        Grid delta(c.density_mesh, 0, 1);
        FML::INTERPOLATION::particles_to_grid(particles.get_particles_ptr(), particles.get_npart(),
                                               particles.get_npart_total(), delta, "CIC");
        return delta;
    };
    const auto & nodes = controls.scale_factors;
    for (size_t j = 0; j < nodes.size(); ++j) {
        auto delta = density(); delta.fftw_r2c(); std::array<Grid, 3> f;
        FML::NBODY::compute_force_from_density_fourier<3>(delta, f, "CIC", 1.);
        const double old = j == 0 ? nodes[j] : .5 * (nodes[j - 1] + nodes[j]);
        const double next = j + 1 == nodes.size() ? nodes[j] : .5 * (nodes[j] + nodes[j + 1]);
        FML::NBODY::KickParticles<3>(f, particles, controls.background.kick(old, next), "CIC");
        if (j + 1 < nodes.size()) FML::NBODY::DriftParticles<3>(particles, controls.background.drift(nodes[j], nodes[j + 1]));
    }
    auto w = m.workspace(); const auto & result = m.evolve(s, {}, w); const auto actual = snapshot(result);
    double e = 0;
    for (auto & particle : particles) for (int d = 0; d < 3; ++d) {
        const size_t q = particle.id;
        const int coord = d == 0 ? int(q / (c.mesh * c.mesh)) : d == 1 ? int(q / c.mesh % c.mesh) : int(q % c.mesh);
        double dx = particle.Pos[d] - double(coord) / c.mesh - actual.displacement[d][q];
        dx -= std::round(dx);
        e = std::max({e, std::abs(dx), std::abs(particle.Vel[d] - actual.momentum[d][q])});
    }
    auto final = density(); Field packed;
    for (auto i : final.get_real_range()) packed.push_back(final.get_real_from_index(i));
    e = std::max(e, error(global(result.density_layout, packed), actual.density));
    e = m.maximum(e);
    check(e < 2e-11, "PM matches native migrating MPIParticles/CIC/force/KickParticles/DriftParticles pipeline");
    if (FML::ThisTask == 0) std::cout << "PM_NATIVE_ERROR " << std::setprecision(17) << e << '\n';
}
#endif
int main(int argc, char ** argv) {
    auto slab = FML::Domain::uniform_slab(FML::NTasks); FML::set_domain(slab);
    FF::PMBackground eds{1.}, lcdm{.3};
    check(eds.growth(.1) == .1 && eds.growth_rate(.2) == 1 && std::abs(lcdm.growth(1.) - 1) < 1e-15,
          "background exact EdS and normalized LCDM growth");
    check(std::abs(lcdm.growth(.1) - .1283184315445443) < 1e-8 && std::abs(lcdm.growth_rate(1.) - .5127962477302378) < 1e-8,
          "background LCDM growth and rate match independent high-accuracy integral values");
    FF::ParticleMeshForward uniform(config(8, 8), spectrum(), settings()); auto uw = uniform.workspace();
    auto u = uniform.evolve(Field(uniform.cells(), 0), {}, uw);
    double norm = 0; for (auto x : u.matter_density) norm = std::max(norm, std::abs(x));
    for (const auto & d : u.displacement) for (double x : d) norm = std::max(norm, std::abs(x));
    check(uniform.maximum(norm) < 1e-14, "uniform lattice remains exactly homogeneous and stationary");
    check(FML::TESTS::throws([&] { auto options = settings(); options.force_window = static_cast<FF::PMForceWindow>(9);
        FF::ParticleMeshForward bad(config(), spectrum(), options); }), "invalid force window rejected");
    const auto baseline = exercise<Grid>(slab, "slab");
    direct_reference();
    direct_reference(8, 4);
    direct_reference(8, 4, true);
    const auto dense = exercise<Grid>(slab, "dense-slab", .3, 8, 4);
#ifdef USE_GSL
    native_reference();
#endif
    check(FML::TESTS::throws([&] { auto s = settings(); s.scale_factors[2] = s.scale_factors[1]; FF::ParticleMeshForward bad(config(), spectrum(), s); }),
          "non-increasing timestep schedule rejected");
    check(FML::TESTS::throws([&] { auto s = settings(); s.background.omega_m = -1; FF::ParticleMeshForward bad(config(), spectrum(), s); }),
          "invalid fixed cosmology rejected");
    check(FML::TESTS::throws([&] { auto c = config(8, 4); c.initial_cutoff = 13.; FF::ParticleMeshForward bad(c, spectrum(), settings()); }),
          "IC support above coarser force Nyquist rejected");
    check(FML::TESTS::throws([&] { FF::ParticleMeshForward bad(config(8, 2), spectrum(), settings()); }),
          "force mesh below minimum rejected");
    check(FML::TESTS::throws([&] { auto c = config(); c.growth_rate = .5; FF::ParticleMeshForward bad(c, spectrum(), settings()); }),
          "initial redshift-space response rejected");
    {
        auto power = spectrum();
        power.parameters = {{"log_amplitude", 0., 0., 1.}};
        power.evaluate = [](double, const auto & p) { return FF::Power{std::exp(p[0]), {1.}}; };
        FF::ParticleMeshForward m(config(), power, settings()); auto w = m.workspace(); auto s = initial(m);
        const auto first = snapshot(m.evolve(s, {0.}, w));
        const auto second = snapshot(m.evolve(s, {std::log(4.)}, w));
        auto twice = first.delta; for (auto & x : twice) x *= 2;
        check(m.parameters().size() == 1 && error(twice, second.delta) < 1e-13 && error(first, second) > 1e-5,
              "only spectrum parameters are exposed and their transfer normalization is respected");
        check(FML::TESTS::throws([&] { m.evolve(s, {}, w); }), "wrong spectrum parameter dimension rejected");
    }
#ifdef USE_MPI
    if (FML::NTasks > 1) {
        check(FML::TESTS::throws([&] { auto s = settings(); if (FML::ThisTask == 0) s.scale_factors.pop_back(); FF::ParticleMeshForward bad(config(), spectrum(), s); }),
              "rank-dependent schedule size rejected before trajectory collectives");
        check(FML::TESTS::throws([&] { auto s = settings(); if (FML::ThisTask == 0) s.background.omega_m = .4; FF::ParticleMeshForward bad(config(), spectrum(), s); }),
              "rank-dependent background rejected");
    }
    for (auto shape : {std::array<int, 2>{FML::NTasks, 1}, {1, FML::NTasks}}) {
        if (shape[0] > 8 || shape[1] > 5) continue;
        auto domain = FML::Domain::uniform_pencil(shape[0], shape[1]); FML::set_domain(domain);
        if (shape[1] <= 3)
            check(error(exercise<FML::GRID::PencilGrid<3>>(domain, "dense-pencil", .3, 8, 4), dense) < 2e-12,
                  "oversampled slab/pencil forward equality");
        else
            check(FML::TESTS::throws([&] { FF::ParticleMeshForwardT<FML::GRID::PencilGrid<3>> bad(config(8,4), spectrum(), settings(), domain); }),
                  "force Fourier process grid too narrow rejects before workspace allocation");
        auto pencil = exercise<FML::GRID::PencilGrid<3>>(domain, "pencil");
        const double e = error(pencil, baseline);
        check(e < 2e-12, "slab/pencil trajectory and output equality");
        if (FML::ThisTask == 0) std::cout << "PM_BACKEND_ERROR " << e << '\n';
    }
    if (FML::NTasks == 4) {
        auto domain = FML::Domain::uniform_pencil(2, 2); FML::set_domain(domain);
        check(error(exercise<FML::GRID::PencilGrid<3>>(domain, "2x2"), baseline) < 2e-12, "2x2 pencil equals slab");
        auto uneven = exercise<FML::GRID::PencilGrid<3>>(domain, "uneven-N10", 1., 10, 20);
        check(!uneven.density.empty(), "unequal pencil particle and force boxes supported");
        auto incompatible = FML::Domain::pencil({0., .4, 1.}, {0., .5, 1.});
        check(FML::TESTS::throws([&] {
            FF::ParticleMeshForwardT<FML::GRID::PencilGrid<3>> bad(config(10, 16), spectrum(), settings(), incompatible);
        }), "force-grid domain edges must align on cells");
    }
#endif
    FML::set_domain(slab);
    if (argc > 1) {
        if (FML::ThisTask == 0) {
            std::ofstream data(std::string(argv[1]) + ".snapshot"); data << std::setprecision(17);
            for (const auto * v : {&baseline.density, &baseline.delta}) for (double x : *v) data << x << '\n';
            for (const auto & v : baseline.displacement) for (double x : v) data << x << '\n';
            for (const auto & v : baseline.momentum) for (double x : v) data << x << '\n';
        }
        // Bounded fixed-mesh convergence evidence; CSV output is root-only.
        std::vector<Snapshot> outputs;
        for (int steps : {4, 8, 16, 32, 64}) {
            FF::ParticleMeshForward m(config(), spectrum(), settings(1., steps)); auto w = m.workspace();
            outputs.push_back(snapshot(m.evolve(initial(m, .3), {}, w)));
        }
        std::ofstream file; if (FML::ThisTask == 0) { file.open(argv[1]); file << "steps,max_displacement_error\n"; }
        std::vector<double> errors;
        for (size_t j = 0; j + 1 < outputs.size(); ++j) {
            double e = 0; for (int d = 0; d < 3; ++d) e = std::max(e, error(outputs[j].displacement[d], outputs.back().displacement[d]));
            errors.push_back(e); if (FML::ThisTask == 0) file << (4 << j) << ',' << std::setprecision(17) << e << '\n';
        }
        check(errors[0] > 2.5 * errors[1] && errors[1] > 2.5 * errors[2] && errors[2] > 2.5 * errors[3],
              "fixed-mesh nonlinear displacement has second-order timestep refinement");
    }
    return FML::TESTS::summary("FieldLevel PM forward");
}
