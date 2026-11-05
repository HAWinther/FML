#include "../src/ShortRangeForce.h"
#include "../src/Particles.h"
#include <FML/NBody/NBody.h>
#include <FML/tests/TestHelpers.h>
#include <iostream>
#include "ShortRangeReference.h"
using FML::TESTS::check;
using FML::TESTS::throws;
using SIMCODE::TEST_REFERENCE::ewald;
namespace SR = FML::SHORT_RANGE_GRAVITY;
using P = ParticleWithMass<ParticleBasic<3>>;
using Container = FML::PARTICLE::MPIParticles<P>;
using Vector = SR::Vector;
double length(Vector x) { return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]); }
std::vector<P> catalogue() {
    std::vector<Vector> x{{.013, .12, .11},
                          {.042, .13, .12},
                          {.989, .11, .12},
                          {.512, .52, .53},
                          {.529, .54, .55},
                          {.751, .25, .25},
                          {.731, .7, .7}};
    std::vector<P> p(x.size());
    for (size_t i = 0; i < p.size(); ++i) {
        p[i].id = i + 1;
        p[i].mass = 1 + .3 * i;
        for (int d = 0; d < 3; ++d) {
            p[i].pos[d] = x[i][d];
            p[i].vel[d] = .1 * (i + d + 1);
        }
    }
    return p;
}
Container local(std::vector<P> all) {
    Container p;
    p.create(all, all.size() + 32, [](P & p) { return p.pos[0] >= FML::xmin_domain && p.pos[0] < FML::xmax_domain; });
    return p;
}
// Independent Ewald image/Fourier sum, using a reference split unrelated to the solver's chosen split.
std::vector<Vector> mesh_force(Container & p, SIMCODE::ShortRangeSettings settings, int nmesh, int order) {
    using Grid = FML::GRID::FFTWGrid<3>;
    const auto shifts = FML::INTERPOLATION::generate_interlacing_shifts<3>(order, 1. / nmesh);
    auto * a = p.get_particles_ptr();
    std::vector<Vector> saved(p.get_npart()), out(p.get_npart());
    for (size_t i = 0; i < p.get_npart(); ++i)
        for (int d = 0; d < 3; ++d)
            saved[i][d] = a[i].pos[d];
    const auto halo = FML::INTERPOLATION::get_extra_slices("PCS");
    for (auto shift : shifts) {
        for (size_t i = 0; i < p.get_npart(); ++i)
            for (int d = 0; d < 3; ++d) {
                a[i].pos[d] = saved[i][d] + shift[d];
                if (d > 0 || FML::NTasks == 1)
                    a[i].pos[d] -= std::floor(a[i].pos[d]);
            }
        Grid density(nmesh, halo.first, halo.second + 1);
        FML::INTERPOLATION::particles_to_grid(a, p.get_npart(), p.get_npart_total(), density, "PCS");
        density.fftw_r2c();
        SIMCODE::filter_short_range_long_force(density, settings, 100.);
        std::array<Grid, 3> force;
        FML::NBODY::ForceOptions opts;
        opts.deconvolve = true;
        FML::NBODY::compute_force_from_density_fourier<3>(density, force, "PCS", .45, opts);
        for (auto & f : force)
            f.communicate_boundaries();
        FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions_apply<3, P>(force,
                                                                                      a,
                                                                                      p.get_npart(),
                                                                                      "PCS",
                                                                                      [&](size_t i, const auto & f) {
                                                                                          for (int d = 0; d < 3; ++d)
                                                                                              out[i][d] -=
                                                                                                  f[d] / shifts.size();
                                                                                          return 0.;
                                                                                      });
        for (size_t i = 0; i < p.get_npart(); ++i)
            for (int d = 0; d < 3; ++d)
                a[i].pos[d] = saved[i][d];
    }
    return out;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    auto all = catalogue();
    auto p = local(all);
    int empty = p.get_npart() == 0;
    FML::SumOverTasks(&empty);
    check(FML::NTasks != 4 || empty > 0, "four-rank sparse fixture exercises an empty owned-particle rank");
    SIMCODE::ShortRangeSettings settings;
    settings.enabled = true;
    settings.split_scale = 4;
    settings.softening_support = .2;
    settings.cutoff_tolerance = 1e-10;
    settings.table_points = 0;
    double M = 0;
    for (auto & x : all)
        M += x.mass;
    SR::Kernel kernel(settings.options(100., .45 / (4 * M_PI * M)));
    std::vector<SR::Particle> global(all.size()), owned(p.get_npart());
    for (size_t i = 0; i < all.size(); ++i)
        global[i] = {{all[i].pos[0], all[i].pos[1], all[i].pos[2]}, all[i].mass, std::uint64_t(all[i].id)};
    for (size_t i = 0; i < p.get_npart(); ++i) {
        auto & x = p.get_particles_ptr()[i];
        owned[i] = {{x.pos[0], x.pos[1], x.pos[2]}, x.mass, std::uint64_t(x.id)};
    }
    const auto reference = SR::direct_sum(owned, global, kernel);
    for (std::string backend : {"cell", "tree"}) {
        settings.backend = backend;
        settings.opening_angle = 0;
        auto actual = SIMCODE::short_range_acceleration(p, settings, 100, .45);
        double err = 0;
        for (size_t i = 0; i < owned.size(); ++i)
            for (int d = 0; d < 3; ++d)
                err = std::max(err, std::abs(actual.acceleration[i][d] - reference.acceleration[i][d]));
        FML::MaxOverTasks(&err);
        check(err < 1e-12,
              "physical scale conversion/global mass and empty-rank ghosts match replicated direct reference: " +
                  backend);
    }
    settings.backend = "cell";
    auto before = all;
    SIMCODE::kick_short_range(p, settings, 100., .45, .013);
    double err = 0;
    for (size_t i = 0; i < owned.size(); ++i) {
        auto & x = p.get_particles_ptr()[i];
        for (int d = 0; d < 3; ++d)
            err =
                std::max(err,
                         std::abs(x.vel[d] - (before[size_t(x.id - 1)].vel[d] + .013 * reference.acceleration[i][d])));
    }
    FML::MaxOverTasks(&err);
    check(err < 1e-13, "short kick adds acceleration with plus sign and does not rescale existing velocity");
    auto old = p.get_particles_ptr();
    std::vector<Vector> v(p.get_npart());
    for (size_t i = 0; i < v.size(); ++i)
        for (int d = 0; d < 3; ++d)
            v[i][d] = old[i].vel[d];
    SIMCODE::kick_short_range(p, settings, 100., .45, 0);
    bool same = true;
    for (size_t i = 0; i < v.size(); ++i)
        for (int d = 0; d < 3; ++d)
            same &= v[i][d] == old[i].vel[d];
    check(same, "zero-length short kick is a no-op");
    auto disabled_kick = settings;
    disabled_kick.enabled = false;
    SIMCODE::kick_short_range(p, disabled_kick, 100., .45, .5);
    same = true;
    for (size_t i = 0; i < v.size(); ++i)
        for (int d = 0; d < 3; ++d)
            same &= v[i][d] == old[i].vel[d];
    check(same, "disabled short-range kick is a no-op even for a nonzero timestep");
    std::vector<Vector> truth(owned.size());
    for (size_t i = 0; i < owned.size(); ++i)
        for (auto & q : global)
            if (q.id != owned[i].id) {
                Vector dx;
                for (int d = 0; d < 3; ++d) {
                    dx[d] = owned[i].position[d] - q.position[d];
                    dx[d] -= std::round(dx[d]);
                }
                auto force = ewald(dx);
                for (int d = 0; d < 3; ++d)
                    truth[i][d] += .45 / (4 * M_PI * M) * q.mass * force[d];
            }
    auto error = [&](const std::vector<Vector> & f) {
        double a = 0, b = 0;
        for (size_t i = 0; i < f.size(); ++i)
            for (int d = 0; d < 3; ++d) {
                a += std::pow(f[i][d] - truth[i][d], 2);
                b += truth[i][d] * truth[i][d];
            }
        FML::SumOverTasks(&a);
        FML::SumOverTasks(&b);
        return std::sqrt(a / b);
    };
    std::vector<double> errors;
    for (int n : {16, 32, 64}) {
        auto f = mesh_force(p, settings, n, 1);
        for (size_t i = 0; i < f.size(); ++i)
            for (int d = 0; d < 3; ++d)
                f[i][d] += reference.acceleration[i][d];
        errors.push_back(error(f));
        if (FML::ThisTask == 0)
            std::cout << "MESH_ERROR " << n << ' ' << errors.back() << '\n';
        Vector momentum{};
        for (size_t i = 0; i < f.size(); ++i)
            for (int d = 0; d < 3; ++d)
                momentum[d] += owned[i].mass * f[i][d];
        FML::SumArrayOverTasks(momentum.data(), 3);
        check(length(momentum) < 1e-9, "mass-weighted combined force conserves momentum mesh=" + std::to_string(n));
    }
    check(errors.back() < .005 && errors.back() < .2 * errors.front(),
          "off-grid deposition/interpolation plus short correction converges to independent Ewald forces");
    auto interlaced = mesh_force(p, settings, 32, 2);
    for (size_t i = 0; i < owned.size(); ++i)
        for (int d = 0; d < 3; ++d)
            interlaced[i][d] += reference.acceleration[i][d];
    double ei = error(interlaced);
    check(ei < .01, "interlaced long force plus one unshifted short contribution matches Ewald");
    auto disabled = settings;
    disabled.enabled = false;
    auto pm = mesh_force(p, disabled, 16, 1);
    double epm = error(pm);
    check(errors.front() < epm * .5,
          "coarse combined force improves on unsplit PM at the same mesh in clustered fixture");
    if (FML::ThisTask == 0)
        std::cout << "INTERLACED_ERROR " << ei << " PM_ONLY_ERROR " << epm << '\n';
    auto scaled = settings;
    scaled.split_scale *= 2;
    scaled.softening_support *= 2;
    auto scale_force = SIMCODE::short_range_acceleration(p, scaled, 200, .45);
    double scale_error = 0;
    for (size_t i = 0; i < owned.size(); ++i)
        for (int d = 0; d < 3; ++d)
            scale_error =
                std::max(scale_error, std::abs(scale_force.acceleration[i][d] - reference.acceleration[i][d]));
    FML::MaxOverTasks(&scale_error);
    check(scale_error < 1e-13, "rescaling physical box and lengths preserves unit-box acceleration");
    check(throws([&] {
              auto bad = settings;
              bad.split_scale = -1;
              bad.validate(100.);
          }),
          "invalid physical split scale is rejected");
    check(throws([&] {
              auto bad = settings;
              if (FML::ThisTask == 0)
                  bad.split_scale = -1;
              SIMCODE::short_range_acceleration(p, bad, 100, .45);
          }),
          "one-rank invalid kernel is rejected collectively");
    return FML::TESTS::summary("SimulationCode short-range force");
}
