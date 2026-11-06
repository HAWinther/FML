// Standalone accuracy experiment, deliberately outside the fast default test suite.
#include "../src/GravityModel_GR.h"
#include "../src/Particles.h"
#include "ClusteredForceReference.h"
#include "ShortRangeReference.h"
#include <FML/tests/TestHelpers.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
namespace R = SIMCODE::CLUSTER_REFERENCE;
using Vector = R::Vector;
using P = ParticleWithMass<ParticleBasic<3>>;
using Container = FML::PARTICLE::MPIParticles<P>;
using Grid = FML::GRID::FFTWGrid<3>;
using FML::TESTS::check;
constexpr double box = 100., epoch = .8;
double uniform(std::uint64_t key) {
    key += 0x9e3779b97f4a7c15ULL;
    key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
    key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
    key ^= key >> 31;
    return (double(key >> 11) + .5) * 0x1.0p-53;
}
std::vector<P> catalogue(int n, const std::string & fixture) {
    std::vector<P> all(n);
    const double pi = std::acos(-1.), scale = .0125, maxr = .14;
    const double maximum = std::pow(maxr / std::sqrt(maxr * maxr + scale * scale), 3);
    for (int i = 0; i < n; ++i) {
        const double u = maximum * uniform(8 * i), r = scale / std::sqrt(std::pow(u, -2. / 3) - 1);
        const double z = 2 * uniform(8 * i + 1) - 1, phi = 2 * pi * uniform(8 * i + 2);
        Vector center{.5, .5, .5};
        if (fixture == "merger")
            center[0] += (i % 2 == 0 ? -.045 : .045);
        if (fixture == "boundary")
            center[0] = .995;
        if (fixture == "shifted")
            center = {.5 + 1. / 96, .5 + .007, .5 + .013};
        Vector offset{r * std::sqrt(1 - z * z) * std::cos(phi), r * std::sqrt(1 - z * z) * std::sin(phi), r * z};
        all[i].id = i + 1;
        all[i].mass = .5 + uniform(8 * i + 3);
        for (int d = 0; d < 3; ++d) {
            all[i].pos[d] = center[d] + offset[d];
            all[i].pos[d] -= std::floor(all[i].pos[d]);
            all[i].vel[d] = 0;
        }
    }
    return all;
}
std::shared_ptr<Cosmology> cosmology() {
    auto c = std::make_shared<Cosmology>();
    FML::UTILS::ParameterMap p;
    p["cosmology_OmegaCDM"] = .25;
    p["cosmology_Omegab"] = .05;
    p["cosmology_OmegaMNu"] = 0.;
    p["cosmology_neutrino_masses_eV"] = std::vector<double>{};
    p["cosmology_OmegaK"] = 0.;
    p["cosmology_h"] = .7;
    p["cosmology_As"] = 2.1e-9;
    p["cosmology_ns"] = .96;
    p["cosmology_kpivot_mpc"] = .05;
    p["cosmology_Neffective"] = 3.046;
    p["cosmology_TCMB_kelvin"] = 2.7255;
    FML::TESTS::quietly([&] {
        c->read_parameters(p);
        c->init();
    });
    return c;
}
struct Case {
    std::string name;
    int mesh, interlace;
    double split, softening;
    std::string backend;
    double theta;
    bool enabled;
    int table{0};
};
std::vector<Vector> combined(Container & p, GravityModelGR<3> & model, const Case & test) {
    SIMCODE::ShortRangeSettings
        settings{test.enabled, test.backend, test.split, test.softening, 1e-8, test.theta, test.table};
    FML::NBODY::ForceOptions options;
    options.deconvolve = true;
    const auto halo = FML::INTERPOLATION::get_extra_slices("PCS");
    const auto shifts = FML::INTERPOLATION::generate_interlacing_shifts<3>(test.interlace, 1. / test.mesh);
    auto * a = p.get_particles_ptr();
    std::vector<Vector> saved(p.get_npart()), out(p.get_npart());
    for (size_t i = 0; i < p.get_npart(); ++i)
        for (int d = 0; d < 3; ++d)
            saved[i][d] = a[i].pos[d];
    for (const auto & shift : shifts) {
        for (size_t i = 0; i < p.get_npart(); ++i)
            for (int d = 0; d < 3; ++d) {
                a[i].pos[d] = saved[i][d] + shift[d];
                // MPI deposition accepts the neighboring shifted x slab; do not migrate shifted copies.
                if (d > 0 || FML::NTasks == 1)
                    a[i].pos[d] -= std::floor(a[i].pos[d]);
            }
        Grid density(test.mesh, halo.first, halo.second + 1);
        FML::INTERPOLATION::particles_to_grid(a, p.get_npart(), p.get_npart_total(), density, "PCS");
        density.fftw_r2c();
        std::array<Grid, 3> force;
        model.compute_force_for_step(epoch, model.H0_hmpc * box, density, "PCS", force, options, settings);
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
    // Exercise the actual model-owned kick. Each benchmark case starts from zero velocity.
    for (size_t i = 0; i < p.get_npart(); ++i)
        for (int d = 0; d < 3; ++d)
            a[i].vel[d] = 0;
    model.kick_short_range(p, settings, epoch, box, 1.);
    std::vector<double> global(3 * p.get_npart_total(), 0.);
    for (size_t i = 0; i < p.get_npart(); ++i)
        for (int d = 0; d < 3; ++d)
            global[3 * (a[i].id - 1) + d] = out[i][d] + a[i].vel[d];
    FML::SumArrayOverTasks(global.data(), global.size());
    std::vector<Vector> result(p.get_npart_total());
    for (size_t i = 0; i < result.size(); ++i)
        for (int d = 0; d < 3; ++d)
            result[i][d] = global[3 * i + d];
    return result;
}
double relative_rms(const std::vector<Vector> & a, const std::vector<Vector> & b) {
    double error = 0, truth = 0;
    for (size_t i = 0; i < a.size(); ++i)
        for (int d = 0; d < 3; ++d) {
            error += std::pow(a[i][d] - b[i][d], 2);
            truth += b[i][d] * b[i][d];
        }
    return std::sqrt(error / truth);
}
double percentile(std::vector<double> x, double p) {
    std::sort(x.begin(), x.end());
    return x[size_t(std::ceil(p * (x.size() - 1)))];
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    if (argc < 2 || argc > 4) {
        if (FML::ThisTask == 0)
            std::cerr << "Usage: benchmark_clustered_force output-prefix [particles=2048] [targets=128]\n";
        return 1;
    }
    const int n = argc > 2 ? std::stoi(argv[2]) : 2048, ntarget = argc > 3 ? std::stoi(argv[3]) : 128;
    FML::assert_mpi(n >= 128 && n <= 16384 && ntarget >= 16 && ntarget <= n &&
                        (FML::NTasks == 1 || FML::NTasks == 2 || FML::NTasks == 4),
                    "Use 128..16384 particles, 16..N targets and 1/2/4 ranks");
    std::vector<size_t> targets;
    for (int i = 0; i < ntarget; ++i) {
        const size_t low = size_t(i) * n / ntarget, high = size_t(i + 1) * n / ntarget;
        targets.push_back(low + size_t(uniform(700000 + i) * (high - low)));
    }
    auto c = cosmology();
    GravityModelGR<3> model(c);
    std::ofstream summary, details;
    if (FML::ThisTask == 0) {
        summary.open(std::string(argv[1]) + "-summary.csv");
        details.open(std::string(argv[1]) + "-particles.csv");
    }
    int writable = FML::ThisTask != 0 || (summary && details);
    FML::MinOverTasks(&writable);
    FML::assert_mpi(writable, "Cannot open benchmark outputs");
    if (FML::ThisTask == 0) {
        summary << std::setprecision(17)
                << "fixture,case,n,ranks,mesh,interlace,split_mpch,h_mpch,theta,table,rms,p50,p95,p99,max,angle_p95_"
                   "deg,momentum_ratio,reference_delta,softened_pairs,seconds\n";
        details << std::setprecision(17)
                << "fixture,case,id,radius,mass,ref_x,ref_y,ref_z,force_x,force_y,force_z,relative_error,scaled_"
                   "absolute_error,angle_deg\n";
    }
    const std::vector<Case> cases{{"pm", 32, 1, 4, .25, "cell", 0, false},
                                  {"mesh16", 16, 1, 4, .25, "cell", 0, true},
                                  {"mesh32", 32, 1, 4, .25, "cell", 0, true},
                                  {"mesh64", 64, 1, 4, .25, "cell", 0, true},
                                  {"interlaced", 32, 2, 4, .25, "cell", 0, true},
                                  {"tree0", 64, 1, 4, .25, "tree", 0, true},
                                  {"tree03", 64, 1, 4, .25, "tree", .3, true},
                                  {"tree06", 64, 1, 4, .25, "tree", .6, true},
                                  {"split3", 64, 1, 3, .25, "cell", 0, true},
                                  {"split5", 64, 1, 5, .25, "cell", 0, true},
                                  {"soft05", 64, 1, 4, .5, "cell", 0, true},
                                  {"table", 64, 1, 4, .25, "cell", 0, true, 16384}};
    std::vector<Vector> halo_reference, halo_force;
    for (const std::string fixture : {"halo", "merger", "boundary", "shifted"}) {
        auto all = catalogue(n, fixture);
        std::vector<R::Source> sources;
        double mass = 0;
        for (const auto & p : all) {
            sources.push_back({{p.pos[0], p.pos[1], p.pos[2]}, p.mass});
            mass += p.mass;
        }
        const double normalization = 1.5 * c->get_OmegaM() * epoch / (4 * std::acos(-1.) * mass);
        Container p;
        p.create(all, n + 32, [](P & p) { return p.pos[0] >= FML::xmin_domain && p.pos[0] < FML::xmax_domain; });
        // Small replicated catalogues/references are intentional test-only O(N^2) work.
        R::Ewald reference(sources, 5, 10), refined(sources, 7, 16);
        std::vector<Vector> truth(ntarget), fine(ntarget), soft(ntarget);
        for (int t = 0; t < ntarget; ++t) {
            truth[t] = reference.force(targets[t], .25 / box);
            fine[t] = refined.force(targets[t], .25 / box);
            soft[t] = refined.force(targets[t], .5 / box);
            for (int d = 0; d < 3; ++d) {
                truth[t][d] *= normalization;
                fine[t][d] *= normalization;
                soft[t][d] *= normalization;
            }
        }
        const double reference_delta = relative_rms(truth, fine);
        check(reference_delta < 1e-10, "independent Ewald split/truncation convergence: " + fixture);
        // Cross-check the structure-factor implementation against the older pairwise oracle,
        // including an independently integrated compact-softening correction.
        Vector pairwise{};
        const auto ti = targets.front();
        for (size_t j = 0; j < all.size(); ++j)
            if (j != ti) {
                const auto dx = R::displacement(sources[ti].x, sources[j].x);
                const double r = R::norm(dx);
                auto f = SIMCODE::TEST_REFERENCE::ewald(dx);
                const double correction = r < .25 / box ? R::softened_coefficient(r, .25 / box) - 1 / (r * r * r) : 0;
                for (int d = 0; d < 3; ++d)
                    pairwise[d] += normalization * all[j].mass * (f[d] - correction * dx[d]);
            }
        check(R::norm({pairwise[0] - fine[0][0], pairwise[1] - fine[0][1], pairwise[2] - fine[0][2]}) <
                  1e-10 * std::max(1., R::norm(fine[0])),
              "structure-factor oracle matches pairwise oracle: " + fixture);
        std::vector<Vector> cell64;
        double coarse = 0, finer = 0, pm = 0;
        for (const auto & test : cases) {
            const auto start = std::chrono::steady_clock::now();
            auto force = combined(p, model, test);
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            FML::MaxOverTasks(&seconds);
            const auto & expected = test.name == "soft05" ? soft : fine;
            size_t softened_pairs = 0;
            for (auto ti : targets)
                for (size_t j = 0; j < all.size(); ++j)
                    if (j != ti && R::norm(R::displacement(sources[ti].x, sources[j].x)) < test.softening / box)
                        ++softened_pairs;
            if (n >= 1024 && ntarget >= 64)
                check(softened_pairs > 0, "sample exercises compact softening: " + fixture + "/" + test.name);
            double ref2 = 0;
            for (auto f : expected)
                ref2 += R::norm(f) * R::norm(f);
            const double scale = std::sqrt(ref2 / ntarget);
            std::vector<Vector> sampled;
            std::vector<double> rel, angle;
            for (int t = 0; t < ntarget; ++t) {
                const size_t i = targets[t];
                sampled.push_back(force[i]);
                Vector delta, cross;
                for (int d = 0; d < 3; ++d)
                    delta[d] = force[i][d] - expected[t][d];
                for (int d = 0; d < 3; ++d)
                    cross[d] = force[i][(d + 1) % 3] * expected[t][(d + 2) % 3] -
                               force[i][(d + 2) % 3] * expected[t][(d + 1) % 3];
                double dot = 0;
                for (int d = 0; d < 3; ++d)
                    dot += force[i][d] * expected[t][d];
                const double error = R::norm(delta) / std::max(R::norm(expected[t]), 1e-3 * scale);
                const double degrees = std::atan2(R::norm(cross), dot) * 180 / std::acos(-1.);
                rel.push_back(error);
                angle.push_back(degrees);
                Vector center{.5, .5, .5};
                if (fixture == "merger")
                    center[0] += (i % 2 == 0 ? -.045 : .045);
                if (fixture == "boundary")
                    center[0] = .995;
                if (fixture == "shifted")
                    center = {.5 + 1. / 96, .507, .513};
                if (FML::ThisTask == 0) {
                    details << fixture << ',' << test.name << ',' << i + 1 << ','
                            << R::norm(R::displacement(sources[i].x, center)) << ',' << all[i].mass;
                    for (double x : expected[t])
                        details << ',' << x;
                    for (double x : force[i])
                        details << ',' << x;
                    details << ',' << error << ',' << R::norm(delta) / scale << ',' << degrees << '\n';
                }
            }
            const double rms = relative_rms(sampled, expected);
            Vector momentum{};
            double normsum = 0;
            for (size_t i = 0; i < force.size(); ++i) {
                for (int d = 0; d < 3; ++d)
                    momentum[d] += all[i].mass * force[i][d];
                normsum += all[i].mass * R::norm(force[i]);
            }
            const double drift = R::norm(momentum) / normsum;
            if (FML::ThisTask == 0) {
                summary << fixture << ',' << test.name << ',' << n << ',' << FML::NTasks << ',' << test.mesh << ','
                        << test.interlace << ',' << test.split << ',' << test.softening << ',' << test.theta << ','
                        << test.table << ',' << rms << ',' << percentile(rel, .5) << ',' << percentile(rel, .95) << ','
                        << percentile(rel, .99) << ',' << percentile(rel, 1) << ',' << percentile(angle, .95) << ','
                        << drift << ',' << reference_delta << ',' << softened_pairs << ',' << seconds << '\n';
                summary.flush();
                std::cout << "CLUSTER " << fixture << ' ' << test.name << " rms=" << rms
                          << " p99=" << percentile(rel, .99) << " momentum=" << drift << std::endl;
            }
            check(std::isfinite(rms) && std::isfinite(drift), "finite force diagnostics: " + fixture + "/" + test.name);
            if (test.name == "tree03" || test.name == "tree06") {
                const double budget = test.name == "tree03" ? .005 : .02;
                check(rms < budget && percentile(rel, .99) < 5 * budget,
                      "finite-opening tree accuracy budget: " + fixture + "/" + test.name);
                if (n >= 1024 && test.name == "tree06")
                    check(relative_rms(force, cell64) > 1e-12,
                          "finite-opening fixture exercises tree approximation: " + fixture);
            }
            if (test.name == "pm")
                pm = rms;
            if (test.name == "mesh16")
                coarse = rms;
            if (test.name == "mesh64") {
                finer = rms;
                cell64 = force;
                if (fixture == "halo") {
                    halo_force = sampled;
                    halo_reference = fine;
                }
            }
            if (test.name == "tree0")
                check(relative_rms(force, cell64) < 1e-12, "exact tree/cell agreement: " + fixture);
            if (test.backend == "cell" || test.theta == 0)
                check(drift < 1e-10, "exact-backend momentum balance: " + fixture + "/" + test.name);
            if (test.name == "mesh64" || test.name == "interlaced" || test.name == "split3" || test.name == "split5" ||
                test.name == "soft05" || test.name == "table")
                check(rms < .005 && percentile(rel, .99) < .02,
                      "resolved combined-force accuracy: " + fixture + "/" + test.name);
            if (fixture == "shifted" && test.name == "mesh64") {
                check(relative_rms(fine, halo_reference) < 1e-10,
                      "reference is invariant under global mesh-relative translation");
                check(relative_rms(sampled, halo_force) < .005,
                      "resolved force is insensitive to global mesh-relative translation");
            }
        }
        check(finer < .2 * coarse && finer < .01 * pm, "mesh convergence and improvement over unsplit PM: " + fixture);
    }
    return FML::TESTS::summary("clustered PM plus short-range force");
}
