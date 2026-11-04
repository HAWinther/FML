#include <FML/GlassIC/GlassIC.h>
#include <FML/GlassIC/LoadIO.h>
#include <FML/tests/TestHelpers.h>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdio>
using namespace FML::GLASSIC;
using FML::TESTS::check;
// Direct summation over particles, no mesh deposition/window correction; small validation catalogues only.
static double structure(Load & l, std::array<int, 3> wave) {
    double re = 0, im = 0;
    for (size_t i = 0; i < l.particles.get_npart(); ++i) {
        const auto & p = l.particles.get_particles_ptr()[i];
        double phase = 0;
        for (int d = 0; d < 3; ++d)
            phase += 2 * std::acos(-1.) * wave[d] * p.position[d];
        re += std::cos(phase);
        im += std::sin(phase);
    }
    FML::SumOverTasks(&re);
    FML::SumOverTasks(&im);
    return (re * re + im * im) / l.particles.get_npart_total(); // Poisson expectation 1.
}
static std::vector<Particle> all_particles(Load & l) {
    std::vector<Particle> p;
    if (l.particles.get_npart())
        p.assign(l.particles.get_particles_ptr(), l.particles.get_particles_ptr() + l.particles.get_npart());
    return FML::COMMUNICATION::gather_records_checked(p, true);
}
// Independent periodic Ewald acceleration with alpha=6 and unit total mass, G=1/(4pi).
// Test softening is far below all pair distances. Real images and reciprocal waves converge exponentially.
static Vector ewald(const Particle & target, const std::vector<Particle> & all) {
    const double pi = std::acos(-1.), alpha = 6.;
    Vector a{};
    for (const auto & p : all)
        for (int nx = -1; nx <= 1; ++nx)
            for (int ny = -1; ny <= 1; ++ny)
                for (int nz = -1; nz <= 1; ++nz) {
                    if (p.id == target.id)
                        continue;
                    Vector dx{p.position[0] - target.position[0] + nx,
                              p.position[1] - target.position[1] + ny,
                              p.position[2] - target.position[2] + nz};
                    const double r = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
                    const double f =
                        (std::erfc(alpha * r) + 2 * alpha * r / std::sqrt(pi) * std::exp(-alpha * alpha * r * r)) /
                        (4 * pi * all.size() * r * r * r);
                    for (int d = 0; d < 3; ++d)
                        a[d] -= f * dx[d];
                }
    for (int nx = -10; nx <= 10; ++nx)
        for (int ny = -10; ny <= 10; ++ny)
            for (int nz = -10; nz <= 10; ++nz) {
                Vector k{2 * pi * nx, 2 * pi * ny, 2 * pi * nz};
                const double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                if (k2 == 0)
                    continue;
                double s = 0;
                for (const auto & p : all) {
                    double phase = 0;
                    for (int d = 0; d < 3; ++d)
                        phase += k[d] * (p.position[d] - target.position[d]);
                    s += std::sin(phase);
                }
                for (int d = 0; d < 3; ++d)
                    a[d] -= k[d] * s * std::exp(-k2 / (4 * alpha * alpha)) / (all.size() * k2);
            }
    return a;
}
static void remove_load(const std::string & prefix) {
    std::remove(detail::shard(prefix, FML::ThisTask).c_str());
    if (FML::ThisTask == 0)
        std::remove((prefix + ".manifest").c_str());
}
struct TestProvider : FML::INITIALCONDITIONS::TransferProvider {
    bool bad_gauge = false, bad_power = false;
    double h() const override { return .7; }
    FML::INITIALCONDITIONS::TransferRange range() const override { return {1e-5, 100, .01, 1}; }
    FML::INITIALCONDITIONS::Gauge density_gauge() const override {
        return bad_gauge ? FML::INITIALCONDITIONS::Gauge::Newtonian : FML::INITIALCONDITIONS::Gauge::SynchronousCDM;
    }
    bool has_density(FML::INITIALCONDITIONS::Species) const override { return true; }
    double density_transfer(FML::INITIALCONDITIONS::Species, double, double) const override { return 2.; }
    double primordial_power(double) const override { return bad_power ? NAN : 3.; }
};
int main(int argc, char ** argv) {
    std::cout << std::setprecision(17);
    if (argc == 3 && std::string(argv[1]) == "--write") {
        auto l = poisson_load(257, 12345, 250);
        save_load(l, argv[2]);
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--read") {
        auto l = load_load(argv[2], 13);
        auto expected = poisson_load(257, 12345, 250);
        bool good = l.particles.get_npart() == expected.particles.get_npart();
        for (size_t i = 0; good && i < l.particles.get_npart(); ++i)
            good &= l.particles.get_particles_ptr()[i].position == expected.particles.get_particles_ptr()[i].position &&
                    l.particles.get_particles_ptr()[i].id == expected.particles.get_particles_ptr()[i].id;
        check(good, "cross-rank-count load readback matches ID-keyed seed exactly");
        return FML::TESTS::summary("GlassIC cross-rank IO");
    }
    const int mesh = 48;
    auto load = poisson_load(257, 42, 100);
    bool valid = true;
    for (size_t i = 0; i < load.particles.get_npart(); ++i) {
        const auto & p = load.particles.get_particles_ptr()[i];
        FML::RANDOM::COUNTER_BASED::Stream rng(42, p.id, 0x474c4153u);
        for (int d = 0; d < 3; ++d)
            valid &= p.position[d] == rng.uniform() && p.q[d] == p.position[d] && p.velocity[d] == 0;
        valid &= p.position[0] >= FML::xmin_domain && p.position[0] < FML::xmax_domain;
    }
    check(valid && load.particles.get_npart_total() == 257,
          "seed positions, IDs and ownership independent of rank count");
    check(FML::TESTS::throws([] { poisson_load(0, 1); }), "zero load rejected collectively");
    auto tiny = poisson_load(1, 11);
    Force one(mesh, 1);
    auto af = one(tiny);
    auto rr = residuals(tiny, af);
    check(rr.force_rms < 2e-7, "one periodic particle has zero net self force (including empty particle slabs)");
    auto pair = poisson_load(2, 1);
    for (size_t i = 0; i < pair.particles.get_npart(); ++i) {
        auto & p = pair.particles.get_particles_ptr()[i];
        p.position = p.q = {.5 + (p.id ? .01 : -.01), .4, .3};
    }
    detail::migrate(pair.particles);
    Force pf(mesh, 2);
    auto pa = pf(pair);
    valid = true;
    for (size_t i = 0; i < pa.size(); ++i)
        valid &= pa[i][0] * (pair.particles.get_particles_ptr()[i].id ? 1 : -1) > 0;
    check(valid, "softened pair repels rather than attracts");
    auto sample = poisson_load(64, 933);
    auto global = all_particles(sample);
    Force coarse(24, 64, 1.25, .001), fine(mesh, 64, 1.25, .001);
    auto ac = coarse(sample), ar = fine(sample);
    double err = 0, norm = 0, errc = 0;
    for (size_t i = 0; i < ar.size(); ++i) {
        const auto exact = ewald(sample.particles.get_particles_ptr()[i], global);
        for (int d = 0; d < 3; ++d) {
            err += std::pow(ar[i][d] - exact[d], 2);
            errc += std::pow(ac[i][d] - exact[d], 2);
            norm += exact[d] * exact[d];
        }
    }
    FML::SumOverTasks(&err);
    FML::SumOverTasks(&errc);
    FML::SumOverTasks(&norm);
    if (FML::ThisTask == 0)
        std::cout << "Ewald RMS relative coarse/fine " << std::sqrt(errc / norm) << ' ' << std::sqrt(err / norm)
                  << '\n';
    check(std::sqrt(err / norm) < .006, "P3M sign/normalization agree with independent periodic Ewald");
    check(err < errc, "off-grid P3M force improves with mesh refinement");
    auto overflow = poisson_load(2, 99);
    for (size_t i = 0; i < overflow.particles.get_npart(); ++i) {
        auto & p = overflow.particles.get_particles_ptr()[i];
        p.q = p.position = {p.id ? 1e-100 : 0, 0, 0};
    }
    detail::migrate(overflow.particles);
    Options extreme;
    extreme.mesh = mesh;
    extreme.softening_spacing = 1e-100;
    extreme.max_steps = 1;
    const auto failed = relax(overflow, extreme);
    check(failed.status == Status::NonFinite && failed.steps == 0,
          "overflowing residuals fail closed before any drift");
    // Evolve Poisson to a quiet load: test requirements are measurable suppression, not an assumed k^4 fit.
    auto glass = poisson_load(512, 73);
    Options o;
    o.mesh = mesh;
    o.max_steps = 400;
    o.force_tolerance = 0;
    o.velocity_tolerance = 0;
    double initial = 0;
    for (auto k : std::vector<std::array<int, 3>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}})
        initial += structure(glass, k);
    auto relaxed = relax(glass, o);
    double final = 0;
    for (auto k : std::vector<std::array<int, 3>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}})
        final += structure(glass, k);
    if (FML::ThisTask == 0)
        std::cout << "Relaxation: low-k ratio " << final / initial << " force ratio "
                  << relaxed.residuals.force_rms / relaxed.history.front().force_rms << '\n';
    check(relaxed.status == Status::MaximumSteps && relaxed.steps == 400 && relaxed.history.size() == 401,
          "iteration exhaustion reported honestly");
    check(final < initial * .03, "large-scale discreteness power suppressed relative to seeded Poisson");
    check(relaxed.residuals.force_rms < .1 * relaxed.history.front().force_rms,
          "force residual decreases during damped relaxation");
    // Direct pair/nearest-neighbour diagnostics, independent of PM and search kernels.
    const auto all_glass = all_particles(glass);
    double close_pairs = 0;
    Vector directions{};
    double nearest_mean = 0;
    for (size_t i = 0; i < glass.particles.get_npart(); ++i) {
        const auto & target = glass.particles.get_particles_ptr()[i];
        double best = 10;
        Vector nearest{};
        for (const auto & source : all_glass)
            if (source.id != target.id) {
                Vector r{};
                double r2 = 0;
                for (int d = 0; d < 3; ++d) {
                    r[d] = source.position[d] - target.position[d];
                    r[d] -= std::round(r[d]);
                    r2 += r[d] * r[d];
                }
                if (r2 < .25 / 64)
                    ++close_pairs; // Half the mean spacing 1/8.
                if (r2 < best) {
                    best = r2;
                    nearest = r;
                }
            }
        nearest_mean += std::sqrt(best);
        for (int d = 0; d < 3; ++d)
            directions[d] += nearest[d] * nearest[d] / best;
    }
    FML::SumOverTasks(&close_pairs);
    FML::SumOverTasks(&nearest_mean);
    for (auto & x : directions) {
        FML::SumOverTasks(&x);
        x /= 512;
    }
    check(close_pairs == 0, "relaxation removes close pairs below half the mean spacing");
    check(*std::max_element(directions.begin(), directions.end()) -
                  *std::min_element(directions.begin(), directions.end()) <
              .1,
          "nearest-neighbour directional moments show no strong Cartesian preference");
    if (FML::ThisTask == 0)
        std::cout << "Glass quality: mean nearest/spacing " << nearest_mean / 512 * 8 << " direction moments "
                  << directions[0] << ' ' << directions[1] << ' ' << directions[2] << '\n';
    // Same manufactured density mode on a glass and lattice; subtract each original load's discreteness mode.
    std::vector<Particle> lattice;
    for (int ix = 0; ix < 8; ++ix)
        for (int iy = 0; iy < 8; ++iy)
            for (int iz = 0; iz < 8; ++iz) {
                Particle p;
                p.id = (ix * 8 + iy) * 8 + iz;
                p.q = p.position = {ix / 8., iy / 8., iz / 8.};
                if (p.position[0] >= FML::xmin_domain && p.position[0] < FML::xmax_domain)
                    lattice.push_back(p);
            }
    Load gridload;
    gridload.particles.move_from(std::move(lattice));
    auto displaced_glass = glass;
    Grid imposed(mesh, 1, 1);
    imposed.fill_real_grid([](auto & x) { return FML::GRID::FloatType(.01 * std::cos(2 * std::acos(-1.) * x[0])); });
    imposed.fftw_r2c();
    double responses[2]{};
    int j = 0;
    for (auto * l : {&displaced_glass, &gridload}) {
        double before = 0, after = 0;
        for (size_t i = 0; i < l->particles.get_npart(); ++i)
            before += std::cos(2 * std::acos(-1.) * l->particles.get_particles_ptr()[i].position[0]);
        FML::SumOverTasks(&before);
        apply_3lpt(*l, imposed, 1.);
        for (size_t i = 0; i < l->particles.get_npart(); ++i)
            after += std::cos(2 * std::acos(-1.) * l->particles.get_particles_ptr()[i].position[0]);
        FML::SumOverTasks(&after);
        responses[j++] = (after - before) / 512;
    }
    check(std::abs(responses[0] / .005 - 1) < .015 && std::abs(responses[1] / .005 - 1) < .001,
          "glass and lattice recover the same imposed linear density mode");
    // Refining the force mesh on the same load changes forces slightly but cannot hide a stalled state.
    Force refined(72, 512, 1.875);
    auto refined_force = refined(glass);
    auto refined_residual = residuals(glass, refined_force);
    check(refined_residual.force_rms < .02 * relaxed.history.front().force_rms,
          "quiet load remains quiet under fixed-split force-mesh refinement");
    o.max_steps = 0;
    auto unchanged = relax(glass, o);
    check(unchanged.steps == 0 && unchanged.history.size() == 1, "zero-step diagnostic preserves load");
    o.force_tolerance = 1e8;
    o.velocity_tolerance = 1e8;
    o.min_steps = 0;
    o.stable_steps = 1;
    check(relax(glass, o).status == Status::Converged, "convergence requires explicit residual thresholds");
    o.damping = -1;
    check(FML::TESTS::throws([&] { relax(glass, o); }), "invalid damping rejected");
    const std::string prefix = "glass_test_" + std::to_string(FML::NTasks);
    remove_load(prefix);
    save_load(glass, prefix);
    auto read = load_load(prefix, 17);
    valid = read.seed == 73 && read.boxsize == 1 && read.particles.get_npart() == glass.particles.get_npart();
    for (size_t i = 0; valid && i < read.particles.get_npart(); ++i) {
        const auto &a = read.particles.get_particles_ptr()[i], &b = glass.particles.get_particles_ptr()[i];
        valid &= a.id == b.id && a.position == b.position && a.q == a.position && a.velocity == Vector{};
    }
    check(valid, "versioned sharded IO preserves positions/IDs and resets artificial velocities");
    check(FML::TESTS::throws([&] { save_load(glass, prefix); }), "existing output prefix rejected");
    // Corruption checks: every rank reaches the same rejection even when only one reader sees damage.
    if (FML::ThisTask == 0) {
        std::ofstream out(detail::shard(prefix, 0));
        out << "FML_GLASS_SHARD 1 2\n0 .1 .2 .3\n";
    }
    check(FML::TESTS::throws([&] { load_load(prefix, 1); }), "truncated shard rejected collectively");
    remove_load(prefix);
    auto duplicate = poisson_load(2, 4);
    for (size_t i = 0; i < duplicate.particles.get_npart(); ++i)
        duplicate.particles.get_particles_ptr()[i].id = 0;
    check(FML::TESTS::throws([&] { save_load(duplicate, prefix); }),
          "duplicate IDs rejected without a global ID array");
    // Linear interpolation reference: sampled sinusoid interpolated independently between its two mesh nodes.
    auto ic = poisson_load(257, 193);
    std::array<Grid, 3> dx, dv;
    for (int d = 0; d < 3; ++d) {
        dx[d] = Grid(mesh, 1, 1);
        dv[d] = Grid(mesh, 1, 1);
        dx[d].fill_real_grid(
            [d](auto & x) { return FML::GRID::FloatType(.03 * (d + 1) * std::sin(2 * std::acos(-1.) * x[d])); });
        dv[d].fill_real_grid(
            [d](auto & x) { return FML::GRID::FloatType(.2 * (d + 1) * std::sin(2 * std::acos(-1.) * x[d])); });
    }
    apply_fields(ic, dx, dv);
    valid = true;
    double error = 0;
    for (size_t i = 0; i < ic.particles.get_npart(); ++i) {
        const auto & p = ic.particles.get_particles_ptr()[i];
        FML::RANDOM::COUNTER_BASED::Stream rng(193, p.id, 0x474c4153u);
        for (int d = 0; d < 3; ++d) {
            const double q = rng.uniform(), x = q * mesh, lo = std::floor(x), t = x - lo;
            const double s =
                (1 - t) * std::sin(2 * std::acos(-1.) * lo / mesh) + t * std::sin(2 * std::acos(-1.) * (lo + 1) / mesh);
            error = std::max(error, std::abs(p.position[d] - detail::wrap(q + .03 * (d + 1) * s)));
            error = std::max(error, std::abs(p.velocity[d] - .2 * (d + 1) * s));
            valid &= p.q[d] == q;
        }
    }
    FML::MaxOverTasks(&error);
    check(valid && error < 3e-6, "CIC at irregular q, wrapped displacement, independent velocity and migration");
    check(FML::TESTS::throws([&] { apply_fields(ic, dx, dv); }), "double IC application rejected");
    // Existing 3LPT on a plane wave must reduce to Zel'dovich exactly; validates adapter/velocity convention.
    auto lpt = poisson_load(128, 14);
    Grid delta(mesh, 1, 1);
    delta.fill_real_grid([](auto & x) { return FML::GRID::FloatType(.02 * std::cos(2 * std::acos(-1.) * x[0])); });
    delta.fftw_r2c();
    apply_3lpt(lpt, delta, 2.);
    error = 0;
    for (size_t i = 0; i < lpt.particles.get_npart(); ++i) {
        const auto & p = lpt.particles.get_particles_ptr()[i];
        const double x = p.q[0] * mesh, t = x - std::floor(x), k = 2 * std::acos(-1.);
        const double u =
            -.02 / k * ((1 - t) * std::sin(k * std::floor(x) / mesh) + t * std::sin(k * (std::floor(x) + 1) / mesh));
        error = std::max(error, std::abs(p.position[0] - detail::wrap(p.q[0] + u)));
        error = std::max(error, std::abs(p.velocity[0] - 2 * u));
        error = std::max(error, std::abs(p.velocity[1]) + std::abs(p.velocity[2]));
    }
    FML::MaxOverTasks(&error);
    check(error < 2e-7, "full 3LPT glass adapter obeys exact plane-parallel displacement/velocity");
    TestProvider provider;
    Grid field(mesh, 1, 1);
    density_from_provider(field, provider, FML::INITIALCONDITIONS::Species::CDMBaryon, .1, 100, 55, true);
    valid = true;
    for (auto i : field.get_fourier_range()) {
        auto k = field.get_fourier_wavevector_from_index(i);
        double k2 = 0;
        bool nyquist = false;
        for (auto x : k) {
            k2 += x * x;
            nyquist |= std::abs(x) >= std::acos(-1.) * mesh * (1 - 1e-10);
        }
        const double p = std::norm(field.get_fourier_from_index(i));
        valid &= (k2 == 0 || nyquist) ? p == 0 : std::abs(p - 12e-6) < 3e-11;
    }
    check(valid, "provider power divided by physical box volume, fixed amplitudes and zero/Nyquist modes");
    provider.bad_gauge = true;
    check(FML::TESTS::throws(
              [&] { density_from_provider(field, provider, FML::INITIALCONDITIONS::Species::CDMBaryon, .1, 100, 55); }),
          "unsupported provider gauge rejected before parallel generation");
    provider.bad_gauge = false;
    provider.bad_power = true;
    check(FML::TESTS::throws(
              [&] { density_from_provider(field, provider, FML::INITIALCONDITIONS::Species::CDMBaryon, .1, 100, 55); }),
          "nonfinite provider power rejected collectively");
    provider.bad_power = false;
    check(FML::TESTS::throws([&] {
              density_from_provider(field, provider, FML::INITIALCONDITIONS::Species::CDMBaryon, .001, 100, 55);
          }),
          "provider time coverage enforced");
    if (FML::NTasks > 1) {
        Options bad;
        bad.mesh = mesh;
        bad.max_steps = FML::ThisTask;
        check(FML::TESTS::throws([&] { relax(glass, bad); }), "rank-inconsistent options rejected collectively");
    }
    return FML::TESTS::summary("GlassIC");
}
