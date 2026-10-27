#include <FML/Baryonification/Baryonification.h>
#include <FML/Global/Global.h>

#include <array>
#include <iostream>
#include <limits>
#include <random>
#include <string>

using namespace FML::BARYONIFICATION;
int baryonification_odr_check();

namespace {
    int failures = 0;
    void check(bool ok, const std::string & what) {
        if (!ok) ++failures;
        if (FML::ThisTask == 0) std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << '\n';
    }
    bool close(double a, double b, double tolerance = 1e-10) {
        return std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tolerance;
    }
    template <class F> bool rejects(F f) {
        try { f(); } catch (const std::invalid_argument &) { return true; }
        return false;
    }
    struct Particle {
        double pos[3];
        double vel[3]{1, 2, 3};
        double mass = 7;
        long long id = 0;
        double * get_pos() { return pos; }
        int get_ndim() const { return 3; }
    };
    RadialMapping analytic_mapping() {
        std::vector<double> r, source, target;
        for (int i = 0; i <= 10000; ++i) {
            const double x = i / 10000.0;
            r.push_back(x); source.push_back(x * x * x); target.push_back(x);
        }
        return {EnclosedMassProfile(r, source), EnclosedMassProfile(r, target)};
    }
}

int main() {
    check(baryonification_odr_check() == 1, "headers are usable from multiple translation units");
    EnclosedMassProfile profile({0, 1, 2, 3}, {0, 1, 1, 4});
    check(close(profile.mass(1.5), 1) && close(profile.radius(1), 1) && close(profile.radius(2.5), 2.5),
          "monotone interpolation and inverse handle a mass plateau");
    auto map = analytic_mapping();
    double error = 0;
    for (int i = 1; i < 200; ++i) {
        const double r = (i + 0.37) / 201.0;
        error = std::max(error, std::abs(map.mapped_radius(r) - r * r * r));
    }
    if (FML::ThisTask == 0) std::cout << "Analytic mapping maximum absolute error: " << error << '\n';
    check(error < 1e-8, "mass-shell transport agrees with independent analytic r_new = r_old^3");
    check(map.displacement(0) == 0 && map.displacement(1) == 0 && map.displacement(5) == 0,
          "center and exterior have zero displacement");

    ModelParameters model;
    HaloProperties halo{1e14, 0.7, 5};
    auto profiles = make_halo_profiles(halo, model, 8e10);
    const double total = profiles.total_halo_mass;
    check(close(profiles.reference.mass(halo.radius200) / halo.mass200, 1, 1e-14),
          "reference NFW is normalized to the input M200");
    check(close((profiles.gas.total_mass() + profiles.central_stars.total_mass() +
                 profiles.collisionless.total_mass()) / total, 1, 1e-14),
          "gas, central stars and collisionless fractions conserve halo mass");
    check(close((profiles.source.total_mass() - total) /
                    ((4 * detail::pi / 3) * 8e10 * std::pow(profiles.source.maximum_radius(), 3)), 1, 1e-14),
          "source includes the specified uniform background");
    check(profiles.source.total_mass() == profiles.target.total_mass(), "outer source and target masses match exactly");
    bool monotone = true, matched = true;
    auto physical_map = profiles.mapping();
    double previous = 0;
    for (double r : profiles.reference.radii()) {
        const double moved = physical_map.mapped_radius(r);
        monotone = monotone && moved >= previous;
        matched = matched && close(profiles.target.mass(moved) / profiles.source.total_mass(),
                                   profiles.source.mass(r) / profiles.source.total_mass(), 2e-14);
        previous = moved;
    }
    check(monotone && matched, "physical profile transport is monotone and matches enclosed masses");

    // Independent references: adaptive SciPy quadrature of the analytic densities, then brentq for
    // ri at fixed rf (the reverse of the C++ shell solver). Values are divided by total halo mass.
    const double reference_values[][4] = {
        {0.01, 5.3400455315336145e-6, 0.0012342231240362528, 0.0016546005658815046},
        {0.1, 0.001625532753937916, 0.002471761222398064, 0.056837630749359457},
        {0.7, 0.03859275475158374, 0.002471761222438765, 0.4524484651219186},
        {2.0, 0.1050881078302447, 0.002471761222438765, 0.7540392666618353}};
    bool independent = true;
    for (const auto & row : reference_values)
        independent = independent && close(profiles.gas.mass(row[0]) / total, row[1], 1e-5) &&
            close(profiles.central_stars.mass(row[0]) / total, row[2], 1e-5) &&
            close(profiles.collisionless.mass(row[0]) / total, row[3], 1e-5);
    check(independent, "gas, stars and relaxed matter agree with independent quadrature/root-solve references");

    // Independently verify quadrature at R200 with a much finer table, including the relaxation solution.
    ProfileNumerics fine;
    fine.radial_bins = 8192;
    auto converged = make_halo_profiles(halo, model, 8e10, fine);
    double convergence = 0;
    for (double r : {0.001, 0.01, 0.1, 0.7, 2.0, 10.0}) {
        const double coarse = profiles.gas.mass(r) + profiles.central_stars.mass(r) + profiles.collisionless.mass(r);
        const double fine_mass = converged.gas.mass(r) + converged.central_stars.mass(r) + converged.collisionless.mass(r);
        convergence = std::max(convergence, std::abs(coarse - fine_mass) / total);
    }
    if (FML::ThisTask == 0) std::cout << "Radial convergence error / halo mass: " << convergence << '\n';
    check(convergence < 3e-5, "profiles including relaxation converge with radial resolution");
    model.relax_collisionless = false;
    auto unrelaxed = make_halo_profiles(halo, model, 0);
    check(close(unrelaxed.collisionless.mass(0.1) / unrelaxed.reference.mass(0.1),
                unrelaxed.collisionless_fraction, 1e-14), "optional unrelaxed matter follows the reference profile");
    check(profiles.collisionless.mass(0.001) > unrelaxed.collisionless.mass(0.001),
          "central stellar condensation contracts the collisionless core");

    // Exact null limit and qualitative limits of the feedback model.
    model.baryon_fraction = 0;
    model.stellar_amplitude = 0;
    auto identity = make_halo_profiles(halo, model, 0).mapping();
    bool unchanged = true;
    for (double r : profiles.reference.radii()) unchanged = unchanged && identity.displacement(r) == 0;
    for (double r : {0.0000321, 0.13456, 2.3456, 7.89012}) unchanged = unchanged && identity.displacement(r) == 0;
    check(unchanged, "zero baryon fraction gives an exactly identity transport");
    model.baryon_fraction = 0.157;
    auto gas_only = make_halo_profiles(halo, model, 0).mapping();
    check(gas_only.displacement(halo.radius200) > 0, "extended gas moves mass outward near R200");
    model.stellar_amplitude = model.baryon_fraction;
    model.stellar_total_exponent = model.stellar_central_exponent = 0;
    auto stars_only = make_halo_profiles(halo, model, 0).mapping();
    check(stars_only.displacement(0.05 * halo.radius200) < 0, "central stars move mass inward");

    // Every rank has the full halo list; particles are partitioned by global ID.
    // This also tests halos whose centers lie on another MPI task.
    const std::vector<HaloDisplacement> halos{{{0.1, 1, 1}, map}, {{0.7, 1, 1}, map}};
    Baryonifier field(halos, 4);
    auto reversed = halos;
    std::reverse(reversed.begin(), reversed.end());
    Baryonifier reverse_field(reversed, 4);
    Baryonifier one({halos[0]}, 4), two({halos[1]}, 4);
    std::vector<Particle> particles;
    for (int i = 0; i < 100; ++i)
        if (i % FML::NTasks == FML::ThisTask)
            particles.push_back({{0.01 + 0.039 * i, 1.01, 1.02}, {1, 2, 3}, 7, i});
    const auto original = particles;
    const auto shifts = field.displacements(particles.data(), particles.size());
    field.apply(particles.data(), particles.size());
    bool partition_ok = true, attributes_ok = true, overlap_ok = true;
    for (size_t i = 0; i < original.size(); ++i) {
        const std::array<double, 3> pos{original[i].pos[0], original[i].pos[1], original[i].pos[2]};
        const auto a = one.displacement_at(pos), b = two.displacement_at(pos), rev = reverse_field.displacement_at(pos);
        for (size_t d = 0; d < 3; ++d) {
            // Independent brute-force sum over halos at the original particle position.
            double brute = 0;
            for (const auto & h : halos) {
                std::array<double, 3> dx{};
                double r2 = 0;
                for (size_t j = 0; j < 3; ++j) {
                    dx[j] = pos[j] - h.center[j]; dx[j] -= 4 * std::round(dx[j] / 4);
                    r2 += dx[j] * dx[j];
                }
                const double r = std::sqrt(r2);
                if (r > 0 && r < 1) brute += (r * r * r - r) * dx[d] / r;
            }
            partition_ok = partition_ok && close(shifts[i][d], brute, 3e-7);
            partition_ok = partition_ok && close(particles[i].pos[d],
                FML::KDTREE::periodic_wrap(pos[d] + shifts[i][d], 4), 1e-14);
            overlap_ok = overlap_ok && close(shifts[i][d], a[d] + b[d], 1e-14) && close(shifts[i][d], rev[d], 1e-14);
            attributes_ok = attributes_ok && original[i].vel[d] == particles[i].vel[d];
        }
        attributes_ok = attributes_ok && original[i].id == particles[i].id && original[i].mass == particles[i].mass;
    }
    check(partition_ok, "KD-tree displacements and MPI partitions agree with an analytic periodic brute-force calculation");
    check(overlap_ok, "overlapping halos add at original positions and are independent of halo order");
    check(attributes_ok, "particle masses, IDs and velocities are preserved");
    const auto edge = one.displacement_at({3.9, 1, 1});
    check(edge[0] > 0 && edge[1] == 0 && edge[2] == 0, "periodic boundary uses the closest halo image");
    check(one.displacement_at({0.1, 1, 1}) == std::array<double, 3>{}, "particle at a halo center stays fixed");
    Baryonifier empty({}, 4);
    empty.apply<Particle>(nullptr, 0);
    auto copy = original;
    empty.apply(copy.data(), copy.size());
    bool empty_ok = true;
    for (size_t i = 0; i < copy.size(); ++i)
        for (size_t d = 0; d < 3; ++d) empty_ok = empty_ok && copy[i].pos[d] == original[i].pos[d];
    check(empty_ok, "empty halo and particle lists are valid and leave wrapped positions unchanged");
    Baryonifier isolated({halos[0]}, 4, false);
    check(isolated.displacement_at({3.9, 1, 1}) == std::array<double, 3>{}, "nonperiodic mode does not use halo images");
    // Invert a genuinely nonlinear cumulative table, independently of the cubic-source transport above.
    std::vector<double> inverse_r, inverse_m;
    for (int i = 0; i <= 10000; ++i) {
        const double x = i / 10000.0;
        inverse_r.push_back(x); inverse_m.push_back(x * x);
    }
    EnclosedMassProfile square_mass(inverse_r, inverse_m);
    bool inverse_ok = true;
    for (double mass : {0.00123, 0.0234, 0.345, 0.789})
        inverse_ok = inverse_ok && close(square_mass.radius(mass), std::sqrt(mass), 1e-7);
    check(inverse_ok, "nonlinear cumulative inverse agrees with sqrt(mass)");

    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(rejects([] { EnclosedMassProfile({0, 1, 2}, {0, 2, 1}); }) &&
          rejects([] { EnclosedMassProfile({0, 1, 1}, {0, 1, 2}); }) &&
          rejects([&] { EnclosedMassProfile({0, 1}, {0, nan}); }), "invalid profile tables are rejected");
    check(rejects([] { RadialMapping(EnclosedMassProfile({0, 1}, {0, 1}), EnclosedMassProfile({0, 1}, {0, 2})); }) &&
          rejects([&] { map.displacement(nan); }), "invalid radial mappings and queries are rejected");
    check(rejects([&] { Baryonifier(halos, 2); }) && rejects([&] { field.displacement_at({nan, 0, 0}); }),
          "ambiguous periodic supports and non-finite positions are rejected");
    check(rejects([&] { make_halo_profiles({0, 1, 5}, {}, 0); }) &&
          rejects([&] { make_halo_profiles({1e8, 1, 5}, {}, 0); }) &&
          rejects([&] { make_halo_profiles(halo, {}, -1); }), "invalid halos, stellar budgets and backgrounds are rejected");
    auto bad = original;
    if (!bad.empty()) {
        bad.back().pos[0] = nan;
        check(rejects([&] { field.apply(bad.data(), bad.size()); }) &&
              (bad.size() == 1 || bad.front().pos[0] == original.front().pos[0]),
              "bad input is rejected before any particles are modified");
    }
    FML::SumOverTasks(&failures);
    if (FML::ThisTask == 0) std::cout << (failures ? "BARYONIFICATION FAILED\n" : "ALL BARYONIFICATION TESTS PASSED\n");
    return failures ? 1 : 0;
}
