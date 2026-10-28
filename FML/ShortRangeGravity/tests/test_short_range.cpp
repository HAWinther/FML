#include <FML/NBody/NBody.h>
#include <FML/ShortRangeGravity/LongRangePM.h>
#include <FML/ShortRangeGravity/Octree.h>
#include <FML/ShortRangeGravity/SlabGhosts.h>
#include <chrono>
#include <iostream>
#include <random>
#include <set>

using namespace FML::SHORT_RANGE_GRAVITY;
int failures = 0;
void check(bool good, const char * label) {
    int bad = !good;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (FML::ThisTask == 0)
        std::cout << (bad ? "FAIL " : "PASS ") << label << '\n';
}
bool near(double a, double b, double relative = 1e-12, double absolute = 1e-12) {
    return std::abs(a - b) <= absolute + relative * std::max(std::abs(a), std::abs(b));
}
double norm(Vector a) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }
double error(const std::vector<Vector> & a, const std::vector<Vector> & b) {
    double difference = 0, reference = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        for (int d = 0; d < 3; ++d) {
            difference += (a[i][d] - b[i][d]) * (a[i][d] - b[i][d]);
            reference += b[i][d] * b[i][d];
        }
    return std::sqrt(difference / std::max(reference, 1e-100));
}
std::vector<Particle> catalogue(std::size_t n, bool clustered = false) {
    std::mt19937 rng(12);
    std::uniform_real_distribution<double> u(0, 1);
    std::vector<Particle> p(n);
    for (std::size_t i = 0; i < n; ++i) {
        p[i].id = i;
        p[i].mass = .5 + u(rng);
        for (auto & x : p[i].position)
            x = clustered && i % 2 ? .3 + .01 * u(rng) : u(rng);
    }
    return p;
}
void kernel_tests() {
    Kernel k;
    const auto & o = k.options();
    // Independently integrate the normalized cubic-spline density to obtain enclosed mass.
    bool correct = true;
    for (double u : {1e-5, .01, .25, .499, .5, .501, .75, .999, 1., 1.5}) {
        const double r = u * o.softening_support;
        const int intervals = 20000;
        double enclosed = 0;
        for (int i = 0; i < intervals; ++i) {
            const double x = u * (i + .5) / intervals;
            const double density = x < .5 ? 1 - 6 * x * x + 6 * x * x * x : x < 1 ? 2 * std::pow(1 - x, 3) : 0;
            enclosed += 32 * x * x * density * u / intervals;
        }
        const double expected = enclosed / (r * r * r);
        correct &= near(k.softened_coefficient(r), expected, 2e-8, 1e-8);
        correct &= near(k.untruncated_coefficient(r) + o.coupling * k.long_range_coefficient(r),
                        o.coupling * k.softened_coefficient(r),
                        1e-12,
                        1e-8);
    }
    check(correct, "spline force from independent enclosed-mass quadrature; exact force split");
    check(std::isfinite(k.coefficient(0)) &&
              near(k.long_range_coefficient(0), 1 / (6 * std::sqrt(std::acos(-1.0)) * std::pow(o.split_scale, 3))),
          "finite zero-separation Gaussian and spline limits");
    check(near(short_range_window(k.cutoff() / (2 * o.split_scale)), o.cutoff_tolerance) &&
              k.coefficient(k.cutoff()) == 0 && k.coefficient(2 * k.cutoff()) == 0,
          "tolerance-derived cutoff");
    check(near(k.long_range_filter(0), 1) &&
              near(k.long_range_filter(1 / (o.split_scale * o.split_scale)), std::exp(-1.)),
          "Gaussian Fourier filter normalization");
    bool rejected = true;
    for (int problem = 0; problem < 6; ++problem) {
        auto bad = o;
        if (problem == 0)
            bad.split_scale = 1;
        if (problem == 1)
            bad.softening_support = 0;
        if (problem == 2)
            bad.cutoff_tolerance = 1;
        if (problem == 3)
            bad.boxsize = -1;
        if (problem == 4)
            bad.coupling = std::numeric_limits<double>::quiet_NaN();
        if (problem == 5)
            bad.max_cells = 0;
        try {
            Kernel invalid(bad);
            rejected = false;
        } catch (const std::invalid_argument &) {
        }
    }
    check(rejected, "invalid kernel options rejected");
}
void backends() {
    bool cells = true, exact = true, tree = true;
    std::uint64_t accepted_nodes = 0;
    for (bool clustered : {false, true}) {
        auto p = catalogue(1200, clustered);
        // Include periodic neighbours, coincident distinct particles and zero mass.
        p[0].position = {.001, .2, .2};
        p[1].position = {.999, .2, .2};
        p[2].position = p[0].position;
        p[2].mass = 0;
        for (std::size_t budget : {1, 8, 100000}) {
            Options o;
            o.max_cells = budget;
            o.split_scale = .04;
            Kernel k(o);
            const auto reference = direct_sum(p, p, k);
            const auto c = CellList(p, k).compute(p);
            cells &= error(c.acceleration, reference.acceleration) < 2e-14 &&
                     c.statistics.accepted_pairs == reference.statistics.accepted_pairs;
            if (budget != 100000)
                continue;
            const auto t0 = Octree(p, k, {0, 8, 40}).compute(p);
            const auto t = Octree(p, k, {.3, 8, 40}).compute(p);
            exact &= error(t0.acceleration, reference.acceleration) < 2e-14;
            tree &= error(t.acceleration, reference.acceleration) < 1e-3;
            if (FML::ThisTask == 0)
                std::cout << "Tree error " << error(t.acceleration, reference.acceleration) << " accepted nodes "
                          << t.accepted_nodes << " clustered " << clustered << "\n";
            accepted_nodes += t.accepted_nodes;
            Vector momentum{};
            for (std::size_t i = 0; i < p.size(); ++i)
                for (int d = 0; d < 3; ++d)
                    momentum[d] += p[i].mass * c.acceleration[i][d];
            cells &= norm(momentum) < 1e-6;
#ifdef USE_OMP
            const int threads = omp_get_max_threads();
            omp_set_num_threads(1);
            const auto serial = CellList(p, k).compute(p);
            omp_set_num_threads(threads);
            cells &= serial.acceleration == c.acceleration;
#endif
        }
    }
    check(cells, "cell lists match direct sums, momentum and thread determinism, including 1/2-cell aliases");
    auto sources = catalogue(1000, true);
    std::vector<Particle> probes{{{.34, .31, .31}, 1, 999999}, {{.34, .31, .31}, 1, 3}};
    Kernel k;
    const auto truth = direct_sum(probes, sources, k);
    const auto approx = Octree(sources, k, {.3, 4, 40}).compute(probes);
    const auto tighter = Octree(sources, k, {.1, 4, 40}).compute(probes);
    tree &=
        error(approx.acceleration, truth.acceleration) < 1e-3 && error(tighter.acceleration, truth.acceleration) < 1e-4;
    check(exact, "zero-opening octree matches direct sums");
    check(tree && accepted_nodes > 0, "monopole tree accuracy and nontrivial node acceptance");
    std::vector<Particle> empty;
    check(CellList(empty, Kernel()).compute(empty).acceleration.empty() &&
              Octree(empty, Kernel()).compute(empty).acceleration.empty(),
          "empty catalogues");
    auto duplicate = catalogue(2);
    duplicate[1].id = duplicate[0].id;
    bool rejected = false;
    try {
        CellList invalid(duplicate, Kernel());
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "duplicate source identity rejected");
}
// Independent periodic Ewald pair acceleration: a different split parameter and explicit image/mode sums.
Vector ewald(Vector displacement, double h, double alpha = 6) {
    const double pi = std::acos(-1.0);
    Vector result{};
    for (int a = -2; a <= 2; ++a)
        for (int b = -2; b <= 2; ++b)
            for (int c = -2; c <= 2; ++c) {
                Vector r{displacement[0] + a, displacement[1] + b, displacement[2] + c};
                const double radius = norm(r);
                if (radius == 0)
                    continue;
                const double factor = (std::erfc(alpha * radius) + 2 * alpha * radius / std::sqrt(pi) *
                                                                       std::exp(-alpha * alpha * radius * radius)) /
                                      std::pow(radius, 3);
                for (int d = 0; d < 3; ++d)
                    result[d] -= factor * r[d];
            }
    for (int a = -12; a <= 12; ++a)
        for (int b = -12; b <= 12; ++b)
            for (int c = -12; c <= 12; ++c) {
                const double n2 = a * a + b * b + c * c;
                if (n2 == 0)
                    continue;
                Vector k{2 * pi * a, 2 * pi * b, 2 * pi * c};
                const double phase = k[0] * displacement[0] + k[1] * displacement[1] + k[2] * displacement[2];
                const double factor =
                    -4 * pi * std::exp(-pi * pi * n2 / (alpha * alpha)) / (4 * pi * pi * n2) * std::sin(phase);
                for (int d = 0; d < 3; ++d)
                    result[d] += factor * k[d];
            }
    const double r = norm(displacement);
    if (r > 0 && r < h) {
        Options o;
        o.softening_support = h;
        Kernel k(o);
        const double correction = k.softened_coefficient(r) - 1 / std::pow(r, 3);
        for (int d = 0; d < 3; ++d)
            result[d] -= correction * displacement[d];
    }
    return result;
}
// Point-source Fourier density -> FML PM Poisson solver, evaluated at arbitrary particle positions by exact DFT.
// Avoids assignment/interpolation errors, so this tests split, sign, units and periodicity independently.
void periodic_force() {
    const double pi = std::acos(-1.0);
    double worst = 0;
    for (double rs : {.025, .04})
        for (double support : {.006, .025}) {
            Options o;
            o.split_scale = rs;
            o.softening_support = support;
            o.cutoff_tolerance = 1e-10;
            o.coupling = fml_acceleration_coupling(4 * pi, 1);
            Kernel k(o);
            FML::GRID::FFTWGrid<3> field(64);
            field.set_grid_status_real(false);
            for (auto i : field.get_fourier_range())
                field.set_fourier_from_index(i, 1.);
            apply_long_range_filter(field, k);
            std::array<FML::GRID::FFTWGrid<3>, 3> force_grid;
            FML::NBODY::compute_force_from_density_fourier<3>(field, force_grid, "CIC", 4 * pi, {});
            for (std::array<int, 3> coord :
                 std::vector<std::array<int, 3>>{{1, 0, 0}, {4, 2, 1}, {10, 6, 2}, {31, 31, 31}}) {
                Vector x{double(coord[0]) / 64, double(coord[1]) / 64, double(coord[2]) / 64};
                Vector force{};
                const int local_x = coord[0] - int(field.get_local_x_start());
                if (local_x >= 0 && local_x < field.get_local_nx()) {
                    const auto index = force_grid[0].get_index_real({local_x, coord[1], coord[2]});
                    for (int d = 0; d < 3; ++d)
                        force[d] = -force_grid[d].get_real_from_index(index);
                }
                FML::SumArrayOverTasks(force.data(), 3);
                for (int d = 0; d < 3; ++d)
                    force[d] -= k.coefficient(norm(x)) * x[d];
                const auto truth = ewald(x, o.softening_support);
                Vector difference;
                for (int d = 0; d < 3; ++d)
                    difference[d] = force[d] - truth[d];
                worst = std::max(worst, norm(difference) / std::max(norm(truth), 1e-8));
            }
            // Real-to-complex multiplicity with explicit Nyquist handling; Nyquist terms are exponentially suppressed.
            for (Vector x : std::vector<Vector>{{.006, .002, .001},
                                                {.08, .01, .02},
                                                {.02, .08, .01},
                                                {.15, .09, .02},
                                                {.45, .01, .03},
                                                {.49, .49, .49}}) {
                Vector force{};
                for (auto i : field.get_fourier_range()) {
                    const auto wave = field.get_fourier_wavevector_from_index(i);
                    const double k2 = wave[0] * wave[0] + wave[1] * wave[1] + wave[2] * wave[2];
                    if (k2 == 0)
                        continue;
                    const double multiplicity = wave[2] == 0 || std::abs(wave[2]) == 64 * pi ? 1 : 2;
                    const double phase = wave[0] * x[0] + wave[1] * x[1] + wave[2] * x[2];
                    const double factor =
                        -multiplicity * 4 * pi / k2 * double(field.get_fourier_from_index(i).real()) * std::sin(phase);
                    for (int d = 0; d < 3; ++d)
                        force[d] += factor * wave[d];
                }
                FML::SumArrayOverTasks(force.data(), 3);
                const double r = norm(x);
                for (int d = 0; d < 3; ++d)
                    force[d] -= k.coefficient(r) * x[d];
                const auto truth = ewald(x, o.softening_support);
                Vector difference;
                for (int d = 0; d < 3; ++d)
                    difference[d] = force[d] - truth[d];
                worst = std::max(worst, norm(difference) / std::max(norm(truth), 1e-8));
            }
        }
    // Float FFT rounding is amplified at the weak, nearly cancelling corner probes. Keep the double criterion strict.
    const double tolerance = sizeof(FML::GRID::FloatType) == sizeof(float) ? 5e-5 : 2e-7;
    check(worst < tolerance, "periodic Gaussian-filtered Fourier plus softened short force matches independent Ewald");
    if (FML::ThisTask == 0)
        std::cout << "Ewald maximum relative force error " << worst << '\n';
    const double C = .7, M = 12, L = 250;
    check(near(fml_acceleration_coupling(C, M, L) / (L * L), C / (4 * pi * M)),
          "FML unit-box Poisson and mass normalization");
}
void ghosts() {
    auto global = catalogue(600);
    // Width-spanning cutoff with >=4 tasks, periodic edges, and deliberately empty rank for a second case.
    Options o;
    o.cutoff_tolerance = 1e-8;
    for (double rs : {.02, .048}) {
        o.split_scale = rs;
        Kernel k(o);
        for (bool empty_rank : {false, true}) {
            auto all = global;
            if (empty_rank)
                all.erase(
                    std::remove_if(all.begin(), all.end(), [](const Particle & p) { return p.position[0] < .25; }),
                    all.end());
            const double lo = double(FML::ThisTask) / FML::NTasks, hi = double(FML::ThisTask + 1) / FML::NTasks;
            std::vector<Particle> local;
            for (const auto & p : all)
                if (p.position[0] >= lo && p.position[0] < hi)
                    local.push_back(p);
            auto sources = exchange_slab_ghosts(local, lo, hi, k);
            const auto expected = direct_sum(local, all, k);
            const auto actual = CellList(sources.particles, k).compute(local);
            const auto tree = Octree(sources.particles, k, {0, 8, 40}).compute(local);
            std::set<std::uint64_t> ids;
            for (const auto & p : sources.particles)
                ids.insert(p.id);
            check(ids.size() == sources.particles.size() && error(actual.acceleration, expected.acceleration) < 3e-14 &&
                      error(tree.acceleration, expected.acceleration) < 3e-14,
                  "MPI slab ghosts match replicated reference without duplicates, including empty/nonadjacent slabs");
        }
    }
}
void normalization_and_scaling() {
    auto p = catalogue(100);
    double total_mass = 0;
    for (const auto & x : p)
        total_mass += x.mass;
    Options o;
    o.coupling = fml_acceleration_coupling(.7, total_mass);
    const auto base = CellList(p, Kernel(o)).compute(p);
    const double L = 250, mass_factor = 1e12;
    for (auto & x : p) {
        for (auto & pos : x.position)
            pos *= L;
        x.mass *= mass_factor;
    }
    o.boxsize *= L;
    o.split_scale *= L;
    o.softening_support *= L;
    o.coupling = fml_acceleration_coupling(.7, total_mass * mass_factor, L);
    const auto scaled = CellList(p, Kernel(o)).compute(p);
    check(error(scaled.acceleration, base.acceleration) < 2e-14,
          "length and mass unit scaling preserves FML acceleration");
}
void chunked_ghosts_and_errors() {
    auto all = catalogue(40000);
    const double lo = double(FML::ThisTask) / FML::NTasks, hi = double(FML::ThisTask + 1) / FML::NTasks;
    std::vector<Particle> local;
    for (const auto & p : all)
        if (p.position[0] >= lo && p.position[0] < hi)
            local.push_back(p);
    Options o;
    o.split_scale = .048;
    o.cutoff_tolerance = 1e-8;
    Kernel k(o);
    auto result = exchange_slab_ghosts(local, lo, hi, k);
    std::set<std::uint64_t> actual, expected;
    for (const auto & p : result.particles)
        actual.insert(p.id);
    for (const auto & p : all)
        if (detail::distance_to_periodic_slab(p.position[0], lo, hi, 1) < k.cutoff())
            expected.insert(p.id);
    check(actual == expected, "chunked ghost payloads preserve complete source identities");
    if (FML::ThisTask == 0 && !local.empty())
        local[0].mass = -1;
    bool rejected = false;
    try {
        exchange_slab_ghosts(local, lo, hi, k);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "rank-local invalid inputs rejected collectively");
}
void synchronize_benchmark() {
#ifdef USE_MPI
    MPI_Barrier(FML::communicator);
#endif
}
void benchmark() {
    for (bool clustered : {false, true}) {
        auto global = catalogue(8000, clustered);
        const double lo = double(FML::ThisTask) / FML::NTasks, hi = double(FML::ThisTask + 1) / FML::NTasks;
        std::vector<Particle> p;
        for (const auto & source : global)
            if (source.position[0] >= lo && source.position[0] < hi)
                p.push_back(source);
        Options o;
        Kernel k(o);
        synchronize_benchmark();
        const auto start = std::chrono::steady_clock::now();
        const auto reference = direct_sum(p, global, k);
        const auto a = std::chrono::steady_clock::now();
        synchronize_benchmark();
        const auto ghost_start = std::chrono::steady_clock::now();
        auto sources = exchange_slab_ghosts(p, lo, hi, k);
        const auto ghosts = std::chrono::steady_clock::now();
        synchronize_benchmark();
        const auto cell_start = std::chrono::steady_clock::now();
        const auto cell = CellList(sources.particles, k).compute(p);
        const auto b = std::chrono::steady_clock::now();
        synchronize_benchmark();
        const auto tree_start = std::chrono::steady_clock::now();
        const auto tree = Octree(sources.particles, k, {.3, 8, 40}).compute(p);
        const auto end = std::chrono::steady_clock::now();
        const auto seconds = [](auto x, auto y) { return std::chrono::duration<double>(y - x).count(); };
        double direct_time = seconds(start, a), ghost_time = seconds(ghost_start, ghosts),
               cell_time = seconds(cell_start, b), tree_time = seconds(tree_start, end);
        double cell_error = error(cell.acceleration, reference.acceleration),
               tree_error = error(tree.acceleration, reference.acceleration);
        double ghost_mib = double(sources.received_particles * sizeof(Particle)) / (1024 * 1024);
        for (auto value : {&direct_time, &ghost_time, &cell_time, &tree_time, &cell_error, &tree_error, &ghost_mib})
            FML::MaxOverTasks(value);
        auto cell_pairs = cell.statistics.candidate_pairs, direct_pairs = reference.statistics.candidate_pairs,
             tree_pairs = tree.statistics.candidate_pairs, nodes = tree.accepted_nodes;
        for (auto count : {&cell_pairs, &direct_pairs, &tree_pairs, &nodes})
            FML::SumOverTasks(count);
        if (FML::ThisTask == 0)
            std::cout << "BENCH n=8000 ranks=" << FML::NTasks << " clustered=" << clustered
                      << " direct_s=" << direct_time << " ghost_s=" << ghost_time << " cell_build_force_s=" << cell_time
                      << " tree_build_force_s=" << tree_time << " cell_candidates=" << cell_pairs
                      << " direct_candidates=" << direct_pairs << " tree_candidates=" << tree_pairs
                      << " accepted_nodes=" << nodes << " ghost_payload_mib_max=" << ghost_mib
                      << " cell_relative_rms_max=" << cell_error << " tree_relative_rms_max=" << tree_error << '\n';
    }
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        benchmark();
        return 0;
    }

    kernel_tests();
    backends();
    periodic_force();
    ghosts();
    normalization_and_scaling();
    chunked_ghosts_and_errors();
    if (FML::ThisTask == 0)
        std::cout << "ShortRangeGravity failures: " << failures << '\n';
    return failures ? 1 : 0;
}
