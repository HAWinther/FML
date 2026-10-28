#include <FML/NBody/NBody.h>
#include <FML/ShortRangeGravity/LongRangePM.h>
#include <FML/ShortRangeGravity/Octree.h>
#include <FML/ShortRangeGravity/SlabGhosts.h>
#include "YukawaReference.h"
#include <iostream>
#include <random>

using namespace FML::SHORT_RANGE_GRAVITY;
int failures = 0;
void check(bool good, const char * label) {
    int bad = !good;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (FML::ThisTask == 0)
        std::cout << (bad ? "FAIL " : "PASS ") << label << '\n';
}
double norm(Vector v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
bool close(double a, double b, double tol = 2e-12, double floor = 1) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tol * std::max({floor, std::abs(a), std::abs(b)});
}
void reference_values() {
    bool good = true;
    double worst = 0;
    for (const auto & row : yukawa_reference) {
        const double q = row[0], z = row[1];
        const auto long_part = detail::yukawa_long(2 * q, 1, 1 / z);
        for (int d = 0; d < 2; ++d) {
            const double err = std::abs(long_part[d] - row[2 + d]) / std::max(1e-8, std::abs(row[2 + d]));
            worst = std::max(worst, err);
            good &= err < 3e-12;
        }
        const auto dl = detail::yukawa_long_difference(2 * q, 1, 1 / z);
        for (int d = 0; d < 2; ++d)
            good &= close(dl[d], row[6 + d], 3e-12, 1e-20);
        if (q > 0) {
            const auto ds = detail::yukawa_short_difference(2 * q, 1, 1 / z);
            for (int d = 0; d < 2; ++d)
                good &= close(ds[d], row[8 + d], 3e-12, 1e-20);
            const auto short_part = detail::yukawa_short(2 * q, 1, 1 / z);
            for (int d = 0; d < 2; ++d)
                good &= close(short_part[d], row[4 + d], 3e-12, 1e-20);
        }
    }
    check(good, "Ewald Yukawa kernels and cancellation differences match independent 100-digit reference values");
    if (FML::ThisTask == 0)
        std::cout << "Maximum scaled long-kernel reference error " << worst << '\n';
}
double derivative(const Kernel & k, double r, int part) {
    const double step = std::max(r, k.options().softening_support) * 1e-5;
    const auto potential = [&](double x) {
        return part == 0 ? k.potential(x) : part == 1 ? k.long_range_potential(x) : k.untruncated_potential(x);
    };
    return (potential(r - 2 * step) - 8 * potential(r - step) + 8 * potential(r + step) - potential(r + 2 * step)) /
           (12 * step * r);
}
void kernels() {
    Options o;
    Kernel n(o), explicit_n(o, Newtonian{}), zero(o, NewtonYukawa{0, .003});
    bool unchanged = true, consistent = true, positive = true;
    for (double r : {0., .0001, .001, .0025, .005, .02, .05, .1}) {
        unchanged &= n.coefficient(r) == explicit_n.coefficient(r) && n.coefficient(r) == zero.coefficient(r) &&
                     n.potential(r) == zero.potential(r) && n.cutoff() == zero.cutoff();
    }
    for (double k2 : {0., 1., 100., 1e6})
        unchanged &= n.long_range_filter(k2) == zero.long_range_filter(k2);
    check(unchanged, "default/explicit Newtonian and alpha=0 preserve identical results");
    for (double lambda : {1e-6, .001, .01, .1, 10., 1e6})
        for (double alpha : {-2., -1., -.5, .2, 3.}) {
            Kernel k(o, NewtonYukawa{alpha, lambda});
            consistent &= std::isfinite(k.coefficient(0)) && std::isfinite(k.potential(0));
            for (double r : {.0005, .00249, .00251, .00499, .00501, .02, .07}) {
                const double scale = (1 + std::abs(alpha)) * n.full_coefficient(r);
                consistent &= std::abs(k.full_coefficient(r) - derivative(k, r, 0)) < 2e-8 * scale;
                consistent &= std::abs(o.coupling * k.long_range_coefficient(r) - derivative(k, r, 1)) < 2e-8 * scale;
                consistent &= std::abs(k.untruncated_coefficient(r) - derivative(k, r, 2)) < 2e-8 * scale;
                consistent &= std::abs(k.untruncated_coefficient(r) + o.coupling * k.long_range_coefficient(r) -
                                       k.full_coefficient(r)) < 2e-11 * scale;
                if (r >= o.softening_support) {
                    const double expected =
                        o.coupling / (r * r * r) * (1 + alpha * (1 + r / lambda) * std::exp(-r / lambda));
                    consistent &= std::abs(k.full_coefficient(r) - expected) < 2e-12 * scale;
                    const auto ys = detail::yukawa_short(r, o.split_scale, lambda);
                    positive &= ys[1] >= -1e-14 / (r * r * r) &&
                                ys[1] <= n.untruncated_coefficient(r) / o.coupling + 1e-12 / (r * r * r);
                }
            }
            const double radius = k.cutoff();
            const double envelope = (1 + std::abs(alpha)) * short_range_window(radius / (2 * o.split_scale));
            consistent &= close(envelope, o.cutoff_tolerance, 2e-12, 1e-12) && k.coefficient(radius) == 0;
        }
    // Also exercise softening that crosses q=1, where the long-range evaluation changes representation.
    Options broad = o;
    broad.softening_support = .06;
    Kernel broad_n(broad);
    for (double alpha : {-1., .5})
        for (double lambda : {.01, .1, 1e6}) {
            Kernel k(broad, NewtonYukawa{alpha, lambda});
            for (double radius : {.025, .04, .05999, .06001}) {
                const double scale = (1 + std::abs(alpha)) * broad_n.full_coefficient(radius);
                consistent &= std::abs(k.full_coefficient(radius) - derivative(k, radius, 0)) < 2e-8 * scale &&
                              std::abs(k.untruncated_coefficient(radius) - derivative(k, radius, 2)) < 2e-8 * scale;
            }
        }
    check(consistent, "potential derivatives, softened force split, point-force law and componentwise cutoff bound");
    check(positive, "Yukawa short force is positive and bounded by the Newtonian short force");
    Kernel large(o, NewtonYukawa{.7, 1e8}), small(o, NewtonYukawa{.7, 1e-8});
    bool limits = true;
    for (double r : {0., .001, .003, .005, .05}) {
        limits &= close(large.full_coefficient(r), 1.7 * n.full_coefficient(r), 1e-12);
        limits &= small.full_coefficient(r) == n.full_coefficient(r);
        limits &= close(large.untruncated_coefficient(r), 1.7 * n.untruncated_coefficient(r), 1e-11);
    }
    check(limits, "large/small lambda recover enhanced/Newtonian gravity, including softened radii");
    Kernel cancellation(o, NewtonYukawa{-1, 1e8});
    const double r = .02;
    const double expected = 1 / (2 * r * 1e16); // Leading mu^2/(2r), corrections O(r/lambda).
    const double split = cancellation.untruncated_coefficient(r) + o.coupling * cancellation.long_range_coefficient(r);
    check(close(cancellation.full_coefficient(r), expected, 3e-10, 1e-30) && close(split, expected, 3e-10, 1e-30) &&
              close(cancellation.long_range_filter(100),
                    std::exp(-100 * o.split_scale * o.split_scale) * (1.0 / 100 + o.split_scale * o.split_scale) / 1e16,
                    2e-12,
                    1e-30),
          "alpha=-1 large-range forces retain the small physical remainder");
    bool rejected = true;
    for (auto model : {NewtonYukawa{1, 0}, NewtonYukawa{1, -1}, NewtonYukawa{NAN, 1}, NewtonYukawa{1, INFINITY}}) {
        try {
            Kernel bad(o, model);
            rejected = false;
        } catch (const std::invalid_argument &) {
        }
    }
    check(rejected, "invalid Yukawa model parameters rejected");
}
// Independent periodic image sum of the unsplit Yukawa point-source force; exponentially convergent.
Vector periodic_yukawa(Vector x, double lambda) {
    Vector result{};
    for (int a = -8; a <= 8; ++a)
        for (int b = -8; b <= 8; ++b)
            for (int c = -8; c <= 8; ++c) {
                const Vector r{x[0] + a, x[1] + b, x[2] + c};
                const double radius = norm(r);
                if (radius == 0)
                    continue;
                const double t = radius / lambda;
                const double coefficient = (1 + t) * std::exp(-t) / (radius * radius * radius);
                for (int d = 0; d < 3; ++d)
                    result[d] -= coefficient * r[d];
            }
    return result;
}
void pm() {
    constexpr int mesh = 64;
    const double pi = std::acos(-1.0);
    double worst = 0;
    for (double rs : {.025, .04})
        for (double lambda : {.015, .06, .2})
            for (double alpha : {-.75, 2.}) {
                Options o;
                o.split_scale = rs;
                o.cutoff_tolerance = 1e-11;
                Kernel y(o, NewtonYukawa{alpha, lambda}), n(o);
                FML::GRID::FFTWGrid<3> original(mesh);
                original.set_grid_status_real(false);
                for (auto i : original.get_fourier_range())
                    original.set_fourier_from_index(i, 1.);
                auto fy = original, fn = original;
                apply_long_range_filter(fy, y);
                apply_long_range_filter(fn, n);
                std::array<FML::GRID::FFTWGrid<3>, 3> gy, gn;
                FML::NBODY::compute_force_from_density_fourier<3>(fy, gy, "CIC", 4 * pi, {});
                FML::NBODY::compute_force_from_density_fourier<3>(fn, gn, "CIC", 4 * pi, {});
                for (std::array<int, 3> coord :
                     std::vector<std::array<int, 3>>{{2, 0, 0}, {4, 2, 1}, {10, 6, 2}, {31, 31, 31}}) {
                    Vector x{double(coord[0]) / mesh, double(coord[1]) / mesh, double(coord[2]) / mesh}, difference{};
                    const int local = coord[0] - int(fy.get_local_x_start());
                    if (local >= 0 && local < fy.get_local_nx()) {
                        const auto index = gy[0].get_index_real({local, coord[1], coord[2]});
                        for (int d = 0; d < 3; ++d)
                            difference[d] =
                                -double(gy[d].get_real_from_index(index)) + double(gn[d].get_real_from_index(index));
                    }
                    FML::SumArrayOverTasks(difference.data(), 3);
                    const double short_coefficient = y.coefficient(norm(x)) - n.coefficient(norm(x));
                    // The models can have different cutoffs; subtract their actual truncated corrections.
                    for (int d = 0; d < 3; ++d)
                        difference[d] -= short_coefficient * x[d];
                    const auto truth = periodic_yukawa(x, lambda);
                    Vector residual;
                    for (int d = 0; d < 3; ++d)
                        residual[d] = difference[d] - alpha * truth[d];
                    worst = std::max(worst, norm(residual) / std::max(1., std::abs(alpha) * norm(truth)));
                }
            }
    check(worst < (sizeof(FML::GRID::FloatType) == sizeof(float) ? 2e-4 : 2e-8),
          "PM plus short correction matches independent periodic Yukawa image sums");
    if (FML::ThisTask == 0)
        std::cout << "Maximum scaled periodic Yukawa force error " << worst << '\n';
    Options o;
    o.boxsize = 250;
    o.split_scale = 5;
    o.softening_support = 1;
    Kernel scaled(o, NewtonYukawa{.6, 12.5});
    Options unit;
    unit.split_scale = .02;
    unit.softening_support = .004;
    Kernel base(unit, NewtonYukawa{.6, .05});
    bool good = true;
    for (double k2 : {0., 1., 100., 1e6})
        good &= close(scaled.long_range_filter(k2 / 250 / 250), base.long_range_filter(k2));
    check(good, "PM kernel and lambda use explicit coordinate/box units");
}
void backends_and_ghosts() {
    std::mt19937 rng(76);
    std::uniform_real_distribution<double> u(0, 1);
    std::vector<Particle> all(1000);
    for (std::size_t i = 0; i < all.size(); ++i) {
        all[i].id = i;
        all[i].mass = .5 + u(rng);
        for (auto & x : all[i].position)
            x = i % 2 ? .3 + .01 * u(rng) : u(rng);
    }
    const double lo = double(FML::ThisTask) / FML::NTasks, hi = double(FML::ThisTask + 1) / FML::NTasks;
    std::vector<Particle> local;
    for (const auto & p : all)
        if (p.position[0] >= lo && p.position[0] < hi)
            local.push_back(p);
    bool good = true;
    std::uint64_t accepted_nodes = 0;
    double max_tree_error = 0, max_fine_error = 0;
    for (double alpha : {-1., .6})
        for (double lambda : {.002, .02, .2}) {
            Kernel k({}, NewtonYukawa{alpha, lambda});
            auto ghosts = exchange_slab_ghosts(local, lo, hi, k);
            const auto truth = direct_sum(local, all, k), cell = CellList(ghosts.particles, k).compute(local);
            const auto exact = Octree(ghosts.particles, k, {0, 8, 40}).compute(local);
            const auto tree = Octree(ghosts.particles, k, {.3, 1, 40}).compute(local);
            const auto fine = Octree(ghosts.particles, k, {.1, 1, 40}).compute(local);
            Kernel n;
            const auto scale = direct_sum(local, all, n);
            accepted_nodes += tree.accepted_nodes;
            for (std::size_t i = 0; i < local.size(); ++i) {
                Vector dc, de, dt, df;
                for (int d = 0; d < 3; ++d) {
                    dc[d] = cell.acceleration[i][d] - truth.acceleration[i][d];
                    de[d] = exact.acceleration[i][d] - truth.acceleration[i][d];
                    dt[d] = tree.acceleration[i][d] - truth.acceleration[i][d];
                    df[d] = fine.acceleration[i][d] - truth.acceleration[i][d];
                }
                const double floor = std::max(1., norm(scale.acceleration[i]));
                max_tree_error = std::max(max_tree_error, norm(dt) / floor);
                max_fine_error = std::max(max_fine_error, norm(df) / floor);
                good &= norm(dc) < 2e-12 * floor && norm(de) < 2e-12 * floor;
            }
        }
    FML::SumOverTasks(&accepted_nodes);
    FML::MaxOverTasks(&max_tree_error);
    FML::MaxOverTasks(&max_fine_error);
    if (FML::ThisTask == 0)
        std::cout << "Yukawa accepted nodes " << accepted_nodes << " max tree component-scaled error " << max_tree_error
                  << " fine " << max_fine_error << '\n';
    check(good && accepted_nodes > 0 && max_tree_error < 1e-2 && max_fine_error < 1e-3 &&
              max_fine_error < max_tree_error,
          "Yukawa cell/tree/MPI ghosts agree with direct forces, including alpha=-1 cancellations");
    bool rejected = true;
    if (FML::NTasks > 1) {
        try {
            exchange_slab_ghosts(local, lo, hi, Kernel({}, NewtonYukawa{FML::ThisTask == 0 ? .4 : .5, .1}));
            rejected = false;
        } catch (const std::invalid_argument &) {
        }
    }
    check(rejected, "MPI collectively rejects mismatched Yukawa parameters");
}
int main() {
    reference_values();
    kernels();
    pm();
    backends_and_ghosts();
    if (FML::ThisTask == 0)
        std::cout << "Yukawa failures: " << failures << '\n';
    return failures ? 1 : 0;
}
