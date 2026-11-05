#ifndef FML_THREE_POINT_CORRELATION_H
#define FML_THREE_POINT_CORRELATION_H
#include <FML/KDTree/KDTree.h>
#include <array>
#include <cstring>
#include <set>

namespace FML::THREEPOINT {
    using Point = std::array<double, 3>;
    struct Sample {
        Point position{};
        std::uint64_t id{};
    };
    struct Options {
        double boxsize{1};
        std::vector<double> edges{.05, .1, .2};
        int lmax{4};
    };
    struct Result {
        Options options;
        std::array<size_t, 2> catalogue_sizes{};
        // Layout: [center type * 4 + first leg type * 2 + second leg type][ell][bin1][bin2].
        // Type 0 = data, 1 = random. Ordered distinct-object triplets, sum of P_ell(cos angle).
        std::vector<double> moments;
        // zeta = sum_ell zeta_ell P_ell(mu), averaged over both radial shell volumes.
        std::vector<double> zeta;
        size_t index(int ell, size_t a, size_t b) const {
            const size_t n = options.edges.size() - 1;
            if (ell < 0 || ell > options.lmax || a >= n || b >= n)
                throw std::out_of_range("3PCF multipole/bin index");
            return (size_t(ell) * n + a) * n + b;
        }
        double moment(int center, int leg1, int leg2, int ell, size_t a, size_t b) const {
            if (center < 0 || center > 1 || leg1 < 0 || leg1 > 1 || leg2 < 0 || leg2 > 1)
                throw std::out_of_range("3PCF catalogue index");
            return moments.at(size_t(center * 4 + leg1 * 2 + leg2) * zeta.size() + index(ell, a, b));
        }
    };
    namespace detail {
        inline void require(bool valid, const char * message) {
            int ok = valid;
            FML::MinOverTasks(&ok);
            if (!ok)
                throw std::invalid_argument(message);
        }
        inline void word(unsigned long long & h, unsigned long long x) { h = (h ^ x) * 1099511628211ULL; }
        inline void real(unsigned long long & h, double x) {
            std::uint64_t bits;
            std::memcpy(&bits, &x, sizeof(bits));
            word(h, bits);
        }
        inline void validate(const std::array<const std::vector<Sample> *, 2> & cats, const Options & o) {
            bool valid = std::isfinite(o.boxsize * o.boxsize) && o.boxsize > 0 && o.boxsize * o.boxsize > 0 &&
                         o.lmax >= 0 && o.lmax <= 16 && o.edges.size() >= 2 && o.edges.size() <= 129;
            for (size_t i = 0; i < o.edges.size(); ++i) {
                valid &= std::isfinite(o.edges[i]) && o.edges[i] > 0 && o.edges[i] < o.boxsize / 2;
                if (i) {
                    const double a = o.edges[i - 1] / o.boxsize, b = o.edges[i] / o.boxsize;
                    valid &= o.edges[i] > o.edges[i - 1] && b * b * b - a * a * a > 0;
                }
            }
            if (o.edges.size() >= 2) {
                double smallest = 1;
                for (size_t i = 1; i < o.edges.size(); ++i) {
                    const double a = o.edges[i - 1] / o.boxsize, b = o.edges[i] / o.boxsize;
                    smallest = std::min(smallest, 4 * M_PI / 3 * (b * b * b - a * a * a));
                }
                valid &=
                    o.edges.front() * o.edges.front() > 0 && smallest > 0 && std::isfinite(1 / smallest / smallest);
            }
            for (const auto * cat : cats) {
                valid &= cat->size() >= 3;
                std::set<std::uint64_t> ids;
                for (const auto & p : *cat) {
                    valid &= ids.insert(p.id).second;
                    for (double x : p.position)
                        valid &= std::isfinite(x) && x >= 0 && x < o.boxsize;
                }
            }
            require(valid,
                    "3PCF needs two catalogues with >=3 unique IDs, positions in [0,L), 0<edges<L/2, 0<=lmax<=16 and "
                    "<=128 bins");
            unsigned long long h = 1469598103934665603ULL;
            real(h, o.boxsize);
            word(h, o.lmax);
            word(h, o.edges.size());
            for (double x : o.edges)
                real(h, x);
            for (const auto * cat : cats) {
                word(h, cat->size());
                for (const auto & p : *cat) {
                    word(h, p.id);
                    for (double x : p.position)
                        real(h, x);
                }
            }
            auto lo = h, hi = h;
            FML::MinOverTasks(&lo);
            FML::MaxOverTasks(&hi);
            require(lo == hi, "3PCF requires identical replicated catalogues and options on every MPI rank");
        }
        // Orthonormal real spherical harmonics, index ell^2 + ell + m.
        // Condon-Shortley phase is immaterial to the addition theorem; include it consistently.
        inline void harmonics(const Point & u, int lmax, std::vector<double> & y) {
            y.assign(size_t(lmax + 1) * (lmax + 1), 0);
            const double z = std::clamp(u[2], -1., 1.);
            const double phi = std::atan2(u[1], u[0]);
            double pmm = 1;
            for (int m = 0; m <= lmax; ++m) {
                if (m)
                    pmm *= -(2 * m - 1) * std::sqrt(std::max(0., 1 - z * z));
                double prev = 0, p = pmm;
                for (int l = m; l <= lmax; ++l) {
                    if (l > m) {
                        const double next = ((2 * l - 1) * z * p - (l + m - 1) * prev) / (l - m);
                        prev = p;
                        p = next;
                    }
                    const double norm = std::sqrt((2 * l + 1) / (4 * M_PI) *
                                                  std::exp(std::lgamma(l - m + 1.) - std::lgamma(l + m + 1.)));
                    if (m == 0)
                        y[l * l + l] = norm * p;
                    else {
                        y[l * l + l + m] = std::sqrt(2.) * norm * p * std::cos(m * phi);
                        y[l * l + l - m] = std::sqrt(2.) * norm * p * std::sin(m * phi);
                    }
                }
            }
        }
        inline double falling(size_t n, int k) {
            double x = 1;
            for (int i = 0; i < k; ++i)
                x *= double(n - i);
            return x;
        }
    } // namespace detail
    // Collective, replicated inputs. Each rank owns a disjoint set of primary indices.
    // No OpenMP region is entered: one reusable coefficient buffer per rank.
    inline Result periodic(const std::vector<Sample> & data, const std::vector<Sample> & random, Options o = {}) {
        const std::array<const std::vector<Sample> *, 2> cats{&data, &random};
        detail::validate(cats, o);
        std::array<FML::KDTREE::KDTree, 2> trees;
        for (int t = 0; t < 2; ++t) {
            FML::KDTREE::pointVec points;
            points.reserve(cats[t]->size());
            for (const auto & p : *cats[t])
                points.emplace_back(p.position.begin(), p.position.end());
            trees[t] = FML::KDTREE::KDTree(points, o.boxsize);
        }
        const size_t nb = o.edges.size() - 1, nh = size_t(o.lmax + 1) * (o.lmax + 1),
                     stride = size_t(o.lmax + 1) * nb * nb;
        Result result{o, {data.size(), random.size()}, std::vector<double>(8 * stride), std::vector<double>(stride)};
        std::array<std::vector<double>, 2> a{std::vector<double>(nb * nh), std::vector<double>(nb * nh)};
        std::array<std::vector<double>, 2> number{std::vector<double>(nb), std::vector<double>(nb)};
        std::vector<double> y, d2;
        std::vector<size_t> ids;
        for (int c = 0; c < 2; ++c)
            for (size_t center = FML::ThisTask; center < cats[c]->size(); center += FML::NTasks) {
                for (int t = 0; t < 2; ++t) {
                    std::fill(a[t].begin(), a[t].end(), 0);
                    std::fill(number[t].begin(), number[t].end(), 0);
                    const auto & x = (*cats[c])[center].position;
                    trees[t].neighborhood_indices_and_distances({x.begin(), x.end()}, o.edges.back(), ids, d2);
                    for (size_t q = 0; q < ids.size(); ++q) {
                        if (t == c && ids[q] == center)
                            continue;
                        Point u{};
                        for (int d = 0; d < 3; ++d) {
                            u[d] = (*cats[t])[ids[q]].position[d] - x[d];
                            if (u[d] > o.boxsize / 2)
                                u[d] -= o.boxsize;
                            if (u[d] < -o.boxsize / 2)
                                u[d] += o.boxsize;
                        }
                        // Recompute from the same minimum-image vector used for the angular direction.
                        const double r = std::hypot(u[0], u[1], u[2]);
                        if (r < o.edges.front() || r >= o.edges.back())
                            continue;
                        const size_t b = std::upper_bound(o.edges.begin(), o.edges.end(), r) - o.edges.begin() - 1;
                        for (double & v : u)
                            v /= r;
                        detail::harmonics(u, o.lmax, y);
                        for (size_t h = 0; h < nh; ++h)
                            a[t][b * nh + h] += y[h];
                        number[t][b] += 1;
                    }
                }
                for (int t = 0; t < 2; ++t)
                    for (int v = 0; v < 2; ++v)
                        for (int l = 0; l <= o.lmax; ++l)
                            for (size_t b = 0; b < nb; ++b)
                                for (size_t d = 0; d < nb; ++d) {
                                    double sum = 0;
                                    for (int m = -l; m <= l; ++m)
                                        sum += a[t][b * nh + l * l + l + m] * a[v][d * nh + l * l + l + m];
                                    sum *= 4 * M_PI / (2 * l + 1);
                                    // A neighbor cannot occupy both legs of a distinct-object triplet.
                                    if (t == v && b == d)
                                        sum -= number[t][b];
                                    result.moments[(c * 4 + t * 2 + v) * stride + result.index(l, b, d)] += sum;
                                }
            }
        FML::SumArrayOverTasks(result.moments.data(), result.moments.size());
        std::vector<double> fraction(nb);
        for (size_t b = 0; b < nb; ++b) {
            const double x = o.edges[b] / o.boxsize, z = o.edges[b + 1] / o.boxsize;
            fraction[b] = 4 * M_PI / 3 * (z * z * z - x * x * x);
        }
        for (int c = 0; c < 2; ++c)
            for (int t = 0; t < 2; ++t)
                for (int v = 0; v < 2; ++v) {
                    const int nr = c + t + v;
                    const double norm = detail::falling(data.size(), 3 - nr) * detail::falling(random.size(), nr);
                    const double sign = nr % 2 ? -1 : 1;
                    for (int l = 0; l <= o.lmax; ++l)
                        for (size_t b = 0; b < nb; ++b)
                            for (size_t d = 0; d < nb; ++d) {
                                const size_t i = result.index(l, b, d);
                                result.zeta[i] += sign * (2 * l + 1) *
                                                  result.moments[(c * 4 + t * 2 + v) * stride + i] / norm /
                                                  fraction[b] / fraction[d];
                            }
                }
        return result;
    }
} // namespace FML::THREEPOINT
#endif
