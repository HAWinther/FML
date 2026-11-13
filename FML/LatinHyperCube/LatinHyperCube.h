#ifndef FML_LATINHYPERCUBE_HEADER
#define FML_LATINHYPERCUBE_HEADER

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <FML/KDTree/KDTree.h>

// LHS: McKay, Beckman & Conover (1979), doi:10.1080/00401706.1979.10489755.
// OA-LHS: Tang (1993), doi:10.1080/01621459.1993.10476423.
// Independent implementation following SciPy; optimization sources and differences: README.md.
namespace FML {

    /// Space-filling designs of experiments (sampling of parameter spaces)
    namespace SAMPLING {

        //============================================================================
        /// Latin hypercube sampling (LHS) of the unit hypercube [0,1)^d, after scipy.stats.qmc.LatinHypercube
        /// (scipy 1.18). See README.md for the details and the differences from scipy.
        ///
        ///   strength 1: n points, every one of the n equal strata of every dimension holds exactly one point.
        ///   strength 2: orthogonal-array based LHS (Bose construction, OA(p^2, p+1, p, 2)); requires n = p^2 with p
        ///               prime and d <= p + 1. Every 2D projection also has one point in each of the p x p cells.
        ///   scramble  : true: the points are uniformly random within their cells; false: at the cell centres.
        ///   optimization (applied after the LHS is drawn):
        ///     random_cd : swap coordinates of two points (within one dimension) when that lowers the centered
        ///                 discrepancy; keeps the LHS property (not the strength-2 orthogonality)
        ///     lloyd     : a few damped Lloyd (centroidal Voronoi) iterations; spreads the points more evenly but
        ///                 does not keep the LHS property. Requires d >= 2.
        ///
        /// The random numbers come from std::mt19937_64 with portable conversions, so a given seed gives the same
        /// points with every compiler and standard library (but not the same points as scipy/numpy).
        ///
        /// Usage:
        ///   FML::SAMPLING::LatinHyperCube lhs(3, true, 1, FML::SAMPLING::LHSOptimization::random_cd, 1234);
        ///   auto points = lhs.random(100);                                  // points[i][j], i < n, j < d
        ///   auto params = FML::SAMPLING::scale(points, {0.1, 0.6, 0.9}, {0.5, 1.0, 1.0}); // to [l, u)
        //============================================================================

        /// A sample: sample[i] is point i, sample[i][j] its coordinate j
        using Sample = std::vector<std::vector<double>>;

        enum class LHSOptimization { none, random_cd, lloyd };

        /// Settings of the optimizations (the defaults are scipy's where scipy has the setting)
        struct LHSOptimizationOptions {
            /// random_cd: at most this many proposed swaps
            size_t random_cd_max_iterations = 10000;
            /// random_cd: stop after this many proposed swaps in a row without improvement
            size_t random_cd_max_no_improvement = 100;
            /// lloyd: at most this many iterations
            int lloyd_max_iterations = 10;
            /// lloyd: stop when the minimum L1 distance between the points changes by less than this
            double lloyd_tolerance = 1e-5;
            /// lloyd: the Voronoi centroids (clipped to the unit cube) are estimated from this many uniform Monte
            /// Carlo points per sample point
            size_t lloyd_points_per_sample = 200;
        };

        namespace detail {

            /// mt19937_64 with portable conversions (the std:: distributions differ between standard libraries)
            class PortableRNG {
                std::mt19937_64 gen;

              public:
                explicit PortableRNG(uint64_t seed = 0) : gen(seed) {}
                void seed(uint64_t s) { gen.seed(s); }
                /// Uniform in [0,1) with 53 random bits
                double uniform() { return double(gen() >> 11) * 0x1.0p-53; }
                /// Uniform integer in [0, range) without modulo bias (range > 0)
                uint64_t below(uint64_t range) {
                    const uint64_t threshold = (0 - range) % range; // 2^64 mod range
                    while (true) {
                        const uint64_t r = gen();
                        if (r >= threshold)
                            return r % range;
                    }
                }
                /// Fisher-Yates shuffle
                template <class T>
                void shuffle(std::vector<T> & v) {
                    for (size_t i = v.size(); i > 1; --i)
                        std::swap(v[i - 1], v[size_t(below(i))]);
                }
            };

            inline bool is_prime(long long p) {
                if (p < 2)
                    return false;
                for (long long q = 2; q * q <= p; q++)
                    if (p % q == 0)
                        return false;
                return true;
            }

            /// Throws unless all the points have d coordinates in [0,1]
            inline void check_unit_sample(const Sample & sample, const std::string & who) {
                if (sample.empty())
                    return;
                const size_t d = sample[0].size();
                for (const auto & x : sample) {
                    if (x.size() != d)
                        throw std::invalid_argument(who + ": the points have different numbers of coordinates");
                    for (double v : x)
                        if (not(v >= 0.0 and v <= 1.0))
                            throw std::invalid_argument(who + ": the sample is not in the unit hypercube [0,1]^d");
                }
            }

            // Centered L2 discrepancy (Hickernell 1998), the factors of its three terms. z = x - 1/2
            inline double cd_row_factor(double z) { return 1.0 + 0.5 * std::abs(z) - 0.5 * z * z; }
            inline double cd_pair_factor(double zi, double zj) {
                return 1.0 + 0.5 * std::abs(zi) + 0.5 * std::abs(zj) - 0.5 * std::abs(zi - zj);
            }

            /// The change of the (squared) centered discrepancy when sample[i1][k] and sample[i2][k] are swapped.
            /// O(n d): only the terms with i1 or i2 change (the (i1, i2) pair term is symmetric under the swap)
            inline double cd_swap_change(const Sample & sample, size_t i1, size_t i2, size_t k) {
                if (i1 == i2)
                    return 0.0;
                const size_t n = sample.size();
                const size_t d = sample[0].size();
                const double a = sample[i1][k] - 0.5; // old z of i1 in column k (the new z of i2)
                const double b = sample[i2][k] - 0.5; // old z of i2 in column k (the new z of i1)

                double r1 = 1.0, r2 = 1.0, k11 = 1.0, k22 = 1.0;
                for (size_t m = 0; m < d; m++) {
                    const double z1 = sample[i1][m] - 0.5, z2 = sample[i2][m] - 0.5;
                    r1 *= cd_row_factor(z1);
                    r2 *= cd_row_factor(z2);
                    k11 *= 1.0 + std::abs(z1);
                    k22 *= 1.0 + std::abs(z2);
                }
                const double d_rows =
                    r1 * (cd_row_factor(b) / cd_row_factor(a) - 1.0) + r2 * (cd_row_factor(a) / cd_row_factor(b) - 1.0);
                const double d_diag =
                    k11 * ((1.0 + std::abs(b)) / (1.0 + std::abs(a)) - 1.0) + k22 * ((1.0 + std::abs(a)) / (1.0 + std::abs(b)) - 1.0);

                double d_pairs = 0.0;
                for (size_t j = 0; j < n; j++) {
                    if (j == i1 or j == i2)
                        continue;
                    double p1 = 1.0, p2 = 1.0;
                    for (size_t m = 0; m < d; m++) {
                        const double zj = sample[j][m] - 0.5;
                        p1 *= cd_pair_factor(sample[i1][m] - 0.5, zj);
                        p2 *= cd_pair_factor(sample[i2][m] - 0.5, zj);
                    }
                    const double zjk = sample[j][k] - 0.5;
                    d_pairs += p1 * (cd_pair_factor(b, zjk) / cd_pair_factor(a, zjk) - 1.0) +
                               p2 * (cd_pair_factor(a, zjk) / cd_pair_factor(b, zjk) - 1.0);
                }
                const double nn = double(n);
                return -2.0 / nn * d_rows + (d_diag + 2.0 * d_pairs) / (nn * nn);
            }

            /// The smallest L1 (cityblock) distance between two points (scipy's convergence measure for Lloyd)
            inline double min_l1_distance(const Sample & sample) {
                double best = std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < sample.size(); i++)
                    for (size_t j = i + 1; j < sample.size(); j++) {
                        double dist = 0.0;
                        for (size_t m = 0; m < sample[i].size(); m++)
                            dist += std::abs(sample[i][m] - sample[j][m]);
                        best = std::min(best, dist);
                    }
                return best;
            }
        } // namespace detail

        //============================================================================
        /// The centered L2 discrepancy of a sample in [0,1]^d (squared, as scipy.stats.qmc.discrepancy with
        /// method = "CD"). Lower is more uniform. O(n^2 d)
        //============================================================================
        inline double centered_discrepancy(const Sample & sample) {
            detail::check_unit_sample(sample, "centered_discrepancy");
            if (sample.empty())
                throw std::invalid_argument("centered_discrepancy: empty sample");
            const size_t n = sample.size();
            const size_t d = sample[0].size();
            double rows = 0.0, pairs = 0.0;
            for (size_t i = 0; i < n; i++) {
                double r = 1.0, kii = 1.0;
                for (size_t m = 0; m < d; m++) {
                    const double z = sample[i][m] - 0.5;
                    r *= detail::cd_row_factor(z);
                    kii *= 1.0 + std::abs(z);
                }
                rows += r;
                pairs += kii;
                for (size_t j = i + 1; j < n; j++) {
                    double p = 1.0;
                    for (size_t m = 0; m < d; m++)
                        p *= detail::cd_pair_factor(sample[i][m] - 0.5, sample[j][m] - 0.5);
                    pairs += 2.0 * p;
                }
            }
            const double nn = double(n);
            return std::pow(13.0 / 12.0, double(d)) - 2.0 / nn * rows + pairs / (nn * nn);
        }

        //============================================================================
        /// Map a sample from [0,1)^d to [l_bounds, u_bounds) (as scipy.stats.qmc.scale). reverse = true maps from
        /// [l_bounds, u_bounds) back to the unit hypercube
        //============================================================================
        inline Sample scale(const Sample & sample,
                            const std::vector<double> & l_bounds,
                            const std::vector<double> & u_bounds,
                            bool reverse = false) {
            const size_t d = l_bounds.size();
            if (u_bounds.size() != d)
                throw std::invalid_argument("scale: l_bounds and u_bounds have different sizes");
            for (size_t m = 0; m < d; m++)
                if (not(l_bounds[m] < u_bounds[m]))
                    throw std::invalid_argument("scale: we need l_bounds < u_bounds");
            for (const auto & x : sample)
                if (x.size() != d)
                    throw std::invalid_argument("scale: the bounds and the sample have different dimensions");
            if (not reverse)
                detail::check_unit_sample(sample, "scale");

            Sample result(sample);
            for (auto & x : result)
                for (size_t m = 0; m < d; m++)
                    x[m] = reverse ? (x[m] - l_bounds[m]) / (u_bounds[m] - l_bounds[m])
                                   : l_bounds[m] + x[m] * (u_bounds[m] - l_bounds[m]);
            return result;
        }

        //============================================================================
        /// Optimization: random coordinate swaps that lower the centered discrepancy (scipy's "random-cd").
        /// Keeps the LHS property
        //============================================================================
        inline void optimize_random_cd(Sample & sample,
                                       detail::PortableRNG & rng,
                                       size_t max_iterations = 10000,
                                       size_t max_no_improvement = 100) {
            const size_t n = sample.size();
            if (n <= 1 or sample[0].size() <= 1) // the discrepancy cannot be lowered by swaps
                return;
            const size_t d = sample[0].size();
            size_t no_improvement = 0;
            for (size_t it = 0; it < max_iterations and no_improvement < max_no_improvement; it++) {
                const size_t col = size_t(rng.below(d));
                const size_t row1 = size_t(rng.below(n));
                const size_t row2 = size_t(rng.below(n));
                if (detail::cd_swap_change(sample, row1, row2, col) < 0.0) {
                    std::swap(sample[row1][col], sample[row2][col]);
                    no_improvement = 0;
                } else {
                    no_improvement++;
                }
            }
        }

        //============================================================================
        /// Optimization: damped Lloyd iterations towards a centroidal Voronoi tessellation of [0,1]^d (scipy's
        /// "lloyd", with scipy's damping schedule, validity rule and stopping rule). The centroids of the Voronoi
        /// cells clipped to the unit cube are estimated with Monte Carlo points (scipy uses the mean of the Voronoi
        /// vertices from Qhull). Requires d >= 2. Does not keep the LHS property
        //============================================================================
        inline void optimize_lloyd(Sample & sample,
                                   detail::PortableRNG & rng,
                                   int max_iterations = 10,
                                   double tolerance = 1e-5,
                                   size_t points_per_sample = 200) {
            detail::check_unit_sample(sample, "optimize_lloyd");
            const size_t n = sample.size();
            if (n == 0)
                return;
            const size_t d = sample[0].size();
            if (d < 2)
                throw std::invalid_argument("optimize_lloyd: the sample must have d >= 2 (as in scipy)");
            if (n == 1 or max_iterations <= 0)
                return;
            points_per_sample = std::max<size_t>(points_per_sample, 1);

            // The damping factor goes from 1.9 (over-relaxation) to ~1.0
            const double root = -double(max_iterations) / std::log(0.1);
            double l1_old = detail::min_l1_distance(sample);
            std::vector<double> sums(n * d);
            std::vector<size_t> counts(n);
            std::vector<double> u(d);
            for (int iter = 0; iter < max_iterations; iter++) {
                const double decay = std::exp(-double(iter) / root) + 0.9;

                FML::KDTREE::KDTree tree(sample);
                std::fill(sums.begin(), sums.end(), 0.0);
                std::fill(counts.begin(), counts.end(), size_t(0));
                const size_t nmc = points_per_sample * n;
                for (size_t q = 0; q < nmc; q++) {
                    for (auto & v : u)
                        v = rng.uniform();
                    const size_t i = tree.nearest_index(u);
                    counts[i]++;
                    for (size_t m = 0; m < d; m++)
                        sums[i * d + m] += u[m];
                }

                // Move towards (beyond, while decay > 1) the centroid; keep the old point if the new one is outside
                std::vector<double> x(d);
                for (size_t i = 0; i < n; i++) {
                    if (counts[i] == 0)
                        continue;
                    bool valid = true;
                    for (size_t m = 0; m < d; m++) {
                        const double centroid = sums[i * d + m] / double(counts[i]);
                        x[m] = sample[i][m] + (centroid - sample[i][m]) * decay;
                        valid = valid and x[m] >= 0.0 and x[m] < 1.0;
                    }
                    if (valid)
                        sample[i] = x;
                }

                const double l1_new = detail::min_l1_distance(sample);
                if (std::abs(l1_new - l1_old) < tolerance)
                    break;
                l1_old = l1_new;
            }
        }

        //============================================================================
        /// Latin hypercube sampler (see the top of this file)
        //============================================================================
        class LatinHyperCube {
          public:
            /// d: the dimension. seed: the seed of the random numbers (none: from std::random_device; the seed used
            /// is available from get_seed())
            explicit LatinHyperCube(int d,
                                    bool scramble = true,
                                    int strength = 1,
                                    LHSOptimization optimization = LHSOptimization::none,
                                    std::optional<uint64_t> seed = std::nullopt,
                                    LHSOptimizationOptions options = {})
                : d(d), scramble(scramble), strength(strength), optimization(optimization), options(options) {
                if (d < 1)
                    throw std::invalid_argument("LatinHyperCube: the dimension d must be >= 1");
                if (strength != 1 and strength != 2)
                    throw std::invalid_argument("LatinHyperCube: strength must be 1 or 2");
                if (optimization == LHSOptimization::lloyd and d < 2)
                    throw std::invalid_argument("LatinHyperCube: the lloyd optimization requires d >= 2");
                if (seed) {
                    base_seed = *seed;
                } else {
                    std::random_device rd;
                    base_seed = (uint64_t(rd()) << 32) ^ uint64_t(rd());
                }
                reset();
            }

            /// Draw n points in [0,1)^d: result[i][j] is coordinate j of point i
            Sample random(size_t n = 1) {
                Sample sample = (strength == 1) ? random_lhs(n) : random_oa_lhs(n);
                if (optimization == LHSOptimization::random_cd)
                    optimize_random_cd(
                        sample, rng, options.random_cd_max_iterations, options.random_cd_max_no_improvement);
                else if (optimization == LHSOptimization::lloyd)
                    optimize_lloyd(sample,
                                   rng,
                                   options.lloyd_max_iterations,
                                   options.lloyd_tolerance,
                                   options.lloyd_points_per_sample);
                num_generated += n;
                return sample;
            }

            /// Draw n points with integer coordinates in [l_bounds, u_bounds) ([l_bounds, u_bounds] if endpoint),
            /// as scipy's QMCEngine.integers: floor of the scaled sample
            std::vector<std::vector<long long>> integers(const std::vector<long long> & l_bounds,
                                                         const std::vector<long long> & u_bounds,
                                                         size_t n = 1,
                                                         bool endpoint = false) {
                if (l_bounds.size() != size_t(d) or u_bounds.size() != size_t(d))
                    throw std::invalid_argument("LatinHyperCube::integers: the bounds must have d elements");
                std::vector<double> lo(d), hi(d);
                for (int m = 0; m < d; m++) {
                    lo[m] = double(l_bounds[m]);
                    hi[m] = double(u_bounds[m] + (endpoint ? 1 : 0));
                }
                const Sample scaled = scale(random(n), lo, hi);
                std::vector<std::vector<long long>> result(n, std::vector<long long>(d));
                for (size_t i = 0; i < n; i++)
                    for (int m = 0; m < d; m++)
                        result[i][m] = std::min((long long)std::floor(scaled[i][m]), (long long)hi[m] - 1);
                return result;
            }

            /// Draw n points with integer coordinates in [0, u_bounds)
            std::vector<std::vector<long long>> integers(const std::vector<long long> & u_bounds,
                                                         size_t n = 1,
                                                         bool endpoint = false) {
                return integers(std::vector<long long>(d, 0), u_bounds, n, endpoint);
            }

            /// Back to the state after construction (the same points again)
            void reset() {
                rng.seed(base_seed);
                num_generated = 0;
            }

            /// Skip n points (as scipy: draws and discards a sample of n points)
            void fast_forward(size_t n) { random(n); }

            int get_dimension() const { return d; }
            int get_strength() const { return strength; }
            bool get_scramble() const { return scramble; }
            LHSOptimization get_optimization() const { return optimization; }
            uint64_t get_seed() const { return base_seed; }
            /// The number of points drawn since construction or the last reset
            size_t get_num_generated() const { return num_generated; }

          private:
            int d;
            bool scramble;
            int strength;
            LHSOptimization optimization;
            LHSOptimizationOptions options;
            uint64_t base_seed{0};
            detail::PortableRNG rng;
            size_t num_generated{0};

            /// Strength 1: in every dimension a random permutation of the n strata
            Sample random_lhs(size_t n) {
                Sample sample(n, std::vector<double>(d));
                std::vector<size_t> perm(n);
                for (int m = 0; m < d; m++) {
                    for (size_t i = 0; i < n; i++)
                        perm[i] = i;
                    rng.shuffle(perm);
                    for (size_t i = 0; i < n; i++) {
                        const double offset = scramble ? rng.uniform() : 0.5;
                        sample[i][m] = (double(perm[i]) + offset) / double(n);
                    }
                }
                return sample;
            }

            /// Strength 2: the Bose orthogonal array OA(p^2, p+1, p, 2) with the symbols of every column randomly
            /// relabelled, each symbol's p rows refined with a 1D LHS of size p (Tang 1993; as scipy)
            Sample random_oa_lhs(size_t n) {
                const long long p = std::llround(std::sqrt(double(n)));
                if (p * p != (long long)n or not detail::is_prime(p))
                    throw std::invalid_argument("LatinHyperCube: strength 2 needs n = p^2 with p prime (n = " +
                                                std::to_string(n) + ")");
                if (d > p + 1)
                    throw std::invalid_argument("LatinHyperCube: n is too small for d with strength 2: we need "
                                                "d <= sqrt(n) + 1");

                // Bose construction: row (a, b) has columns a, b, a + b, a + 2b, ..., a + (p-1) b (mod p)
                const size_t ncol = size_t(p + 1);
                std::vector<std::vector<long long>> oa(n, std::vector<long long>(ncol));
                for (long long a = 0; a < p; a++)
                    for (long long b = 0; b < p; b++) {
                        auto & row = oa[size_t(a * p + b)];
                        row[0] = a;
                        row[1] = b;
                        for (long long c = 1; c < p; c++)
                            row[size_t(c + 1)] = (a + c * b) % p;
                    }

                // Random bijection of the symbols in each column
                std::vector<long long> relabel(p);
                for (size_t j = 0; j < ncol; j++) {
                    for (long long s = 0; s < p; s++)
                        relabel[s] = s;
                    rng.shuffle(relabel);
                    for (auto & row : oa)
                        row[j] = relabel[row[j]];
                }

                // The p rows with symbol s in column j get the values (s + LHS_p) / p
                Sample sample(n, std::vector<double>(d));
                std::vector<size_t> perm(p);
                std::vector<size_t> rows;
                for (int j = 0; j < d; j++) {
                    for (long long s = 0; s < p; s++) {
                        rows.clear();
                        for (size_t i = 0; i < n; i++)
                            if (oa[i][j] == s)
                                rows.push_back(i);
                        for (long long t = 0; t < p; t++)
                            perm[t] = size_t(t);
                        rng.shuffle(perm);
                        for (size_t r = 0; r < rows.size(); r++) {
                            const double offset = scramble ? rng.uniform() : 0.5;
                            const double lhs = (double(perm[r]) + offset) / double(p);
                            sample[rows[r]][j] = (double(s) + lhs) / double(p);
                        }
                    }
                }
                return sample;
            }
        };

    } // namespace SAMPLING
} // namespace FML

#endif
