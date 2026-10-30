#ifndef FML_SURVEY_MESH_HEADER
#define FML_SURVEY_MESH_HEADER
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/ChunkedExchange.h>
#include <array>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace FML::SURVEY {
    using Vector = std::array<double, 3>;
    using Mesh = FML::GRID::FFTWGrid<3>;
    using Complex = FML::GRID::ComplexType;
    using Real = FML::GRID::FloatType;

    inline void require(bool ok, const char * message) {
        int valid = ok;
        FML::MinOverTasks(&valid);
        if (!valid)
            throw std::invalid_argument(message);
    }
    inline double dot(const Vector & a, const Vector & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
    inline Vector unit(Vector v) {
        const double norm = std::hypot(v[0], v[1], v[2]);
        if (norm > 0)
            for (auto & x : v)
                x /= norm;
        return v;
    }
    struct CatalogPoint {
        Vector position{}; // Physical Cartesian coordinates; never periodic-wrapped.
        double weight = 1; // Complete effective weight, including FKP if desired.
        double nbar = 0;   // Physical selection density, for optional I2 normalization.
    };
    inline double fkp_weight(double nbar, double p0) {
        if (!(nbar >= 0) || !(p0 >= 0) || !std::isfinite(nbar) || !std::isfinite(p0))
            throw std::invalid_argument("FKP requires finite nonnegative nbar and P0");
        return 1.0 / (1.0 + nbar * p0);
    }
    struct Geometry {
        int nmesh = 64;
        double boxsize = 1000;
        Vector minimum{};
        Vector observer{};
        bool local_los = true;
        Vector fixed_los{0, 0, 1};
        double cellsize() const { return boxsize / nmesh; }
        double volume() const { return boxsize * boxsize * boxsize; }
        Vector position(const Mesh & mesh, FML::IndexIntType index) const {
            auto c = mesh.get_coord_from_index(index);
            c[0] += mesh.get_local_x_start();
            Vector x;
            for (int d = 0; d < 3; ++d)
                x[d] = minimum[d] + c[d] * cellsize();
            return x;
        }
        Vector los(const Vector & position) const {
            if (!local_los)
                return unit(fixed_los);
            Vector r;
            for (int d = 0; d < 3; ++d)
                r[d] = position[d] - observer[d];
            return unit(r);
        }
        void validate() const {
            bool ok = nmesh >= 4 && nmesh % FML::NTasks == 0 && boxsize > 0 && volume() > 0 && std::isfinite(volume());
            for (int d = 0; d < 3; ++d)
                ok &= std::isfinite(minimum[d]) && std::isfinite(observer[d]) && std::isfinite(fixed_los[d]);
            require(ok && (local_los || dot(fixed_los, fixed_los) > 0),
                    "Invalid survey geometry/LOS; mesh must be divisible by MPI ranks");
        }
    };
    namespace detail {
        // FFTWGrid currently exposes this read-only query through a non-const method.
        inline bool is_real(const Mesh & mesh) { return const_cast<Mesh &>(mesh).get_grid_status_real(); }
        template <class Record>
        std::vector<Record> route(std::vector<std::vector<Record>> & outgoing) {
            std::vector<unsigned long long> counts(FML::NTasks), received(FML::NTasks);
            for (int t = 0; t < FML::NTasks; ++t)
                counts[t] = outgoing[t].size();
#ifdef USE_MPI
            MPI_Alltoall(counts.data(),
                         1,
                         MPI_UNSIGNED_LONG_LONG,
                         received.data(),
                         1,
                         MPI_UNSIGNED_LONG_LONG,
                         FML::communicator);
#else
            received = counts;
#endif
            std::vector<Record> result;
            size_t total = 0;
            bool fits = true;
            for (auto n : received) {
                if (n > result.max_size() - total) {
                    fits = false;
                    break;
                }
                total += size_t(n);
            }
            require(fits, "Survey exchange exceeds addressable vector size");
            result.reserve(total);
            FML::COMMUNICATION::exchange_records_chunked(outgoing, [&](const Record * p, size_t n) noexcept {
                result.insert(result.end(), p, p + n);
            });
            return result;
        }
        inline void validate_catalog(const std::vector<CatalogPoint> & points, const Geometry & geometry) {
            bool ok = true;
            for (const auto & p : points) {
                ok &= p.weight >= 0 && std::isfinite(p.weight) && p.nbar >= 0 && std::isfinite(p.nbar);
                for (int d = 0; d < 3; ++d) {
                    const double u = (p.position[d] - geometry.minimum[d]) / geometry.cellsize();
                    // Retain a mesh cell at each edge: no CIC cloud wraps around the box.
                    ok &= std::isfinite(u) && u >= 1 && u < geometry.nmesh - 2;
                }
                if (geometry.local_los && p.weight > 0) {
                    Vector r;
                    for (int d = 0; d < 3; ++d)
                        r[d] = p.position[d] - geometry.observer[d];
                    ok &= dot(r, r) > 0;
                }
            }
            require(ok, "Invalid survey catalogue: weights/densities or unpadded position/observer coincidence");
        }
        template <class Function>
        void cic(const Vector & x, const Geometry & g, Function function) {
            std::array<int, 3> base;
            Vector fraction;
            for (int d = 0; d < 3; ++d) {
                const double u = (x[d] - g.minimum[d]) / g.cellsize();
                base[d] = int(std::floor(u));
                fraction[d] = u - base[d];
            }
            for (int mask = 0; mask < 8; ++mask) {
                auto c = base;
                double w = 1;
                for (int d = 0; d < 3; ++d) {
                    const int side = (mask >> d) & 1;
                    c[d] += side;
                    w *= side ? fraction[d] : 1 - fraction[d];
                }
                if (w != 0)
                    function(c, w);
            }
        }
        inline Mesh empty_mesh(const Geometry & g, int ghosts = 0) {
            Mesh grid(g.nmesh, ghosts, ghosts);
            grid.set_grid_status_real(true);
            for (auto i : grid.get_real_range())
                grid.set_real_from_index(i, 0);
            return grid;
        }
        inline std::vector<double> legendre_coefficients(int ell) {
            std::vector<double> p0(ell + 1), p1(ell + 1);
            p0[0] = 1;
            if (!ell)
                return p0;
            p1[1] = 1;
            for (int n = 2; n <= ell; ++n) {
                std::vector<double> next(ell + 1);
                for (int j = 0; j < n; ++j)
                    next[j + 1] += (2 * n - 1.0) / n * p1[j];
                for (int j = 0; j <= n - 2; ++j)
                    next[j] -= (n - 1.0) / n * p0[j];
                p0 = std::move(p1);
                p1 = std::move(next);
            }
            return p1;
        }
        inline double legendre(int ell, double mu) {
            auto c = legendre_coefficients(ell);
            double value = 0;
            for (int j = ell; j >= 0; --j)
                value = value * mu + c[j];
            return value;
        }
        struct Moment {
            std::array<int, 3> exponent;
            double multiplicity;
            int degree;
        };
        inline std::vector<Moment> moments(int maxell) {
            auto factorial = [](int n) {
                double f = 1;
                for (int i = 2; i <= n; ++i)
                    f *= i;
                return f;
            };
            std::vector<Moment> result;
            for (int p = 0; p <= maxell; p += 2)
                for (int x = 0; x <= p; ++x)
                    for (int y = 0; y <= p - x; ++y) {
                        const int z = p - x - y;
                        result.push_back({{x, y, z}, factorial(p) / (factorial(x) * factorial(y) * factorial(z)), p});
                    }
            return result;
        }
        inline double monomial(const Vector & x, const Moment & m) {
            double v = 1;
            for (int d = 0; d < 3; ++d)
                for (int j = 0; j < m.exponent[d]; ++j)
                    v *= x[d];
            return v;
        }
        inline double cic_window(const Vector & k, const Geometry & g) {
            double w = 1;
            for (auto v : k) {
                const double x = .5 * v * g.cellsize();
                const double sinc = x == 0 ? 1 : std::sin(x) / x;
                w *= sinc * sinc;
            }
            return w;
        }
        inline void validate_edges(const std::vector<double> & edges, bool collective = true) {
            bool ok = edges.size() > 1;
            for (size_t i = 0; i < edges.size(); ++i)
                ok &= std::isfinite(edges[i]) && edges[i] >= 0 && (i == 0 || edges[i] > edges[i - 1]);
            if (collective)
                require(ok, "Survey bin edges must be finite, nonnegative and strictly increasing");
            else if (!ok)
                throw std::invalid_argument("Survey bin edges must be finite, nonnegative and strictly increasing");
        }
        inline int bin(double value, const std::vector<double> & edges) {
            if (value < edges.front() || value >= edges.back())
                return -1;
            return int(std::upper_bound(edges.begin(), edges.end(), value) - edges.begin() - 1);
        }
        inline void sum(std::vector<double> & a) {
#ifdef USE_MPI
            // Chunked even for ancillary bin/moment arrays.
            for (size_t offset = 0; offset < a.size();) {
                const int n = int(std::min(a.size() - offset, size_t(INT_MAX)));
                MPI_Allreduce(MPI_IN_PLACE, a.data() + offset, n, MPI_DOUBLE, MPI_SUM, FML::communicator);
                offset += n;
            }
#else
            (void)a;
#endif
        }
    } // namespace detail
    inline double weight_sum(const std::vector<CatalogPoint> & points) {
        double value = 0;
        for (const auto & p : points)
            value += p.weight;
        FML::SumOverTasks(&value);
        return value;
    }
    inline Mesh paint_catalog(const std::vector<CatalogPoint> & points, const Geometry & g) {
        g.validate();
        detail::validate_catalog(points, g);
        struct PaintPoint {
            Vector position;
            double weight;
        };
        std::vector<std::vector<PaintPoint>> outgoing(FML::NTasks);
        const int slab = g.nmesh / FML::NTasks;
        for (const auto & p : points) {
            if (p.weight == 0)
                continue;
            const double x = (p.position[0] - g.minimum[0]) / g.cellsize();
            const int left = int(std::floor(x));
            const int first = left / slab, second = (left + 1) / slab;
            outgoing[first].push_back({p.position, p.weight});
            // Send the particle once to each intersecting slab, rather than staging
            // eight full cell records. Only an x-boundary CIC cloud needs a duplicate.
            if (second != first && x > left)
                outgoing[second].push_back({p.position, p.weight});
        }
        auto received = detail::route(outgoing);
        auto grid = detail::empty_mesh(g);
        const int start = int(grid.get_local_x_start());
        for (const auto & p : received)
            detail::cic(p.position, g, [&](auto c, double w) {
                c[0] -= start;
                if (c[0] >= 0 && c[0] < grid.get_local_nx())
                    grid.set_real(c, grid.get_real(c) + Real(p.weight * w));
            });
        return grid;
    }
    // Normalization is explicit: I2 = alpha sum_random nbar*w^2, unless overridden.
    inline double normalization(const std::vector<CatalogPoint> & randoms, double alpha, double override_i2 = 0) {
        double i2 = 0;
        for (const auto & p : randoms)
            i2 += p.nbar * p.weight * p.weight;
        FML::SumOverTasks(&i2);
        i2 = override_i2 > 0 ? override_i2 : alpha * i2;
        require(i2 > 0 && std::isfinite(i2), "Supply positive I2 or positive random selection densities nbar");
        return i2;
    }
} // namespace FML::SURVEY
#endif
