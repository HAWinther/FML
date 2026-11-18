#ifndef FML_FIELD_LEVEL_EULERIAN_BIAS_H
#define FML_FIELD_LEVEL_EULERIAN_BIAS_H

#include <FML/FFTWGrid/GridInterface.h>
#include <FML/Global/MPICounts.h>
#include <array>
#include <stdexcept>

namespace FML::FIELDLEVEL::EULERIAN {
    enum class BandGeometry { Sphere, Cube };

    // Physical wavenumbers; strict boundary. DC and every Nyquist plane are excluded.
    struct SharpKBand {
        double cutoff;
        BandGeometry geometry = BandGeometry::Sphere;
        bool contains(const std::array<int, 3> & n, double boxsize, int mesh) const {
            double n2 = 0, largest = 0;
            for (int v : n) {
                if (mesh % 2 == 0 && std::abs(v) == mesh / 2)
                    return false;
                n2 += double(v) * v;
                largest = std::max(largest, double(std::abs(v)));
            }
            const double radius = geometry == BandGeometry::Sphere ? std::sqrt(n2) : largest;
            return n2 > 0 && 2 * M_PI * radius / boxsize < cutoff;
        }
    };

    // A local real field is ordered exactly as Grid::get_real_range(), without ghosts.
    // All public operations are collective on FML::communicator, with replicated configuration.
    // Input: evolved matter overdensity. Output: Eulerian (not advected Lagrangian) operators.
    template <class Grid = FML::GRID::FFTWGrid<3>>
    class QuadraticBias {
        static_assert(std::tuple_size<decltype(std::declval<const Grid &>().get_fourier_integer_wavevector_from_index(
                              0))>::value == 3,
                      "Eulerian bias requires a three-dimensional grid");

      public:
        using Field = std::vector<double>;
        using Columns = std::array<Field, 4>;
        enum Column { Density, DensitySquared, TidalSquared, Laplacian };

      private:
        const double boxsize_;
        const SharpKBand band_;
        const int mesh_;
        const size_t cells_;
        const std::array<std::array<int, 2>, 3> ranges_;
        static constexpr int pairs_[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
        static void require(bool ok, const char * message) {
            int bad = !ok;
            FML::MaxOverTasks(&bad);
            if (bad)
                throw std::invalid_argument(message);
        }
        static std::array<std::array<int, 2>, 3> ranges(const Grid & grid) {
            std::array<std::array<int, 2>, 3> out;
            for (int d = 0; d < 3; ++d) {
                auto r = grid.get_local_range(d);
                out[d] = {int(r[0]), int(r[1])};
            }
            return out;
        }
        static size_t cells(const Grid & grid) {
            size_t n = 0;
            for (auto i : grid.get_real_range()) {
                (void)i;
                ++n;
            }
            return n;
        }
        void validate(const Field & f) const {
            bool ok = f.size() == cells_;
            for (double v : f)
                ok &= std::isfinite(v);
            require(ok, "[EulerianBias] Invalid local field size or nonfinite value");
        }
        Field allocate() const {
            Field f;
            FML::COMMUNICATION::collective_try([&] { f.resize(cells_); }, "[EulerianBias] Allocate local field");
            return f;
        }
        double mean(const Field & f) const {
            double s = 0;
            for (double v : f)
                s += v;
            FML::SumOverTasks(&s);
            return s / (double(mesh_) * mesh_ * mesh_);
        }
        Field centered(const Field & f) const {
            auto out = allocate();
            std::copy(f.begin(), f.end(), out.begin());
            double m = mean(f);
            for (double & v : out)
                v -= m;
            return out;
        }

      public:
        struct Workspace {
            friend class QuadraticBias;

          private:
            const QuadraticBias * owner;
            Grid scratch;
            Columns columns;
            std::array<Field, 6> tidal;
            bool valid = false;
            Workspace(const QuadraticBias * o, const Grid & prototype)
                : owner(o), scratch(make_grid_like(prototype, o->mesh_, 0, 0)) {}

          public:
            Workspace(Workspace && other) noexcept
                : owner(std::exchange(other.owner, nullptr)), scratch(std::move(other.scratch)),
                  columns(std::move(other.columns)), tidal(std::move(other.tidal)),
                  valid(std::exchange(other.valid, false)) {}
            Workspace(const Workspace &) = delete;
        };

      private:
        void check(const Workspace & w) const { require(w.owner == this, "[EulerianBias] Foreign workspace"); }
        // Real, even multipliers and the sharp band are self-adjoint under the global cell dot product.
        template <class Kernel>
        Field transform(const Field & f, Workspace & w, Kernel kernel) const {
            size_t q = 0;
            for (auto i : w.scratch.get_real_range())
                w.scratch.set_real_from_index(i, f[q++]);
            w.scratch.set_grid_status_real(true);
            w.scratch.fftw_r2c();
            for (auto i : w.scratch.get_fourier_range()) {
                auto n = w.scratch.get_fourier_integer_wavevector_from_index(i);
                const double mult = band_.contains(n, boxsize_, mesh_) ? kernel(n) : 0.;
                w.scratch.set_fourier_from_index(i, w.scratch.get_fourier_from_index(i) * mult);
            }
            w.scratch.fftw_c2r();
            Field out = allocate();
            q = 0;
            for (auto i : w.scratch.get_real_range())
                out[q++] = w.scratch.get_real_from_index(i);
            return out;
        }
        double tidal_kernel(const std::array<int, 3> & n, int a) const {
            double n2 = 0;
            for (int v : n)
                n2 += double(v) * v;
            return double(n[pairs_[a][0]]) * n[pairs_[a][1]] / n2 - (a < 3 ? 1. / 3 : 0.);
        }
        double laplacian_kernel(const std::array<int, 3> & n) const {
            double n2 = 0;
            for (int v : n)
                n2 += double(v) * v;
            return -std::pow(2 * M_PI / boxsize_, 2) * n2;
        }

      public:
        QuadraticBias(const Grid & prototype, double boxsize, SharpKBand matter_band)
            : boxsize_(boxsize), band_(matter_band), mesh_(prototype.get_nmesh()), cells_(cells(prototype)),
              ranges_(ranges(prototype)) {
            static_assert(FML::GRID::has_fourier_grid_interface<Grid>::value, "Fourier grid required");
            require(prototype.get_ndim() == 3 && mesh_ >= 4 && std::isfinite(boxsize_) && boxsize_ > 0 &&
                        std::isfinite(band_.cutoff) && band_.cutoff > 0 &&
                        (band_.geometry == BandGeometry::Sphere || band_.geometry == BandGeometry::Cube),
                    "[EulerianBias] Invalid mesh, boxsize or matter band");
            // Conservative padding: all quadratic product modes fit strictly below Nyquist.
            require(band_.cutoff <= M_PI * mesh_ / (2 * boxsize_),
                    "[EulerianBias] Matter cutoff requires a larger product mesh");
#ifdef USE_MPI
            double v[4] = {boxsize_, band_.cutoff, double(band_.geometry == BandGeometry::Cube), double(mesh_)};
            double lo[4], hi[4];
            MPI_Allreduce(v, lo, 4, MPI_DOUBLE, MPI_MIN, FML::communicator);
            MPI_Allreduce(v, hi, 4, MPI_DOUBLE, MPI_MAX, FML::communicator);
            require(std::equal(lo, lo + 4, hi), "[EulerianBias] Configuration differs across ranks");
#endif
        }
        // The prototype must have the same real-space domain/layout as the constructor's grid.
        Workspace workspace(const Grid & prototype) const {
            require(prototype.get_nmesh() == mesh_ && ranges(prototype) == ranges_,
                    "[EulerianBias] Workspace prototype mismatch");
            return Workspace(this, prototype);
        }
        const Columns & columns(const Workspace & w) const {
            check(w);
            require(w.valid, "[EulerianBias] Columns require a successful evaluate");
            return w.columns;
        }
        // No factor 1/2 in delta^2; K^2 sums all nine tensor entries.
        // Laplacian uses physical length units, so its bias coefficient has units length^2.
        const Columns & evaluate(const Field & matter, Workspace & w) const {
            check(w);
            w.valid = false;
            validate(matter);
            auto & c = w.columns;
            c[Density] = transform(matter, w, [](auto) { return 1.; });
            c[Laplacian] = transform(matter, w, [&](auto n) { return laplacian_kernel(n); });
            for (int a = 0; a < 6; ++a)
                w.tidal[a] = transform(matter, w, [&](auto n) { return tidal_kernel(n, a); });
            FML::COMMUNICATION::collective_try(
                [&] {
                    c[DensitySquared].resize(cells_);
                    c[TidalSquared].assign(cells_, 0.);
                },
                "[EulerianBias] Allocate quadratic columns");
            for (size_t q = 0; q < cells_; ++q) {
                c[DensitySquared][q] = c[Density][q] * c[Density][q];
                for (int a = 0; a < 6; ++a)
                    c[TidalSquared][q] += (a < 3 ? 1 : 2) * w.tidal[a][q] * w.tidal[a][q];
            }
            c[DensitySquared] = centered(c[DensitySquared]);
            c[TidalSquared] = centered(c[TidalSquared]);
            for (const auto & column : c)
                validate(column);
            w.valid = true;
            return c;
        }
        // Transpose Jacobian of all four columns at the latest successful evaluate().
        // Cotangents are local real-cell derivatives; no Fourier/volume factors are implicit.
        Field pullback(const Columns & cotangent, Workspace & w) const {
            check(w);
            require(w.valid, "[EulerianBias] Pullback requires a successful evaluate");
            for (const auto & c : cotangent)
                validate(c);
            const auto g2 = centered(cotangent[DensitySquared]);
            const auto gk = centered(cotangent[TidalSquared]);
            Field source = allocate();
            for (size_t q = 0; q < cells_; ++q)
                source[q] = cotangent[Density][q] + 2 * w.columns[Density][q] * g2[q];
            Field result = transform(source, w, [](auto) { return 1.; });
            auto lap = transform(cotangent[Laplacian], w, [&](auto n) { return laplacian_kernel(n); });
            for (size_t q = 0; q < cells_; ++q)
                result[q] += lap[q];
            for (int a = 0; a < 6; ++a) {
                for (size_t q = 0; q < cells_; ++q)
                    source[q] = 2 * (a < 3 ? 1 : 2) * w.tidal[a][q] * gk[q];
                auto term = transform(source, w, [&](auto n) { return tidal_kernel(n, a); });
                for (size_t q = 0; q < cells_; ++q)
                    result[q] += term[q];
            }
            validate(result);
            return result;
        }
    };
} // namespace FML::FIELDLEVEL::EULERIAN
#endif
