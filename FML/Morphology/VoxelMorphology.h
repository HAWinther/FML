#ifndef FML_VOXEL_MORPHOLOGY_HEADER
#define FML_VOXEL_MORPHOLOGY_HEADER
#include <FML/FFTWGrid/GhostedCells.h>
#include <FML/Global/Global.h>
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
// Cubical-complex / integral-geometric morphology: Schmalzing & Buchert (1997),
// arXiv:astro-ph/9702130; Schmalzing et al. (1999), arXiv:astro-ph/9904384.
// FML boundary, normalization and threshold conventions: README.md.
namespace FML::MORPHOLOGY {
    using Count = unsigned long long;
    enum class Boundary { Periodic, Open };
    enum class ThresholdRule { AtLeast, Greater };
    struct BinarySlab {
        int nmesh = 0;
        size_t x_start = 0, local_nx = 0;
        std::vector<std::uint8_t> cells; // No padding. z fastest; zero=empty, nonzero=closed occupied voxel.
    };
    /// The local cells of a decomposition split in x and y (e.g. pencils): [x_start, x_start + local_nx) x [y_start,
    /// y_start + local_ny) x [0, nmesh). The boxes of the tasks must tile the complete box
    struct BinaryBox {
        int nmesh = 0;
        size_t x_start = 0, local_nx = 0, y_start = 0, local_ny = 0;
        std::vector<std::uint8_t> cells; // No padding. z fastest; zero=empty, nonzero=closed occupied voxel.
    };
    struct CubicalCounts {
        Count vertices = 0, edges = 0, faces = 0, cubes = 0;
    };
    struct Measures {
        CubicalCounts counts;
        Boundary boundary = Boundary::Periodic;
        int nmesh = 0;
        double boxsize = 0, volume = 0, surface_area = 0, integrated_mean_curvature = 0;
        std::int64_t euler_characteristic = 0;
        std::array<double, 4> minkowski_density{}; // V/Vbox, A/(6 Vbox), H/(3pi Vbox), chi/Vbox.
        std::array<double, 4> crofton_density{};   // Isotropic lattice Crofton convention; see README.
    };
    struct Excursion {
        double threshold = 0;
        ThresholdRule rule = ThresholdRule::AtLeast;
        Measures measures;
    };
    namespace detail {
        inline void require(bool ok, const char * message) {
            int valid = ok;
            FML::MinOverTasks(&valid);
            if (!valid)
                throw std::invalid_argument(message);
        }
        inline void same(Count h) {
            Count lo = h, hi = h;
            FML::MinOverTasks(&lo);
            FML::MaxOverTasks(&hi);
            require(lo == hi, "Morphology schemas differ across ranks");
        }
        inline void hash(Count & h, double x) {
            std::uint64_t bits;
            std::memcpy(&bits, &x, sizeof(bits));
            h ^= bits;
            h *= 1099511628211ULL;
        }
        inline int threads() {
#ifdef USE_OMP
            return omp_get_max_threads();
#else
            return 1;
#endif
        }
        inline int thread() {
#ifdef USE_OMP
            return omp_get_thread_num();
#else
            return 0;
#endif
        }
        template <class Grid, class = void>
        struct HasStatus : std::false_type {};
        template <class Grid>
        struct HasStatus<Grid, std::void_t<decltype(std::declval<Grid &>().get_grid_status_real())>> : std::true_type {
        };
        struct alignas(64) SliceCounts {
            Count v = 0, e = 0, f = 0, c = 0;
        };
    } // namespace detail
    namespace detail {
        inline Measures measures_from_counts(const std::array<Count, 4> & counts, int n, double boxsize,
                                             Boundary boundary) {
            Measures result;
            result.nmesh = n;
            result.boxsize = boxsize;
            result.boundary = boundary;
            result.counts = {counts[0], counts[1], counts[2], counts[3]};
            const auto v = std::int64_t(counts[0]), e = std::int64_t(counts[1]), f = std::int64_t(counts[2]),
                       c = std::int64_t(counts[3]);
            const double h = boxsize / n, volume = boxsize * boxsize * boxsize;
            result.volume = double(c) * h * h * h;
            result.surface_area = double(2 * f - 6 * c) * h * h;
            result.integrated_mean_curvature = M_PI * double(3 * c - 2 * f + e) * h;
            result.euler_characteristic = v - e + f - c;
            result.minkowski_density = {result.volume / volume,
                                        result.surface_area / (6 * volume),
                                        result.integrated_mean_curvature / (3 * M_PI * volume),
                                        double(result.euler_characteristic) / volume};
            result.crofton_density = {result.volume / volume,
                                      (2.0 / 9) * double(f - 3 * c) * h * h / volume,
                                      (2.0 / 9) * double(e - 2 * f + 3 * c) * h / volume,
                                      double(result.euler_characteristic) / volume};
            return result;
        }

    } // namespace detail
    /// Collective on FML::communicator. Rank-ordered slabs must tile the complete box;
    /// empty slabs are allowed. No global voxel replication is performed.
    inline Measures measure(const BinarySlab & slab, double boxsize, Boundary boundary = Boundary::Periodic) {
        const int n = slab.nmesh;
        bool valid = n > 0 && boxsize > 0 && std::isfinite(boxsize * boxsize * boxsize) &&
                     boxsize * boxsize * boxsize > 0 && (boundary == Boundary::Periodic || boundary == Boundary::Open);
        if (n > 0) {
            const Count width = Count(n) + 1;
            valid &= width <= Count(std::numeric_limits<std::int64_t>::max()) / 8 / width / width;
            valid &= slab.x_start <= size_t(n) && slab.local_nx <= size_t(n) - slab.x_start;
            valid &= size_t(n) <= std::numeric_limits<size_t>::max() / size_t(n);
            if (valid) {
                const size_t plane = size_t(n) * n;
                valid &= slab.local_nx <= std::numeric_limits<size_t>::max() / plane;
                if (valid)
                    valid &= slab.cells.size() == slab.local_nx * plane;
            }
        }
        detail::require(valid, "Invalid morphology geometry, count range or local binary slab size");
        Count schema = Count(n) * 4 + Count(boundary == Boundary::Open);
        detail::hash(schema, boxsize);
        detail::same(schema);
        std::array<Count, 2> layout{Count(slab.x_start), Count(slab.local_nx)};
        const auto layouts = FML::GatherFromTasks(&layout);
        size_t prefix = 0;
        int first = -1, last = -1;
        valid = true;
        for (int t = 0; t < FML::NTasks; ++t) {
            valid &= layouts[t][0] == prefix && layouts[t][1] <= size_t(n) - std::min(prefix, size_t(n));
            prefix += size_t(layouts[t][1]);
            if (layouts[t][1]) {
                if (first < 0)
                    first = t;
                last = t;
            }
        }
        detail::require(valid && prefix == size_t(n) && first >= 0,
                        "Morphology slabs must form a rank-ordered partition");
        const size_t plane = size_t(n) * n;
        std::vector<std::uint8_t> left(slab.local_nx ? plane : 0);
        if (slab.local_nx) {
            int source = FML::ThisTask, destination = FML::ThisTask;
            do {
                source = (source - 1 + FML::NTasks) % FML::NTasks;
            } while (!layouts[source][1]);
            do {
                destination = (destination + 1) % FML::NTasks;
            } while (!layouts[destination][1]);
            const auto * send = slab.cells.data() + (slab.local_nx - 1) * plane;
            if (source == FML::ThisTask)
                std::copy(send, send + plane, left.begin());
#ifdef USE_MPI
            else
                for (size_t offset = 0; offset < plane;) {
                    const int amount = int(std::min(plane - offset, size_t(8 * 1024 * 1024)));
                    MPI_Sendrecv(send + offset,
                                 amount,
                                 MPI_BYTE,
                                 destination,
                                 13736,
                                 left.data() + offset,
                                 amount,
                                 MPI_BYTE,
                                 source,
                                 13736,
                                 FML::communicator,
                                 MPI_STATUS_IGNORE);
                    offset += size_t(amount);
                }
#endif
        }
        const bool open = boundary == Boundary::Open;
        const size_t anchors = slab.local_nx + (open && FML::ThisTask == last ? 1 : 0);
        const int side = n + int(open);
        std::vector<detail::SliceCounts> counters(static_cast<size_t>(detail::threads()));
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t ix = 0; ix < anchors; ++ix) {
            auto get = [&](int localx, int y, int z) {
                const long long globalx = static_cast<long long>(slab.x_start) + localx;
                if (open && (globalx < 0 || globalx >= n || y < 0 || y >= n || z < 0 || z >= n))
                    return false;
                if (!open) {
                    y = (y + n) % n;
                    z = (z + n) % n;
                }
                const size_t yz = size_t(y) * n + z;
                if (localx < 0)
                    return left[yz] != 0;
                if (size_t(localx) >= slab.local_nx)
                    return false;
                return slab.cells[size_t(localx) * plane + yz] != 0;
            };
            auto & count = counters[detail::thread()];
            for (int y = 0; y < side; ++y)
                for (int z = 0; z < side; ++z) {
                    unsigned pattern = 0;
                    for (int xbit = 0; xbit < 2; ++xbit)
                        for (int ybit = 0; ybit < 2; ++ybit)
                            for (int zbit = 0; zbit < 2; ++zbit)
                                if (get(int(ix) + xbit - 1, y + ybit - 1, z + zbit - 1))
                                    pattern |= 1u << (4 * xbit + 2 * ybit + zbit);
                    count.v += pattern != 0;
                    count.e += bool(pattern & 0xF0) + bool(pattern & 0xCC) + bool(pattern & 0xAA);
                    count.f += bool(pattern & 0x88) + bool(pattern & 0xA0) + bool(pattern & 0xC0);
                    count.c += bool(pattern & 0x80);
                }
        }
        std::array<Count, 4> counts{};
        for (const auto & c : counters) {
            counts[0] += c.v;
            counts[1] += c.e;
            counts[2] += c.f;
            counts[3] += c.c;
        }
        FML::SumArrayOverTasks(counts.data(), counts.size());
        return detail::measures_from_counts(counts, n, boxsize, boundary);
    }
    /// Collective on FML::communicator. The boxes of the tasks (split in x and y, e.g. pencils) must tile the
    /// complete box; empty boxes are allowed. The same counts as measure(BinarySlab) of the same field
    inline Measures measure(const BinaryBox & box, double boxsize, Boundary boundary = Boundary::Periodic) {
        const int n = box.nmesh;
        bool valid = n > 0 && boxsize > 0 && std::isfinite(boxsize * boxsize * boxsize) &&
                     boxsize * boxsize * boxsize > 0 && (boundary == Boundary::Periodic || boundary == Boundary::Open);
        if (n > 0) {
            const Count width = Count(n) + 1;
            valid &= width <= Count(std::numeric_limits<std::int64_t>::max()) / 8 / width / width;
            valid &= box.x_start <= size_t(n) && box.local_nx <= size_t(n) - box.x_start;
            valid &= box.y_start <= size_t(n) && box.local_ny <= size_t(n) - box.y_start;
            valid &= box.cells.size() == box.local_nx * box.local_ny * size_t(n);
        }
        detail::require(valid, "Invalid morphology geometry, count range or local binary box size");
        Count schema = Count(n) * 4 + Count(boundary == Boundary::Open);
        detail::hash(schema, boxsize);
        detail::same(schema);
        Count volume = Count(box.local_nx) * box.local_ny * Count(n);
        FML::SumOverTasks(&volume);
        detail::require(volume == Count(n) * n * n, "Morphology boxes must tile the box");

        // Our cells with one ghost layer below along the split axes (the vertex at the lower corner of a cell needs
        // the 8 cells around it)
        const bool empty = box.local_nx == 0 || box.local_ny == 0;
        const long long nx = empty ? 0 : (long long)(box.local_nx), ny = empty ? 0 : (long long)(box.local_ny);
        FML::GRID::GhostedCells<std::uint8_t, 3> cells(
            n, {{{(long long)(box.x_start), nx}, {(long long)(box.y_start), ny}, {0, empty ? 0 : n}}}, 1, 0, 0);
        for (long long ix = 0; ix < nx; ++ix)
            for (long long iy = 0; iy < ny; ++iy)
                for (int iz = 0; iz < n; ++iz)
                    cells.at({int(ix), int(iy), iz}) = box.cells[(size_t(ix) * ny + iy) * n + iz] != 0;
        cells.communicate();

        const bool open = boundary == Boundary::Open;
        // The vertices on the upper faces of the box only exist for open boundaries (the task at the upper end)
        const long long xanchors = nx + (open && !empty && box.x_start + box.local_nx == size_t(n) ? 1 : 0);
        const long long yanchors = ny + (open && !empty && box.y_start + box.local_ny == size_t(n) ? 1 : 0);
        const int side = n + int(open);
        std::vector<detail::SliceCounts> counters(static_cast<size_t>(detail::threads()));
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
        for (long long ix = 0; ix < xanchors; ++ix) {
            auto get = [&](int localx, int localy, int z) {
                const long long globalx = (long long)(box.x_start) + localx;
                const long long globaly = (long long)(box.y_start) + localy;
                if (open && (globalx < 0 || globalx >= n || globaly < 0 || globaly >= n || z < 0 || z >= n))
                    return false;
                // (open: the cell above the last one is outside the box, done above)
                return cells.at({localx, localy, z}) != 0;
            };
            auto & count = counters[detail::thread()];
            for (long long y = 0; y < yanchors; ++y)
                for (int z = 0; z < side; ++z) {
                    unsigned pattern = 0;
                    for (int xbit = 0; xbit < 2; ++xbit)
                        for (int ybit = 0; ybit < 2; ++ybit)
                            for (int zbit = 0; zbit < 2; ++zbit)
                                if (get(int(ix) + xbit - 1, int(y) + ybit - 1, z + zbit - 1))
                                    pattern |= 1u << (4 * xbit + 2 * ybit + zbit);
                    count.v += pattern != 0;
                    count.e += bool(pattern & 0xF0) + bool(pattern & 0xCC) + bool(pattern & 0xAA);
                    count.f += bool(pattern & 0x88) + bool(pattern & 0xA0) + bool(pattern & 0xC0);
                    count.c += bool(pattern & 0x80);
                }
        }
        std::array<Count, 4> counts{};
        for (const auto & c : counters) {
            counts[0] += c.v;
            counts[1] += c.e;
            counts[2] += c.f;
            counts[3] += c.c;
        }
        FML::SumArrayOverTasks(counts.data(), counts.size());
        return detail::measures_from_counts(counts, n, boxsize, boundary);
    }
    namespace detail {
        template <class Grid, class = void>
        struct HasDomain : std::false_type {};
        template <class Grid>
        struct HasDomain<Grid, std::void_t<decltype(std::declval<const Grid &>().get_domain())>> : std::true_type {};
    } // namespace detail
    /// Grid protocol: get_nmesh/get_local_nx/get_local_x_start/get_real({ix,iy,iz}) (slabs), and for grids with a
    /// domain (get_domain(), e.g. PencilGrid) get_local_range(axis) (x and y split, measured with BinaryBox).
    /// Only physical cells are read; FFT padding and existing ghosts are ignored.
    template <class Grid>
    std::vector<Excursion> excursions(const Grid & field,
                                      const std::vector<double> & thresholds,
                                      double boxsize,
                                      Boundary boundary = Boundary::Periodic,
                                      ThresholdRule rule = ThresholdRule::AtLeast) {
        constexpr bool split_xy = detail::HasDomain<Grid>::value;
        bool valid = !thresholds.empty() && (rule == ThresholdRule::AtLeast || rule == ThresholdRule::Greater);
        for (double t : thresholds)
            valid &= std::isfinite(t);
        if constexpr (detail::HasStatus<Grid>::value)
            valid &= const_cast<Grid &>(field).get_grid_status_real();
        const int n = field.get_nmesh();
        long long nx, start, ny = n, ystart = 0;
        if constexpr (split_xy) {
            nx = (long long)(field.get_local_range(0)[1]);
            start = (long long)(field.get_local_range(0)[0]);
            ny = (long long)(field.get_local_range(1)[1]);
            ystart = (long long)(field.get_local_range(1)[0]);
        } else {
            nx = (long long)(field.get_local_nx());
            start = (long long)(field.get_local_x_start());
        }
        valid &= n > 0 && nx >= 0 && start >= 0 && start <= n && nx <= n - start;
        valid &= ny >= 0 && ystart >= 0 && ystart <= n && ny <= n - ystart;
        detail::require(valid, "Invalid morphology thresholds, field status or slab metadata");
        Count schema = thresholds.size() * 2 + Count(rule == ThresholdRule::Greater);
        for (double t : thresholds)
            detail::hash(schema, t);
        detail::same(schema);
        detail::require(size_t(n) <= std::vector<std::uint8_t>().max_size() / size_t(n) &&
                            size_t(nx) * size_t(ny) <= std::vector<std::uint8_t>().max_size() / size_t(n),
                        "Morphology field allocation exceeds addressable storage");
        std::vector<std::uint8_t> cells(size_t(nx) * size_t(ny) * size_t(n));
        bool finite = true;
        for (int ix = 0; ix < nx; ++ix)
            for (int y = 0; y < ny; ++y)
                for (int z = 0; z < n; ++z)
                    finite &= std::isfinite(double(field.get_real({ix, y, z})));
        detail::require(finite, "Morphology excursion fields must contain finite physical cells");
        BinarySlab slab;
        BinaryBox box;
        if constexpr (split_xy) {
            box.nmesh = n;
            box.local_nx = size_t(nx);
            box.x_start = size_t(start);
            box.local_ny = size_t(ny);
            box.y_start = size_t(ystart);
        } else {
            slab.nmesh = n;
            slab.local_nx = size_t(nx);
            slab.x_start = size_t(start);
        }
        std::vector<Excursion> out;
        for (double threshold : thresholds) {
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
            for (int ix = 0; ix < nx; ++ix)
                for (int y = 0; y < ny; ++y)
                    for (int z = 0; z < n; ++z) {
                        const double value = field.get_real({ix, y, z});
                        cells[(size_t(ix) * ny + y) * n + z] =
                            rule == ThresholdRule::AtLeast ? value >= threshold : value > threshold;
                    }
            if constexpr (split_xy) {
                box.cells = cells;
                out.push_back({threshold, rule, measure(box, boxsize, boundary)});
            } else {
                slab.cells = cells;
                out.push_back({threshold, rule, measure(slab, boxsize, boundary)});
            }
        }
        return out;
    }
} // namespace FML::MORPHOLOGY
#endif
