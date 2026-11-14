#ifndef FML_GLOBAL_DOMAIN_H
#define FML_GLOBAL_DOMAIN_H

// =====================================================================================================================
// The decomposition of the periodic unit box [0,1)^NDIM over the MPI tasks: which task owns a point (of particles,
// and for Pencil and PeanoHilbert also of the cells of a grid). Three classes and a wrapper:
//
//   SlabDomain          task i owns [x_min[i], x_max[i]) (the ranges partition [0,1) in rank order; empty slabs
//                       allowed). The default, and for now the only type that the rest of FML supports (grids,
//                       density assignment, interpolation, FoF, ... assume x-slabs given by FML::xmin_domain,
//                       FML::xmax_domain)
//   PencilDomain        a P0 x P1 process grid in (x, y): the task with process coordinates (ix, iy) is rank
//                       ix * P1 + iy and owns [x_edges[ix], x_edges[ix+1]) x [y_edges[iy], y_edges[iy+1]) x [0,1).
//                       Needs NDIM >= 2. The decomposition of a PencilGrid (PencilDomain/PencilGrid.h), whose edges
//                       lie on its cells
//   PeanoHilbertDomain  contiguous ranges of the Peano-Hilbert key (HilbertCurve<3> of PeanoHilbertDomain/
//                       HilbertCurve.h, `bits` bits per coordinate): task i owns the keys [boundaries[i],
//                       boundaries[i+1]). Built load-balanced from particle keys (build), or from boundaries. Needs
//                       NDIM = 3. The layout of DistributedGrid3D (PeanoHilbertDomain/DistributedGrid3D.h)
//   Domain              one of the three (a std::variant): what MPIParticles and FML::domain hold. owner(pos),
//                       owner_near(task, pos) and contains(task, pos) are inline (a switch on the type, no virtual
//                       calls); they read the first ncoordinates() coordinates of pos (1, 2 or 3), which must be in
//                       [0,1) (wrap them first), comparing in the position's own type. get<T>() gives the class
//
// Pencil and PeanoHilbert are the preparation for supporting them in more of FML: choosing them (FML::set_domain, or
// an MPIParticles constructed with one) gives a warning, and code that needs slabs asserts it (require_slab).
// The global instance is FML::domain (Global.h), a Slab whose ranges are FML::xmin_domain, FML::xmax_domain of each
// task (set by init_fml, or by set_domain). This header is included at the end of Global.h.
// =====================================================================================================================

#include <FML/Global/Global.h>
#include <FML/PeanoHilbertDomain/HilbertCurve.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace FML {

    class Domain;

    namespace DOMAIN_DETAIL {
        /// The interval [e[i], e[i+1]) that holds v (the first non-empty one), compared in the type of v
        template <class X>
        int edge_index(const std::vector<double> & e, X v) {
            return int(std::upper_bound(e.begin() + 1, e.end(), v, [](const X & a, double b) { return a < b; }) -
                       (e.begin() + 1));
        }
        inline void check_edges(const std::vector<double> & e, const std::string & name) {
            bool ok = e.size() >= 2 and e.front() == 0.0 and e.back() == 1.0;
            for (size_t i = 1; ok and i < e.size(); i++)
                ok = std::isfinite(e[i]) and e[i] >= e[i - 1];
            if (not ok)
                throw std::invalid_argument("[PencilDomain] " + name +
                                            " must increase from 0 to 1 (n + 1 values for n process coordinates)");
        }
    } // namespace DOMAIN_DETAIL

    //==================================================================================================================
    // SlabDomain
    //==================================================================================================================

    class SlabDomain {
      public:
        /// One task: [0,1)
        SlabDomain() : xmin{0.0}, xmax{1.0} {}
        /// Task i owns [x_min[i], x_max[i]). The ranges are not checked here (valid_partition() does)
        static SlabDomain from_ranges(std::vector<double> x_min, std::vector<double> x_max) {
            if (x_min.size() != x_max.size() or x_min.empty())
                throw std::invalid_argument("[SlabDomain] x_min and x_max need one entry per task");
            SlabDomain d;
            d.xmin = std::move(x_min);
            d.xmax = std::move(x_max);
            return d;
        }
        /// Equal slabs [i/ntasks, (i+1)/ntasks) (as init_fml)
        static SlabDomain uniform(int ntasks) {
            if (ntasks < 1)
                throw std::invalid_argument("[SlabDomain::uniform] Need ntasks >= 1");
            std::vector<double> lo(ntasks), hi(ntasks);
            for (int i = 0; i < ntasks; i++) {
                lo[i] = i / double(ntasks);
                hi[i] = (i + 1) / double(ntasks);
            }
            return from_ranges(std::move(lo), std::move(hi));
        }

        int ntasks() const { return int(xmin.size()); }
        static constexpr int ncoordinates() { return 1; }
        const std::vector<double> & x_min_per_task() const { return xmin; }
        const std::vector<double> & x_max_per_task() const { return xmax; }
        double x_min(int task) const { return xmin[task]; }
        double x_max(int task) const { return xmax[task]; }
        /// The ranges partition [0,1] in rank order (finite, x_min[0] = 0, x_max[n-1] = 1, contiguous; empty slabs
        /// allowed), as the binary search of owner() needs
        bool valid_partition() const {
            if (xmin.empty() or xmin.front() != 0.0 or xmax.back() != 1.0)
                return false;
            for (size_t t = 0; t < xmin.size(); t++)
                if (not(std::isfinite(xmin[t]) and std::isfinite(xmax[t]) and xmin[t] <= xmax[t] and
                        (t == 0 or xmin[t] == xmax[t - 1])))
                    return false;
            return true;
        }

        /// The task whose range holds x (the first non-empty one with x < x_max), compared in the type of x.
        /// x must be in [0,1): for x >= 1 or NaN the result is NTasks (not a valid rank)
        template <class X>
        int owner_x(X x) const {
            return int(std::upper_bound(xmax.begin(), xmax.end(), x, [](const X & v, double e) { return v < e; }) -
                       xmax.begin());
        }
        template <class Pos>
        int owner(const Pos * pos) const {
            return owner_x(pos[0]);
        }
        /// The owner, checking the neighbours of `task` first (where most particles that leave a task go)
        template <class Pos>
        int owner_near(int task, const Pos * pos) const {
            const auto x = pos[0];
            const int n = ntasks();
            const int right = (task + 1) % n, left = (task + n - 1) % n;
            if (x >= xmin[right] and x < xmax[right])
                return right;
            if (x >= xmin[left] and x < xmax[left])
                return left;
            return owner_x(x);
        }
        template <class Pos>
        bool contains(int task, const Pos * pos) const {
            return pos[0] >= xmin[task] and pos[0] < xmax[task];
        }
        bool operator==(const SlabDomain & o) const { return xmin == o.xmin and xmax == o.xmax; }

      private:
        std::vector<double> xmin, xmax;
    };

    //==================================================================================================================
    // PencilDomain
    //==================================================================================================================

    class PencilDomain {
      public:
        /// One task
        PencilDomain() = default;
        /// Pencils with the given edges in x (P0 + 1 values from 0 to 1) and y (P1 + 1 values from 0 to 1)
        static PencilDomain from_edges(std::vector<double> x_edges, std::vector<double> y_edges) {
            DOMAIN_DETAIL::check_edges(x_edges, "x_edges");
            DOMAIN_DETAIL::check_edges(y_edges, "y_edges");
            PencilDomain d;
            d.process_grid_ = {int(x_edges.size()) - 1, int(y_edges.size()) - 1};
            d.xedges = std::move(x_edges);
            d.yedges = std::move(y_edges);
            return d;
        }
        /// Equal pencils on a P0 x P1 process grid
        static PencilDomain uniform(int px, int py) {
            if (px < 1 or py < 1)
                throw std::invalid_argument("[PencilDomain::uniform] Need P0, P1 >= 1");
            auto edges = [](int n) {
                std::vector<double> e(n + 1);
                for (int i = 0; i <= n; i++)
                    e[i] = i / double(n);
                e[n] = 1.0;
                return e;
            };
            return from_edges(edges(px), edges(py));
        }

        int ntasks() const { return process_grid_[0] * process_grid_[1]; }
        static constexpr int ncoordinates() { return 2; }
        /// The process grid {P0, P1}
        std::array<int, 2> process_grid() const { return process_grid_; }
        const std::vector<double> & x_edges() const { return xedges; }
        const std::vector<double> & y_edges() const { return yedges; }
        /// The rank of the process coordinates (periodic) and the inverse
        int rank_of(int process_x, int process_y) const {
            return wrap(process_x, process_grid_[0]) * process_grid_[1] + wrap(process_y, process_grid_[1]);
        }
        std::array<int, 2> coords_of(int rank) const {
            if (rank < 0 or rank >= ntasks())
                throw std::out_of_range("[PencilDomain::coords_of] invalid rank");
            return {rank / process_grid_[1], rank % process_grid_[1]};
        }

        /// The task holding pos. pos[0], pos[1] must be in [0,1): otherwise the result is not a valid rank
        template <class Pos>
        int owner(const Pos * pos) const {
            return DOMAIN_DETAIL::edge_index(xedges, pos[0]) * process_grid_[1] +
                   DOMAIN_DETAIL::edge_index(yedges, pos[1]);
        }
        template <class Pos>
        int owner_near(int, const Pos * pos) const {
            return owner(pos);
        }
        template <class Pos>
        bool contains(int task, const Pos * pos) const {
            return owner(pos) == task;
        }
        /// The same pencils
        bool operator==(const PencilDomain & o) const {
            return process_grid_ == o.process_grid_ and xedges == o.xedges and yedges == o.yedges;
        }

        /// The domain of particles that matches these pencils: a SlabDomain in x if P1 = 1 (e.g. a 2D PencilGrid;
        /// supported by all of FML), else this PencilDomain. For MPIParticles, so that particles live on the rank of
        /// their cells
        Domain particle_domain() const;

      private:
        static int wrap(int i, int n) {
            i %= n;
            return i < 0 ? i + n : i;
        }
        std::array<int, 2> process_grid_{1, 1};
        std::vector<double> xedges{0.0, 1.0}, yedges{0.0, 1.0};
    };

    //==================================================================================================================
    // PeanoHilbertDomain
    //==================================================================================================================

    /// Rank intervals in Hilbert-key order: from a collective key histogram (build: load-balanced), or from the
    /// boundaries. A default-constructed one is not built (its methods throw)
    class PeanoHilbertDomain {
      public:
        using Key = DOMAIN_DECOMPOSITION::PEANO_HILBERT::Key;
        using Position = DOMAIN_DECOMPOSITION::PEANO_HILBERT::Position;

        PeanoHilbertDomain() = default;

        /// Build from each task's local particle keys: collective over FML::communicator, with identical bits and
        /// histogram_bits on all tasks. histogram_bits controls the memory/quantile resolution (default 16, at most
        /// 20). It is a prefix histogram over the full Hilbert key. Points with identical keys cannot be split across
        /// tasks.
        static PeanoHilbertDomain
        build(const std::vector<Key> & local_keys, unsigned bits = 21, unsigned histogram_bits = 16);

        /// From the key boundaries (ntasks + 1 increasing keys from 0 to 2^(3 bits)); rank: this task
        static PeanoHilbertDomain
        from_boundaries(std::vector<Key> boundaries, unsigned bits, int rank = FML::ThisTask) {
            namespace PH = DOMAIN_DECOMPOSITION::PEANO_HILBERT;
            PH::detail::validate_bits(bits);
            if (boundaries.size() < 2 or boundaries.front() != 0 or boundaries.back() != (Key(1) << (3 * bits)) or
                not std::is_sorted(boundaries.begin(), boundaries.end()))
                throw std::invalid_argument("[PeanoHilbertDomain] Need ntasks + 1 increasing key boundaries from 0 to "
                                            "2^(3 bits)");
            PeanoHilbertDomain d;
            d.nranks_ = int(boundaries.size()) - 1;
            d.boundaries_ = std::move(boundaries);
            d.bits_ = bits;
            d.rank_ = rank;
            return d;
        }
        /// Equal key ranges (equal volumes)
        static PeanoHilbertDomain uniform(int ntasks, unsigned bits = 21) {
            if (ntasks < 1)
                throw std::invalid_argument("[PeanoHilbertDomain::uniform] Need ntasks >= 1");
            DOMAIN_DECOMPOSITION::PEANO_HILBERT::detail::validate_bits(bits);
            const Key nkeys = Key(1) << (3 * bits);
            std::vector<Key> b(ntasks + 1);
            for (int i = 0; i <= ntasks; i++)
                b[i] = Key((nkeys / Key(ntasks)) * Key(i) + ((nkeys % Key(ntasks)) * Key(i)) / Key(ntasks));
            return from_boundaries(std::move(b), bits);
        }

        int owner_of_key(Key key) const {
            require_built();
            if (key >= key_limit())
                throw std::out_of_range("Peano-Hilbert key is outside this decomposition");
            return int(std::upper_bound(boundaries_.begin(), boundaries_.end(), key) - boundaries_.begin()) - 1;
        }
        int owner_of_position(const Position & position) const {
            return owner_of_key(DOMAIN_DECOMPOSITION::PEANO_HILBERT::key_from_position(position, bits_));
        }
        Key key_begin(int rank) const {
            require_rank(rank);
            return boundaries_[size_t(rank)];
        }
        Key key_end(int rank) const {
            require_rank(rank);
            return boundaries_[size_t(rank) + 1];
        }
        const std::vector<Key> & key_boundaries() const { return boundaries_; }
        unsigned bits() const {
            require_built();
            return bits_;
        }
        unsigned histogram_bits() const {
            require_built();
            return histogram_bits_;
        }
        int rank() const {
            require_built();
            return rank_;
        }
        int nranks() const {
            require_built();
            return nranks_;
        }
        uint64_t total_particles() const {
            require_built();
            return total_particles_;
        }

        // The Domain interface (positions in [0,1))
        int ntasks() const { return nranks_; }
        static constexpr int ncoordinates() { return 3; }
        /// The key of a point (its cell at `bits` bits per coordinate; no checks: for the owner lookups)
        template <class Pos>
        Key key_of(const Pos * pos) const {
            const double ngrid = double(Key(1) << bits_);
            const uint32_t nmax = (uint32_t(1) << bits_) - 1;
            std::array<uint32_t, 3> cell{};
            for (int axis = 0; axis < 3; axis++)
                cell[axis] = std::min(uint32_t(std::max(double(pos[axis]), 0.0) * ngrid), nmax);
            return GRID::HilbertCurve<3>::key_from_coord(cell, int(bits_));
        }
        template <class Pos>
        int owner(const Pos * pos) const {
            const Key k = key_of(pos);
            return int(std::upper_bound(boundaries_.begin(), boundaries_.end(), k) - boundaries_.begin()) - 1;
        }
        template <class Pos>
        int owner_near(int, const Pos * pos) const {
            return owner(pos);
        }
        template <class Pos>
        bool contains(int task, const Pos * pos) const {
            const Key k = key_of(pos);
            return k >= boundaries_[task] and k < boundaries_[task + 1];
        }
        bool operator==(const PeanoHilbertDomain & o) const {
            return bits_ == o.bits_ and boundaries_ == o.boundaries_;
        }

      private:
        Key key_limit() const { return Key(1) << (3 * bits_); }
        void require_built() const {
            if (nranks_ == 0)
                throw std::logic_error("Peano-Hilbert decomposition has not been built");
        }
        void require_rank(int rank) const {
            require_built();
            if (rank < 0 || rank >= nranks_)
                throw std::out_of_range("Peano-Hilbert rank is outside the decomposition");
        }

        unsigned bits_{0};
        unsigned histogram_bits_{0};
        int rank_{0};
        int nranks_{0};
        uint64_t total_particles_{0};
        std::vector<Key> boundaries_;
    };

    //==================================================================================================================
    // Domain: one of the three
    //==================================================================================================================

    enum class DomainType { Slab, Pencil, PeanoHilbert };

    class Domain {
      public:
        using Key = PeanoHilbertDomain::Key;

        /// A slab domain of one task ([0,1))
        Domain() = default;
        Domain(SlabDomain d) : d(std::move(d)) {}
        Domain(PencilDomain d) : d(std::move(d)) {}
        Domain(PeanoHilbertDomain d) : d(std::move(d)) {
            std::get<PeanoHilbertDomain>(this->d).nranks(); // must be built
        }

        //========================================================
        // Construction (as the classes)
        //========================================================

        static Domain slab(std::vector<double> x_min, std::vector<double> x_max) {
            return SlabDomain::from_ranges(std::move(x_min), std::move(x_max));
        }
        static Domain uniform_slab(int ntasks) { return SlabDomain::uniform(ntasks); }
        static Domain pencil(std::vector<double> x_edges, std::vector<double> y_edges) {
            return PencilDomain::from_edges(std::move(x_edges), std::move(y_edges));
        }
        static Domain uniform_pencil(int px, int py) { return PencilDomain::uniform(px, py); }
        static Domain peano_hilbert(std::vector<Key> boundaries, unsigned bits) {
            return PeanoHilbertDomain::from_boundaries(std::move(boundaries), bits);
        }
        static Domain peano_hilbert(const PeanoHilbertDomain & dec) { return dec; }
        static Domain uniform_peano_hilbert(int ntasks, unsigned bits = 21) {
            return PeanoHilbertDomain::uniform(ntasks, bits);
        }

        //========================================================
        // The class
        //========================================================

        template <class T>
        bool holds() const {
            return std::holds_alternative<T>(d);
        }
        /// The class (throws std::bad_variant_access if it is another)
        template <class T>
        const T & get() const {
            return std::get<T>(d);
        }
        DomainType type() const { return DomainType(d.index()); }
        bool is_slab() const { return d.index() == 0; }
        std::string type_name() const { return name_of(type()); }
        static std::string name_of(DomainType t) {
            switch (t) {
                case DomainType::Slab:
                    return "Slab";
                case DomainType::Pencil:
                    return "Pencil";
                default:
                    return "PeanoHilbert";
            }
        }

        //========================================================
        // Information
        //========================================================

        int ntasks() const {
            return std::visit([](const auto & x) { return x.ntasks(); }, d);
        }
        /// The number of coordinates of the position the domain uses: 1 (Slab: x), 2 (Pencil: x, y), 3 (PeanoHilbert)
        int ncoordinates() const { return int(d.index()) + 1; }
        /// The number of dimensions the type needs at least (Pencil: 2) or exactly (PeanoHilbert: 3)
        void check_ndim(int ndim) const {
            if (holds<PencilDomain>() and ndim < 2)
                throw std::invalid_argument("[Domain] Pencil needs NDIM >= 2");
            if (holds<PeanoHilbertDomain>() and ndim != 3)
                throw std::invalid_argument("[Domain] PeanoHilbert needs NDIM = 3");
        }
        /// Throw if the domain is not a slab (for code that only supports slabs)
        void require_slab(const std::string & where) const {
            if (not is_slab())
                throw std::runtime_error(where + ": only implemented for the Slab domain (this is " + type_name() +
                                         ")");
        }
        /// Slab: the ranges (as SlabDomain); valid_slab_partition is false for the other types
        const std::vector<double> & x_min_per_task() const { return get<SlabDomain>().x_min_per_task(); }
        const std::vector<double> & x_max_per_task() const { return get<SlabDomain>().x_max_per_task(); }
        double x_min(int task) const { return get<SlabDomain>().x_min(task); }
        double x_max(int task) const { return get<SlabDomain>().x_max(task); }
        bool valid_slab_partition() const { return is_slab() and get<SlabDomain>().valid_partition(); }
        /// Pencil: the process grid and the edges
        int pencil_px() const { return get<PencilDomain>().process_grid()[0]; }
        int pencil_py() const { return get<PencilDomain>().process_grid()[1]; }
        const std::vector<double> & x_edges() const { return get<PencilDomain>().x_edges(); }
        const std::vector<double> & y_edges() const { return get<PencilDomain>().y_edges(); }
        /// PeanoHilbert: the key boundaries, bits per coordinate and the key of a point
        const std::vector<Key> & key_boundaries() const { return get<PeanoHilbertDomain>().key_boundaries(); }
        unsigned bits() const { return get<PeanoHilbertDomain>().bits(); }
        template <class Pos>
        Key key_of(const Pos * pos) const {
            return get<PeanoHilbertDomain>().key_of(pos);
        }

        //========================================================
        // Ownership (pos in [0,1) in the coordinates used)
        //========================================================

        /// The task that owns the point
        template <class Pos>
        int owner(const Pos * pos) const {
            switch (d.index()) {
                case 0:
                    return std::get<0>(d).owner(pos);
                case 1:
                    return std::get<1>(d).owner(pos);
                default:
                    return std::get<2>(d).owner(pos);
            }
        }
        /// The task that owns the point, for a Slab checking the neighbours of `task` first (where most particles
        /// that leave a task go: no binary search)
        template <class Pos>
        int owner_near(int task, const Pos * pos) const {
            switch (d.index()) {
                case 0:
                    return std::get<0>(d).owner_near(task, pos);
                case 1:
                    return std::get<1>(d).owner(pos);
                default:
                    return std::get<2>(d).owner(pos);
            }
        }
        /// Whether the task owns the point
        template <class Pos>
        bool contains(int task, const Pos * pos) const {
            switch (d.index()) {
                case 0:
                    return std::get<0>(d).contains(task, pos);
                case 1:
                    return std::get<1>(d).contains(task, pos);
                default:
                    return std::get<2>(d).contains(task, pos);
            }
        }
        /// Slab: the task whose range holds x
        template <class X>
        int slab_owner(X x) const {
            return get<SlabDomain>().owner_x(x);
        }

        bool operator==(const Domain & o) const { return d == o.d; }
        bool operator!=(const Domain & o) const { return not(*this == o); }

        //========================================================
        // Binary I/O (for files of particles): the type, then for a Slab the number of tasks and the ranges, for a
        // Pencil P0, P1 and the edges, for PeanoHilbert the bits, the number of tasks and the key boundaries
        //========================================================

        void write(std::ostream & os) const {
            const int32_t t = int32_t(d.index());
            os.write(reinterpret_cast<const char *>(&t), sizeof(t));
            auto write_vector = [&](const auto & v) {
                if (not v.empty())
                    os.write(reinterpret_cast<const char *>(v.data()), std::streamsize(sizeof(v[0]) * v.size()));
            };
            if (holds<SlabDomain>()) {
                const int32_t n = ntasks();
                os.write(reinterpret_cast<const char *>(&n), sizeof(n));
                write_vector(x_min_per_task());
                write_vector(x_max_per_task());
            } else if (holds<PencilDomain>()) {
                const int32_t g[2] = {pencil_px(), pencil_py()};
                os.write(reinterpret_cast<const char *>(g), sizeof(g));
                write_vector(x_edges());
                write_vector(y_edges());
            } else {
                const uint32_t b = bits();
                const int32_t n = ntasks();
                os.write(reinterpret_cast<const char *>(&b), sizeof(b));
                os.write(reinterpret_cast<const char *>(&n), sizeof(n));
                write_vector(key_boundaries());
            }
        }
        /// (throws on a bad stream or invalid content)
        static Domain read(std::istream & is) {
            auto read_value = [&](auto & v) { is.read(reinterpret_cast<char *>(&v), sizeof(v)); };
            auto read_vector = [&](auto & v, int32_t n) {
                if (n < 0 or n > (1 << 24))
                    throw std::runtime_error("[Domain::read] Invalid size");
                v.resize(size_t(n));
                if (n > 0)
                    is.read(reinterpret_cast<char *>(v.data()), std::streamsize(sizeof(v[0]) * v.size()));
            };
            int32_t t = -1;
            read_value(t);
            Domain dom;
            if (t == 0) {
                int32_t n = 0;
                read_value(n);
                std::vector<double> lo, hi;
                read_vector(lo, n);
                read_vector(hi, n);
                if (is)
                    dom = SlabDomain::from_ranges(std::move(lo), std::move(hi));
            } else if (t == 1) {
                int32_t g[2] = {0, 0};
                is.read(reinterpret_cast<char *>(g), sizeof(g));
                std::vector<double> xe, ye;
                read_vector(xe, g[0] + 1);
                read_vector(ye, g[1] + 1);
                if (is)
                    dom = PencilDomain::from_edges(std::move(xe), std::move(ye));
            } else if (t == 2) {
                uint32_t b = 0;
                int32_t n = 0;
                read_value(b);
                read_value(n);
                std::vector<Key> keys;
                read_vector(keys, n + 1);
                if (is)
                    dom = PeanoHilbertDomain::from_boundaries(std::move(keys), b);
            } else {
                throw std::runtime_error("[Domain::read] Unknown domain type");
            }
            if (not is)
                throw std::runtime_error("[Domain::read] Failed to read the domain");
            return dom;
        }

      private:
        std::variant<SlabDomain, PencilDomain, PeanoHilbertDomain> d{};
    };

    inline Domain PencilDomain::particle_domain() const {
        if (process_grid_[1] == 1)
            return SlabDomain::from_ranges(std::vector<double>(xedges.begin(), xedges.end() - 1),
                                           std::vector<double>(xedges.begin() + 1, xedges.end()));
        return *this;
    }

    inline PeanoHilbertDomain
    PeanoHilbertDomain::build(const std::vector<Key> & local_keys, unsigned bits, unsigned histogram_bits) {
        namespace PHD = DOMAIN_DECOMPOSITION::PEANO_HILBERT::detail;
        const int nranks = FML::NTasks;
        const int rank = FML::ThisTask;
#ifdef USE_MPI
        MPI_Comm comm = FML::communicator;
#endif

#ifdef USE_MPI
        int local_config[2] = {int(bits), int(histogram_bits)};
        int min_config[2] = {}, max_config[2] = {};
        MPI_Allreduce(local_config, min_config, 2, MPI_INT, MPI_MIN, comm);
        MPI_Allreduce(local_config, max_config, 2, MPI_INT, MPI_MAX, comm);
        if (min_config[0] != max_config[0] || min_config[1] != max_config[1])
            throw std::invalid_argument("All ranks must use the same Peano-Hilbert resolution");
#endif
        PHD::validate_bits(bits);
        const unsigned key_bits = 3 * bits;
        if (histogram_bits == 0 || histogram_bits > 20 || histogram_bits > key_bits)
            throw std::invalid_argument("Peano-Hilbert histogram_bits must be in [1, min(20, 3*bits)]");

        const Key key_limit = Key(1) << key_bits;
        bool local_valid = true;
        for (const Key key : local_keys)
            local_valid = local_valid && key < key_limit;
#ifdef USE_MPI
        int valid = local_valid ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, comm);
        if (!valid)
            throw std::out_of_range("A rank supplied a key outside the configured Hilbert resolution");
#else
        if (!local_valid)
            throw std::out_of_range("A key is outside the configured Hilbert resolution");
#endif

        const size_t nbins = size_t(1) << histogram_bits;
        std::vector<uint64_t> histogram(nbins, 0);
        const unsigned shift = key_bits - histogram_bits;
        for (const Key key : local_keys) {
            auto & count = histogram[size_t(key >> shift)];
            if (count == std::numeric_limits<uint64_t>::max())
                throw std::overflow_error("Local Peano-Hilbert histogram overflow");
            ++count;
        }
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, histogram.data(), int(nbins), MPI_UINT64_T, MPI_SUM, comm);
#endif

        uint64_t total = 0;
        for (const auto count : histogram) {
            if (count > std::numeric_limits<uint64_t>::max() - total)
                throw std::overflow_error("Global Peano-Hilbert particle count overflow");
            total += count;
        }

        PeanoHilbertDomain result;
        result.bits_ = bits;
        result.histogram_bits_ = histogram_bits;
        result.rank_ = rank;
        result.nranks_ = nranks;
        result.total_particles_ = total;
        result.boundaries_.reserve(size_t(nranks) + 1);
        result.boundaries_.push_back(0);

        // Place each cut at whichever edge of the quantile bin is closer to its target. This avoids assigning an
        // entire large bin to one rank where the opposite edge would produce a better balance.
        uint64_t cumulative = 0;
        size_t bin = 0;
        for (int r = 1; r < nranks; ++r) {
            const uint64_t target = (total / uint64_t(nranks)) * uint64_t(r) +
                                    ((total % uint64_t(nranks)) * uint64_t(r)) / uint64_t(nranks);
            while (bin < nbins && cumulative + histogram[bin] < target) {
                cumulative += histogram[bin];
                ++bin;
            }
            size_t cut_bin = bin;
            if (bin < nbins) {
                const uint64_t before = cumulative;
                const uint64_t after = cumulative + histogram[bin];
                if (target - before > after - target)
                    cut_bin = bin + 1;
            }
            const Key cut = Key(cut_bin) << shift;
            result.boundaries_.push_back(std::max(result.boundaries_.back(), cut));
        }
        result.boundaries_.push_back(key_limit);
        return result;
    }

    namespace DOMAIN_DECOMPOSITION::PEANO_HILBERT {
        /// (the class was in this namespace)
        using FML::PeanoHilbertDomain;
    } // namespace DOMAIN_DECOMPOSITION::PEANO_HILBERT

} // namespace FML

#endif
