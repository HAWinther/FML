#ifndef ADAPTIVEMPIGRID_HEADER
#define ADAPTIVEMPIGRID_HEADER

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include <FML/PeanoHilbertDomain/HilbertCurve.h>
#include <FML/Global/Global.h>

namespace FML {
    namespace GRID {

        using IndexInt = long long int;

        //==========================================================================================
        /// A leaf cell in the linear tree. Identified by (key, level). The anchor is the integer
        /// coordinate of the lower-left corner in units of 2^-LMAX of the box [0,1)^NDIM.
        //==========================================================================================
        template <int NDIM>
        struct AMRCell {
            uint64_t key{0};                     // Hilbert key of the first max-depth point in the cell
            std::array<uint32_t, NDIM> anchor{}; // Lower-left corner (integer coordinates)
            int8_t level{0};                     // Refinement level (cellsize = 2^-level)
        };

        //==========================================================================================
        /// Reference to a cell: either a local leaf, a ghost leaf or nothing
        //==========================================================================================
        struct AMRCellRef {
            enum Type : int8_t { NONE = 0, LOCAL = 1, GHOST = 2 };
            Type type{NONE};
            IndexInt index{-1};
            bool is_local() const { return type == LOCAL; }
            bool is_ghost() const { return type == GHOST; }
            explicit operator bool() const { return type != NONE; }
        };

        //==========================================================================================
        ///
        /// An adaptively refined grid stored as a linear tree (only leaves are stored), sorted along
        /// a Peano-Hilbert curve and distributed over MPI tasks as contiguous segments of the curve.
        ///
        /// * Domain is the unit box [0,1)^NDIM, periodic or not along each axis.
        /// * 2:1 balance (including edges and corners) is always enforced, so neighbouring leaves
        ///   differ by at most one level.
        /// * One layer of ghost leaves: every remote leaf that touches a local leaf (face, edge or
        ///   corner), at whatever level it has. Neighbours across a periodic boundary owned by the
        ///   task itself are found among the local leaves, so no "self-ghosts" are needed.
        /// * T must be trivially copyable (it is sent as raw bytes).
        ///
        /// Typical use:
        ///   AdaptiveMPIGrid<2,double> grid(levelmin, levelmax, periodic);
        ///   for(i < get_n_local()) grid.get_data(i) = ...;
        ///   grid.set_flag(i, REFINE / COARSEN);
        ///   grid.adapt();          // balance + refine/coarsen + rebuild ghosts
        ///   grid.load_balance();   // repartition the curve + migrate + rebuild ghosts
        ///   grid.exchange_ghosts(); // whenever the data has changed
        ///
        /// Overlapping communication with computation:
        ///   auto h = grid.begin_exchange_ghosts();
        ///   for (auto i : grid.get_interior_cells()) ...  // neighbours are all local
        ///   h.wait();
        ///   for (auto i : grid.get_boundary_cells()) ...  // may use ghosts
        ///
        //==========================================================================================
        template <int NDIM, class T>
        class AdaptiveMPIGrid {
            static_assert(std::is_trivially_copyable<T>::value, "AdaptiveMPIGrid: T must be trivially copyable");

          public:
            using HC = HilbertCurve<NDIM>;
            using key_type = uint64_t;
            using coord_type = uint32_t;
            using Coord = std::array<coord_type, NDIM>;
            using Cell = AMRCell<NDIM>;
            using CellRef = AMRCellRef;

            static constexpr int LMAX = HC::LMAX;
            static constexpr key_type KEY_END = key_type(1) << (NDIM * LMAX);
            static constexpr int NCHILD = 1 << NDIM;

            enum Flag : int8_t { COARSEN = -1, KEEP = 0, REFINE = 1 };

            /// Irregular exchanges (ghost-pattern setup, migration) use the NBX algorithm when running on
            /// at least this many tasks, otherwise an MPI_Alltoall of counts. NBX avoids the O(NTasks)
            /// collective but polls, so on few (or oversubscribed) tasks the collective is faster.
            /// Set to 0 to always use NBX.
            static inline int nbx_min_tasks = 64;

            /// Prolongation: value in a child from the parent value
            using ProlongFunction = std::function<T(const T & parent_value, const Cell & parent, const Cell & child)>;
            /// Restriction: value in the parent from the NCHILD children (in key order)
            using RestrictFunction =
                std::function<T(const std::vector<T> & child_values, const std::vector<Cell> & children, const Cell & parent)>;
            /// Weight of a cell for load balancing
            using WeightFunction = std::function<double(IndexInt index)>;

            /// The face neighbours of a cell in a given direction
            struct FaceNeighbors {
                enum Relation : int8_t { BOUNDARY = -2, COARSER = -1, SAME = 0, FINER = 1 };
                Relation relation{BOUNDARY};
                int n{0};
                std::array<CellRef, (1 << (NDIM - 1))> cells{};
            };

          private:
            int _levelmin{0};
            int _levelmax{0};
            std::array<bool, NDIM> _periodic{};

            // Local leaves (sorted by key) and their data
            std::vector<Cell> _cells;
            std::vector<T> _data;
            std::vector<int8_t> _flags;

            // Ghost leaves (sorted by key) and their data
            std::vector<Cell> _ghost_cells;
            std::vector<T> _ghost_data;

            // Task r owns keys in [_domain_start[r], _domain_start[r+1])
            std::vector<key_type> _domain_start;

            // Ghost communication pattern (rebuilt after every change of the grid)
            std::vector<int> _send_ranks;
            std::vector<std::vector<IndexInt>> _send_indices;
            std::vector<int> _recv_ranks;
            std::vector<IndexInt> _recv_offset;
            std::vector<IndexInt> _recv_count;

            // Local leaves touching a remote leaf (need ghosts for a stencil) and the rest
            std::vector<IndexInt> _boundary_cells;
            std::vector<IndexInt> _interior_cells;

            // All grids can see each other's internals (truncated() builds a grid with another data type)
            template <int, class>
            friend class AdaptiveMPIGrid;

          public:
            //==========================================================================================
            /// Handle for a ghost exchange in flight (returned by begin_communicate_ghosts).
            /// Move-only. The destructor waits if wait() was never called, so an exchange is never
            /// left dangling. Values are packed when the exchange starts, so the local vector may be
            /// changed afterwards, but the ghost vector must not be resized, read or destroyed until
            /// wait() (or a successful test()) returns.
            //==========================================================================================
            template <class U>
            class GhostExchange {
                friend class AdaptiveMPIGrid;
                std::vector<std::vector<U>> _sendbuf;
#ifdef USE_MPI
                std::vector<MPI_Request> _req;
                MPI_Datatype _type{};
#endif
                bool _active{false};

              public:
                GhostExchange() = default;
                GhostExchange(const GhostExchange &) = delete;
                GhostExchange & operator=(const GhostExchange &) = delete;
                GhostExchange(GhostExchange && o) noexcept { *this = std::move(o); }
                GhostExchange & operator=(GhostExchange && o) noexcept {
                    if (this != &o) {
                        wait();
                        _sendbuf = std::move(o._sendbuf);
#ifdef USE_MPI
                        _req = std::move(o._req);
                        _type = o._type;
#endif
                        _active = o._active;
                        o._active = false;
                    }
                    return *this;
                }
                ~GhostExchange() { wait(); }

                /// Is the exchange still in flight (wait() not yet called)?
                bool active() const { return _active; }

                /// Non-blocking check. Returns true (and completes the exchange) if all messages arrived.
                bool test() {
                    if (not _active)
                        return true;
#ifdef USE_MPI
                    int flag = 0;
                    MPI_Testall(int(_req.size()), _req.data(), &flag, MPI_STATUSES_IGNORE);
                    if (not flag)
                        return false;
                    finish();
#else
                    _active = false;
#endif
                    return true;
                }

                /// Block until the ghost vector is filled
                void wait() {
                    if (not _active)
                        return;
#ifdef USE_MPI
                    MPI_Waitall(int(_req.size()), _req.data(), MPI_STATUSES_IGNORE);
                    finish();
#else
                    _active = false;
#endif
                }

              private:
                void finish() {
#ifdef USE_MPI
                    MPI_Type_free(&_type);
#endif
                    _sendbuf.clear();
                    _active = false;
                }
            };


            AdaptiveMPIGrid() = default;

            /// Create a uniform grid at levelmin with refinement allowed up to levelmax
            AdaptiveMPIGrid(int levelmin, int levelmax, std::array<bool, NDIM> periodic);
            AdaptiveMPIGrid(int levelmin, int levelmax, bool periodic = true)
                : AdaptiveMPIGrid(levelmin, levelmax, filled_array(periodic)) {}
            /// Create a uniform grid at levelmin split as the domain (NDIM = 3: a PeanoHilbertDomain, e.g. that of the
            /// particles or of another grid; its boundaries must fall on cells at levelmin)
            AdaptiveMPIGrid(int levelmin, int levelmax, std::array<bool, NDIM> periodic, const FML::Domain & domain);
            AdaptiveMPIGrid(int levelmin, int levelmax, bool periodic, const FML::Domain & domain)
                : AdaptiveMPIGrid(levelmin, levelmax, filled_array(periodic), domain) {}

            //======================================================================
            // Access
            //======================================================================
            IndexInt get_n_local() const { return IndexInt(_cells.size()); }
            IndexInt get_n_ghost() const { return IndexInt(_ghost_cells.size()); }
            IndexInt get_n_global() const;
            int get_levelmin() const { return _levelmin; }
            int get_levelmax() const { return _levelmax; }
            const std::array<bool, NDIM> & get_periodic() const { return _periodic; }

            const Cell & get_cell(IndexInt i) const { return _cells[i]; }
            T & get_data(IndexInt i) { return _data[i]; }
            const T & get_data(IndexInt i) const { return _data[i]; }
            std::vector<T> & get_data() { return _data; }

            const Cell & get_ghost_cell(IndexInt i) const { return _ghost_cells[i]; }
            const T & get_ghost_data(IndexInt i) const { return _ghost_data[i]; }

            const Cell & get_cell(const CellRef & r) const { return r.is_local() ? _cells[r.index] : _ghost_cells[r.index]; }
            const T & get_data(const CellRef & r) const { return r.is_local() ? _data[r.index] : _ghost_data[r.index]; }

            const std::vector<key_type> & get_domain_start() const { return _domain_start; }

            /// The decomposition as an FML::Domain (3D only: a PeanoHilbertDomain with LMAX = 21 bits, the same keys),
            /// e.g. for particles that follow the grid:
            /// [ grid.load_balance(); particles.set_domain(grid.get_domain()); ]
            /// A position x in [0,1)^3 is owned by the task that owns the leaf containing x
            FML::Domain get_domain() const {
                static_assert(NDIM == 3, "AdaptiveMPIGrid::get_domain: only for NDIM = 3 (PeanoHilbertDomain is 3D)");
                return FML::Domain::peano_hilbert(_domain_start, unsigned(LMAX));
            }

            //======================================================================
            // Geometry
            //======================================================================
            static double get_cell_size(int level) { return std::ldexp(1.0, -level); }
            static std::array<double, NDIM> get_cell_center(const Cell & c);
            static std::array<double, NDIM> get_cell_corner(const Cell & c);
            static key_type key_range(int level) { return key_type(1) << (NDIM * (LMAX - level)); }
            static coord_type size_int(int level) { return coord_type(1) << (LMAX - level); }
            static Cell make_cell(const Coord & anchor, int level);
            static Cell make_cell_from_key(key_type key, int level);

            /// Task that owns the point with a given key
            int owner_of_key(key_type key) const;

            /// Find the local or ghost leaf containing a given integer point (NONE if not available)
            CellRef find_leaf(const Coord & x) const;
            /// Find the local or ghost leaf whose key range contains the key (NONE if not available)
            CellRef find_leaf_by_key(key_type key) const;
            /// Find the local or ghost leaf containing a point in [0,1)^NDIM
            CellRef find_leaf(const std::array<double, NDIM> & x) const;
            /// find_leaf for a point near local leaf `hint` (e.g. in a neighbour of it): the leaves within a few
            /// places of the hint along the curve are searched first, which is faster (same result). With level < LMAX
            /// the point stands for the whole cell at that level containing it: the result is the leaf containing
            /// that cell if the cell is not refined, else one of the leaves inside it (cheaper key)
            CellRef find_leaf_near(const Coord & x, IndexInt hint, int level = LMAX) const;

            /// The neighbours across the face in direction dim, side sign = +-1
            FaceNeighbors get_face_neighbors(IndexInt i, int dim, int sign) const;

            /// Call f(Coord box_anchor, int box_level) for every box at level min(l+1,LMAX) that touches
            /// the cell (faces, edges and corners) and lies inside the domain (periodic wrapping applied).
            /// Under 2:1 balance every leaf touching the cell contains or equals one of these boxes.
            template <class F>
            void for_each_touching_box(const Cell & c, F && f) const;

            //======================================================================
            // Adaptivity and load balancing
            //======================================================================
            void set_flag(IndexInt i, int8_t flag) { _flags[i] = flag; }
            int8_t get_flag(IndexInt i) const { return _flags[i]; }

            /// Enforce 2:1 balance on the flags, refine/coarsen, rebuild ghosts and fill them
            void adapt(ProlongFunction prolong = nullptr, RestrictFunction restrict_func = nullptr);

            /// Repartition the Hilbert curve by weight (default 1 per cell), migrate cells and rebuild ghosts
            void load_balance(WeightFunction weight = nullptr);

            //======================================================================
            // Coarser grids (e.g. for multigrid)
            //======================================================================
            /// The grid truncated at a level (collective): every leaf finer than level is replaced by its ancestor at
            /// level. For level < levelmin this is the uniform grid at level (a full grid coarser than the base grid),
            /// and for level >= the finest leaf it is a copy of the leaf set (same partition, same local indices).
            /// The truncation of a 2:1 balanced grid is 2:1 balanced. The partition is this one with every cut rounded
            /// down to a cell at level (so a coarse cell is never split), which means a few leaves can change task.
            /// The cell data is U{}: use AdaptiveMPIGridTransfer to move values between the grids.
            /// Note: the cuts of the result can fall inside a sibling family, so do not adapt() a truncated grid
            /// if the leaf set must be independent of the number of tasks.
            template <class U = T>
            AdaptiveMPIGrid<NDIM, U> truncated(int level) const;

            //======================================================================
            // Ghosts
            //======================================================================
            /// Fill the ghost data from the owners (blocking)
            void exchange_ghosts() { communicate_ghosts(_data, _ghost_data); }

            /// Start filling the ghost data. Overlap with work on get_interior_cells(), then call
            /// wait() on the handle before touching get_boundary_cells():
            ///   auto h = grid.begin_exchange_ghosts();
            ///   for (auto i : grid.get_interior_cells()) ...;   // only local neighbours
            ///   h.wait();
            ///   for (auto i : grid.get_boundary_cells()) ...;   // may use ghosts
            /// Do not modify the cell data until wait() has returned.
            GhostExchange<T> begin_exchange_ghosts() { return begin_communicate_ghosts(_data, _ghost_data); }

            /// Communicate any per-cell quantity to the ghost layer (blocking)
            template <class U>
            void communicate_ghosts(const std::vector<U> & local, std::vector<U> & ghost) const {
                begin_communicate_ghosts(local, ghost).wait();
            }

            /// Start communicating any per-cell quantity to the ghost layer (non-blocking).
            /// The values are packed immediately, so local may be modified right after this returns.
            template <class U>
            GhostExchange<U> begin_communicate_ghosts(const std::vector<U> & local, std::vector<U> & ghost) const;

            /// Local leaves that touch at least one remote leaf (sorted). These need ghosts for a stencil.
            const std::vector<IndexInt> & get_boundary_cells() const { return _boundary_cells; }
            /// Local leaves whose neighbours are all local (sorted). These can be processed while
            /// a ghost exchange is in flight.
            const std::vector<IndexInt> & get_interior_cells() const { return _interior_cells; }

            //======================================================================
            // Diagnostics
            //======================================================================
            /// Check sorting, global coverage, ownership, ghost completeness and 2:1 balance
            bool check_invariants(bool verbose = true) const;
            /// Order independent hash of the global leaf set (equal for any number of tasks)
            uint64_t global_hash() const;
            /// Write "x y [z] cellsize level task" for every local leaf to prefix.task
            void write_ascii(std::string prefix) const;
            void info() const;

          private:
            static std::array<bool, NDIM> filled_array(bool b) {
                std::array<bool, NDIM> a;
                a.fill(b);
                return a;
            }
            static CellRef search(const std::vector<Cell> & v, key_type k, CellRef::Type type);
            bool wrap(std::array<int64_t, NDIM> & x) const;
            void update_domain_start();
            void check_levels() const;
            void create_uniform(std::vector<key_type> start);
            void build_ghost_pattern();
            bool neighbourhood_is_local(const Cell & c) const;

          public:
            /// Buffers for sparse exchanges: task -> buffer (only the tasks we talk to)
            template <class U>
            using SparseBuffers = std::map<int, std::vector<U>>;
            /// Sparse personalized exchange (collective): send[r] goes to task r, the result holds what every task
            /// sent to us (keyed by sender, in task order). Buffers to ourselves are copied. See nbx_min_tasks.
            template <class U>
            SparseBuffers<U> alltoallv_sparse(const SparseBuffers<U> & send) const;
        };

        //==========================================================================================
        // MPI helpers
        //==========================================================================================
        namespace AMRDETAIL {
#ifdef USE_MPI
            template <class U>
            struct MPIType {
                MPI_Datatype type;
                MPIType() {
                    MPI_Type_contiguous(int(sizeof(U)), MPI_BYTE, &type);
                    MPI_Type_commit(&type);
                }
                ~MPIType() { MPI_Type_free(&type); }
            };
#endif
            inline bool allreduce_or(bool b) {
#ifdef USE_MPI
                int v = b ? 1 : 0;
                MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_MAX, FML::communicator);
                return v != 0;
#else
                return b;
#endif
            }
            inline uint64_t splitmix64(uint64_t x) {
                x += 0x9e3779b97f4a7c15ULL;
                x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
                x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
                return x ^ (x >> 31);
            }
            // Tag counters shared by all grids (and AdaptiveMPIGridTransfer), so exchanges on different grids
            // never use the same tag. Collective calls are made in the same order on all tasks, so they agree.
            inline int ghost_tag_counter = 0;
            inline int nbx_tag_counter = 0;
            inline int transfer_tag_counter = 0;
        } // namespace AMRDETAIL

        //==========================================================================================
        // Implementation
        //==========================================================================================

        template <int NDIM, class T>
        AdaptiveMPIGrid<NDIM, T>::AdaptiveMPIGrid(int levelmin, int levelmax, std::array<bool, NDIM> periodic)
            : _levelmin(levelmin), _levelmax(levelmax), _periodic(periodic) {
            check_levels();
            // Uniform grid at levelmin split into equal parts of the curve
            const key_type ncells = key_type(1) << (NDIM * levelmin);
            const int shift = NDIM * (LMAX - levelmin);
            std::vector<key_type> start(FML::NTasks + 1);
            for (int r = 0; r <= FML::NTasks; r++)
                start[r] = key_type((__uint128_t(ncells) * r) / FML::NTasks) << shift;
            start[FML::NTasks] = KEY_END;
            create_uniform(std::move(start));
        }

        template <int NDIM, class T>
        AdaptiveMPIGrid<NDIM, T>::AdaptiveMPIGrid(int levelmin,
                                                  int levelmax,
                                                  std::array<bool, NDIM> periodic,
                                                  const FML::Domain & domain)
            : _levelmin(levelmin), _levelmax(levelmax), _periodic(periodic) {
            static_assert(NDIM == 3, "AdaptiveMPIGrid: a domain only for NDIM = 3 (PeanoHilbertDomain is 3D)");
            check_levels();
            assert_mpi(domain.holds<FML::PeanoHilbertDomain>(),
                       "AdaptiveMPIGrid: the domain must be a PeanoHilbertDomain (the grid is split along the curve)");
            assert_mpi(domain.ntasks() == FML::NTasks, "AdaptiveMPIGrid: the domain is not made for NTasks tasks");
            const auto & ph = domain.get<FML::PeanoHilbertDomain>();
            assert_mpi(int(ph.bits()) <= LMAX, "AdaptiveMPIGrid: the domain has more bits than LMAX");

            // The boundaries in the grid's keys (LMAX bits: the same curve, refined) must fall on levelmin cells
            const int shift = NDIM * (LMAX - int(ph.bits()));
            std::vector<key_type> start(FML::NTasks + 1);
            bool on_cells = true;
            for (int r = 0; r <= FML::NTasks; r++) {
                start[r] = r == FML::NTasks ? KEY_END : key_type(ph.key_begin(r)) << shift;
                on_cells = on_cells and start[r] % key_range(levelmin) == 0;
            }
            assert_mpi(on_cells, "AdaptiveMPIGrid: the boundaries of the domain must fall on cells at levelmin");
            create_uniform(std::move(start));
        }

        template <int NDIM, class T>
        void AdaptiveMPIGrid<NDIM, T>::check_levels() const {
            assert_mpi(_levelmin >= 1 and _levelmin <= _levelmax and _levelmax <= LMAX,
                       "AdaptiveMPIGrid: need 1 <= levelmin <= levelmax <= LMAX");
            for (int d = 0; d < NDIM; d++)
                assert_mpi(not _periodic[d] or _levelmin >= 2,
                           "AdaptiveMPIGrid: periodic axes need levelmin >= 2 (at least 4 cells per dimension)");
        }

        template <int NDIM, class T>
        void AdaptiveMPIGrid<NDIM, T>::create_uniform(std::vector<key_type> start) {
            // The cells at levelmin in this task's part of the curve
            _domain_start = std::move(start);
            const key_type step = key_range(_levelmin);
            const key_type kbegin = _domain_start[FML::ThisTask];
            const key_type kend = _domain_start[FML::ThisTask + 1];
            _cells.clear();
            _cells.reserve((kend - kbegin) / step);
            for (key_type k = kbegin; k < kend; k += step)
                _cells.push_back(make_cell_from_key(k, _levelmin));
            _data.assign(_cells.size(), T{});
            _flags.assign(_cells.size(), KEEP);
            build_ghost_pattern();
        }

        template <int NDIM, class T>
        IndexInt AdaptiveMPIGrid<NDIM, T>::get_n_global() const {
            IndexInt n = get_n_local();
#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_LONG_LONG, MPI_SUM, FML::communicator);
#endif
            return n;
        }

        template <int NDIM, class T>
        std::array<double, NDIM> AdaptiveMPIGrid<NDIM, T>::get_cell_corner(const Cell & c) {
            std::array<double, NDIM> x;
            for (int d = 0; d < NDIM; d++)
                x[d] = std::ldexp(double(c.anchor[d]), -LMAX);
            return x;
        }

        template <int NDIM, class T>
        std::array<double, NDIM> AdaptiveMPIGrid<NDIM, T>::get_cell_center(const Cell & c) {
            std::array<double, NDIM> x;
            const double h = get_cell_size(c.level);
            for (int d = 0; d < NDIM; d++)
                x[d] = std::ldexp(double(c.anchor[d]), -LMAX) + 0.5 * h;
            return x;
        }

        template <int NDIM, class T>
        typename AdaptiveMPIGrid<NDIM, T>::Cell AdaptiveMPIGrid<NDIM, T>::make_cell(const Coord & anchor, int level) {
            Cell c;
            c.level = int8_t(level);
            const coord_type mask = ~(size_int(level) - 1);
            for (int d = 0; d < NDIM; d++)
                c.anchor[d] = anchor[d] & mask;
            c.key = HC::key_from_coord(c.anchor) & ~(key_range(level) - 1);
            return c;
        }

        template <int NDIM, class T>
        typename AdaptiveMPIGrid<NDIM, T>::Cell AdaptiveMPIGrid<NDIM, T>::make_cell_from_key(key_type key, int level) {
            Cell c;
            c.level = int8_t(level);
            c.key = key & ~(key_range(level) - 1);
            const coord_type mask = ~(size_int(level) - 1);
            c.anchor = HC::coord_from_key(c.key);
            for (int d = 0; d < NDIM; d++)
                c.anchor[d] &= mask;
            return c;
        }

        template <int NDIM, class T>
        int AdaptiveMPIGrid<NDIM, T>::owner_of_key(key_type key) const {
            // Last task with start <= key. Empty tasks have zero width and are skipped automatically.
            auto it = std::upper_bound(_domain_start.begin(), _domain_start.end() - 1, key);
            return int(it - _domain_start.begin()) - 1;
        }

        template <int NDIM, class T>
        AMRCellRef AdaptiveMPIGrid<NDIM, T>::search(const std::vector<Cell> & v, key_type k, CellRef::Type type) {
            auto it = std::upper_bound(v.begin(), v.end(), k, [](key_type a, const Cell & c) { return a < c.key; });
            if (it == v.begin())
                return CellRef{};
            --it;
            if (k - it->key < key_range(it->level))
                return CellRef{type, IndexInt(it - v.begin())};
            return CellRef{};
        }

        template <int NDIM, class T>
        AMRCellRef AdaptiveMPIGrid<NDIM, T>::find_leaf(const Coord & x) const {
            const key_type k = HC::key_from_coord(x);
            if (k >= _domain_start[FML::ThisTask] and k < _domain_start[FML::ThisTask + 1])
                return search(_cells, k, CellRef::LOCAL);
            return search(_ghost_cells, k, CellRef::GHOST);
        }

        template <int NDIM, class T>
        AMRCellRef AdaptiveMPIGrid<NDIM, T>::find_leaf_near(const Coord & x, IndexInt hint, int level) const {
            const key_type k = HC::cell_key(x, level);
            if (k >= _domain_start[FML::ThisTask] and k < _domain_start[FML::ThisTask + 1]) {
                const IndexInt n = IndexInt(_cells.size());
                if (hint >= 0 and hint < n) {
                    // The local leaves tile the local key range in order, so a key inside the window is in one of them
                    const IndexInt lo = std::max(IndexInt(0), hint - 16), hi = std::min(n - 1, hint + 16);
                    if (k >= _cells[lo].key and (k < _cells[hi].key or k - _cells[hi].key < key_range(_cells[hi].level))) {
                        auto it = std::upper_bound(_cells.begin() + lo, _cells.begin() + hi + 1, k,
                                                   [](key_type a, const Cell & c) { return a < c.key; });
                        return CellRef{CellRef::LOCAL, IndexInt(it - _cells.begin()) - 1};
                    }
                }
                return search(_cells, k, CellRef::LOCAL);
            }
            return search(_ghost_cells, k, CellRef::GHOST);
        }

        template <int NDIM, class T>
        AMRCellRef AdaptiveMPIGrid<NDIM, T>::find_leaf_by_key(key_type k) const {
            if (k >= _domain_start[FML::ThisTask] and k < _domain_start[FML::ThisTask + 1])
                return search(_cells, k, CellRef::LOCAL);
            return search(_ghost_cells, k, CellRef::GHOST);
        }

        template <int NDIM, class T>
        AMRCellRef AdaptiveMPIGrid<NDIM, T>::find_leaf(const std::array<double, NDIM> & x) const {
            std::array<int64_t, NDIM> xi;
            for (int d = 0; d < NDIM; d++)
                xi[d] = int64_t(std::floor(std::ldexp(x[d], LMAX)));
            if (not wrap(xi))
                return CellRef{};
            Coord c;
            for (int d = 0; d < NDIM; d++)
                c[d] = coord_type(xi[d]);
            return find_leaf(c);
        }

        template <int NDIM, class T>
        bool AdaptiveMPIGrid<NDIM, T>::wrap(std::array<int64_t, NDIM> & x) const {
            const int64_t D = int64_t(1) << LMAX;
            for (int d = 0; d < NDIM; d++) {
                if (x[d] >= 0 and x[d] < D)
                    continue;
                if (not _periodic[d])
                    return false;
                x[d] = ((x[d] % D) + D) % D;
            }
            return true;
        }

        template <int NDIM, class T>
        template <class F>
        void AdaptiveMPIGrid<NDIM, T>::for_each_touching_box(const Cell & c, F && f) const {
            const int m = std::min(int(c.level) + 1, LMAX);
            const int64_t s = int64_t(size_int(m));
            const int nin = int(size_int(c.level) / s); // boxes per axis inside the cell (2 or 1)
            const int nper = nin + 2;                   // including one box on each side
            int ntot = 1;
            for (int d = 0; d < NDIM; d++)
                ntot *= nper;

            std::array<int64_t, NDIM> x;
            Coord xc;
            for (int idx = 0; idx < ntot; idx++) {
                int rest = idx;
                bool inside = true;
                for (int d = 0; d < NDIM; d++) {
                    const int k = rest % nper - 1;
                    rest /= nper;
                    inside = inside and (k >= 0 and k < nin);
                    x[d] = int64_t(c.anchor[d]) + k * s;
                }
                if (inside or not wrap(x))
                    continue;
                // With wrapping a box could in principle land inside the cell itself
                bool self = true;
                for (int d = 0; d < NDIM; d++) {
                    xc[d] = coord_type(x[d]);
                    self = self and (xc[d] - c.anchor[d] < size_int(c.level));
                }
                if (self)
                    continue;
                f(xc, m);
            }
        }

        template <int NDIM, class T>
        typename AdaptiveMPIGrid<NDIM, T>::FaceNeighbors AdaptiveMPIGrid<NDIM, T>::get_face_neighbors(IndexInt i,
                                                                                                   int dim,
                                                                                                   int sign) const {
            FaceNeighbors res;
            const Cell & c = _cells[i];
            const int64_t h = size_int(c.level);

            std::array<int64_t, NDIM> x;
            for (int d = 0; d < NDIM; d++)
                x[d] = c.anchor[d];
            x[dim] += sign > 0 ? h : -1;
            if (not wrap(x))
                return res; // BOUNDARY

            Coord xc;
            for (int d = 0; d < NDIM; d++)
                xc[d] = coord_type(x[d]);
            // The leaf containing the point next to our anchor (it touches the cell, so it is local or a ghost)
            CellRef r = find_leaf_near(xc, i);
            assert_mpi(bool(r), "get_face_neighbors: neighbour not found (ghosts not built?)");
            const int nlevel = get_cell(r).level;
            if (nlevel <= c.level) {
                res.relation = nlevel < c.level ? FaceNeighbors::COARSER : FaceNeighbors::SAME;
                res.n = 1;
                res.cells[0] = r;
                return res;
            }

            // Finer neighbours: the 2^(NDIM-1) children of the same-size box that touch the face
            res.relation = FaceNeighbors::FINER;
            const int64_t s = h / 2;
            for (int sub = 0; sub < (1 << (NDIM - 1)); sub++) {
                int bit = 0;
                for (int d = 0; d < NDIM; d++) {
                    if (d == dim) {
                        x[d] = int64_t(c.anchor[d]) + (sign > 0 ? h : -s);
                    } else {
                        x[d] = int64_t(c.anchor[d]) + ((sub >> bit) & 1) * s;
                        bit++;
                    }
                }
                wrap(x);
                for (int d = 0; d < NDIM; d++)
                    xc[d] = coord_type(x[d]);
                // The finer neighbours are leaves at our level + 1 touching us (2:1 balance): a cell key will do
                res.cells[res.n++] = find_leaf_near(xc, i, c.level + 1);
            }
            return res;
        }

        //==========================================================================================
        // Communication
        //==========================================================================================

        //==========================================================================================
        /// Sparse personalized exchange: every task sends a buffer to a few other tasks and does not
        /// know who will send to it. Uses the NBX algorithm (Hoefler, Siebert & Lumsdaine, PPoPP 2010):
        /// synchronous sends + probing for incoming messages + a non-blocking barrier that is entered
        /// once all our sends have been matched. No O(NTasks) collective or O(NTasks) memory is
        /// needed; cost scales with the number of tasks we actually talk to.
        /// Buffers to ourselves are copied locally. Empty buffers are not sent.
        /// Below nbx_min_tasks tasks a dense count exchange is used instead (see nbx_min_tasks).
        //==========================================================================================
        template <int NDIM, class T>
        template <class U>
        typename AdaptiveMPIGrid<NDIM, T>::template SparseBuffers<U>
        AdaptiveMPIGrid<NDIM, T>::alltoallv_sparse(const SparseBuffers<U> & send) const {
            SparseBuffers<U> recv;
            auto self = send.find(FML::ThisTask);
            if (self != send.end() and not self->second.empty())
                recv[FML::ThisTask] = self->second;
#ifdef USE_MPI
            // Unique tag per call: a task leaving one exchange may start the next while others are
            // still probing for the previous one. Calls happen in the same order on all tasks.
            const int tag = 31000 + (AMRDETAIL::nbx_tag_counter++ % 1000);
            AMRDETAIL::MPIType<U> mtype;

            if (FML::NTasks < nbx_min_tasks) {
                // Dense variant: exchange counts with a collective, then point-to-point
                std::vector<int> nsend(FML::NTasks, 0), nrecv(FML::NTasks, 0);
                for (auto & [r, buf] : send)
                    if (r != FML::ThisTask) {
                        assert_mpi(buf.size() <= size_t(INT_MAX), "alltoallv_sparse: message count exceeds INT_MAX");
                        nsend[r] = int(buf.size());
                    }
                MPI_Alltoall(nsend.data(), 1, MPI_INT, nrecv.data(), 1, MPI_INT, FML::communicator);
                std::vector<MPI_Request> req;
                for (int r = 0; r < FML::NTasks; r++) {
                    if (nrecv[r] == 0)
                        continue;
                    auto & buf = recv[r];
                    buf.resize(nrecv[r]);
                    req.emplace_back();
                    MPI_Irecv(buf.data(), nrecv[r], mtype.type, r, tag, FML::communicator, &req.back());
                }
                for (auto & [r, buf] : send) {
                    if (r == FML::ThisTask or buf.empty())
                        continue;
                    req.emplace_back();
                    MPI_Isend(buf.data(), int(buf.size()), mtype.type, r, tag, FML::communicator, &req.back());
                }
                MPI_Waitall(int(req.size()), req.data(), MPI_STATUSES_IGNORE);
                return recv;
            }

            std::vector<MPI_Request> sreq;
            for (auto & [r, buf] : send) {
                if (r == FML::ThisTask or buf.empty())
                    continue;
                assert_mpi(buf.size() <= size_t(INT_MAX), "alltoallv_sparse: message count exceeds INT_MAX");
                sreq.emplace_back();
                MPI_Issend(buf.data(), int(buf.size()), mtype.type, r, tag, FML::communicator, &sreq.back());
            }

            MPI_Request barrier;
            bool barrier_active = false;
            bool done = false;
            while (not done) {
                int flag = 0;
                MPI_Status status;
                MPI_Iprobe(MPI_ANY_SOURCE, tag, FML::communicator, &flag, &status);
                if (flag) {
                    int count = 0;
                    MPI_Get_count(&status, mtype.type, &count);
                    auto & buf = recv[status.MPI_SOURCE];
                    buf.resize(count);
                    MPI_Recv(buf.data(), count, mtype.type, status.MPI_SOURCE, tag, FML::communicator,
                             MPI_STATUS_IGNORE);
                }
                if (barrier_active) {
                    int bdone = 0;
                    MPI_Test(&barrier, &bdone, MPI_STATUS_IGNORE);
                    done = bdone != 0;
                } else {
                    // Synchronous sends complete only once matched by a receive on the other side
                    int sdone = 0;
                    MPI_Testall(int(sreq.size()), sreq.data(), &sdone, MPI_STATUSES_IGNORE);
                    if (sdone) {
                        MPI_Ibarrier(FML::communicator, &barrier);
                        barrier_active = true;
                    }
                }
            }
#endif
            return recv;
        }

        template <int NDIM, class T>
        template <class U>
        typename AdaptiveMPIGrid<NDIM, T>::template GhostExchange<U>
        AdaptiveMPIGrid<NDIM, T>::begin_communicate_ghosts(const std::vector<U> & local, std::vector<U> & ghost) const {
            static_assert(std::is_trivially_copyable<U>::value, "communicate_ghosts: U must be trivially copyable");
            assert_mpi(local.size() == _cells.size(), "communicate_ghosts: local vector has wrong size");
            ghost.resize(_ghost_cells.size());
            GhostExchange<U> h;
            h._active = true;
#ifdef USE_MPI
            const int tag = 100 + (AMRDETAIL::ghost_tag_counter++ % 30000);
            MPI_Type_contiguous(int(sizeof(U)), MPI_BYTE, &h._type);
            MPI_Type_commit(&h._type);
            h._req.resize(_recv_ranks.size() + _send_ranks.size());
            int nreq = 0;
            for (size_t j = 0; j < _recv_ranks.size(); j++)
                assert_mpi(size_t(_recv_count[j]) <= size_t(INT_MAX), "communicate_ghosts: count exceeds INT_MAX");
            for (size_t j = 0; j < _send_indices.size(); j++)
                assert_mpi(_send_indices[j].size() <= size_t(INT_MAX), "communicate_ghosts: count exceeds INT_MAX");
            for (size_t j = 0; j < _recv_ranks.size(); j++)
                MPI_Irecv(ghost.data() + _recv_offset[j], int(_recv_count[j]), h._type, _recv_ranks[j], tag,
                          FML::communicator, &h._req[nreq++]);
            h._sendbuf.resize(_send_ranks.size());
            for (size_t j = 0; j < _send_ranks.size(); j++) {
                auto & buf = h._sendbuf[j];
                buf.reserve(_send_indices[j].size());
                for (auto i : _send_indices[j])
                    buf.push_back(local[i]);
                MPI_Isend(
                    buf.data(), int(buf.size()), h._type, _send_ranks[j], tag, FML::communicator, &h._req[nreq++]);
            }
#else
            (void)local;
#endif
            return h;
        }

        template <int NDIM, class T>
        bool AdaptiveMPIGrid<NDIM, T>::neighbourhood_is_local(const Cell & c) const {
            // The boxes of for_each_touching_box lie in the cell grown by one box (s = one level finer) on
            // every side. If that region (without wrapping) lies inside an aligned box whose key range
            // (contiguous on the curve) is inside our part of the curve, every box, and so every leaf
            // touching the cell, is ours. The smallest aligned box containing [lo, hi] along an axis
            // has the size of the highest bit where lo and hi differ.
            const int me = FML::ThisTask;
            const key_type kbegin = _domain_start[me], kend = _domain_start[me + 1];
            if (kbegin == 0 and kend == KEY_END)
                return true; // We own everything (periodic neighbours included)
            if (kbegin == kend)
                return true; // No cells
            const int64_t D = int64_t(1) << LMAX;
            const int64_t h = int64_t(size_int(c.level));
            const int64_t s = int64_t(size_int(std::min(int(c.level) + 1, LMAX)));
            uint64_t diff = 0;
            Coord lo;
            for (int d = 0; d < NDIM; d++) {
                const int64_t a = int64_t(c.anchor[d]) - s;
                const int64_t b = int64_t(c.anchor[d]) + h + s - 1;
                if (a < 0 or b >= D)
                    return false; // Wraps or leaves the domain: use the full test
                lo[d] = coord_type(a);
                diff |= uint64_t(a) ^ uint64_t(b);
            }
            int nbits = 0; // Size of the containing aligned box is 2^nbits
            while (diff >> nbits)
                nbits++;
            const key_type range = key_type(1) << (NDIM * nbits);
            const key_type kbox = HC::key_from_coord(lo) & ~(range - 1);
            return kbox >= kbegin and kbox + range <= kend;
        }

        template <int NDIM, class T>
        void AdaptiveMPIGrid<NDIM, T>::build_ghost_pattern() {
            const int me = FML::ThisTask;
            SparseBuffers<IndexInt> sendlists;

            // A local leaf is a ghost on every other task whose part of the curve intersects one of the
            // boxes (one level finer) touching the leaf. Under 2:1 balance this is exactly the set of
            // tasks owning a leaf that touches it. Leaves whose neighbourhood is local are skipped
            // (see neighbourhood_is_local), which is most of them.
            for (IndexInt i = 0; i < get_n_local(); i++) {
                if (neighbourhood_is_local(_cells[i]))
                    continue;
                for_each_touching_box(_cells[i], [&](const Coord & x, int m) {
                    const key_type kb = HC::key_from_coord(x) & ~(key_range(m) - 1);
                    const int rlo = owner_of_key(kb);
                    const int rhi = owner_of_key(kb + key_range(m) - 1);
                    for (int r = rlo; r <= rhi; r++) {
                        if (r == me or _domain_start[r] == _domain_start[r + 1])
                            continue;
                        auto & list = sendlists[r];
                        if (list.empty() or list.back() != i)
                            list.push_back(i);
                    }
                });
            }

            // Boundary leaves are those sent to someone (touching relation is symmetric under 2:1 balance)
            std::vector<char> is_boundary(_cells.size(), 0);
            for (auto & [r, list] : sendlists)
                for (auto i : list)
                    is_boundary[i] = 1;
            _boundary_cells.clear();
            _interior_cells.clear();
            for (IndexInt i = 0; i < get_n_local(); i++)
                (is_boundary[i] ? _boundary_cells : _interior_cells).push_back(i);

            // Send the cell descriptions
            SparseBuffers<Cell> sendcells;
            _send_ranks.clear();
            _send_indices.clear();
            for (auto & [r, list] : sendlists) {
                _send_ranks.push_back(r);
                auto & buf = sendcells[r];
                for (auto i : list)
                    buf.push_back(_cells[i]);
                _send_indices.push_back(std::move(list));
            }
            auto recvcells = alltoallv_sparse(sendcells);

            // Ghosts from task r are contiguous and the tasks are ordered along the curve, so the
            // concatenation in task order is sorted by key
            _ghost_cells.clear();
            _recv_ranks.clear();
            _recv_offset.clear();
            _recv_count.clear();
            for (auto & [r, buf] : recvcells) {
                _recv_ranks.push_back(r);
                _recv_offset.push_back(IndexInt(_ghost_cells.size()));
                _recv_count.push_back(IndexInt(buf.size()));
                _ghost_cells.insert(_ghost_cells.end(), buf.begin(), buf.end());
            }
            _ghost_data.assign(_ghost_cells.size(), T{});
        }

        template <int NDIM, class T>
        void AdaptiveMPIGrid<NDIM, T>::update_domain_start() {
            // First key of every task; empty tasks get the start of the next task (zero width)
            key_type first = _cells.empty() ? KEY_END : _cells.front().key;
            _domain_start.assign(FML::NTasks + 1, KEY_END);
#ifdef USE_MPI
            MPI_Allgather(&first, 1, MPI_UINT64_T, _domain_start.data(), 1, MPI_UINT64_T, FML::communicator);
#else
            _domain_start[0] = first;
#endif
            _domain_start[FML::NTasks] = KEY_END;
            for (int r = FML::NTasks - 1; r >= 0; r--)
                if (_domain_start[r] == KEY_END)
                    _domain_start[r] = _domain_start[r + 1];
            _domain_start[0] = 0;
        }

        //==========================================================================================
        // Adaptivity
        //==========================================================================================

        template <int NDIM, class T>
        void AdaptiveMPIGrid<NDIM, T>::adapt(ProlongFunction prolong, RestrictFunction restrict_func) {
            const IndexInt n = get_n_local();

            // 1. Target levels from the refinement flags
            std::vector<int8_t> target(n), ghost_target;
            for (IndexInt i = 0; i < n; i++)
                target[i] = int8_t(_cells[i].level + ((_flags[i] == REFINE and _cells[i].level < _levelmax) ? 1 : 0));
            auto target_of = [&](const CellRef & r) { return r.is_local() ? target[r.index] : ghost_target[r.index]; };

            // 2. Parallel 2:1 balance (faces, edges, corners): refine until no target exceeds a
            // neighbour's target by more than one. Iterate until no task changes anything.
            // Work list: the grid is balanced before adapt, so a leaf can only need refining if it touches a
            // leaf whose target went up. Only those are checked (first the neighbours of the leaves flagged
            // for refinement, then the neighbours of the leaves raised in the last round, local or ghost).
            // The result is the same least fixed point as checking every leaf in every round, at a cost
            // proportional to the number of refined leaves instead of all leaves
            std::vector<char> check(n, 0);
            auto mark_neighbours = [&](const Cell & c) {
                for_each_touching_box(c, [&](const Coord & x, int) {
                    CellRef r = find_leaf(x);
                    if (r.is_local())
                        check[r.index] = 1;
                });
            };
            communicate_ghosts(target, ghost_target);
            for (IndexInt i = 0; i < n; i++)
                if (target[i] > _cells[i].level)
                    mark_neighbours(_cells[i]);
            for (IndexInt g = 0; g < get_n_ghost(); g++)
                if (ghost_target[g] > _ghost_cells[g].level)
                    mark_neighbours(_ghost_cells[g]);
            std::vector<IndexInt> raised;
            std::vector<int8_t> ghost_target_old;
            while (true) {
                raised.clear();
                for (IndexInt i = 0; i < n; i++) {
                    if (not check[i])
                        continue;
                    check[i] = 0;
                    if (target[i] > _cells[i].level)
                        continue;
                    int maxnb = 0;
                    for_each_touching_box(_cells[i], [&](const Coord & x, int) {
                        CellRef r = find_leaf(x);
                        assert_mpi(bool(r), "adapt: missing neighbour leaf");
                        maxnb = std::max(maxnb, int(target_of(r)));
                    });
                    if (maxnb > target[i] + 1) {
                        target[i] = int8_t(_cells[i].level + 1);
                        raised.push_back(i);
                    }
                }
                if (not AMRDETAIL::allreduce_or(not raised.empty()))
                    break;
                ghost_target_old = ghost_target;
                communicate_ghosts(target, ghost_target);
                for (auto i : raised)
                    mark_neighbours(_cells[i]);
                for (IndexInt g = 0; g < get_n_ghost(); g++)
                    if (ghost_target[g] != ghost_target_old[g])
                        mark_neighbours(_ghost_cells[g]);
            }
            // ghost_target is now up to date (last round made no changes)

            // 3. Coarsening: complete local families, all flagged, none refining, and no leaf touching the
            // family has target > l (conservative since neighbours can only end up coarser than target)
            std::vector<char> family_start(n, 0);
            for (IndexInt i = 0; i + NCHILD <= n;) {
                const Cell & c = _cells[i];
                const int l = c.level;
                bool ok = l > _levelmin and (c.key & (key_range(l - 1) - 1)) == 0;
                for (int j = 0; ok and j < NCHILD; j++) {
                    const IndexInt k = i + j;
                    ok = _cells[k].level == l and _flags[k] == COARSEN and target[k] == l;
                }
                ok = ok and _cells[i + NCHILD - 1].key + key_range(l) == c.key + key_range(l - 1);
                for (int j = 0; ok and j < NCHILD; j++) {
                    for_each_touching_box(_cells[i + j], [&](const Coord & x, int) {
                        if (not ok)
                            return;
                        CellRef r = find_leaf(x);
                        assert_mpi(bool(r), "adapt: missing neighbour leaf");
                        ok = target_of(r) <= l;
                    });
                }
                if (ok) {
                    family_start[i] = 1;
                    i += NCHILD;
                } else {
                    i++;
                }
            }

            // 4. Build the new leaf list (stays sorted: children replace their parent within its key range)
            std::vector<Cell> newcells;
            std::vector<T> newdata;
            newcells.reserve(n);
            newdata.reserve(n);
            std::vector<Cell> kids(NCHILD);
            std::vector<T> kidvals(NCHILD);
            for (IndexInt i = 0; i < n;) {
                const Cell & c = _cells[i];
                if (family_start[i]) {
                    Cell parent = make_cell(c.anchor, c.level - 1);
                    for (int j = 0; j < NCHILD; j++) {
                        kids[j] = _cells[i + j];
                        kidvals[j] = _data[i + j];
                    }
                    T val;
                    if (restrict_func) {
                        val = restrict_func(kidvals, kids, parent);
                    } else if constexpr (std::is_arithmetic<T>::value) {
                        double sum = 0.0;
                        for (auto & v : kidvals)
                            sum += double(v);
                        val = T(sum / NCHILD);
                    } else {
                        val = kidvals[0];
                    }
                    newcells.push_back(parent);
                    newdata.push_back(val);
                    i += NCHILD;
                    continue;
                }
                if (target[i] > c.level) {
                    const coord_type s = size_int(c.level + 1);
                    for (int j = 0; j < NCHILD; j++) {
                        Coord a = c.anchor;
                        for (int d = 0; d < NDIM; d++)
                            a[d] += ((j >> d) & 1) * s;
                        kids[j] = make_cell(a, c.level + 1);
                    }
                    std::sort(kids.begin(), kids.end(), [](const Cell & a, const Cell & b) { return a.key < b.key; });
                    for (int j = 0; j < NCHILD; j++) {
                        newcells.push_back(kids[j]);
                        newdata.push_back(prolong ? prolong(_data[i], c, kids[j]) : _data[i]);
                    }
                } else {
                    newcells.push_back(c);
                    newdata.push_back(_data[i]);
                }
                i++;
            }
            _cells = std::move(newcells);
            _data = std::move(newdata);
            _flags.assign(_cells.size(), KEEP);

            // 5. Splitters are unchanged (all changes stay inside a task's key range); rebuild ghosts
            build_ghost_pattern();
            exchange_ghosts();
        }

        template <int NDIM, class T>
        void AdaptiveMPIGrid<NDIM, T>::load_balance(WeightFunction weight) {
            const IndexInt n = get_n_local();
            std::vector<double> w(n, 1.0);
            if (weight)
                for (IndexInt i = 0; i < n; i++)
                    w[i] = weight(i);

            double wlocal = 0.0;
            for (auto x : w)
                wlocal += x;
            double wbefore = 0.0, wtot = wlocal;
#ifdef USE_MPI
            MPI_Exscan(&wlocal, &wbefore, 1, MPI_DOUBLE, MPI_SUM, FML::communicator);
            if (FML::ThisTask == 0)
                wbefore = 0.0;
            MPI_Allreduce(MPI_IN_PLACE, &wtot, 1, MPI_DOUBLE, MPI_SUM, FML::communicator);
#endif

            // Each cell goes to the task whose equal share of the total weight contains its midpoint.
            // This is monotone along the curve so tasks get contiguous segments.
            SparseBuffers<Cell> sendcells;
            SparseBuffers<T> senddata;
            SparseBuffers<int8_t> sendflags;
            double wcum = wbefore;
            for (IndexInt i = 0; i < n; i++) {
                const double mid = wcum + 0.5 * w[i];
                wcum += w[i];
                int r = wtot > 0.0 ? int(mid / wtot * FML::NTasks) : 0;
                r = std::max(0, std::min(FML::NTasks - 1, r));
                sendcells[r].push_back(_cells[i]);
                senddata[r].push_back(_data[i]);
                sendflags[r].push_back(_flags[i]);
            }
            auto recvcells = alltoallv_sparse(sendcells);
            auto recvdata = alltoallv_sparse(senddata);
            auto recvflags = alltoallv_sparse(sendflags);

            _cells.clear();
            _data.clear();
            _flags.clear();
            // Maps are ordered by task, and tasks are ordered along the curve
            for (auto & [r, buf] : recvcells) {
                _cells.insert(_cells.end(), buf.begin(), buf.end());
                _data.insert(_data.end(), recvdata[r].begin(), recvdata[r].end());
                _flags.insert(_flags.end(), recvflags[r].begin(), recvflags[r].end());
            }

            update_domain_start();

            // Never cut the curve inside a sibling family, otherwise that family could not be coarsened
            // and the grid would depend on the number of tasks. If our first leaf is not the first child
            // of its parent we hand the leaves up to the end of the parent to the previous (non-empty)
            // task. Repeat until no task moves anything (a parent can span several small tasks).
            bool moved = true;
            while (moved) {
                const int me = FML::ThisTask;
                int prev = me - 1;
                while (prev >= 0 and _domain_start[prev] == _domain_start[prev + 1])
                    prev--;
                SparseBuffers<Cell> sc;
                SparseBuffers<T> sd;
                SparseBuffers<int8_t> sf;
                size_t nmove = 0;
                if (not _cells.empty() and prev >= 0) {
                    const Cell & first = _cells.front();
                    const key_type prange = key_range(first.level - 1);
                    if (first.level > 0 and (first.key & (prange - 1)) != 0) {
                        const key_type pend = (first.key & ~(prange - 1)) + prange;
                        while (nmove < _cells.size() and _cells[nmove].key < pend)
                            nmove++;
                        sc[prev].assign(_cells.begin(), _cells.begin() + nmove);
                        sd[prev].assign(_data.begin(), _data.begin() + nmove);
                        sf[prev].assign(_flags.begin(), _flags.begin() + nmove);
                        _cells.erase(_cells.begin(), _cells.begin() + nmove);
                        _data.erase(_data.begin(), _data.begin() + nmove);
                        _flags.erase(_flags.begin(), _flags.begin() + nmove);
                    }
                }
                auto rc = alltoallv_sparse(sc);
                auto rd = alltoallv_sparse(sd);
                auto rf = alltoallv_sparse(sf);
                // Received leaves come from tasks later on the curve, so they are appended
                for (auto & [r, buf] : rc) {
                    if (r == me)
                        continue;
                    _cells.insert(_cells.end(), buf.begin(), buf.end());
                    _data.insert(_data.end(), rd[r].begin(), rd[r].end());
                    _flags.insert(_flags.end(), rf[r].begin(), rf[r].end());
                }
                moved = AMRDETAIL::allreduce_or(nmove > 0);
                update_domain_start();
            }

            build_ghost_pattern();
            exchange_ghosts();
        }

        template <int NDIM, class T>
        template <class U>
        AdaptiveMPIGrid<NDIM, U> AdaptiveMPIGrid<NDIM, T>::truncated(int level) const {
            AdaptiveMPIGrid<NDIM, U> g;
            g._levelmin = std::min(_levelmin, level);
            g._levelmax = level;
            g._periodic = _periodic;
            g.check_levels();

            // Nothing finer than level: the same leaves and partition, so copy the ghost pattern too
            bool finer = false;
            for (auto & c : _cells)
                finer = finer or c.level > level;
            if (not AMRDETAIL::allreduce_or(finer)) {
                g._cells = _cells;
                g._domain_start = _domain_start;
                g._send_ranks = _send_ranks;
                g._send_indices = _send_indices;
                g._recv_ranks = _recv_ranks;
                g._recv_offset = _recv_offset;
                g._recv_count = _recv_count;
                g._ghost_cells = _ghost_cells;
                g._boundary_cells = _boundary_cells;
                g._interior_cells = _interior_cells;
                g._data.assign(g._cells.size(), U{});
                g._ghost_data.assign(g._ghost_cells.size(), U{});
                g._flags.assign(g._cells.size(), KEEP);
                return g;
            }

            // Round every cut down to a cell at level. A cut that is not on such a cell lies inside a refined box
            // at level (the leaf after the cut is finer than level), so the rounded cut is a leaf boundary.
            const key_type step = key_range(level);
            g._domain_start.resize(_domain_start.size());
            for (size_t r = 0; r < _domain_start.size(); r++)
                g._domain_start[r] = (_domain_start[r] / step) * step;
            g._domain_start.back() = KEY_END;

            // Truncate the local leaves and send each to its new owner. Leaves only move to later tasks (the tail
            // of our part of the curve), so the buffers received in task order are sorted. Several tasks can make
            // the same coarse leaf (it straddled a cut), so remove duplicates.
            SparseBuffers<Cell> send;
            for (auto & c : _cells) {
                const Cell t = c.level > level ? make_cell_from_key(c.key, level) : c;
                auto & buf = send[g.owner_of_key(t.key)];
                if (buf.empty() or buf.back().key != t.key)
                    buf.push_back(t);
            }
            auto recv = alltoallv_sparse(send);
            for (auto & [r, buf] : recv)
                for (auto & c : buf)
                    if (g._cells.empty() or g._cells.back().key != c.key)
                        g._cells.push_back(c);
            g._data.assign(g._cells.size(), U{});
            g._flags.assign(g._cells.size(), KEEP);
            g.build_ghost_pattern();
            return g;
        }

        //==========================================================================================
        // Diagnostics
        //==========================================================================================

        template <int NDIM, class T>
        bool AdaptiveMPIGrid<NDIM, T>::check_invariants(bool verbose) const {
            bool ok = true;
            auto fail = [&](const std::string & msg) {
                if (verbose and ok)
                    std::cout << "Task " << FML::ThisTask << " check_invariants failed: " << msg << "\n";
                ok = false;
            };
            const int me = FML::ThisTask;
            const IndexInt n = get_n_local();

            // Local cells: valid, sorted, contiguous, inside our part of the curve
            for (IndexInt i = 0; i < n; i++) {
                const Cell & c = _cells[i];
                Cell ref = make_cell(c.anchor, c.level);
                if (ref.key != c.key or ref.anchor != c.anchor)
                    fail("inconsistent key/anchor");
                if (c.level < _levelmin or c.level > _levelmax)
                    fail("level out of range");
                if (i > 0 and _cells[i - 1].key + key_range(_cells[i - 1].level) != c.key)
                    fail("local cells not contiguous along the curve");
                if (c.key < _domain_start[me] or c.key + key_range(c.level) > _domain_start[me + 1])
                    fail("cell outside own domain");
            }
            if (n > 0) {
                if (_cells.front().key != _domain_start[me])
                    fail("first cell does not start the domain");
                if (_cells.back().key + key_range(_cells.back().level) != _domain_start[me + 1])
                    fail("last cell does not end the domain");
            } else if (_domain_start[me] != _domain_start[me + 1]) {
                fail("empty task with nonzero domain");
            }

            // Ghosts: sorted, not owned by us
            for (IndexInt i = 0; i < get_n_ghost(); i++) {
                if (i > 0 and _ghost_cells[i - 1].key >= _ghost_cells[i].key)
                    fail("ghosts not sorted");
                if (owner_of_key(_ghost_cells[i].key) == me)
                    fail("ghost owned by self");
            }

            // Every leaf touching a local leaf is available and 2:1 balance holds. Interior leaves touch
            // only local leaves, boundary leaves touch at least one ghost.
            if (IndexInt(_boundary_cells.size() + _interior_cells.size()) != n)
                fail("boundary + interior != local cells");
            std::vector<char> is_boundary(n, 0);
            for (auto i : _boundary_cells)
                is_boundary[i] = 1;
            for (IndexInt i = 0; i < n; i++) {
                bool touches_ghost = false;
                for_each_touching_box(_cells[i], [&](const Coord & x, int) {
                    CellRef r = find_leaf(x);
                    if (not r) {
                        fail("missing neighbour (local or ghost)");
                        return;
                    }
                    touches_ghost = touches_ghost or r.is_ghost();
                    if (std::abs(int(get_cell(r).level) - int(_cells[i].level)) > 1)
                        fail("2:1 balance violated");
                });
                if (touches_ghost != bool(is_boundary[i]))
                    fail("boundary/interior classification wrong");
            }

            // Global coverage: total volume in units of the finest cell must be KEY_END
            __uint128_t vol = 0;
            for (auto & c : _cells)
                vol += key_range(c.level);
            uint64_t vol_lo = uint64_t(vol);
#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &vol_lo, 1, MPI_UINT64_T, MPI_SUM, FML::communicator);
#endif
            if (vol_lo != KEY_END)
                fail("leaves do not cover the domain");

            return not AMRDETAIL::allreduce_or(not ok);
        }

        template <int NDIM, class T>
        uint64_t AdaptiveMPIGrid<NDIM, T>::global_hash() const {
            uint64_t h = 0;
            for (auto & c : _cells)
                h += AMRDETAIL::splitmix64(c.key ^ (uint64_t(c.level) << 1) ^ AMRDETAIL::splitmix64(c.level));
#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &h, 1, MPI_UINT64_T, MPI_SUM, FML::communicator);
#endif
            return h;
        }

        template <int NDIM, class T>
        void AdaptiveMPIGrid<NDIM, T>::write_ascii(std::string prefix) const {
            std::ofstream fp(prefix + "." + std::to_string(FML::ThisTask));
            for (auto & c : _cells) {
                auto x = get_cell_corner(c);
                for (int d = 0; d < NDIM; d++)
                    fp << x[d] << " ";
                fp << get_cell_size(c.level) << " " << int(c.level) << " " << FML::ThisTask << "\n";
            }
        }

        template <int NDIM, class T>
        void AdaptiveMPIGrid<NDIM, T>::info() const {
            std::vector<IndexInt> count(_levelmax + 1, 0);
            for (auto & c : _cells)
                count[c.level]++;
            IndexInt nmin = get_n_local(), nmax = get_n_local(), nghost = get_n_ghost();
            IndexInt nboundary = IndexInt(_boundary_cells.size());
#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, count.data(), int(count.size()), MPI_LONG_LONG, MPI_SUM, FML::communicator);
            MPI_Allreduce(MPI_IN_PLACE, &nmin, 1, MPI_LONG_LONG, MPI_MIN, FML::communicator);
            MPI_Allreduce(MPI_IN_PLACE, &nmax, 1, MPI_LONG_LONG, MPI_MAX, FML::communicator);
            MPI_Allreduce(MPI_IN_PLACE, &nghost, 1, MPI_LONG_LONG, MPI_SUM, FML::communicator);
            MPI_Allreduce(MPI_IN_PLACE, &nboundary, 1, MPI_LONG_LONG, MPI_SUM, FML::communicator);
#endif
            if (FML::ThisTask == 0) {
                IndexInt ntot = 0;
                std::cout << "AdaptiveMPIGrid<" << NDIM << "> levels:";
                for (int l = _levelmin; l <= _levelmax; l++) {
                    std::cout << " " << l << ":" << count[l];
                    ntot += count[l];
                }
                std::cout << " | total " << ntot << " | per task min/max " << nmin << "/" << nmax << " | ghosts "
                          << nghost << " | boundary cells " << nboundary << "\n";
            }
        }

        //==========================================================================================
        ///
        /// Moves cell values between a grid ("fine") and a coarsening of it ("coarse"), e.g. the grid and
        /// grid.truncated(level): every leaf of the fine grid must lie inside a leaf of the coarse grid. The two
        /// grids can be split differently over the tasks; the communication pattern is built once.
        ///
        ///   AdaptiveMPIGridTransfer<NDIM> transfer(fine, coarse);
        ///   transfer.restrict_average(fine_values, coarse_values);  // volume average over each coarse leaf
        ///   transfer.gather_from_coarse(coarse_values, fine_values); // every fine leaf gets its coarse leaf's value
        ///
        /// The vectors are indexed by the local leaves of the respective grid. Both calls are collective.
        /// The pattern is invalid once either grid is adapted or load balanced.
        ///
        //==========================================================================================
        template <int NDIM>
        class AdaptiveMPIGridTransfer {
          private:
            IndexInt _nfine{0};
            IndexInt _ncoarse{0};

            // Pairs where both leaves are on this task: fine index, coarse index, volume ratio
            std::vector<IndexInt> _local_fine;
            std::vector<IndexInt> _local_coarse;
            std::vector<double> _local_weight;

            // Fine side: the fine leaves whose coarse leaf is on another task (per task, in the order sent)
            std::vector<int> _send_ranks;
            std::vector<std::vector<IndexInt>> _send_fine;

            // Coarse side: the coarse leaf (and volume ratio) of every fine leaf we receive (per task, same order)
            std::vector<int> _recv_ranks;
            std::vector<std::vector<IndexInt>> _recv_coarse;
            std::vector<std::vector<double>> _recv_weight;

            // Point-to-point exchange along the pattern: fine -> coarse (forward) or coarse -> fine (backward)
            template <class U>
            void exchange(bool forward,
                          const std::vector<std::vector<U>> & sendbufs,
                          std::vector<std::vector<U>> & recvbufs) const;

          public:
            AdaptiveMPIGridTransfer() = default;

            template <class TF, class TC>
            AdaptiveMPIGridTransfer(const AdaptiveMPIGrid<NDIM, TF> & fine, const AdaptiveMPIGrid<NDIM, TC> & coarse);

            IndexInt get_n_fine() const { return _nfine; }
            IndexInt get_n_coarse() const { return _ncoarse; }

            /// coarse[j] = volume average of the fine leaves inside coarse leaf j
            void restrict_average(const std::vector<double> & fine, std::vector<double> & coarse) const;

            /// fine[i] = value of the coarse leaf containing fine leaf i (U trivially copyable)
            template <class U>
            void gather_from_coarse(const std::vector<U> & coarse, std::vector<U> & fine) const;

            //======================================================================
            // Transfers to a subset of the leaves (e.g. the unknowns of a partial multigrid solve). The local
            // loops only visit the given pairs; the exchange with other tasks is the full one (it only involves
            // leaves at the boundaries of the partitions). Collective like the full transfers
            //======================================================================
            /// The local pairs (indices into the list of fine/coarse pairs on this task) whose coarse leaf is
            /// selected, and those whose fine leaf is selected
            std::vector<IndexInt> local_pairs_with_coarse(const std::vector<char> & coarse_selected) const;
            std::vector<IndexInt> local_pairs_with_fine(const std::vector<char> & fine_selected) const;
            /// restrict_average into the selected coarse leaves only (pairs = local_pairs_with_coarse(selected));
            /// the other coarse values are left as they are (coarse must have the size of the coarse grid)
            void restrict_average_selected(const std::vector<double> & fine,
                                           std::vector<double> & coarse,
                                           const std::vector<char> & coarse_selected,
                                           const std::vector<IndexInt> & pairs) const;
            /// gather_from_coarse into the selected fine leaves only (pairs = local_pairs_with_fine(selected))
            template <class U>
            void gather_from_coarse_selected(const std::vector<U> & coarse,
                                             std::vector<U> & fine,
                                             const std::vector<char> & fine_selected,
                                             const std::vector<IndexInt> & pairs) const;
        };

        template <int NDIM>
        template <class TF, class TC>
        AdaptiveMPIGridTransfer<NDIM>::AdaptiveMPIGridTransfer(const AdaptiveMPIGrid<NDIM, TF> & fine,
                                                               const AdaptiveMPIGrid<NDIM, TC> & coarse)
            : _nfine(fine.get_n_local()), _ncoarse(coarse.get_n_local()) {
            using Grid = AdaptiveMPIGrid<NDIM, TF>;
            struct KeyLevel {
                uint64_t key;
                int level;
            };
            const int me = FML::ThisTask;

            // Every fine leaf goes to the owner of its first key in the coarse partition
            typename Grid::template SparseBuffers<KeyLevel> sendkeys;
            std::map<int, std::vector<IndexInt>> sendfine;
            for (IndexInt i = 0; i < _nfine; i++) {
                const auto & c = fine.get_cell(i);
                const int r = coarse.owner_of_key(c.key);
                if (r == me) {
                    auto ref = coarse.find_leaf_by_key(c.key);
                    assert_mpi(ref.is_local(), "AdaptiveMPIGridTransfer: coarse leaf not found");
                    const int lc = coarse.get_cell(ref.index).level;
                    assert_mpi(lc <= c.level, "AdaptiveMPIGridTransfer: the coarse grid is finer than the fine grid");
                    _local_fine.push_back(i);
                    _local_coarse.push_back(ref.index);
                    _local_weight.push_back(std::ldexp(1.0, -NDIM * (c.level - lc)));
                } else {
                    sendkeys[r].push_back({c.key, int(c.level)});
                    sendfine[r].push_back(i);
                }
            }
            for (auto & [r, list] : sendfine) {
                _send_ranks.push_back(r);
                _send_fine.push_back(std::move(list));
            }

            auto recvkeys = fine.alltoallv_sparse(sendkeys);
            for (auto & [r, buf] : recvkeys) {
                if (r == me)
                    continue;
                _recv_ranks.push_back(r);
                _recv_coarse.emplace_back();
                _recv_weight.emplace_back();
                for (auto & kl : buf) {
                    auto ref = coarse.find_leaf_by_key(kl.key);
                    assert_mpi(ref.is_local(), "AdaptiveMPIGridTransfer: coarse leaf not found on its owner");
                    const int lc = coarse.get_cell(ref.index).level;
                    assert_mpi(lc <= kl.level, "AdaptiveMPIGridTransfer: the coarse grid is finer than the fine grid");
                    _recv_coarse.back().push_back(ref.index);
                    _recv_weight.back().push_back(std::ldexp(1.0, -NDIM * (kl.level - lc)));
                }
            }
        }

        template <int NDIM>
        template <class U>
        void AdaptiveMPIGridTransfer<NDIM>::exchange(bool forward,
                                                     const std::vector<std::vector<U>> & sendbufs,
                                                     std::vector<std::vector<U>> & recvbufs) const {
            static_assert(std::is_trivially_copyable<U>::value, "AdaptiveMPIGridTransfer: U must be trivially copyable");
            const auto & sranks = forward ? _send_ranks : _recv_ranks;
            const auto & rranks = forward ? _recv_ranks : _send_ranks;
            recvbufs.resize(rranks.size());
            for (size_t j = 0; j < rranks.size(); j++)
                recvbufs[j].resize(forward ? _recv_coarse[j].size() : _send_fine[j].size());
#ifdef USE_MPI
            const int tag = 32100 + (AMRDETAIL::transfer_tag_counter++ % 600);
            std::vector<MPI_Request> req(sranks.size() + rranks.size());
            int nreq = 0;
            for (size_t j = 0; j < rranks.size(); j++) {
                assert_mpi(recvbufs[j].size() * sizeof(U) <= size_t(INT_MAX), "AdaptiveMPIGridTransfer: message too large");
                MPI_Irecv(recvbufs[j].data(), int(recvbufs[j].size() * sizeof(U)), MPI_BYTE, rranks[j], tag,
                          FML::communicator, &req[nreq++]);
            }
            for (size_t j = 0; j < sranks.size(); j++) {
                assert_mpi(sendbufs[j].size() * sizeof(U) <= size_t(INT_MAX), "AdaptiveMPIGridTransfer: message too large");
                MPI_Isend(sendbufs[j].data(), int(sendbufs[j].size() * sizeof(U)), MPI_BYTE, sranks[j], tag,
                          FML::communicator, &req[nreq++]);
            }
            MPI_Waitall(nreq, req.data(), MPI_STATUSES_IGNORE);
#else
            (void)sendbufs;
            (void)sranks;
#endif
        }

        template <int NDIM>
        void AdaptiveMPIGridTransfer<NDIM>::restrict_average(const std::vector<double> & fine,
                                                             std::vector<double> & coarse) const {
            assert_mpi(IndexInt(fine.size()) == _nfine, "restrict_average: fine vector has wrong size");
            std::vector<std::vector<double>> sendbufs(_send_ranks.size()), recvbufs;
            for (size_t j = 0; j < _send_ranks.size(); j++) {
                sendbufs[j].reserve(_send_fine[j].size());
                for (auto i : _send_fine[j])
                    sendbufs[j].push_back(fine[i]);
            }
            exchange(true, sendbufs, recvbufs);
            coarse.assign(_ncoarse, 0.0);
            for (size_t k = 0; k < _local_fine.size(); k++)
                coarse[_local_coarse[k]] += _local_weight[k] * fine[_local_fine[k]];
            for (size_t j = 0; j < _recv_ranks.size(); j++)
                for (size_t k = 0; k < _recv_coarse[j].size(); k++)
                    coarse[_recv_coarse[j][k]] += _recv_weight[j][k] * recvbufs[j][k];
        }

        template <int NDIM>
        template <class U>
        void AdaptiveMPIGridTransfer<NDIM>::gather_from_coarse(const std::vector<U> & coarse, std::vector<U> & fine) const {
            assert_mpi(IndexInt(coarse.size()) == _ncoarse, "gather_from_coarse: coarse vector has wrong size");
            std::vector<std::vector<U>> sendbufs(_recv_ranks.size()), recvbufs;
            for (size_t j = 0; j < _recv_ranks.size(); j++) {
                sendbufs[j].reserve(_recv_coarse[j].size());
                for (auto i : _recv_coarse[j])
                    sendbufs[j].push_back(coarse[i]);
            }
            exchange(false, sendbufs, recvbufs);
            fine.resize(_nfine);
            for (size_t k = 0; k < _local_fine.size(); k++)
                fine[_local_fine[k]] = coarse[_local_coarse[k]];
            for (size_t j = 0; j < _send_ranks.size(); j++)
                for (size_t k = 0; k < _send_fine[j].size(); k++)
                    fine[_send_fine[j][k]] = recvbufs[j][k];
        }

        template <int NDIM>
        std::vector<IndexInt> AdaptiveMPIGridTransfer<NDIM>::local_pairs_with_coarse(const std::vector<char> & sel) const {
            std::vector<IndexInt> pairs;
            for (size_t k = 0; k < _local_fine.size(); k++)
                if (sel[_local_coarse[k]])
                    pairs.push_back(IndexInt(k));
            return pairs;
        }

        template <int NDIM>
        std::vector<IndexInt> AdaptiveMPIGridTransfer<NDIM>::local_pairs_with_fine(const std::vector<char> & sel) const {
            std::vector<IndexInt> pairs;
            for (size_t k = 0; k < _local_fine.size(); k++)
                if (sel[_local_fine[k]])
                    pairs.push_back(IndexInt(k));
            return pairs;
        }

        template <int NDIM>
        void AdaptiveMPIGridTransfer<NDIM>::restrict_average_selected(const std::vector<double> & fine,
                                                                      std::vector<double> & coarse,
                                                                      const std::vector<char> & sel,
                                                                      const std::vector<IndexInt> & pairs) const {
            assert_mpi(IndexInt(fine.size()) == _nfine and IndexInt(coarse.size()) == _ncoarse,
                       "restrict_average_selected: vectors have the wrong size");
            std::vector<std::vector<double>> sendbufs(_send_ranks.size()), recvbufs;
            for (size_t j = 0; j < _send_ranks.size(); j++) {
                sendbufs[j].reserve(_send_fine[j].size());
                for (auto i : _send_fine[j])
                    sendbufs[j].push_back(fine[i]);
            }
            exchange(true, sendbufs, recvbufs);
            for (auto k : pairs)
                coarse[_local_coarse[k]] = 0.0;
            for (size_t j = 0; j < _recv_ranks.size(); j++)
                for (auto c : _recv_coarse[j])
                    if (sel[c])
                        coarse[c] = 0.0;
            for (auto k : pairs)
                coarse[_local_coarse[k]] += _local_weight[k] * fine[_local_fine[k]];
            for (size_t j = 0; j < _recv_ranks.size(); j++)
                for (size_t k = 0; k < _recv_coarse[j].size(); k++)
                    if (sel[_recv_coarse[j][k]])
                        coarse[_recv_coarse[j][k]] += _recv_weight[j][k] * recvbufs[j][k];
        }

        template <int NDIM>
        template <class U>
        void AdaptiveMPIGridTransfer<NDIM>::gather_from_coarse_selected(const std::vector<U> & coarse,
                                                                       std::vector<U> & fine,
                                                                       const std::vector<char> & sel,
                                                                       const std::vector<IndexInt> & pairs) const {
            assert_mpi(IndexInt(coarse.size()) == _ncoarse, "gather_from_coarse_selected: coarse vector has wrong size");
            std::vector<std::vector<U>> sendbufs(_recv_ranks.size()), recvbufs;
            for (size_t j = 0; j < _recv_ranks.size(); j++) {
                sendbufs[j].reserve(_recv_coarse[j].size());
                for (auto i : _recv_coarse[j])
                    sendbufs[j].push_back(coarse[i]);
            }
            exchange(false, sendbufs, recvbufs);
            fine.resize(_nfine);
            for (auto k : pairs)
                fine[_local_fine[k]] = coarse[_local_coarse[k]];
            for (size_t j = 0; j < _send_ranks.size(); j++)
                for (size_t k = 0; k < _send_fine[j].size(); k++)
                    if (sel[_send_fine[j][k]])
                        fine[_send_fine[j][k]] = recvbufs[j][k];
        }

    } // namespace GRID
} // namespace FML

#endif
