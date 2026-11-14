#ifndef MPIGRID_HEADER
#define MPIGRID_HEADER

#include <array>
#include <climits>
#include <complex>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <type_traits>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef USE_OMP
#include <omp.h>
#endif

#include <FML/Global/Global.h>

namespace FML {
    namespace GRID {

        using IndexInt = long long int;

// The index helpers are in the inner loops of the solvers: inline them (their rare paths are separate functions)
#if defined(__GNUC__)
#define FML_GRID_INLINE [[gnu::always_inline]] inline
#else
#define FML_GRID_INLINE inline
#endif

        // The absolute value of the type, must be implemented for non-standard types
        template <class T>
        double AbsoluteValue(T & x) {
            return std::abs(x);
        }

        //==========================================================================================
        /// A simple multidimensional grid-class for any type that works over MPI
        ///
        /// Bounds-check for array lookups: BOUNDSCHECK
        ///
        /// The grid is split over the tasks along x (slabs, the fiducial) or, with a FML::Domain, along x and y
        /// (pencils). Every index is an index in the local main grid unless specified otherwise: the main cells are
        /// index = ((ix * NLocal_y + iy) * N + iz) in [0, NtotLocal) for the local coordinates (ix, iy, iz) (NLocal_y
        /// = N for slabs), which is the cell (ix + xStart, iy + yStart, iz) in the global grid
        ///
        /// Every coord is a local coordinate unless specified otherwise
        ///
        /// The extra cells (ghosts) of the neighboring tasks: n_extra_slices_left (right) layers along each axis
        /// that is split over the tasks. The x-slices are before (after) the main grid in memory, so index - k
        /// stride_x is the cell k slices to the left also into the extra slices. The y-layers (pencils) are stored
        /// after the extra x-slices and include the extra x-slices (the corners) so their local index is >=
        /// NtotLocal: use index_from_coord, get_neighbor_gridindex or get_shifted_index to find them. An axis that
        /// is not split (the whole axis is on the task) wraps periodically in the local grid
        ///
        /// External methods we rely on:
        ///
        ///   using IndexInt = long long int;
        ///
        ///   double AbsoluteValue(T &x);
        ///
        ///   T power(T base, int exponent);
        ///
        ///   assert_mpi(Expr, Msg);
        ///
        //==========================================================================================

        template <int NDIM, class T>
        class MPIGrid {
            static_assert(std::is_trivially_copyable_v<T>, "[MPIGrid] T is sent over MPI as bytes: it must be trivially copyable");

          private:
            bool _periodic{true};         // Is the box periodic or not
            int _N{0};                    // Gridcells per dimension in the global grid
            int _NLocal{0};               // Gridcells in the x-dimension for current task
            int _xStartLocal{0};          // The ix-index the local task starts at
            int _LeftTask{0};             // The id of the task on the left
            int _RightTask{0};            // The id of the task on the right
            int _n_extra_slices_left{0};  // Extra x-slices to the left
            int _n_extra_slices_right{0}; // Extra x-slices to the right

            // The local part of every axis, the extra layers along it and the index strides of the main grid
            std::array<int, NDIM> _start{};
            std::array<int, NDIM> _nloc{};
            std::array<int, NDIM> _nextra_left{};
            std::array<int, NDIM> _nextra_right{};
            std::array<IndexInt, NDIM> _stride{};
            std::array<IndexInt, NDIM> _wrap{}; // N stride along a periodic axis held whole by the task (else 0)
            bool _ysplit{false};                // Is the y-axis split over the tasks (extra y-layers)?
            int _DownTask{0}; // The task below (y) with pencils
            int _UpTask{0};   // The task above (y) with pencils
            std::optional<FML::Domain> _domain{};

            IndexInt _Ntot{0};           // How many cells in main grid in total
            IndexInt _NtotLocalLeft{0};  // Total cells in extra left slices
            IndexInt _NtotLocalRight{0}; // Total cells in extra right slices
            IndexInt _NtotLocal{0};      // How many cells in main grid on local task
            IndexInt _NtotLocalAlloc{0}; // How many cells we have allocated locally (includes extra cells)
            IndexInt _NperSlice{0};      // How many cells per x-slice
            IndexInt _NxExt{0};          // The number of x-slices including the extra slices
            IndexInt _YExtraStart{0};    // Where the extra y-layers start in the array
            IndexInt _NperYLayer{0};     // Cells in an extra y-layer (all x-slices including the extra ones)

            // std::vector<T> _y;           // The grid data
            Vector<T> _y{}; // The grid data

            // Helper functions for bounds-checking
            void assert_index(IndexInt index) const;
            void assert_coord(const std::array<int, NDIM> & coord) const;

            // Set the strides and sizes from _start, _nloc and the extra layers and allocate
            void allocate();

            // The index and the coordinate of a cell in the extra y-layers (pencils)
            IndexInt index_from_coord_in_ylayers(const std::array<int, NDIM> & coord) const;
            std::array<int, NDIM> coord_from_index_in_ylayers(IndexInt index) const;

            // Send a block of cells to dest and receive one from source (a copy if dest is this task)
            void sendrecv_block(T * send, T * recv, IndexInt n, int dest, int source);

#ifdef USE_MPI
            // MPI_Sendrecv of a byte buffer of any size (split into chunks whose count fits in an int)
            static void sendrecv_bytes(char * sendbuf, char * recvbuf, size_t bytes, int dest, int source);
#endif

          public:
            // Constructors
            MPIGrid() = default;
            MPIGrid(int N, bool periodic, int n_extra_slices_left = 0, int n_extra_slices_right = 0);

            /// A grid on the given domain (Slab or Pencil): the edges of the domain must lie on the cells and every
            /// task must have cells. With a Pencil domain the grid is split along x and y (NDIM >= 2)
            MPIGrid(int N,
                    bool periodic,
                    const FML::Domain & domain,
                    int n_extra_slices_left = 0,
                    int n_extra_slices_right = 0);

            /// Does the domain split N cells per dim into whole cells with cells on every task?
            static bool fits_domain(int N, const FML::Domain & domain);

            // Get a pointer to the start of the main grid
            T * get_y();
            const T * get_y() const;

            // Get a reference to the cell at a given index
            T & get_y(IndexInt index);

            // Allow syntax grid[i] to get/set cells with a given index
            T & operator[](IndexInt index);
            const T & operator[](IndexInt index) const;

            // Assign value in the grid by index
            void set_y(IndexInt index, const T & value);

            // Assign value in the grid by coordinate
            void set_y(std::array<int, NDIM> & coord, const T & value);

            // Assign the whole grid from a function
            void set_y(const std::function<T(std::array<double, NDIM> &)> & func);

            // Grid-index -> Local coordinate list
            // e.g. index = ix1_local * N^2 + ix2 * N + ix3 -> (ix1_local, ix2, ix3)
            FML_GRID_INLINE std::array<int, NDIM> coord_from_index(IndexInt index) const;

            // Local coordinates -> Grid-index (also the extra cells)
            // e.g. index = ix1_local * N^2 + ix2 * N + ix3
            FML_GRID_INLINE IndexInt index_from_coord(const std::array<int, NDIM> & coord) const;

            // From local index to global coordinates
            std::array<int, NDIM> globalcoord_from_index(IndexInt index) const;

            // From global coordinates to local index
            FML_GRID_INLINE IndexInt index_from_globalcoord(const std::array<int, NDIM> & globalcoord) const;

            // The closest 2NDIM+1 cells to a given cell (including the cell itself at 0). NB: for a non-periodic
            // grid the neighbours outside the box (of the cells on the global boundary) are not meaningful (they can
            // be cells in another row): treat the boundary cells separately (as MultiGridSolver does with Dirichlet
            // boundary cells)
            FML_GRID_INLINE std::array<IndexInt, 2 * NDIM + 1> get_neighbor_gridindex(IndexInt index) const;

            /// The index of the cell offset cells along axis from the cell (index, coord) (coord =
            /// coord_from_index(index)): periodic in the local grid along an axis that is not split, in the extra
            /// cells otherwise. Shift along the axes in increasing order to reach the corners (x first)
            IndexInt get_shifted_index(IndexInt index, const std::array<int, NDIM> & coord, int axis, int offset) const;

            // Gradient at a given cell (simple 2-point symmetric stensil). Not meaningful on the global boundary of a
            // non-periodic grid (see get_neighbor_gridindex)
            std::array<T, NDIM> get_gradient(IndexInt index) const;

            // Get some info about the grid
            IndexInt get_Ntot();
            IndexInt get_NtotLocal() const;
            int get_N() const;
            int get_NLocal();
            int get_xStartLocal() const;
            int get_n_extra_slices_left() const;
            int get_n_extra_slices_right() const;
            /// Is the box periodic?
            bool get_periodic() const { return _periodic; }
            /// The index step between neighbouring cells along an axis (in the main grid and the extra x-slices)
            IndexInt get_stride(int axis) const { return _stride[axis]; }
            /// The global cells along an axis on this task: {first, count} (as FFTWGrid and PencilGrid)
            std::array<int, 2> get_local_range(int axis) const { return {_start[axis], _nloc[axis]}; }
            /// The domain the grid was made on (none for the fiducial slabs)
            const std::optional<FML::Domain> & get_domain() const { return _domain; }

            // Returns the position of the cell in the global grid
            std::array<double, NDIM> get_pos(IndexInt index);
            std::array<double, NDIM> get_pos(const std::array<int, NDIM> & coord);

            // Distance from a point to the cell. Fiducial choice is center of the box
            double get_radial_distance(IndexInt index, std::array<double, NDIM> & point);

            // The rms of the main grid sqrt(Sum y^2/N)
            double norm();

            // Communicate all the extra cells. With more extra layers than local layers the layers are forwarded
            // from tasks further away. With one task along an axis (periodic) the extra cells are copies of the
            // periodic images of our own cells. For a non-periodic grid the extra cells outside the box are not
            // touched. Along x first and then along y (the y-layers include the extra x-slices: the corners)
            void communicate_boundaries();

            // Send slice ix (local index, can be an extra slice) of the grid to the left or right task and receive
            // the corresponding slice from the other neighbor in recv_slice (resized to one slice).
            // With one task the slice itself is returned
            void send_slice_left(int ix, std::vector<T> & recv_slice);
            void send_slice_right(int ix, std::vector<T> & recv_slice);

            // Free up all memory and reset all variables
            void free();

            // Arithmetic operators
            MPIGrid<NDIM, T> & operator+=(const MPIGrid<NDIM, T> & rhs);
            MPIGrid<NDIM, T> & operator-=(const MPIGrid<NDIM, T> & rhs);
            MPIGrid<NDIM, T> & operator*=(const MPIGrid<NDIM, T> & rhs);
            MPIGrid<NDIM, T> & operator/=(const MPIGrid<NDIM, T> & rhs);

            template <class U>
            MPIGrid<NDIM, T> & operator+=(const U & rhs);
            template <class U>
            MPIGrid<NDIM, T> & operator-=(const U & rhs);
            template <class U>
            MPIGrid<NDIM, T> & operator*=(const U & rhs);
            template <class U>
            MPIGrid<NDIM, T> & operator/=(const U & rhs);

            // For memory monitoring
            void add_memory_label(std::string label);

            // Show some info
            void info();
        };

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::info() {
            if (FML::ThisTask == 0) {
                std::cout << "\n";
                std::cout << "#=====================================================\n";
                std::cout << "#\n";
                std::cout << "#            .___        _____          \n";
                std::cout << "#            |   | _____/ ____\\____     \n";
                std::cout << "#            |   |/    \\   __\\/  _ \\    \n";
                std::cout << "#            |   |   |  \\  | (  <_> )   \n";
                std::cout << "#            |___|___|  /__|  \\____/    \n";
                std::cout << "#                     \\/                \n";
                std::cout << "#\n";
                std::cout << "# Info about MPIGrid NDIM [" << NDIM << "] Size of cells [" << sizeof(T) << "] bytes\n";
                std::cout << "# Periodic?            : " << std::boolalpha << _periodic << "\n";
                std::cout << "# N                    : " << _N << "\n";
                std::cout << "# NLocal               : " << _NLocal << "\n";
                std::cout << "# n_extra_slices_left  : " << _n_extra_slices_left << "\n";
                std::cout << "# n_extra_slices_right : " << _n_extra_slices_right << "\n";
                std::cout << "# Cells allocated      : " << _y.size() << " per task\n";
                std::cout << "# Memory allocated     : " << _y.size() * sizeof(T) / 1e6 << " MB per task\n";
                std::cout << "#\n";
                std::cout << "#=====================================================\n";
                std::cout << "\n";
            }
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::add_memory_label([[maybe_unused]] std::string label) {
#ifdef MEMORY_LOGGING
            FML::MemoryLog::get()->add_label(_y.data(), _y.capacity() * sizeof(T), label);
#endif
        }

#define OPS(OP)                                                                                                        \
    template <int NDIM, class T>                                                                                       \
    auto MPIGrid<NDIM, T>::operator OP(const MPIGrid<NDIM, T> & rhs)->MPIGrid<NDIM, T> & {                             \
        assert_mpi(_NtotLocal == rhs._NtotLocal, "[MPIGrid::operator" #OP "] Grids have different sizes\n");           \
        for (IndexInt i = 0; i < _NtotLocal; i++) {                                                                    \
            this->_y[_NtotLocalLeft + i] OP rhs._y[rhs._NtotLocalLeft + i];                                            \
        }                                                                                                              \
        return *this;                                                                                                  \
    }
        OPS(+=);
        OPS(-=);
        OPS(*=);
        OPS(/=)
#undef OPS

#define OPS(OP)                                                                                                        \
    template <int NDIM, class T>                                                                                       \
    template <class U>                                                                                                 \
    auto MPIGrid<NDIM, T>::operator OP(const U & rhs)->MPIGrid<NDIM, T> & {                                            \
        for (IndexInt i = 0; i < _NtotLocal; i++)                                                                      \
            this->_y[_NtotLocalLeft + i] OP rhs;                                                                       \
        return *this;                                                                                                  \
    }
        OPS(+=);
        OPS(-=);
        OPS(*=);
        OPS(/=)
#undef OPS

#define OPS(OP)                                                                                                        \
    template <int NDIM, class T>                                                                                       \
    auto operator OP(MPIGrid<NDIM, T> lhs, const MPIGrid<NDIM, T> & rhs)->MPIGrid<NDIM, T> {                           \
        lhs OP## = rhs;                                                                                                \
        return lhs;                                                                                                    \
    }
        OPS(+);
        OPS(-);
        OPS(*);
        OPS(/)
#undef OPS

#define OPS(OP)                                                                                                        \
    template <int NDIM, class T>                                                                                       \
    auto operator OP(MPIGrid<NDIM, T> lhs, const T & rhs)->MPIGrid<NDIM, T> {                                          \
        lhs OP## = rhs;                                                                                                \
        return lhs;                                                                                                    \
    }
        OPS(+);
        OPS(-);
        OPS(*);
        OPS(/)
#undef OPS

        // Integer log2 i.e. floor(log2(n))
        inline int intlog2(int n) {
            assert_mpi(n > 0, "[intlog2] Can't take a log of argument <= 0\n");
            if (n == 1)
                return 0;
            int res = 0;
            while (n > 1) {
                n /= 2;
                res++;
            }
            return res;
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::allocate() {
            // The strides of the main grid (the last axis is contiguous)
            _stride[NDIM - 1] = 1;
            for (int idim = NDIM - 2; idim >= 0; idim--)
                _stride[idim] = _stride[idim + 1] * _nloc[idim + 1];
            _NLocal = _nloc[0];
            _xStartLocal = _start[0];
            _n_extra_slices_left = _nextra_left[0];
            _n_extra_slices_right = _nextra_right[0];
            _NperSlice = _stride[0];
            _NtotLocalLeft = _NperSlice * _nextra_left[0];
            _NtotLocalRight = _NperSlice * _nextra_right[0];
            _NtotLocal = _NperSlice * _nloc[0];
            _Ntot = FML::power(IndexInt(_N), NDIM);
            _NxExt = _nextra_left[0] + _nloc[0] + _nextra_right[0];
            _YExtraStart = _NtotLocalLeft + _NtotLocal + _NtotLocalRight;
            _NperYLayer = NDIM > 1 ? _NxExt * _stride[NDIM > 1 ? 1 : 0] : 0;
            for (int idim = 0; idim < NDIM; idim++)
                _wrap[idim] = (_periodic and _nloc[idim] == _N) ? IndexInt(_N) * _stride[idim] : 0;
            _ysplit = NDIM > 1 and _nloc[NDIM > 1 ? 1 : 0] < _N;
            const IndexInt nylayers = NDIM > 1 ? _nextra_left[NDIM > 1 ? 1 : 0] + _nextra_right[NDIM > 1 ? 1 : 0] : 0;
            _NtotLocalAlloc = _YExtraStart + nylayers * _NperYLayer;
            _y.resize(_NtotLocalAlloc);
            add_memory_label("MPIGrid");
        }

        // Constructor with initial value
        template <int NDIM, class T>
        MPIGrid<NDIM, T>::MPIGrid(int N, bool periodic, int n_extra_slices_left, int n_extra_slices_right) {
            assert_mpi(N > 0, "[MPIGrid] We need Ngrid > 0\n");
            assert_mpi(n_extra_slices_left >= 0 and n_extra_slices_right >= 0,
                       "[MPIGrid] Number of extra slices cannot be negative\n");
            if (N % FML::NTasks != 0 and FML::ThisTask == 0)
                std::cout << "[MPIGrid] Warning: FML::NTasks should divide N to be compatible with other MPI methods\n";

            // Compute slices for task
            std::vector<int> slices_per_task(FML::NTasks, N / FML::NTasks);
            int nmore = N % FML::NTasks;
            int sumslicesbeforethistask = 0;
            for (int task = 0; task < FML::NTasks; task++) {
                if (task < nmore)
                    slices_per_task[task] += 1;

                if (task < FML::ThisTask)
                    sumslicesbeforethistask += slices_per_task[task];
            }

            _periodic = periodic;
            _N = N;
            _start.fill(0);
            _nloc.fill(N);
            _nextra_left.fill(0);
            _nextra_right.fill(0);
            _start[0] = sumslicesbeforethistask;
            _nloc[0] = slices_per_task[FML::ThisTask];
            _nextra_left[0] = n_extra_slices_left;
            _nextra_right[0] = n_extra_slices_right;
            _RightTask = (FML::ThisTask + 1) % FML::NTasks;
            _LeftTask = (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks;
            _DownTask = _UpTask = FML::ThisTask;

            if (_nloc[0] == 0) {
                // Tasks with 0 cells does nothing
                _RightTask = _LeftTask = FML::ThisTask;
                _nextra_left[0] = _nextra_right[0] = 0;
            } else {
                // Find right CPU with cells
                while (slices_per_task[_RightTask] == 0) {
                    _RightTask = (_RightTask + 1) % FML::NTasks;
                }
                // Find left CPU with cells
                while (slices_per_task[_LeftTask] == 0) {
                    _LeftTask = (_LeftTask - 1 + FML::NTasks) % FML::NTasks;
                }

                // This is maybe something we should have... but fails with multigrid
                // assert_mpi(_n_extra_slices_right <= slices_per_task[_RightTask], "");
                // assert_mpi(_n_extra_slices_left  <= slices_per_task[_LeftTask], "");
            }

            // Allocate memory
            allocate();

            // Show some info
#ifdef DEBUG
            if (FML::ThisTask == 0) {
                std::cout << "\n=======================\nCreating Grid:\n";
                std::cout << "FML::ThisTask: " << FML::ThisTask << " NLocal: " << _NLocal
                          << " xStartLocal: " << _xStartLocal << " LeftTask: " << _LeftTask
                          << " RightTask: " << _RightTask << "\n"
                          << std::flush;
            }
#endif
        }

        namespace detail {
            // The cell of a domain edge (which must lie on a cell): -1 if it does not
            inline int cell_of_edge(double edge, int N) {
                const double x = edge * N;
                const double c = std::round(x);
                return std::abs(x - c) < 1e-6 ? int(c) : -1;
            }
            // The cell ranges [start, end) along x and y (N for an axis that is not split) of a task
            inline bool domain_cell_ranges(int N, const FML::Domain & domain, int task, std::array<int, 2> & start,
                                           std::array<int, 2> & end) {
                start = {0, 0};
                end = {N, N};
                if (domain.is_slab()) {
                    start[0] = cell_of_edge(domain.x_min(task), N);
                    end[0] = cell_of_edge(domain.x_max(task), N);
                } else if (domain.holds<FML::PencilDomain>()) {
                    const auto c = domain.get<FML::PencilDomain>().coords_of(task);
                    start[0] = cell_of_edge(domain.x_edges()[c[0]], N);
                    end[0] = cell_of_edge(domain.x_edges()[c[0] + 1], N);
                    start[1] = cell_of_edge(domain.y_edges()[c[1]], N);
                    end[1] = cell_of_edge(domain.y_edges()[c[1] + 1], N);
                } else {
                    return false;
                }
                for (int i = 0; i < 2; i++)
                    if (start[i] < 0 or end[i] < 0 or end[i] <= start[i])
                        return false;
                return true;
            }
        } // namespace detail

        template <int NDIM, class T>
        bool MPIGrid<NDIM, T>::fits_domain(int N, const FML::Domain & domain) {
            if (N <= 0 or domain.ntasks() != FML::NTasks)
                return false;
            if (not domain.is_slab() and (NDIM < 2 or not domain.holds<FML::PencilDomain>()))
                return false;
            for (int task = 0; task < FML::NTasks; task++) {
                std::array<int, 2> start, end;
                if (not detail::domain_cell_ranges(N, domain, task, start, end))
                    return false;
            }
            return true;
        }

        template <int NDIM, class T>
        MPIGrid<NDIM, T>::MPIGrid(
            int N, bool periodic, const FML::Domain & domain, int n_extra_slices_left, int n_extra_slices_right) {
            assert_mpi(N > 0, "[MPIGrid] We need Ngrid > 0\n");
            assert_mpi(n_extra_slices_left >= 0 and n_extra_slices_right >= 0,
                       "[MPIGrid] Number of extra slices cannot be negative\n");
            assert_mpi(domain.is_slab() or (NDIM >= 2 and domain.holds<FML::PencilDomain>()),
                       "[MPIGrid] The domain must be a Slab or (NDIM >= 2) a Pencil domain\n");
            assert_mpi(fits_domain(N, domain),
                       "[MPIGrid] The edges of the domain must lie on the cells and every task must have cells\n");
            _periodic = periodic;
            _N = N;
            _domain = domain;
            _start.fill(0);
            _nloc.fill(N);
            _nextra_left.fill(0);
            _nextra_right.fill(0);

            // The local ranges and the neighbors along x and y (the tasks whose range starts where ours ends and
            // the other way around, periodic)
            std::array<int, 2> start, end;
            detail::domain_cell_ranges(N, domain, FML::ThisTask, start, end);
            const int nsplit = NDIM >= 2 ? 2 : 1;
            for (int i = 0; i < nsplit; i++) {
                _start[i] = start[i];
                _nloc[i] = end[i] - start[i];
            }
            _LeftTask = _RightTask = _DownTask = _UpTask = FML::ThisTask;
            for (int task = 0; task < FML::NTasks; task++) {
                std::array<int, 2> s, e;
                detail::domain_cell_ranges(N, domain, task, s, e);
                if (s[1] == start[1] and e[0] % N == start[0])
                    _LeftTask = task;
                if (s[1] == start[1] and s[0] == end[0] % N)
                    _RightTask = task;
                if (s[0] == start[0] and e[1] % N == start[1])
                    _DownTask = task;
                if (s[0] == start[0] and s[1] == end[1] % N)
                    _UpTask = task;
            }

            // The extra slices along x (also on one task, as for the slabs) and the extra layers along y if split
            _nextra_left[0] = n_extra_slices_left;
            _nextra_right[0] = n_extra_slices_right;
            if (NDIM >= 2 and _nloc[NDIM >= 2 ? 1 : 0] < N) {
                _nextra_left[NDIM >= 2 ? 1 : 0] = n_extra_slices_left;
                _nextra_right[NDIM >= 2 ? 1 : 0] = n_extra_slices_right;
            }
            allocate();
        }

        template <int NDIM, class T>
        T * MPIGrid<NDIM, T>::get_y() {
            return &_y[_NtotLocalLeft];
        }

        template <int NDIM, class T>
        const T * MPIGrid<NDIM, T>::get_y() const {
            return &_y[_NtotLocalLeft];
        }

        template <int NDIM, class T>
        T & MPIGrid<NDIM, T>::get_y(IndexInt index) {
#ifdef BOUNDSCHECK
            assert_index(index);
#endif
            index += _NtotLocalLeft;
            return _y[index];
        }

        template <int NDIM, class T>
        T & MPIGrid<NDIM, T>::operator[](IndexInt index) {
            return get_y(index);
        }

        template <int NDIM, class T>
        const T & MPIGrid<NDIM, T>::operator[](IndexInt index) const {
#ifdef BOUNDSCHECK
            assert_index(index);
#endif
            return _y[_NtotLocalLeft + index];
        }

        // Set a cell denoted by a (local) index in the grid
        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::set_y(IndexInt index, const T & value) {
#ifdef BOUNDSCHECK
            assert_index(index);
#endif
            index += _NtotLocalLeft;
            _y[index] = value;
        }

        // Set a cell denoted by a coordinate in the grid
        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::set_y(std::array<int, NDIM> & coord, const T & value) {
            IndexInt index = index_from_coord(coord);
            set_y(index, value);
        }

        // Set the whole grid with a function
        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::set_y(const std::function<T(std::array<double, NDIM> &)> & func) {
            T * y = get_y();
            for (IndexInt index = 0; index < _NtotLocal; index++) {
                auto pos = get_pos(index);
                y[index] = func(pos);
            }
            communicate_boundaries();
        }

        template <int NDIM, class T>
        std::array<int, NDIM> MPIGrid<NDIM, T>::coord_from_index(IndexInt index) const {
#ifdef BOUNDSCHECK
            assert_index(index);
#endif
            index += _NtotLocalLeft;
            if (index >= _YExtraStart)
                return coord_from_index_in_ylayers(index - _YExtraStart);
            std::array<int, NDIM> coord;
            for (int idim = NDIM - 1; idim >= 1; idim--) {
                coord[idim] = index % _nloc[idim];
                index = (index / _nloc[idim]);
            }
            coord[0] = index - _n_extra_slices_left;
            return coord;
        }

        template <int NDIM, class T>
        std::array<int, NDIM> MPIGrid<NDIM, T>::coord_from_index_in_ylayers(IndexInt index) const {
            // In the extra y-layers: [layer][x][the other axes]
            std::array<int, NDIM> coord{};
            if constexpr (NDIM > 1) {
                for (int idim = NDIM - 1; idim >= 2; idim--) {
                    coord[idim] = index % _nloc[idim];
                    index = (index / _nloc[idim]);
                }
                coord[0] = int(index % _NxExt) - _nextra_left[0];
                const int layer = int(index / _NxExt);
                coord[1] = layer < _nextra_left[1] ? -1 - layer : _nloc[1] + layer - _nextra_left[1];
            }
            return coord;
        }

        template <int NDIM, class T>
        std::array<int, NDIM> MPIGrid<NDIM, T>::globalcoord_from_index(IndexInt index) const {
#ifdef BOUNDSCHECK
            assert_index(index);
#endif
            auto globalcoord = coord_from_index(index);
            for (int idim = 0; idim < NDIM; idim++)
                globalcoord[idim] += _start[idim];
            return globalcoord;
        }

        template <int NDIM, class T>
        IndexInt MPIGrid<NDIM, T>::index_from_globalcoord(const std::array<int, NDIM> & globalcoord) const {
            auto coord = globalcoord;
            for (int idim = 0; idim < NDIM; idim++)
                coord[idim] -= _start[idim];
            return index_from_coord(coord);
        }

        template <int NDIM, class T>
        IndexInt MPIGrid<NDIM, T>::index_from_coord(const std::array<int, NDIM> & coord) const {
#ifdef BOUNDSCHECK
            assert_coord(coord);
#endif
            if constexpr (NDIM > 1) {
                if (coord[1] < 0 or coord[1] >= _nloc[1])
                    return index_from_coord_in_ylayers(coord);
            }
            IndexInt index = 0;
            for (int idim = 0; idim < NDIM; idim++)
                index += IndexInt(coord[idim]) * _stride[idim];
            return index;
        }

        template <int NDIM, class T>
        IndexInt MPIGrid<NDIM, T>::index_from_coord_in_ylayers(const std::array<int, NDIM> & coord) const {
            if constexpr (NDIM > 1) {
                IndexInt rest = 0;
                for (int idim = 2; idim < NDIM; idim++)
                    rest = rest * _nloc[idim] + coord[idim];
                const IndexInt layer = coord[1] < 0 ? -1 - coord[1] : _nextra_left[1] + coord[1] - _nloc[1];
                return _YExtraStart - _NtotLocalLeft + (layer * _NxExt + coord[0] + _nextra_left[0]) * _stride[1] +
                       rest;
            }
            return 0;
        }

        template <int NDIM, class T>
        std::array<double, NDIM> MPIGrid<NDIM, T>::get_pos(IndexInt index) {
            return get_pos(coord_from_index(index));
        }

        template <int NDIM, class T>
        double MPIGrid<NDIM, T>::get_radial_distance(IndexInt index, std::array<double, NDIM> & point) {
            auto pos = get_pos(coord_from_index(index));
            double r2 = 0;
            for (int idim = 0; idim < NDIM; idim++) {
                auto dist = point[idim] - pos[idim];
                if (_periodic) {
                    if (dist > 0.5)
                        dist -= 1.0;
                    if (dist < -0.5)
                        dist += 1.0;
                }
                r2 += dist * dist;
            }
            return std::sqrt(r2);
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::assert_index(IndexInt index) const {
            index += _NtotLocalLeft;
            assert_mpi(index >= 0,
                       "[MPIGrid::assert_index] OutOfBounds. Index corresponds to cells to the left of the extra left "
                       "slices\n");
            assert_mpi(index < _NtotLocalAlloc,
                       "[MPIGrid::assert_index] OutOfBounds. Index corresponds to cells to the right of the extra "
                       "cells\n");
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::assert_coord(const std::array<int, NDIM> & coord) const {
            assert_mpi(coord.size() == NDIM, "");
            for (int idim = NDIM - 1; idim >= 0; idim--) {
                assert_mpi(coord[idim] + _nextra_left[idim] >= 0,
                           "[MPIGrid::assert_coord] Coordinate is too small. Do not correspond to a cell in the grid\n");
                assert_mpi(coord[idim] - _nextra_right[idim] < _nloc[idim],
                           "[MPIGrid::assert_coord] Coordinate is too large. Do not correspond to a cell in the grid\n");
            }
        }

        template <int NDIM, class T>
        std::array<double, NDIM> MPIGrid<NDIM, T>::get_pos(const std::array<int, NDIM> & coord) {
#ifdef BOUNDSCHECK
            assert_coord(coord);
#endif
            std::array<double, NDIM> pos;
#pragma GCC unroll 4
            for (int idim = NDIM - 1; idim >= 0; idim--) {
                pos[idim] = (coord[idim] + _start[idim]) / double(_N);
                if (_periodic) {
                    pos[idim] += pos[idim] < 0.0 ? 1.0 : 0.0;
                    pos[idim] -= pos[idim] >= 1.0 ? 1.0 : 0.0;
                }
            }
            return pos;
        }

        // Returns number of cells per dim
        template <int NDIM, class T>
        int MPIGrid<NDIM, T>::get_N() const {
            return _N;
        }

        template <int NDIM, class T>
        int MPIGrid<NDIM, T>::get_NLocal() {
            return _NLocal;
        }

        // Return total number of cells
        template <int NDIM, class T>
        IndexInt MPIGrid<NDIM, T>::get_Ntot() {
            return _Ntot;
        }

        template <int NDIM, class T>
        IndexInt MPIGrid<NDIM, T>::get_NtotLocal() const {
            return _NtotLocal;
        }

        template <int NDIM, class T>
        int MPIGrid<NDIM, T>::get_xStartLocal() const {
            return _xStartLocal;
        }

        template <int NDIM, class T>
        int MPIGrid<NDIM, T>::get_n_extra_slices_left() const {
            return _n_extra_slices_left;
        }

        template <int NDIM, class T>
        int MPIGrid<NDIM, T>::get_n_extra_slices_right() const {
            return _n_extra_slices_right;
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::free() {
            // Release the memory and reset all variables (an empty, default constructed grid)
            *this = MPIGrid<NDIM, T>();
        }

#ifdef USE_MPI
        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::sendrecv_bytes(char * sendbuf, char * recvbuf, size_t bytes, int dest, int source) {
            // MPI counts are int so send large buffers in chunks
            const size_t maxchunk = size_t(1) << 30;
            size_t offset = 0;
            do {
                const int nbytes = int(std::min(maxchunk, bytes - offset));
                MPI_Status status;
                MPI_Sendrecv(sendbuf + offset,
                             nbytes,
                             MPI_CHAR,
                             dest,
                             0,
                             recvbuf + offset,
                             nbytes,
                             MPI_CHAR,
                             source,
                             0,
                             FML::communicator,
                             &status);
                offset += size_t(nbytes);
            } while (offset < bytes);
        }
#endif

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::sendrecv_block(T * send, T * recv, IndexInt n, int dest, int source) {
            if (dest == FML::ThisTask and source == FML::ThisTask) {
                std::copy(send, send + n, recv);
                return;
            }
#ifdef USE_MPI
            sendrecv_bytes(reinterpret_cast<char *>(send),
                           reinterpret_cast<char *>(recv),
                           size_t(n) * sizeof(T),
                           dest,
                           source);
#endif
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::send_slice_right(int ix, std::vector<T> & recv_slice) {
            recv_slice.resize(_NperSlice);
            T * slice_tosend = _y.data() + _NtotLocalLeft + _NperSlice * ix;
            if (FML::NTasks == 1) {
                // We are our own right and left neighbor
                std::copy(slice_tosend, slice_tosend + _NperSlice, recv_slice.data());
                return;
            }
            sendrecv_block(slice_tosend, recv_slice.data(), _NperSlice, _RightTask, _LeftTask);
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::send_slice_left(int ix, std::vector<T> & recv_slice) {
            recv_slice.resize(_NperSlice);
            T * slice_tosend = _y.data() + _NtotLocalLeft + _NperSlice * ix;
            if (FML::NTasks == 1) {
                // We are our own right and left neighbor
                std::copy(slice_tosend, slice_tosend + _NperSlice, recv_slice.data());
                return;
            }
            sendrecv_block(slice_tosend, recv_slice.data(), _NperSlice, _LeftTask, _RightTask);
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::communicate_boundaries() {
            const int nsend_to_left = _n_extra_slices_right;
            const int nsend_to_right = _n_extra_slices_left;

            // Buffer to receive data in
            std::vector<T> recv_array(_NperSlice);

            // Send rightmost slices right and store in extra left slices. We fill the extra left slices from the
            // nearest (slice -1) and outwards: slice -1-i is slice NLocal-1-i of the left task. If we have more
            // extra slices than the left task has slices (NLocal-1-i < 0) we get the left tasks extra slice which
            // it received in an earlier iteration
            for (int i = 0; i < nsend_to_right; i++) {
                // (the edges of the box: with N < NTasks some tasks have no cells, so test the slices, not the task id)
                const bool do_not_store = not _periodic and _xStartLocal == 0;
                T * slice_left_torecv = _y.data() + _NperSlice * (_n_extra_slices_left - 1 - i);
                send_slice_right(_NLocal - 1 - i, recv_array);
                if (not do_not_store)
                    std::memcpy(slice_left_torecv, recv_array.data(), _NperSlice * sizeof(T));
            }

            // Send leftmost slices left and store in extra right slices: slice NLocal+i is slice i of the right
            // task (or its extra right slice i - NLocal received in an earlier iteration)
            for (int i = 0; i < nsend_to_left; i++) {
                const bool do_not_store = not _periodic and _xStartLocal + _NLocal == _N;
                T * slice_right_torecv = _y.data() + _NtotLocalLeft + _NtotLocal + _NperSlice * i;
                send_slice_left(i, recv_array);
                if (not do_not_store)
                    std::memcpy(slice_right_torecv, recv_array.data(), _NperSlice * sizeof(T));
            }

            // The extra y-layers (pencils): the rows of all x-slices (with the extra ones, i.e. the corners) the
            // same way. A row of fixed (x, y) is contiguous (the other axes) both in the main grid and in the layers
            if constexpr (NDIM > 1) {
                if (_nextra_left[1] + _nextra_right[1] == 0)
                    return;
                const IndexInt rowsize = _stride[1];
                std::vector<T> send(_NperYLayer), recv(_NperYLayer);
                auto pack_row = [&](int iy) {
                    std::array<int, NDIM> coord{};
                    coord[1] = iy;
                    for (int ix = -_nextra_left[0]; ix < _nloc[0] + _nextra_right[0]; ix++) {
                        coord[0] = ix;
                        const T * row = &_y[_NtotLocalLeft + index_from_coord(coord)];
                        std::copy(row, row + rowsize, send.data() + (ix + _nextra_left[0]) * rowsize);
                    }
                };
                for (int i = 0; i < _nextra_left[1]; i++) {
                    pack_row(_nloc[1] - 1 - i);
                    sendrecv_block(send.data(), recv.data(), _NperYLayer, _UpTask, _DownTask);
                    if (_periodic or _start[1] != 0)
                        std::copy(recv.begin(), recv.end(), _y.data() + _YExtraStart + i * _NperYLayer);
                }
                for (int i = 0; i < _nextra_right[1]; i++) {
                    pack_row(i);
                    sendrecv_block(send.data(), recv.data(), _NperYLayer, _DownTask, _UpTask);
                    if (_periodic or _start[1] + _nloc[1] != _N)
                        std::copy(recv.begin(),
                                  recv.end(),
                                  _y.data() + _YExtraStart + (_nextra_left[1] + i) * _NperYLayer);
                }
            }
        }

        template <int NDIM, class T>
        double MPIGrid<NDIM, T>::norm() {
            double norm2 = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : norm2)
#endif
            for (IndexInt i = 0; i < _NtotLocal; i++) {
                norm2 += AbsoluteValue(get_y(i)) * AbsoluteValue(get_y(i));
            }
            FML::SumOverTasks(&norm2);
            return sqrt(norm2 / double(_Ntot));
        }

        template <int NDIM, class T>
        IndexInt MPIGrid<NDIM, T>::get_shifted_index(IndexInt index,
                                                     const std::array<int, NDIM> & coord,
                                                     int axis,
                                                     int offset) const {
            const int c = coord[axis] + offset;
            // The whole axis is on this task: periodic in the local grid
            if (_nloc[axis] == _N) {
                int cnew = c;
                if (_periodic)
                    cnew = ((c % _N) + _N) % _N;
                return index + IndexInt(cnew - coord[axis]) * _stride[axis];
            }
            // In the extra cells: the x-slices continue the main grid in memory, the y-layers are elsewhere
            if (axis == 0 or (c >= 0 and c < _nloc[axis]))
                return index + IndexInt(offset) * _stride[axis];
            auto coord_new = coord;
            coord_new[axis] = c;
            return index_from_coord(coord_new);
        }

        // The closted 2NDIM+1 cells (including the cell itself at 0) to a given cell
        template <int NDIM, class T>
        std::array<IndexInt, 2 * NDIM + 1> MPIGrid<NDIM, T>::get_neighbor_gridindex(IndexInt index) const {
            std::array<IndexInt, 2 * NDIM + 1> index_list;
            index_list[0] = index;
            const auto coord = coord_from_index(index);
            for (int idim = NDIM - 1; idim >= 0; idim--) {
                const IndexInt st = _stride[idim];
                const int c = coord[idim];
                // (the whole axis on this task: periodic in the local grid)
                IndexInt minus = index - st + (c == 0 ? _wrap[idim] : 0);
                IndexInt plus = index + st - (c == _nloc[idim] - 1 ? _wrap[idim] : 0);
                if (idim == 1 and _ysplit and (c == 0 or c == _nloc[idim] - 1)) {
                    // In the extra y-layers
                    minus = get_shifted_index(index, coord, idim, -1);
                    plus = get_shifted_index(index, coord, idim, +1);
                }
                index_list[2 * idim + 1] = minus;
                index_list[2 * idim + 2] = plus;
            }
            return index_list;
        }

        template <int NDIM, class T>
        std::array<T, NDIM> MPIGrid<NDIM, T>::get_gradient(IndexInt index) const {
            auto index_list = get_neighbor_gridindex(index);
            std::array<T, NDIM> gradient;
            for (int idim = 0; idim < NDIM; idim++) {
                gradient[idim] =
                    (_y[_NtotLocalLeft + index_list[2 * idim + 2]] - _y[_NtotLocalLeft + index_list[2 * idim + 1]]) /
                    2.0 * double(_N);
            }
            return gradient;
        }
    } // namespace GRID
} // namespace FML
#endif
