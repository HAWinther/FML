#ifndef MPIGRID_HEADER
#define MPIGRID_HEADER

#include <array>
#include <climits>
#include <complex>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
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
        /// Every index is an index in the local main grid unless specified otherwise
        /// e.g. index = ix*N^2 + iy*N + iz corresponds to (ix + xStartLocal, iy, iz, ...) in the global grid
        ///
        /// Every coord is a local coordinate unless specified otherwise
        /// e.g. (ix, iy, iz, ...) corresponds to (ix + xStartLocal, iy, iz, ...) in the global grid
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
          private:
            bool _periodic{true};         // Is the box periodic or not
            int _N{0};                    // Gridcells per dimension in the global grid
            int _NLocal{0};               // Gridcells in the x-dimension for current task
            int _xStartLocal{0};          // The ix-index the local task starts at
            int _LeftTask{0};             // The id of the task on the right
            int _RightTask{0};            // The id of the task on the left
            int _n_extra_slices_left{0};  // Extra x-slices to the left
            int _n_extra_slices_right{0}; // Extra x-slices to the right

            IndexInt _Ntot{0};           // How many cells in main grid in total
            IndexInt _NtotLocalLeft{0};  // Total cells in extra left slices
            IndexInt _NtotLocalRight{0}; // Total cells in extra right slices
            IndexInt _NtotLocal{0};      // How many cells in main grid on local task
            IndexInt _NtotLocalAlloc{0}; // How many cells we have allocated locally (includes extra slices)
            IndexInt _NperSlice{0};      // How many cells per x-slice

            // std::vector<T> _y;           // The grid data
            Vector<T> _y{}; // The grid data

            // Helper functions for bounds-checking
            void assert_index(IndexInt index) const;
            void assert_coord(const std::array<int, NDIM> & coord) const;

          public:
            // Constructors
            MPIGrid() = default;
            MPIGrid(int N, bool periodic, int n_extra_slices_left = 0, int n_extra_slices_right = 0);

            // Get a pointer to the start of the main grid
            T * get_y();

            // Get a reference to the cell at a given index
            T & get_y(IndexInt index);

            // Allow syntax grid[i] to get/set cells with a given index
            T & operator[](IndexInt index);

            // Assign value in the grid by index
            void set_y(IndexInt index, const T & value);

            // Assign value in the grid by coordinate
            void set_y(std::array<int, NDIM> & coord, const T & value);

            // Assign the whole grid from a function
            void set_y(std::function<T(std::array<double, NDIM> &)> & func);

            // Grid-index -> Local coordinate list
            // e.g. index = ix1_local * N^2 + ix2 * N + ix3 -> (ix1_local, ix2, ix3)
            std::array<int, NDIM> coord_from_index(IndexInt index) const;

            // Local coordiates -> Grid-index
            // e.g. index = ix1_local * N^2 + ix2 * N + ix3
            IndexInt index_from_coord(const std::array<int, NDIM> & coord) const;

            // From local index to global coordinates
            std::array<int, NDIM> globalcoord_from_index(IndexInt index) const;

            // From global coordinates to local index
            IndexInt index_from_globalcoord(const std::array<int, NDIM> & globalcoord) const;

            // The closest 2NDIM+1 cells to a given cell (including the cell itself at 0)
            std::array<IndexInt, 2 * NDIM + 1> get_neighbor_gridindex(IndexInt index) const;

            // Gradient at a given cell (simple 2-point symmetric stensil)
            std::array<T, NDIM> get_gradient(IndexInt index) const;

            // Get some info about the grid
            IndexInt get_Ntot();
            IndexInt get_NtotLocal();
            int get_N();
            int get_NLocal();
            int get_xStartLocal();
            int get_n_extra_slices_left();
            int get_n_extra_slices_right();

            // Returns the position of the cell in the global grid
            std::array<double, NDIM> get_pos(IndexInt index);
            std::array<double, NDIM> get_pos(const std::array<int, NDIM> & coord);

            // Distance from a point to the cell. Fiducial choice is center of the box
            double get_radial_distance(IndexInt index, std::array<double, NDIM> & point);

            // The rms of the main grid sqrt(Sum y^2/N)
            double norm();

            // Communicate all the extra slices left and right
            void communicate_boundaries();

            // Send a slice of the grid to the left or right task
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
            FML::MemoryLog::get()->add_label(_y.data(), _y.capacity() * sizeof(ComplexType), label);
#endif
        }

#define OPS(OP)                                                                                                        \
    template <int NDIM, class T>                                                                                       \
    auto MPIGrid<NDIM, T>::operator OP(const MPIGrid<NDIM, T> & rhs)->MPIGrid<NDIM, T> & {                             \
        assert_mpi(_NtotLocal == rhs._NtotLocal, "");                                                                  \
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
    auto operator OP(MPIGrid<NDIM, T> lhs, const MPIGrid<NDIM, T> & rhs)->MPIGrid<NDIM, T> & {                         \
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

        // Constructor with intial value
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
            _NLocal = slices_per_task[FML::ThisTask];
            _xStartLocal = sumslicesbeforethistask;
            _n_extra_slices_left = n_extra_slices_left;
            _n_extra_slices_right = n_extra_slices_right;
            _NperSlice = FML::power(_N, NDIM - 1);
            _NtotLocalLeft = _NperSlice * _n_extra_slices_left;
            _NtotLocalRight = _NperSlice * _n_extra_slices_right;
            _NtotLocal = _NperSlice * _NLocal;
            _Ntot = _NperSlice * _N;
            _NtotLocalAlloc = _NtotLocalLeft + _NtotLocal + _NtotLocalRight;
            _RightTask = (FML::ThisTask + 1) % FML::NTasks;
            _LeftTask = (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks;

            if (_NLocal == 0) {
                // Tasks with 0 cells does nothing
                _RightTask = _LeftTask = FML::ThisTask;
                _n_extra_slices_left = _n_extra_slices_right = 0;
                _NtotLocalLeft = _NtotLocalRight = _NtotLocal = _NtotLocalAlloc = 0;
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
            _y.resize(_NtotLocalAlloc);
            add_memory_label("MPIGrid");

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

        template <int NDIM, class T>
        T * MPIGrid<NDIM, T>::get_y() {
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
        void MPIGrid<NDIM, T>::set_y(std::function<T(std::array<double, NDIM> &)> & func) {
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
            std::array<int, NDIM> coord;
            for (int idim = NDIM - 1; idim >= 1; idim--) {
                coord[idim] = index % _N;
                index = (index / _N);
            }
            coord[0] = index - _n_extra_slices_left;
            return coord;
        }

        template <int NDIM, class T>
        std::array<int, NDIM> MPIGrid<NDIM, T>::globalcoord_from_index(IndexInt index) const {
#ifdef BOUNDSCHECK
            assert_index(index);
#endif
            auto globalcoord = coord_from_index(index);
            globalcoord[0] += _xStartLocal;
            return globalcoord;
        }

        template <int NDIM, class T>
        IndexInt MPIGrid<NDIM, T>::index_from_globalcoord(const std::array<int, NDIM> & globalcoord) const {
#ifdef BOUNDSCHECK
            auto coord = globalcoord;
            coord[0] -= _xStartLocal;
            assert_coord(coord);
#endif
            IndexInt index = 0;
            for (int idim = NDIM - 1, n = 1; idim >= 0; idim--, n *= _N) {
                index += globalcoord[idim] * n;
                if (idim == 0)
                    index -= _xStartLocal * n;
            }
            return index;
        }

        template <int NDIM, class T>
        IndexInt MPIGrid<NDIM, T>::index_from_coord(const std::array<int, NDIM> & coord) const {
#ifdef BOUNDSCHECK
            assert_coord(coord);
#endif
            IndexInt index = 0;
            for (int idim = NDIM - 1, n = 1; idim >= 0; idim--, n *= _N)
                index += coord[idim] * n;
            return index;
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
                       "right slices\n");
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::assert_coord(const std::array<int, NDIM> & coord) const {
            assert_mpi(coord.size() == NDIM, "");
            for (int idim = NDIM - 1; idim >= 1; idim--) {
                assert_mpi(coord[idim] >= 0,
                           "[MPIGrid::assert_coord] Coordinate < 0. Do not correspond to a cell in the grid\n");
                assert_mpi(coord[idim] < _N,
                           "[MPIGrid::assert_coord] Coordinate >= N. Do not correspond to a cell in the grid\n");
            }
            assert_mpi(coord[0] + _n_extra_slices_left >= 0,
                       "[MPIGrid::assert_coord] x-coordinate is too small. Do not correspond to a cell in the grid\n");
            assert_mpi(coord[0] - _n_extra_slices_right < _NLocal,
                       "[MPIGrid::assert_coord] x-coordinate is too small. Do not correspond to a cell in the grid\n");
        }

        template <int NDIM, class T>
        std::array<double, NDIM> MPIGrid<NDIM, T>::get_pos(const std::array<int, NDIM> & coord) {
#ifdef BOUNDSCHECK
            assert_coord(coord);
#endif
            std::array<double, NDIM> pos;
            for (int idim = NDIM - 1; idim >= 1; idim--) {
                pos[idim] = coord[idim] / double(_N);
            }
            pos[0] = (coord[0] + _xStartLocal) / double(_N);

            if (_periodic) {
                if (pos[0] < 0.0)
                    pos[0] += 1.0;
                if (pos[0] >= 1.0)
                    pos[0] -= 1.0;
            }

            return pos;
        }

        // Returns number of cells per dim
        template <int NDIM, class T>
        int MPIGrid<NDIM, T>::get_N() {
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
        IndexInt MPIGrid<NDIM, T>::get_NtotLocal() {
            return _NtotLocal;
        }

        template <int NDIM, class T>
        int MPIGrid<NDIM, T>::get_xStartLocal() {
            return _xStartLocal;
        }

        template <int NDIM, class T>
        int MPIGrid<NDIM, T>::get_n_extra_slices_left() {
            return _n_extra_slices_left;
        }

        template <int NDIM, class T>
        int MPIGrid<NDIM, T>::get_n_extra_slices_right() {
            return _n_extra_slices_right;
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::free() {
            _y.clear();
            _y.shrink_to_fit();
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::send_slice_right(int ix, std::vector<T> & recv_slice) {
            if (FML::NTasks == 1)
                return;

#ifdef USE_MPI
            const int bytes_slice = _NperSlice * sizeof(T);
            recv_slice.resize(bytes_slice);

            T * slice_tosend = _y.data() + _NtotLocalLeft + _NperSlice * ix;
            char * sendbuf = reinterpret_cast<char *>(slice_tosend);

            T * slice_torecv = recv_slice.data();
            char * recvbuf = reinterpret_cast<char *>(slice_torecv);

            MPI_Status status;
            MPI_Sendrecv(sendbuf,
                         bytes_slice,
                         MPI_CHAR,
                         _RightTask,
                         0,
                         recvbuf,
                         bytes_slice,
                         MPI_CHAR,
                         _LeftTask,
                         0,
                         MPI_COMM_WORLD,
                         &status);
#endif
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::send_slice_left(int ix, std::vector<T> & recv_slice) {
            if (FML::NTasks == 1)
                return;

#ifdef USE_MPI
            const int bytes_slice = _NperSlice * sizeof(T);
            recv_slice.resize(bytes_slice);

            T * slice_tosend = _y.data() + _NtotLocalLeft + _NperSlice * ix;
            char * sendbuf = reinterpret_cast<char *>(slice_tosend);

            T * slice_torecv = recv_slice.data();
            char * recvbuf = reinterpret_cast<char *>(slice_torecv);

            MPI_Status status;
            MPI_Sendrecv(sendbuf,
                         bytes_slice,
                         MPI_CHAR,
                         _LeftTask,
                         0,
                         recvbuf,
                         bytes_slice,
                         MPI_CHAR,
                         _RightTask,
                         0,
                         MPI_COMM_WORLD,
                         &status);
#endif
        }

        template <int NDIM, class T>
        void MPIGrid<NDIM, T>::communicate_boundaries() {
            const int nsend_to_left = _n_extra_slices_right;
            const int nsend_to_right = _n_extra_slices_left;

            // Buffer to recive data in
            std::vector<T> recv_array(_NperSlice);

            // Send rightmost slices right and store in extra left slices
            for (int i = 0; i < nsend_to_right; i++) {
                const bool do_not_store = not _periodic and FML::ThisTask == 0;
                T * slice_left_torecv = _y.data() + _NperSlice * i;
                send_slice_right(_NLocal - nsend_to_right + i, recv_array);
                if (not do_not_store)
                    std::memcpy(slice_left_torecv, recv_array.data(), _NperSlice * sizeof(T));
            }

            // Send leftmost slices left and store in extra right slices
            for (int i = 0; i < nsend_to_left; i++) {
                const bool do_not_store = not _periodic and FML::ThisTask == FML::NTasks - 1;
                T * slice_right_torecv = _y.data() + _NtotLocalLeft + _NtotLocal + _NperSlice * i;
                send_slice_left(i, recv_array);
                if (not do_not_store)
                    std::memcpy(slice_right_torecv, recv_array.data(), _NperSlice * sizeof(T));
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
#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
            return sqrt(norm2 / double(_Ntot));
        }

        // The closted 2NDIM+1 cells (including the cell itself at 0) to a given cell
        template <int NDIM, class T>
        std::array<IndexInt, 2 * NDIM + 1> MPIGrid<NDIM, T>::get_neighbor_gridindex(IndexInt index) const {
            std::array<IndexInt, 2 * NDIM + 1> index_list;
            index_list[0] = index;

            // Local coordinates
            auto coord = coord_from_index(index);
            IndexInt Npow = 1;
            for (int idim = NDIM - 1; idim >= 0; idim--, Npow *= _N) {
                int coord_minus = coord[idim] - 1;
                int coord_plus = coord[idim] + 1;
                if (_periodic and not(idim == 0 and FML::NTasks > 1)) {
                    coord_minus = (coord_minus + _N) % _N;
                    coord_plus = coord_plus % _N;
                }
                if (idim == 0 and FML::NTasks > 1) {
                    index_list[2 * idim + 1] = index - Npow;
                    index_list[2 * idim + 2] = index + Npow;
                } else {
                    index_list[2 * idim + 1] = index + (coord_minus - coord[idim]) * Npow;
                    index_list[2 * idim + 2] = index + (coord_plus - coord[idim]) * Npow;
                }
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
