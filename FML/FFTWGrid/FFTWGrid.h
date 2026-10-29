#ifndef FFTWGRIDMPI_HEADER
#define FFTWGRIDMPI_HEADER
#include <array>
#include <cassert>
#include <complex>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>
#ifdef USE_FFTW
#include <fftw3.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#ifdef USE_FFTW
#include <fftw3-mpi.h>
#endif
#endif
#ifdef USE_GSL
#include <FML/Spline/Spline.h>
#endif

#include <FML/Global/Global.h>
#include <FML/Performance/Counters.h>
#ifdef FML_ENABLE_PERFORMANCE_COUNTERS
#include <FML/Performance/Storage.h>
#endif

namespace FML {

    /// This namespace contains various grids for holding data shared across different tasks.
    namespace GRID {
        /// Number of full-complex modes represented by a stored r2c coefficient.
        inline int fourier_mode_multiplicity(int last_coord, int nmesh) {
            return last_coord == 0 || (nmesh % 2 == 0 && last_coord == nmesh / 2) ? 1 : 2;
        }


        // Include type definitions
#include "FFTWGlobal.h"

        // Forward declaration of range classes
        class FourierRange;
        class RealRange;

        //==========================================================================
        ///
        /// Class for holding grids and performing real-to-complex and complex-to-real
        /// FFTs using FFTW with MPI. Templated on dimension.
        ///
        /// The real and fourier grid is stored in the same array to save memory
        /// and all transforms are done in-place. Keeps track of the status of the grid, i.e.
        /// if its in real-space or fourier-space (set with set_grid_status_real)
        ///
        /// The default constructor arguments in (Nmesh, n_extra_left, n_extra_right) is:
        ///
        ///   N                : Dimension of the grid
        ///
        ///   Nmesh            : Number of grid-nodes per dimension (assuming the same)
        ///
        ///   n_extra          : Alloc extra slices of the grid in the x-dimension (left and/or right)
        ///
        /// Compile-time defines:
        ///
        ///   BOUNDSCHECK_FFTWGRID       : bound checks when setting and getting values
        ///
        ///   SINGLE_PRECISION_FFTW      : use float instead of double
        ///
        ///   LONG_DOUBLE_PRECISION_FFTW : use load double instead of double
        ///
        ///   DEBUG_FFTWGRID             : Show some info while running
        ///
        ///   USE_FFTW_THREADS           : Use threads if possible. With this we assume the maximum number of threads
        ///                                If you want to use fewer then you can call create_wisdom(..., nthreads) to
        ///                                change this
        ///
        /// See FFTWGlobal.h for type definitions and the defines that deal with the precision we want.
        ///
        /// External variables/methods we rely on:
        ///
        ///    ThisTask             : The MPI task number
        ///
        ///    NTasks               : Number of MPI tasks
        ///
        ///    assert_mpi(Expr, Msg)
        ///
        ///    T power(T base, int exponent);
        ///
        /// The grid-stucture is only compatible with in-place FFTW transforms so
        /// for using the methods that FT from one grid to another we copy the grid and
        /// transforms that.
        ///
        //==========================================================================

        template <int N>
        class FFTWGrid {
            // The slab layout (slices of Nmesh^(N-2) * (Nmesh/2+1) complex cells) needs N >= 2
            static_assert(N >= 2, "FFTWGrid: only N >= 2 dimensions are supported");

          private:
            // Index for local cells. Using long long int as standard
            using IndexIntType = FML::IndexIntType;

            // The raw data vectors. These have the format [extra slices left][main grid][extra slices right]
            // Vector = std::vector<ComplexType> with possible custom allocator
#ifdef FML_ENABLE_PERFORMANCE_COUNTERS
            using PayloadStorage = FML::PERFORMANCE::Storage<ComplexType>;
#else
            using PayloadStorage = Vector<ComplexType>;
#endif
            PayloadStorage fourier_grid_raw{};

            // Mesh size and the dimension of the grid
            int Nmesh{0};
            // Number of local slices in real and Fourier space (the same)
            ptrdiff_t Local_nx{0};
            // The index in the global grid the local grid starts at
            ptrdiff_t Local_x_start{0};

            // The total number of grid-cells we allocate
            ptrdiff_t NmeshTotRealAlloc{0};
            ptrdiff_t NmeshTotComplexAlloc{0};

            // The number of grid-cells that is active in the main part of the grid
            ptrdiff_t NmeshTotReal{0};
            ptrdiff_t NmeshTotComplex{0};

            // XXX Add below and remove non-needed things
            // NComplexCellsBeforeMain
            // NComplexCellsBeforeRight

            // Number of extra cells per slice
            ptrdiff_t NmeshTotRealSlice{0};
            ptrdiff_t NmeshTotComplexSlice{0};

            // Number of extra slices to the left and right of the grid
            int n_extra_x_slices_left{0};
            int n_extra_x_slices_right{0};

            // If you want to keep track of the field is in real space or in Fourier space
            bool grid_is_in_real_space{true};

            std::string name{""};

#ifdef USE_FFTW
            // Get a (cached) FFTW plan for transforming this grid. r2c = true for real to complex
            // (the grid data is preserved and stays at the same address)
            my_fftw_plan get_fftw_plan(bool r2c);
#endif

            // The local slab decomposition for a given Nmesh. Returns the number of complex cells FFTW needs
            // locally for the main grid (can be larger than Local_nx * NmeshTotComplexSlice)
            static ptrdiff_t compute_local_size(int Nmesh, ptrdiff_t & Local_nx, ptrdiff_t & Local_x_start);

            // Number of reals at the start of the right extra slices that FFTW may use as scratch space
            // in an in-place transform (and that we therefore must back up)
            ptrdiff_t get_nreal_right_to_backup() const;

#ifdef USE_MPI
            // MPI_Sendrecv of a byte buffer of any size (split into chunks whose count fits in an int)
            static void sendrecv_bytes(char * sendbuf, char * recvbuf, size_t bytes, int dest, int source);
#endif

          public:
            FFTWGrid() = default;

            /// Has the grid been allocated
            explicit operator bool() const;

            /// Constructor
            /// @param[in] Nmesh Total number of grid-cells per dimension
            /// @param[in] n_extra_x_slices_left Number of extra x-slices to the left of the grid (for keeping boundary
            /// data from neighbor)
            /// @param[in] n_extra_x_slices_right Number of extra x-slices to the right of the grid (for keeping
            /// boundary data from neighbor)
            FFTWGrid(int Nmesh, int n_extra_x_slices_left = 0, int n_extra_x_slices_right = 0);

            // Copying and assignment from grids with the same dimension only is allowed
            FFTWGrid(const FFTWGrid & rhs) = default;
            FFTWGrid & operator=(const FFTWGrid & rhs) = default;
            // Moving takes over the memory. The moved-from grid is left empty (as a default constructed grid)
            FFTWGrid(FFTWGrid && rhs) noexcept;
            FFTWGrid & operator=(FFTWGrid && rhs) noexcept;

            // Pointers to various parts of the grid
            FloatType * get_real_grid_left(); /// The left most slice (slice ix = -nleft_extra,...,-2,-1)
            FloatType * get_real_grid();      /// The main grid       (slice ix = 0...NLocal_x-1)
            FloatType *
            get_real_grid_right(); /// The right grid      (slice ix = NLocal_x,NLocal_x+1,...,NLocal_x+nright_extra-1)
            FloatType *
            get_real_grid_by_slice(int slice); /// Get the ix'th slice (i.e. -nleft_extra <= ix < NLocal_x+nright_extra)
            ComplexType * get_fourier_grid();  /// The Fourier grid (aligns with the main real grid)
#ifdef USE_FFTW
            my_fftw_complex * get_fftw_grid(); /// The fftw_complex cast of get_fourier_gride
#endif
            /// Free all the memory associated with the grid and reset it to an empty grid (as if default
            /// constructed, only the label is kept)
            void free();

            /// Perform real-to-complex fourier transform
            void fftw_r2c();

            /// Perform complex-to-real fourier transform
            void fftw_c2r();

            /// Fill the whole real-grid with a constant value
            void fill_real_grid(const FloatType val);
            /// Fill the whole fourier-grid with a constant value
            void fill_fourier_grid(const ComplexType val);

            /// Fill the main grid from a function specifying the value at a given position
            void fill_real_grid(const std::function<FloatType(std::array<double, N> &)> & func);
            /// Fill the main grid from a function specifying the value at a given fourier wave-vector
            void fill_fourier_grid(const std::function<ComplexType(std::array<double, N> &)> & func);

            /// Get the (local) cell coordinates from a local index
            std::array<int, N> get_coord_from_index(const IndexIntType index_real) const;
            /// Get the (local) cell coordinates from a local index
            std::array<int, N> get_fourier_coord_from_index(const IndexIntType index_fourier) const;

            /// From integer (local) coordinate of a cell in Local_nx x [0,Nmesh)^(Ndim-1) to (local) index in allocated
            /// grid
            IndexIntType get_index_real(const std::array<int, N> & coord) const;

            /// From integer (local) coordinate in fourier grid in Local_nx x [0,Nmesh)^(Ndim-2) x [0,Nmesh/2+1) to
            /// index in allocated grid
            IndexIntType get_index_fourier(const std::array<int, N> & coord) const;

            /// Fetch value in grid by (local) integer coordinate
            FloatType get_real(const std::array<int, N> & coord) const;
            /// Fetch value in grid by (local) index
            FloatType get_real_from_index(const IndexIntType index) const;

            /// Fetch value in fourier grid by (local) integer coordinate
            ComplexType get_fourier(const std::array<int, N> & coord) const;
            /// Fetch value in fourier grid by (local) index
            ComplexType get_fourier_from_index(const IndexIntType index) const;

            /// The (global) position of a real grid-cell in [0,1)^Ndim
            std::array<double, N> get_real_position(const std::array<int, N> & coord) const;

            /// Get the wave-vector of a grid-cell in Fourier space (for physical [k] multiply by 1/Boxsize)
            std::array<double, N> get_fourier_wavevector(const std::array<int, N> & coord) const;
            /// Get the wave-vector of a grid-cell in Fourier space (for physical [k] multiply by 1/Boxsize)
            std::array<double, N> get_fourier_wavevector_from_index(const IndexIntType index) const;
            std::array<int, N> get_fourier_integer_wavevector_from_index(const IndexIntType index) const;

            /// Set value in grid from (local) integer coordinate in Local_nx x [0,Nmesh)^(Ndim-1)
            void set_real(const std::array<int, N> & coord, const FloatType value);
            /// Set value in grid from (local) index
            void set_real_from_index(const IndexIntType ind, const FloatType value);
            /// Add to value in grid
            void add_real(const std::array<int, N> & coord, const FloatType value);

            /// Set value of cell in fourier grid using (local) coordinate in Local_nx x [0,Nmesh)^(Ndim-2) x
            /// [0,Nmesh/2+1)
            void set_fourier(const std::array<int, N> & coord, const ComplexType value);
            /// Set value of cell in fourier grid using (local) index
            void set_fourier_from_index(const IndexIntType ind, const ComplexType value);

            /// How many extra slices we have allocated to the left
            int get_n_extra_slices_left() const;
            /// How many extra slices we have allocated to the right
            int get_n_extra_slices_right() const;
            /// Number of grid-cells per dimension
            int get_nmesh() const;
            /// Dimension of the grid
            int get_ndim() const;

            /// Number of local x-slices in the real grid
            ptrdiff_t get_local_nx() const;
            /// The x-slice the first local slice has in the global grid
            ptrdiff_t get_local_x_start() const;

            /// The number of active real cells (with padding)
            ptrdiff_t get_ntot_real() const;
            /// Total number of active Fourier space grid-nodes [ e.g. Local_nx * N * N * ... * (N/2 + 1) ]
            ptrdiff_t get_ntot_fourier() const;
            /// Total number of Fourier space grid-nodes we allocate (same as above + the extra slices)
            ptrdiff_t get_ntot_fourier_alloc() const;

            /// From index in the grid get the k-vector and the magnitude of it
            void get_fourier_wavevector_and_norm_by_index(const IndexIntType ind,
                                                          std::array<double, N> & kvec,
                                                          double & kmag) const;
            /// From index in the grid get the k-vector and the norm of it (square magnitude)
            void get_fourier_wavevector_and_norm2_by_index(const IndexIntType ind,
                                                           std::array<double, N> & kvec,
                                                           double & kmag2) const;

            /// Range iterator for going through all active cells in the main real real grid by index
            /// [ e.g. for(auto and real_index: grid.real_range()) ]
            /// If you add the slice numbers we only loop over the given slice range
            RealRange get_real_range(int islice_begin = 0, int islice_end = 0) const;
            /// Range iterator for going through all active cells in the main fourier grid by index
            /// [ e.g. for(auto and real_index: grid.real_range()) ]
            /// If you add the slice numbers we only loop over the given slice range
            /// For the Fourier range islice denotes the ikx value
            FourierRange get_fourier_range(int islice_begin = 0, int islice_end = 0) const;

            /// The number of cells per slice that we alloc. Useful to jump from slice to slice
            ptrdiff_t get_ntot_real_slice_alloc() const;

            /// Check if we have NaN or Inf anywhere in the allocated grid (including the extra slices)
            bool nan_in_grids() const;

            /// Actual retained payload capacity, including halos/padding; excludes object/allocator overhead.
            size_t allocated_storage_bytes() const { return fourier_grid_raw.capacity() * sizeof(ComplexType); }

            /// Local allocation for a zero-halo grid, without allocating a mesh.
            static size_t storage_bytes(int nmesh) {
                assert_mpi(nmesh > 0 && nmesh % FML::NTasks == 0, "[FFTWGrid::storage_bytes] Invalid mesh size");
                ptrdiff_t nx, start;
                return size_t(compute_local_size(nmesh, nx, start)) * sizeof(ComplexType);
            }

            /// Send slices to the neighboring CPUs which stores them in the left and right extra slice storage
            /// (with more extra slices than local slices they are forwarded from tasks further away)
            void communicate_boundaries();

            /// This creates FFTW wisdom (but overwrites the arrays so must be done before setting the arrays)
            void create_wisdom(int planner_flag, int numthreads = FML::NThreads);
            /// Load FFTW wisdom
            void load_wisdom(std::string filename) const;
            /// Save FFTW wisdom
            void save_wisdom(std::string filename) const;

            /// Print some info about the grid
            void info();

            /// Get the status of the grid: is it currently a real grid or a fourier grid?
            bool get_grid_status_real();
            /// Set the status of the grid: is it currently a real grid or a fourier grid?
            void set_grid_status_real(bool grid_is_a_real_grid);

            /// For memory logging: add a label to the grid
            void add_memory_label(std::string label);

            /// Save to file (adds .X to fileprefix where X is ThisTask). A raw copy of the grid meant for
            /// checkpointing: read it back with load_from_file with the same number of tasks.
            /// Aborts (assert_mpi) if the file cannot be written
            void dump_to_file(std::string fileprefix);
            /// Load a grid written by dump_to_file (same number of tasks). Aborts (assert_mpi) if the file
            /// cannot be read, is truncated or does not match the current domain decomposition
            void load_from_file(std::string fileprefix);

            /// Allocate matching storage with zero owned field values, preserving label/status,
            /// halos, FFTW excess allocation contents and real-row padding (not the owned field).
            FFTWGrid clone_empty() const {
                assert_mpi(Nmesh > 0, "[FFTWGrid::clone_empty] Source must be allocated");
                FFTWGrid result(Nmesh, n_extra_x_slices_left, n_extra_x_slices_right);
                result.grid_is_in_real_space = grid_is_in_real_space;
                result.add_memory_label(name);
                const ptrdiff_t begin = n_extra_x_slices_left * NmeshTotComplexSlice;
                const ptrdiff_t end = begin + Local_nx * NmeshTotComplexSlice;
                std::copy(fourier_grid_raw.begin(), fourier_grid_raw.begin() + begin, result.fourier_grid_raw.begin());
                std::copy(fourier_grid_raw.begin() + end, fourier_grid_raw.end(),
                          result.fourier_grid_raw.begin() + end);
                if (grid_is_in_real_space) {
                    const auto *source = reinterpret_cast<const FloatType *>(fourier_grid_raw.data()) + 2 * begin;
                    auto *target = result.get_real_grid();
                    const ptrdiff_t stride = 2 * (Nmesh / 2 + 1);
                    for (ptrdiff_t row = 0; row < 2 * (end - begin); row += stride)
                        std::copy(source + row + Nmesh, source + row + stride, target + row + Nmesh);
                }
                if constexpr (FML::PERFORMANCE::enabled) {
                    size_t copied = (fourier_grid_raw.size() - size_t(end - begin)) * sizeof(ComplexType);
                    if (grid_is_in_real_space)
                        copied += size_t(2 * (end - begin) / (2 * (Nmesh / 2 + 1))) *
                                  size_t(2 * (Nmesh / 2 + 1) - Nmesh) * sizeof(FloatType);
                    FML::PERFORMANCE::grid_copy(copied);
                }
                return result;
            }

            void reallocate(int Nmesh, int nleft, int nright) { *this = FFTWGrid(Nmesh, nleft, nright); }

#ifdef USE_GSL
            // For making a spline that we can later use to evaluate a function of the magnitude of the fourier vector
            // in each cell
            FML::INTERPOLATION::SPLINE::Spline make_fourier_spline(std::function<double(double)> function_of_kBox,
                                                                   std::string label,
                                                                   int sampling_factor = 4) const;
#else
            // If we don't have GSL we simply return the function. This way we can use the same method in both cases
            // without having a lot of ifdefs
            std::function<double(double)> make_fourier_spline(std::function<double(double)> function_of_kBox,
                                                              [[maybe_unused]] std::string label,
                                                              [[maybe_unused]] int sampling_factor = 4) const {
                return function_of_kBox;
            }
#endif
        };

        template <int N>
        FFTWGrid<N>::operator bool() const {
            return fourier_grid_raw.size() > 0;
        }

        template <int N>
        void FFTWGrid<N>::add_memory_label([[maybe_unused]] std::string label) {
            name = label;
#ifdef MEMORY_LOGGING
            FML::MemoryLog::get()->add_label(fourier_grid_raw.data(),
                                             fourier_grid_raw.capacity() * sizeof(ComplexType),
                                             label);
#endif
        }

        // Perform a real-to-complex FFT from one grid to another
        template <int N>
        void fftw_r2c(FFTWGrid<N> & in_grid, FFTWGrid<N> & out_grid);

        // Perform a complex-to-real FFT from one grid to another
        template <int N>
        void fftw_c2r(FFTWGrid<N> & in_grid, FFTWGrid<N> & out_grid);

        //===================================================================================
        // For range based loop over the real grid
        // For In-Place FFTW arrays there are 2 extra cells per dimension in the last dimension
        // that we need to skip when looping through all the real cells
        //===================================================================================

        /// An iterator that deal with looping through a real grid. The real grid has padding so we need to skip
        /// some cells when looping through the real grid
        class LoopIteratorReal {
          private:
            IndexIntType real_index, index;
            int Nmesh, odd;

          public:
            LoopIteratorReal(IndexIntType _index, int _Nmesh)
                : real_index(_index + (_Nmesh % 2 == 1 ? 1 : 2) * (_index / _Nmesh)), index(_index), Nmesh(_Nmesh),
                  odd(_Nmesh % 2) {}
            bool operator!=(LoopIteratorReal const & other) const { return index != other.index; }
            const IndexIntType & operator*() const { return real_index; }
            LoopIteratorReal & operator++() {
                ++index;
                if (index % Nmesh == 0)
                    real_index += odd ? 1 : 2;
                ++real_index;
                return *this;
            }
        };

        /// For range based for-loops over the real-grid. Loop over all local cells.
        class RealRange {
          private:
            const IndexIntType from, to;
            const int Nmesh;

          public:
            RealRange(IndexIntType _from, IndexIntType _to, int _Nmesh) : from(_from), to(_to), Nmesh(_Nmesh) {}
            LoopIteratorReal begin() const { return {from, Nmesh}; }
            LoopIteratorReal end() const { return {to, Nmesh}; }
        };

        /// An iterator that deal with looping through the fourier grid
        class LoopIteratorFourier {
          private:
            IndexIntType index;

          public:
            LoopIteratorFourier(IndexIntType _index) : index(_index) {}
            bool operator!=(LoopIteratorFourier const & other) const { return index != other.index; }
            IndexIntType const & operator*() const { return index; }
            LoopIteratorFourier & operator++() {
                ++index;
                return *this;
            }
        };

        /// For range based for-loops over the fourier-grid. Loop over all local cells.
        class FourierRange {
          private:
            const IndexIntType from, to;

          public:
            FourierRange(IndexIntType _from, IndexIntType _to) : from(_from), to(_to) {}
            LoopIteratorFourier begin() const { return {from}; }
            LoopIteratorFourier end() const { return {to}; }
        };

        template <int N>
        RealRange FFTWGrid<N>::get_real_range(int islice_begin, int islice_end) const {

            // If fiducial parameters are used then we loop over all cells
            if (islice_begin == 0 and islice_end == 0)
                islice_end = int(Local_nx);

#ifdef DEBUG_FFTWGRID
            if (not grid_is_in_real_space) {
                if (FML::ThisTask == 0)
                    std::cout << "Warning: [FFTWGrid::get_real_range] The grid status is [Fourierspace]. Label: " +
                                     name + "\n";
            }
#endif
            // Here NmeshTotReal = LocalNx * Nmesh^N-1
            IndexIntType cellsperslice = NmeshTotReal / Local_nx;
            return RealRange(cellsperslice * islice_begin, cellsperslice * islice_end, Nmesh);
        }

        template <int N>
        FourierRange FFTWGrid<N>::get_fourier_range(int islice_begin, int islice_end) const {

            // If fiducial parameters are used then we loop over all cells
            if (islice_begin == 0 and islice_end == 0)
                islice_end = int(Local_nx);

#ifdef DEBUG_FFTWGRID
            if (grid_is_in_real_space) {
                if (FML::ThisTask == 0)
                    std::cout << "Warning: [FFTWGrid::get_fourier_range] The grid status is [Realspace]. Label: " +
                                     name + "\n";
            }
#endif

            IndexIntType cellsperslice = FML::power(Nmesh, N - 2) * IndexIntType(Nmesh / 2 + 1);
            return FourierRange(cellsperslice * islice_begin, cellsperslice * islice_end);
            // return FourierRange(0, NmeshTotComplex);
        }

        template <int N>
        FloatType * FFTWGrid<N>::get_real_grid_left() {
            return reinterpret_cast<FloatType *>(fourier_grid_raw.data());
        }

        template <int N>
        FloatType * FFTWGrid<N>::get_real_grid() {
            return get_real_grid_left() + NmeshTotRealSlice * n_extra_x_slices_left;
        }

        template <int N>
        FloatType * FFTWGrid<N>::get_real_grid_by_slice(int slice) {
#ifdef BOUNDSCHECK_FFTWGRID
            assert_mpi(-n_extra_x_slices_left <= slice and slice < Local_nx + n_extra_x_slices_right,
                       "[FFTWGrid::get_real_grid] Bounds check failed\n");
#endif
            return get_real_grid_left() + NmeshTotRealSlice * (n_extra_x_slices_left + slice);
        }

        template <int N>
        FloatType * FFTWGrid<N>::get_real_grid_right() {
            return get_real_grid_left() + NmeshTotRealSlice * (n_extra_x_slices_left + Local_nx);
        }

        template <int N>
        ComplexType * FFTWGrid<N>::get_fourier_grid() {
            return fourier_grid_raw.data() + NmeshTotComplexSlice * n_extra_x_slices_left;
        }

        template <int N>
        void FFTWGrid<N>::set_grid_status_real(bool grid_is_a_real_grid) {
            grid_is_in_real_space = grid_is_a_real_grid;
        }

        template <int N>
        bool FFTWGrid<N>::get_grid_status_real() {
            return grid_is_in_real_space;
        }

        template <int N>
        void FFTWGrid<N>::info() {
            if (FML::ThisTask > 0)
                return;
            std::string myfloattype = "[Unknown]";
            std::string status = grid_is_in_real_space ? "[Realspace]" : "[Fourierspace]";
            if (sizeof(FloatType) == sizeof(float))
                myfloattype = "[Float]";
            if (sizeof(FloatType) == sizeof(double))
                myfloattype = "[Double]";
            if (sizeof(FloatType) == sizeof(long double))
                myfloattype = "[Long Double]";
            double memory_in_mb = double(fourier_grid_raw.size() * sizeof(ComplexType)) / 1e6;
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
                std::cout << "# Info about FFTWGrid. Grid label: [" << name << "]\n";
                std::cout << "# Status of grid: " << status << " NDIM: [" << N << "] FloatType: " << myfloattype
                          << "\n";
                std::cout << "# Grid has allocated " << memory_in_mb << " MB\n";
                std::cout << "# Nmesh                  " << Nmesh << "\n";
                std::cout << "# Local_nx               " << Local_nx << "\n";
                std::cout << "# n_extra_x_slices_left  " << n_extra_x_slices_left << "\n";
                std::cout << "# n_extra_x_slices_right " << n_extra_x_slices_right << "\n";
                std::cout << "# NmeshTotComplexAlloc   " << NmeshTotComplexAlloc << "\n";
                std::cout << "# NmeshTotComplex        " << NmeshTotComplex << "\n";
                std::cout << "# NmeshTotComplexSlice   " << NmeshTotComplexSlice << "\n";
                std::cout << "# NmeshTotRealAlloc      " << NmeshTotRealAlloc << "\n";
                std::cout << "# NmeshTotReal           " << NmeshTotReal << "\n";
                std::cout << "# NmeshTotRealSlice      " << NmeshTotRealSlice << "\n";
                std::cout << "#\n";
                std::cout << "#=====================================================\n";
                std::cout << "\n";
            }
        }

        // Make FFTW plans with FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE
        template <int N>
        void FFTWGrid<N>::create_wisdom([[maybe_unused]] int planner_flag, [[maybe_unused]] int nthreads) {
#ifdef USE_FFTW
            if (planner_flag == FFTW_ESTIMATE)
                return;
#ifdef USE_FFTW_THREADS
            set_fftw_nthreads(nthreads);
#endif
#ifdef DEBUG_FFTWGRID
            if (FML::ThisTask == 0) {
                std::cout << "[FFTWGrid::create_wisdom] Planning flag " << planner_flag << ". Label: " + name + "\n";
            }
#endif

#ifdef USE_MPI
            std::vector<ptrdiff_t> NmeshPerDim(N, Nmesh);
            my_fftw_plan plan_r2c =
                MAKE_PLAN_R2C(N, NmeshPerDim.data(), get_real_grid(), get_fftw_grid(), FML::communicator, planner_flag);
#else
            std::vector<int> NmeshPerDim(N, Nmesh);
            my_fftw_plan plan_r2c =
                MAKE_PLAN_R2C(N, NmeshPerDim.data(), get_real_grid(), get_fftw_grid(), planner_flag);
#endif
            if (FML::ThisTask == 0)
                std::cout << "[FFTWGrid::create_wisdom] Warning this will clear data in the grid. Label: " + name +
                                 "\n";
            DESTROY_PLAN(plan_r2c);
#endif
        }

        template <int N>
        void FFTWGrid<N>::load_wisdom([[maybe_unused]] std::string filename) const {
#ifdef USE_FFTW
#ifdef DEBUG_FFTWGRID
            if (FML::ThisTask == 0) {
                std::cout << "[FFTWGrid::load_wisdom] Filename " << filename << ". Label: " + name + "\n";
            }
#endif
            if (FML::ThisTask == 0)
                IMPORT_WISDOM_FROM_FILENAME(filename.c_str());
#ifdef USE_MPI
            MPI_BROADCAST_WISDOM(FML::communicator);
#endif
#endif
        }

        template <int N>
        void FFTWGrid<N>::save_wisdom([[maybe_unused]] std::string filename) const {
#ifdef USE_FFTW
#ifdef USE_MPI
            MPI_GATHER_WISDOM(FML::communicator);
#endif
            if (FML::ThisTask == 0)
                EXPORT_WISDOM_TO_FILENAME(filename.c_str());
#ifdef DEBUG_FFTWGRID
            if (FML::ThisTask == 0) {
                std::cout << "[FFTWGrid::save_wisdom] Filename " << filename << ". Label: " + name + "\n";
            }
#endif
#endif
        }

#ifdef USE_FFTW
        template <int N>
        my_fftw_plan FFTWGrid<N>::get_fftw_plan(bool r2c) {
            const unsigned int flag = FML::GRID::get_fftw_planner_flag();
            const FML::GRID::FFTWPlanKey key{N,
                                             Nmesh,
                                             r2c ? 1 : -1,
                                             FML::GRID::get_fftw_nthreads(),
                                             ALIGNMENT_OF(get_real_grid()),
                                             flag};
            if (auto * plan = FML::GRID::find_cached_fftw_plan(key))
                return *plan;

            // Planning with anything but FFTW_ESTIMATE overwrites the arrays so keep a copy
            PayloadStorage backup;
            if (flag != FFTW_ESTIMATE)
                backup = fourier_grid_raw;

#ifdef USE_MPI
            std::vector<ptrdiff_t> NmeshPerDim(N, Nmesh);
            my_fftw_plan plan =
                r2c ? MAKE_PLAN_R2C(N, NmeshPerDim.data(), get_real_grid(), get_fftw_grid(), FML::communicator, flag) :
                      MAKE_PLAN_C2R(N, NmeshPerDim.data(), get_fftw_grid(), get_real_grid(), FML::communicator, flag);
#else
            std::vector<int> NmeshPerDim(N, Nmesh);
            my_fftw_plan plan = r2c ? MAKE_PLAN_R2C(N, NmeshPerDim.data(), get_real_grid(), get_fftw_grid(), flag) :
                                      MAKE_PLAN_C2R(N, NmeshPerDim.data(), get_fftw_grid(), get_real_grid(), flag);
#endif
            assert_mpi(plan != nullptr, "[FFTWGrid::get_fftw_plan] FFTW failed to create a plan\n");
            FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::FFTPlans);

            // Restore the data in place: the buffer must keep its address (callers may hold pointers into it)
            // and its alignment (the plan is cached for this alignment)
            if (flag != FFTW_ESTIMATE) {
                std::copy(backup.begin(), backup.end(), fourier_grid_raw.begin());
                FML::PERFORMANCE::grid_copy(backup.size() * sizeof(ComplexType));
            }

            FML::GRID::store_fftw_plan(key, plan);
            return plan;
        }
#endif

        template <int N>
        ptrdiff_t FFTWGrid<N>::get_ntot_real_slice_alloc() const {
            return NmeshTotRealSlice;
        }

        template <int N>
        void FFTWGrid<N>::fill_real_grid(const FloatType val) {
#ifdef DEBUG_FFTWGRID
            if (not grid_is_in_real_space) {
                if (FML::ThisTask == 0)
                    std::cout << "Warning: [FFTWGrid::fill_real_grid] The grid status is [Fourierspace]. Label: " +
                                     name + "\n";
            }
#endif
            FloatType * begin = (FloatType *)fourier_grid_raw.data();
            FloatType * end = begin + NmeshTotRealAlloc;
            std::fill(begin, end, val);
            grid_is_in_real_space = true;
        }

        template <int N>
        void FFTWGrid<N>::fill_real_grid(const std::function<FloatType(std::array<double, N> &)> & func) {
#ifdef DEBUG_FFTWGRID
            if (not grid_is_in_real_space) {
                if (FML::ThisTask == 0)
                    std::cout << "Warning: [FFTWGrid::fill_real_grid] The grid status is [Fourierspace]. Label: " +
                                     name + "\n";
            }
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && real_index : get_real_range(islice, islice + 1)) {
                    auto coord = get_coord_from_index(real_index);
                    auto pos = get_real_position(coord);
                    auto value = func(pos);
                    set_real_from_index(real_index, value);
                }
            }
            grid_is_in_real_space = true;

            communicate_boundaries();
        }

        template <int N>
        void FFTWGrid<N>::fill_fourier_grid(const ComplexType val) {
#ifdef DEBUG_FFTWGRID
            if (grid_is_in_real_space) {
                if (FML::ThisTask == 0)
                    std::cout << "Warning: [FFTWGrid::fill_fourier_grid] The grid status is [Realspace]. Label: " +
                                     name + "\n";
            }
#endif
            std::fill(fourier_grid_raw.begin(), fourier_grid_raw.end(), val);
            grid_is_in_real_space = false;
        }

        template <int N>
        void FFTWGrid<N>::fill_fourier_grid(const std::function<ComplexType(std::array<double, N> &)> & func) {
#ifdef DEBUG_FFTWGRID
            if (grid_is_in_real_space) {
                if (FML::ThisTask == 0)
                    std::cout << "Warning: [FFTWGrid::fill_fourier_grid] The grid status is [Realspace]. Label: " +
                                     name + "\n";
            }
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && fourier_index : get_fourier_range(islice, islice + 1)) {
                    auto kvec = get_fourier_wavevector_from_index(fourier_index);
                    auto value = func(kvec);
                    set_fourier_from_index(fourier_index, value);
                }
            }
            grid_is_in_real_space = false;
        }

#ifdef USE_MPI
        template <int N>
        void FFTWGrid<N>::sendrecv_bytes(char * sendbuf, char * recvbuf, size_t bytes, int dest, int source) {
            // MPI counts are int so send large buffers in chunks (a slice can be > 2GB for huge grids)
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

        // We copy over slices. If we have more extra slices than local slices (thin slabs) then the slices
        // beyond the local ones are taken from the neighbors extra slices, which have already been received at that
        // point (the right slices are filled from the nearest one and outwards, the left ones likewise), so the
        // ghosts always hold the (periodic) neighbor slices no matter how many we have. This relies on all tasks
        // having the same Local_nx (true as we require NTasks to divide Nmesh)
        template <int N>
        void FFTWGrid<N>::communicate_boundaries() {
            const int n_to_recv_right = n_extra_x_slices_right;
            const int n_to_recv_left = n_extra_x_slices_left;
            if (n_to_recv_right + n_to_recv_left == 0)
                return;
            assert_mpi(Local_nx > 0, "[FFTWGrid::communicate_boundaries] Grid is not allocated\n");

#ifdef DEBUG_FFTWGRID
            if (FML::ThisTask == 0) {
                std::cout << "[FFTWGrid::communicate_boundaries] Receiving " << n_to_recv_right
                          << " from the right and " << n_to_recv_left << " slices from the left. Label: " + name + "\n";
            }
#endif

#ifdef USE_MPI
            int rightcpu = (FML::ThisTask + 1) % FML::NTasks;
            int leftcpu = (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks;
#endif
            const size_t bytes_slice = size_t(NmeshTotRealSlice) * sizeof(FloatType);

            // Slice i to the right of the main grid is slice i of the right task (or for i >= Local_nx its extra
            // slice i - Local_nx which was received in iteration i - Local_nx)
            for (int i = 0; i < n_to_recv_right; i++) {
                FloatType * slice_left_tosend = get_real_grid() + NmeshTotRealSlice * (i);
                FloatType * slice_right_torecv = get_real_grid_right() + NmeshTotRealSlice * (i);
                char * sendbuf = reinterpret_cast<char *>(slice_left_tosend);
                char * recvbuf = reinterpret_cast<char *>(slice_right_torecv);
#ifdef USE_MPI
                sendrecv_bytes(sendbuf, recvbuf, bytes_slice, leftcpu, rightcpu);
#else
                std::memcpy(recvbuf, sendbuf, bytes_slice);
#endif
            }

            // Slice -1-i is slice Local_nx-1-i of the left task (or for i >= Local_nx its extra slice
            // Local_nx-1-i < 0 which was received in iteration i - Local_nx)
            for (int i = 0; i < n_to_recv_left; i++) {
                FloatType * slice_right_tosend = get_real_grid() + NmeshTotRealSlice * (Local_nx - 1 - i);
                FloatType * slice_left_torecv =
                    get_real_grid_left() + NmeshTotRealSlice * (n_extra_x_slices_left - 1 - i);
                char * sendbuf = reinterpret_cast<char *>(slice_right_tosend);
                char * recvbuf = reinterpret_cast<char *>(slice_left_torecv);
#ifdef USE_MPI
                sendrecv_bytes(sendbuf, recvbuf, bytes_slice, rightcpu, leftcpu);
#else
                std::memcpy(recvbuf, sendbuf, bytes_slice);
#endif
            }
        }

        template <int N>
        ptrdiff_t FFTWGrid<N>::compute_local_size(int Nmesh, ptrdiff_t & Local_nx, ptrdiff_t & Local_x_start) {
            ptrdiff_t NmeshTotComplex = 0;
            std::vector<ptrdiff_t> NmeshPerDimFourier(N, Nmesh);
            NmeshPerDimFourier[N - 1] = Nmesh / 2 + 1;
#ifdef USE_MPI
#ifdef USE_FFTW
            NmeshTotComplex =
                MPI_FFTW_LOCAL_SIZE(N, NmeshPerDimFourier.data(), FML::communicator, &Local_nx, &Local_x_start);
#else
            // If we don't have FFTW, but want to use this class
            Local_nx = Nmesh / FML::NTasks;
            Local_x_start = FML::ThisTask * Local_nx;
            NmeshTotComplex = ptrdiff_t(Nmesh / 2 + 1) * ptrdiff_t(FML::power(Nmesh, N - 1)) / ptrdiff_t(FML::NTasks);
#endif
#else
            Local_nx = Nmesh;
            Local_x_start = 0;
            NmeshTotComplex = 1;
            for (int i = 0; i < N; i++)
                NmeshTotComplex *= NmeshPerDimFourier[i];
#endif
            return NmeshTotComplex;
        }

        template <int N>
        ptrdiff_t FFTWGrid<N>::get_nreal_right_to_backup() const {
            if (n_extra_x_slices_right == 0)
                return 0;
            // FFTW may use all of its local array (NmeshTotComplex complex numbers) in an in-place transform. With MPI
            // this can be larger than the Local_nx slices we use (e.g. in 2D when the transposed dimension
            // Nmesh/2+1 is not divisible by NTasks) and then the excess overlaps with the right extra slices.
            // We also always keep the first Nmesh/2+1 reals (as has always been done)
            const ptrdiff_t nreal_right = NmeshTotRealSlice * n_extra_x_slices_right;
            const ptrdiff_t nreal_excess = 2 * (NmeshTotComplex - Local_nx * NmeshTotComplexSlice);
            return std::min(nreal_right, std::max(ptrdiff_t(Nmesh / 2 + 1), nreal_excess));
        }

        template <int N>
        FFTWGrid<N>::FFTWGrid(FFTWGrid && rhs) noexcept {
            *this = std::move(rhs);
        }

        template <int N>
        FFTWGrid<N> & FFTWGrid<N>::operator=(FFTWGrid && rhs) noexcept {
            if (this == &rhs)
                return *this;
            // Take over the buffer (keeps its address and alignment) and leave rhs empty
            fourier_grid_raw = std::move(rhs.fourier_grid_raw);
            Vector<ComplexType>().swap(rhs.fourier_grid_raw);
            Nmesh = std::exchange(rhs.Nmesh, 0);
            Local_nx = std::exchange(rhs.Local_nx, 0);
            Local_x_start = std::exchange(rhs.Local_x_start, 0);
            NmeshTotRealAlloc = std::exchange(rhs.NmeshTotRealAlloc, 0);
            NmeshTotComplexAlloc = std::exchange(rhs.NmeshTotComplexAlloc, 0);
            NmeshTotReal = std::exchange(rhs.NmeshTotReal, 0);
            NmeshTotComplex = std::exchange(rhs.NmeshTotComplex, 0);
            NmeshTotRealSlice = std::exchange(rhs.NmeshTotRealSlice, 0);
            NmeshTotComplexSlice = std::exchange(rhs.NmeshTotComplexSlice, 0);
            n_extra_x_slices_left = std::exchange(rhs.n_extra_x_slices_left, 0);
            n_extra_x_slices_right = std::exchange(rhs.n_extra_x_slices_right, 0);
            grid_is_in_real_space = std::exchange(rhs.grid_is_in_real_space, true);
            name = std::move(rhs.name);
            rhs.name.clear();
            return *this;
        }

        template <int N>
        FFTWGrid<N>::FFTWGrid(int Nmesh, int n_extra_x_slices_left, int n_extra_x_slices_right)
            : Nmesh(Nmesh), Local_nx(Nmesh), n_extra_x_slices_left(n_extra_x_slices_left),
              n_extra_x_slices_right(n_extra_x_slices_right) {

#ifdef USE_MPI
            // FFTW with MPI not implemented in FFTW for Ndim = 1 so abort
            assert_mpi(N > 1, "[FFTWGrid] FFTW r2c and c2r with MPI currently not supported for 1D\n");
            assert_mpi(Nmesh % FML::NTasks == 0,
                       "[FFTWGrid] The number of CPUs should divide the gridsize. Otherwise there might be issues with "
                       "extra padding in FFTW and in they way we divide the domain\n");
#endif

            // Total number of complex gridcells in the local grid
            NmeshTotComplex = compute_local_size(Nmesh, Local_nx, Local_x_start);

            // Total number of real gridcells in the local grid.
            // NB: This will differ from the number of real cells we allocate!
            NmeshTotReal = ptrdiff_t(Local_nx) * ptrdiff_t(FML::power(Nmesh, N - 1));
            // Number of cells for one extra slice
            NmeshTotComplexSlice = ptrdiff_t(Nmesh / 2 + 1) * ptrdiff_t(FML::power(Nmesh, N - 2));
            NmeshTotRealSlice = 2 * NmeshTotComplexSlice;
            // Total number of grid-nodes we allocate including extra slices
            NmeshTotComplexAlloc =
                NmeshTotComplex + NmeshTotComplexSlice * (n_extra_x_slices_left + n_extra_x_slices_right);
            NmeshTotRealAlloc = 2 * NmeshTotComplexAlloc;

            // Allocate memory and initialize to 0
            fourier_grid_raw.resize(NmeshTotComplexAlloc);
            add_memory_label("FFTWGrid");
            std::fill(fourier_grid_raw.begin(), fourier_grid_raw.end(), 0.0);

            // Check alignment (relevant for SIMD instructions) since we don't use fftw_malloc
            // If you have SIMD and the alignment is off then changing to an alignment allocator
            // will help speed up the transforms
#ifdef USE_FFTW
            if (((unsigned long)get_fftw_grid() & (16 - 1))) {
                std::cout << "Warning: FFTWGrid is not 16 byte aligned (only relevant for speed if you have SIMD)\n";
            }
#endif

#ifdef DEBUG_FFTWGRID
            if (FML::ThisTask == 0) {
                std::cout << "[FFTWGrid] Creating grid Nmesh = " << Nmesh << " Local_nx = " << Local_nx << " n_extra: ("
                          << n_extra_x_slices_left << " + " << n_extra_x_slices_right << ")\n";
            }
#endif
        }

        template <int N>
        std::array<int, N> FFTWGrid<N>::get_coord_from_index(const IndexIntType index_real) const {
#ifdef BOUNDSCHECK_FFTWGRID
            assert_mpi(index_real >= -NmeshTotRealSlice * n_extra_x_slices_left and
                           index_real < NmeshTotRealSlice * (Local_nx + n_extra_x_slices_right),
                       "[FFTWGrid::get_coord_from_index] Bounds check failed\n");
#endif
            std::array<int, N> coord;
            int nmesh_plus_padding = (2 * (Nmesh / 2 + 1));
            IndexIntType index = index_real;
            coord[N - 1] = int(index % nmesh_plus_padding);
            index /= nmesh_plus_padding;

            if constexpr (N == 2) {
                coord[0] = index;
            } else if (N == 3) {
                coord[1] = int(index % Nmesh);
                coord[0] = int(index / Nmesh);
            } else {
                for (int idim = N - 2; idim >= 1; idim--) {
                    coord[idim] = int(index % Nmesh);
                    index /= Nmesh;
                }
                coord[0] = index;
            }
            return coord;
        }

        template <int N>
        IndexIntType FFTWGrid<N>::get_index_real(const std::array<int, N> & coord) const {
#ifdef BOUNDSCHECK_FFTWGRID
            assert_mpi(coord.size() == N, "[FFTWGrid::get_index_real] Coord has wrong size\n");
            assert_mpi(-n_extra_x_slices_left <= coord[0] and coord[0] < Local_nx + n_extra_x_slices_right,
                       "[FFTWGrid::get_index_real] Bounds check failed (first coordinate)\n");
            for (int idim = 1; idim < N; idim++) {
                assert_mpi(0 <= coord[idim] and coord[idim] < Nmesh,
                           "[FFTWGrid::get_index_real] Bounds check failed\n");
            }
#endif

            if constexpr (N == 2) {
                return IndexIntType(coord[0]) * (2 * (Nmesh / 2 + 1)) + coord[1];
            } else if (N == 3) {
                return (IndexIntType(coord[0]) * Nmesh + coord[1]) * (2 * (Nmesh / 2 + 1)) + coord[2];
            } else {

                IndexIntType index = coord[0];
                for (int idim = 1; idim < N - 1; idim++) {
                    index = index * Nmesh + coord[idim];
                }
                index = index * (2 * (Nmesh / 2 + 1)) + coord[N - 1];
                return index;
            }
            return 0;
        }

        template <int N>
        IndexIntType FFTWGrid<N>::get_index_fourier(const std::array<int, N> & coord) const {

#ifdef BOUNDSCHECK_FFTWGRID
            assert_mpi(0 <= coord[0] and coord[0] < Local_nx, "[FFTWGrid::get_index_fourier] Bounds check failed\n");
            for (int idim = 1; idim < N; idim++) {
                assert_mpi(0 <= coord[idim] and coord[idim] < Nmesh,
                           "[FFTWGrid::get_index_fourier] Bounds check failed\n");
            }
#endif

            if constexpr (N == 2) {
                return IndexIntType(coord[0]) * (Nmesh / 2 + 1) + coord[1];
            } else if (N == 3) {
                return (IndexIntType(coord[0]) * Nmesh + coord[1]) * (Nmesh / 2 + 1) + coord[2];
            } else {
                IndexIntType index = coord[0];
                for (int idim = 1; idim < N - 1; idim++) {
                    index = index * Nmesh + coord[idim];
                }
                index = index * (Nmesh / 2 + 1) + coord[N - 1];
                return index;
            }
        }

        template <int N>
        void FFTWGrid<N>::fftw_r2c() {
#ifdef USE_FFTW

#ifdef DEBUG_FFTWGRID
            if (FML::ThisTask == 0) {
                std::cout << "[FFTWGrid::fftw_r2c] Transforming grid to fourier space. Label: " + name + "\n";
            }
            if (not grid_is_in_real_space) {
                if (FML::ThisTask == 0)
                    std::cout << "Warning: [FFTWGrid::fftw_r2c] Transforming grid whose status is already "
                                 "[Fourierspace]\n";
            }
#endif

            // Plans are cached so this is only expensive the first time
            // (planning preserves the data and the buffer so we fetch pointers after this)
            my_fftw_plan plan_r2c = get_fftw_plan(true);

            //=================================================================================
            // Make a copy of the cells in the right extra slices that FFTW might use as scratch
            // space when doing the in-place FFT (see get_nreal_right_to_backup)
            //=================================================================================
            FloatType * real_grid_right = get_real_grid_right();
            const ptrdiff_t nright_copy = get_nreal_right_to_backup();
            std::vector<FloatType> right_copy(real_grid_right, real_grid_right + nright_copy);
            FML::PERFORMANCE::grid_allocation(right_copy.capacity() * sizeof(FloatType));
            FML::PERFORMANCE::grid_copy(size_t(nright_copy) * sizeof(FloatType));

            EXECUTE_R2C_NEW_ARRAY(plan_r2c, get_real_grid(), get_fftw_grid());
            FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::ForwardFFTs);
            grid_is_in_real_space = false;

            // Normalize
            const FML::GRID::FloatType norm = 1.0 / std::pow(double(Nmesh), N);
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && fourier_index : get_fourier_range(islice, islice + 1)) {
                    set_fourier_from_index(fourier_index, norm * get_fourier_from_index(fourier_index));
                }
            }

            //=================================================================================
            // Copy back data we copied
            //=================================================================================
            std::copy(right_copy.begin(), right_copy.end(), real_grid_right);
            FML::PERFORMANCE::grid_copy(size_t(nright_copy) * sizeof(FloatType));

#else
            assert_mpi(false, "[FFTWGrid::fftw_r2c] Compiled without FFTW support so cannot take Fourier transforms\n");
#endif
        }

        template <int N>
        void FFTWGrid<N>::fftw_c2r() {
#ifdef USE_FFTW

#ifdef DEBUG_FFTWGRID
            if (FML::ThisTask == 0) {
                std::cout << "[FFTWGrid::fftw_c2r] Transforming grid to real space\n";
            }
            if (grid_is_in_real_space) {
                if (FML::ThisTask == 0)
                    std::cout
                        << "Warning: [FFTWGrid::fftw_c2r] Transforming grid whose status is already [Realspace]\n";
            }
#endif

            // Plans are cached so this is only expensive the first time
            // (planning preserves the data and the buffer so we fetch pointers after this)
            my_fftw_plan plan_c2r = get_fftw_plan(false);

            //=================================================================================
            // Make a copy of the cells in the right extra slices that FFTW might use as scratch
            // space when doing the in-place FFT (see get_nreal_right_to_backup)
            //=================================================================================
            FloatType * real_grid_right = get_real_grid_right();
            const ptrdiff_t nright_copy = get_nreal_right_to_backup();
            std::vector<FloatType> right_copy(real_grid_right, real_grid_right + nright_copy);
            FML::PERFORMANCE::grid_allocation(right_copy.capacity() * sizeof(FloatType));
            FML::PERFORMANCE::grid_copy(size_t(nright_copy) * sizeof(FloatType));

            EXECUTE_C2R_NEW_ARRAY(plan_c2r, get_fftw_grid(), get_real_grid());
            FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::InverseFFTs);
            grid_is_in_real_space = true;

            //=================================================================================
            // Copy back data we copied
            //=================================================================================
            std::copy(right_copy.begin(), right_copy.end(), real_grid_right);
            FML::PERFORMANCE::grid_copy(size_t(nright_copy) * sizeof(FloatType));

#else
            assert_mpi(false, "[FFTWGrid::fftw_c2r] Compiled without FFTW support so cannot take Fourier transforms\n");
#endif
        }

        template <int N>
        FloatType FFTWGrid<N>::get_real(const std::array<int, N> & coord) const {
            IndexIntType index = get_index_real(coord);
            const FloatType * grid = reinterpret_cast<const FloatType *>(fourier_grid_raw.data()) +
                                     NmeshTotRealSlice * n_extra_x_slices_left;
            return grid[index];
        }

        template <int N>
        void FFTWGrid<N>::set_real(const std::array<int, N> & coord, const FloatType value) {
            IndexIntType index = get_index_real(coord);
            get_real_grid()[index] = value;
        }

        template <int N>
        void FFTWGrid<N>::add_real(const std::array<int, N> & coord, const FloatType value) {
            IndexIntType index = get_index_real(coord);
            get_real_grid()[index] += value;
        }

        template <int N>
        void FFTWGrid<N>::set_real_from_index(const IndexIntType index, const FloatType value) {
            get_real_grid()[index] = value;
        }

        template <int N>
        ComplexType FFTWGrid<N>::get_fourier(const std::array<int, N> & coord) const {
            IndexIntType index = get_index_fourier(coord);
            return fourier_grid_raw[NmeshTotComplexSlice * n_extra_x_slices_left + index];
        }

        template <int N>
        ComplexType FFTWGrid<N>::get_fourier_from_index(const IndexIntType index) const {
            return fourier_grid_raw[NmeshTotComplexSlice * n_extra_x_slices_left + index];
        }

        template <int N>
        void FFTWGrid<N>::set_fourier_from_index(const IndexIntType index, const ComplexType value) {
            get_fourier_grid()[index] = value;
        }

        template <int N>
        void FFTWGrid<N>::set_fourier(const std::array<int, N> & coord, const ComplexType value) {
            IndexIntType index = get_index_fourier(coord);
            get_fourier_grid()[index] = value;
        }

#ifdef USE_FFTW
        template <int N>
        my_fftw_complex * FFTWGrid<N>::get_fftw_grid() {
            return reinterpret_cast<my_fftw_complex *>(get_fourier_grid());
        }
#endif

        template <int N>
        int FFTWGrid<N>::get_nmesh() const {
            return Nmesh;
        }

        template <int N>
        int FFTWGrid<N>::get_ndim() const {
            return N;
        }

        template <int N>
        ptrdiff_t FFTWGrid<N>::get_ntot_real() const {
            return NmeshTotRealSlice * Local_nx;
        }

        template <int N>
        ptrdiff_t FFTWGrid<N>::get_ntot_fourier() const {
            return NmeshTotComplexSlice * Local_nx;
        }

        template <int N>
        ptrdiff_t FFTWGrid<N>::get_ntot_fourier_alloc() const {
            return NmeshTotComplexAlloc;
        }

        template <int N>
        ptrdiff_t FFTWGrid<N>::get_local_nx() const {
            return Local_nx;
        }

        template <int N>
        ptrdiff_t FFTWGrid<N>::get_local_x_start() const {
            return Local_x_start;
        }

        template <int N>
        std::array<double, N> FFTWGrid<N>::get_real_position(const std::array<int, N> & coord) const {
            std::array<double, N> xcoord;
#ifdef CELLCENTERSHIFTED
            const constexpr double shift = 0.5;
#else
            const constexpr double shift = 0.0;
#endif
            xcoord[0] = (double(Local_x_start + coord[0]) + shift) / double(Nmesh);
            for (int idim = 1; idim < N; idim++)
                xcoord[idim] = (coord[idim] + shift) / double(Nmesh);
            return xcoord;
        }

        template <int N>
        void FFTWGrid<N>::get_fourier_wavevector_and_norm_by_index(const IndexIntType index,
                                                                   std::array<double, N> & kvec,
                                                                   double & kmag) const {
            get_fourier_wavevector_and_norm2_by_index(index, kvec, kmag);
            kmag = std::sqrt(kmag);
        }

        template <int N>
        std::array<int, N> FFTWGrid<N>::get_fourier_coord_from_index(const IndexIntType index) const {
            const int nover2plus1 = Nmesh / 2 + 1;
            std::array<int, N> coord;
            coord[N - 1] = index % nover2plus1;
            IndexIntType n = nover2plus1;
            for (int idim = N - 2; idim >= 0; idim--, n *= Nmesh) {
                coord[idim] = (index / n) % Nmesh;
            }
            return coord;
        }

        template <int N>
        void FFTWGrid<N>::get_fourier_wavevector_and_norm2_by_index(const IndexIntType index,
                                                                    std::array<double, N> & kvec,
                                                                    double & kmag2) const {
            const double twopi = 2.0 * M_PI;
            const int nover2plus1 = Nmesh / 2 + 1;
            [[maybe_unused]] const int nover2 = Nmesh / 2;

#ifdef BOUNDSCHECK_FFTWGRID
            assert_mpi(kvec.size() == N,
                       "[FFTWGrid::get_fourier_wavevector_and_norm2_by_index] kvec has the wrong size\n");
#endif

            if (N == 3) {
                auto iz = index % nover2plus1;
                auto iy = index / nover2plus1;
                auto ix = (iy / Nmesh) + Local_x_start;
                iy = iy % Nmesh;
                kvec[0] = twopi * double(ix <= nover2 ? ix : ix - Nmesh);
                kvec[1] = twopi * double(iy <= nover2 ? iy : iy - Nmesh);
                kvec[2] = twopi * double(iz <= nover2 ? iz : iz - Nmesh);
                kmag2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
            } else if (N == 2) {
                auto iy = index % nover2plus1;
                auto ix = (index / nover2plus1) + Local_x_start;
                kvec[0] = twopi * double(ix <= nover2 ? ix : ix - Nmesh);
                kvec[1] = twopi * double(iy <= nover2 ? iy : iy - Nmesh);
                kmag2 = kvec[0] * kvec[0] + kvec[1] * kvec[1];
            } else if (N == 1) {
                auto ix = index + Local_x_start;
                kvec[0] = twopi * double(ix <= nover2 ? ix : ix - Nmesh);
                kmag2 = kvec[0] * kvec[0];
            } else {
                std::array<int, N> coord;
                coord[N - 1] = index % nover2plus1;
                IndexIntType n = nover2plus1;
                for (int idim = N - 2; idim >= 0; idim--, n *= Nmesh) {
                    coord[idim] = (index / n) % Nmesh;
                }
                coord[0] += Local_x_start;
                kmag2 = 0.0;
                for (int idim = 0; idim < N; idim++) {
                    kvec[idim] = twopi * double(coord[idim] <= nover2 ? coord[idim] : coord[idim] - Nmesh);
                    kmag2 += kvec[idim] * kvec[idim];
                }
            }
        }

        template <int N>
        std::array<double, N> FFTWGrid<N>::get_fourier_wavevector(const std::array<int, N> & coord) const {
            const double twopi = 2.0 * M_PI;
            std::array<double, N> fcoord;
            fcoord[0] = coord[0] + Local_x_start;
            fcoord[0] = twopi * (fcoord[0] <= Nmesh / 2 ? fcoord[0] : fcoord[0] - Nmesh);
            for (int idim = 1; idim < N; idim++)
                fcoord[idim] = twopi * (coord[idim] <= Nmesh / 2 ? coord[idim] : coord[idim] - Nmesh);
            return fcoord;
        }

        template <int N>
        std::array<double, N> FFTWGrid<N>::get_fourier_wavevector_from_index(const IndexIntType index) const {
            const double twopi = 2.0 * M_PI;
            const int nover2plus1 = Nmesh / 2 + 1;
            const int nover2 = Nmesh / 2;
            std::array<double, N> fcoord;

            if constexpr (N == 3) {
                auto iz = index % nover2plus1;
                auto iy = index / nover2plus1;
                auto ix = (iy / Nmesh) + Local_x_start;
                iy = iy % Nmesh;
                fcoord[0] = twopi * double(ix <= nover2 ? ix : ix - Nmesh);
                fcoord[1] = twopi * double(iy <= nover2 ? iy : iy - Nmesh);
                fcoord[2] = twopi * double(iz <= nover2 ? iz : iz - Nmesh);
            } else if (N == 2) {
                auto iy = index % nover2plus1;
                auto ix = (index / nover2plus1) + Local_x_start;
                fcoord[0] = twopi * double(ix <= nover2 ? ix : ix - Nmesh);
                fcoord[1] = twopi * double(iy <= nover2 ? iy : iy - Nmesh);
            } else if (N == 1) {
                auto ix = (index % nover2plus1) + Local_x_start;
                fcoord[0] = twopi * double(ix <= nover2 ? ix : ix - Nmesh);
            } else {
                fcoord[N - 1] = index % nover2plus1;
                IndexIntType n = nover2plus1;
                for (int idim = N - 2; idim >= 0; idim--, n *= Nmesh) {
                    fcoord[idim] = double((index / n) % Nmesh);
                }
                fcoord[0] += Local_x_start;
                for (int idim = 0; idim < N; idim++) {
                    fcoord[idim] = twopi * double(fcoord[idim] <= nover2 ? fcoord[idim] : fcoord[idim] - Nmesh);
                }
            }
            return fcoord;
        }

        template <int N>
        std::array<int, N> FFTWGrid<N>::get_fourier_integer_wavevector_from_index(const IndexIntType index) const {
            const int nover2plus1 = Nmesh / 2 + 1;
            const int nover2 = Nmesh / 2;
            std::array<int, N> fcoord;

            if constexpr (N == 3) {
                auto iz = index % nover2plus1;
                auto iy = index / nover2plus1;
                auto ix = (iy / Nmesh) + Local_x_start;
                iy = iy % Nmesh;
                fcoord[0] = (ix <= nover2 ? ix : ix - Nmesh);
                fcoord[1] = (iy <= nover2 ? iy : iy - Nmesh);
                fcoord[2] = (iz <= nover2 ? iz : iz - Nmesh);
            } else if (N == 2) {
                auto iy = index % nover2plus1;
                auto ix = (index / nover2plus1) + Local_x_start;
                fcoord[0] = (ix <= nover2 ? ix : ix - Nmesh);
                fcoord[1] = (iy <= nover2 ? iy : iy - Nmesh);
            } else if (N == 1) {
                auto ix = (index % nover2plus1) + Local_x_start;
                fcoord[0] = (ix <= nover2 ? ix : ix - Nmesh);
            } else {
                fcoord[N - 1] = index % nover2plus1;
                IndexIntType n = nover2plus1;
                for (int idim = N - 2; idim >= 0; idim--, n *= Nmesh) {
                    fcoord[idim] = ((index / n) % Nmesh);
                }
                fcoord[0] += Local_x_start;
                for (int idim = 0; idim < N; idim++) {
                    fcoord[idim] = (fcoord[idim] <= nover2 ? fcoord[idim] : fcoord[idim] - Nmesh);
                }
            }
            return fcoord;
        }

        template <int N>
        void fftw_c2r(FFTWGrid<N> & in_grid, FFTWGrid<N> & out_grid) {
#ifdef DEBUG_FFTWGRID
            if (FML::ThisTask == 0) {
                std::cout << "[fftw_c2r] Transforming grid to real space\n";
            }
#endif
            out_grid = in_grid;
            out_grid.fftw_c2r();
        }

        template <int N>
        void fftw_r2c(FFTWGrid<N> & in_grid, FFTWGrid<N> & out_grid) {
#ifdef DEBUG_FFTWGRID
            if (FML::ThisTask == 0) {
                std::cout << "[fftw_r2c] Transforming grid to real space\n";
            }
#endif
            out_grid = in_grid;
            out_grid.fftw_r2c();
        }

        template <int N>
        void FFTWGrid<N>::free() {
            // Reset to an empty grid (releases the memory), but keep the label
            std::string label = name;
            *this = FFTWGrid<N>();
            name = label;
        }

        template <int N>
        FloatType FFTWGrid<N>::get_real_from_index(const IndexIntType index) const {
            const FloatType * grid = reinterpret_cast<const FloatType *>(fourier_grid_raw.data()) +
                                     NmeshTotRealSlice * n_extra_x_slices_left;
            return grid[index];
        }

        template <int N>
        int FFTWGrid<N>::get_n_extra_slices_left() const {
            return n_extra_x_slices_left;
        }

        template <int N>
        int FFTWGrid<N>::get_n_extra_slices_right() const {
            return n_extra_x_slices_right;
        }

        template <int N>
        bool FFTWGrid<N>::nan_in_grids() const {
            bool found = false;
            for (ptrdiff_t i = 0; i < NmeshTotComplexAlloc; i++) {
                const ComplexType & z = fourier_grid_raw[i];
                if (not std::isfinite(z.real()) or not std::isfinite(z.imag())) {
                    std::cout << "[FFTWGrid::nan_in_grids] Found NaN or Inf in grid. Index = " << i << "\n";
                    found = true;
                    break;
                }
            }
            return found;
        }

        template <int N>
        void FFTWGrid<N>::dump_to_file(std::string fileprefix) {
            std::ios_base::sync_with_stdio(false);
            std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
            auto myfile = std::fstream(filename, std::ios::out | std::ios::binary);

            // If we fail to open the file throw an error (a checkpoint we cannot read back is useless)
            if (not myfile.good()) {
                std::string error = "[FFTWGrid::dump_to_file] Failed to open file for writing on task " +
                                    std::to_string(FML::ThisTask) + " Filename: " + filename + "\n";
                assert_mpi(false, error.c_str());
            }

            // Write header data
            int NDIM = N;
            myfile.write((char *)&NDIM, sizeof(NDIM));
            myfile.write((char *)&Nmesh, sizeof(Nmesh));
            myfile.write((char *)&n_extra_x_slices_left, sizeof(n_extra_x_slices_left));
            myfile.write((char *)&n_extra_x_slices_right, sizeof(n_extra_x_slices_right));
            myfile.write((char *)&Local_nx, sizeof(Local_nx));
            myfile.write((char *)&Local_x_start, sizeof(Local_x_start));
            myfile.write((char *)&NmeshTotComplexAlloc, sizeof(NmeshTotComplexAlloc));
            myfile.write((char *)&NmeshTotRealAlloc, sizeof(NmeshTotRealAlloc));
            myfile.write((char *)&NmeshTotComplex, sizeof(NmeshTotComplex));
            myfile.write((char *)&NmeshTotComplexSlice, sizeof(NmeshTotComplexSlice));
            myfile.write((char *)&NmeshTotRealSlice, sizeof(NmeshTotRealSlice));
            myfile.write((char *)&grid_is_in_real_space, sizeof(grid_is_in_real_space));

            // Write main grid
            size_t bytes = sizeof(ComplexType) * NmeshTotComplexAlloc;
            myfile.write((char *)fourier_grid_raw.data(), bytes);

            myfile.close();
            if (myfile.fail()) {
                std::string error = "[FFTWGrid::dump_to_file] Failed to write the grid data on task " +
                                    std::to_string(FML::ThisTask) + " Filename: " + filename + "\n";
                assert_mpi(false, error.c_str());
            }
        }

        template <int N>
        void FFTWGrid<N>::load_from_file(std::string fileprefix) {
            std::ios_base::sync_with_stdio(false);
            std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
            auto myfile = std::ifstream(filename, std::ios::binary);

            // If we fail to load a file throw an error
            if (not myfile.good()) {
                std::string error = "[FFTWGrid::load_from_file] Failed to read the grid from file on task " +
                                    std::to_string(FML::ThisTask) + " Filename: " + filename + "\n";
                assert_mpi(false, error.c_str());
            }
            auto check_read = [&](const char * what) {
                if (not myfile.good()) {
                    std::string error = "[FFTWGrid::load_from_file] File is truncated or unreadable (failed to read " +
                                        std::string(what) + ") on task " + std::to_string(FML::ThisTask) +
                                        " Filename: " + filename + "\n";
                    assert_mpi(false, error.c_str());
                }
            };

            // Read header data (into temporaries, the grid is only changed once everything is validated)
            int NDIM, _Nmesh, _nleft, _nright;
            ptrdiff_t _Local_nx, _Local_x_start, _NmeshTotComplexAlloc, _NmeshTotRealAlloc, _NmeshTotComplex,
                _NmeshTotComplexSlice, _NmeshTotRealSlice;
            bool _grid_is_in_real_space;
            myfile.read((char *)&NDIM, sizeof(NDIM));
            check_read("the header");
            assert_mpi(N == NDIM,
                       "[FFTWGrid::load_from_file] The dimension of the grid does not match what is in the file\n");
            myfile.read((char *)&_Nmesh, sizeof(_Nmesh));
            myfile.read((char *)&_nleft, sizeof(_nleft));
            myfile.read((char *)&_nright, sizeof(_nright));
            myfile.read((char *)&_Local_nx, sizeof(_Local_nx));
            myfile.read((char *)&_Local_x_start, sizeof(_Local_x_start));
            myfile.read((char *)&_NmeshTotComplexAlloc, sizeof(_NmeshTotComplexAlloc));
            myfile.read((char *)&_NmeshTotRealAlloc, sizeof(_NmeshTotRealAlloc));
            myfile.read((char *)&_NmeshTotComplex, sizeof(_NmeshTotComplex));
            myfile.read((char *)&_NmeshTotComplexSlice, sizeof(_NmeshTotComplexSlice));
            myfile.read((char *)&_NmeshTotRealSlice, sizeof(_NmeshTotRealSlice));
            myfile.read((char *)&_grid_is_in_real_space, sizeof(_grid_is_in_real_space));
            check_read("the header");

            // The file must have been written with the same domain decomposition as we have now
            bool ok = _Nmesh > 0 and _nleft >= 0 and _nright >= 0;
#ifdef USE_MPI
            ok = ok and _Nmesh % FML::NTasks == 0;
#endif
            ptrdiff_t expected_Local_nx = 0, expected_Local_x_start = 0, expected_NmeshTotComplex = 0;
            if (ok)
                expected_NmeshTotComplex = compute_local_size(_Nmesh, expected_Local_nx, expected_Local_x_start);
            const ptrdiff_t expected_slice = ptrdiff_t(_Nmesh / 2 + 1) * ptrdiff_t(FML::power(_Nmesh, N - 2));
            ok = ok and _Local_nx == expected_Local_nx and _Local_x_start == expected_Local_x_start and
                 _NmeshTotComplex == expected_NmeshTotComplex and _NmeshTotComplexSlice == expected_slice and
                 _NmeshTotRealSlice == 2 * expected_slice and
                 _NmeshTotComplexAlloc == expected_NmeshTotComplex + expected_slice * (_nleft + _nright) and
                 _NmeshTotRealAlloc == 2 * _NmeshTotComplexAlloc;
            if (not ok) {
                std::string error =
                    "[FFTWGrid::load_from_file] The grid in the file does not match the current domain decomposition "
                    "(it must be read with the same number of tasks as it was written with) on task " +
                    std::to_string(FML::ThisTask) + " File has Nmesh = " + std::to_string(_Nmesh) +
                    " Local_nx = " + std::to_string(_Local_nx) + " Local_x_start = " + std::to_string(_Local_x_start) +
                    ", expected Local_nx = " + std::to_string(expected_Local_nx) +
                    " Local_x_start = " + std::to_string(expected_Local_x_start) + " Filename: " + filename + "\n";
                assert_mpi(false, error.c_str());
            }

            // Allocate and read main grid
            PayloadStorage data(_NmeshTotComplexAlloc);
            size_t bytes = sizeof(ComplexType) * _NmeshTotComplexAlloc;
            myfile.read((char *)data.data(), bytes);
            check_read("the grid data");
            myfile.close();

            fourier_grid_raw = std::move(data);
            Nmesh = _Nmesh;
            n_extra_x_slices_left = _nleft;
            n_extra_x_slices_right = _nright;
            Local_nx = _Local_nx;
            Local_x_start = _Local_x_start;
            NmeshTotComplexAlloc = _NmeshTotComplexAlloc;
            NmeshTotRealAlloc = _NmeshTotRealAlloc;
            NmeshTotComplex = _NmeshTotComplex;
            NmeshTotComplexSlice = _NmeshTotComplexSlice;
            NmeshTotRealSlice = _NmeshTotRealSlice;
            NmeshTotReal = ptrdiff_t(Local_nx) * ptrdiff_t(FML::power(Nmesh, N - 1));
            grid_is_in_real_space = _grid_is_in_real_space;
            add_memory_label(name.empty() ? "FFTWGrid" : name);
        }

#ifdef USE_GSL
        /// std::function can be slow so for looping through a fourier grid and evaluating a function f(k)
        /// in every cell its faster to make a spline and use this instead. This method makes such a spline.
        /// @param[in] function_of_kBox The function f(kBox)
        /// @param[in] label The label given to the spline. Useful for giving error messages if the spline ever
        /// throw errors
        /// @param[in] sampling_factor How many points per integer wavenumber should we sample the function at when
        /// making the spline?
        template <int N>
        FML::INTERPOLATION::SPLINE::Spline
        FFTWGrid<N>::make_fourier_spline(std::function<double(double)> function_of_kBox,
                                         std::string label,
                                         int sampling_factor) const {

            FML::assert_mpi(Nmesh > 0, "FFTWGrid::make_fourier_spline Grid is not allocated");

            sampling_factor = std::max(sampling_factor, 1);
            const int npts = Nmesh * sampling_factor + 1;
            const double kmin = 0.0;
            const double kmax = 2.0 * M_PI * Nmesh / 2.0 * std::sqrt(double(N));
            std::vector<double> k_vec(npts);
            std::vector<double> func_vec(npts);
            for (int i = 0; i < npts; i++) {
                k_vec[i] = kmin + (kmax - kmin) * i / double(npts - 1);
                func_vec[i] = function_of_kBox(k_vec[i]);
                if (func_vec[i] != func_vec[i] or std::isinf(func_vec[i])) {
                    throw std::runtime_error("FFTWGrid::make_fourier_spline The function we asked to make a spline "
                                             "of evaluated to NaN or Inf at kBox = " +
                                             std::to_string(k_vec[i]));
                }
            }
            return FML::INTERPOLATION::SPLINE::Spline(k_vec, func_vec, label);
        }
#endif
    } // namespace GRID
} // namespace FML
#endif
