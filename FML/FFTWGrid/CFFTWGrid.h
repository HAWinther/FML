#ifndef FML_CFFTWGRID_HEADER
#define FML_CFFTWGRID_HEADER

// =====================================================================================================================
// CFFTWGrid<N>: a complex field on an N-dimensional grid distributed in x-slabs (FFTW-MPI, untransposed in both
// spaces) with complex-to-complex Fourier transforms. As close to FFTWGrid<N> as possible:
//
//   - The same constructor (Nmesh, extra x-slices left/right), slabs, extra slices and communicate_boundaries
//   - The same index and coordinate functions, ranges (by rows for OpenMP), status, label, dump/load, clone_empty
//   - The same normalization: fftw_forward() divides by Nmesh^N (Fourier-series amplitudes), fftw_backward() does not
//
// FFTW may give a task no slices when Nmesh is not divisible by the number of tasks (FFTWGrid requires it to be);
// such grids work but cannot have extra slices.
//
// The differences: the values in position ("real") space are complex (get_real returns a ComplexType), there is no
// padding, and the Fourier grid has all the modes: the coordinates are Local_nx x [0,Nmesh)^(N-1) as in position
// space and every mode has multiplicity 1 (the integer wave-vector is in (-Nmesh/2, Nmesh/2] in every direction).
// The transforms are fftw_forward / fftw_backward (FFTWGrid: fftw_r2c / fftw_c2r). The pencil counterpart is
// FML::GRID::CPencilGrid (PencilDomain/CPencilGrid.h).
// =====================================================================================================================

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_FFTW
#if defined(SINGLE_PRECISION_FFTW)
#define FML_CFFT(name) fftwf_##name
#define FML_CFFT_MPI(name) fftwf_mpi_##name
#elif defined(LONG_DOUBLE_PRECISION_FFTW)
#define FML_CFFT(name) fftwl_##name
#define FML_CFFT_MPI(name) fftwl_mpi_##name
#else
#define FML_CFFT(name) fftw_##name
#define FML_CFFT_MPI(name) fftw_mpi_##name
#endif
#endif

namespace FML {
    namespace GRID {

        /// A complex field on an N-dimensional grid in FFTW-MPI x-slabs with complex-to-complex FFTs (see the top of this file)
        template <int N>
        class CFFTWGrid {
            static_assert(N >= 2, "CFFTWGrid: only N >= 2 dimensions are supported");

          public:
            using IndexIntType = FML::IndexIntType;

          private:
            Vector<ComplexType> raw{};
            int Nmesh{0};
            ptrdiff_t Local_nx{0};
            ptrdiff_t Local_x_start{0};
            ptrdiff_t NmeshTotSlice{0}; // Nmesh^(N-1)
            ptrdiff_t NmeshTotAlloc{0}; // what FFTW needs for the main grid (>= Local_nx * NmeshTotSlice)
            int n_extra_x_slices_left{0};
            int n_extra_x_slices_right{0};
            bool grid_is_in_real_space{true};
            std::string name{};

            ptrdiff_t main_offset() const { return ptrdiff_t(n_extra_x_slices_left) * NmeshTotSlice; }

            static ptrdiff_t compute_local_size(int Nmesh, ptrdiff_t & Local_nx, ptrdiff_t & Local_x_start) {
                ptrdiff_t slice = 1;
                for (int i = 1; i < N; i++)
                    slice *= Nmesh;
#if defined(USE_MPI) && defined(USE_FFTW)
                std::vector<ptrdiff_t> n(N, Nmesh);
                return FML_CFFT_MPI(local_size)(N, n.data(), FML::communicator, &Local_nx, &Local_x_start);
#else
                Local_nx = Nmesh;
                Local_x_start = 0;
                return Local_nx * slice;
#endif
            }

#ifdef USE_FFTW
            // A (cached) plan for an in-place transform of the main grid (sign: FFTW_FORWARD / FFTW_BACKWARD)
            my_fftw_plan get_fftw_plan(int sign) {
                const unsigned int flag = FML::GRID::get_fftw_planner_flag();
                const FML::GRID::FFTWPlanKey key{N,
                                                 Nmesh,
                                                 sign == FFTW_FORWARD ? 2 : -2,
                                                 FML::GRID::get_fftw_nthreads(),
                                                 FML_CFFT(alignment_of)(reinterpret_cast<FloatType *>(get_fourier_grid())),
                                                 flag};
                if (auto * plan = FML::GRID::find_cached_fftw_plan(key))
                    return *plan;
                // Planning with anything but FFTW_ESTIMATE overwrites the arrays so keep a copy
                Vector<ComplexType> backup;
                if (flag != FFTW_ESTIMATE)
                    backup = raw;
                auto * data = get_fftw_grid();
#ifdef USE_MPI
                std::vector<ptrdiff_t> n(N, Nmesh);
                my_fftw_plan plan = FML_CFFT_MPI(plan_dft)(N, n.data(), data, data, FML::communicator, sign, flag);
#else
                std::vector<int> n(N, Nmesh);
                my_fftw_plan plan = FML_CFFT(plan_dft)(N, n.data(), data, data, sign, flag);
#endif
                assert_mpi(plan != nullptr, "[CFFTWGrid::get_fftw_plan] FFTW failed to create a plan\n");
                if (flag != FFTW_ESTIMATE)
                    std::copy(backup.begin(), backup.end(), raw.begin());
                FML::GRID::store_fftw_plan(key, plan);
                return plan;
            }
            void execute(int sign) {
                auto plan = get_fftw_plan(sign);
                auto * data = get_fftw_grid();
#ifdef USE_MPI
                FML_CFFT_MPI(execute_dft)(plan, data, data);
#else
                FML_CFFT(execute_dft)(plan, data, data);
#endif
            }
#endif

#ifdef USE_MPI
            static void sendrecv_bytes(const char * sendbuf, char * recvbuf, size_t bytes, int dest, int source) {
                const size_t maxchunk = size_t(1) << 30;
                size_t done = 0;
                do {
                    const int n = int(std::min(maxchunk, bytes - done));
                    MPI_Sendrecv(sendbuf + done, n, MPI_BYTE, dest, 0, recvbuf + done, n, MPI_BYTE, source, 0,
                                 FML::communicator, MPI_STATUS_IGNORE);
                    done += size_t(n);
                } while (done < bytes);
            }
#endif

          public:
            /// An empty grid (not allocated)
            CFFTWGrid() = default;
            /// Has the grid been allocated
            explicit operator bool() const { return raw.size() > 0; }

            /// Constructor
            /// @param[in] Nmesh Total number of grid-cells per dimension
            /// @param[in] n_extra_x_slices_left Number of extra x-slices to the left of the grid
            /// @param[in] n_extra_x_slices_right Number of extra x-slices to the right of the grid
            CFFTWGrid(int Nmesh_, int n_extra_x_slices_left_ = 0, int n_extra_x_slices_right_ = 0)
                : Nmesh(Nmesh_), n_extra_x_slices_left(n_extra_x_slices_left_),
                  n_extra_x_slices_right(n_extra_x_slices_right_) {
                assert_mpi(Nmesh > 0 and n_extra_x_slices_left >= 0 and n_extra_x_slices_right >= 0,
                           "[CFFTWGrid] Invalid grid size or number of extra slices\n");
                NmeshTotSlice = 1;
                for (int i = 1; i < N; i++)
                    NmeshTotSlice *= Nmesh;
                NmeshTotAlloc = compute_local_size(Nmesh, Local_nx, Local_x_start);
                NmeshTotAlloc = std::max(NmeshTotAlloc, Local_nx * NmeshTotSlice);
                const ptrdiff_t total = NmeshTotAlloc +
                                        ptrdiff_t(n_extra_x_slices_left + n_extra_x_slices_right) * NmeshTotSlice;
                raw.assign(size_t(std::max<ptrdiff_t>(total, 1)), ComplexType(0));
            }

            /// Copies and moves (a copy has its own memory)
            CFFTWGrid(const CFFTWGrid &) = default;
            CFFTWGrid & operator=(const CFFTWGrid &) = default;
            CFFTWGrid(CFFTWGrid && rhs) noexcept { *this = std::move(rhs); }
            CFFTWGrid & operator=(CFFTWGrid && rhs) noexcept {
                if (this != &rhs) {
                    raw = std::move(rhs.raw);
                    Nmesh = std::exchange(rhs.Nmesh, 0);
                    Local_nx = std::exchange(rhs.Local_nx, 0);
                    Local_x_start = std::exchange(rhs.Local_x_start, 0);
                    NmeshTotSlice = std::exchange(rhs.NmeshTotSlice, 0);
                    NmeshTotAlloc = std::exchange(rhs.NmeshTotAlloc, 0);
                    n_extra_x_slices_left = std::exchange(rhs.n_extra_x_slices_left, 0);
                    n_extra_x_slices_right = std::exchange(rhs.n_extra_x_slices_right, 0);
                    grid_is_in_real_space = std::exchange(rhs.grid_is_in_real_space, true);
                    name = std::move(rhs.name);
                    rhs.raw = Vector<ComplexType>();
                }
                return *this;
            }

            /// A grid with the same size, extra slices and status with all values zero
            CFFTWGrid clone_empty() const {
                assert_mpi(Nmesh > 0, "[CFFTWGrid::clone_empty] Source must be allocated");
                CFFTWGrid result(Nmesh, n_extra_x_slices_left, n_extra_x_slices_right);
                result.grid_is_in_real_space = grid_is_in_real_space;
                result.name = name;
                return result;
            }
            /// Reallocate as a new grid of Nmesh cells per dimension with extra slices (the values are zero)
            void reallocate(int Nmesh_, int nleft, int nright) { *this = CFFTWGrid(Nmesh_, nleft, nright); }
            /// Free all the memory (an empty grid; the label is kept)
            void free() {
                const auto label = name;
                *this = CFFTWGrid();
                name = label;
            }

            // Pointers to the parts of the grid
            ComplexType * get_real_grid_left() { return raw.data(); }
            /// The first cell of the main grid in position space (after the left extra slices)
            ComplexType * get_real_grid() { return raw.data() + main_offset(); }
            /// The first cell of the main grid in position space
            const ComplexType * get_real_grid() const { return raw.data() + main_offset(); }
            /// The first cell of the right extra slices
            ComplexType * get_real_grid_right() { return get_real_grid() + Local_nx * NmeshTotSlice; }
            /// The first cell of a local x-slice (negative or >= Local_nx: the extra slices)
            ComplexType * get_real_grid_by_slice(int slice) { return get_real_grid() + slice * NmeshTotSlice; }
            /// The Fourier grid (the same memory as the main grid in position space)
            ComplexType * get_fourier_grid() { return get_real_grid(); }
            /// The Fourier grid
            const ComplexType * get_fourier_grid() const { return get_real_grid(); }
#ifdef USE_FFTW
            /// The Fourier grid as an FFTW array
            my_fftw_complex * get_fftw_grid() { return reinterpret_cast<my_fftw_complex *>(get_fourier_grid()); }
#endif

            /// Forward transform (position -> Fourier space), normalized by 1/Nmesh^N as FFTWGrid::fftw_r2c
            void fftw_forward() {
#ifdef USE_FFTW
                assert_mpi(Nmesh > 0, "[CFFTWGrid::fftw_forward] The grid is not allocated");
                execute(FFTW_FORWARD);
                const FloatType norm = FloatType(1.0 / std::pow(double(Nmesh), N));
                ComplexType * data = get_fourier_grid();
                const ptrdiff_t n = Local_nx * NmeshTotSlice;
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (ptrdiff_t i = 0; i < n; i++)
                    data[i] *= norm;
                grid_is_in_real_space = false;
#else
                assert_mpi(false, "[CFFTWGrid::fftw_forward] Requires USE_FFTW");
#endif
            }
            /// Backward transform (Fourier -> position space), not normalized as FFTWGrid::fftw_c2r
            void fftw_backward() {
#ifdef USE_FFTW
                assert_mpi(Nmesh > 0, "[CFFTWGrid::fftw_backward] The grid is not allocated");
                execute(FFTW_BACKWARD);
                grid_is_in_real_space = true;
#else
                assert_mpi(false, "[CFFTWGrid::fftw_backward] Requires USE_FFTW");
#endif
            }

            /// Fill the whole grid (the extra slices too) with a constant value
            void fill_real_grid(const ComplexType val) {
                std::fill(raw.begin(), raw.end(), val);
                grid_is_in_real_space = true;
            }
            /// Fill the whole grid (the extra slices too) with a constant value and set the status to Fourier space
            void fill_fourier_grid(const ComplexType val) {
                std::fill(raw.begin(), raw.end(), val);
                grid_is_in_real_space = false;
            }
            /// Fill the main grid from a function of the position
            void fill_real_grid(const std::function<ComplexType(std::array<double, N> &)> & func) {
                for (auto && index : get_real_range()) {
                    auto pos = get_real_position(get_coord_from_index(index));
                    set_real_from_index(index, func(pos));
                }
                grid_is_in_real_space = true;
            }
            /// Fill the main grid from a function of the wave-vector
            void fill_fourier_grid(const std::function<ComplexType(std::array<double, N> &)> & func) {
                for (auto && index : get_fourier_range()) {
                    auto kvec = get_fourier_wavevector_from_index(index);
                    set_fourier_from_index(index, func(kvec));
                }
                grid_is_in_real_space = false;
            }

            /// The (local) cell coordinates from a local index (the same in position and Fourier space)
            std::array<int, N> get_coord_from_index(const IndexIntType index) const {
                std::array<int, N> coord;
                IndexIntType rest = index;
                for (int idim = N - 1; idim >= 1; idim--) {
                    coord[idim] = int(rest % Nmesh);
                    rest /= Nmesh;
                }
                coord[0] = int(rest);
                return coord;
            }
            /// The (local) Fourier coordinates from a local Fourier index (the same layout as in position space)
            std::array<int, N> get_fourier_coord_from_index(const IndexIntType index) const {
                return get_coord_from_index(index);
            }
            /// From (local) coordinate in [-n_left, Local_nx + n_right) x [0,Nmesh)^(N-1) to the index
            IndexIntType get_index_real(const std::array<int, N> & coord) const {
                IndexIntType index = coord[0];
                for (int idim = 1; idim < N; idim++)
                    index = index * Nmesh + coord[idim];
                return index;
            }
            /// The local Fourier index from (local) Fourier coordinates
            IndexIntType get_index_fourier(const std::array<int, N> & coord) const { return get_index_real(coord); }
            /// The index of a Fourier cell from its global Fourier coordinates (must be local)
            IndexIntType get_index_fourier_global(std::array<int, N> coord) const {
                coord[0] -= int(Local_x_start);
                return get_index_fourier(coord);
            }

            /// The value of a cell in position space from its local coordinates
            ComplexType get_real(const std::array<int, N> & coord) const { return get_real_grid()[get_index_real(coord)]; }
            /// The value of a cell in position space from its local index
            ComplexType get_real_from_index(const IndexIntType index) const { return get_real_grid()[index]; }
            /// The value of a Fourier cell from its local coordinates
            ComplexType get_fourier(const std::array<int, N> & coord) const {
                return get_fourier_grid()[get_index_fourier(coord)];
            }
            /// The value of a Fourier cell from its local index
            ComplexType get_fourier_from_index(const IndexIntType index) const { return get_fourier_grid()[index]; }
            /// Set the value of a cell in position space (local coordinates)
            void set_real(const std::array<int, N> & coord, const ComplexType value) {
                get_real_grid()[get_index_real(coord)] = value;
            }
            /// Set the value of a cell in position space (local index)
            void set_real_from_index(const IndexIntType index, const ComplexType value) { get_real_grid()[index] = value; }
            /// Add to the value of a cell in position space (local coordinates)
            void add_real(const std::array<int, N> & coord, const ComplexType value) {
                get_real_grid()[get_index_real(coord)] += value;
            }
            /// Set the value of a Fourier cell (local coordinates)
            void set_fourier(const std::array<int, N> & coord, const ComplexType value) {
                get_fourier_grid()[get_index_fourier(coord)] = value;
            }
            /// Set the value of a Fourier cell (local index)
            void set_fourier_from_index(const IndexIntType index, const ComplexType value) {
                get_fourier_grid()[index] = value;
            }

            /// The (global) position of a cell in [0,1)^N
            std::array<double, N> get_real_position(const std::array<int, N> & coord) const {
#ifdef CELLCENTERSHIFTED
                const constexpr double shift = 0.5;
#else
                const constexpr double shift = 0.0;
#endif
                std::array<double, N> pos;
                pos[0] = (double(Local_x_start + coord[0]) + shift) / Nmesh;
                for (int idim = 1; idim < N; idim++)
                    pos[idim] = (double(coord[idim]) + shift) / Nmesh;
                return pos;
            }
            /// The integer wave-vector of a Fourier cell, in (-Nmesh/2, Nmesh/2] in every direction
            std::array<int, N> get_fourier_integer_wavevector_from_index(const IndexIntType index) const {
                auto k = get_coord_from_index(index);
                k[0] += int(Local_x_start);
                for (auto & ki : k)
                    if (ki > Nmesh / 2)
                        ki -= Nmesh;
                return k;
            }
            /// The wave-vector (2 pi times the integer wave-vector; for physical k divide by the box size)
            std::array<double, N> get_fourier_wavevector(const std::array<int, N> & coord) const {
                return get_fourier_wavevector_from_index(get_index_fourier(coord));
            }
            /// The wave-vector 2 pi n of a Fourier cell (local index; for physical k divide by the box size)
            std::array<double, N> get_fourier_wavevector_from_index(const IndexIntType index) const {
                const auto k = get_fourier_integer_wavevector_from_index(index);
                std::array<double, N> kvec;
                for (int idim = 0; idim < N; idim++)
                    kvec[idim] = 2.0 * M_PI * k[idim];
                return kvec;
            }
            /// The wave-vector 2 pi n and its norm |k| of a Fourier cell (local index)
            void get_fourier_wavevector_and_norm_by_index(const IndexIntType index,
                                                          std::array<double, N> & kvec,
                                                          double & kmag) const {
                get_fourier_wavevector_and_norm2_by_index(index, kvec, kmag);
                kmag = std::sqrt(kmag);
            }
            /// The wave-vector 2 pi n and its squared norm k^2 of a Fourier cell (local index)
            void get_fourier_wavevector_and_norm2_by_index(const IndexIntType index,
                                                           std::array<double, N> & kvec,
                                                           double & kmag2) const {
                kvec = get_fourier_wavevector_from_index(index);
                kmag2 = 0.0;
                for (int idim = 0; idim < N; idim++)
                    kmag2 += kvec[idim] * kvec[idim];
            }

            /// The number of extra x-slices to the left
            int get_n_extra_slices_left() const { return n_extra_x_slices_left; }
            /// The number of extra x-slices to the right
            int get_n_extra_slices_right() const { return n_extra_x_slices_right; }
            /// The number of cells per dimension
            int get_nmesh() const { return Nmesh; }
            /// The number of dimensions N
            int get_ndim() const { return N; }
            /// The number of local x-slices (can be 0)
            ptrdiff_t get_local_nx() const { return Local_nx; }
            /// The global x-index of the first local x-slice
            ptrdiff_t get_local_x_start() const { return Local_x_start; }
            /// The local cells along an axis: {first, count} (x: the slab, the other axes: all)
            std::array<ptrdiff_t, 2> get_local_range(int axis) const {
                return axis == 0 ? std::array<ptrdiff_t, 2>{Local_x_start, Local_nx} :
                                   std::array<ptrdiff_t, 2>{0, Nmesh};
            }
            /// The local Fourier cells along an axis (the same as in position space)
            std::array<ptrdiff_t, 2> get_local_fourier_range(int axis) const { return get_local_range(axis); }
            /// Number of local cells (position space; the same in Fourier space)
            ptrdiff_t get_ntot_real() const { return Local_nx * NmeshTotSlice; }
            /// The number of local Fourier cells
            ptrdiff_t get_ntot_fourier() const { return Local_nx * NmeshTotSlice; }
            /// The number of complex numbers allocated (main grid, the FFTW padding and the extra slices)
            ptrdiff_t get_ntot_fourier_alloc() const { return ptrdiff_t(raw.size()); }
            /// The number of cells in an x-slice (Nmesh^(N-1))
            ptrdiff_t get_ntot_real_slice_alloc() const { return NmeshTotSlice; }
            /// The memory allocated for the grid in bytes
            size_t allocated_storage_bytes() const { return raw.capacity() * sizeof(ComplexType); }

            /// The local cells (index ranges; there is no padding)
            FourierRange get_real_range() const { return FourierRange(0, get_ntot_real()); }
            /// The local Fourier cells (index range)
            FourierRange get_fourier_range() const { return FourierRange(0, get_ntot_fourier()); }
            /// The number of local rows (lines along the last dimension) and the cells of a range of rows
            IndexIntType get_local_nrows() const { return IndexIntType(Local_nx * (NmeshTotSlice / Nmesh)); }
            /// The number of local Fourier rows (the same as in position space)
            IndexIntType get_local_nrows_fourier() const { return get_local_nrows(); }
            /// The axis the Fourier rows run along (the last one, as in position space)
            static constexpr int fourier_row_axis = N - 1;
            /// The cells of the rows [row_begin, row_end) in position space (for OpenMP loops over rows)
            FourierRange get_real_range_of_rows(IndexIntType row_begin, IndexIntType row_end) const {
                return FourierRange(row_begin * Nmesh, row_end * Nmesh);
            }
            /// The cells of the Fourier rows [row_begin, row_end)
            FourierRange get_fourier_range_of_rows(IndexIntType row_begin, IndexIntType row_end) const {
                return FourierRange(row_begin * Nmesh, row_end * Nmesh);
            }

            /// Fill the extra slices with the slices of the neighboring tasks (periodic). Every task needs at least
            /// as many slices as extra slices
            void communicate_boundaries() {
                const int nl = n_extra_x_slices_left, nr = n_extra_x_slices_right;
                if (nl + nr == 0)
                    return;
                assert_mpi(Local_nx >= std::max(nl, nr),
                           "[CFFTWGrid::communicate_boundaries] More extra slices than local slices");
                const size_t slice_bytes = size_t(NmeshTotSlice) * sizeof(ComplexType);
#ifdef USE_MPI
                const int right = (FML::ThisTask + 1) % FML::NTasks, left = (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks;
                if (nr > 0)
                    sendrecv_bytes(reinterpret_cast<const char *>(get_real_grid()),
                                   reinterpret_cast<char *>(get_real_grid_right()), nr * slice_bytes, left, right);
                if (nl > 0)
                    sendrecv_bytes(reinterpret_cast<const char *>(get_real_grid_by_slice(int(Local_nx) - nl)),
                                   reinterpret_cast<char *>(get_real_grid_left()), nl * slice_bytes, right, left);
#else
                if (nr > 0)
                    std::memcpy(get_real_grid_right(), get_real_grid(), nr * slice_bytes);
                if (nl > 0)
                    std::memcpy(get_real_grid_left(), get_real_grid_by_slice(int(Local_nx) - nl), nl * slice_bytes);
#endif
            }

            /// Is there a NaN or Inf anywhere in the allocated grid?
            bool nan_in_grids() const {
                for (auto & z : raw)
                    if (not std::isfinite(z.real()) or not std::isfinite(z.imag()))
                        return true;
                return false;
            }
            /// Is the grid in position space (true) or in Fourier space (false)?
            bool get_grid_status_real() const { return grid_is_in_real_space; }
            /// Set the status of the grid: position space (true) or Fourier space (false)
            void set_grid_status_real(bool grid_is_a_real_grid) { grid_is_in_real_space = grid_is_a_real_grid; }
            /// A label for the grid (shown by info)
            void add_memory_label(std::string label) { name = label; }
            /// Print the size, the extra slices and the status of the grid (task 0)
            void info() const {
                if (FML::ThisTask == 0)
                    std::cout << "CFFTWGrid<" << N << "> " << name << ": Nmesh = " << Nmesh
                              << " extra slices = " << n_extra_x_slices_left << " " << n_extra_x_slices_right
                              << " status = " << (grid_is_in_real_space ? "position" : "Fourier") << " space\n";
            }

            /// Save to file (fileprefix.X with X = ThisTask): a raw copy for checkpointing; read it back with
            /// load_from_file with the same number of tasks
            void dump_to_file(std::string fileprefix) const {
                const std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
                std::ofstream out(filename, std::ios::binary | std::ios::trunc);
                const char magic[8] = {'F', 'M', 'L', 'C', 'G', 'R', 'D', '1'};
                const int64_t header[8] = {N, Nmesh, n_extra_x_slices_left, n_extra_x_slices_right, Local_nx,
                                           Local_x_start, int64_t(raw.size()), grid_is_in_real_space};
                out.write(magic, 8);
                out.write(reinterpret_cast<const char *>(header), sizeof(header));
                out.write(reinterpret_cast<const char *>(raw.data()), std::streamsize(raw.size() * sizeof(ComplexType)));
                out.close();
                int good = out.fail() ? 0 : 1;
                FML::MinOverTasks(&good);
                assert_mpi(good == 1, ("[CFFTWGrid::dump_to_file] Failed to write " + fileprefix + ".*").c_str());
            }
            /// Load a grid written by dump_to_file (same number of tasks)
            void load_from_file(std::string fileprefix) {
                const std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
                std::ifstream in(filename, std::ios::binary);
                char magic[8] = {};
                int64_t header[8] = {};
                in.read(magic, 8);
                in.read(reinterpret_cast<char *>(header), sizeof(header));
                bool ok = in.good() and std::memcmp(magic, "FMLCGRD1", 8) == 0 and header[0] == N and header[1] > 0;
                int good = ok ? 1 : 0;
                FML::MinOverTasks(&good);
                assert_mpi(good == 1, ("[CFFTWGrid::load_from_file] Cannot read " + fileprefix + ".*").c_str());
                *this = CFFTWGrid(int(header[1]), int(header[2]), int(header[3]));
                ok = Local_nx == header[4] and Local_x_start == header[5] and int64_t(raw.size()) == header[6];
                if (ok) {
                    in.read(reinterpret_cast<char *>(raw.data()), std::streamsize(raw.size() * sizeof(ComplexType)));
                    // Nothing may be left: a longer file was written with another precision (e.g. double vs float)
                    ok = in.good() and in.peek() == std::ifstream::traits_type::eof();
                }
                grid_is_in_real_space = header[7] != 0;
                good = ok ? 1 : 0;
                FML::MinOverTasks(&good);
                assert_mpi(good == 1, ("[CFFTWGrid::load_from_file] The data in " + fileprefix +
                                       ".* does not match the grid (truncated, another decomposition or another precision)")
                                          .c_str());
            }
        };

        /// A grid like another (CFFTWGrid: the slabs)
        template <int N>
        CFFTWGrid<N> make_grid_like(const CFFTWGrid<N> &, int Nmesh, int n_extra_left, int n_extra_right) {
            return CFFTWGrid<N>(Nmesh, n_extra_left, n_extra_right);
        }
        /// A CFFTWGrid made by make_grid: the FFTW slabs (the domain must be a SlabDomain)
        template <int N>
        CFFTWGrid<N> make_grid_on_domain(grid_type_tag<CFFTWGrid<N>>, int Nmesh, int n_extra_left, int n_extra_right,
                                         const FML::Domain & domain) {
            assert_mpi(domain.is_slab(), "[make_grid] A CFFTWGrid is decomposed in slabs: the domain must be a SlabDomain");
            return CFFTWGrid<N>(Nmesh, n_extra_left, n_extra_right);
        }

    } // namespace GRID
} // namespace FML

#undef FML_CFFT
#undef FML_CFFT_MPI
#endif
