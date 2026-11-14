#ifndef FML_CPENCIL_GRID_H
#define FML_CPENCIL_GRID_H

#include <FML/PencilDomain/PencilGrid.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace FML {
    namespace GRID {

        struct CPencilTransforms;

        //==========================================================================
        ///
        /// A complex field on a grid distributed in pencils (a PencilDomain) with complex-to-complex FFTs by MPI
        /// transposes: the complex counterpart of PencilGrid, as CFFTWGrid is of FFTWGrid. The same interface and
        /// conventions as PencilGrid, except that:
        ///
        ///   - the values in position ("real") space are complex (get_real returns a ComplexType) and the rows have
        ///     no padding
        ///   - the Fourier grid has all the modes: in 3D task (p0, p1) has all kx, the ky in its part of [0, Nmesh)
        ///     (split over P0) and the kz in its part of [0, Nmesh) (split over P1); in 2D all kx and the ky in its
        ///     part of [0, Nmesh). Every mode has multiplicity 1 and the integer wave-vector is in (-Nmesh/2, Nmesh/2]
        ///   - the transforms are fftw_forward (normalized by 1/Nmesh^N) and fftw_backward (not normalized)
        ///
        /// Fourier coordinates (kx, iky, ikz) are global in kx and local in ky and kz; the Fourier rows are lines
        /// along kx (index (iky * Local_nkz + ikz) * Nmesh + kx), as for PencilGrid.
        ///
        //==========================================================================

        template <int N>
        class CPencilGrid {
          public:
            static_assert(N == 2 or N == 3, "CPencilGrid: only N = 2 and N = 3 are supported");
            using IndexIntType = FML::IndexIntType;

            /// An empty grid (not allocated)
            CPencilGrid() = default;
            /// The pencils of FML::pencil_domain() (3D) / slabs (2D) with the edges on cells
            CPencilGrid(int Nmesh, int n_extra_left = 0, int n_extra_right = 0);
            /// The pencils of a domain (the edges must lie on cells)
            CPencilGrid(int Nmesh, int n_extra_left, int n_extra_right, const PencilDomain & domain);
            /// Copies (own memory, the shared transforms) and moves
            CPencilGrid(const CPencilGrid & rhs);
            CPencilGrid & operator=(const CPencilGrid & rhs);
            CPencilGrid(CPencilGrid && rhs) noexcept;
            CPencilGrid & operator=(CPencilGrid && rhs) noexcept;
            ~CPencilGrid();

            /// Has the grid been allocated
            explicit operator bool() const { return data != nullptr; }
            /// A grid with the same size, extra cells, pencils and status with all values zero
            CPencilGrid clone_empty() const {
                CPencilGrid result(Nmesh, n_extra_left, n_extra_right, domain);
                result.set_grid_status_real(grid_is_in_real_space);
                result.add_memory_label(name);
                return result;
            }
            /// Reallocate as a new grid on the same pencils (the values are zero)
            void reallocate(int Nmesh_, int nleft, int nright) { *this = CPencilGrid(Nmesh_, nleft, nright, domain); }
            /// Free the memory (an empty grid)
            void free();

            /// The main grid in position space (the cell with local coordinates 0)
            ComplexType * get_real_grid() { return data + real_offset; }
            /// The main grid in position space (the cell with local coordinates 0)
            const ComplexType * get_real_grid() const { return data + real_offset; }
            /// The Fourier grid
            ComplexType * get_fourier_grid() { return data; }
            /// The Fourier grid
            const ComplexType * get_fourier_grid() const { return data; }

            /// Forward transform (position -> Fourier space), normalized by 1/Nmesh^N
            void fftw_forward();
            /// Backward transform (Fourier -> position space), not normalized
            void fftw_backward();

            /// Fill all the allocated memory with a value (and set the status)
            void fill_real_grid(const ComplexType val);
            /// Fill all the allocated memory with a value and set the status to Fourier space
            void fill_fourier_grid(const ComplexType val);
            /// Fill the main grid from a function of the position / the wave-vector
            void fill_real_grid(const std::function<ComplexType(std::array<double, N> &)> & func);
            /// Fill the Fourier grid from a function of the wave-vector 2 pi n
            void fill_fourier_grid(const std::function<ComplexType(std::array<double, N> &)> & func);

            /// Local coordinates from a local index and the reverse (position space: x and in 3D y local, z global)
            std::array<int, N> get_coord_from_index(const IndexIntType index_real) const {
                const IndexIntType index = index_real + real_offset;
                std::array<int, N> coord;
                coord[0] = int(index / slice_stride) - n_extra_left;
                const IndexIntType rest = index % slice_stride;
                if constexpr (N == 3) {
                    coord[1] = int(rest / row_stride) - n_extra_left;
                    coord[2] = int(rest % row_stride);
                } else {
                    coord[1] = int(rest);
                }
                return coord;
            }
            /// The local index of a cell in position space from its local coordinates (extra cells: negative / beyond the local range)
            IndexIntType get_index_real(const std::array<int, N> & coord) const {
                if constexpr (N == 3)
                    return coord[0] * slice_stride + coord[1] * row_stride + coord[2];
                return coord[0] * slice_stride + coord[1];
            }
            /// Fourier coordinates (kx global, iky and ikz local) from a local index and the reverse
            std::array<int, N> get_fourier_coord_from_index(const IndexIntType index) const {
                std::array<int, N> coord;
                coord[0] = int(index % Nmesh);
                const IndexIntType row = index / Nmesh;
                if constexpr (N == 3) {
                    coord[1] = int(row / Local_nkz);
                    coord[2] = int(row % Local_nkz);
                } else {
                    coord[1] = int(row);
                }
                return coord;
            }
            /// The local Fourier index from Fourier coordinates (kx global, iky and ikz local)
            IndexIntType get_index_fourier(const std::array<int, N> & coord) const {
                if constexpr (N == 3)
                    return (IndexIntType(coord[1]) * Local_nkz + coord[2]) * Nmesh + coord[0];
                return IndexIntType(coord[1]) * Nmesh + coord[0];
            }
            /// The index of a Fourier cell from its global Fourier coordinates (must be local)
            IndexIntType get_index_fourier_global(std::array<int, N> coord) const {
                coord[1] -= int(Local_ky_start);
                if constexpr (N == 3)
                    coord[2] -= int(Local_kz_start);
                return get_index_fourier(coord);
            }

            /// The value of a cell in position space from its local coordinates
            ComplexType get_real(const std::array<int, N> & coord) const { return get_real_grid()[get_index_real(coord)]; }
            /// The value of a cell in position space from its local index
            ComplexType get_real_from_index(const IndexIntType index) const { return get_real_grid()[index]; }
            /// The value of a Fourier cell from its Fourier coordinates
            ComplexType get_fourier(const std::array<int, N> & coord) const { return data[get_index_fourier(coord)]; }
            /// The value of a Fourier cell from its local index
            ComplexType get_fourier_from_index(const IndexIntType index) const { return data[index]; }
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
            /// Set the value of a Fourier cell (Fourier coordinates)
            void set_fourier(const std::array<int, N> & coord, const ComplexType value) {
                data[get_index_fourier(coord)] = value;
            }
            /// Set the value of a Fourier cell (local index)
            void set_fourier_from_index(const IndexIntType index, const ComplexType value) { data[index] = value; }

            /// The (global) position of a cell in [0,1)^N
            std::array<double, N> get_real_position(const std::array<int, N> & coord) const {
#ifdef CELLCENTERSHIFTED
                const constexpr double shift = 0.5;
#else
                const constexpr double shift = 0.0;
#endif
                std::array<double, N> x;
                x[0] = (double(Local_x_start + coord[0]) + shift) / double(Nmesh);
                x[1] = (double(Local_y_start + coord[1]) + shift) / double(Nmesh);
                if constexpr (N == 3)
                    x[2] = (double(coord[2]) + shift) / double(Nmesh);
                return x;
            }
            /// The integer wave-vector of a Fourier cell, in (-Nmesh/2, Nmesh/2] in every direction
            std::array<int, N> get_fourier_integer_wavevector_from_index(const IndexIntType index) const {
                auto k = get_fourier_coord_from_index(index);
                k[1] += int(Local_ky_start);
                if constexpr (N == 3)
                    k[2] += int(Local_kz_start);
                for (auto & ki : k)
                    if (ki > Nmesh / 2)
                        ki -= Nmesh;
                return k;
            }
            /// The wave-vector 2 pi n of a Fourier cell (local index; for physical k divide by the box size)
            std::array<double, N> get_fourier_wavevector_from_index(const IndexIntType index) const {
                const auto k = get_fourier_integer_wavevector_from_index(index);
                std::array<double, N> kvec;
                for (int idim = 0; idim < N; idim++)
                    kvec[idim] = 2.0 * M_PI * k[idim];
                return kvec;
            }
            /// The wave-vector 2 pi n of a Fourier cell (Fourier coordinates)
            std::array<double, N> get_fourier_wavevector(const std::array<int, N> & coord) const {
                return get_fourier_wavevector_from_index(get_index_fourier(coord));
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
            /// The wave-vector 2 pi n and its norm |k| of a Fourier cell (local index)
            void get_fourier_wavevector_and_norm_by_index(const IndexIntType index,
                                                          std::array<double, N> & kvec,
                                                          double & kmag) const {
                get_fourier_wavevector_and_norm2_by_index(index, kvec, kmag);
                kmag = std::sqrt(kmag);
            }

            /// The number of extra cells to the left in x (and in 3D in y)
            int get_n_extra_slices_left() const { return n_extra_left; }
            /// The number of extra cells to the right in x (and in 3D in y)
            int get_n_extra_slices_right() const { return n_extra_right; }
            /// The number of cells per dimension
            int get_nmesh() const { return Nmesh; }
            /// The number of dimensions N
            int get_ndim() const { return N; }
            /// The pencils of the grid
            const PencilDomain & get_domain() const { return domain; }
            /// The number of local cells in x
            ptrdiff_t get_local_nx() const { return Local_nx; }
            /// The global x-index of the first local cell
            ptrdiff_t get_local_x_start() const { return Local_x_start; }
            /// The number of local cells in y (2D: Nmesh)
            ptrdiff_t get_local_ny() const { return Local_ny; }
            /// The global y-index of the first local cell (2D: 0)
            ptrdiff_t get_local_y_start() const { return Local_y_start; }
            /// The number of local ky in Fourier space
            ptrdiff_t get_local_nky() const { return Local_nky; }
            /// The first local ky (in [0, Nmesh))
            ptrdiff_t get_local_ky_start() const { return Local_ky_start; }
            /// The number of local kz in Fourier space (2D: 1)
            ptrdiff_t get_local_nkz() const { return Local_nkz; }
            /// The first local kz (in [0, Nmesh); 2D: 0)
            ptrdiff_t get_local_kz_start() const { return Local_kz_start; }
            /// The local cells along an axis: {first, count} (x and in 3D y: the pencil, the other axes: all)
            std::array<ptrdiff_t, 2> get_local_range(int axis) const {
                if (axis == 0)
                    return {Local_x_start, Local_nx};
                if (axis == 1)
                    return {Local_y_start, Local_ny};
                return {0, Nmesh};
            }
            /// The local Fourier cells along an axis: {first, count} (the integer wave-vector modulo Nmesh)
            std::array<ptrdiff_t, 2> get_local_fourier_range(int axis) const {
                if (axis == 0)
                    return {0, Nmesh};
                if (axis == 1)
                    return {Local_ky_start, Local_nky};
                return {Local_kz_start, Local_nkz};
            }

            /// The number of local cells in position and in Fourier space
            ptrdiff_t get_ntot_real() const { return get_local_nrows() * Nmesh; }
            /// The number of local Fourier cells
            ptrdiff_t get_ntot_fourier() const { return Local_nky * Local_nkz * Nmesh; }
            /// The number of complex numbers allocated
            ptrdiff_t get_ntot_fourier_alloc() const { return NComplexAlloc; }
            /// The memory allocated for the grid in bytes
            size_t allocated_storage_bytes() const { return size_t(NComplexAlloc) * sizeof(ComplexType); }

            /// The cells in position space (skips the extra cells in y) and in Fourier space
            PencilRealRange get_real_range() const { return get_real_range_of_rows(0, get_local_nrows()); }
            /// The local Fourier cells (index range)
            FourierRange get_fourier_range() const { return FourierRange(0, get_ntot_fourier()); }
            /// Rows: lines of cells along the last dimension (position space) and along kx (Fourier space)
            IndexIntType get_local_nrows() const { return IndexIntType(Local_nx) * (N == 3 ? Local_ny : 1); }
            /// The cells of the rows [row_begin, row_end) in position space (skips the extra cells)
            PencilRealRange get_real_range_of_rows(IndexIntType row_begin, IndexIntType row_end) const {
                return PencilRealRange(row_begin, row_end, Nmesh, N == 3 ? int(Local_ny) : 1, row_stride, slice_stride);
            }
            /// The number of local Fourier rows (lines along kx)
            IndexIntType get_local_nrows_fourier() const { return IndexIntType(Local_nky) * Local_nkz; }
            /// The cells of the Fourier rows [row_begin, row_end)
            FourierRange get_fourier_range_of_rows(IndexIntType row_begin, IndexIntType row_end) const {
                return FourierRange(row_begin * Nmesh, row_end * Nmesh);
            }
            /// The axis the Fourier rows run along (kx)
            static constexpr int fourier_row_axis = 0;

            /// Is there a NaN or Inf in the active cells
            bool nan_in_grids() const;
            /// Fill the extra cells with the cells of the neighboring tasks (periodic)
            void communicate_boundaries();
            /// Add the extra cells to the cells of the neighboring tasks they belong to and set them to zero
            void add_contribution_from_extra_cells();
            /// Print the size, the process grid, the local cells and the status of the grid (task 0)
            void info() const;
            /// Raw copy for checkpointing (fileprefix.X, X = ThisTask); read back with the same number of tasks
            void dump_to_file(std::string fileprefix) const;
            /// Load a grid written by dump_to_file (the same number of tasks)
            void load_from_file(std::string fileprefix);

            /// Is the grid in position space (true) or in Fourier space (false)?
            bool get_grid_status_real() const { return grid_is_in_real_space; }
            /// Set the status of the grid: position space (true) or Fourier space (false)
            void set_grid_status_real(bool grid_is_a_real_grid) { grid_is_in_real_space = grid_is_a_real_grid; }
            /// A label for the grid (shown by info)
            void add_memory_label(std::string label) { name = std::move(label); }

          private:
            int Nmesh{0};
            int n_extra_left{0};
            int n_extra_right{0};
            PencilDomain domain{};
            ptrdiff_t Local_nx{0}, Local_x_start{0}, Local_ny{0}, Local_y_start{0};
            ptrdiff_t Local_nky{0}, Local_ky_start{0}, Local_nkz{1}, Local_kz_start{0};
            // The layout in position space (in complex numbers): rows of Nmesh cells, rows_per_slice rows per x-slice
            // (3D: the y-values with the extra ones)
            IndexIntType row_stride{0};
            IndexIntType slice_stride{0};
            IndexIntType real_offset{0};
            ComplexType * data{nullptr};
            ptrdiff_t NComplexAlloc{0};
            bool grid_is_in_real_space{true};
            std::string name{"CPencilGrid"};
            std::shared_ptr<CPencilTransforms> transforms{};

            void allocate();
            void exchange_x(bool add);
            void exchange_y(bool add);
        };

        /// A grid of another size on the same pencils (the edges must lie on cells of the new size)
        template <int N>
        CPencilGrid<N> make_grid_like(const CPencilGrid<N> & grid, int Nmesh, int n_extra_left, int n_extra_right) {
            return CPencilGrid<N>(Nmesh, n_extra_left, n_extra_right, grid.get_domain());
        }

        /// A CPencilGrid on a domain: a PencilDomain, or the slabs of a SlabDomain (P1 = 1)
        template <int N>
        CPencilGrid<N> make_grid_on_domain(grid_type_tag<CPencilGrid<N>>, int Nmesh, int n_extra_left, int n_extra_right,
                                           const FML::Domain & domain) {
            if (domain.holds<PencilDomain>())
                return CPencilGrid<N>(Nmesh, n_extra_left, n_extra_right, domain.get<PencilDomain>());
            assert_mpi(domain.is_slab(), "[make_grid] A CPencilGrid needs a PencilDomain or a SlabDomain");
            std::vector<double> xedges(FML::NTasks + 1);
            for (int task = 0; task < FML::NTasks; task++) {
                assert_mpi(task == 0 or std::abs(domain.x_min(task) - domain.x_max(task - 1)) < 1e-12,
                           "[make_grid] The slabs must be in the order of the tasks");
                xedges[task] = domain.x_min(task);
            }
            xedges[FML::NTasks] = 1.0;
            return CPencilGrid<N>(Nmesh, n_extra_left, n_extra_right, PencilDomain::from_edges(xedges, {0.0, 1.0}));
        }

    } // namespace GRID
} // namespace FML

#endif
