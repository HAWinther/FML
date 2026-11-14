#ifndef FML_PENCIL_GRID_H
#define FML_PENCIL_GRID_H

// =====================================================================================================================
// PencilGrid<N>: a real field distributed in pencils (2D: slabs) with real-to-complex FFTs by MPI transposes, with the
// interface of FFTWGrid. The library's grid code is written against this common interface (FFTWGrid/GridInterface.h),
// so it runs on slabs or on pencils:
//
//   FML::set_domain(FML::pencil_domain());                         // the particles and grids on the pencils
//   auto grid = FML::GRID::make_grid<FML::GRID::PencilGrid<3>>(Nmesh, n_extra_left, n_extra_right);
//   grid.fill_real_grid(f); grid.fftw_r2c(); ...; grid.fftw_c2r(); grid.communicate_boundaries();
//
// Pencils allow up to Nmesh^2 tasks and any Nmesh (slabs: at most Nmesh tasks dividing Nmesh). The complex counterpart
// is CPencilGrid (CPencilGrid.h). README.md in this folder: the layout, the interface, how to run the codes of FML on
// pencils (SimulationCode / SnapshotAnalyzer: make USE_PENCIL_GRID=true) and how to write code for both.
// =====================================================================================================================

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/Global/Global.h>

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#ifndef USE_MPI
#error "PencilGrid requires USE_MPI=true"
#endif
#ifndef USE_FFTW
#error "PencilGrid requires USE_FFTW=true"
#endif

namespace FML {
    namespace GRID {

        // The communicators, MPI datatypes and FFTW plans of a pencil grid layout (PencilGrid.cpp). Shared by the
        // copies of a grid
        struct PencilTransforms;

        /// Range iterator over the (local) indices of the real cells of a PencilGrid: the rows (lines of cells
        /// along the last dimension) have padding at the end and in 3D the extra cells in y lie between them, both
        /// are skipped. Like RealRange of FFTWGrid
        class PencilRealRange {
          public:
            class iterator {
              public:
                iterator(IndexIntType row, IndexIntType index, int column, int row_in_slice)
                    : row(row), index(index), column(column), row_in_slice(row_in_slice) {}
                const IndexIntType & operator*() const { return index; }
                bool operator!=(const iterator & other) const { return row != other.row or column != other.column; }
                iterator & operator++() {
                    ++index;
                    if (++column == nlast) {
                        column = 0;
                        ++row;
                        index += row_stride - nlast;
                        if (++row_in_slice == rows_per_slice) {
                            row_in_slice = 0;
                            index += slice_stride - IndexIntType(rows_per_slice) * row_stride;
                        }
                    }
                    return *this;
                }

              private:
                friend class PencilRealRange;
                IndexIntType row, index;
                int column, row_in_slice;
                int nlast{0}, rows_per_slice{1};
                IndexIntType row_stride{0}, slice_stride{0};
            };

            /// The rows [row_begin, row_end): nlast cells per row, rows_per_slice rows per x-slice, the strides (in
            /// reals) from row to row and from x-slice to x-slice
            PencilRealRange(IndexIntType row_begin,
                            IndexIntType row_end,
                            int nlast,
                            int rows_per_slice,
                            IndexIntType row_stride,
                            IndexIntType slice_stride)
                : row_begin(row_begin), row_end(row_end), nlast(nlast), rows_per_slice(rows_per_slice),
                  row_stride(row_stride), slice_stride(slice_stride) {}
            iterator begin() const { return make(row_begin); }
            iterator end() const { return make(row_end); }

          private:
            iterator make(IndexIntType row) const {
                const int row_in_slice = int(row % rows_per_slice);
                iterator it(row, (row / rows_per_slice) * slice_stride + row_in_slice * row_stride, 0, row_in_slice);
                it.nlast = nlast;
                it.rows_per_slice = rows_per_slice;
                it.row_stride = row_stride;
                it.slice_stride = slice_stride;
                return it;
            }
            IndexIntType row_begin, row_end;
            int nlast, rows_per_slice;
            IndexIntType row_stride, slice_stride;
        };

        //==========================================================================
        ///
        /// A grid distributed in pencils (a PencilDomain: P0 x P1 tasks) with in-place real-to-complex FFTs by MPI
        /// transposes. The same interface and conventions as FFTWGrid (positions in [0,1), wave-vectors 2 pi n, the
        /// forward transform normalized by 1/Nmesh^N, local integer coordinates and indices into the grid, range
        /// loops over indices).
        ///
        /// N = 3: task (p0, p1) has the cells x in its x-range, y in its y-range and all z (a pencil along z). In
        /// Fourier space it has all kx, the ky in its part of [0, Nmesh) (split over P0) and the kz in its part of
        /// [0, Nmesh/2] (split over P1). Real coordinates (ix, iy, iz) are local in x and y; Fourier coordinates
        /// (kx, iky, ikz) are global in kx and local in ky and kz (see get_local_ky_start, get_local_kz_start).
        ///
        /// N = 2: the domain has P1 = 1, so task p0 has the x in its x-range and all y (a slab). In Fourier space it
        /// has all kx and the ky in its part of [0, Nmesh/2] (split over P0). Coordinates (ix, iy) and (kx, iky).
        ///
        /// The x (and in 3D y) range of each task are given by the domain, whose edges must lie on cells. The extra
        /// cells (n_extra_left / n_extra_right) are added on both sides in x (and in 3D in y) and filled by
        /// communicate_boundaries. They are not kept through a Fourier transform: the Fourier data uses the same
        /// memory.
        ///
        //==========================================================================

        template <int N>
        class PencilGrid {
            static_assert(N == 2 or N == 3, "PencilGrid: only N = 2 and N = 3 are supported");

          public:
            PencilGrid() = default;

            /// A grid with the default pencils: the process grid N = 3: that of FML::pencil_domain()
            /// (MPI_Dims_create), N = 2: NTasks x 1, with edges on cells as even as possible
            /// @param[in] Nmesh Number of grid-cells per dimension
            /// @param[in] n_extra_left Number of extra cells to the left in x (and in 3D in y)
            /// @param[in] n_extra_right Number of extra cells to the right in x (and in 3D in y)
            PencilGrid(int Nmesh, int n_extra_left = 0, int n_extra_right = 0);
            /// A grid on the given pencils (made for NTasks tasks; edges on cells; N = 2: P1 = 1)
            PencilGrid(int Nmesh, int n_extra_left, int n_extra_right, const PencilDomain & domain);

            PencilGrid(const PencilGrid & rhs);
            PencilGrid & operator=(const PencilGrid & rhs);
            PencilGrid(PencilGrid && rhs) noexcept;
            PencilGrid & operator=(PencilGrid && rhs) noexcept;
            ~PencilGrid();

#ifdef USE_GSL
            /// A spline of a function of |k| (in units of 1/Box) over the wavenumbers of the grid (as FFTWGrid). (A member
            /// template: not made by the explicit instantiation of the class, so PencilGrid.o needs no Spline.o)
            template <class F = std::function<double(double)>>
            FML::INTERPOLATION::SPLINE::Spline
            make_fourier_spline(F function_of_kBox, std::string label, int sampling_factor = 4) const {
                FML::assert_mpi(Nmesh > 0, "PencilGrid::make_fourier_spline Grid is not allocated");
                sampling_factor = std::max(sampling_factor, 1);
                const int npts = Nmesh * sampling_factor + 1;
                const double kmax = 2.0 * M_PI * Nmesh / 2.0 * std::sqrt(double(N));
                std::vector<double> k_vec(npts), func_vec(npts);
                for (int i = 0; i < npts; i++) {
                    k_vec[i] = kmax * i / double(npts - 1);
                    func_vec[i] = function_of_kBox(k_vec[i]);
                    if (func_vec[i] != func_vec[i] or std::isinf(func_vec[i]))
                        throw std::runtime_error("PencilGrid::make_fourier_spline The function evaluated to NaN or Inf "
                                                 "at kBox = " +
                                                 std::to_string(k_vec[i]));
                }
                return FML::INTERPOLATION::SPLINE::Spline(k_vec, func_vec, label);
            }
#else
            /// Without GSL: the function itself (as FFTWGrid)
            std::function<double(double)> make_fourier_spline(std::function<double(double)> function_of_kBox,
                                                              [[maybe_unused]] std::string label,
                                                              [[maybe_unused]] int sampling_factor = 4) const {
                return function_of_kBox;
            }
#endif

            /// A grid of the same size, extra cells and pencils with zero values (keeps the status real/Fourier)
            PencilGrid clone_empty() const {
                PencilGrid result(Nmesh, n_extra_left, n_extra_right, domain);
                result.set_grid_status_real(grid_is_in_real_space);
                result.add_memory_label(name);
                return result;
            }

            /// Has the grid been allocated
            explicit operator bool() const { return data != nullptr; }
            /// Free all the memory associated with the grid (as if default constructed)
            void free();

            /// The real grid (the cell with local coordinate 0; the extra cells are at negative indices and past
            /// the main grid)
            FloatType * get_real_grid();
            /// The Fourier grid
            ComplexType * get_fourier_grid() { return data; }
            /// The fftw_complex cast of get_fourier_grid
            my_fftw_complex * get_fftw_grid() { return reinterpret_cast<my_fftw_complex *>(data); }

            /// Perform real-to-complex fourier transform (normalized by 1/Nmesh^N)
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
            /// Get the (local) Fourier coordinates from a local index
            std::array<int, N> get_fourier_coord_from_index(const IndexIntType index_fourier) const;
            /// From (local) coordinate of a cell (extra cells: -n_extra_left <= ix < Local_nx + n_extra_right,
            /// likewise for y in 3D) to the index in the real grid
            IndexIntType get_index_real(const std::array<int, N> & coord) const;
            /// From (local) coordinate in the Fourier grid to the index in the Fourier grid
            IndexIntType get_index_fourier(const std::array<int, N> & coord) const;

            /// Fetch value in grid by (local) integer coordinate
            FloatType get_real(const std::array<int, N> & coord) const;
            /// Fetch value in grid by (local) index
            FloatType get_real_from_index(const IndexIntType index) const;
            /// Fetch value in fourier grid by (local) integer coordinate
            ComplexType get_fourier(const std::array<int, N> & coord) const;
            /// Fetch value in fourier grid by (local) index
            ComplexType get_fourier_from_index(const IndexIntType index) const;

            /// Set value in grid from (local) integer coordinate
            void set_real(const std::array<int, N> & coord, const FloatType value);
            /// Set value in grid from (local) index
            void set_real_from_index(const IndexIntType index, const FloatType value);
            /// Add to value in grid
            void add_real(const std::array<int, N> & coord, const FloatType value);
            /// Set value of cell in fourier grid using (local) coordinate
            void set_fourier(const std::array<int, N> & coord, const ComplexType value);
            /// Set value of cell in fourier grid using (local) index
            void set_fourier_from_index(const IndexIntType index, const ComplexType value);

            /// The (global) position of a real grid-cell in [0,1)^N
            std::array<double, N> get_real_position(const std::array<int, N> & coord) const;
            /// Get the wave-vector of a grid-cell in Fourier space (for physical [k] multiply by 1/Boxsize)
            std::array<double, N> get_fourier_wavevector(const std::array<int, N> & coord) const;
            /// Get the wave-vector of a grid-cell in Fourier space (for physical [k] multiply by 1/Boxsize)
            std::array<double, N> get_fourier_wavevector_from_index(const IndexIntType index) const;
            /// The integer wave-vector (in [-Nmesh/2, Nmesh/2] and the last component in [0, Nmesh/2])
            std::array<int, N> get_fourier_integer_wavevector_from_index(const IndexIntType index) const;
            /// From index in the grid get the k-vector and the magnitude of it
            void get_fourier_wavevector_and_norm_by_index(const IndexIntType index,
                                                          std::array<double, N> & kvec,
                                                          double & kmag) const;
            /// From index in the grid get the k-vector and the norm of it (square magnitude)
            void get_fourier_wavevector_and_norm2_by_index(const IndexIntType index,
                                                           std::array<double, N> & kvec,
                                                           double & kmag2) const;

            /// Number of extra cells to the left in x (and in 3D in y)
            int get_n_extra_slices_left() const { return n_extra_left; }
            /// Number of extra cells to the right in x (and in 3D in y)
            int get_n_extra_slices_right() const { return n_extra_right; }
            /// Number of grid-cells per dimension
            int get_nmesh() const { return Nmesh; }
            /// Dimension of the grid
            int get_ndim() const { return N; }
            /// The pencils
            const PencilDomain & get_domain() const { return domain; }

            /// Number of local x-slices in the real grid and the first one
            ptrdiff_t get_local_nx() const { return Local_nx; }
            ptrdiff_t get_local_x_start() const { return Local_x_start; }
            /// Number of local y-values in the real grid and the first one (N = 2: Nmesh and 0)
            ptrdiff_t get_local_ny() const { return Local_ny; }
            ptrdiff_t get_local_y_start() const { return Local_y_start; }
            /// Number of local ky-values in the Fourier grid and the first one
            ptrdiff_t get_local_nky() const { return Local_nky; }
            ptrdiff_t get_local_ky_start() const { return Local_ky_start; }
            /// The local cells along an axis: {first, count} (x and in 3D y: the pencil, the other axes: all)
            std::array<ptrdiff_t, 2> get_local_range(int axis) const {
                if (axis == 0)
                    return {Local_x_start, Local_nx};
                if (axis == 1)
                    return {Local_y_start, Local_ny};
                return {0, Nmesh};
            }
            /// The local Fourier cells along an axis: {first, count} in the global Fourier coordinates (the integer
            /// wavevector modulo Nmesh; the last axis: [0, Nmesh/2]). x: all, y (and z): the local ones
            std::array<ptrdiff_t, 2> get_local_fourier_range(int axis) const {
                if (axis == 0)
                    return {0, Nmesh};
                if (axis == 1)
                    return {Local_ky_start, Local_nky};
                return {Local_kz_start, Local_nkz};
            }
            /// The index of a Fourier cell from its global Fourier coordinates (must be local)
            IndexIntType get_index_fourier_global(std::array<int, N> coord) const {
                coord[1] -= int(Local_ky_start);
                if constexpr (N == 3)
                    coord[2] -= int(Local_kz_start);
                return get_index_fourier(coord);
            }
            /// Number of local kz-values in the Fourier grid and the first one (N = 3)
            ptrdiff_t get_local_nkz() const { return Local_nkz; }
            ptrdiff_t get_local_kz_start() const { return Local_kz_start; }

            /// The number of active real cells (with padding)
            ptrdiff_t get_ntot_real() const { return get_local_nrows() * row_stride; }
            /// Total number of active Fourier space grid-nodes
            ptrdiff_t get_ntot_fourier() const { return Local_nky * Local_nkz * Nmesh; }
            /// Total number of complex numbers allocated
            ptrdiff_t get_ntot_fourier_alloc() const { return NComplexAlloc; }
            /// The allocated memory
            size_t allocated_storage_bytes() const { return size_t(NComplexAlloc) * sizeof(ComplexType); }
            /// The memory a grid without extra cells on a domain would allocate on this task (without allocating it)
            static size_t storage_bytes(int Nmesh, const PencilDomain & domain);

            /// Range iterator over all the active cells of the main real grid by index
            /// [ e.g. for(auto && real_index : grid.get_real_range()) ]
            PencilRealRange get_real_range() const { return get_real_range_of_rows(0, get_local_nrows()); }
            /// Range iterator over all the active cells of the Fourier grid by index
            FourierRange get_fourier_range() const { return FourierRange(0, get_ntot_fourier()); }

            /// The number of local rows: lines of cells along the last dimension (real: Local_nx * Local_ny in 3D,
            /// Local_nx in 2D). Parallelize loops over rows
            /// [ #pragma omp parallel for
            ///   for (IndexIntType irow = 0; irow < grid.get_local_nrows(); irow++)
            ///       for (auto && real_index : grid.get_real_range_of_rows(irow, irow + 1)) ... ]
            IndexIntType get_local_nrows() const { return IndexIntType(Local_nx) * (N == 3 ? Local_ny : 1); }
            /// Range iterator over the real grid cells of the rows [row_begin, row_end)
            PencilRealRange get_real_range_of_rows(IndexIntType row_begin, IndexIntType row_end) const {
                return PencilRealRange(row_begin, row_end, Nmesh, N == 3 ? int(Local_ny) : 1, row_stride, slice_stride);
            }
            /// The number of rows of the Fourier grid (lines along kx: Local_nky * Local_nkz)
            IndexIntType get_local_nrows_fourier() const { return IndexIntType(Local_nky) * Local_nkz; }
            /// Range iterator over the Fourier grid cells of the rows [row_begin, row_end) (rows along kx)
            FourierRange get_fourier_range_of_rows(IndexIntType row_begin, IndexIntType row_end) const {
                return FourierRange(row_begin * Nmesh, row_end * Nmesh);
            }

            /// Check if we have NaN or Inf in the active cells
            bool nan_in_grids() const;

            /// Fill the extra cells with the cells of the neighboring tasks (periodic)
            void communicate_boundaries();
            /// The reverse: add the extra cells to the cells of the neighboring tasks they belong to (e.g. after
            /// assigning particles to the grid) and set them to zero
            void add_contribution_from_extra_cells();

            /// Print some info about the grid
            void info() const;

            /// Save to file (adds .X to fileprefix where X is ThisTask): a raw copy of the local data (the cells, the
            /// extra cells and the padding) with the size, the extra cells, the pencil edges and the status, for
            /// checkpointing. Read it back with load_from_file with the same number of tasks. Collective
            void dump_to_file(std::string fileprefix) const;
            /// Load a grid written by dump_to_file (same number of tasks): the grid is made with the size, extra cells
            /// and pencils of the file. Collective; aborts if a file cannot be read or does not match
            void load_from_file(std::string fileprefix);

            /// Get the status of the grid: is it currently a real grid or a fourier grid?
            bool get_grid_status_real() const { return grid_is_in_real_space; }
            /// Set the status of the grid: is it currently a real grid or a fourier grid?
            void set_grid_status_real(bool grid_is_a_real_grid) { grid_is_in_real_space = grid_is_a_real_grid; }

            /// For memory logging / info: add a label to the grid
            void add_memory_label(std::string label) { name = std::move(label); }

          private:
            int Nmesh{0};
            int n_extra_left{0};
            int n_extra_right{0};
            PencilDomain domain{};

            // The real cells: x in [Local_x_start, + Local_nx), y in [Local_y_start, + Local_ny) (2D: all y)
            ptrdiff_t Local_nx{0};
            ptrdiff_t Local_x_start{0};
            ptrdiff_t Local_ny{0};
            ptrdiff_t Local_y_start{0};
            // The Fourier cells: all kx, ky in [Local_ky_start, + Local_nky), kz in [Local_kz_start, + Local_nkz)
            // (2D: Local_nkz = 1)
            ptrdiff_t Local_nky{0};
            ptrdiff_t Local_ky_start{0};
            ptrdiff_t Local_nkz{1};
            ptrdiff_t Local_kz_start{0};

            // The real layout (in reals): rows of Nmesh cells and padding, rows_per_slice rows per x-slice (3D: the
            // y-values with the extra ones)
            IndexIntType row_stride{0};
            IndexIntType slice_stride{0};
            // The offset of the main grid in the real data (the extra cells before it)
            IndexIntType real_offset{0};

            ComplexType * data{nullptr};
            ptrdiff_t NComplexAlloc{0};
            bool grid_is_in_real_space{true};
            std::string name{"PencilGrid"};
            std::shared_ptr<PencilTransforms> transforms{};

            void allocate();
            void exchange_x(bool add);
            void exchange_y(bool add);
        };

        /// Perform a real-to-complex FFT from one grid to another
        template <int N>
        void fftw_r2c(const PencilGrid<N> & in_grid, PencilGrid<N> & out_grid) {
            out_grid = in_grid;
            out_grid.fftw_r2c();
        }

        /// Perform a complex-to-real FFT from one grid to another
        template <int N>
        void fftw_c2r(const PencilGrid<N> & in_grid, PencilGrid<N> & out_grid) {
            out_grid = in_grid;
            out_grid.fftw_c2r();
        }

        /// A grid of another size on the same pencils (NUFFT: the fine grid). See fits_grid_like
        template <int N>
        PencilGrid<N> make_grid_like(const PencilGrid<N> & grid, int Nmesh, int n_extra_left, int n_extra_right) {
            return PencilGrid<N>(Nmesh, n_extra_left, n_extra_right, grid.get_domain());
        }
        /// Can a grid of Nmesh cells be made on the same pencils (the edges of the pencils on its cells)?
        template <int N>
        bool fits_grid_like(const PencilGrid<N> & grid, int Nmesh) {
            auto on_cells = [&](const std::vector<double> & edges) {
                for (double e : edges)
                    if (std::abs(e * Nmesh - std::round(e * Nmesh)) > 1e-6)
                        return false;
                for (size_t i = 0; i + 1 < edges.size(); i++)
                    if (std::round(edges[i + 1] * Nmesh) <= std::round(edges[i] * Nmesh))
                        return false;
                return true;
            };
            return on_cells(grid.get_domain().x_edges()) and (N == 2 or on_cells(grid.get_domain().y_edges()));
        }

        /// A PencilGrid on a domain: its pencils, or for a SlabDomain the slabs as P0 x 1 pencils (the tasks in the
        /// order of the slabs)
        template <int N>
        PencilGrid<N> make_grid_on_domain(grid_type_tag<PencilGrid<N>>, int Nmesh, int n_extra_left, int n_extra_right,
                                          const FML::Domain & domain) {
            if (domain.holds<PencilDomain>())
                return PencilGrid<N>(Nmesh, n_extra_left, n_extra_right, domain.get<PencilDomain>());
            assert_mpi(domain.is_slab(), "[make_grid] A PencilGrid needs a PencilDomain or a SlabDomain");
            std::vector<double> xedges(FML::NTasks + 1);
            for (int task = 0; task < FML::NTasks; task++) {
                assert_mpi(task == 0 or std::abs(domain.x_min(task) - domain.x_max(task - 1)) < 1e-12,
                           "[make_grid] The slabs must be in the order of the tasks");
                xedges[task] = domain.x_min(task);
            }
            xedges[FML::NTasks] = 1.0;
            return PencilGrid<N>(Nmesh, n_extra_left, n_extra_right, PencilDomain::from_edges(xedges, {0.0, 1.0}));
        }

    } // namespace GRID
} // namespace FML

#endif
