#include "PencilGrid.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <vector>

namespace FML {
    namespace GRID {

        namespace {

            //=====================================================================
            // FFTW and MPI types of the precision we use
            //=====================================================================
#if defined(SINGLE_PRECISION_FFTW)
            using fftw_iodim_type = fftwf_iodim;
            inline my_fftw_complex * alloc_complex(size_t n) { return fftwf_alloc_complex(n); }
            inline void free_complex(my_fftw_complex * p) { fftwf_free(p); }
            inline my_fftw_plan plan_r2c(fftw_iodim_type * dims,
                                         int howmany_rank,
                                         fftw_iodim_type * howmany,
                                         FloatType * in,
                                         my_fftw_complex * out,
                                         unsigned flags) {
                return fftwf_plan_guru_dft_r2c(1, dims, howmany_rank, howmany, in, out, flags);
            }
            inline my_fftw_plan plan_c2r(fftw_iodim_type * dims,
                                         int howmany_rank,
                                         fftw_iodim_type * howmany,
                                         my_fftw_complex * in,
                                         FloatType * out,
                                         unsigned flags) {
                return fftwf_plan_guru_dft_c2r(1, dims, howmany_rank, howmany, in, out, flags);
            }
            inline my_fftw_plan plan_many(int n, int howmany, my_fftw_complex * data, int sign, unsigned flags) {
                return fftwf_plan_many_dft(1, &n, howmany, data, nullptr, 1, n, data, nullptr, 1, n, sign, flags);
            }
            inline void execute_r2c(my_fftw_plan p, FloatType * in, my_fftw_complex * out) {
                fftwf_execute_dft_r2c(p, in, out);
            }
            inline void execute_c2r(my_fftw_plan p, my_fftw_complex * in, FloatType * out) {
                fftwf_execute_dft_c2r(p, in, out);
            }
            inline void execute_dft(my_fftw_plan p, my_fftw_complex * data) { fftwf_execute_dft(p, data, data); }
            inline void destroy_plan(my_fftw_plan p) { fftwf_destroy_plan(p); }
            const MPI_Datatype mpi_complex_type = MPI_C_FLOAT_COMPLEX;
#elif defined(LONG_DOUBLE_PRECISION_FFTW)
            using fftw_iodim_type = fftwl_iodim;
            inline my_fftw_complex * alloc_complex(size_t n) { return fftwl_alloc_complex(n); }
            inline void free_complex(my_fftw_complex * p) { fftwl_free(p); }
            inline my_fftw_plan plan_r2c(fftw_iodim_type * dims,
                                         int howmany_rank,
                                         fftw_iodim_type * howmany,
                                         FloatType * in,
                                         my_fftw_complex * out,
                                         unsigned flags) {
                return fftwl_plan_guru_dft_r2c(1, dims, howmany_rank, howmany, in, out, flags);
            }
            inline my_fftw_plan plan_c2r(fftw_iodim_type * dims,
                                         int howmany_rank,
                                         fftw_iodim_type * howmany,
                                         my_fftw_complex * in,
                                         FloatType * out,
                                         unsigned flags) {
                return fftwl_plan_guru_dft_c2r(1, dims, howmany_rank, howmany, in, out, flags);
            }
            inline my_fftw_plan plan_many(int n, int howmany, my_fftw_complex * data, int sign, unsigned flags) {
                return fftwl_plan_many_dft(1, &n, howmany, data, nullptr, 1, n, data, nullptr, 1, n, sign, flags);
            }
            inline void execute_r2c(my_fftw_plan p, FloatType * in, my_fftw_complex * out) {
                fftwl_execute_dft_r2c(p, in, out);
            }
            inline void execute_c2r(my_fftw_plan p, my_fftw_complex * in, FloatType * out) {
                fftwl_execute_dft_c2r(p, in, out);
            }
            inline void execute_dft(my_fftw_plan p, my_fftw_complex * data) { fftwl_execute_dft(p, data, data); }
            inline void destroy_plan(my_fftw_plan p) { fftwl_destroy_plan(p); }
            const MPI_Datatype mpi_complex_type = MPI_C_LONG_DOUBLE_COMPLEX;
#else
            using fftw_iodim_type = fftw_iodim;
            inline my_fftw_complex * alloc_complex(size_t n) { return fftw_alloc_complex(n); }
            inline void free_complex(my_fftw_complex * p) { fftw_free(p); }
            inline my_fftw_plan plan_r2c(fftw_iodim_type * dims,
                                         int howmany_rank,
                                         fftw_iodim_type * howmany,
                                         FloatType * in,
                                         my_fftw_complex * out,
                                         unsigned flags) {
                return fftw_plan_guru_dft_r2c(1, dims, howmany_rank, howmany, in, out, flags);
            }
            inline my_fftw_plan plan_c2r(fftw_iodim_type * dims,
                                         int howmany_rank,
                                         fftw_iodim_type * howmany,
                                         my_fftw_complex * in,
                                         FloatType * out,
                                         unsigned flags) {
                return fftw_plan_guru_dft_c2r(1, dims, howmany_rank, howmany, in, out, flags);
            }
            inline my_fftw_plan plan_many(int n, int howmany, my_fftw_complex * data, int sign, unsigned flags) {
                return fftw_plan_many_dft(1, &n, howmany, data, nullptr, 1, n, data, nullptr, 1, n, sign, flags);
            }
            inline void execute_r2c(my_fftw_plan p, FloatType * in, my_fftw_complex * out) {
                fftw_execute_dft_r2c(p, in, out);
            }
            inline void execute_c2r(my_fftw_plan p, my_fftw_complex * in, FloatType * out) {
                fftw_execute_dft_c2r(p, in, out);
            }
            inline void execute_dft(my_fftw_plan p, my_fftw_complex * data) { fftw_execute_dft(p, data, data); }
            inline void destroy_plan(my_fftw_plan p) { fftw_destroy_plan(p); }
            const MPI_Datatype mpi_complex_type = MPI_C_DOUBLE_COMPLEX;
#endif

            inline my_fftw_complex * as_fftw(ComplexType * p) { return reinterpret_cast<my_fftw_complex *>(p); }

            ComplexType * allocate_complex(size_t n) {
                auto * p = alloc_complex(std::max(n, size_t(1)));
                if (p == nullptr)
                    throw std::bad_alloc();
                return reinterpret_cast<ComplexType *>(p);
            }
            void free_complex_array(ComplexType * p) {
                if (p != nullptr)
                    free_complex(reinterpret_cast<my_fftw_complex *>(p));
            }

            // FFTW-aligned scratch memory for a transform
            struct Scratch {
                explicit Scratch(size_t n) : data(allocate_complex(n)) {}
                ~Scratch() { free_complex_array(data); }
                Scratch(const Scratch &) = delete;
                Scratch & operator=(const Scratch &) = delete;
                ComplexType * data;
            };

            //=====================================================================
            // MPI datatypes for the transposes (in units of complex numbers)
            //=====================================================================

            // A block of a C-ordered array
            template <int R>
            MPI_Datatype
            subarray_type(std::array<int, R> sizes, std::array<int, R> subsizes, std::array<int, R> starts) {
                MPI_Datatype type;
                MPI_Type_create_subarray(R,
                                         sizes.data(),
                                         subsizes.data(),
                                         starts.data(),
                                         MPI_ORDER_C,
                                         mpi_complex_type,
                                         &type);
                MPI_Type_commit(&type);
                return type;
            }

            // The elements start + i * stride_outer + j * stride_middle + k * stride_inner for i < n_outer,
            // j < n_middle, k < n_inner, in this order (k fastest)
            MPI_Datatype strided_type(int n_outer,
                                      MPI_Aint stride_outer,
                                      int n_middle,
                                      MPI_Aint stride_middle,
                                      int n_inner,
                                      MPI_Aint stride_inner,
                                      MPI_Aint start) {
                const MPI_Aint bytes = sizeof(ComplexType);
                MPI_Datatype inner, inner_resized, middle, middle_resized, outer, type;
                MPI_Type_create_hvector(n_inner, 1, stride_inner * bytes, mpi_complex_type, &inner);
                MPI_Type_create_resized(inner, 0, stride_middle * bytes, &inner_resized);
                MPI_Type_contiguous(n_middle, inner_resized, &middle);
                MPI_Type_create_resized(middle, 0, stride_outer * bytes, &middle_resized);
                MPI_Type_contiguous(n_outer, middle_resized, &outer);
                int one = 1;
                MPI_Aint displacement = start * bytes;
                MPI_Type_create_hindexed(1, &one, &displacement, outer, &type);
                MPI_Type_commit(&type);
                for (auto * t : {&inner, &inner_resized, &middle, &middle_resized, &outer})
                    MPI_Type_free(t);
                return type;
            }

            // Every task sends one of its datatypes to every task of the communicator
            void alltoallw(const ComplexType * send,
                           const std::vector<MPI_Datatype> & send_types,
                           ComplexType * recv,
                           const std::vector<MPI_Datatype> & recv_types,
                           MPI_Comm comm) {
                const int n = int(send_types.size());
                std::vector<int> counts(n, 1), displacements(n, 0);
                MPI_Alltoallw(send,
                              counts.data(),
                              displacements.data(),
                              send_types.data(),
                              recv,
                              counts.data(),
                              displacements.data(),
                              recv_types.data(),
                              comm);
            }

            // MPI_Sendrecv of a buffer of any size (in chunks whose count fits in an int)
            void sendrecv_bytes(const void * send, void * recv, size_t bytes, int dest, int source, int tag) {
                const size_t maxchunk = size_t(1) << 30;
                const char * s = static_cast<const char *>(send);
                char * r = static_cast<char *>(recv);
                size_t offset = 0;
                do {
                    const int n = int(std::min(maxchunk, bytes - offset));
                    MPI_Sendrecv(s + offset,
                                 n,
                                 MPI_BYTE,
                                 dest,
                                 tag,
                                 r + offset,
                                 n,
                                 MPI_BYTE,
                                 source,
                                 tag,
                                 FML::communicator,
                                 MPI_STATUS_IGNORE);
                    offset += size_t(n);
                } while (offset < bytes);
            }

            // The split of n values over parts: the first n % parts parts get one more. Returns {start, count}
            std::array<ptrdiff_t, 2> split(int n, int parts, int part) {
                const int base = n / parts;
                const int rem = n % parts;
                return {ptrdiff_t(part) * base + std::min(part, rem), base + (part < rem ? 1 : 0)};
            }

            // The first cell of each part from the edges of the domain (they must lie on cells)
            std::vector<ptrdiff_t> cell_edges(const std::vector<double> & edges, int Nmesh, const char * axis) {
                std::vector<ptrdiff_t> cells(edges.size());
                for (size_t i = 0; i < edges.size(); i++) {
                    const double x = edges[i] * Nmesh;
                    cells[i] = ptrdiff_t(std::llround(x));
                    const std::string msg = std::string("[PencilGrid] The ") + axis +
                                            "-edges of the domain must lie on cells (edge * Nmesh an integer)";
                    assert_mpi(std::fabs(x - double(cells[i])) < 1e-6, msg.c_str());
                }
                return cells;
            }

            int fold(int k, int n) { return k <= n / 2 ? k : k - n; }

            // Edges on cells, as even as possible (the first Nmesh % parts parts get one more cell)
            std::vector<double> edges_on_cells(int Nmesh, int parts) {
                std::vector<double> edges(parts + 1);
                for (int i = 0; i < parts; i++)
                    edges[i] = double(split(Nmesh, parts, i)[0]) / double(Nmesh);
                edges[parts] = 1.0;
                return edges;
            }

            // The default pencils: 3D the process grid of FML::pencil_domain() (MPI_Dims_create), 2D NTasks x 1
            PencilDomain default_domain(int Nmesh, int ndim) {
                const auto P = ndim == 3 ? FML::pencil_domain().get<PencilDomain>().process_grid() :
                                           std::array<int, 2>{FML::NTasks, 1};
                return PencilDomain::from_edges(edges_on_cells(Nmesh, P[0]), edges_on_cells(Nmesh, P[1]));
            }

        } // namespace

        //=========================================================================
        // The communicators, datatypes and plans of a layout (shared by copies)
        //=========================================================================
        struct PencilTransforms {
            // Row: the tasks with the same p0 (rank p1), column: the tasks with the same p1 (rank p0)
            MPI_Comm row{MPI_COMM_NULL};
            MPI_Comm column{MPI_COMM_NULL};
            // Real (after the r2c) -> y-pencils, and y-pencils -> x-pencils (3D); real -> x-pencils (2D)
            std::vector<MPI_Datatype> send_row, recv_row, send_column, recv_column;
            my_fftw_plan r2c{nullptr}, c2r{nullptr};
            my_fftw_plan y_forward{nullptr}, y_backward{nullptr};
            my_fftw_plan x_forward{nullptr}, x_backward{nullptr};
            // The number of complex numbers of scratch memory a transform needs
            size_t nscratch{0};

            PencilTransforms() = default;
            PencilTransforms(const PencilTransforms &) = delete;
            PencilTransforms & operator=(const PencilTransforms &) = delete;
            ~PencilTransforms() {
                for (auto p : {r2c, c2r, y_forward, y_backward, x_forward, x_backward})
                    if (p != nullptr)
                        destroy_plan(p);
                int finalized = 0;
                MPI_Finalized(&finalized);
                if (finalized)
                    return;
                for (auto * types : {&send_row, &recv_row, &send_column, &recv_column})
                    for (auto & t : *types)
                        MPI_Type_free(&t);
                for (auto * comm : {&row, &column})
                    if (*comm != MPI_COMM_NULL)
                        MPI_Comm_free(comm);
            }
        };

        //=========================================================================
        // Construction
        //=========================================================================

        template <int N>
        PencilGrid<N>::PencilGrid(int Nmesh, int n_extra_left, int n_extra_right)
            : PencilGrid(Nmesh, n_extra_left, n_extra_right, default_domain(Nmesh, N)) {}

        template <int N>
        PencilGrid<N>::PencilGrid(int Nmesh, int n_extra_left, int n_extra_right, const PencilDomain & domain)
            : Nmesh(Nmesh), n_extra_left(n_extra_left), n_extra_right(n_extra_right), domain(domain) {
            assert_mpi(Nmesh > 0 and n_extra_left >= 0 and n_extra_right >= 0,
                       "[PencilGrid] Need Nmesh > 0 and n_extra >= 0");
            assert_mpi(domain.ntasks() == FML::NTasks, "[PencilGrid] The domain is not made for NTasks tasks");
            const auto P = domain.process_grid();
            const auto p = domain.coords_of(FML::ThisTask);
            assert_mpi(N == 3 or P[1] == 1, "[PencilGrid] In 2D the domain must have P1 = 1 (slabs in x)");

            // The real cells: from the edges of the domain
            const auto xcells = cell_edges(domain.x_edges(), Nmesh, "x");
            Local_x_start = xcells[p[0]];
            Local_nx = xcells[p[0] + 1] - xcells[p[0]];
            if constexpr (N == 3) {
                const auto ycells = cell_edges(domain.y_edges(), Nmesh, "y");
                Local_y_start = ycells[p[1]];
                Local_ny = ycells[p[1] + 1] - ycells[p[1]];
            } else {
                Local_y_start = 0;
                Local_ny = Nmesh;
            }
            const int nextra = std::max(n_extra_left, n_extra_right);
            assert_mpi(Local_nx >= std::max(1, nextra) and Local_ny >= std::max(1, N == 3 ? nextra : 0),
                       "[PencilGrid] Every task needs at least one cell (and as many as the extra cells) in x and y");

            // The Fourier cells: 3D ky in [0, Nmesh) over P0 and kz in [0, Nmesh/2] over P1, 2D ky in [0, Nmesh/2]
            // over P0
            const int nlast_fourier = Nmesh / 2 + 1;
            if constexpr (N == 3) {
                assert_mpi(Nmesh >= P[0] and nlast_fourier >= P[1],
                           "[PencilGrid] Need Nmesh >= P0 and Nmesh/2+1 >= P1 (every task has Fourier cells)");
                const auto ky = split(Nmesh, P[0], p[0]);
                const auto kz = split(nlast_fourier, P[1], p[1]);
                Local_ky_start = ky[0];
                Local_nky = ky[1];
                Local_kz_start = kz[0];
                Local_nkz = kz[1];
            } else {
                assert_mpi(nlast_fourier >= P[0],
                           "[PencilGrid] Need Nmesh/2+1 >= NTasks (every task has Fourier cells)");
                const auto ky = split(nlast_fourier, P[0], p[0]);
                Local_ky_start = ky[0];
                Local_nky = ky[1];
                Local_kz_start = 0;
                Local_nkz = 1;
            }

            // The real layout: rows of 2 (Nmesh/2+1) reals; in 3D the extra cells in y are in each x-slice
            row_stride = 2 * nlast_fourier;
            if constexpr (N == 3) {
                slice_stride = (Local_ny + n_extra_left + n_extra_right) * row_stride;
                real_offset = n_extra_left * slice_stride + n_extra_left * row_stride;
            } else {
                slice_stride = row_stride;
                real_offset = n_extra_left * slice_stride;
            }
            const ptrdiff_t ncomplex_real = (Local_nx + n_extra_left + n_extra_right) * slice_stride / 2;
            NComplexAlloc = std::max(ncomplex_real, get_ntot_fourier());

            allocate();
            fill_real_grid(0.0);
        }

        template <int N>
        void PencilGrid<N>::allocate() {
            data = allocate_complex(size_t(NComplexAlloc));

            const auto P = domain.process_grid();
            const auto p = domain.coords_of(FML::ThisTask);
            const int nx = int(Local_nx), ny = int(Local_ny), nky = int(Local_nky), nkz = int(Local_nkz);
            const int nl = n_extra_left;
            const auto xcells = cell_edges(domain.x_edges(), Nmesh, "x");

            transforms = std::make_shared<PencilTransforms>();
            auto & t = *transforms;
            MPI_Comm_split(FML::communicator, p[0], p[1], &t.row);
            MPI_Comm_split(FML::communicator, p[1], p[0], &t.column);

            // The plans are made with FFTW_ESTIMATE (the data is not touched) on FFTW-aligned memory, and executed
            // on the memory of any grid with this layout (the same alignment)
            const unsigned flags = FFTW_ESTIMATE;
            FloatType * real = reinterpret_cast<FloatType *>(data) + real_offset;
            my_fftw_complex * complex = as_fftw(data) + real_offset / 2;

            if constexpr (N == 3) {
                const auto ycells = cell_edges(domain.y_edges(), Nmesh, "y");
                const int nzc = Nmesh / 2 + 1;
                const int nxa = nx + n_extra_left + n_extra_right;
                const int nya = ny + n_extra_left + n_extra_right;

                // Row transpose: the real pencils [nxa][nya][nzc] (after the r2c) -> y-pencils [nx][nkz][Nmesh]
                for (int j = 0; j < P[1]; j++) {
                    const auto kz = split(nzc, P[1], j);
                    const int ystart = int(ycells[j]);
                    const int ny_j = int(ycells[j + 1] - ycells[j]);
                    t.send_row.push_back(subarray_type<3>({nxa, nya, nzc}, {nx, ny, int(kz[1])}, {nl, nl, int(kz[0])}));
                    t.recv_row.push_back(strided_type(nx, MPI_Aint(nkz) * Nmesh, ny_j, 1, nkz, Nmesh, ystart));
                }
                // Column transpose: the y-pencils [nx][nkz][Nmesh] -> x-pencils [nky][nkz][Nmesh]
                for (int i = 0; i < P[0]; i++) {
                    const auto ky = split(Nmesh, P[0], i);
                    const int xstart = int(xcells[i]);
                    const int nx_i = int(xcells[i + 1] - xcells[i]);
                    t.send_column.push_back(
                        subarray_type<3>({nx, nkz, Nmesh}, {nx, nkz, int(ky[1])}, {0, 0, int(ky[0])}));
                    t.recv_column.push_back(strided_type(nx_i, 1, nkz, Nmesh, nky, MPI_Aint(nkz) * Nmesh, xstart));
                }
                t.nscratch = size_t(nx) * nkz * Nmesh;
                Scratch scratch(t.nscratch);

                fftw_iodim_type dim{Nmesh, 1, 1};
                fftw_iodim_type forward[2] = {{nx, int(slice_stride), int(slice_stride / 2)},
                                              {ny, int(row_stride), int(row_stride / 2)}};
                fftw_iodim_type backward[2] = {{nx, int(slice_stride / 2), int(slice_stride)},
                                               {ny, int(row_stride / 2), int(row_stride)}};
                t.r2c = plan_r2c(&dim, 2, forward, real, complex, flags);
                t.c2r = plan_c2r(&dim, 2, backward, complex, real, flags);
                t.y_forward = plan_many(Nmesh, nx * nkz, as_fftw(scratch.data), FFTW_FORWARD, flags);
                t.y_backward = plan_many(Nmesh, nx * nkz, as_fftw(scratch.data), FFTW_BACKWARD, flags);
                t.x_forward = plan_many(Nmesh, nky * nkz, as_fftw(data), FFTW_FORWARD, flags);
                t.x_backward = plan_many(Nmesh, nky * nkz, as_fftw(data), FFTW_BACKWARD, flags);
                assert_mpi(t.r2c and t.c2r and t.y_forward and t.y_backward and t.x_forward and t.x_backward,
                           "[PencilGrid] FFTW failed to make a plan");
            } else {
                const int nyc = Nmesh / 2 + 1;
                const int nxa = nx + n_extra_left + n_extra_right;

                // Column transpose: the real slabs [nxa][nyc] (after the r2c) -> x-pencils [nky][Nmesh]
                for (int i = 0; i < P[0]; i++) {
                    const auto ky = split(nyc, P[0], i);
                    const int xstart = int(xcells[i]);
                    const int nx_i = int(xcells[i + 1] - xcells[i]);
                    t.send_column.push_back(subarray_type<2>({nxa, nyc}, {nx, int(ky[1])}, {nl, int(ky[0])}));
                    t.recv_column.push_back(strided_type(nx_i, 1, 1, 1, nky, Nmesh, xstart));
                }
                t.nscratch = size_t(nky) * Nmesh;
                Scratch scratch(t.nscratch);

                fftw_iodim_type dim{Nmesh, 1, 1};
                fftw_iodim_type forward[1] = {{nx, int(row_stride), int(row_stride / 2)}};
                fftw_iodim_type backward[1] = {{nx, int(row_stride / 2), int(row_stride)}};
                t.r2c = plan_r2c(&dim, 1, forward, real, complex, flags);
                t.c2r = plan_c2r(&dim, 1, backward, complex, real, flags);
                t.x_forward = plan_many(Nmesh, nky, as_fftw(scratch.data), FFTW_FORWARD, flags);
                t.x_backward = plan_many(Nmesh, nky, as_fftw(scratch.data), FFTW_BACKWARD, flags);
                assert_mpi(t.r2c and t.c2r and t.x_forward and t.x_backward, "[PencilGrid] FFTW failed to make a plan");
            }
        }

        template <int N>
        PencilGrid<N>::PencilGrid(const PencilGrid & rhs)
            : Nmesh(rhs.Nmesh), n_extra_left(rhs.n_extra_left), n_extra_right(rhs.n_extra_right), domain(rhs.domain),
              Local_nx(rhs.Local_nx), Local_x_start(rhs.Local_x_start), Local_ny(rhs.Local_ny),
              Local_y_start(rhs.Local_y_start), Local_nky(rhs.Local_nky), Local_ky_start(rhs.Local_ky_start),
              Local_nkz(rhs.Local_nkz), Local_kz_start(rhs.Local_kz_start), row_stride(rhs.row_stride),
              slice_stride(rhs.slice_stride), real_offset(rhs.real_offset), NComplexAlloc(rhs.NComplexAlloc),
              grid_is_in_real_space(rhs.grid_is_in_real_space), name(rhs.name), transforms(rhs.transforms) {
            if (rhs.data != nullptr) {
                data = allocate_complex(size_t(NComplexAlloc));
                std::memcpy(data, rhs.data, size_t(NComplexAlloc) * sizeof(ComplexType));
            }
        }

        template <int N>
        PencilGrid<N> & PencilGrid<N>::operator=(const PencilGrid & rhs) {
            if (this != &rhs)
                *this = PencilGrid(rhs);
            return *this;
        }

        template <int N>
        PencilGrid<N>::PencilGrid(PencilGrid && rhs) noexcept {
            *this = std::move(rhs);
        }

        template <int N>
        PencilGrid<N> & PencilGrid<N>::operator=(PencilGrid && rhs) noexcept {
            if (this == &rhs)
                return *this;
            free_complex_array(data);
            Nmesh = std::exchange(rhs.Nmesh, 0);
            n_extra_left = std::exchange(rhs.n_extra_left, 0);
            n_extra_right = std::exchange(rhs.n_extra_right, 0);
            domain = std::move(rhs.domain);
            Local_nx = std::exchange(rhs.Local_nx, 0);
            Local_x_start = std::exchange(rhs.Local_x_start, 0);
            Local_ny = std::exchange(rhs.Local_ny, 0);
            Local_y_start = std::exchange(rhs.Local_y_start, 0);
            Local_nky = std::exchange(rhs.Local_nky, 0);
            Local_ky_start = std::exchange(rhs.Local_ky_start, 0);
            Local_nkz = std::exchange(rhs.Local_nkz, 1);
            Local_kz_start = std::exchange(rhs.Local_kz_start, 0);
            row_stride = std::exchange(rhs.row_stride, 0);
            slice_stride = std::exchange(rhs.slice_stride, 0);
            real_offset = std::exchange(rhs.real_offset, 0);
            data = std::exchange(rhs.data, nullptr);
            NComplexAlloc = std::exchange(rhs.NComplexAlloc, 0);
            grid_is_in_real_space = std::exchange(rhs.grid_is_in_real_space, true);
            name = std::move(rhs.name);
            transforms = std::move(rhs.transforms);
            return *this;
        }

        template <int N>
        PencilGrid<N>::~PencilGrid() {
            free_complex_array(data);
        }

        template <int N>
        void PencilGrid<N>::free() {
            *this = PencilGrid();
        }

        //=========================================================================
        // Fourier transforms
        //=========================================================================

        template <int N>
        void PencilGrid<N>::fftw_r2c() {
            assert_mpi(data != nullptr, "[PencilGrid::fftw_r2c] The grid is not allocated");
            auto & t = *transforms;
            Scratch scratch(t.nscratch);
            FloatType * real = get_real_grid();
            my_fftw_complex * complex = as_fftw(data) + real_offset / 2;

            execute_r2c(t.r2c, real, complex);
            if constexpr (N == 3) {
                alltoallw(data, t.send_row, scratch.data, t.recv_row, t.row);
                execute_dft(t.y_forward, as_fftw(scratch.data));
                alltoallw(scratch.data, t.send_column, data, t.recv_column, t.column);
                execute_dft(t.x_forward, as_fftw(data));
            } else {
                alltoallw(data, t.send_column, scratch.data, t.recv_column, t.column);
                execute_dft(t.x_forward, as_fftw(scratch.data));
                std::copy(scratch.data, scratch.data + get_ntot_fourier(), data);
            }
            FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::ForwardFFTs);
            grid_is_in_real_space = false;

            // Normalize
            const FloatType norm = FloatType(1.0 / std::pow(double(Nmesh), N));
            const ptrdiff_t ntot = get_ntot_fourier();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (ptrdiff_t i = 0; i < ntot; i++)
                data[i] *= norm;
        }

        template <int N>
        void PencilGrid<N>::fftw_c2r() {
            assert_mpi(data != nullptr, "[PencilGrid::fftw_c2r] The grid is not allocated");
            auto & t = *transforms;
            Scratch scratch(t.nscratch);
            FloatType * real = get_real_grid();
            my_fftw_complex * complex = as_fftw(data) + real_offset / 2;

            if constexpr (N == 3) {
                execute_dft(t.x_backward, as_fftw(data));
                alltoallw(data, t.recv_column, scratch.data, t.send_column, t.column);
                execute_dft(t.y_backward, as_fftw(scratch.data));
                alltoallw(scratch.data, t.recv_row, data, t.send_row, t.row);
            } else {
                std::copy(data, data + get_ntot_fourier(), scratch.data);
                execute_dft(t.x_backward, as_fftw(scratch.data));
                alltoallw(scratch.data, t.recv_column, data, t.send_column, t.column);
            }
            execute_c2r(t.c2r, complex, real);
            FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::InverseFFTs);
            grid_is_in_real_space = true;
        }

        //=========================================================================
        // The extra cells
        //=========================================================================

        template <int N>
        void PencilGrid<N>::communicate_boundaries() {
            if (n_extra_left + n_extra_right == 0)
                return;
            assert_mpi(data != nullptr, "[PencilGrid::communicate_boundaries] The grid is not allocated");
            // y first (the main x-slices), then x (whole slices with the extra y cells: fills the corners)
            if constexpr (N == 3)
                exchange_y(false);
            exchange_x(false);
        }

        template <int N>
        void PencilGrid<N>::add_contribution_from_extra_cells() {
            if (n_extra_left + n_extra_right == 0)
                return;
            assert_mpi(data != nullptr, "[PencilGrid::add_contribution_from_extra_cells] The grid is not allocated");
            // The reverse order: x first (the corners go to the extra y cells of the x-neighbors), then y
            exchange_x(true);
            if constexpr (N == 3)
                exchange_y(true);
        }

        // The extra x-slices: copy (add = false) or add back (add = true). An x-slice is contiguous in memory
        template <int N>
        void PencilGrid<N>::exchange_x(bool add) {
            const auto p = domain.coords_of(FML::ThisTask);
            const int left = domain.rank_of(p[0] - 1, p[1]);
            const int right = domain.rank_of(p[0] + 1, p[1]);
            FloatType * grid = get_real_grid();
            auto slices = [&](ptrdiff_t ix) {
                return grid + ix * slice_stride - (N == 3 ? n_extra_left * row_stride : 0);
            };
            const size_t bytes_per_slice = size_t(slice_stride) * sizeof(FloatType);
            const ptrdiff_t nl = n_extra_left, nr = n_extra_right;

            if (not add) {
                // My first nr slices are the right extra slices of the left task, my last nl its left extra ones
                if (nr > 0)
                    sendrecv_bytes(slices(0), slices(Local_nx), nr * bytes_per_slice, left, right, 30);
                if (nl > 0)
                    sendrecv_bytes(slices(Local_nx - nl), slices(-nl), nl * bytes_per_slice, right, left, 31);
                return;
            }

            std::vector<FloatType> buffer;
            auto add_to = [&](FloatType * target, size_t n) {
                for (size_t i = 0; i < n; i++)
                    target[i] += buffer[i];
            };
            if (nr > 0) {
                const size_t n = size_t(nr * slice_stride);
                buffer.resize(n);
                sendrecv_bytes(slices(Local_nx), buffer.data(), n * sizeof(FloatType), right, left, 32);
                add_to(slices(0), n);
            }
            if (nl > 0) {
                const size_t n = size_t(nl * slice_stride);
                buffer.resize(n);
                sendrecv_bytes(slices(-nl), buffer.data(), n * sizeof(FloatType), left, right, 33);
                add_to(slices(Local_nx - nl), n);
            }
            std::fill(slices(-nl), slices(0), FloatType(0));
            std::fill(slices(Local_nx), slices(Local_nx + nr), FloatType(0));
        }

        // The extra y-cells of the main x-slices (3D): copy (add = false) or add back (add = true)
        template <int N>
        void PencilGrid<N>::exchange_y(bool add) {
            if constexpr (N == 3) {
                const auto p = domain.coords_of(FML::ThisTask);
                const int down = domain.rank_of(p[0], p[1] - 1);
                const int up = domain.rank_of(p[0], p[1] + 1);
                FloatType * grid = get_real_grid();
                const ptrdiff_t nx = Local_nx, ny = Local_ny;

                // Pack / unpack the rows [iy0, iy0 + n) of all the main x-slices
                auto pack = [&](ptrdiff_t iy0, ptrdiff_t n, std::vector<FloatType> & buf) {
                    const ptrdiff_t block = n * row_stride;
                    buf.resize(size_t(nx * block));
                    for (ptrdiff_t ix = 0; ix < nx; ix++)
                        std::copy(grid + ix * slice_stride + iy0 * row_stride,
                                  grid + ix * slice_stride + iy0 * row_stride + block,
                                  buf.begin() + ix * block);
                };
                auto unpack = [&](ptrdiff_t iy0, ptrdiff_t n, const std::vector<FloatType> & buf, bool add_values) {
                    const ptrdiff_t block = n * row_stride;
                    for (ptrdiff_t ix = 0; ix < nx; ix++) {
                        FloatType * target = grid + ix * slice_stride + iy0 * row_stride;
                        for (ptrdiff_t i = 0; i < block; i++)
                            target[i] = add_values ? target[i] + buf[ix * block + i] : buf[ix * block + i];
                    }
                };
                auto exchange =
                    [&](ptrdiff_t send_iy0, ptrdiff_t recv_iy0, ptrdiff_t n, int dest, int source, int tag) {
                        if (n == 0)
                            return;
                        std::vector<FloatType> send, recv(size_t(nx * n * row_stride));
                        pack(send_iy0, n, send);
                        sendrecv_bytes(send.data(), recv.data(), send.size() * sizeof(FloatType), dest, source, tag);
                        unpack(recv_iy0, n, recv, add);
                    };
                const ptrdiff_t nl = n_extra_left, nr = n_extra_right;
                if (not add) {
                    // My first nr rows are the right extra rows of the task below, my last nl its left extra ones
                    exchange(0, ny, nr, down, up, 40);
                    exchange(ny - nl, -nl, nl, up, down, 41);
                } else {
                    exchange(ny, 0, nr, up, down, 42);
                    exchange(-nl, ny - nl, nl, down, up, 43);
                    std::vector<FloatType> zeros;
                    for (auto [iy0, n] : {std::array<ptrdiff_t, 2>{-nl, nl}, std::array<ptrdiff_t, 2>{ny, nr}}) {
                        zeros.assign(size_t(nx * n * row_stride), FloatType(0));
                        unpack(iy0, n, zeros, false);
                    }
                }
            } else {
                (void)add;
            }
        }

        //=========================================================================
        // Access
        //=========================================================================

        template <int N>
        FloatType * PencilGrid<N>::get_real_grid() {
            return reinterpret_cast<FloatType *>(data) + real_offset;
        }

        template <int N>
        IndexIntType PencilGrid<N>::get_index_real(const std::array<int, N> & coord) const {
#ifdef BOUNDSCHECK_FFTWGRID
            assert_mpi(-n_extra_left <= coord[0] and coord[0] < Local_nx + n_extra_right and 0 <= coord[N - 1] and
                           coord[N - 1] < Nmesh,
                       "[PencilGrid::get_index_real] Bounds check failed\n");
            if constexpr (N == 3)
                assert_mpi(-n_extra_left <= coord[1] and coord[1] < Local_ny + n_extra_right,
                           "[PencilGrid::get_index_real] Bounds check failed\n");
#endif
            if constexpr (N == 3)
                return coord[0] * slice_stride + coord[1] * row_stride + coord[2];
            return coord[0] * slice_stride + coord[1];
        }

        template <int N>
        std::array<int, N> PencilGrid<N>::get_coord_from_index(const IndexIntType index_real) const {
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

        template <int N>
        IndexIntType PencilGrid<N>::get_index_fourier(const std::array<int, N> & coord) const {
#ifdef BOUNDSCHECK_FFTWGRID
            assert_mpi(0 <= coord[0] and coord[0] < Nmesh and 0 <= coord[1] and coord[1] < Local_nky,
                       "[PencilGrid::get_index_fourier] Bounds check failed\n");
            if constexpr (N == 3)
                assert_mpi(0 <= coord[2] and coord[2] < Local_nkz,
                           "[PencilGrid::get_index_fourier] Bounds check failed\n");
#endif
            if constexpr (N == 3)
                return (IndexIntType(coord[1]) * Local_nkz + coord[2]) * Nmesh + coord[0];
            return IndexIntType(coord[1]) * Nmesh + coord[0];
        }

        template <int N>
        std::array<int, N> PencilGrid<N>::get_fourier_coord_from_index(const IndexIntType index) const {
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

        template <int N>
        FloatType PencilGrid<N>::get_real(const std::array<int, N> & coord) const {
            return get_real_from_index(get_index_real(coord));
        }

        template <int N>
        FloatType PencilGrid<N>::get_real_from_index(const IndexIntType index) const {
            return reinterpret_cast<const FloatType *>(data)[real_offset + index];
        }

        template <int N>
        ComplexType PencilGrid<N>::get_fourier(const std::array<int, N> & coord) const {
            return data[get_index_fourier(coord)];
        }

        template <int N>
        ComplexType PencilGrid<N>::get_fourier_from_index(const IndexIntType index) const {
            return data[index];
        }

        template <int N>
        void PencilGrid<N>::set_real(const std::array<int, N> & coord, const FloatType value) {
            get_real_grid()[get_index_real(coord)] = value;
        }

        template <int N>
        void PencilGrid<N>::set_real_from_index(const IndexIntType index, const FloatType value) {
            get_real_grid()[index] = value;
        }

        template <int N>
        void PencilGrid<N>::add_real(const std::array<int, N> & coord, const FloatType value) {
            get_real_grid()[get_index_real(coord)] += value;
        }

        template <int N>
        void PencilGrid<N>::set_fourier(const std::array<int, N> & coord, const ComplexType value) {
            data[get_index_fourier(coord)] = value;
        }

        template <int N>
        void PencilGrid<N>::set_fourier_from_index(const IndexIntType index, const ComplexType value) {
            data[index] = value;
        }

        //=========================================================================
        // Positions and wave-vectors
        //=========================================================================

        template <int N>
        std::array<double, N> PencilGrid<N>::get_real_position(const std::array<int, N> & coord) const {
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

        template <int N>
        std::array<int, N> PencilGrid<N>::get_fourier_integer_wavevector_from_index(const IndexIntType index) const {
            const auto coord = get_fourier_coord_from_index(index);
            std::array<int, N> k;
            k[0] = fold(coord[0], Nmesh);
            if constexpr (N == 3) {
                k[1] = fold(int(Local_ky_start) + coord[1], Nmesh);
                k[2] = int(Local_kz_start) + coord[2];
            } else {
                k[1] = int(Local_ky_start) + coord[1];
            }
            return k;
        }

        template <int N>
        std::array<double, N> PencilGrid<N>::get_fourier_wavevector(const std::array<int, N> & coord) const {
            return get_fourier_wavevector_from_index(get_index_fourier(coord));
        }

        template <int N>
        std::array<double, N> PencilGrid<N>::get_fourier_wavevector_from_index(const IndexIntType index) const {
            const auto k = get_fourier_integer_wavevector_from_index(index);
            std::array<double, N> kvec;
            for (int idim = 0; idim < N; idim++)
                kvec[idim] = 2.0 * M_PI * k[idim];
            return kvec;
        }

        template <int N>
        void PencilGrid<N>::get_fourier_wavevector_and_norm2_by_index(const IndexIntType index,
                                                                      std::array<double, N> & kvec,
                                                                      double & kmag2) const {
            kvec = get_fourier_wavevector_from_index(index);
            kmag2 = 0.0;
            for (int idim = 0; idim < N; idim++)
                kmag2 += kvec[idim] * kvec[idim];
        }

        template <int N>
        void PencilGrid<N>::get_fourier_wavevector_and_norm_by_index(const IndexIntType index,
                                                                     std::array<double, N> & kvec,
                                                                     double & kmag) const {
            get_fourier_wavevector_and_norm2_by_index(index, kvec, kmag);
            kmag = std::sqrt(kmag);
        }

        //=========================================================================
        // Filling and checking
        //=========================================================================

        template <int N>
        void PencilGrid<N>::fill_real_grid(const FloatType val) {
            FloatType * begin = reinterpret_cast<FloatType *>(data);
            std::fill(begin, begin + 2 * NComplexAlloc, val);
            grid_is_in_real_space = true;
        }

        template <int N>
        void PencilGrid<N>::fill_fourier_grid(const ComplexType val) {
            std::fill(data, data + NComplexAlloc, val);
            grid_is_in_real_space = false;
        }

        template <int N>
        void PencilGrid<N>::fill_real_grid(const std::function<FloatType(std::array<double, N> &)> & func) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (IndexIntType irow = 0; irow < get_local_nrows(); irow++) {
                for (auto && real_index : get_real_range_of_rows(irow, irow + 1)) {
                    auto pos = get_real_position(get_coord_from_index(real_index));
                    set_real_from_index(real_index, func(pos));
                }
            }
            grid_is_in_real_space = true;
            communicate_boundaries();
        }

        template <int N>
        void PencilGrid<N>::fill_fourier_grid(const std::function<ComplexType(std::array<double, N> &)> & func) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (IndexIntType irow = 0; irow < get_local_nrows_fourier(); irow++) {
                for (auto && fourier_index : get_fourier_range_of_rows(irow, irow + 1)) {
                    auto kvec = get_fourier_wavevector_from_index(fourier_index);
                    set_fourier_from_index(fourier_index, func(kvec));
                }
            }
            grid_is_in_real_space = false;
        }

        template <int N>
        bool PencilGrid<N>::nan_in_grids() const {
            if (grid_is_in_real_space) {
                for (auto && real_index : get_real_range())
                    if (not std::isfinite(get_real_from_index(real_index)))
                        return true;
                return false;
            }
            for (auto && fourier_index : get_fourier_range()) {
                const auto z = get_fourier_from_index(fourier_index);
                if (not std::isfinite(z.real()) or not std::isfinite(z.imag()))
                    return true;
            }
            return false;
        }

        template <int N>
        void PencilGrid<N>::info() const {
            if (FML::ThisTask != 0)
                return;
            const auto P = domain.process_grid();
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Info about PencilGrid. Grid label: [" << name << "]\n";
            std::cout << "# Status of grid: " << (grid_is_in_real_space ? "[Realspace]" : "[Fourierspace]")
                      << " NDIM: [" << N << "]\n";
            std::cout << "# Nmesh                  " << Nmesh << "\n";
            std::cout << "# Process grid           " << P[0] << " x " << P[1] << "\n";
            std::cout << "# Local x, y (task 0)    " << Local_nx << " x " << Local_ny << " (+ " << n_extra_left
                      << " left, " << n_extra_right << " right)\n";
            std::cout << "# Local ky, kz (task 0)  " << Local_nky << " x " << Local_nkz << "\n";
            std::cout << "# Allocated (task 0)     " << double(allocated_storage_bytes()) / 1e6 << " MB\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }

        template class PencilGrid<2>;
        template class PencilGrid<3>;

    } // namespace GRID
} // namespace FML
