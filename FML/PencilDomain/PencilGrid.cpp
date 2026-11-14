#include "PencilGrid.h"
#include "CPencilGrid.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
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
            inline my_fftw_plan plan_dft(fftw_iodim_type * dims, int howmany_rank, fftw_iodim_type * howmany,
                                         my_fftw_complex * data, int sign, unsigned flags) {
                return fftwf_plan_guru_dft(1, dims, howmany_rank, howmany, data, data, sign, flags);
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
            inline my_fftw_plan plan_dft(fftw_iodim_type * dims, int howmany_rank, fftw_iodim_type * howmany,
                                         my_fftw_complex * data, int sign, unsigned flags) {
                return fftwl_plan_guru_dft(1, dims, howmany_rank, howmany, data, data, sign, flags);
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
            inline my_fftw_plan plan_dft(fftw_iodim_type * dims, int howmany_rank, fftw_iodim_type * howmany,
                                         my_fftw_complex * data, int sign, unsigned flags) {
                return fftw_plan_guru_dft(1, dims, howmany_rank, howmany, data, data, sign, flags);
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

            //=====================================================================
            // The transposes: blocks of a 3D index space copied between strided layouts
            //=====================================================================

            // The elements offset + i * stride[0] + j * stride[1] + k * stride[2] for i < n[0], j < n[1], k < n[2] (in
            // units of complex numbers)
            struct BlockView {
                ptrdiff_t offset{0};
                std::array<ptrdiff_t, 3> stride{};
                std::array<int, 3> n{};
                size_t count() const { return size_t(n[0]) * size_t(n[1]) * size_t(n[2]); }
                // The same block stored contiguously (in this order) at offset
                static BlockView contiguous(std::array<int, 3> n, ptrdiff_t offset = 0) {
                    return {offset, {ptrdiff_t(n[1]) * n[2], n[2], 1}, n};
                }
            };

            // Copy a block from one layout to another. When the two layouts are contiguous along different indices
            // (a transpose) the copy is done in tiles, so that both are read / written a cache line at a time
            void copy_block(ComplexType * dst, const BlockView & d, const ComplexType * src, const BlockView & s) {
                const auto & n = s.n;
                if (n[0] == 0 or n[1] == 0 or n[2] == 0)
                    return;
                // The index along which a layout is contiguous (the smallest stride among the indices with n > 1)
                auto fastest = [&](const std::array<ptrdiff_t, 3> & stride) {
                    int best = -1;
                    for (int i = 0; i < 3; i++)
                        if (n[i] > 1 and (best < 0 or std::abs(stride[i]) < std::abs(stride[best])))
                            best = i;
                    return best < 0 ? 2 : best;
                };
                const int si = fastest(s.stride), di = fastest(d.stride);
                ComplexType * d0 = dst + d.offset;
                const ComplexType * s0 = src + s.offset;

                if (si == di) {
                    // The same fast index: copy lines
                    const int a = si == 0 ? 1 : 0, b = 3 - si - a;
                    const ptrdiff_t ds = d.stride[si], ss = s.stride[si];
#ifdef USE_OMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
                    for (int i = 0; i < n[a]; i++)
                        for (int j = 0; j < n[b]; j++) {
                            ComplexType * dp = d0 + i * d.stride[a] + j * d.stride[b];
                            const ComplexType * sp = s0 + i * s.stride[a] + j * s.stride[b];
                            if (ds == 1 and ss == 1)
                                std::copy(sp, sp + n[si], dp);
                            else
                                for (int k = 0; k < n[si]; k++)
                                    dp[k * ds] = sp[k * ss];
                        }
                    return;
                }

                // A transpose of the indices si and di (o the third one): tiles of tile x tile
                constexpr int tile = 32;
                const int o = 3 - si - di;
                const int ntiles = (n[di] + tile - 1) / tile;
#ifdef USE_OMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
                for (int io = 0; io < n[o]; io++)
                    for (int td = 0; td < ntiles; td++) {
                        const int a_end = std::min(n[di], (td + 1) * tile);
                        for (int b0 = 0; b0 < n[si]; b0 += tile) {
                            const int b_end = std::min(n[si], b0 + tile);
                            for (int a = td * tile; a < a_end; a++) {
                                ComplexType * dp = d0 + io * d.stride[o] + a * d.stride[di];
                                const ComplexType * sp = s0 + io * s.stride[o] + a * s.stride[di];
                                for (int b = b0; b < b_end; b++)
                                    dp[b * d.stride[si]] = sp[b * s.stride[si]];
                            }
                        }
                    }
            }

            // One all-to-all redistribution between the tasks of a communicator. For every partner p: send[p] is
            // where the data this task sends to p is in the source, recv[p] where the data from p goes in the
            // destination, both with the indices in the order the sender packs them. The reverse redistribution
            // (backward transform) swaps the two
            struct Exchange {
                MPI_Comm comm{MPI_COMM_NULL};
                int size{1};
                std::vector<BlockView> send, recv;
                std::vector<int> send_counts, send_displacements, recv_counts, recv_displacements;
                size_t send_total{0}, recv_total{0};

                void finalize() {
                    auto layout = [](const std::vector<BlockView> & views,
                                     std::vector<int> & counts,
                                     std::vector<int> & displacements) {
                        size_t total = 0;
                        for (const auto & v : views) {
                            assert_mpi(v.count() <= size_t(std::numeric_limits<int>::max()) and
                                           total <= size_t(std::numeric_limits<int>::max()),
                                       "[PencilGrid] A transpose block is too large for MPI (int counts)");
                            counts.push_back(int(v.count()));
                            displacements.push_back(int(total));
                            total += v.count();
                        }
                        return total;
                    };
                    send_total = layout(send, send_counts, send_displacements);
                    recv_total = layout(recv, recv_counts, recv_displacements);
                }
            };

            // Redistribute from src to dst (reverse: the backward direction): pack into send_buffer, exchange into
            // recv_buffer, unpack. With one task the block is copied directly (if src and dst differ)
            void run_exchange(const Exchange & ex,
                              bool reverse,
                              const ComplexType * src,
                              ComplexType * dst,
                              ComplexType * send_buffer,
                              ComplexType * recv_buffer) {
                const auto & from = reverse ? ex.recv : ex.send;
                const auto & to = reverse ? ex.send : ex.recv;
                if (ex.size == 1 and src != dst) {
                    copy_block(dst, to[0], src, from[0]);
                    return;
                }
                const auto & scounts = reverse ? ex.recv_counts : ex.send_counts;
                const auto & sdispl = reverse ? ex.recv_displacements : ex.send_displacements;
                const auto & rcounts = reverse ? ex.send_counts : ex.recv_counts;
                const auto & rdispl = reverse ? ex.send_displacements : ex.recv_displacements;
                for (int p = 0; p < ex.size; p++)
                    copy_block(send_buffer, BlockView::contiguous(from[p].n, sdispl[p]), src, from[p]);
                const ComplexType * received = send_buffer;
                if (ex.size > 1) {
                    MPI_Alltoallv(send_buffer,
                                  scounts.data(),
                                  sdispl.data(),
                                  mpi_complex_type,
                                  recv_buffer,
                                  rcounts.data(),
                                  rdispl.data(),
                                  mpi_complex_type,
                                  ex.comm);
                    received = recv_buffer;
                }
                for (int p = 0; p < ex.size; p++)
                    copy_block(dst, to[p], received, BlockView::contiguous(to[p].n, rdispl[p]));
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
            // 3D: real pencils (after the r2c) -> y-pencils in the row (the tasks with the same p0), y-pencils ->
            // x-pencils in the column (the same p1). 2D: real slabs -> x-pencils in the column (all tasks)
            Exchange row, column;
            my_fftw_plan r2c{nullptr}, c2r{nullptr};
            my_fftw_plan y_forward{nullptr}, y_backward{nullptr};
            my_fftw_plan x_forward{nullptr}, x_backward{nullptr};
            // FFTW-aligned work memory of the transforms (kept: shared by the grids with this layout, so they must not
            // be transformed at the same time from different threads)
            ComplexType * work{nullptr};
            size_t nwork{0};

            PencilTransforms() = default;
            PencilTransforms(const PencilTransforms &) = delete;
            PencilTransforms & operator=(const PencilTransforms &) = delete;
            ~PencilTransforms() {
                for (auto p : {r2c, c2r, y_forward, y_backward, x_forward, x_backward})
                    if (p != nullptr)
                        destroy_plan(p);
                free_complex_array(work);
                int finalized = 0;
                MPI_Finalized(&finalized);
                if (finalized)
                    return;
                for (auto * comm : {&row.comm, &column.comm})
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
            // The grid also receives the blocks of the transposes (3D: the y-pencils [Local_nx][Local_nkz][Nmesh])
            const ptrdiff_t ncomplex_real = (Local_nx + n_extra_left + n_extra_right) * slice_stride / 2;
            const ptrdiff_t ncomplex_ypencils = N == 3 ? Local_nx * Local_nkz * Nmesh : 0;
            NComplexAlloc = std::max({ncomplex_real, ncomplex_ypencils, get_ntot_fourier()});

            allocate();
            fill_real_grid(0.0);
        }

        template <int N>
        size_t PencilGrid<N>::storage_bytes(int Nmesh, const PencilDomain & domain) {
            const auto P = domain.process_grid();
            const auto p = domain.coords_of(FML::ThisTask);
            const auto xcells = cell_edges(domain.x_edges(), Nmesh, "x");
            const ptrdiff_t nx = xcells[p[0] + 1] - xcells[p[0]];
            const int nlast = Nmesh / 2 + 1;
            ptrdiff_t ny = Nmesh, nky = 0, nkz = 1;
            if constexpr (N == 3) {
                const auto ycells = cell_edges(domain.y_edges(), Nmesh, "y");
                ny = ycells[p[1] + 1] - ycells[p[1]];
                nky = split(Nmesh, P[0], p[0])[1];
                nkz = split(nlast, P[1], p[1])[1];
            } else {
                nky = split(nlast, P[0], p[0])[1];
            }
            // As the constructor: the real rows of 2 (Nmesh/2+1) reals, the y-pencils (3D) and the Fourier cells
            const ptrdiff_t ncomplex_real = nx * (N == 3 ? ny : 1) * nlast;
            const ptrdiff_t ncomplex_ypencils = N == 3 ? nx * nkz * Nmesh : 0;
            return size_t(std::max({ncomplex_real, ncomplex_ypencils, nky * nkz * Nmesh})) * sizeof(ComplexType);
        }

        template <int N>
        void PencilGrid<N>::allocate() {
            data = allocate_complex(size_t(NComplexAlloc));

            const auto P = domain.process_grid();
            const auto p = domain.coords_of(FML::ThisTask);
            const int nx = int(Local_nx), ny = int(Local_ny), nky = int(Local_nky), nkz = int(Local_nkz);
            const int nl = n_extra_left;
            const int nlast = Nmesh / 2 + 1;
            const auto xcells = cell_edges(domain.x_edges(), Nmesh, "x");

            transforms = std::make_shared<PencilTransforms>();
            auto & t = *transforms;
            MPI_Comm_split(FML::communicator, p[0], p[1], &t.row.comm);
            MPI_Comm_split(FML::communicator, p[1], p[0], &t.column.comm);
            t.row.size = P[1];
            t.column.size = P[0];

            // The layouts (complex numbers). The real grid after the r2c: (ix, iy, kz) at
            // (ix + nl) * slice + (iy + nl) * row + kz (2D: (ix, ky) at (ix + nl) * row + ky)
            const ptrdiff_t row = row_stride / 2, slice = slice_stride / 2;
            if constexpr (N == 3) {
                const auto ycells = cell_edges(domain.y_edges(), Nmesh, "y");
                // Row: (ix, iy, kz) of the real pencils -> the y-pencils (ix, kz, y) at (ix * nkz + kz) * Nmesh + y
                for (int j = 0; j < P[1]; j++) {
                    const auto kz = split(nlast, P[1], j);
                    const int ny_j = int(ycells[j + 1] - ycells[j]);
                    t.row.send.push_back({nl * slice + nl * row + kz[0], {slice, row, 1}, {nx, ny, int(kz[1])}});
                    t.row.recv.push_back({ptrdiff_t(ycells[j]), {ptrdiff_t(nkz) * Nmesh, 1, Nmesh}, {nx, ny_j, nkz}});
                }
                // Column: (ix, kz, ky) of the y-pencils -> the x-pencils (ky, kz, x) at (ky * nkz + kz) * Nmesh + x
                for (int i = 0; i < P[0]; i++) {
                    const auto ky = split(Nmesh, P[0], i);
                    const int nx_i = int(xcells[i + 1] - xcells[i]);
                    t.column.send.push_back({ky[0], {ptrdiff_t(nkz) * Nmesh, Nmesh, 1}, {nx, nkz, int(ky[1])}});
                    t.column.recv.push_back(
                        {ptrdiff_t(xcells[i]), {1, Nmesh, ptrdiff_t(nkz) * Nmesh}, {nx_i, nkz, nky}});
                }
                t.row.finalize();
                t.column.finalize();
                // The work memory holds the packed blocks and the y-pencils (the grid the received blocks)
                t.nwork = std::max({t.row.send_total, size_t(nx) * nkz * Nmesh, t.column.recv_total});
            } else {
                // Column: (1, ix, ky) of the real slabs -> the x-pencils (ky, x) at ky * Nmesh + x
                for (int i = 0; i < P[0]; i++) {
                    const auto ky = split(nlast, P[0], i);
                    const int nx_i = int(xcells[i + 1] - xcells[i]);
                    t.column.send.push_back({nl * row + ky[0], {0, row, 1}, {1, nx, int(ky[1])}});
                    t.column.recv.push_back({ptrdiff_t(xcells[i]), {0, 1, Nmesh}, {1, nx_i, nky}});
                }
                t.column.finalize();
                // The work memory holds the packed blocks to send and the received ones
                t.nwork = t.column.send_total + t.column.recv_total;
            }
            t.work = allocate_complex(t.nwork);

            // The plans are made with FML's planner flag (FML::GRID::get_fftw_planner_flag(), as FFTWGrid;
            // FFTW_ESTIMATE by default) on FFTW-aligned memory, before the grid is filled (the other flags overwrite
            // the arrays), and executed on the memory of any grid with this layout (the same alignment)
            const unsigned flags = FML::GRID::get_fftw_planner_flag();
            FloatType * real = reinterpret_cast<FloatType *>(data) + real_offset;
            my_fftw_complex * complex = as_fftw(data) + real_offset / 2;
            fftw_iodim_type dim{Nmesh, 1, 1};
            if constexpr (N == 3) {
                fftw_iodim_type forward[2] = {{nx, int(slice_stride), int(slice_stride / 2)},
                                              {ny, int(row_stride), int(row_stride / 2)}};
                fftw_iodim_type backward[2] = {{nx, int(slice_stride / 2), int(slice_stride)},
                                               {ny, int(row_stride / 2), int(row_stride)}};
                t.r2c = plan_r2c(&dim, 2, forward, real, complex, flags);
                t.c2r = plan_c2r(&dim, 2, backward, complex, real, flags);
                t.y_forward = plan_many(Nmesh, nx * nkz, as_fftw(t.work), FFTW_FORWARD, flags);
                t.y_backward = plan_many(Nmesh, nx * nkz, as_fftw(t.work), FFTW_BACKWARD, flags);
            } else {
                fftw_iodim_type forward[1] = {{nx, int(row_stride), int(row_stride / 2)}};
                fftw_iodim_type backward[1] = {{nx, int(row_stride / 2), int(row_stride)}};
                t.r2c = plan_r2c(&dim, 1, forward, real, complex, flags);
                t.c2r = plan_c2r(&dim, 1, backward, complex, real, flags);
            }
            t.x_forward = plan_many(Nmesh, nky * nkz, as_fftw(data), FFTW_FORWARD, flags);
            t.x_backward = plan_many(Nmesh, nky * nkz, as_fftw(data), FFTW_BACKWARD, flags);
            assert_mpi(t.r2c and t.c2r and t.x_forward and t.x_backward and (N == 2 or (t.y_forward and t.y_backward)),
                       "[PencilGrid] FFTW failed to make a plan");
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

        // The forward transform: r2c along the last axis, then (3D) the row transpose and the FFT along y, then the
        // column transpose and the FFT along x. The grid (D) and the work memory (W) take turns: each transpose packs
        // from one into the other, exchanges back, and unpacks into the other again
        template <int N>
        void PencilGrid<N>::fftw_r2c() {
            assert_mpi(data != nullptr, "[PencilGrid::fftw_r2c] The grid is not allocated");
            auto & t = *transforms;
            ComplexType * W = t.work;
            execute_r2c(t.r2c, get_real_grid(), as_fftw(data) + real_offset / 2);
            if constexpr (N == 3) {
                run_exchange(t.row, false, data, W, W, data);
                execute_dft(t.y_forward, as_fftw(W));
                run_exchange(t.column, false, W, data, data, W);
            } else {
                run_exchange(t.column, false, data, data, W, W + t.column.send_total);
            }
            execute_dft(t.x_forward, as_fftw(data));
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

        // The backward transform: the same steps in reverse
        template <int N>
        void PencilGrid<N>::fftw_c2r() {
            assert_mpi(data != nullptr, "[PencilGrid::fftw_c2r] The grid is not allocated");
            auto & t = *transforms;
            ComplexType * W = t.work;
            execute_dft(t.x_backward, as_fftw(data));
            if constexpr (N == 3) {
                run_exchange(t.column, true, data, W, W, data);
                execute_dft(t.y_backward, as_fftw(W));
                run_exchange(t.row, true, W, data, data, W);
            } else {
                run_exchange(t.column, true, data, data, W, W + t.column.recv_total);
            }
            execute_c2r(t.c2r, as_fftw(data) + real_offset / 2, get_real_grid());
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

        namespace {
            constexpr char pencil_grid_magic[8] = {'F', 'M', 'L', 'P', 'G', 'R', 'D', '1'};
        }

        template <int N>
        void PencilGrid<N>::dump_to_file(std::string fileprefix) const {
            const std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
            std::ofstream out(filename, std::ios::binary | std::ios::trunc);
            bool ok = out.good();
            if (ok) {
                const int32_t header[6] = {N, int32_t(Nmesh), int32_t(n_extra_left), int32_t(n_extra_right),
                                           int32_t(FML::NTasks), int32_t(grid_is_in_real_space)};
                out.write(pencil_grid_magic, 8);
                out.write(reinterpret_cast<const char *>(header), sizeof(header));
                for (const auto * edges : {&domain.x_edges(), &domain.y_edges()}) {
                    const int64_t n = int64_t(edges->size());
                    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
                    out.write(reinterpret_cast<const char *>(edges->data()), std::streamsize(n * sizeof(double)));
                }
                const int64_t nalloc = NComplexAlloc;
                out.write(reinterpret_cast<const char *>(&nalloc), sizeof(nalloc));
                if (nalloc > 0)
                    out.write(reinterpret_cast<const char *>(data), std::streamsize(nalloc * sizeof(ComplexType)));
                out.close();
                ok = not out.fail();
            }
            int good = ok ? 1 : 0;
            FML::MinOverTasks(&good);
            assert_mpi(good == 1, ("[PencilGrid::dump_to_file] Failed to write " + fileprefix + ".*").c_str());
        }

        template <int N>
        void PencilGrid<N>::load_from_file(std::string fileprefix) {
            const std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
            std::ifstream in(filename, std::ios::binary);
            char magic[8] = {};
            int32_t header[6] = {};
            std::vector<double> edges[2];
            int64_t nalloc = -1;
            in.read(magic, 8);
            in.read(reinterpret_cast<char *>(header), sizeof(header));
            bool ok = in.good() and std::memcmp(magic, pencil_grid_magic, 8) == 0 and header[0] == N and
                      header[4] == FML::NTasks and header[1] > 0;
            for (int k = 0; ok and k < 2; k++) {
                int64_t n = 0;
                in.read(reinterpret_cast<char *>(&n), sizeof(n));
                ok = in.good() and n >= 2 and n <= int64_t(FML::NTasks) + 1;
                if (ok) {
                    edges[k].resize(size_t(n));
                    in.read(reinterpret_cast<char *>(edges[k].data()), std::streamsize(n * sizeof(double)));
                    ok = in.good();
                }
            }
            if (ok)
                in.read(reinterpret_cast<char *>(&nalloc), sizeof(nalloc));
            ok = ok and in.good();
            int good = ok ? 1 : 0;
            FML::MinOverTasks(&good);
            assert_mpi(good == 1,
                       ("[PencilGrid::load_from_file] Cannot read " + fileprefix +
                        ".* (missing, not a PencilGrid file, other dimension or other number of tasks)")
                           .c_str());
            // All tasks have the same header (the same grid): make it and read our part
            *this = PencilGrid<N>(header[1], header[2], header[3], PencilDomain::from_edges(edges[0], edges[1]));
            set_grid_status_real(header[5] != 0);
            ok = nalloc == int64_t(NComplexAlloc);
            if (ok and nalloc > 0) {
                in.read(reinterpret_cast<char *>(data), std::streamsize(nalloc * sizeof(ComplexType)));
                ok = in.good();
            }
            // Nothing may be left: a longer file was written with another precision (e.g. double vs float)
            ok = ok and in.peek() == std::ifstream::traits_type::eof();
            good = ok ? 1 : 0;
            FML::MinOverTasks(&good);
            assert_mpi(good == 1,
                       ("[PencilGrid::load_from_file] The data in " + fileprefix +
                        ".* does not match the grid (truncated, or written with another layout or precision)")
                           .c_str());
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

        //=========================================================================
        // CPencilGrid: the complex-to-complex pencil grid (the same transposes, no r2c / c2r)
        //=========================================================================

        struct CPencilTransforms {
            // 3D: position-space pencils (after the FFT along z) -> y-pencils in the row, y-pencils -> x-pencils in
            // the column. 2D: slabs (after the FFT along y) -> x-pencils in the column
            Exchange row, column;
            my_fftw_plan last_forward{nullptr}, last_backward{nullptr};
            my_fftw_plan y_forward{nullptr}, y_backward{nullptr};
            my_fftw_plan x_forward{nullptr}, x_backward{nullptr};
            ComplexType * work{nullptr};
            size_t nwork{0};

            CPencilTransforms() = default;
            CPencilTransforms(const CPencilTransforms &) = delete;
            CPencilTransforms & operator=(const CPencilTransforms &) = delete;
            ~CPencilTransforms() {
                for (auto p : {last_forward, last_backward, y_forward, y_backward, x_forward, x_backward})
                    if (p != nullptr)
                        destroy_plan(p);
                free_complex_array(work);
                int finalized = 0;
                MPI_Finalized(&finalized);
                if (finalized)
                    return;
                for (auto * comm : {&row.comm, &column.comm})
                    if (*comm != MPI_COMM_NULL)
                        MPI_Comm_free(comm);
            }
        };

        template <int N>
        CPencilGrid<N>::CPencilGrid(int Nmesh, int n_extra_left, int n_extra_right)
            : CPencilGrid(Nmesh, n_extra_left, n_extra_right, default_domain(Nmesh, N)) {}

        template <int N>
        CPencilGrid<N>::CPencilGrid(int Nmesh, int n_extra_left, int n_extra_right, const PencilDomain & domain)
            : Nmesh(Nmesh), n_extra_left(n_extra_left), n_extra_right(n_extra_right), domain(domain) {
            assert_mpi(Nmesh > 0 and n_extra_left >= 0 and n_extra_right >= 0,
                       "[CPencilGrid] Need Nmesh > 0 and n_extra >= 0");
            assert_mpi(domain.ntasks() == FML::NTasks, "[CPencilGrid] The domain is not made for NTasks tasks");
            const auto P = domain.process_grid();
            const auto p = domain.coords_of(FML::ThisTask);
            assert_mpi(N == 3 or P[1] == 1, "[CPencilGrid] In 2D the domain must have P1 = 1 (slabs in x)");

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
                       "[CPencilGrid] Every task needs at least one cell (and as many as the extra cells) in x and y");

            // The Fourier cells: all kx, ky split over P0 and (3D) kz split over P1, both over [0, Nmesh)
            assert_mpi(Nmesh >= P[0] and Nmesh >= P[1], "[CPencilGrid] Need Nmesh >= P0 and Nmesh >= P1");
            const auto ky = split(Nmesh, P[0], p[0]);
            Local_ky_start = ky[0];
            Local_nky = ky[1];
            if constexpr (N == 3) {
                const auto kz = split(Nmesh, P[1], p[1]);
                Local_kz_start = kz[0];
                Local_nkz = kz[1];
            } else {
                Local_kz_start = 0;
                Local_nkz = 1;
            }

            row_stride = Nmesh;
            if constexpr (N == 3) {
                slice_stride = (Local_ny + n_extra_left + n_extra_right) * row_stride;
                real_offset = n_extra_left * slice_stride + n_extra_left * row_stride;
            } else {
                slice_stride = row_stride;
                real_offset = n_extra_left * slice_stride;
            }
            const ptrdiff_t ncomplex_real = (Local_nx + n_extra_left + n_extra_right) * slice_stride;
            const ptrdiff_t ncomplex_ypencils = N == 3 ? Local_nx * Local_nkz * Nmesh : 0;
            NComplexAlloc = std::max({ncomplex_real, ncomplex_ypencils, get_ntot_fourier()});

            allocate();
            fill_real_grid(ComplexType(0.0));
        }

        template <int N>
        void CPencilGrid<N>::allocate() {
            data = allocate_complex(size_t(NComplexAlloc));

            const auto P = domain.process_grid();
            const auto p = domain.coords_of(FML::ThisTask);
            const int nx = int(Local_nx), ny = int(Local_ny), nky = int(Local_nky), nkz = int(Local_nkz);
            const int nl = n_extra_left;
            const auto xcells = cell_edges(domain.x_edges(), Nmesh, "x");

            transforms = std::make_shared<CPencilTransforms>();
            auto & t = *transforms;
            MPI_Comm_split(FML::communicator, p[0], p[1], &t.row.comm);
            MPI_Comm_split(FML::communicator, p[1], p[0], &t.column.comm);
            t.row.size = P[1];
            t.column.size = P[0];

            const ptrdiff_t row = row_stride, slice = slice_stride;
            if constexpr (N == 3) {
                const auto ycells = cell_edges(domain.y_edges(), Nmesh, "y");
                // Row: (ix, iy, kz) of the pencils -> the y-pencils (ix, kz, y) at (ix * nkz + kz) * Nmesh + y
                for (int j = 0; j < P[1]; j++) {
                    const auto kz = split(Nmesh, P[1], j);
                    const int ny_j = int(ycells[j + 1] - ycells[j]);
                    t.row.send.push_back({nl * slice + nl * row + kz[0], {slice, row, 1}, {nx, ny, int(kz[1])}});
                    t.row.recv.push_back({ptrdiff_t(ycells[j]), {ptrdiff_t(nkz) * Nmesh, 1, Nmesh}, {nx, ny_j, nkz}});
                }
                // Column: (ix, kz, ky) of the y-pencils -> the x-pencils (ky, kz, x) at (ky * nkz + kz) * Nmesh + x
                for (int i = 0; i < P[0]; i++) {
                    const auto ky = split(Nmesh, P[0], i);
                    const int nx_i = int(xcells[i + 1] - xcells[i]);
                    t.column.send.push_back({ky[0], {ptrdiff_t(nkz) * Nmesh, Nmesh, 1}, {nx, nkz, int(ky[1])}});
                    t.column.recv.push_back(
                        {ptrdiff_t(xcells[i]), {1, Nmesh, ptrdiff_t(nkz) * Nmesh}, {nx_i, nkz, nky}});
                }
                t.row.finalize();
                t.column.finalize();
                t.nwork = std::max({t.row.send_total, size_t(nx) * nkz * Nmesh, t.column.recv_total});
            } else {
                // Column: (1, ix, ky) of the slabs -> the x-pencils (ky, x) at ky * Nmesh + x
                for (int i = 0; i < P[0]; i++) {
                    const auto ky = split(Nmesh, P[0], i);
                    const int nx_i = int(xcells[i + 1] - xcells[i]);
                    t.column.send.push_back({nl * row + ky[0], {0, row, 1}, {1, nx, int(ky[1])}});
                    t.column.recv.push_back({ptrdiff_t(xcells[i]), {0, 1, Nmesh}, {1, nx_i, nky}});
                }
                t.column.finalize();
                t.nwork = t.column.send_total + t.column.recv_total;
            }
            t.work = allocate_complex(t.nwork);

            // The plans (made before the grid is filled: the planner flags other than FFTW_ESTIMATE overwrite it)
            const unsigned flags = FML::GRID::get_fftw_planner_flag();
            my_fftw_complex * main = as_fftw(data) + real_offset;
            fftw_iodim_type dim{Nmesh, 1, 1};
            if constexpr (N == 3) {
                fftw_iodim_type howmany[2] = {{nx, int(slice), int(slice)}, {ny, int(row), int(row)}};
                t.last_forward = plan_dft(&dim, 2, howmany, main, FFTW_FORWARD, flags);
                t.last_backward = plan_dft(&dim, 2, howmany, main, FFTW_BACKWARD, flags);
                t.y_forward = plan_many(Nmesh, nx * nkz, as_fftw(t.work), FFTW_FORWARD, flags);
                t.y_backward = plan_many(Nmesh, nx * nkz, as_fftw(t.work), FFTW_BACKWARD, flags);
            } else {
                fftw_iodim_type howmany[1] = {{nx, int(slice), int(slice)}};
                t.last_forward = plan_dft(&dim, 1, howmany, main, FFTW_FORWARD, flags);
                t.last_backward = plan_dft(&dim, 1, howmany, main, FFTW_BACKWARD, flags);
            }
            t.x_forward = plan_many(Nmesh, nky * nkz, as_fftw(data), FFTW_FORWARD, flags);
            t.x_backward = plan_many(Nmesh, nky * nkz, as_fftw(data), FFTW_BACKWARD, flags);
            assert_mpi(t.last_forward and t.last_backward and t.x_forward and t.x_backward and
                           (N == 2 or (t.y_forward and t.y_backward)),
                       "[CPencilGrid] FFTW failed to make a plan");
        }

        template <int N>
        CPencilGrid<N>::CPencilGrid(const CPencilGrid & rhs)
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
        CPencilGrid<N> & CPencilGrid<N>::operator=(const CPencilGrid & rhs) {
            if (this == &rhs)
                return *this;
            // The same layout: copy the values into the memory we have
            if (data != nullptr and rhs.data != nullptr and transforms == rhs.transforms and
                NComplexAlloc == rhs.NComplexAlloc) {
                std::memcpy(data, rhs.data, size_t(NComplexAlloc) * sizeof(ComplexType));
                grid_is_in_real_space = rhs.grid_is_in_real_space;
                name = rhs.name;
                return *this;
            }
            *this = CPencilGrid(rhs);
            return *this;
        }

        template <int N>
        CPencilGrid<N>::CPencilGrid(CPencilGrid && rhs) noexcept {
            *this = std::move(rhs);
        }

        template <int N>
        CPencilGrid<N> & CPencilGrid<N>::operator=(CPencilGrid && rhs) noexcept {
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
        CPencilGrid<N>::~CPencilGrid() {
            free_complex_array(data);
        }

        template <int N>
        void CPencilGrid<N>::free() {
            *this = CPencilGrid();
        }

        // Forward: the FFT along the last axis on the pencils, then (3D) the row transpose and the FFT along y, then
        // the column transpose and the FFT along x
        template <int N>
        void CPencilGrid<N>::fftw_forward() {
            assert_mpi(data != nullptr, "[CPencilGrid::fftw_forward] The grid is not allocated");
            auto & t = *transforms;
            ComplexType * W = t.work;
            execute_dft(t.last_forward, as_fftw(data) + real_offset);
            if constexpr (N == 3) {
                run_exchange(t.row, false, data, W, W, data);
                execute_dft(t.y_forward, as_fftw(W));
                run_exchange(t.column, false, W, data, data, W);
            } else {
                run_exchange(t.column, false, data, data, W, W + t.column.send_total);
            }
            execute_dft(t.x_forward, as_fftw(data));
            FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::ForwardFFTs);
            grid_is_in_real_space = false;

            const FloatType norm = FloatType(1.0 / std::pow(double(Nmesh), N));
            const ptrdiff_t ntot = get_ntot_fourier();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (ptrdiff_t i = 0; i < ntot; i++)
                data[i] *= norm;
        }

        template <int N>
        void CPencilGrid<N>::fftw_backward() {
            assert_mpi(data != nullptr, "[CPencilGrid::fftw_backward] The grid is not allocated");
            auto & t = *transforms;
            ComplexType * W = t.work;
            execute_dft(t.x_backward, as_fftw(data));
            if constexpr (N == 3) {
                run_exchange(t.column, true, data, W, W, data);
                execute_dft(t.y_backward, as_fftw(W));
                run_exchange(t.row, true, W, data, data, W);
            } else {
                run_exchange(t.column, true, data, data, W, W + t.column.recv_total);
            }
            execute_dft(t.last_backward, as_fftw(data) + real_offset);
            FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::InverseFFTs);
            grid_is_in_real_space = true;
        }

        template <int N>
        void CPencilGrid<N>::communicate_boundaries() {
            if (n_extra_left + n_extra_right == 0)
                return;
            assert_mpi(data != nullptr, "[CPencilGrid::communicate_boundaries] The grid is not allocated");
            if constexpr (N == 3)
                exchange_y(false);
            exchange_x(false);
        }

        template <int N>
        void CPencilGrid<N>::add_contribution_from_extra_cells() {
            if (n_extra_left + n_extra_right == 0)
                return;
            assert_mpi(data != nullptr, "[CPencilGrid::add_contribution_from_extra_cells] The grid is not allocated");
            exchange_x(true);
            if constexpr (N == 3)
                exchange_y(true);
        }

        template <int N>
        void CPencilGrid<N>::exchange_x(bool add) {
            const auto p = domain.coords_of(FML::ThisTask);
            const int left = domain.rank_of(p[0] - 1, p[1]);
            const int right = domain.rank_of(p[0] + 1, p[1]);
            ComplexType * grid = get_real_grid();
            auto slices = [&](ptrdiff_t ix) {
                return grid + ix * slice_stride - (N == 3 ? n_extra_left * row_stride : 0);
            };
            const size_t bytes_per_slice = size_t(slice_stride) * sizeof(ComplexType);
            const ptrdiff_t nl = n_extra_left, nr = n_extra_right;
            if (not add) {
                if (nr > 0)
                    sendrecv_bytes(slices(0), slices(Local_nx), nr * bytes_per_slice, left, right, 50);
                if (nl > 0)
                    sendrecv_bytes(slices(Local_nx - nl), slices(-nl), nl * bytes_per_slice, right, left, 51);
                return;
            }
            std::vector<ComplexType> buffer;
            auto add_to = [&](ComplexType * target, size_t n) {
                for (size_t i = 0; i < n; i++)
                    target[i] += buffer[i];
            };
            if (nr > 0) {
                const size_t n = size_t(nr * slice_stride);
                buffer.resize(n);
                sendrecv_bytes(slices(Local_nx), buffer.data(), n * sizeof(ComplexType), right, left, 52);
                add_to(slices(0), n);
            }
            if (nl > 0) {
                const size_t n = size_t(nl * slice_stride);
                buffer.resize(n);
                sendrecv_bytes(slices(-nl), buffer.data(), n * sizeof(ComplexType), left, right, 53);
                add_to(slices(Local_nx - nl), n);
            }
            std::fill(slices(-nl), slices(0), ComplexType(0));
            std::fill(slices(Local_nx), slices(Local_nx + nr), ComplexType(0));
        }

        template <int N>
        void CPencilGrid<N>::exchange_y(bool add) {
            if constexpr (N == 3) {
                const auto p = domain.coords_of(FML::ThisTask);
                const int down = domain.rank_of(p[0], p[1] - 1);
                const int up = domain.rank_of(p[0], p[1] + 1);
                ComplexType * grid = get_real_grid();
                const ptrdiff_t nx = Local_nx, ny = Local_ny;
                auto pack = [&](ptrdiff_t iy0, ptrdiff_t n, std::vector<ComplexType> & buf) {
                    const ptrdiff_t block = n * row_stride;
                    buf.resize(size_t(nx * block));
                    for (ptrdiff_t ix = 0; ix < nx; ix++)
                        std::copy(grid + ix * slice_stride + iy0 * row_stride,
                                  grid + ix * slice_stride + iy0 * row_stride + block,
                                  buf.begin() + ix * block);
                };
                auto unpack = [&](ptrdiff_t iy0, ptrdiff_t n, const std::vector<ComplexType> & buf, bool add_values) {
                    const ptrdiff_t block = n * row_stride;
                    for (ptrdiff_t ix = 0; ix < nx; ix++) {
                        ComplexType * target = grid + ix * slice_stride + iy0 * row_stride;
                        for (ptrdiff_t i = 0; i < block; i++)
                            target[i] = add_values ? target[i] + buf[ix * block + i] : buf[ix * block + i];
                    }
                };
                auto exchange =
                    [&](ptrdiff_t send_iy0, ptrdiff_t recv_iy0, ptrdiff_t n, int dest, int source, int tag) {
                        if (n == 0)
                            return;
                        std::vector<ComplexType> send, recv(size_t(nx * n * row_stride));
                        pack(send_iy0, n, send);
                        sendrecv_bytes(send.data(), recv.data(), send.size() * sizeof(ComplexType), dest, source, tag);
                        unpack(recv_iy0, n, recv, add);
                    };
                const ptrdiff_t nl = n_extra_left, nr = n_extra_right;
                if (not add) {
                    exchange(0, ny, nr, down, up, 60);
                    exchange(ny - nl, -nl, nl, up, down, 61);
                } else {
                    exchange(ny, 0, nr, up, down, 62);
                    exchange(-nl, ny - nl, nl, down, up, 63);
                    std::vector<ComplexType> zeros;
                    for (auto [iy0, n] : {std::array<ptrdiff_t, 2>{-nl, nl}, std::array<ptrdiff_t, 2>{ny, nr}}) {
                        zeros.assign(size_t(nx * n * row_stride), ComplexType(0));
                        unpack(iy0, n, zeros, false);
                    }
                }
            } else {
                (void)add;
            }
        }

        template <int N>
        void CPencilGrid<N>::fill_real_grid(const ComplexType val) {
            std::fill(data, data + NComplexAlloc, val);
            grid_is_in_real_space = true;
        }

        template <int N>
        void CPencilGrid<N>::fill_fourier_grid(const ComplexType val) {
            std::fill(data, data + NComplexAlloc, val);
            grid_is_in_real_space = false;
        }

        template <int N>
        void CPencilGrid<N>::fill_real_grid(const std::function<ComplexType(std::array<double, N> &)> & func) {
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
        void CPencilGrid<N>::fill_fourier_grid(const std::function<ComplexType(std::array<double, N> &)> & func) {
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
        bool CPencilGrid<N>::nan_in_grids() const {
            auto bad = [](ComplexType z) { return not std::isfinite(z.real()) or not std::isfinite(z.imag()); };
            if (grid_is_in_real_space) {
                for (auto && index : get_real_range())
                    if (bad(get_real_from_index(index)))
                        return true;
                return false;
            }
            for (auto && index : get_fourier_range())
                if (bad(get_fourier_from_index(index)))
                    return true;
            return false;
        }

        namespace {
            constexpr char cpencil_grid_magic[8] = {'F', 'M', 'L', 'C', 'P', 'G', 'R', '1'};
        }

        template <int N>
        void CPencilGrid<N>::dump_to_file(std::string fileprefix) const {
            const std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
            std::ofstream out(filename, std::ios::binary | std::ios::trunc);
            bool ok = out.good();
            if (ok) {
                const int32_t header[6] = {N, int32_t(Nmesh), int32_t(n_extra_left), int32_t(n_extra_right),
                                           int32_t(FML::NTasks), int32_t(grid_is_in_real_space)};
                out.write(cpencil_grid_magic, 8);
                out.write(reinterpret_cast<const char *>(header), sizeof(header));
                for (const auto * edges : {&domain.x_edges(), &domain.y_edges()}) {
                    const int64_t n = int64_t(edges->size());
                    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
                    out.write(reinterpret_cast<const char *>(edges->data()), std::streamsize(n * sizeof(double)));
                }
                const int64_t nalloc = NComplexAlloc;
                out.write(reinterpret_cast<const char *>(&nalloc), sizeof(nalloc));
                if (nalloc > 0)
                    out.write(reinterpret_cast<const char *>(data), std::streamsize(nalloc * sizeof(ComplexType)));
                out.close();
                ok = not out.fail();
            }
            int good = ok ? 1 : 0;
            FML::MinOverTasks(&good);
            assert_mpi(good == 1, ("[CPencilGrid::dump_to_file] Failed to write " + fileprefix + ".*").c_str());
        }

        template <int N>
        void CPencilGrid<N>::load_from_file(std::string fileprefix) {
            const std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
            std::ifstream in(filename, std::ios::binary);
            char magic[8] = {};
            int32_t header[6] = {};
            std::vector<double> edges[2];
            int64_t nalloc = -1;
            in.read(magic, 8);
            in.read(reinterpret_cast<char *>(header), sizeof(header));
            bool ok = in.good() and std::memcmp(magic, cpencil_grid_magic, 8) == 0 and header[0] == N and
                      header[4] == FML::NTasks and header[1] > 0;
            for (int k = 0; ok and k < 2; k++) {
                int64_t n = 0;
                in.read(reinterpret_cast<char *>(&n), sizeof(n));
                ok = in.good() and n >= 2 and n <= int64_t(FML::NTasks) + 1;
                if (ok) {
                    edges[k].resize(size_t(n));
                    in.read(reinterpret_cast<char *>(edges[k].data()), std::streamsize(n * sizeof(double)));
                    ok = in.good();
                }
            }
            if (ok)
                in.read(reinterpret_cast<char *>(&nalloc), sizeof(nalloc));
            ok = ok and in.good();
            int good = ok ? 1 : 0;
            FML::MinOverTasks(&good);
            assert_mpi(good == 1, ("[CPencilGrid::load_from_file] Cannot read " + fileprefix + ".*").c_str());
            *this = CPencilGrid<N>(header[1], header[2], header[3], PencilDomain::from_edges(edges[0], edges[1]));
            set_grid_status_real(header[5] != 0);
            ok = nalloc == int64_t(NComplexAlloc);
            if (ok and nalloc > 0) {
                in.read(reinterpret_cast<char *>(data), std::streamsize(nalloc * sizeof(ComplexType)));
                ok = in.good();
            }
            // Nothing may be left: a longer file was written with another precision (e.g. double vs float)
            ok = ok and in.peek() == std::ifstream::traits_type::eof();
            good = ok ? 1 : 0;
            FML::MinOverTasks(&good);
            assert_mpi(good == 1, ("[CPencilGrid::load_from_file] The data in " + fileprefix +
                                   ".* does not match the grid (truncated, or written with another layout or precision)")
                                      .c_str());
        }

        template <int N>
        void CPencilGrid<N>::info() const {
            if (FML::ThisTask != 0)
                return;
            const auto P = domain.process_grid();
            std::cout << "\n#=====================================================\n";
            std::cout << "# Info about CPencilGrid. Grid label: [" << name << "]\n";
            std::cout << "# Status of grid: " << (grid_is_in_real_space ? "[Realspace]" : "[Fourierspace]")
                      << " NDIM: [" << N << "]\n";
            std::cout << "# Nmesh                  " << Nmesh << "\n";
            std::cout << "# Process grid           " << P[0] << " x " << P[1] << "\n";
            std::cout << "# Local x, y (task 0)    " << Local_nx << " x " << Local_ny << " (+ " << n_extra_left
                      << " left, " << n_extra_right << " right)\n";
            std::cout << "# Local ky, kz (task 0)  " << Local_nky << " x " << Local_nkz << "\n";
            std::cout << "# Allocated (task 0)     " << double(allocated_storage_bytes()) / 1e6 << " MB\n";
            std::cout << "#=====================================================\n\n";
        }

        template class CPencilGrid<2>;
        template class CPencilGrid<3>;


    } // namespace GRID
} // namespace FML
