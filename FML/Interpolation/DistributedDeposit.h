#ifndef FML_INTERPOLATION_DISTRIBUTEDDEPOSIT_H
#define FML_INTERPOLATION_DISTRIBUTEDDEPOSIT_H

// =====================================================================================================================
// Density assignment when the particles and the grid have different domain decompositions, e.g. particles on a
// Peano-Hilbert domain and the grid in slabs or pencils (or particles anywhere: no assumption is made about which task
// holds which particle). This is how GADGET does its PM step (Springel 2005, pm_periodic.c; GADGET-4's PM_ZOOM
// variant does the same with cell lists):
//
//   1. Each task deposits its own particles into a local dense patch of the mesh: the periodic bounding box of the
//      cells its particles touch (for a compact domain such as a Peano-Hilbert one, about the size of a slab).
//   2. The part of the patch that each task's box of the grid overlaps is sent to that task (all-zero blocks are
//      skipped), and the receiver adds it to its grid.
//
// The patch has at most DepositOptions::max_patch_cells cells: a larger bounding box (a domain that is not compact) is
// processed in several passes over slices of it, so the memory is bounded whatever the particle distribution.
//
// The grid is described by a MeshPartition: the mesh size along each axis and the box of global cells each task owns
// (the boxes tile the mesh). mesh_partition_from_grid gives it for any grid with the interface of
// FFTWGrid/GridInterface.h (FFTWGrid slabs, PencilGrid pencils: the box of a task is its get_local_range along each
// axis). deposit_to_partitioned_mesh works for any box decomposition, with a callback that adds a value to a local cell.
//
// This is the density assignment for particles on a PeanoHilbertDomain (Global/Domain.h), which the grid code cannot
// use directly (ParticleGridInterpolation(Generic).h need the particles in the domain of the grid), without moving
// the particles: particles_to_grid_any_domain(MPIParticles, grid, "CIC").
//
// The assignment kernels are those of ParticleGridInterpolation.h (B-splines of order 1-5: NGP, CIC, TSC, PCS, PQS,
// the same cells and weights), and particles_to_grid_any_domain gives the same overdensity field as particles_to_grid
// (mass or delta-f weighted, normalized to the mean) up to the order of the floating-point sums, but without
// requiring the particles to be in the domain of the grid. It needs no extra slices; the extra slices of the grid are
// set to the background value. assignment_offset (in box units, as in particles_to_grid; used for interlacing) wraps
// periodically along every axis.
//
// Results are deterministic: the same for any number of OpenMP threads > 1 (the patch is filled in two phases of
// non-overlapping x-chunks, as in particles_to_grid), and identical to one thread up to the order of the sums.
//
// Communication: one count exchange and one chunked pairwise exchange per pass (FML::COMMUNICATION::
// exchange_records_chunked, at most max_message_bytes per message). All tasks must call the functions (collective).
// =====================================================================================================================

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/Global/ChunkedExchange.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif

namespace FML::INTERPOLATION {

    /// A box of global mesh cells: [start[d], start[d] + count[d]) along each axis (no periodic wrap)
    template <int N>
    struct MeshBox {
        std::array<int, N> start{};
        std::array<int, N> count{};
        size_t volume() const {
            size_t v = 1;
            for (int d = 0; d < N; d++)
                v *= size_t(std::max(count[d], 0));
            return v;
        }
    };

    /// How a mesh is distributed over the tasks: the mesh size along each axis and the box of global cells of each
    /// task (box[rank]; empty boxes are allowed). The boxes must tile the mesh
    template <int N>
    struct MeshPartition {
        std::array<int, N> mesh{};
        std::vector<MeshBox<N>> box;

        /// Throws unless there is one box per task, inside the mesh, with the total volume of the mesh
        void validate() const {
            assert_mpi(box.size() == size_t(FML::NTasks), "[MeshPartition] Need one box per task");
            size_t total = 1, sum = 0;
            for (int d = 0; d < N; d++) {
                assert_mpi(mesh[d] > 0, "[MeshPartition] The mesh size must be positive");
                total *= size_t(mesh[d]);
            }
            for (const auto & b : box) {
                for (int d = 0; d < N; d++)
                    assert_mpi(b.count[d] >= 0 and b.start[d] >= 0 and b.start[d] + b.count[d] <= mesh[d],
                               "[MeshPartition] A box is outside the mesh");
                sum += b.volume();
            }
            assert_mpi(sum == total, "[MeshPartition] The boxes do not tile the mesh");
        }
    };

    /// The partition of a distributed grid (FFTWGrid, PencilGrid, ...: anything with get_nmesh and
    /// get_local_range(axis) = {first global cell, count}): task r owns the product of its local ranges. Collective
    template <int N, template <int> class Grid>
    MeshPartition<N> mesh_partition_from_grid(const Grid<N> & grid) {
        MeshPartition<N> p;
        p.mesh.fill(grid.get_nmesh());
        std::array<long long, 2 * N> mine;
        for (int d = 0; d < N; d++) {
            const auto range = grid.get_local_range(d);
            mine[2 * d] = (long long)range[0];
            mine[2 * d + 1] = (long long)range[1];
        }
        std::vector<long long> all(2 * N * size_t(FML::NTasks));
#ifdef USE_MPI
        MPI_Allgather(mine.data(), 2 * N, MPI_LONG_LONG, all.data(), 2 * N, MPI_LONG_LONG, FML::communicator);
#else
        std::copy(mine.begin(), mine.end(), all.begin());
#endif
        p.box.resize(FML::NTasks);
        for (int r = 0; r < FML::NTasks; r++)
            for (int d = 0; d < N; d++) {
                p.box[r].start[d] = int(all[2 * N * size_t(r) + 2 * d]);
                p.box[r].count[d] = int(all[2 * N * size_t(r) + 2 * d + 1]);
            }
        p.validate();
        return p;
    }

    /// Settings of the distributed deposit
    struct DepositOptions {
        /// The largest local patch (cells; 0: automatic, twice the cells of the mesh per task, at least 2^18). A larger
        /// bounding box of the local particles is processed in several passes
        size_t max_patch_cells = 0;
        /// The largest MPI message (bytes)
        size_t max_message_bytes = 8 * 1024 * 1024;
        /// Do not send blocks of the patch that are all zero (empty regions of the bounding box)
        bool skip_zero_blocks = true;
    };

    /// What a deposit did (for diagnostics and tests)
    struct DepositInfo {
        int passes = 0;                    ///< Passes over the patch on this task
        std::array<int, 3> patch_count{};  ///< The size of the bounding box of this task (first 3 axes)
        size_t values_sent = 0;            ///< Cell values sent to other tasks (and to itself)
        size_t blocks_sent = 0;            ///< Blocks sent
    };

    namespace detail {
        /// The weight of a particle (mass, delta-f weight or 1) as in particles_to_grid
        template <class T>
        double deposit_weight(const T & p) {
            if constexpr (FML::PARTICLE::has_get_deltaf_weight<T>())
                return FML::PARTICLE::GetDeltafWeight(const_cast<T &>(p));
            else if constexpr (FML::PARTICLE::has_get_mass<T>())
                return FML::PARTICLE::GetMass(const_cast<T &>(p));
            else
                return 1.0;
        }

        /// The stencil of a particle: the first (leftmost, unwrapped) cell and the ORDER weights along each axis, the
        /// same cells and weights as particles_to_grid
        template <int N, int ORDER>
        struct Stencil {
            std::array<int, N> first;
            std::array<std::array<double, ORDER>, N> w;
        };

        template <int N, int ORDER, class T>
        Stencil<N, ORDER> stencil_of(const T & particle,
                                     const std::array<int, N> & mesh,
                                     const std::array<double, N> & offset,
                                     bool shifted) {
            Stencil<N, ORDER> s;
            const auto * pos = FML::PARTICLE::GetPos(const_cast<T &>(particle));
            for (int d = 0; d < N; d++) {
                double x;
                if (shifted) {
                    // As particles_to_grid: the shifted coordinate in the particle's precision, wrapped periodically
                    auto c = pos[d];
                    c += offset[d];
                    c = FML::PARTICLE::wrap_coordinate_periodic(c);
                    x = c * mesh[d];
                } else {
                    x = pos[d] * mesh[d];
                }
                int ix = int(x);
                x -= ix;
                // Offset of the first cell relative to the particle's cell
                int xstart;
                if constexpr (ORDER % 2 == 0) {
                    xstart = -ORDER / 2 + 1;
#ifdef CELLCENTERSHIFTED
                    xstart = -ORDER / 2;
                    if (x > 0.5)
                        xstart += 1;
#endif
                } else {
#ifndef CELLCENTERSHIFTED
                    xstart = -ORDER / 2;
                    if (x > 0.5)
                        xstart += 1;
#else
                    xstart = -ORDER / 2;
#endif
                }
                for (int j = 0; j < ORDER; j++) {
                    const int go = xstart + j;
                    if constexpr (ORDER == 1) {
                        s.w[d][j] = 1.0;
                    } else {
#ifdef CELLCENTERSHIFTED
                        s.w[d][j] = kernel<ORDER>(std::fabs(-x + go + 0.5));
#else
                        s.w[d][j] = kernel<ORDER>(std::fabs(-x + go));
#endif
                    }
                }
                // NGP: the cell the particle is in (as particles_to_grid)
                s.first[d] = ORDER == 1 ? ix : ix + xstart;
            }
            return s;
        }

        inline int positive_mod(long long i, int n) {
            long long r = i % n;
            return int(r < 0 ? r + n : r);
        }

        /// The smallest periodic interval [start, start + count) (mod n) covering the marked cells (count = 0 if none)
        inline std::pair<int, int> covering_interval(const std::vector<char> & marked) {
            const int n = int(marked.size());
            int first_marked = -1;
            for (int i = 0; i < n; i++)
                if (marked[i]) {
                    first_marked = i;
                    break;
                }
            if (first_marked < 0)
                return {0, 0};
            // The longest run of unmarked cells (cyclic), scanning from a marked cell
            int best_len = 0, best_end = first_marked; // best_end: the first marked cell after the gap
            int run = 0;
            for (int k = 1; k <= n; k++) {
                const int i = (first_marked + k) % n;
                if (not marked[i]) {
                    run++;
                } else {
                    if (run > best_len) {
                        best_len = run;
                        best_end = i;
                    }
                    run = 0;
                }
            }
            if (best_len == 0)
                return {0, n};
            return {best_end, n - best_len};
        }

        /// The pieces of the periodic interval [a, a + L) (mod n, 0 <= a < n, L <= n) inside [b, b + c): pairs of
        /// (global start, offset in the interval) with a length
        struct Piece {
            int global_start, local_start, count;
        };
        inline int interval_pieces(int a, int L, int b, int c, int n, Piece out[2]) {
            int np = 0;
            for (int shift : {0, -n}) {
                const int lo = std::max(b, a + shift);
                const int hi = std::min(b + c, a + L + shift);
                if (lo < hi)
                    out[np++] = Piece{lo, lo - (a + shift), hi - lo};
            }
            return np;
        }
    } // namespace detail

    /// Deposit the particles part[0..NumPart) (anywhere in the box, on any task) to a distributed mesh: every cell gets
    /// the sum over all particles on all tasks of factor * weight * W, with W the B-spline kernel of order ORDER and the
    /// weight the particle's mass (get_mass), delta-f weight (get_deltaf_weight) or 1. add_local(global_cell, value)
    /// is called on the task that owns the cell (partition.box[ThisTask]) and must add value to it; it may be called
    /// several times for a cell (once per contributing task and pass). Collective
    template <int N, int ORDER, class T, class AddLocal>
    DepositInfo deposit_to_partitioned_mesh(const T * part,
                                            size_t NumPart,
                                            const MeshPartition<N> & partition,
                                            double factor,
                                            AddLocal && add_local,
                                            const std::array<double, N> & assignment_offset = {},
                                            const DepositOptions & options = {}) {
        static_assert(ORDER >= 1 and ORDER <= 5, "[deposit_to_partitioned_mesh] ORDER must be 1..5");
        partition.validate();
        const auto & mesh = partition.mesh;
        DepositInfo info;

        bool shifted = false;
        for (int d = 0; d < N; d++) {
            assert_mpi(std::isfinite(assignment_offset[d]) and assignment_offset[d] >= 0.0 and
                           assignment_offset[d] < 1.0 / mesh[d],
                       "[deposit_to_partitioned_mesh] Assignment offsets must be in [0, 1/Nmesh)");
            shifted = shifted or assignment_offset[d] != 0.0;
        }
        auto stencil = [&](size_t i) {
            return detail::stencil_of<N, ORDER, T>(part[i], mesh, assignment_offset, shifted);
        };

        //==========================================================
        // 1. The periodic bounding box of the cells touched by the local particles
        //==========================================================
        std::array<int, N> pstart{}, pcount{};
        size_t patch_volume = 1;
        {
            std::array<std::vector<char>, N> marked;
            for (int d = 0; d < N; d++)
                marked[d].assign(mesh[d], 0);
            for (size_t i = 0; i < NumPart; i++) {
                const auto s = stencil(i);
                for (int d = 0; d < N; d++)
                    for (int j = 0; j < ORDER; j++)
                        marked[d][detail::positive_mod(s.first[d] + j, mesh[d])] = 1;
            }
            for (int d = 0; d < N; d++) {
                const auto [a, L] = detail::covering_interval(marked[d]);
                pstart[d] = a;
                pcount[d] = L;
                patch_volume *= size_t(L);
            }
        }
        for (int d = 0; d < std::min(N, 3); d++)
            info.patch_count[d] = pcount[d];

        // The size of a pass: whole x-slices of the patch, at most max_patch_cells cells (at least one slice)
        size_t budget = options.max_patch_cells;
        if (budget == 0) {
            double cells = 1.0;
            for (int d = 0; d < N; d++)
                cells *= mesh[d];
            budget = std::max(size_t(1) << 18, size_t(2.0 * cells / FML::NTasks) + 1);
        }
        const size_t slice_cells = pcount[0] > 0 ? patch_volume / size_t(pcount[0]) : 0;
        const int slices_per_pass =
            pcount[0] > 0 ? int(std::max<size_t>(1, std::min<size_t>(pcount[0], budget / std::max<size_t>(1, slice_cells))))
                          : 1;
        info.passes = pcount[0] > 0 ? (pcount[0] + slices_per_pass - 1) / slices_per_pass : 0;
        int npasses = info.passes;
        FML::MaxOverTasks(&npasses);

        // Strides of the patch (row-major, the last axis fastest)
        std::array<size_t, N> stride{};
        stride[N - 1] = 1;
        for (int d = N - 2; d >= 0; d--)
            stride[d] = stride[d + 1] * size_t(pcount[d + 1]);

        // The patch coordinate of the first cell of the stencil along an axis (in [0, mesh) relative to pstart)
        auto patch_coord = [&](int global, int d) { return detail::positive_mod((long long)global - pstart[d], mesh[d]); };

#ifdef USE_OMP
        const int nthreads = omp_get_max_threads();
#else
        const int nthreads = 1;
#endif

        std::vector<double> patch;
        for (int pass = 0; pass < npasses; pass++) {
            const int x0 = pass * slices_per_pass;                            // first patch slice of this pass
            const int nx = pass < info.passes ? std::min(slices_per_pass, pcount[0] - x0) : 0;
            std::vector<std::vector<double>> outgoing(FML::NTasks);

            if (nx > 0) {
                patch.assign(size_t(nx) * slice_cells, 0.0);

                //==========================================================
                // 2. Deposit into the patch (only the cells in this pass's slices)
                //==========================================================
                auto add_particle = [&](size_t i) {
                    const auto s = stencil(i);
                    const double m = factor * detail::deposit_weight(part[i]);
                    // Patch coordinates along each axis (the first axis relative to this pass)
                    std::array<std::array<int, ORDER>, N> pc;
                    for (int d = 0; d < N; d++)
                        for (int j = 0; j < ORDER; j++)
                            pc[d][j] = patch_coord(s.first[d] + j, d) - (d == 0 ? x0 : 0);
                    constexpr int ncells = FML::power(ORDER, N);
                    for (int c = 0; c < ncells; c++) {
                        double w = 1.0;
                        size_t index = 0;
                        bool inside = true;
                        for (int d = 0, n = 1; d < N; d++, n *= ORDER) {
                            const int j = c / n % ORDER;
                            const int coord = pc[d][j];
                            if (d == 0 and (coord < 0 or coord >= nx)) {
                                inside = false;
                                break;
                            }
                            w *= s.w[d][j];
                            index += size_t(coord) * stride[d];
                        }
                        if (inside)
                            patch[index] += w * m;
                    }
                };

                // Several threads: the particles in x-chunks of at least ORDER + 1 slices, even chunks in parallel,
                // then odd ones (chunks c and c + 2 never write the same cells), each chunk in particle order
                int nchunks = nthreads > 1 ? nx / (ORDER + 1) : 0;
                const bool wraps = pcount[0] == mesh[0] and nx == pcount[0]; // stencils can wrap around the patch
                if (wraps and nchunks % 2 == 1)
                    nchunks--;
                if (nchunks < 4 or NumPart >= size_t(std::numeric_limits<uint32_t>::max())) {
                    for (size_t i = 0; i < NumPart; i++)
                        add_particle(i);
                } else {
                    // The chunk of a particle from the first slice of its stencil (stencils reaching into this pass
                    // from the left are put in chunk 0, those starting to the right of it are skipped)
                    std::vector<uint32_t> chunk(NumPart);
                    const uint32_t skip = uint32_t(nchunks);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                    for (size_t i = 0; i < NumPart; i++) {
                        const auto s = stencil(i);
                        const int first = patch_coord(s.first[0], 0) - x0;
                        int c;
                        if (first >= nx) {
                            // Starts to the right of the pass: it can still reach into it if the patch wraps
                            c = (first - mesh[0] + ORDER - 1 >= 0) ? 0 : -1;
                        } else if (first < 0) {
                            c = (first + ORDER - 1 >= 0) ? 0 : -1;
                        } else {
                            c = int((long long)first * nchunks / nx);
                        }
                        chunk[i] = c < 0 ? skip : uint32_t(c);
                    }
                    std::vector<size_t> start(nchunks + 2, 0);
                    for (size_t i = 0; i < NumPart; i++)
                        start[chunk[i] + 1]++;
                    for (int c = 0; c <= nchunks; c++)
                        start[c + 1] += start[c];
                    std::vector<uint32_t> order(NumPart);
                    {
                        std::vector<size_t> next(start.begin(), start.end() - 1);
                        for (size_t i = 0; i < NumPart; i++)
                            order[next[chunk[i]]++] = uint32_t(i);
                    }
                    std::vector<uint32_t>().swap(chunk);
                    for (int phase = 0; phase < 2; phase++) {
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                        for (int c = phase; c < nchunks; c += 2)
                            for (size_t k = start[c]; k < start[c + 1]; k++)
                                add_particle(order[k]);
                    }
                }

                //==========================================================
                // 3. The blocks of the patch in the box of each task: N global starts and N counts, then the values
                // (row-major, last axis fastest)
                //==========================================================
                for (int r = 0; r < FML::NTasks; r++) {
                    const auto & box = partition.box[r];
                    if (box.volume() == 0)
                        continue;
                    std::array<detail::Piece[2], N> pieces;
                    std::array<int, N> npieces{};
                    bool empty = false;
                    for (int d = 0; d < N; d++) {
                        const int a = d == 0 ? (pstart[0] + x0) % mesh[0] : pstart[d];
                        const int L = d == 0 ? nx : pcount[d];
                        npieces[d] = detail::interval_pieces(a, L, box.start[d], box.count[d], mesh[d], pieces[d]);
                        // The offset of a piece in the pass (for d = 0) or the patch
                        empty = empty or npieces[d] == 0;
                    }
                    if (empty)
                        continue;
                    int ncombinations = 1;
                    for (int d = 0; d < N; d++)
                        ncombinations *= npieces[d];
                    auto & out = outgoing[r];
                    for (int comb = 0; comb < ncombinations; comb++) {
                        std::array<detail::Piece, N> p;
                        for (int d = 0, n = 1; d < N; d++) {
                            p[d] = pieces[d][comb / n % npieces[d]];
                            n *= npieces[d];
                        }
                        size_t count = 1;
                        for (int d = 0; d < N; d++)
                            count *= size_t(p[d].count);
                        // The values of the block (iterate over its cells)
                        auto patch_index = [&](size_t k) {
                            size_t index = 0;
                            for (int d = N - 1; d >= 0; d--) {
                                const size_t c = k % size_t(p[d].count);
                                k /= size_t(p[d].count);
                                index += (size_t(p[d].local_start) + c) * stride[d];
                            }
                            return index;
                        };
                        if (options.skip_zero_blocks) {
                            bool nonzero = false;
                            for (size_t k = 0; k < count and not nonzero; k++)
                                nonzero = patch[patch_index(k)] != 0.0;
                            if (not nonzero)
                                continue;
                        }
                        for (int d = 0; d < N; d++)
                            out.push_back(double(p[d].global_start));
                        for (int d = 0; d < N; d++)
                            out.push_back(double(p[d].count));
                        for (size_t k = 0; k < count; k++)
                            out.push_back(patch[patch_index(k)]);
                        info.values_sent += count;
                        info.blocks_sent++;
                    }
                }
            }

            //==========================================================
            // 4. Send the blocks to their tasks and add them to the grid. The stream of each source is parsed in order
            // (a block can be split over several messages)
            //==========================================================
            struct Parser {
                int nheader = 0;
                std::array<double, 2 * N> header{};
                size_t remaining = 0, done = 0;
                std::array<int, N> start{}, count{};
            } parser;
            auto consume = [&](const double * values, size_t n) noexcept {
                for (size_t i = 0; i < n; i++) {
                    if (parser.remaining == 0) {
                        parser.header[parser.nheader++] = values[i];
                        if (parser.nheader == 2 * N) {
                            parser.remaining = 1;
                            for (int d = 0; d < N; d++) {
                                parser.start[d] = int(parser.header[d]);
                                parser.count[d] = int(parser.header[N + d]);
                                parser.remaining *= size_t(parser.count[d]);
                            }
                            parser.done = 0;
                            parser.nheader = 0;
                        }
                        continue;
                    }
                    size_t k = parser.done;
                    std::array<int, N> cell;
                    for (int d = N - 1; d >= 0; d--) {
                        cell[d] = parser.start[d] + int(k % size_t(parser.count[d]));
                        k /= size_t(parser.count[d]);
                    }
                    add_local(cell, values[i]);
                    parser.done++;
                    parser.remaining--;
                }
            };
            FML::COMMUNICATION::exchange_records_chunked<double>(outgoing, consume, options.max_message_bytes);
        }
        return info;
    }

    /// The overdensity field delta of the particles on a distributed grid (FFTWGrid, PencilGrid, ...), as
    /// particles_to_grid (the same kernels and normalization: weight / mean weight, delta-f particles relative to 0),
    /// for particles on any task (e.g. a PeanoHilbertDomain). The grid needs no extra cells; they are set to the
    /// background. Collective
    template <int N, int ORDER, class T, template <int> class Grid>
    DepositInfo particles_to_grid_any_domain(const T * part,
                                             size_t NumPart,
                                             size_t NumPartTot,
                                             Grid<N> & density,
                                             const std::array<double, N> & assignment_offset = {},
                                             const DepositOptions & options = {}) {
        constexpr bool has_deltaf_weight = FML::PARTICLE::has_get_deltaf_weight<T>();
        constexpr bool has_mass = FML::PARTICLE::has_get_mass<T>() and not has_deltaf_weight;
        const int Nmesh = density.get_nmesh();
        assert_mpi(Nmesh > 0, "[particles_to_grid_any_domain] The grid is not allocated");
        assert_mpi(NumPartTot > 0, "[particles_to_grid_any_domain] No particles");

        double factor = std::pow(double(Nmesh), N) / double(NumPartTot);
        if constexpr (has_mass) {
            double mean_mass = 0.0;
            for (size_t i = 0; i < NumPart; i++)
                mean_mass += FML::PARTICLE::GetMass(const_cast<T &>(part[i]));
            SumOverTasks(&mean_mass);
            mean_mass /= double(NumPartTot);
            factor /= mean_mass;
        }

        density.fill_real_grid(has_deltaf_weight ? 0.0 : -1.0);
        const auto partition = mesh_partition_from_grid(density);
        std::array<int, N> local_start;
        for (int d = 0; d < N; d++)
            local_start[d] = int(density.get_local_range(d)[0]);
        FloatType * grid = density.get_real_grid();
        auto add_local = [&](const std::array<int, N> & cell, double value) {
            std::array<int, N> local;
            for (int d = 0; d < N; d++)
                local[d] = cell[d] - local_start[d];
            grid[density.get_index_real(local)] += FloatType(value);
        };
        return deposit_to_partitioned_mesh<N, ORDER, T>(
            part, NumPart, partition, factor, add_local, assignment_offset, options);
    }

    /// As above with the method by name (NGP, CIC, TSC, PCS, PQS)
    template <int N, class T, template <int> class Grid>
    DepositInfo particles_to_grid_any_domain(const T * part,
                                             size_t NumPart,
                                             size_t NumPartTot,
                                             Grid<N> & density,
                                             const std::string & density_assignment_method,
                                             const std::array<double, N> & assignment_offset = {},
                                             const DepositOptions & options = {}) {
        switch (interpolation_order_from_name(density_assignment_method)) {
            case 1:
                return particles_to_grid_any_domain<N, 1, T>(part, NumPart, NumPartTot, density, assignment_offset, options);
            case 2:
                return particles_to_grid_any_domain<N, 2, T>(part, NumPart, NumPartTot, density, assignment_offset, options);
            case 3:
                return particles_to_grid_any_domain<N, 3, T>(part, NumPart, NumPartTot, density, assignment_offset, options);
            case 4:
                return particles_to_grid_any_domain<N, 4, T>(part, NumPart, NumPartTot, density, assignment_offset, options);
            case 5:
                return particles_to_grid_any_domain<N, 5, T>(part, NumPart, NumPartTot, density, assignment_offset, options);
        }
        throw std::runtime_error("[particles_to_grid_any_domain] Unknown density assignment method [" +
                                 density_assignment_method + "]");
    }

    /// The overdensity field of MPIParticles on any domain (Slab, Pencil, PeanoHilbert) on a distributed grid
    template <int N, class T, template <int> class Grid>
    DepositInfo particles_to_grid_any_domain(FML::PARTICLE::MPIParticles<T> & part,
                                             Grid<N> & density,
                                             const std::string & density_assignment_method,
                                             const std::array<double, N> & assignment_offset = {},
                                             const DepositOptions & options = {}) {
        return particles_to_grid_any_domain<N, T>(part.get_particles_ptr(),
                                                  part.get_npart(),
                                                  part.get_npart_total(),
                                                  density,
                                                  density_assignment_method,
                                                  assignment_offset,
                                                  options);
    }

} // namespace FML::INTERPOLATION

#endif
