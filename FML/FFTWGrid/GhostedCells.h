#ifndef FML_GHOSTEDCELLS_H
#define FML_GHOSTEDCELLS_H

// =====================================================================================================================
// GhostedCells<T, N>: per-cell data of any (trivially copyable) type T on the local cells of a distributed grid
// (FFTWGrid slabs, PencilGrid pencils, ...), with ghost layers from the neighbouring tasks. What the float grids do
// with communicate_boundaries, for labels, flags, indices, ...
//
//   - The local cells are those of the grid: get_local_range(axis) = {first, count} (global coordinates)
//   - An axis that is split over tasks (count < Nmesh) has n_left ghost layers below and n_right above. An axis that
//     is whole wraps: any coordinate is taken modulo Nmesh. Cells are addressed by their local coordinates (global
//     minus the first local one), from -n_left to count + n_right - 1 along the split axes
//   - communicate() fills the ghost layers periodically from the neighbours, corners included (the higher axes are
//     exchanged first, then the lower ones carry the ghosts of the higher ones)
//   - Tasks with no cells take no part in the exchange (the neighbours are the tasks with cells next to ours)
//
// The neighbours along a split axis must have at least n_left (n_right) layers along it.
// =====================================================================================================================

#include <FML/Global/Global.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <type_traits>
#include <vector>

namespace FML {
    namespace GRID {

        /// Per-cell data of type T on the local cells of a distributed grid with ghost layers on the split axes (see the top
        /// of this file)
        template <class T, int N>
        class GhostedCells {
            static_assert(std::is_trivially_copyable_v<T>, "GhostedCells: T must be trivially copyable");

          public:
            using IndexType = long long int;

            /// An empty container
            GhostedCells() = default;

            /// The local cells of a grid (anything with get_nmesh() and get_local_range(axis)), ghost layers on the
            /// split axes, all cells set to value
            template <class Grid>
            GhostedCells(const Grid & layout, int n_left = 1, int n_right = 1, T value = T{}) {
                std::array<std::array<long long, 2>, N> ranges;
                for (int a = 0; a < N; a++) {
                    const auto r = layout.get_local_range(a);
                    ranges[a] = {(long long)(r[0]), (long long)(r[1])};
                }
                init(layout.get_nmesh(), ranges, n_left, n_right, value);
            }

            /// Explicit local ranges {first, count} per axis (all tasks call this)
            GhostedCells(int nmesh,
                         const std::array<std::array<long long, 2>, N> & ranges,
                         int n_left = 1,
                         int n_right = 1,
                         T value = T{}) {
                init(nmesh, ranges, n_left, n_right, value);
            }

            /// The number of cells per dimension of the grid
            int get_nmesh() const { return nmesh; }
            /// The local cells along an axis: {first, count}
            std::array<long long, 2> get_local_range(int axis) const { return {start[axis], count[axis]}; }
            /// Does the axis have ghost layers (is it split over tasks)?
            bool is_split(int axis) const { return split[axis]; }
            /// Number of local (non-ghost) cells
            IndexType get_nlocal() const {
                IndexType n = 1;
                for (int a = 0; a < N; a++)
                    n *= count[a];
                return n;
            }
            /// Number of stored cells (with the ghosts)
            IndexType size() const { return IndexType(data_.size()); }

            /// The storage index of a cell from its local coordinates (whole axes wrap)
            IndexType index(std::array<int, N> c) const {
                IndexType i = 0;
                for (int a = 0; a < N; a++) {
                    long long x = c[a];
                    if (not split[a]) {
                        x %= nmesh;
                        if (x < 0)
                            x += nmesh;
                    }
                    i += (x + offset[a]) * stride[a];
                }
                return i;
            }
            /// The local coordinates of a stored cell (ghosts: outside [0, count))
            std::array<int, N> coord(IndexType i) const {
                std::array<int, N> c;
                for (int a = 0; a < N; a++) {
                    c[a] = int(i / stride[a] - offset[a]);
                    i %= stride[a];
                }
                return c;
            }
            /// Is a stored cell a ghost?
            bool is_ghost(IndexType i) const {
                const auto c = coord(i);
                for (int a = 0; a < N; a++)
                    if (c[a] < 0 or c[a] >= count[a])
                        return true;
                return false;
            }
            /// The global coordinates of a cell from its local ones (wrapped to [0, Nmesh))
            std::array<int, N> global_coord(std::array<int, N> c) const {
                for (int a = 0; a < N; a++)
                    c[a] = int(((start[a] + c[a]) % nmesh + nmesh) % nmesh);
                return c;
            }

            T & operator[](IndexType i) { return data_[i]; }
            const T & operator[](IndexType i) const { return data_[i]; }
            /// The value of a cell from its local coordinates (ghost cells: -n_left .. count + n_right - 1 along the split axes;
            /// whole axes wrap)
            T & at(const std::array<int, N> & c) { return data_[index(c)]; }
            const T & at(const std::array<int, N> & c) const { return data_[index(c)]; }
            /// All the values (local cells and ghost layers) in the internal order (index: see operator[])
            std::vector<T> & data() { return data_; }
            const std::vector<T> & data() const { return data_; }

            /// Fill the ghost layers from the neighbouring tasks (periodic, corners included)
            void communicate() {
                for (int a = N - 1; a >= 0; a--)
                    if (split[a])
                        exchange(a);
            }

          private:
            int nmesh{0};
            int nl{0}, nr{0};
            std::array<long long, N> start{}, count{}, offset{}, extent{};
            std::array<IndexType, N> stride{};
            std::array<bool, N> split{};
            std::array<int, N> lower{}, upper{};
            std::vector<T> data_;

            void init(int nmesh_,
                      const std::array<std::array<long long, 2>, N> & ranges,
                      int n_left,
                      int n_right,
                      T value) {
                nmesh = nmesh_;
                nl = n_left;
                nr = n_right;
                assert_mpi(nl >= 0 and nr >= 0, "[GhostedCells] The number of ghost layers must be >= 0");
                bool empty = false;
                for (int a = 0; a < N; a++) {
                    start[a] = ranges[a][0];
                    count[a] = ranges[a][1];
                    empty = empty or count[a] == 0;
                }
                // An axis is split if some task does not have all of it
                std::array<char, N> partial{};
                for (int a = 0; a < N; a++)
                    partial[a] = char(not empty and count[a] < nmesh);
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, partial.data(), N, MPI_CHAR, MPI_MAX, FML::communicator);
#endif
                for (int a = 0; a < N; a++) {
                    split[a] = partial[a] != 0;
                    offset[a] = split[a] ? nl : 0;
                    extent[a] = count[a] + (split[a] ? nl + nr : 0);
                }
                if (empty)
                    extent.fill(0);
                IndexType s = 1;
                for (int a = N - 1; a >= 0; a--) {
                    stride[a] = s;
                    s *= std::max<long long>(extent[a], 1);
                }
                data_.assign(empty ? 0 : size_t(s), value);
                find_neighbours();
            }

            // The task that has a cell (global coordinates), among the tasks with cells
            void find_neighbours() {
                std::array<long long, 2 * N> mine;
                for (int a = 0; a < N; a++) {
                    mine[2 * a] = start[a];
                    mine[2 * a + 1] = data_.empty() ? 0 : count[a];
                }
                std::vector<long long> all(size_t(2 * N) * FML::NTasks);
#ifdef USE_MPI
                MPI_Allgather(mine.data(), 2 * N, MPI_LONG_LONG, all.data(), 2 * N, MPI_LONG_LONG, FML::communicator);
#else
                std::copy(mine.begin(), mine.end(), all.begin());
#endif
                auto owner = [&](const std::array<long long, N> & g) {
                    for (int t = 0; t < FML::NTasks; t++) {
                        const long long * b = &all[size_t(2 * N) * t];
                        bool inside = true;
                        for (int a = 0; a < N; a++)
                            inside = inside and b[2 * a + 1] > 0 and g[a] >= b[2 * a] and g[a] < b[2 * a] + b[2 * a + 1];
                        if (inside)
                            return t;
                    }
                    return -1;
                };
                lower.fill(FML::ThisTask);
                upper.fill(FML::ThisTask);
                if (data_.empty())
                    return;
                for (int a = 0; a < N; a++) {
                    if (not split[a])
                        continue;
                    std::array<long long, N> g = start;
                    g[a] = (start[a] - 1 + nmesh) % nmesh;
                    lower[a] = owner(g);
                    g[a] = (start[a] + count[a]) % nmesh;
                    upper[a] = owner(g);
                    assert_mpi(lower[a] >= 0 and upper[a] >= 0, "[GhostedCells] No task has the neighbouring cells");
                    assert_mpi(count[a] >= std::max(nl, nr),
                               "[GhostedCells] A task has fewer cells along a split axis than ghost layers");
                }
            }

            // Copy the layers [lo, lo + w) (local coordinates) along axis a to/from a buffer (all the other cells,
            // ghosts included)
            void pack(int a, long long lo, long long w, std::vector<T> & buf) const {
                const IndexType inner = stride[a], outer = IndexType(data_.size()) / (extent[a] * inner);
                buf.resize(size_t(outer * w * inner));
                for (IndexType o = 0; o < outer; o++)
                    std::copy_n(&data_[size_t(o * extent[a] * inner + (lo + offset[a]) * inner)],
                                size_t(w * inner),
                                &buf[size_t(o * w * inner)]);
            }
            void unpack(int a, long long lo, long long w, const std::vector<T> & buf) {
                const IndexType inner = stride[a], outer = IndexType(data_.size()) / (extent[a] * inner);
                for (IndexType o = 0; o < outer; o++)
                    std::copy_n(&buf[size_t(o * w * inner)],
                                size_t(w * inner),
                                &data_[size_t(o * extent[a] * inner + (lo + offset[a]) * inner)]);
            }

            static void sendrecv(const std::vector<T> & send, std::vector<T> & recv, int dest, int source, int tag) {
                recv.resize(send.size());
                if (dest == FML::ThisTask and source == FML::ThisTask) {
                    recv = send;
                    return;
                }
#ifdef USE_MPI
                const size_t bytes = send.size() * sizeof(T), maxchunk = size_t(1) << 30;
                const char * s = reinterpret_cast<const char *>(send.data());
                char * r = reinterpret_cast<char *>(recv.data());
                size_t done = 0;
                do {
                    const int n = int(std::min(maxchunk, bytes - done));
                    MPI_Sendrecv(s + done, n, MPI_BYTE, dest, tag, r + done, n, MPI_BYTE, source, tag,
                                 FML::communicator, MPI_STATUS_IGNORE);
                    done += size_t(n);
                } while (done < bytes);
#endif
            }

            void exchange(int a) {
                if (data_.empty())
                    return;
                std::vector<T> send, recv;
                // Our first nr layers are the upper ghosts of the lower task, our last nl the lower ghosts of the upper
                if (nr > 0) {
                    pack(a, 0, nr, send);
                    sendrecv(send, recv, lower[a], upper[a], 7300 + 2 * a);
                    unpack(a, count[a], nr, recv);
                }
                if (nl > 0) {
                    pack(a, count[a] - nl, nl, send);
                    sendrecv(send, recv, upper[a], lower[a], 7301 + 2 * a);
                    unpack(a, -nl, nl, recv);
                }
            }
        };

        /// The local coordinates of a neighbouring cell of a grid for reading with get_real: the axes the grid has
        /// whole (count = Nmesh) wrap, the split ones use the extra cells (FFTWGrid: x; PencilGrid: x and y)
        template <int N, class Grid>
        std::array<int, N> wrap_whole_axes(const Grid & grid, std::array<int, N> c) {
            const int n = grid.get_nmesh();
            for (int a = 0; a < N; a++)
                if (grid.get_local_range(a)[1] == n)
                    c[a] = (c[a] % n + n) % n;
            return c;
        }

    } // namespace GRID
} // namespace FML
#endif
