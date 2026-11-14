#ifndef FML_GLOBAL_DOMAIN_BUFFER_H
#define FML_GLOBAL_DOMAIN_BUFFER_H

// =====================================================================================================================
// The buffer exchange of the particle methods (FoF, Delaunay tessellation, ...) for any box domain (Slab or
// Pencil). A task owns the box [lo, hi) along the axes the domain splits over the tasks (all of the other axes) and gets
// copies of the particles of the neighbor tasks within `buffer` of its box: Slab: the 2 neighbors in x; Pencil: up to
// 8 in the x-y process grid (the corners included). An axis with one task along it is not split (no buffer).
//
// One message per neighbor offset o in {-1, 0, +1}^(split axes) \ {0}: the particles within the buffer of the faces on
// the side o go to the task at coord + o, while the task at coord - o sends to us. With shift_positions the copies
// are moved by +-1 across the periodic box so that they are next to the box of the receiver (for methods that do not
// wrap positions themselves); without periodic, nothing is sent across the edge of the box.
//
//   DomainBuffer<NDIM> buf(domain, buffer, periodic);
//   std::vector<T> copies;
//   buf.exchange(p, NumPart, copies);                          // the copies, message by message
//   auto v = buf.forward(values_of_local_particles);           // values of the copies (same order)
//   buf.backward(values_of_copies, [&](size_t i, Q v) {...});  // back to the local particle i of each copy
// =====================================================================================================================

#include <FML/Global/Global.h>
#include <FML/Global/MPICounts.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

namespace FML {

    /// The buffer exchange of particles (copies within a buffer of the box of each task) on a Slab or Pencil domain; see
    /// the top of this file
    template <int NDIM>
    class DomainBuffer {
      public:
        /// An empty exchange
        DomainBuffer() = default;

        /// buffer: the width of the buffer along each axis (used along the split axes)
        DomainBuffer(const FML::Domain & domain, std::array<double, NDIM> buffer, bool periodic)
            : domain(domain), buffer(buffer), periodic(periodic) {
            assert_mpi(domain.ntasks() == FML::NTasks, "[DomainBuffer] The domain is not made for NTasks tasks");
            for (int d = 0; d < NDIM; d++) {
                nparts[d] = 1;
                edges[d] = {0.0, 1.0};
                coord[d] = 0;
            }
            if (domain.is_slab()) {
                const auto & slab = domain.get<FML::SlabDomain>();
                nparts[0] = slab.ntasks();
                edges[0].resize(nparts[0] + 1);
                for (int t = 0; t < nparts[0]; t++)
                    edges[0][t] = slab.x_min(t);
                edges[0][nparts[0]] = 1.0;
                coord[0] = FML::ThisTask;
            } else if (domain.holds<FML::PencilDomain>()) {
                assert_mpi(NDIM >= 2, "[DomainBuffer] A Pencil domain needs NDIM >= 2");
                const auto & pencil = domain.get<FML::PencilDomain>();
                const auto P = pencil.process_grid();
                const auto c = pencil.coords_of(FML::ThisTask);
                for (int d = 0; d < 2 and d < NDIM; d++) {
                    nparts[d] = P[d];
                    edges[d] = d == 0 ? pencil.x_edges() : pencil.y_edges();
                    coord[d] = c[d];
                }
            } else {
                assert_mpi(false, "[DomainBuffer] Only Slab and Pencil domains are supported");
            }
            // The buffer comes from the nearest neighbors only, and the two sides of a task do not overlap
            for (int d = 0; d < NDIM; d++)
                if (split(d))
                    for (int t = 0; t < nparts[d]; t++) {
                        const double w = edges[d][t + 1] - edges[d][t];
                        assert_mpi(buffer[d] >= 0.0 and buffer[d] <= w and (nparts[d] > 2 or 2 * buffer[d] <= w),
                                   "[DomainBuffer] The buffer is larger than a neighbor's part of an axis");
                    }
            // The messages
            std::vector<int> split_axes;
            for (int d = 0; d < NDIM; d++)
                if (split(d))
                    split_axes.push_back(d);
            int ncomb = 1;
            for (size_t k = 0; k < split_axes.size(); k++)
                ncomb *= 3;
            for (int comb = 0; comb < ncomb; comb++) {
                std::array<int, NDIM> o{};
                bool zero = true;
                for (int k = 0, n = comb; k < int(split_axes.size()); k++, n /= 3) {
                    o[split_axes[k]] = n % 3 - 1;
                    zero = zero and o[split_axes[k]] == 0;
                }
                if (not zero)
                    offsets.push_back(o);
            }
            send_index.resize(offsets.size());
            send_shift.resize(offsets.size());
            nrecv.assign(offsets.size(), 0);
            recv_start.assign(offsets.size(), 0);
        }

        /// Along an axis: is it split over the tasks, the box of this task and the buffer
        bool split(int d) const { return nparts[d] > 1; }
        /// The number of tasks along an axis
        int get_nparts(int d) const { return nparts[d]; }
        /// The lower edge of the box of this task along an axis (in [0,1))
        double lo(int d) const { return edges[d][coord[d]]; }
        /// The upper edge of the box of this task along an axis
        double hi(int d) const { return edges[d][coord[d] + 1]; }
        /// The width of the buffer along an axis (0 along an axis that is not split)
        double get_buffer(int d) const { return split(d) ? buffer[d] : 0.0; }
        /// The domain the exchange is made for
        const FML::Domain & get_domain() const { return domain; }

        /// The coordinate unwrapped to the periodic image closest to the box of this task
        double unwrap(double x, int d) const {
            if (not split(d) or not periodic)
                return x;
            return x - std::round(x - 0.5 * (lo(d) + hi(d)));
        }
        /// Is the position in the box of this task (along the split axes)?
        template <class Pos>
        bool in_box(const Pos * pos) const {
            for (int d = 0; d < NDIM; d++)
                if (split(d) and not(pos[d] >= lo(d) and pos[d] < hi(d)))
                    return false;
            return true;
        }
        /// Is the position strictly inside the box plus the buffer (periodic, along the split axes)?
        template <class Pos>
        bool in_covered(const Pos * pos, double tolerance = 0.0) const {
            for (int d = 0; d < NDIM; d++)
                if (split(d)) {
                    const double x = unwrap(double(pos[d]), d);
                    if (not(x > lo(d) - buffer[d] + tolerance and x < hi(d) + buffer[d] - tolerance))
                        return false;
                }
            return true;
        }

        /// Send the particles within the buffer of the neighbor tasks; receive theirs in copies (message by message)
        template <class T>
        void exchange(T * p, size_t NumPart, std::vector<T> & copies, bool shift_positions = false) {
            copies.clear();
            for (size_t m = 0; m < offsets.size(); m++) {
                send_index[m].clear();
                send_shift[m].clear();
                for (size_t i = 0; i < NumPart; i++) {
                    std::array<double, NDIM> shift;
                    if (sent_along(FML::PARTICLE::GetPos(p[i]), offsets[m], shift)) {
                        send_index[m].push_back(i);
                        send_shift[m].push_back(shift);
                    }
                }
            }
#ifdef USE_MPI
            for (size_t m = 0; m < offsets.size(); m++) {
                // Pack (the particles can have different sizes)
                size_t bytes = 0;
                for (auto i : send_index[m])
                    bytes += size_t(FML::PARTICLE::GetSize(p[i]));
                std::vector<char> sendbuf(bytes);
                char * b = sendbuf.data();
                for (size_t k = 0; k < send_index[m].size(); k++) {
                    T copy = p[send_index[m][k]];
                    if (shift_positions) {
                        auto * pos = FML::PARTICLE::GetPos(copy);
                        for (int d = 0; d < NDIM; d++)
                            pos[d] += send_shift[m][k][d];
                    }
                    FML::PARTICLE::AppendToBuffer(copy, b);
                    b += FML::PARTICLE::GetSize(copy);
                }
                // The counts and the bytes, then the particles
                unsigned long long info[2] = {send_index[m].size(), bytes}, rinfo[2] = {0, 0};
                MPI_Sendrecv(info, 2, MPI_UNSIGNED_LONG_LONG, dest(m), 20, rinfo, 2, MPI_UNSIGNED_LONG_LONG,
                             source(m), 20, FML::communicator, MPI_STATUS_IGNORE);
                std::vector<char> recvbuf(static_cast<size_t>(rinfo[1]));
                detail_sendrecv(sendbuf.data(), bytes, dest(m), recvbuf.data(), recvbuf.size(), source(m), 21);
                nrecv[m] = size_t(rinfo[0]);
                recv_start[m] = copies.size();
                copies.resize(copies.size() + nrecv[m]);
                const char * rb = recvbuf.data();
                for (size_t k = 0; k < nrecv[m]; k++) {
                    T & q = copies[recv_start[m] + k];
                    FML::PARTICLE::AssignFromBuffer(q, const_cast<char *>(rb));
                    rb += FML::PARTICLE::GetSize(q);
                }
            }
#else
            (void)p;
            (void)shift_positions;
#endif
        }

        /// The values of the copies (in the order of exchange) from the values of the local particles
        template <class Q>
        std::vector<Q> forward(const std::vector<Q> & local_values) const {
            static_assert(std::is_trivially_copyable_v<Q>, "[DomainBuffer] Only trivially copyable values");
            size_t ncopies = 0;
            for (auto n : nrecv)
                ncopies += n;
            std::vector<Q> result(ncopies);
#ifdef USE_MPI
            for (size_t m = 0; m < offsets.size(); m++) {
                std::vector<Q> send(send_index[m].size());
                for (size_t k = 0; k < send.size(); k++)
                    send[k] = local_values[send_index[m][k]];
                detail_sendrecv(send.data(), send.size() * sizeof(Q), dest(m), result.data() + recv_start[m],
                                nrecv[m] * sizeof(Q), source(m), 22);
            }
#else
            (void)local_values;
#endif
            return result;
        }

        /// Send a value of each copy back to its task: f(local index, value) for each local particle that was sent
        /// (once per message it was in)
        template <class Q, class F>
        void backward(const std::vector<Q> & copy_values, F && f) const {
            static_assert(std::is_trivially_copyable_v<Q>, "[DomainBuffer] Only trivially copyable values");
#ifdef USE_MPI
            for (size_t m = 0; m < offsets.size(); m++) {
                std::vector<Q> mine(send_index[m].size());
                detail_sendrecv(copy_values.data() + recv_start[m], nrecv[m] * sizeof(Q), source(m), mine.data(),
                                mine.size() * sizeof(Q), dest(m), 23);
                for (size_t k = 0; k < mine.size(); k++)
                    f(send_index[m][k], mine[k]);
            }
#else
            (void)copy_values;
            (void)f;
#endif
        }

        /// The number of messages (neighbour offsets) of an exchange
        size_t num_messages() const { return offsets.size(); }

      private:
        FML::Domain domain{};
        std::array<double, NDIM> buffer{};
        bool periodic{true};
        std::array<int, NDIM> nparts{};
        std::array<std::vector<double>, NDIM> edges{};
        std::array<int, NDIM> coord{};
        std::vector<std::array<int, NDIM>> offsets;
        std::vector<std::vector<size_t>> send_index;
        std::vector<std::vector<std::array<double, NDIM>>> send_shift;
        std::vector<size_t> nrecv, recv_start;

        int rank_of(std::array<int, NDIM> c) const {
            if (domain.is_slab())
                return ((c[0] % nparts[0]) + nparts[0]) % nparts[0];
            return domain.get<FML::PencilDomain>().rank_of(c[0], c[1]);
        }
        int dest(size_t m) const {
            auto c = coord;
            for (int d = 0; d < NDIM; d++)
                c[d] += offsets[m][d];
            return rank_of(c);
        }
        int source(size_t m) const {
            auto c = coord;
            for (int d = 0; d < NDIM; d++)
                c[d] -= offsets[m][d];
            return rank_of(c);
        }

        // Is the position sent along the offset (and its shift for the receiver)?
        template <class Pos>
        bool sent_along(const Pos * pos, const std::array<int, NDIM> & o, std::array<double, NDIM> & shift) const {
            for (int d = 0; d < NDIM; d++) {
                shift[d] = 0.0;
                if (o[d] == 0)
                    continue;
                const bool near = o[d] < 0 ? pos[d] < lo(d) + buffer[d] : pos[d] >= hi(d) - buffer[d];
                if (not near)
                    return false;
                const bool wraps = (o[d] < 0 and coord[d] == 0) or (o[d] > 0 and coord[d] == nparts[d] - 1);
                if (wraps and not periodic)
                    return false;
                if (wraps)
                    shift[d] = o[d] < 0 ? 1.0 : -1.0;
            }
            return true;
        }

#ifdef USE_MPI
        // Send sbytes to dest and receive rbytes from source, any number of bytes (in chunks whose count fits in an
        // int). The send and the receive peers differ, so each side posts exactly the chunks of its own message
        // (non-blocking): the number of messages to dest is set by sbytes alone (= the rbytes dest expects) and
        // likewise for the receive. Messages between a pair with the same tag arrive in order
        static void detail_sendrecv(const void * send, size_t sbytes, int dest, void * recv, size_t rbytes, int source,
                                    int tag) {
            const size_t chunk = size_t(1) << 30;
            std::vector<MPI_Request> requests;
            requests.reserve((sbytes + chunk - 1) / chunk + (rbytes + chunk - 1) / chunk);
            for (size_t ro = 0; ro < rbytes; ro += chunk) {
                requests.emplace_back();
                MPI_Irecv(static_cast<char *>(recv) + ro, int(std::min(chunk, rbytes - ro)), MPI_BYTE, source, tag,
                          FML::communicator, &requests.back());
            }
            for (size_t so = 0; so < sbytes; so += chunk) {
                requests.emplace_back();
                MPI_Isend(static_cast<const char *>(send) + so, int(std::min(chunk, sbytes - so)), MPI_BYTE, dest, tag,
                          FML::communicator, &requests.back());
            }
            MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        }
#endif
    };

} // namespace FML

#endif
