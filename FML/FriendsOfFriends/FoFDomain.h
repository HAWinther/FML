#ifndef FML_FOF_DOMAIN_H
#define FML_FOF_DOMAIN_H

// =====================================================================================================================
// Friends-of-friends for particles on any box domain (Slab or Pencil; FoF.h is for the global slabs).
//
// The algorithm of FoF.h generalized from x-slabs to boxes: a task owns the box [lo, hi) (along the axes the domain
// splits over the tasks; all of the other axes). It gets a copy of the particles of the neighbor tasks within the
// buffer of its box (Slab: the 2 neighbors in x; Pencil: up to 8 in the x-y process grid, the corners included), links
// its particles and the copies on a grid covering the box with the buffer, and keeps the groups whose center it owns
// (domain.owner(center)). Groups larger than the buffer can be cut (counted in
// FoF_number_of_groups_reaching_buffer_edge, as in FoF.h). The FoF IDs of the copies go back to the tasks they came
// from.
//
// An axis with one task along it (a Slab on one task, a Pencil with P1 = 1, z) is periodic within the task: no buffer.
// =====================================================================================================================

#include <FML/FriendsOfFriends/FoF.h>
#include <FML/Global/Domain.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace FML {
    namespace FOF {

        namespace detail {
            // The box a task owns and the process grid along each axis (axes with one task: [0,1))
            template <int NDIM>
            struct DomainBoxes {
                std::array<int, NDIM> nparts{};                // tasks along the axis (1: not split)
                std::array<std::vector<double>, NDIM> edges{}; // nparts + 1 edges
                std::array<int, NDIM> coord{};                 // the process coordinates of this task

                explicit DomainBoxes(const FML::Domain & domain) {
                    for (int d = 0; d < NDIM; d++) {
                        nparts[d] = 1;
                        edges[d] = {0.0, 1.0};
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
                        static_assert(NDIM >= 2, "FoF on a Pencil domain needs NDIM >= 2");
                        const auto & pencil = domain.get<FML::PencilDomain>();
                        const auto P = pencil.process_grid();
                        const auto c = pencil.coords_of(FML::ThisTask);
                        nparts[0] = P[0];
                        nparts[1] = P[1];
                        edges[0] = pencil.x_edges();
                        edges[1] = pencil.y_edges();
                        coord[0] = c[0];
                        coord[1] = c[1];
                    } else {
                        assert_mpi(false, "[FriendsOfFriends] Only Slab and Pencil domains are supported");
                    }
                }
                double lo(int d) const { return edges[d][coord[d]]; }
                double hi(int d) const { return edges[d][coord[d] + 1]; }
                bool split(int d) const { return nparts[d] > 1; }
            };
        } // namespace detail

        /// Friends-of-friends of particles on a domain (Slab or Pencil): the same arguments and results as
        /// FriendsOfFriends in FoF.h, with the domain of the particles first (e.g. MPIParticles::get_domain())
        template <class T, int NDIM, class FoFHaloClass = FoFHalo<T, NDIM>>
        void FriendsOfFriends(const FML::Domain & domain,
                              T * part,
                              size_t NumPart,
                              double linking_length,
                              int nmin_FoF_group,
                              bool periodic,
                              double Buffersize_over_Boxsize,
                              std::vector<FoFHaloClass> & LocalFoFGroups,
                              int Ngrid,
                              double mean_particle_separation) {

            LocalFoFGroups.clear();
            FoF_number_of_groups_reaching_buffer_edge = 0;
            assert_mpi(domain.ntasks() == FML::NTasks, "[FriendsOfFriends] The domain is not made for NTasks tasks");
            assert_mpi(FML::PARTICLE::GetNDIM(T()) == NDIM,
                       "[FriendsOfFriends] The dimensions of particle and method do not match");
            size_t NumPart_tot = NumPart;
            FML::SumOverTasks(&NumPart_tot);
            if (NumPart_tot == 0)
                return;

            // The linking distance and the grid (as in FoF.h)
            if (not(mean_particle_separation > 0.0))
                mean_particle_separation = 1.0 / std::pow(double(NumPart_tot), 1.0 / NDIM);
            const double fof_distance = linking_length * mean_particle_separation;
            const double fof_distance2 = fof_distance * fof_distance;
            if (Ngrid <= 0)
                Ngrid = FoF_Ngrid_max;
            Ngrid = std::min(int(0.5 / fof_distance), Ngrid);
            Ngrid = std::max(1, std::min(Ngrid, int(1.0 / mean_particle_separation)));
            assert_mpi(1.0 / Ngrid > 2 * fof_distance,
                       "[FriendsOfFriends] The gridsize is too large (larger than the 2x linking distance)");

            // The boxes and the buffer
            const detail::DomainBoxes<NDIM> boxes(domain);
            const double b = Buffersize_over_Boxsize;
            for (int d = 0; d < NDIM; d++)
                if (boxes.split(d))
                    for (int t = 0; t < boxes.nparts[d]; t++)
                        assert_mpi(b <= boxes.edges[d][t + 1] - boxes.edges[d][t],
                                   "[FriendsOfFriends] The buffer is larger than a task's part of an axis: reduce the "
                                   "buffer (or the number of tasks)");
            for (size_t i = 0; i < NumPart; i++) {
                const auto * pos = FML::PARTICLE::GetPos(part[i]);
                bool inside = true;
                for (int d = 0; d < NDIM; d++)
                    if (boxes.split(d))
                        inside = inside and pos[d] >= boxes.lo(d) and pos[d] < boxes.hi(d);
                assert_mpi(inside, "[FriendsOfFriends] A particle is not in the domain of the task");
            }

            // Sort the particles by y (faster linking, as in FoF.h)
            std::sort(part, part + NumPart, [](const T & a, const T & c) {
                return FML::PARTICLE::GetPos(const_cast<T &>(c))[1 % NDIM] >
                       FML::PARTICLE::GetPos(const_cast<T &>(a))[1 % NDIM];
            });

            //=========================================================================
            // The buffer. One message per neighbor offset o in {-1, 0, +1}^(split axes) \ {0}: the particles within b
            // of the faces of the box on the side o (unwrapped across the periodic box to be next to the box of the
            // neighbor), to the task at coord + o, while receiving from the task at coord - o
            //=========================================================================
            std::vector<int> split_axes;
            for (int d = 0; d < NDIM; d++)
                if (boxes.split(d))
                    split_axes.push_back(d);
            std::vector<std::array<int, NDIM>> offsets;
            {
                const int nsplit = int(split_axes.size());
                int ncomb = 1;
                for (int k = 0; k < nsplit; k++)
                    ncomb *= 3;
                for (int c = 0; c < ncomb; c++) {
                    std::array<int, NDIM> o{};
                    bool zero = true;
                    for (int k = 0, n = c; k < nsplit; k++, n /= 3) {
                        o[split_axes[k]] = n % 3 - 1;
                        zero = zero and o[split_axes[k]] == 0;
                    }
                    if (not zero)
                        offsets.push_back(o);
                }
            }
            auto rank_of = [&](std::array<int, NDIM> c) {
                if (domain.is_slab())
                    return ((c[0] % boxes.nparts[0]) + boxes.nparts[0]) % boxes.nparts[0];
                return domain.get<FML::PencilDomain>().rank_of(c[0], c[1]);
            };
            // Is the particle sent along the offset (and the shift of its position for the neighbor)? Non-periodic:
            // not across the edge of the box
            auto sent_along = [&](const T & p, const std::array<int, NDIM> & o, std::array<double, NDIM> & shift) {
                const auto * pos = FML::PARTICLE::GetPos(const_cast<T &>(p));
                for (int d = 0; d < NDIM; d++) {
                    shift[d] = 0.0;
                    if (o[d] == 0)
                        continue;
                    const bool near = o[d] < 0 ? pos[d] < boxes.lo(d) + b : pos[d] >= boxes.hi(d) - b;
                    if (not near)
                        return false;
                    const bool wraps =
                        (o[d] < 0 and boxes.coord[d] == 0) or (o[d] > 0 and boxes.coord[d] == boxes.nparts[d] - 1);
                    if (wraps and not periodic)
                        return false;
                    if (wraps)
                        shift[d] = o[d] < 0 ? 1.0 : -1.0;
                }
                return true;
            };

            const size_t bytes_per_particle = NumPart > 0 ? size_t(FML::PARTICLE::GetSize(part[0])) : 0;
            FML::COMMUNICATION::Count record_bytes = bytes_per_particle;
            const auto record_sizes = FML::GatherFromTasks(&record_bytes);
            const size_t common_bytes = size_t(*std::max_element(record_sizes.begin(), record_sizes.end()));

            std::vector<T> boundary_particles;
            std::vector<size_t> nsend(offsets.size(), 0), nrecv(offsets.size(), 0), recv_start(offsets.size(), 0);
            std::vector<std::array<int, NDIM>> neighbor_send(offsets.size()), neighbor_recv(offsets.size());
            for (size_t m = 0; m < offsets.size(); m++) {
                std::array<int, NDIM> to = boxes.coord, from = boxes.coord;
                for (int d = 0; d < NDIM; d++) {
                    to[d] += offsets[m][d];
                    from[d] -= offsets[m][d];
                }
                neighbor_send[m] = to;
                neighbor_recv[m] = from;
            }
#ifdef USE_MPI
            for (size_t m = 0; m < offsets.size(); m++) {
                // Pack
                std::vector<char> sendbuf, recvbuf;
                std::array<double, NDIM> shift;
                for (size_t i = 0; i < NumPart; i++)
                    if (sent_along(part[i], offsets[m], shift))
                        nsend[m]++;
                sendbuf.resize(nsend[m] * common_bytes);
                char * buf = sendbuf.data();
                for (size_t i = 0; i < NumPart; i++)
                    if (sent_along(part[i], offsets[m], shift)) {
                        T copy = part[i];
                        auto * pos = FML::PARTICLE::GetPos(copy);
                        for (int d = 0; d < NDIM; d++)
                            pos[d] += shift[d];
                        FML::PARTICLE::AppendToBuffer(copy, buf);
                        buf += common_bytes;
                    }
                // Exchange the counts, then the particles
                const int dest = rank_of(neighbor_send[m]), source = rank_of(neighbor_recv[m]);
                unsigned long long ns = nsend[m], nr = 0;
                MPI_Sendrecv(&ns,
                             1,
                             MPI_UNSIGNED_LONG_LONG,
                             dest,
                             10,
                             &nr,
                             1,
                             MPI_UNSIGNED_LONG_LONG,
                             source,
                             10,
                             FML::communicator,
                             MPI_STATUS_IGNORE);
                nrecv[m] = size_t(nr);
                recvbuf.resize(nrecv[m] * common_bytes);
                MPI_Sendrecv(sendbuf.data(),
                             int(FML::COMMUNICATION::checked_bytes(nsend[m], common_bytes, INT_MAX)),
                             MPI_BYTE,
                             dest,
                             11,
                             recvbuf.data(),
                             int(FML::COMMUNICATION::checked_bytes(nrecv[m], common_bytes, INT_MAX)),
                             MPI_BYTE,
                             source,
                             11,
                             FML::communicator,
                             MPI_STATUS_IGNORE);
                recv_start[m] = boundary_particles.size();
                boundary_particles.resize(boundary_particles.size() + nrecv[m]);
                buf = recvbuf.data();
                for (size_t i = 0; i < nrecv[m]; i++) {
                    FML::PARTICLE::AssignFromBuffer(boundary_particles[recv_start[m] + i], buf);
                    buf += common_bytes;
                }
            }
#endif
            const size_t NumPart_boundary = boundary_particles.size();
            const size_t NumPart_total = NumPart + NumPart_boundary;

            //=========================================================================
            // The grid over the box and the buffer (split axes: from lo - b, not periodic; the others: periodic)
            //=========================================================================
            std::array<double, NDIM> origin{};
            std::array<int, NDIM> ncells{};
            for (int d = 0; d < NDIM; d++) {
                origin[d] = boxes.split(d) ? boxes.lo(d) - b : 0.0;
                ncells[d] = boxes.split(d) ? int(std::ceil((boxes.hi(d) - boxes.lo(d) + 2 * b) * Ngrid)) : Ngrid;
            }
            size_t ncells_total = 1;
            for (int d = 0; d < NDIM; d++)
                ncells_total *= size_t(ncells[d]);
            auto particle = [&](size_t g) -> T & {
                return g < NumPart_boundary ? boundary_particles[g] : part[g - NumPart_boundary];
            };
            auto cell_coord = [&](const T & p, std::array<int, NDIM> & c, std::array<int, NDIM> * side) {
                const auto * pos = FML::PARTICLE::GetPos(const_cast<T &>(p));
                size_t index = 0;
                for (int d = 0; d < NDIM; d++) {
                    const double x = (pos[d] - origin[d]) * Ngrid;
                    c[d] = std::clamp(int(x), 0, ncells[d] - 1);
                    if (side)
                        (*side)[d] = (x - c[d] > 0.5) ? +1 : -1;
                    index = index * ncells[d] + c[d];
                }
                return index;
            };
            // Counting sort of the particles (boundary first, as in FoF.h) into the cells
            std::vector<size_t> cell_start(ncells_total + 1, 0), cell_count(ncells_total, 0),
                cell_particles(NumPart_total);
            {
                std::array<int, NDIM> c;
                for (size_t g = 0; g < NumPart_total; g++)
                    cell_start[cell_coord(particle(g), c, nullptr) + 1]++;
                for (size_t k = 0; k < ncells_total; k++)
                    cell_start[k + 1] += cell_start[k];
                std::vector<size_t> next(cell_start.begin(), cell_start.end() - 1);
                for (size_t g = 0; g < NumPart_total; g++)
                    cell_particles[next[cell_coord(particle(g), c, nullptr)]++] = g;
                for (size_t k = 0; k < ncells_total; k++)
                    cell_count[k] = cell_start[k + 1] - cell_start[k];
            }

            //=========================================================================
            // The linking (as in FoF.h: only the 2^NDIM cells on the particle's side; linked particles are removed
            // from their cells)
            //=========================================================================
            std::vector<size_t> fofid(NumPart_total, no_FoF_ID);
            std::vector<size_t> stack;
            auto find_friends = [&](size_t g, size_t id) {
                const auto * pos1 = FML::PARTICLE::GetPos(particle(g));
                std::array<int, NDIM> c, side, nb;
                cell_coord(particle(g), c, &side);
                for (int nbcell = 0; nbcell < FML::power(2, NDIM); nbcell++) {
                    bool skip = false;
                    size_t index = 0;
                    for (int d = 0, n = 1; d < NDIM; d++, n *= 2) {
                        nb[d] = c[d] + (nbcell / n % 2) * side[d];
                        if (nb[d] < 0 or nb[d] >= ncells[d]) {
                            if (boxes.split(d) or not periodic) {
                                skip = true;
                                break;
                            }
                            nb[d] = (nb[d] + ncells[d]) % ncells[d];
                        }
                        index = index * ncells[d] + nb[d];
                    }
                    if (skip)
                        continue;
                    size_t * list = cell_particles.data() + cell_start[index];
                    size_t & n_active = cell_count[index];
                    for (size_t ii = 0; ii < n_active;) {
                        const size_t h = list[ii];
                        if (fofid[h] != no_FoF_ID) {
                            std::swap(list[ii], list[n_active - 1]);
                            n_active--;
                            continue;
                        }
                        ii++;
                        if (h == g)
                            continue;
                        const auto * pos2 = FML::PARTICLE::GetPos(particle(h));
                        double dist2 = 0.0;
                        for (int d = 0; d < NDIM; d++) {
                            double dx = std::abs(pos2[d] - pos1[d]);
                            if (dx > 0.5 and periodic)
                                dx -= 1.0;
                            dist2 += dx * dx;
                        }
                        if (dist2 < fof_distance2) {
                            fofid[h] = id;
                            if (fofid[g] == no_FoF_ID)
                                fofid[g] = id;
                            stack.push_back(h);
                        }
                    }
                }
            };
            // Is a particle within a linking length of the outer edge of the buffer (its group might continue)?
            auto reaches_edge = [&](const T & p) {
                const auto * pos = FML::PARTICLE::GetPos(const_cast<T &>(p));
                for (int d = 0; d < NDIM; d++)
                    if (boxes.split(d) and
                        (pos[d] < boxes.lo(d) - b + fof_distance or pos[d] > boxes.hi(d) + b - fof_distance))
                        return true;
                return false;
            };

            size_t FoFID = 0;
            std::vector<char> kept_group;
            for (size_t g = 0; g < NumPart_total; g++) {
                if (fofid[g] != no_FoF_ID)
                    continue;
                stack.clear();
                find_friends(g, FoFID);
                if (stack.empty())
                    continue;
                FoFHaloClass halo(FoFID);
                halo.add(particle(g), periodic);
                bool edge = reaches_edge(particle(g));
                while (not stack.empty()) {
                    const size_t h = stack.back();
                    stack.pop_back();
                    halo.add(particle(h), periodic);
                    edge = edge or reaches_edge(particle(h));
                    find_friends(h, FoFID);
                }
                bool kept = false;
                if (int(halo.np) >= nmin_FoF_group) {
                    // Kept by the task that owns its center
                    std::array<double, NDIM> center;
                    const auto * hp = halo.get_pos();
                    for (int d = 0; d < NDIM; d++) {
                        center[d] = hp[d];
                        if (periodic)
                            center[d] -= std::floor(center[d]);
                        if (center[d] >= 1.0)
                            center[d] = 0.0;
                    }
                    if (domain.owner(center.data()) == FML::ThisTask) {
                        LocalFoFGroups.push_back(halo);
                        kept = true;
                        if (edge)
                            FoF_number_of_groups_reaching_buffer_edge++;
                    }
                }
                kept_group.push_back(kept ? 1 : 0);
                FoFID++;
            }
            FML::SumOverTasks(&FoF_number_of_groups_reaching_buffer_edge);
            if (FoF_number_of_groups_reaching_buffer_edge > 0 and FML::ThisTask == 0 and FoF_verbose)
                std::cout << "WARNING: [FriendsOfFriends] " << FoF_number_of_groups_reaching_buffer_edge
                          << " groups reach the edge of the buffer: increase the buffer\n";

            // Unique IDs over the tasks
            std::vector<long long int> addfofid(FML::NTasks, 0);
            for (int i = FML::ThisTask + 1; i < FML::NTasks; i++)
                addfofid[i] = (long long int)FoFID;
            FML::SumArrayOverTasks(addfofid.data(), addfofid.size());
            const size_t add = size_t(addfofid[FML::ThisTask]);
            for (auto & h : LocalFoFGroups) {
                h.id += add;
                h.finalize(periodic);
            }

            // The FoF IDs in the particles (the copies' IDs back to the tasks they came from, in the same order)
            if constexpr (FML::PARTICLE::has_set_fofid<T>()) {
                auto global_id = [&](size_t localid) {
                    return (localid != no_FoF_ID and kept_group[localid]) ? localid + add : no_FoF_ID;
                };
                for (size_t i = 0; i < NumPart; i++)
                    FML::PARTICLE::SetFoFID(part[i], global_id(fofid[NumPart_boundary + i]));
#ifdef USE_MPI
                for (size_t m = 0; m < offsets.size(); m++) {
                    std::vector<size_t> back(nrecv[m]), mine(nsend[m]);
                    for (size_t i = 0; i < nrecv[m]; i++)
                        back[i] = global_id(fofid[recv_start[m] + i]);
                    MPI_Sendrecv(back.data(),
                                 int(FML::COMMUNICATION::checked_bytes(nrecv[m], sizeof(size_t), INT_MAX)),
                                 MPI_BYTE,
                                 rank_of(neighbor_recv[m]),
                                 12,
                                 mine.data(),
                                 int(FML::COMMUNICATION::checked_bytes(nsend[m], sizeof(size_t), INT_MAX)),
                                 MPI_BYTE,
                                 rank_of(neighbor_send[m]),
                                 12,
                                 FML::communicator,
                                 MPI_STATUS_IGNORE);
                    std::array<double, NDIM> shift;
                    size_t k = 0;
                    for (size_t i = 0; i < NumPart; i++)
                        if (sent_along(part[i], offsets[m], shift)) {
                            if (mine[k] != no_FoF_ID)
                                FML::PARTICLE::SetFoFID(part[i], mine[k]);
                            k++;
                        }
                }
#endif
            }
        }

    } // namespace FOF
} // namespace FML

#endif
