#ifndef FOF_HEADER
#define FOF_HEADER
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <iterator>
#include <map>

#include <FML/FriendsOfFriends/FoFBinning.h>
#include <FML/Global/Global.h>

namespace FML {

    //========================================================================================
    /// This namespace deals how to link together particles in groups
    //========================================================================================

    namespace FOF {

        // If a particle belongs to no FoF group it is given this FoF ID
        const constexpr size_t no_FoF_ID = SIZE_MAX;

        // The maximum grid we are allowed to use to bin the particle to
        // This is just to have a memory limit in what the algorithm is
        // allowed to allocate
        int FoF_Ngrid_max = 256;

        // If we are to merge FoF groups across boundaries in parallel or not
        // False is guaranteed to be save. True works as long as a FoF group
        // do not span more than 2 tasks
        bool merging_in_parallel_default = false;

        // For communicating boundary particles. Use float to reduce communication cost
        // at no big precision loss
        using FoFPosType = double;

        template <class T, int NDIM, class FoFHaloClass = FoFHalo<T, NDIM>>
        void FriendsOfFriends(T * part,
                              size_t NumPart,
                              double fof_distance,
                              int nmin_FoF_group,
                              bool periodic,
                              std::vector<FoFHaloClass> & LocalFoFGroups);

        template <class T, int NDIM>
        void ParticlesToCells(int Ngrid, T * part, size_t NumPart, std::vector<FoFCells> & PartCells);

        template <class T, int NDIM>
        void FriendsOfFriendsLinkingLocal(T * part,
                                          size_t NumPart,
                                          std::vector<FoFCells> & PartCells,
                                          int Ngrid,
                                          int Local_nx,
                                          double fof_distance,
                                          std::vector<size_t> & particle_id_FoF,
                                          bool periodic);

        template <class T, int NDIM>
        void BoundaryLinking(double fof_distance,
                             T * part,
                             size_t NumPart,
                             int Ngrid,
                             const std::vector<FoFCells> & PartCells,
                             std::vector<size_t> & particle_id_FoF,
                             std::vector<size_t> & BoundaryParticleIndex,
                             std::vector<size_t> & BoundaryParticleRightFoFIndex,
                             std::vector<char> & isShared,
                             bool periodic,
                             bool merging_in_parallel);

        //=========================================================================
        // Bin local particles to cells in a grid
        //=========================================================================
        template <class T, int NDIM>
        void ParticlesToCells(int Ngrid, T * part, size_t NumPart, std::vector<FoFCells> & PartCells) {

            // Adjust Ngrid if it doesnt divide the gridsize
            Ngrid = Ngrid - Ngrid % FML::NTasks;
            if (Ngrid % FML::NTasks > Ngrid / 2)
                Ngrid *= 2;
            const int Local_nx = Ngrid / FML::NTasks;
            const size_t NgridTot = size_t(Local_nx) * FML::power(Ngrid, NDIM - 1);

            // Make a grid containing particles
            PartCells.resize(NgridTot);

            // Get index of the cell the particle lives in
            auto get_cell_index = [&](size_t ipart, std::vector<int> & coord) -> size_t {
                auto * pos = part[ipart].get_pos();
                size_t index_cell = 0;
                for (int idim = 0; idim < NDIM; idim++) {
                    coord[idim] = int((pos[idim] - FML::xmin_domain * (idim == 0 ? 1 : 0)) * Ngrid);
                    index_cell = index_cell * Ngrid + coord[idim];
                    assert(coord[idim] >= 0 and coord[idim] < (idim == 0 ? Local_nx : Ngrid));
                }
                assert(index_cell >= 0 and index_cell < NgridTot);
                return index_cell;
            };

            // Add up number of particles in cells
            std::vector<int> coord(NDIM);
            for (size_t i = 0; i < NumPart; i++) {
                const size_t index_cell = get_cell_index(i, coord);
                PartCells[index_cell].np++;
            }

            // Allocate cells
            for (auto & cell : PartCells) {
                cell.ParticleIndex.resize(cell.np);
                cell.np = 0;
            }

            for (size_t i = 0; i < NumPart; i++) {
                const size_t index_cell = get_cell_index(i, coord);
                auto & np = PartCells[index_cell].np;
                PartCells[index_cell].ParticleIndex[np] = i;
                np++;
            }
        }

        template <class T, int NDIM>
        void BoundaryLinking(double fof_distance,
                             T * part,
                             size_t NumPart,
                             int Ngrid,
                             const std::vector<FoFCells> & PartCells,
                             std::vector<size_t> & particle_id_FoF,
                             std::vector<size_t> & BoundaryParticleIndex,
                             std::vector<size_t> & BoundaryParticleRightFoFIndex,
                             std::vector<char> & isShared,
                             bool periodic,
                             bool merging_in_parallel) {

            //=========================================================================
            // We find the particles that are close to the right boundary of the domain
            // communicate them to the right, do the FoF linking and send back particles
            // that have links with the right task. In the process we link together
            // FoF groups that are only linked together via the boundary particles
            // This is done one pair of CPUs at the time to ensure that this is correct
            // However if you know a priori that no FoF groups can span more than 2
            // tasks then this can be done in parallel ( merging_in_parallel )
            // Periodic means the box wraps around
            //=========================================================================

            if (FML::NTasks == 1)
                return;

#ifdef USE_MPI

            const double fof_distance2 = fof_distance * fof_distance;
            const constexpr int nblocksearchpartgrid = 3;
            const constexpr int threetondim = FML::power(nblocksearchpartgrid, NDIM);

            // Count boundary particles
            size_t nboundary_right = 0;
            for (size_t i = 0; i < NumPart; i++) {
                const auto * pos = part[i].get_pos();
                if (FML::xmax_domain - pos[0] < fof_distance)
                    nboundary_right++;
            }

            // Allocate and gather the data
            const int bytes_per_partice = sizeof(FoFPosType) * NDIM + 2 * sizeof(size_t);
            long long int bytes_to_send = bytes_per_partice * nboundary_right;
            std::vector<char> CommBufferSend(bytes_to_send);

            // Gather data. We wrap it all up in a byte array and use get_pos and get_id
            // defined below to extract the data
            char * data = CommBufferSend.data();
            for (size_t i = 0; i < NumPart; i++) {
                FoFPosType pos[NDIM];
                if (FML::xmax_domain - part[i].get_pos()[0] < fof_distance) {
                    for (int idim = 0; idim < NDIM; idim++) {
                        pos[idim] = FoFPosType(part[i].get_pos()[idim]);
                    }
                    std::memcpy(data, pos, sizeof(FoFPosType) * NDIM);
                    data += sizeof(FoFPosType) * NDIM;
                    std::memcpy(data, &i, sizeof(size_t));
                    data += sizeof(size_t);
                    std::memcpy(data, &particle_id_FoF[i], sizeof(size_t));
                    data += sizeof(size_t);
                }
            }

            // Communicate how much to send and recieve
            MPI_Status status;
            const int RightTask = (FML::ThisTask + 1) % FML::NTasks;
            const int LeftTask = (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks;
            long long int bytes_to_recv;
            MPI_Sendrecv(&bytes_to_send,
                         sizeof(bytes_to_send),
                         MPI_BYTE,
                         RightTask,
                         0,
                         &bytes_to_recv,
                         sizeof(bytes_to_recv),
                         MPI_BYTE,
                         LeftTask,
                         0,
                         MPI_COMM_WORLD,
                         &status);

            // In a non-periodic domain we don't need to do the left and rightmost task
            if (!periodic) {
                if (FML::ThisTask == 0)
                    bytes_to_recv = 0;
                if (FML::ThisTask == FML::NTasks - 1)
                    bytes_to_send = 0;
            }

            // Do the merging 1 by 1 or if merging_in_parallel do it in parallel
            // Merging in parallel will give wrong results if FoF groups span more than 2 tasks
            const int nmerging_steps = merging_in_parallel ? 1 : (periodic ? FML::NTasks : FML::NTasks - 1);
            // for(int istep = 0; istep < nmerging_steps; istep++){
            for (int ss = 0; ss < nmerging_steps; ss++) {
                // We go alternating right and left from task 0 if periodic to ensure
                // all groups gets find even if they span the whole box
                int istep = 0;
                if (periodic)
                    istep = ss % 2 == 0 ? ss / 2 : FML::NTasks - (ss + 1) / 2;
                else
                    istep = ss;

                const int sendTask = istep % FML::NTasks;
                const int recvTask = ((istep + 1) % FML::NTasks);

                if (FML::ThisTask == sendTask or FML::ThisTask == recvTask or merging_in_parallel) {

#ifdef DEBUG_FOF
                    if (FML::ThisTask == sendTask or merging_in_parallel) {
                        std::cout << "Sending boundary particles from " << FML::ThisTask << " to " << RightTask
                                  << " and merging FoF groups" << std::endl;
                        std::cout << FML::ThisTask << " will send " << bytes_to_send / 1e3 << " kb of data to task "
                                  << RightTask << std::endl;
                    }
                    if (FML::ThisTask == RightTask or merging_in_parallel) {
                        std::cout << FML::ThisTask << " will recieve " << bytes_to_recv / 1e3
                                  << " kb of data from task " << LeftTask << std::endl;
                    }
#endif

                    // Send and recieve the data
                    std::vector<char> CommBufferRecv;
                    if (FML::ThisTask == recvTask or merging_in_parallel)
                        CommBufferRecv.resize(bytes_to_recv);

                    const size_t nboundary_left_recv = bytes_to_recv / bytes_per_partice;

                    if (merging_in_parallel) {
                        MPI_Sendrecv(CommBufferSend.data(),
                                     bytes_to_send,
                                     MPI_BYTE,
                                     RightTask,
                                     0,
                                     CommBufferRecv.data(),
                                     bytes_to_recv,
                                     MPI_BYTE,
                                     LeftTask,
                                     0,
                                     MPI_COMM_WORLD,
                                     &status);
                    } else {
                        if (FML::ThisTask == sendTask)
                            MPI_Send(CommBufferSend.data(), bytes_to_send, MPI_BYTE, RightTask, 0, MPI_COMM_WORLD);
                        if (FML::ThisTask == recvTask) {
                            MPI_Recv(
                                CommBufferRecv.data(), bytes_to_recv, MPI_BYTE, LeftTask, 0, MPI_COMM_WORLD, &status);
                        }
                    }

                    // Methods to extract data from the stuff we communicated
                    auto get_pos = [&](int i) -> FoFPosType * {
                        char * p = &CommBufferRecv[bytes_per_partice * i];
                        assert(FML::ThisTask == recvTask or merging_in_parallel);
                        assert(size_t(i) >= 0 and size_t(i) < nboundary_left_recv);
                        return (FoFPosType *)p;
                    };
                    auto get_fof_id = [&](int i) -> size_t * {
                        assert(FML::ThisTask == recvTask or merging_in_parallel);
                        assert(size_t(i) >= 0 and size_t(i) < nboundary_left_recv);
                        return (size_t *)&CommBufferRecv[bytes_per_partice * i + sizeof(FoFPosType) * NDIM +
                                                         sizeof(size_t)];
                    };
                    auto get_ind = [&](int i) -> size_t * {
                        assert(FML::ThisTask == recvTask or merging_in_parallel);
                        assert(size_t(i) >= 0 and size_t(i) < nboundary_left_recv);
                        return (size_t *)&CommBufferRecv[bytes_per_partice * i + sizeof(FoFPosType) * NDIM];
                    };

                    // To keep track of if a boundary particle on the left has a link on current task
                    std::vector<char> BoundaryParticle;
                    if (FML::ThisTask == recvTask or merging_in_parallel) {
                        BoundaryParticle = std::vector<char>(nboundary_left_recv, 0);
                        isShared = std::vector<char>(NumPart, 0);
                    }

                    // Loop over boundary particles and determine which ones has a link
                    bool skip = false;
                    if (FML::ThisTask == recvTask or merging_in_parallel) {
                        for (size_t i = 0; i < nboundary_left_recv; i++) {
                            const FoFPosType * pos2 = get_pos(i);

                            // Compute cell coordinate
                            std::vector<int> coord(NDIM);
                            coord[0] = 0;
                            for (int idim = 1; idim < NDIM; idim++) {
                                coord[idim] = int(pos2[idim] * Ngrid);
                            }

                            for (int nbcell = 0; nbcell < threetondim; nbcell++) {
                                std::array<int, NDIM> icoord;
                                for (int idim = 0, threepow = 1; idim < NDIM;
                                     idim++, threepow *= nblocksearchpartgrid) {
                                    int go_left_right_of_stay =
                                        -nblocksearchpartgrid / 2 + (nbcell / threepow % nblocksearchpartgrid);
                                    icoord[idim] = coord[idim] + go_left_right_of_stay;
                                }
                                if (icoord[0] != 0)
                                    continue;

                                // Compute cell-index of nbor cell
                                size_t index_nbor_cell = 0;
                                if (!periodic)
                                    skip = false;
                                for (int idim = 0; idim < NDIM; idim++) {
                                    // Periodic boundary conditions
                                    if (periodic) {
                                        if (icoord[idim] < 0)
                                            icoord[idim] += Ngrid;
                                        if (icoord[idim] >= Ngrid)
                                            icoord[idim] -= Ngrid;
                                    } else {
                                        if (icoord[idim] < 0)
                                            skip = true;
                                        if (icoord[idim] >= Ngrid)
                                            skip = true;
                                    }
                                    index_nbor_cell = index_nbor_cell * Ngrid + icoord[idim];
                                }
                                if (!periodic and skip)
                                    continue;

                                // Loop over all particles in nbor cell
                                const size_t np = size_t(PartCells[index_nbor_cell].np);
                                for (size_t ii = 0; ii < np; ii++) {
                                    const auto pindex = PartCells[index_nbor_cell].ParticleIndex[ii];
                                    const auto * pos1 = part[pindex].get_pos();

                                    double dist2 = 0.0;
                                    for (int idim = 0; idim < NDIM; idim++) {
                                        double dx = double(pos1[idim]) - double(pos2[idim]);
                                        if (periodic) {
                                            if (dx > 0.5)
                                                dx -= 1.0;
                                            if (dx < -0.5)
                                                dx += 1.0;
                                        }
                                        dist2 += dx * dx;
                                        if (dist2 > fof_distance2)
                                            break;
                                    }

                                    // Check if we should link
                                    if (dist2 < fof_distance2) {

                                        size_t * fof_id1 = &particle_id_FoF[pindex];
                                        size_t * fof_id2 = get_fof_id(i);
                                        size_t original_fof_id1 = *fof_id1;
                                        size_t original_fof_id2 = *fof_id2;

                                        //=======================================================================
                                        // Here we tie together the groups that are linked via the boundary
                                        //=======================================================================
                                        if (original_fof_id1 == no_FoF_ID and original_fof_id2 == no_FoF_ID) {
                                            // The particle and the boundary particle does not belong to a FoF group
                                            // They together form a new group of size 2
                                            // (we currently ignore this case as its only relevant if we want 2 particle
                                            // FoF groups the way to deal with it is to have some free indexes for each
                                            // task to be able to assign then assign it here and increase counter)
                                            continue;
                                        }

                                        // For communicating back that the boundary particle forms a link
                                        BoundaryParticle[i] = 1;
                                        // Mark that the particle on the current task has a link to the left task
                                        isShared[pindex] = 1;

                                        if (original_fof_id1 == original_fof_id2)
                                            continue;
                                        if (original_fof_id1 == no_FoF_ID) {
                                            // The particle has no FoF group, but belong to the FoF group of the
                                            // boundary particle
                                            *fof_id1 = original_fof_id2;
                                        } else if (original_fof_id2 == no_FoF_ID) {
                                            // The left particle belongs to the same FoFID as the current particle
                                            *fof_id2 = original_fof_id1;
                                        } else {

                                            size_t new_fof_id = std::min(original_fof_id1, original_fof_id2);

                                            for (size_t jj = 0; jj < nboundary_left_recv; jj++) {
                                                auto * tmp = get_fof_id(jj);
                                                if (*tmp == original_fof_id1 or *tmp == original_fof_id2) {
                                                    *tmp = new_fof_id;
                                                }
                                            }

#ifdef USE_OMP
#pragma omp parallel for
#endif
                                            for (size_t jj = 0; jj < NumPart; jj++)
                                                if (particle_id_FoF[jj] == original_fof_id1 or
                                                    particle_id_FoF[jj] == original_fof_id2) {
                                                    particle_id_FoF[jj] = new_fof_id;
                                                }
                                        }
                                        //=======================================================================
                                    }
                                }
                            }
                        }
                    }

                    size_t count = 0;
                    std::vector<size_t> BoundaryDataToSendBack_pindex;
                    std::vector<size_t> BoundaryDataToSendBack_fofindex;
                    if (FML::ThisTask == recvTask or merging_in_parallel) {
                        // Count how many boundary particles
                        for (size_t i = 0; i < nboundary_left_recv; i++) {
                            count += BoundaryParticle[i];
                        }

                        // Gather the data to send back
                        BoundaryDataToSendBack_pindex.reserve(count);
                        BoundaryDataToSendBack_fofindex.reserve(count);
                        for (size_t i = 0; i < nboundary_left_recv; i++) {
                            if (BoundaryParticle[i] == 1) {
                                auto * ind = get_ind(i);
                                auto * id = get_fof_id(i);
                                BoundaryDataToSendBack_pindex.push_back(*ind);
                                BoundaryDataToSendBack_fofindex.push_back(*id);
                            }
                        }
                        assert(BoundaryDataToSendBack_pindex.size() == count);
                        assert(BoundaryDataToSendBack_fofindex.size() == count);
                    }

                    // Send back how many particles to recieve
                    size_t count_recv = 0;
                    if (merging_in_parallel) {
                        MPI_Sendrecv(&count,
                                     sizeof(count),
                                     MPI_BYTE,
                                     LeftTask,
                                     0,
                                     &count_recv,
                                     sizeof(count_recv),
                                     MPI_BYTE,
                                     RightTask,
                                     0,
                                     MPI_COMM_WORLD,
                                     &status);
                    } else {
                        if (FML::ThisTask == recvTask)
                            MPI_Send(&count, sizeof(count), MPI_BYTE, LeftTask, 0, MPI_COMM_WORLD);
                        if (FML::ThisTask == sendTask)
                            MPI_Recv(&count_recv, sizeof(count_recv), MPI_BYTE, RightTask, 0, MPI_COMM_WORLD, &status);
                    }

#ifdef DEBUG_FOF
                    if (merging_in_parallel) {
                        std::cout << "Merging " << sendTask << " <-> " << recvTask << " We have " << count
                                  << " and will recieve " << count_recv << " boundary links\n";
                    } else {
                        if (FML::ThisTask == sendTask)
                            std::cout << "Merging " << sendTask << " -> " << recvTask << " We have " << count_recv
                                      << " boundary links\n";
                    }
#endif

                    // Allocate memory to recieve
                    if (FML::ThisTask == sendTask or merging_in_parallel) {
                        BoundaryParticleIndex = std::vector<size_t>(count_recv, 0);
                        BoundaryParticleRightFoFIndex = std::vector<size_t>(count_recv, 0);
                    }

                    // Send back the final data
                    const int bytes_send2 = count * sizeof(size_t);
                    const int bytes_recv2 = count_recv * sizeof(size_t);

                    if (merging_in_parallel) {
                        MPI_Sendrecv(BoundaryDataToSendBack_pindex.data(),
                                     bytes_send2,
                                     MPI_BYTE,
                                     LeftTask,
                                     0,
                                     BoundaryParticleIndex.data(),
                                     bytes_recv2,
                                     MPI_BYTE,
                                     RightTask,
                                     0,
                                     MPI_COMM_WORLD,
                                     &status);
                        MPI_Sendrecv(BoundaryDataToSendBack_fofindex.data(),
                                     bytes_send2,
                                     MPI_BYTE,
                                     LeftTask,
                                     0,
                                     BoundaryParticleRightFoFIndex.data(),
                                     bytes_recv2,
                                     MPI_BYTE,
                                     RightTask,
                                     0,
                                     MPI_COMM_WORLD,
                                     &status);
                    } else {
                        if (FML::ThisTask == recvTask)
                            MPI_Send(BoundaryDataToSendBack_pindex.data(),
                                     bytes_send2,
                                     MPI_BYTE,
                                     LeftTask,
                                     0,
                                     MPI_COMM_WORLD);
                        if (FML::ThisTask == sendTask)
                            MPI_Recv(BoundaryParticleIndex.data(),
                                     bytes_recv2,
                                     MPI_BYTE,
                                     RightTask,
                                     0,
                                     MPI_COMM_WORLD,
                                     &status);

                        if (FML::ThisTask == recvTask)
                            MPI_Send(BoundaryDataToSendBack_fofindex.data(),
                                     bytes_send2,
                                     MPI_BYTE,
                                     LeftTask,
                                     0,
                                     MPI_COMM_WORLD);
                        if (FML::ThisTask == sendTask)
                            MPI_Recv(BoundaryParticleRightFoFIndex.data(),
                                     bytes_recv2,
                                     MPI_BYTE,
                                     RightTask,
                                     0,
                                     MPI_COMM_WORLD,
                                     &status);
                    }

                    // Update FoF group indices
                    if (FML::ThisTask == sendTask or merging_in_parallel) {

                        for (size_t i = 0; i < count_recv; i++) {
                            const size_t pindex = BoundaryParticleIndex[i];
                            assert(pindex < NumPart);
                            const size_t NewFoFIndex = BoundaryParticleRightFoFIndex[i];
                            assert(NewFoFIndex != no_FoF_ID);
                            const size_t OldFoFIndex = particle_id_FoF[pindex];
                            if (OldFoFIndex == NewFoFIndex)
                                continue;
                            if (OldFoFIndex == no_FoF_ID) {
                                particle_id_FoF[pindex] = NewFoFIndex;
                            } else {
#ifdef USE_OMP
#pragma omp parallel for
#endif
                                for (size_t j = 0; j < NumPart; j++)
                                    if (particle_id_FoF[j] == OldFoFIndex) {
                                        particle_id_FoF[j] = NewFoFIndex;
                                    }
                            }
                        }
                    }

                    // We now finally have the list of particles with links
#ifdef DEBUG_FOF
                    if (FML::ThisTask == sendTask or merging_in_parallel)
                        std::cout << FML::ThisTask << " has " << nboundary_right << " boundary particles and "
                                  << count_recv << " of them have links" << std::endl;
#endif
                }
                if (!merging_in_parallel)
                    MPI_Barrier(MPI_COMM_WORLD);
            }
#endif
        }

        template <class T, int NDIM>
        void FriendsOfFriendsLinkingLocal(T * part,
                                          size_t NumPart,
                                          std::vector<FoFCells> & PartCells,
                                          int Ngrid,
                                          int Local_nx,
                                          double fof_distance,
                                          std::vector<size_t> & particle_id_FoF,
                                          bool periodic) {

            //=========================================================================
            // Link together all particles in the local domain that are located within
            // a linkking distance of linking_length times the mean particle separation.
            // The result is a list of all the particles and an associated FoF group ID
            // This ID can be arbitrary but is guaranteed to be unique across tasks
            //=========================================================================

            const double fof_distance2 = fof_distance * fof_distance;
            const size_t NgridTot = size_t(Local_nx) * FML::power(Ngrid, NDIM - 1);
            const constexpr int nblocksearchpartgrid = 3;
            const constexpr int threetondim = FML::power(nblocksearchpartgrid, NDIM);

            // Some basic checks
            assert(part[0].get_ndim() == NDIM);

#ifdef DEBUG_FOF
            if (FML::ThisTask == 0) {
                std::cout << "FriendsOfFriendsLinkingLocal\n";
                std::cout << "FoF Linking Distance: " << fof_distance << "\n";
                std::cout << "FoF Gridsize Ngrid = " << Ngrid << " Local: " << Local_nx << "\n";
            }
#endif

            // Check that the grid is not too small
            assert(1.0 / Ngrid > fof_distance);

            // Get index of the cell the particle lives in
            auto get_cell_index = [&](size_t ipart, std::vector<int> & coord) -> size_t {
                auto * pos = part[ipart].get_pos();
                size_t index_cell = 0;
                for (int idim = 0; idim < NDIM; idim++) {
                    coord[idim] = int((pos[idim] - FML::xmin_domain * (idim == 0 ? 1 : 0)) * Ngrid);
                    index_cell = index_cell * Ngrid + coord[idim];
                    assert(coord[idim] >= 0 and coord[idim] < (idim == 0 ? Local_nx : Ngrid));
                }
                assert(index_cell >= 0 and index_cell < NgridTot);
                return index_cell;
            };

            // ID of the FoF groups
            int nFoF = 0;

            // Allocate the FoF id for each particle
            particle_id_FoF = std::vector<size_t>(NumPart, no_FoF_ID);

            // We now have cells with particles in them. Start linking
            bool skip = false;
            std::vector<int> coord(NDIM);
            for (size_t i = 0; i < NumPart; i++) {

                // Current particle
                const auto * pos1 = part[i].get_pos();

                // Compute coord of cell particle is in
                get_cell_index(i, coord);

                size_t id_FoF_group = particle_id_FoF[i];

                // If the current particle already have a FoF ID then we don't
                // increase nFoF
                if (id_FoF_group == no_FoF_ID) {
                    id_FoF_group = nFoF;
                    particle_id_FoF[i] = id_FoF_group;
                    nFoF++;
                }

                // Loop over all 3^NDIM neighbor cells (center cell is included)
                for (int nbcell = 0; nbcell < threetondim; nbcell++) {
                    std::array<int, NDIM> icoord;
                    for (int idim = 0, threepow = 1; idim < NDIM; idim++, threepow *= nblocksearchpartgrid) {
                        int go_left_right_of_stay =
                            -nblocksearchpartgrid / 2 + (nbcell / threepow % nblocksearchpartgrid);
                        icoord[idim] = coord[idim] + go_left_right_of_stay;
                    }

                    // For boundary cells in the x direction we don't have a left or a right nbor
                    // unless we only have 1 task
                    if (FML::NTasks > 1) {
                        if (icoord[0] < 0)
                            continue;
                        if (icoord[0] >= Local_nx)
                            continue;
                    }

                    // Compute cell-index or nbor cell
                    size_t index_nbor_cell = 0;
                    if (!periodic)
                        skip = false;
                    for (int idim = 0; idim < NDIM; idim++) {
                        // Periodic boundary conditions
                        if (periodic) {
                            if (icoord[idim] < 0)
                                icoord[idim] += Ngrid;
                            if (icoord[idim] >= Ngrid)
                                icoord[idim] -= Ngrid;
                        } else {
                            if (icoord[idim] < 0)
                                skip = true;
                            if (icoord[idim] >= Ngrid)
                                skip = true;
                        }
                        index_nbor_cell = index_nbor_cell * Ngrid + icoord[idim];
                    }
                    if (!periodic and skip)
                        continue;

                    // Loop over all particles in nbor cell
                    const size_t np = PartCells[index_nbor_cell].np;
                    for (size_t ii = 0; ii < np; ii++) {
                        const auto pindex = PartCells[index_nbor_cell].ParticleIndex[ii];
                        if (pindex == i)
                            continue;
                        const auto * pos2 = part[pindex].get_pos();

                        // Do the linking
                        double dist2 = 0.0;
                        for (int idim = 0; idim < NDIM; idim++) {
                            double dx = pos2[idim] - pos1[idim];
                            if (periodic) {
                                if (dx > 0.5)
                                    dx -= 1.0;
                                if (dx < -0.5)
                                    dx += 1.0;
                            }
                            dist2 += dx * dx;
                            if (dist2 > fof_distance2) {
                                break;
                            }
                        }

                        // Check if particle pair should be linked
                        if (dist2 < fof_distance2) {

                            const auto id_existing_FoF_group = particle_id_FoF[pindex];

                            if (id_existing_FoF_group == no_FoF_ID) {

                                // The particle has not been touched before and is added to current FoF group
                                particle_id_FoF[pindex] = id_FoF_group;

                            } else {

                                if (id_existing_FoF_group == id_FoF_group)
                                    continue;

                                // We merge the current FoF group with an existing one
                                size_t new_FoF_id = std::min(id_FoF_group, id_existing_FoF_group);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                                for (size_t jj = 0; jj < NumPart; jj++)
                                    if (particle_id_FoF[jj] == id_FoF_group or
                                        particle_id_FoF[jj] == id_existing_FoF_group)
                                        particle_id_FoF[jj] = new_FoF_id;

                                // The current group is set to be the exisiting group
                                id_FoF_group = new_FoF_id;
                            }
                        }
                    }
                }
            }

            //=============================================================================
            // Change the FoF ID so its unique across tasks
            //=============================================================================
            size_t FoF_id_start_local = 0;
#ifdef USE_MPI
            std::vector<long long int> max_FoF_id_over_tasks(FML::NTasks, 0);
            max_FoF_id_over_tasks[FML::ThisTask] = nFoF;
            MPI_Allreduce(
                MPI_IN_PLACE, max_FoF_id_over_tasks.data(), FML::NTasks, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

            // Update FoF ID so its unique over tasks
            for (int i = 1; i < FML::NTasks; i++) {
                max_FoF_id_over_tasks[i] += max_FoF_id_over_tasks[i - 1];
            }
            if (FML::ThisTask > 0)
                FoF_id_start_local = size_t(max_FoF_id_over_tasks[FML::ThisTask - 1]);
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t jj = 0; jj < NumPart; jj++)
                if (particle_id_FoF[jj] != no_FoF_ID)
                    particle_id_FoF[jj] += FoF_id_start_local;

#ifdef DEBUG_FOF
            std::cout << "FoFIndex on " << FML::ThisTask << " is in the range " << FoF_id_start_local << " -> "
                      << FoF_id_start_local + nFoF - 1 << "\n";
#endif
        }

        template <class T, int NDIM, class FoFHaloClass>
        void FriendsOfFriends(T * part,
                              size_t NumPart,
                              double fof_distance,
                              int nmin_FoF_group,
                              bool periodic,
                              std::vector<FoFHaloClass> & LocalFoFGroups) {

            //============================================================================
            // If we can assume a FoF group do not span more than 2 tasks then
            // we can merge in parallel which is faster, but will be wrong if
            // the FoF groups are too wide
            //============================================================================

            [[maybe_unused]] const bool merging_in_parallel = merging_in_parallel_default;

            //============================================================================
            // Make a grid where the grid-size is atleast fof_distance to ease the linking
            // For convenience with parallelization adjust so the grid is divided by ntasks
            //============================================================================

            int Ngrid = int(1.0 / fof_distance);
            Ngrid = Ngrid - Ngrid % FML::NTasks;
            if (Ngrid % FML::NTasks > Ngrid / 2)
                Ngrid *= 2;
            if (Ngrid > FoF_Ngrid_max)
                Ngrid = (FoF_Ngrid_max / FML::NTasks) * FML::NTasks;
            if (Ngrid < 3 * FML::NTasks)
                Ngrid = 3 * FML::NTasks;
            const int Local_nx = Ngrid / FML::NTasks;

#ifdef DEBUG_FOF
            if (FML::ThisTask == 0) {
                std::cout << "FriendsOfFriends linking\n";
                std::cout << "FoF Linking Distance: " << fof_distance << "\n";
                std::cout << "FoF Gridsize Ngrid = " << Ngrid << " Local: " << Local_nx << "\n";
            }
#endif
            assert(1.0 / Ngrid > fof_distance);

            //=============================================================================
            // Add particles to cells to speed up the linking below
            //=============================================================================
            std::vector<FoFCells> PartCells;
            ParticlesToCells<T, NDIM>(Ngrid, part, NumPart, PartCells);

            //=============================================================================
            // Do local FoF linking
            //=============================================================================
            std::vector<size_t> particle_id_FoF;
            FriendsOfFriendsLinkingLocal<T, NDIM>(
                part, NumPart, PartCells, Ngrid, Local_nx, fof_distance, particle_id_FoF, periodic);

            //=============================================================================
            // Link across boundaries
            // Index in part of the particles that have a link across the boundary on the right
            // and the index of the FoF group these belong to
            // isShared is the index of particles that has a link to boundary particles on the left
            //=============================================================================

            std::vector<size_t> BoundaryParticleIndex;
            std::vector<size_t> BoundaryParticleRightFoFIndex;
            std::vector<char> isShared(NumPart, 0);
#ifdef USE_MPI
            BoundaryLinking<T, NDIM>(fof_distance,
                                     part,
                                     NumPart,
                                     Ngrid,
                                     PartCells,
                                     particle_id_FoF,
                                     BoundaryParticleIndex,
                                     BoundaryParticleRightFoFIndex,
                                     isShared,
                                     periodic,
                                     merging_in_parallel);
#endif

            // Free memory no longer needed
            PartCells.clear();
            PartCells.shrink_to_fit();

            // Deal with the left boundary. First figure out FoFIDs of boundary particles
            std::vector<size_t> BoundaryParticleLeftFoFIndex;
            for (size_t i = 0; i < NumPart; i++) {
                if (isShared[i] == 1) {
                    BoundaryParticleLeftFoFIndex.push_back(particle_id_FoF[i]);
                }
            }

            // ...and make sure we only have unique FoFIDs
            std::vector<size_t>::iterator it;
            it = std::unique(BoundaryParticleLeftFoFIndex.begin(), BoundaryParticleLeftFoFIndex.end());
            BoundaryParticleLeftFoFIndex.resize(std::distance(BoundaryParticleLeftFoFIndex.begin(), it));

            //==========================================================
            // Count number of FoF groups on local task and the number
            // of particles in the group. Also record which FoF groups
            // that are shared across tasks
            //==========================================================

            // Sort the FoFIDs from small to large and count how many local FoF groups we have
            std::vector<size_t> tmp;
            for (auto & i : particle_id_FoF)
                if (i != no_FoF_ID)
                    tmp.push_back(i);
            std::sort(tmp.begin(), tmp.end(), [](const size_t & a, const size_t & b) -> bool { return a < b; });

            size_t curFoFID = tmp[0];
            size_t count = 0;
            size_t groupsize = 1;
            std::vector<size_t> FoFIndex;
            std::vector<size_t> ningroup;
            for (size_t i = 1; i < NumPart; i++) {
                if (tmp[i] == no_FoF_ID)
                    continue;
                if (tmp[i] == curFoFID and i != NumPart - 1) {
                    groupsize++;
                } else {
                    ningroup.push_back(groupsize);
                    FoFIndex.push_back(curFoFID);
                    curFoFID = tmp[i];
                    groupsize = 1;
                    count++;
                }
            }
            tmp.clear();
            tmp.shrink_to_fit();

            // Compute what are the shared groups
            std::vector<char> GroupIsShared(ningroup.size(), 0);
            size_t nshared_groups = 0;
            for (size_t i = 0; i < FoFIndex.size(); i++) {
                for (size_t j = 0; j < BoundaryParticleRightFoFIndex.size(); j++) {
                    if (BoundaryParticleRightFoFIndex[j] == FoFIndex[i]) {
                        GroupIsShared[i] = 1;
                        nshared_groups++;
                    }
                }
                for (size_t j = 0; j < BoundaryParticleLeftFoFIndex.size(); j++) {
                    if (BoundaryParticleLeftFoFIndex[j] == FoFIndex[i]) {
                        GroupIsShared[i] = 1;
                        nshared_groups++;
                    }
                }
            }
            BoundaryParticleLeftFoFIndex.clear();
            BoundaryParticleRightFoFIndex.clear();
            BoundaryParticleLeftFoFIndex.shrink_to_fit();
            BoundaryParticleRightFoFIndex.shrink_to_fit();

            // Count local non-shared groups
            int nnonshared = 0;
            std::vector<size_t> LocalNonSharedHalos;
            for (size_t i = 0; i < ningroup.size(); i++) {
                if (ningroup[i] >= size_t(nmin_FoF_group) and GroupIsShared[i] == 0) {
                    nnonshared++;
                    LocalNonSharedHalos.push_back(FoFIndex[i]);
                }
            }

            // Sum up total number of local non-shared groups
            int non_shared_total = nnonshared;

#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &non_shared_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
#ifdef DEBUG_FOF
            std::cout << FML::ThisTask << " has " << non_shared_total << " nonshared halos\n";
            std::cout << FML::ThisTask << " has " << count << " FoF groups before merging. The number of shared groups "
                      << nshared_groups << " Nlocal groups: "
                      << "\n";
            std::cout << FML::ThisTask << " has " << BoundaryParticleRightFoFIndex.size() << " boundary particles\n";
#endif

#ifdef USE_MPI
            // We are here is we have more than 1 task

            // Gather data to send
            std::vector<size_t> FoFIDSharedGroup;
            std::vector<size_t> ninSharedGroup;
            for (size_t i = 0; i < GroupIsShared.size(); i++) {
                if (GroupIsShared[i] == 1) {
                    FoFIDSharedGroup.push_back(FoFIndex[i]);
                    ninSharedGroup.push_back(ningroup[i]);
                }
            }
            GroupIsShared.clear();
            GroupIsShared.shrink_to_fit();

            // Gather data about shared groups
            std::vector<int> shared_groups_in_task(FML::NTasks, 0);
            shared_groups_in_task[FML::ThisTask] = ninSharedGroup.size();
            MPI_Allreduce(MPI_IN_PLACE, shared_groups_in_task.data(), FML::NTasks, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            std::vector<std::vector<size_t>> FoFIDSharedGroupFromOtherTasks(FML::NTasks);
            std::vector<std::vector<size_t>> ninSharedGroupFromOtherTasks(FML::NTasks);
            if (FML::ThisTask == 0) {
                FoFIDSharedGroupFromOtherTasks[0] = FoFIDSharedGroup;
                ninSharedGroupFromOtherTasks[0] = ninSharedGroup;
            }

            // Send from task i and recieve in task 0, the rest just sit quiet and wait
            for (int i = 1; i < FML::NTasks; i++) {
                const int bytes = sizeof(size_t) * shared_groups_in_task[i];
                MPI_Barrier(MPI_COMM_WORLD);
                if (FML::ThisTask == i) {
                    MPI_Send(FoFIDSharedGroup.data(), bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                    MPI_Send(ninSharedGroup.data(), bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                }
                if (FML::ThisTask == 0) {
                    FoFIDSharedGroupFromOtherTasks[i] = std::vector<size_t>(shared_groups_in_task[i]);
                    ninSharedGroupFromOtherTasks[i] = std::vector<size_t>(shared_groups_in_task[i]);
                    MPI_Status status;
                    MPI_Recv(FoFIDSharedGroupFromOtherTasks[i].data(), bytes, MPI_BYTE, i, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(ninSharedGroupFromOtherTasks[i].data(), bytes, MPI_BYTE, i, 0, MPI_COMM_WORLD, &status);
                }
            }

            // Count total number of halos
            std::vector<size_t> SharedHalos;
            std::vector<size_t> nSharedHalos;

            // This variable only correct on task 0
            int ntotal_halos = 0;
            if (FML::ThisTask == 0) {
                int nshared_halos = 0;

                // Make a map to count up the number of particles in each shared group
                std::map<size_t, size_t> groupmap;
                for (int i = 0; i < FML::NTasks; i++) {
                    for (int j = 0; j < shared_groups_in_task[i]; j++) {
                        size_t fofid = FoFIDSharedGroupFromOtherTasks[i][j];
                        size_t n = ninSharedGroupFromOtherTasks[i][j];
                        groupmap[fofid] += n;
                    }
                }

                // Go through the map and check if any is larger.
                for (auto & it : groupmap) {
                    if (it.second >= size_t(nmin_FoF_group)) {
#ifdef DEBUG_FOF
                        std::cout << "Shared group " << it.first << " has " << it.second << " particles\n";
#endif
                        SharedHalos.push_back(it.first);
                        nSharedHalos.push_back(it.second);
                        nshared_halos++;
                    }
                }
#ifdef DEBUG_FOF
                std::cout << "We have " << nshared_halos << " acceptable shared halos for a total of "
                          << nshared_halos + non_shared_total << "\n";
#endif

                ntotal_halos = nshared_halos + non_shared_total;
            }
            FoFIDSharedGroupFromOtherTasks.clear();
            FoFIDSharedGroupFromOtherTasks.shrink_to_fit();

            // Communicate number of groups to all tasks
            int nsharedhalos = SharedHalos.size();
            MPI_Bcast(&nsharedhalos, 1, MPI_INT, 0, MPI_COMM_WORLD);
            SharedHalos.resize(nsharedhalos);
            MPI_Bcast(SharedHalos.data(), sizeof(size_t) * nsharedhalos, MPI_BYTE, 0, MPI_COMM_WORLD);

            // Figure out which shared halos we have a part of on current task
            std::vector<size_t> LocalSharedHalos;
            for (int i = 0; i < nsharedhalos; i++) {
                for (auto & id : FoFIDSharedGroup)
                    if (id == SharedHalos[i]) {
                        LocalSharedHalos.push_back(id);
                    }
            }
            SharedHalos.clear();
            SharedHalos.shrink_to_fit();
#else
            // No MPI no shared halos
            std::vector<size_t> LocalSharedHalos;
            size_t ntotal_halos = non_shared_total;
#endif

            //=============================================================================
            // Create the halos (here we will have redunancies as we havent merged shared halos yet)
            //=============================================================================

            std::vector<FoFHaloClass> halos;
            halos.reserve(LocalSharedHalos.size() + LocalNonSharedHalos.size());

            // Process non-shared halos
            for (auto & id : LocalNonSharedHalos) {
                const bool shared = false;
                FoFHaloClass g(id, shared);
                for (size_t i = 0; i < NumPart; i++) {
                    if (particle_id_FoF[i] == id) {
                        g.add(part[i], periodic);
                    }
                }
                halos.push_back(g);
            }

            // Process shared halos
            for (auto & id : LocalSharedHalos) {
                const bool shared = true;
                FoFHaloClass g(id, shared);
                for (size_t i = 0; i < NumPart; i++) {
                    if (particle_id_FoF[i] == id) {
                        g.add(part[i], periodic);
                    }
                }
                halos.push_back(g);
            }
            LocalSharedHalos.clear();
            LocalNonSharedHalos.clear();
            LocalSharedHalos.shrink_to_fit();
            LocalNonSharedHalos.shrink_to_fit();

            // Send halos to task 0 which merges and returns
            std::vector<FoFHaloClass> myhalos = halos;
#ifdef USE_MPI
            std::vector<int> nhalosontask(FML::NTasks, 0);
            nhalosontask[FML::ThisTask] = halos.size();
            MPI_Allreduce(MPI_IN_PLACE, nhalosontask.data(), FML::NTasks, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            for (int i = 1; i < FML::NTasks; i++) {
                if (FML::ThisTask == i) {
                    MPI_Send(halos.data(), sizeof(FoFHaloClass) * nhalosontask[i], MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                } else if (FML::ThisTask == 0) {
                    std::vector<FoFHaloClass> tmp(nhalosontask[i]);
                    MPI_Status status;
                    MPI_Recv(
                        tmp.data(), sizeof(FoFHaloClass) * nhalosontask[i], MPI_BYTE, i, 0, MPI_COMM_WORLD, &status);
                    for (auto & g : tmp) {
                        myhalos.push_back(g);
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
#endif

            // Merge the halos
            if (FML::ThisTask == 0) {
                for (auto & g : myhalos) {
                    if (g.np == 0)
                        continue;
                    g.merged = true;
                    for (auto & gg : myhalos) {
                        if (gg.np == 0)
                            continue;
                        if (gg.id == g.id and !gg.merged) {
                            g.merge(gg, periodic);
                            gg.merged = true;
                        }
                    }
                }

                LocalFoFGroups.reserve(ntotal_halos);
                for (auto & g : myhalos) {
                    if (g.np > 0) {
                        LocalFoFGroups.push_back(g);
                    }
                }
            }

            // We are now done. Task 0 has all halos. But we should in principle send the shared halos back to where
            // they belong so all tasks just has the halos they are in charge of, i.e. the ones that fall into their
            // domain
        }
    } // namespace FOF
} // namespace FML
#endif
