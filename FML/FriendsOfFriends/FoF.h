#ifndef FOF_HEADER
#define FOF_HEADER
#include <FML/FriendsOfFriends/FoFBinning.h>
#include <FML/Global/Global.h>
#include <FML/Timing/Timings.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <map>

namespace FML {

    //========================================================================================
    /// This namespace deals how to link together particles in groups
    //========================================================================================
    namespace FOF {

        // Memory for particlegrid. Allocated in ParticlesToCells and freed in FriendsOfFriends
        size_t * particlegriddata{nullptr};

        /// If a particle belongs to no FoF group it is given this FoF ID
        const constexpr size_t no_FoF_ID = std::numeric_limits<size_t>::max();

        /// Type of particle for book-keeping
        enum ParticleType { BOUNDARY_PARTICLE = -1, REGULAR_PARTICLE = 0 };

        /// The maximum grid we are allowed to use to bin the particle to
        /// This is just to have a memory limit in what the algorithm is
        /// allowed to allocate. This value is used if user do not provide it.
        int FoF_Ngrid_max = 1024;

        using Timings = FML::UTILS::Timings;
        /// Records timings of the FoF search
        Timings foftimer;

        //========================================================================================
        /// Locate friends of friends groups and bin what you want over these groups.
        ///
        /// @tparam T The particle class
        /// @tparam NDIM The number of dimensions
        /// @tparam FoFHaloClass The FoF group class, determined what to be binned up over the FoF group
        ///
        /// @param[in] part Pointer to particles. If the particle has a set_fofid then we will also store the FoFID in
        /// the particles.
        /// @param[in] NumPart Number of local particles
        /// @param[in] linking_length The distance in units of mean particle seperation (in [0,1)) required to have a
        /// link between two particles. Typical value is 0.2.
        /// @param[in] nmin_FoF_group Minimum number of particles in a FoF group to store it. Typical value is 20.
        /// @param[in] periodic Is the box periodic?
        /// @param[in] Buffersize_over_Boxsize The buffer we communicate on both sides of the domain in units of the
        /// boxsize. Typical value for N-body simulations is (3 Mpc/h) / boxsize.
        /// @param[out] LocalFoFGroups The results: list of FoF groups.
        /// @param[in] Ngrid (Optional) The maximum gridsize we use to bin the particles to in order to speed up the
        /// calculation. The default FoF_Ngrid_max is set in the header. Larger value means more memory needed.
        ///
        //========================================================================================
        template <class T, int NDIM, class FoFHaloClass = FoFHalo<T, NDIM>>
        void FriendsOfFriends(T * part,
                              size_t NumPart,
                              double linking_length,
                              int nmin_FoF_group,
                              bool periodic,
                              double Buffersize_over_Boxsize,
                              std::vector<FoFHaloClass> & LocalFoFGroups,
                              int Ngrid = FoF_Ngrid_max);

        //=========================================================================
        /// A gridcell in a grid used for speeding up the linking of particles
        /// The ID of the particles is their position in the particle list
        //=========================================================================
        class FoFCell {
          public:
            /// Number of particles in the cell
            int np{0};
            /// Number of boundary particles in the cell
            int np_boundary{0};
            /// List of indices of particles that are in the cell
            size_t * ParticleIndex;
            /// List of indices of particles that are in the cell
            size_t * ParticleIndexBoundary;
            /// List of indices of boundary particles that are in the cell
            FoFCell() = default;
        };

        //=========================================================================
        /// Internal method: bin particles to a grid.
        /// NB: this method assumes the boundary particles are *unwrapped*
        /// when we run with a periodic box.
        /// i.e. on task 1 they should have x<0 (-0.1, not 0.9 if inside the box)
        /// and on task N-1 they should have x>1 (1.1, not 0.1 if inside the box)
        //=========================================================================
        template <class T, int NDIM>
        void ParticlesToCells(int Ngrid,
                              int & Local_nx,
                              double dx_boundary,
                              T * part,
                              size_t NumPart,
                              T * part_boundary,
                              size_t NumPart_boundary,
                              std::vector<FoFCell> & PartCells) {

            // Total length of domain
            const double dx_cells = 1.0 / Ngrid;
            const double xmin_box = FML::xmin_domain - dx_boundary;
            const double xmax_box = FML::xmax_domain + dx_boundary;
            const double dx_box_domain = xmax_box - xmin_box;

            // Total number of local-cells
            Local_nx = std::ceil(dx_box_domain / dx_cells);

            // Total number of local cells to allocate
            const size_t NgridTot = size_t(Local_nx) * FML::power(Ngrid, NDIM - 1);

            // Make a grid containing particles
            PartCells.resize(NgridTot);

            // Get index of the cell the particle lives in
            auto get_cell_index = [&](size_t ipart, std::vector<int> & coord, int type) {
                decltype(part[0].get_pos()) pos;
                if (type == BOUNDARY_PARTICLE)
                    pos = FML::PARTICLE::GetPos(part_boundary[ipart]);
                else if (type == REGULAR_PARTICLE)
                    pos = FML::PARTICLE::GetPos(part[ipart]);
                else {
                    throw std::runtime_error("FoF::The particle is somehow marked as not being left/regular/right! "
                                             "Should not happen ever!\n");
                }

                size_t index_cell = 0;
                for (int idim = 0; idim < NDIM; idim++) {
                    auto dx = (pos[idim] - xmin_box * (idim == 0 ? 1 : 0));
                    coord[idim] = int(dx * Ngrid);
                    index_cell = index_cell * Ngrid + coord[idim];
                }
                return index_cell;
            };

            // Count number of particles in cells
            std::vector<int> coord(NDIM);
            for (size_t i = 0; i < NumPart; i++) {
                const size_t index_cell = get_cell_index(i, coord, REGULAR_PARTICLE);
                PartCells[index_cell].np++;
            }
            for (size_t i = 0; i < NumPart_boundary; i++) {
                const size_t index_cell = get_cell_index(i, coord, BOUNDARY_PARTICLE);
                PartCells[index_cell].np_boundary++;
            }

            // Allocate all the data
            particlegriddata = new size_t[NumPart_boundary + NumPart];
            size_t * ptr = &particlegriddata[0];

            // Allocate cells
            int np_max = 0;
            size_t np_total = 0;
            size_t np_boundary_total = 0;
            for (auto & cell : PartCells) {
                cell.ParticleIndexBoundary = ptr;
                ptr += cell.np_boundary;
                cell.ParticleIndex = ptr;
                ptr += cell.np;

                np_max = std::max(np_max, cell.np);
                np_total += cell.np;
                np_boundary_total += cell.np_boundary;
                cell.np = 0;
                cell.np_boundary = 0;
            }

            // Fill up cells with info about particles
            for (size_t i = 0; i < NumPart; i++) {
                const size_t index_cell = get_cell_index(i, coord, 0);
                auto & np = PartCells[index_cell].np;
                PartCells[index_cell].ParticleIndex[np] = i;
                np++;
            }
            for (size_t i = 0; i < NumPart_boundary; i++) {
                const size_t index_cell = get_cell_index(i, coord, -1);
                auto & np_boundary = PartCells[index_cell].np_boundary;
                PartCells[index_cell].ParticleIndexBoundary[np_boundary] = i;
                np_boundary++;
            }
        }

        template <class T, int NDIM, class FoFHaloClass = FoFHalo<T, NDIM>>
        void FriendsOfFriends(T * part,
                              size_t NumPart,
                              double linking_length,
                              int nmin_FoF_group,
                              bool periodic,
                              double Buffersize_over_Boxsize,
                              std::vector<FoFHaloClass> & LocalFoFGroups,
                              int Ngrid) {

            LocalFoFGroups.clear();
            if (NumPart == 0)
                return;
            const bool debug = false;

            // Some basic checks
            assert_mpi(FML::PARTICLE::GetNDIM(T()) == NDIM,
                       "FriendsOfFriends::The dimensions of particle and method do not match!\n");

            // Compute fof search distance
            size_t NumPart_tot = NumPart;
            FML::SumOverTasks(&NumPart_tot);
            const double mean_particle_separation = 1.0 / std::pow(NumPart_tot, 1.0 / 3.0);
            const double fof_distance = linking_length * mean_particle_separation;
            const double fof_distance2 = fof_distance * fof_distance;

            // Compute grid-size for faster linking
            if(Ngrid == 0) Ngrid = FoF_Ngrid_max;
            Ngrid = std::min(int(0.5 / fof_distance), Ngrid);

            // Set how the size of the grid-cube we need to search
            const constexpr int twotondim = FML::power(2, NDIM);

            // Check that the grid is not too small
            assert_mpi(1.0 / Ngrid > 2 * fof_distance,
                       "FriendsOfFriends::The gridsize is too large (larger than the 2x linking distance)");

            // Sort particles by x position
            // This speed it up when doing the linking, typically by ~20-30%
            foftimer.StartTiming("Sort particles");
            std::sort(part, part + NumPart, [](const T & a, const T & b) {
                // The y axis seems fastest in benchmarks
                constexpr int axis = 1;
                T * f = const_cast<T *>(&a);
                T * g = const_cast<T *>(&b);
                const auto pos_a = FML::PARTICLE::GetPos(*f);
                const auto pos_b = FML::PARTICLE::GetPos(*g);
                return pos_b[axis] > pos_a[axis];
            });
            foftimer.EndTiming("Sort particles");

            // Compute ranges for which to copy over boundary partices
            const double xmin = FML::xmin_domain;
            const double xmax = FML::xmax_domain;
            const double dx_boundary = FML::NTasks == 1 ? 0.0 : Buffersize_over_Boxsize;

            // If the boundary we ask for is too large
            if (dx_boundary > xmax - xmin) {
                throw std::runtime_error("FoF::The boundary asked for is larger than the neighbor task. Reduce "
                                         "boundary (or the number of CPUs)\n");
            }

            //=========================================================================
            // Fetch all the boundary particles
            // We gather particles within dx_boundary of each side
            //=========================================================================
            std::vector<T> boundary_particles{};
            const size_t bytes_per_particle = FML::PARTICLE::GetSize(part[0]);
#ifdef USE_MPI
            foftimer.StartTiming("Communicate boundary");
            if (FML::NTasks > 1) {
                // Count how may particles to send left and right
                size_t n_send_left = 0;
                size_t n_send_right = 0;
                size_t noutside = 0;
                for (size_t i = 0; i < NumPart; i++) {
                    const auto * pos = FML::PARTICLE::GetPos(part[i]);
                    if (pos[0] < xmin + dx_boundary)
                        ++n_send_left;
                    if (pos[0] > xmax - dx_boundary)
                        ++n_send_right;
                    if (pos[0] < xmin or pos[0] > xmax)
                        noutside++;
                }
                if (noutside > 0)
                    throw std::runtime_error("FoF::Particle detected outside local domain, please ensure all particles "
                                             "have been moved inside domain\n");

                // The MPI id of the left and right task
                MPI_Status status;
                const int RightTask = (FML::ThisTask + 1) % FML::NTasks;
                const int LeftTask = (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks;

                // Communicate how many to send and get how many to recieve
                size_t n_recv_left = 0;
                size_t n_recv_right = 0;

                // Send left recieve from right
                MPI_Sendrecv(&n_send_left,
                             sizeof(n_send_left),
                             MPI_BYTE,
                             LeftTask,
                             0,
                             &n_recv_right,
                             sizeof(n_recv_right),
                             MPI_BYTE,
                             RightTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);
                // Send right recieve from left
                MPI_Sendrecv(&n_send_right,
                             sizeof(n_send_right),
                             MPI_BYTE,
                             RightTask,
                             0,
                             &n_recv_left,
                             sizeof(n_recv_left),
                             MPI_BYTE,
                             LeftTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);

                // If not periodic then left boundary on first task and right boundary on last task is not there
                if (not periodic) {
                    if (FML::ThisTask == 0)
                        n_recv_left = n_send_left = 0;
                    if (FML::ThisTask == FML::NTasks - 1)
                        n_recv_right = n_send_right = 0;
                }

                // Debug
                if (debug) {
                    std::cout << "Task " << FML::ThisTask << " will send " << n_send_right << " to   " << RightTask
                              << "\n";
                    std::cout << "Task " << FML::ThisTask << " will send " << n_send_left << " to   " << LeftTask
                              << "\n";
                    std::cout << "Task " << FML::ThisTask << " will recv " << n_recv_right << " from " << RightTask
                              << "\n";
                    std::cout << "Task " << FML::ThisTask << " will recv " << n_recv_left << " from " << LeftTask
                              << "\n";
                    std::cout << "Send fraction " << (n_send_left + n_send_right) / double(NumPart) << "\n";
                }

                // Compute how many bytes to send/recv
                const size_t bytes_send_left = bytes_per_particle * n_send_left;
                const size_t bytes_send_right = bytes_per_particle * n_send_right;
                const size_t bytes_recv_left = bytes_per_particle * n_recv_left;
                const size_t bytes_recv_right = bytes_per_particle * n_recv_right;

                // Set up communication buffer
                std::vector<char> CommBufferSend;
                std::vector<char> CommBufferRecv;
                CommBufferSend.resize(std::max(bytes_send_left, bytes_send_right));
                CommBufferRecv.resize(std::max(bytes_recv_left, bytes_recv_right));

                // Set up storage for boundary particles
                boundary_particles.resize(n_recv_left + n_recv_right);

                // Gather data to send left
                char * buffer = CommBufferSend.data();
                if (n_send_left > 0)
                    for (size_t i = 0; i < NumPart; i++) {
                        const auto * pos = FML::PARTICLE::GetPos(part[i]);
                        if (pos[0] < xmin + dx_boundary) {
                            FML::PARTICLE::AppendToBuffer(part[i], buffer);
                            buffer += bytes_per_particle;
                        }
                    }

                // Send left recieve from right
                MPI_Sendrecv(CommBufferSend.data(),
                             bytes_send_left,
                             MPI_BYTE,
                             LeftTask,
                             0,
                             CommBufferRecv.data(),
                             bytes_recv_right,
                             MPI_BYTE,
                             RightTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);

                // Generate particles
                buffer = CommBufferRecv.data();
                for (size_t i = 0; i < n_recv_right; i++) {
                    FML::PARTICLE::AssignFromBuffer(boundary_particles[i], buffer);
                    buffer += bytes_per_particle;
                }

                // Gather data to send right
                buffer = CommBufferSend.data();
                if (n_send_right > 0)
                    for (size_t i = 0; i < NumPart; i++) {
                        const auto * pos = FML::PARTICLE::GetPos(part[i]);
                        if (pos[0] > xmax - dx_boundary) {
                            FML::PARTICLE::AppendToBuffer(part[i], buffer);
                            buffer += bytes_per_particle;
                        }
                    }

                // Send right recieve from left
                MPI_Sendrecv(CommBufferSend.data(),
                             bytes_send_right,
                             MPI_BYTE,
                             RightTask,
                             0,
                             CommBufferRecv.data(),
                             bytes_recv_left,
                             MPI_BYTE,
                             LeftTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);

                // Generate particles
                buffer = CommBufferRecv.data();
                for (size_t i = 0; i < n_recv_left; i++) {
                    FML::PARTICLE::AssignFromBuffer(boundary_particles[n_recv_right + i], buffer);
                    buffer += bytes_per_particle;
                }

                // Free up memory used for communication
                CommBufferSend = std::vector<char>();
                CommBufferRecv = std::vector<char>();

                // Make sure the left particles are wrapped to the left of the box on first task
                // If we don't do this then we need to modify dx[0] comp in get_cell_index in ParticlesInGrid
                // to wrap + the same in FindAllFriends
                if (FML::ThisTask == 0) {
                    for (size_t i = 0; i < n_recv_left; i++) {
                        auto * pos = FML::PARTICLE::GetPos(boundary_particles[n_recv_right + i]);
                        pos[0] -= 1.0;
                    }
                }

                // Make sure the right particles are wrapped to the right of the box on last task
                if (FML::ThisTask == FML::NTasks - 1) {
                    for (size_t i = 0; i < n_recv_right; i++) {
                        auto * pos = FML::PARTICLE::GetPos(boundary_particles[i]);
                        pos[0] += 1.0;
                    }
                }
            }
            foftimer.EndTiming("Communicate boundary", FML::ThisTask == 0);
#endif

            const auto NumPart_boundary = boundary_particles.size();
            const auto NumPart_total = NumPart + NumPart_boundary;

            //=========================================================================
            // Bin particles to a grid for faster linking
            //=========================================================================
            std::vector<FoFCell> PartCells;
            int Local_nx{};
            foftimer.StartTiming("ParticlesToCells");
            ParticlesToCells<T, NDIM>(Ngrid,
                                      Local_nx,
                                      dx_boundary,
                                      part,
                                      NumPart,
                                      boundary_particles.data(),
                                      boundary_particles.size(),
                                      PartCells);
            foftimer.EndTiming("ParticlesToCells");

            //=========================================================================
            // We now have all the boundary particles and can do the FoF linking
            //=========================================================================
            if (FML::ThisTask == 0) {
                std::cout << "\n";
                std::cout << "#=====================================================\n";
                std::cout << "#\n";
                std::cout << "#            ___________________  ___________     \n";
                std::cout << "#            \\_   _____/\\_____  \\ \\_   _____/ \n";
                std::cout << "#             |    __)   /   |   \\ |    __)      \n";
                std::cout << "#             |     \\   /    |    \\|     \\     \n";
                std::cout << "#             \\___  /   \\_______  /\\___  /     \n";
                std::cout << "#                 \\/            \\/     \\/      \n";
                std::cout << "#\n";
                std::cout << "# FriendsOfFriends linking\n";
                std::cout << "# FoF Linking Length: " << linking_length << "\n";
                std::cout << "# FoF Linking Distance: " << fof_distance << "\n";
                std::cout << "# dx_boundary / Boxsize: " << Buffersize_over_Boxsize << "\n";
                std::cout << "# Periodic box: " << std::boolalpha << periodic << "\n";
                std::cout << "# FoF linking Gridsize = " << Ngrid << " Local_nx: " << Local_nx << "\n";
                std::cout << "# Maximum possible FoF gridsize = " << int(0.5 / fof_distance) << "\n";
                std::cout << "# Npart_boundary : " << NumPart_boundary << " ( "
                          << NumPart_boundary / std::pow(1024.0, 2) * bytes_per_particle << " MB )\n";
                std::cout << "# Grid memory per task: "
                          << (double(Local_nx) * FML::power(Ngrid, NDIM - 1) * sizeof(FoFCell) +
                              NumPart_total * sizeof(size_t)) /
                                 std::pow(1024.0, 2)
                          << " MB\n";
                std::cout << "# ...compared to particle memory: " << NumPart * bytes_per_particle / std::pow(1024.0, 2)
                          << " MB\n";
                std::cout << "#\n";
                std::cout << "#=====================================================\n";
                std::cout << "\n";
            }

            //=========================================================================
            // Link together all particles in the local domain that are located within
            // a linkking distance of linking_length times the mean particle separation.
            // The result is a list of all the particles and an associated FoF group ID
            // This ID can be arbitrary but is guaranteed to be unique across tasks
            //=========================================================================

            // Allocate the FoF id for each particle
            auto particle_id_FoF = std::vector<size_t>(NumPart + NumPart_boundary, no_FoF_ID);

            // Picks out the particle
            auto type_from_globalindex = [&](size_t globalindex) {
                if (globalindex < NumPart_boundary)
                    return int(BOUNDARY_PARTICLE);
                return int(REGULAR_PARTICLE);
            };
            auto localindex_from_globalindex = [&](size_t globalindex) {
                if (globalindex < NumPart_boundary)
                    return globalindex;
                return globalindex - NumPart_boundary;
            };
            auto globalindex_from_localindex_type = [&](size_t localindex, int type) {
                if (type == BOUNDARY_PARTICLE)
                    return localindex;
                return localindex + NumPart_boundary;
            };
            auto particle_from_localindex_and_type = [&](size_t localindex, int type) {
                if (type == BOUNDARY_PARTICLE)
                    return &boundary_particles[localindex];
                return &part[localindex];
            };

            // Reserve space for 1000 particles in a FoF group
            std::vector<size_t> friend_local_index_list;
            std::vector<signed char> friend_type_list;
            friend_local_index_list.reserve(1000);
            friend_type_list.reserve(1000);

            // Function to locate and tag all closest friends
            // This is then used recursively in the next method we have
            auto FindAllFriends = [&](T * curpart,
                                      [[maybe_unused]] int type,
                                      [[maybe_unused]] size_t localindex,
                                      size_t globalindex,
                                      size_t FoFID,
                                      std::vector<size_t> & friend_local_index_list,
                                      std::vector<signed char> & friend_type_list) {
                // Get the particle
                const auto * pos1 = FML::PARTICLE::GetPos(*curpart);

                // Compute coord of cell particle is in
                std::array<int, NDIM> coord;
                std::array<int, NDIM> di;
                for (int idim = 0; idim < NDIM; idim++) {
                    double ix = (pos1[idim] - (xmin - dx_boundary) * (idim == 0 ? 1 : 0)) * Ngrid;
                    coord[idim] = int(ix);
                    di[idim] = (ix - coord[idim] > 0.5) ? +1 : -1;
                }

                // Loop through all 2^NDIM nbor cells
                // We only go to the cells on the same side as the particle position
                // As long as the cell-size is larger than 2*fof_distance we don't need
                // to visit the other cells
                std::array<int, NDIM> icoord;
                for (int nbcell = 0; nbcell < twotondim; nbcell++) {
                    for (int idim = 0, n = 1; idim < NDIM; idim++, n *= 2) {
                        int go_left_right_or_stay = (nbcell / n % 2);
                        icoord[idim] = coord[idim] + go_left_right_or_stay * di[idim];
                    }

                    // For boundary cells in the x direction we don't have a left or a right nbor
                    // unless we only have 1 task
                    if (FML::NTasks > 1) {
                        if (icoord[0] < 0 or icoord[0] >= Local_nx)
                            continue;
                    } else {
                        if (icoord[0] < 0)
                            icoord[0] += Ngrid;
                        if (icoord[0] >= Ngrid)
                            icoord[0] -= Ngrid;
                    }

                    // Compute cell-index or nbor cell
                    bool skip = false;
                    size_t index_nbor_cell = icoord[0];
                    for (int idim = 1; idim < NDIM; idim++) {
                        // Periodic boundary conditions
                        if (periodic) {
                            if (icoord[idim] < 0)
                                icoord[idim] += Ngrid;
                            if (icoord[idim] >= Ngrid)
                                icoord[idim] -= Ngrid;
                        } else {
                            if (icoord[idim] < 0 or icoord[idim] >= Ngrid){
                                skip = true;
                                break;
                            }
                        }
                        index_nbor_cell = index_nbor_cell * Ngrid + icoord[idim];
                    }

                    // Cell do not exist in this case
                    if (skip)
                        continue;

                    // Get total number of particles in cell
                    const size_t np_boundary = PartCells[index_nbor_cell].np_boundary;
                    const size_t np = PartCells[index_nbor_cell].np;
                    const size_t np_total = np + np_boundary;

                    // Loop over all particles in nbor cell
                    for (size_t ii = 0; ii < np_total; ii++) {
                        T * nborpart{};
                        int nbortype{};
                        size_t nborglobalindex{};
                        size_t nborlocalindex{};

                        // Type of particle
                        if (ii < np_boundary) {
                            nborlocalindex = PartCells[index_nbor_cell].ParticleIndexBoundary[ii];
                            nborglobalindex = nborlocalindex;
                            nborpart = &boundary_particles[nborlocalindex];
                            nbortype = BOUNDARY_PARTICLE;
                        } else {
                            nborlocalindex = PartCells[index_nbor_cell].ParticleIndex[ii - np_boundary];
                            nborglobalindex = nborlocalindex + NumPart_boundary;
                            nborpart = &part[nborlocalindex];
                            nbortype = REGULAR_PARTICLE;
                        }

                        // If same particle or already processed so that it has an ID skip going further
                        if (particle_id_FoF[nborglobalindex] != no_FoF_ID)
                            continue;
                        if (nborglobalindex == globalindex)
                            continue;

                        // Compute distance
                        double dist2 = 0.0;
                        const auto * pos2 = FML::PARTICLE::GetPos(*nborpart);
                        std::array<double, NDIM> dx2;
                        for (int idim = 0; idim < NDIM; idim++) {
                            dx2[idim] = std::abs(pos2[idim] - pos1[idim]);
                            if (dx2[idim] > 0.5 and periodic)
                                dx2[idim] -= 1.0;
                        }
                        for (int idim = 0; idim < NDIM; idim++)
                            dist2 += dx2[idim] * dx2[idim];

                        // Check if we have a link. Set the FoF ID in the particle list
                        if (dist2 < fof_distance2) {
                            particle_id_FoF[nborglobalindex] = FoFID;
                            // Only need to set this one time so avoid a write by adding if test
                            if (particle_id_FoF[globalindex] == no_FoF_ID)
                                particle_id_FoF[globalindex] = FoFID;
                            // Push partice info to FoF list
                            friend_local_index_list.push_back(nborlocalindex);
                            friend_type_list.push_back(nbortype);
                        }
                    }
                }
            };

            // ID of the FoF groups
            unsigned int FoFID = 0;

            // Recursively find all friends of friends
            foftimer.StartTiming("FoFLinking");
            for (size_t i = 0; i < NumPart_total; i++) {

                // Show progress bar
                if (FML::ThisTask == 0)
                    if ((10 * i) / NumPart_total != (10 * i + 10) / NumPart_total or i == 0) {
                        std::cout << (100 * i + 100) / NumPart_total << "% " << std::flush;
                        if (i == NumPart_total - 1) {
                            std::cout << std::endl;
                        }
                    }

                // Fetch the particle
                size_t globalindex = i;
                size_t localindex = localindex_from_globalindex(globalindex);
                int type = type_from_globalindex(globalindex);
                T * curpart = particle_from_localindex_and_type(localindex, type);

                // Check if particle is not already processed
                if (particle_id_FoF[i] == no_FoF_ID) {
                    // Start new FoF group
                    friend_local_index_list.clear();
                    friend_type_list.clear();

                    // Do linking
                    FindAllFriends(
                        curpart, type, localindex, globalindex, FoFID, friend_local_index_list, friend_type_list);

                    if (friend_local_index_list.size() == 0)
                        continue;

                    // We are here if we found a group (2 or more particles)
                    // Go through all friends and the friend of these friends
                    // until we have found all particles in the halo
                    FoFHalo<T, NDIM> newhalo(FoFID);
                    newhalo.add(*curpart, periodic);

                    while (friend_local_index_list.size() > 0) {
                        // Fetch a particle
                        auto localindex = friend_local_index_list.back();
                        type = int(friend_type_list.back());
                        globalindex = globalindex_from_localindex_type(localindex, type);
                        curpart = particle_from_localindex_and_type(localindex, type);

                        // Fetch position relative to first particle
                        newhalo.add(*curpart, periodic);

                        // Remove particle from list
                        friend_local_index_list.pop_back();
                        friend_type_list.pop_back();

                        // Do linking
                        FindAllFriends(
                            curpart, type, localindex, globalindex, FoFID, friend_local_index_list, friend_type_list);
                    }

                    // Save halo if it lies in the local domain and has more than nmin particles
                    if (int(newhalo.np) >= nmin_FoF_group) {
                        double xhalo = newhalo.get_pos()[0];
                        if (periodic) // Periodic wrap
                            xhalo += (xhalo < 0.0) ? 1.0 : ((xhalo >= 1.0) ? -1.0 : 0.0);
                        if (xhalo >= FML::xmin_domain and xhalo < FML::xmax_domain) {
                            LocalFoFGroups.push_back(newhalo);
                        }
                    }

                    // Increase the FoF ID
                    FoFID++;
                }
            }
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            foftimer.EndTiming("FoFLinking");

            // Grid no longer needed, free up memory
            delete[] particlegriddata;

            // Show timings
            if (FML::ThisTask == 0)
                foftimer.PrintAllTimings();

            // Print how many particles we found
            auto nhalos = LocalFoFGroups.size();
            FML::SumOverTasks(&nhalos);
            if (FML::ThisTask == 0)
                std::cout << "# We found a total of " << nhalos << " (" << LocalFoFGroups.size()
                          << " on task 0) "
                             "\n";

            // We need to fix the FoFID of the halos as they start from 0 on each task
            // so that all just have a unique ID
            // We do this such that after mpi-sum we have the sum of the FoFIDs
            std::vector<int> addfofid(FML::NTasks, 0);
            for (int i = FML::ThisTask + 1; i < FML::NTasks; i++)
                addfofid[i] = FoFID;
#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, addfofid.data(), NTasks, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
     
            // Finalize computation of halos
            auto add = addfofid[FML::ThisTask];
            for (auto & h : LocalFoFGroups) {
                h.id += add;
                h.finalize(periodic);
            }

            // If particles have a set_fofid method then set the ID in the particles
            // This sets it to no_FoF_id if the particle is not part of a group
            if constexpr (FML::PARTICLE::has_set_fofid<T>()) {
                for (size_t i = 0; i < NumPart_total; i++) {
                    auto FoFID = particle_id_FoF[i] + addfofid[FML::ThisTask];
                    if (type_from_globalindex(i) == BOUNDARY_PARTICLE)
                        FML::PARTICLE::SetFoFID(boundary_particles[localindex_from_globalindex(i)], FoFID);
                    else if (type_from_globalindex(i) == REGULAR_PARTICLE)
                        FML::PARTICLE::SetFoFID(part[localindex_from_globalindex(i)], FoFID);
                }
                // TODO - id should be communicated back for the boundary particles
            }

            // Boundary particles are finally freed here
        }
    } // namespace FOF
} // namespace FML
#endif
