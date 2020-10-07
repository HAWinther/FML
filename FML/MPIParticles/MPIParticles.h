#ifndef MPIPARTICLES_HEADER
#define MPIPARTICLES_HEADER

#include <cassert>
#include <cstdio>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

namespace FML {

    /// This namespace contains things related to particles: different types, containers for holding shared particles
    /// across tasks and algorithms for doing stuff with particles.
    namespace PARTICLE {

        //===========================================================
        ///
        /// A container class for holding particles that are distributed
        /// across many CPUs and to easily deal with communication
        /// of these particles if they cross the domain boundary (simply
        /// call communicate_particles() at any time to do this)
        ///
        /// Contains methods for setting up MPIParticles from a set of particles
        /// or creating regular grids of particles and so on
        ///
        /// Templated on particle class. Particle must at a minimum have the methods
        ///
        ///    get_pos()            : Ptr to position
        ///
        ///    int get_ndim()       : How many dimensions pos have
        ///
        /// If the particle has any dynamical allocations you must also provide (these are automatically provided
        /// otherwise):
        ///
        ///    get_particle_byte_size()   : How many bytes does the particle store
        ///
        ///    append_to_buffer(char *)   : append all the particle data to a char array moving buffer forward as we
        ///    read
        ///
        ///    assign_from_buffer(char *) : assign data to a particle after it has been recieved moving buffer forward
        ///    as we do this
        ///
        /// Compile time defines:
        ///
        ///    DEBUG_MPIPARTICLES : Show some info when running
        ///
        /// External variables/methods we rely on:
        ///
        ///    int ThisTask;
        ///
        ///    int NTasks;
        ///
        ///    assert_mpi(Expr, Msg)
        ///
        ///    T power(T base, int exponent);
        ///
        //===========================================================

        template <class T>
        class MPIParticles {
          private:
            // Particle container
            // std::vector<T> p;
            Vector<T> p{};

            // Info about the particle distribution
            size_t NpartTotal{0};        // Total number of particles across all tasks
            size_t NpartLocal_in_use{0}; // Number of particles in use (total allocated given by p.size())

            // The range of x-values in [0,1] that belongs to each task
            std::vector<double> x_min_per_task{};
            std::vector<double> x_max_per_task{};

            // If we create particles from scratch according to a grid
            double buffer_factor{1.0}; // Allocate a factor of this more particles than needed

            // If we created a uniform grid of particles
            int Npart_1D{0};       // Total number of particles slices
            int Local_Npart_1D{0}; // Number of slices in current task
            int Local_p_start{0};  // Start index of local particle slice in [0,Npart_1D)

            // Swap particles
            void swap_particles(T & a, T & b);

            // The byte-size of particle ipart
            size_t get_particle_byte_size(size_t ipart);

            // For communication
            void copy_over_recieved_data(std::vector<char> & recv_buffer, size_t Npart_recieved);

          public:
            /// Iterator for looping through all the active particles i.e. allow for(auto &&p: mpiparticles)
            class iterator {
              public:
                iterator(T * ptr) : ptr(ptr) {}
                iterator operator++() {
                    ++ptr;
                    return *this;
                }
                bool operator!=(const iterator & other) { return ptr != other.ptr; }
                T & operator*() { return *ptr; }

              private:
                T * ptr;
            };

            /// Iterator: points to the first local particle
            iterator begin() { return iterator(p.data()); }
            /// Iterator: points to one past the last active local particle
            iterator end() { return iterator(p.data() + NpartLocal_in_use); }

            /// Create MPIParticles from a set of particles. We only keep the particles in part that are in the correct
            /// x-range if all_tasks_has_the_same_particles is true nallocate is the total amount of particles we
            /// allocate for locally (we do buffered read if only one task reads particles to avoid having to allocate
            /// too much).
            /// @param[in] part Pointer to a set of particles
            /// @param[in] NumParts Number of particles we have
            /// @param[in] nallocate How many particles to allocate for locally (in case we want to have a buffer)
            /// @param[in] xmin_local The min of the x-range in [0,1] the current task is responsible for
            /// @param[in] xmax_local The max of the x-range in [0,1] the current task is responsible for
            /// @param[in] all_tasks_has_the_same_particles Set this to be true if all tasks has the same set of
            /// particles (e.g. they come from reading the same file). If false then we assume all tasks that has
            /// particles has distinct particles
            void create(T * part,
                        size_t NumParts,
                        size_t nallocate,
                        double xmin_local,
                        double xmax_local,
                        bool all_tasks_has_the_same_particles);

            /// Create from a vector of particles with a given selection function
            /// This could for example be that xmin < x < xmax and if particles have a type only select a particular
            /// type
            void create(std::vector<T> & part, size_t nallocate, std::function<bool(T &)> selection_function);

            /// Moves a vector of particles into internal storage (so no copies are being done).
            /// The extra storage (if needed) needs to allready be in part!
            /// Assumes we have distinct particles on different tasks. This saves having to do allocations
            /// and copies, but its safer and flexible to just use create() so its best to not use this unless
            /// you really need to
            void move_from(Vector<T> && part);

            /// Create \f$ {\rm Npart}_{1D}^{\rm NDIM} \f$ particles in a rectangular grid spread across all tasks
            /// buffer_factor is how much extra to allocate in case particles moves
            /// xmin, xmax specifies the domain in [0,1] that the current task is responsible for
            /// This method only sets the positions of the particles not id or vel or anything else
            void create_particle_grid(int Npart_1D, double buffer_factor, double xmin_local, double xmax_local);

            MPIParticles() = default;

            /// Get reference to particle vector. NB: due to we allow a buffer the size of the vector returned is
            /// not equal to the number of active particles!
            Vector<T> & get_particles();
            /// Get a pointer to the first particle.
            T * get_particles_ptr();

            /// Access particles through indexing operator
            T & operator[](size_t i);
            /// Get a reference to the i'th particle
            T & get_part(int i);

            /// Get the idim component of the position for particle ipart
            auto get_pos(size_t ipart, int idim);
            /// Get the idim component of the velocity for particle ipart (if its availiable)
            auto get_vel(size_t ipart, int idim);

            /// Total number of active particles across all tasks
            size_t get_npart_total() const;
            /// Number of active particles on the local task
            size_t get_npart() const;

            /// Communicate particles across CPU boundaries
            void communicate_particles();

            /// Get a vector of xmin of the domain for each task
            std::vector<double> get_x_min_per_task();
            /// Get a vector of xmax of the domain for each task
            std::vector<double> get_x_max_per_task();

            /// Free all memory of the stored particles
            void free();

            /// For memory logging add a tag to the vector we have allocated
            void add_memory_label(std::string name);

            /// Dump data to file (internal format)
            void dump_to_file(std::string fileprefix, size_t max_bytesize_buffer = 100 * 1000 * 1000);
            /// Load data from file (internal format)
            void load_from_file(std::string fileprefix);

            /// Show some info about the class
            void info();
        };

        template <class T>
        void MPIParticles<T>::info() {
            T tmp{};
            auto NDIM = FML::PARTICLE::GetNDIM(tmp);
            double memory_in_mb = 0;
            for (auto & part : p)
                memory_in_mb += FML::PARTICLE::GetSize(part);
            memory_in_mb /= 1e6;
            double max_memory_in_mb = memory_in_mb;
            double min_memory_in_mb = memory_in_mb;
            double mean_memory_in_mb = memory_in_mb / double(FML::NTasks);
            FML::MaxOverTasks(&max_memory_in_mb);
            FML::MinOverTasks(&min_memory_in_mb);
            FML::SumOverTasks(&mean_memory_in_mb);
            double fraction_filled = double(NpartLocal_in_use) / double(p.size()) * 100;
            double max_fraction_filled = fraction_filled;
            double min_fraction_filled = fraction_filled;
            double mean_fraction_filled = fraction_filled / double(FML::NTasks);
            FML::MaxOverTasks(&max_fraction_filled);
            FML::MinOverTasks(&min_fraction_filled);
            FML::SumOverTasks(&mean_fraction_filled);

            if (FML::ThisTask == 0) {
                std::cout << "\n";
                std::cout << "#=====================================================\n";
                std::cout << "#\n";
                std::cout << "#            .___        _____          \n";
                std::cout << "#            |   | _____/ ____\\____     \n";
                std::cout << "#            |   |/    \\   __\\/  _ \\    \n";
                std::cout << "#            |   |   |  \\  | (  <_> )   \n";
                std::cout << "#            |___|___|  /__|  \\____/    \n";
                std::cout << "#                     \\/                \n";
                std::cout << "#\n";
                std::cout << "# Info about MPIParticles. Dimension of particles is " << NDIM << "\n";
                std::cout << "# We have allocated " << mean_memory_in_mb << " MB (mean), " << min_memory_in_mb
                          << " MB (min), " << max_memory_in_mb << " MB (max)\n";
                std::cout << "# Total particles across all tasks is " << NpartTotal << "\n";
                std::cout << "# Task 0 has " << NpartLocal_in_use << " particles in use. Capacity is " << p.size()
                          << "\n";
                std::cout << "# The buffer is " << mean_fraction_filled << "%% (mean), " << min_fraction_filled
                          << " %% (min), " << max_fraction_filled << " %% (max) filled across tasks\n";
                std::cout << "#\n";
                std::cout << "#=====================================================\n";
                std::cout << "\n";
            }
        }

        template <class T>
        void MPIParticles<T>::add_memory_label([[maybe_unused]] std::string name) {
#ifdef MEMORY_LOGGING
            FML::MemoryLog::get()->add_label(p.data(), p.capacity(), name);
#endif
        }

        template <class T>
        void MPIParticles<T>::free() {
            p.clear();
            p.shrink_to_fit();
        }

        template <class T>
        void MPIParticles<T>::move_from(Vector<T> && part) {
            // Set the xmin/xmax
            x_min_per_task = FML::GatherFromTasks(&FML::xmin_domain);
            x_max_per_task = FML::GatherFromTasks(&FML::xmax_domain);
            // Move data from particles into internal storage
            p = std::move(part);
            // Set number of particles
            NpartLocal_in_use = p.size();
            NpartTotal = NpartLocal_in_use;
            FML::SumOverTasks(&NpartTotal);
            // Resize to capacity - we keep size in NpartLocal_in_use
            p.resize(p.capacity());
        }

        // Create from a vector of particles with a given selection function
        template <class T>
        void
        MPIParticles<T>::create(std::vector<T> & part, size_t nallocate, std::function<bool(T &)> selection_function) {

            // Set the xmin/xmax
            x_min_per_task = FML::GatherFromTasks(&FML::xmin_domain);
            x_max_per_task = FML::GatherFromTasks(&FML::xmax_domain);
            p.clear();
            p.reserve(nallocate + 1);

            size_t count = 0;
            for (auto & curpart : part) {
                if (selection_function(curpart)) {
                    p.push_back(curpart);
                    count++;
                    if (count > nallocate) {
                        assert_mpi(false, "[MPIParticle::create] Reached allocation limit. Increase nallocate\n");
                    }
                }
            }
            // Set number of particles
            NpartLocal_in_use = count;
            NpartTotal = NpartLocal_in_use;
            FML::SumOverTasks(&NpartTotal);
            // Resize to capacity - we keep size in NpartLocal_in_use
            p.resize(p.capacity());
        }

        template <class T>
        void MPIParticles<T>::create(T * part,
                                     size_t NumPartinpart,
                                     size_t nallocate,
                                     double xmin,
                                     double xmax,
                                     bool all_tasks_has_the_same_particles) {
            if (FML::NTasks == 1)
                all_tasks_has_the_same_particles = true;

            // Set the xmin/xmax
            x_min_per_task = FML::GatherFromTasks(&xmin);
            x_max_per_task = FML::GatherFromTasks(&xmax);

#ifdef DEBUG_MPIPARTICLES
            if (ThisTask == 0) {
                for (int i = 0; i < NTasks; i++) {
                    std::cout << "Task: " << i << " / " << NTasks << " xmin: " << x_min_per_task[i]
                              << " xmax: " << x_max_per_task[i] << "\n";
                }
            }
#endif

            // Allocate memory
            p.resize(nallocate);
            add_memory_label("MPIPartices::create");

            // If all tasks has the same particles
            // we read all the particles and only keep the particles in range
            if (all_tasks_has_the_same_particles) {
                size_t count = 0;
                for (size_t i = 0; i < NumPartinpart; i++) {
                    auto * pos = FML::PARTICLE::GetPos(part[i]);
                    if (pos[0] >= xmin and pos[0] < xmax) {
                        p[count] = part[i];
                        count++;
                    }
                }

                // Check that we are not past allocation limit
                if (count > nallocate) {
                    assert_mpi(false, "[MPIParticle::create] Reached allocation limit. Increase nallocate\n");
                }

                NpartLocal_in_use = count;

                NpartTotal = NpartLocal_in_use;
                FML::SumOverTasks(&NpartTotal);
            }

#ifdef USE_MPI
            std::cout << std::flush;
            MPI_Barrier(MPI_COMM_WORLD);
#endif

            // If only one or more task has particles then read in batches and communicate as we go along
            // just in case the total amount of particles are too large
            if (not all_tasks_has_the_same_particles) {

                const auto nmax_per_batch = nallocate;
                // Read in batches
                size_t start = 0;
                size_t count = 0;
                bool more_to_process_globally = true;
                bool more_to_process_locally = NumPartinpart > 0;
                while (more_to_process_globally) {
                    if (more_to_process_locally) {
                        size_t nbatch = start + nmax_per_batch < NumPartinpart ? start + nmax_per_batch : NumPartinpart;
                        more_to_process_locally = nbatch < NumPartinpart;

                        for (size_t i = start; i < nbatch; i++) {
                            p[count] = part[i];
                            count++;
                            if (count > nallocate) {
                                assert_mpi(false,
                                           "[MPIParticle::create] Reached allocation limit. Increase nallocate\n");
                            }
                        }
                        start = nbatch;
                    }

                    // Set the number of particles read
                    NpartLocal_in_use = count;

#ifdef DEBUG_MPIPARTICLES
                    std::cout << "Task: " << ThisTask << " NpartLocal_in_use: " << NpartLocal_in_use << " precomm\n";
#endif

                    // Send particles to where they belong
                    communicate_particles();

#ifdef DEBUG_MPIPARTICLES
                    std::cout << "Task: " << ThisTask << " NpartLocal_in_use: " << NpartLocal_in_use << " postcomm\n";
#endif

                    // Update how many particles we now have
                    count = NpartLocal_in_use;

                    // The while loop continues until all tasks are done reading particles
                    int moretodo = more_to_process_locally ? 1 : 0;
                    FML::SumOverTasks(&moretodo);
                    more_to_process_globally = (moretodo >= 1);
                }
            }

            // Set total number of particles
            NpartTotal = NpartLocal_in_use;
            FML::SumOverTasks(&NpartTotal);

#ifdef DEBUG_MPIPARTICLES
            std::cout << "Task: " << ThisTask << " NpartLocal_in_use: " << NpartLocal_in_use << "\n";
#endif
        }

        template <class T>
        T & MPIParticles<T>::get_part(int i) {
            return p[i];
        }

        template <class T>
        size_t MPIParticles<T>::get_npart() const {
            return NpartLocal_in_use;
        }

        template <class T>
        size_t MPIParticles<T>::get_npart_total() const {
            return NpartTotal;
        }

        template <class T>
        Vector<T> & MPIParticles<T>::get_particles() {
            return p;
        }

        template <class T>
        T * MPIParticles<T>::get_particles_ptr() {
            return p.data();
        }

        template <class T>
        std::vector<double> MPIParticles<T>::get_x_min_per_task() {
            return x_min_per_task;
        }

        template <class T>
        std::vector<double> MPIParticles<T>::get_x_max_per_task() {
            return x_max_per_task;
        }

        template <class T>
        T & MPIParticles<T>::operator[](size_t i) {
            return p[i];
        }

        template <class T>
        void MPIParticles<T>::swap_particles(T & a, T & b) {
            T tmp = a;
            a = b;
            b = tmp;
        }

        template <class T>
        void MPIParticles<T>::create_particle_grid(int Npart_1D,
                                                   double buffer_factor,
                                                   double xmin_local,
                                                   double xmax_local) {
            this->Npart_1D = Npart_1D;
            this->buffer_factor = buffer_factor;

            // Use the local xmin,xmax values to compute how many slices per task
            int imin = 0;
            while (imin / double(Npart_1D) < xmin_local) {
                imin++;
            }
            int imax = imin;
            while (imax / double(Npart_1D) < xmax_local) {
                imax++;
            }
            Local_p_start = imin;
            Local_Npart_1D = imax - imin;

            // Sanity check
            auto Local_Npart_1D_per_task = FML::GatherFromTasks(&Local_Npart_1D);
            int Local_p_start_computed = 0, Npart_1D_computed = 0;
            for (int i = 0; i < NTasks; i++) {
                if (i < ThisTask)
                    Local_p_start_computed += Local_Npart_1D_per_task[i];
                Npart_1D_computed += Local_Npart_1D_per_task[i];
            }
            assert_mpi(Npart_1D_computed == Npart_1D,
                       "[MPIParticles::create_particle_grid] Computed Npart does not match Npart\n");
            assert_mpi(Local_p_start_computed == Local_p_start,
                       "[MPIParticles::create_particle_grid] Computed Local_p_start does not match Local_p_start\n");

            // Get the min and max x-positions (in [0,1]) that each of the tasks is responsible for
            x_min_per_task = std::vector<double>(NTasks, 0.0);
            x_max_per_task = std::vector<double>(NTasks, 0.0);

            // If we don't have xmin,xmax availiable
            // x_min_per_task[ThisTask] = Local_p_start / double(Npart_1D);
            // x_max_per_task[ThisTask] = (Local_p_start + Local_Npart_1D) / double(Npart_1D);

            // Fetch these values
            x_min_per_task = FML::GatherFromTasks(&xmin_local);
            x_max_per_task = FML::GatherFromTasks(&xmax_local);

#ifdef DEBUG_MPIPARTICLES
            // Output some info
            for (int taskid = 0; taskid < NTasks; taskid++) {
                if (ThisTask == 0) {
                    std::cout << "Task[" << taskid << "]  xmin: " << x_min_per_task[taskid]
                              << " xmax: " << x_max_per_task[taskid] << "\n";
                }
            }
#endif

            // Total number of particles
            T tmp{};
            int ndim = FML::PARTICLE::GetNDIM(tmp);
            NpartTotal = FML::power(Npart_1D, ndim);
            NpartLocal_in_use = Local_Npart_1D * FML::power(Npart_1D, ndim - 1);

            // Allocate particle struct
            size_t NpartToAllocate =
                buffer_factor <= 1.0 ? NpartLocal_in_use : size_t(double(NpartLocal_in_use) * buffer_factor);
            assert(NpartToAllocate >= NpartLocal_in_use);
            p.resize(NpartToAllocate);
            add_memory_label("MPIPartices::create_particle_grid");

            // Initialize the coordinate to the first cell in the local grid
            std::vector<double> Pos(ndim, 0.0);
            std::vector<int> coord(ndim, 0);
            coord[0] = Local_p_start;
            for (size_t i = 0; i < NpartLocal_in_use; i++) {
                auto * Pos = FML::PARTICLE::GetPos(p[i]);

                // Position regular grid
                for (int idim = 0; idim < ndim; idim++) {
                    Pos[idim] = coord[idim] / double(Npart_1D);
                }

                // This is adding 1 very time in base Npart_1D storing the digits in reverse order in [coord]
                int idim = ndim - 1;
                while (++coord[idim] == Npart_1D) {
                    coord[idim--] = 0;
                    if (idim < 0)
                        break;
                }
            }
        }

        template <class T>
        auto MPIParticles<T>::get_pos(size_t ipart, int i) {
            auto pos = FML::PARTICLE::GetPos(p[ipart]);
            if (pos == nullptr)
                assert(false);
            return pos[i];
        }

        template <class T>
        auto MPIParticles<T>::get_vel(size_t ipart, int i) {
            auto vel = FML::PARTICLE::GetVel(p[ipart]);
            if (vel == nullptr)
                assert(false);
            return vel[i];
        }

        template <class T>
        size_t MPIParticles<T>::get_particle_byte_size(size_t ipart) {
            return FML::PARTICLE::GetSize(p[ipart]);
        }

        template <class T>
        void MPIParticles<T>::copy_over_recieved_data(std::vector<char> & recv_buffer, size_t Npart_recv) {
            assert_mpi(NpartLocal_in_use + Npart_recv <= p.size(),
                       "[MPIParticles::copy_over_recieved_data] Too many particles recieved! Increase buffer\n");

            char * buffer = recv_buffer.data();
            size_t bytes_processed = 0;
            for (size_t i = 0; i < Npart_recv; i++) {
                FML::PARTICLE::AssignFromBuffer(p[NpartLocal_in_use + i], buffer);
                auto size = FML::PARTICLE::GetSize(p[NpartLocal_in_use + i]);
                buffer += size;
                bytes_processed += size;
                assert(bytes_processed <= recv_buffer.size());
            }

            // Update the total number of particles in use
            NpartLocal_in_use += Npart_recv;
        }

        template <class T>
        void MPIParticles<T>::communicate_particles() {
            if (FML::NTasks == 1)
                return;
#ifdef USE_MPI

            // The number of particles we start with
            size_t NpartLocal_in_use_pre_comm = NpartLocal_in_use;

#ifdef DEBUG_MPIPARTICLES
            if (ThisTask == 0) {
                std::cout << "Communicating particles task: " << ThisTask
                          << " Nparticles: " << NpartLocal_in_use_pre_comm << "\n"
                          << std::flush;
            }
#endif

            // Count how many particles to send to each task
            // and move the particles to be send to the back of the array
            // and reduce the NumPartLocal_in_use if a partice is to be sent
            // After this is done we have all the particles to be send in
            // location [NpartLocal_in_use, NpartLocal_in_use_pre_comm)
            std::vector<int> n_to_send(NTasks, 0);
            std::vector<int> n_to_recv(NTasks, 0);
            std::vector<int> nbytes_to_send(NTasks, 0);
            std::vector<int> nbytes_to_recv(NTasks, 0);
            size_t i = 0;
            while (i < NpartLocal_in_use) {
                auto x = FML::PARTICLE::GetPos(p[i])[0];
                if (x >= x_max_per_task[ThisTask]) {
                    int taskid = ThisTask;
                    while (1) {
                        ++taskid;
                        if (x < x_max_per_task[taskid])
                            break;
                    }

                    n_to_send[taskid]++;
                    nbytes_to_send[taskid] += FML::PARTICLE::GetSize(p[i]);
                    swap_particles(p[i], p[--NpartLocal_in_use]);

                } else if (x < x_min_per_task[ThisTask]) {
                    int taskid = ThisTask;
                    while (1) {
                        --taskid;
                        if (x >= x_min_per_task[taskid])
                            break;
                    }

                    n_to_send[taskid]++;
                    nbytes_to_send[taskid] += FML::PARTICLE::GetSize(p[i]);
                    swap_particles(p[i], p[--NpartLocal_in_use]);

                } else {
                    i++;
                }
            }

            // Communicate to get how many to recieve from each task
            for (int i = 1; i < NTasks; i++) {
                int send_request_to = (ThisTask + i) % NTasks;
                int get_request_from = (ThisTask - i + NTasks) % NTasks;

                // Send to the right, recieve from left
                MPI_Status status;
                MPI_Sendrecv(&n_to_send[send_request_to],
                             1,
                             MPI_INT,
                             send_request_to,
                             0,
                             &n_to_recv[get_request_from],
                             1,
                             MPI_INT,
                             get_request_from,
                             0,
                             MPI_COMM_WORLD,
                             &status);

                // If part can have variable size
                MPI_Sendrecv(&nbytes_to_send[send_request_to],
                             1,
                             MPI_INT,
                             send_request_to,
                             0,
                             &nbytes_to_recv[get_request_from],
                             1,
                             MPI_INT,
                             get_request_from,
                             0,
                             MPI_COMM_WORLD,
                             &status);
            }

#ifdef DEBUG_MPIPARTICLES
            // Show some info
            if (ThisTask == 0) {
                for (int i = 0; i < NTasks; i++) {
                    std::cout << "Task " << ThisTask << " send to   " << i << " : " << n_to_send[i] << "\n"
                              << std::flush;
                    std::cout << "Task " << ThisTask << " recv from " << i << " : " << n_to_recv[i] << "\n"
                              << std::flush;
                }
            }
#endif

            // Total number to send and recv
            size_t ntot_to_send = 0;
            size_t ntot_to_recv = 0;
            size_t ntot_bytes_to_send = 0;
            size_t ntot_bytes_to_recv = 0;
            for (int i = 0; i < NTasks; i++) {
                ntot_to_send += n_to_send[i];
                ntot_to_recv += n_to_recv[i];
                ntot_bytes_to_send += nbytes_to_send[i];
                ntot_bytes_to_recv += nbytes_to_recv[i];
            }

            // Sanity check
            assert_mpi(NpartLocal_in_use_pre_comm == NpartLocal_in_use + ntot_to_send,
                       "[MPIParticles::communicate_particles] Number to particles to communicate does not match\n");

            // Allocate send buffer
            std::vector<char> send_buffer(ntot_bytes_to_send);
            std::vector<char> recv_buffer(ntot_bytes_to_recv);

            // Pointers to each send-recv place in the send-recv buffer
            std::vector<size_t> offset_in_send_buffer(NTasks, 0);
            std::vector<size_t> offset_in_recv_buffer(NTasks, 0);
            std::vector<char *> send_buffer_by_task(NTasks, send_buffer.data());
            std::vector<char *> recv_buffer_by_task(NTasks, recv_buffer.data());
            for (int i = 1; i < NTasks; i++) {
                offset_in_send_buffer[i] = offset_in_send_buffer[i - 1] + nbytes_to_send[i - 1];
                offset_in_recv_buffer[i] = offset_in_recv_buffer[i - 1] + nbytes_to_recv[i - 1];
                send_buffer_by_task[i] = &send_buffer.data()[offset_in_send_buffer[i]];
                recv_buffer_by_task[i] = &recv_buffer.data()[offset_in_recv_buffer[i]];
            }

            // Gather particle data
            for (size_t i = 0; i < ntot_to_send; i++) {
                size_t index = NpartLocal_in_use + i;
                auto x = FML::PARTICLE::GetPos(p[index])[0];
                if (x >= x_max_per_task[ThisTask]) {
                    int taskid = ThisTask;
                    while (1) {
                        ++taskid;
                        if (x < x_max_per_task[taskid])
                            break;
                    }

                    FML::PARTICLE::AppendToBuffer(p[index], send_buffer_by_task[taskid]);
                    send_buffer_by_task[taskid] += FML::PARTICLE::GetSize(p[index]);

                } else if (x < x_min_per_task[ThisTask]) {
                    int taskid = ThisTask;
                    while (1) {
                        --taskid;
                        if (x >= x_min_per_task[taskid])
                            break;
                    }

                    FML::PARTICLE::AppendToBuffer(p[index], send_buffer_by_task[taskid]);
                    send_buffer_by_task[taskid] += FML::PARTICLE::GetSize(p[index]);

                } else {

                    // We should not be here as particles are moved
                    assert_mpi(false,
                               "[MPIParticles::communicate_particles] Error in communicate_particles. After moving "
                               "particles we still have particles out of range\n");
                }
            }

            // We changed the send-recv pointers above so reset them
            for (int i = 0; i < NTasks; i++) {
                send_buffer_by_task[i] = &send_buffer.data()[offset_in_send_buffer[i]];
                recv_buffer_by_task[i] = &recv_buffer.data()[offset_in_recv_buffer[i]];
            }

            // Communicate the particle data
            for (int i = 1; i < NTasks; i++) {
                int send_request_to = (ThisTask + i) % NTasks;
                int get_request_from = (ThisTask - i + NTasks) % NTasks;

                // Send to the right, recieve from left
                MPI_Status status;
                MPI_Sendrecv(send_buffer_by_task[send_request_to],
                             nbytes_to_send[send_request_to],
                             MPI_CHAR,
                             send_request_to,
                             0,
                             recv_buffer_by_task[get_request_from],
                             nbytes_to_recv[get_request_from],
                             MPI_CHAR,
                             get_request_from,
                             0,
                             MPI_COMM_WORLD,
                             &status);
            }

            // Copy over the particle data (this also updates the total number of particles)
            copy_over_recieved_data(recv_buffer, ntot_to_recv);
#endif
        }

        template <class T>
        void MPIParticles<T>::dump_to_file(std::string fileprefix, size_t max_bytesize_buffer) {
            std::ios_base::sync_with_stdio(false);
            std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
            auto myfile = std::fstream(filename, std::ios::out | std::ios::binary);

            // If we fail to write give a warning, but continue
            if (not myfile.good()) {
                std::string error = "[MPIParticles::dump_to_file] Failed to save the particle data on task " +
                                    std::to_string(FML::ThisTask) + " Filename: " + filename;
                std::cout << error << "\n";
                std::ios_base::sync_with_stdio(true);
                return;
            }

            T tmp;
            int ndim = FML::PARTICLE::GetNDIM(tmp);

            // Compute total bytes to write
            size_t total_bytes_to_write = 0;
            for (auto & part : p)
                total_bytes_to_write += FML::PARTICLE::GetSize(part);

            // Write header data
            size_t NpartLocalAllocated = p.size();
            myfile.write((char *)&ndim, sizeof(ndim));
            myfile.write((char *)&NpartTotal, sizeof(NpartTotal));
            myfile.write((char *)&NpartLocal_in_use, sizeof(NpartLocal_in_use));
            myfile.write((char *)&NpartLocalAllocated, sizeof(NpartLocalAllocated));
            myfile.write((char *)x_min_per_task.data(), sizeof(double) * FML::NTasks);
            myfile.write((char *)x_max_per_task.data(), sizeof(double) * FML::NTasks);

            if (NpartLocal_in_use == 0)
                return;

            // Allocate a write buffer
            std::vector<char> buffer_data(max_bytesize_buffer);
            assert_mpi(max_bytesize_buffer > size_t(100 * FML::PARTICLE::GetSize(p[0])),
                       "[MPIParticles::dump_to_file] Buffer size is likely too small, can't even fit 100 particles\n");

            // Write in chunks
            size_t nwritten = 0;
            while (nwritten < NpartLocal_in_use) {

                size_t n_to_write = NpartLocal_in_use;
                size_t nbytes_to_write = 0;

                std::cout << "Writing " << nwritten << std::endl;

                char * buffer = buffer_data.data();
                for (size_t i = 0; i < n_to_write; i++) {
                    auto bytes = FML::PARTICLE::GetSize(p[nwritten + i]);
                    if (nbytes_to_write + bytes > max_bytesize_buffer) {
                        n_to_write = i;
                        break;
                    }
                    FML::PARTICLE::AppendToBuffer(p[nwritten + i], buffer);
                    buffer += bytes;
                    nbytes_to_write += bytes;
                }
                myfile.write((char *)&n_to_write, sizeof(n_to_write));
                myfile.write((char *)&nbytes_to_write, sizeof(nbytes_to_write));
                myfile.write((char *)buffer_data.data(), nbytes_to_write);

                nwritten += n_to_write;
            }
            myfile.close();
            std::ios_base::sync_with_stdio(true);
        }

        template <class T>
        void MPIParticles<T>::load_from_file(std::string fileprefix) {
            std::ios_base::sync_with_stdio(false);
            std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
            auto myfile = std::ifstream(filename, std::ios::binary);

            // If we fail to load a file throw an error
            if (not myfile.good()) {
                std::string error = "[MPIParticles::load_from_file] Failed to read the particles on task " +
                                    std::to_string(FML::ThisTask) + " Filename: " + filename;
                assert_mpi(false, error.c_str());
            }

            T tmp;
            int ndim_expected = FML::PARTICLE::GetNDIM(tmp);
            int ndim;

            // Read header data
            myfile.read((char *)&ndim, sizeof(ndim));
            assert_mpi(ndim == ndim_expected,
                       "[MPIParticles::load_from_file] Particle dimension do not match the one in the file");
            myfile.read((char *)&NpartTotal, sizeof(NpartTotal));
            myfile.read((char *)&NpartLocal_in_use, sizeof(NpartLocal_in_use));
            size_t NpartLocalAllocated;
            myfile.read((char *)&NpartLocalAllocated, sizeof(NpartLocalAllocated));
            x_min_per_task.resize(FML::NTasks);
            x_max_per_task.resize(FML::NTasks);
            myfile.read((char *)x_min_per_task.data(), sizeof(double) * FML::NTasks);
            myfile.read((char *)x_max_per_task.data(), sizeof(double) * FML::NTasks);

            // Allocate memory
            p.resize(NpartLocalAllocated);
            if (NpartLocal_in_use == 0)
                return;

            // Allocate a read buffer
            std::vector<char> buffer_data;

            // Read in chunks
            size_t nread = 0;
            while (nread < NpartLocal_in_use) {

                size_t n_to_read{};
                size_t nbytes_to_read{};
                myfile.read((char *)&n_to_read, sizeof(n_to_read));
                myfile.read((char *)&nbytes_to_read, sizeof(nbytes_to_read));
                if (buffer_data.size() < nbytes_to_read)
                    buffer_data.resize(1.25 * nbytes_to_read);
                char * buffer = buffer_data.data();
                std::cout << "Reading " << n_to_read << " / " << NpartLocal_in_use << " " << nbytes_to_read << " "
                          << buffer_data.size() << std::endl;
                myfile.read(buffer, nbytes_to_read);
                for (size_t i = 0; i < n_to_read; i++) {
                    FML::PARTICLE::AssignFromBuffer(p[nread + i], buffer);
                    buffer += FML::PARTICLE::GetSize(p[nread + i]);
                }

                nread += n_to_read;
            }
            myfile.close();
            std::ios_base::sync_with_stdio(true);
        }

    } // namespace PARTICLE
} // namespace FML
#endif
