#ifndef MPIPARTICLES_HEADER
#define MPIPARTICLES_HEADER

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <FML/Global/Global.h>
#include <FML/Global/MPICounts.h>
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
        ///    assign_from_buffer(char *) : assign data to a particle after it has been received moving buffer forward
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
            void copy_over_received_data(std::vector<char> & recv_buffer, size_t Npart_received);

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
            /// The extra storage (if needed) needs to already be in part!
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

            // Has the mpiparticles been created?
            explicit operator bool() const;

            /// Get reference to particle vector. NB: due to we allow a buffer the size of the vector returned is
            /// not equal to the number of active particles!
            Vector<T> & get_particles();
            /// Get a pointer to the first particle.
            T * get_particles_ptr();

            /// Access particles through indexing operator
            T & operator[](size_t i);
            /// Get a reference to the i'th particle
            T & get_part(size_t i);

            /// Get the idim component of the position for particle ipart
            auto get_pos(size_t ipart, int idim);
            /// Get the idim component of the velocity for particle ipart (if its available)
            auto get_vel(size_t ipart, int idim);

            /// Total number of active particles across all tasks
            size_t get_npart_total() const;
            /// Number of active particles on the local task
            size_t get_npart() const;

            /// Communicate particles across CPU boundaries
            void communicate_particles(size_t max_message_bytes = std::numeric_limits<int>::max());

            /// Get a vector of xmin of the domain for each task
            std::vector<double> get_x_min_per_task();
            /// Get a vector of xmax of the domain for each task
            std::vector<double> get_x_max_per_task();

            /// Free all memory of the stored particles
            void free();

            /// For memory logging add a tag to the vector we have allocated
            void add_memory_label(std::string name);

            /// Dump data to file (internal format: a raw copy of the particles, one file fileprefix.X per task).
            /// Meant for checkpointing within the same program: the files can only be read back with
            /// load_from_file using the same particle type, the same number of tasks and the same machine
            /// architecture. For portable output use e.g. GADGET (HDF5) files (GadgetUtils)
            void dump_to_file(std::string fileprefix, size_t max_bytesize_buffer = 100 * 1000 * 1000);
            /// Load data written by dump_to_file (same particle type and number of tasks)
            void load_from_file(std::string fileprefix);

            /// Show some info about the class
            void info();

            /// Reorder the local particles by position (see sort_particles_by_position) so that particles close in
            /// space are close in memory. Speeds up assigning particles to a grid and interpolating from it
            void sort_by_position(int nblocks_per_dim);
        };

        /// @brief Reorder particles by position: stable counting sort by the block of a nblocks_per_dim^N grid the
        /// particle is in, the blocks in x-major order (the memory order of FFTWGrid real grids). Particles close in
        /// space end up close in memory, which makes assigning them to a grid and interpolating from it faster (4x for
        /// shuffled vs ordered particles with CIC). The reordering is done in place (following the cycles of the
        /// permutation) and needs 4 bytes per particle (8 for more than 2^32 particles) plus a counter per block in
        /// the local x-range. The particles keep all their data (IDs etc.), only the order changes.
        template <class T, class IndexType>
        void sort_particles_by_position_impl(T * part, size_t NumPart, int nblocks_per_dim) {
            if (NumPart < 2)
                return;
            T tmp{};
            constexpr int N = FML::PARTICLE::GetNDIM(tmp);
            const int nb = nblocks_per_dim;
            auto block_coord = [&](double x) { return std::max(0, std::min(int(x * nb), nb - 1)); };

            // The x-range of the blocks of the local particles
            int bxmin = nb, bxmax = -1;
            for (size_t i = 0; i < NumPart; i++) {
                const int bx = block_coord(FML::PARTICLE::GetPos(part[i])[0]);
                bxmin = std::min(bxmin, bx);
                bxmax = std::max(bxmax, bx);
            }
            size_t nblocks = size_t(bxmax - bxmin + 1);
            for (int idim = 1; idim < N; idim++)
                nblocks *= size_t(nb);
            auto block_of = [&](size_t i) {
                const auto * pos = FML::PARTICLE::GetPos(part[i]);
                size_t b = size_t(block_coord(pos[0]) - bxmin);
                for (int idim = 1; idim < N; idim++)
                    b = b * nb + size_t(block_coord(pos[idim]));
                return b;
            };

            // Destination of each particle (stable counting sort)
            std::vector<size_t> start(nblocks + 1, 0);
            for (size_t i = 0; i < NumPart; i++)
                start[block_of(i) + 1]++;
            for (size_t b = 0; b < nblocks; b++)
                start[b + 1] += start[b];
            std::vector<IndexType> dest(NumPart);
            for (size_t i = 0; i < NumPart; i++)
                dest[i] = IndexType(start[block_of(i)]++);
            std::vector<size_t>().swap(start);

            // Apply the permutation in place: put the particle at i where it should go until the one at i belongs there
            for (size_t i = 0; i < NumPart; i++) {
                while (size_t(dest[i]) != i) {
                    const size_t j = dest[i];
                    std::swap(part[i], part[j]);
                    std::swap(dest[i], dest[j]);
                }
            }
        }

        template <class T>
        void sort_particles_by_position(T * part, size_t NumPart, int nblocks_per_dim) {
            assert_mpi(nblocks_per_dim > 0, "[sort_particles_by_position] nblocks_per_dim must be positive\n");
            if (NumPart < size_t(std::numeric_limits<uint32_t>::max()))
                sort_particles_by_position_impl<T, uint32_t>(part, NumPart, nblocks_per_dim);
            else
                sort_particles_by_position_impl<T, size_t>(part, NumPart, nblocks_per_dim);
        }

        template <class T>
        void MPIParticles<T>::sort_by_position(int nblocks_per_dim) {
            sort_particles_by_position(p.data(), NpartLocal_in_use, nblocks_per_dim);
        }

        template <class T>
        MPIParticles<T>::operator bool() const {
            return p.size() > 0;
        }

        template <class T>
        void MPIParticles<T>::info() {
            auto NDIM = FML::PARTICLE::GetNDIM(T());
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
                std::cout << "# We have allocated " << mean_memory_in_mb << " MB (mean)\n";
                std::cout << "#                   " << min_memory_in_mb << " MB (min)\n";
                std::cout << "#                   " << max_memory_in_mb << " MB (max)\n";
                std::cout << "# Total particles across all tasks is " << NpartTotal << "\n";
                std::cout << "# Task 0 has " << NpartLocal_in_use << " parts. Capacity: " << p.size() << "\n";
                std::cout << "# The buffer is " << mean_fraction_filled << " % (mean)\n";
                std::cout << "#               " << min_fraction_filled << " % (min)\n";
                std::cout << "#               " << max_fraction_filled << " % (max) filled across tasks\n";
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

        /// Bring the x-coordinate of a particle given to MPIParticles::create into [0,1): x = 1.0 (or a tiny
        /// negative x that wraps to 1.0 after rounding, e.g. -1e-17) is a valid periodic position that would otherwise
        /// be in no task's range [xmin,xmax). x = 1.0 is treated as communicate_particles does (just below 1.0) and a
        /// tiny negative x becomes 0.0. Anything else outside [0,1) (or NaN) is an error
        template <class U>
        void mpiparticles_wrap_boundary_position(U & x) {
            if (x == U(1.0))
                x = std::nextafter(x, U(0.0));
            else if (x < U(0.0) and x + U(1.0) == U(1.0))
                x = U(0.0);
            if (not(x >= U(0.0) and x < U(1.0))) {
                std::string errmsg = "[MPIParticles::create] Particle with x = " + std::to_string(double(x)) +
                                     " is outside [0,1). Particles must be wrapped (periodic) before calling create\n";
                assert_mpi(false, errmsg.c_str());
            }
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
            add_memory_label("MPIParticles::create");

            // If all tasks has the same particles
            // we read all the particles and only keep the particles in range
            if (all_tasks_has_the_same_particles) {
                size_t count = 0;
                for (size_t i = 0; i < NumPartinpart; i++) {
                    auto x = FML::PARTICLE::GetPos(part[i])[0];
                    mpiparticles_wrap_boundary_position(x);
                    if (x >= xmin and x < xmax) {
                        // Check that we are not past allocation limit before writing
                        assert_mpi(count < nallocate,
                                   "[MPIParticle::create] Reached allocation limit. Increase nallocate\n");
                        p[count] = part[i];
                        FML::PARTICLE::GetPos(p[count])[0] = x;
                        count++;
                    }
                }

                NpartLocal_in_use = count;

                NpartTotal = NpartLocal_in_use;
                FML::SumOverTasks(&NpartTotal);
            }

#ifdef USE_MPI
            std::cout << std::flush;
            MPI_Barrier(FML::communicator);
#endif

            // If only one or more task has particles then read in batches and communicate as we go along
            // just in case the total amount of particles are too large
            if (not all_tasks_has_the_same_particles) {

                // Read in batches
                size_t start = 0;
                size_t count = 0;
                bool more_to_process_globally = true;
                bool more_to_process_locally = NumPartinpart > 0;
                while (more_to_process_globally) {
                    if (more_to_process_locally) {
                        // We can only read as many as we have free space for (we keep the particles that belong
                        // to us from earlier batches and receive particles from other tasks)
                        const size_t nmax_per_batch = nallocate - count;
                        assert_mpi(nmax_per_batch > 0,
                                   "[MPIParticle::create] Reached allocation limit while reading particles in batches "
                                   "(no room for more particles). Increase nallocate\n");
                        size_t nbatch = start + nmax_per_batch < NumPartinpart ? start + nmax_per_batch : NumPartinpart;
                        more_to_process_locally = nbatch < NumPartinpart;

                        for (size_t i = start; i < nbatch; i++) {
                            // Check that we are not past allocation limit before writing
                            assert_mpi(count < nallocate,
                                       "[MPIParticle::create] Reached allocation limit. Increase nallocate\n");
                            p[count] = part[i];
                            mpiparticles_wrap_boundary_position(FML::PARTICLE::GetPos(p[count])[0]);
                            count++;
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
        T & MPIParticles<T>::get_part(size_t i) {
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

            // If we don't have xmin,xmax available
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
            T tmp_ndim; // (GetNDIM(T()) needs a particle that can be built at compile time)
            constexpr int ndim = FML::PARTICLE::GetNDIM(tmp_ndim);
            NpartTotal = FML::power(Npart_1D, ndim);
            NpartLocal_in_use = Local_Npart_1D * FML::power(Npart_1D, ndim - 1);

            // Allocate particle struct. The capacity is buffer_factor times the largest number of lattice slices
            // that the local x-range [xmin_local, xmax_local) can hold (for any uniform shift of the lattice), not
            // times the number of slices we got: with Npart_1D not divisible by NTasks a task can get fewer slices
            // than its share of the volume (or no slice at all if Npart_1D < NTasks) and would otherwise have no room
            // for particles moving in. When the domain holds a whole number of slices (the usual case) this is the
            // same as buffer_factor * NpartLocal_in_use
            const double nslices_domain = double(Npart_1D) * (xmax_local - xmin_local);
            const size_t max_slices = size_t(std::max(0.0, std::ceil(nslices_domain - 1e-9)));
            const size_t npart_per_slice = size_t(FML::power(Npart_1D, ndim - 1));
            size_t NpartToAllocate =
                size_t(std::ceil(std::max(buffer_factor, 1.0) * double(max_slices * npart_per_slice)));
            NpartToAllocate = std::max(NpartToAllocate, size_t(NpartLocal_in_use));
            assert_mpi(NpartToAllocate >= NpartLocal_in_use,
                       "[MPIParticles::create_particle_grid] Allocation is smaller than the number of particles\n");
            p.resize(NpartToAllocate);
            add_memory_label("MPIParticles::create_particle_grid");

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
        void MPIParticles<T>::copy_over_received_data(std::vector<char> & recv_buffer, size_t Npart_recv) {
            assert_mpi(NpartLocal_in_use <= p.size() && Npart_recv <= p.size() - NpartLocal_in_use,
                       "[MPIParticles::copy_over_received_data] Too many particles received! Increase buffer\n");

            char * buffer = recv_buffer.data();
            size_t bytes_processed = 0;
            for (size_t i = 0; i < Npart_recv; i++) {
                FML::PARTICLE::AssignFromBuffer(p[NpartLocal_in_use + i], buffer);
                auto size = FML::PARTICLE::GetSize(p[NpartLocal_in_use + i]);
                assert_mpi(size > 0 && size_t(size) <= recv_buffer.size() - bytes_processed,
                           "[MPIParticles] Invalid deserialized particle byte size");
                buffer += size;
                bytes_processed += size_t(size);
            }

            // Update the total number of particles in use
            NpartLocal_in_use += Npart_recv;
        }

        template <class T> void MPIParticles<T>::communicate_particles([[maybe_unused]] size_t max_message_bytes) {
            if (FML::NTasks == 1)
                return;
#ifdef USE_MPI

            FML::COMMUNICATION::Count max_chunk_bytes =
                std::min(max_message_bytes, size_t(std::numeric_limits<int>::max()));
            FML::MinOverTasks(&max_chunk_bytes);
            FML::COMMUNICATION::require_count(max_chunk_bytes > 0, "[MPIParticles] Message limit must be positive");

            // The number of particles we start with
            size_t NpartLocal_in_use_pre_comm = NpartLocal_in_use;

#ifdef DEBUG_MPIPARTICLES
            if (ThisTask == 0) {
                std::cout << "Communicating particles task: " << ThisTask
                          << " Nparticles: " << NpartLocal_in_use_pre_comm << "\n"
                          << std::flush;
            }
#endif

            T tmp{};
            using PosType = std::remove_reference_t<decltype(tmp.get_pos()[0])>;

            // Count how many particles to send to each task
            // and move the particles to be send to the back of the array
            // and reduce the NumPartLocal_in_use if a partice is to be sent
            // After this is done we have all the particles to be send in
            // location [NpartLocal_in_use, NpartLocal_in_use_pre_comm)
            using Count = FML::COMMUNICATION::Count;
            std::vector<Count> n_to_send(NTasks, 0), n_to_recv(NTasks, 0);
            std::vector<Count> nbytes_to_send(NTasks, 0), nbytes_to_recv(NTasks, 0);
            bool valid_counts = true;
            auto count_outgoing = [&](int task, T &particle) {
                auto bytes = FML::PARTICLE::GetSize(particle);
                valid_counts = valid_counts && bytes > 0;
                if (bytes > 0) {
                    valid_counts &= FML::COMMUNICATION::add_count(n_to_send[task], 1);
                    valid_counts &= FML::COMMUNICATION::add_count(nbytes_to_send[task], Count(bytes));
                }
            };
            size_t i = 0;
            while (i < NpartLocal_in_use) {
                auto & x = FML::PARTICLE::GetPos(p[i])[0];

                // To fix issues appearing when we are exactly on the boundary
                if (x == 1.0)
                    x = std::nextafter(x, PosType(0.0));
                if (x == 0.0)
                    x = std::numeric_limits<PosType>::min();

                // Particles must be wrapped inside [0,1) (this also catches NaN)
                if (not(x >= 0.0 and x < 1.0)) {
                    std::string errmsg =
                        "[MPIParticles::communicate_particles] Particle with x = " + std::to_string(double(x)) +
                        " is outside [0,1). Particles must be wrapped (periodic) before communication\n";
                    assert_mpi(false, errmsg.c_str());
                }

                if (x >= x_max_per_task[ThisTask]) {
                    int taskid = ThisTask;
                    while (taskid < NTasks - 1) {
                        ++taskid;
                        if (x < x_max_per_task[taskid])
                            break;
                    }

                    count_outgoing(taskid, p[i]);
                    swap_particles(p[i], p[--NpartLocal_in_use]);

                } else if (x < x_min_per_task[ThisTask] or x == 0.0) {
                    int taskid = ThisTask;
                    while (taskid > 0) {
                        --taskid;
                        if (x >= x_min_per_task[taskid])
                            break;
                    }

                    count_outgoing(taskid, p[i]);
                    swap_particles(p[i], p[--NpartLocal_in_use]);

                } else {
                    i++;
                }
            }

            // Communicate to get how many particles and bytes to receive from each task
            FML::COMMUNICATION::require_count(valid_counts, "[MPIParticles] Outgoing particle/byte count overflow");
            MPI_Alltoall(n_to_send.data(), 1, MPI_UNSIGNED_LONG_LONG, n_to_recv.data(), 1, MPI_UNSIGNED_LONG_LONG,
                         FML::communicator);
            MPI_Alltoall(nbytes_to_send.data(), 1, MPI_UNSIGNED_LONG_LONG, nbytes_to_recv.data(), 1,
                         MPI_UNSIGNED_LONG_LONG, FML::communicator);

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
            const auto send_particles = FML::COMMUNICATION::checked_size_layout(n_to_send, p.max_size());
            const auto recv_particles = FML::COMMUNICATION::checked_size_layout(n_to_recv, p.max_size());
            const auto send_bytes =
                FML::COMMUNICATION::checked_size_layout(nbytes_to_send, std::vector<char>().max_size());
            const auto recv_bytes =
                FML::COMMUNICATION::checked_size_layout(nbytes_to_recv, std::vector<char>().max_size());
            const size_t ntot_to_send = send_particles.total, ntot_to_recv = recv_particles.total;
            const size_t ntot_bytes_to_send = send_bytes.total, ntot_bytes_to_recv = recv_bytes.total;

            // Sanity check
            std::string error =
                "[MPIParticles::communicate_particles] Number to particles to communicate does not match\n" +
                std::to_string(NpartLocal_in_use_pre_comm) + " vs " + std::to_string(NpartLocal_in_use + ntot_to_send) +
                "\n";
            error += "Possible cause is particles not being wrapped inside [0,1) or not communicated after they have "
                     "moved across boundaries!\n";
            assert_mpi(NpartLocal_in_use <= NpartLocal_in_use_pre_comm &&
                           ntot_to_send == NpartLocal_in_use_pre_comm - NpartLocal_in_use,
                       error.c_str());

            FML::COMMUNICATION::require_count(NpartLocal_in_use <= p.size() &&
                                                  ntot_to_recv <= p.size() - NpartLocal_in_use,
                                              "[MPIParticles] Receive count exceeds allocated particle capacity");

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
                send_buffer_by_task[i] =
                    (send_buffer.empty() ? nullptr : send_buffer.data() + offset_in_send_buffer[i]);
                recv_buffer_by_task[i] =
                    (recv_buffer.empty() ? nullptr : recv_buffer.data() + offset_in_recv_buffer[i]);
            }

            std::vector<size_t> packed(NTasks, 0);
            auto append_particle = [&](size_t index, int task) {
                auto bytes = FML::PARTICLE::GetSize(p[index]);
                if (bytes <= 0 || size_t(bytes) > size_t(nbytes_to_send[task]) - packed[task]) {
                    valid_counts = false;
                    return;
                }
                FML::PARTICLE::AppendToBuffer(p[index], send_buffer_by_task[task]);
                send_buffer_by_task[task] += size_t(bytes);
                packed[task] += size_t(bytes);
            };

            // Gather particle data
            for (size_t i = 0; i < ntot_to_send; i++) {
                size_t index = NpartLocal_in_use + i;
                auto x = FML::PARTICLE::GetPos(p[index])[0];

                if (x >= x_max_per_task[ThisTask]) {
                    int taskid = ThisTask;
                    while (taskid < NTasks - 1) {
                        ++taskid;
                        if (x < x_max_per_task[taskid])
                            break;
                    }

                    append_particle(index, taskid);

                } else if (x < x_min_per_task[ThisTask]) {
                    int taskid = ThisTask;
                    while (taskid > 0) {
                        --taskid;
                        if (x >= x_min_per_task[taskid])
                            break;
                    }

                    append_particle(index, taskid);

                } else {

                    // We should not be here as particles are moved
                    assert_mpi(false,
                               "[MPIParticles::communicate_particles] Error in communicate_particles. After moving "
                               "particles we still have particles out of range\n");
                }
            }

            for (int task = 0; task < NTasks; ++task)
                valid_counts &= packed[task] == size_t(nbytes_to_send[task]);
            FML::COMMUNICATION::require_count(valid_counts,
                                              "[MPIParticles] Serialized particle sizes changed while packing");

            // We changed the send-recv pointers above so reset them
            for (int i = 0; i < NTasks; i++) {
                send_buffer_by_task[i] =
                    (send_buffer.empty() ? nullptr : send_buffer.data() + offset_in_send_buffer[i]);
                recv_buffer_by_task[i] =
                    (recv_buffer.empty() ? nullptr : recv_buffer.data() + offset_in_recv_buffer[i]);
            }

            // Communicate the particle data. MPI counts are int so we send in chunks of
            // at most INT_MAX bytes. All tasks do the same number of chunks so that the
            // send/recv pairs always match up (most of the time this is just 1 chunk)
            Count max_bytes = 0;
            for (int i = 0; i < NTasks; i++)
                max_bytes = std::max(max_bytes, std::max(nbytes_to_send[i], nbytes_to_recv[i]));
            FML::MaxOverTasks(&max_bytes);
            const Count nchunks = max_bytes / max_chunk_bytes + (max_bytes % max_chunk_bytes != 0);

            for (int i = 1; i < NTasks; i++) {
                int send_request_to = (ThisTask + i) % NTasks;
                int get_request_from = (ThisTask - i + NTasks) % NTasks;

                for (Count ichunk = 0; ichunk < nchunks; ichunk++) {
                    const Count offset = ichunk * max_chunk_bytes;
                    const int nsend = int(offset < nbytes_to_send[send_request_to]
                                              ? std::min(nbytes_to_send[send_request_to] - offset, max_chunk_bytes)
                                              : 0);
                    const int nrecv = int(offset < nbytes_to_recv[get_request_from]
                                              ? std::min(nbytes_to_recv[get_request_from] - offset, max_chunk_bytes)
                                              : 0);
                    char *send_ptr = nsend ? send_buffer_by_task[send_request_to] + size_t(offset) : nullptr;
                    char *recv_ptr = nrecv ? recv_buffer_by_task[get_request_from] + size_t(offset) : nullptr;

                    // Send to the right, receive from left
                    MPI_Status status;
                    MPI_Sendrecv(send_ptr,
                                 nsend,
                                 MPI_CHAR,
                                 send_request_to,
                                 0,
                                 recv_ptr,
                                 nrecv,
                                 MPI_CHAR,
                                 get_request_from,
                                 0,
                                 FML::communicator,
                                 &status);
                }
            }

            // Copy over the particle data (this also updates the total number of particles)
            copy_over_received_data(recv_buffer, ntot_to_recv);
#endif
        }

        template <class T>
        void MPIParticles<T>::dump_to_file(std::string fileprefix, size_t max_bytesize_buffer) {
            std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
            auto myfile = std::fstream(filename, std::ios::out | std::ios::binary);

            // If we fail to write give a warning, but continue
            if (not myfile.good()) {
                std::string error = "[MPIParticles::dump_to_file] Failed to save the particle data on task " +
                                    std::to_string(FML::ThisTask) + " Filename: " + filename;
                std::cout << error << "\n";
                return;
            }

            int ndim = FML::PARTICLE::GetNDIM(T());

            // Write header data
            size_t NpartLocalAllocated = p.size();
            myfile.write((char *)&ndim, sizeof(ndim));
            myfile.write((char *)&NpartTotal, sizeof(NpartTotal));
            myfile.write((char *)&NpartLocal_in_use, sizeof(NpartLocal_in_use));
            myfile.write((char *)&NpartLocalAllocated, sizeof(NpartLocalAllocated));
            myfile.write((char *)x_min_per_task.data(), sizeof(double) * FML::NTasks);
            myfile.write((char *)x_max_per_task.data(), sizeof(double) * FML::NTasks);

            if (NpartLocal_in_use == 0) {
                myfile.close();
                return;
            }

            // Allocate a write buffer
            std::vector<char> buffer_data(max_bytesize_buffer);
            assert_mpi(max_bytesize_buffer > 0 && FML::PARTICLE::GetSize(p[0]) > 0 &&
                           size_t(FML::PARTICLE::GetSize(p[0])) <= (max_bytesize_buffer - 1) / 100,
                       "[MPIParticles::dump_to_file] Buffer size is likely too small, can't even fit 100 particles\n");

            if (FML::ThisTask == 0)
                std::cout << "# Writing particles to " << fileprefix << ".X\n";

            // Write in chunks (only the particles in use)
            size_t nwritten = 0;
            while (nwritten < NpartLocal_in_use) {

                size_t n_to_write = NpartLocal_in_use - nwritten;
                size_t nbytes_to_write = 0;

                char * buffer = buffer_data.data();
                for (size_t i = 0; i < n_to_write; i++) {
                    auto bytes = FML::PARTICLE::GetSize(p[nwritten + i]);
                    if (bytes <= 0 || size_t(bytes) > max_bytesize_buffer - nbytes_to_write) {
                        n_to_write = i;
                        break;
                    }
                    FML::PARTICLE::AppendToBuffer(p[nwritten + i], buffer);
                    buffer += bytes;
                    nbytes_to_write += bytes;
                }
                assert_mpi(n_to_write > 0,
                           "[MPIParticles::dump_to_file] A particle does not fit in the buffer. Increase "
                           "max_bytesize_buffer\n");
                myfile.write((char *)&n_to_write, sizeof(n_to_write));
                myfile.write((char *)&nbytes_to_write, sizeof(nbytes_to_write));
                myfile.write((char *)buffer_data.data(), nbytes_to_write);

                nwritten += n_to_write;
            }
            if (not myfile.good())
                std::cout << "[MPIParticles::dump_to_file] Error writing the particle data on task " +
                                 std::to_string(FML::ThisTask) + " Filename: " + filename + "\n";
            myfile.close();
        }

        template <class T>
        void MPIParticles<T>::load_from_file(std::string fileprefix) {
            std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
            auto myfile = std::ifstream(filename, std::ios::binary);

            // If we fail to load a file throw an error
            if (not myfile.good()) {
                std::string error = "[MPIParticles::load_from_file] Failed to read the particles on task " +
                                    std::to_string(FML::ThisTask) + " Filename: " + filename;
                assert_mpi(false, error.c_str());
            }

            int ndim_expected = FML::PARTICLE::GetNDIM(T());
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
            assert_mpi(myfile.good() and NpartLocal_in_use <= NpartLocalAllocated,
                       "[MPIParticles::load_from_file] Error reading the header of the file\n");

            // Allocate memory
            p.resize(NpartLocalAllocated);
            if (NpartLocal_in_use == 0)
                return;

            if (FML::ThisTask == 0)
                std::cout << "# Reading particles from " << fileprefix << ".X\n";

            // Allocate a read buffer
            std::vector<char> buffer_data;

            // Read in chunks
            size_t nread = 0;
            while (nread < NpartLocal_in_use) {

                size_t n_to_read{};
                size_t nbytes_to_read{};
                myfile.read((char *)&n_to_read, sizeof(n_to_read));
                myfile.read((char *)&nbytes_to_read, sizeof(nbytes_to_read));
                // Never read more particles than the file says we have (files from older versions could contain
                // more, and n_to_read = 0 would never finish)
                assert_mpi(myfile.good() and n_to_read > 0 and n_to_read <= NpartLocal_in_use - nread,
                           "[MPIParticles::load_from_file] Corrupt file: the chunks do not add up to the number of "
                           "particles in the header\n");
                assert_mpi(nbytes_to_read <= buffer_data.max_size() &&
                               nbytes_to_read <= size_t(std::numeric_limits<std::streamsize>::max()),
                           "[MPIParticles::load_from_file] Chunk byte count exceeds supported storage");
                if (buffer_data.size() < nbytes_to_read)
                    buffer_data.resize(nbytes_to_read);
                char * buffer = buffer_data.data();
                myfile.read(buffer, nbytes_to_read);
                assert_mpi(myfile.good(), "[MPIParticles::load_from_file] Error reading particle data\n");
                size_t bytes_processed = 0;
                for (size_t i = 0; i < n_to_read; i++) {
                    FML::PARTICLE::AssignFromBuffer(p[nread + i], buffer);
                    auto bytes = FML::PARTICLE::GetSize(p[nread + i]);
                    assert_mpi(bytes > 0 && size_t(bytes) <= nbytes_to_read - bytes_processed,
                               "[MPIParticles::load_from_file] Corrupt file: particle data larger than the chunk\n");
                    buffer += size_t(bytes);
                    bytes_processed += size_t(bytes);
                }

                nread += n_to_read;
            }
            myfile.close();
        }

    } // namespace PARTICLE
} // namespace FML
#endif
