#ifndef MPIPARTICLES_HEADER
#define MPIPARTICLES_HEADER

#include <FML/Performance/Counters.h>
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
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <stdexcept>
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

        enum class MigrationRoute { Serial, NoTraffic, Neighbors, General };

        /// The header of the files of MPIParticles::dump_to_file (then the domain, Domain::write, and the particles)
        struct MPIParticlesFileHeader {
            char magic[8] = {'F', 'M', 'L', 'P', 'A', 'R', 'T', '\0'};
            uint32_t version = 2; // (1: the format before the magic string, not readable)
            int32_t ntasks = 0;
            int32_t thistask = 0;
            int32_t ndim = 0;
            uint64_t sizeof_particle = 0;
            uint64_t npart_total = 0;
            uint64_t npart_local = 0;
            uint64_t npart_allocated = 0;
        };

        /// How the particles given to MPIParticles::create are spread over the tasks
        enum class Input {
            /// Every task has the same particles (e.g. all read the same file): each keeps those it owns
            Replicated,
            /// The tasks have different particles, anywhere in the box: they are sent to the tasks that own them
            /// (read in batches, communicating as we go, so one task can hold all of them)
            Distributed,
            /// The particles are this task's: taken as given (no ownership check, no communication; the caller makes
            /// sure they are on the right task, or communicates afterwards)
            Local
        };

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
        /// The particles are distributed according to a Domain (Global/Domain.h): the one given to
        /// the constructor, or else the global FML::domain (a Slab by default) when the particles are created. Every
        /// method supports every domain (create_particle_grid: Slab and Pencil). With Pencil the grid code needs
        /// PencilGrids (FML::GRID::make_grid, particle_domain_of); with PeanoHilbert the grid code (grids, density
        /// assignment, ...) cannot be used on the particles directly (move them with set_domain first), except the
        /// density assignment of Interpolation/DistributedDeposit.h (particles_to_grid_any_domain: any domain to
        /// slabs or pencils, as GADGET's PM)
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
            size_t NpartTotal{0}; // Total number of particles across all tasks
            MigrationRoute last_migration_route{MigrationRoute::Serial};
            size_t NpartLocal_in_use{0}; // Number of particles in use (total allocated given by p.size())

            // The decomposition of the particles over the tasks (Domain.h): the one given to the constructor (then
            // domain_given), or else the global FML::domain (for a Slab its current ranges xmin_domain, xmax_domain)
            // taken when the particles are created or loaded. Empty before
            std::optional<FML::Domain> domain{};
            bool domain_given{false};
            // The domain to use when creating particles: the given one, else the global one (collective)
            void set_domain_for_create();
            // The domain is made for NTasks tasks and (Pencil, PeanoHilbert) the particles have its dimension
            static void check_domain(const FML::Domain & d);
            // Bring the coordinates the domain uses into [0,1) (x = 1.0 just below; a tiny negative x to 0)
            void wrap_boundary_positions(T & particle) const;

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

            /// Create MPIParticles from a set of particles, distributed over the tasks according to the domain.
            /// @param[in] part Pointer to the particles (positions in [0,1); x = 1.0 is allowed)
            /// @param[in] NumParts Number of particles in part
            /// @param[in] nallocate How many particles to allocate for locally (room for particles moving in)
            /// @param[in] input How the particles are spread over the tasks: Input::Replicated (every task has the
            /// same particles: each keeps those it owns), Input::Distributed (different particles anywhere: sent to
            /// their owners, in batches) or Input::Local (already this task's: taken as given)
            /// @param[in] keep Optional filter: only the particles for which it is true are used (e.g. a type)
            void create(T * part,
                        size_t NumParts,
                        size_t nallocate,
                        Input input,
                        std::function<bool(T &)> keep = std::function<bool(T &)>{});
            /// The same from a vector of particles
            void create(std::vector<T> & part,
                        size_t nallocate,
                        Input input,
                        std::function<bool(T &)> keep = std::function<bool(T &)>{});

            /// Moves a vector of particles into internal storage (so no copies are being done).
            /// The extra storage (if needed) needs to already be in part!
            /// Assumes we have distinct particles on different tasks. This saves having to do allocations
            /// and copies, but its safer and flexible to just use create() so its best to not use this unless
            /// you really need to
            void move_from(Vector<T> && part);

            /// Create \f$ {\rm Npart}_{1D}^{\rm NDIM} \f$ particles in a rectangular grid, each task those in its
            /// part of the domain (a Slab or a Pencil domain; other domains throw std::invalid_argument: create the
            /// lattice on slabs and set_domain to move it). buffer_factor is how much extra to allocate in case
            /// particles move. This method only sets the positions of the particles not id or vel or anything else
            void create_particle_grid(int Npart_1D, double buffer_factor);

            //----------------------------------------------------------------------------------------------------
            // Deprecated (the old interface): thin wrappers around the methods above, kept so old code compiles
            //----------------------------------------------------------------------------------------------------

            /// Deprecated: the particles get the Slab domain made from every task's [xmin_local, xmax_local)
            /// (collective). Same as constructing with that domain and create(..., Input::Replicated) if
            /// all_tasks_has_the_same_particles, else create(..., Input::Distributed)
            [[deprecated("use create(part, NumParts, nallocate, Input::Replicated / Input::Distributed) with the "
                         "domain given to the constructor")]] void
            create(T * part,
                   size_t NumParts,
                   size_t nallocate,
                   double xmin_local,
                   double xmax_local,
                   bool all_tasks_has_the_same_particles) {
                set_slab_domain_from_local_range(xmin_local, xmax_local);
                create(part,
                       NumParts,
                       nallocate,
                       all_tasks_has_the_same_particles ? Input::Replicated : Input::Distributed);
            }
            /// Deprecated: the same as create(part, nallocate, Input::Local, selection_function)
            [[deprecated("use create(part, nallocate, Input::Local, selection_function)")]] void
            create(std::vector<T> & part, size_t nallocate, std::function<bool(T &)> selection_function) {
                create(part, nallocate, Input::Local, std::move(selection_function));
            }
            /// Deprecated: the particles get the Slab domain made from every task's [xmin_local, xmax_local)
            /// (collective), then the same as create_particle_grid(Npart_1D, buffer_factor)
            [[deprecated("use create_particle_grid(Npart_1D, buffer_factor) with the domain given to the "
                         "constructor")]] void
            create_particle_grid(int Npart_1D, double buffer_factor, double xmin_local, double xmax_local) {
                set_slab_domain_from_local_range(xmin_local, xmax_local);
                create_particle_grid(Npart_1D, buffer_factor);
            }

          private:
            // For the deprecated methods: use the Slab domain made from every task's local range (collective)
            void set_slab_domain_from_local_range(double xmin_local, double xmax_local) {
                domain = FML::slab_domain_from_local_range(xmin_local, xmax_local);
                domain_given = true;
            }

          public:
            /// The particles will use the global particle domain (FML::domain, taken when they are created)
            MPIParticles() = default;
            /// The particles will use this domain (made for NTasks tasks): Slab, Pencil or PeanoHilbert
            explicit MPIParticles(const FML::Domain & particle_domain_in);

            /// The domain of the particles (once created or loaded; else the one given to the constructor, or the
            /// global one)
            const FML::Domain & get_domain() const { return domain ? *domain : FML::domain; }

            /// Change the domain of the particles and move them to their new owners (collective; made for NTasks
            /// tasks). For particles that follow a grid whose decomposition changes, e.g. after a load balance:
            /// [ grid.load_balance(); particles.set_domain(grid.get_domain()); ]
            void set_domain(const FML::Domain & new_domain);

            // Has the mpiparticles been created?
            explicit operator bool() const;

            /// Get reference to particle vector. NB: due to we allow a buffer the size of the vector returned is
            /// not equal to the number of active particles (it is get_capacity(); the active particles are the first
            /// get_npart(), as begin() to end())!
            Vector<T> & get_particles();
            /// The number of particles we have room for locally (the size of get_particles())
            size_t get_capacity() const { return p.size(); }
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
            void communicate_particles(size_t max_message_bytes = std::numeric_limits<int>::max(),
                                       bool force_general = false);
            /// Last route selected by communicate_particles (same on every rank).
            MigrationRoute get_last_migration_route() const { return last_migration_route; }

            /// Free all memory of the stored particles
            void free();

            /// For memory logging add a tag to the vector we have allocated
            void add_memory_label(std::string name);

            /// Dump data to file (internal format, one file fileprefix.X per task: a header with a magic string and
            /// version, the number of tasks, NDIM, sizeof(T), the particle counts and the domain, then the particles in
            /// chunks). Meant for checkpointing within the same program: the files can only be read back with
            /// load_from_file using the same particle type, the same number of tasks and the same machine
            /// architecture (all checked). For portable output use e.g. GADGET (HDF5) files (GadgetUtils)
            void dump_to_file(std::string fileprefix, size_t max_bytesize_buffer = 100 * 1000 * 1000);
            /// Load data written by dump_to_file (same particle type and number of tasks; with a domain given to the
            /// constructor, the file's domain must be the same)
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
            // The domain (the given one, else the global one)
            set_domain_for_create();
            // Move data from particles into internal storage
            p = std::move(part);
            // Set number of particles
            NpartLocal_in_use = p.size();
            NpartTotal = NpartLocal_in_use;
            FML::SumOverTasks(&NpartTotal);
            // Resize to capacity - we keep size in NpartLocal_in_use
            p.resize(p.capacity());
        }

        /// The number of dimensions of a particle: get_ndim() (some particle types only have a non-const one), or
        /// GetNDIM's default for particles without it
        template <class P, class = void>
        struct mpiparticles_has_ndim : std::false_type {};
        template <class P>
        struct mpiparticles_has_ndim<P, std::void_t<decltype(std::declval<P &>().get_ndim())>> : std::true_type {};
        template <class P>
        int mpiparticles_ndim(P & particle) {
            if constexpr (mpiparticles_has_ndim<P>::value)
                return int(particle.get_ndim());
            else
                return FML::PARTICLE::GetNDIM(particle);
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
        void
        MPIParticles<T>::create(std::vector<T> & part, size_t nallocate, Input input, std::function<bool(T &)> keep) {
            create(part.data(), part.size(), nallocate, input, std::move(keep));
        }

        template <class T>
        void MPIParticles<T>::create(T * part,
                                     size_t NumPartinpart,
                                     size_t nallocate,
                                     Input input,
                                     std::function<bool(T &)> keep) {
            // The domain (the given one, else the global one)
            set_domain_for_create();
            if (domain->ncoordinates() > 1) {
                T tmp{};
                const std::string msg = "[MPIParticles::create] The " + domain->type_name() + " domain needs NDIM " +
                                        (domain->ncoordinates() == 2 ? ">= 2" : "= 3");
                const int ndim = mpiparticles_ndim(tmp);
                assert_mpi(domain->ncoordinates() == 2 ? ndim >= 2 : ndim == 3, msg.c_str());
            }
            // With one task every particle is ours
            if (FML::NTasks == 1 and input == Input::Distributed)
                input = Input::Replicated;
            auto use = [&](T & particle) { return not keep or keep(particle); };

#ifdef DEBUG_MPIPARTICLES
            if (ThisTask == 0) {
                for (int i = 0; i < NTasks; i++)
                    if (domain->is_slab())
                        std::cout << "Task: " << i << " / " << NTasks << " xmin: " << domain->x_min(i)
                                  << " xmax: " << domain->x_max(i) << "\n";
            }
#endif

            // Allocate memory
            p.resize(nallocate);
            add_memory_label("MPIParticles::create");

            // The particles are ours: take them as given
            if (input == Input::Local) {
                size_t count = 0;
                for (size_t i = 0; i < NumPartinpart; i++) {
                    if (not use(part[i]))
                        continue;
                    assert_mpi(count < nallocate,
                               "[MPIParticle::create] Reached allocation limit. Increase nallocate\n");
                    p[count++] = part[i];
                }
                NpartLocal_in_use = count;
            }

            // If all tasks has the same particles we keep the particles we own
            if (input == Input::Replicated) {
                size_t count = 0;
                for (size_t i = 0; i < NumPartinpart; i++) {
                    if (not use(part[i]))
                        continue;
                    T particle = part[i];
                    wrap_boundary_positions(particle);
                    if (domain->contains(ThisTask, FML::PARTICLE::GetPos(particle))) {
                        // Check that we are not past allocation limit before writing
                        assert_mpi(count < nallocate,
                                   "[MPIParticle::create] Reached allocation limit. Increase nallocate\n");
                        p[count++] = particle;
                    }
                }
                NpartLocal_in_use = count;
            }

#ifdef USE_MPI
            std::cout << std::flush;
            MPI_Barrier(FML::communicator);
#endif

            // If the tasks have different particles then read in batches and communicate as we go along
            // just in case the total amount of particles are too large
            if (input == Input::Distributed) {

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
                        size_t i = start;
                        for (; i < NumPartinpart and count < nallocate; i++) {
                            if (not use(part[i]))
                                continue;
                            p[count] = part[i];
                            wrap_boundary_positions(p[count]);
                            count++;
                        }
                        start = i;
                        more_to_process_locally = start < NumPartinpart;
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
        void MPIParticles<T>::check_domain(const FML::Domain & d) {
            const std::string msg = "[MPIParticles] The particle domain is made for " + std::to_string(d.ntasks()) +
                                    " tasks, but we have " + std::to_string(FML::NTasks);
            assert_mpi(d.ntasks() == FML::NTasks, msg.c_str());
            if (not d.is_slab()) {
                T tmp{};
                try {
                    d.check_ndim(mpiparticles_ndim(tmp));
                } catch (const std::exception & e) {
                    assert_mpi(false, e.what());
                }
            }
        }

        template <class T>
        void MPIParticles<T>::set_domain(const FML::Domain & new_domain) {
            check_domain(new_domain);
            // (has_value is the same on all tasks: set by the constructor, create and load)
            const bool communicate = domain.has_value();
            domain = new_domain;
            domain_given = true;
            if (communicate)
                communicate_particles();
        }

        template <class T>
        MPIParticles<T>::MPIParticles(const FML::Domain & particle_domain_in)
            : domain(particle_domain_in), domain_given(true) {
            check_domain(*domain);
        }

        template <class T>
        void MPIParticles<T>::set_domain_for_create() {
            if (not domain_given)
                domain = FML::get_domain();
        }

        template <class T>
        void MPIParticles<T>::wrap_boundary_positions(T & particle) const {
            auto * pos = FML::PARTICLE::GetPos(particle);
            for (int idim = 0; idim < domain->ncoordinates(); idim++)
                mpiparticles_wrap_boundary_position(pos[idim]);
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
        void MPIParticles<T>::create_particle_grid(int Npart_1D, double buffer_factor) {
            // The domain (the given one, else the global one): a Slab (the lattice slices in [xmin, xmax)) or a Pencil
            // (the lattice points in the pencil [x0, x1) x [y0, y1))
            set_domain_for_create();
            // (the domain is the same on every task, so every task throws)
            if (not domain->is_slab() and not domain->template holds<FML::PencilDomain>())
                throw std::invalid_argument("[MPIParticles::create_particle_grid] only implemented for the Slab and "
                                            "Pencil particle domains, not " +
                                            domain->type_name());
            T tmp_ndim; // (GetNDIM(T()) needs a particle that can be built at compile time)
            constexpr int ndim = FML::PARTICLE::GetNDIM(tmp_ndim);

            // The local edges along x and y
            std::array<double, 2> lo{0.0, 0.0}, hi{1.0, 1.0};
            if (domain->is_slab()) {
                lo[0] = domain->x_min(ThisTask);
                hi[0] = domain->x_max(ThisTask);
            } else {
                assert_mpi(ndim >= 2, "[MPIParticles::create_particle_grid] A Pencil domain needs ndim >= 2");
                const auto & pencils = domain->template get<FML::PencilDomain>();
                const auto c = pencils.coords_of(ThisTask);
                lo = {pencils.x_edges()[c[0]], pencils.y_edges()[c[1]]};
                hi = {pencils.x_edges()[c[0] + 1], pencils.y_edges()[c[1] + 1]};
            }

            // The lattice points i / Npart_1D in [lo, hi) along x and y
            std::array<int, 2> Local_p_start{0, 0}, Local_Npart_1D{Npart_1D, Npart_1D};
            for (int idim = 0; idim < std::min(ndim, 2); idim++) {
                int imin = 0;
                while (imin / double(Npart_1D) < lo[idim]) {
                    imin++;
                }
                int imax = imin;
                while (imax / double(Npart_1D) < hi[idim]) {
                    imax++;
                }
                Local_p_start[idim] = imin;
                Local_Npart_1D[idim] = imax - imin;
            }

            // Sanity check (Slab: the slices of the tasks follow each other; all: every lattice point is somewhere)
            long long int local_count = Local_Npart_1D[0] * (ndim >= 2 ? (long long int)Local_Npart_1D[1] : 1LL);
            for (int idim = 2; idim < ndim; idim++)
                local_count *= Npart_1D;
            if (domain->is_slab()) {
                auto Local_Npart_1D_per_task = FML::GatherFromTasks(&Local_Npart_1D[0]);
                int Local_p_start_computed = 0, Npart_1D_computed = 0;
                for (int i = 0; i < NTasks; i++) {
                    if (i < ThisTask)
                        Local_p_start_computed += Local_Npart_1D_per_task[i];
                    Npart_1D_computed += Local_Npart_1D_per_task[i];
                }
                assert_mpi(Npart_1D_computed == Npart_1D,
                           "[MPIParticles::create_particle_grid] Computed Npart does not match Npart\n");
                assert_mpi(Local_p_start_computed == Local_p_start[0],
                           "[MPIParticles::create_particle_grid] Computed Local_p_start does not match Local_p_start\n");
            } else {
                long long int total = local_count;
                FML::SumOverTasks(&total);
                assert_mpi(total == FML::power((long long int)Npart_1D, ndim),
                           "[MPIParticles::create_particle_grid] The pencils do not have all the lattice points\n");
            }

#ifdef DEBUG_MPIPARTICLES
            // Output some info
            for (int taskid = 0; taskid < NTasks; taskid++) {
                if (ThisTask == 0 and domain->is_slab()) {
                    std::cout << "Task[" << taskid << "]  xmin: " << domain->x_min(taskid)
                              << " xmax: " << domain->x_max(taskid) << "\n";
                }
            }
#endif

            // Total number of particles
            NpartTotal = FML::power(Npart_1D, ndim);
            NpartLocal_in_use = size_t(local_count);

            // Allocate particle struct. The capacity is buffer_factor times the largest number of lattice points that
            // the local region can hold (for any uniform shift of the lattice), not times the number we got: with
            // Npart_1D not divisible by the number of parts a task can get fewer points than its share of the volume
            // (or none) and would otherwise have no room for particles moving in. When the domain holds a whole number
            // of lattice planes (the usual case) this is the same as buffer_factor * NpartLocal_in_use
            size_t max_points = 1;
            for (int idim = 0; idim < ndim; idim++) {
                const double width = idim < 2 ? hi[idim] - lo[idim] : 1.0;
                const double nplanes = double(Npart_1D) * width;
                max_points *= size_t(std::max(0.0, std::ceil(nplanes - 1e-9)));
            }
            size_t NpartToAllocate = size_t(std::ceil(std::max(buffer_factor, 1.0) * double(max_points)));
            NpartToAllocate = std::max(NpartToAllocate, size_t(NpartLocal_in_use));
            assert_mpi(NpartToAllocate >= NpartLocal_in_use,
                       "[MPIParticles::create_particle_grid] Allocation is smaller than the number of particles\n");
            p.resize(NpartToAllocate);
            add_memory_label("MPIParticles::create_particle_grid");

            // The local lattice points (x slowest, the last coordinate fastest)
            std::vector<int> coord(ndim, 0), cstart(ndim, 0), cend(ndim, Npart_1D);
            for (int idim = 0; idim < std::min(ndim, 2); idim++) {
                cstart[idim] = Local_p_start[idim];
                cend[idim] = Local_p_start[idim] + Local_Npart_1D[idim];
            }
            coord = cstart;
            for (size_t i = 0; i < NpartLocal_in_use; i++) {
                auto * Pos = FML::PARTICLE::GetPos(p[i]);

                // Position regular grid
                for (int idim = 0; idim < ndim; idim++) {
                    Pos[idim] = coord[idim] / double(Npart_1D);
                }

                // The next lattice point
                int idim = ndim - 1;
                while (++coord[idim] == cend[idim]) {
                    coord[idim] = cstart[idim];
                    if (--idim < 0)
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
            assert_mpi(NpartLocal_in_use <= p.size() && Npart_recv <= p.max_size() - NpartLocal_in_use,
                       "[MPIParticles::copy_over_received_data] Too many particles received\n");

            // More particles than we have room for: grow the storage (with 25% headroom) rather than abort. NB: this
            // invalidates pointers and references to the particles (get_particles_ptr() etc. must be called again)
            if (Npart_recv > p.size() - NpartLocal_in_use) {
                const size_t needed = NpartLocal_in_use + Npart_recv;
                const size_t newsize = std::max(needed, needed + needed / 4);
                std::cout << "[MPIParticles] Task " << FML::ThisTask << " receives more particles than it has room "
                          << "for: growing the capacity from " << p.size() << " to " << newsize << "\n";
                p.resize(newsize);
#ifdef MEMORY_LOGGING
                FML::MemoryLog::get()->add_label(p.data(), p.capacity(), "MPIParticles");
#endif
            }

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

        template <class T>
        void MPIParticles<T>::communicate_particles([[maybe_unused]] size_t max_message_bytes,
                                                    [[maybe_unused]] bool force_general) {
            if (FML::NTasks == 1) {
                last_migration_route = MigrationRoute::Serial;
                return;
            }
#ifdef USE_MPI

            FML::COMMUNICATION::Count max_chunk_bytes =
                std::min(max_message_bytes, size_t(std::numeric_limits<int>::max()));
            FML::MinOverTasks(&max_chunk_bytes);
            FML::COMMUNICATION::require_count(max_chunk_bytes > 0, "[MPIParticles] Message limit must be positive");

            // The domain must be made for these tasks (a Slab: a finite, contiguous rank-ordered partition of [0,1]
            // for the binary lookup; empty slabs are allowed)
            FML::COMMUNICATION::require_count(domain.has_value(),
                                              "[MPIParticles] The particles have not been created (no domain)");
            const FML::Domain & dom = *domain;
            bool valid_partition = dom.ntasks() == NTasks;
            if (valid_partition && dom.is_slab())
                valid_partition = dom.valid_slab_partition();
            FML::COMMUNICATION::require_count(valid_partition,
                                              dom.is_slab() ?
                                                  "[MPIParticles] Slabs must partition [0,1] in rank order" :
                                                  "[MPIParticles] The particle domain is not made for NTasks");
            // The domain reads the first ncoord coordinates of the positions (Slab: x; Pencil: x, y; PeanoHilbert:
            // x, y, z). The particles need that many (get_ndim is only asked for then: some particle types have none)
            const int ncoord = dom.ncoordinates();
            if (ncoord > 1) {
                T tmp_ndim{};
                const std::string msg = "[MPIParticles::communicate_particles] The " + dom.type_name() +
                                        " domain needs NDIM " + (ncoord == 2 ? ">= 2" : "= 3");
                const int ndim = mpiparticles_ndim(tmp_ndim);
                assert_mpi(ncoord == 2 ? ndim >= 2 : ndim == 3, msg.c_str());
            }
            // The neighbours in rank order (all traffic to them: the neighbour route)
            const int left = int((static_cast<long long>(ThisTask) + NTasks - 1) % NTasks);
            const int right = int((static_cast<long long>(ThisTask) + 1) % NTasks);
            // The task that owns a particle (positions in [0,1)), the neighbours first (most departures end there)
            auto task_of = [&](T & particle) { return dom.owner_near(ThisTask, FML::PARTICLE::GetPos(particle)); };
            int route = force_general ? 2 : 0; // 0: no traffic, 1: neighbors, 2: general (including forced).

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
            auto count_outgoing = [&](int task, T & particle) {
                route = std::max(route, task == left || task == right ? 1 : 2);
                auto bytes = FML::PARTICLE::GetSize(particle);
                valid_counts = valid_counts && bytes > 0;
                if (bytes > 0) {
                    valid_counts &= FML::COMMUNICATION::add_count(n_to_send[task], 1);
                    valid_counts &= FML::COMMUNICATION::add_count(nbytes_to_send[task], Count(bytes));
                }
            };
            size_t i = 0;
            while (i < NpartLocal_in_use) {
                auto * pos = FML::PARTICLE::GetPos(p[i]);
                for (int idim = 0; idim < ncoord; idim++) {
                    auto & x = pos[idim];

                    // To fix issues appearing when we are exactly on the boundary
                    if (x == 1.0)
                        x = std::nextafter(x, PosType(0.0));
                    if (x == 0.0)
                        x = std::numeric_limits<PosType>::min();

                    // Particles must be wrapped inside [0,1) (this also catches NaN)
                    if (not(x >= 0.0 and x < 1.0)) {
                        std::string errmsg = "[MPIParticles::communicate_particles] Particle with " +
                                             std::string(idim == 0 ? "x" : (idim == 1 ? "y" : "z")) + " = " +
                                             std::to_string(double(x)) +
                                             " is outside [0,1). Particles must be wrapped (periodic) before "
                                             "communication\n";
                        assert_mpi(false, errmsg.c_str());
                    }
                }

                if (not dom.contains(ThisTask, pos)) {
                    const int taskid = task_of(p[i]);
                    count_outgoing(taskid, p[i]);
                    swap_particles(p[i], p[--NpartLocal_in_use]);
                } else {
                    i++;
                }
            }

            // Communicate to get how many particles and bytes to receive from each task
            FML::COMMUNICATION::require_count(valid_counts, "[MPIParticles] Outgoing particle/byte count overflow");
            FML::MaxOverTasks(&route);
            if (route == 0) {
                last_migration_route = MigrationRoute::NoTraffic;
                return;
            }
            last_migration_route = route == 1 ? MigrationRoute::Neighbors : MigrationRoute::General;
            std::vector<int> peer_steps;
            if (route == 1) {
                peer_steps.push_back(1);
                if (NTasks > 2)
                    peer_steps.push_back(NTasks - 1);
                for (int step : peer_steps) {
                    const int dest = int((static_cast<long long>(ThisTask) + step) % NTasks);
                    const int source = int((static_cast<long long>(ThisTask) - step + NTasks) % NTasks);
                    const Count outgoing[2] = {n_to_send[dest], nbytes_to_send[dest]};
                    Count incoming[2]{};
                    MPI_Sendrecv(outgoing,
                                 2,
                                 MPI_UNSIGNED_LONG_LONG,
                                 dest,
                                 0,
                                 incoming,
                                 2,
                                 MPI_UNSIGNED_LONG_LONG,
                                 source,
                                 0,
                                 FML::communicator,
                                 MPI_STATUS_IGNORE);
                    n_to_recv[source] = incoming[0];
                    nbytes_to_recv[source] = incoming[1];
                }
            } else {
                for (int step = 1; step < NTasks; ++step)
                    peer_steps.push_back(step);
                MPI_Alltoall(n_to_send.data(),
                             1,
                             MPI_UNSIGNED_LONG_LONG,
                             n_to_recv.data(),
                             1,
                             MPI_UNSIGNED_LONG_LONG,
                             FML::communicator);
                MPI_Alltoall(nbytes_to_send.data(),
                             1,
                             MPI_UNSIGNED_LONG_LONG,
                             nbytes_to_recv.data(),
                             1,
                             MPI_UNSIGNED_LONG_LONG,
                             FML::communicator);
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

            // More particles arrive than fit: grow the storage (copy_over_received_data does it)
            FML::COMMUNICATION::require_count(NpartLocal_in_use <= p.size() &&
                                                  ntot_to_recv <= p.max_size() - NpartLocal_in_use,
                                              "[MPIParticles] Receive count exceeds the maximum particle capacity");

            // Allocate send buffer
            std::vector<char> send_buffer(ntot_bytes_to_send);
            std::vector<char> recv_buffer(ntot_bytes_to_recv);
            FML::PERFORMANCE::transport_allocation(send_buffer.capacity());
            FML::PERFORMANCE::transport_allocation(recv_buffer.capacity());

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
                const int taskid = task_of(p[index]);
                assert_mpi(taskid != ThisTask, "[MPIParticles] Outgoing particle is still locally owned");
                append_particle(index, taskid);
            }

            for (int task = 0; task < NTasks; ++task)
                valid_counts &= packed[task] == size_t(nbytes_to_send[task]);
            FML::COMMUNICATION::require_count(valid_counts,
                                              "[MPIParticles] Serialized particle sizes changed while packing");
            FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::TransportPackedBytes, ntot_bytes_to_send);

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

            for (int step : peer_steps) {
                int send_request_to = int((static_cast<long long>(ThisTask) + step) % NTasks);
                int get_request_from = int((static_cast<long long>(ThisTask) - step + NTasks) % NTasks);

                for (Count ichunk = 0; ichunk < nchunks; ichunk++) {
                    const Count offset = ichunk * max_chunk_bytes;
                    const int nsend = int(offset < nbytes_to_send[send_request_to] ?
                                              std::min(nbytes_to_send[send_request_to] - offset, max_chunk_bytes) :
                                              0);
                    const int nrecv = int(offset < nbytes_to_recv[get_request_from] ?
                                              std::min(nbytes_to_recv[get_request_from] - offset, max_chunk_bytes) :
                                              0);
                    char * send_ptr = nsend ? send_buffer_by_task[send_request_to] + size_t(offset) : nullptr;
                    char * recv_ptr = nrecv ? recv_buffer_by_task[get_request_from] + size_t(offset) : nullptr;

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
                    FML::PERFORMANCE::payload(size_t(nsend), size_t(nrecv));
                }
            }

            // Copy over the particle data (this also updates the total number of particles)
            copy_over_received_data(recv_buffer, ntot_to_recv);
            FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::TransportUnpackedBytes, ntot_bytes_to_recv);
#endif
        }

        template <class T>
        void MPIParticles<T>::dump_to_file(std::string fileprefix, size_t max_bytesize_buffer) {
            // Write to a temporary file and rename it when complete, so a crash or full disk while writing leaves
            // any previous file (e.g. the last checkpoint) intact
            const std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
            const std::string tmpfilename = filename + ".tmp";
            auto myfile = std::fstream(tmpfilename, std::ios::out | std::ios::binary | std::ios::trunc);

            // A checkpoint we cannot write is an error (as for FFTWGrid::dump_to_file)
            if (not myfile.good()) {
                std::string error = "[MPIParticles::dump_to_file] Failed to open the file for writing on task " +
                                    std::to_string(FML::ThisTask) + " Filename: " + tmpfilename + "\n";
                assert_mpi(false, error.c_str());
            }
            auto finish = [&]() {
                myfile.close();
                if (myfile.fail()) {
                    std::string error = "[MPIParticles::dump_to_file] Error writing the particle data on task " +
                                        std::to_string(FML::ThisTask) + " Filename: " + tmpfilename + "\n";
                    assert_mpi(false, error.c_str());
                }
                if (std::rename(tmpfilename.c_str(), filename.c_str()) != 0) {
                    std::string error = "[MPIParticles::dump_to_file] Failed to rename " + tmpfilename + " to " +
                                        filename + " on task " + std::to_string(FML::ThisTask) + "\n";
                    assert_mpi(false, error.c_str());
                }
            };

            // Write the header: magic string, version, tasks, NDIM, sizeof(T), the counts and the domain
            MPIParticlesFileHeader header;
            header.ntasks = FML::NTasks;
            header.thistask = FML::ThisTask;
            header.ndim = FML::PARTICLE::GetNDIM(T());
            header.sizeof_particle = sizeof(T);
            header.npart_total = NpartTotal;
            header.npart_local = NpartLocal_in_use;
            header.npart_allocated = p.size();
            myfile.write(reinterpret_cast<const char *>(&header), sizeof(header));
            get_domain().write(myfile);

            if (NpartLocal_in_use == 0) {
                finish();
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
            finish();
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

            // Read and check the header
            MPIParticlesFileHeader header;
            const MPIParticlesFileHeader expected;
            myfile.read(reinterpret_cast<char *>(&header), sizeof(header));
            assert_mpi(myfile.good() and
                           std::equal(std::begin(header.magic), std::end(header.magic), std::begin(expected.magic)),
                       "[MPIParticles::load_from_file] Not a file of MPIParticles::dump_to_file (or one written by "
                       "an older version of FML)\n");
            assert_mpi(header.version == expected.version,
                       "[MPIParticles::load_from_file] The file has another version of the format\n");
            assert_mpi(header.ntasks == FML::NTasks and header.thistask == FML::ThisTask,
                       "[MPIParticles::load_from_file] The file was written with another number of tasks\n");
            assert_mpi(header.ndim == FML::PARTICLE::GetNDIM(T()) and header.sizeof_particle == sizeof(T),
                       "[MPIParticles::load_from_file] The file has another particle type (NDIM or size)\n");
            assert_mpi(header.npart_local <= header.npart_allocated,
                       "[MPIParticles::load_from_file] Error reading the header of the file\n");
            FML::Domain file_domain;
            try {
                file_domain = FML::Domain::read(myfile);
            } catch (const std::exception & e) {
                assert_mpi(false, e.what());
            }
            assert_mpi(not domain_given or file_domain == *domain,
                       "[MPIParticles::load_from_file] The file has another domain than the one given to the "
                       "constructor\n");
            assert_mpi(file_domain.ntasks() == FML::NTasks,
                       "[MPIParticles::load_from_file] The domain of the file is not made for this number of tasks\n");
            domain = file_domain;
            NpartTotal = header.npart_total;
            NpartLocal_in_use = header.npart_local;
            const size_t NpartLocalAllocated = header.npart_allocated;

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
