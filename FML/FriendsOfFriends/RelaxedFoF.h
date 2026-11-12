#ifndef FML_RELAXED_FOF_HEADER
#define FML_RELAXED_FOF_HEADER

#include <FML/FriendsOfFriends/FoF.h>
#include <FML/FriendsOfFriends/FoFBinning.h>
#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif

namespace FML {
    namespace FOF {

        //=============================================================================================================
        // Relaxed friends-of-friends (Dai, Feng, Seljak & Singh 2019, arXiv:1908.05276, sec. 2.1, Algorithm 1):
        // FoF with a linking length that depends on the size of the halo (longer for small halos, which are too
        // diffuse in low-resolution simulations) and the redshift, and rejection of "fake" halos (unbound clumps in
        // dense regions linked by the longer linking length) by their velocity dispersion.
        //
        // The halos are split into bins by particle number, N < N_1 = 20, 40, 80, 160, 320, inf (eq. 2.2; nbins =
        // bin_edges.size() + 1). Pass i = 1..nbins links the particles that are still active with
        //     l_1(z)     = l1 - a1 / (1 + z)                            (eq. 2.3, the smallest halos)
        //     l_n(z)     = max(l6 - a2 / (1 + z), linking_length_min)    (eq. 2.4, the largest halos)
        //     l_i(z)     = [(n - i) l_1 + (i - 1) l_n] / (n - 1)       (eq. 2.5, in between)
        // in units of the mean particle separation of the whole simulation. A group (of at least nmin particles) is
        // saved if it has fewer than N_i particles and is not fake:
        //     sigma_v < r0(z) V_std(M, z),   V_std = V0 (E(z) M / 1e15 Msun/h)^(1/3)  (eq. 2.1, Evrard et al. 2008),
        //     r0(z) = b1 - b2 ln(1 + z)                                                (eq. 2.6)
        // with sigma_v the 3D velocity dispersion sqrt(sum over axes of the dispersions) in peculiar km/s and M the
        // halo mass in Msun/h. Its particles are then removed. The particles that are in no group are removed too; the
        // others (in larger groups or in rejected ones) go on to the next pass. Each pass therefore only splits groups
        // of the previous one, and every particle ends up in at most one halo.
        //
        // Modes:
        //   fastpm: as FastPM's implementation (libfastpm/rfof.c): the nbins passes only; the particles of halos
        //           rejected in the last pass are dropped. The default parameters are FastPM's (fitted to Illustris
        //           TNG300-2-Dark with FastPM at a mean separation of 0.3 Mpc/h and 40 steps).
        //   paper:  as Algorithm 1 of the paper: after the nbins passes the linking length is reduced by
        //           shrink_factor (0.9) per pass, saving the halos that pass the velocity cut (any size), until no
        //           particles are left (or the linking length is below shrink_linking_length_min).
        //
        // NB: with the default b1, b2, r0(z) is 0.40 at z = 2 and negative for z > 2.2 (all halos rejected); the
        // parameters were fitted at z <= 2. Set r0_min to put a floor on r0.
        //
        // The particles are not reordered (unlike FriendsOfFriends) and not changed, except that the halo ID is set
        // if the particle has a set_fofid method. MPI: the same slab decomposition as FriendsOfFriends.
        //=============================================================================================================

        /// The settings of RelaxedFriendsOfFriends (the defaults are FastPM's)
        struct RelaxedFoFParameters {
            enum class Mode { fastpm, paper };
            Mode mode{Mode::fastpm};
            /// The linking lengths (units of the mean particle separation), eqs. 2.3-2.5
            double l1{0.25};
            double l6{0.24};
            double a1{0.012};
            double a2{0.06};
            double linking_length_min{0.2};
            /// The upper edges N_1, ..., N_{n-1} of the halo bins (the last bin has no upper edge), eq. 2.2
            std::vector<size_t> bin_edges{20, 40, 80, 160, 320};
            /// The smallest group (particles)
            int nmin{8};
            /// The velocity-dispersion cut, eqs. 2.1 and 2.6
            bool velocity_rejection{true};
            double b1{7.02};
            double b2{6.025};
            double r0_min{-std::numeric_limits<double>::infinity()};
            double V0_kms{1100.0};
            /// true: sigma_v = sqrt(sum of the per-axis dispersions^2) (FastPM); false: the 1D dispersion
            bool velocity_dispersion_3d{true};
            /// paper mode: the linking length is multiplied by this after the bins until this minimum
            double shrink_factor{0.9};
            double shrink_linking_length_min{0.02};
            /// Print a line per pass on task 0
            bool verbose{true};

            size_t nbins() const { return bin_edges.size() + 1; }

            /// The linking length of bin i = 1..nbins() at redshift z
            double linking_length(size_t i, double z) const {
                const double first = l1 - a1 / (1.0 + z);
                const double last = std::max(l6 - a2 / (1.0 + z), linking_length_min);
                const size_t n = nbins();
                if (i < 1 or i > n)
                    throw std::invalid_argument("RelaxedFoFParameters::linking_length: bin out of range");
                if (n == 1 or i == 1)
                    return first;
                if (i == n)
                    return last;
                return (double(n - i) * first + double(i - 1) * last) / double(n - 1);
            }

            /// The largest number of particles + 1 of a halo saved in bin i (no limit for the last bin)
            size_t bin_upper_edge(size_t i) const {
                return i < nbins() ? bin_edges[i - 1] : std::numeric_limits<size_t>::max();
            }

            /// The rejection threshold r0(z)
            double r0(double z) const { return std::max(b1 - b2 * std::log(1.0 + z), r0_min); }

            /// The velocity dispersion of the scaling relation (km/s) for mass M (Msun/h) and E(z) = H(z)/H0
            double V_std(double M_msunh, double E_of_z) const { return V0_kms * std::cbrt(E_of_z * M_msunh / 1e15); }

            void validate() const {
                if (nmin < 2)
                    throw std::invalid_argument("RelaxedFoF: nmin must be >= 2");
                for (size_t k = 0; k < bin_edges.size(); k++)
                    if (bin_edges[k] < 1 or (k > 0 and bin_edges[k] <= bin_edges[k - 1]))
                        throw std::invalid_argument("RelaxedFoF: bin_edges must be increasing and >= 1");
                if (not(shrink_factor > 0.0 and shrink_factor < 1.0))
                    throw std::invalid_argument("RelaxedFoF: shrink_factor must be in (0, 1)");
                if (not(V0_kms > 0.0))
                    throw std::invalid_argument("RelaxedFoF: V0_kms must be > 0");
            }
        };

        /// The particle the passes link (a copy of the position, velocity and mass of the active particles)
        template <int NDIM>
        struct RelaxedFoFParticle {
            double pos[NDIM];
            double vel[NDIM];
            double mass;
            size_t index; // in the particles given to RelaxedFriendsOfFriends
            size_t fofid;
            double * get_pos() { return pos; }
            double * get_vel() { return vel; }
            double get_mass() const { return mass; }
            void set_fofid(size_t id) { fofid = id; }
            size_t get_fofid() const { return fofid; }
            constexpr int get_ndim() const { return NDIM; }
        };

        /// A halo of RelaxedFriendsOfFriends: the FoFHalo quantities (id, np, mass, pos, vel, vel_rms in the units of
        /// the particles) and where and how it was found
        template <int NDIM>
        struct RelaxedFoFHalo : public FoFHalo<RelaxedFoFParticle<NDIM>, NDIM> {
            using FoFHalo<RelaxedFoFParticle<NDIM>, NDIM>::FoFHalo;
            int pass{0};               // 0, 1, ...
            int bin{0};                // 1..nbins, or 0 for the shrinking passes (paper mode)
            double linking_length{0};  // in units of the mean particle separation
            double mass_msunh{0};      // mass * mass_unit_msunh
            double vdisp_kms{0};       // the velocity dispersion used for the cut
            double vdisp_ratio{0};     // vdisp_kms / V_std(mass_msunh)
        };

        /// What happened in each pass (sums over the tasks)
        struct RelaxedFoFPass {
            int pass{0}, bin{0};
            double linking_length{0}, r0{0};
            size_t active_particles{0}; // linked in this pass
            size_t groups{0};           // with at least nmin particles
            size_t saved{0};
            size_t rejected_velocity{0}; // fake: failed the velocity cut
            size_t too_large{0};         // passed on to the next bin
            size_t particles_saved{0};
        };

        namespace detail {
            /// Send a list to the left and right neighbor tasks and return the lists of both neighbors
            inline std::vector<size_t> exchange_with_neighbors([[maybe_unused]] const std::vector<size_t> & mine) {
                std::vector<size_t> result;
#ifdef USE_MPI
                if (FML::NTasks == 1)
                    return result;
                const int right = (FML::ThisTask + 1) % FML::NTasks;
                const int left = (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks;
                for (int dir = 0; dir < 2; dir++) {
                    const int to = dir == 0 ? right : left;
                    const int from = dir == 0 ? left : right;
                    unsigned long long nsend = mine.size(), nrecv = 0;
                    MPI_Sendrecv(&nsend, 1, MPI_UNSIGNED_LONG_LONG, to, 10 + dir, &nrecv, 1, MPI_UNSIGNED_LONG_LONG,
                                 from, 10 + dir, FML::communicator, MPI_STATUS_IGNORE);
                    std::vector<size_t> recv(static_cast<size_t>(nrecv));
                    const int bytes_send = int(FML::COMMUNICATION::checked_bytes(mine.size(), sizeof(size_t), INT_MAX));
                    const int bytes_recv = int(FML::COMMUNICATION::checked_bytes(recv.size(), sizeof(size_t), INT_MAX));
                    MPI_Sendrecv(mine.data(), bytes_send, MPI_BYTE, to, 20 + dir, recv.data(), bytes_recv, MPI_BYTE,
                                 from, 20 + dir, FML::communicator, MPI_STATUS_IGNORE);
                    result.insert(result.end(), recv.begin(), recv.end());
                }
#endif
                return result;
            }

            /// Restores FoF_verbose when it goes out of scope
            struct QuietFoF {
                bool old{FoF_verbose};
                QuietFoF() { FoF_verbose = false; }
                ~QuietFoF() { FoF_verbose = old; }
            };
        } // namespace detail

        //=============================================================================================================
        /// Relaxed friends-of-friends (see the top of this file). Returns the halos kept by this task (as
        /// FriendsOfFriends: a halo is kept by the task where its center is).
        ///
        /// @param[in] part The particles (not reordered; positions in [0,1)^NDIM). The halo ID is stored with
        ///   set_fofid if the particle has it (no_FoF_ID if the particle is in no halo).
        /// @param[in] NumPart The number of local particles (may be zero on some of the tasks)
        /// @param[in] params The settings
        /// @param[in] redshift, E_of_z The redshift and H(z)/H0
        /// @param[in] mass_unit_msunh Multiplies the halo mass (the sum of get_mass of its particles, 1 per particle
        ///   without get_mass) to give Msun/h
        /// @param[in] velocity_unit_kms Multiplies the particle velocities to give peculiar km/s
        /// @param[in] periodic, Buffersize_over_Boxsize, Ngrid As FriendsOfFriends
        /// @param[out] particle_halo_id (Optional) For each particle the ID of its halo or no_FoF_ID. The IDs are
        ///   unique over the tasks but not contiguous
        /// @param[out] passes (Optional) What happened in each pass
        /// @param[in] mean_particle_separation (Optional) The mean separation the linking lengths are in units of
        ///   (box units). <= 0: (total number of particles)^(-1/NDIM)
        //=============================================================================================================
        template <class T, int NDIM>
        std::vector<RelaxedFoFHalo<NDIM>> RelaxedFriendsOfFriends(T * part,
                                                                  size_t NumPart,
                                                                  const RelaxedFoFParameters & params,
                                                                  double redshift,
                                                                  double E_of_z,
                                                                  double mass_unit_msunh,
                                                                  double velocity_unit_kms,
                                                                  bool periodic,
                                                                  double Buffersize_over_Boxsize,
                                                                  std::vector<size_t> * particle_halo_id = nullptr,
                                                                  std::vector<RelaxedFoFPass> * passes = nullptr,
                                                                  double mean_particle_separation = 0.0,
                                                                  int Ngrid = FoF_Ngrid_max) {
            params.validate();
            using WorkParticle = RelaxedFoFParticle<NDIM>;
            using Halo = RelaxedFoFHalo<NDIM>;

            size_t NumPart_tot = NumPart;
            FML::SumOverTasks(&NumPart_tot);
            if (not(mean_particle_separation > 0.0))
                mean_particle_separation = NumPart_tot > 0 ? 1.0 / std::pow(double(NumPart_tot), 1.0 / NDIM) : 1.0;

            std::vector<char> active(NumPart, 1);
            std::vector<size_t> halo_of(NumPart, no_FoF_ID);
            std::vector<Halo> halos;
            std::vector<RelaxedFoFPass> info;
            size_t id_offset = 0;
            const double r0 = params.r0(redshift);
            if (params.velocity_rejection and r0 <= 0.0 and FML::ThisTask == 0)
                std::cout << "WARNING: [RelaxedFoF] r0(z = " << redshift << ") = " << r0
                          << " <= 0: every halo is rejected (set r0_min or the parameters b1, b2 for this redshift)\n";

            // One FoF pass with linking length ll; save(halo) decides which groups are saved
            auto run_pass = [&](int pass, int bin, double ll, size_t upper_edge) {
                RelaxedFoFPass p;
                p.pass = pass;
                p.bin = bin;
                p.linking_length = ll;
                p.r0 = r0;

                std::vector<WorkParticle> work;
                for (size_t i = 0; i < NumPart; i++) {
                    if (not active[i])
                        continue;
                    WorkParticle w{};
                    const auto * pos = FML::PARTICLE::GetPos(part[i]);
                    for (int d = 0; d < NDIM; d++)
                        w.pos[d] = pos[d];
                    if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                        const auto * vel = FML::PARTICLE::GetVel(part[i]);
                        for (int d = 0; d < NDIM; d++)
                            w.vel[d] = vel[d];
                    }
                    w.mass = FML::PARTICLE::GetMass(part[i]);
                    w.index = i;
                    w.fofid = no_FoF_ID;
                    work.push_back(w);
                }
                p.active_particles = work.size();
                FML::SumOverTasks(&p.active_particles);
                if (p.active_particles == 0) {
                    info.push_back(p);
                    return size_t(0);
                }

                std::vector<Halo> groups;
                {
                    detail::QuietFoF quiet;
                    FriendsOfFriends<WorkParticle, NDIM, Halo>(work.data(),
                                                               work.size(),
                                                               ll,
                                                               params.nmin,
                                                               periodic,
                                                               Buffersize_over_Boxsize,
                                                               groups,
                                                               Ngrid,
                                                               mean_particle_separation);
                }

                // Decide which of the groups kept here are saved
                std::vector<size_t> saved_ids;
                size_t max_id = 0;
                for (auto & g : groups) {
                    max_id = std::max(max_id, g.id + 1);
                    double s2 = 0.0;
                    for (int d = 0; d < NDIM; d++)
                        s2 += g.vel_rms[d] * g.vel_rms[d];
                    const double sigma =
                        velocity_unit_kms * std::sqrt(params.velocity_dispersion_3d ? s2 : s2 / double(NDIM));
                    g.mass_msunh = g.mass * mass_unit_msunh;
                    g.vdisp_kms = sigma;
                    g.vdisp_ratio = sigma / params.V_std(g.mass_msunh, E_of_z);
                    g.pass = pass;
                    g.bin = bin;
                    g.linking_length = ll;
                    const bool small_enough = g.np < upper_edge;
                    const bool real = not params.velocity_rejection or sigma < r0 * params.V_std(g.mass_msunh, E_of_z);
                    p.groups++;
                    if (small_enough and real) {
                        saved_ids.push_back(g.id);
                        p.saved++;
                        p.particles_saved += g.np;
                    } else if (not real) {
                        p.rejected_velocity++;
                    } else {
                        p.too_large++;
                    }
                }

                // The particles of a saved group can be on a neighbor task (the group is kept by the task of its
                // center, whose buffer reaches at most into the neighbor tasks)
                std::sort(saved_ids.begin(), saved_ids.end());
                std::vector<size_t> saved = saved_ids;
                const auto from_neighbors = detail::exchange_with_neighbors(saved_ids);
                saved.insert(saved.end(), from_neighbors.begin(), from_neighbors.end());
                std::sort(saved.begin(), saved.end());

                for (auto & w : work) {
                    if (w.fofid == no_FoF_ID) {
                        active[w.index] = 0; // in no group (of nmin or more particles)
                    } else if (std::binary_search(saved.begin(), saved.end(), w.fofid)) {
                        active[w.index] = 0;
                        halo_of[w.index] = w.fofid + id_offset;
                    }
                }
                for (auto & g : groups)
                    if (std::binary_search(saved_ids.begin(), saved_ids.end(), g.id)) {
                        g.id += id_offset;
                        halos.push_back(g);
                    }

                FML::MaxOverTasks(&max_id);
                id_offset += max_id;
                FML::SumOverTasks(&p.groups);
                FML::SumOverTasks(&p.saved);
                FML::SumOverTasks(&p.rejected_velocity);
                FML::SumOverTasks(&p.too_large);
                FML::SumOverTasks(&p.particles_saved);
                info.push_back(p);
                return p.active_particles;
            };

            int pass = 0;
            const size_t nbins = params.nbins();
            for (size_t i = 1; i <= nbins; i++, pass++)
                run_pass(pass, int(i), params.linking_length(i, redshift), params.bin_upper_edge(i));

            if (params.mode == RelaxedFoFParameters::Mode::paper) {
                double ll = params.linking_length(nbins, redshift);
                while (true) {
                    ll *= params.shrink_factor;
                    if (ll < params.shrink_linking_length_min)
                        break;
                    size_t remaining = 0;
                    for (size_t i = 0; i < NumPart; i++)
                        remaining += active[i] ? 1 : 0;
                    FML::SumOverTasks(&remaining);
                    if (remaining == 0)
                        break;
                    run_pass(pass++, 0, ll, std::numeric_limits<size_t>::max());
                }
            }

            if (params.verbose and FML::ThisTask == 0) {
                std::cout << "# RelaxedFoF (" << (params.mode == RelaxedFoFParameters::Mode::fastpm ? "fastpm" : "paper")
                          << " mode) z = " << redshift << " r0 = " << r0 << "\n";
                std::cout << "# pass bin  linking_length    active    groups     saved  rejected(v)  too_large\n";
                for (auto & p : info)
                    std::cout << "# " << std::setw(4) << p.pass << std::setw(4) << p.bin << std::setw(16)
                              << p.linking_length << std::setw(10) << p.active_particles << std::setw(10) << p.groups
                              << std::setw(10) << p.saved << std::setw(13) << p.rejected_velocity << std::setw(11)
                              << p.too_large << "\n";
            }

            if constexpr (FML::PARTICLE::has_set_fofid<T>())
                for (size_t i = 0; i < NumPart; i++)
                    FML::PARTICLE::SetFoFID(part[i], halo_of[i]);
            if (particle_halo_id)
                *particle_halo_id = std::move(halo_of);
            if (passes)
                *passes = info;
            return halos;
        }

    } // namespace FOF
} // namespace FML

#endif
