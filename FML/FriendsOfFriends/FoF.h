#ifndef FOF_HEADER
#define FOF_HEADER
#include <FML/FriendsOfFriends/FoFBinning.h>
#include <FML/Global/Domain.h>
#include <FML/Global/Global.h>
#include <FML/Global/MPICounts.h>
#include <FML/Timing/Timings.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <vector>

namespace FML {

    //========================================================================================
    /// This namespace deals how to link together particles in groups
    //========================================================================================
    namespace FOF {

        /// If a particle belongs to no FoF group it is given this FoF ID
        const constexpr size_t no_FoF_ID = std::numeric_limits<size_t>::max();

        /// Type of particle for book-keeping
        enum ParticleType { BOUNDARY_PARTICLE = -1, REGULAR_PARTICLE = 0 };

        /// The maximum grid we are allowed to use to bin the particle to
        /// This is just to have a memory limit in what the algorithm is
        /// allowed to allocate. This value is used if user do not provide it.
        inline int FoF_Ngrid_max = 1024;

        using Timings = FML::UTILS::Timings;
        /// Records timings of the FoF search
        inline Timings foftimer;

        /// The number of groups found in the last call to FriendsOfFriends (summed over tasks) that reach the edge of
        /// the buffer of particles copied from the neighbor tasks. These might be cut off (and then also found as a
        /// second, partial group on the neighbor task): increase the buffer if this is not zero
        inline size_t FoF_number_of_groups_reaching_buffer_edge{0};

        /// Print the banner, progress, timings and the number of groups (on task 0). Set to false to run quietly
        /// (e.g. when FoF is called many times, as in RelaxedFriendsOfFriends)
        inline bool FoF_verbose{true};

        //========================================================================================
        /// Locate friends of friends groups and bin what you want over these groups.
        /// The FoF linking definition is the standard cosmological method of Davis et al. (1985),
        /// https://doi.org/10.1086/163168; the grid search and MPI boundary handling here are FML's implementation.
        ///
        /// @tparam T The particle class
        /// @tparam NDIM The number of dimensions
        /// @tparam FoFHaloClass The FoF group class, determined what to be binned up over the FoF group. It needs the
        /// members of FoFHalo used here: a constructor from the (size_t) id, the members id and np, add(particle,
        /// periodic), finalize(periodic) and get_pos() (the center, used to decide which task keeps the group).
        /// A class derived from FoFHalo (with using FoFHalo::FoFHalo;) works.
        ///
        /// @param[in] part Pointer to particles. If the particle has a set_fofid then we will also store the FoFID in
        /// the particles: the id member of its group (FoFHalo::id), or no_FoF_ID if it is in no group that is kept.
        /// NB: the IDs are numbered before the groups smaller than nmin_FoF_group are removed, so they are not
        /// contiguous and not the index of the group in LocalFoFGroups: map them with LocalFoFGroups[i].id (over all
        /// tasks with MPI). NB: the particles are sorted (by their y-coordinate) in place.
        /// @param[in] NumPart Number of local particles (may be zero on some of the tasks)
        /// @param[in] linking_length The distance in units of mean particle separation (in [0,1)) required to have a
        /// link between two particles. Typical value is 0.2.
        /// @param[in] nmin_FoF_group Minimum number of particles in a FoF group to store it. Typical value is 20.
        /// @param[in] periodic Is the box periodic?
        /// @param[in] Buffersize_over_Boxsize The buffer we communicate on both sides of the domain in units of the
        /// boxsize. Typical value for N-body simulations is (3 Mpc/h) / boxsize.
        /// @param[out] LocalFoFGroups The results: list of FoF groups.
        /// @param[in] Ngrid (Optional) The maximum gridsize we use to bin the particles to in order to speed up the
        /// calculation. The default FoF_Ngrid_max is set in the header. Larger value means more memory needed. The
        /// grid used has at most about one cell per particle (the result does not depend on the grid).
        /// @param[in] mean_particle_separation (Optional) The mean particle separation (in units of the box) that the
        /// linking length is in units of. <= 0 (the default): from the number of particles given,
        /// (sum of NumPart)^(-1/NDIM). Give it to link a subset of the particles of a simulation with the linking
        /// length of the full simulation.
        ///
        //========================================================================================
        template <class T, int NDIM, class FoFHaloClass>
        void FriendsOfFriends(T * part,
                              size_t NumPart,
                              double linking_length,
                              int nmin_FoF_group,
                              bool periodic,
                              double Buffersize_over_Boxsize,
                              std::vector<FoFHaloClass> & LocalFoFGroups,
                              int Ngrid = FoF_Ngrid_max,
                              double mean_particle_separation = 0.0);

        //========================================================================================
        /// Friends-of-friends of particles on a domain (Slab or Pencil; FoFDomain.h): the same as above with the
        /// domain of the particles first (e.g. MPIParticles::get_domain()). The version above uses the default
        /// particle domain
        //========================================================================================
        template <class T, int NDIM, class FoFHaloClass>
        void FriendsOfFriends(const FML::Domain & domain,
                              T * part,
                              size_t NumPart,
                              double linking_length,
                              int nmin_FoF_group,
                              bool periodic,
                              double Buffersize_over_Boxsize,
                              std::vector<FoFHaloClass> & LocalFoFGroups,
                              int Ngrid = FoF_Ngrid_max,
                              double mean_particle_separation = 0.0);

        template <class T, int NDIM, class FoFHaloClass = FoFHalo<T, NDIM>>
        void FriendsOfFriends(T * part,
                              size_t NumPart,
                              double linking_length,
                              int nmin_FoF_group,
                              bool periodic,
                              double Buffersize_over_Boxsize,
                              std::vector<FoFHaloClass> & LocalFoFGroups,
                              int Ngrid,
                              double mean_particle_separation) {
            // The default particle domain
            FriendsOfFriends<T, NDIM, FoFHaloClass>(FML::get_default_particle_domain(),
                                                    part,
                                                    NumPart,
                                                    linking_length,
                                                    nmin_FoF_group,
                                                    periodic,
                                                    Buffersize_over_Boxsize,
                                                    LocalFoFGroups,
                                                    Ngrid,
                                                    mean_particle_separation);
        }
    } // namespace FOF
} // namespace FML
// FoF on a domain (Slab or Pencil)
#include <FML/FriendsOfFriends/FoFDomain.h>
#endif
