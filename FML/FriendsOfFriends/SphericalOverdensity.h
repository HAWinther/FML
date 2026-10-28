#ifndef SPHERICALOVERDENSITY_HEADER
#define SPHERICALOVERDENSITY_HEADER

#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace FML {
    namespace FOF {

        //========================================================================================
        /// The spherical overdensity (SO) properties of a halo around a given (guess for the) center
        //========================================================================================
        template <int NDIM>
        struct SOHalo {
            /// The center (shrinking sphere center of mass) in [0,1)
            std::array<double, NDIM> pos{};
            /// Mass weighted mean velocity and velocity dispersion (per dimension) of the particles inside the radius
            /// of the first overdensity
            std::array<double, NDIM> vel{};
            std::array<double, NDIM> vel_rms{};
            /// For each overdensity: radius (box units), mass (units of the particle masses) and number of particles
            std::vector<double> radius;
            std::vector<double> mass;
            std::vector<long long int> npart;
            /// The sphere did not reach the overdensity within the search radius (radius, mass are lower limits)
            bool truncated{false};
        };

        /// The number of halos in the last call to SphericalOverdensity (summed over tasks) whose sphere did not reach
        /// the overdensity within the search radius (the buffer)
        inline size_t SO_number_of_truncated_halos{0};

        //========================================================================================
        /// Copy the particles within dx of the left and right edge of the local domain from the neighbor tasks.
        /// The copies are unwrapped (x < 0 on the first task, x >= 1 on the last task) so that distances in x can be
        /// computed without wrapping when we have more than one task.
        //========================================================================================
        template <class T>
        std::vector<T> fetch_boundary_particles([[maybe_unused]] T * part,
                                                [[maybe_unused]] size_t NumPart,
                                                [[maybe_unused]] double dx) {
            std::vector<T> boundary;
#ifdef USE_MPI
            if (FML::NTasks == 1)
                return boundary;
            const double xmin = FML::xmin_domain;
            const double xmax = FML::xmax_domain;
            if (dx > xmax - xmin)
                throw std::runtime_error("[fetch_boundary_particles] The buffer is larger than the domain of a task. "
                                         "Reduce it (or the number of tasks)");
            const int RightTask = (FML::ThisTask + 1) % FML::NTasks;
            const int LeftTask = (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks;
            T tmp{};
            const size_t bytes = FML::PARTICLE::GetSize(tmp);
            auto exchange = [&](bool to_left) {
                std::vector<char> sendbuf;
                size_t nsend = 0;
                for (size_t i = 0; i < NumPart; i++) {
                    const double x = FML::PARTICLE::GetPos(part[i])[0];
                    if ((to_left and x < xmin + dx) or (not to_left and x >= xmax - dx)) {
                        sendbuf.resize((nsend + 1) * bytes);
                        FML::PARTICLE::AppendToBuffer(part[i], sendbuf.data() + nsend * bytes);
                        nsend++;
                    }
                }
                const int dest = to_left ? LeftTask : RightTask;
                const int src = to_left ? RightTask : LeftTask;
                size_t nrecv = 0;
                MPI_Status status;
                MPI_Sendrecv(&nsend,
                             sizeof(size_t),
                             MPI_BYTE,
                             dest,
                             10,
                             &nrecv,
                             sizeof(size_t),
                             MPI_BYTE,
                             src,
                             10,
                             FML::communicator,
                             &status);
                std::vector<char> recvbuf(nrecv * bytes);
                // In chunks below 2^31 bytes. All tasks do the same number of exchanges (the largest needed)
                const size_t chunk = (size_t(1) << 30) / bytes * bytes;
                const size_t sbytes = nsend * bytes, rbytes = nrecv * bytes;
                long long int nchunks = (long long int)((std::max(sbytes, rbytes) + chunk - 1) / chunk);
                MPI_Allreduce(MPI_IN_PLACE, &nchunks, 1, MPI_LONG_LONG, MPI_MAX, FML::communicator);
                for (long long int ic = 0; ic < nchunks; ic++) {
                    const size_t s0 = std::min(sbytes, size_t(ic) * chunk);
                    const size_t r0 = std::min(rbytes, size_t(ic) * chunk);
                    const size_t ns = std::min(chunk, sbytes - s0), nr = std::min(chunk, rbytes - r0);
                    MPI_Sendrecv(sendbuf.data() + s0,
                                 int(ns),
                                 MPI_BYTE,
                                 dest,
                                 11,
                                 recvbuf.data() + r0,
                                 int(nr),
                                 MPI_BYTE,
                                 src,
                                 11,
                                 FML::communicator,
                                 &status);
                }
                for (size_t i = 0; i < nrecv; i++) {
                    T p{};
                    FML::PARTICLE::AssignFromBuffer(p, recvbuf.data() + i * bytes);
                    auto * pos = FML::PARTICLE::GetPos(p);
                    // Particles from the right task (sent left) are to the right of us
                    if (to_left and FML::ThisTask == FML::NTasks - 1)
                        pos[0] += 1.0;
                    if (not to_left and FML::ThisTask == 0)
                        pos[0] -= 1.0;
                    boundary.push_back(p);
                }
            };
            exchange(true);
            exchange(false);
#endif
            return boundary;
        }

        //========================================================================================
        /// Spherical overdensity properties of halos around the given centers (e.g. the centers of FoF groups found
        /// on this task). For each halo:
        ///   1. the center is refined with a shrinking sphere (Power et al. 2003): starting from the guess and radius
        ///      initial_radius[i], move to the center of mass of the particles inside and shrink the radius by 10%
        ///      until fewer than max(nmin_center, 10% of the starting number) particles are left
        ///   2. the radius for each threshold (in units of the mean density of the particles, i.e. Delta) is the
        ///      OUTERMOST radius where the mean enclosed density is still above the threshold (as e.g. Rockstar):
        ///      with the particles sorted by distance r_i and M_i the mass of particles 0..i, the halo is particles
        ///      0..i for the largest i with M_i / (4pi/3 r_i^3) >= Delta rho_mean, and the radius is defined by
        ///      M_i / (4pi/3 R^3) = Delta rho_mean, so it lies between particle i and the first particle outside.
        ///      Growing the sphere from the center and stopping at the FIRST particle below the threshold is not
        ///      well defined at simulation resolution: the few innermost particles are Poisson noise around a
        ///      (shrinking sphere) center that is not on a particle, so the sphere often stops after 1-2 particles
        ///      even for halos with hundreds of particles inside R_Delta.
        /// All particles (on this task and within search_radius of it on the neighbor tasks) are used, not only the
        /// members of the FoF group. The search radius (box units) is the same for any number of tasks so the result
        /// does not depend on it; halos whose sphere has not reached the overdensity within it are flagged as truncated
        /// (and counted in SO_number_of_truncated_halos). Masses are in units of the particle masses (get_mass if the
        /// particle has it, otherwise 1).
        ///
        /// @param[in] part The particles on this task (positions in [0,1), all inside the local domain)
        /// @param[in] NumPart Number of particles on this task
        /// @param[in] total_mass Total mass of all the particles on all tasks (in the same units as the masses), i.e.
        /// the mean density in box units
        /// @param[in] centers Guesses for the centers
        /// @param[in] initial_radius The radius of the first shrinking sphere for each halo (box units)
        /// @param[in] thresholds The overdensities Delta in units of the mean density of the particles
        /// @param[in] search_radius The largest radius we look at (box units); also the buffer copied from the
        /// neighbor tasks
        /// @param[in] nmin_center Stop the shrinking sphere when fewer particles than this are left (fiducial 20)
        //========================================================================================
        template <class T, int NDIM>
        std::vector<SOHalo<NDIM>> SphericalOverdensity(T * part,
                                                       size_t NumPart,
                                                       double total_mass,
                                                       const std::vector<std::array<double, NDIM>> & centers,
                                                       const std::vector<double> & initial_radius,
                                                       const std::vector<double> & thresholds,
                                                       double search_radius,
                                                       int nmin_center = 20) {
            static_assert(NDIM >= 1, "SphericalOverdensity: need NDIM >= 1");
            const bool many_tasks = FML::NTasks > 1;
            auto boundary = fetch_boundary_particles(part, NumPart, search_radius);

            // A grid of cells (size >= search_radius / 2) over the local domain + buffer in x (periodic in x with one
            // task) and the whole box in the other directions
            const double xlo = many_tasks ? FML::xmin_domain - search_radius : 0.0;
            const double xhi = many_tasks ? FML::xmax_domain + search_radius : 1.0;
            const int ncell = std::max(1, std::min(512, int(2.0 / search_radius)));
            const double cellsize = 1.0 / ncell;
            const int ncellx = many_tasks ? std::max(1, int(std::ceil((xhi - xlo) / cellsize))) : ncell;
            auto cell_coord = [&](const double * pos, int idim) {
                if (idim == 0) {
                    int i = int((pos[0] - xlo) / cellsize);
                    return many_tasks ? std::min(std::max(i, 0), ncellx - 1) : (i % ncell + ncell) % ncell;
                }
                int i = int(pos[idim] / cellsize);
                return (i % ncell + ncell) % ncell;
            };
            // Particle pointers per cell (sparse: only cells with particles)
            std::map<long long int, std::vector<T *>> cells;
            auto cell_key = [&](const std::array<int, NDIM> & c) {
                long long int key = c[0];
                for (int idim = 1; idim < NDIM; idim++)
                    key = key * ncell + c[idim];
                return key;
            };
            auto add_to_cell = [&](T & p) {
                const auto * pos = FML::PARTICLE::GetPos(p);
                std::array<int, NDIM> c;
                for (int idim = 0; idim < NDIM; idim++)
                    c[idim] = cell_coord(pos, idim);
                cells[cell_key(c)].push_back(&p);
            };
            for (size_t i = 0; i < NumPart; i++)
                add_to_cell(part[i]);
            for (auto & p : boundary)
                add_to_cell(p);

            // Separation vector x - c (periodic except in x with several tasks)
            auto separation = [&](const double * x, const std::array<double, NDIM> & c, std::array<double, NDIM> & d) {
                double r2 = 0.0;
                for (int idim = 0; idim < NDIM; idim++) {
                    d[idim] = x[idim] - c[idim];
                    if (idim > 0 or not many_tasks)
                        d[idim] -= std::round(d[idim]);
                    r2 += d[idim] * d[idim];
                }
                return r2;
            };

            // The particles within radius r of c: (r^2, particle)
            const int nsearch = int(std::ceil(search_radius / cellsize));
            auto particles_within = [&](const std::array<double, NDIM> & c, double r) {
                std::vector<std::pair<double, T *>> result;
                std::array<int, NDIM> c0;
                for (int idim = 0; idim < NDIM; idim++)
                    c0[idim] = cell_coord(c.data(), idim);
                const int nr = std::min(nsearch, int(std::ceil(r / cellsize)));
                const double r2max = r * r;
                // The cells to visit in each direction (all of them if the search covers the whole periodic box)
                std::array<std::vector<int>, NDIM> cellrange;
                for (int idim = 0; idim < NDIM; idim++) {
                    const bool periodic_dim = idim > 0 or not many_tasks;
                    const int n = (idim == 0) ? ncellx : ncell;
                    if (periodic_dim and 2 * nr + 1 >= n) {
                        for (int i = 0; i < n; i++)
                            cellrange[idim].push_back(i);
                    } else {
                        for (int i = c0[idim] - nr; i <= c0[idim] + nr; i++) {
                            if (periodic_dim)
                                cellrange[idim].push_back((i % n + n) % n);
                            else if (i >= 0 and i < n)
                                cellrange[idim].push_back(i);
                        }
                    }
                }
                std::array<size_t, NDIM> index{};
                std::array<double, NDIM> d;
                for (int idim = 0; idim < NDIM; idim++)
                    if (cellrange[idim].empty())
                        return result;
                while (true) {
                    std::array<int, NDIM> cc;
                    for (int idim = 0; idim < NDIM; idim++)
                        cc[idim] = cellrange[idim][index[idim]];
                    auto it = cells.find(cell_key(cc));
                    if (it != cells.end())
                        for (auto * p : it->second) {
                            const double r2 = separation(FML::PARTICLE::GetPos(*p), c, d);
                            if (r2 <= r2max)
                                result.push_back({r2, p});
                        }
                    int idim = 0;
                    while (idim < NDIM) {
                        if (++index[idim] < cellrange[idim].size())
                            break;
                        index[idim] = 0;
                        idim++;
                    }
                    if (idim == NDIM)
                        break;
                }
                return result;
            };

            const double volume_factor = (NDIM == 3) ? 4.0 * M_PI / 3.0 : (NDIM == 2 ? M_PI : 2.0);
            auto ball_volume = [&](double r) { return volume_factor * std::pow(r, NDIM); };

            std::vector<SOHalo<NDIM>> halos(centers.size());
            size_t ntruncated = 0;
            for (size_t ih = 0; ih < centers.size(); ih++) {
                auto & h = halos[ih];
                h.radius.assign(thresholds.size(), 0.0);
                h.mass.assign(thresholds.size(), 0.0);
                h.npart.assign(thresholds.size(), 0);

                // 1. Shrinking sphere
                std::array<double, NDIM> c = centers[ih];
                double r = std::min(initial_radius[ih], search_radius);
                size_t n0 = 0;
                std::array<double, NDIM> d;
                for (int iter = 0; iter < 200; iter++) {
                    auto inside = particles_within(c, r);
                    if (iter == 0)
                        n0 = inside.size();
                    const size_t nstop = std::max(size_t(nmin_center), n0 / 10);
                    if (inside.size() < nstop or inside.empty())
                        break;
                    std::array<double, NDIM> com{};
                    double mtot = 0.0;
                    for (auto & ip : inside) {
                        const double m = FML::PARTICLE::GetMass(*ip.second);
                        separation(FML::PARTICLE::GetPos(*ip.second), c, d);
                        for (int idim = 0; idim < NDIM; idim++)
                            com[idim] += m * d[idim];
                        mtot += m;
                    }
                    for (int idim = 0; idim < NDIM; idim++) {
                        c[idim] += com[idim] / mtot;
                        // Keep y, z (and x with one task) in [0,1); x stays in the local domain + buffer
                        if (idim > 0 or not many_tasks)
                            c[idim] -= std::floor(c[idim]);
                    }
                    r *= 0.9;
                }
                // The center is in [0,1) (with several tasks x can only move a little outside the domain)
                for (int idim = 0; idim < NDIM; idim++)
                    h.pos[idim] = c[idim] - std::floor(c[idim]);

                // 2. Grow the sphere
                auto all = particles_within(c, search_radius);
                std::sort(all.begin(), all.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
                std::vector<double> cummass(all.size());
                double m = 0.0;
                for (size_t i = 0; i < all.size(); i++) {
                    m += FML::PARTICLE::GetMass(*all[i].second);
                    cummass[i] = m;
                }
                for (size_t it = 0; it < thresholds.size(); it++) {
                    const double rho_threshold = thresholds[it] * total_mass;
                    // The outermost particle where the mean density inside (including it) is still above the
                    // threshold (see above). Search from the outside in
                    size_t n = all.size();
                    while (n > 0 and cummass[n - 1] / ball_volume(std::sqrt(all[n - 1].first)) < rho_threshold)
                        n--;
                    if (n == all.size() and n > 0) {
                        // Still above the threshold at the last particle: did not reach it within the search radius
                        h.truncated = true;
                        h.mass[it] = cummass.back();
                        h.npart[it] = (long long int)all.size();
                        h.radius[it] = search_radius;
                        continue;
                    }
                    // n = 0 (no particle has an enclosed density above the threshold): radius and mass 0
                    const double Min = n > 0 ? cummass[n - 1] : 0.0;
                    h.mass[it] = Min;
                    h.npart[it] = (long long int)n;
                    h.radius[it] = std::pow(Min / (rho_threshold * volume_factor), 1.0 / NDIM);
                }
                if (h.truncated)
                    ntruncated++;

                // Velocity and dispersion inside the radius of the first threshold
                if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                    const double R = thresholds.empty() ? 0.0 : h.radius[0];
                    double mtot = 0.0;
                    for (auto & ip : all) {
                        if (ip.first > R * R)
                            break;
                        const double mp = FML::PARTICLE::GetMass(*ip.second);
                        const auto * v = FML::PARTICLE::GetVel(*ip.second);
                        for (int idim = 0; idim < NDIM; idim++) {
                            h.vel[idim] += mp * v[idim];
                            h.vel_rms[idim] += mp * v[idim] * v[idim];
                        }
                        mtot += mp;
                    }
                    for (int idim = 0; idim < NDIM; idim++) {
                        if (mtot > 0.0) {
                            h.vel[idim] /= mtot;
                            h.vel_rms[idim] =
                                std::sqrt(std::max(0.0, h.vel_rms[idim] / mtot - h.vel[idim] * h.vel[idim]));
                        }
                    }
                }
            }

            SO_number_of_truncated_halos = ntruncated;
            FML::SumOverTasks(&SO_number_of_truncated_halos);
            if (SO_number_of_truncated_halos > 0 and FML::ThisTask == 0)
                std::cout << "WARNING: [SphericalOverdensity] " << SO_number_of_truncated_halos
                          << " halos did not reach the overdensity within the search radius (" << search_radius
                          << " of the box): their radius and mass are lower limits. Increase the buffer\n";
            return halos;
        }

    } // namespace FOF
} // namespace FML

#endif
