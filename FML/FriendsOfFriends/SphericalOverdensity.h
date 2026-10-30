#ifndef SPHERICALOVERDENSITY_HEADER
#define SPHERICALOVERDENSITY_HEADER

#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
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
        /// Collective buffer expansions beyond the initial selective exchange.
        inline size_t SO_number_of_buffer_refreshes{0};

        namespace SO_DETAIL {
            using Count = unsigned long long;
            inline void require(bool valid, const char * message) {
                int good = valid;
                FML::MinOverTasks(&good);
                if (!good)
                    throw std::invalid_argument(message);
            }
            inline double distance_to_slab(double x, double lo, double hi) {
                double best = 1;
                for (int image = -1; image <= 1; ++image)
                    best = std::min(best, std::max({lo - (x + image), x + image - hi, 0.0}));
                return best;
            }
            inline int nearest_image(double x, double lo, double hi) {
                int shift = 0;
                double best = std::numeric_limits<double>::infinity();
                for (int image = -1; image <= 1; ++image) {
                    double d = std::max({lo - (x + image), x + image - hi, 0.0});
                    if (d < best) {
                        best = d;
                        shift = image;
                    }
                }
                return shift;
            }
        } // namespace SO_DETAIL

        /// Collective selective exchange from every intersecting slab, including
        /// nonadjacent/empty ranks. Each remote source record is received once.
        /// Default: choose the image closest to this slab (legacy unwrapped x).
        /// Wrapped mode preserves coordinates exactly for minimum-image searches.
        /// Particle serialization may have variable record sizes. A complete framed
        /// record must fit max_message_bytes; the smallest rank limit is used.
        template <class T>
        std::vector<T> fetch_boundary_particles(T * part,
                                                size_t NumPart,
                                                double dx,
                                                size_t max_message_bytes = 8 * 1024 * 1024,
                                                bool unwrap_periodic_images = true) {
            using SO_DETAIL::Count;
            const double lo = FML::xmin_domain, hi = FML::xmax_domain;
            bool valid = std::isfinite(dx) && dx >= 0 && lo >= 0 && hi <= 1 && lo <= hi && (NumPart == 0 || part);
            for (size_t i = 0; part && i < NumPart; ++i) {
                double x = FML::PARTICLE::GetPos(part[i])[0];
                valid &= std::isfinite(x) && x >= lo && x < hi;
            }
            valid &= FML::NTasks != 1 || (lo == 0 && hi == 1);
            Count capacity = std::min(max_message_bytes, size_t(INT_MAX));
            FML::MinOverTasks(&capacity);
            SO_DETAIL::require(valid && capacity > sizeof(Count),
                               "[SO exchange] Invalid owned particles/slab, buffer or message limit");
            std::vector<T> boundary;
#ifdef USE_MPI
            if (FML::NTasks > 1) {
                std::array<double, 3> local{lo, hi, dx};
                const auto slabs = FML::GatherFromTasks(&local);
                valid = slabs.front()[0] == 0 && slabs.back()[1] == 1;
                for (int t = 0; t < FML::NTasks; ++t)
                    valid &= slabs[t][0] <= slabs[t][1] && (!t || slabs[t][0] == slabs[t - 1][1]);
                SO_DETAIL::require(valid, "[SO exchange] Rank-ordered slabs must partition [0,1]");
                std::vector<size_t> sizes(NumPart);
                for (size_t i = 0; i < NumPart; ++i) {
                    auto bytes = FML::PARTICLE::GetSize(part[i]);
                    valid &= bytes > 0 && Count(bytes) <= capacity - sizeof(Count);
                    sizes[i] = size_t(bytes);
                }
                SO_DETAIL::require(valid, "[SO exchange] Serialized particle plus length header exceeds message limit");
                auto needed = [&](size_t i, int dest) {
                    const auto & slab = slabs[dest];
                    return slab[0] < slab[1] &&
                           SO_DETAIL::distance_to_slab(FML::PARTICLE::GetPos(part[i])[0], slab[0], slab[1]) <= slab[2];
                };
                std::vector<Count> sends(FML::NTasks), receives(FML::NTasks);
                for (int dest = 0; dest < FML::NTasks; ++dest)
                    if (dest != FML::ThisTask)
                        for (size_t i = 0; i < NumPart; ++i)
                            sends[dest] += needed(i, dest);
                MPI_Alltoall(sends.data(),
                             1,
                             MPI_UNSIGNED_LONG_LONG,
                             receives.data(),
                             1,
                             MPI_UNSIGNED_LONG_LONG,
                             FML::communicator);
                size_t total = 0;
                valid = true;
                for (Count count : receives) {
                    if (count > boundary.max_size() - total) {
                        valid = false;
                        break;
                    }
                    total += size_t(count);
                }
                SO_DETAIL::require(valid, "[SO exchange] Receive particle count exceeds addressable storage");
                boundary.reserve(total);
                std::vector<char> send_buffer, receive_buffer;
                for (int offset = 1; offset < FML::NTasks; ++offset) {
                    const int dest = int((static_cast<long long>(FML::ThisTask) + offset) % FML::NTasks),
                              source =
                                  int((static_cast<long long>(FML::ThisTask) - offset + FML::NTasks) % FML::NTasks);
                    std::vector<size_t> indices;
                    indices.reserve(size_t(sends[dest]));
                    Count rounds = 0;
                    size_t used = 0;
                    for (size_t i = 0; i < NumPart; ++i)
                        if (needed(i, dest)) {
                            indices.push_back(i);
                            size_t framed = sizeof(Count) + sizes[i];
                            if (used && framed > size_t(capacity) - used) {
                                ++rounds;
                                used = 0;
                            }
                            used += framed;
                        }
                    if (used)
                        ++rounds;
                    FML::MaxOverTasks(&rounds);
                    size_t cursor = 0;
                    Count received = 0;
                    for (Count round = 0; round < rounds; ++round) {
                        send_buffer.clear();
                        size_t begin = cursor;
                        used = 0;
                        while (cursor < indices.size()) {
                            size_t bytes = sizeof(Count) + sizes[indices[cursor]];
                            if (bytes > size_t(capacity) - used)
                                break;
                            used += bytes;
                            ++cursor;
                        }
                        if (used > send_buffer.capacity()) {
                            std::vector<char> exact(used);
                            send_buffer.swap(exact);
                        } else
                            send_buffer.resize(used);
                        size_t position = 0;
                        valid = true;
                        try {
                            for (size_t j = begin; j < cursor; ++j) {
                                size_t i = indices[j];
                                Count bytes = sizes[i];
                                std::memcpy(send_buffer.data() + position, &bytes, sizeof(bytes));
                                position += sizeof(bytes);
                                FML::PARTICLE::AppendToBuffer(part[i], send_buffer.data() + position);
                                position += size_t(bytes);
                                if (size_t(FML::PARTICLE::GetSize(part[i])) != sizes[i])
                                    valid = false;
                            }
                        } catch (...) {
                            valid = false;
                        }
                        SO_DETAIL::require(valid,
                                           "[SO exchange] Particle serialization failed or changed its record size");
                        Count ns = send_buffer.size(), nr = 0;
                        MPI_Sendrecv(&ns,
                                     1,
                                     MPI_UNSIGNED_LONG_LONG,
                                     dest,
                                     13510,
                                     &nr,
                                     1,
                                     MPI_UNSIGNED_LONG_LONG,
                                     source,
                                     13510,
                                     FML::communicator,
                                     MPI_STATUS_IGNORE);
                        SO_DETAIL::require(nr <= capacity, "[SO exchange] Invalid received byte count");
                        if (nr > receive_buffer.capacity()) {
                            std::vector<char> exact(static_cast<size_t>(nr));
                            receive_buffer.swap(exact);
                        } else
                            receive_buffer.resize(size_t(nr));
                        MPI_Sendrecv(ns ? send_buffer.data() : nullptr,
                                     int(ns),
                                     MPI_BYTE,
                                     dest,
                                     13511,
                                     nr ? receive_buffer.data() : nullptr,
                                     int(nr),
                                     MPI_BYTE,
                                     source,
                                     13511,
                                     FML::communicator,
                                     MPI_STATUS_IGNORE);
                        position = 0;
                        valid = true;
                        try {
                            while (position < receive_buffer.size()) {
                                if (receive_buffer.size() - position < sizeof(Count)) {
                                    valid = false;
                                    break;
                                }
                                Count bytes = 0;
                                std::memcpy(&bytes, receive_buffer.data() + position, sizeof(bytes));
                                position += sizeof(bytes);
                                if (bytes == 0 || bytes > receive_buffer.size() - position ||
                                    received >= receives[source]) {
                                    valid = false;
                                    break;
                                }
                                T particle{};
                                FML::PARTICLE::AssignFromBuffer(particle, receive_buffer.data() + position);
                                if (Count(FML::PARTICLE::GetSize(particle)) != bytes) {
                                    valid = false;
                                    break;
                                }
                                const double x = FML::PARTICLE::GetPos(particle)[0];
                                if (!std::isfinite(x) || x < slabs[source][0] || x >= slabs[source][1]) {
                                    valid = false;
                                    break;
                                }
                                if (unwrap_periodic_images) {
                                    auto * x = FML::PARTICLE::GetPos(particle);
                                    x[0] += SO_DETAIL::nearest_image(x[0], lo, hi);
                                }
                                boundary.push_back(std::move(particle));
                                position += size_t(bytes);
                                ++received;
                            }
                        } catch (...) {
                            valid = false;
                        }
                        SO_DETAIL::require(valid, "[SO exchange] Particle deserialization failed or frame was invalid");
                    }
                    SO_DETAIL::require(cursor == indices.size() && received == receives[source],
                                       "[SO exchange] Particle count mismatch");
                }
            }
#else
            (void)unwrap_periodic_images;
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
        /// All particles required by the search spheres, including nonadjacent MPI slabs, are used, not only the
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
        /// intersecting MPI slabs
        /// @param[in] nmin_center Stop the shrinking sphere when fewer particles than this are left (fiducial 20)
        /// @param[in] max_exchange_bytes Maximum serialized message bytes, including record headers (smallest rank
        /// limit) This call is collective. Particle coordinates remain wrapped; centering can collectively expand the
        /// selectively exchanged buffer before a query would leave its coverage. See SphericalOverdensity.md.
        //========================================================================================
        template <class T, int NDIM>
        std::vector<SOHalo<NDIM>> SphericalOverdensity(T * part,
                                                       size_t NumPart,
                                                       double total_mass,
                                                       const std::vector<std::array<double, NDIM>> & centers,
                                                       const std::vector<double> & initial_radius,
                                                       const std::vector<double> & thresholds,
                                                       double search_radius,
                                                       int nmin_center = 20,
                                                       size_t max_exchange_bytes = 8 * 1024 * 1024) {
            static_assert(NDIM >= 1, "SphericalOverdensity: need NDIM >= 1");
            bool valid = centers.size() == initial_radius.size() && total_mass > 0 && std::isfinite(total_mass) &&
                         search_radius > 0 && std::isfinite(search_radius) &&
                         std::isfinite(std::pow(search_radius, NDIM)) && nmin_center >= 0 && (NumPart == 0 || part);
            valid &= FML::xmin_domain < FML::xmax_domain || centers.empty();
            for (size_t i = 0; i < centers.size(); ++i) {
                for (double x : centers[i])
                    valid &= std::isfinite(x);
                if (i < initial_radius.size())
                    valid &= initial_radius[i] >= 0 && std::isfinite(initial_radius[i]);
            }
            for (double delta : thresholds)
                valid &= delta > 0 && std::isfinite(delta * total_mass);
            for (size_t i = 0; part && i < NumPart; ++i) {
                const auto * x = FML::PARTICLE::GetPos(part[i]);
                for (int d = 0; d < NDIM; ++d)
                    valid &= std::isfinite(x[d]) && x[d] >= 0 && x[d] < 1;
                valid &= std::isfinite(FML::PARTICLE::GetMass(part[i])) && FML::PARTICLE::GetMass(part[i]) > 0;
            }
            SO_DETAIL::require(valid, "[SphericalOverdensity] Invalid centers, radii, masses or thresholds");
            std::vector<T> boundary;
            // Sparse periodic cell keys; only owned and selectively received records are stored.
            const int ncell = int(std::max(1., std::min(512., 2. / search_radius)));
            const double cellsize = 1. / ncell;
            auto cell_coord = [&](const auto * pos, int dim) {
                double x = pos[dim] - std::floor(pos[dim]);
                return std::min(ncell - 1, int(x * ncell));
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
            auto exchange_and_rebuild = [&](double coverage) {
                cells.clear();
                std::vector<T>().swap(boundary);
                boundary =
                    fetch_boundary_particles(part,
                                             NumPart,
                                             std::min(.5, coverage + 32 * std::numeric_limits<double>::epsilon()),
                                             max_exchange_bytes,
                                             false);
                for (size_t i = 0; i < NumPart; ++i)
                    add_to_cell(part[i]);
                for (auto & p : boundary)
                    add_to_cell(p);
            };

            // Minimum-image separation in every dimension, independent of MPI layout.
            auto separation = [&](const auto * x, const std::array<double, NDIM> & c, std::array<double, NDIM> & d) {
                double r2 = 0.0;
                for (int idim = 0; idim < NDIM; idim++) {
                    d[idim] = x[idim] - c[idim];
                    d[idim] -= std::round(d[idim]);
                    r2 += d[idim] * d[idim];
                }
                return r2;
            };

            // The particles within radius r of c: (r^2, particle)
            const int nsearch = int(std::min(double(ncell), std::ceil(search_radius / cellsize)));
            auto particles_within = [&](const std::array<double, NDIM> & c, double r) {
                std::vector<std::pair<double, T *>> result;
                std::array<int, NDIM> c0;
                for (int idim = 0; idim < NDIM; idim++)
                    c0[idim] = cell_coord(c.data(), idim);
                const int nr = int(std::min(double(nsearch), std::ceil(r / cellsize)));
                const double r2max = r * r;
                // The cells to visit in each direction (all of them if the search covers the whole periodic box)
                std::array<std::vector<int>, NDIM> cellrange;
                for (int idim = 0; idim < NDIM; idim++) {
                    const int n = ncell;
                    if (2 * nr + 1 >= n) {
                        for (int i = 0; i < n; i++)
                            cellrange[idim].push_back(i);
                    } else {
                        for (int i = c0[idim] - nr; i <= c0[idim] + nr; i++) {
                            cellrange[idim].push_back((i % n + n) % n);
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

            struct CenterState {
                std::array<double, NDIM> center;
                double radius;
                size_t initial_count = 0;
                int iteration = 0;
                bool done = false;
            };
            std::vector<CenterState> states;
            states.reserve(centers.size());
            double coverage = 0;
            auto requirement = [&](const auto & c, double r) {
                return r + SO_DETAIL::distance_to_slab(c[0], FML::xmin_domain, FML::xmax_domain);
            };
            for (size_t i = 0; i < centers.size(); ++i) {
                auto wrapped = centers[i];
                for (double & x : wrapped)
                    x -= std::floor(x);
                states.push_back({wrapped, std::min(initial_radius[i], search_radius)});
                coverage = std::max(coverage, requirement(wrapped, search_radius));
            }
            coverage = FML::NTasks == 1 ? .5 : std::min(.5, coverage);
            SO_number_of_buffer_refreshes = 0;
            exchange_and_rebuild(coverage);
            for (;;) {
                double needed = coverage;
                for (auto & state : states) {
                    while (!state.done && state.iteration < 200) {
                        double required = requirement(state.center, state.radius);
                        if (coverage < .5 && required > coverage) {
                            needed = std::max(needed, required);
                            break;
                        }
                        auto inside = particles_within(state.center, state.radius);
                        if (state.iteration == 0)
                            state.initial_count = inside.size();
                        size_t stop = std::max(size_t(nmin_center), state.initial_count / 10);
                        if (inside.empty() || inside.size() < stop) {
                            state.done = true;
                            break;
                        }
                        std::array<double, NDIM> com{}, d{};
                        double mass = 0;
                        for (const auto & particle : inside) {
                            double m = FML::PARTICLE::GetMass(*particle.second);
                            separation(FML::PARTICLE::GetPos(*particle.second), state.center, d);
                            for (int dim = 0; dim < NDIM; ++dim)
                                com[dim] += m * d[dim];
                            mass += m;
                        }
                        for (int dim = 0; dim < NDIM; ++dim) {
                            state.center[dim] += com[dim] / mass;
                            state.center[dim] -= std::floor(state.center[dim]);
                        }
                        state.radius *= .9;
                        ++state.iteration;
                    }
                    if (state.iteration == 200)
                        state.done = true;
                    if (state.done)
                        needed = std::max(needed, requirement(state.center, search_radius));
                }
                int expand = coverage < .5 && needed > coverage;
                FML::MaxOverTasks(&expand);
                if (!expand)
                    break;
                if (needed > coverage)
                    coverage = std::min(.5, std::max(needed, 1.5 * coverage));
                exchange_and_rebuild(coverage);
                ++SO_number_of_buffer_refreshes;
            }

            std::vector<SOHalo<NDIM>> halos(centers.size());
            size_t ntruncated = 0;
            for (size_t ih = 0; ih < centers.size(); ih++) {
                auto & h = halos[ih];
                h.radius.assign(thresholds.size(), 0.0);
                h.mass.assign(thresholds.size(), 0.0);
                h.npart.assign(thresholds.size(), 0);

                const auto & c = states[ih].center;
                h.pos = c;

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
