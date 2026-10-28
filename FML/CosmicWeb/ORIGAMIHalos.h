#ifndef FML_COSMICWEB_ORIGAMIHALOS_HEADER
#define FML_COSMICWEB_ORIGAMIHALOS_HEADER

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <FML/CosmicWeb/ORIGAMI.h>
#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Triangulation/PeriodicDelaunay.h>

namespace FML {
    namespace COSMICWEB {

        /// A halo from ORIGAMIHalos (positions in [0,1), densities in units of the mean)
        struct OrigamiHalo {
            /// Number in the catalogue (sorted by the number of particles, largest first)
            long long int id{0};
            long long int npart{0};
            /// Number of core particles (density above the threshold), 0 for groups without a core
            long long int ncore{0};
            double mass{0.0};
            /// Center of mass
            std::array<double, 3> pos{};
            /// The densest particle
            std::array<double, 3> pos_densest{};
            double density_max{0.0};
        };

        /// Options for ORIGAMIHalos (defaults as in Falck et al. 2012)
        struct OrigamiHaloOptions {
            /// Core particles are halo particles with Voronoi density above this (in units of the mean)
            double core_density_threshold{200.0};
            /// Halos with fewer particles are not kept
            long long int min_particles{20};
            /// Keep the groups of halo particles that are not connected to any core
            bool keep_groups_without_core{true};
            /// Tessellation (MPI): how much of the neighbor domains we include, and the guard points
            double buffer_fraction{0.5};
            double random_fraction{0.3};
        };

        namespace detail {
            struct OrigamiVertexData {
                long long int index{-1};
            };
            struct OrigamiGroupInfo {
                long long int label;
                long long int n;
                long long int ncore;
                double mass;
                double density_max;
                long long int key_densest;
                double pos_densest[3];
            };
            struct OrigamiGroupSum {
                long long int label;
                double mass;
                double mdx[3];
            };

            // Gather a vector of trivially copyable things on task 0
            template <class S>
            std::vector<S> gather_on_task0(const std::vector<S> & local) {
#ifdef USE_MPI
                if (FML::NTasks > 1) {
                    MPI_Datatype type;
                    MPI_Type_contiguous(int(sizeof(S)), MPI_BYTE, &type);
                    MPI_Type_commit(&type);
                    std::vector<int> counts(FML::NTasks), displs(FML::NTasks, 0);
                    int n = int(local.size());
                    MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, FML::communicator);
                    for (int i = 1; i < FML::NTasks; i++)
                        displs[i] = displs[i - 1] + counts[i - 1];
                    std::vector<S> all;
                    if (FML::ThisTask == 0)
                        all.resize(size_t(displs.back()) + counts.back());
                    MPI_Gatherv(local.data(),
                                n,
                                type,
                                all.data(),
                                counts.data(),
                                displs.data(),
                                type,
                                0,
                                FML::communicator);
                    MPI_Type_free(&type);
                    return all;
                }
#endif
                return local;
            }
            template <class S>
            void broadcast_from_task0([[maybe_unused]] std::vector<S> & v) {
#ifdef USE_MPI
                long long int n = (long long int)(v.size());
                MPI_Bcast(&n, 1, MPI_LONG_LONG, 0, FML::communicator);
                v.resize(size_t(n));
                const size_t chunk = (size_t(1) << 28) / sizeof(S) + 1;
                for (size_t offset = 0; offset < v.size(); offset += chunk) {
                    const size_t count = std::min(chunk, v.size() - offset);
                    MPI_Bcast(v.data() + offset, int(count * sizeof(S)), MPI_BYTE, 0, FML::communicator);
                }
#endif
            }
        } // namespace detail

        //========================================================================================
        /// The halo grouping of ORIGAMI (Falck, Neyrinck & Szalay 2012). The halo particles (morphology 3 from ORIGAMI)
        /// are linked on the Delaunay tessellation of all the particles:
        ///   1. The density of every particle is mass / Voronoi volume in units of the mean (VTFE).
        ///   2. Halo particles above core_density_threshold that are Delaunay neighbors form the cores.
        ///   3. The other halo particles are added to the cores through their Delaunay neighbors, in rounds: in each
        ///      round every halo particle that is not in a halo yet and has a neighbor in a halo joins the halo of its
        ///      densest such neighbor (from the previous round, so the result does not depend on the order or the
        ///      number of tasks). Every halo then has exactly one core.
        ///   4. The halo particles that are not connected to a core form groups by themselves (connected pieces).
        ///   5. Groups with fewer than min_particles are not kept. No unbinding is done.
        ///
        /// Groups are labeled by the smallest lattice index of their particles internally; the catalogue is sorted by
        /// the number of particles. The particles must be in [0,1)^3 and inside the local domain (e.g. MPIParticles).
        ///
        /// @param[in] p Pointer to the particles
        /// @param[in] NumPart Number of local particles
        /// @param[in] Nlattice The number of lattice points per dimension of the initial lattice
        /// @param[in] morphology The ORIGAMI morphology of the local particles (from ORIGAMI)
        /// @param[out] halo_id For each local particle the id of its halo in the catalogue (-1 if none)
        /// @param[out] halos The halo catalogue (on all tasks)
        /// @param[in] options Options
        /// @param[in] lattice_coordinate Optional: the lattice point (ix,iy,iz) of a particle (default from get_id
        /// with id = (ix * Nlattice + iy) * Nlattice + iz)
        ///
        //========================================================================================
        template <class T>
        void ORIGAMIHalos(T * p,
                          size_t NumPart,
                          int Nlattice,
                          const std::vector<int> & morphology,
                          std::vector<long long int> & halo_id,
                          std::vector<OrigamiHalo> & halos,
                          const OrigamiHaloOptions & options = OrigamiHaloOptions(),
                          std::function<std::array<int, 3>(T &)> lattice_coordinate = nullptr) {

            using VD = detail::OrigamiVertexData;
            using Delaunay = FML::TRIANGULATION::MPIPeriodicDelaunay<T, VD>;
            using Vertex_handle = typename Delaunay::Vertex_handle;
            static_assert(CGAL_NDIM == 3, "ORIGAMIHalos is only implemented in 3D");
            assert_mpi(morphology.size() == NumPart, "[ORIGAMIHalos] morphology must have size NumPart");
            const long long int N = Nlattice;
            const long long int no_label = std::numeric_limits<long long int>::max();
            if (not lattice_coordinate) {
                if constexpr (FML::PARTICLE::has_get_id<T>()) {
                    lattice_coordinate = [N](T & part) -> std::array<int, 3> {
                        const long long int id = (long long int)(FML::PARTICLE::GetID(part));
                        return {int(id / N / N), int((id / N) % N), int(id % N)};
                    };
                } else {
                    assert_mpi(false, "[ORIGAMIHalos] The particles have no get_id, provide lattice_coordinate");
                }
            }

            // Tessellation and Voronoi densities
            Delaunay D;
            D.create(p, NumPart, options.buffer_fraction, options.random_fraction);
            const auto & vs = D.get_vertex_handles_regular();
            const auto & vsb = D.get_vertex_handles_boundary();
            const size_t nb = vsb.size();
            for (size_t i = 0; i < NumPart; i++)
                vs[i]->info().index = (long long int)(i);
            for (size_t j = 0; j < nb; j++)
                vsb[j]->info().index = (long long int)(NumPart + j);
            std::vector<double> volume;
            D.VoronoiVolume(volume);
            double total_mass = 0.0;
            std::vector<double> mass(NumPart, 1.0);
            for (size_t i = 0; i < NumPart; i++) {
                if constexpr (FML::PARTICLE::has_get_mass<T>())
                    mass[i] = FML::PARTICLE::GetMass(p[i]);
                total_mass += mass[i];
            }
            FML::SumOverTasks(&total_mass);

            // Values for the regular particles and their copies on the neighbor tasks (index i < NumPart: regular,
            // NumPart + j: boundary particle j)
            auto with_boundary = [&](const auto & regular) {
                auto boundary = D.communicate_boundary_values(regular);
                auto all = regular;
                all.insert(all.end(), boundary.begin(), boundary.end());
                return all;
            };
            std::vector<double> density_local(NumPart);
            std::vector<long long int> key_local(NumPart);
            std::vector<int> halo_local(NumPart);
            for (size_t i = 0; i < NumPart; i++) {
                density_local[i] = mass[i] / volume[i] / total_mass;
                const auto q = lattice_coordinate(p[i]);
                key_local[i] = (q[0] * N + q[1]) * N + q[2];
                halo_local[i] = morphology[i] == ORIGAMI_HALO ? 1 : 0;
            }
            const auto density = with_boundary(density_local);
            const auto key = with_boundary(key_local);
            const auto is_halo = with_boundary(halo_local);
            const size_t ntot = NumPart + nb;

            // Adjacency (only from the regular particles; links between two boundary particles are not reliable)
            std::vector<size_t> adj_offset(NumPart + 1, 0);
            std::vector<long long int> adj;
            {
                std::vector<Vertex_handle> work;
                for (size_t i = 0; i < NumPart; i++) {
                    D.for_each_adjacent_vertex(vs[i], work, [&](Vertex_handle w) {
                        const long long int k = w->info().index;
                        if (k >= 0)
                            adj.push_back(k);
                    });
                    adj_offset[i + 1] = adj.size();
                }
            }

            // The connected pieces of the 'active' particles: every active particle gets the smallest label of its
            // piece. Local union-find over the links of our particles, then the labels of our particles are sent to
            // their copies on the neighbor tasks, until nothing changes
            auto connected_labels = [&](const std::vector<char> & active, std::vector<long long int> & label) {
                std::vector<long long int> uf(ntot);
                for (int iteration = 0;; iteration++) {
                    std::iota(uf.begin(), uf.end(), 0LL);
                    std::function<long long int(long long int)> find = [&](long long int a) {
                        while (uf[a] != a) {
                            uf[a] = uf[uf[a]];
                            a = uf[a];
                        }
                        return a;
                    };
                    for (size_t i = 0; i < NumPart; i++) {
                        if (not active[i])
                            continue;
                        for (size_t k = adj_offset[i]; k < adj_offset[i + 1]; k++)
                            if (active[adj[k]]) {
                                const long long int a = find((long long int)(i)), b = find(adj[k]);
                                if (a != b)
                                    uf[std::max(a, b)] = std::min(a, b);
                            }
                    }
                    std::vector<long long int> minlabel(ntot, no_label);
                    for (size_t i = 0; i < ntot; i++)
                        if (active[i])
                            minlabel[find(i)] = std::min(minlabel[find(i)], label[i]);
                    long long int nchanged = 0;
                    for (size_t i = 0; i < ntot; i++) {
                        if (not active[i])
                            continue;
                        const long long int l = minlabel[find(i)];
                        if (i < NumPart and l != label[i])
                            nchanged++;
                        label[i] = l;
                    }
                    // Our labels to the copies on the other tasks
                    std::vector<long long int> regular(label.begin(), label.begin() + NumPart);
                    const auto boundary = D.communicate_boundary_values(regular);
                    for (size_t j = 0; j < nb; j++)
                        if (active[NumPart + j])
                            label[NumPart + j] = boundary[j];
                    FML::SumOverTasks(&nchanged);
                    if (nchanged == 0 and iteration > 0)
                        break;
                }
            };

            // 1. The cores
            std::vector<char> active(ntot, 0);
            std::vector<long long int> label(ntot, no_label);
            std::vector<char> core(ntot, 0);
            for (size_t i = 0; i < ntot; i++) {
                core[i] = is_halo[i] and density[i] > options.core_density_threshold;
                active[i] = core[i];
                if (core[i])
                    label[i] = key[i];
            }
            connected_labels(active, label);

            // 2. Add the other halo particles in rounds
            for (int round = 0;; round++) {
                std::vector<long long int> new_label(label.begin(), label.begin() + NumPart);
                long long int nadded = 0;
                for (size_t i = 0; i < NumPart; i++) {
                    if (not is_halo[i] or label[i] != no_label)
                        continue;
                    long long int best = -1;
                    for (size_t k = adj_offset[i]; k < adj_offset[i + 1]; k++) {
                        const long long int j = adj[k];
                        if (not is_halo[j] or label[j] == no_label)
                            continue;
                        if (best < 0 or density[j] > density[best] or
                            (density[j] == density[best] and key[j] < key[best]))
                            best = j;
                    }
                    if (best >= 0) {
                        new_label[i] = label[best];
                        nadded++;
                    }
                }
                std::copy(new_label.begin(), new_label.end(), label.begin());
                const auto boundary = D.communicate_boundary_values(new_label);
                std::copy(boundary.begin(), boundary.end(), label.begin() + NumPart);
                FML::SumOverTasks(&nadded);
                if (nadded == 0)
                    break;
            }

            // 3. The halo particles not connected to a core
            if (options.keep_groups_without_core) {
                std::vector<long long int> label2(ntot, no_label);
                for (size_t i = 0; i < ntot; i++) {
                    active[i] = is_halo[i] and label[i] == no_label;
                    if (active[i])
                        label2[i] = key[i];
                }
                connected_labels(active, label2);
                for (size_t i = 0; i < ntot; i++)
                    if (active[i])
                        label[i] = label2[i];
            }

            // 4. Bin up the groups: number of particles, core particles, mass and the densest particle
            std::vector<detail::OrigamiGroupInfo> info;
            {
                std::unordered_map<long long int, size_t> where;
                for (size_t i = 0; i < NumPart; i++) {
                    if (label[i] == no_label)
                        continue;
                    auto it = where.find(label[i]);
                    if (it == where.end()) {
                        it = where.emplace(label[i], info.size()).first;
                        detail::OrigamiGroupInfo g{};
                        g.label = label[i];
                        g.density_max = -1.0;
                        info.push_back(g);
                    }
                    auto & g = info[it->second];
                    g.n++;
                    g.ncore += core[i];
                    g.mass += mass[i];
                    if (density[i] > g.density_max or (density[i] == g.density_max and key[i] < g.key_densest)) {
                        g.density_max = density[i];
                        g.key_densest = key[i];
                        const auto * pos = FML::PARTICLE::GetPos(p[i]);
                        for (int d = 0; d < 3; d++)
                            g.pos_densest[d] = double(pos[d]);
                    }
                }
            }
            auto all_info = detail::gather_on_task0(info);
            std::vector<detail::OrigamiGroupInfo> kept;
            if (FML::ThisTask == 0) {
                std::map<long long int, detail::OrigamiGroupInfo> merged;
                for (auto & g : all_info) {
                    auto it = merged.find(g.label);
                    if (it == merged.end()) {
                        merged.emplace(g.label, g);
                        continue;
                    }
                    auto & m = it->second;
                    m.n += g.n;
                    m.ncore += g.ncore;
                    m.mass += g.mass;
                    if (g.density_max > m.density_max or
                        (g.density_max == m.density_max and g.key_densest < m.key_densest)) {
                        m.density_max = g.density_max;
                        m.key_densest = g.key_densest;
                        for (int d = 0; d < 3; d++)
                            m.pos_densest[d] = g.pos_densest[d];
                    }
                }
                for (auto & [l, g] : merged)
                    if (g.n >= options.min_particles)
                        kept.push_back(g);
                std::sort(kept.begin(), kept.end(), [](const auto & a, const auto & b) {
                    return a.n != b.n ? a.n > b.n : a.label < b.label;
                });
            }
            detail::broadcast_from_task0(kept);
            std::unordered_map<long long int, long long int> id_of_label;
            for (size_t k = 0; k < kept.size(); k++)
                id_of_label[kept[k].label] = (long long int)(k);

            // The halo of each particle and the center of mass (relative to the densest particle)
            halo_id.assign(NumPart, -1);
            std::vector<double> sums(4 * kept.size(), 0.0);
            for (size_t i = 0; i < NumPart; i++) {
                if (label[i] == no_label)
                    continue;
                auto it = id_of_label.find(label[i]);
                if (it == id_of_label.end())
                    continue;
                const long long int h = it->second;
                halo_id[i] = h;
                const auto * pos = FML::PARTICLE::GetPos(p[i]);
                sums[4 * h] += mass[i];
                for (int d = 0; d < 3; d++) {
                    double dx = double(pos[d]) - kept[h].pos_densest[d];
                    dx -= std::round(dx);
                    sums[4 * h + 1 + d] += mass[i] * dx;
                }
            }
            if (not sums.empty())
                FML::SumArrayOverTasks(sums.data(), int(sums.size()));
            halos.resize(kept.size());
            for (size_t h = 0; h < kept.size(); h++) {
                auto & H = halos[h];
                H.id = (long long int)(h);
                H.npart = kept[h].n;
                H.ncore = kept[h].ncore;
                H.mass = kept[h].mass;
                H.density_max = kept[h].density_max;
                for (int d = 0; d < 3; d++) {
                    H.pos_densest[d] = kept[h].pos_densest[d];
                    double x = kept[h].pos_densest[d] + sums[4 * h + 1 + d] / sums[4 * h];
                    x -= std::floor(x);
                    H.pos[d] = x >= 1.0 ? 0.0 : x;
                }
            }

            if (FML::ThisTask == 0) {
                long long int nin = 0, nwithcore = 0;
                for (auto & H : halos) {
                    nin += H.npart;
                    nwithcore += H.ncore > 0;
                }
                std::cout << "[ORIGAMIHalos] " << halos.size() << " halos with >= " << options.min_particles
                          << " particles (" << nwithcore << " with a core), " << nin << " particles in halos\n";
            }
        }

    } // namespace COSMICWEB
} // namespace FML

#endif
