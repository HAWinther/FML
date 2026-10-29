#ifndef FML_COSMICWEB_ORIGAMI_HEADER
#define FML_COSMICWEB_ORIGAMI_HEADER

#include <FML/Global/MPICounts.h>
#include <array>
#include <cmath>
#include <deque>
#include <functional>
#include <vector>

#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

namespace FML {
    namespace COSMICWEB {

        /// ORIGAMI morphology of a particle (number of orthogonal axes along which it has been crossed)
        enum OrigamiMorphology : int { ORIGAMI_VOID = 0, ORIGAMI_WALL = 1, ORIGAMI_FILAMENT = 2, ORIGAMI_HALO = 3 };

        /// The 9 axes on the Lagrangian lattice: x, y, z and the two 45 degree diagonals in each coordinate plane
        inline const std::array<std::array<int, 3>, 9> & origami_axes() {
            static const std::array<std::array<int, 3>, 9> axes{
                {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, -1}, {1, 0, 1}, {1, 0, -1}, {1, 1, 0}, {1, -1, 0}}};
            return axes;
        }
        /// The 4 sets of orthogonal axes: the Cartesian axes, and one Cartesian axis with the two diagonals in the
        /// plane perpendicular to it
        inline const std::array<std::array<int, 3>, 4> & origami_triplets() {
            static const std::array<std::array<int, 3>, 4> t{{{0, 1, 2}, {0, 3, 4}, {1, 5, 6}, {2, 7, 8}}};
            return t;
        }

        /// The morphology from the bitmask of crossed axes (bit a = crossed along origami_axes()[a])
        inline int origami_morphology(unsigned int crossed, bool use_diagonal_axes = true) {
            int M = 0;
            const int ntriplets = use_diagonal_axes ? 4 : 1;
            for (int t = 0; t < ntriplets; t++) {
                int m = 0;
                for (int a : origami_triplets()[t])
                    m += (crossed >> a) & 1u;
                M = std::max(M, m);
            }
            return M;
        }

        namespace detail {
            struct OrigamiRecord {
                double x[3];
                int q[3];
                int task;
                long long int local_index;
            };
            struct OrigamiResult {
                long long int local_index;
                unsigned int crossed;
            };

            // Is the particle crossed along a row: u are the (unwrapped) coordinates along the row, the row is
            // periodic with u[s + n] = u[s] + period. Crossed if there is a j in (s, s+W] with u_j < u_s or
            // a j in [s-W, s) with u_j > u_s
            inline void
            crossed_in_row(const std::vector<double> & u, double period, int W, std::vector<char> & crossed) {
                const int n = int(u.size());
                auto ext = [&](int t) {
                    const int k = ((t % n) + n) % n;
                    return u[k] + double((t - k) / n) * period;
                };
                crossed.assign(n, 0);
                // Forward: sliding window minimum of ext(s+1 .. s+W)
                std::deque<int> dq;
                for (int t = 1; t <= W; t++) {
                    while (not dq.empty() and ext(dq.back()) >= ext(t))
                        dq.pop_back();
                    dq.push_back(t);
                }
                for (int s = 0; s < n; s++) {
                    if (ext(dq.front()) < u[s])
                        crossed[s] = 1;
                    // slide: remove s+1, add s+W+1
                    if (dq.front() == s + 1)
                        dq.pop_front();
                    const int t = s + W + 1;
                    while (not dq.empty() and ext(dq.back()) >= ext(t))
                        dq.pop_back();
                    dq.push_back(t);
                }
                // Backward: sliding window maximum of ext(s-W .. s-1)
                dq.clear();
                for (int t = -W; t <= -1; t++) {
                    while (not dq.empty() and ext(dq.back()) <= ext(t))
                        dq.pop_back();
                    dq.push_back(t);
                }
                for (int s = 0; s < n; s++) {
                    if (ext(dq.front()) > u[s])
                        crossed[s] = 1;
                    if (dq.front() == s - W)
                        dq.pop_front();
                    const int t = s;
                    while (not dq.empty() and ext(dq.back()) <= ext(t))
                        dq.pop_back();
                    dq.push_back(t);
                }
            }
        } // namespace detail

        /// Options for ORIGAMI
        struct OrigamiOptions {
            /// Use the 3 extra sets of axes with the 45 degree diagonals (as in Falck et al. 2012). If false only the
            /// Cartesian axes are used
            bool use_diagonal_axes{true};
            /// How far along a row of the lattice we look for crossings, as a fraction of the row (Falck et al.: 1/4)
            double search_fraction{0.25};
        };

        //========================================================================================
        /// ORIGAMI (Falck, Neyrinck & Szalay 2012): the morphology of the particles from stream crossings in
        /// Lagrangian space. The particles must have started on a regular lattice with Nlattice^3 points. Two
        /// particles on the same row of the lattice have crossed along that row if their order along the row
        /// direction is now reversed. A particle is crossed along an axis if it has crossed with a particle within
        /// search_fraction of the row (in either direction). We check 9 axes (x, y, z and the two diagonals in each
        /// coordinate plane) and the morphology is the largest number of crossed axes in one of the 4 sets of
        /// orthogonal axes: 0 = void, 1 = wall, 2 = filament, 3 = halo.
        ///
        /// Positions must be in [0,1) and the displacement of a particle from its lattice point must be less than
        /// half the box. With MPI the particles can be anywhere (e.g. MPIParticles); we move the rows around.
        ///
        /// @param[in] p Pointer to the particles
        /// @param[in] NumPart Number of local particles
        /// @param[in] Nlattice The number of lattice points per dimension (the total number of particles is Nlattice^3)
        /// @param[out] morphology The morphology of each local particle
        /// @param[in] options Options
        /// @param[in] lattice_coordinate Optional: the lattice point (ix,iy,iz) of a particle. The default uses the
        /// particle ID with id = (ix * Nlattice + iy) * Nlattice + iz, which is how MPIParticles::create_particle_grid
        /// and the NBody initial conditions number the particles
        /// @param[out] crossed_axes Optional: the bitmask of the axes each particle is crossed along (bit a is the
        /// axis origami_axes()[a])
        ///
        //========================================================================================
        template <class T>
        void ORIGAMI(T * p,
                     size_t NumPart,
                     int Nlattice,
                     std::vector<int> & morphology,
                     const OrigamiOptions & options = OrigamiOptions(),
                     std::function<std::array<int, 3>(T &)> lattice_coordinate = nullptr,
                     std::vector<unsigned int> * crossed_axes = nullptr) {
            using detail::OrigamiRecord;
            using detail::OrigamiResult;
            const int N = Nlattice;
            assert_mpi(N > 1, "[ORIGAMI] Nlattice must be > 1");
            FML::COMMUNICATION::Count nlocal = NumPart;
            const auto local_counts = FML::GatherFromTasks(&nlocal);
            const auto total = FML::COMMUNICATION::checked_size_layout(local_counts, size_t(LLONG_MAX));
            FML::COMMUNICATION::Count square = 0, cube = 0;
            bool valid_lattice = FML::COMMUNICATION::multiply_count(N, N, square, LLONG_MAX);
            valid_lattice = valid_lattice && FML::COMMUNICATION::multiply_count(square, N, cube, LLONG_MAX);
            FML::COMMUNICATION::require_count(valid_lattice && cube == total.total,
                                              "[ORIGAMI] Lattice count mismatch or overflow");
            const size_t NumPartTot = total.total;
            if (not lattice_coordinate) {
                if constexpr (FML::PARTICLE::has_get_id<T>()) {
                    lattice_coordinate = [N](T & part) -> std::array<int, 3> {
                        const long long int id = (long long int)(FML::PARTICLE::GetID(part));
                        return {int(id / (long long int)(N) / N), int((id / N) % N), int(id % N)};
                    };
                } else {
                    assert_mpi(false, "[ORIGAMI] The particles have no get_id, provide lattice_coordinate");
                }
            }
            const int W = std::max(1, int(options.search_fraction * N));
            assert_mpi(W < N / 2 + 1, "[ORIGAMI] search_fraction must be at most 1/2");

            // The records
            std::vector<OrigamiRecord> records(NumPart);
            for (size_t i = 0; i < NumPart; i++) {
                const auto * pos = FML::PARTICLE::GetPos(p[i]);
                const auto q = lattice_coordinate(p[i]);
                for (int j = 0; j < 3; j++) {
                    records[i].x[j] = double(pos[j]);
                    records[i].q[j] = q[j];
                    assert_mpi(q[j] >= 0 and q[j] < N, "[ORIGAMI] Lattice coordinate out of range");
                }
                records[i].task = FML::ThisTask;
                records[i].local_index = (long long int)(i);
            }
            std::vector<unsigned int> crossed(NumPart, 0u);

            // The rows in a plane of fixed q[fixed]: the plane coordinates (a, b) are the two other axes and we do
            // the rows along the given 2D directions, each mapped to one of the 9 axes
            struct PlaneDirection {
                int da, db, axis;
            };
            auto process = [&](int fixed, const std::vector<PlaneDirection> & directions) {
                const int ia = fixed == 0 ? 1 : 0;
                const int ib = fixed == 2 ? 1 : 2;
                // Planes are distributed over tasks in blocks
                auto owner = [&](int plane) { return int((long long int)(plane)*FML::NTasks / N); };
                std::vector<OrigamiRecord> mine;
#ifdef USE_MPI
                if (FML::NTasks > 1) {
                    FML::COMMUNICATION::require_count(records.size() <= size_t(INT_MAX),
                                                      "[ORIGAMI] Outgoing record count exceeds MPI int");
                    MPI_Datatype rtype;
                    MPI_Type_contiguous(int(sizeof(OrigamiRecord)), MPI_BYTE, &rtype);
                    MPI_Type_commit(&rtype);
                    std::vector<int> scount(FML::NTasks, 0), rcount(FML::NTasks), sdispl(FML::NTasks, 0),
                        rdispl(FML::NTasks, 0);
                    for (auto & r : records)
                        scount[owner(r.q[fixed])]++;
                    for (int t = 1; t < FML::NTasks; t++)
                        sdispl[t] = sdispl[t - 1] + scount[t - 1];
                    std::vector<OrigamiRecord> sendbuf(records.size());
                    std::vector<int> fill(sdispl);
                    for (auto & r : records)
                        sendbuf[fill[owner(r.q[fixed])]++] = r;
                    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, FML::communicator);
                    const auto receive_layout = FML::COMMUNICATION::checked_int_layout(rcount);
                    rdispl = receive_layout.offsets;
                    mine.resize(receive_layout.total);
                    MPI_Alltoallv(sendbuf.data(),
                                  scount.data(),
                                  sdispl.data(),
                                  rtype,
                                  mine.data(),
                                  rcount.data(),
                                  rdispl.data(),
                                  rtype,
                                  FML::communicator);
                    MPI_Type_free(&rtype);
                } else {
                    mine = records;
                }
#else
                mine = records;
#endif
                // Our planes
                int plane_begin = 0;
                while (plane_begin < N and owner(plane_begin) < FML::ThisTask)
                    plane_begin++;
                int plane_end = plane_begin;
                while (plane_end < N and owner(plane_end) == FML::ThisTask)
                    plane_end++;
                const long long int nplanes = plane_end - plane_begin;
                assert_mpi((long long int)(mine.size()) == nplanes * N * N,
                           "[ORIGAMI] Missing or duplicate lattice points (are the lattice coordinates right?)");
                std::vector<long long int> where(nplanes * N * N, -1);
                for (size_t k = 0; k < mine.size(); k++) {
                    const auto & r = mine[k];
                    long long int & w = where[((long long int)(r.q[fixed] - plane_begin) * N + r.q[ia]) * N + r.q[ib]];
                    assert_mpi(w == -1, "[ORIGAMI] Two particles have the same lattice coordinate");
                    w = (long long int)(k);
                }
                std::vector<unsigned int> mine_crossed(mine.size(), 0u);
                std::vector<double> u(N);
                std::vector<char> rowcrossed;
                std::vector<long long int> rowindex(N);
                for (long long int pl = 0; pl < nplanes; pl++) {
                    for (const auto & dir : directions) {
                        const auto & D = origami_axes()[dir.axis];
                        const double norm = std::sqrt(double(D[0] * D[0] + D[1] * D[1] + D[2] * D[2]));
                        // Rows: start at a = 0 (b = 0..N-1), except for the direction (0,1) that starts at b = 0
                        for (int start = 0; start < N; start++) {
                            const int a0 = dir.da == 0 ? start : 0;
                            const int b0 = dir.da == 0 ? 0 : start;
                            for (int s = 0; s < N; s++) {
                                const int a = ((a0 + s * dir.da) % N + N) % N;
                                const int b = ((b0 + s * dir.db) % N + N) % N;
                                const long long int k = where[(pl * N + a) * N + b];
                                rowindex[s] = k;
                                const auto & r = mine[k];
                                double proj = 0.0;
                                for (int j = 0; j < 3; j++) {
                                    double disp = r.x[j] - double(r.q[j]) / N;
                                    disp -= std::round(disp);
                                    proj += disp * D[j];
                                }
                                u[s] = s * norm / N + proj / norm;
                            }
                            detail::crossed_in_row(u, norm, W, rowcrossed);
                            for (int s = 0; s < N; s++)
                                if (rowcrossed[s])
                                    mine_crossed[rowindex[s]] |= (1u << dir.axis);
                        }
                    }
                }

                // Send the results back
#ifdef USE_MPI
                if (FML::NTasks > 1) {
                    FML::COMMUNICATION::require_count(mine.size() <= size_t(INT_MAX),
                                                      "[ORIGAMI] Outgoing record count exceeds MPI int");
                    MPI_Datatype otype;
                    MPI_Type_contiguous(int(sizeof(OrigamiResult)), MPI_BYTE, &otype);
                    MPI_Type_commit(&otype);
                    std::vector<int> scount(FML::NTasks, 0), rcount(FML::NTasks), sdispl(FML::NTasks, 0),
                        rdispl(FML::NTasks, 0);
                    for (auto & r : mine)
                        scount[r.task]++;
                    for (int t = 1; t < FML::NTasks; t++)
                        sdispl[t] = sdispl[t - 1] + scount[t - 1];
                    std::vector<OrigamiResult> sendbuf(mine.size());
                    std::vector<int> fill(sdispl);
                    for (size_t k = 0; k < mine.size(); k++)
                        sendbuf[fill[mine[k].task]++] = {mine[k].local_index, mine_crossed[k]};
                    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, FML::communicator);
                    const auto receive_layout = FML::COMMUNICATION::checked_int_layout(rcount);
                    rdispl = receive_layout.offsets;
                    std::vector<OrigamiResult> recvbuf(receive_layout.total);
                    MPI_Alltoallv(sendbuf.data(),
                                  scount.data(),
                                  sdispl.data(),
                                  otype,
                                  recvbuf.data(),
                                  rcount.data(),
                                  rdispl.data(),
                                  otype,
                                  FML::communicator);
                    MPI_Type_free(&otype);
                    for (auto & res : recvbuf)
                        crossed[res.local_index] |= res.crossed;
                    return;
                }
#endif
                for (size_t k = 0; k < mine.size(); k++)
                    crossed[mine[k].local_index] |= mine_crossed[k];
            };

            // Planes of fixed x: y, z and the diagonals in the y-z plane
            // Planes of fixed y: x and the diagonals in the x-z plane. Planes of fixed z: the diagonals in the x-y
            // plane
            if (options.use_diagonal_axes) {
                process(0, {{1, 0, 1}, {0, 1, 2}, {1, 1, 3}, {1, -1, 4}});
                process(1, {{1, 0, 0}, {1, 1, 5}, {1, -1, 6}});
                process(2, {{1, 1, 7}, {1, -1, 8}});
            } else {
                process(0, {{1, 0, 1}, {0, 1, 2}});
                process(1, {{1, 0, 0}});
            }

            morphology.resize(NumPart);
            std::array<long long int, 4> count{};
            for (size_t i = 0; i < NumPart; i++) {
                morphology[i] = origami_morphology(crossed[i], options.use_diagonal_axes);
                count[morphology[i]]++;
            }
            FML::SumArrayOverTasks(count.data(), 4);
            if (FML::ThisTask == 0) {
                std::cout << "[ORIGAMI] Fraction of particles in voids " << double(count[0]) / NumPartTot << " walls "
                          << double(count[1]) / NumPartTot << " filaments " << double(count[2]) / NumPartTot
                          << " halos " << double(count[3]) / NumPartTot << "\n";
            }
            if (crossed_axes)
                *crossed_axes = std::move(crossed);
        }

    } // namespace COSMICWEB
} // namespace FML

#endif
