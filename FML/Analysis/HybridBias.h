#ifndef FML_ANALYSIS_HYBRIDBIAS_H
#define FML_ANALYSIS_HYBRIDBIAS_H

#include <FML/Analysis/AnalysisContext.h>
#include <FML/Analysis/Output.h>
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/ComputePowerSpectra/HybridBiasLinearFieldIO.h>
#include <FML/ComputePowerSpectra/HybridBiasSpectra.h>
#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Smoothing/SmoothingFourier.h>

#include <array>
#include <climits>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif

//=============================================================================
// Hybrid bias spectra (the Lagrangian bias basis; Modi, Chen & White 2020, Kokron et al. 2021) of a snapshot: the
// cross spectra of the operators 1, delta_L, delta_L^2, s^2, nabla^2 delta_L at the starting lattice points of the
// particles, advected with the particles (FML/ComputePowerSpectra/HybridBiasSpectra.h, as SimulationCode's
// hybridbias). A snapshot does not have the linear field, so it comes from
//
//   ic_snapshot: an initial-conditions snapshot of the same particles (matched by ID). Each particle's lattice point
//                is its starting position rounded to the lattice (offset found from the data); the displacements
//                Psi = x - q give delta_L = -div Psi, with (by default) the 2LPT term removed:
//                delta_L = -div Psi - 3/7 sum_{i<j} (phi_ii phi_jj - phi_ij^2), nabla^2 phi = -div Psi. Works for any
//                code's snapshots (no ID convention). The IC must be early enough that no particle moved half a
//                lattice cell.
//   file:        delta_L on the lattice in an FML linear field file (HybridBiasLinearFieldIO.h; SimulationCode writes
//                it with hybridbias_output_linear_field). The lattice point of a particle is from its ID
//                (id - hybridbias_id_offset = (ix N + iy) N + iz) or, if hybridbias_ic_snapshot is set, from the IC
//                snapshot.
//
// delta_L is scaled to the snapshot with the linear growth D(z) / D(z_L) of the snapshot's expansion history (scale
// independent; hybridbias_growth_ratio overrides it).
//=============================================================================

namespace FML {
    namespace ANALYSIS {

        //=============================================================================
        /// Hybrid bias settings. Keys hybridbias_<name>, SimulationCode's names where it has them
        //=============================================================================
        struct HybridBiasSettings {
            static constexpr const char * prefix = "hybridbias_";
            int nmesh;
            int noperators;
            std::string density_assignment_method;
            bool interlacing;
            bool subtract_shotnoise;
            double smoothing_scale;
            std::string smoothing_filter;
            double kcut_over_nyquist;
            std::string linear_field_source;
            std::string ic_snapshot;
            std::string linear_field_file;
            std::string ic_lpt_correction;
            int id_offset;
            double growth_ratio;

            HybridBiasSettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("nmesh", nmesh, 128, "Grid size for the advected fields (divisible by the number of MPI tasks)");
                f("noperators", noperators, 5, "Use the first noperators of (1, delta, delta^2, s^2, nabla^2 delta)");
                f("density_assignment_method",
                  density_assignment_method,
                  std::string("CIC"),
                  "Density assignment method (NGP, CIC, TSC, PCS, PQS)");
                f("interlacing", interlacing, true, "Use interlaced grids to reduce aliasing");
                f("subtract_shotnoise", subtract_shotnoise, false, "Subtract the Poisson shot noise");
                f("smoothing_scale", smoothing_scale, 0.0, "Smooth the linear field with this scale (Mpc/h; 0: none)");
                f("smoothing_filter", smoothing_filter, std::string("gaussian"), "tophat, sharpk or gaussian");
                f("kcut_over_nyquist",
                  kcut_over_nyquist,
                  1.0,
                  "Keep the modes of the linear field with k < this times the Nyquist frequency of the lattice");
                f("linear_field_source",
                  linear_field_source,
                  std::string("ic_snapshot"),
                  "ic_snapshot: from the displacements in hybridbias_ic_snapshot; file: hybridbias_linear_field_file");
                f("ic_snapshot",
                  ic_snapshot,
                  std::string(""),
                  "Initial-conditions snapshot of the same particles (prefix; snapshot_format), matched by ID");
                f("linear_field_file",
                  linear_field_file,
                  std::string(""),
                  "FML linear field file (e.g. SimulationCode's hybridbias_output_linear_field)");
                f("ic_lpt_correction",
                  ic_lpt_correction,
                  std::string("2lpt"),
                  "ic_snapshot: 2lpt (remove the 2LPT term from -div Psi) or none (1LPT initial conditions)");
                f("id_offset",
                  id_offset,
                  0,
                  "file without ic_snapshot: the lattice index of a particle is id - id_offset = (ix N + iy) N + iz");
                f("growth_ratio",
                  growth_ratio,
                  0.0,
                  "D(z_snapshot) / D(z of the linear field) (0: from the expansion history, scale-independent)");
            }

            void validate() const {
                if (nmesh < 2 or nmesh % FML::NTasks != 0)
                    throw std::runtime_error("hybridbias_nmesh must be >= 2 and divisible by the number of MPI tasks");
                if (noperators < 1 or noperators > 5)
                    throw std::runtime_error("hybridbias_noperators must be 1-5");
                const auto & m = density_assignment_method;
                if (m != "NGP" and m != "CIC" and m != "TSC" and m != "PCS" and m != "PQS")
                    throw std::runtime_error("hybridbias_density_assignment_method must be NGP, CIC, TSC, PCS or PQS");
                if (smoothing_filter != "gaussian" and smoothing_filter != "sharpk" and smoothing_filter != "tophat")
                    throw std::runtime_error("hybridbias_smoothing_filter must be gaussian, sharpk or tophat");
                if (not(smoothing_scale >= 0.0) or not(kcut_over_nyquist >= 0.0) or not(growth_ratio >= 0.0))
                    throw std::runtime_error("hybridbias_smoothing_scale, hybridbias_kcut_over_nyquist and "
                                             "hybridbias_growth_ratio must be >= 0");
                if (linear_field_source == "ic_snapshot") {
                    if (ic_snapshot.empty())
                        throw std::runtime_error("hybridbias_linear_field_source = ic_snapshot needs hybridbias_ic_snapshot");
                } else if (linear_field_source == "file") {
                    if (linear_field_file.empty())
                        throw std::runtime_error("hybridbias_linear_field_source = file needs hybridbias_linear_field_file");
                } else {
                    throw std::runtime_error("hybridbias_linear_field_source must be ic_snapshot or file");
                }
                if (ic_lpt_correction != "2lpt" and ic_lpt_correction != "none")
                    throw std::runtime_error("hybridbias_ic_lpt_correction must be 2lpt or none");
            }
        };

        /// The linear field on the lattice and where it came from
        struct HybridBiasLinearField {
            FML::GRID::FFTWGrid<3> delta_fourier;     // delta_L(k) at the redshift below (Fourier space, nlattice^3)
            int nlattice{0};
            double redshift{0.0};
            std::string source;
            // ic_snapshot: the lattice offset (cells) and the largest displacement (cells)
            std::array<double, 3> lattice_offset{};
            double max_displacement_cells{0.0};
        };

        namespace detail {
            /// All-to-all of POD records: out[r] goes to task r. Returns the records received, in task order, and
            /// (if from is given) the number received from each task. Collective
            template <class R>
            std::vector<R> exchange_records(const std::vector<std::vector<R>> & out,
                                            std::vector<long long int> * from = nullptr) {
#ifdef USE_MPI
                const int NT = FML::NTasks;
                std::vector<long long int> nsend(static_cast<size_t>(NT)), nrecv(static_cast<size_t>(NT));
                for (int r = 0; r < NT; r++)
                    nsend[size_t(r)] = (long long int)out[size_t(r)].size();
                MPI_Alltoall(nsend.data(), 1, MPI_LONG_LONG, nrecv.data(), 1, MPI_LONG_LONG, FML::communicator);
                std::vector<int> sc(static_cast<size_t>(NT)), sd(static_cast<size_t>(NT)), rc(static_cast<size_t>(NT)), rd(static_cast<size_t>(NT));
                long long int stot = 0, rtot = 0;
                for (int r = 0; r < NT; r++) {
                    if (double(nsend[size_t(r)]) * sizeof(R) > double(INT_MAX) or
                        double(stot + nsend[size_t(r)]) * sizeof(R) > double(INT_MAX) or
                        double(rtot + nrecv[size_t(r)]) * sizeof(R) > double(INT_MAX))
                        throw std::runtime_error("exchange_records: more than 2 GB in one exchange");
                    sc[size_t(r)] = int(nsend[size_t(r)] * (long long int)sizeof(R));
                    rc[size_t(r)] = int(nrecv[size_t(r)] * (long long int)sizeof(R));
                    sd[size_t(r)] = int(stot * (long long int)sizeof(R));
                    rd[size_t(r)] = int(rtot * (long long int)sizeof(R));
                    stot += nsend[size_t(r)];
                    rtot += nrecv[size_t(r)];
                }
                std::vector<R> sendbuf;
                sendbuf.reserve(static_cast<size_t>(stot));
                for (int r = 0; r < NT; r++)
                    sendbuf.insert(sendbuf.end(), out[size_t(r)].begin(), out[size_t(r)].end());
                std::vector<R> recvbuf(static_cast<size_t>(rtot));
                MPI_Alltoallv(sendbuf.data(), sc.data(), sd.data(), MPI_BYTE, recvbuf.data(), rc.data(), rd.data(),
                              MPI_BYTE, FML::communicator);
                if (from)
                    *from = nrecv;
                return recvbuf;
#else
                if (from)
                    *from = {(long long int)out[0].size()};
                return out[0];
#endif
            }

            inline int hash_task(long long int id) {
                uint64_t x = uint64_t(id) + 0x9e3779b97f4a7c15ULL;
                x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
                x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
                x ^= x >> 31;
                return int(x % uint64_t(FML::NTasks));
            }

            /// The task that owns each x slice of grids with the layout of this one
            inline std::vector<int> slab_owners(const FML::GRID::FFTWGrid<3> & layout) {
                const int N = layout.get_nmesh();
                long long int mine[2] = {layout.get_local_x_start(), layout.get_local_nx()};
                std::vector<long long int> all(2 * size_t(FML::NTasks), 0);
#ifdef USE_MPI
                MPI_Allgather(mine, 2, MPI_LONG_LONG, all.data(), 2, MPI_LONG_LONG, FML::communicator);
#else
                all[0] = mine[0];
                all[1] = mine[1];
#endif
                std::vector<int> owner(static_cast<size_t>(N), -1);
                for (int t = 0; t < FML::NTasks; t++)
                    for (long long int ix = all[2 * size_t(t)]; ix < all[2 * size_t(t)] + all[2 * size_t(t) + 1]; ix++)
                        owner[size_t(ix)] = t;
                return owner;
            }

            /// The linear growth factor D(a) (unnormalized) of an expansion history: D'' + (2 + dlnE/dlna) D' =
            /// 3/2 Omega_m(a) D in ln a, from the growing mode D = a deep in matter domination
            inline double linear_growth(const Background & bg, double a, double a_start) {
                const double x0 = std::log(a_start), x1 = std::log(a);
                const int n = 4000;
                const double h = (x1 - x0) / n;
                auto dlnE = [&](double x) {
                    const double e = 1e-4;
                    return (std::log(bg.E(std::exp(x + e))) - std::log(bg.E(std::exp(x - e)))) / (2 * e);
                };
                auto rhs = [&](double x, double D, double Dp, double & dD, double & dDp) {
                    dD = Dp;
                    dDp = -(2.0 + dlnE(x)) * Dp + 1.5 * bg.Omega_m(std::exp(x)) * D;
                };
                double D = a_start, Dp = a_start, x = x0;
                for (int i = 0; i < n; i++) {
                    double k1, l1, k2, l2, k3, l3, k4, l4;
                    rhs(x, D, Dp, k1, l1);
                    rhs(x + h / 2, D + h / 2 * k1, Dp + h / 2 * l1, k2, l2);
                    rhs(x + h / 2, D + h / 2 * k2, Dp + h / 2 * l2, k3, l3);
                    rhs(x + h, D + h * k3, Dp + h * l3, k4, l4);
                    D += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
                    Dp += h / 6 * (l1 + 2 * l2 + 2 * l3 + l4);
                    x += h;
                }
                return D;
            }

            struct LatticeRecord {
                long long int lattice;
                double psi[3];
            };
            struct IdLattice {
                long long int id;
                long long int lattice;
            };
            struct IdQuery {
                long long int id;
                long long int index;
            };
            struct IndexLattice {
                long long int index;
                long long int lattice;
            };
        } // namespace detail

        /// D(a) / D(a_L) of an expansion history (scale independent). a_start: where D = a is assumed
        inline double growth_ratio(const Background & bg, double a, double a_L) {
            const double a_start = std::min(1e-3, 0.02 * std::min(a, a_L));
            return detail::linear_growth(bg, a, a_start) / detail::linear_growth(bg, a_L, a_start);
        }

        //=============================================================================
        /// Option A: the linear field from an initial-conditions snapshot (positions in box units, IDs). Returns
        /// the field (at the redshift of the IC) and, in id_lattice, (id, lattice index) of the local IC particles.
        /// Collective
        //=============================================================================
        template <class Q>
        HybridBiasLinearField linear_field_from_ic(Q * ic,
                                                   size_t nic,
                                                   double ic_redshift,
                                                   bool correct_2lpt,
                                                   std::vector<detail::IdLattice> & id_lattice) {
            static_assert(FML::PARTICLE::has_get_id<Q>(), "linear_field_from_ic: the particles need IDs");
            size_t ntot = nic;
            FML::SumOverTasks(&ntot);
            const int N = int(std::llround(std::cbrt(double(ntot))));
            if ((long long int)N * N * N != (long long int)ntot)
                throw std::runtime_error("Hybrid bias: the initial-conditions snapshot has " + std::to_string(ntot) +
                                         " particles, not a lattice N^3");
            if (N < FML::NTasks)
                throw std::runtime_error("Hybrid bias: the lattice is smaller than the number of MPI tasks");

            // The lattice offset: the circular mean of the position within a lattice cell
            std::array<double, 6> cs{};
            for (size_t i = 0; i < nic; i++) {
                const auto * x = FML::PARTICLE::GetPos(ic[i]);
                for (int d = 0; d < 3; d++) {
                    const double f = 2.0 * M_PI * (x[d] * N - std::floor(x[d] * N));
                    cs[2 * d] += std::cos(f);
                    cs[2 * d + 1] += std::sin(f);
                }
            }
            FML::SumArrayOverTasks(cs.data(), cs.size());
            HybridBiasLinearField out;
            out.nlattice = N;
            out.redshift = ic_redshift;
            out.source = "ic_snapshot";
            for (int d = 0; d < 3; d++) {
                double o = std::atan2(cs[2 * d + 1], cs[2 * d]) / (2.0 * M_PI);
                out.lattice_offset[d] = o - std::floor(o);
            }

            // Lattice point and displacement of each IC particle, sent to the task owning the x slice
            std::array<FML::GRID::FFTWGrid<3>, 3> psi;
            for (auto & g : psi)
                g = FML::GRID::FFTWGrid<3>(N);
            const auto owner = detail::slab_owners(psi[0]);
            std::vector<std::vector<detail::LatticeRecord>> out_records(static_cast<size_t>(FML::NTasks));
            id_lattice.clear();
            id_lattice.reserve(nic);
            double maxdisp = 0.0;
            for (size_t i = 0; i < nic; i++) {
                const auto * x = FML::PARTICLE::GetPos(ic[i]);
                std::array<long long int, 3> q;
                detail::LatticeRecord rec;
                for (int d = 0; d < 3; d++) {
                    const double u = x[d] * N - out.lattice_offset[d];
                    const double qu = std::floor(u + 0.5);
                    rec.psi[d] = (u - qu) / N; // box units, |psi| <= half a cell
                    maxdisp = std::max(maxdisp, std::abs(u - qu));
                    q[d] = ((long long int)qu % N + N) % N;
                }
                rec.lattice = (q[0] * N + q[1]) * N + q[2];
                out_records[size_t(owner[size_t(q[0])])].push_back(rec);
                id_lattice.push_back({(long long int)FML::PARTICLE::GetID(ic[i]), rec.lattice});
            }
            FML::MaxOverTasks(&maxdisp);
            out.max_displacement_cells = maxdisp;
            const auto received = detail::exchange_records(out_records);

            const long long int x_start = psi[0].get_local_x_start();
            const long long int local_nx = psi[0].get_local_nx();
            std::vector<int> occupancy(static_cast<size_t>(local_nx) * size_t(N) * size_t(N), 0);
            for (auto & rec : received) {
                const long long int ix = rec.lattice / ((long long int)N * N) - x_start;
                const int iy = int(rec.lattice / N % N), iz = int(rec.lattice % N);
                occupancy[(static_cast<size_t>(ix) * size_t(N) + size_t(iy)) * size_t(N) + size_t(iz)]++;
                for (int d = 0; d < 3; d++)
                    psi[size_t(d)].set_real({int(ix), iy, iz}, FML::GRID::FloatType(rec.psi[d]));
            }
            long long int bad = 0;
            for (int o : occupancy)
                bad += (o != 1);
            FML::SumOverTasks(&bad);
            if (bad > 0)
                throw std::runtime_error("Hybrid bias: " + std::to_string(bad) +
                                         " lattice points of the initial-conditions snapshot do not have exactly one "
                                         "particle (largest displacement " +
                                         std::to_string(maxdisp) +
                                         " cells): the IC must be at a redshift where no particle moved half a cell");

            // delta = -div Psi (k in units of 1/box; the Nyquist component of a derivative is ambiguous: 0)
            for (auto & g : psi)
                g.fftw_r2c();
            const double knyquist = M_PI * N * (1.0 - 1e-10);
            FML::GRID::FFTWGrid<3> delta = psi[0];
            FML::GRID::for_each_fourier_mode<3>(delta, [&](size_t index, double) {
                std::array<double, 3> kvec;
                double kmag;
                delta.get_fourier_wavevector_and_norm_by_index(index, kvec, kmag);
                FML::GRID::ComplexType div = 0.0;
                for (int d = 0; d < 3; d++)
                    if (std::abs(kvec[d]) < knyquist)
                        div += FML::GRID::ComplexType(0, kvec[d]) * psi[size_t(d)].get_fourier_from_index(index);
                delta.set_fourier_from_index(index, -div);
            });
            for (auto & g : psi)
                g = FML::GRID::FFTWGrid<3>();

            // 2LPT: -div Psi = D1 delta_1 - D2 S[phi_1] with D2 = -3/7 D1^2, so the linear field is
            // -div Psi - 3/7 S[phi] with nabla^2 phi = -div Psi, S = sum_{i<j} (phi_ii phi_jj - phi_ij^2)
            if (correct_2lpt) {
                const int ii[6] = {0, 1, 2, 0, 0, 1};
                const int jj[6] = {0, 1, 2, 1, 2, 2};
                std::array<FML::GRID::FFTWGrid<3>, 6> phi;
                for (int c = 0; c < 6; c++) {
                    phi[size_t(c)] = delta;
                    auto & g = phi[size_t(c)];
                    FML::GRID::for_each_fourier_mode<3>(g, [&](size_t index, double) {
                        std::array<double, 3> kvec;
                        double kmag;
                        g.get_fourier_wavevector_and_norm_by_index(index, kvec, kmag);
                        double fac = 0.0;
                        const bool nyquist = ii[c] != jj[c] and
                                             (std::abs(kvec[size_t(ii[c])]) >= knyquist or
                                              std::abs(kvec[size_t(jj[c])]) >= knyquist);
                        if (kmag > 0.0 and not nyquist)
                            fac = kvec[size_t(ii[c])] * kvec[size_t(jj[c])] / (kmag * kmag);
                        g.set_fourier_from_index(index, g.get_fourier_from_index(index) * FML::GRID::FloatType(fac));
                    });
                    g.fftw_c2r();
                }
                FML::GRID::FFTWGrid<3> S(N);
                for (int ix = 0; ix < local_nx; ix++)
                    for (int iy = 0; iy < N; iy++)
                        for (int iz = 0; iz < N; iz++) {
                            const std::array<int, 3> c{ix, iy, iz};
                            const double xx = phi[0].get_real(c), yy = phi[1].get_real(c), zz = phi[2].get_real(c);
                            const double xy = phi[3].get_real(c), xz = phi[4].get_real(c), yz = phi[5].get_real(c);
                            S.set_real(c, FML::GRID::FloatType(xx * yy + xx * zz + yy * zz - xy * xy - xz * xz - yz * yz));
                        }
                for (auto & g : phi)
                    g = FML::GRID::FFTWGrid<3>();
                S.fftw_r2c();
                FML::GRID::linear_combination_fourier<3>(delta, S, 1.0, -3.0 / 7.0);
            }
            out.delta_fourier = std::move(delta);
            return out;
        }

        /// Option C: the linear field from an FML linear field file (checks the box size). Collective
        inline HybridBiasLinearField linear_field_from_file(const std::string & filename, double boxsize) {
            namespace HB = FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
            HB::LinearFieldFileHeader header;
            auto grid = HB::read_linear_field<3>(filename, header);
            if (std::abs(header.boxsize / boxsize - 1.0) > 1e-6)
                throw std::runtime_error("Hybrid bias: the linear field file has boxsize " +
                                         std::to_string(header.boxsize) + " Mpc/h, the snapshot " +
                                         std::to_string(boxsize));
            grid.fftw_r2c();
            HybridBiasLinearField out;
            out.delta_fourier = std::move(grid);
            out.nlattice = int(header.nmesh);
            out.redshift = header.redshift;
            out.source = "file";
            return out;
        }

        /// The lattice index of each local particle from (id, lattice) pairs held on any task (hashed exchange by
        /// ID). Throws if a particle has no match or an ID is duplicated. Collective
        template <class P>
        std::vector<long long int> lattice_index_by_id(const P * part,
                                                       size_t npart,
                                                       const std::vector<detail::IdLattice> & id_lattice) {
            const int NT = FML::NTasks;
            std::vector<std::vector<detail::IdLattice>> pairs_out(static_cast<size_t>(NT));
            for (auto & p : id_lattice)
                pairs_out[size_t(detail::hash_task(p.id))].push_back(p);
            const auto pairs = detail::exchange_records(pairs_out);
            std::unordered_map<long long int, long long int> map;
            map.reserve(pairs.size() * 2 + 1);
            long long int duplicates = 0;
            for (auto & p : pairs)
                duplicates += not map.emplace(p.id, p.lattice).second;
            FML::SumOverTasks(&duplicates);
            if (duplicates > 0)
                throw std::runtime_error("Hybrid bias: " + std::to_string(duplicates) +
                                         " duplicate particle IDs in the initial-conditions snapshot");

            std::vector<std::vector<detail::IdQuery>> queries_out(static_cast<size_t>(NT));
            for (size_t i = 0; i < npart; i++) {
                const long long int id = (long long int)FML::PARTICLE::GetID(const_cast<P &>(part[i]));
                queries_out[size_t(detail::hash_task(id))].push_back({id, (long long int)i});
            }
            std::vector<long long int> from;
            const auto queries = detail::exchange_records(queries_out, &from);
            std::vector<std::vector<detail::IndexLattice>> answers_out(static_cast<size_t>(NT));
            size_t k = 0;
            for (int t = 0; t < NT; t++)
                for (long long int j = 0; j < from[size_t(t)]; j++, k++) {
                    const auto it = map.find(queries[k].id);
                    answers_out[size_t(t)].push_back({queries[k].index, it == map.end() ? -1 : it->second});
                }
            const auto answers = detail::exchange_records(answers_out);
            std::vector<long long int> lattice(npart, -1);
            for (auto & a : answers)
                lattice[size_t(a.index)] = a.lattice;
            long long int missing = 0;
            for (auto l : lattice)
                missing += (l < 0);
            FML::SumOverTasks(&missing);
            if (missing > 0)
                throw std::runtime_error("Hybrid bias: " + std::to_string(missing) +
                                         " particles have no match (by ID) in the initial-conditions snapshot");
            return lattice;
        }

        /// The lattice index of each local particle from its ID: id - offset (checked to be in [0, N^3))
        template <class P>
        std::vector<long long int> lattice_index_from_ids(const P * part, size_t npart, int N, long long int offset) {
            std::vector<long long int> lattice(npart);
            long long int bad = 0;
            const long long int n3 = (long long int)N * N * N;
            for (size_t i = 0; i < npart; i++) {
                lattice[i] = (long long int)FML::PARTICLE::GetID(const_cast<P &>(part[i])) - offset;
                bad += (lattice[i] < 0 or lattice[i] >= n3);
            }
            FML::SumOverTasks(&bad);
            if (bad > 0)
                throw std::runtime_error("Hybrid bias: " + std::to_string(bad) +
                                         " particle IDs minus hybridbias_id_offset are outside [0, N^3) of the lattice "
                                         "of the linear field file");
            return lattice;
        }

        /// The result: the spectra in physical units and what was used
        struct HybridBiasResult {
            FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::HybridBiasSpectra<3> spectra;
            std::vector<double> plinear; // D^2 |delta_L(k)|^2 of this realization in the bins ((Mpc/h)^3)
            int nlattice{0};
            double linear_field_redshift{0.0};
            double growth_ratio{1.0};
            std::string source;
            std::array<double, 3> lattice_offset{};
            double max_displacement_cells{0.0};
        };

        //=============================================================================
        /// The hybrid bias spectra of the particles (positions in box units, IDs). ic: the initial-conditions
        /// particles (nic may be 0 when the linear field is from a file and the lattice from the IDs). Collective
        //=============================================================================
        template <class P, class Q>
        HybridBiasResult measure_hybrid_bias(const P * part,
                                             size_t npart,
                                             Q * ic,
                                             size_t nic,
                                             double ic_redshift,
                                             const HybridBiasSettings & settings,
                                             const AnalysisContext & context) {
            namespace HB = FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
            settings.validate();
            size_t nic_tot = nic;
            FML::SumOverTasks(&nic_tot);
            const bool have_ic = nic_tot > 0;

            HybridBiasLinearField field;
            std::vector<detail::IdLattice> id_lattice;
            if (settings.linear_field_source == "ic_snapshot") {
                if (not have_ic)
                    throw std::runtime_error("Hybrid bias: the initial-conditions snapshot has no particles");
                field = linear_field_from_ic(ic, nic, ic_redshift, settings.ic_lpt_correction == "2lpt", id_lattice);
            } else {
                field = linear_field_from_file(settings.linear_field_file, context.boxsize);
                if (have_ic) {
                    // The lattice from the IC snapshot (its field is not used)
                    std::vector<detail::IdLattice> pairs;
                    auto ic_field = linear_field_from_ic(ic, nic, ic_redshift, false, pairs);
                    if (ic_field.nlattice != field.nlattice)
                        throw std::runtime_error("Hybrid bias: the IC snapshot lattice (" +
                                                 std::to_string(ic_field.nlattice) + ") differs from the file's (" +
                                                 std::to_string(field.nlattice) + ")");
                    field.lattice_offset = ic_field.lattice_offset;
                    field.max_displacement_cells = ic_field.max_displacement_cells;
                    id_lattice = std::move(pairs);
                }
            }
            const int N = field.nlattice;
            size_t ntot = npart;
            FML::SumOverTasks(&ntot);
            if (double(ntot) != std::pow(double(N), 3))
                throw std::runtime_error("Hybrid bias: the snapshot has " + std::to_string(ntot) +
                                         " particles, the lattice " + std::to_string(N) + "^3");

            const auto lattice = have_ic ? lattice_index_by_id(part, npart, id_lattice)
                                         : lattice_index_from_ids(part, npart, N, settings.id_offset);
            auto lattice_coordinate = [&lattice, part, N](const P & p) -> std::array<int, 3> {
                const long long int L = lattice[size_t(&p - part)];
                return {int(L / ((long long int)N * N)), int(L / N % N), int(L % N)};
            };

            const double a_L = 1.0 / (1.0 + field.redshift);
            const double ratio = settings.growth_ratio > 0.0 ? settings.growth_ratio
                                                             : growth_ratio(context.background, context.scale_factor, a_L);

            HB::HybridBiasOptions options;
            options.noperators = settings.noperators;
            options.density_assignment_method = settings.density_assignment_method;
            options.interlacing = settings.interlacing;
            options.subtract_shotnoise = settings.subtract_shotnoise;
            options.smoothing_scale = settings.smoothing_scale / context.boxsize;
            options.smoothing_method = settings.smoothing_filter;
            options.kcut = settings.kcut_over_nyquist * M_PI * N;

            HybridBiasResult result;
            HB::compute_hybrid_bias_spectra<3, P>(part,
                                                  npart,
                                                  ntot,
                                                  field.delta_fourier,
                                                  N,
                                                  settings.nmesh,
                                                  options,
                                                  result.spectra,
                                                  [ratio](double) { return ratio; },
                                                  lattice_coordinate);

            // The power spectrum of the linear field itself in the same bins (before scaling to physical units)
            auto plin = result.spectra.pofk[0];
            std::fill(plin.pofk.begin(), plin.pofk.end(), 0.0);
            std::fill(plin.count.begin(), plin.count.end(), 0.0);
            FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum<3>(field.delta_fourier, plin);
            plin.scale(context.boxsize);
            result.spectra.scale(context.boxsize);
            result.plinear.resize(plin.pofk.size());
            for (size_t i = 0; i < plin.pofk.size(); i++)
                result.plinear[i] = ratio * ratio * plin.pofk[i];

            result.nlattice = N;
            result.linear_field_redshift = field.redshift;
            result.growth_ratio = ratio;
            result.source = field.source;
            result.lattice_offset = field.lattice_offset;
            result.max_displacement_cells = field.max_displacement_cells;
            return result;
        }

        /// Write the spectra (SimulationCode's hybridbias_z<z>.txt columns, with the realization's linear P(k) in
        /// place of the model's). Collective
        inline void write_hybrid_bias(const std::string & filename,
                                      const HybridBiasResult & r,
                                      const HybridBiasSettings & settings,
                                      double redshift,
                                      double boxsize,
                                      const std::string & header_text = "") {
            std::ostringstream header;
            header << header_text;
            header << "# Hybrid bias spectra at z = " << redshift << " (boxsize " << boxsize << " Mpc/h, lattice "
                   << r.nlattice << "^3, nmesh " << settings.nmesh << ", " << settings.density_assignment_method
                   << (settings.interlacing ? " with interlacing" : "") << ", shot noise subtracted: " << std::boolalpha
                   << settings.subtract_shotnoise << ")\n";
            header << "# Linear field: " << r.source << " at z = " << r.linear_field_redshift
                   << ", scaled with D(z)/D(z_L) = " << r.growth_ratio;
            if (r.source == "ic_snapshot")
                header << " (lattice offset " << r.lattice_offset[0] << " " << r.lattice_offset[1] << " "
                       << r.lattice_offset[2] << " cells, largest displacement " << r.max_displacement_cells
                       << " cells, " << settings.ic_lpt_correction << " correction)";
            header << "\n";
            header << "# The tracer power spectrum is P_tt = sum_ab b_a b_b P_ab with b = (1, b1, b2, bs, bnabla) for "
                      "the operators (1, delta_L, delta_L^2 - <delta_L^2>, s^2 - <s^2>, nabla^2 delta_L)\n";
            header << "# Plinear(k): D^2 |delta_L(k)|^2 of this realization's linear field in the same bins\n";
            r.spectra.write(filename, header.str(), {{"Plinear(k)", r.plinear}});
#ifdef USE_MPI
            MPI_Barrier(FML::communicator);
#endif
        }

    } // namespace ANALYSIS
} // namespace FML
#endif
