#ifndef FML_HYBRIDBIASSPECTRA_HEADER
#define FML_HYBRIDBIASSPECTRA_HEADER

#include <FML/Global/MPICounts.h>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/ComputePowerSpectra/HybridBiasOperators.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/NUFFT/NUFFT.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Smoothing/SmoothingFourier.h>

namespace FML {
    namespace CORRELATIONFUNCTIONS {

        //=====================================================================================================
        ///
        /// Hybrid bias (Lagrangian bias basis) spectra, Modi, Chen & White 2020 (1910.07097), Kokron et al. 2021
        /// (2101.11014), Zennaro et al. 2021. The tracer field is written as a sum of Lagrangian bias operators
        /// advected with the particles of a simulation:
        ///
        ///   delta_t(k) = sum_a b_a O_a(k), O_a(x) = int d^3q O_a(q) delta_D(x - q - Psi(q))
        ///
        /// with the operators (b_1 = 1, b_delta = b_1^L, ...)
        ///
        ///   1, delta_L, delta_L^2 - <delta_L^2>, s^2 - <s^2>, nabla^2 delta_L
        ///
        /// where delta_L(q) is the linear density field (scaled to the redshift of the snapshot), s_ij = (k_i k_j / k^2
        /// - delta_ij / N) delta_L(k) the tidal field and s^2 = s_ij s_ij. The 15 cross spectra P_ab(k) of the advected
        /// fields give the tracer power spectrum for any bias parameters: P_tt = sum_ab b_a b_b P_ab (with b_1 = 1 for
        /// the field "1"), and P_tm = sum_a b_a P_1a.
        ///
        /// The simulation particles must have started on a regular lattice of Nlattice^N points (as in N-body
        /// initial conditions); the Lagrangian field of a particle is the field at its lattice point. The fields are
        /// computed on a grid of size Nlattice (all Fourier modes |m_i| < min(Nmesh_lin, Nlattice) / 2 of the linear
        /// field are kept) so the values at the lattice points are exact. Particle i then carries the weight O_a(q_i)
        /// and the advected field is the weighted density: O_a(x) = (1 / nbar) sum_i O_a(q_i) W(x - x_i).
        ///
        /// Units: positions in [0,1) (units of the box). nabla^2 delta_L is computed with k in units of 1/box; scale()
        /// puts the spectra in physical units (k in 1/box -> k / B, P -> B^N P, and an extra 1/B^2 for every nabla^2
        /// field so that nabla^2 delta_L has units of length^2).
        ///
        /// The lattice point of a particle is by default taken from its ID: id = (ix Nlattice + iy) Nlattice + iz
        /// (which is how MPIParticles::create_particle_grid and NBodyInitialConditions number the particles).
        /// Everything works with MPI (the particles can be anywhere, we fetch the Lagrangian fields from the task that
        /// owns them).
        ///
        /// Example:
        ///
        ///   using namespace FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
        ///   HybridBiasOptions options;
        ///   HybridBiasSpectra<3> spectra;
        ///   compute_hybrid_bias_spectra<3>(part.get_particles_ptr(), part.get_npart(), part.get_npart_total(),
        ///                                  delta_linear_fourier, Npart_1D, Nmesh, options, spectra);
        ///   spectra.scale(boxsize);
        ///   auto & P_delta_s2 = spectra.get("delta", "s2"); // a PowerSpectrumBinning
        ///
        //=====================================================================================================
        namespace HYBRIDBIAS {

            template <int N>
            using FFTWGrid = FML::GRID::FFTWGrid<N>;


            /// Options
            struct HybridBiasOptions {
                /// How many of the operators (1, delta, delta2, s2, nabla2delta) to use (1 to 5)
                int noperators{5};
                /// Density assignment of the advected fields (NGP, CIC, TSC, PCS or PQS)
                std::string density_assignment_method{"CIC"};
                /// Interlaced grids for the advected fields to reduce aliasing
                bool interlacing{true};
                /// Sharp spherical cut of the linear field before we form the operators: keep |k| < kcut (in units of
                /// 1/box, 0: no cut, i.e. all modes of the lattice grid, |m_i| < Nlattice / 2). With kcut = the Nyquist
                /// frequency of the lattice, pi Nlattice, the operators are isotropic (as assumed in the LPT
                /// predictions)
                double kcut{0.0};
                /// Smoothing of the linear field before we form the operators (scale in units of the box, 0: none)
                double smoothing_scale{0.0};
                /// gaussian, sharpk or tophat
                std::string smoothing_method{"gaussian"};
                /// Subtract the mean of delta^2 and s^2 over the lattice (the usual definition)
                bool subtract_lagrangian_means{true};
                /// Subtract the Poisson shot noise mean(w_a w_b) / Npart from P_ab. The usual choice for the basis
                /// spectra is false (the shot noise of the simulation particles is then part of the spectra)
                bool subtract_shotnoise{false};
                /// Number of k-bins (0: Nmesh / 2). The bins are linearly spaced from the fundamental mode to the
                /// Nyquist frequency
                int nbins{0};
            };

            /// The result: the cross spectra of the advected operators for all pairs a <= b
            template <int N>
            struct HybridBiasSpectra {
                std::vector<std::string> names{};
                /// The pairs (a,b), a <= b, in the order the spectra are stored: (0,0), (0,1), ..., (1,1), ...
                std::vector<std::array<int, 2>> pairs{};
                std::vector<PowerSpectrumBinning<N>> pofk{};
                /// The Poisson shot noise mean(w_a w_b) / Npart (subtracted if subtract_shotnoise)
                std::vector<double> shotnoise{};
                bool scaled{false};

                int index(int a, int b) const {
                    if (a > b)
                        std::swap(a, b);
                    for (size_t i = 0; i < pairs.size(); i++)
                        if (pairs[i][0] == a and pairs[i][1] == b)
                            return int(i);
                    throw std::runtime_error("[HybridBiasSpectra] Pair not found");
                }
                int operator_index(const std::string & name) const {
                    for (size_t i = 0; i < names.size(); i++)
                        if (names[i] == name)
                            return int(i);
                    throw std::runtime_error("[HybridBiasSpectra] Unknown operator " + name);
                }
                PowerSpectrumBinning<N> & get(int a, int b) { return pofk[index(a, b)]; }
                PowerSpectrumBinning<N> & get(const std::string & a, const std::string & b) {
                    return get(operator_index(a), operator_index(b));
                }

                /// To physical units: k -> k / B, P -> B^N P, and 1/B^2 for every nabla^2 delta in the pair
                void scale(double boxsize) {
                    assert_mpi(not scaled, "[HybridBiasSpectra::scale] Already scaled");
                    for (size_t i = 0; i < pairs.size(); i++) {
                        pofk[i].scale(boxsize);
                        int nlap = (pairs[i][0] == NABLA2DELTA) + (pairs[i][1] == NABLA2DELTA);
                        const double fac = std::pow(boxsize, -2.0 * nlap);
                        for (auto & p : pofk[i].pofk)
                            p *= fac;
                        shotnoise[i] *= std::pow(boxsize, N) * fac;
                    }
                    scaled = true;
                }

                /// The tracer power spectrum P_tt(k) = sum_ab b_a b_b P_ab for the bias parameters
                /// (1, b1, b2, bs, bnabla) (a missing operator has bias 0). Use b = {1, b1, b2/2, bs, bnabla} etc.
                /// depending on your convention for the bias expansion
                std::vector<double> tracer_power_spectrum(const std::vector<double> & b) const {
                    std::vector<double> result(pofk[0].n, 0.0);
                    for (size_t i = 0; i < pairs.size(); i++) {
                        const int a = pairs[i][0], c = pairs[i][1];
                        const double ba = a < int(b.size()) ? b[a] : 0.0;
                        const double bc = c < int(b.size()) ? b[c] : 0.0;
                        const double fac = (a == c ? 1.0 : 2.0) * ba * bc;
                        for (int k = 0; k < pofk[i].n; k++)
                            result[k] += fac * pofk[i].pofk[k];
                    }
                    return result;
                }

                /// Write all the spectra to file (only task 0): k, the number of modes, P_ab for all pairs and then
                /// the extra columns (name, values in each bin) if any
                void write(const std::string & filename,
                           const std::string & header = "",
                           const std::vector<std::pair<std::string, std::vector<double>>> & extra_columns = {}) const {
                    if (FML::ThisTask != 0)
                        return;
                    std::ofstream fp(filename);
                    assert_mpi(fp.good(), ("[HybridBiasSpectra::write] Could not open " + filename).c_str());
                    if (header != "")
                        fp << header;
                    fp << "# Hybrid bias (Lagrangian bias basis) spectra P_ab(k) of the advected operators "
                       << (scaled ? "(k in h/Mpc, P in (Mpc/h)^" + std::to_string(N) + ", nabla2delta in (Mpc/h)^2)" :
                                    "(box units)")
                       << "\n# Column 1: k, column 2: number of modes, then P_ab for (a,b) = ";
                    for (auto & p : pairs)
                        fp << "(" << names[p[0]] << "," << names[p[1]] << ") ";
                    for (auto & c : extra_columns)
                        fp << "then " << c.first << " ";
                    fp << "\n# Poisson shot noise mean(w_a w_b)/Npart:";
                    for (auto s : shotnoise)
                        fp << " " << s;
                    fp << "\n";
                    fp << std::setprecision(8);
                    for (int k = 0; k < pofk[0].n; k++) {
                        fp << std::setw(15) << pofk[0].kbin[k] << " " << std::setw(15) << pofk[0].count[k];
                        for (auto & p : pofk)
                            fp << " " << std::setw(15) << p.pofk[k];
                        for (auto & c : extra_columns)
                            fp << " " << std::setw(15) << c.second[k];
                        fp << "\n";
                    }
                }
            };

            namespace detail {
                /// To make a function argument non-deduced (std::type_identity in C++20)
                template <class U>
                struct identity {
                    using type = U;
                };
                template <int N, class T>
                using LatticeCoordinateFunction = typename identity<std::function<std::array<int, N>(const T &)>>::type;

                /// The particle we use for assigning the advected fields: the weight is used without being
                /// normalized by its mean (as for delta-f particles)
                template <int N>
                struct WeightedParticle {
                    double pos[N];
                    double weight;
                    double * get_pos() { return pos; }
                    constexpr int get_ndim() const { return N; }
                    double get_deltaf_weight() const { return weight; }
                };

                /// Mean of a real grid (over all tasks)
                template <int N>
                double mean_of_real_grid(const FFTWGrid<N> & grid) {
                    double sum = 0.0;
                    const auto Local_nx = grid.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : sum)
#endif
                    for (int islice = 0; islice < Local_nx; islice++)
                        for (auto && real_index : grid.get_real_range(islice, islice + 1))
                            sum += grid.get_real_from_index(real_index);
                    FML::SumOverTasks(&sum);
                    return sum / std::pow(double(grid.get_nmesh()), N);
                }

                /// The lattice point of a particle from its id = (ix Nlattice + iy) Nlattice + iz
                template <int N, class T>
                std::function<std::array<int, N>(const T &)> lattice_coordinate_from_id(int Nlattice) {
                    if constexpr (FML::PARTICLE::has_get_id<T>()) {
                        return [Nlattice](const T & part) -> std::array<int, N> {
                            long long int id = (long long int)(FML::PARTICLE::GetID(const_cast<T &>(part)));
                            std::array<int, N> coord;
                            for (int idim = N - 1; idim >= 0; idim--) {
                                coord[idim] = int(id % Nlattice);
                                id /= Nlattice;
                            }
                            return coord;
                        };
                    } else {
                        assert_mpi(false, "[HybridBias] The particles have no get_id, provide lattice_coordinate");
                        return nullptr;
                    }
                }
            } // namespace detail

            //=====================================================================================================
            /// @brief The Lagrangian operators delta_L, delta_L^2 - <delta_L^2>, s^2 - <s^2> and nabla^2 delta_L on
            /// a real grid of size Nlattice (the values at the lattice points q = (i_1,...,i_N) / Nlattice).
            ///
            /// @param[in] delta_linear_fourier The linear density field in Fourier space (any grid size)
            /// @param[in] Nlattice The size of the particle lattice
            /// @param[out] fields The fields delta, delta2, s2, nabla2delta (the first noperators - 1 of them) in
            /// real space
            /// @param[in] options The options (noperators, smoothing, subtract_lagrangian_means)
            /// @param[in] growth_factor_of_kBox Optional: multiply delta_linear by this function of k (in units of
            /// 1/box), e.g. D(k,z) / D(k,zini) to scale the initial field to the redshift of the snapshot
            //=====================================================================================================
            template <int N>
            void compute_lagrangian_fields(const FFTWGrid<N> & delta_linear_fourier,
                                           int Nlattice,
                                           std::vector<FFTWGrid<N>> & fields,
                                           const HybridBiasOptions & options = HybridBiasOptions(),
                                           std::function<double(double)> growth_factor_of_kBox = nullptr) {
                assert_mpi(options.noperators >= 1 and options.noperators <= max_operators,
                           "[compute_lagrangian_fields] noperators must be 1, ..., 5");
                const int nfields = options.noperators - 1;
                fields.clear();
                if (nfields == 0)
                    return;

                // The linear field on the lattice grid (and scaled by the growth factor)
                FFTWGrid<N> delta(Nlattice);
                delta.add_memory_label("HybridBias::delta_L(q)");
                std::function<double(const std::array<int, N> &)> factor = [&](const std::array<int, N> & m) {
                    if (not growth_factor_of_kBox)
                        return 1.0;
                    double k2 = 0.0;
                    for (int idim = 0; idim < N; idim++)
                        k2 += double(m[idim]) * m[idim];
                    return growth_factor_of_kBox(2.0 * M_PI * std::sqrt(k2));
                };
                FML::NUFFT::copy_fourier_modes<N>(delta_linear_fourier, delta, factor);
                if (FML::ThisTask == 0)
                    delta.set_fourier_from_index(0, 0.0);
                if (options.kcut > 0.0) {
                    const double kcut = options.kcut;
                    FML::GRID::multiply_fourier_by_function_of_k<N>(delta, [kcut](double kmag) {
                        return kmag < kcut ? 1.0 : 0.0;
                    });
                }
                if (options.smoothing_scale > 0.0)
                    FML::GRID::smoothing_filter_fourier_space<N>(delta,
                                                                 options.smoothing_scale,
                                                                 options.smoothing_method);

                const auto Local_nx = delta.get_local_nx();

                // nabla^2 delta (k in units of 1/box)
                FFTWGrid<N> nabla2;
                if (nfields >= 4) {
                    nabla2 = delta;
                    FML::GRID::multiply_fourier_by_function_of_k<N>(nabla2, [](double kmag) { return -kmag * kmag; });
                    nabla2.fftw_c2r();
                }

                // s^2 = sum_ij s_ij^2 with s_ij = (k_i k_j / k^2 - delta_ij / N) delta(k)
                FFTWGrid<N> s2;
                if (nfields >= 3) {
                    s2 = FFTWGrid<N>(Nlattice);
                    s2.add_memory_label("HybridBias::s2(q)");
                    s2.fill_real_grid(0.0);
                    FFTWGrid<N> sij(Nlattice);
                    for (int i = 0; i < N; i++) {
                        for (int j = i; j < N; j++) {
                            sij.set_grid_status_real(false);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                            for (int islice = 0; islice < Local_nx; islice++) {
                                [[maybe_unused]] double kmag2;
                                [[maybe_unused]] std::array<double, N> kvec;
                                for (auto && fourier_index : delta.get_fourier_range(islice, islice + 1)) {
                                    delta.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                                    double fac = 0.0;
                                    if (kmag2 > 0.0)
                                        fac = kvec[i] * kvec[j] / kmag2 - (i == j ? 1.0 / N : 0.0);
                                    sij.set_fourier_from_index(fourier_index,
                                                               delta.get_fourier_from_index(fourier_index) *
                                                                   FML::GRID::FloatType(fac));
                                }
                            }
                            sij.fftw_c2r();
                            const double w = i == j ? 1.0 : 2.0;
#ifdef USE_OMP
#pragma omp parallel for
#endif
                            for (int islice = 0; islice < Local_nx; islice++) {
                                for (auto && real_index : sij.get_real_range(islice, islice + 1)) {
                                    const double v = sij.get_real_from_index(real_index);
                                    s2.set_real_from_index(real_index, s2.get_real_from_index(real_index) + w * v * v);
                                }
                            }
                        }
                    }
                }

                // delta and delta^2
                delta.fftw_c2r();
                FFTWGrid<N> delta2;
                if (nfields >= 2) {
                    delta2 = delta;
#ifdef USE_OMP
#pragma omp parallel for
#endif
                    for (int islice = 0; islice < Local_nx; islice++) {
                        for (auto && real_index : delta2.get_real_range(islice, islice + 1)) {
                            const double v = delta2.get_real_from_index(real_index);
                            delta2.set_real_from_index(real_index, v * v);
                        }
                    }
                }

                // Subtract the means
                auto subtract_mean = [&](FFTWGrid<N> & grid) {
                    const double mean = detail::mean_of_real_grid<N>(grid);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                    for (int islice = 0; islice < Local_nx; islice++)
                        for (auto && real_index : grid.get_real_range(islice, islice + 1))
                            grid.set_real_from_index(real_index, grid.get_real_from_index(real_index) - mean);
                };
                if (options.subtract_lagrangian_means) {
                    if (nfields >= 2)
                        subtract_mean(delta2);
                    if (nfields >= 3)
                        subtract_mean(s2);
                }

                fields.push_back(std::move(delta));
                if (nfields >= 2)
                    fields.push_back(std::move(delta2));
                if (nfields >= 3)
                    fields.push_back(std::move(s2));
                if (nfields >= 4)
                    fields.push_back(std::move(nabla2));
            }

            //=====================================================================================================
            /// @brief The values of the Lagrangian fields at the lattice points of the particles. With MPI the
            /// fields are fetched from the task that owns the lattice point.
            ///
            /// @param[in] p Pointer to the particles
            /// @param[in] NumPart Number of local particles
            /// @param[in] fields Real grids of size Nlattice (e.g. from compute_lagrangian_fields)
            /// @param[out] values values[nfields * i + f] is field f at the lattice point of particle i
            /// @param[in] lattice_coordinate Optional: the lattice point of a particle (default: from get_id)
            //=====================================================================================================
            template <int N, class T>
            void lagrangian_fields_at_particles(const T * p,
                                                size_t NumPart,
                                                const std::vector<FFTWGrid<N>> & fields,
                                                std::vector<double> & values,
                                                detail::LatticeCoordinateFunction<N, T> lattice_coordinate = nullptr) {
                FML::COMMUNICATION::require_count(fields.size() <= size_t(INT_MAX), "[hybrid fields] Too many fields");
                FML::COMMUNICATION::Count field_count = fields.size();
                const auto field_counts = FML::GatherFromTasks(&field_count);
                FML::COMMUNICATION::require_count(
                    std::all_of(field_counts.begin(), field_counts.end(), [&](auto n) { return n == field_count; }),
                    "[hybrid fields] Field counts differ between ranks");
                const int nfields = int(fields.size());
                const size_t nvalues = FML::COMMUNICATION::checked_bytes(NumPart, size_t(nfields), values.max_size());
                values.assign(nvalues, 0.0);
                if (nfields == 0)
                    return;
                const int Nlattice = fields[0].get_nmesh();
                if (not lattice_coordinate)
                    lattice_coordinate = detail::lattice_coordinate_from_id<N, T>(Nlattice);
                [[maybe_unused]] long long int local_x_start = fields[0].get_local_x_start();
                [[maybe_unused]] long long int local_nx = fields[0].get_local_nx();

                // The local index (in the slab of the owner) of a lattice point
                auto local_index = [&](const std::array<int, N> & coord, long long int x_start) {
                    auto c = coord;
                    c[0] = int(coord[0] - x_start);
                    return fields[0].get_index_real(c);
                };
                auto check = [&](const std::array<int, N> & coord) {
                    for (int idim = 0; idim < N; idim++)
                        assert_mpi(coord[idim] >= 0 and coord[idim] < Nlattice,
                                   "[lagrangian_fields_at_particles] Lattice coordinate out of range (do the IDs "
                                   "match the lattice?)");
                };

                if (FML::NTasks == 1) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
                    for (size_t i = 0; i < NumPart; i++) {
                        const auto coord = lattice_coordinate(p[i]);
                        check(coord);
                        const auto index = local_index(coord, 0);
                        for (int f = 0; f < nfields; f++)
                            values[i * nfields + f] = fields[f].get_real_from_index(index);
                    }
                    return;
                }

#ifdef USE_MPI
                // The slabs of the lattice grid
                auto x_start_of_task = FML::GatherFromTasks(&local_x_start);
                auto nx_of_task = FML::GatherFromTasks(&local_nx);
                std::vector<int> owner_of_x(Nlattice, -1);
                for (int t = 0; t < FML::NTasks; t++)
                    for (long long int ix = x_start_of_task[t]; ix < x_start_of_task[t] + nx_of_task[t]; ix++)
                        owner_of_x[ix] = t;

                // We do it in chunks of particles to keep the MPI buffers below 2 GB
                const size_t chunk = size_t(1) << 24;
                long long int nchunks = (long long int)(NumPart / chunk + (NumPart % chunk != 0));
                FML::MaxOverTasks(&nchunks);
                std::vector<int> owner;
                std::vector<long long int> index_in_owner;
                for (long long int c = 0; c < nchunks; c++) {
                    const size_t i0 = std::min(NumPart, size_t(c) * chunk);
                    const size_t i1 = i0 + std::min(chunk, NumPart - i0);
                    const size_t n = i1 - i0;

                    // Who has the fields of the particles?
                    owner.resize(n);
                    index_in_owner.resize(n);
                    std::vector<int> nsend(FML::NTasks, 0), nrecv(FML::NTasks, 0);
                    for (size_t i = 0; i < n; i++) {
                        const auto coord = lattice_coordinate(p[i0 + i]);
                        check(coord);
                        owner[i] = owner_of_x[coord[0]];
                        index_in_owner[i] = local_index(coord, x_start_of_task[owner[i]]);
                        nsend[owner[i]]++;
                    }
                    MPI_Alltoall(nsend.data(), 1, MPI_INT, nrecv.data(), 1, MPI_INT, FML::communicator);
                    std::vector<int> sdispl(FML::NTasks, 0), rdispl(FML::NTasks, 0);
                    const auto outgoing = FML::COMMUNICATION::checked_int_layout(nsend);
                    const auto incoming = FML::COMMUNICATION::checked_int_layout(nrecv);
                    sdispl = outgoing.offsets;
                    rdispl = incoming.offsets;
                    const size_t ntotrecv = incoming.total;
                    const auto answer_layout = FML::COMMUNICATION::checked_int_layout(nrecv, size_t(nfields));
                    const auto result_layout = FML::COMMUNICATION::checked_int_layout(nsend, size_t(nfields));

                    // Send the requests (sorted by owner)
                    std::vector<long long int> requests(n), received(ntotrecv);
                    std::vector<size_t> position(n);
                    {
                        auto next = sdispl;
                        for (size_t i = 0; i < n; i++) {
                            position[i] = size_t(next[owner[i]]++);
                            requests[position[i]] = index_in_owner[i];
                        }
                    }
                    MPI_Alltoallv(requests.data(),
                                  nsend.data(),
                                  sdispl.data(),
                                  MPI_LONG_LONG,
                                  received.data(),
                                  nrecv.data(),
                                  rdispl.data(),
                                  MPI_LONG_LONG,
                                  FML::communicator);

                    // Answer them
                    std::vector<double> answers(size_t(ntotrecv) * nfields);
                    for (size_t r = 0; r < ntotrecv; r++)
                        for (int f = 0; f < nfields; f++)
                            answers[size_t(r) * nfields + f] = fields[f].get_real_from_index(received[r]);
                    std::vector<int> nsend_f(FML::NTasks), nrecv_f(FML::NTasks), sdispl_f(FML::NTasks),
                        rdispl_f(FML::NTasks);
                    nsend_f = answer_layout.counts;
                    sdispl_f = answer_layout.offsets;
                    nrecv_f = result_layout.counts;
                    rdispl_f = result_layout.offsets;
                    std::vector<double> result(n * nfields);
                    MPI_Alltoallv(answers.data(),
                                  nsend_f.data(),
                                  sdispl_f.data(),
                                  MPI_DOUBLE,
                                  result.data(),
                                  nrecv_f.data(),
                                  rdispl_f.data(),
                                  MPI_DOUBLE,
                                  FML::communicator);
                    for (size_t i = 0; i < n; i++)
                        for (int f = 0; f < nfields; f++)
                            values[(i0 + i) * nfields + f] = result[position[i] * nfields + f];
                }
                (void)local_nx;
#endif
            }

            //=====================================================================================================
            /// @brief The advected operators in Fourier space: O_a(x) = (1 / nbar) sum_i w_a(i) W(x - x_i) with
            /// w = 1 for the first ("1") and the Lagrangian field values for the others. The window function is
            /// deconvolved and the k = 0 mode is set to zero.
            ///
            /// @param[in] p Pointer to the particles
            /// @param[in] NumPart Number of local particles
            /// @param[in] NumPartTot Total number of particles
            /// @param[in] values The Lagrangian fields at the particles (values[nfields * i + f])
            /// @param[in] nfields Number of fields in values (the result has nfields + 1 grids)
            /// @param[in] Nmesh The grid size for the advected fields
            /// @param[out] advected The nfields + 1 grids in Fourier space
            /// @param[in] options The options (density_assignment_method, interlacing)
            //=====================================================================================================
            template <int N, class T>
            void compute_advected_fields(const T * p,
                                         size_t NumPart,
                                         size_t NumPartTot,
                                         const std::vector<double> & values,
                                         int nfields,
                                         int Nmesh,
                                         std::vector<FFTWGrid<N>> & advected,
                                         const HybridBiasOptions & options = HybridBiasOptions()) {
                assert_mpi(values.size() == NumPart * size_t(nfields),
                           "[compute_advected_fields] values has the wrong size");
                std::vector<detail::WeightedParticle<N>> wp(NumPart);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t i = 0; i < NumPart; i++) {
                    const auto * pos = FML::PARTICLE::GetPos(const_cast<T &>(p[i]));
                    for (int idim = 0; idim < N; idim++)
                        wp[i].pos[idim] = pos[idim];
                }
                const auto nleftright = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(
                    options.density_assignment_method);
                advected.clear();
                for (int f = -1; f < nfields; f++) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
                    for (size_t i = 0; i < NumPart; i++)
                        wp[i].weight = f < 0 ? 1.0 : values[i * nfields + f];
                    FFTWGrid<N> grid(Nmesh, nleftright.first, nleftright.second + (options.interlacing ? 1 : 0));
                    grid.add_memory_label("HybridBias::advected field");
                    FML::INTERPOLATION::particles_to_fourier_grid<N>(wp.data(),
                                                                     NumPart,
                                                                     NumPartTot,
                                                                     grid,
                                                                     options.density_assignment_method,
                                                                     options.interlacing);
                    FML::INTERPOLATION::deconvolve_window_function_fourier<N>(grid, options.density_assignment_method);
                    if (FML::ThisTask == 0)
                        grid.set_fourier_from_index(0, 0.0);
                    advected.push_back(std::move(grid));
                }
            }

            //=====================================================================================================
            /// @brief The cross spectra of a set of grids (in Fourier space) for all pairs a <= b
            ///
            /// @param[in] grids The grids in Fourier space (e.g. the advected operators)
            /// @param[out] spectra The spectra (names, pairs and pofk are set; shotnoise is set to zero)
            /// @param[in] nbins Number of bins (0: Nmesh / 2), linearly spaced from the fundamental mode to the Nyquist
            /// frequency
            //=====================================================================================================
            template <int N>
            void cross_spectra_of_grids(const std::vector<FFTWGrid<N>> & grids,
                                        HybridBiasSpectra<N> & spectra,
                                        int nbins = 0) {
                const int n = int(grids.size());
                assert_mpi(n > 0, "[cross_spectra_of_grids] No grids");
                const int Nmesh = grids[0].get_nmesh();
                if (nbins <= 0)
                    nbins = Nmesh / 2;
                spectra = HybridBiasSpectra<N>();
                const auto & all_names = operator_names();
                for (int a = 0; a < n; a++)
                    spectra.names.push_back(a < max_operators ? all_names[a] : "field" + std::to_string(a));
                for (int a = 0; a < n; a++) {
                    for (int b = a; b < n; b++) {
                        PowerSpectrumBinning<N> binning(2.0 * M_PI,
                                                        2.0 * M_PI * Nmesh / 2,
                                                        nbins,
                                                        PowerSpectrumBinning<N>::LINEAR_SPACING);
                        if (a == b)
                            bin_up_power_spectrum<N>(grids[a], binning);
                        else
                            bin_up_cross_power_spectrum<N>(grids[a], grids[b], binning);
                        spectra.pairs.push_back({a, b});
                        spectra.pofk.push_back(binning);
                        spectra.shotnoise.push_back(0.0);
                    }
                }
            }

            //=====================================================================================================
            /// @brief Compute the hybrid bias spectra from particles and the linear density field. This is
            /// compute_lagrangian_fields + lagrangian_fields_at_particles + compute_advected_fields +
            /// cross_spectra_of_grids. The spectra are in box units, call spectra.scale(boxsize) for physical units.
            ///
            /// @param[in] p Pointer to the particles (they must have started on a lattice of Nlattice^N points)
            /// @param[in] NumPart Number of local particles
            /// @param[in] NumPartTot Total number of particles (Nlattice^N)
            /// @param[in] delta_linear_fourier The linear density field in Fourier space
            /// @param[in] Nlattice The size of the particle lattice
            /// @param[in] Nmesh The grid size for the advected fields
            /// @param[in] options The options
            /// @param[out] spectra The result
            /// @param[in] growth_factor_of_kBox Optional: multiply delta_linear by this (e.g. D(k,z) / D(k,zini))
            /// @param[in] lattice_coordinate Optional: the lattice point of a particle (default from get_id)
            /// @param[out] advected_out Optional: return the advected fields (Fourier space)
            //=====================================================================================================
            template <int N, class T>
            void compute_hybrid_bias_spectra(const T * p,
                                             size_t NumPart,
                                             size_t NumPartTot,
                                             const FFTWGrid<N> & delta_linear_fourier,
                                             int Nlattice,
                                             int Nmesh,
                                             const HybridBiasOptions & options,
                                             HybridBiasSpectra<N> & spectra,
                                             std::function<double(double)> growth_factor_of_kBox = nullptr,
                                             detail::LatticeCoordinateFunction<N, T> lattice_coordinate = nullptr,
                                             std::vector<FFTWGrid<N>> * advected_out = nullptr) {
                assert_mpi(double(NumPartTot) == std::pow(double(Nlattice), N),
                           "[compute_hybrid_bias_spectra] We need Nlattice^N particles");
                std::vector<double> values;
                int nfields = 0;
                {
                    std::vector<FFTWGrid<N>> fields;
                    compute_lagrangian_fields<N>(delta_linear_fourier,
                                                 Nlattice,
                                                 fields,
                                                 options,
                                                 growth_factor_of_kBox);
                    nfields = int(fields.size());
                    lagrangian_fields_at_particles<N, T>(p, NumPart, fields, values, lattice_coordinate);
                }
                std::vector<FFTWGrid<N>> advected;
                compute_advected_fields<N, T>(p, NumPart, NumPartTot, values, nfields, Nmesh, advected, options);
                cross_spectra_of_grids<N>(advected, spectra, options.nbins);

                // The Poisson shot noise mean(w_a w_b) / Npart (in box units)
                const int nop = nfields + 1;
                std::vector<double> sums(size_t(nop) * nop, 0.0);
                for (size_t i = 0; i < NumPart; i++) {
                    for (int a = 0; a < nop; a++) {
                        const double wa = a == 0 ? 1.0 : values[i * nfields + a - 1];
                        for (int b = a; b < nop; b++) {
                            const double wb = b == 0 ? 1.0 : values[i * nfields + b - 1];
                            sums[a * nop + b] += wa * wb;
                        }
                    }
                }
                for (auto & s : sums)
                    FML::SumOverTasks(&s);
                for (size_t i = 0; i < spectra.pairs.size(); i++) {
                    const int a = spectra.pairs[i][0], b = spectra.pairs[i][1];
                    spectra.shotnoise[i] = sums[a * nop + b] / (double(NumPartTot) * double(NumPartTot));
                    if (options.subtract_shotnoise)
                        for (auto & pk : spectra.pofk[i].pofk)
                            pk -= spectra.shotnoise[i];
                }
                if (advected_out)
                    *advected_out = std::move(advected);
            }

            //=====================================================================================================
            /// @brief The Fourier modes of a grid that go into each bin of a binning (as in bin_up_power_spectrum),
            /// for the bins below kmax_exact: result[i] is a list of (|k|, number of modes) in bin i (|k| in the
            /// units of the binning, i.e. box units k = 2 pi |m|). Bins that extend above kmax_exact are left empty.
            /// Useful for comparing a theory prediction with a measurement in bins that contain few modes: average
            /// the theory over the modes, sum_m P(|k_m|) / sum_m 1, instead of evaluating it at the mean k of the bin
            //=====================================================================================================
            template <int N>
            std::vector<std::vector<std::pair<double, double>>> modes_in_bins(PowerSpectrumBinning<N> binning,
                                                                              double kmax_exact) {
                const int n = binning.n;
                std::vector<std::vector<std::pair<double, double>>> result(n);
                const int M = int(std::ceil(kmax_exact / (2.0 * M_PI)));
                // The bins that are fully below kmax_exact
                std::vector<bool> exact(n, false);
                for (int i = 0; i < n; i++) {
                    const double upper =
                        binning.get_k_from_bin_index(i, binning.kmin, binning.kmax, n, binning.bin_type) +
                        0.5 * (binning.kmax - binning.kmin) / double(n - 1);
                    exact[i] = upper <= kmax_exact and binning.bin_type == PowerSpectrumBinning<N>::LINEAR_SPACING;
                }
                // Count the modes with each |m|^2
                std::map<long long int, double> count;
                std::array<int, N> m;
                const int side = 2 * M + 1;
                long long int ntot = 1;
                for (int idim = 0; idim < N; idim++)
                    ntot *= side;
                for (long long int index = 0; index < ntot; index++) {
                    long long int rest = index, m2 = 0;
                    for (int idim = 0; idim < N; idim++) {
                        m[idim] = int(rest % side) - M;
                        rest /= side;
                        m2 += (long long int)(m[idim]) * m[idim];
                    }
                    if (m2 > 0)
                        count[m2] += 1.0;
                }
                for (auto & c : count) {
                    const double k = 2.0 * M_PI * std::sqrt(double(c.first));
                    if (k > kmax_exact)
                        continue;
                    const int i = binning.get_bin_index(k, binning.kmin, binning.kmax, n, binning.bin_type);
                    if (i >= 0 and i < n and exact[i])
                        result[i].push_back({k, c.second});
                }
                return result;
            }

            //=====================================================================================================
            /// @brief The cross spectra P_ta(k) of a tracer field (e.g. halos) with the advected operators. Together
            /// with the basis spectra this is what one fits for the bias parameters: P_tm = sum_a b_a P_1a and
            /// P_tt = sum_ab b_a b_b P_ab.
            ///
            /// @param[in] tracer_fourier The tracer density contrast in Fourier space (same grid size)
            /// @param[in] advected The advected operators (from compute_advected_fields or compute_hybrid_bias_spectra)
            /// @param[out] result result[a] is the cross spectrum of the tracer with operator a (box units)
            /// @param[in] nbins Number of bins (0: Nmesh / 2)
            //=====================================================================================================
            template <int N>
            void tracer_cross_spectra(const FFTWGrid<N> & tracer_fourier,
                                      const std::vector<FFTWGrid<N>> & advected,
                                      std::vector<PowerSpectrumBinning<N>> & result,
                                      int nbins = 0) {
                const int Nmesh = tracer_fourier.get_nmesh();
                if (nbins <= 0)
                    nbins = Nmesh / 2;
                result.clear();
                for (auto & grid : advected) {
                    assert_mpi(grid.get_nmesh() == Nmesh, "[tracer_cross_spectra] The grids must have the same size");
                    PowerSpectrumBinning<N> binning(2.0 * M_PI,
                                                    2.0 * M_PI * Nmesh / 2,
                                                    nbins,
                                                    PowerSpectrumBinning<N>::LINEAR_SPACING);
                    bin_up_cross_power_spectrum<N>(tracer_fourier, grid, binning);
                    result.push_back(binning);
                }
            }

        } // namespace HYBRIDBIAS
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif
