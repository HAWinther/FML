#ifndef FML_HYBRIDBIASREDSHIFTSPACE_HEADER
#define FML_HYBRIDBIASREDSHIFTSPACE_HEADER

#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/ComputePowerSpectra/HybridBiasSpectra.h>

#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <string>
#include <vector>

namespace FML {
    namespace CORRELATIONFUNCTIONS {

        //=====================================================================================================
        ///
        /// Hybrid bias (Lagrangian bias basis) spectra in redshift space: the multipoles P_ab^l(k) of the cross
        /// spectra of the Lagrangian operators 1, delta_L, delta_L^2 - <delta_L^2>, s^2 - <s^2>, nabla^2 delta_L
        /// (see HybridBiasSpectra.h) advected with the particles to their redshift-space positions
        ///
        ///   s = x + (v . n) n velocity_to_displacement     (plane-parallel, n a coordinate axis)
        ///
        /// The operators are those of the Lagrangian (linear) field, exactly as in real space; only the positions the
        /// weights are assigned at change. For bias parameters b_a the tracer multipoles are P_tt^l = sum_ab b_a b_b
        /// P_ab^l (tracer_multipoles). Galaxy-specific small-scale velocities (fingers of god) are not in the basis
        /// spectra (the particles move with the dark matter): they enter the fit as counterterms k^2 mu^(2n) and
        /// stochastic terms. The theory predictions are in HybridBiasZeldovich.h (Zel'dovich, exact, in redshift space:
        /// zeldovich_hybrid_bias_multipoles) and HybridBiasOneLoopRSD.h (one-loop LPT).
        ///
        ///   compute_hybrid_bias_multipoles: from simulation particles (with velocities) and the linear field, the
        ///     mean over the line-of-sight axes (all by default)
        ///   compute_zeldovich_hybrid_bias_multipoles: the same for the Zel'dovich field of the initial conditions in
        ///     redshift space (x = q + Psi + f (Psi . n) n): its mean is the Zel'dovich prediction, so it is the
        ///     control variate of the measured multipoles (as in real space, Kokron et al. 2022)
        ///
        /// The multipoles are computed with compute_cross_power_spectrum_multipoles_fourier (ComputePowerSpectrum.h)
        /// on the window-deconvolved (and interlaced) advected fields, binned as the real-space spectra (linear bins
        /// from the fundamental mode to the Nyquist frequency). Units: box units until scale(boxsize) (k -> k / B,
        /// P -> B^N P and 1/B^2 per nabla^2 delta in the pair).
        ///
        //=====================================================================================================
        namespace HYBRIDBIAS {

            /// The multipoles of the hybrid bias spectra: pofk[pair][ell] for ell = 0, ..., ellmax (odd ell are zero)
            template <int N>
            struct HybridBiasMultipoles {
                std::vector<std::string> names{};
                std::vector<std::array<int, 2>> pairs{};
                int ellmax{4};
                std::vector<std::vector<PowerSpectrumBinning<N>>> pofk{};
                /// The Poisson shot noise mean(w_a w_b) / Npart of the monopole (subtracted from ell = 0 if
                /// options.subtract_shotnoise)
                std::vector<double> shotnoise{};
                /// The line-of-sight axes we averaged over
                std::vector<int> los_axes{};
                bool scaled{false};

                int index(int a, int b) const {
                    if (a > b)
                        std::swap(a, b);
                    for (size_t i = 0; i < pairs.size(); i++)
                        if (pairs[i][0] == a and pairs[i][1] == b)
                            return int(i);
                    throw std::runtime_error("[HybridBiasMultipoles] Pair not found");
                }
                int operator_index(const std::string & name) const {
                    for (size_t i = 0; i < names.size(); i++)
                        if (names[i] == name)
                            return int(i);
                    throw std::runtime_error("[HybridBiasMultipoles] Unknown operator " + name);
                }
                PowerSpectrumBinning<N> & get(int a, int b, int ell) {
                    assert_mpi(ell >= 0 and ell <= ellmax, "[HybridBiasMultipoles::get] ell out of range");
                    return pofk[index(a, b)][ell];
                }
                PowerSpectrumBinning<N> & get(const std::string & a, const std::string & b, int ell) {
                    return get(operator_index(a), operator_index(b), ell);
                }

                /// To physical units: k -> k / B, P -> B^N P, and 1/B^2 for every nabla^2 delta in the pair
                void scale(double boxsize) {
                    assert_mpi(not scaled, "[HybridBiasMultipoles::scale] Already scaled");
                    for (size_t i = 0; i < pairs.size(); i++) {
                        const int nlap = (pairs[i][0] == NABLA2DELTA) + (pairs[i][1] == NABLA2DELTA);
                        const double fac = std::pow(boxsize, -2.0 * nlap);
                        for (auto & p : pofk[i]) {
                            p.scale(boxsize);
                            for (auto & v : p.pofk)
                                v *= fac;
                        }
                        shotnoise[i] *= std::pow(boxsize, N) * fac;
                    }
                    scaled = true;
                }

                /// The tracer multipole P_tt^l(k) = sum_ab b_a b_b P_ab^l for the bias parameters b (b[0] = 1 for
                /// the field "1"; a missing operator has bias 0)
                std::vector<double> tracer_multipole(const std::vector<double> & b, int ell) const {
                    std::vector<double> result(pofk[0][0].n, 0.0);
                    for (size_t i = 0; i < pairs.size(); i++) {
                        const int a = pairs[i][0], c = pairs[i][1];
                        const double ba = a < int(b.size()) ? b[a] : 0.0;
                        const double bc = c < int(b.size()) ? b[c] : 0.0;
                        const double fac = (a == c ? 1.0 : 2.0) * ba * bc;
                        for (int k = 0; k < pofk[i][ell].n; k++)
                            result[k] += fac * pofk[i][ell].pofk[k];
                    }
                    return result;
                }

                /// Write the multipoles (only task 0): k, the number of modes, then for every pair P_ab^0, P_ab^2,
                /// ..., P_ab^ellmax (the even ell)
                void write(const std::string & filename, const std::string & header = "") const {
                    if (FML::ThisTask != 0)
                        return;
                    std::ofstream fp(filename);
                    assert_mpi(fp.good(), ("[HybridBiasMultipoles::write] Could not open " + filename).c_str());
                    if (header != "")
                        fp << header;
                    fp << "# Hybrid bias spectra in redshift space: multipoles P_ab^l(k) of the advected operators "
                       << (scaled ? "(k in h/Mpc, P in (Mpc/h)^" + std::to_string(N) + ", nabla2delta in (Mpc/h)^2)" :
                                    "(box units)")
                       << ", mean over the line-of-sight axes";
                    for (auto a : los_axes)
                        fp << " " << a;
                    fp << "\n# Column 1: k, column 2: number of modes, then for (a,b) = ";
                    for (auto & p : pairs)
                        fp << "(" << names[p[0]] << "," << names[p[1]] << ") ";
                    fp << "the multipoles l =";
                    for (int ell = 0; ell <= ellmax; ell += 2)
                        fp << " " << ell;
                    fp << "\n# Poisson shot noise mean(w_a w_b)/Npart:";
                    for (auto s : shotnoise)
                        fp << " " << s;
                    fp << "\n" << std::setprecision(8);
                    for (int k = 0; k < pofk[0][0].n; k++) {
                        fp << std::setw(15) << pofk[0][0].kbin[k] << " " << std::setw(15) << pofk[0][0].count[k];
                        for (auto & pair : pofk)
                            for (int ell = 0; ell <= ellmax; ell += 2)
                                fp << " " << std::setw(15) << pair[ell].pofk[k];
                        fp << "\n";
                    }
                }
            };

            namespace detail {
                /// A point with the operator weights of its lattice point (assigned in redshift space)
                template <int N>
                struct RSDPoint {
                    double pos[N];
                    double w[max_operators - 1];
                    double * get_pos() { return pos; }
                    constexpr int get_ndim() const { return N; }
                };

                /// The cross multipoles of all pairs a <= b of the grids, added to result with the weight (the mean
                /// over the axes)
                template <int N, template <int> class Grid>
                void add_cross_multipoles(const std::vector<Grid<N>> & grids,
                                          int axis,
                                          int ellmax,
                                          int nbins,
                                          double weight,
                                          HybridBiasMultipoles<N> & result) {
                    const int n = int(grids.size());
                    const int Nmesh = grids[0].get_nmesh();
                    if (nbins <= 0)
                        nbins = Nmesh / 2;
                    std::vector<double> los(N, 0.0);
                    los[axis] = 1.0;
                    int ipair = 0;
                    for (int a = 0; a < n; a++) {
                        for (int b = a; b < n; b++, ipair++) {
                            std::vector<PowerSpectrumBinning<N>> Pell(
                                ellmax + 1,
                                PowerSpectrumBinning<N>(2.0 * M_PI,
                                                        2.0 * M_PI * Nmesh / 2,
                                                        nbins,
                                                        PowerSpectrumBinning<N>::LINEAR_SPACING));
                            compute_cross_power_spectrum_multipoles_fourier<N>(grids[a], grids[b], Pell, los);
                            for (auto & p : Pell)
                                for (int i = 0; i < p.n; i++) {
                                    p.pofk[i] *= weight;
                                    p.count[i] *= weight;
                                    p.kbin[i] *= weight;
                                }
                            if (int(result.pofk.size()) <= ipair) {
                                result.pairs.push_back({a, b});
                                result.pofk.push_back(Pell);
                                result.shotnoise.push_back(0.0);
                            } else {
                                for (int ell = 0; ell <= ellmax; ell++)
                                    result.pofk[ipair][ell] += Pell[ell];
                            }
                        }
                    }
                }

                /// The multipoles of points (positions already in redshift space along axis) with the weights
                template <int N, template <int> class Grid>
                void multipoles_of_points(FML::PARTICLE::MPIParticles<RSDPoint<N>> & part,
                                          int nfields,
                                          int Nmesh,
                                          const HybridBiasOptions & options,
                                          int axis,
                                          int ellmax,
                                          double weight,
                                          HybridBiasMultipoles<N> & result) {
                    const size_t NumPart = part.get_npart(), NumPartTot = part.get_npart_total();
                    const RSDPoint<N> * p = part.get_particles_ptr();
                    std::vector<double> values(NumPart * size_t(nfields));
                    for (size_t i = 0; i < NumPart; i++)
                        for (int f = 0; f < nfields; f++)
                            values[i * nfields + f] = p[i].w[f];
                    std::vector<Grid<N>> advected;
                    compute_advected_fields<N, RSDPoint<N>, Grid>(
                        p, NumPart, NumPartTot, values, nfields, Nmesh, advected, options, part.get_domain());
                    add_cross_multipoles<N, Grid>(advected, axis, ellmax, options.nbins, weight, result);
                }

                template <int N>
                void start_result(HybridBiasMultipoles<N> & result, int nop, int ellmax, const std::vector<int> & axes) {
                    result = HybridBiasMultipoles<N>();
                    result.ellmax = ellmax;
                    result.los_axes = axes;
                    const auto & all = operator_names();
                    result.names = std::vector<std::string>(all.begin(), all.begin() + nop);
                }

                /// The shot noise mean(w_a w_b) / Npart (via set_shotnoise of the real-space spectra), subtracted
                /// from the monopole if subtract
                template <int N>
                void set_multipole_shotnoise(const std::vector<double> & values,
                                             int nfields,
                                             size_t NumPart,
                                             size_t NumPartTot,
                                             bool subtract,
                                             HybridBiasMultipoles<N> & result) {
                    HybridBiasSpectra<N> tmp;
                    tmp.pairs = result.pairs;
                    tmp.shotnoise.assign(result.pairs.size(), 0.0);
                    tmp.pofk.resize(result.pairs.size());
                    for (size_t i = 0; i < result.pairs.size(); i++)
                        tmp.pofk[i] = result.pofk[i][0];
                    set_shotnoise(values, nfields, NumPart, NumPartTot, subtract, tmp);
                    for (size_t i = 0; i < result.pairs.size(); i++) {
                        result.shotnoise[i] = tmp.shotnoise[i];
                        result.pofk[i][0] = tmp.pofk[i];
                    }
                }

                inline std::vector<int> default_axes(int N, std::vector<int> axes) {
                    if (axes.empty())
                        for (int idim = 0; idim < N; idim++)
                            axes.push_back(idim);
                    for (auto a : axes)
                        assert_mpi(a >= 0 and a < N, "[hybrid bias multipoles] line-of-sight axis out of range");
                    return axes;
                }
            } // namespace detail

            //=====================================================================================================
            /// @brief The redshift-space hybrid bias multipoles from simulation particles: the Lagrangian operators of
            /// the linear field at the lattice points of the particles (as compute_hybrid_bias_spectra), assigned at
            /// the redshift-space positions s = x + v_n n velocity_to_displacement for each line-of-sight axis n,
            /// the cross multipoles of all pairs, averaged over the axes.
            ///
            /// @param[in] p Pointer to the particles (positions in [0,1), velocities; started on a lattice)
            /// @param[in] NumPart Number of local particles
            /// @param[in] NumPartTot Total number of particles (Nlattice^N)
            /// @param[in] delta_linear_fourier The linear density field in Fourier space
            /// @param[in] Nlattice The size of the particle lattice
            /// @param[in] Nmesh The grid size for the advected fields
            /// @param[in] velocity_to_displacement Converts a velocity to a shift in box units (as for
            /// compute_power_spectrum_multipoles; for SimulationCode's particles 1 / (a^2 H(a) / H0 ... ), see there)
            /// @param[in] options The options (as for compute_hybrid_bias_spectra; the shot noise is subtracted from
            /// the monopole only)
            /// @param[out] result The multipoles (box units, call result.scale(boxsize))
            /// @param[in] ellmax The largest multipole (even; 4: monopole, quadrupole, hexadecapole)
            /// @param[in] los_axes The line-of-sight axes to average over (empty: all)
            /// @param[in] growth_factor_of_kBox Optional: multiply delta_linear by this (as for the real-space spectra)
            /// @param[in] lattice_coordinate Optional: the lattice point of a particle (default from get_id)
            //=====================================================================================================
            template <int N, class T, template <int> class Grid = FFTWGrid>
            void compute_hybrid_bias_multipoles(const T * p,
                                                size_t NumPart,
                                                size_t NumPartTot,
                                                const Grid<N> & delta_linear_fourier,
                                                int Nlattice,
                                                int Nmesh,
                                                double velocity_to_displacement,
                                                const HybridBiasOptions & options,
                                                HybridBiasMultipoles<N> & result,
                                                int ellmax = 4,
                                                std::vector<int> los_axes = {},
                                                std::function<double(double)> growth_factor_of_kBox = nullptr,
                                                detail::LatticeCoordinateFunction<N, T> lattice_coordinate = nullptr) {
                static_assert(FML::PARTICLE::has_get_vel<T>(),
                              "[compute_hybrid_bias_multipoles] The particles need velocities");
                assert_mpi(double(NumPartTot) == std::pow(double(Nlattice), N),
                           "[compute_hybrid_bias_multipoles] We need Nlattice^N particles");
                assert_mpi(ellmax >= 0 and ellmax % 2 == 0, "[compute_hybrid_bias_multipoles] ellmax must be even");
                los_axes = detail::default_axes(N, los_axes);
                using Point = detail::RSDPoint<N>;

                // The operators at the particles
                std::vector<double> values;
                int nfields = 0;
                {
                    std::vector<Grid<N>> fields;
                    compute_lagrangian_fields<N, Grid>(delta_linear_fourier,
                                                       Nlattice,
                                                       fields,
                                                       options,
                                                       growth_factor_of_kBox);
                    nfields = int(fields.size());
                    lagrangian_fields_at_particles<N, T, Grid>(p, NumPart, fields, values, lattice_coordinate);
                }
                detail::start_result(result, nfields + 1, ellmax, los_axes);

                const auto domain = FML::GRID::particle_domain_of(delta_linear_fourier);
                const double weight = 1.0 / double(los_axes.size());
                for (int axis : los_axes) {
                    // The points in redshift space (with their weights), on the tasks of their positions
                    Vector<Point> points(NumPart);
                    for (size_t i = 0; i < NumPart; i++) {
                        auto * pos = FML::PARTICLE::GetPos(const_cast<T &>(p[i]));
                        auto * vel = FML::PARTICLE::GetVel(const_cast<T &>(p[i]));
                        for (int idim = 0; idim < N; idim++)
                            points[i].pos[idim] = pos[idim];
                        points[i].pos[axis] = FML::PARTICLE::wrap_coordinate_periodic(
                            double(pos[axis]) + double(vel[axis]) * velocity_to_displacement);
                        for (int f = 0; f < nfields; f++)
                            points[i].w[f] = values[i * nfields + f];
                    }
                    FML::PARTICLE::MPIParticles<Point> part(domain);
                    part.move_from(std::move(points));
                    part.communicate_particles();
                    assert_mpi(double(part.get_npart_total()) == double(NumPartTot),
                               "[compute_hybrid_bias_multipoles] Lost particles in the communication");
                    detail::multipoles_of_points<N, Grid>(part, nfields, Nmesh, options, axis, ellmax, weight, result);
                }
                detail::set_multipole_shotnoise(values, nfields, NumPart, NumPartTot, options.subtract_shotnoise, result);
            }

            //=====================================================================================================
            /// @brief The redshift-space hybrid bias multipoles of the Zel'dovich field of the initial conditions:
            /// the lattice points q are moved to s = q + Psi(q) + f (Psi(q) . n) n with the 1LPT displacement of the
            /// linear field (scaled with growth_factor_of_kBox, spherical cut kcut_displacement) and carry the operator
            /// weights of q. Its mean is the redshift-space Zel'dovich prediction (zeldovich_hybrid_bias_multipoles in
            /// HybridBiasZeldovich.h with W_displacement = the cut, W_operators = the cut of options): an exact test
            /// of the estimator and the theory, and the control variate of the measured multipoles. The shot noise is
            /// never subtracted (see compute_zeldovich_hybrid_bias_spectra).
            ///
            /// @param[in] delta_linear_fourier The linear density field in Fourier space
            /// @param[in] Nlattice The size of the lattice
            /// @param[in] Nmesh The grid size for the advected fields
            /// @param[in] growth_rate f = dlnD/dlna at the output (the redshift-space displacement is (1 + f n n) Psi)
            /// @param[in] options The options (as for compute_hybrid_bias_spectra)
            /// @param[out] result The multipoles (box units, call result.scale(boxsize))
            /// @param[in] ellmax The largest multipole (even)
            /// @param[in] los_axes The line-of-sight axes to average over (empty: all)
            /// @param[in] growth_factor_of_kBox Optional: multiply delta_linear by this (operators and displacement)
            /// @param[in] kcut_displacement The spherical cut of the displacement (units of 1/box, 0: none)
            //=====================================================================================================
            template <int N, template <int> class Grid = FFTWGrid>
            void compute_zeldovich_hybrid_bias_multipoles(const Grid<N> & delta_linear_fourier,
                                                          int Nlattice,
                                                          int Nmesh,
                                                          double growth_rate,
                                                          const HybridBiasOptions & options,
                                                          HybridBiasMultipoles<N> & result,
                                                          int ellmax = 4,
                                                          std::vector<int> los_axes = {},
                                                          std::function<double(double)> growth_factor_of_kBox = nullptr,
                                                          double kcut_displacement = 0.0) {
                assert_mpi(ellmax >= 0 and ellmax % 2 == 0,
                           "[compute_zeldovich_hybrid_bias_multipoles] ellmax must be even");
                los_axes = detail::default_axes(N, los_axes);
                using Point = detail::RSDPoint<N>;

                // The operators and the Zel'dovich displacement on the lattice
                std::vector<Grid<N>> fields;
                compute_lagrangian_fields<N, Grid>(delta_linear_fourier, Nlattice, fields, options, growth_factor_of_kBox);
                const int nfields = int(fields.size());
                std::array<Grid<N>, N> psi;
                {
                    HybridBiasOptions doptions;
                    doptions.noperators = 2;
                    doptions.kcut = kcut_displacement;
                    doptions.smoothing_scale = 0.0;
                    std::vector<Grid<N>> delta;
                    compute_lagrangian_fields<N, Grid>(delta_linear_fourier, Nlattice, delta, doptions, growth_factor_of_kBox);
                    delta[0].fftw_r2c();
                    Grid<N> phi;
                    FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier<N>(delta[0], phi);
                    delta.clear();
                    FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<N>(phi, psi);
                }
                detail::start_result(result, nfields + 1, ellmax, los_axes);

                const auto domain = FML::GRID::particle_domain_of(psi[0]);
                const double weight = 1.0 / double(los_axes.size());
                const size_t nlocal = size_t(psi[0].get_local_nrows()) * size_t(std::pow(Nlattice, N - 1));
                std::vector<double> values;
                size_t NumPartLocal = 0;
                for (size_t iaxis = 0; iaxis < los_axes.size(); iaxis++) {
                    const int axis = los_axes[iaxis];
                    Vector<Point> points;
                    points.reserve(size_t(1.5 * double(nlocal)) + 1024);
                    for (FML::IndexIntType irow = 0; irow < psi[0].get_local_nrows(); irow++) {
                        for (auto && real_index : psi[0].get_real_range_of_rows(irow, irow + 1)) {
                            const auto coord = psi[0].get_coord_from_index(real_index);
                            const auto q = psi[0].get_real_position(coord);
                            Point point;
                            for (int idim = 0; idim < N; idim++) {
                                const double d = double(psi[idim].get_real_from_index(real_index));
                                point.pos[idim] = FML::PARTICLE::wrap_coordinate_periodic(
                                    q[idim] + d * (idim == axis ? 1.0 + growth_rate : 1.0));
                            }
                            for (int f = 0; f < nfields; f++)
                                point.w[f] = double(fields[f].get_real(coord));
                            points.push_back(point);
                        }
                    }
                    // The weights of the lattice points (for the shot noise), once
                    if (iaxis == 0) {
                        NumPartLocal = points.size();
                        values.resize(NumPartLocal * nfields);
                        for (size_t i = 0; i < NumPartLocal; i++)
                            for (int f = 0; f < nfields; f++)
                                values[i * nfields + f] = points[i].w[f];
                    }
                    FML::PARTICLE::MPIParticles<Point> part(domain);
                    part.move_from(std::move(points));
                    part.communicate_particles();
                    assert_mpi(double(part.get_npart_total()) == std::pow(double(Nlattice), N),
                               "[compute_zeldovich_hybrid_bias_multipoles] Lost points in the communication");
                    detail::multipoles_of_points<N, Grid>(part, nfields, Nmesh, options, axis, ellmax, weight, result);
                }
                const size_t NumPartTot = size_t(std::llround(std::pow(double(Nlattice), N)));
                detail::set_multipole_shotnoise(values, nfields, NumPartLocal, NumPartTot, false, result);
            }

        } // namespace HYBRIDBIAS
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif
