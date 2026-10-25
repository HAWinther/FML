#ifndef FML_COSMICWEB_HEADER
#define FML_COSMICWEB_HEADER

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>

namespace FML {

    //========================================================================================
    /// Classification of the cosmic web into voids, sheets, filaments and knots from the eigenvalues of a
    /// tensor on a grid:
    ///
    /// T-web (Hahn et al. 2007, Forero-Romero et al. 2009): the tidal tensor T_ij = d_i d_j phi with
    /// laplace(phi) = delta (the density contrast, here smoothed with a Gaussian). In Fourier space
    /// T_ij(k) = k_i k_j / k^2 delta(k), and the trace is delta.
    ///
    /// V-web (Hoffman et al. 2012): the velocity shear Sigma_ij = -(d_i v_j + d_j v_i) / (2 H), e.g. from the DTFE.
    ///
    /// A grid cell is a void, sheet, filament or knot if 0, 1, 2 or 3 of the eigenvalues are above the threshold
    /// lambda_threshold (Forero-Romero et al. 2009 and Hoffman et al. 2012 use a threshold > 0).
    /// TidalTensor/Hessian.h in FML computes Hessians and eigensystems with GSL; this module does not need GSL
    /// and also does the classification.
    ///
    /// Grid positions: cell (ix,iy,iz) is at (ix,iy,iz)/Nmesh. Gradients are with respect to the positions in
    /// units of the box (so delta is dimensionless and k = 2 pi n).
    //========================================================================================

    namespace COSMICWEB {

        /// The classes
        enum WebType : int { VOID = 0, SHEET = 1, FILAMENT = 2, KNOT = 3 };
        inline const char * web_type_name(int type) {
            static const char * names[4] = {"void", "sheet", "filament", "knot"};
            return (type >= 0 and type < 4) ? names[type] : "unknown";
        }

        /// The result of the classification
        struct CosmicWeb {
            using Grid = FML::GRID::FFTWGrid<3>;
            /// The class of every cell (0 = void, 1 = sheet, 2 = filament, 3 = knot), stored as a number
            Grid classification{};
            /// The eigenvalues lambda_1 >= lambda_2 >= lambda_3 (only if asked for)
            std::array<Grid, 3> eigenvalues{};
            /// The fraction of the volume in voids, sheets, filaments and knots
            std::array<double, 4> volume_fraction{};
            /// The fraction of the mass (only from the particle methods, otherwise zero)
            std::array<double, 4> mass_fraction{};
        };

        /// Options for the T-web
        struct TWebOptions {
            /// Gaussian smoothing scale of the density field in units of the box (0: no smoothing)
            double smoothing_scale{0.0};
            /// Threshold for the eigenvalues
            double lambda_threshold{0.0};
            /// Also return the eigenvalues
            bool store_eigenvalues{false};
            /// Density assignment for the particle method (NGP, CIC, TSC, PCS, PQS)
            std::string density_assignment_method{"CIC"};
        };

        /// The eigenvalues e[0] >= e[1] >= e[2] of the symmetric matrix [[a00,a01,a02],[a01,a11,a12],[a02,a12,a22]].
        /// Cyclic Jacobi rotations: accurate to rounding also for (nearly) degenerate eigenvalues
        inline void
        symmetric_eigenvalues(double a00, double a01, double a02, double a11, double a12, double a22, double * e) {
            double a[3][3] = {{a00, a01, a02}, {a01, a11, a12}, {a02, a12, a22}};
            const double scale = std::abs(a00) + std::abs(a11) + std::abs(a22) + std::abs(a01) + std::abs(a02) +
                                 std::abs(a12);
            for (int sweep = 0; sweep < 50; sweep++) {
                const double off = std::abs(a[0][1]) + std::abs(a[0][2]) + std::abs(a[1][2]);
                if (off <= 1e-17 * scale or off == 0.0)
                    break;
                for (int p = 0; p < 2; p++)
                    for (int q = p + 1; q < 3; q++) {
                        if (a[p][q] == 0.0)
                            continue;
                        // Rotation that zeros a[p][q]
                        const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                        const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                        const double c = 1.0 / std::sqrt(t * t + 1.0), sn = t * c;
                        for (int k = 0; k < 3; k++) {
                            const double akp = a[k][p], akq = a[k][q];
                            a[k][p] = c * akp - sn * akq;
                            a[k][q] = sn * akp + c * akq;
                        }
                        for (int k = 0; k < 3; k++) {
                            const double apk = a[p][k], aqk = a[q][k];
                            a[p][k] = c * apk - sn * aqk;
                            a[q][k] = sn * apk + c * aqk;
                        }
                    }
            }
            e[0] = a[0][0];
            e[1] = a[1][1];
            e[2] = a[2][2];
            if (e[1] > e[0])
                std::swap(e[0], e[1]);
            if (e[2] > e[1])
                std::swap(e[1], e[2]);
            if (e[1] > e[0])
                std::swap(e[0], e[1]);
        }

        //========================================================================================
        /// Classify the cells given the 6 components of a symmetric tensor (xx, xy, xz, yy, yz, zz) on a grid.
        /// This is the general method (T-web, V-web, or any other tensor)
        ///
        /// @param[in] tensor The components xx, xy, xz, yy, yz, zz
        /// @param[in] lambda_threshold The threshold for the eigenvalues
        /// @param[out] result The classification, the volume fractions (and the eigenvalues if store_eigenvalues)
        /// @param[in] store_eigenvalues Also return the eigenvalues
        /// @param[in] density Optional: the density (in units of the mean) to get the mass fractions
        ///
        //========================================================================================
        inline void classify_tensor(const std::array<FML::GRID::FFTWGrid<3>, 6> & tensor,
                                    double lambda_threshold,
                                    CosmicWeb & result,
                                    bool store_eigenvalues = false,
                                    const FML::GRID::FFTWGrid<3> * density = nullptr) {
            const int N = tensor[0].get_nmesh();
            result = CosmicWeb();
            result.classification = FML::GRID::FFTWGrid<3>(N);
            if (store_eigenvalues)
                for (auto & g : result.eigenvalues)
                    g = FML::GRID::FFTWGrid<3>(N);
            std::array<double, 4> volume{}, mass{};
            for (int ix = 0; ix < tensor[0].get_local_nx(); ix++)
                for (int iy = 0; iy < N; iy++)
                    for (int iz = 0; iz < N; iz++) {
                        const std::array<int, 3> c{ix, iy, iz};
                        double e[3];
                        symmetric_eigenvalues(tensor[0].get_real(c),
                                              tensor[1].get_real(c),
                                              tensor[2].get_real(c),
                                              tensor[3].get_real(c),
                                              tensor[4].get_real(c),
                                              tensor[5].get_real(c),
                                              e);
                        const int type = int(e[0] > lambda_threshold) + int(e[1] > lambda_threshold) +
                                         int(e[2] > lambda_threshold);
                        result.classification.set_real(c, FML::GRID::FloatType(type));
                        if (store_eigenvalues)
                            for (int k = 0; k < 3; k++)
                                result.eigenvalues[k].set_real(c, FML::GRID::FloatType(e[k]));
                        volume[type] += 1.0;
                        if (density)
                            mass[type] += density->get_real(c);
                    }
            FML::SumArrayOverTasks(volume.data(), 4);
            FML::SumArrayOverTasks(mass.data(), 4);
            const double vtot = volume[0] + volume[1] + volume[2] + volume[3];
            const double mtot = mass[0] + mass[1] + mass[2] + mass[3];
            for (int k = 0; k < 4; k++) {
                result.volume_fraction[k] = volume[k] / vtot;
                result.mass_fraction[k] = mtot > 0.0 ? mass[k] / mtot : 0.0;
            }
        }

        //========================================================================================
        /// The tidal tensor T_ij = k_i k_j / k^2 delta(k) of a (smoothed) density contrast (requires FFTW)
        ///
        /// @param[in] delta The density contrast in real space
        /// @param[in] smoothing_scale Gaussian smoothing exp(-k^2R^2/2) with R in units of the box (0: none)
        /// @param[out] tensor The components xx, xy, xz, yy, yz, zz in real space
        ///
        //========================================================================================
        inline void tidal_tensor(const FML::GRID::FFTWGrid<3> & delta,
                                 double smoothing_scale,
                                 std::array<FML::GRID::FFTWGrid<3>, 6> & tensor) {
#ifdef USE_FFTW
            const int N = delta.get_nmesh();
            FML::GRID::FFTWGrid<3> delta_k(N);
            for (int ix = 0; ix < delta.get_local_nx(); ix++)
                for (int iy = 0; iy < N; iy++)
                    for (int iz = 0; iz < N; iz++)
                        delta_k.set_real({ix, iy, iz}, delta.get_real({ix, iy, iz}));
            delta_k.fftw_r2c();
            const int ii[6] = {0, 0, 0, 1, 1, 2};
            const int jj[6] = {0, 1, 2, 1, 2, 2};
            for (int c = 0; c < 6; c++) {
                tensor[c] = delta_k;
                std::array<double, 3> kvec;
                double k2;
                for (auto && index : tensor[c].get_fourier_range()) {
                    tensor[c].get_fourier_wavevector_and_norm2_by_index(index, kvec, k2);
                    if (k2 == 0.0) {
                        tensor[c].set_fourier_from_index(index, 0.0);
                        continue;
                    }
                    const double filter = std::exp(-0.5 * k2 * smoothing_scale * smoothing_scale);
                    tensor[c].set_fourier_from_index(index,
                                                     tensor[c].get_fourier_from_index(index) *
                                                         FML::GRID::FloatType(kvec[ii[c]] * kvec[jj[c]] / k2 * filter));
                }
                tensor[c].fftw_c2r();
            }
#else
            (void)delta;
            (void)smoothing_scale;
            (void)tensor;
            assert_mpi(false, "[tidal_tensor] Requires FFTW (USE_FFTW)");
#endif
        }

        //========================================================================================
        /// T-web from a density contrast grid
        ///
        /// @param[in] delta The density contrast delta in real space
        /// @param[in] options Smoothing scale, threshold, if we want the eigenvalues
        /// @param[out] result The classification and the volume fractions (mass fractions are computed from delta)
        ///
        //========================================================================================
        inline void TWeb(const FML::GRID::FFTWGrid<3> & delta, const TWebOptions & options, CosmicWeb & result) {
            std::array<FML::GRID::FFTWGrid<3>, 6> tensor;
            tidal_tensor(delta, options.smoothing_scale, tensor);
            // The (unsmoothed) density 1 + delta for the mass fractions
            FML::GRID::FFTWGrid<3> density(delta.get_nmesh());
            for (int ix = 0; ix < delta.get_local_nx(); ix++)
                for (int iy = 0; iy < delta.get_nmesh(); iy++)
                    for (int iz = 0; iz < delta.get_nmesh(); iz++)
                        density.set_real({ix, iy, iz}, 1.0 + delta.get_real({ix, iy, iz}));
            classify_tensor(tensor, options.lambda_threshold, result, options.store_eigenvalues, &density);
            if (FML::ThisTask == 0) {
                std::cout << "[TWeb] lambda_th = " << options.lambda_threshold << " R = " << options.smoothing_scale
                          << ". Volume (mass) fractions:";
                for (int k = 0; k < 4; k++)
                    std::cout << " " << web_type_name(k) << " " << result.volume_fraction[k] << " ("
                              << result.mass_fraction[k] << ")";
                std::cout << "\n";
            }
        }

        //========================================================================================
        /// T-web from particles: assign them to a grid (CIC by default) and classify.
        /// The particles must be in [0,1)^3 and inside the local domain (e.g. MPIParticles)
        ///
        //========================================================================================
        template <class T>
        void TWeb(T * p, size_t NumPart, int Nmesh, const TWebOptions & options, CosmicWeb & result) {
            long long int NumPartTot = (long long int)(NumPart);
            FML::SumOverTasks(&NumPartTot);
            const auto nextra =
                FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(options.density_assignment_method);
            FML::GRID::FFTWGrid<3> delta(Nmesh, nextra.first, nextra.second);
            FML::INTERPOLATION::particles_to_grid(
                p, NumPart, size_t(NumPartTot), delta, options.density_assignment_method);
            TWeb(delta, options, result);
        }

        //========================================================================================
        /// V-web from the velocity gradient (e.g. from the DTFE: shear sigma_ij and divergence theta, then
        /// d_i v_j + d_j v_i = 2 (sigma_ij + delta_ij theta / 3)). The tensor we classify is
        /// Sigma_ij = normalization * (d_i v_j + d_j v_i) / 2, with normalization = -1/H (in the units of the
        /// velocity and positions) as in Hoffman et al. 2012
        ///
        /// @param[in] shear The traceless shear in the order xx, xy, xz, yy, yz, zz
        /// @param[in] divergence The divergence
        /// @param[in] normalization The factor (e.g. -1/H)
        /// @param[in] lambda_threshold The threshold
        /// @param[out] result The classification and volume fractions
        /// @param[in] store_eigenvalues Also return the eigenvalues
        ///
        //========================================================================================
        inline void VWeb(const std::array<FML::GRID::FFTWGrid<3>, 6> & shear,
                         const FML::GRID::FFTWGrid<3> & divergence,
                         double normalization,
                         double lambda_threshold,
                         CosmicWeb & result,
                         bool store_eigenvalues = false) {
            const int N = divergence.get_nmesh();
            std::array<FML::GRID::FFTWGrid<3>, 6> tensor;
            const bool diagonal[6] = {true, false, false, true, false, true};
            for (int c = 0; c < 6; c++) {
                tensor[c] = FML::GRID::FFTWGrid<3>(N);
                for (int ix = 0; ix < divergence.get_local_nx(); ix++)
                    for (int iy = 0; iy < N; iy++)
                        for (int iz = 0; iz < N; iz++) {
                            const std::array<int, 3> co{ix, iy, iz};
                            double v = shear[c].get_real(co);
                            if (diagonal[c])
                                v += divergence.get_real(co) / 3.0;
                            tensor[c].set_real(co, FML::GRID::FloatType(normalization * v));
                        }
            }
            classify_tensor(tensor, lambda_threshold, result, store_eigenvalues);
            if (FML::ThisTask == 0) {
                std::cout << "[VWeb] lambda_th = " << lambda_threshold << ". Volume fractions:";
                for (int k = 0; k < 4; k++)
                    std::cout << " " << web_type_name(k) << " " << result.volume_fraction[k];
                std::cout << "\n";
            }
        }

    } // namespace COSMICWEB
} // namespace FML

#endif
