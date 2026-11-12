#ifndef FML_ANALYSIS_MINKOWSKI_H
#define FML_ANALYSIS_MINKOWSKI_H

#include <FML/Analysis/AnalysisContext.h>
#include <FML/Analysis/DensityGrid.h>
#include <FML/Analysis/Output.h>
#include <FML/Morphology/VoxelMorphology.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

//=============================================================================
// Minkowski functionals of the excursion sets of the Gaussian-smoothed density contrast (FML/Morphology: exact
// counts of the voxel cell complex). For each threshold delta_th = nu sigma_0 (or given in delta) the densities
//     V0 = volume fraction, V1 = A / (6 V), V2 = (integrated mean curvature) / (3 pi V), V3 = Euler characteristic / V
// with the isotropic lattice (Crofton) estimator of Morphology, in (h/Mpc)^k, and next to them the Gaussian random
// field predictions (Tomita 1986; Schmalzing & Buchert 1997) for the sigma_0 = <delta^2>^(1/2) and
// sigma_1 = <|grad delta|^2>^(1/2) of the same smoothed field:
//     V0 = erfc(nu / sqrt 2) / 2,                      V1 = (2/3) lambda / sqrt(2 pi) exp(-nu^2/2),
//     V2 = (2/3) lambda^2 / sqrt(2 pi) nu exp(-nu^2/2), V3 = lambda^3 / sqrt(2 pi) (nu^2 - 1) exp(-nu^2/2),
// lambda = sigma_1 / sqrt(6 pi sigma_0^2). Departures from them measure non-Gaussianity. The excursion sets are
// cells with delta >= delta_th in the periodic box. The density is not deconvolved by the assignment window (use a
// smoothing scale of several cells).
//=============================================================================

namespace FML {
    namespace ANALYSIS {

        //=============================================================================
        /// Minkowski functional settings. Keys minkowski_<name>
        //=============================================================================
        struct MinkowskiSettings {
            static constexpr const char * prefix = "minkowski_";
            int nmesh;
            double smoothing_scale_mpch;
            std::string density_assignment_method;
            std::string threshold_variable;
            double threshold_min;
            double threshold_max;
            int nthresholds;

            MinkowskiSettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("nmesh", nmesh, 128, "Density grid size (divisible by the number of MPI tasks)");
                f("smoothing_scale_mpch",
                  smoothing_scale_mpch,
                  5.0,
                  "Gaussian smoothing scale R of the density, exp(-k^2 R^2 / 2) (Mpc/h); several cells");
                f("density_assignment_method",
                  density_assignment_method,
                  std::string("CIC"),
                  "Density assignment method (NGP, CIC, TSC, PCS, PQS)");
                f("threshold_variable",
                  threshold_variable,
                  std::string("nu"),
                  "nu: thresholds in units of sigma_0 of the smoothed field (delta_th = nu sigma_0); delta: in delta");
                f("threshold_min", threshold_min, -3.0, "Smallest threshold");
                f("threshold_max", threshold_max, 3.0, "Largest threshold");
                f("nthresholds", nthresholds, 25, "Number of thresholds (linear from min to max)");
            }

            void validate() const {
                detail::validate_grid_settings(prefix, nmesh, smoothing_scale_mpch, density_assignment_method);
                if (threshold_variable != "nu" and threshold_variable != "delta")
                    throw std::runtime_error("minkowski_threshold_variable must be nu or delta");
                if (not std::isfinite(threshold_min) or not std::isfinite(threshold_max) or nthresholds < 1 or
                    (nthresholds > 1 and not(threshold_max > threshold_min)))
                    throw std::runtime_error("Invalid minkowski_threshold_min, minkowski_threshold_max or "
                                             "minkowski_nthresholds");
            }

            std::vector<double> thresholds() const {
                std::vector<double> t(static_cast<size_t>(nthresholds));
                for (int i = 0; i < nthresholds; i++)
                    t[size_t(i)] = nthresholds == 1 ? threshold_min
                                                    : threshold_min + (threshold_max - threshold_min) * i /
                                                                          double(nthresholds - 1);
                return t;
            }
        };

        /// The Minkowski functionals for each threshold, in physical units (lengths in Mpc/h)
        struct MinkowskiFunctionals {
            double sigma0{0.0}; // of the smoothed density contrast
            double sigma1{0.0}; // <|grad delta|^2>^(1/2) (h/Mpc)
            double mean{0.0};   // of the smoothed density contrast (~0)
            double smoothing_scale_mpch{0.0};
            double boxsize{0.0};
            int nmesh{0};
            std::vector<double> nu, delta_threshold;
            std::vector<std::array<double, 4>> measured;       // Crofton densities V0..V3 ((h/Mpc)^k)
            std::vector<std::array<double, 4>> measured_cubic;  // exact voxel-surface densities (see Morphology)
            std::vector<std::array<double, 4>> gaussian;       // Gaussian random field predictions
            std::vector<long long int> euler_characteristic;   // of the excursion set in the box (integer)
        };

        /// The Gaussian random field predictions V0..V3 at nu for lambda = sigma_1 / sqrt(6 pi sigma_0^2)
        inline std::array<double, 4> minkowski_gaussian_prediction(double nu, double lambda) {
            const double g = std::exp(-0.5 * nu * nu) / std::sqrt(2.0 * M_PI);
            return {0.5 * std::erfc(nu / std::sqrt(2.0)),
                    2.0 / 3.0 * lambda * g,
                    2.0 / 3.0 * lambda * lambda * nu * g,
                    lambda * lambda * lambda * (nu * nu - 1.0) * g};
        }

        /// The Minkowski functionals of a density contrast grid (real space; smoothed here, not changed). Collective
        inline MinkowskiFunctionals minkowski_from_delta(const FML::GRID::FFTWGrid<3> & delta,
                                                         const MinkowskiSettings & settings,
                                                         double boxsize) {
            settings.validate();
            const int N = delta.get_nmesh();
            const double R = settings.smoothing_scale_mpch / boxsize;

            // The smoothed field and its Fourier modes (for the gradient)
            FML::GRID::FFTWGrid<3> field(N);
            for (int ix = 0; ix < delta.get_local_nx(); ix++)
                for (int iy = 0; iy < N; iy++)
                    for (int iz = 0; iz < N; iz++)
                        field.set_real({ix, iy, iz}, delta.get_real({ix, iy, iz}));
            field.fftw_r2c();
            if (R > 0.0)
                FML::GRID::multiply_fourier_by_function_of_k<3>(field,
                                                                [R](double k) { return std::exp(-0.5 * k * k * R * R); });

            // sigma_1^2 = <|grad delta|^2> from the spectral gradient (i k_d delta_k, k in h/Mpc), one axis at a time
            double grad2 = 0.0;
            const double knyquist = M_PI * N * (1.0 - 1e-10);
            for (int d = 0; d < 3; d++) {
                auto g = field;
                FML::GRID::for_each_fourier_mode<3>(g, [&](size_t index, double) {
                    std::array<double, 3> kvec;
                    double kmag;
                    g.get_fourier_wavevector_and_norm_by_index(index, kvec, kmag);
                    // The Nyquist mode is its own mirror image: its derivative is ambiguous, set to zero
                    const double kd = std::abs(kvec[d]) >= knyquist ? 0.0 : kvec[d] / boxsize;
                    g.set_fourier_from_index(index, g.get_fourier_from_index(index) * FML::GRID::ComplexType(0, kd));
                });
                g.fftw_c2r();
                double s = 0.0;
#ifdef USE_OMP
#pragma omp parallel for collapse(2) reduction(+ : s)
#endif
                for (int ix = 0; ix < g.get_local_nx(); ix++)
                    for (int iy = 0; iy < N; iy++)
                        for (int iz = 0; iz < N; iz++) {
                            const double v = g.get_real({ix, iy, iz});
                            s += v * v;
                        }
                grad2 += s;
            }
            field.fftw_c2r();

            double sum = 0.0, sum2 = 0.0;
#ifdef USE_OMP
#pragma omp parallel for collapse(2) reduction(+ : sum, sum2)
#endif
            for (int ix = 0; ix < field.get_local_nx(); ix++)
                for (int iy = 0; iy < N; iy++)
                    for (int iz = 0; iz < N; iz++) {
                        const double v = field.get_real({ix, iy, iz});
                        sum += v;
                        sum2 += v * v;
                    }
            FML::SumOverTasks(&sum);
            FML::SumOverTasks(&sum2);
            FML::SumOverTasks(&grad2);
            const double ncells = double(N) * double(N) * double(N);

            MinkowskiFunctionals out;
            out.mean = sum / ncells;
            out.sigma0 = std::sqrt(std::max(0.0, sum2 / ncells - out.mean * out.mean));
            out.sigma1 = std::sqrt(grad2 / ncells);
            out.smoothing_scale_mpch = settings.smoothing_scale_mpch;
            out.boxsize = boxsize;
            out.nmesh = N;
            if (settings.threshold_variable == "nu" and not(out.sigma0 > 0.0))
                throw std::runtime_error("Minkowski functionals: sigma_0 = 0, cannot use thresholds in nu");

            const bool in_nu = settings.threshold_variable == "nu";
            for (double t : settings.thresholds()) {
                out.delta_threshold.push_back(in_nu ? t * out.sigma0 : t);
                out.nu.push_back(in_nu ? t : (out.sigma0 > 0.0 ? t / out.sigma0 : 0.0));
            }
            const auto ex = FML::MORPHOLOGY::excursions(field,
                                                        out.delta_threshold,
                                                        boxsize,
                                                        FML::MORPHOLOGY::Boundary::Periodic,
                                                        FML::MORPHOLOGY::ThresholdRule::AtLeast);
            const double lambda = out.sigma0 > 0.0 ? out.sigma1 / std::sqrt(6.0 * M_PI * out.sigma0 * out.sigma0) : 0.0;
            for (size_t i = 0; i < ex.size(); i++) {
                out.measured.push_back(ex[i].measures.crofton_density);
                out.measured_cubic.push_back(ex[i].measures.minkowski_density);
                out.euler_characteristic.push_back((long long int)ex[i].measures.euler_characteristic);
                out.gaussian.push_back(minkowski_gaussian_prediction(out.nu[i], lambda));
            }
            return out;
        }

        /// The Minkowski functionals of the particles. Collective
        template <class P>
        MinkowskiFunctionals measure_minkowski_functionals(P * part,
                                                           size_t npart,
                                                           const MinkowskiSettings & settings,
                                                           const AnalysisContext & context) {
            settings.validate();
            const auto delta = density_contrast_grid(part, npart, settings.nmesh, settings.density_assignment_method);
            return minkowski_from_delta(delta, settings, context.boxsize);
        }

        /// Write the Minkowski functionals: nu, delta_th, V0..V3 (Crofton), V0..V3 (Gaussian prediction), the Euler
        /// characteristic of the excursion set, and the exact voxel-surface V1, V2
        inline void write_minkowski_functionals(const std::string & filename,
                                                const MinkowskiFunctionals & mf,
                                                const std::string & header = "") {
            write_on_root(filename, [&](std::ostream & fp) {
                fp << header;
                fp << "# Minkowski functionals of the excursion sets delta >= delta_th of the density contrast smoothed "
                      "with a Gaussian of R = "
                   << mf.smoothing_scale_mpch << " Mpc/h on a " << mf.nmesh << "^3 grid (box " << mf.boxsize
                   << " Mpc/h)\n";
                fp << "# sigma_0 = " << mf.sigma0 << "  sigma_1 = " << mf.sigma1
                   << " h/Mpc  lambda = sigma_1 / sqrt(6 pi sigma_0^2) = "
                   << (mf.sigma0 > 0.0 ? mf.sigma1 / std::sqrt(6.0 * M_PI * mf.sigma0 * mf.sigma0) : 0.0)
                   << " h/Mpc  mean = " << mf.mean << "\n";
                fp << "# V_k in (h/Mpc)^k: V0 volume fraction, V1 = A/(6V), V2 = int H dA/(3 pi V), V3 = chi/V "
                      "(isotropic Crofton estimator); G: the Gaussian random field prediction; chi: the Euler "
                      "characteristic of the excursion set; V1_cubic, V2_cubic: of the exact voxel surface\n";
                fp << "#        nu       delta_th             V0             V1             V2             V3"
                      "           G_V0           G_V1           G_V2           G_V3       chi       V1_cubic"
                      "       V2_cubic\n";
                for (size_t i = 0; i < mf.nu.size(); i++) {
                    fp << std::setw(11) << mf.nu[i] << " " << std::setw(14) << mf.delta_threshold[i];
                    for (double v : mf.measured[i])
                        fp << " " << std::setw(14) << v;
                    for (double v : mf.gaussian[i])
                        fp << " " << std::setw(14) << v;
                    fp << " " << std::setw(9) << mf.euler_characteristic[i] << " " << std::setw(14)
                       << mf.measured_cubic[i][1] << " " << std::setw(14) << mf.measured_cubic[i][2] << "\n";
                }
            });
        }

    } // namespace ANALYSIS
} // namespace FML
#endif
