#ifndef FML_ANALYSIS_COSMICWEB_H
#define FML_ANALYSIS_COSMICWEB_H

#include <FML/Analysis/AnalysisContext.h>
#include <FML/Analysis/DensityGrid.h>
#include <FML/Analysis/Output.h>
#include <FML/CosmicWeb/CosmicWeb.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

//=============================================================================
// Cosmic-web classification with the T-web (Hahn et al. 2007, Forero-Romero et al. 2009; FML/CosmicWeb): the
// eigenvalues lambda_1 >= lambda_2 >= lambda_3 of the tidal tensor d_i d_j phi, nabla^2 phi = delta, of the
// Gaussian-smoothed density contrast; a cell is a void, sheet, filament or knot if 0, 1, 2 or 3 eigenvalues are
// above the threshold lambda_th. The output: the volume and mass fractions of the four types for each threshold
// (the mass fractions weight the cells by the unsmoothed density 1 + delta on the same grid).
//=============================================================================

namespace FML {
    namespace ANALYSIS {

        //=============================================================================
        /// T-web settings. Keys cosmicweb_<name>
        //=============================================================================
        struct CosmicWebSettings {
            static constexpr const char * prefix = "cosmicweb_";
            int nmesh;
            double smoothing_scale_mpch;
            std::vector<double> lambda_thresholds;
            std::string density_assignment_method;

            CosmicWebSettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("nmesh", nmesh, 128, "Density grid size (divisible by the number of MPI tasks)");
                f("smoothing_scale_mpch",
                  smoothing_scale_mpch,
                  2.0,
                  "Gaussian smoothing scale R of the density, exp(-k^2 R^2 / 2) (Mpc/h; 0: none)");
                f("lambda_thresholds",
                  lambda_thresholds,
                  std::vector<double>{0.0, 0.1, 0.2, 0.3, 0.4, 0.5},
                  "Eigenvalue thresholds lambda_th (one row of fractions each)");
                f("density_assignment_method",
                  density_assignment_method,
                  std::string("CIC"),
                  "Density assignment method (NGP, CIC, TSC, PCS, PQS)");
            }

            void validate() const {
                detail::validate_grid_settings(prefix, nmesh, smoothing_scale_mpch, density_assignment_method);
                if (lambda_thresholds.empty())
                    throw std::runtime_error("cosmicweb_lambda_thresholds is empty");
                for (double t : lambda_thresholds)
                    if (not std::isfinite(t))
                        throw std::runtime_error("cosmicweb_lambda_thresholds must be finite");
            }
        };

        /// The volume and mass fractions (void, sheet, filament, knot) for each threshold
        struct CosmicWebFractions {
            std::vector<double> lambda_thresholds;
            std::vector<std::array<double, 4>> volume_fraction, mass_fraction;
            double smoothing_scale_mpch{0.0};
            int nmesh{0};
        };

        /// The T-web fractions of a density contrast grid (real space; not changed). Collective
        inline CosmicWebFractions cosmic_web_from_delta(const FML::GRID::FFTWGrid<3> & delta,
                                                        const CosmicWebSettings & settings,
                                                        double boxsize) {
            settings.validate();
            namespace CW = FML::COSMICWEB;
            const int N = delta.get_nmesh();
            // The tidal tensor once (smoothed in Fourier space), the density 1 + delta for the mass fractions
            std::array<FML::GRID::FFTWGrid<3>, 6> tensor;
            CW::tidal_tensor(delta, settings.smoothing_scale_mpch / boxsize, tensor);
            FML::GRID::FFTWGrid<3> density(N);
#ifdef USE_OMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
            for (int ix = 0; ix < delta.get_local_nx(); ix++)
                for (int iy = 0; iy < N; iy++)
                    for (int iz = 0; iz < N; iz++)
                        density.set_real({ix, iy, iz}, 1.0 + delta.get_real({ix, iy, iz}));

            CosmicWebFractions out;
            out.smoothing_scale_mpch = settings.smoothing_scale_mpch;
            out.nmesh = N;
            for (double th : settings.lambda_thresholds) {
                CW::CosmicWeb result;
                CW::classify_tensor(tensor, th, result, CW::ClassificationStorage::Bytes, false, &density);
                out.lambda_thresholds.push_back(th);
                out.volume_fraction.push_back(result.volume_fraction);
                out.mass_fraction.push_back(result.mass_fraction);
            }
            return out;
        }

        /// The T-web fractions of the particles. Collective
        template <class P>
        CosmicWebFractions measure_cosmic_web(P * part,
                                              size_t npart,
                                              const CosmicWebSettings & settings,
                                              const AnalysisContext & context) {
            settings.validate();
            const auto delta = density_contrast_grid(part, npart, settings.nmesh, settings.density_assignment_method);
            return cosmic_web_from_delta(delta, settings, context.boxsize);
        }

        /// Write the fractions: lambda_th, the volume and the mass fractions of voids, sheets, filaments, knots
        inline void write_cosmic_web(const std::string & filename,
                                     const CosmicWebFractions & web,
                                     const std::string & header = "") {
            write_on_root(filename, [&](std::ostream & fp) {
                fp << header;
                fp << "# T-web: nmesh = " << web.nmesh << ", Gaussian smoothing R = " << web.smoothing_scale_mpch
                   << " Mpc/h\n";
                fp << "#  lambda_th     V_void     V_sheet    V_filament   V_knot     M_void     M_sheet    M_filament   "
                      "M_knot   (V: volume fractions, M: mass fractions)\n";
                for (size_t i = 0; i < web.lambda_thresholds.size(); i++) {
                    fp << std::setw(12) << web.lambda_thresholds[i];
                    for (double v : web.volume_fraction[i])
                        fp << " " << std::setw(12) << v;
                    for (double m : web.mass_fraction[i])
                        fp << " " << std::setw(12) << m;
                    fp << "\n";
                }
            });
        }

    } // namespace ANALYSIS
} // namespace FML
#endif
