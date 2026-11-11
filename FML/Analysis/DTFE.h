#ifndef FML_ANALYSIS_DTFE_H
#define FML_ANALYSIS_DTFE_H

#include <FML/Analysis/AnalysisContext.h>
#include <FML/Analysis/Output.h>
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/ComputePowerSpectra/PowerSpectrumBinning.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#ifdef USE_CGAL
#include <FML/Triangulation/DTFE.h>
#endif

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

//=============================================================================
// DTFE (Schaap & van de Weygaert 2000; Cautun & van de Weygaert 2011; needs CGAL) density and velocity fields of a
// particle set on a grid and their power spectra: the density contrast, the peculiar velocity components (km/s),
// the divergence and the vorticity (km/s per Mpc/h). Files: dtfe_spectra_z<z>.txt and, optionally, the fields as
// one text file per task, dtfe_fields_z<z>.rank<NNNNN>.txt.
//=============================================================================

namespace FML {
    namespace ANALYSIS {

        /// Keys dtfe_<name>
        struct DTFESettings {
            static constexpr const char * prefix = "dtfe_";
            int nmesh;
            double buffer_fraction;
            double random_fraction;
            std::string sampling;
            std::string weighting;
            int nsub;
            int nbins;
            bool write_fields;

            DTFESettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("nmesh", nmesh, 128, "Grid size; divisible by the number of MPI tasks");
                f("buffer_fraction", buffer_fraction, 0.5, "Buffer copied from the neighbor tasks (fraction of a slab)");
                f("random_fraction", random_fraction, 0.3, "Guard points (fraction)");
                f("sampling", sampling, std::string("cell_average"), "cell_average or grid_point");
                f("weighting", weighting, std::string("volume"), "Velocity cell average: volume or mass weighted");
                f("nsub", nsub, 2, "Samples per dimension per cell for cell_average");
                f("nbins", nbins, 40, "Logarithmic k-bins of the spectra, from 2pi/B to the Nyquist k");
                f("write_fields", write_fields, false, "Also write the fields (one text file per task)");
            }

            void validate() const {
                if (nmesh < 3 or nmesh % FML::NTasks != 0 or nsub < 1 or nbins < 1 or
                    not(buffer_fraction >= 0.0) or not(random_fraction >= 0.0))
                    throw std::runtime_error("Invalid DTFE settings: dtfe_nmesh must divide across the tasks, "
                                             "dtfe_nsub and dtfe_nbins >= 1 and the fractions >= 0");
                if (sampling != "grid_point" and sampling != "cell_average")
                    throw std::runtime_error("dtfe_sampling must be grid_point or cell_average");
                if (weighting != "volume" and weighting != "mass")
                    throw std::runtime_error("dtfe_weighting must be volume or mass");
            }
        };

#ifdef USE_CGAL
        /// The DTFE fields on a grid: density in units of the mean, velocity in peculiar km/s, divergence and
        /// vorticity in km/s per Mpc/h
        struct DTFEFields {
            FML::GRID::FFTWGrid<3> density;
            FML::TRIANGULATION::DTFEVelocityGrids velocity;
        };

        template <class P>
        DTFEFields compute_dtfe_fields(P * part, size_t npart, const DTFESettings & settings, const AnalysisContext & context) {
            using namespace FML::TRIANGULATION;
            settings.validate();
            context.validate();
            const auto sampling = settings.sampling == "grid_point" ? DTFESampling::GridPoint : DTFESampling::CellAverage;
            const auto weighting = settings.weighting == "mass" ? DTFEWeighting::Mass : DTFEWeighting::Volume;
            DTFE<P> dtfe;
            dtfe.create(part, npart, settings.buffer_fraction, settings.random_fraction);
            DTFEFields f;
            dtfe.density_to_grid(settings.nmesh, f.density, sampling, settings.nsub);
            dtfe.velocity_to_grid(part, settings.nmesh, f.velocity, sampling, weighting, settings.nsub, true, true, true,
                                  false);
            // Particle velocity units -> km/s, and the gradients (with respect to box units) per Mpc/h
            const double vnorm = context.velocity_to_kms;
            const double gnorm = context.velocity_to_kms / context.boxsize;
            const int n = settings.nmesh;
            for (int ix = 0; ix < f.density.get_local_nx(); ix++)
                for (int iy = 0; iy < n; iy++)
                    for (int iz = 0; iz < n; iz++) {
                        const std::array<int, 3> c{ix, iy, iz};
                        for (auto & g : f.velocity.velocity)
                            g.set_real(c, g.get_real(c) * vnorm);
                        f.velocity.divergence.set_real(c, f.velocity.divergence.get_real(c) * gnorm);
                        for (auto & g : f.velocity.vorticity)
                            g.set_real(c, g.get_real(c) * gnorm);
                    }
            return f;
        }

        /// The power spectrum of a field (a copy; subtract_mean for the density in mean units)
        inline FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3>
        dtfe_field_spectrum(const FML::GRID::FFTWGrid<3> & field, bool subtract_mean, int nbins, double boxsize) {
            auto grid = field;
            const int nmesh = grid.get_nmesh();
            if (subtract_mean)
                for (int ix = 0; ix < grid.get_local_nx(); ix++)
                    for (int iy = 0; iy < nmesh; iy++)
                        for (int iz = 0; iz < nmesh; iz++) {
                            const std::array<int, 3> c{ix, iy, iz};
                            grid.set_real(c, grid.get_real(c) - 1.0);
                        }
            grid.fftw_r2c();
            FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> pofk(
                2.0 * M_PI, M_PI * nmesh, nbins, FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3>::LOG_SPACING);
            pofk.subtract_shotnoise = false;
            FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum<3>(grid, pofk);
            pofk.scale(boxsize);
            return pofk;
        }

        /// The spectra of all the fields: (name, binning). Collective
        inline std::vector<std::pair<std::string, FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3>>>
        dtfe_spectra(const DTFEFields & f, const DTFESettings & settings, const AnalysisContext & context) {
            std::vector<std::pair<std::string, FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3>>> spectra;
            const int nb = settings.nbins;
            const double B = context.boxsize;
            spectra.emplace_back("density_delta", dtfe_field_spectrum(f.density, true, nb, B));
            for (int i = 0; i < 3; i++)
                spectra.emplace_back("velocity_" + std::string(1, "xyz"[i]),
                                     dtfe_field_spectrum(f.velocity.velocity[i], false, nb, B));
            spectra.emplace_back("velocity_divergence", dtfe_field_spectrum(f.velocity.divergence, false, nb, B));
            for (int i = 0; i < 3; i++)
                spectra.emplace_back("vorticity_" + std::string(1, "xyz"[i]),
                                     dtfe_field_spectrum(f.velocity.vorticity[i], false, nb, B));
            return spectra;
        }

        /// The spectra file (task 0). Collective
        inline void write_dtfe_spectra(
            const std::string & filename,
            const std::vector<std::pair<std::string, FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3>>> & spectra,
            const AnalysisContext & context) {
            write_on_root(filename, [&](std::ostream & out) {
                out << "# DTFE spectra at z = " << context.redshift()
                    << "; density in mean-density units, velocity in peculiar km/s, divergence and vorticity in km/s "
                       "per Mpc/h\n"
                    << "# k (h/Mpc)  P  modes\n"
                    << std::setprecision(12);
                for (auto & [name, pofk] : spectra) {
                    out << "\n# " << name << "\n";
                    for (int i = 0; i < pofk.n; i++)
                        out << pofk.kbin[i] << " " << pofk.pofk[i] << " " << pofk.count[i] << "\n";
                }
            });
        }

        /// The fields as one text file per task: <prefix>.rank<NNNNN>.txt (the local slab). Collective
        inline void write_dtfe_fields(const std::string & prefix, const DTFEFields & f) {
            std::ostringstream rank;
            rank << std::setw(5) << std::setfill('0') << FML::ThisTask;
            const std::string filename = prefix + ".rank" + rank.str() + ".txt";
            std::string error;
            {
                std::ofstream field(filename);
                if (field.good()) {
                    const auto & density = f.density;
                    const int nmesh = density.get_nmesh();
                    field << "# local DTFE field slab; x0 " << density.get_local_x_start() << " nx "
                          << density.get_local_nx() << " Nmesh " << nmesh << "\n"
                          << "# ix iy iz density vx vy vz divergence wx wy wz\n"
                          << std::setprecision(12);
                    for (int ix = 0; ix < density.get_local_nx(); ix++)
                        for (int iy = 0; iy < nmesh; iy++)
                            for (int iz = 0; iz < nmesh; iz++) {
                                const std::array<int, 3> c{ix, iy, iz};
                                field << density.get_local_x_start() + ix << " " << iy << " " << iz << " "
                                      << density.get_real(c);
                                for (const auto & component : f.velocity.velocity)
                                    field << " " << component.get_real(c);
                                field << " " << f.velocity.divergence.get_real(c);
                                for (const auto & component : f.velocity.vorticity)
                                    field << " " << component.get_real(c);
                                field << "\n";
                            }
                    field.flush();
                }
                if (not field.good())
                    error = "Could not write the DTFE field file " + filename;
            }
            throw_if_any_task_failed(error, "Writing the DTFE field files");
        }
#endif

    } // namespace ANALYSIS
} // namespace FML
#endif
