#ifndef FML_ANALYSIS_POWERSPECTRUM_H
#define FML_ANALYSIS_POWERSPECTRUM_H

#include <FML/Analysis/AnalysisContext.h>
#include <FML/Analysis/Output.h>
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/ComputePowerSpectra/PowerSpectrumBinning.h>
#include <FML/MPIParticles/MPIParticles.h>

#include <cmath>
#include <functional>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace FML {
    namespace ANALYSIS {

        /// An extra output column: its name and its value in bin i
        using Column = std::pair<std::string, std::function<double(int)>>;

        namespace detail {
            inline void validate_spectrum_settings(const std::string & prefix,
                                                   int nmesh,
                                                   int nbins,
                                                   const std::string & method) {
                if (nmesh < 2)
                    throw std::runtime_error(prefix + "nmesh must be >= 2");
                if (nmesh % FML::NTasks != 0)
                    throw std::runtime_error(prefix + "nmesh must be divisible by the number of MPI tasks");
                if (nbins < 0)
                    throw std::runtime_error(prefix + "nbins must be >= 0 (0: nmesh/2 linear bins)");
                if (method != "NGP" and method != "CIC" and method != "TSC" and method != "PCS" and method != "PQS")
                    throw std::runtime_error(prefix + "density_assignment_method must be NGP, CIC, TSC, PCS or PQS");
            }

            /// nbins = 0: nmesh/2 linear bins of width 2pi (box units) from k = 2pi, as SimulationCode. Otherwise
            /// nbins bins from the fundamental mode to the Nyquist frequency, linear or logarithmic
            template <int NDIM>
            FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> make_binning(int nmesh, int nbins, bool log_binning) {
                using Binning = FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>;
                if (nbins == 0)
                    return Binning(nmesh / 2);
                return Binning(2.0 * M_PI,
                               M_PI * nmesh,
                               nbins,
                               log_binning ? Binning::LOG_SPACING : Binning::LINEAR_SPACING);
            }
        } // namespace detail

        //=============================================================================
        /// P(k) of the particles (mass weighted if the particles have masses). Keys pofk_<name>, the same as
        /// SimulationCode's where they exist (nbins and log_binning are extra)
        //=============================================================================
        struct PowerSpectrumSettings {
            static constexpr const char * prefix = "pofk_";
            int nmesh;
            bool interlacing;
            bool subtract_shotnoise;
            std::string density_assignment_method;
            int nbins;
            bool log_binning;

            PowerSpectrumSettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("nmesh", nmesh, 256, "Grid size");
                f("interlacing", interlacing, false, "Use interlaced grids to reduce aliasing");
                f("subtract_shotnoise", subtract_shotnoise, false, "Subtract shot-noise");
                f("density_assignment_method",
                  density_assignment_method,
                  std::string("CIC"),
                  "Density assignment method (NGP, CIC, TSC, PCS, PQS)");
                f("nbins", nbins, 0, "Number of k-bins (0: nmesh/2 linear bins of width 2pi/Box from 2pi/Box)");
                f("log_binning", log_binning, false, "Logarithmic bins (nbins > 0) from 2pi/Box to the Nyquist k");
            }

            void validate() const {
                detail::validate_spectrum_settings(prefix, nmesh, nbins, density_assignment_method);
            }
        };

        //=============================================================================
        /// Redshift-space multipoles P_ell(k) (the mean over the three axes as line of sight). Keys pofk_multipole_<name>
        //=============================================================================
        struct PowerSpectrumMultipoleSettings {
            static constexpr const char * prefix = "pofk_multipole_";
            int nmesh;
            bool interlacing;
            bool subtract_shotnoise;
            int ellmax;
            std::string density_assignment_method;
            int nbins;
            bool log_binning;

            PowerSpectrumMultipoleSettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("nmesh", nmesh, 256, "Grid size");
                f("interlacing", interlacing, false, "Use interlaced grids to reduce aliasing");
                f("subtract_shotnoise", subtract_shotnoise, false, "Subtract shot-noise from P0");
                f("ellmax", ellmax, 4, "Compute P_ell for ell = 0, 2, ..., ellmax");
                f("density_assignment_method",
                  density_assignment_method,
                  std::string("CIC"),
                  "Density assignment method (NGP, CIC, TSC, PCS, PQS)");
                f("nbins", nbins, 0, "Number of k-bins (0: nmesh/2 linear bins of width 2pi/Box from 2pi/Box)");
                f("log_binning", log_binning, false, "Logarithmic bins (nbins > 0) from 2pi/Box to the Nyquist k");
            }

            void validate() const {
                detail::validate_spectrum_settings(prefix, nmesh, nbins, density_assignment_method);
                if (ellmax < 0 or ellmax % 2 != 0)
                    throw std::runtime_error("pofk_multipole_ellmax must be a non-negative even integer");
            }
        };

        //=============================================================================
        /// The bispectrum B(k1, k2, k3) of the particles (mass weighted if they have masses), as SimulationCode's
        /// (bispectrum = true): nbins linear bins in each k from 0 to the Nyquist frequency of the grid. Keys
        /// bispectrum_<name>, SimulationCode's names and defaults (SimulationCode requires nmesh and nbins). Memory and
        /// time: nbins grids of nmesh^3 and nbins^3 integrals
        //=============================================================================
        struct BispectrumSettings {
            static constexpr const char * prefix = "bispectrum_";
            int nmesh;
            int nbins;
            std::string density_assignment_method;
            bool interlacing;
            bool subtract_shotnoise;

            BispectrumSettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("nmesh", nmesh, 128, "Grid size");
                f("nbins",
                  nbins,
                  16,
                  "Number of linear k-bins from 0 to the Nyquist frequency (memory and time scale as nbins grids and "
                  "nbins^3 integrals)");
                f("density_assignment_method",
                  density_assignment_method,
                  std::string("CIC"),
                  "Density assignment method (NGP, CIC, TSC, PCS, PQS)");
                f("interlacing", interlacing, true, "Use interlaced grids to reduce aliasing");
                f("subtract_shotnoise", subtract_shotnoise, false, "Subtract shot-noise");
            }

            void validate() const {
                detail::validate_spectrum_settings(prefix, nmesh, nbins, density_assignment_method);
                if (nbins < 1)
                    throw std::runtime_error("bispectrum_nbins must be >= 1");
            }
        };

        /// P(k) in physical units (k in h/Mpc, P in (Mpc/h)^3)
        template <int NDIM, class P>
        FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> measure_power_spectrum(P * particles,
                                                                                   size_t npart,
                                                                                   size_t npart_total,
                                                                                   const PowerSpectrumSettings & settings,
                                                                                   const AnalysisContext & context) {
            settings.validate();
            auto binning = detail::make_binning<NDIM>(settings.nmesh, settings.nbins, settings.log_binning);
            binning.subtract_shotnoise = settings.subtract_shotnoise;
            FML::CORRELATIONFUNCTIONS::compute_power_spectrum<NDIM, P>(settings.nmesh,
                                                                       particles,
                                                                       npart,
                                                                       npart_total,
                                                                       binning,
                                                                       settings.density_assignment_method,
                                                                       settings.interlacing);
            binning.scale(context.boxsize);
            return binning;
        }

        /// The bispectrum in physical units (k in h/Mpc, B in (Mpc/h)^6), as SimulationCode's. Collective
        template <int NDIM, class P>
        FML::CORRELATIONFUNCTIONS::BispectrumBinning<NDIM> measure_bispectrum(P * particles,
                                                                            size_t npart,
                                                                            size_t npart_total,
                                                                            const BispectrumSettings & settings,
                                                                            const AnalysisContext & context) {
            settings.validate();
            using Binning = FML::CORRELATIONFUNCTIONS::BispectrumBinning<NDIM>;
            Binning bofk(0.0, 2.0 * M_PI * settings.nmesh / 2, settings.nbins, Binning::LINEAR_SPACING);
            bofk.subtract_shotnoise = settings.subtract_shotnoise;
            FML::CORRELATIONFUNCTIONS::compute_bispectrum(settings.nmesh,
                                                          particles,
                                                          npart,
                                                          npart_total,
                                                          bofk,
                                                          settings.density_assignment_method,
                                                          settings.interlacing);
            bofk.scale(context.boxsize);
            return bofk;
        }

        /// Write the bispectrum with SimulationCode's columns: i1 i2 i3, k1 k2 k3 (h/Mpc, the mean k of the bins),
        /// B (Mpc/h)^6, the reduced Q and the number of triangles N, for all nbins^3 bins (collective)
        template <int NDIM>
        void write_bispectrum(const std::string & filename,
                              const FML::CORRELATIONFUNCTIONS::BispectrumBinning<NDIM> & bispectrum,
                              const std::string & header = "") {
            auto bofk = bispectrum; // the accessors are not const
            write_on_root(filename, [&](std::ostream & fp) {
                fp << header;
                fp << "# ShotnoiseSubtracted = " << std::boolalpha << bofk.subtract_shotnoise << "\n";
                fp << "#  i1    i2   i3       k1  (h/Mpc)       k2 (h/Mpc)       k3 (h/Mpc)       B(k1,k2,k3) "
                      "(Mpc/h)^6       Q(k1,k2,k3)       N(k1,k2,k3) \n";
                for (int i = 0; i < bofk.n; i++)
                    for (int j = 0; j < bofk.n; j++)
                        for (int k = 0; k < bofk.n; k++) {
                            fp << std::setw(3) << i << " " << std::setw(3) << j << " " << std::setw(3) << k << " ";
                            fp << std::setw(15) << bofk.kmean[i] << " " << std::setw(15) << bofk.kmean[j] << " "
                               << std::setw(15) << bofk.kmean[k] << " ";
                            fp << std::setw(15) << bofk.get_spectrum(i, j, k) << " ";
                            fp << std::setw(15) << bofk.get_reduced_spectrum(i, j, k) << " ";
                            fp << std::setw(15) << bofk.get_bincount(i, j, k) << "\n";
                        }
            });
        }

        /// P_ell(k) for ell = 0, 1, ..., ellmax (element ell; the odd ones vanish) in physical units. The velocities
        /// are converted to redshift-space displacements with the context (no user-supplied factor)
        template <int NDIM, class P>
        std::vector<FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>>
        measure_power_spectrum_multipoles(FML::PARTICLE::MPIParticles<P> & particles,
                                          const PowerSpectrumMultipoleSettings & settings,
                                          const AnalysisContext & context) {
            settings.validate();
            std::vector<FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>> pell(
                settings.ellmax + 1,
                detail::make_binning<NDIM>(settings.nmesh, settings.nbins, settings.log_binning));
            pell[0].subtract_shotnoise = settings.subtract_shotnoise;
            FML::CORRELATIONFUNCTIONS::compute_power_spectrum_multipoles<NDIM, P>(settings.nmesh,
                                                                                  particles,
                                                                                  context.velocity_to_displacement(),
                                                                                  pell,
                                                                                  settings.density_assignment_method,
                                                                                  settings.interlacing);
            for (auto & p : pell)
                p.scale(context.boxsize);
            return pell;
        }

        /// Write P(k): k (h/Mpc), P (Mpc/h)^3 and extra columns (collective)
        template <int NDIM>
        void write_power_spectrum(const std::string & filename,
                                  const FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> & binning,
                                  const std::string & header = "",
                                  const std::vector<Column> & extra = {}) {
            write_on_root(filename, [&](std::ostream & fp) {
                fp << header;
                fp << "# ShotnoiseSubtracted = " << std::boolalpha << binning.subtract_shotnoise << "\n";
                fp << "#  k (h/Mpc)       P(k) (Mpc/h)^3";
                for (auto & c : extra)
                    fp << "   " << c.first;
                fp << "\n";
                for (int i = 0; i < binning.n; i++) {
                    fp << std::setw(15) << binning.kbin[i] << " " << std::setw(15) << binning.pofk[i];
                    for (auto & c : extra)
                        fp << " " << std::setw(15) << c.second(i);
                    fp << "\n";
                }
            });
        }

        /// Write P_ell(k) for the even ell: k (h/Mpc), P0, P2, ... (Mpc/h)^3 and extra columns (collective)
        template <int NDIM>
        void write_power_spectrum_multipoles(const std::string & filename,
                                             const std::vector<FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>> & pell,
                                             const std::string & header = "",
                                             const std::vector<Column> & extra = {}) {
            if (pell.empty())
                throw std::runtime_error("write_power_spectrum_multipoles: no multipoles");
            write_on_root(filename, [&](std::ostream & fp) {
                fp << header;
                fp << "# ShotnoiseSubtracted = " << std::boolalpha << pell[0].subtract_shotnoise << "\n";
                fp << "#  k (h/Mpc)";
                for (size_t ell = 0; ell < pell.size(); ell += 2)
                    fp << "      P" << ell << "(k) (Mpc/h)^3";
                for (auto & c : extra)
                    fp << "   " << c.first;
                fp << "\n";
                for (int i = 0; i < pell[0].n; i++) {
                    fp << std::setw(15) << pell[0].kbin[i];
                    for (size_t ell = 0; ell < pell.size(); ell += 2)
                        fp << " " << std::setw(15) << pell[ell].pofk[i];
                    for (auto & c : extra)
                        fp << " " << std::setw(15) << c.second(i);
                    fp << "\n";
                }
            });
        }

    } // namespace ANALYSIS
} // namespace FML
#endif
