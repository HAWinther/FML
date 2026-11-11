#ifndef FML_ANALYSIS_ANALYSISCONTEXT_H
#define FML_ANALYSIS_ANALYSISCONTEXT_H

#include <FML/Units/Units.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML {
    namespace ANALYSIS {

        //=============================================================================
        /// The expansion history the analyses need: E(a) = H(a)/H0 and the total matter density parameter today
        /// (Omega_m(a) = Omega_m0 / (a^3 E^2), used for the virial overdensity). Independent of how the particles were
        /// made: from the snapshot header (LCDM with optional radiation), from a table, or from a simulation's own
        /// cosmology.
        //=============================================================================
        struct Background {
            std::function<double(double)> E_of_a;
            double Omega_m0{0.0};

            double E(double a) const { return E_of_a(a); }
            double Omega_m(double a) const { return Omega_m0 / (a * a * a * E(a) * E(a)); }

            /// Matter, radiation, curvature (from closure) and Lambda
            static Background LCDM(double Omega_m0, double Omega_Lambda0, double Omega_r0 = 0.0) {
                if (not(Omega_m0 > 0.0) or not std::isfinite(Omega_Lambda0) or not(Omega_r0 >= 0.0))
                    throw std::runtime_error("Background::LCDM: need Omega_m > 0, finite Omega_Lambda, Omega_r >= 0");
                const double Omega_k0 = 1.0 - Omega_m0 - Omega_Lambda0 - Omega_r0;
                Background b;
                b.Omega_m0 = Omega_m0;
                b.E_of_a = [=](double a) {
                    return std::sqrt(Omega_r0 / (a * a * a * a) + Omega_m0 / (a * a * a) + Omega_k0 / (a * a) +
                                     Omega_Lambda0);
                };
                return b;
            }

            /// E(a) from a table with two columns a and H(a)/H0 ('#' comments), interpolated linearly in log a
            static Background from_table(const std::string & filename, double Omega_m0) {
                std::ifstream in(filename);
                if (not in)
                    throw std::runtime_error("Background::from_table: cannot open " + filename);
                std::vector<double> loga, E;
                std::string line;
                while (std::getline(in, line)) {
                    if (line.empty() or line[0] == '#')
                        continue;
                    std::istringstream row(line);
                    double a, e;
                    if (not(row >> a >> e))
                        continue;
                    if (not(a > 0.0) or not(e > 0.0))
                        throw std::runtime_error("Background::from_table: need a > 0 and H/H0 > 0 in " + filename);
                    if (not loga.empty() and std::log(a) <= loga.back())
                        throw std::runtime_error("Background::from_table: a must increase in " + filename);
                    loga.push_back(std::log(a));
                    E.push_back(e);
                }
                if (loga.size() < 2)
                    throw std::runtime_error("Background::from_table: need at least two rows in " + filename);
                Background b;
                b.Omega_m0 = Omega_m0;
                b.E_of_a = [loga, E](double a) {
                    const double x = std::log(a);
                    if (x < loga.front() or x > loga.back())
                        throw std::runtime_error("Background::from_table: a = " + std::to_string(a) +
                                                 " outside the table");
                    size_t i = size_t(std::upper_bound(loga.begin(), loga.end(), x) - loga.begin());
                    i = std::min(std::max(i, size_t(1)), loga.size() - 1);
                    const double t = (x - loga[i - 1]) / (loga[i] - loga[i - 1]);
                    return E[i - 1] + t * (E[i] - E[i - 1]);
                };
                return b;
            }
        };

        //=============================================================================
        /// What an analysis needs to know about a particle set besides the particles. Particle positions are always
        /// in box units [0,1) (the FML convention); masses and velocities are in whatever units the particles have,
        /// with the factors below converting them. Standalone (a snapshot) and in a simulation the same analysis code
        /// is used with a different context.
        //=============================================================================
        struct AnalysisContext {
            /// Comoving box size (Mpc/h)
            double boxsize{0.0};
            /// Scale factor of the particles
            double scale_factor{1.0};
            /// Expansion history
            Background background;
            /// Particle mass units -> Msun/h
            double mass_to_msunh{1.0};
            /// Particle velocity units -> peculiar velocity in km/s
            double velocity_to_kms{1.0};

            double redshift() const { return 1.0 / scale_factor - 1.0; }

            /// Particle velocity -> redshift-space displacement along the line of sight in box units:
            /// s = v_pec / (a H(a) B) with H in km/s/(Mpc/h) = 100 E(a)
            double velocity_to_displacement() const {
                return velocity_to_kms / (100.0 * scale_factor * background.E(scale_factor) * boxsize);
            }

            /// The critical density today in Msun/h / (Mpc/h)^3
            static double critical_density0_msunh_mpch3() {
                return 3.0 * FML::UTILS::SI::MplMpl_over_H0_msunh / std::pow(FML::UTILS::SI::c_over_H0_mpch, 3);
            }

            /// The density parameter today of a set of particles with total mass total_mass (particle mass units)
            double Omega_of_particles(double total_mass) const {
                return total_mass * mass_to_msunh / (critical_density0_msunh_mpch3() * std::pow(boxsize, 3));
            }

            void validate() const {
                if (not(boxsize > 0.0) or not std::isfinite(boxsize))
                    throw std::runtime_error("AnalysisContext: invalid boxsize");
                if (not(scale_factor > 0.0) or not std::isfinite(scale_factor))
                    throw std::runtime_error("AnalysisContext: invalid scale factor");
                if (not background.E_of_a)
                    throw std::runtime_error("AnalysisContext: no expansion history");
                const double E = background.E(scale_factor);
                if (not(E > 0.0) or not std::isfinite(E))
                    throw std::runtime_error("AnalysisContext: invalid H(a)/H0 at the scale factor of the particles");
                if (not(mass_to_msunh > 0.0) or not(velocity_to_kms > 0.0))
                    throw std::runtime_error("AnalysisContext: mass and velocity unit factors must be positive");
            }
        };

        /// The redshift label used in the output file names (the SimulationCode convention): 3 decimals
        inline std::string redshift_label(double redshift) {
            std::ostringstream s;
            s << std::fixed << std::setprecision(3) << redshift;
            return s.str();
        }

    } // namespace ANALYSIS
} // namespace FML
#endif
