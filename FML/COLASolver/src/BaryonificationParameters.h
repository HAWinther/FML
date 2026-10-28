#ifndef COLASOLVER_BARYONIFICATION_PARAMETERS_H
#define COLASOLVER_BARYONIFICATION_PARAMETERS_H

#include <FML/Baryonification/Profiles.h>
#include <ostream>
#include <string>

namespace COLASOLVER {
    /// One list of defaults/descriptions shared by reading, schema generation and output provenance.
    /// Masses use Msun/h, lengths use comoving Mpc/h; radius ratios explicitly say _over_r200.
    struct BaryonificationSettings {
        bool enabled = false, output_particles, relax_collisionless;
        std::string model;
        double min_halo_mass_msunh, gas_mass_scale_msunh, stellar_mass_scale_msunh;
        double gas_mass_exponent, gas_core_radius_over_r200, gas_ejection_radius_over_r200;
        double truncation_radius_over_r200, stellar_radius_over_r200, stellar_amplitude;
        double stellar_total_exponent, stellar_central_exponent, relaxation_amplitude, relaxation_exponent;
        double concentration_norm, concentration_pivot_mass_msunh, concentration_mass_slope,
            concentration_redshift_power;
        double fof_linking_length, fof_buffer_mpch, min_radius_over_r200, max_radius_over_r200;
        int fof_min_particles, fof_nmesh_max, profile_radial_bins;

        BaryonificationSettings() {
            parameters(
                [](const char *, auto & value, const auto & default_value, const char *) { value = default_value; });
        }

        template <class F>
        void parameters(F && f) {
            f("model",
              model,
              std::string("schneider2019"),
              "Finite-support Schneider2019 profile family; illustrative, uncalibrated defaults");
            f("output_particles",
              output_particles,
              false,
              "Write a separate corrected snapshot; velocities are unchanged");
            f("min_halo_mass_msunh", min_halo_mass_msunh, 1e12, "Minimum host M200c (Msun/h)");
            f("gas_mass_scale_msunh", gas_mass_scale_msunh, 1e14, "Gas-slope mass scale M_c (Msun/h)");
            f("stellar_mass_scale_msunh", stellar_mass_scale_msunh, 2.5e11, "Stellar-fraction pivot M1 (Msun/h)");
            f("gas_mass_exponent", gas_mass_exponent, 0.3, "Exponent mu in beta=3-(M_c/M200c)^mu");
            f("gas_core_radius_over_r200", gas_core_radius_over_r200, 0.1, "Gas core radius / R200c");
            f("gas_ejection_radius_over_r200", gas_ejection_radius_over_r200, 4.0, "Gas ejection radius / R200c");
            f("truncation_radius_over_r200", truncation_radius_over_r200, 4.0, "NFW truncation radius / R200c");
            f("stellar_radius_over_r200", stellar_radius_over_r200, 0.015, "Central stellar scale R_h / R200c");
            f("stellar_amplitude", stellar_amplitude, 0.09, "Stellar mass fraction at M1");
            f("stellar_total_exponent", stellar_total_exponent, 0.32, "Total stellar fraction power-law exponent");
            f("stellar_central_exponent", stellar_central_exponent, 0.6, "Central stellar fraction power-law exponent");
            f("relax_collisionless", relax_collisionless, true, "Apply collisionless contraction/expansion");
            f("relaxation_amplitude", relaxation_amplitude, 0.3, "Dimensionless collisionless relaxation amplitude");
            f("relaxation_exponent", relaxation_exponent, 2.0, "Dimensionless collisionless relaxation exponent");
            f("concentration_norm", concentration_norm, 5.0, "Illustrative c200c normalization; not a fitted relation");
            f("concentration_pivot_mass_msunh",
              concentration_pivot_mass_msunh,
              1e14,
              "Concentration pivot mass (Msun/h)");
            f("concentration_mass_slope",
              concentration_mass_slope,
              -0.1,
              "c=c0*(M200c/Mpivot)^slope*(1+z)^redshift_power");
            f("concentration_redshift_power",
              concentration_redshift_power,
              -1.0,
              "Power of (1+z) in the concentration relation");
            f("fof_linking_length", fof_linking_length, 0.2, "FoF seed linking length / mean particle separation");
            f("fof_min_particles", fof_min_particles, 20, "Minimum particles in a FoF seed halo");
            f("fof_nmesh_max", fof_nmesh_max, 0, "Maximum FoF search mesh (0: automatic)");
            f("fof_buffer_mpch",
              fof_buffer_mpch,
              3.0,
              "Halo/SO search buffer (comoving Mpc/h); larger than halo radii");
            f("profile_radial_bins", profile_radial_bins, 2048, "Number of positive profile-table radii");
            f("min_radius_over_r200", min_radius_over_r200, 1e-6, "First positive profile radius / R200c");
            f("max_radius_over_r200",
              max_radius_over_r200,
              30.0,
              "Finite profile support / R200c; must be smaller than half the box");
        }

        template <class Reader>
        void read(Reader & r) {
            r.get("baryonification", enabled, false);
            if (enabled)
                parameters([&](const char * name, auto & value, const auto & default_value, const char *) {
                    r.get(std::string("baryonification_") + name, value, default_value);
                });
        }

        FML::BARYONIFICATION::ModelParameters profile_model(double baryon_fraction) const {
            FML::BARYONIFICATION::ModelParameters p;
            p.baryon_fraction = baryon_fraction;
            p.gas_mass_scale = gas_mass_scale_msunh;
            p.stellar_mass_scale = stellar_mass_scale_msunh;
            p.gas_mass_exponent = gas_mass_exponent;
            p.gas_core_radius = gas_core_radius_over_r200;
            p.gas_ejection_radius = gas_ejection_radius_over_r200;
            p.truncation_radius = truncation_radius_over_r200;
            p.stellar_radius = stellar_radius_over_r200;
            p.stellar_amplitude = stellar_amplitude;
            p.stellar_total_exponent = stellar_total_exponent;
            p.stellar_central_exponent = stellar_central_exponent;
            p.relax_collisionless = relax_collisionless;
            p.relaxation_amplitude = relaxation_amplitude;
            p.relaxation_exponent = relaxation_exponent;
            return p;
        }

        void validate() const {
            using namespace FML::BARYONIFICATION::detail;
            require(model == "schneider2019", "baryonification_model must be schneider2019");
            require(positive(min_halo_mass_msunh) && positive(concentration_pivot_mass_msunh) &&
                        positive(concentration_norm),
                    "Baryonification masses (_msunh) and concentration must be finite and positive");
            require(std::isfinite(concentration_mass_slope) && std::isfinite(concentration_redshift_power),
                    "Baryonification concentration exponents must be finite");
            require(positive(fof_linking_length) && fof_min_particles >= 2 && fof_nmesh_max >= 0 &&
                        positive(fof_buffer_mpch),
                    "Baryonification: invalid FoF linking length, particle count, mesh or buffer_mpch");
            require(profile_radial_bins >= 32 && positive(min_radius_over_r200) && min_radius_over_r200 < 1.0 &&
                        positive(max_radius_over_r200) && max_radius_over_r200 > 1.0,
                    "Baryonification: invalid profile table radii or radial-bin count");
            profile_model(0.157).validate(); // cosmological baryon fraction checked separately
        }

        void write(std::ostream & out) const {
            auto copy = *this;
            copy.parameters([&](const char * name, const auto & value, const auto &, const char *) {
                out << "# baryonification_" << name << " = " << value << '\n';
            });
        }
    };
} // namespace COLASOLVER
#endif
