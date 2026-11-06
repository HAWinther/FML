#ifndef FML_AXIONHMCODE_H
#define FML_AXIONHMCODE_H

#include <vector>

namespace FML::COSMOLOGY {

    // Native mixed cold/ultralight-axion halo model (Vogt et al. 2022; Dome et al. 2024).
    // Independent of the Einstein--Boltzmann solver. k: h/Mpc; P: (Mpc/h)^3; M: Msun/h.
    class AxionHMcode {
      public:
        enum class Version { Basic, Dome2024 };
        enum class Transition { Recommended, Sum, Smoothed };
        enum class JeansDensity { ReferenceCode, OmegaMatter };
        struct Cosmology {
            double h = 0.674;
            double Omega_b = 0.049, Omega_cdm = 0.23, Omega_axion = 0.031;
            double mass_eV = 3.162277660168379e-25;
            double z = 2.0;
        };
        struct Input {
            std::vector<double> k, cold, axion;
        };
        struct Options {
            Version version = Version::Dome2024;
            Transition transition = Transition::Recommended; // Dome: smoothed; Basic: sum
            // The pinned reference passes omega_m=Omega_m*h^2 into its axion-mass Jeans relation.
            // Keep that convention explicitly selectable; OmegaMatter uses the density in the published formula.
            JeansDensity jeans_density = JeansDensity::ReferenceCode;
            bool allow_extrapolation = false;
            bool one_halo_damping = true, two_halo_damping = false, halo_bloating = false, full_two_halo = false;
            bool reference_density_cut = true; // reference rejection |central amplitude - initial guess| > 100
            double concentration_amplitude = 4.0;
            double mass_min = 1e7, mass_max = 1e18;
            int n_mass = 257, n_radius = 2000;
        };
        struct Halo {
            double cold_mass{}, axion_mass{}, concentration{}, virial_radius{}, core_radius{};
            // Central amplitude refers to the unscaled soliton/envelope shape; the final density is
            // multiplied by profile_rescaling so its mass is exactly axion_mass.
            double central_density_parameter{}, central_density_guess{}, normalization_error{};
            double raw_profile_mass_error{}, profile_rescaling{};
        };
        struct Spectrum {
            std::vector<double> k, linear_total, total, cold, cold_axion, axion;
            // These are physical component one-halo terms (no clustered-fraction denominators).
            std::vector<double> cold_one_halo, cross_one_halo, axion_one_halo;
            std::vector<Halo> haloes;
            // Cold hosts retained in the cold term but excluded from axion terms by the reference
            // central-density rejection criterion.
            std::vector<double> rejected_axion_hosts;
            double sigma8_cold{}, delta_c{}, Delta_vir{}, cutoff_mass{}, beta2{}, clustered_fraction{};
            double alpha_cold{}, alpha_cross{};
            bool extrapolated{};
        };

        static void validate(const Cosmology &, const Input &, const Options &);
        static Spectrum spectrum(const Cosmology &, const Input &, const std::vector<double> & output_k);
        static Spectrum spectrum(const Cosmology &, const Input &, const std::vector<double> & output_k, const Options &);
        // Diagnostics also usable without a power spectrum.
        static double beta2(double z, double axion_fraction_of_matter, bool allow_extrapolation = false);
        static double jeans_mass(double z, double Omega_m, double h, double mass_eV);
    };

} // namespace FML::COSMOLOGY
#endif
