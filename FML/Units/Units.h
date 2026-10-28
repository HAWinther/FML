#ifndef UNITS_HEADER
#define UNITS_HEADER

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef USE_MPI
#include <mpi.h>
namespace FML {
    // The MPI communicator used by the library (defined in Global.cpp)
    extern MPI_Comm communicator;
} // namespace FML
#endif

namespace FML {
    namespace UTILS {

        // Shared SI constants. Compile-time values are independent of ConstantsAndUnits instances.
        // The cosmology solver keeps its Constants namespace as an alias to this namespace.
        // Retain C++14 support for this standalone header; C++17 gives constants one address across TUs.
#if (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L
#define FML_UNITS_INLINE inline
#else
#define FML_UNITS_INLINE
#endif
        namespace SI {
            FML_UNITS_INLINE constexpr double m = 1.0;
            FML_UNITS_INLINE constexpr double s = 1.0;
            FML_UNITS_INLINE constexpr double kg = 1.0;
            FML_UNITS_INLINE constexpr double K = 1.0;
            FML_UNITS_INLINE constexpr double Co = 1.0;
            FML_UNITS_INLINE constexpr double mm = 1e-3 * m;
            FML_UNITS_INLINE constexpr double cm = 1e-2 * m;
            FML_UNITS_INLINE constexpr double km = 1e3 * m;
            FML_UNITS_INLINE constexpr double N = kg * m / (s * s);
            FML_UNITS_INLINE constexpr double J = 1.0;
            FML_UNITS_INLINE constexpr double W = J / s;
            FML_UNITS_INLINE constexpr double eV = 1.602176634e-19 * J;
            FML_UNITS_INLINE constexpr double MeV = 1e6 * eV;
            FML_UNITS_INLINE constexpr double pc = 3.08567758149e16 * m;
            FML_UNITS_INLINE constexpr double kpc = 1e3 * pc;
            FML_UNITS_INLINE constexpr double Mpc = 1e6 * pc;
            FML_UNITS_INLINE constexpr double Gpc = 1e3 * Mpc;
            FML_UNITS_INLINE constexpr double Msun = 1.98847e30 * kg;
            FML_UNITS_INLINE constexpr double yr = 365.25 * 24.0 * 3600.0 * s; // Julian year
            FML_UNITS_INLINE constexpr double Gyr = 1e9 * yr;
            FML_UNITS_INLINE constexpr double velocity = m / s;
            FML_UNITS_INLINE constexpr double density = kg / (m * m * m);

            // Fundamental and atomic constants. Values preserve the cosmology solver's SI table.
            FML_UNITS_INLINE constexpr double c = 2.99792458e8;
            FML_UNITS_INLINE constexpr double G = 6.67430e-11;
            FML_UNITS_INLINE constexpr double k_b = 1.380649e-23;
            FML_UNITS_INLINE constexpr double k_e = 8.9875517923e9;
            FML_UNITS_INLINE constexpr double hbar = 1.054571817e-34;
            FML_UNITS_INLINE constexpr double m_e = 9.1093837015e-31;
            FML_UNITS_INLINE constexpr double m_H = 1.6735575e-27;
            FML_UNITS_INLINE constexpr double sigma_T = 6.6524587321e-29;
            FML_UNITS_INLINE constexpr double alpha = 7.2973525693e-3;
            FML_UNITS_INLINE constexpr double lambda_2s1s = 8.227;
            FML_UNITS_INLINE constexpr double epsilon_0 = 13.605693122994 * eV;
            FML_UNITS_INLINE constexpr double xhi0 = 24.587387 * eV;
            FML_UNITS_INLINE constexpr double xhi1 = 4.0 * epsilon_0;
            FML_UNITS_INLINE constexpr double H0_over_h = 100.0 * km / s / Mpc;
        } // namespace SI
#undef FML_UNITS_INLINE

        /// Class for dealing with physical units. Contains physical constants and conversion factors from SI to
        /// whatever units you want.
        class ConstantsAndUnits {
          private:
            std::string description = "SI units";

            void throw_error(std::string errormessage) const {
#ifdef USE_MPI
                std::cout << errormessage << std::flush;
                MPI_Abort(MPI_COMM_WORLD, 1);
                abort();
#else
                throw std::runtime_error(errormessage);
#endif
            }

          public:
            // NB: the units below are what one meter, second, ... is in your units, i.e. a length of x meters is
            // x * m in your units (e.g. m = 1e-3 for km as the length unit)
            /// One meter in your length unit
            double m = 1.0;
            /// One second in your time unit
            double s = 1.0;
            /// One kilogram in your mass unit
            double kg = 1.0;
            /// One kelvin in your temperature unit
            double K = 1.0;
            /// One coulomb in your charge unit
            double Co = 1.0;

            // Physical constants set by the units
            /// Boltzmanns constant
            double k_b{0.0};
            /// Fine-structure constant (dimensionless in every unit system)
            double alpha{0.0};
            /// Coloumbs constant
            double k_e{0.0};
            /// The gravitational constant
            double G{0.0};
            /// Reduced Placks constant h/2pi
            double hbar{0.0};
            /// Speed of light
            double c{0.0};
            /// Mass of electron
            double m_e{0.0};
            /// Mass of hydrogen
            double m_H{0.0};
            /// Thompson cross section
            double sigma_T{0.0};
            /// Transition rate from 2s to 1s in hydrogen
            double lambda_2s1s{0.0};
            /// Binding energy of hydrogen
            double epsilon_0{0.0};
            /// Hubble factor today without the h
            double H0_over_h{0.0};
            /// Binding energy of helium
            double xhi0{0.0};
            /// Binding energy of ionized helium
            double xhi1{0.0};

            // Derived units
            /// One year
            double yr{0.0};
            /// One gigayear
            double Gyr{0.0};
            /// Millimeter
            double mm{0.0};
            /// Centimeter
            double cm{0.0};
            /// Kilometer
            double km{0.0};
            /// Newton
            double N{0.0};
            /// Joule
            double J{0.0};
            // Watt
            double W{0.0};
            // Electronvolt
            double eV{0.0};
            /// Mega electronvolt
            double MeV{0.0};
            /// Megaparsec
            double Mpc{0.0};
            /// Kiloparsec
            double kpc{0.0};
            /// Gigaparsec
            double Gpc{0.0};
            /// Solar mass
            double Msun{0.0};
            /// Velocity m / s
            double velocity{0.0};
            /// Density kg / m^3
            double density{0.0};

            /// Construct units by type: SI, Planck, ParticlePhysics or Cosmology
            ConstantsAndUnits(std::string type = "SI") { init(type); }

            //=================================================================================
            /// User units: specify what length, time, mass, temperature and charge unit you want.
            /// E.g. if you want km to be your length unit then length = 1/1000
            /// When all are unity we have SI
            /// @param[in] meter_over_lengthunit How many length units to make 1 meters
            /// @param[in] seconds_over_timeunit How many time units to make 1 second
            /// @param[in] kilogram_over_massunit How many mass units to make 1 kilogram
            /// @param[in] kelvin_over_temperatureunit How many temperature units to make 1 kelvin
            /// @param[in] coloumb_over_chargeunit How many charge units to make 1 coloumb
            ///
            //=================================================================================
            ConstantsAndUnits(double meter_over_lengthunit,
                              double seconds_over_timeunit,
                              double kilogram_over_massunit,
                              double kelvin_over_temperatureunit,
                              double coloumb_over_chargeunit) {
                m = meter_over_lengthunit;
                s = seconds_over_timeunit;
                kg = kilogram_over_massunit;
                K = kelvin_over_temperatureunit;
                Co = coloumb_over_chargeunit;
                init("User units");
            }

            ConstantsAndUnits(const ConstantsAndUnits &) = default;
            ConstantsAndUnits(ConstantsAndUnits &&) = default;
            ConstantsAndUnits & operator=(const ConstantsAndUnits &) = default;
            ConstantsAndUnits & operator=(ConstantsAndUnits && other) = default;

            /// Initialize the units and compute all the constants in the desired unit system
            void init(std::string type) {

                if (type == "User units") {
                    description = "User units";
                    // The m, s, kg, K, Co constants
                    // have already been set so continue
                } else if (type == "SI") {
                    description = "SI units";
                    m = 1.0;
                    s = 1.0;
                    kg = 1.0;
                    K = 1.0;
                    Co = 1.0;
                } else if (type == "Planck") {
                    description = "Planck Units (c=hbar=G=kb=1)";
                    m = sqrt((SI::c * SI::c * SI::c) / (SI::hbar * SI::G));
                    s = SI::c * m;
                    kg = SI::G * (m * m * m) / (s * s);
                    K = SI::k_b * kg * (m * m) / (s * s);
                    Co = sqrt(SI::k_e * kg * (m * m * m) / (s * s));
                } else if (type == "ParticlePhysics") {
                    description = "Particle Physics Units (c=hbar=kb=1) with eV = 1";
                    kg = (SI::c * SI::c) / SI::eV;
                    s = (SI::c * SI::c) / (kg * SI::hbar);
                    m = s / SI::c;
                    K = SI::k_b / (s * s) * m * m * kg;
                    Co = 1.0;
                } else if (type == "Cosmology") {
                    description = "Cosmo Units (Mpc, Hubble time, Solar-masses, Kelvin)";
                    m = 1.0 / (1e6 * SI::pc); // Megaparsecs
                    s = 1e5 / (1e6 * SI::pc); // Hubble-time 1/(H0/h)
                    kg = 1.0 / SI::Msun;      // Solar masses
                    K = 1.0;                  // Kelvin
                    Co = 1.0;                 // Coulomb
                } else {
                    std::string errormessage = "[ConstantsAndUnits::init] Unknown units: " + type +
                                               " Choose between: SI, Planck, ParticlePhysics, Cosmology\n";
                    throw_error(errormessage);
                }

                // Derived units
                mm = 1e-3 * m;              // Millimeters when SI
                cm = 1e-2 * m;              // Centimeters when SI
                km = 1e3 * m;               // Kilometers when SI
                N = kg * m / (s * s);       // Newton when SI
                J = N * m;                  // Joule when SI
                W = J / s;                  // Watt when SI
                eV = SI::eV * J;            // Electronvolt when SI
                MeV = eV * 1e6;             // Megaelectronvolt when SI
                Mpc = 1e6 * SI::pc * m;     // Megaparsec when SI
                kpc = 1e-3 * Mpc;           // kiloparsec when SI
                Gpc = 1e3 * Mpc;            // Gigaparsecs when SI
                Msun = SI::Msun * kg;       // Mass of sun
                velocity = m / s;           // Velocity
                density = kg / (m * m * m); // Density

                // The physical constants we use in the code in the desired units
                alpha = SI::alpha;
                k_b = SI::k_b * J / K;                 // Bolzmanns constant
                k_e = SI::k_e * N * m * m / (Co * Co); // Columbs constant
                G = SI::G * N * m * m / (kg * kg);     // Gravitational constant
                hbar = SI::hbar * J * s;               // Reduced Plancks constant
                c = SI::c * m / s;                     // Speed of light
                m_e = SI::m_e * kg;                    // Mass of electron
                m_H = SI::m_H * kg;                    // Mass of hydrogen atom
                sigma_T = SI::sigma_T * m * m;         // Thomson scattering cross-section
                lambda_2s1s = SI::lambda_2s1s / s;     // Transition rate 2s->1s for hydrogen
                epsilon_0 = SI::epsilon_0 * J;         // Ionization energy for the ground state of hydrogen
                H0_over_h = 100 * km / s / Mpc;        // Hubble constant without little 'h', i.e. H0 / h
                xhi0 = SI::xhi0 * J;                   // Ionization energy for neutral Helium
                xhi1 = SI::xhi1 * J;                   // Ionization energy for singly ionized Helium
                yr = SI::yr * s;                       // One year
                Gyr = 1e9 * yr;                        // Gigayear
            }

            /// Convert a length in your unit to SI unit
            double length_to_SI(double x) { return x / m; }
            /// Convert a length in SI to your unit
            double length_to_user_units(double x) { return x * m; }
            /// Convert a time in your unit to SI unit
            double time_to_SI(double x) { return x / s; }
            /// Convert a time in SI to your unit
            double time_to_user_units(double x) { return x * s; }
            /// Convert a mass in your unit to SI unit
            double mass_to_SI(double x) { return x / kg; }
            /// Convert a mass in SI to your unit
            double mass_to_user_units(double x) { return x * kg; }
            /// Convert a temperature in your unit to SI unit
            double temperature_to_SI(double x) { return x / K; }
            /// Convert a temperature in SI to your unit
            double temperature_to_user_units(double x) { return x * K; }
            /// Convert a charge in your unit to SI unit
            double charge_to_SI(double x) { return x / Co; }
            /// Convert a charge in SI to your unit
            double charge_to_user_units(double x) { return x * Co; }
            /// Convert an energy in your unit to SI unit
            double energy_to_SI(double x) { return x / J; }
            /// Convert an energy in SI to your unit
            double energy_to_user_units(double x) { return x * J; }
            /// Convert a density in your unit to SI unit
            double density_to_SI(double x) { return x / density; }
            /// Convert a density in SI to your unit
            double density_to_user_units(double x) { return x * density; }
            /// Convert a velocity in your unit to SI unit
            double velocity_to_SI(double x) { return x / velocity; }
            /// Convert a velocity in SI to your unit
            double velocity_to_user_units(double x) { return x * velocity; }

            /// Show info about the given set of units
            void info() {
#ifdef USE_MPI
                int ThisTask = 0;
                int initialized = 0;
                MPI_Initialized(&initialized);
                if (initialized)
                    MPI_Comm_rank(FML::communicator, &ThisTask);
                if (ThisTask > 0)
                    return;
#endif
                std::cout << "\n============================================\n";
                std::cout << "Using units [" << description << "]: \n";
                std::cout << "============================================\n";
                std::cout << "\n";
                std::cout << "1 Length unit:    " << length_to_SI(1.0) << " meters = " << length_to_SI(1.0) * (m / Mpc)
                          << " Mpc\n";
                std::cout << "1 Time unit:      " << time_to_SI(1.0) << " seconds\n";
                std::cout << "1 Mass unit:      " << mass_to_SI(1.0) << " kg = " << mass_to_SI(1.0) * (kg / Msun)
                          << " solar masses\n";
                std::cout << "1 Temp unit:      " << temperature_to_SI(1.0) << " Kelvin\n";
                std::cout << "1 Energy unit:    " << energy_to_SI(1.0) << " Joules = " << energy_to_SI(1.0) * (J / eV)
                          << " eV\n";
                std::cout << "1 Density unit:   " << density_to_SI(1.0) << " kg/m^3\n";
                std::cout << "1 Velocity unit:  " << velocity_to_SI(1.0) << " m/s\n";
                std::cout << "\n";
                std::cout << "1 m   equals:   " << length_to_user_units(1.0) << " length units\n";
                std::cout << "1 s   equals:   " << time_to_user_units(1.0) << " time units\n";
                std::cout << "1 kg  equals:   " << mass_to_user_units(1.0) << " mass units\n";
                std::cout << "1 K   equals:   " << temperature_to_user_units(1.0) << " temperature units\n";
                std::cout << "1 C   equals:   " << charge_to_user_units(1.0) << " charge units\n";
                std::cout << "1 J   equals:   " << energy_to_user_units(1.0) << " energy units\n";
                std::cout << "1 m/s  equals:  " << velocity_to_user_units(1.0) << " velocity units\n";
                std::cout << "\n";
                std::cout << "Fundamental Constants in user units: \n";
                std::cout << "Speed of light     c  = " << c << "\n";
                std::cout << "Boltzmann constant kb = " << k_b << "\n";
                std::cout << "Coloumbs constant  ke = " << k_e << "\n";
                std::cout << "Newtons constant   G  = " << G << "\n";
                std::cout << "Plancks constant hbar = " << hbar << "\n";
                std::cout << "============================================\n\n";
            }
        };
    } // namespace UTILS
} // namespace FML

#endif
