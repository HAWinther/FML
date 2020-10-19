#ifndef UNITS_HEADER
#define UNITS_HEADER

#include <cmath>
#include <iostream>
#ifdef USE_MPI
#include <mpi.h>
#endif

namespace FML {
    namespace UTILS {

        /// Class for dealing with physical units. Contains physical constants and conversion factors from SI to
        /// whatever units you want.
        class ConstantsAndUnits {
          private:
            // Constants in SI units (do not change)
            // These are used to set the units below
            double G_SI = 6.67430e-11;
            double c_SI = 2.99792458e8;
            double k_b_SI = 1.38064852e-23;
            double k_e_SI = 8.9875517923e9;
            double hbar_SI = 1.054571817e-34;
            double eV_SI = 1.60217653e-19;
            double m_e_SI = 9.10938356e-31;
            double m_H_SI = 1.6735575e-27;
            double Msun_SI = 1.98847e30;
            double sigma_T_SI = 6.6524587158e-29;
            double lambda_2s1s_SI = 8.227;
            double epsilon_0_eV = 13.605693122994;
            double xhi0_eV = 24.587387;
            double xhi1_eV = 4.0 * epsilon_0_eV;
            double pc_SI = 3.08567758e16;

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
            /// Size of your length unit in meters
            double m = 1.0;
            /// Size of your time unit in seconds
            double s = 1.0;
            /// Size of your mass unit in kg
            double kg = 1.0;
            /// Size of your temperature unit in kelvin
            double K = 1.0;
            /// Size of your charge unit in coloumbs
            double Co = 1.0;

            // Physical constants set by the units
            /// Boltzmanns constant
            double k_b{0.0};
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
            /// Milimeter
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
                    m = sqrt((c_SI * c_SI * c_SI) / (hbar_SI * G_SI));
                    s = c_SI * m;
                    kg = G_SI * (m * m * m) / (s * s);
                    K = k_b_SI * kg * (m * m) / (s * s);
                    Co = sqrt(k_e_SI * kg * (m * m * m) / (s * s));
                } else if (type == "ParticlePhysics") {
                    description = "Particle Physics Units (c=hbar=kb=1) with eV = 1";
                    m = eV_SI / hbar_SI / c_SI;
                    s = c_SI * m;
                    kg = 1.0 / hbar_SI * s / (m * m);
                    K = k_b_SI * G_SI * (m * m * m) / (s * s) * (m * m) / (s * s);
                    Co = 1.0;
                } else if (type == "Cosmology") {
                    description = "Cosmo Units (Mpc, Hubble time, Solar-masses, Kelvin)";
                    m = 1.0 / (1e6 * pc_SI); // Megaparsecs
                    s = 1e5 / (1e6 * pc_SI); // Hubble-time 1/(H0/h)
                    kg = 1.0 / Msun_SI;      // Solar masses
                    K = 1.0;                 // Kelvin
                    Co = 1.0;                // Coulomb
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
                eV = eV_SI * J;             // Electronvolt when SI
                MeV = eV * 1e6;             // Megaelectronvolt when SI
                Mpc = 1e6 * pc_SI * m;      // Megaparsec when SI
                kpc = 1e-3 * Mpc;           // kiloparsec when SI
                Gpc = 1e3 * Mpc;            // Gigaparsecs when SI
                Msun = Msun_SI * kg;        // Mass of sun
                velocity = m / s;           // Velocity
                density = kg / (m * m * m); // Density

                // The physical constants we use in the code in the desired units
                k_b = k_b_SI * J / K;                 // Bolzmanns constant
                k_e = k_e_SI * N * m * m / (Co * Co); // Columbs constant
                G = G_SI * N * m * m / (kg * kg);     // Gravitational constant
                hbar = hbar_SI * J * s;               // Reduced Plancks constant
                c = c_SI * m / s;                     // Speed of light
                m_e = m_e_SI * kg;                    // Mass of electron
                m_H = m_H_SI * kg;                    // Mass of hydrogen atom
                sigma_T = sigma_T_SI * m * m;         // Thomas scattering cross-section
                lambda_2s1s = lambda_2s1s_SI / s;     // Transition rate 2s->1s for hydrogen
                epsilon_0 = epsilon_0_eV * eV;        // Ionization energy for the ground state of hydrogen
                H0_over_h = 100 * km / s / Mpc;       // Hubble constant without little 'h', i.e. H0 / h
                xhi0 = xhi0_eV * eV;                  // Ionization energy for neutral Helium
                xhi1 = xhi1_eV * eV;                  // Ionization energy for singly ionized Helium
                yr = 60. * 60. * 24 * 365.24 * s;     // One year
                Gyr = 1e9 * yr;                       // Gigayear
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
                MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
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
