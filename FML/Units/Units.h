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
            // m is the size of your length unit in meters,
            // s the size of your time unit in seconds, etc.
            double m = 1.0;
            double s = 1.0;
            double kg = 1.0;
            double K = 1.0;
            double Co = 1.0;

            // Physical constants set by the units
            double k_b{0.0};
            double k_e{0.0};
            double G{0.0};
            double hbar{0.0};
            double c{0.0};
            double m_e{0.0};
            double m_H{0.0};
            double sigma_T{0.0};
            double lambda_2s1s{0.0};
            double epsilon_0{0.0};
            double H0_over_h{0.0};
            double xhi0{0.0};
            double xhi1{0.0};
            double yr{0.0};
            double Gyr{0.0};

            // Derived units
            double mm{0.0};
            double cm{0.0};
            double km{0.0};
            double N{0.0};
            double J{0.0};
            double W{0.0};
            double eV{0.0};
            double MeV{0.0};
            double Mpc{0.0};
            double kpc{0.0};
            double Gpc{0.0};
            double Msun{0.0};
            double velocity{0.0};
            double density{0.0};

            ConstantsAndUnits(std::string type = "SI") { init(type); }

            // User units: specify what lenght, time, mass and charge unit you want
            ConstantsAndUnits(double L, double T, double M, double C) {
                m = L;
                s = T;
                kg = M;
                Co = C;
                init("User units");
            }

            ConstantsAndUnits & operator=(ConstantsAndUnits && other) = default;

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

            double length_to_SI(double x) { return x / m; }
            double length_to_user_units(double x) { return x * m; }
            double time_to_SI(double x) { return x / s; }
            double time_to_user_units(double x) { return x * s; }
            double mass_to_SI(double x) { return x / kg; }
            double mass_to_user_units(double x) { return x * kg; }
            double temperature_to_SI(double x) { return x / K; }
            double temperature_to_user_units(double x) { return x * K; }
            double charge_to_SI(double x) { return x / Co; }
            double charge_to_user_units(double x) { return x * Co; }
            double energy_to_SI(double x) { return x / J; }
            double energy_to_user_units(double x) { return x * J; }
            double density_to_SI(double x) { return x / density; }
            double density_to_user_units(double x) { return x * density; }
            double velocity_to_SI(double x) { return x / velocity; }
            double velocity_to_user_units(double x) { return x * velocity; }

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
