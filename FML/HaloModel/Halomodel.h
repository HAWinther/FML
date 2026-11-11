#ifndef HALOMODEL_INCLUDE
#define HALOMODEL_INCLUDE

//=====================================================================================================================
// The general halo model for P(k, z), n(M, z) and b(M, z), with HMcode-2016 (Mead et al. 2015, 2016) as a
// prescription. HMcode-2020 is HMcode2020.h. Shared building blocks (variance, profiles, mass functions, collapse
// fits) are in HaloModelTools.h. See README.md in this folder.
//
// Units: k in h/Mpc, R in Mpc/h, M in Msun/h, P(k) in (Mpc/h)^3, n in (h/Mpc)^3. x = ln a.
//
// Usage:
//   auto cosmology = cosmology_from_pofk_file(collapse_model, "pofk_lin.txt");   // or cosmology_from_background
//   HaloModelOptions options;
//   options.prescription = Prescription::HMcode2016;
//   HaloModel hm(cosmology, options);
//   hm.compute_at_redshift(0.0);
//   double P = hm.pofk(1.0);
//=====================================================================================================================

#include <FML/HaloModel/HaloModelTools.h>
#include <FML/SphericalCollapse/SphericalCollapse.h>
#include <FML/Spline/Spline.h>
#include <FML/Timing/Timings.h>

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>

namespace FML {
    namespace COSMOLOGY {
        namespace HALOMODEL {

            class HaloModel;

            using Spline = FML::INTERPOLATION::SPLINE::Spline;
            using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
            using DVector = std::vector<double>;
            using Func1 = std::function<double(double)>;
            using Func2 = std::function<double(double, double)>;
            using SphericalCollapseModel = FML::COSMOLOGY::SPHERICALCOLLAPSE::SphericalCollapseModel;

            /// A linear power spectrum P(x, k): x = ln a, k in h/Mpc, P in (Mpc/h)^3
            using LinearPower = std::function<double(double, double)>;

            //=========================================================================================================
            /// What the halo model needs from the cosmology
            //=========================================================================================================
            struct HaloModelCosmology {
                /// E(x), Omega_m(x), dln E/dx, mu(x) and w(x): the growth factor, spherical collapse and Omega_m(z)
                SphericalCollapseModel collapse;
                /// Total matter density today (with massive neutrinos): the mean density in M(R)
                double Omega_m{0.3};
                /// Massive-neutrino fraction of Omega_m
                double f_nu{0.0};
                /// Linear total-matter power spectrum
                LinearPower P_m;
                /// Linear cold-matter (CDM + baryons) power spectrum, relative to the mean cold density. It gives
                /// sigma(R), the peak heights and the concentrations. Empty: P_m (no massive neutrinos)
                LinearPower P_cb;
            };

            /// The linear P(k) in a file with columns k [h/Mpc] and P [(Mpc/h)^3] at the time x_pofk, scaled to
            /// other times with the growth factor of the collapse model (scale-independent growth) and power-law
            /// extrapolated in k. normalization multiplies P(k). Omega_m is collapse.OmegaM.
            HaloModelCosmology cosmology_from_pofk_file(const SphericalCollapseModel & collapse,
                                                        const std::string & filename,
                                                        double x_pofk = 0.0,
                                                        double normalization = 1.0,
                                                        int col_k = 0,
                                                        int col_pofk = 1);
            /// As above from a spline of ln Delta^2(ln k) (as made by read_pofk_file)
            HaloModelCosmology cosmology_from_pofk_spline(const SphericalCollapseModel & collapse,
                                                          const Spline & logDelta_of_logk,
                                                          double x_pofk = 0.0);
            // From an FML BackgroundCosmology (and the solver's P(k)): HaloModelBackground.h

            /// Read a file [k, P(k)] in (h/Mpc, (Mpc/h)^3) and make a spline of ln Delta^2(ln k), extrapolated as a power
            /// law at both ends (to k = 1e-12 and 1e12)
            void read_pofk_file(const std::string & filename_pofk,
                                Spline & logDelta_of_logk_spline,
                                double normalization = 1.0,
                                int col_k = 0,
                                int col_pofk = 1);

            /// The Fourier transform u(k) of a halo profile given as rho_r3(r) = 4 pi r^3 rho(r) / M (any normalisation),
            /// zero or not outside Rvir, normalised to u = 1 at k_array[0] (choose k_array[0] Rvir << 1)
            DVector fourier_transform_profile(const std::function<double(double)> & rho_r3,
                                              const DVector & k_array,
                                              double Rvir,
                                              int accuracy_boost = 1);

            //=========================================================================================================
            // Options
            //=========================================================================================================

            enum class Prescription {
                /// The halo model with deltac and DeltaVir from spherical collapse in the collapse model
                Standard,
                /// HMcode-2016 (Mead et al. 2015, 2016, as in CAMB's halofit_version = mead2016): fitted deltac,
                /// DeltaVir, concentration amplitude, one-halo damping, two-halo damping, halo bloating and transition
                HMcode2016
            };
            enum class ConcentrationModel {
                /// c = A (1 + z_f) / (1 + z), z_f where D(z_f)/D(z) sigma(f M) = deltac (Bullock et al. 2001)
                Bullock2001,
                /// c = c0 (M / M0)^-beta / (1 + z)
                PowerLaw
            };

            // Customisation. The functions get the halo model at the current redshift (state(), sigma(R), nu_of_M(M),
            // growth_factor(x), ...)
            /// f(nu): f(nu) dnu is the mass fraction in haloes with peak height in dnu
            using MultiplicityFunction = std::function<double(double nu, const HaloModel & hm)>;
            /// Linear halo bias b(nu)
            using BiasFunction = std::function<double(double nu, const HaloModel & hm)>;
            /// Concentration c(M) of the (NFW) profile
            using ConcentrationFunction = std::function<double(double M, const HaloModel & hm)>;
            /// A halo profile as the dimensionless 4 pi r^3 rho(r, M) / M (any normalisation: the Fourier transform is
            /// normalised to 1 at k = 0). Transformed numerically
            using DensityProfile = std::function<double(double r, double M, const HaloModel & hm)>;
            /// The smoothing filter W(kR) of sigma(R)
            using WindowFunction = std::function<double(double kR)>;

            struct HaloModelOptions {
                Prescription prescription{Prescription::Standard};

                //======================================
                // Mass function and bias. Default (empty functions): Sheth & Tormen with these parameters (the
                // normalisation A is irrelevant for P(k), the halo integrals are normalised)
                //======================================
                ShethTormen sheth_tormen{};
                MultiplicityFunction multiplicity{};
                BiasFunction bias{};

                //======================================
                // Profile: NFW (analytic transform) with the concentration below, or a custom density_profile
                // (numerical transform). HMcode-2016 uses NFW with Bullock2001 and its own A
                //======================================
                DensityProfile density_profile{};
                ConcentrationModel concentration_model{ConcentrationModel::Bullock2001};
                double bullock_A{4.0};
                double bullock_mass_fraction{0.01};
                double powerlaw_c0{4.75};
                double powerlaw_M0{1e14};
                double powerlaw_beta{0.13};
                ConcentrationFunction concentration{};

                WindowFunction window{};

                //======================================
                // Rescalings of the fiducial values (also for HMcode-2016)
                //======================================
                double deltac_multiplier{1.0};
                double DeltaVir_multiplier{1.0};
                double concentration_multiplier{1.0};

                //======================================
                // HMcode-like modifications of the Standard model (HMcode2016 sets its own at each redshift)
                //======================================
                /// One-halo damping 1 - exp(-(k/kstar)^2) (0: none; 0.01 h/Mpc avoids a constant P_1h at low k)
                double kstar{0.01};
                /// Transition (P_1h^alpha + P_2h^alpha)^(1/alpha)
                double alpha{1.0};
                /// Halo bloating u(k, M) -> u(nu^eta k, M)
                double eta{0.0};
                /// Two-halo damping P_2h (1 - f tanh^2(k sigma_v / sqrt f)) (0: none)
                double f_damp{0.0};

                //======================================
                // Ranges and accuracy
                //======================================
                /// The P(k) splines cover [kmin, kmax] (h/Mpc)
                double kmin{1e-5};
                double kmax{100.0};
                int npts_k{1000};
                /// Points in the halo integrals and profile transforms (runtime is quadratic in this)
                int accuracy_boost{1};
                /// Spherical collapse (Standard): deltac(z) and DeltaVir(z) are computed for z <= zmax
                double zmax_spherical_collapse{10.0};
                double xini_spherical_collapse{std::log(1e-5)};
                int npts_spherical_collapse{1000};

                bool verbose{false};
            };

            //=========================================================================================================
            /// Values at the current redshift
            //=========================================================================================================
            struct HaloModelState {
                double z{std::numeric_limits<double>::quiet_NaN()};
                double x{std::numeric_limits<double>::quiet_NaN()};
                double deltac{};
                double DeltaVir{};
                /// sigma(8 Mpc/h) of the cold matter (the halo variance) and of the total matter
                double sigma8{};
                double sigma8_m{};
                /// sigma_v(R = 0) and sigma_v(R = 100 Mpc/h) of the total matter (Mpc/h)
                double sigmav{};
                double sigmav100{};
                /// sigma(rnl) = deltac and n_eff = -3 - dln sigma^2 / dln R there (cold matter)
                double rnl{};
                double neff{};
                /// The HMcode parameters in use (HMcode-2016 fits or the Standard options)
                double kstar{};
                double alpha{};
                double eta{};
                double f_damp{};
                /// The amplitude of the Bullock et al. concentration (with the multiplier and, for HMcode-2016, the Dolag
                /// correction)
                double concentration_A{};
                /// Mean comoving matter density (Msun/h)/(Mpc/h)^3
                double rho_mean{};
            };

            //=========================================================================================================
            /// @brief The halo model: P(k, z) (one- and two-halo terms), n(M, z), b(M, z), from a linear power
            /// spectrum and a spherical-collapse model (expansion and mu(x) for modified gravity). HMcode-2016 is
            /// Prescription::HMcode2016.
            ///
            /// With massive neutrinos (cosmology.f_nu > 0 and P_cb given) the peak heights, mass function and
            /// concentrations use the cold-matter variance and the two-halo term the total-matter P(k). Standard
            /// multiplies the one-halo term by (1 - f_nu)^2 (the neutrinos do not cluster in haloes); HMcode-2016
            /// follows CAMB (no such factor, the fits have neutrino corrections).
            //=========================================================================================================
            class HaloModel {
              public:
                HaloModel() = default;
                explicit HaloModel(HaloModelCosmology cosmology, HaloModelOptions options = {});

                /// New options (re-runs the growth and spherical-collapse set-up; call compute_at_redshift again)
                void set_options(HaloModelOptions options);
                const HaloModelOptions & get_options() const { return options; }
                const HaloModelCosmology & get_cosmology() const { return cosmology; }

                /// Compute everything at redshift z
                void compute_at_redshift(double z);
                bool computed() const { return not std::isnan(current.x); }
                const HaloModelState & state() const { return current; }

                //=====================================================
                // At the current redshift
                //=====================================================
                /// The halo-model P(k), its linear (total matter), one- and two-halo parts. k in [kmin, kmax]
                double pofk(double k) const;
                double pofk_linear(double k) const;
                double pofk_onehalo(double k) const;
                double pofk_twohalo(double k) const;
                /// dn/dln M and the cumulative n(> M) in (h/Mpc)^3 (1e5 <= M <= 1e18 Msun/h)
                double dndlogM(double M) const;
                double n_of_M(double M) const;
                /// sigma(R) (cold matter), the Lagrangian and virial radius of M, nu = deltac / sigma(R(M))
                double sigma(double R) const;
                double R_of_M(double M) const;
                double Rvir_of_M(double M) const;
                double nu_of_M(double M) const;
                /// f(nu), b(nu) and b(M)
                double multiplicity(double nu) const;
                double bias(double nu) const;
                double bias_of_M(double M) const;
                /// Concentration (with the multiplier) and the Bullock et al. formation redshift
                double concentration(double M) const;
                double formation_redshift(double M) const;
                /// The normalised Fourier transform of the halo profile u(k|M) (without bloating)
                double profile_fourier(double k, double M) const;

                //=====================================================
                // Any time
                //=====================================================
                /// The (unnormalised) growth factor of the collapse model and the growth rate dln D / dln a
                double growth_factor(double x) const { return growthfactor_of_x_spline(x); }
                double growth_rate(double x) const { return growthrate_of_x_spline(x); }
                /// deltac(z) and DeltaVir(z) of the prescription (with the multipliers)
                double deltac_of_z(double z) const;
                double DeltaVir_of_z(double z) const;

                //=====================================================
                // Output
                //=====================================================
                /// k, P_NL, P_lin, P_1h, P_2h (in [kmin, kmax])
                void output_pofk(const std::string & filename, double kmin = 1e-5, double kmax = 100, int npts_k = 1000) const;
                /// M, dn/dlnM, n(>M), R, Rvir, nu, c, b
                void output_nofM(const std::string & filename, double Mmin = 1e8, double Mmax = 1e16, int npts_M = 100) const;
                /// z, deltac, DeltaVir (and the other HMcode-2016 parameters)
                void output_deltac(const std::string & filename, double zmin = 0.0, double zmax = 10.0, int npts_z = 100) const;
                void info() const;

                FML::UTILS::Timings & timings() { return timer; }

              private:
                HaloModelCosmology cosmology;
                HaloModelOptions options;
                HaloModelState current{};
                FML::UTILS::Timings timer;

                // Set up by init()
                Spline growthfactor_of_x_spline{"D(x)"};
                Spline growthrate_of_x_spline{"f(x)"};
                Spline deltac_of_x_spline{"deltac(x)"};
                Spline DeltaVir_of_x_spline{"DeltaVir(x)"};
                Spline xta_of_x_spline{"xta(x)"};
                Spline xnl_of_x_spline{"xnl(x)"};
                Spline xvir_of_x_spline{"xvir(x)"};
                Spline delta_ini_of_x_spline{"deltaini(x)"};
                double dolag_2016{1.0};

                // Set up by compute_at_redshift()
                Spline logsigma_of_logR_spline{"ln sigma(ln R)"};
                Spline lognu_of_logR_spline{"ln nu(ln R)"};
                Spline logR_of_lognu_spline{"ln R(ln nu)"};
                Spline formationredshift_of_logM_spline{"zf(ln M)"};
                Spline dndlogM_of_logM_spline{"dn/dlnM(ln M)"};
                Spline n_of_logM_spline{"n(ln M)"};
                Spline logDelta_lin_of_logk_spline{"ln Delta_lin(ln k)"};
                Spline logDelta_onehalo_of_logk_spline{"ln Delta_1h(ln k)"};
                Spline logDelta_twohalo_of_logk_spline{"ln Delta_2h(ln k)"};
                Spline logDelta_full_of_logk_spline{"ln Delta(ln k)"};
                Spline2D y_of_logR_and_logk_spline{"y(ln R, ln k)"};

                void init();
                void check_computed(const std::string & where) const;
                void check_k(double k, const std::string & where) const;
                double Delta2_m(double x, double k) const;
                double Delta2_cb(double x, double k) const;
                /// The HMcode-2016 sigma8, sigma_v, n_eff, ... at any x (for output_deltac)
                HaloModelState hmcode2016_state_at(double x) const;
            };

        } // namespace HALOMODEL
    } // namespace COSMOLOGY
} // namespace FML

#endif
