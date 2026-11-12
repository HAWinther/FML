#ifndef FML_HALOMODEL_HALOMODELREACTION_H
#define FML_HALOMODEL_HALOMODELREACTION_H

#include <FML/SphericalCollapse/ScreenedSphericalCollapse.h>
#include <FML/Spline/Spline.h>

#include <functional>
#include <vector>

//=====================================================================================================================
///
/// The halo-model reaction R(k) at one redshift (Cataneo et al. 2019, arXiv:1812.05594; Bose et al. 2020,
/// arXiv:2005.12184; massive neutrinos: Bose et al. 2021, arXiv:2105.12114): the ratio of the nonlinear P(k) of a
/// target cosmology (modified gravity, dark energy, massive neutrinos) to that of its pseudo cosmology (LCDM
/// background, GR, the same linear P(k) as the target), P_NL = R P_NL,pseudo:
///
///   R(k) = [(1 - f_nu)^2 P_h + 2 f_nu (1 - f_nu) sqrt(P_h P_nu) + f_nu^2 P_nu] / [P_L + P_1h,pseudo(k)]
///   P_h  = [(1 - E) exp(-k / k*) + E] P_L,cb + P_1h,real(k)
///
/// with f_nu = Omega_nu / Omega_m and the linear target spectra P_L (total matter), P_L,cb and P_nu (from the
/// transfer functions: sqrt(P_nu) = [Omega_m sqrt(P_L) - Omega_cb sqrt(P_L,cb)] / Omega_nu).
///
/// One-halo terms: P_1h = int dln M n(M) (M / rho)^2 u(k, M)^2 with
///  - real: rho = rho_cb, the spherical collapse of SphericalCollapse/ScreenedSphericalCollapse.h in the target
///    background with the fifth force (screened, depending on the mass and on the most probable environment of
///    Lombriser et al. 2013 from sigma(8 Mpc/h) of P_L,cb^LCDM); delta_c(M) extrapolated with the linear growth of
///    the LCDM background (OmegaM total in H, Omega_cb as the source) and sigma(M) of P_L,cb^LCDM, the LCDM cold
///    spectrum with the target's primordial parameters (and neutrinos)
///  - pseudo: rho = rho_m, GR collapse in the LCDM background with OmegaM as the clustering matter, sigma(M) of
///    the target P_L
///  - Sheth-Tormen n(M) with nu = delta_c(M) / sigma(M) (a = 0.75, p = 0.3, as Cataneo et al. 2019); NFW
///    profiles truncated at the virial radius of Delta_vir = (1 + delta(a_vir)) (a / a_vir)^3 with
///    c = g c0 (M / M*)^-alpha a, sigma(M*) = delta_c(M), c0 = 9, alpha = 0.13 (Bullock et al. 2001) and
///    g = D_LCDM(1) / D_target(1) for the real halos (Dolag et al. 2004; 1 for the pseudo)
///
/// E = (1 - f_nu)^2 P_1h,real(k_E) / P_1h,pseudo(k_E) (k_E = 0.01 h/Mpc) and k* (matching R at k0 = 0.06 h/Mpc to
/// one-loop perturbation theory: set_one_loop) are only used for modified gravity (a fifth force given); otherwise
/// E = 1 and k* is irrelevant. The one-loop P(k0): PerturbationTheory/MGOneLoop.h (mg_one_loop with the model's
/// kernels and P_cb for the real, mg_lcdm and P for the pseudo cosmology), for all redshifts in one call.
///
/// As in ReACT: sigma(M) integrates P(k) over 1e-4 <= k <= 50 h/Mpc, the masses are 1e5 - 1e20 Msun/h (n_mass
/// collapses), R = 1 below k = 0.01 h/Mpc. Units: k [h/Mpc], P [(Mpc/h)^3], M [Msun/h], R [Mpc/h].
///
/// Validated against ReACT (ACTio-ReACTio a36dda02, HALO.cpp) in tests/test_halomodel_reaction.cpp.
///
//=====================================================================================================================

namespace FML {
    namespace COSMOLOGY {
        namespace HALOMODEL {

            /// Options of HaloModelReaction
            struct HaloModelReactionOptions {
                int n_mass{30};             ///< Real collapses (ReACT's mass_loop)
                int n_mass_pseudo{30};      ///< Pseudo masses (GR: delta_c does not depend on the mass)
                double log10M_min{5.0}, log10M_max{20.0};
                int n_mass_integral{400};   ///< Simpson points in log M for the one-halo integrals
                double kmin_sigma{1e-4}, kmax_sigma{50.0};
                int n_sigma{4000};          ///< Points (log k) of the sigma(M) integrals
                double st_a{0.75}, st_p{0.3}, st_A{1.0 / 3.056 / 1.012};
                double c0{9.0}, alpha{0.13};
                double k_E{0.01}, k0{0.06};
                double kmin_reaction{0.01}; ///< R = 1 below this k
                SPHERICALCOLLAPSE::ScreenedCollapseOptions collapse{};
            };

            class HaloModelReaction {
              public:
                using Spline = FML::INTERPOLATION::SPLINE::Spline;
                using Func1 = std::function<double(double)>;
                using FifthForce = SPHERICALCOLLAPSE::FifthForce;
                /// The fifth force of a halo of comoving Lagrangian radius R_th [Mpc/h] (e.g.
                /// [](double R) { return fifth_force_hu_sawicki(fR0, 1, OmegaM, R); }); empty: GR
                using FifthForceOfRadius = std::function<FifthForce(double R_th)>;

                struct Cosmology {
                    double OmegaM{0.3};  ///< Total matter today (including massive neutrinos)
                    double OmegaNu{0.0}; ///< Massive neutrinos today
                    /// The target background E(x) = H/H0 and dlog E / dx (x = log a); empty: flat LCDM with OmegaM
                    Func1 Eofx{}, logEprimeofx{};
                };

                /// Linear spectra at the redshift: k [h/Mpc] -> P [(Mpc/h)^3]
                struct LinearSpectra {
                    Func1 P;        ///< Target total matter (also the pseudo cosmology's)
                    Func1 Pcb;      ///< Target cold (CDM + baryons); empty: P
                    Func1 Pcb_lcdm; ///< LCDM cold spectrum with the target's primordial parameters and neutrinos
                };

                using Options = HaloModelReactionOptions;

                /// The collapse and halo quantities of one mass
                struct Halo {
                    double log10M{}, R_th{}, deltac{}, sigma{}, a_vir{}, delta_vir{}, DeltaVir{};
                };

                HaloModelReaction(const Cosmology & cosmology,
                                  double a,
                                  const LinearSpectra & spectra,
                                  FifthForceOfRadius fifth_force = {},
                                  const Options & options = {});

                bool modified_gravity() const { return mg; }
                double scale_factor() const { return a; }
                /// The one-halo terms [(Mpc/h)^3]
                double one_halo(double k) const { return one_halo_term(k, real); }
                double one_halo_pseudo(double k) const { return one_halo_term(k, pseudo); }
                /// Modified gravity: set k* from the one-loop P_cb of the target and P of the pseudo cosmology at k0
                /// (linear + one loop, [(Mpc/h)^3]). If no k* reproduces them, k* and E fall back to the GR values
                /// (as ReACT); returns false then
                bool set_one_loop(double P_spt_real_k0, double P_spt_pseudo_k0);
                double E() const { return E_; }
                double kstar() const { return kstar_; }
                /// The reaction
                double reaction(double k) const;
                /// Concentration correction g = D_LCDM(1) / D_target(1) of the real halos
                double g_de() const { return gde; }
                /// The collapse quantities on the mass grids
                const std::vector<Halo> & real_halos() const { return real.halos; }
                const std::vector<Halo> & pseudo_halos() const { return pseudo.halos; }
                /// delta_c, sigma and Delta_vir as functions of log10 M (cubic splines through the mass grids)
                double deltac(double log10M) const { return real.deltac(log10M); }
                double sigma(double log10M) const { return real.sigma(log10M); }
                double DeltaVir(double log10M) const { return real.DeltaVir(log10M); }
                double deltac_pseudo(double log10M) const { return pseudo.deltac(log10M); }
                double sigma_pseudo(double log10M) const { return pseudo.sigma(log10M); }
                double DeltaVir_pseudo(double log10M) const { return pseudo.DeltaVir(log10M); }
                /// The most probable environment (linear overdensity at a) of the real collapse
                double environment() const { return zeta; }

              private:
                struct Population {
                    std::vector<Halo> halos;
                    double rho{}, g{1.0};
                    Spline deltac, sigma, DeltaVir; // of log10 M
                    // Quadrature in log10 M: nodes, Simpson weights, n(M) M^2 / rho^2 dlnM / dlog10M, r_vir, c
                    std::vector<double> lm, weight, rvir, conc;
                };

                Cosmology cosmo;
                double a{}, zeta{};
                LinearSpectra spec;
                Options opt;
                bool mg{false};
                double E_{1.0}, kstar_{1.0}, gde{1.0};
                bool kstar_set{false};
                Population real, pseudo;

                void build(Population & pop, bool is_real, const FifthForceOfRadius & fifth_force);
                double one_halo_term(double k, const Population & pop) const;
                double Pnu(double k) const;
            };

            //=========================================================================================================
            /// A table of R(z, k) for the solver: a bicubic spline of ln R in (x = log a, log k). R = 1 below the
            /// table's k, the endpoint value above it; outside the redshift range it throws. Safe to evaluate from
            /// several threads.
            //=========================================================================================================
            class ReactionTable {
              public:
                ReactionTable() = default;
                /// z increasing (>= 4 values), k in h/Mpc increasing (>= 4), R[iz][ik] > 0
                ReactionTable(const std::vector<double> & z,
                              const std::vector<double> & k,
                              const std::vector<std::vector<double>> & R);
                /// R at x = log a and k in h/Mpc
                double operator()(double x, double k) const;
                double x_min() const { return xmin; }
                double x_max() const { return xmax; }

              private:
                double xmin{}, xmax{}, lkmin{}, lkmax{};
                FML::INTERPOLATION::SPLINE::Spline2D lnR;
            };

        } // namespace HALOMODEL
    } // namespace COSMOLOGY
} // namespace FML

#endif
