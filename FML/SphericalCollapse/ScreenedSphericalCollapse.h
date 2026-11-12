#ifndef SCREENEDSPHERICALCOLLAPSE_INCLUDE
#define SCREENEDSPHERICALCOLLAPSE_INCLUDE

#include <FML/SphericalCollapse/SphericalCollapseModel.h>
#include <FML/Spline/Spline.h>

#include <functional>

//=====================================================================================================================
///
/// Spherical collapse of a top-hat in modified gravity with screening: the fifth force depends on the halo's own
/// state (density, mass) and on its environment, so the collapse threshold delta_c(M, z) and the virial quantities
/// depend on the halo mass.
///
/// The halo radius R = a y R_th (R_th the comoving Lagrangian radius, 1 + delta = y^-3) obeys
///
///     y'' + (2 + dlogE/dx) y' + OmegaM(x) / 2 (1 + F) (y^-3 - 1) y = 0                 (x = log a)
///
/// (Schmidt et al. 2010, arXiv:0911.5178; Lombriser et al. 2013, arXiv:1304.6395; Cataneo et al. 2019,
/// arXiv:1812.05594, appendix A): the background acceleration cancels and the fifth force F (G_eff/G - 1 for the
/// mass excess) acts on the overdensity. Here it is written for delta, as in SphericalCollapse.h:
///
///     delta'' = -(2 + dlogE/dx) delta' + 4/3 delta'^2 / (1 + delta) + 3/2 OmegaM(x) (1 + F) delta (1 + delta)
///
/// F(x, delta, delta_env) is any function (FifthForce). Provided:
///  - fifth_force_hu_sawicki: chameleon thin shell of Hu-Sawicki f(R) (n = 1 is the usual model):
///        Delta R / R = (|f_R(rho_env)| - |f_R(rho_halo)|) / (2 Phi_N),  Phi_N = G M / R = (H0 R_th)^2 OmegaM / (2 a y)
///        F = 1/3 [1 - (1 - Delta R / R)^3] for Delta R / R < 1 (the mass fraction in the shell), 1/3 otherwise
///    with |f_R(rho)| = |f_R0| [(OmegaM + 4 OmegaL) / (OmegaM a^-3 (1 + delta) + 4 OmegaL)]^(n + 1) (large-curvature
///    limit, LCDM background)
///  - fifth_force_ndgp: Vainshtein screening of normal-branch DGP (Schmidt 2010, arXiv:0911.5178):
///        F = 2 / (3 beta) (sqrt(1 + eps) - 1) / eps,  eps = 8 (r_c H)^2 OmegaM(x) delta / (9 beta^2),
///        beta = 1 + 2 H r_c (1 + dlogE/dx / 3),  H0 r_c = 1 / (2 sqrt(Omega_rc))
///    -> 1 / (3 beta) (linear theory) for small eps
///  - fifth_force_unscreened: F = mu(x) - 1 (any scale-independent G_eff/G)
///
/// The environment: a top-hat of linear overdensity delta_env_lin (extrapolated to the collapse time with the linear
/// growth of the model, i.e. model.muofx) evolves with the same equation and mu = model.muofx (no screening: GR for
/// f(R) and DGP with muofx = 1, as in Cataneo et al. 2019), from the same initial time. For f(R) its density sets
/// f_R outside the halo. environment_most_probable gives the most probable environment of Lombriser et al. 2013.
///
/// The linear overdensity (and so delta_c) is extrapolated with model.muofx: set it to 1 for f(R) and DGP to get
/// delta_c with the GR growth (as ReACT, which uses it with sigma(M) of the LCDM linear spectrum).
///
/// Validated against ReACT's spherical collapse (ACTio-ReACTio a36dda02, SCOL.cpp; tests/test_screened_collapse.cpp):
/// the most probable environment to 2e-6, delta_c to 8e-5, a_vir to 2e-4 and delta(a_vir) to 1.4e-3 for f(R) and
/// nDGP at z <= 1. (ReACT brackets delta_ini by [1.1, 4.4] zeta_i / a for modified gravity, zeta_i the initial
/// environment; at high redshift its delta_c is pinned at the bracket edge, e.g. 5% low at z = 2.3. This code
/// brackets by [0, 1].) A collapse takes ~0.05 s.
///
//=====================================================================================================================

namespace FML {
    namespace COSMOLOGY {
        namespace SPHERICALCOLLAPSE {

            using Spline = FML::INTERPOLATION::SPLINE::Spline;

            /// The fifth force F (G_eff/G - 1 for the mass excess) of a top-hat at x = log a with nonlinear
            /// overdensity delta in an environment of nonlinear overdensity delta_env
            using FifthForce = std::function<double(double x, double delta, double delta_env)>;

            /// c / H0 in Mpc/h
            constexpr double c_over_H0_Mpch = 2997.92458;

            /// No fifth force (GR)
            FifthForce fifth_force_none();

            /// F = mu(x) - 1: a scale-independent, unscreened G_eff/G
            FifthForce fifth_force_unscreened(Func1 muofx);

            /// Hu-Sawicki f(R) thin shell for a halo of comoving Lagrangian radius R_th [Mpc/h]. fR0 = |f_R0|, n the
            /// Hu-Sawicki index, OmegaM the matter density today (LCDM background)
            FifthForce fifth_force_hu_sawicki(double fR0, double n, double OmegaM, double R_th);

            /// Normal-branch DGP with Omega_rc = 1 / (4 r_c^2 H0^2) on the background of model (E(x), dlogE/dx,
            /// OmegaM(x))
            FifthForce fifth_force_ndgp(double Omega_rc, const SphericalCollapseModel & model);

            /// Comoving Lagrangian radius [Mpc/h] of mass M [Msun/h] in a universe with matter density OmegaM
            double lagrangian_radius(double M, double OmegaM);

            /// The linear threshold for the collapse of the environment, delta_c(OmegaM, z) of LCDM (fit, Kitayama &
            /// Suto 1996)
            double environment_collapse_threshold(double OmegaM, double z);

            /// The most probable linear environment overdensity (at redshift z) of Lombriser et al. 2013 (arXiv:1304.6395;
            /// as in ReACT): the maximum of the distribution of the environment density in spheres of Eulerian
            /// radius 5 Mpc/h, from sigma^2(R) and its derivative d sigma^2 / dR at R = 8 Mpc/h (at redshift z, R
            /// in Mpc/h)
            double environment_most_probable(double sigma2_8, double dsigma2dR_8, double OmegaM, double z);

            /// The result of a screened spherical collapse
            struct ScreenedCollapseResult {
                double xcollapse{};      ///< The collapse time x = log a
                double delta_ini{};      ///< Initial (linear) overdensity at xini
                double deltac{};         ///< Linear overdensity at the collapse (growth with model.muofx)
                double delta_env_ini{};  ///< Initial (linear) overdensity of the environment
                double xta{};            ///< Turnaround
                double xvir{};           ///< Virialization (2 E_kin + E_pot = 0, with the fifth force)
                double delta_vir{};      ///< Nonlinear overdensity at xvir
                double DeltaVir{};       ///< (1 + delta_vir) (a_collapse / a_vir)^3: w.r.t. the mean density at the collapse
                double F_vir{};          ///< The fifth force at xvir
                Spline delta_of_x;       ///< Nonlinear delta(x) of the halo up to the collapse
                Spline delta_env_of_x;   ///< Nonlinear delta(x) of the environment
            };

            /// Accuracy settings
            struct ScreenedCollapseOptions {
                double xini = -11.512925464970229; ///< log(1e-5)
                int npts = 1000;                   ///< Output points (linear in a) up to the collapse
                double epsilon_delta = 1e-13;      ///< Absolute accuracy of delta_ini (bisection)
                double delta_infinity = 1e16;      ///< delta at which the halo counts as collapsed
            };

            /// Find the initial overdensity that collapses at xcollapse with the fifth force F in an environment of
            /// linear overdensity delta_env_lin (at xcollapse); compute delta_c, turnaround and virial quantities
            ScreenedCollapseResult screened_spherical_collapse(const SphericalCollapseModel & model,
                                                               const FifthForce & F,
                                                               double xcollapse,
                                                               double delta_env_lin = 0.0,
                                                               const ScreenedCollapseOptions & options = {});

        } // namespace SPHERICALCOLLAPSE
    } // namespace COSMOLOGY
} // namespace FML

#endif
