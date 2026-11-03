#ifndef GRAVITYMODELSETTINGS_HEADER
#define GRAVITYMODELSETTINGS_HEADER

#include "ParameterFields.h"

#include <string>
#include <vector>

namespace SIMCODE {

    //========================================================================
    /// The gravity_model_* and multigrid_* parameters, listed once in fields() (the parameter table, ParameterSchema.h,
    /// and the reading, GravityModel::read_parameters, come from it; see ParameterFields.h). gravity_model itself
    /// selects the GravityModel (Models.h). The parameters of a model are only read for that model (for_gravity). The
    /// checks of the values are in the models' read_parameters
    //========================================================================
    struct GravityModelSettings {
        std::string mu_parametrization{};
        double mu_mu0{};
        std::vector<double> mu_BZ_params{};
        std::string mu_filename{};
        std::vector<double> mbeta_params{};
        double fofr_fofr0{};
        double fofr_nfofr{};
        double symmetron_assb{};
        double symmetron_beta{};
        double symmetron_L_mpch{};
        double dgp_rcH0overc{};
        bool screening{};
        std::string dgp_smoothing_filter{};
        double dgp_smoothing_scale_over_boxsize{};
        bool screening_enforce_largescale_linear{};
        double screening_linear_scale_hmpc{};
        double screening_efficiency{};
        bool fofr_exact_solution{};
        std::string fofr_multigrid_method{};
        bool symmetron_exact_solution{};
        bool dgp_exact_solution{};
        int multigrid_nsweeps{};
        int multigrid_nsweeps_first_step{};
        double multigrid_solver_residual_convergence{};
        std::string multigrid_convergence_criterion{};
        std::string multigrid_coarse_restriction{};
        double multigrid_start_redshift{};
        std::string multigrid_force_before_start{};
        int multigrid_coarse_factor{};

        template <class F>
        void fields(F && f) {
            f.for_gravity({"mu"}, [&]() {
                f.required("gravity_model_mu_parametrization",
                           mu_parametrization,
                           "DE",
                           "Poisson equation with G_eff/G = mu(a,k), no screening. DE: mu = 1 + mu0 "
                           "Omega_DE(a)/Omega_DE(a=1). BZ: mu = (1 + beta1 lambda1^2 k^2 a^s) / (1 + lambda1^2 k^2 "
                           "a^s) (Bertschinger & Zukin 2008). table: mu(a) or mu(a,k) from a file",
                           {"DE", "BZ", "table"});
                f.when(is_one_of("gravity_model_mu_parametrization", {"DE"}),
                       "gravity_model_mu_parametrization = DE",
                       [&]() {
                           f.required("gravity_model_mu_mu0",
                                      mu_mu0,
                                      0.0,
                                      "mu0 in mu = 1 + mu0 Omega_DE(a)/Omega_DE(a=1)");
                       });
                f.when(is_one_of("gravity_model_mu_parametrization", {"BZ"}),
                       "gravity_model_mu_parametrization = BZ",
                       [&]() {
                           f.required("gravity_model_mu_BZ_params",
                                      mu_BZ_params,
                                      std::vector<double>{1.0, 0.0, 4.0},
                                      "{beta1, lambda1 (Mpc/h), s} in mu = (1 + beta1 lambda1^2 k^2 a^s) / (1 + "
                                      "lambda1^2 k^2 a^s) (k in h/Mpc)",
                                      check::list_of(3));
                       });
                f.when(is_one_of("gravity_model_mu_parametrization", {"table"}),
                       "gravity_model_mu_parametrization = table",
                       [&]() {
                           f.required("gravity_model_mu_filename",
                                      mu_filename,
                                      "",
                                      "File with rows 'a mu' or 'a k mu' (k in h/Mpc, a regular grid in a and k; "
                                      "e.g. from hi_class). Splined in log a, log k; the closest value outside");
                       });
            });
            f.for_gravity({"mbeta"}, [&]() {
                f.required("gravity_model_mbeta_params",
                           mbeta_params,
                           std::vector<double>{},
                           "The parameters (beta0, n, m0, m) of beta(a) = beta0 a^n and m(a) = m0 H0 a^m",
                           check::list_of(4));
            });
            f.for_gravity({"f(R)"}, [&]() {
                f.required("gravity_model_fofr_fofr0", fofr_fofr0, 1e-5, "The value of f_R today", check::positive());
                f.optional("gravity_model_fofr_nfofr", fofr_nfofr, 1.0, "The index n of the Hu-Sawicky model");
            });
            f.for_gravity({"Symmetron"}, [&]() {
                f.optional("gravity_model_symmetron_assb",
                           symmetron_assb,
                           0.5,
                           "Symmetry breaking scale factor (no fifth-force for a < assb)");
                f.optional("gravity_model_symmetron_beta",
                           symmetron_beta,
                           1.0,
                           "Coupling strength relative to gravity");
                f.optional("gravity_model_symmetron_L_mpch",
                           symmetron_L_mpch,
                           1.0,
                           "Range of the fifth force in the background today "
                           "(Mpc/h)");
            });
            f.for_gravity({"DGP"}, [&]() {
                f.required("gravity_model_dgp_rcH0overc", dgp_rcH0overc, 1.0, "The cross-over scale rc H0 / c");
            });
            f.for_gravity({"mbeta", "f(R)", "Symmetron", "DGP"}, [&]() {
                f.optional("gravity_model_screening",
                           screening,
                           true,
                           "Use the approximate screening model (otherwise linear)");
                f.if_true("gravity_model_screening", [&]() {
                    f.for_gravity({"DGP"}, [&]() {
                        f.optional("gravity_model_dgp_smoothing_filter",
                                   dgp_smoothing_filter,
                                   "tophat",
                                   "Screening approximation: the filter used to smooth the density field",
                                   choices::smoothing_filters);
                        f.optional("gravity_model_dgp_smoothing_scale_over_boxsize",
                                   dgp_smoothing_scale_over_boxsize,
                                   0.0,
                                   "Screening approximation: the smoothing scale R/boxsize (0: no smoothing; 1 smooths "
                                   "over the whole box, which makes the screening very weak)");
                    });
                    f.optional("gravity_model_screening_enforce_largescale_linear",
                               screening_enforce_largescale_linear,
                               false,
                               "Combine the screened solution with the linear solution to get the correct linear "
                               "evolution on large scales");
                    f.optional("gravity_model_screening_linear_scale_hmpc",
                               screening_linear_scale_hmpc,
                               0.05,
                               "Use the linear solution for k < k* and the screened solution for k > k* (h/Mpc)");
                    f.for_gravity({"mbeta", "f(R)"}, [&]() {
                        f.optional("gravity_model_screening_efficiency",
                                   screening_efficiency,
                                   1.0,
                                   "Screening efficiency (1 is standard). Increases or decreases the amount of "
                                   "screening");
                    });
                });
            });
            f.for_gravity({"f(R)"}, [&]() {
                f.optional("gravity_model_fofr_exact_solution",
                           fofr_exact_solution,
                           false,
                           "Solve the exact f(R) equation with the multigrid solver (slow, mainly for testing)");
                f.when(is_true("gravity_model_fofr_exact_solution"), "gravity_model_fofr_exact_solution = true", [&]() {
                    f.always_optional("gravity_model_fofr_multigrid_method",
                                      fofr_multigrid_method,
                                      "sqrt",
                                      "The field the multigrid solver solves for. sqrt: u = sqrt(|f_R|) with the "
                                      "equation in a cell solved exactly (a cubic for n = 1), robust for any f_R0, "
                                      "n and redshift. log: ln|f_R| with Newton steps (the original method; can give "
                                      "NaN for small f_R0 or at high redshift)",
                                      {"sqrt", "log"});
                });
            });
            f.for_gravity({"Symmetron"}, [&]() {
                f.optional("gravity_model_symmetron_exact_solution",
                           symmetron_exact_solution,
                           false,
                           "Solve the exact symmetron equation with the multigrid solver (slow, mainly for testing)");
            });
            f.for_gravity({"DGP"}, [&]() {
                f.optional("gravity_model_dgp_exact_solution",
                           dgp_exact_solution,
                           false,
                           "Solve the exact DGP equation with the multigrid solver (slow, mainly for testing; can "
                           "be hard to get to converge)");
            });
            // (a map without gravity_model, e.g. of a unit test: no exact solution)
            auto exact_solution = [](const ParameterMap & p) {
                if (not p.contains("gravity_model"))
                    return false;
                auto model = p.get<std::string>("gravity_model");
                return (model == "f(R)" and p.get<bool>("gravity_model_fofr_exact_solution")) or
                       (model == "Symmetron" and p.get<bool>("gravity_model_symmetron_exact_solution")) or
                       (model == "DGP" and p.get<bool>("gravity_model_dgp_exact_solution"));
            };
            f.when(exact_solution, "solving the exact equation of the gravity model", [&]() {
                f.optional("multigrid_nsweeps",
                           multigrid_nsweeps,
                           3,
                           "Multigrid solver: Newton-Gauss-Seidel sweeps on every level");
                f.optional("multigrid_nsweeps_first_step",
                           multigrid_nsweeps_first_step,
                           20,
                           "Multigrid solver: sweeps on the first step (increase if the solver fails to converge)");
                f.optional("multigrid_solver_residual_convergence",
                           multigrid_solver_residual_convergence,
                           1e-6,
                           "Multigrid solver: converged when the residual is below this");
                f.always_optional("multigrid_convergence_criterion",
                                  multigrid_convergence_criterion,
                                  "residual",
                                  "Multigrid solver: converged when the rms residual is below "
                                  "multigrid_solver_residual_convergence (residual) or below it times the rms of the "
                                  "linear source of the equation (relative; the residual scale changes by orders of "
                                  "magnitude over a simulation)",
                                  {"residual", "relative"});
                f.always_optional("multigrid_coarse_restriction",
                                  multigrid_coarse_restriction,
                                  "sharpk",
                                  "Multigrid solver with multigrid_coarse_factor > 1: the density on the coarse grid "
                                  "is its Fourier modes below the coarse Nyquist frequency (sharpk; delta < -1 set to "
                                  "-1) or a CIC deposit of the force-grid cells (cic; conserves the mass, delta >= -1)",
                                  {"sharpk", "cic"});
                f.always_optional("multigrid_start_redshift",
                                  multigrid_start_redshift,
                                  -1.0,
                                  "Multigrid solver: solve the field equation only below this redshift; above it the "
                                  "fifth force is set by multigrid_force_before_start. < 0: at all redshifts");
                f.always_optional("multigrid_force_before_start",
                                  multigrid_force_before_start,
                                  "linear",
                                  "Multigrid solver with multigrid_start_redshift >= 0: the fifth force above that "
                                  "redshift is the linear one (linear; no screening) or none (GR; the growth factors, "
                                  "IC and COLA frame still have the modified gravity)",
                                  {"linear", "GR"});
                f.optional("multigrid_coarse_factor",
                           multigrid_coarse_factor,
                           1,
                           "Multigrid solver: solve the field equation on a grid coarser by this factor than the "
                           "force grid (the density and the solution go between the grids as their Fourier modes below "
                           "the coarse Nyquist frequency). 1: on the force grid",
                           check::at_least(1));
            });
        }
    };

} // namespace SIMCODE

#endif
