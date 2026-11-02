#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Perturbations/Presets.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace FML::COSMOLOGY {

    using namespace Constants;

    namespace {
        // The parameters of the background of a run: deg_ncdm massive species of mass m_nu
        BACKGROUND::BackgroundParameters background_parameters(const CosmologyParameters & p) {
            BACKGROUND::BackgroundParameters b;
            b.h = p.h;
            b.Omega_b = p.OmegaB;
            b.Omega_cdm = p.OmegaCDM;
            b.Omega_k = p.OmegaK;
            b.T_CMB = p.TCMB;
            b.N_eff = p.Neff;
            if (p.has_massive_nu())
                b.m_nu_eV.assign(p.deg_ncdm, p.m_nu);
            b.T_ncdm_over_TCMB = p.T_ncdm_over_TCMB;
            return b;
        }

        // The ComponentBackground is built on this range unless solve() asks for more
        constexpr double default_x_min = -25.0, default_x_max = 5.0;
        constexpr int default_points_per_unit_x = 200;

        // 5-point central difference, step 1e-3 (error ~ h^4 f^(5) / 30)
        template <class F>
        double derivative(const F & f, double x) {
            const double h = 1e-3;
            return (f(x - 2.0 * h) - 8.0 * f(x - h) + 8.0 * f(x + h) - f(x + 2.0 * h)) / (12.0 * h);
        }
    } // namespace

    // =============================================================================
    // Construction
    // =============================================================================

    BackgroundCosmology::BackgroundCosmology(const CosmologyParameters & params_in)
        : params(params_in), H0(params_in.H0()), OmegaK0(params_in.OmegaK),
          K(-params_in.OmegaK * params_in.H0() * params_in.H0() / (Constants::c * Constants::c)),
          L_unit(Constants::c / params_in.H0() / Constants::c_over_H0_mpch),
          range{default_x_min, default_x_max, default_points_per_unit_x} {
        params.validate();
        build_background();
        find_massive_neutrinos();
    }

    BackgroundCosmology::BackgroundCosmology(const CosmologyParameters & params_in,
                                             PERTURBATIONS::PerturbationModelPtr model_in)
        : params(params_in), H0(params_in.H0()), OmegaK0(params_in.OmegaK),
          K(-params_in.OmegaK * params_in.H0() * params_in.H0() / (Constants::c * Constants::c)),
          L_unit(Constants::c / params_in.H0() / Constants::c_over_H0_mpch), pmodel(std::move(model_in)),
          range{default_x_min, default_x_max, default_points_per_unit_x}, explicit_pmodel(true) {
        params.validate();
        if (!pmodel)
            throw std::invalid_argument("BackgroundCosmology: no perturbation model");
        bg = pmodel->background_ptr();
        // The cosmological parameters must be those of the model's background
        const auto p = background_parameters(params), &q = bg->parameters();
        if (p.h != q.h || p.Omega_b != q.Omega_b || p.Omega_cdm != q.Omega_cdm || p.Omega_k != q.Omega_k ||
            p.T_CMB != q.T_CMB || p.N_eff != q.N_eff || p.m_nu_eV != q.m_nu_eV ||
            (!p.m_nu_eV.empty() && p.T_ncdm_over_TCMB != q.T_ncdm_over_TCMB))
            throw std::invalid_argument("BackgroundCosmology: the parameters are not those of the background of the "
                                        "perturbation model " +
                                        pmodel->name());
        range.x_min = bg->x_min();
        range.x_max = bg->x_max();
        tensor_fe = pmodel->field_equation_perturbations(pmodel->context());
        find_massive_neutrinos();
    }

    BackgroundCosmology::BackgroundCosmology(BackgroundCosmology &&) noexcept = default;

    BackgroundCosmology::~BackgroundCosmology() = default;

    void BackgroundCosmology::build_background() {
        bg.reset();
        pmodel.reset();
        tensor_fe.reset();
        const auto p = background_parameters(params);
        bg = PERTURBATIONS::make_model_background(params.model, p, params.model_params, range);
        pmodel = PERTURBATIONS::make_perturbation_model(params.model, bg, params.model_params);
        tensor_fe = pmodel->field_equation_perturbations(pmodel->context());
    }

    void BackgroundCosmology::find_massive_neutrinos() {
        // Degenerate species (the same mass and temperature) count as one with the sum of their factors
        ncdm = nullptr;
        ncdm_factor = 0.0;
        for (const auto & s : bg->species()) {
            if (s->component() != BACKGROUND::Component::MassiveNeutrinos)
                continue;
            double factor = 1.0;
            const auto * typed = BACKGROUND::as_massive_neutrinos(*s, factor);
            if (!typed)
                throw std::invalid_argument("BackgroundCosmology: massive neutrinos must be a MassiveNeutrinoSpecies");
            if (ncdm && !BACKGROUND::degenerate(*ncdm, *typed))
                throw std::invalid_argument("BackgroundCosmology: only one massive neutrino species (or several "
                                            "degenerate ones) is supported");
            ncdm = typed;
            ncdm_factor += factor;
        }
        if (!ncdm)
            ncdm_factor = 1.0;
    }

    std::string BackgroundCosmology::gravity_name() const { return tensor_fe->name(); }

    TensorCoefficients BackgroundCosmology::tensor_coefficients(double x) const { return tensor_fe->tensor(x); }

    // =============================================================================
    // Species sums: from the ComponentBackground (second derivatives: 5-point differences of the first)
    // =============================================================================

    double BackgroundCosmology::rho_species(double x) const { return bg->rho_total(x); }
    double BackgroundCosmology::drho_species(double x) const { return bg->drho_total(x); }
    double BackgroundCosmology::ddrho_species(double x) const {
        return derivative([&](double xx) { return bg->drho_total(xx); }, x);
    }
    double BackgroundCosmology::rho_of(BACKGROUND::Component c, double x) const { return bg->rho(c, x); }
    double BackgroundCosmology::drho_of(BACKGROUND::Component c, double x) const { return bg->drhodx(c, x); }
    double BackgroundCosmology::ddrho_of(BACKGROUND::Component c, double x) const {
        return derivative([&](double xx) { return bg->drhodx(c, xx); }, x);
    }
    double BackgroundCosmology::p_of(BACKGROUND::Component c, double x) const { return bg->p(c, x); }

    double BackgroundCosmology::pNcdm(double x) const {
        return has_massive_nu() ? p_of(BACKGROUND::Component::MassiveNeutrinos, x) : 0.0;
    }
    double BackgroundCosmology::dpNcdmdx(double x) const { return ncdm ? ncdm_factor * ncdm->dpdx(x) : 0.0; }
    double BackgroundCosmology::pseudo_pNcdm(double x) const { return ncdm ? ncdm_factor * ncdm->pseudo_p(x) : 0.0; }
    double BackgroundCosmology::M_ncdm_of_x(double x) const { return ncdm ? ncdm->M_of_x(x) : 0.0; }
    double BackgroundCosmology::ncdm_prefactor() const { return ncdm ? ncdm_factor * ncdm->prefactor() : 0.0; }

    // =============================================================================
    // Expansion rate, from the ComponentBackground: E and dlnE/dx analytic (species and field equations),
    // d^2 lnE/dx^2 from a 5-point difference of dlnE/dx. With Hp = H0 e^x E:
    //   Hp'  = Hp (1 + dlnE/dx)
    //   Hp'' = Hp [ (1 + dlnE/dx)^2 + d^2lnE/dx^2 ]
    //   E2' = 2 E2 dlnE/dx,  E2'' = E2 [ 4 (dlnE/dx)^2 + 2 d^2lnE/dx^2 ]
    // =============================================================================

    double BackgroundCosmology::E2_of_x(double x) const {
        const double E = bg->E(x);
        return E * E;
    }
    double BackgroundCosmology::dE2dx_of_x(double x) const { return 2.0 * E2_of_x(x) * bg->dlogEdx(x); }
    double BackgroundCosmology::ddE2ddx_of_x(double x) const {
        const double L = bg->dlogEdx(x);
        return E2_of_x(x) * (4.0 * L * L + 2.0 * ddlogEdx(x));
    }
    double BackgroundCosmology::ddlogEdx(double x) const {
        return derivative([&](double xx) { return bg->dlogEdx(xx); }, x);
    }

    double BackgroundCosmology::H_of_x(double x) const { return H0 * bg->E(x); }

    double BackgroundCosmology::Hp_of_x(double x) const { return H0 * std::exp(x) * bg->E(x); }

    double BackgroundCosmology::dHpdx_of_x(double x) const { return Hp_of_x(x) * (1.0 + bg->dlogEdx(x)); }

    double BackgroundCosmology::ddHpddx_of_x(double x) const {
        const double L = 1.0 + bg->dlogEdx(x);
        return Hp_of_x(x) * (L * L + ddlogEdx(x));
    }

    double BackgroundCosmology::OmegaB_of_x(double x) const { return rhoB(x) / E2_of_x(x); }
    double BackgroundCosmology::OmegaCDM_of_x(double x) const { return rhoCDM(x) / E2_of_x(x); }
    double BackgroundCosmology::OmegaGamma_of_x(double x) const { return rhoGamma(x) / E2_of_x(x); }
    double BackgroundCosmology::OmegaNu_of_x(double x) const { return rhoNu(x) / E2_of_x(x); }
    double BackgroundCosmology::OmegaNcdm_of_x(double x) const { return rhoNcdm(x) / E2_of_x(x); }
    double BackgroundCosmology::OmegaK_of_x(double x) const { return rhoK(x) / E2_of_x(x); }
    double BackgroundCosmology::OmegaDE_of_x(double x) const { return rhoDE(x) / E2_of_x(x); }

    double BackgroundCosmology::TCMB_of_x(double x) const { return params.TCMB * std::exp(-x); }

    // =============================================================================
    // Times and distances: tabulated by the ComponentBackground (Gauss-Legendre integrals of 1/(aE), 1/E)
    // =============================================================================

    void BackgroundCosmology::solve(double x_start_in, double x_end_in, int n_points) {
        Utils::Timer timer("BackgroundCosmology::solve");
        if (!(x_end_in > x_start_in))
            throw std::invalid_argument("BackgroundCosmology::solve: need x_end > x_start");
        const int ppx = std::max(default_points_per_unit_x, int(std::ceil(n_points / (x_end_in - x_start_in))));
        if (explicit_pmodel) {
            if (x_start_in < bg->x_min() || x_end_in > bg->x_max())
                throw std::invalid_argument("BackgroundCosmology::solve: [" + std::to_string(x_start_in) + ", " +
                                            std::to_string(x_end_in) + "] outside the range of the background of " +
                                            "the perturbation model");
        } else if (x_start_in < bg->x_min() || x_end_in > bg->x_max() || ppx > range.points_per_unit_x) {
            range.x_min = std::min(x_start_in, default_x_min);
            range.x_max = std::max(x_end_in, default_x_max);
            range.points_per_unit_x = ppx;
            build_background();
            find_massive_neutrinos();
        }
        x_start = x_start_in;
        x_end = x_end_in;
        solved = true;
        timer.print();
    }

    void BackgroundCosmology::require_solved(const char * func) const {
        if (!solved)
            throw std::runtime_error(std::string("BackgroundCosmology::") + func + ": call solve() first");
    }

    void BackgroundCosmology::require_in_range(const char * func, double x) const {
        const double tol = 1e-10 * (1.0 + std::fabs(x));
        if (x < x_start - tol || x > x_end + tol)
            throw std::out_of_range(std::string("BackgroundCosmology::") + func + ": x = " + std::to_string(x) +
                                    " outside the solved range [" + std::to_string(x_start) + ", " +
                                    std::to_string(x_end) + "] (bg_x_start, bg_x_end)");
    }

    double BackgroundCosmology::eta_of_x(double x) const {
        require_solved("eta_of_x");
        require_in_range("eta_of_x", x);
        return bg->eta(x) * L_unit;
    }

    double BackgroundCosmology::detadx_of_x(double x) const { return c / Hp_of_x(x); }

    double BackgroundCosmology::t_of_x(double x) const {
        require_solved("t_of_x");
        require_in_range("t_of_x", x);
        return bg->t_H0(x) / H0;
    }

    double BackgroundCosmology::eta0() const { return eta_of_x(0.0); }
    double BackgroundCosmology::t0() const { return t_of_x(0.0); }

    double BackgroundCosmology::comoving_distance_of_x(double x) const { return eta0() - eta_of_x(x); }

    double BackgroundCosmology::sinK(double chi) const { return bg->sinK(chi / L_unit) * L_unit; }

    double BackgroundCosmology::cotK(double chi) const { return bg->cotK(chi / L_unit) / L_unit; }

    double BackgroundCosmology::transverse_distance_of_x(double x) const { return sinK(comoving_distance_of_x(x)); }

    double BackgroundCosmology::angular_distance_of_x(double x) const {
        return std::exp(x) * transverse_distance_of_x(x);
    }

    double BackgroundCosmology::luminosity_distance_of_x(double x) const {
        return transverse_distance_of_x(x) * std::exp(-x);
    }

    // =============================================================================
    // Characteristic epochs
    // =============================================================================

    namespace {
        // Root of f on [a,b], or NaN if there is no sign change
        double root_or_nan(const std::function<double(double)> & f, double a, double b) {
            if (f(a) * f(b) > 0.0)
                return std::numeric_limits<double>::quiet_NaN();
            return Utils::find_root(f, a, b);
        }
    } // namespace

    double BackgroundCosmology::x_matter_radiation_equality() const {
        // Massive neutrinos split as in CLASS: 3 p_ncdm is radiation, rho_ncdm - 3 p_ncdm matter (they were left out)
        return root_or_nan(
            [&](double x) {
                const double r3p = 3.0 * pNcdm(x);
                return std::log((rhoB(x) + rhoCDM(x) + (rhoNcdm(x) - r3p)) / (rhoGamma(x) + rhoNu(x) + r3p));
            },
            x_start,
            x_end);
    }

    double BackgroundCosmology::x_matter_DE_equality() const {
        return root_or_nan([&](double x) { return (rhoB(x) + rhoCDM(x)) - rhoDE(x); }, std::max(x_start, -10.0), x_end);
    }

    double BackgroundCosmology::x_acceleration_onset() const {
        return root_or_nan([&](double x) { return dHpdx_of_x(x) / Hp_of_x(x); }, std::max(x_start, -10.0), x_end);
    }

    // =============================================================================
    // Output
    // =============================================================================

    void BackgroundCosmology::info() const {
        const auto f = std::cout.flags();
        std::cout << std::setprecision(6);
        std::cout << "\nBackgroundCosmology: model " << (explicit_pmodel ? pmodel->name() : params.model)
                  << ", gravity " << bg->field_equations().info() << ", perturbations " << gravity_name() << "\n";
        std::cout << "  Species: ";
        for (size_t i = 0; i < bg->species().size(); i++)
            std::cout << (i ? "\n           " : "") << bg->species()[i]->info();
        std::cout << "\n";
        std::cout << "  E^2(x=0) (should be 1)   = " << E2_of_x(0.0) << "\n";
        std::cout << "  OmegaB, CDM, DE (today)  = " << OmegaB_of_x(0.0) << ", " << OmegaCDM_of_x(0.0) << ", "
                  << OmegaDE_of_x(0.0) << "\n";
        if (has_massive_nu())
            std::cout << "  OmegaNcdm (today)        = " << OmegaNcdm_of_x(0.0) << "\n";
        std::cout << "  OmegaGamma, Nu, K        = " << OmegaGamma_of_x(0.0) << ", " << OmegaNu_of_x(0.0) << ", "
                  << OmegaK_of_x(0.0) << "\n";
        if (K != 0.0) {
            std::cout << "  Curvature radius 1/sqrt|K| = " << 1.0 / std::sqrt(std::fabs(K)) / Mpc << " Mpc ("
                      << (K > 0.0 ? "closed" : "open") << ")";
            if (solved)
                std::cout << ", sqrt|K| eta0 = " << std::sqrt(std::fabs(K)) * eta0();
            std::cout << "\n";
        }

        auto print_epoch = [&](const char * name, double x) {
            std::cout << "  " << std::left << std::setw(25) << name << "x = " << std::setw(11) << x
                      << "  z = " << std::setw(11) << std::exp(-x) - 1.0;
            if (solved && std::isfinite(x))
                std::cout << "  t = " << t_of_x(x) / yr << " yr";
            std::cout << "\n";
        };
        print_epoch("Matter-radiation eq.:", x_matter_radiation_equality());
        print_epoch("Matter-DE equality:", x_matter_DE_equality());
        print_epoch("Acceleration onset:", x_acceleration_onset());

        if (solved) {
            std::cout << "  Age t0                   = " << t0() / Gyr << " Gyr\n";
            std::cout << "  Conformal time eta0      = " << eta0() / Mpc << " Mpc  (eta0/c = " << eta0() / c / Gyr
                      << " Gyr)\n";
        }
        std::cout << "\n";
        std::cout.flags(f);
    }

    void BackgroundCosmology::output(const std::string & filename, double x_min, double x_max, int n_points) const {
        require_solved("output");
        if (x_min == 0.0 && x_max == 0.0) {
            x_min = x_start;
            x_max = x_end;
        }

        std::ofstream fp(filename);
        if (!fp)
            throw std::runtime_error("BackgroundCosmology::output: cannot open " + filename);

        const double H0unit = H0_over_h; // 100 km/s/Mpc
        fp << "# Background cosmology. Units: eta, chi, d_A, d_L in Mpc; t in Gyr; H, Hp, dHp, ddHp in (100 km/s/Mpc)\n"
           << "# x  eta  t  H  Hp  dHpdx  ddHpddx  OmegaB  OmegaCDM  OmegaGamma  OmegaNu  OmegaK  OmegaDE"
              "  chi  d_A  d_L  OmegaNcdm  w_ncdm\n";
        fp << std::scientific << std::setprecision(10);
        for (double x : Utils::linspace(x_min, x_max, n_points)) {
            fp << x << " " << eta_of_x(x) / Mpc << " " << t_of_x(x) / Gyr << " " << H_of_x(x) / H0unit << " "
               << Hp_of_x(x) / H0unit << " " << dHpdx_of_x(x) / H0unit << " " << ddHpddx_of_x(x) / H0unit << " "
               << OmegaB_of_x(x) << " " << OmegaCDM_of_x(x) << " " << OmegaGamma_of_x(x) << " " << OmegaNu_of_x(x)
               << " " << OmegaK_of_x(x) << " " << OmegaDE_of_x(x) << " " << comoving_distance_of_x(x) / Mpc << " "
               << angular_distance_of_x(x) / Mpc << " " << luminosity_distance_of_x(x) / Mpc << " " << OmegaNcdm_of_x(x)
               << " " << (has_massive_nu() ? pNcdm(x) / rhoNcdm(x) : 0.0) << "\n";
        }
    }

} // namespace FML::COSMOLOGY
