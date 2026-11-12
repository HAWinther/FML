#ifndef GRAVITYMODEL_HEADER
#define GRAVITYMODEL_HEADER

#include "Cosmology.h"
#include "GravityModelSettings.h"
#include "ParameterDefaults.h"
#include "ShortRangeForce.h"

#include "TransferInput.h"
#include <FML/Cosmology/Growth/LPTGrowthFactors.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Spline/Spline.h>
#include <FML/Timing/Timings.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

/// Base class for gravity models. A model provides its quasi-static theory (QuasiStaticGravity, built in init()) or
/// overrides GeffOverG; the defaults here then give G_eff/G = mu(a, k) of the theory (1 without one: GR), the force
/// from 1.5 OmegaM a G_eff/G delta (k-dependent if the theory is) and scale-dependent growth if mu depends on k or
/// there are massive neutrinos. Screened models (GravityModel_Screened.h) override compute_force.
template <int NDIM>
class GravityModel {
  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;
    using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
    using ParameterMap = FML::UTILS::ParameterMap;
    using DVector = FML::INTERPOLATION::SPLINE::DVector;
    using DVector2D = FML::INTERPOLATION::SPLINE::DVector2D;
    using TransferProvider = SIMCODE::TransferProvider;
    static constexpr double H0_hmpc = 1.0 / FML::UTILS::SI::c_over_H0_mpch;

    //========================================================================
    // Constructors
    //========================================================================
    GravityModel(std::shared_ptr<Cosmology> cosmo, std::string name) : cosmo(cosmo), name(name) {}

    //========================================================================
    // Name of the model
    //========================================================================
    std::string get_name() const { return name; }
    bool uses_finite_difference_force() const { return force_use_finite_difference_force; }

    std::shared_ptr<Cosmology> get_cosmo() { return cosmo; }

    //========================================================================
    // The theory's quasi-static limit (mu, Sigma, the second-order source) shared with the Einstein-Boltzmann solver
    // (FML/Cosmology/Perturbations/QuasiStaticGravity.h), built on the cosmology's background in init() by the models
    // that have one (f(R), DGP, JBD, mu with the DE parametrisation); nullptr otherwise
    //========================================================================
    using QuasiStaticGravity = FML::COSMOLOGY::PERTURBATIONS::QuasiStaticGravity;
    std::shared_ptr<const QuasiStaticGravity> get_quasi_static_gravity() const { return quasi_static; }

    //========================================================================
    // Growth functions
    //========================================================================
    // The growth factors (FML/Cosmology/Growth/LPTGrowthFactors.h; k below 1e-4 h/Mpc is the value there)
    double get_D_1LPT(double a, double koverH0 = 0.0) const { return lpt_growth().D1(std::log(a), koverH0); }
    double get_Dmnu_1LPT(double a, double koverH0 = 0.0) const {
        if (not(scaledependent_growth and has_neutrino_transfers()))
            return get_D_1LPT(a, koverH0);
        koverH0 = std::max(koverH0, koverH0low);
        return Dmnu_1LPT_of_logkoverH0_loga(std::log(koverH0), std::log(a));
    }
    double get_D_2LPT(double a, double koverH0 = 0.0) const { return lpt_growth().D2(std::log(a), koverH0); }
    double get_D_3LPTa(double a, double koverH0 = 0.0) const { return lpt_growth().D3a(std::log(a), koverH0); }
    double get_D_3LPTb(double a, double koverH0 = 0.0) const { return lpt_growth().D3b(std::log(a), koverH0); }

    //========================================================================
    // Is growth scaledpendent?
    //========================================================================
    bool is_growth_scaledependent() const { return scaledependent_growth; }

    //========================================================================
    // Growth rates
    //========================================================================
    double get_f_1LPT(double a, double koverH0 = 0.0) const { return lpt_growth().f1(std::log(a), koverH0); }
    double get_f_2LPT(double a, double koverH0 = 0.0) const { return lpt_growth().f2(std::log(a), koverH0); }
    double get_f_3LPTa(double a, double koverH0 = 0.0) const { return lpt_growth().f3a(std::log(a), koverH0); }
    double get_f_3LPTb(double a, double koverH0 = 0.0) const { return lpt_growth().f3b(std::log(a), koverH0); }

    /// The growth factors (built in init)
    using LPTGrowthFactors = FML::COSMOLOGY::GROWTH::LPTGrowthFactors;
    const LPTGrowthFactors & lpt_growth() const {
        if (not growth)
            throw std::runtime_error("GravityModel [" + name + "]: the growth factors are computed in init()");
        return *growth;
    }

    //========================================================================
    // Output the stuff we compute: the columns (name and value at (a, k c / H0)) of the table of output()
    //========================================================================
    struct OutputColumn {
        std::string name;
        std::function<double(double, double)> value;
    };
    virtual std::vector<OutputColumn> output_columns() const {
        std::vector<OutputColumn> columns{{"a", [](double a, double) { return a; }},
                                          {"GeffG(a,k)", [this](double a, double k) { return GeffOverG(a, k); }},
                                          {"D1(a,k)", [this](double a, double k) { return get_D_1LPT(a, k); }},
                                          {"D1mnu(a,k)", [this](double a, double k) { return get_Dmnu_1LPT(a, k); }}};
        if (transfers) {
            // D1_Tk = T_cb(k, a) / T_cb(k, aini); D1mnu_Tk the same for the neutrinos (NaN without massive neutrinos)
            auto transfer_growth = [this](SIMCODE::TransferSpecies s, double a, double koverH0) {
                const double k = koverH0 * H0_hmpc;
                return transfers->density_over_k_mpc2(s, k, a) / transfers->density_over_k_mpc2(s, k, aini);
            };
            columns.push_back({"D1_Tk(a,k)", [transfer_growth](double a, double k) {
                                   return transfer_growth(SIMCODE::TransferSpecies::CDMBaryon, a, k);
                               }});
            columns.push_back({"D1mnu_Tk(a,k)", [this, transfer_growth](double a, double k) {
                                   return has_neutrino_transfers() ?
                                              transfer_growth(SIMCODE::TransferSpecies::MassiveNeutrino, a, k) :
                                              std::numeric_limits<double>::quiet_NaN();
                               }});
        }
        columns.push_back({"D2(a,k)", [this](double a, double k) { return get_D_2LPT(a, k); }});
        columns.push_back({"D3a(a,k)", [this](double a, double k) { return get_D_3LPTa(a, k); }});
        columns.push_back({"D3b(a,k)", [this](double a, double k) { return get_D_3LPTb(a, k); }});
        columns.push_back({"source_1LPT", [this](double a, double k) { return source_factor_1LPT(a, k); }});
        columns.push_back({"source_2LPT", [this](double a, double k) { return source_factor_2LPT(a, k); }});
        columns.push_back({"source_3LPTa", [this](double a, double k) { return source_factor_3LPTa(a, k); }});
        columns.push_back({"source_3LPTb", [this](double a, double k) { return source_factor_3LPTb(a, k); }});
        return columns;
    }

    virtual void output(std::string filename, double koverH0) {
        std::ofstream fp(filename.c_str());
        if (not fp.is_open())
            return;

        const auto columns = output_columns();
        fp << "#";
        for (size_t i = 0; i < columns.size(); i++)
            fp << (i > 0 ? " " : "") << std::setw(15) << columns[i].name;
        fp << "\n";
        for (int i = 0; i < npts_loga; i++) {
            double loga = std::log(alow) + std::log(ahigh / alow) * i / double(npts_loga - 1);
            double a = std::exp(loga);
            for (auto & column : columns)
                fp << " " << std::setw(15) << column.value(a, koverH0);
            fp << "\n";
        }
    }

    //========================================================================
    // Initialize
    // In derived class remember to also call the base class (this) unless
    // you solve for the growth factors independently
    //========================================================================
    virtual void init() {
        if (geff_scale_dependent())
            scaledependent_growth = true;
        compute_growth_factors();
    }

    //========================================================================
    // Show some info
    //========================================================================
    void info() const {
        if (FML::ThisTask == 0) {
            std::cout << "#=====================================================\n";
            std::cout << "# GravityModel is [" << name << "] and has the cosmology: [" << this->cosmo->get_name()
                      << "]\n";
            std::cout << "# Growth factors unity at zini = " << 1.0 / aini - 1.0 << "\n";
            std::cout << "# Starting integrating growth factors from z = " << 1.0 / alow - 1.0 << "\n";
            std::cout << "# scaledependent_growth : " << scaledependent_growth << "\n";
            for (const auto & [name, value] : settings_printed)
                std::cout << "# " << std::left << std::setw(49) << name << ": " << value << "\n" << std::right;
        }
        info_model();
        if (FML::ThisTask == 0) {
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }
    /// The model's own lines of info() (what follows from its parameters, which info() prints; task 0 only)
    virtual void info_model() const {}

    /// A fitted nonlinear P(k) boost over GR, P_model(k, z) / P_GR(k, z) with the same initial conditions (k in h/Mpc),
    /// for the nonlinear predictions in the P(k) outputs (Simulation::nonlinear_over_linear_prediction). Empty (the
    /// default) if the model has none
    virtual std::function<double(double)> nonlinear_boost_over_GR([[maybe_unused]] double redshift) const { return {}; }

    //========================================================================
    // The transfer functions (shared with the simulation: see TransferInput.h)
    //========================================================================
    std::shared_ptr<TransferProvider> get_transfers() { return transfers; }
    void set_transfers(std::shared_ptr<TransferProvider> _transfers) { transfers = _transfers; }

    //========================================================================
    // Read the parameters we need
    // In derived class remember to also call the base class (this)
    //========================================================================
    virtual void read_parameters(ParameterMap & param) {
        // The gravity_model_* and multigrid_* parameters (GravityModelSettings::fields)
        // (name is the model's name in gravity_models(), Models.h: the value of gravity_model)
        SIMCODE::SettingsReader reader{param, "", name};
        settings.fields(reader);
        settings_printed = reader.printed;
        aini = 1.0 / (1.0 + param.get<double>("ic_initial_redshift"));
        // The growth factor splines must start before the simulation (e.g. when starting at very high redshift)
        alow = std::min(alow, aini / 10.0);
        // The transfer functions (CAMB, CLASS or the internal solver) are loaded once by the caller and given with
        // set_transfers before this (Main.cpp; the same object is used by the initial conditions and the analysis).
        // The growth factors with massive neutrinos are computed from them (and start at their highest redshift)
        this->scaledependent_growth = this->cosmo->get_fMNu() > 0.0;
        FML::assert_mpi(not(this->cosmo->get_OmegaMNu() > 0.0 and not transfers),
                        "Massive neutrinos need ic_type_of_input = camb, class, internal or tables (the growth factors "
                        "with massive neutrinos are computed from the transfer functions)");

        // Experimental option with finite-difference forces (instead of fourier)
        force_use_finite_difference_force = SIMCODE::get_parameter<bool>(param, "force_use_finite_difference_force");
        force_finite_difference_stencil_order =
            SIMCODE::get_parameter<int>(param, "force_finite_difference_stencil_order");
    }

    //========================================================================
    // Effective gravitational constant and equivalent factors appearing
    // in the LPT growth equations
    //========================================================================
    /// G_eff/G = mu(a, k) of the theory (koverH0 = c k / H0), 1 without one (GR). The G_eff/G of the CDM+baryon
    /// particles (one fluid): with a coupling of the CDM alone (GravityModel_CoupledQuintessence) the momentum-weighted
    /// force of the fluid
    virtual double GeffOverG(double a, double koverH0 = 0) const {
        return quasi_static ? quasi_static->mu(std::log(a), koverH0) : 1.0;
    }
    /// G_eff/G of gravity alone: GeffOverG without a coupling of the CDM alone (the force on the baryons with CDM and
    /// baryons as separate particles)
    virtual double GeffOverG_gravity(double a, double koverH0 = 0) const { return GeffOverG(a, koverH0); }
    /// A force of the CDM on the CDM alone, relative to gravity (coupled quintessence: 2 beta^2; none by default).
    /// With CDM and baryons as separate particles the CDM gets it from the CDM density in a second kick
    /// (NBodySimulation::kick_cdm_fifth_force); with one set of particles it is in GeffOverG
    virtual bool has_cdm_fifth_force() const { return false; }
    virtual double cdm_fifth_force_coupling([[maybe_unused]] double a) const { return 0.0; }

    //========================================================================
    // Velocity friction: dv/dlna = -Gamma(a) v + force in the code velocity v = a^3 E dx/da. In the growth equations
    // (all LPT orders, COLA, the IC velocities) and the kick and drift factors (TimeStepping.h). By default that of
    // the cosmology (Cosmology::velocity_friction: dark scattering); a gravity model with an extra velocity term
    // overrides them (coupled quintessence)
    //========================================================================
    virtual bool has_velocity_friction() const { return cosmo->has_velocity_friction(); }
    /// The friction of the CDM+baryon particles as one fluid
    virtual double velocity_friction(double a) const { return cosmo->velocity_friction(a); }
    /// The friction of the CDM alone (CDM and baryons as separate particles; the baryons have none)
    virtual double velocity_friction_cdm(double a) const { return cosmo->velocity_friction_cdm(a); }
    /// The friction integral I(a) = int_amin^a Gamma dlna (a spline in log a on [amin, amax], constant outside), for
    /// the time stepping. An empty function without friction. cdm_only: of velocity_friction_cdm, else of
    /// velocity_friction
    std::function<double(double)>
    velocity_friction_integral(double amin = 1e-6, double amax = 100.0, bool cdm_only = false) const {
        if (not has_velocity_friction())
            return {};
        const int n = 4000;
        std::vector<double> x(n);
        for (int i = 0; i < n; i++)
            x[i] = std::log(amin) + std::log(amax / amin) * i / double(n - 1);
        FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double xx, [[maybe_unused]] const double * y, double * dydx) {
            dydx[0] = cdm_only ? velocity_friction_cdm(std::exp(xx)) : velocity_friction(std::exp(xx));
            return GSL_SUCCESS;
        };
        FML::SOLVERS::ODESOLVER::ODESolver ode;
        std::vector<double> ini{0.0};
        ode.solve(deriv, x, ini);
        auto spline = std::make_shared<const FML::INTERPOLATION::SPLINE::Spline>(
            x, ode.get_data_by_component(0), "Friction integral I(log(a))");
        return [spline](double a) { return (*spline)(std::log(a)); };
    }
    /// Whether G_eff/G depends on k (then the force and the growth do too)
    virtual bool geff_scale_dependent() const { return quasi_static and quasi_static->scale_dependent(); }

    // Defaults for new models: sample their existing Geff function at a large finite k.
    // Units of k are H0/c, as in GeffOverG. Override for an exact asymptote when known.
    static constexpr double short_range_fiducial_koverH0 = 1e12;
    virtual double GeffOverG_highk(double a) const { return GeffOverG(a, short_range_fiducial_koverH0); }
    // Comoving Mpc/h. The zero sentinel denotes an infinite-range surrogate.
    virtual double short_range_comoving_range(double /*a*/) const { return 0.; }
    virtual bool short_range_screening_active() const { return false; }
    void validate_short_range_support() const {
        if (NDIM != 3 || uses_finite_difference_force() || short_range_screening_active())
            throw std::invalid_argument("force_short_range requires 3D continuous Fourier forces with screening and "
                                        "exact nonlinear solvers disabled");
    }
    SIMCODE::ShortRangeInteraction short_range_interaction(double a) const {
        if (!(a > 0) || !std::isfinite(a))
            throw std::invalid_argument("Invalid short-range force epoch");
        SIMCODE::ShortRangeInteraction model{GeffOverG(a, 0), GeffOverG_highk(a), short_range_comoving_range(a)};
        model.validate();
        return model;
    }
    // Coordinate errors and parameter consistency before any force communication.
    SIMCODE::ShortRangeInteraction checked_short_range_interaction(double a) const {
        SIMCODE::ShortRangeInteraction model;
        int valid = 1;
        try {
            validate_short_range_support();
            model = short_range_interaction(a);
        } catch (const std::exception &) {
            valid = 0;
        }
        FML::MinOverTasks(&valid);
        if (!valid)
            throw std::invalid_argument("Invalid/unsupported short-range model on at least one rank");
        double low[]{a, model.mu0, model.muinf, model.range_mpch}, high[4];
        std::copy_n(low, 4, high);
        FML::MinArrayOverTasks(low, 4);
        FML::MaxArrayOverTasks(high, 4);
        if (!std::equal(low, low + 4, high))
            throw std::invalid_argument("Short-range model or epoch differs across ranks");
        return model;
    }
    template <class P>
    void kick_short_range(FML::PARTICLE::MPIParticles<P> & particles,
                          const SIMCODE::ShortRangeSettings & settings,
                          double a,
                          double boxsize,
                          double dt) const {
        if (!settings.enabled || dt == 0)
            return;
        SIMCODE::kick_short_range(particles,
                                  settings,
                                  boxsize,
                                  1.5 * cosmo->get_OmegaM() * cosmo->clustering_mass_factor(a) * a,
                                  dt,
                                  checked_short_range_interaction(a));
    }

    // Factors in the LPT equations
    virtual double source_factor_1LPT([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        double factor = 1.0;
        const double fnu = f_nu_of_a(a);
        if (fnu > 0.0 and koverH0 > 0.0) {
            if (transfers) {
                const double T_nu =
                    transfers->density_over_k_mpc2(SIMCODE::TransferSpecies::MassiveNeutrino, koverH0 * H0_hmpc, a);
                const double T_cb =
                    transfers->density_over_k_mpc2(SIMCODE::TransferSpecies::CDMBaryon, koverH0 * H0_hmpc, a);
                factor = (1.0 - fnu) + fnu * T_nu / T_cb;
            }
        }
        return factor;
    };
    virtual double source_factor_2LPT([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0 - f_nu_of_a(a);
    };
    // The D1^2 source of the 2LPT equation relative to the D2 term (1 in GR):
    //   D2'' + (2 + dlogH/dloga) D2' = rhs source_factor_2LPT (D2 - nonlinear_source_factor_2LPT D1^2)
    // with rhs = 1.5 OmegaM(a) GeffOverG. In DGP the nonlinear (brane-bending) term only adds to the source, it does
    // not multiply D2
    virtual double nonlinear_source_factor_2LPT([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0;
    };
    virtual double source_factor_3LPTa([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0 - f_nu_of_a(a);
    };
    virtual double source_factor_3LPTb([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0 - f_nu_of_a(a);
    };

    /// The PM force with or without the fifth force of the CDM (has_cdm_fifth_force): without it (gravity alone) with
    /// CDM and baryons as separate particles, where the CDM gets it from the CDM density in a second kick
    /// (NBodySimulation::kick_cdm_fifth_force). G_eff/G of the PM force
    bool cdm_coupling_in_force{true};
    void set_cdm_coupling_in_force(bool in_force) { cdm_coupling_in_force = in_force; }
    double GeffOverG_force(double a, double koverH0 = 0) const {
        return cdm_coupling_in_force ? GeffOverG(a, koverH0) : GeffOverG_gravity(a, koverH0);
    }

    /// The force D Phi from the density field (Fourier space): D^2 Phi = 1.5 OmegaM a G_eff/G(a, k) delta, with
    /// G_eff/G(a, k) multiplying delta(k) if it depends on k (and OmegaM times the clustering_mass_factor(a) of a
    /// varying CDM mass). Screened models override it
    virtual void compute_force(double a,
                               double H0Box,
                               FFTWGrid<NDIM> & density_fourier,
                               std::string density_assignment_method_used,
                               std::array<FFTWGrid<NDIM>, NDIM> & force_real,
                               const FML::NBODY::ForceOptions & options = {}) const {
        const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * this->cosmo->clustering_mass_factor(a) * a;
        if (not geff_scale_dependent()) {
            compute_force_from_total_density(density_fourier,
                                             norm_poisson_equation * GeffOverG_force(a),
                                             density_assignment_method_used,
                                             force_real,
                                             options);
            return;
        }
        // G_eff/G(a, k) delta(k) (kBox = k Box, koverH0 = kBox / H0Box), in a grid kept between steps
        FFTWGrid<NDIM> & density_geff = density_workspace;
        FML::NBODY::compute_delta_fifth_force<NDIM>(density_fourier, density_geff, [&](double kBox) {
            return GeffOverG_force(a, kBox / H0Box);
        });
        compute_force_from_total_density(density_geff,
                                         norm_poisson_equation,
                                         density_assignment_method_used,
                                         force_real,
                                         options);
    }

    // Model-owned split. The resolved remainder keeps arbitrary future Geff(k,a) models intact:
    // mu_long = mu_true - mu_pair + mu_pair_long. For an exact Yukawa model the remainder is zero.
    void compute_force_for_step(double a,
                                double H0Box,
                                FFTWGrid<NDIM> & density,
                                const std::string & assignment,
                                std::array<FFTWGrid<NDIM>, NDIM> & force,
                                const FML::NBODY::ForceOptions & options,
                                const SIMCODE::ShortRangeSettings & settings) const {
        if (!settings.enabled) {
            compute_force(a, H0Box, density, assignment, force, options);
            return;
        }
        const auto model = checked_short_range_interaction(a);
        double boxsize = 0;
        int valid = 1;
        std::optional<FML::SHORT_RANGE_GRAVITY::Kernel> storage;
        try {
            if (!(H0Box > 0) || !std::isfinite(H0Box))
                throw std::invalid_argument("Invalid box scale");
            boxsize = H0Box / H0_hmpc;
            auto filter_settings = settings;
            filter_settings.table_points = 0;
            storage.emplace(model.kernel(filter_settings, boxsize));
        } catch (const std::exception &) {
            valid = 0;
        }
        FML::MinOverTasks(&valid);
        if (!valid)
            throw std::invalid_argument("Invalid short-range force kernel on at least one rank");
        const auto & kernel = *storage;
        FML::NBODY::compute_delta_fifth_force<NDIM>(density, density_workspace, [&](double kbox) {
            const double k2 = kbox * kbox;
            return GeffOverG_force(a, kbox / H0Box) - model.fourier_multiplier(k2, boxsize) +
                   model.prefactor() * kernel.long_range_filter(k2);
        });
        compute_force_from_total_density(
            density_workspace, 1.5 * cosmo->get_OmegaM() * cosmo->clustering_mass_factor(a) * a, assignment, force, options);
    }

    //========================================================================
    // The force D Phi from the (total) density that sources it, D^2 Phi = norm_poisson_equation * density:
    // with finite differences of the potential (force_use_finite_difference_force) or in Fourier space.
    // Shared by all gravity models.
    //========================================================================
    void compute_force_from_total_density(FFTWGrid<NDIM> & density_fourier,
                                          double norm_poisson_equation,
                                          const std::string & density_assignment_method_used,
                                          std::array<FFTWGrid<NDIM>, NDIM> & force_real,
                                          const FML::NBODY::ForceOptions & options = {}) const {
        if (force_use_finite_difference_force) {
            // A stencil of order 2, 4 or 6 (phi(i+/-1), phi(i+/-2), ...) needs stencil_order/2 boundary slices
            const int stencil_order = force_finite_difference_stencil_order;
            const int nboundary_cells = stencil_order / 2;
            FFTWGrid<NDIM> potential_real;
            FML::NBODY::compute_potential_real_from_density_fourier<NDIM>(density_fourier,
                                                                          potential_real,
                                                                          norm_poisson_equation,
                                                                          nboundary_cells,
                                                                          options);
            FML::NBODY::compute_force_from_potential_real<NDIM>(potential_real,
                                                                force_real,
                                                                density_assignment_method_used,
                                                                stencil_order,
                                                                options);
        } else {
            FML::NBODY::compute_force_from_density_fourier<NDIM>(density_fourier,
                                                                 force_real,
                                                                 density_assignment_method_used,
                                                                 norm_poisson_equation,
                                                                 options);
        }
    }

    //========================================================================
    // Compute the growth factors: the LPT growth equations with G_eff/G = GeffOverG, the source factors and the
    // nonlinear 2LPT factor of the model (FML/Cosmology/Growth/LPTGrowthFactors.h). The matter density in the
    // source is the actual rho_m(a) of the background (massive neutrinos with their energy density, relativistic at
    // early times); with massive neutrinos the CDM+baryon source is (1 - f_nu(a)) + f_nu(a) T_nu/T_cb
    // (source_factor_1LPT, f_nu(a) = rho_nu(a)/rho_m(a)) from the transfer functions, so (1 - f_nu(a)) rho_m(a) is the
    // CDM+baryon density, and the initial growth rate is the CDM+baryon one of the transfer functions. Normalized to
    // D1 = 1 at aini.
    //========================================================================
    virtual void compute_growth_factors() {
        // With massive neutrinos the transfer functions are needed: we cannot start earlier than they go
        const bool nu_growth = scaledependent_growth and cosmo->get_fMNu() > 0.0;
        if (nu_growth) {
            const double zstart = SIMCODE::transfer_zmax(*transfers);
            const double astart = 1.0 / (1.0 + zstart);
            if (alow < astart)
                alow = astart;
            if (aini < astart)
                throw std::runtime_error(
                    "Transferdata provided are only up to z = " + std::to_string(zstart) +
                    ", but we want to start simulation at z = " + std::to_string(1.0 / aini - 1.0) +
                    " Provide the required data, put OmegaMNu = 0.0 or set it so that we don't use scaledependent "
                    "growth!");
        }
        const bool use_neutrino_transfer = transfers and nu_growth;

        // f_nu(x) on a fine grid covering the integration (LPTGrowthFactors may start up to ~1e5 earlier than alow in
        // the radiation era; outside the range f_nu is evaluated directly)
        if (cosmo->get_fMNu() > 0.0) {
            const int n = 4000;
            f_nu_xmin = std::log(alow) - std::log(1e5);
            f_nu_xmax = std::log(ahigh);
            std::vector<double> x(n), f(n);
            for (int i = 0; i < n; i++) {
                x[i] = f_nu_xmin + (f_nu_xmax - f_nu_xmin) * i / double(n - 1);
                f[i] = cosmo->f_nu(std::exp(x[i]));
            }
            f_nu_spline = FML::INTERPOLATION::SPLINE::Spline(x, f, "f_nu(x)");
        }

        namespace GROWTH = FML::COSMOLOGY::GROWTH;
        auto eq = GROWTH::LPTGrowthEquations::from_background(cosmo->background());
        eq.mu = [this](double x, double k) { return GeffOverG(std::exp(x), k); };
        eq.second_order_source = [this](double x, double k) { return nonlinear_source_factor_2LPT(std::exp(x), k); };
        eq.source_1 = [this](double x, double k) { return source_factor_1LPT(std::exp(x), k); };
        eq.source_2 = [this](double x, double k) { return source_factor_2LPT(std::exp(x), k); };
        eq.source_3a = [this](double x, double k) { return source_factor_3LPTa(std::exp(x), k); };
        eq.source_3b = [this](double x, double k) { return source_factor_3LPTb(std::exp(x), k); };
        if (has_velocity_friction())
            eq.friction = [this](double x) { return velocity_friction(std::exp(x)); };
        if (use_neutrino_transfer)
            eq.initial_rate_1 = [this](double x, double k) {
                return transfers->density_growth_rate(SIMCODE::TransferSpecies::CDMBaryon, k * H0_hmpc, std::exp(x));
            };

        GROWTH::LPTGrowthSettings gs;
        gs.a_min = alow;
        gs.a_max = ahigh;
        gs.n_a = npts_loga;
        gs.a_normalize = aini;
        // Deep in the radiation era (not possible when we need the transfer data for massive neutrinos)
        gs.start_in_radiation_era = not nu_growth;
        gs.scale_dependent = scaledependent_growth;
        gs.koverH0_min = koverH0low;
        gs.koverH0_max = koverH0high;
        gs.n_k = npts_logk;
        growth = std::make_shared<const LPTGrowthFactors>(eq, gs);
        alow = growth->a_min();

        // The massive neutrino growth D1mnu(k, a) = T_nu(k, a) / T_cb(k, aini) from the transfer functions (else
        // get_Dmnu_1LPT is D1: the neutrinos as CDM)
        if (scaledependent_growth and has_neutrino_transfers()) {
            const auto & xs = growth->x_grid();
            const auto & lks = growth->logk_grid();
            DVector2D D1mnu(lks.size(), DVector(xs.size()));
            for (size_t i = 0; i < lks.size(); i++) {
                const double k = std::exp(lks[i]) * H0_hmpc;
                const double Tcb_ini = transfers->density_over_k_mpc2(SIMCODE::TransferSpecies::CDMBaryon, k, aini);
                for (size_t j = 0; j < xs.size(); j++)
                    D1mnu[i][j] =
                        transfers->density_over_k_mpc2(SIMCODE::TransferSpecies::MassiveNeutrino, k, std::exp(xs[j])) /
                        Tcb_ini;
            }
            Dmnu_1LPT_of_logkoverH0_loga.create(lks, xs, D1mnu, "D1mnuLPT(log(k/H0),log(a))");
        }
    }

    virtual ~GravityModel() = default;

    /// Free the grids compute_force keeps between steps
    virtual void release_workspace() const { density_workspace.free(); }

  protected:
    /// The parameters of the gravity models (read in read_parameters; those of other models keep their defaults)
    SIMCODE::GravityModelSettings settings;
    /// (name, value) of the parameters in settings that are used (printed by info())
    std::vector<std::pair<std::string, std::string>> settings_printed;

    // For massive neutrinos we need transfer functions
    std::shared_ptr<TransferProvider> transfers;
    bool has_neutrino_transfers() const {
        return transfers and transfers->has_density(SIMCODE::TransferSpecies::MassiveNeutrino);
    }

    // The background cosmology
    std::shared_ptr<Cosmology> cosmo;

    // A density-sized grid of compute_force kept between steps (G_eff(k) delta, the fifth-force density); freed by
    // release_workspace (the simulation frees it with its PM grids before the analysis at an output)
    mutable FFTWGrid<NDIM> density_workspace;

    // The massive neutrino fraction f_nu(a) of the source factors, splined in x = ln a by compute_growth_factors
    // (the background sum with the neutrino momentum integral was most of the cost of the scale-dependent growth)
    FML::INTERPOLATION::SPLINE::Spline f_nu_spline;
    double f_nu_xmin{0.0}, f_nu_xmax{-1.0};
    double f_nu_of_a(double a) const {
        const double x = std::log(a);
        return (x >= f_nu_xmin and x <= f_nu_xmax) ? f_nu_spline(x) : this->cosmo->f_nu(a);
    }

    // The theory's quasi-static limit (see get_quasi_static_gravity)
    std::shared_ptr<const QuasiStaticGravity> quasi_static;
    const QuasiStaticGravity & qs() const {
        if (not quasi_static)
            throw std::runtime_error("GravityModel [" + name +
                                     "]: call init() first (the quasi-static theory is built "
                                     "there)");
        return *quasi_static;
    }

    // Name of the gravity model
    std::string name;

    // Scalefactor we normalize D1(aini) = 1 which we take to be the start of the simulation
    double aini;

    // Ranges for splines of growth-factors
    double alow = 1e-4;
    const double ahigh = 1e1;
    const int npts_loga = 1000;
    const double koverH0low = 1e-4 / H0_hmpc;
    // Scale-dependent growth on k = 1e-4..100 h/Mpc; beyond, the splines hold the end values (small boxes can have
    // larger k on the grid, where the growth of f(R) and of massive neutrinos has saturated)
    const double koverH0high = 100.0 / H0_hmpc;
    const int npts_logk = 200;

    // The growth factors (compute_growth_factors)
    std::shared_ptr<const FML::COSMOLOGY::GROWTH::LPTGrowthFactors> growth;
    // The massive neutrino growth from the transfer functions (only with scale-dependent growth and neutrino transfers)
    Spline2D Dmnu_1LPT_of_logkoverH0_loga{"[Dmnu1LPT(log(k/H0),log(a)) not yet created]"};

    // Does it have scaledependent growth?
    bool scaledependent_growth{false};

    // Experimental: the force from finite differences of the potential (in compute_force_from_total_density)
    bool force_use_finite_difference_force{false};
    int force_finite_difference_stencil_order{4};
};
#endif
