#ifndef GRAVITYMODEL_HEADER
#define GRAVITYMODEL_HEADER

#include "Cosmology.h"
#include "ParameterDefaults.h"

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
    const double H0_hmpc = 1.0 / FML::UTILS::SI::c_over_H0_mpch;

    //========================================================================
    // Constructors
    //========================================================================
    GravityModel(std::shared_ptr<Cosmology> cosmo, std::string name) : cosmo(cosmo), name(name) {}

    //========================================================================
    // Name of the model
    //========================================================================
    std::string get_name() const { return name; }

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

    // Output an element (e.g. string/double) in a header/row with a desired alignment width
    template <typename T>
    void output_element(std::ofstream & fp, const T & element, int width = 15) const {
        fp << std::setw(width) << element;
    }

    //========================================================================
    // Output the stuff we compute
    //========================================================================
    virtual void output_header(std::ofstream & fp) {
        fp << "#";
        output_element(fp, "a");
        fp << " ";
        output_element(fp, "GeffG(a,k)");
        fp << " ";
        output_element(fp, "D1(a,k)");
        fp << " ";
        output_element(fp, "D1mnu(a,k)");
        if (transfers) {
            fp << " ";
            output_element(fp, "D1_Tk(a,k)");
            fp << " ";
            output_element(fp, "D1mnu_Tk(a,k)");
        }
        fp << " ";
        output_element(fp, "D2(a,k)");
        fp << " ";
        output_element(fp, "D3a(a,k)");
        fp << " ";
        output_element(fp, "D3b(a,k)");
        fp << " ";
        output_element(fp, "source_1LPT");
        fp << " ";
        output_element(fp, "source_2LPT");
        fp << " ";
        output_element(fp, "source_3LPTa");
        fp << " ";
        output_element(fp, "source_3LPTb");
    }

    virtual void output_row(std::ofstream & fp, double a, double koverH0) const {
        fp << " ";
        output_element(fp, a);
        fp << " ";
        output_element(fp, GeffOverG(a, koverH0));
        fp << " ";
        output_element(fp, get_D_1LPT(a, koverH0));
        fp << " ";
        output_element(fp, get_Dmnu_1LPT(a, koverH0));
        if (transfers) {
            // (D1mnu_Tk: NaN without massive neutrinos)
            const double k = koverH0 * H0_hmpc;
            fp << " ";
            output_element(fp,
                           transfers->density_over_k_mpc2(SIMCODE::TransferSpecies::CDMBaryon, k, a) /
                               transfers->density_over_k_mpc2(SIMCODE::TransferSpecies::CDMBaryon, k, aini));
            fp << " ";
            output_element(fp,
                           has_neutrino_transfers() ?
                               transfers->density_over_k_mpc2(SIMCODE::TransferSpecies::MassiveNeutrino, k, a) /
                                   transfers->density_over_k_mpc2(SIMCODE::TransferSpecies::MassiveNeutrino, k, aini) :
                               std::numeric_limits<double>::quiet_NaN());
        }
        fp << " ";
        output_element(fp, get_D_2LPT(a, koverH0));
        fp << " ";
        output_element(fp, get_D_3LPTa(a, koverH0));
        fp << " ";
        output_element(fp, get_D_3LPTb(a, koverH0));
        fp << " ";
        output_element(fp, source_factor_1LPT(a, koverH0));
        fp << " ";
        output_element(fp, source_factor_2LPT(a, koverH0));
        fp << " ";
        output_element(fp, source_factor_3LPTa(a, koverH0));
        fp << " ";
        output_element(fp, source_factor_3LPTb(a, koverH0));
    }

    virtual void output(std::string filename, double koverH0) {
        std::ofstream fp(filename.c_str());
        if (not fp.is_open())
            return;

        output_header(fp);
        fp << "\n";
        for (int i = 0; i < npts_loga; i++) {
            double loga = std::log(alow) + std::log(ahigh / alow) * i / double(npts_loga - 1);
            double a = std::exp(loga);
            output_row(fp, a, koverH0);
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
        }
        info_model();
        if (FML::ThisTask == 0) {
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }
    /// The model's own lines of info() (its parameters; print on task 0 only)
    virtual void info_model() const {}

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
        aini = 1.0 / (1.0 + param.get<double>("ic_initial_redshift"));
        // The growth factor splines must start before the simulation (e.g. when starting at very high redshift)
        alow = std::min(alow, aini / 10.0);
        // The transfer functions (CAMB, CLASS or the internal solver) are loaded once by the caller and given with
        // set_transfers before this (Main.cpp; the same object is used by the initial conditions and the analysis).
        // The growth factors with massive neutrinos are computed from them (and start at their highest redshift)
        this->scaledependent_growth = this->cosmo->get_fMNu() > 0.0;
        FML::assert_mpi(not(this->cosmo->get_OmegaMNu() > 0.0 and not transfers),
                        "Massive neutrinos need ic_type_of_input = camb, class or internal (the growth factors "
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
    /// G_eff/G = mu(a, k) of the theory (koverH0 = c k / H0), 1 without one (GR)
    virtual double GeffOverG(double a, double koverH0 = 0) const {
        return quasi_static ? quasi_static->mu(std::log(a), koverH0) : 1.0;
    }
    /// Whether G_eff/G depends on k (then the force and the growth do too)
    virtual bool geff_scale_dependent() const { return quasi_static and quasi_static->scale_dependent(); }

    // Factors in the LPT equations
    virtual double source_factor_1LPT([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        double factor = 1.0;
        const double fnu = this->cosmo->f_nu(a);
        if (fnu > 0.0 and koverH0 > 0.0) {
            if (transfers) {
                const double T_nu = transfers->density_over_k_mpc2(SIMCODE::TransferSpecies::MassiveNeutrino, koverH0 * H0_hmpc, a);
                const double T_cb = transfers->density_over_k_mpc2(SIMCODE::TransferSpecies::CDMBaryon, koverH0 * H0_hmpc, a);
                factor = (1.0 - fnu) + fnu * T_nu / T_cb;
            }
        }
        return factor;
    };
    virtual double source_factor_2LPT([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0 - this->cosmo->f_nu(a);
    };
    // The D1^2 source of the 2LPT equation relative to the D2 term (1 in GR):
    //   D2'' + (2 + dlogH/dloga) D2' = rhs source_factor_2LPT (D2 - nonlinear_source_factor_2LPT D1^2)
    // with rhs = 1.5 OmegaM(a) GeffOverG. In DGP the nonlinear (brane-bending) term only adds to the source, it does
    // not multiply D2
    virtual double nonlinear_source_factor_2LPT([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0;
    };
    virtual double source_factor_3LPTa([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0 - this->cosmo->f_nu(a);
    };
    virtual double source_factor_3LPTb([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0 - this->cosmo->f_nu(a);
    };

    /// The force D Phi from the density field (Fourier space): D^2 Phi = 1.5 OmegaM a G_eff/G(a, k) delta, with
    /// G_eff/G(a, k) multiplying delta(k) if it depends on k. Screened models override it
    virtual void compute_force(double a,
                               double H0Box,
                               FFTWGrid<NDIM> & density_fourier,
                               std::string density_assignment_method_used,
                               std::array<FFTWGrid<NDIM>, NDIM> & force_real,
                               const FML::NBODY::ForceOptions & options = {}) const {
        const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * a;
        if (not geff_scale_dependent()) {
            compute_force_from_total_density(density_fourier,
                                             norm_poisson_equation * GeffOverG(a),
                                             density_assignment_method_used,
                                             force_real,
                                             options);
            return;
        }
        // G_eff/G(a, k) delta(k) (kBox = k Box, koverH0 = kBox / H0Box)
        FFTWGrid<NDIM> density_geff;
        FML::NBODY::compute_delta_fifth_force<NDIM>(
            density_fourier, density_geff, [&](double kBox) { return GeffOverG(a, kBox / H0Box); });
        compute_force_from_total_density(
            density_geff, norm_poisson_equation, density_assignment_method_used, force_real, options);
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

        namespace GROWTH = FML::COSMOLOGY::GROWTH;
        auto eq = GROWTH::LPTGrowthEquations::from_background(cosmo->background());
        eq.mu = [this](double x, double k) { return GeffOverG(std::exp(x), k); };
        eq.second_order_source = [this](double x, double k) { return nonlinear_source_factor_2LPT(std::exp(x), k); };
        eq.source_1 = [this](double x, double k) { return source_factor_1LPT(std::exp(x), k); };
        eq.source_2 = [this](double x, double k) { return source_factor_2LPT(std::exp(x), k); };
        eq.source_3a = [this](double x, double k) { return source_factor_3LPTa(std::exp(x), k); };
        eq.source_3b = [this](double x, double k) { return source_factor_3LPTb(std::exp(x), k); };
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

  protected:
    // For massive neutrinos we need transfer functions
    std::shared_ptr<TransferProvider> transfers;
    bool has_neutrino_transfers() const {
        return transfers and transfers->has_density(SIMCODE::TransferSpecies::MassiveNeutrino);
    }

    // The background cosmology
    std::shared_ptr<Cosmology> cosmo;

    // The theory's quasi-static limit (see get_quasi_static_gravity)
    std::shared_ptr<const QuasiStaticGravity> quasi_static;
    const QuasiStaticGravity & qs() const {
        if (not quasi_static)
            throw std::runtime_error("GravityModel [" + name + "]: call init() first (the quasi-static theory is built "
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
