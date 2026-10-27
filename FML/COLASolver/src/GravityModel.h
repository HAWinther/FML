#ifndef GRAVITYMODEL_HEADER
#define GRAVITYMODEL_HEADER

#include "ParameterDefaults.h"

#include <FML/CAMBUtils/CAMBReader.h>
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

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

/// Base class for gravity models
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
    using LinearTransferData = FML::FILEUTILS::LinearTransferData;
    const double H0_hmpc = 1.0 / 2997.92458;

    //========================================================================
    // Constructors
    //========================================================================
    GravityModel(std::string name) : name(name) {}
    GravityModel(std::shared_ptr<Cosmology> cosmo, std::string name) : cosmo(cosmo), name(name) {}

    //========================================================================
    // Name of the model
    //========================================================================
    std::string get_name() const { return name; }

    std::shared_ptr<Cosmology> get_cosmo() { return cosmo; }

    //========================================================================
    // Growth functions
    //========================================================================
    double get_D_1LPT(double a, double koverH0 = 0.0) const {
        koverH0 = std::max(koverH0, koverH0low);
        return not scaledependent_growth ? D_1LPT_of_loga(std::log(a)) :
                                           D_1LPT_of_logkoverH0_loga(std::log(koverH0), std::log(a));
    }
    double get_Dmnu_1LPT(double a, double koverH0 = 0.0) const {
        koverH0 = std::max(koverH0, koverH0low);
        return not scaledependent_growth ? Dmnu_1LPT_of_loga(std::log(a)) :
                                           Dmnu_1LPT_of_logkoverH0_loga(std::log(koverH0), std::log(a));
    }
    double get_D_2LPT(double a, double koverH0 = 0.0) const {
        koverH0 = std::max(koverH0, koverH0low);
        return not scaledependent_growth ? D_2LPT_of_loga(std::log(a)) :
                                           D_2LPT_of_logkoverH0_loga(std::log(koverH0), std::log(a));
    }
    double get_D_3LPTa(double a, double koverH0 = 0.0) const {
        koverH0 = std::max(koverH0, koverH0low);
        return not scaledependent_growth ? D_3LPTa_of_loga(std::log(a)) :
                                           D_3LPTa_of_logkoverH0_loga(std::log(koverH0), std::log(a));
    }
    double get_D_3LPTb(double a, double koverH0 = 0.0) const {
        koverH0 = std::max(koverH0, koverH0low);
        return not scaledependent_growth ? D_3LPTb_of_loga(std::log(a)) :
                                           D_3LPTb_of_logkoverH0_loga(std::log(koverH0), std::log(a));
    }
   
    //========================================================================
    // Is growth scaledpendent?
    //========================================================================
    bool is_growth_scaledependent() const { return scaledependent_growth; }

    //========================================================================
    // Growth rates
    //========================================================================
    double get_f_1LPT(double a, double koverH0 = 0.0) const {
        koverH0 = std::max(koverH0, koverH0low);
        return not scaledependent_growth ? D_1LPT_of_loga.deriv_x(std::log(a)) / D_1LPT_of_loga(std::log(a)) :
                                           D_1LPT_of_logkoverH0_loga.deriv_y(std::log(koverH0), std::log(a)) /
                                               D_1LPT_of_logkoverH0_loga(std::log(koverH0), std::log(a));
    }
    double get_f_2LPT(double a, double koverH0 = 0.0) const {
        koverH0 = std::max(koverH0, koverH0low);
        return not scaledependent_growth ? D_2LPT_of_loga.deriv_x(std::log(a)) / D_2LPT_of_loga(std::log(a)) :
                                           D_2LPT_of_logkoverH0_loga.deriv_y(std::log(koverH0), std::log(a)) /
                                               D_2LPT_of_logkoverH0_loga(std::log(koverH0), std::log(a));
    }
    double get_f_3LPTa(double a, double koverH0 = 0.0) const {
        koverH0 = std::max(koverH0, koverH0low);
        return not scaledependent_growth ? D_3LPTa_of_loga.deriv_x(std::log(a)) / D_3LPTa_of_loga(std::log(a)) :
                                           D_3LPTa_of_logkoverH0_loga.deriv_y(std::log(koverH0), std::log(a)) /
                                               D_3LPTa_of_logkoverH0_loga(std::log(koverH0), std::log(a));
    }
    double get_f_3LPTb(double a, double koverH0 = 0.0) const {
        koverH0 = std::max(koverH0, koverH0low);
        return not scaledependent_growth ? D_3LPTb_of_loga.deriv_x(std::log(a)) / D_3LPTb_of_loga(std::log(a)) :
                                           D_3LPTb_of_logkoverH0_loga.deriv_y(std::log(koverH0), std::log(a)) /
                                               D_3LPTb_of_logkoverH0_loga(std::log(koverH0), std::log(a));
    }
    
    // Output an element (e.g. string/double) in a header/row with a desired alignment width
    template <typename T> void output_element(std::ofstream & fp, const T & element, int width = 15) const {
        fp << std::setw(width) << element;
    }

    //========================================================================
    // Output the stuff we compute
    //========================================================================
    virtual void output_header(std::ofstream & fp) {
        fp << "#"; output_element(fp, "a"); 
        fp << " "; output_element(fp, "GeffG(a,k)"); 
        fp << " "; output_element(fp, "D1(a,k)"); 
        fp << " "; output_element(fp, "D1mnu(a,k)"); 
        if (transferdata) {
          fp << " "; output_element(fp, "D1_Tk(a,k)"); 
          fp << " "; output_element(fp, "D1mnu_Tk(a,k)"); 
        }
        fp << " "; output_element(fp, "D2(a,k)"); 
        fp << " "; output_element(fp, "D3a(a,k)"); 
        fp << " "; output_element(fp, "D3b(a,k)"); 
        fp << " "; output_element(fp, "source_1LPT"); 
        fp << " "; output_element(fp, "source_2LPT"); 
        fp << " "; output_element(fp, "source_3LPTa"); 
        fp << " "; output_element(fp, "source_3LPTb"); 
    }
    
    virtual void output_row(std::ofstream & fp, double a, double koverH0) const {
        fp << " "; output_element(fp, a); 
        fp << " "; output_element(fp, GeffOverG(a, koverH0)); 
        fp << " "; output_element(fp, get_D_1LPT(a, koverH0)); 
        fp << " "; output_element(fp, get_Dmnu_1LPT(a, koverH0)); 
        if (transferdata) {
          fp << " "; output_element(fp,  transferdata->get_cdm_baryon_transfer_function(koverH0 * H0_hmpc, a) /
                      transferdata->get_cdm_baryon_transfer_function(koverH0 * H0_hmpc, aini)); 
          fp << " "; output_element(fp, transferdata->get_massive_neutrino_transfer_function(koverH0 * H0_hmpc, a) /
                      transferdata->get_massive_neutrino_transfer_function(koverH0 * H0_hmpc, aini)); 
        }
        fp << " "; output_element(fp, get_D_2LPT(a, koverH0)); 
        fp << " "; output_element(fp, get_D_3LPTa(a, koverH0)); 
        fp << " "; output_element(fp, get_D_3LPTb(a, koverH0)); 
        fp << " "; output_element(fp, source_factor_1LPT(a, koverH0)); 
        fp << " "; output_element(fp, source_factor_2LPT(a, koverH0)); 
        fp << " "; output_element(fp, source_factor_3LPTa(a, koverH0)); 
        fp << " "; output_element(fp, source_factor_3LPTb(a, koverH0)); 
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
    virtual void init() { compute_growth_factors(); }

    //========================================================================
    // Show some info
    //========================================================================
    virtual void info() const {
        if (FML::ThisTask == 0) {
            std::cout << "#=====================================================\n";
            std::cout << "# GravityModel is [" << name << "] and has the cosmology: [" << this->cosmo->get_name() << "]\n";
            std::cout << "# Growth factors unity at zini = " << 1.0 / aini - 1.0 << "\n";
            std::cout << "# Starting integrating growth factors from z = " << 1.0 / alow - 1.0 << "\n";
            std::cout << "# scaledependent_growth : " << scaledependent_growth << "\n";
        }
    }

    //========================================================================
    // Read the transferinfo data from file
    //========================================================================
    void init_transferdata(std::string transferinfofilename, std::string fileformat = "CAMB") {
        transferdata = std::make_shared<LinearTransferData>(cosmo->get_Omegab(),
                                                            cosmo->get_OmegaCDM(),
                                                            cosmo->get_OmegaMNu(),
                                                            cosmo->get_kpivot_mpc(),
                                                            cosmo->get_As(),
                                                            cosmo->get_ns(),
                                                            cosmo->get_h(),
                                                            fileformat);
        const bool verbose = false; // For testing
        transferdata->read_transfer(transferinfofilename, verbose);
    }
    std::shared_ptr<LinearTransferData> get_transferdata() { return transferdata; }
    void set_transferdata(std::shared_ptr<LinearTransferData> _transferdata) { transferdata = _transferdata; }

    //========================================================================
    // Read the parameters we need
    // In derived class remember to also call the base class (this)
    //========================================================================
    virtual void read_parameters(ParameterMap & param) {
        aini = 1.0 / (1.0 + param.get<double>("ic_initial_redshift"));
        // The growth factor splines must start before the simulation (e.g. when starting at very high redshift)
        alow = std::min(alow, aini / 10.0);
        if (param.get<std::string>("ic_type_of_input") == "transferinfofile") {
            init_transferdata(param.get<std::string>("ic_input_filename"), COLASOLVER::get_parameter<std::string>(param, "ic_type_of_input_fileformat"));
        }
        this->scaledependent_growth = this->cosmo->get_fMNu() > 0.0;
        // The growth factors with massive neutrinos are computed from the transfer functions (and start at their
        // highest redshift). Without them we would dereference a null pointer in compute_growth_factors
        FML::assert_mpi(not(this->cosmo->get_OmegaMNu() > 0.0 and not transferdata),
                        "cosmology_OmegaMNu > 0 needs ic_type_of_input = transferinfofile (the growth factors with "
                        "massive neutrinos are computed from the transfer functions)");
    
        // Experimental option with finite-difference forces (instead of fourier)
        force_use_finite_difference_force = COLASOLVER::get_parameter<bool>(param, "force_use_finite_difference_force");
        force_finite_difference_stencil_order = COLASOLVER::get_parameter<int>(param, "force_finite_difference_stencil_order");
    }

    //========================================================================
    // Effective gravitational constant and equivalent factors appearing
    // in the LPT growth equations
    //========================================================================
    virtual double GeffOverG([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const = 0;

    // Factors in the LPT equations
    virtual double source_factor_1LPT([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        double factor = 1.0;
        const double fnu = this->cosmo->get_fMNu();
        if (fnu > 0.0 and koverH0 > 0.0) {
            if (transferdata) {
                const double T_nu = transferdata->get_massive_neutrino_transfer_function(koverH0 * H0_hmpc, a);
                const double T_cb = transferdata->get_cdm_baryon_transfer_function(koverH0 * H0_hmpc, a);
                factor = (1.0 - fnu) + fnu * T_nu / T_cb;
            }
        }
        return factor;
    };
    virtual double source_factor_2LPT([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0 - this->cosmo->get_fMNu();
    };
    virtual double source_factor_3LPTa([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0 - this->cosmo->get_fMNu();
    };
    virtual double source_factor_3LPTb([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0 - this->cosmo->get_fMNu();
    };

    // This computes the force DPhi from the density field
    virtual void compute_force(double a,
                               double H0Box,
                               FFTWGrid<NDIM> & density_fourier,
                               std::string density_assignment_method_used,
                               std::array<FFTWGrid<NDIM>, NDIM> & force_real) const = 0;

    //========================================================================
    // The force D Phi from the (total) density that sources it, D^2 Phi = norm_poisson_equation * density:
    // with finite differences of the potential (force_use_finite_difference_force) or in Fourier space.
    // Shared by all gravity models.
    //========================================================================
    void compute_force_from_total_density(FFTWGrid<NDIM> & density_fourier,
                                          double norm_poisson_equation,
                                          const std::string & density_assignment_method_used,
                                          std::array<FFTWGrid<NDIM>, NDIM> & force_real) const {
        if (force_use_finite_difference_force) {
            // A stencil of order 2, 4 or 6 (phi(i+/-1), phi(i+/-2), ...) needs stencil_order/2 boundary slices
            const int stencil_order = force_finite_difference_stencil_order;
            const int nboundary_cells = stencil_order / 2;
            FFTWGrid<NDIM> potential_real;
            FML::NBODY::compute_potential_real_from_density_fourier<NDIM>(
                density_fourier, potential_real, norm_poisson_equation, nboundary_cells);
            FML::NBODY::compute_force_from_potential_real<NDIM>(
                potential_real, force_real, density_assignment_method_used, stencil_order);
        } else {
            FML::NBODY::compute_force_from_density_fourier<NDIM>(
                density_fourier, force_real, density_assignment_method_used, norm_poisson_equation);
        }
    }

    //========================================================================
    // a / a_eq at scale factor a, from the background: in a matter + radiation
    // universe 3/2 OmegaM(a) = 3/2 y / (1 + y) with y = a / a_eq.
    // Returns a large number when radiation is negligible.
    //========================================================================
    double a_over_aeq(double a) const {
        const double H = cosmo->HoverH0_of_a(a);
        const double r = cosmo->get_OmegaM() / (H * H * a * a * a);
        if (r >= 1.0 - 1e-12)
            return 1e30;
        return r / (1.0 - r);
    }

    //========================================================================
    // Growing-mode initial conditions for the LPT growth equations in a
    // universe with matter and (smooth) radiation, as a power series in y = a/a_eq.
    // D1 = 1 + 3y/2 is the Meszaros growing mode, and D2, D3a, D3b are the
    // solutions that vanish as y -> 0 (no growth during radiation domination).
    // Returns {D1, dD1/dlna, D2, dD2/dlna, D3a, dD3a/dlna, D3b, dD3b/dlna}.
    // For y -> infinity this tends to the Einstein-de Sitter values
    // D2 = -3/7 D1^2, D3a = -1/3 D1^3, D3b = 5/21 D1^3. Converges for y < 1.
    //========================================================================
    static std::array<double, 8> radiation_matter_growing_mode(double y, int nterms = 200) {
        const int N = nterms;
        auto mult = [N](const std::vector<double> & u, const std::vector<double> & v) {
            std::vector<double> w(N, 0.0);
            for (int i = 0; i < N; i++)
                for (int j = 0; i + j < N; j++)
                    w[i + j] += u[i] * v[j];
            return w;
        };
        // (1+y) D'' + y/2 D' - 3/2 y D + 3/2 y S = 0 with ' = d/dlny, D = sum c_n y^n, c_0 = 0
        auto solve = [N](const std::vector<double> & S) {
            std::vector<double> c(N, 0.0);
            for (int n = 1; n < N; n++) {
                const double m = n - 1;
                c[n] = -((m * m + 0.5 * m - 1.5) * c[n - 1] + 1.5 * S[n - 1]) / double(n * n);
            }
            return c;
        };
        std::vector<double> D1(N, 0.0);
        D1[0] = 1.0;
        D1[1] = 1.5;
        auto D1sq = mult(D1, D1);
        auto D1cube = mult(D1sq, D1);
        auto D2 = solve(D1sq);
        std::vector<double> S3a(N), S3b(N);
        auto D2D1 = mult(D2, D1);
        for (int n = 0; n < N; n++) {
            S3a[n] = 2.0 * D1cube[n];
            S3b[n] = D2D1[n] - D1cube[n];
        }
        auto D3a = solve(S3a);
        auto D3b = solve(S3b);

        std::array<double, 8> res{};
        const std::vector<double> * series[4] = {&D1, &D2, &D3a, &D3b};
        for (int k = 0; k < 4; k++) {
            double val = 0.0, der = 0.0, yn = 1.0;
            for (int n = 0; n < N; n++) {
                val += (*series[k])[n] * yn;
                der += n * (*series[k])[n] * yn;
                yn *= y;
            }
            res[2 * k] = val;
            res[2 * k + 1] = der;
        }
        return res;
    }

    //========================================================================
    // Compute and spline growth-factors
    // Fiducial method is LCDM growth equation with an effective GeffG
    //========================================================================
    virtual void compute_growth_factors() {

        // When we have massive neutrinos we use the transferdata when
        // solving the LPT equations so we cannot start earlier than what we
        // have splines for and we cannot do
        if (scaledependent_growth and cosmo->get_fMNu() > 0.0) {
            const double zstart = transferdata->get_zmax_splines();
            const double astart = 1.0 / (1.0 + zstart);
            if (alow < astart) {
                alow = astart;
            }
            if (aini < astart and scaledependent_growth) {
                throw std::runtime_error(
                    "Transferdata provided are only up to z = " + std::to_string(zstart) +
                    ", but we want to start simulation at z = " + std::to_string(1.0 / aini - 1.0) +
                    " Provide the required data, put OmegaMNu = 0.0 or set it so that we don't use scaledependent "
                    "growth!");
                aini = astart;
            }
        }

        // Start the growth factors deep in the radiation era (a < a_eq / 10) when radiation matters, so that
        // the growing-mode initial conditions below are accurate (not possible when we need transfer data
        // for massive neutrinos)
        if (not(scaledependent_growth and cosmo->get_fMNu() > 0.0)) {
            const double y = a_over_aeq(alow);
            if (y > 0.1 and y < 1e4)
                alow *= 0.1 / y;
        }
        const double y_low = a_over_aeq(alow);

        DVector loga_arr(npts_loga);
        for (int i = 0; i < npts_loga; i++) {
            loga_arr[i] = std::log(alow) + std::log(ahigh / alow) * i / double(npts_loga);
        }

        DVector logkoverH0_arr(npts_logk);
        for (int i = 0; i < npts_logk; i++)
            logkoverH0_arr[i] = std::log(koverH0low) + std::log(koverH0high / koverH0low) * i / double(npts_logk);

        // With the transfer functions the neutrino source in the 1LPT equation is exact: fnu T_nu(k,a) / T_cb(k,a)
        // (source_factor_1LPT), and the initial growth rate is the CDM+baryon growth rate from the transfer functions.
        // The alternative below (solve_for_neutrinos, a fluid approximation for the neutrinos with a free-streaming
        // scale: the equations in 1605.05283) is kept for reference but not used: it overestimates the growth of
        // delta_cb by up to 2% (z = 49 -> 0, 0.4 eV, k ~ 0.01-0.1 h/Mpc) compared with CAMB, while the exact source
        // agrees with CAMB to 0.02% (k > 0.03 h/Mpc) and 0.1% (k = 0.01 h/Mpc)
        const bool use_neutrino_transfer = transferdata and scaledependent_growth and cosmo->get_fMNu() > 0.0;
        const bool solve_for_neutrinos = false;

        // A quite general set of LPT equations up to 3rd order
        auto solve_growth_equations = [&](double koverH0) -> std::tuple<DVector, DVector, DVector, DVector, DVector> {
            const double OmegaM = cosmo->get_OmegaM();
            const double fnu = cosmo->get_fMNu();
            FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double x, const double * y, double * dydx) {
                const double a = std::exp(x);
                const double H = cosmo->HoverH0_of_a(a);
                const double dlogHdx = cosmo->dlogHdloga_of_a(a);

                // source_factor_nLPT also contains the effect of neutrinos ((1-fnu) + fnu * Dnu/Dcb) for 1LPT
                // and (1-fnu) otherwise (i.e. neutrinos don't contribute). If solve_for_neutrinos then we
                // do 1LPT further down
                const double rhs = 1.5 * OmegaM * GeffOverG(a, koverH0) / (H * H * a * a * a);
                const double rhs_1LPT = rhs * source_factor_1LPT(a, koverH0);
                const double rhs_2LPT = rhs * source_factor_2LPT(a, koverH0);
                const double rhs_3LPTa = rhs * source_factor_3LPTa(a, koverH0);
                const double rhs_3LPTb = rhs * source_factor_3LPTb(a, koverH0);

                const double D1 = y[0];
                const double dD1dx = y[1];
                const double D2 = y[2];
                const double dD2dx = y[3];
                const double D3a = y[4];
                const double dD3adx = y[5];
                const double D3b = y[6];
                const double dD3bdx = y[7];

                // CDM+baryon LPT equations
                dydx[0] = dD1dx;
                dydx[1] = rhs_1LPT * D1 - (2.0 + dlogHdx) * dD1dx;
                dydx[2] = dD2dx;
                dydx[3] = rhs_2LPT * (D2 - D1 * D1) - (2.0 + dlogHdx) * dD2dx;
                dydx[4] = dD3adx;
                dydx[5] = rhs_3LPTa * (D3a - 2.0 * D1 * D1 * D1) - (2.0 + dlogHdx) * dD3adx;
                dydx[6] = dD3bdx;
                dydx[7] = rhs_3LPTb * (D3b + D1 * D1 * D1 - D2 * D1) - (2.0 + dlogHdx) * dD3bdx;

                // Massive neutrinos LPT equations
                if (solve_for_neutrinos) {
                    // k over the free-streaming scale kfs
                    const double k_over_kfs = koverH0 * H0_hmpc / cosmo->get_neutrino_free_streaming_scale_hmpc(a);
                    const double k_over_kfs2 = k_over_kfs * k_over_kfs;

                    const double D1mnu = y[8];
                    const double dD1mnudx = y[9];
                    dydx[8] = dD1mnudx;
                    dydx[9] = rhs * (D1 * (1.0 - fnu) + (fnu - k_over_kfs2) * D1mnu) - (2.0 + dlogHdx) * dD1mnudx;

                    // Updating 1LPT equation for baryon+CDM to reflect massive neutrinos
                    dydx[1] = rhs * (D1 * (1.0 - fnu) + fnu * D1mnu) - (2.0 + dlogHdx) * dD1dx;
                } else {
                    dydx[8] = dydx[9] = 0.0;
                }

                return GSL_SUCCESS;
            };

            // Initial conditions: the growing mode in a matter + radiation universe (exact for y = a/a_eq -> 0
            // and the Einstein-de Sitter solution for y -> infinity). The decaying mode is absent.
            // For dD1dx_ini this really changes from 1.0 on large scales to 1-3fnu/5 on small scales in EdS
            std::array<double, 8> ini;
            if (y_low < 0.5) {
                ini = radiation_matter_growing_mode(y_low);
            } else {
                ini = {1.0, 1.0, -3.0 / 7.0, -6.0 / 7.0, -1.0 / 3.0, -1.0, 5.0 / 21.0, 5.0 / 7.0};
            }
            const double D1_ini = ini[0];
            double dD1dx_ini = ini[1];
            const double D2_ini = ini[2];
            const double dD2dx_ini = ini[3];
            const double D3a_ini = ini[4];
            const double dD3adx_ini = ini[5];
            const double D3b_ini = ini[6];
            const double dD3bdx_ini = ini[7];

            // Neutrinos as CDM
            double D1mnu_ini = D1_ini;
            double dD1mnu_ini = dD1dx_ini;

            // To solve properly for massive neutrinos we need the transfer-functions
            // and growth rates at the initial redshift
            if (use_neutrino_transfer and koverH0 > 0.0)
                dD1dx_ini = D1_ini * transferdata->get_cdm_baryon_growth_rate(koverH0 * H0_hmpc, alow);
            if (solve_for_neutrinos and koverH0 > 0.0) {
                D1mnu_ini = D1_ini * transferdata->get_massive_neutrino_transfer_function(koverH0 * H0_hmpc, alow) /
                            transferdata->get_cdm_baryon_transfer_function(koverH0 * H0_hmpc, alow);
                dD1mnu_ini = D1mnu_ini * transferdata->get_massive_neutrino_growth_rate(koverH0 * H0_hmpc, alow);
            }

            std::vector<double> yini{
                D1_ini, dD1dx_ini, D2_ini, dD2dx_ini, D3a_ini, dD3adx_ini, D3b_ini, dD3bdx_ini, D1mnu_ini, dD1mnu_ini};

            // Solve the ODE
            FML::SOLVERS::ODESOLVER::ODESolver ode;
            ode.solve(deriv, loga_arr, yini);
            auto D1 = ode.get_data_by_component(0);
            auto D2 = ode.get_data_by_component(2);
            auto D3a = ode.get_data_by_component(4);
            auto D3b = ode.get_data_by_component(6);
            auto D1mnu = ode.get_data_by_component(8);

            // Normalize such that D = 1 at initial redshift
            FML::INTERPOLATION::SPLINE::Spline tmp;
            tmp.create(loga_arr, D1, "Temporary D1(loga) Spline");
            const double D = tmp(std::log(aini));
            for (size_t i = 0; i < D1.size(); i++) {
                D1[i] /= D;
                D2[i] /= D * D;
                D3a[i] /= D * D * D;
                D3b[i] /= D * D * D;
                D1mnu[i] /= D;
                if (not solve_for_neutrinos) {
                    // Neutrinos treated as CDM
                    D1mnu[i] = D1[i];
                    // Neutrinos treated exactly by using transfer functions
                    if (transferdata and koverH0 > 0.0) {
                        D1mnu[i] = transferdata->get_massive_neutrino_transfer_function(koverH0 * H0_hmpc,
                                                                                        std::exp(loga_arr[i])) /
                                   transferdata->get_cdm_baryon_transfer_function(koverH0 * H0_hmpc, aini);
                    }
                }
            }

            return {D1, D2, D3a, D3b, D1mnu};
        };

        // Compute scaleindependent growth-factors (for scaledependent growth
        // this corresponds to k = 0)
        auto data = solve_growth_equations(0.0);
        D_1LPT_of_loga.create(loga_arr, std::get<0>(data), "D1LPT(log(a))");
        D_2LPT_of_loga.create(loga_arr, std::get<1>(data), "D2LPT(log(a))");
        D_3LPTa_of_loga.create(loga_arr, std::get<2>(data), "D3LPTa(log(a))");
        D_3LPTb_of_loga.create(loga_arr, std::get<3>(data), "D3LPTb(log(a))");
        Dmnu_1LPT_of_loga.create(loga_arr, std::get<4>(data), "Dmnu1LPT(log(a))");

        // Compute the full scaledependent D_iLPT(k,a)
        if (scaledependent_growth) {

            DVector2D D1mnu(npts_logk, DVector(npts_loga));
            DVector2D D1(npts_logk, DVector(npts_loga));
            DVector2D D2(npts_logk, DVector(npts_loga));
            DVector2D D3a(npts_logk, DVector(npts_loga));
            DVector2D D3b(npts_logk, DVector(npts_loga));
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int i = 0; i < npts_logk; i++) {
                auto data = solve_growth_equations(std::exp(logkoverH0_arr[i]));
                D1[i] = std::get<0>(data);
                D2[i] = std::get<1>(data);
                D3a[i] = std::get<2>(data);
                D3b[i] = std::get<3>(data);
                D1mnu[i] = std::get<4>(data);
            }
            D_1LPT_of_logkoverH0_loga.create(logkoverH0_arr, loga_arr, D1, "D1LPT(log(k/H0),log(a))");
            D_2LPT_of_logkoverH0_loga.create(logkoverH0_arr, loga_arr, D2, "D2LPT(log(k/H0),log(a))");
            D_3LPTa_of_logkoverH0_loga.create(logkoverH0_arr, loga_arr, D3a, "D3LPTa(log(k/H0),log(a))");
            D_3LPTb_of_logkoverH0_loga.create(logkoverH0_arr, loga_arr, D3b, "D3LPTb(log(k/H0),log(a))");
            Dmnu_1LPT_of_logkoverH0_loga.create(logkoverH0_arr, loga_arr, D1mnu, "D1mnuLPT(log(k/H0),log(a))");
        }
    }

    virtual ~GravityModel() = default;
   
  protected:
    
    // For massive neutrinos we need transfer functions
    std::shared_ptr<LinearTransferData> transferdata;
    
    // The background cosmology
    std::shared_ptr<Cosmology> cosmo;

    // Name of the gravity model
    std::string name;

    // Scalefactor we normalize D1(aini) = 1 which we take to be the start of the simulation
    double aini;

    // Ranges for splines of growth-factors
    double alow = 1e-4;
    const double ahigh = 1e1;
    const int npts_loga = 1000;
    const double koverH0low = 1e-4 / H0_hmpc;
    const double koverH0high = 100.0 / H0_hmpc;
    const int npts_logk = 200;

    // Scaleindependent growth-factors
    Spline D_1LPT_of_loga{"[D1LPT(log(a)) not yet created]"};
    Spline D_2LPT_of_loga{"[D2LPT(log(a)) not yet created]"};
    Spline D_3LPTa_of_loga{"[D3LPTa(log(a)) not yet created]"};
    Spline D_3LPTb_of_loga{"[D3LPTb(log(a)) not yet created]"};

    // Scaledependent growth factors
    Spline2D D_1LPT_of_logkoverH0_loga{"[D1LPT(log(k/H0),log(a)) not yet created]"};
    Spline2D D_2LPT_of_logkoverH0_loga{"[D2LPT(log(k/H0),log(a)) not yet created]"};
    Spline2D D_3LPTa_of_logkoverH0_loga{"[D3LPTa(log(k/H0),log(a)) not yet created"};
    Spline2D D_3LPTb_of_logkoverH0_loga{"[D3LPTb(log(k/H0),log(a)) not yet created]"};

    Spline Dmnu_1LPT_of_loga{"[Dmnu1LPT(log(a)) not yet created]"};
    Spline2D Dmnu_1LPT_of_logkoverH0_loga{"[Dmnu1LPT(log(k/H0),log(a)) not yet created]"};
    
    // Does it have scaledependent growth?
    bool scaledependent_growth{false};

    // Experimental option (only for GR so far)
    bool force_use_finite_difference_force{false};
    int force_finite_difference_stencil_order{4};
};
#endif
