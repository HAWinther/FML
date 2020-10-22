#ifndef GRAVITYMODEL_HEADER
#define GRAVITYMODEL_HEADER

#include <FML/CAMBUtils/CAMBReader.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
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

using LinearTransferData = FML::FILEUTILS::LinearTransferData;

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

    const double H0_hmpc = 1.0 / 2997.92458;

    // The background cosmology
    std::shared_ptr<Cosmology> cosmo;

    // Does it have scaledependent growth?
    bool scaledependent_growth{false};

    // For massive neutrinos we need transfer functions
    std::shared_ptr<LinearTransferData> transferdata;

    // Name of the gravity model
    std::string name;

    // Scalefactor we normalize D1(aini) = 1 which we take to be the start of the simulation
    double aini;

    // Ranges for splines of growth-factors
    const int npts_loga = 200;
    double alow = 1.0 / 500.0;
    const double ahigh = 2.0;
    const int npts_logk = 100;
    const double koverH0low = 1e-4 / H0_hmpc;
    const double koverH0high = 10.0 / H0_hmpc;

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

    //========================================================================
    // Constructors
    //========================================================================
    GravityModel(std::string name) : name(name) {}
    GravityModel(std::shared_ptr<Cosmology> cosmo, std::string name) : cosmo(cosmo), name(name) {}

    //========================================================================
    // Name of the model
    //========================================================================
    std::string get_name() const { return name; }

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

    //========================================================================
    // Output the stuff we compute
    //========================================================================
    virtual void output(std::string filename, double koverH0) {
        std::ofstream fp(filename.c_str());
        if (not fp.is_open())
            return;
        double k = koverH0 * this->H0_hmpc;
        fp << "#  a  GeffG(a,k)  D1(a,k)  D1mnu(a,k) ( D1_transfer(a,k) D1mnu_transfer(a,k) )  D2(a,k)  D3a(a,k)  "
              "D3b(a,k)  (source terms)  (Output for k = "
           << k << " h/Mpc)\n";
        for (int i = 0; i < npts_loga; i++) {
            double loga = std::log(alow) + std::log(ahigh / alow) * i / double(npts_loga - 1);
            double a = std::exp(loga);
            fp << std::setw(15) << a << "  ";
            fp << std::setw(15) << GeffOverG(a, koverH0) << " ";
            fp << std::setw(15) << get_D_1LPT(a, koverH0) << " ";
            fp << std::setw(15) << get_Dmnu_1LPT(a, koverH0) << " ";
            if (transferdata) {
                fp << std::setw(15)
                   << transferdata->get_cdm_baryon_transfer_function(koverH0 * H0_hmpc, a) /
                          transferdata->get_cdm_baryon_transfer_function(koverH0 * H0_hmpc, aini)
                   << " ";
                fp << std::setw(15)
                   << transferdata->get_massive_neutrino_transfer_function(koverH0 * H0_hmpc, a) /
                          transferdata->get_massive_neutrino_transfer_function(koverH0 * H0_hmpc, aini)
                   << " ";
            }
            fp << std::setw(15) << get_D_2LPT(a, koverH0) << " ";
            fp << std::setw(15) << get_D_3LPTa(a, koverH0) << " ";
            fp << std::setw(15) << get_D_3LPTb(a, koverH0) << " ";
            fp << std::setw(15) << source_factor_1LPT(a, koverH0) << " ";
            fp << std::setw(15) << source_factor_2LPT(a, koverH0) << " ";
            fp << std::setw(15) << source_factor_3LPTa(a, koverH0) << " ";
            fp << std::setw(15) << source_factor_3LPTb(a, koverH0) << " ";
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
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# GravityModel [" << name << "] Cosmology: [" << this->cosmo->get_name() << "]\n";
            std::cout << "# Growth factors unity at zini = " << 1.0 / aini - 1.0 << "\n";
            std::cout << "# Starting integrating growth factors from z = " << 1.0 / alow - 1.0 << "\n";
            std::cout << "# scaledependent_growth : " << scaledependent_growth << "\n";
        }
    }

    //========================================================================
    // Read the transferinfo data from file
    //========================================================================
    void init_transferdata(std::string transferinfofilename) {
        transferdata = std::make_shared<LinearTransferData>(cosmo->get_Omegab(),
                                                            cosmo->get_OmegaCDM(),
                                                            cosmo->get_kpivot_mpc(),
                                                            cosmo->get_As(),
                                                            cosmo->get_ns(),
                                                            cosmo->get_h());
        transferdata->read_transfer(transferinfofilename);
    }
    std::shared_ptr<LinearTransferData> get_transferdata() { return transferdata; }
    void set_transferdata(std::shared_ptr<LinearTransferData> _transferdata) { transferdata = _transferdata; }

    //========================================================================
    // Read the parameters we need
    // In derived class remember to also call the base class (this)
    //========================================================================
    virtual void read_parameters(ParameterMap & param) {
        aini = 1.0 / (1.0 + param.get<double>("ic_initial_redshift"));
        if (param.get<std::string>("ic_type_of_input") == "transferinfofile") {
            init_transferdata(param.get<std::string>("ic_input_filename"));
        }
        this->scaledependent_growth = this->cosmo->get_OmegaMNu() > 0.0;
    }

    //========================================================================
    // Effective gravitational constant and equivalent factors appearing
    // in the LPT growth equations
    //========================================================================
    virtual double GeffOverG([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const = 0;

    // Factors in the LPT equations
    virtual double source_factor_1LPT([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        double factor = 1.0;
        const double OmegaMNu = this->cosmo->get_OmegaMNu();
        if (OmegaMNu > 0.0 and koverH0 > 0.0) {
            if (transferdata) {
                const double OmegaM = this->cosmo->get_OmegaM();
                const double fnu = OmegaMNu / OmegaM;
                const double T_nu = transferdata->get_massive_neutrino_transfer_function(koverH0 * H0_hmpc, a);
                const double T_cb = transferdata->get_cdm_baryon_transfer_function(koverH0 * H0_hmpc, a);
                factor = (1.0 - fnu) + fnu * T_nu / T_cb;
            }
        }
        return factor;
    };
    virtual double source_factor_2LPT([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0 - this->cosmo->get_OmegaMNu() / this->cosmo->get_OmegaM();
    };
    virtual double source_factor_3LPTa([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0 - this->cosmo->get_OmegaMNu() / this->cosmo->get_OmegaM();
    };
    virtual double source_factor_3LPTb([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const {
        return 1.0 - this->cosmo->get_OmegaMNu() / this->cosmo->get_OmegaM();
    };

    // This computes the force DPhi from the density field
    virtual void compute_force(double a,
                               double H0Box,
                               FFTWGrid<NDIM> & density_fourier,
                               std::string density_assignment_method_used,
                               std::array<FFTWGrid<NDIM>, NDIM> & force_real) const = 0;

    //========================================================================
    // Compute and spline growth-factors
    // Fiducial method is LCDM growth equation with an effective GeffG
    //========================================================================
    virtual void compute_growth_factors() {

        // When we have massive neutrinos we use the transferdata when
        // solving the LPT equations so we cannot start earlier than what we
        // have splines for and we cannot do
        if (scaledependent_growth and cosmo->get_OmegaMNu() > 0.0) {
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

        DVector loga_arr(npts_loga);
        for (int i = 0; i < npts_loga; i++) {
            loga_arr[i] = std::log(alow) + std::log(ahigh / alow) * i / double(npts_loga);
        }

        DVector logkoverH0_arr(npts_logk);
        for (int i = 0; i < npts_logk; i++)
            logkoverH0_arr[i] = std::log(koverH0low) + std::log(koverH0high / koverH0low) * i / double(npts_logk);

        // We can include solving for the massive neutrinos below
        // However we need the CAMB data to set the IC so its no point
        // but we leave this as an option (basically the equations in 1605.05283)
        const bool solve_for_neutrinos = transferdata and scaledependent_growth and cosmo->get_OmegaMNu() > 0.0;

        // A quite general set of LPT equations up to 3rd order
        auto solve_growth_equations = [&](double koverH0) -> std::tuple<DVector, DVector, DVector, DVector, DVector> {
            const double OmegaM = cosmo->get_OmegaM();
            const double OmegaMNu = cosmo->get_OmegaMNu();
            const double fnu = OmegaMNu / OmegaM;
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

            // Initial conditions: we are using the analytial solution for
            // GR in Einstein de-Sitter to set these
            // For dD1dx_ini this really changes from 1.0 on large scales to 1-3fnu/5 on small scales in EdS
            const double D1_ini = 1.0;
            double dD1dx_ini = D1_ini * 1.0;
            const double D2_ini = -3.0 / 7.0;
            const double dD2dx_ini = D2_ini * 2.0;
            const double D3a_ini = -1.0 / 3.0;
            const double dD3adx_ini = D3a_ini * 3.0;
            const double D3b_ini = 5.0 / 21.0;
            const double dD3bdx_ini = D3b_ini * 3.0;

            // Neutrinos as CDM
            double D1mnu_ini = D1_ini;
            double dD1mnu_ini = dD1dx_ini;

            // To solve properly for massive neutrinos we need the transfer-functions
            // and growth rates at the initial redshift
            if (solve_for_neutrinos and koverH0 > 0.0) {
                dD1dx_ini = transferdata->get_cdm_baryon_growth_rate(koverH0 * H0_hmpc, alow);
                D1mnu_ini = transferdata->get_massive_neutrino_transfer_function(koverH0 * H0_hmpc, alow) /
                            transferdata->get_cdm_baryon_transfer_function(koverH0 * H0_hmpc, alow);
                dD1mnu_ini = D1mnu_ini * transferdata->get_massive_neutrino_growth_rate(koverH0 * H0_hmpc, alow);
            }

            // The initial conditions
            // D1 = a/aini, D2 = -3/7 D1^2, D3a = -1/3 D1^3 and D3b = 5/21 D1^3 for growing mode in EdS
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
};
#endif
