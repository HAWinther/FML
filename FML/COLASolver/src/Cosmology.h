#ifndef COSMOLOGY_HEADER
#define COSMOLOGY_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <memory>

#ifndef __clang__
const double riemann_zeta3 = std::riemann_zeta(3.0);
const double riemann_zeta4 = std::riemann_zeta(4.0);
const double riemann_zeta5 = std::riemann_zeta(5.0);
#else
// clang does not have riemann_zeta so hardcode it
const double riemann_zeta3 = 1.20205690315959;
const double riemann_zeta4 = 1.08232323371113;
const double riemann_zeta5 = 1.03692775514336;
#endif

/// Base class for a general cosmology
class Cosmology {
  public:
    using ParameterMap = FML::UTILS::ParameterMap;
    using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;
    using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
    using Constants = FML::UTILS::ConstantsAndUnits;
    using DVector = FML::INTERPOLATION::SPLINE::DVector;

    //========================================================================
    // Constructors
    //========================================================================
    Cosmology() = default;

    //========================================================================
    // Print some info.
    // In derived class remember to also call the base class (this)
    //========================================================================
    virtual void info() const {
        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Cosmology [" << name << "]\n";
            std::cout << "# Omegab      : " << Omegab << "\n";
            std::cout << "# OmegaM      : " << OmegaM << "\n";
            std::cout << "# OmegaMNu    : " << OmegaMNu << "\n";
            std::cout << "# OmegaCDM    : " << OmegaCDM << "\n";
            std::cout << "# OmegaLambda : " << OmegaLambda << "\n";
            std::cout << "# OmegaR      : " << OmegaR << "\n";
            std::cout << "# OmegaNu     : " << OmegaNu << "\n";
            std::cout << "# OmegaRtot   : " << OmegaRtot << "\n";
            std::cout << "# OmegaK      : " << OmegaK << "\n";
            std::cout << "# h           : " << h << "\n";
            std::cout << "# N_nu        : " << N_nu << "\n";
            std::cout << "# Neff        : " << Neff << "\n";
            std::cout << "# Mnu         : " << Mnu_eV << " eV\n";
            std::cout << "# TCMB        : " << TCMB_kelvin << " K\n";
            std::cout << "# Tnu         : " << Tnu_kelvin << " K\n";
            std::cout << "# As          : " << As << "\n";
            std::cout << "# ns          : " << ns << "\n";
            std::cout << "# kpivot      : " << kpivot_mpc << " 1/Mpc\n";
        }
    }

    //========================================================================
    // Primodial poweer-spetrum
    //========================================================================
    virtual double get_primordial_pofk(double k_hmpc) {
        return 2.0 * M_PI * M_PI / (k_hmpc * k_hmpc * k_hmpc) * As * std::pow(h * k_hmpc / kpivot_mpc, ns - 1.0);
    }

    //========================================================================
    // Solve the background (for models where this is needed)
    // Here we solve and spline the boltzmann integrals for the neutrino
    // energy density. This should be called for derived classes!
    //========================================================================
    virtual void init() { solve_for_neutrinos(); }

    //========================================================================
    // Neutrino specific things. For exact treatment of neutrinos in the
    // background
    //========================================================================
    void solve_for_neutrinos() {
        // Solve for the neutrino boltzmann factor needed to define the background properly
        auto solve_ode = [&](double val) {
            FML::SOLVERS::ODESOLVER::ODEFunction deriv =
                [&](double x, [[maybe_unused]] const double * y, double * dydx) {
                    dydx[0] = x * x * std::sqrt(x * x + val * val) / (1.0 + std::exp(x));
                    return GSL_SUCCESS;
                };
            FML::SOLVERS::ODESOLVER::ODESolver ode;
            std::vector<double> x_arr{0.0, 20.0};
            std::vector<double> yini{0.0};
            ode.solve(deriv, x_arr, yini);
            return ode.get_final_data_by_component(0);
        };

        // Compute and spline F(y) / (F(0) + C y). This converges to 1 in both ends
        // so no issue with splines evaluated out of bounds (as it returns closest value)
        const int npts = 200;
        const double ymin = 0.01;
        const double ymax = 1000.0;
        std::vector<double> y_arr(npts), f_arr(npts);
        for (int i = 0; i < npts; i++) {
            y_arr[i] = i == 0 ? 0.0 : std::exp(std::log(ymin) + std::log(ymax / ymin) * (i - 1) / double(npts - 2));
            f_arr[i] = solve_ode(y_arr[i]) / (sixeta4 + twoeta3 * y_arr[i]);
        }
        neutrino_boltzmann_integral_spline.create(y_arr, f_arr, "Neutrino boltzmann integral");
    }

    // Boltzmann integral for energy density F(y) (where y is proportional to the mass)
    double get_neutrino_boltzmann_integral(double y) const {
        return neutrino_boltzmann_integral_spline(y) * (sixeta4 + twoeta3 * y);
    }
    // Boltzmann integral for energy density derivativedF(y)/dlogy
    double get_dneutrino_boltzmann_integral_dlogy(double y) const {
        return y * (neutrino_boltzmann_integral_spline.deriv_x(y) * (sixeta4 + twoeta3 * y) +
                    twoeta3 * neutrino_boltzmann_integral_spline(y));
    }
    // rhoNu / rhocrit0 used for exact treatment of neutrinos going from relativistic -> non-relativistic
    double get_rhoNu_exact(double a) const {
        static double norm = get_neutrino_boltzmann_integral(0);
        double y = Mnu_eV / get_neutrino_temperature_eV(a) / N_nu;
        return OmegaNu / (a * a * a * a) * get_neutrino_boltzmann_integral(y) / norm;
    }
    // Derivative of rhoNu
    double get_drhoNudloga_exact(double a) const {
        static double norm = get_neutrino_boltzmann_integral(0);
        double y = Mnu_eV / get_neutrino_temperature_eV(a) / N_nu;
        return OmegaNu / (a * a * a * a) *
               (-4.0 * get_neutrino_boltzmann_integral(y) + get_dneutrino_boltzmann_integral_dlogy(y)) / norm;
    }
    // Sound speed over c in non-relativitic limit (1408.2995). Truncating at the free radiation sounds speed if
    // evaluated for very larger redshifts
    double get_neutrino_sound_speed_cs_over_c(double a) const {
        return std::min(nu_sound_speed_factor * get_neutrino_temperature_eV(a) / Mnu_eV, 1.0 / std::sqrt(3.0));
    }

    // Free streaming scale for the neutrinos (1408.2995)
    double get_neutrino_free_streaming_scale_hmpc(double a) const {
        return std::sqrt(1.5 * OmegaM / a) / get_neutrino_sound_speed_cs_over_c(a) * H0_hmpc;
    }
    // Neutrino temperature in eV
    double get_neutrino_temperature_eV(double a) const { return (Tnu_kelvin * units.K * units.k_b / units.eV) / a; }

    //========================================================================
    // Read the parameters we need
    // In derived class remember to also call the base class (this)
    //========================================================================
    virtual void read_parameters(ParameterMap & param) {
        FML::UTILS::ConstantsAndUnits u;

        OmegaMNu = param.get<double>("cosmology_OmegaMNu");
        Omegab = param.get<double>("cosmology_Omegab");
        OmegaCDM = param.get<double>("cosmology_OmegaCDM");
        OmegaLambda = param.get<double>("cosmology_OmegaLambda");
        OmegaM = Omegab + OmegaCDM + OmegaMNu;
        h = param.get<double>("cosmology_h");
        As = param.get<double>("cosmology_As");
        ns = param.get<double>("cosmology_ns");
        kpivot_mpc = param.get<double>("cosmology_kpivot_mpc");
        Neff = param.get<double>("cosmology_Neffective");
        TCMB_kelvin = param.get<double>("cosmology_TCMB_kelvin");

        // Neutrino to photon temperature today
        Tnu_kelvin = TCMB_kelvin * std::pow(Neff / 3.0, 0.25) * std::pow(4.0 / 11.0, 1.0 / 3.0);

        // Compute photon density parameter
        const double N_photon = 2;
        const double rho_critical_today_over_h2 = 3.0 * u.H0_over_h * u.H0_over_h / (8.0 * M_PI * u.G);
        const double OmegaRh2 = N_photon * sixzeta4 / (2.0 * M_PI * M_PI) *
                                std::pow(u.k_b * TCMB_kelvin * u.K / u.hbar, 4) * u.hbar / std::pow(u.c, 5) /
                                rho_critical_today_over_h2;
        OmegaR = OmegaRh2 / (h * h);

        // Neutrino density parameter
        const double OmegaNuh2 = (7.0 / 8.0) * N_nu * std::pow(Tnu_kelvin / TCMB_kelvin, 4) * OmegaRh2;
        OmegaNu = OmegaNuh2 / (h * h);

        // Set the sum of the masses of the neutrinos
        Mnu_eV = (OmegaMNu / OmegaNu) / twoeta3 * sixeta4 * N_nu * ((Tnu_kelvin * u.K * u.k_b / u.eV));
        // Simpler expression: Mnu_eV = 93.14 * OmegaMNu * h * h;

        // Total radiation density (in the early Universe)
        OmegaRtot = OmegaR + OmegaNu;

        // Curvature is whats left. Just set it to 0 if its super small
        OmegaK = 1.0 - OmegaM - OmegaRtot - OmegaLambda;
        if (std::fabs(OmegaK) < 1e-5) {
            OmegaLambda -= OmegaK;
            OmegaK = 0.0;
        }

        // Well to be super precise its really (to avoid overcounting the neutrinos today which are matter):
        // OmegaK = 1.0 - (OmegaLambda + OmegaR + OmegaCDM + Omegab + OmegaNu * F(y_today)/F(0));
    }

    //========================================================================
    // Functions all models have (but expressions might differ so we make them virtual)
    //========================================================================
    virtual double get_OmegaMNu(double a = 1.0) const {
        double E = HoverH0_of_a(a);
        return OmegaMNu / (a * a * a * E * E);
    }
    virtual double get_Omegab(double a = 1.0) const {
        double E = HoverH0_of_a(a);
        return Omegab / (a * a * a * E * E);
    }
    virtual double get_OmegaM(double a = 1.0) const {
        double E = HoverH0_of_a(a);
        return OmegaM / (a * a * a * E * E);
    }
    virtual double get_OmegaCDM(double a = 1.0) const {
        double E = HoverH0_of_a(a);
        return OmegaCDM / (a * a * a * E * E);
    }
    virtual double get_OmegaR(double a = 1.0) const {
        double E = HoverH0_of_a(a);
        return OmegaR / (a * a * a * a * E * E);
    }
    virtual double get_OmegaNu(double a = 1.0) const {
        double E = HoverH0_of_a(a);
        return OmegaNu / (a * a * a * a * E * E);
    }
    virtual double get_OmegaNu_exact(double a = 1.0) const {
        double E = HoverH0_of_a(a);
        return get_rhoNu_exact(a) / (E * E);
    }
    virtual double get_OmegaRtot(double a = 1.0) const {
        double E = HoverH0_of_a(a);
        return OmegaRtot / (a * a * a * a * E * E);
    }
    virtual double get_OmegaK(double a = 1.0) const {
        double E = HoverH0_of_a(a);
        return OmegaK / (a * a * E * E);
    }
    virtual double get_OmegaLambda(double a = 1.0) const {
        double E = HoverH0_of_a(a);
        return OmegaLambda / (E * E);
    }

    virtual double HoverH0_of_a([[maybe_unused]] double a) const = 0;

    virtual double dlogHdloga_of_a([[maybe_unused]] double a) const = 0;

    //========================================================================
    // Output the stuff we compute
    //========================================================================
    virtual void output(std::string filename) const {
        std::ofstream fp(filename.c_str());
        if (not fp.is_open())
            return;
        fp << "#   a     H/H0    dlogHdloga     OmegaM     OmegaRtot    OmegaLambda\n";
        for (int i = 0; i < npts_loga; i++) {
            double loga = std::log(alow) + std::log(ahigh / alow) * i / double(npts_loga);
            double a = std::exp(loga);
            fp << std::setw(15) << a << "  ";
            fp << std::setw(15) << HoverH0_of_a(a) << " ";
            fp << std::setw(15) << dlogHdloga_of_a(a) << " ";
            fp << std::setw(15) << get_OmegaM(a) << " ";
            fp << std::setw(15) << get_OmegaR(a) << " ";
            fp << std::setw(15) << get_OmegaNu(a) << " ";
            fp << std::setw(15) << get_OmegaMNu(a) << " ";
            fp << std::setw(15) << get_OmegaNu_exact(a) << " ";
            fp << std::setw(15) << get_OmegaLambda(a) << " ";
            fp << "\n";
        }
    }

    double get_h() const { return h; }
    double get_As() const { return As; }
    double get_ns() const { return ns; }
    double get_TCMB_kelvin() const { return TCMB_kelvin; }
    double get_Neff() const { return Neff; }
    double get_kpivot_mpc() const { return kpivot_mpc; }
    std::string get_name() { return name; }

    void set_As(double _As) { As = _As; }
    void set_ns(double _ns) { ns = _ns; }
    void set_kpivot_mpc(double _kpivot_mpc) { kpivot_mpc = _kpivot_mpc; }

    virtual ~Cosmology() = default;

  protected:
    //========================================================================
    // Parameters all models have (Baryons, CDM, neutrinos, Cosmological constant)
    //========================================================================
    const double H0_hmpc = 1.0 / 2997.92458;
    double h;             // Hubble parameter (little h)
    double OmegaMNu;      // Massive neutrinos (in the matter era)
    double Omegab;        // Baryons
    double OmegaM;        // Total matter (in the matter era)
    double OmegaCDM;      // Cold dark matter
    double OmegaLambda;   // Dark energy
    double OmegaR;        // Photons
    double OmegaNu;       // Neutrinos (density set by Neff)
    double OmegaRtot;     // Total relativistic (in the radiation era)
    double OmegaK;        // Curvature. Derived from Sum Omega == 1
    double Neff;          // Effecive number of non-photon relativistic species (3.046)
    double TCMB_kelvin;   // Temperature of the CMB today in Kelvin
    double Tnu_kelvin;    // Temperature of the neutrinos today in Kelvin. Derived from Neff and TCMB
    double Mnu_eV;        // Sum of the neutrino masses in eV. Derived from OmegaMNu and h
    const double N_nu{3}; // Number of neutrinos (3)
    std::string name;

    // For neutrinos in the background 
    const double twoeta3{3.0 / 2.0 * riemann_zeta3};
    const double sixeta4{7.0 / 120.0 * M_PI * M_PI * M_PI * M_PI};
    const double sixzeta4{6.0 * riemann_zeta4};
    const double nu_sound_speed_factor{std::sqrt(25.0 * riemann_zeta5 / riemann_zeta3 / 3.0)};
    Spline neutrino_boltzmann_integral_spline;
    Constants units;

    //========================================================================
    // Primordial power-spectrum
    //========================================================================
    double As;
    double ns;
    double kpivot_mpc;

    //========================================================================
    // Ranges for splines of growth-factors
    //========================================================================
    const int npts_loga = 200;
    const double alow = 1.0 / 1000.0;
    const double ahigh = 100.0;
};

#endif
