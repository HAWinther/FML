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

/// Base class for a general cosmology
class BackgroundCosmology {
  public:
    using ParameterMap = FML::UTILS::ParameterMap;
    using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;
    using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;

    //========================================================================
    // Constructors
    //========================================================================
    BackgroundCosmology() = default;

    //========================================================================
    // Print some info.
    // In derived class remember to also call the base class (this)
    //========================================================================
    virtual void info() const {
        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# BackgroundCosmology [" << name << "]\n";
            std::cout << "# Omegab      : " << Omegab << "\n";
            std::cout << "# OmegaM      : " << OmegaM << "\n";
            std::cout << "# OmegaMNu    : " << OmegaMNu << "\n";
            std::cout << "# OmegaCDM    : " << OmegaCDM << "\n";
            std::cout << "# OmegaLambda : " << OmegaLambda << "\n";
            std::cout << "# OmegaR      : " << OmegaR << "\n";
            std::cout << "# OmegaNu     : " << OmegaNu << "\n";
            std::cout << "# OmegaRtot   : " << OmegaRtot << "\n";
            std::cout << "# OmegaK      : " << OmegaK << "\n";
            std::cout << "# Neff        : " << Neff << "\n";
            std::cout << "# h           : " << h << "\n";
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
    //========================================================================
    virtual void init() {}

    //========================================================================
    // Read the parameters we need
    // In derived class remember to also call the base class (this)
    //========================================================================
    virtual void read_parameters(ParameterMap & param) {
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

        // Compute photon and neutrino density parameter
        FML::UTILS::ConstantsAndUnits u;
        const double Tnu_over_TCMB = std::pow(4.0 / 11.0, 1.0 / 3.0);
        const double rho_critical_today = 3.0 * u.H0_over_h * h * u.H0_over_h * h / (8.0 * M_PI * u.G);
        OmegaR = 2.0 * (M_PI * M_PI / 30.0) * std::pow(u.k_b * TCMB_kelvin * u.K / u.hbar, 4) * u.hbar /
                 std::pow(u.c, 5) / rho_critical_today;
        OmegaNu = (7.0 / 8.0) * Neff * std::pow(Tnu_over_TCMB, 4) * OmegaR;
        OmegaRtot = OmegaR + OmegaNu;
        OmegaK = 1.0 - OmegaM - OmegaRtot - OmegaLambda;
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
        return OmegaNu / (E * E);
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
            fp << std::setw(15) << get_OmegaRtot(a) << " ";
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
    void set_ns(double _ns) { As = _ns; }
    void set_kpivot_mpc(double _kpivot_mpc) { kpivot_mpc = _kpivot_mpc; }

  protected:
    //========================================================================
    // Parameters all models have (Baryons, CDM, neutrinos, Cosmological constant)
    //========================================================================
    double h;           // Hubble parameter (little h)
    double OmegaMNu;    // Massive neutrinos
    double Omegab;      // Baryons
    double OmegaM;      // Total matter
    double OmegaCDM;    // Cold dark matter
    double OmegaLambda; // Dark energy
    double OmegaR;      // Photons
    double OmegaNu;     // Neutrinos (density set by Neff)
    double OmegaRtot;   // Total relativistic
    double OmegaK;      // Curvature
    double Neff;        // Effecive number of non-photn relativistic species (3.046)
    double TCMB_kelvin; // Temperature of the CMB today in Kelvin
    std::string name;

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
