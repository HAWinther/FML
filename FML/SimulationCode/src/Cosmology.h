#ifndef COSMOLOGY_HEADER
#define COSMOLOGY_HEADER

#include "EE2Boost.h"
#include "ParameterDefaults.h"

#include <FML/Cosmology/Background/BackgroundModel_Tabulated.h>
#include <FML/Cosmology/Background/Presets.h>
#include <FML/Global/Global.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>
#include <FML/Units/Units.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

//=============================================================================
/// The cosmology of the simulation: the background expansion of FML's background module
/// (FML/Cosmology/Background, one ComponentBackground per run) and the parameters SimulationCode uses.
///
/// The model is selected by cosmology_model and built from the background registry:
///   LCDM          -> lcdm
///   w0waCDM       -> w0wa (cosmology_w0, cosmology_wa)
///   DGP           -> dgp on the self-accelerating branch (cosmology_dgp_OmegaRC; Lambda from E(0) = 1, with a
///                    warning if it is not zero)
///   JBD           -> jbd (cosmology_JBD_wBD, _GeffG_today, _density_parameter_definition)
///   Quintessence  -> quintessence[_inverse_power_law|_pngb|_hilltop] (cosmology_quintessence_potential, ...)
///   EDE           -> ede_zc (cosmology_ede_fEDE, _zc, _theta_ini, _n)
///   Tabulated     -> a tabulated dark energy (BackgroundModel_Tabulated.h; cosmology_tabulated_filename, _format
///                    = H (z, H/H0), CLASS (a CLASS background file) or w (a, w))
///
/// Neutrinos: N_nu = 3 degenerate species with the temperature T_nu = T_CMB (Neff/3)^(1/4) (4/11)^(1/3), all massive
/// with the total mass Mnu (from cosmology_OmegaMNu with the non-relativistic relation) when OmegaMNu > 0, otherwise
/// massless.
///
/// The Omega getters keep SimulationCode's definitions: at a = 1 the parameter (OmegaM = Omegab + OmegaCDM +
/// OmegaMNu, OmegaR the photons, OmegaNu the neutrinos as if massless), otherwise the parameter scaled as matter
/// (a^-3), radiation (a^-4) or curvature (a^-2) over E^2(a). get_OmegaNu_exact and get_OmegaLambda (all the dark
/// energy) are the actual densities of the background. background() gives the background itself.
//=============================================================================
class Cosmology {
  public:
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;
    using DVector = FML::INTERPOLATION::SPLINE::DVector;
    using Background = FML::COSMOLOGY::BACKGROUND::ComponentBackground;

    /// model: the name of the cosmology_model
    explicit Cosmology(std::string model = "LCDM") : name(std::move(model)) {
        if (name == "Quintessence" or name == "EDE" or name == "Tabulated")
            alow = 1e-9; // the output table (output()) covers the early times where the dark energy can matter
    }

    //========================================================================
    // Read the parameters (the ones of all models, then those of the model)
    //========================================================================
    void read_parameters(ParameterMap & param) {
        using namespace FML::UTILS::SI;
        OmegaMNu = param.get<double>("cosmology_OmegaMNu");
        Omegab = param.get<double>("cosmology_Omegab");
        OmegaCDM = param.get<double>("cosmology_OmegaCDM");
        OmegaM = Omegab + OmegaCDM + OmegaMNu;
        OmegaK = SIMCODE::get_parameter<double>(param, "cosmology_OmegaK");
        h = param.get<double>("cosmology_h");
        As = param.get<double>("cosmology_As");
        ns = param.get<double>("cosmology_ns");
        kpivot_mpc = param.get<double>("cosmology_kpivot_mpc");
        Neff = param.get<double>("cosmology_Neffective");
        TCMB_kelvin = param.get<double>("cosmology_TCMB_kelvin");

        // Neutrino to photon temperature today
        Tnu_kelvin = TCMB_kelvin * std::pow(Neff / 3.0, 0.25) * std::pow(4.0 / 11.0, 1.0 / 3.0);

        // Photons and (massless) neutrinos
        const double rho_critical_today_over_h2 = 3.0 * H0_over_h * H0_over_h / (8.0 * M_PI * G);
        const double OmegaRh2 = 2.0 * sixzeta4 / (2.0 * M_PI * M_PI) * std::pow(k_b * TCMB_kelvin / hbar, 4) * hbar /
                                std::pow(c, 5) / rho_critical_today_over_h2;
        OmegaR = OmegaRh2 / (h * h);
        OmegaNu = (7.0 / 8.0) * N_nu * std::pow(Tnu_kelvin / TCMB_kelvin, 4) * OmegaR;
        OmegaRtot = OmegaR + OmegaNu;

        // The sum of the neutrino masses (non-relativistic relation; no neutrinos, Neff = 0, needs OmegaMNu = 0)
        FML::assert_mpi(OmegaNu > 0.0 or OmegaMNu == 0.0,
                        "cosmology_Neffective = 0 (no neutrinos) needs cosmology_OmegaMNu = 0");
        Mnu_eV = OmegaNu > 0.0 ? (OmegaMNu / OmegaNu) / twoeta3 * sixeta4 * N_nu * (k_b * Tnu_kelvin / eV) : 0.0;

        // The parameters of the model
        model_parameters.clear();
        if (name == "LCDM") {
            registry_name = "lcdm";
        } else if (name == "w0waCDM") {
            registry_name = "w0wa";
            w0 = param.get<double>("cosmology_w0");
            wa = param.get<double>("cosmology_wa");
            model_parameters = {{"w0", w0}, {"wa", wa}};
        } else if (name == "DGP") {
            registry_name = "dgp";
            model_parameters = {{"Omega_rc", param.get<double>("cosmology_dgp_OmegaRC")}, {"dgp_branch", 1.0}};
        } else if (name == "JBD") {
            registry_name = "jbd";
            const auto def = param.get<std::string>("cosmology_JBD_density_parameter_definition");
            const std::vector<std::string> defs{"hi-class", "Gtoday", "Gnewton", "Gbare"};
            const auto it = std::find(defs.begin(), defs.end(), def);
            if (it == defs.end())
                throw std::runtime_error("JBD: unknown cosmology_JBD_density_parameter_definition = " + def +
                                         " (hi-class, Gtoday, Gnewton, Gbare)");
            model_parameters = {{"bd_omega", param.get<double>("cosmology_JBD_wBD")},
                                {"bd_GeffG_today", param.get<double>("cosmology_JBD_GeffG_today")},
                                {"bd_density_parameter_definition", double(it - defs.begin())}};
        } else if (name == "Quintessence") {
            const auto potential = param.get<std::string>("cosmology_quintessence_potential");
            const double phi_ini = SIMCODE::get_parameter<double>(param, "cosmology_quintessence_phi_ini");
            const double dphi_ini = SIMCODE::get_parameter<double>(param, "cosmology_quintessence_dphidlna_ini");
            model_parameters = {{"quint_phi_ini", phi_ini}, {"quint_dphidlna_ini", dphi_ini}};
            if (potential == "exponential") {
                registry_name = "quintessence";
                model_parameters["quint_lambda"] = param.get<double>("cosmology_quintessence_lambda");
            } else if (potential == "inverse_power_law") {
                registry_name = "quintessence_inverse_power_law";
                model_parameters["quint_alpha"] = param.get<double>("cosmology_quintessence_alpha");
                if (not(phi_ini > 0.0))
                    throw std::runtime_error("Quintessence: inverse_power_law needs cosmology_quintessence_phi_ini > 0");
            } else if (potential == "pngb") {
                registry_name = "quintessence_pngb";
                model_parameters["quint_f"] = param.get<double>("cosmology_quintessence_f");
                if (not(model_parameters["quint_f"] > 0.0))
                    throw std::runtime_error("Quintessence: pngb needs cosmology_quintessence_f > 0");
            } else if (potential == "hilltop") {
                registry_name = "quintessence_hilltop";
                model_parameters["quint_mu"] = param.get<double>("cosmology_quintessence_mu");
                model_parameters["quint_p"] = param.get<double>("cosmology_quintessence_p");
                if (not(model_parameters["quint_mu"] > 0.0) or not(model_parameters["quint_p"] > 0.0))
                    throw std::runtime_error("Quintessence: hilltop needs cosmology_quintessence_mu > 0 and _p > 0");
            } else {
                throw std::runtime_error("Quintessence: unknown cosmology_quintessence_potential = " + potential +
                                         " (exponential, inverse_power_law, pngb, hilltop)");
            }
            if (dphi_ini * dphi_ini >= 6.0)
                throw std::runtime_error("Quintessence: need |cosmology_quintessence_dphidlna_ini| < sqrt(6)");
        } else if (name == "EDE") {
            registry_name = "ede_zc";
            const double fEDE = param.get<double>("cosmology_ede_fEDE"), zc = param.get<double>("cosmology_ede_zc");
            const double theta = param.get<double>("cosmology_ede_theta_ini"), n = param.get<double>("cosmology_ede_n");
            if (not(fEDE > 0.0 and fEDE < 0.5))
                throw std::runtime_error("EDE: need 0 < cosmology_ede_fEDE < 0.5");
            if (not(zc > 10.0 and zc < 1e7))
                throw std::runtime_error("EDE: need 10 < cosmology_ede_zc < 1e7");
            if (not(theta > 0.0 and theta < M_PI))
                throw std::runtime_error("EDE: need 0 < cosmology_ede_theta_ini < pi");
            if (not(n > 1.0))
                throw std::runtime_error("EDE: need cosmology_ede_n > 1");
            model_parameters = {{"ede_fEDE", fEDE}, {"ede_zc", zc}, {"ede_theta_i", theta}, {"ede_n", n}};
        } else if (name == "Tabulated") {
            registry_name = "";
            tabulated_filename = param.get<std::string>("cosmology_tabulated_filename");
            tabulated_format = param.get<std::string>("cosmology_tabulated_format");
            if (tabulated_format != "H" and tabulated_format != "CLASS" and tabulated_format != "w")
                throw std::runtime_error("Tabulated cosmology: unknown cosmology_tabulated_format = " +
                                         tabulated_format + " (H, CLASS, w)");
        } else {
            throw std::runtime_error("Unknown cosmology_model [" + name + "]");
        }
    }

    //========================================================================
    // Build the background
    //========================================================================
    void init() {
        namespace BG = FML::COSMOLOGY::BACKGROUND;
        BG::BackgroundParameters p;
        p.h = h;
        p.Omega_b = Omegab;
        p.Omega_cdm = OmegaCDM;
        p.Omega_k = OmegaK;
        p.T_CMB = TCMB_kelvin;
        p.N_eff = Neff;
        if (OmegaMNu > 0.0) {
            p.m_nu_eV.assign(int(N_nu), Mnu_eV / N_nu);
            p.T_ncdm_over_TCMB = Tnu_kelvin / TCMB_kelvin;
        }
        const BG::BackgroundRange range{std::log(1e-10), std::log(100.0), 200};

        if (name == "DGP") {
            const double rho0 = BG::density_today(BG::standard_species(p));
            const double OmegaL = BG::DGPFieldEquations::OmegaLambda(model_parameters.at("Omega_rc"), 1, rho0);
            if (std::fabs(OmegaL) > 1e-6 and FML::ThisTask == 0)
                std::cout << "Warning: DGP: cosmology_dgp_OmegaRC does not give E(0) = 1 by itself: a cosmological "
                             "constant OmegaLambda = "
                          << OmegaL << " is added (cosmology_dgp_OmegaRC = " << 0.25 * (1.0 - rho0) * (1.0 - rho0)
                          << " would need none)\n";
        }
        if (name == "Tabulated")
            bg = make_tabulated(p, range);
        else
            bg = BG::make_background(registry_name, p, model_parameters, range);

        // All the dark energy today (OmegaM, OmegaR, OmegaNu, OmegaK are the parameters)
        OmegaLambda = bg->Omega(BG::Component::DarkEnergy, 0.0);
    }

    //========================================================================
    // Info and output
    //========================================================================
    void info() const {
        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Cosmology [" << name << "]\n";
            std::cout << "# Omegab                  : " << Omegab << "\n";
            std::cout << "# OmegaM                  : " << OmegaM << "\n";
            std::cout << "# OmegaMNu                : " << OmegaMNu << "\n";
            std::cout << "# OmegaCDM                : " << OmegaCDM << "\n";
            std::cout << "# OmegaLambda             : " << OmegaLambda << "\n";
            std::cout << "# OmegaR                  : " << OmegaR << "\n";
            std::cout << "# OmegaNu                 : " << OmegaNu << "\n";
            std::cout << "# OmegaRtot               : " << OmegaRtot << "\n";
            std::cout << "# OmegaK                  : " << OmegaK << "\n";
            std::cout << "# h                       : " << h << "\n";
            std::cout << "# N_nu                    : " << N_nu << "\n";
            std::cout << "# Neff                    : " << Neff << "\n";
            std::cout << "# Mnu                     : " << Mnu_eV << " eV\n";
            std::cout << "# TCMB                    : " << TCMB_kelvin << " K\n";
            std::cout << "# Tnu                     : " << Tnu_kelvin << " K\n";
            std::cout << "# As                      : " << As << "\n";
            std::cout << "# ns                      : " << ns << "\n";
            std::cout << "# kpivot                  : " << kpivot_mpc << " 1/Mpc\n";
            for (const auto & [key, value] : model_parameters)
                std::cout << "# " << std::left << std::setw(24) << key << ": " << value << "\n" << std::right;
            if (name == "Tabulated")
                std::cout << "# Table                   : " << tabulated_filename << " (format " << tabulated_format
                          << ")\n";
            if (bg)
                std::cout << "# Background              : " << bg->description() << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    /// A table of the background quantities from a = alow to ahigh
    void output(std::string filename) const {
        std::ofstream fp(filename.c_str());
        if (not fp.is_open()) {
            std::cout << "Cosmology:: Error opening file " << filename << " for output\n";
            return;
        }
        const std::vector<std::string> columns{
            "a", "H/H0", "dlogH/dloga", "OmegaM", "OmegaR", "OmegaNu", "OmegaMNu", "OmegaNu_exact", "OmegaLambda"};
        fp << "#";
        for (size_t i = 0; i < columns.size(); i++)
            fp << (i > 0 ? " " : "") << std::setw(15) << columns[i];
        fp << "\n";
        for (int i = 0; i < npts_loga; i++) {
            const double a = std::exp(std::log(alow) + std::log(ahigh / alow) * i / double(npts_loga - 1));
            const double values[] = {a,
                                     HoverH0_of_a(a),
                                     dlogHdloga_of_a(a),
                                     get_OmegaM(a),
                                     get_OmegaR(a),
                                     get_OmegaNu(a),
                                     get_OmegaMNu(a),
                                     get_OmegaNu_exact(a),
                                     get_OmegaLambda(a)};
            for (double v : values)
                fp << " " << std::setw(15) << v;
            fp << "\n";
        }
    }

    //========================================================================
    // The background
    //========================================================================
    double HoverH0_of_a(double a) const { return bg->E(std::log(a)); }
    double dlogHdloga_of_a(double a) const { return bg->dlogEdx(std::log(a)); }
    /// The background itself (e.g. for the fields of a theory: background().field_equations_as<...>())
    const Background & background() const { return *bg; }

    //========================================================================
    // Density parameters (see the class comment)
    //========================================================================
    double get_fMNu() const { return OmegaMNu / OmegaM; }
    double get_OmegaMNu(double a = 1.0) const { return a == 1.0 ? OmegaMNu : OmegaMNu / (a * a * a * E2(a)); }
    double get_Omegab(double a = 1.0) const { return a == 1.0 ? Omegab : Omegab / (a * a * a * E2(a)); }
    double get_OmegaM(double a = 1.0) const { return a == 1.0 ? OmegaM : OmegaM / (a * a * a * E2(a)); }
    double get_OmegaCDM(double a = 1.0) const { return a == 1.0 ? OmegaCDM : OmegaCDM / (a * a * a * E2(a)); }
    double get_OmegaR(double a = 1.0) const { return a == 1.0 ? OmegaR : OmegaR / (a * a * a * a * E2(a)); }
    double get_OmegaNu(double a = 1.0) const { return a == 1.0 ? OmegaNu : OmegaNu / (a * a * a * a * E2(a)); }
    double get_OmegaRtot(double a = 1.0) const { return a == 1.0 ? OmegaRtot : OmegaRtot / (a * a * a * a * E2(a)); }
    double get_OmegaK(double a = 1.0) const { return a == 1.0 ? OmegaK : OmegaK / (a * a * E2(a)); }
    /// All the neutrinos (massive and massless), the actual density
    double get_OmegaNu_exact(double a = 1.0) const {
        namespace BG = FML::COSMOLOGY::BACKGROUND;
        const double x = std::log(a);
        return bg->Omega(BG::Component::MassiveNeutrinos, x) + bg->Omega(BG::Component::MasslessNeutrinos, x);
    }
    /// All the dark energy (cosmological constant, fluid or field), the actual density
    double get_OmegaLambda(double a = 1.0) const {
        return a == 1.0 ? OmegaLambda : bg->Omega(FML::COSMOLOGY::BACKGROUND::Component::DarkEnergy, std::log(a));
    }

    /// The growth suppression of the large-scale potential today, g = D(a = 1) / D_EdS(a = 1) with D normalized to a in
    /// the matter era (Heath 1977: D(a) = 5/2 OmegaM E(a) int_0^a da / (a E)^3 for matter, curvature and a cosmological
    /// constant; radiation and the equation of state of the dark energy are ignored here, sub-percent for g)
    double get_potential_growth_suppression_today() const {
        const double OmegaL = 1.0 - OmegaM - OmegaK;
        auto E = [&](double a) { return std::sqrt(OmegaM / (a * a * a) + OmegaK / (a * a) + OmegaL); };
        // Simpson in a: the integrand (a E)^-3 goes as a^(3/2) / OmegaM^(3/2) for a -> 0
        const int n = 20000;
        const double da = 1.0 / n;
        double integral = 0.0;
        for (int i = 0; i <= n; i++) {
            const double a = i * da;
            const double f = a == 0.0 ? 0.0 : std::pow(a * E(a), -3);
            integral += f * (i == 0 or i == n ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0));
        }
        integral *= da / 3.0;
        return 2.5 * OmegaM * E(1.0) * integral;
    }

    //========================================================================
    // Neutrinos
    //========================================================================
    /// Neutrino temperature in eV
    double get_neutrino_temperature_eV(double a) const {
        using namespace FML::UTILS::SI;
        return (Tnu_kelvin * k_b / eV) / a;
    }
    /// Sound speed over c in the non-relativistic limit (1408.2995), at most that of radiation (mass of one neutrino)
    double get_neutrino_sound_speed_cs_over_c(double a) const {
        return std::min(nu_sound_speed_factor * get_neutrino_temperature_eV(a) / (Mnu_eV / N_nu), 1.0 / std::sqrt(3.0));
    }
    /// Free streaming scale for the neutrinos (1408.2995)
    double get_neutrino_free_streaming_scale_hmpc(double a) const {
        return std::sqrt(1.5 * OmegaM / a) / get_neutrino_sound_speed_cs_over_c(a) * H0_hmpc;
    }

    //========================================================================
    // Primordial power spectrum and the nonlinear boost
    //========================================================================
    double get_primordial_pofk(double k_hmpc) const {
        return 2.0 * M_PI * M_PI / (k_hmpc * k_hmpc * k_hmpc) * As * std::pow(h * k_hmpc / kpivot_mpc, ns - 1.0);
    }

    /// Pnl/Plinear from the EuclidEmulator2 for LCDM and w0waCDM (empty spline for the other models)
    Spline get_nonlinear_matter_power_spectrum_boost(double redshift) const {
        if (name != "LCDM" and name != "w0waCDM")
            return Spline();
        // EuclidEmulator2 takes A_s at the pivot 0.05/Mpc: convert from our pivot (the factor is 1 for kpivot = 0.05)
        const double As_ee2 = As * std::pow(0.05 / kpivot_mpc, ns - 1.0);
        return ee2_boost.boost(Omegab, OmegaM, Mnu_eV, ns, h, w0, wa, As_ee2, redshift);
    }

    //========================================================================
    // Parameters
    //========================================================================
    double get_h() const { return h; }
    double get_Mnu_eV() const { return Mnu_eV; }
    double get_N_nu() const { return N_nu; }
    double get_As() const { return As; }
    double get_ns() const { return ns; }
    double get_TCMB_kelvin() const { return TCMB_kelvin; }
    double get_Neff() const { return Neff; }
    double get_kpivot_mpc() const { return kpivot_mpc; }
    std::string get_name() const { return name; }
    double get_H0_hmpc() const { return H0_hmpc; }

    void set_As(double _As) { As = _As; }
    void set_ns(double _ns) { ns = _ns; }
    void set_kpivot_mpc(double _kpivot_mpc) { kpivot_mpc = _kpivot_mpc; }

  private:
    std::string name;
    std::string registry_name;
    std::map<std::string, double> model_parameters;
    std::shared_ptr<const Background> bg;

    const double H0_hmpc = 1.0 / FML::UTILS::SI::c_over_H0_mpch;
    double h;                 // Hubble parameter (little h)
    double OmegaMNu;          // Massive neutrinos (in the matter era)
    double Omegab;            // Baryons
    double OmegaM;            // Total matter (in the matter era)
    double OmegaCDM;          // Cold dark matter
    double OmegaLambda{0.0};  // All the dark energy today
    double OmegaR;            // Photons
    double OmegaNu;           // Neutrinos (as if massless; density set by Neff)
    double OmegaRtot;         // Total relativistic (in the radiation era)
    double OmegaK;            // Curvature
    double Neff;              // Effective number of non-photon relativistic species (3.046)
    double TCMB_kelvin;       // Temperature of the CMB today in Kelvin
    double Tnu_kelvin;        // Temperature of the neutrinos today in Kelvin. Derived from Neff and TCMB
    double Mnu_eV;            // Sum of the neutrino masses in eV. Derived from OmegaMNu and h
    const double N_nu{3};     // Number of neutrinos (3)
    double w0{-1.0}, wa{0.0}; // w0waCDM (for the nonlinear boost)
    std::string tabulated_filename, tabulated_format;

    // zeta(3), zeta(4), zeta(5): the non-relativistic neutrino mass relation and the sound speed
    static constexpr double zeta3 = 1.2020569031595942, zeta4 = 1.0823232337111382, zeta5 = 1.0369277551433699;
    const double twoeta3{3.0 / 2.0 * zeta3};
    const double sixeta4{7.0 / 120.0 * M_PI * M_PI * M_PI * M_PI};
    const double sixzeta4{6.0 * zeta4};
    const double nu_sound_speed_factor{std::sqrt(25.0 * zeta5 / zeta3 / 3.0)};

    // The nonlinear boost (built once, see EE2Boost.h)
    EE2BoostCache ee2_boost;

    // Primordial power spectrum
    double As;
    double ns;
    double kpivot_mpc;

    // The range of the output table (output())
    double alow = 1e-4;
    const double ahigh = 1e1;
    const int npts_loga = 1000;

    double E2(double a) const {
        const double E = HoverH0_of_a(a);
        return E * E;
    }

    //========================================================================
    // Tabulated: read the table (format H: z, H/H0; CLASS: the columns z and H [1/Mpc] of a CLASS background file,
    // normalized by H at z = 0; w: a, w) and make the background
    //========================================================================
    std::shared_ptr<const Background> make_tabulated(const FML::COSMOLOGY::BACKGROUND::BackgroundParameters & p,
                                                     const FML::COSMOLOGY::BACKGROUND::BackgroundRange & range) const {
        namespace BG = FML::COSMOLOGY::BACKGROUND;
        int icol_x = 0, icol_y = 1;
        if (tabulated_format == "CLASS")
            std::tie(icol_x, icol_y) = class_columns();
        auto data = read_columns(icol_x, icol_y);
        std::sort(data.begin(), data.end());
        data.erase(std::unique(data.begin(),
                               data.end(),
                               [](const auto & l, const auto & r) { return l.first == r.first; }),
                   data.end());
        std::vector<double> x, y;
        for (auto & d : data) {
            x.push_back(d.first);
            y.push_back(d.second);
        }
        if (tabulated_format == "w") {
            if (not(x.front() > 0.0))
                throw std::runtime_error("Tabulated cosmology: the scale factors in the w table must be > 0");
            return BG::make_tabulated_w(p, x, y, range);
        }
        // x = z (increasing), y = H
        if (x.front() > 1e-6)
            throw std::runtime_error("Tabulated cosmology: the H table in " + tabulated_filename +
                                     " must reach z = 0 (it starts at z = " + std::to_string(x.front()) + ")");
        if (tabulated_format == "CLASS") {
            const double H0 = y.front();
            for (auto & v : y)
                v /= H0;
        } else if (std::fabs(y.front() - 1.0) > 1e-4) {
            throw std::runtime_error("Tabulated cosmology: H/H0 at z = 0 is " + std::to_string(y.front()) + " in " +
                                     tabulated_filename + " (should be 1)");
        }
        for (double v : y)
            if (not(v > 0.0))
                throw std::runtime_error("Tabulated cosmology: H must be > 0 in " + tabulated_filename);
        auto b = BG::make_tabulated_E(p, x, y, range);
        if (FML::ThisTask == 0) {
            double max_OmegaDE = 0.0, z_max = 0.0;
            for (double z : x)
                if (z > 10.0) {
                    const double O = std::fabs(b->Omega(BG::Component::DarkEnergy, -std::log1p(z)));
                    if (O > max_OmegaDE) {
                        max_OmegaDE = O;
                        z_max = z;
                    }
                }
            std::cout << "Tabulated cosmology: " << tabulated_filename << " (z = " << x.front() << " -> " << x.back()
                      << "), largest |OmegaDE(z)| for z > 10 is " << max_OmegaDE << " at z = " << z_max
                      << " (should be small unless the table has early dark energy or different matter/radiation "
                         "than the parameter file)\n";
        }
        return b;
    }

    std::vector<std::pair<double, double>> read_columns(int icol_x, int icol_y) const {
        std::ifstream fp(tabulated_filename);
        if (not fp.is_open())
            throw std::runtime_error("Tabulated cosmology: cannot open " + tabulated_filename);
        std::vector<std::pair<double, double>> data;
        std::string line;
        while (std::getline(fp, line)) {
            const auto first = line.find_first_not_of(" \t");
            if (first == std::string::npos or line[first] == '#')
                continue;
            std::istringstream ss(line);
            std::vector<double> values;
            double v;
            while (ss >> v)
                values.push_back(v);
            if (int(values.size()) <= std::max(icol_x, icol_y))
                throw std::runtime_error("Tabulated cosmology: too few columns in " + tabulated_filename + ": " + line);
            data.push_back({values[icol_x], values[icol_y]});
        }
        if (data.size() < 4)
            throw std::runtime_error("Tabulated cosmology: need at least 4 rows in " + tabulated_filename);
        return data;
    }

    /// The columns z and H [1/Mpc] of a CLASS background file (from its header)
    std::pair<int, int> class_columns() const {
        std::ifstream fp(tabulated_filename);
        if (not fp.is_open())
            throw std::runtime_error("Tabulated cosmology: cannot open " + tabulated_filename);
        std::string line;
        int iz = -1, iH = -1;
        while (std::getline(fp, line)) {
            if (line.find("1:z") == std::string::npos)
                continue;
            std::size_t pos = 0;
            while ((pos = line.find(':', pos)) != std::string::npos) {
                std::size_t start = pos;
                while (start > 0 and std::isdigit(static_cast<unsigned char>(line[start - 1])))
                    start--;
                if (start == pos) {
                    pos++;
                    continue;
                }
                const int icol = std::stoi(line.substr(start, pos - start)) - 1;
                std::size_t end = line.find(':', pos + 1);
                std::string label = line.substr(pos + 1, (end == std::string::npos ? line.size() : end) - pos - 1);
                while (not label.empty() and
                       (std::isdigit(static_cast<unsigned char>(label.back())) or label.back() == ' '))
                    label.pop_back();
                if (label == "z")
                    iz = icol;
                if (label == "H [1/Mpc]")
                    iH = icol;
                pos++;
            }
        }
        if (iz < 0 or iH < 0)
            throw std::runtime_error("Tabulated cosmology: did not find the columns 'z' and 'H [1/Mpc]' in the "
                                     "CLASS file " +
                                     tabulated_filename);
        return {iz, iH};
    }
};

#endif
