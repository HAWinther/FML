#ifndef COSMOLOGY_HEADER
#define COSMOLOGY_HEADER

#include "EE2Boost.h"
#include "Neutrinos.h"
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
/// Neutrinos: N_nu = 3 species with the temperature T_nu = T_CMB (Neff/3)^(1/4) (4/11)^(1/3). Their masses are
/// either cosmology_neutrino_masses_eV (up to 3 masses, 0 = massless; the species not listed are massless) or, with
/// cosmology_OmegaMNu > 0, 3 equal masses with the total Mnu from OmegaMNu (non-relativistic relation); otherwise all
/// are massless. get_OmegaMNu() follows from the masses with the non-relativistic relation in both cases.
///
/// Density parameters, two kinds:
///   get_OmegaM(), get_Omegab(), get_OmegaCDM(), get_OmegaMNu(), get_fMNu(), get_OmegaK(), get_OmegaR(),
///   get_OmegaLambda(): today, the input definitions used by the N-body part (Poisson normalization 1.5 OmegaM a,
///   particle masses, f_nu, the transfer-function inputs): OmegaM = Omegab + OmegaCDM + OmegaMNu with OmegaMNu the
///   input (the neutrino mass from it with the non-relativistic relation), OmegaR the photons, OmegaLambda all the
///   dark energy of the background today
///   Omega_m(a), Omega_cb(a), Omega_b(a), Omega_cdm(a), Omega_nu(a), Omega_r(a), Omega_k(a), Omega_de(a), f_nu(a):
///   functions of time, the actual densities of the background (rho_i(a) / rho_crit(a); the massive neutrinos with
///   their Fermi-Dirac energy density, relativistic at early times)
/// Nothing is stored besides the background: the parameters (h, the Omegas today, T_CMB, Neff, the neutrino masses)
/// are those of background().parameters(), OmegaMNu from the masses with the non-relativistic relation, OmegaLambda the
/// dark energy of the background today. The getters need init().
/// background() gives the background itself.
//=============================================================================
class Cosmology {
  public:
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;
    using DVector = FML::INTERPOLATION::SPLINE::DVector;
    using Background = FML::COSMOLOGY::BACKGROUND::ComponentBackground;

    /// model: the name of the cosmology_model
    explicit Cosmology(std::string model = "LCDM") : name(std::move(model)) {
        // (throws for an unknown model)
        if (model_info(name).early_dark_energy)
            alow = 1e-9; // the output table (output()) covers the early times where the dark energy can matter
    }

    /// The names of the cosmologies (cosmology_model) in the order of models()
    static std::vector<std::string> model_names() {
        std::vector<std::string> names;
        for (auto & m : models())
            names.push_back(m.name);
        return names;
    }

    //========================================================================
    // Read the parameters (the ones of all models, then those of the model)
    //========================================================================
    void read_parameters(ParameterMap & param) {
        namespace BG = FML::COSMOLOGY::BACKGROUND;
        using namespace FML::UTILS::SI;
        // The background parameters (used by init(); afterwards everything comes from the background)
        BG::BackgroundParameters p;
        p.h = param.get<double>("cosmology_h");
        p.Omega_b = param.get<double>("cosmology_Omegab");
        p.Omega_cdm = param.get<double>("cosmology_OmegaCDM");
        p.Omega_k = SIMCODE::get_parameter<double>(param, "cosmology_OmegaK");
        p.T_CMB = param.get<double>("cosmology_TCMB_kelvin");
        p.N_eff = param.get<double>("cosmology_Neffective");
        As = param.get<double>("cosmology_As");
        ns = param.get<double>("cosmology_ns");
        kpivot_mpc = param.get<double>("cosmology_kpivot_mpc");

        // The neutrino masses: cosmology_neutrino_masses_eV (up to N_nu, 0 = massless), or N_nu equal masses with the
        // total mass from OmegaMNu (non-relativistic relation). No neutrinos (Neff = 0) needs no masses
        const double OmegaMNu = param.get<double>("cosmology_OmegaMNu");
        const auto masses = SIMCODE::get_parameter<std::vector<double>>(param, "cosmology_neutrino_masses_eV");
        const double OmegaMNu_per_eV = OmegaMNu_per_Mnu_eV(p);
        FML::assert_mpi(masses.size() <= size_t(N_nu), "cosmology_neutrino_masses_eV: at most 3 masses");
        FML::assert_mpi(masses.empty() or OmegaMNu == 0.0,
                        "Give either cosmology_neutrino_masses_eV or cosmology_OmegaMNu (the masses give OmegaMNu)");
        if (not masses.empty()) {
            for (double m : masses) {
                FML::assert_mpi(m >= 0.0, "cosmology_neutrino_masses_eV: the masses must be >= 0");
                if (m > 0.0)
                    p.m_nu_eV.push_back(m);
            }
        } else if (OmegaMNu > 0.0) {
            p.m_nu_eV.assign(int(N_nu), OmegaMNu / OmegaMNu_per_eV / N_nu);
        }
        FML::assert_mpi(OmegaMNu_per_eV > 0.0 or p.m_nu_eV.empty(),
                        "cosmology_Neffective = 0 (no neutrinos) cannot have massive neutrinos");
        if (not p.m_nu_eV.empty())
            p.T_ncdm_over_TCMB = neutrino_temperature_kelvin(p) / p.T_CMB;
        input_parameters = p;

        // The parameters of the model (the read function of the model in models())
        model_parameters.clear();
        (this->*model_info(name).read)(param);
    }

    //========================================================================
    // Build the background
    //========================================================================
    void init() {
        namespace BG = FML::COSMOLOGY::BACKGROUND;
        const BG::BackgroundParameters & p = input_parameters;
        const BG::BackgroundRange range{std::log(1e-10), std::log(100.0), 200};

        if (model_info(name).dgp_crossover) {
            const double rho0 = BG::density_today(BG::standard_species(p));
            const double OmegaL = BG::DGPFieldEquations::OmegaLambda(model_parameters.at("Omega_rc"), 1, rho0);
            if (std::fabs(OmegaL) > 1e-6 and FML::ThisTask == 0)
                std::cout << "Warning: DGP: cosmology_dgp_OmegaRC does not give E(0) = 1 by itself: a cosmological "
                             "constant OmegaLambda = "
                          << OmegaL << " is added (cosmology_dgp_OmegaRC = " << 0.25 * (1.0 - rho0) * (1.0 - rho0)
                          << " would need none)\n";
        }
        if (model_info(name).tabulated)
            bg = make_tabulated(p, range);
        else
            bg = BG::make_background(registry_name, p, model_parameters, range);
        massive_neutrinos.clear();
        for (double m : params().m_nu_eV)
            massive_neutrinos.emplace_back(m, params());
    }

    //========================================================================
    // Info and output
    //========================================================================
    void info() const {
        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Cosmology [" << name << "]\n";
            if (bg) {
                std::cout << "# Omegab                  : " << get_Omegab() << "\n";
                std::cout << "# OmegaM                  : " << get_OmegaM() << "\n";
                std::cout << "# OmegaMNu                : " << get_OmegaMNu() << "\n";
                std::cout << "# OmegaCDM                : " << get_OmegaCDM() << "\n";
                std::cout << "# OmegaLambda             : " << get_OmegaLambda() << "\n";
                std::cout << "# OmegaR                  : " << get_OmegaR() << "\n";
                std::cout << "# OmegaNu (as if massless): " << OmegaNu_massless_equivalent(params()) << "\n";
                std::cout << "# OmegaK                  : " << get_OmegaK() << "\n";
                std::cout << "# h                       : " << get_h() << "\n";
                std::cout << "# N_nu                    : " << N_nu << "\n";
                std::cout << "# Neff                    : " << get_Neff() << "\n";
                std::cout << "# Mnu                     : " << get_Mnu_eV() << " eV";
                if (not get_neutrino_masses_eV().empty()) {
                    std::cout << " (masses";
                    for (double m : get_neutrino_masses_eV())
                        std::cout << " " << m;
                    std::cout << " eV)";
                }
                std::cout << "\n";
                std::cout << "# TCMB                    : " << get_TCMB_kelvin() << " K\n";
                std::cout << "# Tnu                     : " << neutrino_temperature_kelvin(params()) << " K\n";
            }
            std::cout << "# As                      : " << As << "\n";
            std::cout << "# ns                      : " << ns << "\n";
            std::cout << "# kpivot                  : " << kpivot_mpc << " 1/Mpc\n";
            for (const auto & [key, value] : model_parameters)
                std::cout << "# " << std::left << std::setw(24) << key << ": " << value << "\n" << std::right;
            if (model_info(name).tabulated)
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
        // (the columns of OmegaM(a) and OmegaDE(a) as before: 4 and 9)
        const std::vector<std::string> columns{
            "a", "H/H0", "dlogH/dloga", "Omega_m", "Omega_r", "Omega_nu", "Omega_cb", "Omega_k", "Omega_de"};
        fp << "#";
        for (size_t i = 0; i < columns.size(); i++)
            fp << (i > 0 ? " " : "") << std::setw(15) << columns[i];
        fp << "\n";
        for (int i = 0; i < npts_loga; i++) {
            const double a = std::exp(std::log(alow) + std::log(ahigh / alow) * i / double(npts_loga - 1));
            const double values[] = {a,
                                     HoverH0_of_a(a),
                                     dlogHdloga_of_a(a),
                                     Omega_m(a),
                                     Omega_r(a),
                                     Omega_nu(a),
                                     Omega_cb(a),
                                     Omega_k(a),
                                     Omega_de(a)};
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
    // Density parameters today: the input definitions (see the class comment)
    //========================================================================
    double get_OmegaM() const { return get_Omegab() + get_OmegaCDM() + get_OmegaMNu(); }
    double get_Omegab() const { return params().Omega_b; }
    double get_OmegaCDM() const { return params().Omega_cdm; }
    /// The massive neutrinos with the non-relativistic relation (the input cosmology_OmegaMNu)
    double get_OmegaMNu() const {
        const auto & p = params();
        return p.m_nu_eV.empty() ? 0.0 : p.sum_m_nu_eV() * OmegaMNu_per_Mnu_eV(p);
    }
    double get_fMNu() const { return get_OmegaMNu() / get_OmegaM(); }
    double get_OmegaK() const { return params().Omega_k; }
    /// The photons
    double get_OmegaR() const { return params().Omega_gamma(); }
    /// All the dark energy (cosmological constant, fluid or field) today
    double get_OmegaLambda() const { return Omega_de(1.0); }

    //========================================================================
    // Density parameters as functions of time: the actual densities of the background (rho_i(a) / rho_crit(a))
    //========================================================================
    /// Matter: baryons + CDM + massive neutrinos
    double Omega_m(double a) const { return bg->Omega_m(std::log(a)); }
    /// Baryons + CDM
    double Omega_cb(double a) const { return Omega_b(a) + Omega_cdm(a); }
    double Omega_b(double a) const { return Omega(FML::COSMOLOGY::BACKGROUND::Component::Baryons, a); }
    double Omega_cdm(double a) const { return Omega(FML::COSMOLOGY::BACKGROUND::Component::CDM, a); }
    /// All the neutrinos (massive and massless)
    double Omega_nu(double a) const {
        namespace BG = FML::COSMOLOGY::BACKGROUND;
        return Omega(BG::Component::MassiveNeutrinos, a) + Omega(BG::Component::MasslessNeutrinos, a);
    }
    /// Radiation: photons + massless neutrinos
    double Omega_r(double a) const { return bg->Omega_r(std::log(a)); }
    double Omega_k(double a) const { return Omega(FML::COSMOLOGY::BACKGROUND::Component::Curvature, a); }
    /// All the dark energy
    double Omega_de(double a) const { return Omega(FML::COSMOLOGY::BACKGROUND::Component::DarkEnergy, a); }
    /// rho_m(a) a^3 / rho_c0: the OmegaM of the growth equations and the COLA frame at a (OmegaM without massive
    /// neutrinos; larger than get_OmegaM() while they are relativistic)
    double Omega_m_a3(double a) const {
        const double E = HoverH0_of_a(a);
        return Omega_m(a) * E * E * a * a * a;
    }
    /// The massive neutrino fraction of the matter rho_nu(a) / rho_m(a) (its energy density: larger than get_fMNu()
    /// while the neutrinos are relativistic)
    double f_nu(double a) const {
        namespace BG = FML::COSMOLOGY::BACKGROUND;
        return params().m_nu_eV.empty() ? 0.0 : Omega(BG::Component::MassiveNeutrinos, a) / Omega_m(a);
    }

    /// The growth suppression of the large-scale potential today, g = D(a = 1) / D_EdS(a = 1) with D normalized to a in
    /// the matter era (Heath 1977: D(a) = 5/2 OmegaM E(a) int_0^a da / (a E)^3 for matter, curvature and a cosmological
    /// constant; radiation and the equation of state of the dark energy are ignored here, sub-percent for g)
    double get_potential_growth_suppression_today() const {
        const double OmegaM = get_OmegaM(), OmegaK = get_OmegaK();
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
        return (neutrino_temperature_kelvin(params()) * k_b / eV) / a;
    }
    /// The massive neutrino species of the background (one per mass; empty without massive neutrinos)
    const SIMCODE::NEUTRINOS::MassiveNeutrinoList & get_massive_neutrino_species() const {
        params(); // (throws if init() has not been called)
        return massive_neutrinos;
    }
    /// The mean energy per mass <E/m>(a) of the massive neutrinos: their energy density over their mass density in
    /// the background (-> 1 when non-relativistic; 1 without massive neutrinos)
    double get_neutrino_mean_energy_factor(double a) const {
        return SIMCODE::NEUTRINOS::mean_energy_factor(get_massive_neutrino_species(), a);
    }
    /// Sound speed over c in the non-relativistic limit (1408.2995), at most that of radiation (for the mean mass of
    /// the massive species)
    double get_neutrino_sound_speed_cs_over_c(double a) const {
        const double m_mean = get_Mnu_eV() / double(get_neutrino_masses_eV().size());
        return std::min(nu_sound_speed_factor * get_neutrino_temperature_eV(a) / m_mean, 1.0 / std::sqrt(3.0));
    }

    //========================================================================
    // Primordial power spectrum and the nonlinear boost
    //========================================================================
    double get_primordial_pofk(double k_hmpc) const {
        return 2.0 * M_PI * M_PI / (k_hmpc * k_hmpc * k_hmpc) * As * std::pow(get_h() * k_hmpc / kpivot_mpc, ns - 1.0);
    }

    /// Pnl/Plinear from the EuclidEmulator2 for LCDM and w0waCDM (empty spline for the other models)
    Spline get_nonlinear_matter_power_spectrum_boost(double redshift) const {
        if (not model_info(name).nonlinear_boost)
            return Spline();
        // EuclidEmulator2 takes A_s at the pivot 0.05/Mpc: convert from our pivot (the factor is 1 for kpivot = 0.05)
        const double As_ee2 = As * std::pow(0.05 / kpivot_mpc, ns - 1.0);
        return ee2_boost.boost(get_Omegab(), get_OmegaM(), get_Mnu_eV(), ns, get_h(), w0, wa, As_ee2, redshift);
    }

    //========================================================================
    // Parameters
    //========================================================================
    double get_h() const { return params().h; }
    double get_Mnu_eV() const { return params().sum_m_nu_eV(); }
    /// The masses of the massive species (equal or not; the other N_nu - size() are massless)
    const std::vector<double> & get_neutrino_masses_eV() const { return params().m_nu_eV; }
    double get_N_nu() const { return N_nu; }
    double get_As() const { return As; }
    double get_ns() const { return ns; }
    double get_TCMB_kelvin() const { return params().T_CMB; }
    double get_Neff() const { return params().N_eff; }
    double get_kpivot_mpc() const { return kpivot_mpc; }
    std::string get_name() const { return name; }
    /// The name of the model in the background (and perturbation) registry ("" for Tabulated) and its parameters
    const std::string & get_registry_name() const { return registry_name; }
    const std::map<std::string, double> & get_model_parameters() const { return model_parameters; }
    double get_H0_hmpc() const { return H0_hmpc; }

    void set_As(double _As) { As = _As; }

  private:
    //========================================================================
    // The cosmologies (cosmology_model): the one list of the models, with how their parameters are read (each sets
    // registry_name, the model of FML's background registry, and model_parameters) and what is special about them.
    // To add a model: a line here and its read function below (and its parameters in ParameterSchema.h)
    //========================================================================
    struct ModelInfo {
        std::string name;                          // cosmology_model
        void (Cosmology::*read)(ParameterMap &);   // reads the parameters of the model
        bool early_dark_energy{false};             // the output table goes to a = 1e-9
        bool nonlinear_boost{false};               // has the EuclidEmulator2 nonlinear P(k) boost
        bool tabulated{false};                     // the background is read from a table (no registry model)
        bool dgp_crossover{false};                 // the DGP crossover scale (a check of E(0) = 1 in init)
    };
    static const std::vector<ModelInfo> & models() {
        static const std::vector<ModelInfo> list{
            {"LCDM", &Cosmology::read_lcdm, false, true},
            {"w0waCDM", &Cosmology::read_w0wa, false, true},
            {"DGP", &Cosmology::read_dgp, false, false, false, true},
            {"JBD", &Cosmology::read_jbd},
            {"Quintessence", &Cosmology::read_quintessence, true},
            {"EDE", &Cosmology::read_ede, true},
            {"Tabulated", &Cosmology::read_tabulated, true, false, true},
        };
        return list;
    }
    static const ModelInfo & model_info(const std::string & model) {
        for (auto & m : models())
            if (m.name == model)
                return m;
        std::string available;
        for (auto & m : models())
            available += " " + m.name;
        throw std::runtime_error("Unknown cosmology [" + model + "]. Available:" + available);
    }

    // The parameters of each model
    void read_lcdm([[maybe_unused]] ParameterMap & param) {
        registry_name = "lcdm";
    }
    void read_w0wa([[maybe_unused]] ParameterMap & param) {
        registry_name = "w0wa";
        w0 = param.get<double>("cosmology_w0");
        wa = param.get<double>("cosmology_wa");
        model_parameters = {{"w0", w0}, {"wa", wa}};
    }
    void read_dgp([[maybe_unused]] ParameterMap & param) {
        registry_name = "dgp";
        model_parameters = {{"Omega_rc", param.get<double>("cosmology_dgp_OmegaRC")}, {"dgp_branch", 1.0}};
    }
    void read_jbd([[maybe_unused]] ParameterMap & param) {
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
    }
    void read_quintessence([[maybe_unused]] ParameterMap & param) {
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
    }
    void read_ede([[maybe_unused]] ParameterMap & param) {
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
    }
    void read_tabulated([[maybe_unused]] ParameterMap & param) {
        registry_name = "";
        tabulated_filename = param.get<std::string>("cosmology_tabulated_filename");
        tabulated_format = param.get<std::string>("cosmology_tabulated_format");
        if (tabulated_format != "H" and tabulated_format != "CLASS" and tabulated_format != "w")
            throw std::runtime_error("Tabulated cosmology: unknown cosmology_tabulated_format = " +
                                     tabulated_format + " (H, CLASS, w)");
    }

    double Omega(FML::COSMOLOGY::BACKGROUND::Component c, double a) const { return bg->Omega(c, std::log(a)); }
    /// The parameters of the background (init() must have been called)
    const FML::COSMOLOGY::BACKGROUND::BackgroundParameters & params() const {
        if (not bg)
            throw std::runtime_error("Cosmology: init() has not been called");
        return bg->parameters();
    }
    /// The neutrino temperature today, T_CMB (Neff/3)^(1/4) (4/11)^(1/3)
    static double neutrino_temperature_kelvin(const FML::COSMOLOGY::BACKGROUND::BackgroundParameters & p) {
        return p.T_CMB * std::pow(p.N_eff / 3.0, 0.25) * std::pow(4.0 / 11.0, 1.0 / 3.0);
    }
    /// The N_nu neutrinos as if massless (density set by Neff)
    double OmegaNu_massless_equivalent(const FML::COSMOLOGY::BACKGROUND::BackgroundParameters & p) const {
        return (7.0 / 8.0) * N_nu * std::pow(neutrino_temperature_kelvin(p) / p.T_CMB, 4) * p.Omega_gamma();
    }
    /// The non-relativistic relation OmegaMNu = Mnu * OmegaMNu_per_Mnu_eV (0 without neutrinos)
    double OmegaMNu_per_Mnu_eV(const FML::COSMOLOGY::BACKGROUND::BackgroundParameters & p) const {
        using namespace FML::UTILS::SI;
        const double Tnu_eV = k_b * neutrino_temperature_kelvin(p) / eV;
        return Tnu_eV > 0.0 ? OmegaNu_massless_equivalent(p) * twoeta3 / sixeta4 / (N_nu * Tnu_eV) : 0.0;
    }

    std::string name;
    std::string registry_name;
    std::map<std::string, double> model_parameters;
    std::shared_ptr<const Background> bg;
    SIMCODE::NEUTRINOS::MassiveNeutrinoList massive_neutrinos;

    const double H0_hmpc = 1.0 / FML::UTILS::SI::c_over_H0_mpch;
    // The parameters read by read_parameters() for init(); after that all of them come from the background
    FML::COSMOLOGY::BACKGROUND::BackgroundParameters input_parameters;
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
