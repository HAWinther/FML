#ifndef COSMOLOGY_HEADER
#define COSMOLOGY_HEADER

#include "EE2Boost.h"
#include "Neutrinos.h"
#include "ParameterDefaults.h"
#include "ParameterFields.h"

#include <FML/Cosmology/Background/BackgroundModel_Tabulated.h>
#include <FML/Cosmology/Background/Presets.h>
#include <FML/Cosmology/Background/TabulatedBackground.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/Global/Global.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>
#include <FML/Units/Units.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
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
///   DarkScattering -> dark_scattering (cosmology_ds_w, _xi, _cs2): wCDM with elastic dark matter - dark energy
///                    scattering (Simpson 2010; Baldi & Simpson 2015). The scattering is a velocity friction of the
///                    dark matter (velocity_friction(a)); cs2 (the sound speed of the dark energy) is only used by the
///                    internal Einstein-Boltzmann solver for the IC
///   CovariantDarkScattering -> covariant_dark_scattering (cosmology_ds_w, cosmology_cds_alpha, cosmology_ds_cs2):
///                    wCDM with the covariantised dark Thomson-like scattering (Beltran Jimenez et al. 2020), the
///                    dark-matter drag Gamma = alpha a^3 / (Omega_c0 E) per ln a (velocity_friction, as DarkScattering)
///   CoupledQuintessence -> coupled_quintessence (cosmology_cq_lambda, _beta): a scalar field with V ~ exp(-lambda
///                    phi) coupled to the CDM, whose particles have the mass m ~ exp(-beta phi) (Wetterich 1995;
///                    Amendola 2000): the background, rho_c(a) = Omega_cdm cdm_mass_factor(a) a^-3. The force and the
///                    velocity term of the CDM are in the gravity model CoupledQuintessence (needs this cosmology)
///   IDE           -> ide (cosmology_ide_w, _xi, _cs2): interacting dark energy, the energy exchange Q = xi H rho_DE
///                    along the dark-matter 4-velocity (Valiviita, Majerotto & Maartens 2008). The CDM particles have
///                    their own masses: each gains xi/3 rho_DE times the change of its physical volume
///                    (Simulation::update_ide_masses); their velocities are not changed. Mean mass cdm_mass_factor(a)
///   CubicGalileon -> cubic_galileon: the cubic Galileon on its tracker (no free parameter beyond those of LCDM; the
///                    dark energy is the Galileon). The force is the gravity model CubicGalileon (needs this cosmology)
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
    /// The cosmology_* parameters, listed once in fields() (the parameter table, ParameterSchema.h, and the reading,
    /// read_parameters, come from it; see ParameterFields.h). cosmology_model itself selects the Cosmology
    /// (Models.h). The parameters of a model are only read for that model (for_cosmology)
    //========================================================================
    struct Settings {
        double Omegab, OmegaMNu, OmegaCDM, OmegaK, Neffective, TCMB_kelvin, h;
        std::vector<double> neutrino_masses_eV;
        double JBD_wBD, JBD_GeffG_today;
        std::string JBD_density_parameter_definition;
        double w0, wa;
        std::string quintessence_potential;
        double quintessence_phi_ini, quintessence_dphidlna_ini, quintessence_lambda, quintessence_alpha, quintessence_f,
            quintessence_mu, quintessence_p;
        double ede_fEDE, ede_zc, ede_theta_ini, ede_n;
        std::string tabulated_filename, tabulated_format;
        double dgp_OmegaRC;
        double ds_w, ds_xi, ds_cs2;
        double cds_alpha;
        double cq_lambda, cq_beta;
        double ide_w, ide_xi, ide_cs2;
        double As, ns, kpivot_mpc; // The primordial power-spectrum

        /// The parameters of the background and of the models
        template <class F>
        void fields(F && f) {
            f.required("cosmology_Omegab", Omegab, 0.0, "Baryon density parameter");
            f.optional("cosmology_OmegaMNu",
                       OmegaMNu,
                       0.0,
                       "Massive neutrino density parameter (3 equal masses; or use cosmology_neutrino_masses_eV)");
            f.optional("cosmology_neutrino_masses_eV",
                       neutrino_masses_eV,
                       std::vector<double>{},
                       "The masses of the (up to 3) neutrinos in eV, e.g. {0.0, 0.0086, 0.05} (0 or not listed: "
                       "massless). Instead of cosmology_OmegaMNu (which then follows from the masses)");
            f.required("cosmology_OmegaCDM", OmegaCDM, 0.0, "CDM density parameter");
            f.required("cosmology_OmegaK", OmegaK, 0.0, "Curvature density parameter (-k/H0^2)");
            f.required("cosmology_Neffective", Neffective, 3.046, "Effective number of relativistic species");
            f.required("cosmology_TCMB_kelvin", TCMB_kelvin, 2.7255, "Temperature of the CMB today (Kelvin)");
            f.required("cosmology_h", h, 0.0, "Hubble parameter H0 = 100h km/s/Mpc");
            f.for_cosmology({"JBD"}, [&]() {
                f.required("cosmology_JBD_wBD", JBD_wBD, 10000.0, "The JBD parameter (wBD -> infinity gives GR)");
                f.optional("cosmology_JBD_GeffG_today",
                           JBD_GeffG_today,
                           1.0,
                           "The value of G today relative to Newtons constant (GeffG_today != 1 effectively gives a "
                           "different Newtons constant)");
                f.optional("cosmology_JBD_density_parameter_definition",
                           JBD_density_parameter_definition,
                           "hi-class",
                           "Definition of the density parameters Omega = 8pi G_* rho / 3H0^2 (which G_* to use)",
                           {"Gbare", "Gnewton", "Gtoday", "hi-class"});
            });
            f.for_cosmology({"w0waCDM"}, [&]() {
                f.required("cosmology_w0", w0, -1.0, "Dark energy equation of state w(a) = w0 + wa(1-a): w0");
                f.required("cosmology_wa", wa, 0.0, "Dark energy equation of state w(a) = w0 + wa(1-a): wa");
            });
            f.for_cosmology({"Quintessence"}, [&]() {
                f.required("cosmology_quintessence_potential",
                           quintessence_potential,
                           "exponential",
                           "The potential V = V0 u(phi), phi in units of Mpl: exponential u = exp(-lambda phi), "
                           "inverse_power_law u = phi^-alpha, pngb u = 1 + cos(phi/f), hilltop u = (1 - (phi/mu)^p)^2. "
                           "V0 is found so that OmegaDE today = 1 - OmegaM - OmegaR - OmegaK",
                           {"exponential", "inverse_power_law", "pngb", "hilltop"});
                f.optional("cosmology_quintessence_phi_ini",
                           quintessence_phi_ini,
                           0.0,
                           "The initial value of the field (units of Mpl; deep in the radiation era)");
                f.optional("cosmology_quintessence_dphidlna_ini",
                           quintessence_dphidlna_ini,
                           0.0,
                           "The initial dphi/dlna (0: frozen)");
                f.when(SIMCODE::is_one_of("cosmology_quintessence_potential", {"exponential"}),
                       "cosmology_quintessence_potential = exponential",
                       [&]() {
                           f.required("cosmology_quintessence_lambda", quintessence_lambda, 1.0, "The slope lambda");
                       });
                f.when(
                    SIMCODE::is_one_of("cosmology_quintessence_potential", {"inverse_power_law"}),
                    "cosmology_quintessence_potential = inverse_power_law",
                    [&]() { f.required("cosmology_quintessence_alpha", quintessence_alpha, 1.0, "The power alpha"); });
                f.when(SIMCODE::is_one_of("cosmology_quintessence_potential", {"pngb"}),
                       "cosmology_quintessence_potential = pngb",
                       [&]() {
                           f.required("cosmology_quintessence_f", quintessence_f, 1.0, "The scale f (units of Mpl)");
                       });
                f.when(SIMCODE::is_one_of("cosmology_quintessence_potential", {"hilltop"}),
                       "cosmology_quintessence_potential = hilltop",
                       [&]() {
                           f.required("cosmology_quintessence_mu",
                                      quintessence_mu,
                                      1.0,
                                      "The position of the minimum mu (units of Mpl)");
                           f.required("cosmology_quintessence_p", quintessence_p, 2.0, "The power p");
                       });
            });
            f.for_cosmology({"EDE"}, [&]() {
                f.required("cosmology_ede_fEDE",
                           ede_fEDE,
                           0.1,
                           "Early dark energy: the maximum of rho_EDE / rho_total");
                f.required("cosmology_ede_zc", ede_zc, 3500.0, "Early dark energy: the redshift of the maximum");
                f.optional("cosmology_ede_theta_ini",
                           ede_theta_ini,
                           2.8,
                           "Early dark energy: the initial angle phi_ini / f");
                f.optional("cosmology_ede_n", ede_n, 3.0, "Early dark energy: V = m^2 f^2 (1 - cos(phi/f))^n");
            });
            f.for_cosmology({"Tabulated"}, [&]() {
                f.required("cosmology_tabulated_filename",
                           tabulated_filename,
                           std::string(""),
                           "The file with the background");
                f.required("cosmology_tabulated_format",
                           tabulated_format,
                           "H",
                           "H: columns z, H/H0. CLASS: a CLASS background file. w: columns a, w(a) of the dark energy",
                           {"H", "CLASS", "w"});
            });
            f.for_cosmology({"DarkScattering", "CovariantDarkScattering"}, [&]() {
                f.required("cosmology_ds_w",
                           ds_w,
                           -0.9,
                           "Dark scattering: the (constant) equation of state w of the dark energy (!= -1)");
                f.optional("cosmology_ds_cs2",
                           ds_cs2,
                           1.0,
                           "Dark scattering: the rest-frame sound speed squared of the dark energy (only for the IC "
                           "from the internal Einstein-Boltzmann solver)");
            });
            f.for_cosmology({"CovariantDarkScattering"}, [&]() {
                f.required("cosmology_cds_alpha",
                           cds_alpha,
                           1.0,
                           "Covariantised dark Thomson-like scattering: alpha = 8 pi G alpha_bar / (3 H0^3) of the "
                           "interaction Q^mu = alpha_bar (u_DE^mu - u_c^mu) (0: wCDM). The dark matter feels the drag "
                           "alpha a^3 / (Omega_cdm E(a)) per ln a (the CDM+baryon particles f_c = OmegaCDM / OmegaCB "
                           "of it)");
            });
            f.for_cosmology({"DarkScattering"}, [&]() {
                f.required("cosmology_ds_xi",
                           ds_xi,
                           10.0,
                           "Dark scattering: xi = sigma_D / m_c in barn/GeV (0: wCDM). The dark matter feels the drag "
                           "(1 + w) c xi rho_DE / H per ln a (the CDM+baryon particles f_c = OmegaCDM / OmegaCB of it)");
            });
            f.for_cosmology({"CoupledQuintessence"}, [&]() {
                f.required("cosmology_cq_lambda",
                           cq_lambda,
                           0.5,
                           "Coupled quintessence: the slope lambda of the potential V ~ exp(-lambda phi) (phi in "
                           "reduced Planck masses)");
                f.required("cosmology_cq_beta",
                           cq_beta,
                           0.05,
                           "Coupled quintessence: the coupling beta (Amendola's sqrt(2/3) beta): the CDM particles have "
                           "the mass m ~ exp(-beta phi) and feel G (1 + 2 beta^2) from the CDM. 0: uncoupled");
            });
            f.for_cosmology({"IDE"}, [&]() {
                f.required("cosmology_ide_w",
                           ide_w,
                           -0.9,
                           "Interacting dark energy: the (constant) equation of state w of the dark energy (!= -1)");
                f.required("cosmology_ide_xi",
                           ide_xi,
                           -0.1,
                           "Interacting dark energy: the coupling xi of the energy exchange Q = xi H rho_DE along the "
                           "dark-matter 4-velocity (> 0: energy flows to the dark matter; 0: wCDM). Needs the doom "
                           "factor xi / (3 (1 + w)) <= 0 (xi < 0 for w > -1, xi > 0 for w < -1)");
                f.optional(
                    "cosmology_ide_cs2",
                    ide_cs2,
                    1.0,
                    "Interacting dark energy: the rest-frame sound speed squared of the dark energy (only for the "
                    "IC from the internal Einstein-Boltzmann solver)");
            });
            f.for_cosmology({"DGP"}, [&]() {
                f.required("cosmology_dgp_OmegaRC",
                           dgp_OmegaRC,
                           0.0,
                           "The cross-over scale of the self-accelerating DGP model "
                           "(Omega_rc = 1/(4 rc^2 H0^2)). A cosmological constant makes up E(0) = 1 (a warning if "
                           "it is not zero; no dark energy for Omega_rc = (1 - Omega_matter - Omega_radiation)^2/4)");
            });
        }
        /// The primordial power-spectrum
        template <class F>
        void primordial_fields(F && f) {
            f.optional("cosmology_As", As, 2e-9, "Primordial amplitude");
            f.optional("cosmology_ns", ns, 1.0, "Spectral index");
            f.optional("cosmology_kpivot_mpc", kpivot_mpc, 0.05, "Pivot scale (1/Mpc)");
        }
    };

    //========================================================================
    // Read the parameters (the ones of all models, then those of the model)
    //========================================================================
    void read_parameters(ParameterMap & param) {
        namespace BG = FML::COSMOLOGY::BACKGROUND;
        using namespace FML::UTILS::SI;
        SIMCODE::SettingsReader reader{param, name, ""};
        settings.fields(reader);
        settings.primordial_fields(reader);
        const auto & s = settings;

        // The background parameters (used by init(); afterwards everything comes from the background)
        BG::BackgroundParameters p;
        p.h = s.h;
        p.Omega_b = s.Omegab;
        p.Omega_cdm = s.OmegaCDM;
        p.Omega_k = s.OmegaK;
        p.T_CMB = s.TCMB_kelvin;
        p.N_eff = s.Neffective;
        As = s.As;
        ns = s.ns;
        kpivot_mpc = s.kpivot_mpc;

        // The neutrino masses: cosmology_neutrino_masses_eV (up to N_nu, 0 = massless), or N_nu equal masses with the
        // total mass from OmegaMNu (non-relativistic relation). No neutrinos (Neff = 0) needs no masses
        const double OmegaMNu = s.OmegaMNu;
        const auto & masses = s.neutrino_masses_eV;
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
        (this->*model_info(name).read)();
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
        else {
            // Only the parameters of the background (model_parameters can also have those of the perturbations, for
            // the internal Einstein-Boltzmann solver)
            BG::ModelParameters background_parameters;
            for (const auto & q : BG::find_model(registry_name).parameters)
                if (model_parameters.count(q.name))
                    background_parameters[q.name] = model_parameters.at(q.name);
            bg = BG::make_background(registry_name, p, background_parameters, range);
        }
        cq_sector = nullptr;
        if (registry_name == "coupled_quintessence")
            for (const auto & s : bg->species())
                if (const auto * field = dynamic_cast<const BG::CoupledFieldSpecies *>(s.get()))
                    cq_sector = &field->sector();
        ide_cdm = nullptr;
        ide_de = nullptr;
        if (registry_name == "ide")
            for (const auto & s : bg->species()) {
                if (const auto * c = dynamic_cast<const BG::InteractingCDMSpecies *>(s.get()))
                    ide_cdm = c;
                if (const auto * d = dynamic_cast<const BG::InteractingDarkEnergySpecies *>(s.get()))
                    ide_de = d;
            }
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
            if (bg and has_velocity_friction())
                std::cout << "# Velocity friction       : Gamma = " << velocity_friction(1.0) << " (a = 1), "
                          << velocity_friction(0.5) << " (a = 0.5) per ln a, of the CDM+baryon particles ("
                          << velocity_friction_cdm(1.0) << ", " << velocity_friction_cdm(0.5)
                          << " of the CDM alone with particle_baryons)\n";
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
        const std::vector<std::string>
            columns{"a", "H/H0", "dlogH/dloga", "Omega_m", "Omega_r", "Omega_nu", "Omega_cb", "Omega_k", "Omega_de"};
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

    /// Velocity friction of the clustering matter (CDM+baryons): its equation of motion is dv/dlna = -Gamma(a) v +
    /// force in the code velocity v = a^3 E dx/da (e.g. dark matter - dark energy scattering). It enters the growth
    /// equations (GravityModel::compute_growth_factors) and the kick and drift factors (TimeStepping.h). No model
    /// has friction yet: Gamma = 0 and has_velocity_friction() is false (then nothing changes)
    /// DarkScattering: the drag of the dark matter (BACKGROUND::dark_scattering_cdm_drag, the rate of the
    /// Einstein-Boltzmann solver) on the CDM+baryon particles as one fluid: f_c = OmegaCDM / OmegaCB of it (the baryons
    /// do not scatter; momentum conservation of the fluid with v_b = v_c)
    /// (The gravity model uses these by default, GravityModel::velocity_friction; coupled quintessence has its velocity
    /// term in GravityModel_CoupledQuintessence)
    /// CovariantDarkScattering: the drag alpha a^3 / (Omega_c0 E) of the dark matter
    /// (BACKGROUND::covariant_dark_scattering_cdm_drag, the rate of the Einstein-Boltzmann solver)
    bool has_velocity_friction() const {
        return (registry_name == "dark_scattering" and ds_xi_SI != 0.0) or
               (registry_name == "covariant_dark_scattering" and cds_alpha != 0.0);
    }
    double velocity_friction(double a) const {
        if (not has_velocity_friction())
            return 0.0;
        const double f_c = get_OmegaCDM() / (get_OmegaCDM() + get_Omegab());
        return f_c * velocity_friction_cdm(a);
    }
    /// The friction of the CDM alone (CDM and baryons as separate particles, particle_baryons; the baryons have none):
    /// DarkScattering, CovariantDarkScattering: the full drag of the dark matter. 0 without friction
    double velocity_friction_cdm(double a) const {
        if (not has_velocity_friction())
            return 0.0;
        if (registry_name == "covariant_dark_scattering")
            return FML::COSMOLOGY::BACKGROUND::covariant_dark_scattering_cdm_drag(*bg, cds_alpha, std::log(a));
        return FML::COSMOLOGY::BACKGROUND::dark_scattering_cdm_drag(*bg, ds_xi_SI, std::log(a));
    }

    //========================================================================
    // Coupled quintessence: the background of the field and the CDM (the force and the velocity term of the CDM are in
    // GravityModel_CoupledQuintessence). The CDM mass factors are exactly 1 for the other models
    //========================================================================
    bool is_coupled_quintessence() const { return cq_sector != nullptr; }
    /// The coupling beta (0 for the other models)
    double cq_beta() const { return cq_sector ? cq_sector->beta : 0.0; }
    /// dphi/dlna of the field (0 for the other models)
    double cq_dphidlna(double a) const { return cq_sector ? cq_sector->dphidx(std::log(a)) : 0.0; }
    /// The CDM mass relative to today, m(a) / m(1) = rho_c(a) a^3 / Omega_cdm (coupled quintessence: exp(-beta (phi(a)
    /// - phi(1))); interacting dark energy: the mean mass, which also changes with the local expansion)
    double cdm_mass_factor(double a) const {
        if (cq_sector)
            return cq_sector->rho_c(std::log(a)) * a * a * a / get_OmegaCDM();
        if (ide_cdm)
            return ide_cdm->rho(std::log(a)) * a * a * a / get_OmegaCDM();
        return 1.0;
    }
    /// The CDM mass changes with time (coupled quintessence, interacting dark energy)
    bool cdm_mass_varies() const { return cq_sector != nullptr or ide_cdm != nullptr; }

    //========================================================================
    // Interacting dark energy (IDE): energy exchange Q = xi H rho_DE along the dark-matter 4-velocity. The CDM momentum
    // is not changed (no velocity term) and the mass a CDM particle gains is xi/3 rho_DE times the change of its
    // physical volume (the exchange follows the local expansion of the dark matter), see Simulation::
    // update_ide_masses. The CDM mass factors above are the background, rho_c(a) a^3 / Omega_cdm
    //========================================================================
    bool is_ide() const { return ide_cdm != nullptr; }
    /// The coupling xi (0 for the other models)
    double ide_xi() const { return ide_de ? ide_de->xi : 0.0; }
    /// rho_DE(a) a^3 in units of rho_crit today (0 for the other models)
    double ide_rho_de_a3(double a) const { return ide_de ? ide_de->rho(std::log(a)) * a * a * a : 0.0; }
    /// The CDM fraction of the CDM+baryon density at a, rho_c(a) / (rho_c(a) + rho_b(a))
    double cdm_fraction_cb(double a) const {
        const double c = get_OmegaCDM() * cdm_mass_factor(a);
        return c / (c + get_Omegab());
    }
    /// rho_m(a) a^3 / OmegaM of the clustering matter: the time dependence of the Poisson source 1.5 OmegaM a delta
    /// from the CDM mass (exactly 1 except for coupled quintessence)
    double clustering_mass_factor(double a) const {
        if (not cdm_mass_varies())
            return 1.0;
        return (get_OmegaCDM() * cdm_mass_factor(a) + get_Omegab() + get_OmegaMNu()) / get_OmegaM();
    }
    /// rho_cb(a) a^3 / OmegaCB: the mass of the CDM+baryon particles relative to today (exactly 1 except for coupled
    /// quintessence)
    double cb_mass_factor(double a) const {
        if (not cdm_mass_varies())
            return 1.0;
        return (get_OmegaCDM() * cdm_mass_factor(a) + get_Omegab()) / (get_OmegaCDM() + get_Omegab());
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
    // To add a model: a line here, its read function below and its parameters in Settings::fields (for_cosmology)
    //========================================================================
    struct ModelInfo {
        std::string name;              // cosmology_model
        void (Cosmology::*read)();     // sets up the model from its parameters (settings)
        bool early_dark_energy{false}; // the output table goes to a = 1e-9
        bool nonlinear_boost{false};   // has the EuclidEmulator2 nonlinear P(k) boost
        bool tabulated{false};         // the background is read from a table (no registry model)
        bool dgp_crossover{false};     // the DGP crossover scale (a check of E(0) = 1 in init)
    };
    static const std::vector<ModelInfo> & models() {
        static const std::vector<ModelInfo> list{
            {"LCDM", &Cosmology::read_lcdm, false, true},
            {"w0waCDM", &Cosmology::read_w0wa, false, true},
            {"DGP", &Cosmology::read_dgp, false, false, false, true},
            {"JBD", &Cosmology::read_jbd},
            {"Quintessence", &Cosmology::read_quintessence, true},
            {"EDE", &Cosmology::read_ede, true},
            {"DarkScattering", &Cosmology::read_dark_scattering},
            {"CovariantDarkScattering", &Cosmology::read_covariant_dark_scattering},
            {"CoupledQuintessence", &Cosmology::read_coupled_quintessence},
            {"IDE", &Cosmology::read_ide},
            {"CubicGalileon", &Cosmology::read_cubic_galileon},
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

    // The setup of each model from its parameters (settings)
    void read_lcdm() { registry_name = "lcdm"; }
    void read_w0wa() {
        registry_name = "w0wa";
        w0 = settings.w0;
        wa = settings.wa;
        model_parameters = {{"w0", w0}, {"wa", wa}};
    }
    void read_dark_scattering() {
        registry_name = "dark_scattering";
        const double w = settings.ds_w, xi = settings.ds_xi, cs2 = settings.ds_cs2;
        if (std::fabs(1.0 + w) < 1e-6)
            throw std::runtime_error("DarkScattering: need cosmology_ds_w != -1 (w = -1 has no scattering: use LCDM)");
        if (not(xi >= 0.0))
            throw std::runtime_error("DarkScattering: need cosmology_ds_xi >= 0");
        if (not(cs2 > 0.0))
            throw std::runtime_error("DarkScattering: need cosmology_ds_cs2 > 0");
        ds_xi_SI = FML::COSMOLOGY::BACKGROUND::dark_scattering_xi_SI(xi);
        model_parameters = {{"ds_w", w}, {"ds_xi", xi}, {"ds_cs2", cs2}};
    }
    void read_covariant_dark_scattering() {
        registry_name = "covariant_dark_scattering";
        const double w = settings.ds_w, alpha = settings.cds_alpha, cs2 = settings.ds_cs2;
        if (std::fabs(1.0 + w) < 1e-6)
            throw std::runtime_error("CovariantDarkScattering: need cosmology_ds_w != -1 (no interaction for w = -1)");
        if (not(alpha >= 0.0))
            throw std::runtime_error("CovariantDarkScattering: need cosmology_cds_alpha >= 0");
        if (not(cs2 > 0.0))
            throw std::runtime_error("CovariantDarkScattering: need cosmology_ds_cs2 > 0");
        cds_alpha = alpha;
        model_parameters = {{"ds_w", w}, {"cds_alpha", alpha}, {"ds_cs2", cs2}};
    }
    void read_coupled_quintessence() {
        registry_name = "coupled_quintessence";
        if (not std::isfinite(settings.cq_lambda) or not std::isfinite(settings.cq_beta))
            throw std::runtime_error("CoupledQuintessence: need finite cosmology_cq_lambda and cosmology_cq_beta");
        model_parameters = {{"cq_lambda", settings.cq_lambda}, {"cq_beta", settings.cq_beta}};
    }
    void read_ide() {
        registry_name = "ide";
        const double w = settings.ide_w, xi = settings.ide_xi, cs2 = settings.ide_cs2;
        if (std::fabs(1.0 + w) < 1e-6)
            throw std::runtime_error("IDE: need cosmology_ide_w != -1");
        if (xi / (3.0 * (1.0 + w)) > 0.0)
            throw std::runtime_error("IDE: the doom factor xi / (3 (1 + w)) > 0 (large-scale instability of the dark "
                                     "energy perturbations): need xi <= 0 for w > -1 and xi >= 0 for w < -1");
        if (not(cs2 > 0.0))
            throw std::runtime_error("IDE: need cosmology_ide_cs2 > 0");
        model_parameters = {{"ide_w", w}, {"ide_xi", xi}, {"ide_cs2", cs2}};
    }
    void read_cubic_galileon() { registry_name = "cubic_galileon"; }
    void read_dgp() {
        registry_name = "dgp";
        model_parameters = {{"Omega_rc", settings.dgp_OmegaRC}, {"dgp_branch", 1.0}};
    }
    void read_jbd() {
        registry_name = "jbd";
        const auto def = settings.JBD_density_parameter_definition;
        const std::vector<std::string> defs{"hi-class", "Gtoday", "Gnewton", "Gbare"};
        const auto it = std::find(defs.begin(), defs.end(), def);
        if (it == defs.end())
            throw std::runtime_error("JBD: unknown cosmology_JBD_density_parameter_definition = " + def +
                                     " (hi-class, Gtoday, Gnewton, Gbare)");
        model_parameters = {{"bd_omega", settings.JBD_wBD},
                            {"bd_GeffG_today", settings.JBD_GeffG_today},
                            {"bd_density_parameter_definition", double(it - defs.begin())}};
    }
    void read_quintessence() {
        const auto potential = settings.quintessence_potential;
        const double phi_ini = settings.quintessence_phi_ini;
        const double dphi_ini = settings.quintessence_dphidlna_ini;
        model_parameters = {{"quint_phi_ini", phi_ini}, {"quint_dphidlna_ini", dphi_ini}};
        if (potential == "exponential") {
            registry_name = "quintessence";
            model_parameters["quint_lambda"] = settings.quintessence_lambda;
        } else if (potential == "inverse_power_law") {
            registry_name = "quintessence_inverse_power_law";
            model_parameters["quint_alpha"] = settings.quintessence_alpha;
            if (not(phi_ini > 0.0))
                throw std::runtime_error("Quintessence: inverse_power_law needs cosmology_quintessence_phi_ini > 0");
        } else if (potential == "pngb") {
            registry_name = "quintessence_pngb";
            model_parameters["quint_f"] = settings.quintessence_f;
            if (not(model_parameters["quint_f"] > 0.0))
                throw std::runtime_error("Quintessence: pngb needs cosmology_quintessence_f > 0");
        } else if (potential == "hilltop") {
            registry_name = "quintessence_hilltop";
            model_parameters["quint_mu"] = settings.quintessence_mu;
            model_parameters["quint_p"] = settings.quintessence_p;
            if (not(model_parameters["quint_mu"] > 0.0) or not(model_parameters["quint_p"] > 0.0))
                throw std::runtime_error("Quintessence: hilltop needs cosmology_quintessence_mu > 0 and _p > 0");
        } else {
            throw std::runtime_error("Quintessence: unknown cosmology_quintessence_potential = " + potential +
                                     " (exponential, inverse_power_law, pngb, hilltop)");
        }
        if (dphi_ini * dphi_ini >= 6.0)
            throw std::runtime_error("Quintessence: need |cosmology_quintessence_dphidlna_ini| < sqrt(6)");
    }
    void read_ede() {
        registry_name = "ede_zc";
        const double fEDE = settings.ede_fEDE, zc = settings.ede_zc;
        const double theta = settings.ede_theta_ini, n = settings.ede_n;
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
    void read_tabulated() {
        registry_name = "";
        tabulated_filename = settings.tabulated_filename;
        tabulated_format = settings.tabulated_format;
        if (tabulated_format != "H" and tabulated_format != "CLASS" and tabulated_format != "w")
            throw std::runtime_error("Tabulated cosmology: unknown cosmology_tabulated_format = " + tabulated_format +
                                     " (H, CLASS, w)");
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
    Settings settings; // The parameters (read_parameters)
    std::string registry_name;
    std::map<std::string, double> model_parameters;
    std::shared_ptr<const Background> bg;
    SIMCODE::NEUTRINOS::MassiveNeutrinoList massive_neutrinos;

    static constexpr double H0_hmpc = 1.0 / FML::UTILS::SI::c_over_H0_mpch;
    // The parameters read by read_parameters() for init(); after that all of them come from the background
    FML::COSMOLOGY::BACKGROUND::BackgroundParameters input_parameters;
    const double N_nu{3};     // Number of neutrinos (3)
    double w0{-1.0}, wa{0.0}; // w0waCDM (for the nonlinear boost)
    double ds_xi_SI{0.0};     // DarkScattering: xi = sigma_D / m_c in m^2/kg
    double cds_alpha{0.0};    // CovariantDarkScattering: alpha
    // CoupledQuintessence: the background solution of the field and the CDM (owned by bg)
    const FML::COSMOLOGY::BACKGROUND::CoupledDarkSector * cq_sector{nullptr};
    // IDE: the interacting CDM and dark energy (owned by bg)
    const FML::COSMOLOGY::BACKGROUND::InteractingCDMSpecies * ide_cdm{nullptr};
    const FML::COSMOLOGY::BACKGROUND::InteractingDarkEnergySpecies * ide_de{nullptr};
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
        auto data = read_columns();
        std::sort(data.begin(), data.end());
        data.erase(
            std::unique(data.begin(), data.end(), [](const auto & l, const auto & r) { return l.first == r.first; }),
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

    /// The rows (x, y) of the table: the columns z and H [1/Mpc] of a CLASS background file, else the first two
    std::vector<std::pair<double, double>> read_columns() const {
        std::vector<std::pair<double, double>> data;
        if (tabulated_format == "CLASS") {
            const auto table = FML::COSMOLOGY::BACKGROUND::read_class_table(tabulated_filename);
            if (table.find("z") < 0 or table.find("H [1/Mpc]") < 0)
                throw std::runtime_error("Tabulated cosmology: did not find the columns 'z' and 'H [1/Mpc]' in the "
                                         "CLASS file " +
                                         tabulated_filename);
            const auto z = table.column("z"), H = table.column("H [1/Mpc]");
            for (size_t i = 0; i < z.size(); i++)
                data.push_back({z[i], H[i]});
        } else {
            for (auto & values : FML::FILEUTILS::loadtxt(tabulated_filename)) {
                if (values.size() < 2)
                    throw std::runtime_error("Tabulated cosmology: too few columns in " + tabulated_filename + " (" +
                                             std::to_string(values.size()) + " in row " + std::to_string(data.size()) +
                                             ")");
                data.push_back({values[0], values[1]});
            }
        }
        if (data.size() < 4)
            throw std::runtime_error("Tabulated cosmology: need at least 4 rows in " + tabulated_filename);
        return data;
    }
};

#endif
