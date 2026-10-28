#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/Models/BransDicke.h>
#include <FML/Cosmology/Models/CoupledQuintessence.h>
#include <FML/Cosmology/Models/CubicGalileon.h>
#include <FML/Cosmology/Models/DGP.h>
#include <FML/Cosmology/Models/DarkScattering.h>
#include <FML/Cosmology/Models/EarlyDarkEnergy.h>
#include <FML/Cosmology/Models/GeneralRelativity.h>
#include <FML/Cosmology/Models/HorndeskiAlpha.h>
#include <FML/Cosmology/Models/HuSawicky.h>
#include <FML/Cosmology/Models/InteractingDarkEnergy.h>
#include <FML/Cosmology/Models/MuSigma.h>
#include <FML/Cosmology/Models/Quintessence.h>
#include <FML/Cosmology/Models/StandardSpecies.h>

#include <FML/Cosmology/Common.h>

#include <cmath>
#include <stdexcept>

namespace FML::COSMOLOGY {

    namespace Models {

        std::vector<std::shared_ptr<const Species>> standard_species(const CosmologyParameters & p) {
            using R = Species::Role;
            std::vector<std::shared_ptr<const Species>> s = {
                std::make_shared<PowerLawSpecies>("baryons", R::Baryons, p.OmegaB, 3.0),
                std::make_shared<ColdDarkMatter>(p.OmegaCDM),
                std::make_shared<PowerLawSpecies>("photons", R::Photons, p.OmegaGamma(), 4.0),
                std::make_shared<MasslessNeutrinos>(p.OmegaNu()),
            };
            if (p.has_massive_nu())
                s.push_back(std::make_shared<MassiveNeutrinoSpecies>(p));
            return s;
        }

        Model lcdm(const CosmologyParameters & p) {
            Model m;
            m.name = "lcdm";
            m.species = standard_species(p);
            m.species.push_back(
                std::make_shared<PowerLawSpecies>("lambda", Species::Role::DarkEnergy, p.OmegaLambda(), 0.0));
            m.gravity = std::make_shared<GeneralRelativity>();
            return m;
        }

        Model musigma(const CosmologyParameters & p) {
            Model m = lcdm(p);
            m.name = "musigma";
            m.gravity = std::make_shared<MuSigmaGravity>(parameter(p, "mu0"), parameter(p, "Sigma0"));
            return m;
        }

        Model dgp(const CosmologyParameters & p) {
            if (p.OmegaK != 0.0)
                throw std::invalid_argument("dgp: only implemented for a flat universe");
            const double b = parameter(p, "dgp_branch");
            if (b != 1.0 && b != -1.0)
                throw std::invalid_argument("dgp: dgp_branch must be +1 or -1");
            const int branch = int(b);
            Model m;
            m.name = "dgp";
            m.species = standard_species(p);
            double rho0 = 0.0;
            for (const auto & s : m.species)
                rho0 += s->rho(0.0);
            double Omega_rc = parameter(p, "Omega_rc");
            const bool lcdm_bg = parameter(p, "dgp_lcdm_background") != 0.0;
            if (lcdm_bg) {
                m = lcdm(p);
                m.name = "dgp";
                m.gravity = std::make_shared<DGPGravity>(Omega_rc, branch, true);
                return m;
            }
            if (Omega_rc == 0.0) { // self-accelerating without dark energy: 1 - 2 sqrt(Omega_rc) = rho0
                if (branch != 1)
                    throw std::invalid_argument("dgp: Omega_rc = 0 (no dark energy) needs dgp_branch = +1");
                Omega_rc = 0.25 * (1.0 - rho0) * (1.0 - rho0);
            }
            const double OmL = DGPGravity::OmegaLambda(Omega_rc, branch, rho0);
            m.species.push_back(std::make_shared<PowerLawSpecies>("lambda", Species::Role::DarkEnergy, OmL, 0.0));
            m.gravity = std::make_shared<DGPGravity>(Omega_rc, branch);
            return m;
        }

        Model quintessence(const CosmologyParameters & p) {
            Model m;
            m.name = "quintessence";
            m.species = standard_species(p);
            m.species.push_back(std::make_shared<QuintessenceSpecies>(parameter(p, "quint_lambda"),
                                                                      m.species,
                                                                      p.OmegaK,
                                                                      p.OmegaLambda()));
            m.gravity = std::make_shared<GeneralRelativity>();
            return m;
        }

        Model coupled_quintessence(const CosmologyParameters & p) {
            Model m;
            m.name = "coupled_quintessence";
            const auto standard = standard_species(p);
            std::vector<std::shared_ptr<const Species>> others;
            for (const auto & s : standard)
                if (s->role() != Species::Role::CDM)
                    others.push_back(s);
            auto d = std::make_shared<const CoupledDarkSector>(parameter(p, "cq_lambda"),
                                                               parameter(p, "cq_beta"),
                                                               others,
                                                               p.OmegaK,
                                                               p.OmegaCDM,
                                                               p.OmegaLambda());
            for (const auto & s : standard)
                m.species.push_back(s->role() == Species::Role::CDM ? std::make_shared<CoupledCDM>(d) : s);
            m.species.push_back(std::make_shared<CoupledField>(d));
            m.gravity = std::make_shared<GeneralRelativity>();
            return m;
        }

        Model dark_scattering(const CosmologyParameters & p) {
            using namespace Constants;
            Model m;
            m.name = "dark_scattering";
            for (const auto & s : standard_species(p)) // the dark matter perturbations are in the dark energy's block
                m.species.push_back(
                    s->role() == Species::Role::CDM ?
                        std::make_shared<PowerLawSpecies>("cdm (scattering)", Species::Role::CDM, p.OmegaCDM, 3.0) :
                        s);
            const double barn_per_GeV = 1e-28 / 1.78266192e-27; // m^2/kg
            const double rho_crit0 = 3.0 * p.H0() * p.H0() / (8.0 * M_PI * G);
            m.species.push_back(std::make_shared<FluidDarkEnergy>(p.OmegaLambda(),
                                                                  parameter(p, "ds_w"),
                                                                  parameter(p, "ds_cs2"),
                                                                  parameter(p, "ds_xi") * barn_per_GeV,
                                                                  rho_crit0));
            m.gravity = std::make_shared<GeneralRelativity>();
            return m;
        }

        Model cubic_galileon(const CosmologyParameters & p) {
            if (p.OmegaK != 0.0)
                throw std::invalid_argument("cubic_galileon: only implemented for a flat universe");
            Model m;
            m.name = "cubic_galileon";
            m.species = standard_species(p);
            const double Om = p.OmegaLambda(); // closure
            m.species.push_back(std::make_shared<CubicGalileonSpecies>(m.species, Om));
            m.gravity = std::make_shared<CubicGalileonGravity>(Om);
            return m;
        }

        Model hu_sawicky(const CosmologyParameters & p) {
            Model m = lcdm(p);
            m.name = "fr";
            m.gravity = std::make_shared<HuSawickyGravity>(parameter(p, "fR0"), parameter(p, "fR_n"));
            return m;
        }

        Model jbd(const CosmologyParameters & p) {
            if (p.OmegaK != 0.0)
                throw std::invalid_argument("jbd: only implemented for a flat universe");
            Model m;
            m.name = "jbd";
            m.species = standard_species(p);
            auto g = std::make_shared<BransDickeGravity>(parameter(p, "bd_omega"), m.species);
            m.species.push_back(
                std::make_shared<PowerLawSpecies>("lambda", Species::Role::DarkEnergy, g->OmegaLambda(), 0.0));
            m.gravity = g;
            return m;
        }

        Model horndeski(const CosmologyParameters & p) {
            if (p.OmegaK != 0.0)
                throw std::invalid_argument("horndeski: only implemented for a flat universe");
            const double w0 = parameter(p, "hd_w0"), wa = parameter(p, "hd_wa"),
                         type = parameter(p, "hd_parametrization");
            if (type != 0.0 && type != 1.0 && type != 2.0)
                throw std::invalid_argument(
                    "horndeski: hd_parametrization must be 0 (propto_omega), 1 (propto_scale) or 2 (constant)");
            Model m;
            m.name = "horndeski";
            m.species = standard_species(p);
            if (w0 == -1.0 && wa == 0.0)
                m.species.push_back(
                    std::make_shared<PowerLawSpecies>("lambda", Species::Role::DarkEnergy, p.OmegaLambda(), 0.0));
            else
                m.species.push_back(std::make_shared<CPLDarkEnergy>(p.OmegaLambda(), w0, wa));
            auto al = std::make_shared<StandardAlphas>(StandardAlphas::Type(int(type)),
                                                       parameter(p, "hd_cK"),
                                                       parameter(p, "hd_cB"),
                                                       parameter(p, "hd_cM"),
                                                       parameter(p, "hd_M2_ini"));
            m.gravity = std::make_shared<HorndeskiAlphaGravity>(al, nullptr, parameter(p, "hd_stability_check") != 0.0);
            return m;
        }

        Model w0wa(const CosmologyParameters & p) {
            Model m;
            m.name = "w0wa";
            m.species = standard_species(p);
            m.species.push_back(
                std::make_shared<CPLDarkEnergy>(p.OmegaLambda(), parameter(p, "w0"), parameter(p, "wa")));
            m.gravity = std::make_shared<GeneralRelativity>();
            return m;
        }

        Model ede(const CosmologyParameters & p) {
            Model m;
            m.name = "ede";
            m.species = standard_species(p);
            auto e = std::make_shared<EarlyDarkEnergySpecies>(parameter(p, "ede_fEDE"),
                                                              parameter(p, "ede_log10zc"),
                                                              parameter(p, "ede_theta_i"),
                                                              parameter(p, "ede_n"),
                                                              m.species,
                                                              p.OmegaK);
            m.species.push_back(e);
            m.species.push_back(
                std::make_shared<PowerLawSpecies>("lambda", Species::Role::DarkEnergy, e->OmegaL(), 0.0));
            m.gravity = std::make_shared<GeneralRelativity>();
            return m;
        }

        Model ide(const CosmologyParameters & p) {
            Model m;
            m.name = "ide";
            auto de = std::make_shared<InteractingDarkEnergy>(p.OmegaLambda(),
                                                              p.OmegaCDM,
                                                              parameter(p, "ide_w"),
                                                              parameter(p, "ide_xi"),
                                                              parameter(p, "ide_cs2"));
            if (parameter(p, "ide_stability_check") != 0.0 && de->doom() > 0.0)
                throw std::invalid_argument(
                    "ide: doom factor d = xi/(3(1 + w)) = " + std::to_string(de->doom()) +
                    " > 0: large-scale instability of the dark-energy perturbations (xi and 1 + w "
                    "need opposite signs; ide_stability_check = 0 to run anyway)");
            for (const auto & s : standard_species(p))
                m.species.push_back(s->role() == Species::Role::CDM ? std::make_shared<InteractingCDM>(de) : s);
            m.species.push_back(de);
            m.gravity = std::make_shared<GeneralRelativity>();
            return m;
        }

        // ---------------------------------------------------------------------------
        // Registry of built-in models
        // ---------------------------------------------------------------------------
        const std::vector<ModelInfo> & registry() {
            static const std::vector<ModelInfo> r = {
                {"lcdm", "LCDM (+ curvature, massive neutrinos) in general relativity", {}, lcdm},
                {"musigma",
                 "LCDM background, perturbations with mu(a), Sigma(a) of Planck/DES (flat)",
                 {{"mu0", 0.0, "mu = 1 + mu0 Omega_DE(a)/Omega_DE,0: modified growth (Psi)"},
                  {"Sigma0", 0.0, "Sigma = 1 + Sigma0 Omega_DE(a)/Omega_DE,0: modified lensing (Psi - Phi)"}},
                 musigma},
                {"dgp",
                 "DGP braneworld (flat): modified Friedmann equation, quasi-static mu(a), Sigma = 1; "
                 "Omega_Lambda from E(0) = 1",
                 {{"Omega_rc",
                   0.25,
                   "c^2/(4 r_c^2 H0^2) (0.25: H0 r_c/c = 1). 0 with dgp_branch = 1: sDGP without dark energy"},
                  {"dgp_branch", -1.0, "-1: normal branch (nDGP, with Lambda), +1: self-accelerating branch (sDGP)"},
                  {"dgp_lcdm_background",
                   0.0,
                   "1: LCDM expansion, only the perturbations modified (nDGP of N-body work)"}},
                 dgp},
                {"quintessence",
                 "scalar field with V = V0 exp(-lambda phi), thawing from rest; V0 from closure; GR",
                 {{"quint_lambda", 1.0, "slope lambda of the potential (phi in reduced Planck masses); 0 = LCDM"}},
                 quintessence},
                {"jbd",
                 "Jordan-Brans-Dicke (flat) with Lambda; phi today gives the Newtonian G, Omega_Lambda from E(0) = 1",
                 {{"bd_omega",
                   1000.0,
                   "Brans-Dicke parameter omega (GR as omega -> infinity; Cassini: omega > 40000)"}},
                 jbd},
                {"fr",
                 "Hu-Sawicky f(R) (flat): LCDM background, quasi-static mu(k, a) and Sigma(a)",
                 {{"fR0", 1e-5, "|f_R0|, the scalar field today (GR for 0)"},
                  {"fR_n", 1.0, "exponent n of the Hu-Sawicky model"}},
                 hu_sawicky},
                {"coupled_quintessence",
                 "quintessence (V ~ exp(-lambda phi)) coupled to CDM (m ~ exp(-beta phi)); GR",
                 {{"cq_lambda", 0.5, "slope lambda of the potential (phi in reduced Planck masses)"},
                  {"cq_beta", 0.05, "coupling beta (Amendola's sqrt(2/3) beta); CDM feels G (1 + 2 beta^2)"}},
                 coupled_quintessence},
                {"dark_scattering",
                 "wCDM (dark-energy fluid with perturbations) with elastic dark matter - dark energy scattering; GR",
                 {{"ds_w", -0.9, "equation of state w of the dark energy (constant, != -1)"},
                  {"ds_xi", 10.0, "xi = sigma_D / m_c [barn/GeV] (0: wCDM without scattering)"},
                  {"ds_cs2", 1.0, "rest-frame sound speed squared of the dark energy"}},
                 dark_scattering},
                {"cubic_galileon",
                 "cubic Galileon on its tracker (flat, no Lambda): Omega_phi from closure, no free parameter",
                 {},
                 cubic_galileon},
                {"horndeski",
                 "Horndeski in the alpha parameterization of hi_class (alpha_T = 0, flat): LCDM or w0-wa expansion, "
                 "alpha_X = c_X f(a); stability checked",
                 {{"hd_parametrization",
                   0.0,
                   "f(a): 0 = Omega_DE(a) (propto_omega), 1 = a (propto_scale), 2 = 1 (constant_alphas)"},
                  {"hd_cK", 1.0, "c_K: kineticity alpha_K = c_K f(a)"},
                  {"hd_cB", 0.0, "c_B: braiding alpha_B = c_B f(a) (Bellini-Sawicki convention, as hi_class)"},
                  {"hd_cM", 0.0, "c_M: Planck-mass run rate alpha_M = dln M*^2/dln a = c_M f(a)"},
                  {"hd_M2_ini", 1.0, "M*^2 at early times (a = 1e-14 for constant_alphas)"},
                  {"hd_w0", -1.0, "w0 of the expansion (w0-wa dark energy, GR Friedmann equation)"},
                  {"hd_wa", 0.0, "wa of the expansion"},
                  {"hd_stability_check",
                   1.0,
                   "1: stop for a ghost (D < 0) or gradient (c_s^2 < 0) instability, M*^2 <= 0"}},
                 horndeski},
                {"w0wa",
                 "dark energy with w(a) = w0 + wa (1 - a), smooth (no perturbations: for background data); GR",
                 {{"w0", -1.0, "w0 of the equation of state"}, {"wa", 0.0, "wa of the equation of state"}},
                 w0wa},
                {"ede",
                 "axion-like early dark energy V ~ (1 - cos(phi/f))^n with its perturbations, + Lambda (closure); GR",
                 {{"ede_fEDE", 0.1, "f_EDE: maximum fraction rho_EDE/rho_tot"},
                  {"ede_log10zc", 3.5, "log10 of the redshift z_c of the maximum"},
                  {"ede_theta_i", 2.8, "initial field theta_i = phi_i/f (0 < theta_i < pi)"},
                  {"ede_n", 3.0, "exponent n of the potential (w -> (n-1)/(n+1) when oscillating)"}},
                 ede},
                {"ide",
                 "dark-energy fluid (w, cs2) exchanging energy with CDM, Q = xi H rho_DE along u_c; GR",
                 {{"ide_xi", -0.1, "coupling xi (> 0: energy flows to the dark matter)"},
                  {"ide_w", -0.95, "equation of state w of the dark energy (constant, != -1)"},
                  {"ide_cs2", 1.0, "rest-frame sound speed squared of the dark energy"},
                  {"ide_stability_check", 1.0, "1: stop if the doom factor xi/(3(1+w)) > 0 (large-scale instability)"}},
                 ide},
            };
            return r;
        }

        const ModelInfo & find(const std::string & name) {
            for (const auto & m : registry())
                if (m.name == name)
                    return m;
            std::string names;
            for (const auto & m : registry())
                names += (names.empty() ? "" : ", ") + m.name;
            throw std::invalid_argument("unknown model '" + name + "' (available: " + names + ")");
        }

        double parameter(const CosmologyParameters & params, const std::string & name) {
            const auto it = params.model_params.find(name);
            if (it != params.model_params.end())
                return it->second;
            for (const auto & p : find(params.model).parameters)
                if (p.name == name)
                    return p.default_value;
            throw std::invalid_argument("model " + params.model + " has no parameter '" + name + "'");
        }

        void validate(const CosmologyParameters & params) {
            const ModelInfo & info = find(params.model);
            for (const auto & [name, value] : params.model_params) {
                bool ok = false;
                for (const auto & p : info.parameters)
                    ok = ok || p.name == name;
                if (!ok)
                    throw std::invalid_argument("model " + params.model + " has no parameter '" + name + "'");
            }
        }

        Model build(const CosmologyParameters & params) {
            validate(params);
            Model m = find(params.model).build(params);
            if (!m.gravity)
                throw std::logic_error("model " + params.model + ": no gravity");
            return m;
        }
    } // namespace Models

} // namespace FML::COSMOLOGY
