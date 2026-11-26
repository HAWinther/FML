#ifndef FML_TREEPMCODE_COSMOLOGY_HEADER
#define FML_TREEPMCODE_COSMOLOGY_HEADER

//==========================================================================================
//
// TreePMCode: the background of the simulation, the models that enter it through the background, the Poisson factor and
// a friction (LCDM, coupled quintessence / the Fading Dark Sector, covariantised dark scattering), the growth factors of
// the initial conditions and the kick and drift integrals. (The same as AMRCode's, kept separate.)
//
// Units: positions in [0,1) (box units), time variable tau = log(a), velocities v = a^2 dx/dt / (H0 B):
//   dx/dtau = v / (a^2 E(a)),   dv/dtau = -Gamma(a) v - (3/2) OmegaM P(a) / (a E(a)) * D phi,   D^2 phi = delta
//
//==========================================================================================

#include <FML/Cosmology/Background/BackgroundModel_CoupledQuintessence.h>
#include <FML/Cosmology/Background/BackgroundModel_DarkScattering.h>
#include <FML/Global/Global.h>

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace FML {
    namespace NBODY {
        namespace TREEPM {


            //======================================================================================
            /// The background of the simulation with the 1LPT/2LPT growth factors and the kick and drift
            /// integrals of the code's equations of motion:
            ///  - LCDM (the default): flat, matter + Lambda (+ optional radiation)
            ///  - Coupled quintessence / the Fading Dark Sector (Obied, Vafa & Wu, arXiv:2611.17422; V ~ exp(-c phi),
            ///    m_DM ~ exp(-c' phi): coupled quintessence with lambda = c, beta = c'). The background (E(a), phi(a), the
            ///    CDM density with the varying mass) is FML's (Cosmology/Background, make_coupled_quintessence). The
            ///    particles are CDM + baryons as one fluid, as SimulationCode's GravityModel_CoupledQuintessence: the
            ///    field is light (no screening, scale-independent on N-body scales), so the model enters only through
            ///      Poisson: G_eff/G = (1 + 2 beta^2 f_c^2) and the source rho_m(a) a^3 / OmegaM (the varying CDM mass)
            ///      friction: dv/dtau = -Gamma v + force, Gamma = -f_c beta dphi/dlna
            ///    with f_c(a) = rho_c(a) / (rho_c(a) + rho_b(a)) (momentum-weighted: only the CDM couples)
            ///    With CDM and baryons as two sets of particles (Fluid::CDM, Fluid::Baryon) both feel gravity
            ///    (P = rho_m(a) a^3 / OmegaM), the CDM also the fifth force 2 beta^2 f_c of the CDM density (a second
            ///    force evaluation with the CDM as the only source, in TreePMSimulation) and the friction Gamma_c = -beta dphi/dlna; the baryons none
            /// For beta = lambda = 0 it is LCDM (with OmegaR = 0: the same background)
            //======================================================================================
            /// The fluid a particle belongs to: CDM+baryons as one set of particles (CB), or CDM and baryons as two
            enum class Fluid { CB, CDM, Baryon };

            class Cosmology {
              public:
                double OmegaM{0.3};
                double OmegaR{0.0};
                double OmegaL{0.7};
                double OmegaB{0.049};
                std::string model{"LCDM"}; // LCDM or CoupledQuintessence
                double cq_lambda{0.0}, cq_beta{0.0};
                double ds_w{-1.0}, cds_alpha{0.0}; // CovariantDarkScattering

                Cosmology() = default;
                Cosmology(double OmegaM_, double OmegaR_ = 0.0) : OmegaM(OmegaM_), OmegaR(OmegaR_), OmegaL(1.0 - OmegaM_ - OmegaR_) {
                    compute_growth();
                }
                /// Coupled quintessence (the Fading Dark Sector: lambda = c, beta = c'), with OmegaB of the OmegaM
                /// (only the CDM couples). The radiation OmegaR > 0 is photons + 3.046 massless neutrinos
                static Cosmology coupled_quintessence(double OmegaM_, double OmegaB_, double lambda, double beta, double OmegaR_ = 0.0) {
                    namespace BG = FML::COSMOLOGY::BACKGROUND;
                    assert_mpi(OmegaB_ >= 0.0 and OmegaB_ < OmegaM_, "Cosmology: need 0 <= OmegaB < OmegaM");
                    Cosmology c;
                    c.OmegaM = OmegaM_;
                    c.OmegaR = OmegaR_;
                    c.OmegaB = OmegaB_;
                    c.OmegaL = 0.0;
                    c.model = "CoupledQuintessence";
                    c.cq_lambda = lambda;
                    c.cq_beta = beta;
                    c.bg = BG::make_coupled_quintessence(background_parameters(OmegaM_, OmegaB_, OmegaR_), lambda, beta);
                    for (const auto & s : c.bg->species())
                        if (const auto * field = dynamic_cast<const BG::CoupledFieldSpecies *>(s.get()))
                            c.cq = &field->sector();
                    assert_mpi(c.cq != nullptr, "Cosmology: no coupled dark sector in the background");
                    c.compute_growth();
                    return c;
                }
                /// The covariantised dark Thomson-like scattering (Beltran Jimenez et al. 2020; N-body: arXiv:2510.12551),
                /// Q^mu = alpha_bar (u_DE^mu - u_c^mu): momentum exchange only, so the background is wCDM (w), and the
                /// CDM has the drag Gamma_c = alpha a^3 / (Omega_c0 E) per ln a (alpha = 8 pi G alpha_bar / (3 H0^3);
                /// BACKGROUND::covariant_dark_scattering_cdm_drag, the rate of the Einstein-Boltzmann solver). The dark
                /// energy is smooth and at rest (as SimulationCode's CovariantDarkScattering: the drag is too strong
                /// for k < ~0.05 h/Mpc, where the solver's dark energy moves with the dark matter). Baryons do not scatter
                static Cosmology covariant_dark_scattering(double OmegaM_, double OmegaB_, double w, double alpha, double OmegaR_ = 0.0) {
                    namespace BG = FML::COSMOLOGY::BACKGROUND;
                    assert_mpi(OmegaB_ >= 0.0 and OmegaB_ < OmegaM_, "Cosmology: need 0 <= OmegaB < OmegaM");
                    assert_mpi(alpha >= 0.0, "Cosmology: CovariantDarkScattering needs cds_alpha >= 0");
                    Cosmology c;
                    c.OmegaM = OmegaM_;
                    c.OmegaR = OmegaR_;
                    c.OmegaB = OmegaB_;
                    c.OmegaL = 0.0;
                    c.model = "CovariantDarkScattering";
                    c.ds_w = w;
                    c.cds_alpha = alpha;
                    c.bg = BG::make_dark_scattering(background_parameters(OmegaM_, OmegaB_, OmegaR_), w);
                    c.cds = true;
                    c.compute_growth();
                    return c;
                }

                double E(double a) const {
                    if (bg)
                        return bg->E(std::log(a));
                    return std::sqrt(OmegaR / (a * a * a * a) + OmegaM / (a * a * a) + OmegaL);
                }
                double dlogEdloga(double a) const {
                    if (bg)
                        return bg->dlogEdx(std::log(a));
                    const double E2 = E(a) * E(a);
                    return (-4.0 * OmegaR / (a * a * a * a) - 3.0 * OmegaM / (a * a * a)) / (2.0 * E2);
                }
                /// OmegaM / (a^3 E^2) (the Poisson source is (3/2) OmegaM_of_a * poisson_factor)
                double OmegaM_of_a(double a) const { return OmegaM / (a * a * a * E(a) * E(a)); }

                /// The CDM mass relative to today (1 for LCDM)
                double cdm_mass_factor(double a) const {
                    if (not cq)
                        return 1.0;
                    return cq->rho_c(std::log(a)) * a * a * a / (OmegaM - OmegaB);
                }
                /// rho_c / (rho_c + rho_b)
                double cdm_fraction(double a) const {
                    const double c = (OmegaM - OmegaB) * cdm_mass_factor(a);
                    return c / (c + OmegaB);
                }
                /// The factor of (3/2) OmegaM / a in the Poisson equation of the one-fluid particles: G_eff/G times
                /// rho_m(a) a^3 / OmegaM (1 for LCDM)
                /// (CDM, Baryon: the gravity of all the CDM+baryons, rho_m(a) a^3 / OmegaM; the fifth force of the CDM is
                /// cdm_fifth_force(a) times the force of the CDM density)
                double poisson_factor(double a, Fluid fluid = Fluid::CB) const {
                    if (not cq)
                        return 1.0;
                    const double fc = cdm_fraction(a);
                    const double mass = ((OmegaM - OmegaB) * cdm_mass_factor(a) + OmegaB) / OmegaM;
                    return fluid == Fluid::CB ? mass * (1.0 + 2.0 * cq_beta * cq_beta * fc * fc) : mass;
                }
                /// CDM and baryons as two fluids: the CDM feels -D(phi + cdm_fifth_force phi_c) with D^2 phi = delta_cb,
                /// D^2 phi_c = delta_c (the same factor poisson_factor(a, CDM) in front): 2 beta^2 f_c(a)
                double cdm_fifth_force(double a) const { return cq ? 2.0 * cq_beta * cq_beta * cdm_fraction(a) : 0.0; }
                bool has_cdm_fifth_force() const { return cq != nullptr and cq_beta != 0.0; }
                /// The friction Gamma of dv/dtau = -Gamma v + force (0 for LCDM). CDM: -beta dphi/dlna (coupled
                /// quintessence) or the drag alpha a^3 / (Omega_c0 E) (CovariantDarkScattering); CB (one fluid): f_c
                /// times that; Baryon: 0
                double friction(double a, Fluid fluid = Fluid::CB) const {
                    if (not has_friction(fluid))
                        return 0.0;
                    const double g = cds ? FML::COSMOLOGY::BACKGROUND::covariant_dark_scattering_cdm_drag(*bg, cds_alpha, std::log(a))
                                         : -cq_beta * cq->dphidx(std::log(a));
                    return fluid == Fluid::CB ? cdm_fraction(a) * g : g;
                }
                bool has_friction(Fluid fluid = Fluid::CB) const {
                    return ((cq != nullptr and cq_beta != 0.0) or (cds and cds_alpha != 0.0)) and fluid != Fluid::Baryon;
                }
                /// dphi/dlna of the field (0 for LCDM)
                double dphidlna(double a) const { return cq ? cq->dphidx(std::log(a)) : 0.0; }
                const FML::COSMOLOGY::BACKGROUND::ComponentBackground * background() const { return bg.get(); }

                /// Growth factors (D1 normalized to a at early times, D2 to -3/7 D1^2) and growth rates dlogD/dloga
                double D1(double a) const { return interp(a, 0); }
                double f1(double a) const { return interp(a, 1) / interp(a, 0); }
                double D2(double a) const { return interp(a, 2); }
                double f2(double a) const { return interp(a, 3) / interp(a, 2); }

                /// The kick from a1 to a2: v(a2) = friction_factor(a1, a2) v(a1) + kick_factor(a1, a2) (-D phi) for a
                /// constant force. kick_factor = (3/2) OmegaM int dtau poisson_factor / (a E) exp(-int_tau^tau2 Gamma)
                double kick_factor(double a1, double a2, Fluid fluid = Fluid::CB) const {
                    const double I2 = friction_integral(a2, fluid);
                    const bool fr = has_friction(fluid);
                    return 1.5 * OmegaM *
                           integrate([&](double a) {
                               const double w = fr ? std::exp(friction_integral(a, fluid) - I2) : 1.0;
                               return poisson_factor(a, fluid) * w / (a * E(a));
                           }, a1, a2);
                }
                /// exp(-int_tau1^tau2 Gamma dtau) (1 without friction)
                double friction_factor(double a1, double a2, Fluid fluid = Fluid::CB) const {
                    return has_friction(fluid) ? std::exp(friction_integral(a1, fluid) - friction_integral(a2, fluid)) : 1.0;
                }
                /// Drift integral int dtau / (a^2 E) from a1 to a2 (multiplies v)
                double drift_factor(double a1, double a2) const {
                    return integrate([&](double a) { return 1.0 / (a * a * E(a)); }, a1, a2);
                }

              private:
                std::shared_ptr<const FML::COSMOLOGY::BACKGROUND::ComponentBackground> bg;
                const FML::COSMOLOGY::BACKGROUND::CoupledDarkSector * cq{nullptr};
                bool cds{false}; // CovariantDarkScattering

                /// The background parameters of FML's models: OmegaB, OmegaM - OmegaB, and the radiation OmegaR (photons
                /// + massless neutrinos with T_CMB scaled to give it; none if OmegaR = 0)
                static FML::COSMOLOGY::BACKGROUND::BackgroundParameters background_parameters(double OmegaM_, double OmegaB_, double OmegaR_) {
                    FML::COSMOLOGY::BACKGROUND::BackgroundParameters bp;
                    bp.Omega_b = OmegaB_;
                    bp.Omega_cdm = OmegaM_ - OmegaB_;
                    bp.m_nu_eV = {};
                    if (OmegaR_ > 0.0) {
                        // The CMB temperature that gives OmegaR (radiation ~ T^4)
                        const double OmegaR_ref = bp.Omega_gamma() + bp.Omega_nu_massless();
                        bp.T_CMB *= std::pow(OmegaR_ / OmegaR_ref, 0.25);
                    } else {
                        bp.T_CMB = 0.0;
                        bp.N_eff = 0.0;
                    }
                    return bp;
                }

                // Table in x = log(a) of D1, dD1/dx, D2, dD2/dx and int_xmin^x Gamma dx of the one fluid and of the CDM
                static constexpr double xmin = -11.0, xmax = 1.0;
                static constexpr int ntab = 4801;
                std::vector<std::array<double, 6>> table;

                double friction_integral(double a, Fluid fluid) const {
                    return has_friction(fluid) ? interp(a, fluid == Fluid::CB ? 4 : 5) : 0.0;
                }

                template <class F>
                static double integrate(F && f, double a1, double a2) {
                    // Simpson in x = log(a)
                    const int n = 64;
                    const double x1 = std::log(a1), x2 = std::log(a2), dx = (x2 - x1) / n;
                    double sum = 0.0;
                    for (int i = 0; i <= n; i++) {
                        const double w = (i == 0 or i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
                        sum += w * f(std::exp(x1 + i * dx));
                    }
                    return sum * dx / 3.0;
                }

                double interp(double a, int k) const {
                    const double x = std::log(a);
                    assert_mpi(x >= xmin and x <= xmax, "Cosmology: scale factor out of range of the growth table");
                    const double u = (x - xmin) / (xmax - xmin) * (ntab - 1);
                    const int i = std::min(int(u), ntab - 2);
                    const double w = u - i;
                    return (1.0 - w) * table[i][k] + w * table[i + 1][k];
                }

                void compute_growth() {
                    // D'' + (2 + dlogE/dx + Gamma) D' = 3/2 OmegaM(a) P(a) D, D2'' + (...) D2' = 3/2 OmegaM(a) P(a) (D2 - D1^2)
                    // (P = poisson_factor) with growing-mode initial conditions deep in matter domination (RK4 in x = log a),
                    // and the integral of the friction Gamma
                    using Y = std::array<double, 6>;
                    auto rhs = [&](double x, const Y & y) {
                        const double a = std::exp(x);
                        const double G = friction(a);
                        const double fr = 2.0 + dlogEdloga(a) + G, om = 1.5 * OmegaM_of_a(a) * poisson_factor(a);
                        return Y{y[1], -fr * y[1] + om * y[0], y[3], -fr * y[3] + om * (y[2] - y[0] * y[0]), G,
                                 friction(a, Fluid::CDM)};
                    };
                    table.resize(ntab);
                    const double a0 = std::exp(xmin);
                    Y y{a0, a0, -3.0 / 7.0 * a0 * a0, -6.0 / 7.0 * a0 * a0, 0.0, 0.0};
                    const double dx = (xmax - xmin) / (ntab - 1);
                    table[0] = y;
                    for (int i = 1; i < ntab; i++) {
                        const int nsub = 4;
                        const double h = dx / nsub;
                        for (int s = 0; s < nsub; s++) {
                            const double x = xmin + (i - 1) * dx + s * h;
                            auto add = [](const Y & p, const Y & q, double c) {
                                Y r;
                                for (int k = 0; k < 6; k++)
                                    r[k] = p[k] + c * q[k];
                                return r;
                            };
                            auto k1 = rhs(x, y);
                            auto k2 = rhs(x + 0.5 * h, add(y, k1, 0.5 * h));
                            auto k3 = rhs(x + 0.5 * h, add(y, k2, 0.5 * h));
                            auto k4 = rhs(x + h, add(y, k3, h));
                            for (int k = 0; k < 6; k++)
                                y[k] += h / 6.0 * (k1[k] + 2 * k2[k] + 2 * k3[k] + k4[k]);
                        }
                        table[i] = y;
                    }
                }
            };

            //======================================================================================
            /// The model of a simulation: the gravity model (LCDM; CoupledQuintessence or FadingDarkSector;
            /// CovariantDarkScattering), the densities and whether CDM and baryons are two sets of particles
            //======================================================================================
            struct ModelParameters {
                double OmegaM{0.3};
                double OmegaR{0.0};
                std::string gravity_model{"LCDM"};
                // Coupled quintessence (gravity_model = "CoupledQuintessence"): V ~ exp(-cq_lambda phi), m_DM ~
                // exp(-cq_beta phi). The Fading Dark Sector (gravity_model = "FadingDarkSector", Obied, Vafa & Wu,
                // arXiv:2611.17422): V ~ exp(-c phi), m_DM ~ exp(-c' phi), the same model with lambda = fds_c, beta =
                // fds_cprime (the paper's best fit c = -0.85, c' = 0.05). The baryon density OmegaB (of OmegaM) does
                // not couple. See Cosmology
                double OmegaB{0.049};
                double cq_lambda{0.5};
                double cq_beta{0.05};
                double fds_c{-0.85};
                double fds_cprime{0.05};
                // The covariantised dark Thomson-like scattering (gravity_model = "CovariantDarkScattering", see
                // Cosmology::covariant_dark_scattering): wCDM with w = ds_w and the CDM drag alpha a^3 / (Omega_c0 E)
                double ds_w{-0.98};
                double cds_alpha{1.0};
                // CDM and baryons as two sets of particles (the same number of each; species 0 and 1) instead of one
                // set of CDM+baryon particles. Both feel the gravity of all the CDM+baryons (collisionless baryons);
                // with coupled quintessence the CDM also feels the fifth force of the CDM and the friction
                bool baryons{false};
                bool coupled_quintessence() const {
                    return gravity_model == "CoupledQuintessence" or gravity_model == "FadingDarkSector";
                }
                bool covariant_dark_scattering() const { return gravity_model == "CovariantDarkScattering"; }
                /// A model of this header (the background, P(a) and the friction carry it): LCDM, CQ/FDS, CDS
                bool background_model() const {
                    return gravity_model == "LCDM" or coupled_quintessence() or covariant_dark_scattering();
                }
            };

            /// The background of the parameters (LCDM, coupled quintessence / the Fading Dark Sector, covariant dark
            /// scattering)
            inline Cosmology make_cosmology(const ModelParameters & p) {
                if (p.gravity_model == "FadingDarkSector")
                    return Cosmology::coupled_quintessence(p.OmegaM, p.OmegaB, p.fds_c, p.fds_cprime, p.OmegaR);
                if (p.gravity_model == "CoupledQuintessence")
                    return Cosmology::coupled_quintessence(p.OmegaM, p.OmegaB, p.cq_lambda, p.cq_beta, p.OmegaR);
                if (p.covariant_dark_scattering())
                    return Cosmology::covariant_dark_scattering(p.OmegaM, p.OmegaB, p.ds_w, p.cds_alpha, p.OmegaR);
                Cosmology c(p.OmegaM, p.OmegaR);
                c.OmegaB = p.OmegaB;
                return c;
            }

        } // namespace TREEPM
    } // namespace NBODY
} // namespace FML
#endif
