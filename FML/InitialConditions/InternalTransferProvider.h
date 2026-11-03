#ifndef FML_INTERNAL_TRANSFER_PROVIDER_H
#define FML_INTERNAL_TRANSFER_PROVIDER_H
#include <FML/InitialConditions/TransferProvider.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Spline/Spline.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
namespace FML::INITIALCONDITIONS {
    /// Owning solver adapter. The converted density uses the CDM rest frame, not each
    /// species' own comoving frame. Velocity transfers remain explicitly Newtonian.
    /// Gauge::NBody (Fidler et al. 2015, 2016; GR): delta_s = delta_s^N + (1 + w_s) S with the shift to the comoving
    /// density S = 3 calH theta_tot / (ck)^2 = 3 (Psi - dPhi/dx) / D, D = 1 - dlnHp/dx - Omega_K(x) (the momentum
    /// constraint and calH^2 - calH' + K c^2 = 4 pi G a^2 (rho + p) = 1.5 Omega_cb calH^2 (rho + p) / rho_cb), and the
    /// source nbody_gauge_source delta_extra = -(Theta' + calH Theta) / (4 pi G a^2 rho_cb) - delta_cb with
    /// Theta = theta_cb^N + H_T' the N-body gauge CDM+baryon velocity (continuity delta_cb' = -Theta holds in this
    /// gauge, so H_T' = (3 Phi - S)'):
    ///   Theta' + calH Theta = (ck)^2 Psi + f_b (Delta' + calH Delta) + H_T'' + calH H_T',   Delta = theta_b - theta_c
    /// (the CDM part is the Newtonian Euler equation). (ck)^2 Psi = (ck)^2 (Psi + Phi) - (ck)^2 Phi with the comoving
    /// Poisson equation (c^2 k^2 - 3 K c^2) Phi = 4 pi G a^2 rho_tot Delta_tot, rho_tot Delta_tot = rho_cb delta_cb +
    /// delta rho_other^N + (rho + p - rho_cb)_tot S (the photons, neutrinos and dark energy), so that delta_extra is a
    /// sum of small terms (not the difference of (ck)^2 Psi and delta_cb, which cancel to 1e-5 on small scales). The
    /// x-derivatives of 3 Phi - S and Delta are local least-squares fits on the x-grid of the solver
    /// (smooth_derivatives; the derivatives of a spline are wrong over ~10 nodes at its ends).
    /// Any model of the solver's registry (settings.cosmo.model; curvature allowed) with its massive neutrinos
    /// (deg_ncdm degenerate species of mass m_nu, or m_nu_list): MassiveNeutrino (all of them) is then available and
    /// TotalMatter includes it. The weights of
    /// the combinations are the background densities rho_i(a) of the model (e.g. a coupled CDM).
    class InternalTransferProvider final : public TransferProvider {
        struct State {
            FML::COSMOLOGY::RunSettings settings;
            FML::COSMOLOGY::BackgroundCosmology background;
            std::unique_ptr<FML::COSMOLOGY::RecombinationHistory> recombination;
            std::unique_ptr<FML::COSMOLOGY::Perturbations> perturbations;
            explicit State(FML::COSMOLOGY::RunSettings options)
                : settings(std::move(options)), background(settings.cosmo) {
                background.solve(settings.bg_x_start, settings.bg_x_end, settings.bg_n_points);
                recombination = std::make_unique<FML::COSMOLOGY::RecombinationHistory>(background, settings.rec);
                recombination->solve();
                perturbations =
                    std::make_unique<FML::COSMOLOGY::Perturbations>(background, *recombination, settings.pert);
                perturbations->solve();
            }
        };
        std::shared_ptr<const State> state;
        Gauge gauge;
        /// The N-body gauge source delta_extra(x, log k [1/m]) (Gauge::NBody)
        std::shared_ptr<const FML::INTERPOLATION::SPLINE::Spline2D> nbody_source;
        double primordial_scale{1.0};
        bool massive_nu() const { return state->settings.cosmo.has_massive_nu(); }
        // delta of one species in the requested gauge. Newtonian -> CDM-synchronous is the time shift that brings CDM
        // to rest: delta_s += 3 (1 + w_s) Hp theta_cdm / (c k)^2 = -3 (1 + w_s) Hp v_cdm / (c k)
        double density_one(Species species, double x, double k_si) const {
            const auto & pert = *state->perturbations;
            double delta = species == Species::CDM    ? pert.get_delta_cdm(x, k_si) :
                           species == Species::Baryon ? pert.get_delta_b(x, k_si) :
                                                        pert.get_delta_ncdm(x, k_si);
            if (gauge == Gauge::SynchronousCDM)
                delta -= 3.0 * (1.0 + w_of(species, x)) * state->background.Hp_of_x(x) /
                         (FML::COSMOLOGY::Constants::c * k_si) * pert.get_v_cdm(x, k_si);
            else if (gauge == Gauge::NBody)
                delta += (1.0 + w_of(species, x)) * nbody_shift(x, k_si);
            return delta;
        }
        /// S = 3 calH theta_tot / (ck)^2: delta^Nb = delta^N + (1 + w) S (see the class comment)
        double nbody_shift(double x, double k_si) const {
            const auto & bg = state->background;
            const auto & pert = *state->perturbations;
            const double D = 1.0 - bg.dHpdx_of_x(x) / bg.Hp_of_x(x) - bg.OmegaK_of_x(x);
            return 3.0 * (pert.get_Psi(x, k_si) - pert.get_dPhidx(x, k_si)) / D;
        }
        /// The first and second derivatives of y(x) on a (non-uniform) grid from a local least-squares quadratic over
        /// the points with |x_j - x_i| < width (at least 3 on each side where there are). The values of the solver
        /// have relative errors ~1e-6 (its ODE tolerances), which a 3-point second derivative on its grid (dx = 0.01)
        /// amplifies to ~1e-2; the fit over ~20 points reduces this by more than 100. The source varies on dx ~ 1
        static void smooth_derivatives(const std::vector<double> & x,
                                       const std::vector<double> & y,
                                       std::vector<double> & dy,
                                       std::vector<double> & ddy,
                                       double width = 0.1) {
            const size_t n = x.size();
            dy.assign(n, 0.0);
            ddy.assign(n, 0.0);
            for (size_t i = 0; i < n; i++) {
                size_t lo = i, hi = i;
                while (lo > 0 and (x[i] - x[lo - 1] < width or i - lo < 3))
                    lo--;
                while (hi + 1 < n and (x[hi + 1] - x[i] < width or hi - i < 3))
                    hi++;
                // y = c0 + c1 t + c2 t^2 with t = x - x_i: the normal equations
                double S[5] = {0, 0, 0, 0, 0}, T[3] = {0, 0, 0};
                for (size_t j = lo; j <= hi; j++) {
                    const double t = x[j] - x[i];
                    double tp = 1.0;
                    for (int m = 0; m < 5; m++) {
                        S[m] += tp;
                        if (m < 3)
                            T[m] += tp * y[j];
                        tp *= t;
                    }
                }
                // Solve [[S0 S1 S2][S1 S2 S3][S2 S3 S4]] c = T (Cramer)
                const double A[3][3] = {{S[0], S[1], S[2]}, {S[1], S[2], S[3]}, {S[2], S[3], S[4]}};
                auto det3 = [](const double M[3][3]) {
                    return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                           M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                           M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
                };
                const double det = det3(A);
                double B1[3][3], B2[3][3];
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++) {
                        B1[r][c] = c == 1 ? T[r] : A[r][c];
                        B2[r][c] = c == 2 ? T[r] : A[r][c];
                    }
                dy[i] = det3(B1) / det;
                ddy[i] = 2.0 * det3(B2) / det;
            }
        }
        /// The N-body gauge source on the x-grid of the solver for each of its k (see the class comment)
        std::shared_ptr<const FML::INTERPOLATION::SPLINE::Spline2D> make_nbody_source() const {
            const auto & bg = state->background;
            const auto & pert = *state->perturbations;
            const auto & xs = pert.get_x_array();
            const auto & ks = pert.get_k_array();
            const double c = FML::COSMOLOGY::Constants::c, H0 = bg.get_H0();
            const size_t nx = xs.size(), nk = ks.size();
            std::vector<double> logk(nk);
            std::vector<std::vector<double>> source(nx, std::vector<double>(nk));
            for (size_t ik = 0; ik < nk; ik++)
                logk[ik] = std::log(ks[ik]);
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (size_t ik = 0; ik < nk; ik++) {
                const double k = ks[ik];
                // S, G = 3 Phi - S (H_T' = Hp dG/dx: the derivatives of G, not the difference of those of Phi and
                // S, which nearly cancel on large scales) and Delta
                std::vector<double> S(nx), G(nx), Delta(nx);
                for (size_t i = 0; i < nx; i++) {
                    S[i] = nbody_shift(xs[i], k);
                    G[i] = 3.0 * pert.get_Phi(xs[i], k) - S[i];
                    Delta[i] = -c * k * (pert.get_v_b(xs[i], k) - pert.get_v_cdm(xs[i], k));
                }
                std::vector<double> dG, ddG, dDelta, unused;
                smooth_derivatives(xs, G, dG, ddG);
                smooth_derivatives(xs, Delta, dDelta, unused);
                for (size_t i = 0; i < nx; i++) {
                    const double x = xs[i], Hp = bg.Hp_of_x(x), dlnHp = bg.dHpdx_of_x(x) / Hp;
                    const double rho_c = bg.rhoCDM(x), rho_b = bg.rhoB(x), rho_cb = rho_c + rho_b, f_b = rho_b / rho_cb;
                    const double Omega_cb = H0 * H0 * std::exp(2.0 * x) * rho_cb / (Hp * Hp);
                    const double OmegaK = bg.OmegaK_of_x(x), D = 1.0 - dlnHp - OmegaK, ckH = c * k / Hp;
                    // The density contrasts of the other species in the Newtonian gauge, delta rho / rho_cb
                    double other = 4.0 * bg.rhoGamma(x) * pert.get_Theta(0, x, k);
                    if (pert.has_neutrinos())
                        other += 4.0 * bg.rhoNu(x) * pert.get_Nu(0, x, k);
                    if (pert.has_massive_nu())
                        other += bg.rhoNcdm(x) * pert.get_delta_ncdm(x, k);
                    if (pert.has("delta_de"))
                        other += bg.rhoDE(x) * pert.get("delta_de", x, k);
                    other /= rho_cb;
                    const double delta_cb =
                        (rho_c * pert.get_delta_cdm(x, k) + rho_b * pert.get_delta_b(x, k)) / rho_cb + S[i];
                    // rho_tot Delta_tot / rho_cb and the Poisson equation (with the curvature)
                    const double Delta_tot = delta_cb + other + (D / (1.5 * Omega_cb) - 1.0) * S[i];
                    const double poisson = 1.0 + 3.0 * OmegaK / (ckH * ckH);
                    // H_T' / Hp and its x-derivative
                    const double U = dG[i], dU = ddG[i];
                    const double anisotropic = ckH * ckH * (pert.get_Psi(x, k) + pert.get_Phi(x, k));
                    source[i][ik] =
                        (Delta_tot / poisson - delta_cb) -
                        (anisotropic + f_b * (dDelta[i] + Delta[i]) / Hp + dU + (1.0 + dlnHp) * U) / (1.5 * Omega_cb);
                }
            }
            auto spline = std::make_shared<FML::INTERPOLATION::SPLINE::Spline2D>();
            spline->create(xs, logk, source, "N-body gauge source delta_extra(x, log k)");
            return spline;
        }
        double w_of(Species species, double x) const {
            if (species != Species::MassiveNeutrino)
                return 0.0;
            return state->background.pNcdm(x) / state->background.rhoNcdm(x);
        }
        /// rho_s a^3 in units of rho_c0 (the weights of the combinations; Omega for CDM and baryons in LCDM)
        double rho_a3_of(Species species, double x) const {
            const auto & bg = state->background;
            const double rho = species == Species::CDM    ? bg.rhoCDM(x) :
                               species == Species::Baryon ? bg.rhoB(x) :
                                                            bg.rhoNcdm(x);
            return rho * std::exp(3.0 * x);
        }
        /// The species a combination is made of
        std::vector<Species> components(Species s) const {
            if (s == Species::CDMBaryon)
                return {Species::CDM, Species::Baryon};
            if (s == Species::TotalMatter) {
                if (massive_nu())
                    return {Species::CDM, Species::Baryon, Species::MassiveNeutrino};
                return {Species::CDM, Species::Baryon};
            }
            return {s};
        }
        static double k_si_of(double k_hmpc, double h) { return k_hmpc * h / FML::COSMOLOGY::Constants::Mpc; }

      public:
        explicit InternalTransferProvider(FML::COSMOLOGY::RunSettings settings, Gauge gauge = Gauge::SynchronousCDM)
            : gauge(gauge) {
            const auto & p = settings.cosmo;
            // alpha_s: primordial_power() and SimulationCode's primordial spectrum have no running
            if (p.alpha_s != 0.0 || settings.pert.ic_adiabatic != 1.0 ||
                std::any_of(settings.pert.iso_amplitude.begin(), settings.pert.iso_amplitude.end(), [](double v) {
                    return v != 0;
                }))
                throw std::invalid_argument("[InternalTransferProvider] Requires no running alpha_s and a unit "
                                            "adiabatic mode (no isocurvature)");
            if (!(p.h > 0) || !std::isfinite(p.h) || !(p.A_s > 0) || !std::isfinite(p.A_s) || !(p.k_pivot > 0) ||
                !std::isfinite(p.k_pivot) || !std::isfinite(p.n_s) || !(p.OmegaB > 0) || !(p.OmegaCDM > 0) ||
                !(p.m_nu >= 0) ||
                (gauge != Gauge::Newtonian && gauge != Gauge::SynchronousCDM && gauge != Gauge::NBody))
                throw std::invalid_argument("[InternalTransferProvider] Invalid species, h or gauge");
            state = std::make_shared<State>(std::move(settings));
            if (gauge == Gauge::NBody)
                nbody_source = make_nbody_source();
        }
        /// Share the solved state; changing the density view never changes velocity gauge.
        InternalTransferProvider with_density_gauge(Gauge requested) const {
            if (requested != Gauge::Newtonian && requested != Gauge::SynchronousCDM && requested != Gauge::NBody)
                throw std::invalid_argument("[InternalTransferProvider] Unsupported density gauge");
            auto result = *this;
            result.gauge = requested;
            if (requested == Gauge::NBody and not result.nbody_source)
                result.nbody_source = make_nbody_source();
            return result;
        }
        bool has_nbody_gauge_source() const override { return gauge == Gauge::NBody; }
        double nbody_gauge_source(double k, double a) const override {
            if (gauge != Gauge::NBody)
                return TransferProvider::nbody_gauge_source(k, a);
            check_query(Species::CDM, k, a);
            return (*nbody_source)(std::log(a), std::log(k_si_of(k, h())));
        }
        double h() const override { return state->settings.cosmo.h; }
        TransferRange range() const override {
            const auto & p = *state->perturbations;
            return {p.get_k_array().front() * FML::COSMOLOGY::Constants::Mpc / h(),
                    p.get_k_array().back() * FML::COSMOLOGY::Constants::Mpc / h(),
                    std::exp(p.get_x_array().front()),
                    std::exp(p.get_x_array().back())};
        }
        Gauge density_gauge() const override { return gauge; }
        bool has_density(Species s) const override {
            return s == Species::CDM || s == Species::Baryon || s == Species::CDMBaryon || s == Species::TotalMatter ||
                   (s == Species::MassiveNeutrino && massive_nu());
        }
        bool has_velocity(Species s) const override { return has_density(s); }
        /// Combinations are weighted by rho_i(a): CDM + baryons, and with massive neutrinos also them in TotalMatter
        double density_transfer(Species s, double k, double a) const override {
            check_query(s, k, a);
            const double x = std::log(a), k_si = k_si_of(k, h());
            double sum = 0.0, weight = 0.0;
            for (auto c : components(s)) {
                const double rho = rho_a3_of(c, x);
                sum += rho * density_one(c, x, k_si);
                weight += rho;
            }
            return sum / weight;
        }
        double density_growth_rate(Species s, double k, double a) const override {
            check_query(s, k, a);
            const double x = std::log(a), lo = std::max(x - 1e-3, std::log(range().amin)),
                         hi = std::min(x + 1e-3, std::log(range().amax));
            const double value = density_transfer(s, k, a);
            if (value == 0 || !(hi > lo))
                throw std::domain_error("[InternalTransferProvider] Undefined density logarithmic derivative");
            return (density_transfer(s, k, std::exp(hi)) - density_transfer(s, k, std::exp(lo))) / ((hi - lo) * value);
        }
        /// Newtonian theta/(aH) = -c k v / Hp; combinations are weighted by rho_i + p_i
        double velocity_divergence_transfer(Species s, double k, double a) const override {
            check_query(s, k, a, true);
            const double x = std::log(a), k_si = k_si_of(k, h());
            const auto & pert = *state->perturbations;
            double sum = 0.0, weight = 0.0;
            for (auto c : components(s)) {
                const double v = c == Species::CDM    ? pert.get_v_cdm(x, k_si) :
                                 c == Species::Baryon ? pert.get_v_b(x, k_si) :
                                                        pert.get_v_ncdm(x, k_si);
                const double rho_plus_p = rho_a3_of(c, x) * (1.0 + w_of(c, x));
                sum += rho_plus_p * v;
                weight += rho_plus_p;
            }
            return -FML::COSMOLOGY::Constants::c * k_si / state->background.Hp_of_x(x) * sum / weight;
        }
        void rescale_primordial_power(double factor) override {
            if (!(factor > 0) || !std::isfinite(factor))
                throw std::invalid_argument("[InternalTransferProvider] Invalid primordial power rescaling");
            primordial_scale *= factor;
        }
        double primordial_power(double k) const override {
            if (!(k > 0) || !std::isfinite(k))
                throw std::invalid_argument("[InternalTransferProvider] Invalid k_hmpc");
            const auto & p = state->settings.cosmo;
            return primordial_scale * 2 * M_PI * M_PI / (k * k * k) * p.A_s * std::pow(k * h() / p.k_pivot, p.n_s - 1);
        }
    };
} // namespace FML::INITIALCONDITIONS
#endif
