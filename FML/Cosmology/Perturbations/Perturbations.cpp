#include <FML/Cosmology/Perturbations/Perturbations.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FML::COSMOLOGY {

    using namespace Constants;

    Perturbations::Perturbations(const BackgroundCosmology & cosmo_in,
                                 const RecombinationHistory & rec_in,
                                 PerturbationOptions options_in)
        : cosmo(cosmo_in), rec(rec_in), options(options_in), K(cosmo_in.get_K()) {
        if (options.lmax < 3)
            throw std::invalid_argument("Perturbations: need lmax >= 3");
        if (!(options.x_start < -8.0 && options.x_end > -6.0))
            throw std::invalid_argument("Perturbations: need x_start < -8 and x_end > -6");
        if (options.x_start < cosmo.get_x_start() || options.x_end > cosmo.get_x_end())
            throw std::invalid_argument("Perturbations: x-range [" + std::to_string(options.x_start) + ", " +
                                        std::to_string(options.x_end) +
                                        "] extends outside the background solution (bg_x_start, bg_x_end)");
        for (double a : options.iso_amplitude)
            if (a != 0.0 && cosmo.gravity().name() != "GR")
                throw std::invalid_argument("Perturbations: isocurvature modes need GR (model " +
                                            cosmo.get_params().model + ")");
        for (const auto & s : cosmo.species()) {
            auto p = s->perturbations(*this);
            if (!p)
                continue;
            for (const auto & name : p->output_names()) {
                if (output_index.count(name))
                    throw std::invalid_argument("Perturbations: output " + name + " defined twice");
                output_index[name] = int(output_names.size());
                output_names.push_back(name);
            }
            species.push_back(std::move(p));
            species_bg.push_back(s.get());
        }
        for (const auto & name : cosmo.gravity().output_names()) {
            if (output_index.count(name))
                throw std::invalid_argument("Perturbations: output " + name + " defined twice");
            output_index[name] = int(output_names.size());
            output_names.push_back(name);
            n_gravity_outputs++;
        }
    }

    Perturbations::~Perturbations() = default;

    // =============================================================================
    // State layout and background at x
    // =============================================================================

    Perturbations::Layout Perturbations::layout(bool tc, const std::vector<int> & modes) const {
        Layout l;
        l.tc = tc;
        l.Theta = Layout::metric + cosmo.gravity().n_vars();
        l.n_Theta = tc ? 2 : options.lmax + 1;
        l.ThetaP = l.Theta + l.n_Theta;
        l.n_ThetaP = (!tc && options.polarization) ? options.lmax + 1 : 0;
        l.mode = modes;
        int n = l.ThetaP + l.n_ThetaP;
        for (size_t i = 0; i < species.size(); i++) {
            l.offset.push_back(n);
            n += species[i]->n_vars(modes[i]);
        }
        l.n_tot = n;
        return l;
    }

    Perturbations::ModeState Perturbations::make_mode(double k) const {
        ModeState st;
        st.b.k = k;
        st.b.s2 = s_ell(k, 2);
        st.b.curv = 1.0 - 3.0 * K / (k * k); // (k^2 - 3K) / k^2
        if (st.b.s2 <= 0.0)
            throw std::runtime_error("Perturbations: s_2 = 0 (closed universe mode with nu <= 2)");
        for (const auto & s : species)
            st.cache.emplace_back(s->cache_size());
        st.gcache.resize(cosmo.gravity().cache_size());
        return st;
    }

    void Perturbations::update(double x, ModeState & st) const {
        ModeAtX & b = st.b;
        if (x == b.x)
            return;
        const double H0 = cosmo.get_H0();
        b.x = x;
        b.a = std::exp(x);
        b.Hp = cosmo.Hp_of_x(x);
        b.dHpH = cosmo.dHpdx_of_x(x) / b.Hp;
        b.ckH = c * b.k / b.Hp;
        b.dtau = rec.dtaudx_of_x(x);
        b.ddtau = rec.ddtauddx_of_x(x);
        b.R = rec.R_of_x(x);
        b.eta = cosmo.eta_of_x(x);
        b.cotK_H = c * cosmo.cotK(b.eta) / b.Hp;
        b.H0a2 = H0 * H0 * b.a * b.a;
        b.H0a2_rho_b = b.H0a2 * cosmo.rhoB(x);
        b.H0a2_rho_gamma = b.H0a2 * cosmo.rhoGamma(x);
        for (size_t i = 0; i < species.size(); i++)
            species[i]->update_cache(b, st.cache[i].data());
        cosmo.gravity().update_cache(cosmo, b, st.gcache.data());
    }

    // Moments of the core: baryons (delta rho = rho_b delta_b, flux = -rho_b v_b, no pressure) and photons
    // (4 rho_gamma Theta_0, delta p = delta rho / 3, 4 rho_gamma Theta_1, 4 rho_gamma Theta_2), plus those of the
    // species
    MetricAtX Perturbations::metric(const ModeState & st,
                                    const Layout & lay,
                                    const double * y,
                                    double Theta2,
                                    double * dydx_metric) const {
        const ModeAtX & b = st.b;
        Moments tot;
        tot.drho = b.H0a2_rho_b * y[Layout::delta_b] + 4.0 * b.H0a2_rho_gamma * y[lay.Theta];
        tot.dp = 4.0 / 3.0 * b.H0a2_rho_gamma * y[lay.Theta];
        tot.flux = -b.H0a2_rho_b * y[Layout::v_b] + 4.0 * b.H0a2_rho_gamma * y[lay.Theta + 1];
        tot.shear = 4.0 * b.H0a2_rho_gamma * Theta2;
        for (size_t i = 0; i < species.size(); i++)
            tot += species[i]->moments(b, st.cache[i].data(), lay.mode[i], y + lay.offset[i]);
        return cosmo.gravity().metric(b, st.gcache.data(), y + Layout::metric, tot, dydx_metric);
    }

    void Perturbations::rhs_species(const ModeState & st,
                                    const Layout & lay,
                                    const MetricAtX & m,
                                    const double * y,
                                    double * dydx) const {
        for (size_t i = 0; i < species.size(); i++)
            species[i]->rhs(st.b, st.cache[i].data(), lay.mode[i], m, y + lay.offset[i], dydx + lay.offset[i]);
    }

    // =============================================================================
    // Initial conditions (adiabatic, superhorizon, radiation domination), R = 1:
    //   Psi = -1/(3/2 + 2 f_nu/5),  Phi = -(1 + 2 f_nu/5) Psi,  delta_b = -3/2 Psi,  v_b = -(ck/2Hp) Psi,
    //   Theta_0 = -Psi/2,  Theta_1 = (ck/6Hp) Psi
    // f_nu is the fraction of free-streaming species (massless and, still relativistic, massive neutrinos)
    // in the radiation. The species set their own variables (SpeciesPerturbations::initial_conditions).
    // =============================================================================

    std::vector<double> Perturbations::initial_conditions(ModeState & st) const {
        const double x = options.x_start;
        update(x, st);
        const ModeAtX & b = st.b;
        const Layout lay = layout(true, std::vector<int>(species.size(), 0));
        std::vector<double> y(lay.n_tot);
        double OmNu = 0.0;
        for (size_t i = 0; i < species.size(); i++)
            if (species_bg[i]->free_streaming())
                OmNu += species_bg[i]->rho(x) * std::exp(4.0 * x);
        const double OmG = cosmo.rhoGamma(0.0);
        InitialData ic;
        ic.f_nu = OmNu / (OmG + OmNu);
        ic.OmNu = OmNu;
        ic.H0 = cosmo.get_H0();
        ic.G_eff = cosmo.gravity().G_eff_initial();
        ic.Psi = -1.0 / (1.5 + 0.4 * ic.f_nu);
        ic.Phi = -(1.0 + 0.4 * ic.f_nu) * ic.Psi;
        cosmo.gravity().initial_conditions(b, ic, y.data() + Layout::metric);
        y[Layout::delta_b] = -1.5 * ic.Psi;
        y[Layout::v_b] = -0.5 * b.ckH * ic.Psi;
        y[lay.Theta + 0] = -0.5 * ic.Psi;
        y[lay.Theta + 1] = b.ckH / 6.0 * ic.Psi;
        for (size_t i = 0; i < species.size(); i++)
            species[i]->initial_conditions(b, ic, y.data() + lay.offset[i]);
        for (double & v : y)
            v *= options.ic_adiabatic;

        // Isocurvature modes: linear superposition with the k-dependent amplitudes
        for (int m = 0; m < 4; m++) {
            if (options.iso_amplitude[m] == 0.0)
                continue;
            const double amp = options.iso_amplitude[m] *
                               std::pow(b.k * Constants::Mpc / cosmo.get_params().k_pivot, options.iso_tilt[m]);
            InitialData icm = ic;
            const IsoValues v = isocurvature_values(m, x, b.k, icm.Phi, icm.Psi);
            icm.iso = &v;
            std::vector<double> ym(lay.n_tot, 0.0);
            cosmo.gravity().initial_conditions(b, icm, ym.data() + Layout::metric);
            ym[Layout::delta_b] = v.delta_b;
            ym[Layout::v_b] = v.v_b;
            ym[lay.Theta + 0] = v.Theta0;
            ym[lay.Theta + 1] = v.Theta1;
            for (size_t i = 0; i < species.size(); i++)
                species[i]->initial_conditions(b, icm, ym.data() + lay.offset[i]);
            for (size_t i = 0; i < y.size(); i++)
                y[i] += amp * ym[i];
        }
        return y;
    }

    // =============================================================================
    // Isocurvature initial conditions (CDM, baryon, neutrino density, neutrino velocity), as CLASS: the leading
    // terms in k tau and omega tau (Bucher, Moodley & Turok 2000) in the synchronous gauge, then the gauge
    // transformation to the Newtonian gauge with alpha = (h' + 6 eta')/(2k^2) (from the series of delta_cdm and eta).
    // CLASS's variables map to ours as Theta_0 = delta_g/4, Theta_1 = theta_g/(3k), v = -theta/k, N_0 = delta_ur/4, N_1
    // = theta_ur/(3k), N_2 = shear_ur/2, Phi = -phi, Psi = psi. Our adiabatic mode is CLASS's with curvature_ini = -1,
    // so the isocurvature modes are CLASS's with entropy_ini = -1: the correlation with the adiabatic mode has the sign
    // convention of CLASS (c_ad_iso). omega = a rho_m/sqrt(rho_r) (CLASS units), with massive neutrinos counted as
    // radiation and rho_m = rho_b + rho_cdm.
    // =============================================================================

    IsoValues Perturbations::isocurvature_values(int m, double x, double k, double & Phi_out, double & Psi_out) const {
        const double E = -1.0;                                                           // entropy_ini (see above)
        const double a = std::exp(x), tau = cosmo.eta_of_x(x), H = cosmo.Hp_of_x(x) / c; // [m], [1/m]
        const double h0 = cosmo.get_H0() / c;                                            // [1/m]
        const double rg = cosmo.rhoGamma(x), rnu = cosmo.rhoNu(x) + cosmo.rhoNcdm(x);
        const double rb = cosmo.rhoB(x), rc = cosmo.rhoCDM(x);
        const double rr = rg + rnu, rm = rb + rc;
        const double fnu = rnu / rr, fg = rg / rr, fb = rb / rm, fc = rc / rm;
        const double om = a * h0 * rm / std::sqrt(rr); // [1/m]
        const double ot = om * tau, kt2 = k * k * tau * tau;
        const double s2sq = 1.0 - 3.0 * cosmo.get_K() / (k * k);
        if ((m == 2 || m == 3) && rnu <= 0.0)
            throw std::invalid_argument("Perturbations: neutrino isocurvature needs neutrinos");
        if (m == 0 && rc <= 0.0)
            throw std::invalid_argument("Perturbations: CDM isocurvature needs CDM");
        double dg = 0, tg = 0, db = 0, tb = 0, dc = 0, tc = 0, dur = 0, tur = 0, sur = 0, eta = 0;
        double ddc = 0, deta = 0; // d delta_cdm/dtau and d eta/dtau of the same series (for alpha, see below)
        if (m == 0 || m == 1) {   // CDI, BI
            const double f = m == 0 ? fc : fb;
            dg = E * f * ot * (-2.0 / 3.0 + ot / 4.0);
            tg = -E * f * om * kt2 / 12.0;
            db = 0.75 * dg + (m == 1 ? E : 0.0);
            tb = tg;
            dc = 0.75 * dg + (m == 0 ? E : 0.0);
            dur = dg;
            tur = tg;
            sur = -E * f * kt2 * ot / 6.0 / (2.0 * fnu + 15.0);
            eta = -E * f * ot * (1.0 / 6.0 - ot / 16.0);
            ddc = 0.75 * E * f * om * (-2.0 / 3.0 + ot / 2.0);
            deta = -E * f * om * (1.0 / 6.0 - ot / 8.0);
        } else if (m == 2) { // NID
            dg = E * fnu / fg * (-1.0 + kt2 / 6.0);
            tg = -E * fnu / fg * k * k * tau * (0.25 - fb / fg * 3.0 / 16.0 * ot);
            db = E * fnu / fg / 8.0 * kt2;
            tb = tg;
            dc = -E * fnu * fb / fg / 80.0 * kt2 * ot;
            dur = E * (1.0 - kt2 / 6.0);
            tur = E * k * k * tau / 4.0;
            sur = E * kt2 / (4.0 * fnu + 15.0) / 2.0;
            eta = -E * fnu / (4.0 * fnu + 15.0) / 6.0 * kt2;
            ddc = -3.0 * E * fnu * fb / fg / 80.0 * k * k * om * tau * tau;
            deta = -E * fnu / (4.0 * fnu + 15.0) / 3.0 * k * k * tau;
        } else { // NIV
            const double kt = k * tau;
            dg = E * kt * fnu / fg * (1.0 - 3.0 / 16.0 * fb * (2.0 + fg) / fg * ot);
            tg = E * fnu / fg * 0.75 * k *
                 (-1.0 + 0.75 * fb / fg * ot + 3.0 / 16.0 * ot * ot * fb / (fg * fg) * (fg - 3.0 * fb) + kt2 / 6.0);
            db = 0.75 * dg;
            tb = tg;
            dc = -E * 9.0 / 64.0 * fnu * fb / fg * kt * ot;
            dur = -E * kt * (1.0 + 3.0 / 16.0 * fb * fnu / fg * ot);
            tur = E * 0.75 * k * (1.0 - kt2 / 6.0 * (4.0 * fnu + 9.0) / (4.0 * fnu + 5.0));
            sur = E / (4.0 * fnu + 15.0) * kt * (1.0 + 3.0 * ot * fnu / (4.0 * fnu + 15.0));
            const double c2 = -3.0 / 64.0 * fb / fg + 15.0 / 4.0 / (4.0 * fnu + 15.0) / (4.0 * fnu + 5.0);
            eta = E * fnu * kt * (-1.0 / (4.0 * fnu + 5.0) + c2 * ot);
            ddc = -E * 9.0 / 32.0 * fnu * fb / fg * k * om * tau;
            deta = E * fnu * (-k / (4.0 * fnu + 5.0) + 2.0 * c2 * k * om * tau);
        }
        // synchronous -> Newtonian gauge with alpha = (h' + 6 eta')/(2k^2), h' = -2 delta_cdm' (CDM at rest in the
        // synchronous gauge). CLASS takes alpha from the Einstein constraints, which amplifies the truncation of the
        // series by (aH/k)^2: for k below omega ~ 0.008/Mpc the error is O(omega^2/k^2) and does not decay; the
        // derivative form has no such factor.
        (void)s2sq;
        const double alpha = (-2.0 * ddc + 6.0 * deta) / (2.0 * k * k);
        const double phi = eta - H * alpha;
        dg -= 4.0 * H * alpha;
        tg += k * k * alpha;
        db -= 3.0 * H * alpha;
        tb += k * k * alpha;
        dc -= 3.0 * H * alpha;
        tc = k * k * alpha;
        dur -= 4.0 * H * alpha;
        tur += k * k * alpha;
        // psi from the shear: k^2 (phi - psi) = 12 pi G a^2 (rho + p) sigma = 6 a^2 rho_ur sigma (CLASS units)
        const double psi = phi - 6.0 * h0 * h0 * a * a * rnu * sur / (k * k);
        IsoValues v;
        v.Theta0 = dg / 4.0;
        v.Theta1 = tg / (3.0 * k);
        v.delta_b = db;
        v.v_b = -tb / k;
        v.delta_c = dc;
        v.v_c = -tc / k;
        v.N0 = dur / 4.0;
        v.N1 = tur / (3.0 * k);
        v.N2 = sur / 2.0;
        Phi_out = -phi;
        Psi_out = psi;
        return v;
    }

    // =============================================================================
    // Tight coupling multipoles (expansion in ck/(Hp tau')), symmetric convention with s_l:
    //   Theta_2   = -(8/15) s_2 ck/(Hp tau') Theta_1    with polarization   [-(20/45) without]
    //   Theta_ell = -ell s_ell/(2 ell + 1) ck/(Hp tau') Theta_{ell-1},  ell >= 3
    //   Theta^P_0 = (5/4) Theta_2,  Theta^P_2 = Theta_2 / 4   (local balance of the collision term)
    //   Theta^P_1 = -ck/(3 Hp tau') (Theta^P_0 - 2 s_2 Theta^P_2) = -(5/4 - s_2/2)/3 ck/(Hp tau') Theta_2
    //   Theta^P_ell = -ell s_ell/(2 ell + 1) ck/(Hp tau') Theta^P_{ell-1},  ell >= 3
    // =============================================================================

    double Perturbations::Theta2_tc(double x, double k, double Theta1) const {
        const double ckH_tau = c * k / (cosmo.Hp_of_x(x) * rec.dtaudx_of_x(x));
        return (options.polarization ? -8.0 / 15.0 : -20.0 / 45.0) * s_ell(k, 2) * ckH_tau * Theta1;
    }

    double Perturbations::multipoles_tc(double x,
                                        double k,
                                        double Theta0,
                                        double Theta1,
                                        std::vector<double> & Th,
                                        std::vector<double> & ThP) const {
        const int L = options.lmax;
        const double ckH_tau = c * k / (cosmo.Hp_of_x(x) * rec.dtaudx_of_x(x));
        Th.assign(L + 1, 0.0);
        ThP.assign(L + 1, 0.0);
        Th[0] = Theta0;
        Th[1] = Theta1;
        Th[2] = Theta2_tc(x, k, Theta1);
        for (int ell = 3; ell <= L; ell++)
            Th[ell] = -ell * s_ell(k, ell) / (2.0 * ell + 1.0) * ckH_tau * Th[ell - 1];
        if (!options.polarization)
            return Th[2];
        ThP[0] = 1.25 * Th[2];
        ThP[2] = 0.25 * Th[2];
        ThP[1] = -ckH_tau / 3.0 * (ThP[0] - 2.0 * s_ell(k, 2) * ThP[2]);
        for (int ell = 3; ell <= L; ell++)
            ThP[ell] = -ell * s_ell(k, ell) / (2.0 * ell + 1.0) * ckH_tau * ThP[ell - 1];
        return Th[2] + ThP[0] + ThP[2];
    }

    // Variables at a segment boundary: tight coupling -> full photon hierarchy, and species switching mode
    std::vector<double> Perturbations::change_layout(ModeState & st,
                                                     const Layout & from,
                                                     const Layout & to,
                                                     const std::vector<double> & y_old) const {
        const double x = st.b.x;
        std::vector<double> y(to.n_tot);
        for (int i = 0; i < to.Theta; i++)
            y[i] = y_old[i];
        if (from.tc && !to.tc) {
            std::vector<double> Th, ThP;
            multipoles_tc(x, st.b.k, y_old[from.Theta], y_old[from.Theta + 1], Th, ThP);
            for (int ell = 0; ell < to.n_Theta; ell++)
                y[to.Theta + ell] = Th[ell];
            for (int ell = 0; ell < to.n_ThetaP; ell++)
                y[to.ThetaP + ell] = ThP[ell];
        } else {
            if (from.tc != to.tc)
                throw std::logic_error("Perturbations: cannot return to tight coupling");
            for (int i = to.Theta; i < to.ThetaP + to.n_ThetaP; i++)
                y[i] = y_old[i];
        }
        for (size_t i = 0; i < species.size(); i++) {
            std::vector<double> yi(y_old.begin() + from.offset[i],
                                   y_old.begin() + from.offset[i] + species[i]->n_vars(from.mode[i]));
            for (int m = from.mode[i] + 1; m <= to.mode[i]; m++) {
                std::vector<double> yn(species[i]->n_vars(m));
                species[i]->switch_mode(st.b, st.cache[i].data(), m, yi.data(), yn.data());
                yi = yn;
            }
            std::copy(yi.begin(), yi.end(), y.begin() + to.offset[i]);
        }
        return y;
    }

    // =============================================================================
    // Right-hand sides of the core
    // =============================================================================

    int
    Perturbations::rhs_tight_coupling(const ModeState & st, const Layout & lay, const double * y, double * dydx) const {
        const ModeAtX & b = st.b;
        const double dHpH = b.dHpH, ckH = b.ckH, dtau = b.dtau, ddtau = b.ddtau, R = b.R;
        const double vb = y[Layout::v_b];
        const double Th0 = y[lay.Theta], Th1 = y[lay.Theta + 1];
        const double s2 = b.s2;
        const double Th2 = (options.polarization ? -8.0 / 15.0 : -20.0 / 45.0) * s2 * ckH / dtau * Th1;

        const MetricAtX m = metric(st, lay, y, Th2, dydx + Layout::metric);
        const double Psi = m.Psi, dPhi = m.dPhi;
        rhs_species(st, lay, m, y, dydx);

        const double dTh0 = -ckH * Th1 - dPhi;
        const double q = (-((1.0 - R) * dtau + (1.0 + R) * ddtau) * (3.0 * Th1 + vb) - ckH * Psi +
                          (1.0 - dHpH) * ckH * (-Th0 + 2.0 * s2 * Th2) - ckH * dTh0) /
                         ((1.0 + R) * dtau + dHpH - 1.0);
        const double dvb = (-vb - ckH * Psi + R * (q + ckH * (-Th0 + 2.0 * s2 * Th2) - ckH * Psi)) / (1.0 + R);

        dydx[Layout::delta_b] = ckH * vb - 3.0 * dPhi;
        dydx[Layout::v_b] = dvb;
        dydx[lay.Theta] = dTh0;
        dydx[lay.Theta + 1] = (q - dvb) / 3.0;
        return GSL_SUCCESS;
    }

    int Perturbations::rhs_full(const ModeState & st, const Layout & lay, const double * y, double * dydx) const {
        const ModeAtX & b = st.b;
        const int L = options.lmax;
        const double ckH = b.ckH, dtau = b.dtau, R = b.R, k = b.k;
        const double trunc = (L + 1.0) * b.cotK_H; // c (L+1) cot_K(eta) / Hp

        const double vb = y[Layout::v_b];
        const double * Th = y + lay.Theta;
        double * dTh = dydx + lay.Theta;
        const double * ThP = y + lay.ThetaP;
        double * dThP = dydx + lay.ThetaP;
        const double Pi = options.polarization ? Th[2] + ThP[0] + ThP[2] : Th[2];

        const MetricAtX m = metric(st, lay, y, Th[2], dydx + Layout::metric);
        const double Psi = m.Psi, dPhi = m.dPhi;
        rhs_species(st, lay, m, y, dydx);

        dydx[Layout::delta_b] = ckH * vb - 3.0 * dPhi;
        dydx[Layout::v_b] = -vb - ckH * Psi + dtau * R * (3.0 * Th[1] + vb);

        // Temperature
        dTh[0] = -ckH * Th[1] - dPhi;
        // Curvature factors s_l (all 1 when flat)
        double s[64] = {};
        if (L + 1 >= 64)
            throw std::runtime_error("Perturbations: lmax too large");
        for (int ell = 0; ell <= L; ell++)
            s[ell] = s_ell(k, ell);

        dTh[1] = ckH / 3.0 * (Th[0] - 2.0 * s[2] * Th[2] + Psi) + dtau * (Th[1] + vb / 3.0);
        for (int ell = 2; ell < L; ell++) {
            dTh[ell] = ckH / (2.0 * ell + 1.0) * (ell * s[ell] * Th[ell - 1] - (ell + 1.0) * s[ell + 1] * Th[ell + 1]) +
                       dtau * (Th[ell] - (ell == 2 ? Pi / 10.0 : 0.0));
        }
        dTh[L] = ckH * s[L] * Th[L - 1] - trunc * Th[L] + dtau * Th[L]; // Truncation (Ma & Bertschinger)

        // Polarization
        if (options.polarization) {
            dThP[0] = -ckH * ThP[1] + dtau * (ThP[0] - 0.5 * Pi);
            for (int ell = 1; ell < L; ell++) {
                dThP[ell] =
                    ckH / (2.0 * ell + 1.0) * (ell * s[ell] * ThP[ell - 1] - (ell + 1.0) * s[ell + 1] * ThP[ell + 1]) +
                    dtau * (ThP[ell] - (ell == 2 ? Pi / 10.0 : 0.0));
            }
            dThP[L] = ckH * s[L] * ThP[L - 1] - trunc * ThP[L] + dtau * ThP[L];
        }
        return GSL_SUCCESS;
    }

    // =============================================================================
    // Grids and tight coupling criterion
    // =============================================================================

    std::vector<double> Perturbations::make_x_grid() const {
        auto a = Utils::linspace(options.x_start, -8.0, options.n_x_early);
        auto b = Utils::linspace(-8.0, -6.0, options.n_x_recomb);
        auto e = Utils::linspace(-6.0, options.x_end, options.n_x_late);
        std::vector<double> x(a);
        x.insert(x.end(), b.begin() + 1, b.end());
        x.insert(x.end(), e.begin() + 1, e.end());
        return x;
    }

    double Perturbations::x_tight_coupling_end(double k) const {
        const double x_rec = -std::log1p(options.tc_z_max);
        const double dx = 1e-3;
        for (double x = options.x_start; x < x_rec; x += dx) {
            const double dtau = rec.dtaudx_of_x(x);
            const double ckH = c * k / cosmo.Hp_of_x(x);
            if (std::fabs(dtau) < options.tc_tau_min || std::fabs(ckH / dtau) > options.tc_k_over_tau)
                return x;
        }
        return x_rec;
    }

    // =============================================================================
    // Solve
    // =============================================================================

    void Perturbations::solve() {
        Utils::Timer timer("Perturbations::solve");
        if (!cosmo.is_solved() || !rec.is_solved())
            throw std::runtime_error("Perturbations::solve: background and recombination must be solved first");

        // Modes are sampled logarithmically in q = sqrt(k^2 + K) on [k_min, k_max] (q = k when flat).
        // Open: k = sqrt(q^2 + |K|) >= sqrt|K|. Closed: q = nu sqrt(K) with integer nu >= 3.
        // Open: q -> 0 (k -> sqrt|K|) still contributes to the lowest multipoles. If the curvature scale is
        // not far below k_min (sqrt|K| > 0.1 k_min) we add 10 extra points, log-spaced in q from
        // 0.01 sqrt|K| up to k_min.
        {
            double q_min = options.k_min;
            if (K > 0.0)
                q_min = std::max(q_min, 3.0 * std::sqrt(K));
            if (!(options.k_max > q_min))
                throw std::invalid_argument("Perturbations: k_max too small for this curvature");
            k_array = Utils::logspace(q_min, options.k_max, options.n_k);
            if (options.dk_max > 0.0) { // log-linear: step min(q dlnk, dk_max), the last step stretched to end at k_max
                const double dlnk = std::log(options.k_max / q_min) / (options.n_k - 1);
                std::vector<double> qs{q_min};
                while (qs.back() < options.k_max)
                    qs.push_back(std::min(qs.back() * std::exp(dlnk), qs.back() + options.dk_max));
                if (qs.size() > 2 && options.k_max - qs[qs.size() - 2] < 0.5 * (qs[qs.size() - 2] - qs[qs.size() - 3]))
                    qs.erase(qs.end() - 2);
                qs.back() = options.k_max;
                k_array = qs;
            }
            const double q_open = 0.01 * std::sqrt(std::fabs(K));
            if (K < 0.0 && std::sqrt(-K) > 0.1 * q_min) {
                auto extra = Utils::logspace(q_open, q_min, 11);
                extra.pop_back();
                k_array.insert(k_array.begin(), extra.begin(), extra.end());
            }
            // Closed: only integer nu = q/sqrt(K) are physical modes. Round every q to the nearest integer nu
            // and remove duplicates; at low nu (where the log spacing is below one unit of nu) this keeps
            // every integer nu, higher up the nodes move by at most half a unit (< 1% in k).
            if (K > 0.0) {
                const double sK = std::sqrt(K);
                std::vector<double> qs;
                for (double q : k_array) {
                    const double qn = std::max(3.0, std::round(q / sK)) * sK;
                    if (qs.empty() || qn > qs.back() * (1.0 + 1e-12))
                        qs.push_back(qn);
                }
                k_array = qs;
            }
            for (double & k : k_array)
                k = std::sqrt(k * k - K);
        }
        x_array = make_x_grid();
        const size_t nx = x_array.size(), nk = k_array.size();
        const int L = options.lmax;
        const bool pol = options.polarization;

        // Background quantities on the x-grid (used for the source functions)
        std::vector<double> Hp(nx), g(nx), exp_mtau(nx);
        for (size_t i = 0; i < nx; i++) {
            const double x = x_array[i];
            Hp[i] = cosmo.Hp_of_x(x);
            g[i] = rec.g_tilde_of_x(x);
            exp_mtau[i] = std::exp(-rec.tau_of_x(x));
        }

        // Results, flat arrays z[ix + nx * ik]
        const size_t N = nx * nk;
        std::vector<double> db(N), vb(N), Phi(N), dPhi(N), Psi(N), Pi(N), ST(N), SE(N), ST0(N);
        const bool comps = options.store_source_components;
        std::vector<double> S_SW(comps ? N : 0), S_ISW(comps ? N : 0), S_Dop(comps ? N : 0), S_Quad(comps ? N : 0);
        std::vector<std::vector<double>> Theta(L + 1, std::vector<double>(N));
        std::vector<std::vector<double>> ThetaP(pol ? L + 1 : 0, std::vector<double>(N));
        std::vector<std::vector<double>> out(output_names.size(), std::vector<double>(N));
        const size_t nsp = species.size();

        std::string error;
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t ik = 0; ik < nk; ik++) {
            try {
                const double k = k_array[ik];
                ModeState st = make_mode(k);
                // Switch points (tight coupling end, species changing their approximation), not before x_array[1]
                const double x_tc = std::max(x_tight_coupling_end(k), x_array[1]);
                std::vector<std::vector<double>> sw(nsp);
                std::vector<double> cand = {x_tc};
                for (size_t i = 0; i < nsp; i++) {
                    for (double xs : species[i]->switch_points(k))
                        sw[i].push_back(std::max(xs, x_array[1]));
                    cand.insert(cand.end(), sw[i].begin(), sw[i].end());
                }
                std::sort(cand.begin(), cand.end());

                // Segments with fixed equations
                struct Segment {
                    double x0, x1;
                    bool tc;
                    std::vector<int> modes;
                    std::vector<double> pts;
                };
                std::vector<double> bp = {options.x_start};
                for (double xb : cand)
                    if (xb > bp.back() && xb < options.x_end)
                        bp.push_back(xb);
                bp.push_back(options.x_end);
                std::vector<Segment> segs;
                for (size_t i = 0; i + 1 < bp.size(); i++) {
                    const double xm = 0.5 * (bp[i] + bp[i + 1]);
                    std::vector<int> modes(nsp);
                    for (size_t j = 0; j < nsp; j++)
                        modes[j] = int(std::count_if(sw[j].begin(), sw[j].end(), [xm](double xs) { return xs <= xm; }));
                    segs.push_back({bp[i], bp[i + 1], xm < x_tc, modes, {}});
                }
                // Grid point ix -> (segment, index in that segment's output). Each segment's points are x0, the
                // grid points inside (x0, x1), and x1; a grid point at a segment boundary belongs to the later segment.
                std::vector<std::pair<int, int>> where(nx);
                for (size_t is = 0; is < segs.size(); is++) {
                    Segment & sg = segs[is];
                    const bool last = (is + 1 == segs.size());
                    sg.pts.push_back(sg.x0);
                    for (size_t ix = 0; ix < nx; ix++) {
                        const double x = x_array[ix];
                        if (x == sg.x0)
                            where[ix] = {int(is), 0};
                        else if (x > sg.x0 && x < sg.x1) {
                            where[ix] = {int(is), int(sg.pts.size())};
                            sg.pts.push_back(x);
                        }
                    }
                    sg.pts.push_back(sg.x1);
                    if (last)
                        where[nx - 1] = {int(is), int(sg.pts.size()) - 1};
                }

                // Integrate segment by segment, changing variables at the start of each
                std::vector<std::vector<std::vector<double>>> seg_data(segs.size());
                std::vector<Layout> lays;
                std::vector<double> y = initial_conditions(st);
                Layout lay_prev = layout(true, std::vector<int>(nsp, 0));
                for (size_t is = 0; is < segs.size(); is++) {
                    const Segment & sg = segs[is];
                    const Layout lay = layout(sg.tc, sg.modes);
                    if (lay.tc != lay_prev.tc || lay.mode != lay_prev.mode) {
                        update(sg.x0, st);
                        y = change_layout(st, lay_prev, lay, y);
                    }
                    ODEFunction rhs = [&](double x, const double * yy, double * dydx) {
                        update(x, st);
                        return lay.tc ? rhs_tight_coupling(st, lay, yy, dydx) : rhs_full(st, lay, yy, dydx);
                    };
                    bool expl = false;
                    for (size_t j = 0; j < nsp; j++)
                        expl = expl || species[j]->prefers_explicit(lay.mode[j]);
                    const ODEStepper method =
                        lay.tc ? options.method_tc : (expl ? options.method_ncdm : options.method_full);
                    ODESolver ode(1e-5, options.ode_abserr, options.ode_relerr);
                    ode.set_finite_difference_jacobian(true);
                    ode.solve(rhs, sg.pts, y, method);
                    seg_data[is] = ode.get_data();
                    y = ode.get_final_data();
                    lays.push_back(lay);
                    lay_prev = lay;
                }

                // Collect results on the grid
                std::vector<double> th, thp, o, dmetric(cosmo.gravity().n_vars());
                for (size_t ix = 0; ix < nx; ix++) {
                    const double x = x_array[ix];
                    const size_t j = ix + nx * ik;
                    const Layout & lay = lays[where[ix].first];
                    const double * yx = seg_data[where[ix].first][where[ix].second].data();
                    update(x, st);
                    double pi;
                    if (lay.tc) {
                        pi = multipoles_tc(x, k, yx[lay.Theta], yx[lay.Theta + 1], th, thp);
                    } else {
                        th.assign(yx + lay.Theta, yx + lay.Theta + L + 1);
                        if (pol)
                            thp.assign(yx + lay.ThetaP, yx + lay.ThetaP + L + 1);
                        pi = pol ? th[2] + thp[0] + thp[2] : th[2];
                    }
                    const MetricAtX m = metric(st, lay, yx, th[2], dmetric.data());
                    db[j] = yx[Layout::delta_b];
                    vb[j] = yx[Layout::v_b];
                    Phi[j] = m.Phi;
                    Psi[j] = m.Psi;
                    dPhi[j] = m.dPhi;
                    Pi[j] = pi;
                    for (int ell = 0; ell <= L; ell++)
                        Theta[ell][j] = th[ell];
                    if (pol)
                        for (int ell = 0; ell <= L; ell++)
                            ThetaP[ell][j] = thp[ell];
                    for (size_t is = 0; is < nsp; is++) {
                        const int i0 = output_index.at(species[is]->output_names().front());
                        o.resize(species[is]->output_names().size());
                        species[is]->outputs(st.b, st.cache[is].data(), lay.mode[is], yx + lay.offset[is], o.data());
                        for (size_t io = 0; io < o.size(); io++)
                            out[i0 + io][j] = o[io];
                    }
                    if (n_gravity_outputs > 0) {
                        const int i0 = int(output_names.size()) - n_gravity_outputs;
                        o.resize(n_gravity_outputs);
                        cosmo.gravity().outputs(st.b, st.gcache.data(), yx + Layout::metric, m, o.data());
                        for (int io = 0; io < n_gravity_outputs; io++)
                            out[i0 + io][j] = o[io];
                    }
                }

                // Temperature source. Derivatives of the composite terms from 1D splines in x.
                std::vector<double> A(nx), B(nx), psi(nx);
                for (size_t ix = 0; ix < nx; ix++) {
                    const size_t j = ix + nx * ik;
                    A[ix] = Hp[ix] * g[ix] * vb[j];
                    B[ix] = Hp[ix] * g[ix] * Pi[j];
                    psi[ix] = Psi[j];
                }
                Spline1D A_s(x_array, A), B_s(x_array, B), Psi_s(x_array, psi);
                std::vector<double> C(nx);
                for (size_t ix = 0; ix < nx; ix++)
                    C[ix] = Hp[ix] * B_s.deriv_x(x_array[ix]);
                Spline1D C_s(x_array, C);

                const double ck = c * k;
                for (size_t ix = 0; ix < nx; ix++) {
                    const double x = x_array[ix];
                    const size_t j = ix + nx * ik;
                    const double sw = g[ix] * (Theta[0][j] + Psi[j]);
                    const double isw = exp_mtau[ix] * (Psi_s.deriv_x(x) - dPhi[j]);
                    const double dop = -A_s.deriv_x(x) / ck;
                    const double quad = g[ix] * 0.25 * Pi[j] + 0.75 / (ck * ck) * C_s.deriv_x(x);
                    ST[j] = sw + isw + dop + quad;
                    ST0[j] = sw + isw;
                    if (comps) {
                        S_SW[j] = sw;
                        S_ISW[j] = isw;
                        S_Dop[j] = dop;
                        S_Quad[j] = quad;
                    }
                    SE[j] = pol ? 0.75 * g[ix] * Pi[j] : 0.0; // numerator of the E-mode source
                }
            } catch (const std::exception & e) {
#pragma omp critical
                error = "k = " + std::to_string(k_array[ik] * Mpc) + "/Mpc: " + e.what();
            }
        }
        if (!error.empty())
            throw std::runtime_error("Perturbations::solve failed for " + error);

        delta_b_spline.create(x_array, k_array, db, "delta_b", spline2d_type(x_array.size(), k_array.size()));
        v_b_spline.create(x_array, k_array, vb, "v_b", spline2d_type(x_array.size(), k_array.size()));
        Phi_spline.create(x_array, k_array, Phi, "Phi", spline2d_type(x_array.size(), k_array.size()));
        dPhidx_spline.create(x_array, k_array, dPhi, "dPhidx", spline2d_type(x_array.size(), k_array.size()));
        Psi_spline.create(x_array, k_array, Psi, "Psi", spline2d_type(x_array.size(), k_array.size()));
        Pi_spline.create(x_array, k_array, Pi, "Pi", spline2d_type(x_array.size(), k_array.size()));
        Source_T_spline.create(x_array, k_array, ST, "Source_T", spline2d_type(x_array.size(), k_array.size()));
        Source_E_num_spline.create(x_array,
                                   k_array,
                                   SE,
                                   "Source_E_numerator",
                                   spline2d_type(x_array.size(), k_array.size()));
        Source_T0_spline.create(x_array, k_array, ST0, "Source_T0", spline2d_type(x_array.size(), k_array.size()));
        if (comps) {
            S_SW_spline.create(x_array, k_array, S_SW, "S_SW", spline2d_type(x_array.size(), k_array.size()));
            S_ISW_spline.create(x_array, k_array, S_ISW, "S_ISW", spline2d_type(x_array.size(), k_array.size()));
            S_Doppler_spline.create(x_array,
                                    k_array,
                                    S_Dop,
                                    "S_Doppler",
                                    spline2d_type(x_array.size(), k_array.size()));
            S_Quad_spline.create(x_array,
                                 k_array,
                                 S_Quad,
                                 "S_Quadrupole",
                                 spline2d_type(x_array.size(), k_array.size()));
        }
        Theta_spline.resize(L + 1);
        for (int ell = 0; ell <= L; ell++)
            Theta_spline[ell].create(x_array,
                                     k_array,
                                     Theta[ell],
                                     "Theta_" + std::to_string(ell),
                                     spline2d_type(x_array.size(), k_array.size()));
        ThetaP_spline.resize(pol ? L + 1 : 0);
        for (int ell = 0; pol && ell <= L; ell++)
            ThetaP_spline[ell].create(x_array,
                                      k_array,
                                      ThetaP[ell],
                                      "ThetaP_" + std::to_string(ell),
                                      spline2d_type(x_array.size(), k_array.size()));
        output_spline.resize(output_names.size());
        for (size_t i = 0; i < output_names.size(); i++)
            output_spline[i].create(x_array,
                                    k_array,
                                    out[i],
                                    output_names[i],
                                    spline2d_type(x_array.size(), k_array.size()));

        solved = true;
        timer.print();
    }

    // =============================================================================
    // Getters and output
    // =============================================================================

    const Spline2D & Perturbations::check(const Spline2D & s) const {
        if (!solved)
            throw std::runtime_error("Perturbations: call solve() first");
        return s;
    }

    double Perturbations::get_Theta(int ell, double x, double k) const {
        if (ell < 0 || ell > options.lmax)
            throw std::out_of_range("Perturbations::get_Theta: ell out of range");
        return check(Theta_spline[ell])(x, k);
    }

    double Perturbations::get_ThetaP(int ell, double x, double k) const {
        if (!options.polarization)
            throw std::runtime_error("Perturbations::get_ThetaP: polarization is off");
        if (ell < 0 || ell > options.lmax)
            throw std::out_of_range("Perturbations::get_ThetaP: ell out of range");
        return check(ThetaP_spline[ell])(x, k);
    }

    double Perturbations::get(const std::string & name, double x, double k) const {
        const auto it = output_index.find(name);
        if (it == output_index.end())
            throw std::runtime_error("Perturbations::get: no species output '" + name + "'");
        return check(output_spline[it->second])(x, k);
    }

    double Perturbations::get_Nu(int ell, double x, double k) const {
        if (ell < 0 || ell > 2)
            throw std::out_of_range("Perturbations::get_Nu: only Nu_0..Nu_2 are stored");
        return get("Nu_" + std::to_string(ell), x, k);
    }

    double Perturbations::get_Source_T_term(SourceTerm term, double x, double k) const {
        if (term == SourceTerm::All)
            return get_Source_T(x, k);
        if (!options.store_source_components)
            throw std::runtime_error("Perturbations::get_Source_T_term: set options.store_source_components = true");
        switch (term) {
            case SourceTerm::SW:
                return check(S_SW_spline)(x, k);
            case SourceTerm::ISW:
                return check(S_ISW_spline)(x, k);
            case SourceTerm::Doppler:
                return check(S_Doppler_spline)(x, k);
            case SourceTerm::Quadrupole:
                return check(S_Quad_spline)(x, k);
            default:
                return get_Source_T(x, k);
        }
    }

    double Perturbations::get_Source_E(double x, double k) const {
        const double z = k * (cosmo.eta0() - cosmo.eta_of_x(x));
        if (z < 1e-6)
            return 0.0; // today: g = 0 (no reionization), and j_ell(z)/z^2 -> 0 for ell > 2
        return get_Source_E_numerator(x, k) / (z * z);
    }

    void Perturbations::info() const {
        std::cout << "\nPerturbations (" << (options.polarization ? "temperature + polarization" : "temperature only")
                  << ", gravity " << cosmo.gravity().info() << "):\n";
        std::cout << "  species: photons, baryons";
        for (const auto & s : species)
            std::cout << ", " << s->info();
        std::cout << "\n"
                  << "  k-grid: " << (k_array.empty() ? options.n_k : int(k_array.size())) << " points in ["
                  << (k_array.empty() ? options.k_min : k_array.front()) * Mpc << ", "
                  << (k_array.empty() ? options.k_max : k_array.back()) * Mpc << "] /Mpc (log in q = sqrt(k^2 + K)"
                  << (K > 0.0 ? ", integer nu" : "") << ")\n"
                  << "  x-grid: " << x_array.size() << " points in [" << options.x_start << ", " << options.x_end
                  << "]\n"
                  << "  lmax = " << options.lmax << "\n"
                  << "  tight coupling ends at x = " << x_tight_coupling_end(options.k_min) << " (k_min), "
                  << x_tight_coupling_end(options.k_max) << " (k_max)\n";
#ifdef _OPENMP
        std::cout << "  OpenMP threads: " << omp_get_max_threads() << "\n";
#endif
        std::cout << "\n";
    }

    void Perturbations::output(double k, const std::string & filename) const {
        check(Phi_spline);
        std::ofstream fp(filename);
        if (!fp)
            throw std::runtime_error("Perturbations::output: cannot open " + filename);
        // Standard columns (0 if a species is absent), then the outputs of any other species
        const std::vector<std::string> standard =
            {"delta_cdm", "v_cdm", "Nu_0", "Nu_1", "Nu_2", "delta_ncdm", "v_ncdm"};
        std::vector<std::string> extra;
        for (const auto & n : output_names)
            if (std::find(standard.begin(), standard.end(), n) == standard.end())
                extra.push_back(n);
        auto opt = [&](const std::string & n, double x) { return has(n) ? get(n, x, k) : 0.0; };
        fp << "# k = " << k * Mpc << " /Mpc. x_tc_end = " << x_tight_coupling_end(k) << "\n"
           << "# x  delta_cdm  v_cdm  delta_b  v_b  Phi  Psi  dPhidx  Theta0  Theta1  Theta2  Source_T"
              "  ThetaP0  ThetaP2  Pi  Source_E  Nu0  Nu2  ThetaP1  Nu1  delta_ncdm  v_ncdm";
        for (const auto & n : extra)
            fp << "  " << n;
        fp << "\n" << std::scientific << std::setprecision(10);
        const bool pol = options.polarization;
        for (double x : x_array) {
            fp << x << " " << opt("delta_cdm", x) << " " << opt("v_cdm", x) << " " << get_delta_b(x, k) << " "
               << get_v_b(x, k) << " " << get_Phi(x, k) << " " << get_Psi(x, k) << " " << get_dPhidx(x, k) << " "
               << get_Theta(0, x, k) << " " << get_Theta(1, x, k) << " " << get_Theta(2, x, k) << " "
               << get_Source_T(x, k) << " " << (pol ? get_ThetaP(0, x, k) : 0.0) << " "
               << (pol ? get_ThetaP(2, x, k) : 0.0) << " " << get_Pi(x, k) << " " << get_Source_E(x, k) << " "
               << opt("Nu_0", x) << " " << opt("Nu_2", x) << " " << (pol ? get_ThetaP(1, x, k) : 0.0) << " "
               << opt("Nu_1", x) << " " << opt("delta_ncdm", x) << " " << opt("v_ncdm", x);
            for (const auto & n : extra)
                fp << " " << get(n, x, k);
            fp << "\n";
        }
    }

} // namespace FML::COSMOLOGY
