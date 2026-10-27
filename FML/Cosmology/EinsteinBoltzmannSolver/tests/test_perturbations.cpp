// =============================================================================
// Tests for Perturbations (temperature only, no neutrinos). Run with `make test`.
// Uses a reduced k-grid to keep the run time short.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <functional>
#include <vector>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

int main() {
  CosmologyParameters p;
  p.Neff = 0.0;
  BackgroundCosmology cosmo(p);
  cosmo.solve();
  RecombinationHistory rec(cosmo);
  rec.solve();
  PerturbationOptions opt;
  opt.n_k = 80;
  Perturbations pert(cosmo, rec, opt);
  pert.solve();

  const auto& ks = pert.get_k_array();
  auto k_near = [&](double k_Mpc) {  // grid k closest to k_Mpc (avoids interpolation in k)
    double best = ks[0];
    for (double k : ks) if (std::fabs(std::log(k * Mpc / k_Mpc)) < std::fabs(std::log(best * Mpc / k_Mpc))) best = k;
    return best;
  };
  const double k_small = ks.front(), k_mid = k_near(0.01), k_large = k_near(0.1);
  const double x0 = opt.x_start;

  std::cout << "Initial conditions (adiabatic, normalised to Phi = 2/3)\n";
  check_close("Phi = 2/3",                pert.get_Phi(x0, k_large), 2.0 / 3.0, 1e-8);
  check_close("Psi = -2/3",               pert.get_Psi(x0, k_large), -2.0 / 3.0, 1e-6);
  check_close("delta_cdm = -3/2 Psi = 1", pert.get_delta_cdm(x0, k_large), 1.0, 1e-8);
  check_close("delta_b = delta_cdm",      pert.get_delta_b(x0, k_large), pert.get_delta_cdm(x0, k_large), 1e-12);
  check_close("delta_cdm = 3 Theta_0 (adiabatic)", 3.0 * pert.get_Theta(0, x0, k_large), pert.get_delta_cdm(x0, k_large), 1e-8);

  std::cout << "Superhorizon evolution\n";
  // Exact superhorizon solution for a matter + radiation perfect fluid (Kodama & Sasaki 1984):
  //   Phi/Phi_0 = [16 sqrt(1+y) + 9 y^3 + 2 y^2 - 8 y - 16] / (10 y^3),  y = a / a_eq.
  // Valid before decoupling; afterwards free-streaming photons add anisotropic stress
  // (Phi + Psi ~ 0.02 around x = -6, confirmed against CLASS), which decays again in MD.
  {
    const double a_eq = p.OmegaR() / p.OmegaM();
    for (double x : {-12.0, -10.0, -8.0}) {
      const double y = std::exp(x) / a_eq;
      const double ks = (16.0 * std::sqrt(1.0 + y) + 9.0 * y * y * y + 2.0 * y * y - 8.0 * y - 16.0) / (10.0 * y * y * y);
      check_rel("Phi superhorizon vs exact fluid solution, x = " + std::to_string(int(x)),
                pert.get_Phi(x, k_small), 2.0 / 3.0 * ks, 5e-4);
    }
  }
  check_rel("Phi(MD, superhorizon) = (9/10)(2/3) at x = -2", pert.get_Phi(-2.0, k_small), 0.6, 1e-3);
  // Sachs-Wolfe: Theta_0 + Psi = Psi/3 in matter domination (radiation still ~20% at recombination)
  {
    const double x_ls = rec.x_last_scattering();
    const double ratio = (pert.get_Theta(0, x_ls, k_small) + pert.get_Psi(x_ls, k_small)) / pert.get_Psi(x_ls, k_small);
    check_close("Sachs-Wolfe: (Theta_0 + Psi)/Psi ~ 1/3 at last scattering", ratio, 1.0 / 3.0, 0.05);
  }

  std::cout << "Subhorizon evolution\n";
  {
    const double h = 1e-3, x = -3.0;
    const double dlogd = (std::log(pert.get_delta_cdm(x + h, k_large)) - std::log(pert.get_delta_cdm(x - h, k_large))) / (2 * h);
    check_close("CDM growth delta ~ a in matter domination (x = -3)", dlogd, 1.0, 0.03);
  }
  check_rel("baryons catch up with CDM after decoupling (x = 0, k = 0.1)",
            pert.get_delta_b(0.0, k_large), pert.get_delta_cdm(0.0, k_large), 0.01);
  check_rel("no anisotropic stress today: Psi = -Phi (k = 0.01)", pert.get_Psi(0.0, k_mid), -pert.get_Phi(0.0, k_mid), 1e-6);

  std::cout << "Constraint equation (not used in the evolution)\n";
  // c^2 k^2 Phi / Hp^2 = (3/2) sum_i Omega_i(x) [delta_i - 3 Hp v_i / (ck)] at late times (radiation negligible)
  for (double k : {k_mid, k_large}) {
    const double x = 0.0;
    const double ckH = c * k / cosmo.Hp_of_x(x);
    const double lhs = ckH * ckH * pert.get_Phi(x, k);
    const double rhs = 1.5 * (cosmo.OmegaCDM_of_x(x) * (pert.get_delta_cdm(x, k) - 3.0 * pert.get_v_cdm(x, k) / ckH) +
                              cosmo.OmegaB_of_x(x) * (pert.get_delta_b(x, k) - 3.0 * pert.get_v_b(x, k) / ckH));
    check_rel("Poisson constraint today, k = " + std::to_string(k * Mpc).substr(0, 6) + "/Mpc", lhs, rhs, 1e-4);
  }

  std::cout << "Tight coupling switch\n";
  for (double k : {k_mid, k_large}) {
    const double xtc = pert.x_tight_coupling_end(k), d = 1e-4;
    const std::string lab = " (k = " + std::to_string(k * Mpc).substr(0, 6) + "/Mpc)";
    double max_jump = 0.0;
    for (int ell = 0; ell <= 2; ell++) {
      const double amp = std::fabs(pert.get_Theta(ell, xtc - 0.05, k)) + std::fabs(pert.get_Theta(ell, xtc + 0.05, k));
      max_jump = std::max(max_jump, std::fabs(pert.get_Theta(ell, xtc + d, k) - pert.get_Theta(ell, xtc - d, k)) / amp);
    }
    check_close("Theta_0,1,2 continuous across TC end" + lab, max_jump, 0.0, 1e-2);
    check("TC ends before recombination" + lab, xtc <= -std::log1p(opt.tc_z_max) + 1e-12);
  }

  std::cout << "Derived quantities\n";
  {
    double max_err = 0.0;
    for (double x = -15.0; x < -0.1; x += 0.37) {
      const double h = 1e-4;
      const double fd = (pert.get_Phi(x + h, k_mid) - pert.get_Phi(x - h, k_mid)) / (2 * h);
      max_err = std::max(max_err, std::fabs(fd - pert.get_dPhidx(x, k_mid)));
    }
    check_close("dPhi/dx vs finite difference of Phi", max_err, 0.0, 1e-5);
  }
  {
    bool finite = true;
    double s_early = 0.0, s_max = 0.0, x_peak = 0.0;
    for (double k : ks)
      for (double x : pert.get_x_array()) finite &= std::isfinite(pert.get_Source_T(x, k));
    for (double x = -12.0; x < 0.0; x += 0.001) {
      const double s = std::fabs(pert.get_Source_T(x, k_large));
      if (x < -9.0) s_early = std::max(s_early, s);
      if (s > s_max) { s_max = s; x_peak = x; }
    }
    check("source function finite everywhere", finite);
    check("source function ~ 0 before recombination", s_early < 1e-6 * s_max);
    check_close("source function peaks at recombination", x_peak, rec.x_visibility_peak(), 0.15);
  }

  std::cout << "Polarization\n";
  {
    // Tight coupling relations: Theta^P_0 = 5/4 Theta_2, Theta^P_2 = Theta_2 / 4, Pi = Theta_2 + P_0 + P_2
    const double x = -10.0;
    const double T2 = pert.get_Theta(2, x, k_mid);
    check_rel("TC: Theta^P_0 = (5/4) Theta_2", pert.get_ThetaP(0, x, k_mid), 1.25 * T2, 1e-6);
    check_rel("TC: Theta^P_2 = Theta_2 / 4",   pert.get_ThetaP(2, x, k_mid), 0.25 * T2, 1e-6);
    check_rel("TC: Pi = (5/2) Theta_2",         pert.get_Pi(x, k_mid), 2.5 * T2, 1e-6);
    check_rel("TC: Theta_2 = -(8/15) ck/(Hp tau') Theta_1", T2,
              -8.0 / 15.0 * c * k_mid / (cosmo.Hp_of_x(x) * rec.dtaudx_of_x(x)) * pert.get_Theta(1, x, k_mid), 1e-6);
    const double xr = -6.9;
    check_rel("full system: Pi = Theta_2 + Theta^P_0 + Theta^P_2", pert.get_Pi(xr, k_large),
              pert.get_Theta(2, xr, k_large) + pert.get_ThetaP(0, xr, k_large) + pert.get_ThetaP(2, xr, k_large), 1e-6);
    double max_jump = 0.0;
    const double xtc = pert.x_tight_coupling_end(k_large), d = 1e-4;
    for (int ell = 0; ell <= 2; ell++) {
      const double amp = std::fabs(pert.get_ThetaP(ell, xtc - 0.05, k_large)) + std::fabs(pert.get_ThetaP(ell, xtc + 0.05, k_large));
      max_jump = std::max(max_jump, std::fabs(pert.get_ThetaP(ell, xtc + d, k_large) - pert.get_ThetaP(ell, xtc - d, k_large)) / amp);
    }
    check_close("Theta^P_0,1,2 continuous across TC end (k = 0.1/Mpc)", max_jump, 0.0, 1e-2);
    double smax = 0.0, x_peak = 0.0;
    for (double x = -8.0; x < -1.0; x += 0.001) {
      const double s = std::fabs(pert.get_Source_E(x, k_large));
      if (s > smax) { smax = s; x_peak = x; }
    }
    check_close("E-mode source peaks at recombination (no reionization)", x_peak, rec.x_visibility_peak(), 0.15);
  }
  {
    PerturbationOptions o2 = opt;
    o2.polarization = false;
    o2.n_k = 20;
    Perturbations pT(cosmo, rec, o2);
    pT.solve();
    const double k = pT.get_k_array()[15], x = -6.9;
    check_rel("temperature-only mode: Pi = Theta_2", pT.get_Pi(x, k), pT.get_Theta(2, x, k), 1e-12);
    bool threw = false;
    try { pT.get_ThetaP(0, x, k); } catch (const std::runtime_error&) { threw = true; }
    check("temperature-only mode: Theta^P not available", threw);
  }

  check("Neff = 0: neutrino hierarchy inactive", !pert.has_neutrinos());

  std::cout << "Massless neutrinos (Neff = 3.046)\n";
  {
    CosmologyParameters pn;  // fiducial, Neff = 3.046
    BackgroundCosmology cosmo_n(pn);
    cosmo_n.solve();
    RecombinationHistory rec_n(cosmo_n);
    rec_n.solve();
    PerturbationOptions on;
    on.n_k = 30;
    Perturbations pn_pert(cosmo_n, rec_n, on);
    pn_pert.solve();
    check("Neff > 0: neutrino hierarchy active", pn_pert.has_neutrinos());

    const double f_nu = pn.OmegaNu() / (pn.OmegaGamma() + pn.OmegaNu());
    const double Psi0 = -1.0 / (1.5 + 0.4 * f_nu);
    const double ks = pn_pert.get_k_array().front(), kl = pn_pert.get_k_array()[25];
    const double xs = on.x_start;
    check_rel("IC: Psi = -1/(3/2 + 2 f_nu/5)", pn_pert.get_Psi(xs, kl), Psi0, 1e-5);
    check_rel("IC: Phi = -(1 + 2 f_nu/5) Psi", pn_pert.get_Phi(xs, kl), -(1.0 + 0.4 * f_nu) * Psi0, 1e-8);
    check_rel("IC: Nu_0 = Theta_0", pn_pert.get_Nu(0, xs, kl), pn_pert.get_Theta(0, xs, kl), 1e-8);
    // Phi + Psi from the neutrino quadrupole (the Nu_2 initial condition) matches the analytic value
    check_rel("IC: neutrino anisotropic stress, Phi + Psi = -(2 f_nu/5) Psi",
              pn_pert.get_Phi(xs, kl) + pn_pert.get_Psi(xs, kl), -0.4 * f_nu * Psi0, 1e-4);
    // Curvature normalisation R = 1: Phi = 3/5 on superhorizon scales in matter domination
    check_rel("superhorizon Phi = 3/5 in matter domination (R = 1)", pn_pert.get_Phi(-2.0, ks), 0.6, 2e-3);
    check_close("neutrino anisotropic stress gone in MD: (Phi + Psi)/Phi",
                (pn_pert.get_Phi(-2.0, ks) + pn_pert.get_Psi(-2.0, ks)) / pn_pert.get_Phi(-2.0, ks), 0.0, 1e-2);
    // Free streaming damps the neutrino density well inside the horizon, unlike CDM
    check("free streaming: |Nu_0| << |delta_cdm| inside the horizon (k = 0.1/Mpc, x = -3)",
          std::fabs(pn_pert.get_Nu(0, -3.0, kl)) < 1e-3 * std::fabs(pn_pert.get_delta_cdm(-3.0, kl)));
  }

  std::cout << "Spatial curvature\n";
  {
    // Run flat, nearly flat and curved models on a small k-grid (fiducial physics)
    auto run = [](double OmK, int n_k, std::vector<double>& out_k,
                  std::function<void(const BackgroundCosmology&, const Perturbations&)> f) {
      CosmologyParameters pc;
      pc.OmegaK = OmK;
      BackgroundCosmology cc(pc);
      cc.solve();
      RecombinationHistory rc(cc);
      rc.solve();
      PerturbationOptions oc;
      oc.n_k = n_k;
      oc.k_min = 0.0002 / Mpc;
      oc.k_max = 0.15 / Mpc;
      Perturbations pc_pert(cc, rc, oc);
      pc_pert.solve();
      out_k = pc_pert.get_k_array();
      f(cc, pc_pert);
    };
    // Flat limit: Omega_K = 1e-8 reproduces Omega_K = 0
    std::vector<double> kf, kn;
    std::vector<double> phi_flat, phi_near;
    auto sample = [](std::vector<double>& v) {
      return [&v](const BackgroundCosmology&, const Perturbations& pp) {
        for (double k : pp.get_k_array())
          for (double x : {-10.0, -7.0, -3.0, 0.0}) v.push_back(pp.get_Phi(x, k) + pp.get_delta_b(x, k) * 1e-3);
      };
    };
    run(0.0, 12, kf, sample(phi_flat));
    run(1e-8, 12, kn, sample(phi_near));
    double max_diff = 0.0;
    for (size_t i = 0; i < phi_flat.size(); i++) max_diff = std::max(max_diff, std::fabs(phi_near[i] / phi_flat[i] - 1.0));
    check_close("flat limit: Omega_K = 1e-8 reproduces flat Phi, delta_b", max_diff, 0.0, 1e-6);

    for (double OmK : {0.05, -0.05}) {
      const std::string lab = std::string("Omega_K = ") + (OmK > 0 ? "+0.05: " : "-0.05: ");
      std::vector<double> kc;
      run(OmK, 12, kc, [&](const BackgroundCosmology& cc, const Perturbations& pp) {
        const double K = cc.get_K();
        check_rel(lab + "K passed on to Perturbations", pp.get_K(), K, 1e-15);
        // Modes sampled in q = sqrt(k^2 + K): first q = 0.01 sqrt|K| (open) or 3 sqrt(K) (closed)
        const double q0 = std::sqrt(kc.front() * kc.front() + K);
        check_rel(lab + "first mode q = 0.01 sqrt|K| (open), nearest integer nu to max(k_min, 3 sqrt K) (closed)", q0,
                  K > 0 ? std::round(std::max(0.0002 / Mpc, 3.0 * std::sqrt(K)) / std::sqrt(K)) * std::sqrt(K)
                        : 0.01 * std::sqrt(-K), 1e-10);
        // s_l = sqrt(q^2 - l^2 K)/k; closed universe: s_nu = 0
        const double k = kc[5], q = std::sqrt(k * k + K);
        check_rel(lab + "s_l = sqrt(q^2 - l^2 K)/k (l = 4)", pp.s_ell(k, 4), std::sqrt(q * q - 16.0 * K) / k, 1e-12);
        if (K > 0) {
          bool integer = true;
          for (double kk : kc) {
            const double nu = std::sqrt(kk * kk / K + 1.0);
            integer &= std::fabs(nu - std::round(nu)) < 1e-6;
          }
          check(lab + "closed: perturbation modes have integer nu", integer);
          const double k3 = std::sqrt(8.0 * K);  // nu = 3
          check_close(lab + "closed: s_3 = 0 for nu = 3", pp.s_ell(k3, 3), 0.0, 1e-7);
        }
        // Initial conditions are those of the flat case (curvature negligible at x_start)
        const double f_nu = cc.get_params().OmegaNu() / (cc.get_params().OmegaGamma() + cc.get_params().OmegaNu());
        check_rel(lab + "IC: Psi = -1/(3/2 + 2 f_nu/5)", pp.get_Psi(-18.0, kc[8]), -1.0 / (1.5 + 0.4 * f_nu), 1e-4);
        // Poisson constraint with curvature at late times:
        //   c^2 (k^2 - 3K) Phi / Hp^2 = (3/2) sum_i Omega_i(x) [delta_i - 3 Hp v_i / (ck)]
        // Well inside the horizon today (radiation perturbations, neglected here, are not negligible
        // for superhorizon modes)
        const int nkc = kc.size();
        for (int ik : {nkc - 8, nkc - 4, nkc - 1}) {
          const double kk = kc[ik], x = 0.0;
          const double ckH = c * kk / cc.Hp_of_x(x);
          const double lhs = ckH * ckH * (1.0 - 3.0 * K / (kk * kk)) * pp.get_Phi(x, kk);
          const double rhs = 1.5 * (cc.OmegaCDM_of_x(x) * (pp.get_delta_cdm(x, kk) - 3.0 * pp.get_v_cdm(x, kk) / ckH) +
                                    cc.OmegaB_of_x(x) * (pp.get_delta_b(x, kk) - 3.0 * pp.get_v_b(x, kk) / ckH));
          check_rel(lab + "Poisson constraint with (k^2 - 3K), k = " + std::to_string(kk * Mpc).substr(0, 7) + "/Mpc",
                    lhs, rhs, 2e-4);
        }
      });
    }
  }

  std::cout << "Log-linear k-grid (dk_max > 0)\n";
  {
    PerturbationOptions o;
    o.k_min = 1e-4 / Mpc; o.k_max = 0.05 / Mpc; o.n_k = 12; o.dk_max = 0.005 / Mpc;
    o.n_x_early = 50; o.n_x_recomb = 200; o.n_x_late = 100;
    Perturbations pl(cosmo, rec, o);
    pl.solve();
    const auto& kl = pl.get_k_array();
    const double r = std::pow(o.k_max / o.k_min, 1.0 / (o.n_k - 1));
    bool ok = true;
    for (size_t i = 1; i + 1 < kl.size(); i++)
      ok = ok && kl[i] > kl[i - 1] && kl[i] - kl[i - 1] <= std::min(kl[i - 1] * (r - 1.0), o.dk_max) * (1 + 1e-12);
    const double last = kl.back() - kl[kl.size() - 2];
    check("ends at k_min and k_max", std::fabs(kl.front() / o.k_min - 1) < 1e-12 && std::fabs(kl.back() / o.k_max - 1) < 1e-12);
    check("steps min(k dlnk, dk_max), last one within [0.5, 1.5] dk_max", ok && last > 0.5 * o.dk_max && last < 1.5 * o.dk_max);
    check("linear part present (more modes than n_k)", int(kl.size()) > o.n_k);
    check_rel("same Phi(x = 0) as the log grid at the shared k_max", pl.get_Phi(0.0, o.k_max), pert.get_Phi(0.0, o.k_max), 2e-3);
  }

  pert.info();
  return Test::summary();
}
