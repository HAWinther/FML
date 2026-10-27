// =============================================================================
// Tests of the tensor modes (Tensors.h). Run with `make test`.
//
//   * Superhorizon: h = 1/sqrt6 (the normalization of r) while k eta << 1
//   * Einstein-de Sitter without anisotropic stress: h = 3 j_1(k eta)/(k eta) h_0 for a mode entering in the
//     matter era (with the radiation shift of eta)
//   * Neutrino free streaming damps modes that enter in the radiation era by the factor 0.80 of Weinberg (2004)
//   * Modified gravity: with alpha_M the subhorizon amplitude scales as 1/(a M*) (WKB), Horndeski alpha_M ~
//     Omega_DE on a LCDM background; f(R) with f_R0 = 1e-5 is GR within 1e-4
//   * Spectra against CLASS (tensor modes alone, r = 0.1, n_t = 0, Planck 2018; python/compare_class_tensors.py)
//   * C_ell proportional to r; the consistency relation for n_t
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/Gravity.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Perturbations/Tensors.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

namespace {
  struct World {
    RunSettings s;
    BackgroundCosmology bg;
    RecombinationHistory rec;
    explicit World(const RunSettings& s_in) : s(s_in), bg(s.cosmo), rec((bg.solve(s.bg_x_start, s.bg_x_end, s.bg_n_points), bg), s.rec) {
      rec.solve();
    }
  };
  // value of the row of mode() closest to x
  std::vector<double> at_x(const std::vector<std::vector<double>>& m, double x) {
    size_t best = 0;
    for (size_t i = 0; i < m.size(); i++) if (std::fabs(m[i][0] - x) < std::fabs(m[best][0] - x)) best = i;
    return m[best];
  }
  // WKB amplitude sqrt(h^2 + (h_x / (ck/Hp))^2) at x
  double amplitude(const World& w, const std::vector<std::vector<double>>& m, double k, double x) {
    const auto r = at_x(m, x);
    const double K = c * k / w.bg.Hp_of_x(r[0]);
    return std::sqrt(r[1] * r[1] + std::pow(r[2] / K, 2));
  }
}

int main() {
  Utils::verbose = false;
  const double h0 = 1.0 / std::sqrt(6.0);

  std::cout << "Superhorizon modes\n";
  {
    World w(ParameterFile::read("", {"preset=planck2018", "r=0.1"}));
    Tensors t(w.bg, w.rec);
    const double k = 1e-5 / Mpc;   // k eta0 = 0.14
    const auto m = t.mode(k);
    check_rel("h = 1/sqrt6 at x = -10", at_x(m, -10.0)[1], h0, 1e-6);
    check_rel("... still at z ~ 10 (k eta << 1)", at_x(m, -2.4)[1], h0, 2e-3);
  }

  std::cout << "Einstein-de Sitter, no anisotropic stress: h = 3 j1(k eta)/(k eta) / sqrt6\n";
  {
    RunSettings s = ParameterFile::read("", {"h=0.7", "OmegaB=0.05", "Neff=0", "reionization=false"});
    s.cosmo.OmegaCDM = 1.0 - s.cosmo.OmegaB - s.cosmo.OmegaGamma();   // Omega_Lambda = 0
    World w(s);
    TensorOptions o;
    o.neutrinos = false;
    Tensors t(w.bg, w.rec, o);
    for (double kM : {3e-4, 1e-3}) {
      const double k = kM / Mpc;
      const auto m = t.mode(k);
      for (double x : {-1.0, 0.0}) {
        // matter + radiation: a(eta) has eta + eta_1 = 2 sqrt(Omega_m a + Omega_r)/(H0 Omega_m) and Hp = 2/(eta + eta_1)
        // up to (eta_1/eta)^2 ~ 1e-4, i.e. the matter-era solution in the shifted time eta + eta_1
        const double Om = w.s.cosmo.OmegaB + w.s.cosmo.OmegaCDM, Or = w.s.cosmo.OmegaGamma();
        const double u = k * 2.0 * c * std::sqrt(Om * std::exp(x) + Or) / (w.bg.get_H0() * Om);
        const double expected = 3.0 * (std::sin(u) / (u * u) - std::cos(u) / u) / u * h0;
        check_close("k = " + std::to_string(kM).substr(0, 6) + "/Mpc, x = " + std::to_string(x).substr(0, 4) +
                    " (k eta = " + std::to_string(u).substr(0, 4) + ")", at_x(m, x)[1], expected, 2e-3 * h0);
      }
    }
  }

  std::cout << "Neutrino free streaming (Weinberg 2004: amplitude x 0.8026 for f_nu = 0.405)\n";
  {
    World w(ParameterFile::read("", {"preset=planck2018", "r=0.1"}));
    TensorOptions on, off;
    off.neutrinos = false;
    // k eta_eq ~ 100: deep in the radiation era (k = 0.05/Mpc, k eta_eq ~ 5, gives 0.84; 0.2/Mpc 0.81)
    const double k = 1.0 / Mpc;
    const auto m1 = Tensors(w.bg, w.rec, on).mode(k), m0 = Tensors(w.bg, w.rec, off).mode(k);
    double ratio = 0.0;
    int n = 0;
    for (double x = -4.0; x <= -3.0; x += 0.05, n++) ratio += amplitude(w, m1, k, x) / amplitude(w, m0, k, x);
    ratio /= n;
    const double fnu = w.bg.rhoNu(-15.0) / (w.bg.rhoNu(-15.0) + w.bg.rhoGamma(-15.0));
    std::cout << std::setprecision(5) << "  f_nu = " << fnu << ", amplitude ratio with/without neutrino stress = " << ratio << "\n";
    check_close("ratio = 0.80 (Weinberg; f_nu = 0.409 here, 0.405 there)", ratio, 0.8026, 0.003);
  }

  std::cout << "Modified gravity\n";
  {
    // Horndeski alpha_M = c_M Omega_DE on the LCDM background: subhorizon amplitude ~ 1/(a M*)
    World g(ParameterFile::read("", {"preset=planck2018", "r=0.1"}));
    World mg(ParameterFile::read("", {"preset=planck2018", "r=0.1", "model=horndeski", "hd_cK=1", "hd_cB=0.2", "hd_cM=0.5"}));
    const double k = 0.05 / Mpc;
    const auto m0 = Tensors(g.bg, g.rec).mode(k), m1 = Tensors(mg.bg, mg.rec).mode(k);
    const double M2_0 = mg.bg.gravity().tensor(0.0, mg.bg).M2, M2_e = mg.bg.gravity().tensor(-5.0, mg.bg).M2;
    const double wkb = std::sqrt(M2_e / M2_0);
    double ratio = amplitude(mg, m1, k, 0.0) / amplitude(g, m0, k, 0.0);
    std::cout << std::setprecision(6) << "  M*^2 today / at z = 150: " << M2_0 / M2_e << ", amplitude ratio today " << ratio
              << " (WKB " << wkb << ")\n";
    check_close("Horndeski c_M = 0.5: h ~ 1/M* (WKB)", ratio, wkb, 2e-3);
    check("... the background is LCDM (same H)", std::fabs(mg.bg.H_of_x(0.0) / g.bg.H_of_x(0.0) - 1.0) < 1e-10);
    // f(R): M*^2 = 1 + f_R ~ 1 - 1e-5
    World fr(ParameterFile::read("", {"preset=planck2018", "r=0.1", "model=fr", "fR0=1e-5"}));
    const auto tc = fr.bg.gravity().tensor(0.0, fr.bg);
    check_close("f(R): M*^2 today = 1 + f_R0", tc.M2, 1.0 - 1e-5, 1e-7);
    const auto mf = Tensors(fr.bg, fr.rec).mode(k);
    check_rel("f(R) f_R0 = 1e-5: h today as in GR (1e-3, different background)", amplitude(fr, mf, k, 0.0),
              amplitude(g, m0, k, 0.0), 1e-3);
  }

  std::cout << "Spectra against CLASS (r = 0.1, n_t = 0, Planck 2018, unlensed tensor contributions)\n";
  {
    World w(ParameterFile::read("", {"preset=planck2018", "r=0.1", "n_t_consistency=false", "n_t=0"}));
    Tensors t(w.bg, w.rec);
    t.solve();
    // CLASS v3.4 with python/compare_class_tensors.py --precise: D_ell [muK^2] (TT, EE, BB)
    struct Ref { int l; double tt, ee, bb; };
    for (const Ref& r : {Ref{2, 49.15, 8.233e-4, 1.509e-3}, Ref{10, 37.33, 2.818e-4, 3.048e-4},
                         Ref{80, 17.61, 8.770e-3, 6.485e-3}, Ref{200, 0.6819, 2.146e-3, 1.254e-3}}) {
      const std::string l = "ell = " + std::to_string(r.l) + ": ";
      check_rel(l + "TT", t.get_Dell_TT(r.l), r.tt, 6e-3);
      check_rel(l + "EE", t.get_Dell_EE(r.l), r.ee, 6e-3);
      check_rel(l + "BB", t.get_Dell_BB(r.l), r.bb, 6e-3);
    }
    check("zero above ell_max", t.get_Dell_BB(t.ell_max() + 1) == 0.0);
    // C_ell proportional to r
    World w2(ParameterFile::read("", {"preset=planck2018", "r=0.05", "n_t_consistency=false", "n_t=0"}));
    Tensors t2(w2.bg, w2.rec);
    t2.solve();
    check_rel("C_ell proportional to r", t2.get_Dell_BB(80), 0.5 * t.get_Dell_BB(80), 1e-10);
    check_close("consistency relation n_t = -r/8 (2 - r/8 - n_s)",
                ParameterFile::read("", {"preset=planck2018", "r=0.1"}).cosmo.tensor_tilt(),
                -0.1 / 8.0 * (2.0 - 0.1 / 8.0 - 0.9665), 1e-15);
  }

  return Test::summary();
}
