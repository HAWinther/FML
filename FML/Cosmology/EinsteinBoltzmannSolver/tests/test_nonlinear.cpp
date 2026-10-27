// =============================================================================
// Tests of the nonlinear matter power spectrum (Halofit.h). Run with `make test`.
//
//   * Power law Delta^2_L = (k/k0)^{n+3}: n_eff = n, C = 0 and sigma(1/k_sigma) = 1 analytically
//   * Linear regime: P_NL = P_L on large scales
//   * P_NL/P_L against CLASS v3.4 (non_linear = halofit) for Planck 2018 at z = 0 and 1 (here with precision = fast;
//     with the default precision and pert_k_max = 1/Mpc P_NL agrees to < 1e-3, python/compare_class_halofit.py)
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Spectra/Halofit.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
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

int main() {
  Utils::verbose = false;

  std::cout << "Power-law spectrum\n";
  {
    std::vector<double> lnk, D2;
    for (double l = std::log(1e-4); l <= std::log(1e3); l += 0.01) lnk.push_back(l);
    const double n = -1.5, k0 = 0.2;   // Delta^2 = (k/k0)^{1.5}
    for (double l : lnk) D2.push_back(std::pow(std::exp(l) / k0, n + 3.0));
    std::vector<double> NL;
    double ks, ne, C;
    const bool ok = Halofit::nonlinear(lnk, D2, 0.3, 0.7, -1.0, NL, ks, ne, C);
    // sigma^2(R) = Gamma((n+3)/2)/2 (k0 R)^{-(n+3)}  ->  k_sigma = k0 [Gamma((n+3)/2)/2]^{-1/(n+3)}
    const double ks_exact = k0 * std::pow(std::tgamma((n + 3.0) / 2.0) / 2.0, -1.0 / (n + 3.0));
    check("nonlinear scale found", ok);
    check_rel("k_sigma", ks, ks_exact, 1e-4);
    check_close("n_eff = n", ne, n, 1e-4);
    check_close("C = 0", C, 0.0, 1e-4);
  }

  std::cout << "Planck 2018: P_NL/P_L against CLASS (non_linear = halofit)\n";
  {
    const RunSettings s = ParameterFile::read("", {"preset=planck2018", "precision=fast", "pert_k_max=1"});
    BackgroundCosmology bg(s.cosmo);
    bg.solve(s.bg_x_start, s.bg_x_end, s.bg_n_points);
    RecombinationHistory rec(bg, s.rec);
    rec.solve();
    Perturbations pert(bg, rec, s.pert);
    pert.solve();
    PowerSpectrum ps(bg, rec, pert, s.ps);
    Halofit hf(bg, [&](double x, double k) { return ps.get_matter_power_spectrum_extrapolated(x, k); });
    const double h = s.cosmo.h;
    // Delta^2_Q ~ Delta^2_L exp(-y/4) for y = k/k_sigma -> 0: linear only as k -> 0 (0.3% below at 0.005 h/Mpc, as CLASS)
    check_close("linear on large scales (k = 2e-4 h/Mpc, z = 0)", hf.ratio(0.0, 2e-4 * h / Mpc), 1.0, 5e-4);
    check_close("ratio = 1 above z_max", hf.ratio(-std::log(20.0), 1.0 / Mpc), 1.0, 0.0);
    // CLASS: pk/pk_lin at k = 0.1, 0.3, 1 h/Mpc
    struct Ref { double z, k, r; };
    for (const Ref& r : {Ref{0.0, 0.1, 1.01289}, Ref{0.0, 0.3, 1.54388}, Ref{0.0, 1.0, 6.00259},
                         Ref{1.0, 0.1, 1.00999}, Ref{1.0, 0.3, 1.24156}, Ref{1.0, 1.0, 3.28870}}) {
      const double v = hf.ratio(-std::log1p(r.z), r.k * h / Mpc);
      check_rel("z = " + std::to_string(int(r.z)) + ", k = " + std::to_string(r.k).substr(0, 3) + " h/Mpc", v, r.r, 5e-3);
    }
  }

  return Test::summary();
}
