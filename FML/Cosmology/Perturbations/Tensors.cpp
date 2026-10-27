#include <FML/Cosmology/Perturbations/Tensors.h>
#include <FML/Cosmology/Background/Gravity.h>
#include <FML/Cosmology/Common.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace FML::COSMOLOGY {

using namespace Constants;

namespace {
  const double SQRT6 = std::sqrt(6.0);
}

Tensors::Tensors(const BackgroundCosmology& c, const RecombinationHistory& r, TensorOptions o)
  : cosmo(c), rec(r), options(o) {
  if (cosmo.get_K() != 0.0) throw std::invalid_argument("Tensors: only implemented for a flat universe");
  if (options.lmax < 5 || options.lmax_nu < 5)
    throw std::invalid_argument("Tensors: lmax and lmax_nu must be >= 5 (the stress needs F_4)");
  if (!(options.k_max > options.k_min && options.k_min > 0.0)) throw std::invalid_argument("Tensors: bad k-range");
}

double Tensors::primordial_power_spectrum(double k) const {
  const auto& p = cosmo.get_params();
  return p.r * p.A_s * std::pow(k * Mpc / p.k_pivot, p.tensor_tilt());
}

// =============================================================================
// Grids
// =============================================================================

void Tensors::make_grids() {
  const double eta0 = cosmo.eta0();
  const double dk_osc = 2.0 * M_PI / (eta0 * options.n_k_per_oscillation);
  const double dlnk = std::log(10.0) / options.n_k_log;
  k_array = {options.k_min};
  while (k_array.back() < options.k_max) {
    const double k = k_array.back();
    const double dk = std::min(k * std::expm1(dlnk), k < options.k_fine ? dk_osc : options.dk_max);
    k_array.push_back(std::min(k + dk, options.k_max));
  }
  // x-grid: coarse early, fine around recombination
  x_array.clear();
  for (double x : Utils::linspace(options.x_start, -8.0, options.n_x_early)) x_array.push_back(x);
  auto mid = Utils::linspace(-8.0, -6.0, options.n_x_recomb);
  x_array.insert(x_array.end(), mid.begin() + 1, mid.end());
  auto late = Utils::linspace(-6.0, 0.0, options.n_x_late);
  x_array.insert(x_array.end(), late.begin() + 1, late.end());
}

// =============================================================================
// One mode
// =============================================================================

void Tensors::evolve(double k, std::vector<double>& h, std::vector<double>& hx, std::vector<double>& ST,
                     std::vector<double>& SP) const {
  const int L = options.lmax, Ln = options.lmax_nu;
  const bool nu = options.neutrinos && cosmo.get_params().Neff > 0.0;
  // state: h, h_x, neutrino F_0..F_Ln, photon F_0..F_L, G_0..G_L
  const int iH = 0, iHX = 1, iN = 2, iF = iN + Ln + 1, iG = iF + L + 1, n = iG + L + 1;
  const Gravity& grav = cosmo.gravity();
  const bool has_ncdm = cosmo.has_massive_nu();

  struct Bg { double K, D, Hp_eta, dlnHp, Og, Ofs, M2, aM, aT; };
  auto background = [&](double x) {
    Bg b;
    const double Hp = cosmo.Hp_of_x(x), E2 = cosmo.E2_of_x(x);
    b.K = c * k / Hp;
    b.D = -rec.dtaudx_of_x(x);
    b.Hp_eta = Hp * cosmo.eta_of_x(x) / c;
    b.dlnHp = cosmo.dHpdx_of_x(x) / Hp;
    b.Og = cosmo.rhoGamma(x) / E2;
    b.Ofs = nu ? (cosmo.rhoNu(x) + (has_ncdm ? 3.0 * cosmo.pNcdm(x) : 0.0)) / E2 : 0.0;
    const TensorCoefficients t = grav.tensor(x, cosmo);
    b.M2 = t.M2; b.aM = t.alpha_M; b.aT = t.alpha_T;
    return b;
  };
  auto stress = [](const double* F) { return F[0] / 15.0 + 2.0 * F[2] / 21.0 + F[4] / 35.0; };
  auto P_full = [&](const double* y) {
    return -(y[iF] / 10.0 + y[iF + 2] / 7.0 + 3.0 * y[iF + 4] / 70.0 - 3.0 * y[iG] / 5.0 + 6.0 * y[iG + 2] / 7.0
             - 3.0 * y[iG + 4] / 70.0) / SQRT6;
  };

  // derivatives; tc: photons tightly coupled (not evolved)
  auto rhs = [&](double x, const double* y, double* dy, bool tc) {
    const Bg b = background(x);
    for (int i = 0; i < n; i++) dy[i] = 0.0;
    const double hx_ = y[iHX];
    double src = 0.0;   // -4 sqrt6 sum Omega_X (F0/15 + 2F2/21 + F4/35), photons in tc: -32/15 Omega_g h'/kappa'
    if (nu) src += -4.0 * SQRT6 * b.Ofs * stress(y + iN);
    if (tc) src += -32.0 / 15.0 * b.Og * hx_ / b.D;
    else    src += -4.0 * SQRT6 * b.Og * stress(y + iF);
    dy[iH] = hx_;
    dy[iHX] = -(2.0 + b.aM + b.dlnHp) * hx_ - (1.0 + b.aT) * b.K * b.K * y[iH] + src / b.M2;
    // neutrinos
    if (nu) {
      const double* F = y + iN;
      double* dF = dy + iN;
      dF[0] = -b.K * F[1] + SQRT6 * hx_;
      for (int l = 1; l < Ln; l++) dF[l] = b.K / (2.0 * l + 1.0) * (l * F[l - 1] - (l + 1.0) * F[l + 1]);
      dF[Ln] = b.K * F[Ln - 1] - (Ln + 1.0) / b.Hp_eta * F[Ln];
    }
    if (!tc) {
      const double P = P_full(y);
      const double* F = y + iF;
      const double* G = y + iG;
      double* dF = dy + iF;
      double* dG = dy + iG;
      dF[0] = -b.K * F[1] - b.D * (F[0] + SQRT6 * P) + SQRT6 * hx_;
      dG[0] = -b.K * G[1] - b.D * (G[0] - SQRT6 * P);
      for (int l = 1; l < L; l++) {
        dF[l] = b.K / (2.0 * l + 1.0) * (l * F[l - 1] - (l + 1.0) * F[l + 1]) - b.D * F[l];
        dG[l] = b.K / (2.0 * l + 1.0) * (l * G[l - 1] - (l + 1.0) * G[l + 1]) - b.D * G[l];
      }
      dF[L] = b.K * F[L - 1] - (L + 1.0) / b.Hp_eta * F[L] - b.D * F[L];
      dG[L] = b.K * G[L - 1] - (L + 1.0) / b.Hp_eta * G[L] - b.D * G[L];
    }
    return 0;
  };

  // End of tight coupling
  double x_tc = -std::log1p(options.tc_z_max);
  for (double x = options.x_start; x < x_tc; x += 1e-3) {
    const double Hp = cosmo.Hp_of_x(x), D = -rec.dtaudx_of_x(x);
    if (D < options.tc_tau_min || c * k / Hp / D > options.tc_k_over_tau) { x_tc = x; break; }
  }
  x_tc = std::max(x_tc, x_array[1]);

  // Initial conditions (CLASS, with the (k tau)^2 correction)
  std::vector<double> y(n, 0.0);
  {
    const double x0 = x_array.front();
    const double kt = k * cosmo.eta_of_x(x0), Hp_eta = cosmo.Hp_of_x(x0) * cosmo.eta_of_x(x0) / c;
    const double h0 = 1.0 / SQRT6;
    const double corr = -h0 * kt * kt / (6.0 + 8.0 / 5.0 * (nu ? nu_fraction : 0.0));
    y[iH] = h0 + corr;
    y[iHX] = 2.0 * corr / Hp_eta;   // h' = 2 corr/tau, h_x = h'/Hp
    if (nu) y[iN] = SQRT6 * corr;
  }

  // Segment points: grid points in [x0, x_tc) and [x_tc, 0]
  std::vector<double> p1, p2;
  for (double x : x_array) (x < x_tc ? p1 : p2).push_back(x);
  p1.push_back(x_tc);
  const bool inserted = p2.empty() || p2.front() != x_tc;   // x_tc is not itself a grid point
  if (inserted) p2.insert(p2.begin(), x_tc);

  std::vector<std::vector<double>> data;
  {
    ODESolver ode(1e-5, options.ode_abserr, options.ode_relerr);
    ode.solve([&](double x, const double* yy, double* dy) { return rhs(x, yy, dy, true); }, p1, y, gsl_odeiv2_step_rk8pd);
    data = ode.get_data();
    y = ode.get_final_data();
    data.pop_back();   // x_tc is not a grid point
  }
  // switch: photon multipoles from tight coupling
  {
    const double D = -rec.dtaudx_of_x(x_tc);
    y[iF] = 4.0 * SQRT6 / 3.0 * y[iHX] / D;
    y[iG] = -SQRT6 / 3.0 * y[iHX] / D;
  }
  const size_t n_tc = data.size();
  {
    ODESolver ode(1e-5, options.ode_abserr, options.ode_relerr);
    ode.set_finite_difference_jacobian(true);
    ode.solve([&](double x, const double* yy, double* dy) { return rhs(x, yy, dy, false); }, p2, y, gsl_odeiv2_step_msbdf);
    const auto& d2 = ode.get_data();
    data.insert(data.end(), d2.begin() + (inserted ? 1 : 0), d2.end());
  }
  const size_t nx = x_array.size();
  if (data.size() != nx) throw std::runtime_error("Tensors: internal grid mismatch");

  h.resize(nx); hx.resize(nx); ST.resize(nx); SP.resize(nx);
  for (size_t i = 0; i < nx; i++) {
    const double x = x_array[i];
    const double* yy = data[i].data();
    const double P = i < n_tc ? -yy[iHX] / (3.0 * -rec.dtaudx_of_x(x)) : P_full(yy);
    h[i] = yy[iH];
    hx[i] = yy[iHX];
    const double g = rec.g_tilde_of_x(x), emt = std::exp(-rec.tau_of_x(x));
    ST[i] = -yy[iHX] * emt + g * P;
    SP[i] = SQRT6 * g * P;
  }
}

std::vector<std::vector<double>> Tensors::mode(double k) const {
  Tensors t(*this);
  if (t.x_array.empty()) t.make_grids();
  if (t.nu_fraction == 0.0) {
    const double x0 = t.x_array.front();
    const double fs = t.cosmo.rhoNu(x0) + (t.cosmo.has_massive_nu() ? 3.0 * t.cosmo.pNcdm(x0) : 0.0);
    t.nu_fraction = fs / (fs + t.cosmo.rhoGamma(x0));
  }
  std::vector<double> h, hx, ST, SP;
  t.evolve(k, h, hx, ST, SP);
  std::vector<std::vector<double>> out;
  for (size_t i = 0; i < h.size(); i++) out.push_back({t.x_array[i], h[i], hx[i]});
  return out;
}

// =============================================================================
// Solve
// =============================================================================

void Tensors::solve() {
  Utils::Timer timer("Tensors::solve");
  make_grids();
  {
    const double x0 = x_array.front();
    const double fs = cosmo.rhoNu(x0) + (cosmo.has_massive_nu() ? 3.0 * cosmo.pNcdm(x0) : 0.0);
    nu_fraction = fs / (fs + cosmo.rhoGamma(x0));
  }
  const size_t nk = k_array.size(), nx = x_array.size();
  std::vector<double> H(nx * nk), HX(nx * nk), S1(nx * nk), S2(nx * nk);
  std::string error;
  #pragma omp parallel for schedule(dynamic, 1)
  for (size_t ik = 0; ik < nk; ik++) {
    try {
      std::vector<double> h, hx, st, sp;
      evolve(k_array[ik], h, hx, st, sp);
      for (size_t ix = 0; ix < nx; ix++) {
        const size_t j = ix + nx * ik;
        H[j] = h[ix]; HX[j] = hx[ix]; S1[j] = st[ix]; S2[j] = sp[ix];
      }
    } catch (const std::exception& e) {
      #pragma omp critical
      error = e.what();
    }
  }
  if (!error.empty()) throw std::runtime_error("Tensors::solve: " + error);
  h_spline.create(x_array, k_array, H, "tensor_h", spline2d_type(x_array.size(), k_array.size()));
  hx_spline.create(x_array, k_array, HX, "tensor_hx", spline2d_type(x_array.size(), k_array.size()));
  ST_spline.create(x_array, k_array, S1, "tensor_ST", spline2d_type(x_array.size(), k_array.size()));
  SP_spline.create(x_array, k_array, S2, "tensor_SP", spline2d_type(x_array.size(), k_array.size()));
  const double t_modes = timer.elapsed();
  line_of_sight();
  solved = true;
  if (Utils::verbose) std::cout << "[Tensors] modes " << t_modes << " s, LOS " << timer.elapsed() - t_modes << " s\n";
  timer.print();
}

// =============================================================================
// Line of sight and C_ell
// =============================================================================

void Tensors::line_of_sight() {
  const int lmax = options.ell_max;
  const double eta0 = cosmo.eta0();
  // ells: all up to 30, then sparser (C_ell are smooth), splined in ell
  std::vector<int> ells;
  for (int l = 2; l <= std::min(30, lmax); l++) ells.push_back(l);
  for (int l = 35; l < std::min(100, lmax); l += 5) ells.push_back(l);
  for (int l = 100; l < std::min(300, lmax); l += 5) ells.push_back(l);   // fine: the TT trough near ell = 170
  for (int l = 300; l < lmax; l += 10) ells.push_back(l);
  if (ells.back() != lmax) ells.push_back(lmax);
  const size_t nl = ells.size();

  // LOS k-grid
  std::vector<double> k_los;
  const double dk_lin = 2.0 * M_PI / (eta0 * options.los_n_k_per_oscillation);
  for (double k = k_array.front(); k < k_array.back(); k += std::min(dk_lin, k * options.los_dlnk_max)) k_los.push_back(k);
  k_los.push_back(k_array.back());
  const size_t nk = k_los.size();

  // LOS x-grid: uniform in eta (Bessel oscillations at k_max) plus a fine grid in x around recombination
  std::vector<double> x_los;
  {
    const double x_lo = -8.5;
    const double eta_lo = cosmo.eta_of_x(x_lo);
    const int n_eta = std::max(200, int(options.los_n_eta_per_oscillation * k_array.back() * (eta0 - eta_lo) / (2.0 * M_PI)));
    const auto xb = Utils::linspace(x_lo - 0.1, 0.0, 20000);
    std::vector<double> etab(xb.size());
    for (size_t i = 0; i < xb.size(); i++) etab[i] = cosmo.eta_of_x(xb[i]);
    Spline1D x_of_eta(etab, xb, "x_of_eta");
    for (double e : Utils::linspace(eta_lo, eta0, n_eta)) x_los.push_back(x_of_eta(e));
    x_los.front() = x_lo;
    x_los.back() = 0.0;
    for (double x = -7.6; x <= -6.2; x += options.los_dx_recomb) x_los.push_back(x);
    std::sort(x_los.begin(), x_los.end());
    std::vector<double> u;
    for (double x : x_los) if (u.empty() || x - u.back() > 1e-10) u.push_back(x);
    x_los = u;
  }
  const size_t nx = x_los.size();
  std::vector<double> w(nx), deta(nx);
  for (size_t i = 0; i < nx; i++) {
    w[i] = 0.5 * ((i > 0 ? x_los[i] - x_los[i - 1] : 0.0) + (i + 1 < nx ? x_los[i + 1] - x_los[i] : 0.0));
    deta[i] = eta0 - cosmo.eta_of_x(x_los[i]);
  }
  deta.back() = 0.0;

  // Radial functions per ell on a uniform z-grid: T: j_l/z^2, E: [-2 j_l + (l+1)(l+2) j_l/z^2 - 2 j_{l+1}/z]/4,
  // B: [(l+2) j_l/z - j_{l+1}]/2 (z -> 0 limits: T = 1/15 for l = 2, E = 1/5 for l = 2, else 0)
  const double z_max = 1.001 * k_los.back() * eta0, dz = 2.0 * M_PI / 32.0;
  const size_t nz = size_t(z_max / dz) + 4;
  std::vector<UniformSpline1D> RT(nl), RE(nl), RB(nl);
  #pragma omp parallel for schedule(dynamic, 1)
  for (size_t il = 0; il < nl; il++) {
    const int l = ells[il];
    std::vector<double> t(nz), e(nz), b(nz);
    for (size_t i = 0; i < nz; i++) {
      const double z = i * dz;
      if (i == 0) {
        t[i] = l == 2 ? 1.0 / 15.0 : 0.0;
        e[i] = l == 2 ? 0.2 : 0.0;
        b[i] = 0.0;
        continue;
      }
      const double jl = Utils::j_ell(l, z), jl1 = Utils::j_ell(l + 1, z);
      t[i] = jl / (z * z);
      e[i] = 0.25 * (-2.0 * jl + (l + 1.0) * (l + 2.0) * jl / (z * z) - 2.0 * jl1 / z);
      b[i] = 0.5 * ((l + 2.0) * jl / z - jl1);
    }
    RT[il].create(0.0, dz, t);
    RE[il].create(0.0, dz, e);
    RB[il].create(0.0, dz, b);
  }

  std::vector<std::vector<double>> DT(nl, std::vector<double>(nk)), DE = DT, DB = DT;
  #pragma omp parallel for schedule(dynamic, 4)
  for (size_t ik = 0; ik < nk; ik++) {
    const double k = k_los[ik];
    std::vector<double> wT(nx), wP(nx), z(nx);
    for (size_t i = 0; i < nx; i++) {
      z[i] = k * deta[i];
      wT[i] = w[i] * ST_spline(x_los[i], k);
      wP[i] = w[i] * SP_spline(x_los[i], k);
    }
    for (size_t il = 0; il < nl; il++) {
      const double nu = ells[il] + 0.5, z_low = std::max(0.0, nu - 10.0 * std::cbrt(nu));
      double sT = 0.0, sE = 0.0, sB = 0.0;
      for (size_t i = 0; i < nx; i++) {
        if (z[i] < z_low) continue;
        sT += wT[i] * RT[il](z[i]);
        sE += wP[i] * RE[il](z[i]);
        sB += wP[i] * RB[il](z[i]);
      }
      const double l = ells[il];
      DT[il][ik] = std::sqrt(3.0 / 8.0 * (l + 2.0) * (l + 1.0) * l * (l - 1.0)) * sT;
      DE[il][ik] = sE;
      DB[il][ik] = sB;
    }
  }

  std::vector<double> P(nk);
  for (size_t ik = 0; ik < nk; ik++) P[ik] = primordial_power_spectrum(k_los[ik]);
  auto cell = [&](const std::vector<double>& X, const std::vector<double>& Y) {
    std::vector<double> f(nk);
    for (size_t ik = 0; ik < nk; ik++) f[ik] = P[ik] * X[ik] * Y[ik] / k_los[ik];
    return 4.0 * M_PI * Utils::trapz(k_los, f);
  };
  // D_ell (dimensionless) in the list, splined in ell (log for the positive spectra)
  std::vector<double> ld(nl), lt(nl), le(nl), lb(nl), dte(nl);
  for (size_t il = 0; il < nl; il++) {
    const double l = ells[il], f = l * (l + 1.0) / (2.0 * M_PI);
    ld[il] = l;
    lt[il] = std::log(f * cell(DT[il], DT[il]));
    le[il] = std::log(f * cell(DE[il], DE[il]));
    lb[il] = std::log(f * cell(DB[il], DB[il]));
    dte[il] = f * cell(DT[il], DE[il]);
  }
  Spline1D sT(ld, lt, "tensor_TT"), sE(ld, le, "tensor_EE"), sB(ld, lb, "tensor_BB"), sX(ld, dte, "tensor_TE");
  for (auto* v : {&tt, &ee, &bb, &te}) v->assign(lmax + 1, 0.0);
  for (int l = 2; l <= lmax; l++) {
    const double f = 2.0 * M_PI / (l * (l + 1.0));
    tt[l] = std::exp(sT(l)) * f;
    ee[l] = std::exp(sE(l)) * f;
    bb[l] = std::exp(sB(l)) * f;
    te[l] = sX(l) * f;
  }
}

// =============================================================================
// Output
// =============================================================================

namespace {
  double dell(const std::vector<double>& v, int l, double T) {
    return l >= 2 && l < int(v.size()) ? l * (l + 1.0) / (2.0 * M_PI) * v[l] * T * T : 0.0;
  }
}
double Tensors::get_Dell_TT(int l) const { return dell(tt, l, cosmo.get_params().TCMB * 1e6); }
double Tensors::get_Dell_EE(int l) const { return dell(ee, l, cosmo.get_params().TCMB * 1e6); }
double Tensors::get_Dell_BB(int l) const { return dell(bb, l, cosmo.get_params().TCMB * 1e6); }
double Tensors::get_Dell_TE(int l) const { return dell(te, l, cosmo.get_params().TCMB * 1e6); }

void Tensors::output_cells(const std::string& filename) const {
  if (!solved) throw std::runtime_error("Tensors::output_cells: call solve() first");
  std::ofstream f(filename);
  if (!f) throw std::runtime_error("Tensors::output_cells: cannot open " + filename);
  const auto& p = cosmo.get_params();
  f << "# tensor modes, r = " << p.r << ", n_t = " << p.tensor_tilt() << "\n";
  f << "# ell  D_ell = ell(ell+1)C_ell/2pi TCMB^2 [muK^2] for TT, EE, BB, TE (unlensed tensor contributions)\n";
  f << std::scientific << std::setprecision(10);
  for (int l = 2; l <= options.ell_max; l++)
    f << l << " " << get_Dell_TT(l) << " " << get_Dell_EE(l) << " " << get_Dell_BB(l) << " " << get_Dell_TE(l) << "\n";
}

void Tensors::info() const {
  const auto& p = cosmo.get_params();
  std::cout << "\nTensors: r = " << p.r << ", n_t = " << p.tensor_tilt() << ", " << k_array.size() << " modes in ["
            << k_array.front() * Mpc << ", " << k_array.back() * Mpc << "] /Mpc, lmax = " << options.lmax
            << ", lmax_nu = " << options.lmax_nu << (options.neutrinos ? "" : " (no neutrino stress)") << "\n";
  if (solved) {
    int lpk = 2;
    for (int l = 2; l <= std::min(200, options.ell_max); l++) if (get_Dell_BB(l) > get_Dell_BB(lpk)) lpk = l;
    std::cout << "  D_ell^TT(10) = " << get_Dell_TT(10) << " muK^2, BB peak at ell = " << lpk << " ("
              << get_Dell_BB(lpk) << " muK^2)\n";
  }
}

} // namespace FML::COSMOLOGY
