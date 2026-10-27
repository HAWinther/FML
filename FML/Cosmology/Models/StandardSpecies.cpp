#include <FML/Cosmology/Models/StandardSpecies.h>
#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/MassiveNeutrinos.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

using namespace Constants;

namespace {
  std::string fmt(double v) {
    std::ostringstream os;
    os << v;
    return os.str();
  }

  // Multipoles N_0..N_lmax of free-streaming radiation in the adiabatic initial conditions:
  //   N_0 = -Psi/2,  N_1 = (ck/6Hp) Psi,
  //   N_2 = -s_2 (c k a)^2 Phi / (12 H0^2 G_eff Omega_nu) / (1 + 5/(2 f_nu)),  N_l = ck s_l/((2l+1)Hp) N_{l-1}
  // (physical shear from the flat-space solution, converted to the symmetric convention with s_2; Omega_nu
  // is a^4 rho of all free-streaming species, G_eff the gravitational constant in units of G)
  std::vector<double> free_streaming_ic(const Perturbations& pert, const ModeAtX& b, const InitialData& ic, int lmax) {
    const double k = b.k;
    std::vector<double> N(lmax + 1, 0.0);
    if (ic.iso) {   // isocurvature: density, velocity and shear given, higher multipoles zero
      N[0] = ic.iso->N0;
      N[1] = ic.iso->N1;
      N[2] = ic.iso->N2;
      return N;
    }
    N[0] = -0.5 * ic.Psi;
    N[1] = b.ckH / 6.0 * ic.Psi;
    N[2] = -pert.s_ell(k, 2) * std::pow(c * k * b.a, 2) * ic.Phi / (12.0 * ic.H0 * ic.H0 * ic.G_eff * ic.OmNu) / (1.0 + 2.5 / ic.f_nu);
    for (int ell = 3; ell <= lmax; ell++) N[ell] = b.ckH * pert.s_ell(k, ell) / (2.0 * ell + 1.0) * N[ell - 1];
    return N;
  }

  // ===========================================================================
  // Cold dark matter (pressureless, no anisotropic stress):
  //   delta' = (ck/Hp) v - 3 Phi',   v' = -v - (ck/Hp) Psi
  // Adiabatic initial conditions: delta = -3/2 Psi, v = -(ck/2Hp) Psi.
  // ===========================================================================
  class CDMPerturbations : public SpeciesPerturbations {
    public:
      explicit CDMPerturbations(const Species& s) : species(s) {}
      std::string name() const override { return "cdm"; }
      int n_vars(int) const override { return 2; }
      int cache_size() const override { return 1; }
      void update_cache(const ModeAtX& b, double* cache) const override { cache[0] = b.H0a2 * species.rho(b.x); }
      void initial_conditions(const ModeAtX& b, const InitialData& ic, double* y) const override {
        if (ic.iso) { y[0] = ic.iso->delta_c; y[1] = ic.iso->v_c; return; }
        y[0] = -1.5 * ic.Psi;
        y[1] = -0.5 * b.ckH * ic.Psi;
      }
      Moments moments(const ModeAtX&, const double* cache, int, const double* y) const override {
        Moments m;
        m.drho = cache[0] * y[0];
        m.flux = -cache[0] * y[1];
        return m;
      }
      void rhs(const ModeAtX& b, const double*, int, const MetricAtX& m, const double* y, double* dydx) const override {
        dydx[0] = b.ckH * y[1] - 3.0 * m.dPhi;
        dydx[1] = -y[1] - b.ckH * m.Psi;
      }
      std::vector<std::string> output_names() const override { return {"delta_cdm", "v_cdm"}; }
      void outputs(const ModeAtX&, const double*, int, const double* y, double* out) const override {
        out[0] = y[0];
        out[1] = y[1];
      }

    private:
      const Species& species;
  };

  // ===========================================================================
  // Massless neutrinos (free streaming), with the curvature factors s_l (= 1 when flat):
  //   Nu_0' = -(ck/Hp) Nu_1 - Phi'
  //   Nu_1' = (ck/3Hp) (Nu_0 - 2 s_2 Nu_2 + Psi)
  //   Nu_l' = ck/((2l+1) Hp) (l s_l Nu_{l-1} - (l+1) s_{l+1} Nu_{l+1})
  //   Nu_L' = (ck/Hp) s_L Nu_{L-1} - c (L+1) cot_K(eta)/Hp Nu_L
  // Moments: delta rho = 4 rho Nu_0, flux = 4 rho Nu_1, shear = 4 rho Nu_2.
  // ===========================================================================
  class MasslessNeutrinoPerturbations : public SpeciesPerturbations {
    public:
      MasslessNeutrinoPerturbations(const Species& s, const Perturbations& p)
        : species(s), pert(p), L(p.get_options().lmax_nu) {
        if (L < 3) throw std::invalid_argument("Perturbations: need lmax_nu >= 3");
      }
      std::string name() const override { return "massless neutrinos"; }
      std::string info() const override { return "massless neutrinos (lmax_nu = " + std::to_string(L) + ")"; }
      int n_vars(int) const override { return L + 1; }
      int cache_size() const override { return 1; }
      void update_cache(const ModeAtX& b, double* cache) const override { cache[0] = b.H0a2 * species.rho(b.x); }
      void initial_conditions(const ModeAtX& b, const InitialData& ic, double* y) const override {
        const auto N = free_streaming_ic(pert, b, ic, L);
        for (int ell = 0; ell <= L; ell++) y[ell] = N[ell];
      }
      Moments moments(const ModeAtX&, const double* cache, int, const double* N) const override {
        Moments m;
        m.drho  = 4.0 * cache[0] * N[0];
        m.dp    = m.drho / 3.0;
        m.flux  = 4.0 * cache[0] * N[1];
        m.shear = 4.0 * cache[0] * N[2];
        return m;
      }
      void rhs(const ModeAtX& b, const double*, int, const MetricAtX& m, const double* N, double* dN) const override {
        const double ckH = b.ckH, k = b.k;
        dN[0] = -ckH * N[1] - m.dPhi;
        dN[1] = ckH / 3.0 * (N[0] - 2.0 * pert.s_ell(k, 2) * N[2] + m.Psi);
        for (int ell = 2; ell < L; ell++)
          dN[ell] = ckH / (2.0 * ell + 1.0) *
                    (ell * pert.s_ell(k, ell) * N[ell - 1] - (ell + 1.0) * pert.s_ell(k, ell + 1) * N[ell + 1]);
        dN[L] = ckH * pert.s_ell(k, L) * N[L - 1] - (L + 1.0) * b.cotK_H * N[L];
      }
      // Only Nu_0..Nu_2 are stored (all that enter the Einstein equations and outputs)
      std::vector<std::string> output_names() const override { return {"Nu_0", "Nu_1", "Nu_2"}; }
      void outputs(const ModeAtX&, const double*, int, const double* N, double* out) const override {
        out[0] = N[0];
        out[1] = N[1];
        out[2] = N[2];
      }

    private:
      const Species& species;
      const Perturbations& pert;
      int L;
  };

  // ===========================================================================
  // Massive neutrinos: the perturbation Psi_ell(q) of the Fermi-Dirac distribution, f = f0(q)[1 + Psi(q)],
  // for n_q_ncdm momenta and ell = 0..lmax_ncdm (Ma & Bertschinger 1995, Newtonian gauge, phi_MB = -Phi,
  // psi_MB = Psi), with eps = sqrt(q^2 + (m a/T)^2):
  //   Psi_0' = -(ck/Hp)(q/eps) Psi_1 + Phi' dln f0/dln q
  //   Psi_1' = (ck/3Hp)(q/eps)(Psi_0 - 2 s_2 Psi_2) - (ck/3Hp)(eps/q) Psi dln f0/dln q
  //   Psi_l' = (ck/Hp)(q/eps)/(2l+1) (l s_l Psi_{l-1} - (l+1) s_{l+1} Psi_{l+1})
  // (mode 0). Deep inside the horizon (k eta > ncdm_fluid_trigger) the hierarchy is replaced by the fluid
  // equations of CLASS's ncdmfa_CLASS (Lesgourgues & Tram 2011) for F = (delta, theta/(ck), sigma) (mode 1):
  //   delta' = -(1 + w)(ck/Hp theta/(ck) + 3 Phi') - 3 (c_a^2 - w) delta
  //   [theta/(ck)]' = -(1 - 3 c_a^2) theta/(ck) + (ck/Hp)[c_a^2 delta/(1 + w) - sigma + Psi]
  //   sigma' = -3 [2/3 - c_a^2 - pseudo_p/(3p) + c/(Hp eta)] sigma
  //            + (8/3) c_vis^2/(1 + w) s_2 (ck/Hp theta/(ck) + 6 Phi'),   c_vis^2 = 3 w c_a^2
  // c_a^2 = w (5 - pseudo_p/p) / (3 (1 + w)) is the adiabatic sound speed.
  //
  // Moments (eqs. 55-56 of Ma & Bertschinger), with C = rho / I_rho:
  //   delta rho      = C sum_i w_i q^2 eps Psi_0,   delta p = C sum_i w_i q^4/(3 eps) Psi_0
  //   (rho+p) theta  = C ck sum_i w_i q^3 Psi_1          (flux = (rho+p) theta / (ck))
  //   (rho+p) sigma  = (2/3) C sum_i w_i q^4/eps Psi_2   (shear = (3/2)(rho+p) sigma)
  // For massless particles Psi_l = -dln f0/dln q Nu_l, and these reduce to 4 rho Nu_l.
  // Initial conditions (relativistic): Psi_l(q) = -dln f0/dln q N_l (times eps/q for l = 1).
  // ===========================================================================
  class MassiveNeutrinoPerturbations : public SpeciesPerturbations {
    public:
      MassiveNeutrinoPerturbations(const MassiveNeutrinoSpecies& s, const Perturbations& p)
        : species(s), pert(p), L(p.get_options().lmax_ncdm), nq(p.get_options().n_q_ncdm) {
        if (L < 3 || nq < 1) throw std::invalid_argument("Perturbations: need lmax_ncdm >= 3 and n_q_ncdm >= 1");
        const MassiveNeutrinos::Quadrature quad(nq);
        q = quad.q;
        wq = quad.w;
        for (double qi : q) dlnf0.push_back(MassiveNeutrinos::dlnf0_dlnq(qi));
      }
      std::string name() const override { return "massive neutrinos"; }
      std::string info() const override {
        return "massive neutrinos (" + std::to_string(nq) + " momenta, lmax_ncdm = " + std::to_string(L) + ")";
      }

      // k eta = ncdm_fluid_trigger
      std::vector<double> switch_points(double k) const override {
        const auto& o = pert.get_options();
        const auto& cosmo = pert.get_cosmo();
        const double trig = o.ncdm_fluid_trigger;
        if (k * cosmo.eta_of_x(o.x_end) <= trig) return {};
        if (k * cosmo.eta_of_x(o.x_start) >= trig) return {o.x_start};
        return {Utils::find_root([&](double x) { return k * cosmo.eta_of_x(x) - trig; }, o.x_start, o.x_end)};
      }
      int n_vars(int mode) const override { return mode == 0 ? nq * (L + 1) : 3; }
      // ~200 variables: the dense linear algebra of an implicit solver makes it ~20x slower
      bool prefers_explicit(int mode) const override { return mode == 0; }

      enum { M_, C_, RHO, P_, W, CA2, PSEUDO, N_CACHE };
      int cache_size() const override { return N_CACHE; }
      void update_cache(const ModeAtX& b, double* cc) const override {
        const double x = b.x;
        cc[M_] = species.M_of_x(x);
        cc[C_] = b.H0a2 * species.prefactor() * std::exp(-4.0 * x);
        cc[RHO] = b.H0a2 * species.rho(x);
        cc[P_] = b.H0a2 * species.p(x);
        const double rho = species.rho(x), p = species.p(x), pp = species.pseudo_p(x);
        cc[W] = p / rho;
        cc[PSEUDO] = pp / p;
        cc[CA2] = cc[W] / (3.0 * (1.0 + cc[W])) * (5.0 - cc[PSEUDO]);
      }

      void initial_conditions(const ModeAtX& b, const InitialData& ic, double* y) const override {
        const auto N = free_streaming_ic(pert, b, ic, L);
        const double M = species.M_of_x(b.x);
        for (int i = 0; i < nq; i++) {
          const double eps_q = std::sqrt(q[i] * q[i] + M * M) / q[i];
          for (int ell = 0; ell <= L; ell++) y[i * (L + 1) + ell] = -dlnf0[i] * N[ell] * (ell == 1 ? eps_q : 1.0);
        }
      }

      void switch_mode(const ModeAtX& b, const double* cc, int, const double* P, double* F) const override {
        const Moments m = moments(b, cc, 0, P);
        const double rpp = cc[RHO] + cc[P_];
        F[0] = m.drho / cc[RHO];
        F[1] = m.flux / rpp;
        F[2] = m.shear / (1.5 * rpp);
      }

      Moments moments(const ModeAtX&, const double* cc, int mode, const double* P) const override {
        Moments m;
        if (mode == 1) {
          const double rpp = cc[RHO] + cc[P_];
          m.drho = cc[RHO] * P[0];
          m.dp = cc[CA2] * m.drho;   // adiabatic, as CLASS's fluid approximation
          m.flux = rpp * P[1];
          m.shear = 1.5 * rpp * P[2];
          return m;
        }
        const double M = cc[M_];
        for (int i = 0; i < nq; i++) {
          const double qi = q[i], eps = std::sqrt(qi * qi + M * M), w = wq[i];
          const double* Y = P + i * (L + 1);
          m.drho  += w * qi * qi * eps * Y[0];
          m.dp    += w * qi * qi * qi * qi / (3.0 * eps) * Y[0];
          m.flux  += w * qi * qi * qi * Y[1];
          m.shear += w * qi * qi * qi * qi / eps * Y[2];
        }
        m.drho *= cc[C_];
        m.dp *= cc[C_];
        m.flux *= cc[C_];
        m.shear *= cc[C_];
        return m;
      }

      void rhs(const ModeAtX& b, const double* cc, int mode, const MetricAtX& met, const double* P,
               double* dP) const override {
        const double k = b.k, s2 = pert.s_ell(k, 2), dPhi = met.dPhi, Psi = met.Psi;
        if (mode == 1) {
          const double w = cc[W], ca2 = cc[CA2], cvis2 = 3.0 * w * ca2;
          dP[0] = -(1.0 + w) * (b.ckH * P[1] + 3.0 * dPhi) - 3.0 * (ca2 - w) * P[0];
          dP[1] = -(1.0 - 3.0 * ca2) * P[1] + b.ckH * (ca2 * P[0] / (1.0 + w) - P[2] + Psi);
          dP[2] = -3.0 * (2.0 / 3.0 - ca2 - cc[PSEUDO] / 3.0 + c / (b.Hp * b.eta)) * P[2] +
                  8.0 / 3.0 * cvis2 / (1.0 + w) * s2 * (b.ckH * P[1] + 6.0 * dPhi);
          return;
        }
        const double M = cc[M_];
        for (int i = 0; i < nq; i++) {
          const double qi = q[i], eps = std::sqrt(qi * qi + M * M);
          const double kq = b.ckH * qi / eps, dlnf = dlnf0[i];
          const double* Y = P + i * (L + 1);
          double* dY = dP + i * (L + 1);
          dY[0] = -kq * Y[1] + dPhi * dlnf;
          dY[1] = kq / 3.0 * (Y[0] - 2.0 * s2 * Y[2]) - b.ckH * eps / (3.0 * qi) * Psi * dlnf;
          for (int ell = 2; ell < L; ell++)
            dY[ell] = kq / (2.0 * ell + 1.0) *
                      (ell * pert.s_ell(k, ell) * Y[ell - 1] - (ell + 1.0) * pert.s_ell(k, ell + 1) * Y[ell + 1]);
          dY[L] = kq * pert.s_ell(k, L) * Y[L - 1] - (L + 1.0) * b.cotK_H * Y[L];
        }
      }

      // delta = delta rho / rho, and v defined like v_cdm: (rho + p) v = -flux
      std::vector<std::string> output_names() const override { return {"delta_ncdm", "v_ncdm"}; }
      void outputs(const ModeAtX& b, const double* cc, int mode, const double* P, double* out) const override {
        const Moments m = moments(b, cc, mode, P);
        out[0] = m.drho / cc[RHO];
        out[1] = -m.flux / (cc[RHO] + cc[P_]);
      }

    private:
      const MassiveNeutrinoSpecies& species;
      const Perturbations& pert;
      int L, nq;
      std::vector<double> q, wq, dlnf0;   // momentum nodes, weights (incl. f0), dln f0/dln q
  };
}

// =============================================================================
// Background
// =============================================================================

std::string PowerLawSpecies::info() const {
  return name_ + " (Omega = " + fmt(Omega0) + (n == 0.0 ? ", constant)" : ", rho ~ a^-" + fmt(n) + ")");
}

std::unique_ptr<SpeciesPerturbations> ColdDarkMatter::perturbations(const Perturbations&) const {
  return std::make_unique<CDMPerturbations>(*this);
}

std::unique_ptr<SpeciesPerturbations> MasslessNeutrinos::perturbations(const Perturbations& pert) const {
  if (Omega_today() <= 0.0) return nullptr;
  if (!pert.get_options().neutrinos) {
    std::cerr << "Perturbations: warning: Omega_nu > 0 but neutrino perturbations are switched off "
                 "(inconsistent model)\n";
    return nullptr;
  }
  return std::make_unique<MasslessNeutrinoPerturbations>(*this, pert);
}

MassiveNeutrinoSpecies::MassiveNeutrinoSpecies(const CosmologyParameters& params)
  : M0(params.M_ncdm()),
    C(params.OmegaGamma() * 15.0 / std::pow(M_PI, 4) * std::pow(params.T_ncdm_over_TCMB, 4)) {}

double MassiveNeutrinoSpecies::rho(double x) const {
  return C * std::exp(-4.0 * x) * MassiveNeutrinos::I_rho(M_of_x(x));
}
double MassiveNeutrinoSpecies::p(double x) const {
  return C * std::exp(-4.0 * x) * MassiveNeutrinos::I_p(M_of_x(x));
}
// p' = C a^-4 (-4 I_p + dI_p/dlnM)
double MassiveNeutrinoSpecies::dpdx(double x) const {
  const double M = M_of_x(x);
  return C * std::exp(-4.0 * x) * (-4.0 * MassiveNeutrinos::I_p(M) + MassiveNeutrinos::dI_p_dlnM(M));
}
double MassiveNeutrinoSpecies::pseudo_p(double x) const {
  return C * std::exp(-4.0 * x) * MassiveNeutrinos::I_pp(M_of_x(x));
}
std::unique_ptr<SpeciesPerturbations> MassiveNeutrinoSpecies::perturbations(const Perturbations& pert) const {
  return std::make_unique<MassiveNeutrinoPerturbations>(*this, pert);
}
std::string MassiveNeutrinoSpecies::info() const {
  return "ncdm (Omega = " + fmt(rho(0.0)) + ", m/T today = " + fmt(M0) + ", non-relativistic (m = 3T) at z = " +
         fmt(M0 / 3.0 - 1.0) + ")";
}

std::string CPLDarkEnergy::info() const {
  return "cpl (Omega = " + fmt(Omega0) + ", w0 = " + fmt(w0) + ", wa = " + fmt(wa) + ", smooth)";
}

} // namespace FML::COSMOLOGY
