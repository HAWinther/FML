#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Background/BBN.h>
#include <FML/Cosmology/Background/MassiveNeutrinos.h>
#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace FML::COSMOLOGY {

double CosmologyParameters::H0() const {
  return h * Constants::H0_over_h;
}

double CosmologyParameters::OmegaGamma() const {
  // rho_gamma c^2 = (pi^2/15) (k_b T)^4 / (hbar c)^3, Omega = 8 pi G rho / (3 H0^2)
  using namespace Constants;
  const double H = H0();
  return 2.0 * (M_PI * M_PI / 30.0) * std::pow(k_b * TCMB, 4) / (std::pow(hbar, 3) * std::pow(c, 5))
         * 8.0 * M_PI * G / (3.0 * H * H);
}

double CosmologyParameters::OmegaNu() const {
  return (Neff - N_ncdm()) * (7.0 / 8.0) * std::pow(4.0 / 11.0, 4.0 / 3.0) * OmegaGamma();
}

double CosmologyParameters::N_ncdm() const {
  return has_massive_nu() ? std::pow(T_ncdm_over_TCMB / std::cbrt(4.0 / 11.0), 4) : 0.0;
}

double CosmologyParameters::M_ncdm() const { return MassiveNeutrinos::M0(m_nu, T_ncdm()); }

double CosmologyParameters::OmegaNcdm() const {
  if (!has_massive_nu()) return 0.0;
  // rho_ncdm / rho_gamma = (15/pi^4) (T_ncdm/TCMB)^4 I_rho(M)
  return OmegaGamma() * 15.0 / std::pow(M_PI, 4) * std::pow(T_ncdm_over_TCMB, 4) * MassiveNeutrinos::I_rho(M_ncdm());
}

double CosmologyParameters::OmegaLambda() const {
  return 1.0 - (OmegaB + OmegaCDM + OmegaGamma() + OmegaNu() + OmegaNcdm() + OmegaK);
}

void CosmologyParameters::validate() const {
  auto require = [](bool ok, const std::string& msg) {
    if (!ok) throw std::invalid_argument("CosmologyParameters: " + msg);
  };
  require(h > 0.0,                    "h must be > 0");
  require(OmegaB > 0.0,               "OmegaB must be > 0");
  require(OmegaCDM >= 0.0,            "OmegaCDM must be >= 0");
  require(Neff >= 0.0,                "Neff must be >= 0");
  require(m_nu >= 0.0,                "m_nu must be >= 0");
  require(!has_massive_nu() || Neff >= N_ncdm(), "Neff must include the massive species (Neff >= N_ncdm)");
  require(TCMB > 0.0,                 "TCMB must be > 0");
  require(Yp >= 0.0 && Yp < 1.0,      "Yp must be in [0,1)");
  if (Yp_bbn) {
    try {
      helium_fraction();
    } catch (const std::exception& e) {
      throw std::invalid_argument(std::string("CosmologyParameters: Yp_bbn: ") + e.what());
    }
  }
  require(A_s > 0.0,                  "A_s must be > 0");
  require(k_pivot > 0.0,              "k_pivot must be > 0");
  require(r >= 0.0,                   "r must be >= 0");
  {
    double c2 = 0.0;
    for (int m = 0; m < 4; m++) {
      require(std::fabs(iso_c[m]) <= 1.0, "isocurvature correlations c_ad_* must be in [-1, 1]");
      if (iso_f[m] != 0.0) c2 += iso_c[m] * iso_c[m];
    }
    require(c2 < 1.0, "isocurvature: the correlations must satisfy sum c_ad^2 < 1");
    require(iso_f[0] == 0.0 || OmegaCDM > 0.0, "CDM isocurvature needs OmegaCDM > 0");
    require((iso_f[2] == 0.0 && iso_f[3] == 0.0) || Neff > 0.0, "neutrino isocurvature needs Neff > 0");
  }
  require(r == 0.0 || OmegaK == 0.0,  "tensor modes (r > 0) are only implemented for a flat universe");
  require(delta_z_reion > 0.0 && delta_z_He_reion > 0.0, "reionization widths must be > 0");
  try {
    Models::validate(*this);
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string("CosmologyParameters: ") + e.what());
  }
}

double CosmologyParameters::helium_fraction() const {
  return Yp_bbn ? BBN::helium_fraction(OmegaB * h * h, Neff) : Yp;
}

double CosmologyParameters::model_param(const std::string& name) const { return Models::parameter(*this, name); }

void CosmologyParameters::print(std::ostream& os) const {
  const auto f = os.flags();
  os << std::setprecision(6);
  os << "Cosmological parameters:\n"
     << "  model        = " << model << "\n";
  for (const auto& p : Models::find(model).parameters)
    os << "  " << std::left << std::setw(12) << p.name << std::right << " = " << model_param(p.name) << "\n";
  os
     << "  h            = " << h        << "\n"
     << "  H0           = " << H0() / Constants::H0_over_h * 100.0 << " km/s/Mpc\n"
     << "  OmegaB       = " << OmegaB   << "\n"
     << "  OmegaCDM     = " << OmegaCDM << "\n"
     << "  OmegaGamma   = " << OmegaGamma() << "\n"
     << "  OmegaNu      = " << OmegaNu()    << "  (Neff = " << Neff << ", massless " << Neff - N_ncdm() << ")\n";
  if (has_massive_nu())
    os << "  OmegaNcdm    = " << OmegaNcdm() << "  (m_nu = " << m_nu << " eV, T_ncdm/TCMB = " << T_ncdm_over_TCMB
       << ", m/T today = " << M_ncdm() << ")\n";
  os
     << "  OmegaK       = " << OmegaK   << "\n"
     << "  OmegaLambda  = " << OmegaLambda() << "  (closure in GR; some models set their own, see BackgroundCosmology)\n"
     << "  TCMB         = " << TCMB     << " K\n"
     << "  Yp           = " << helium_fraction() << (Yp_bbn ? " (BBN)" : "") << "\n"
     << "  reionization = " << (reionization ? "on" : "off");
  if (reionization)
    os << "  (z_reion = " << z_reion << ", dz = " << delta_z_reion
       << "; z_He_reion = " << z_He_reion << ", dz = " << delta_z_He_reion << ")";
  os << "\n"
     << "  A_s          = " << A_s      << "\n"
     << "  n_s          = " << n_s      << "\n"
     << "  k_pivot      = " << k_pivot  << " /Mpc\n";
  for (int m = 0; m < 4; m++)
    if (iso_f[m] != 0.0)
      os << "  isocurvature " << (m == 0 ? "cdi" : m == 1 ? "bi " : m == 2 ? "nid" : "niv") << ": f = " << iso_f[m]
         << ", n = " << iso_n[m] << ", c_ad = " << iso_c[m] << "\n";
  if (r > 0.0)
    os << "  r            = " << r << "\n"
       << "  n_t          = " << tensor_tilt() << (n_t_consistency ? " (consistency relation)" : "") << "\n";
  os.flags(f);
}

void CosmologyParameters::print() const { print(std::cout); }

void CosmologyParameters::write(const std::string& filename) const {
  std::ofstream fp(filename);
  if (!fp) throw std::runtime_error("CosmologyParameters::write: cannot open " + filename);
  fp << std::setprecision(17);
  fp << "h = " << h << "\nOmegaB = " << OmegaB << "\nOmegaCDM = " << OmegaCDM << "\nOmegaK = " << OmegaK
     << "\nNeff = " << Neff << "\nm_nu = " << m_nu << "\nT_ncdm_over_TCMB = " << T_ncdm_over_TCMB << "\nTCMB = " << TCMB << "\nYp = " << helium_fraction()
     << "\nreionization = " << (reionization ? 1 : 0) << "\nz_reion = " << z_reion
     << "\ndelta_z_reion = " << delta_z_reion << "\nz_He_reion = " << z_He_reion
     << "\ndelta_z_He_reion = " << delta_z_He_reion << "\nA_s = " << A_s << "\nn_s = " << n_s
     << "\nk_pivot = " << k_pivot << "\nf_cdi = " << iso_f[0] << "\nn_cdi = " << iso_n[0] << "\nc_ad_cdi = " << iso_c[0]
     << "\nf_bi = " << iso_f[1] << "\nn_bi = " << iso_n[1] << "\nc_ad_bi = " << iso_c[1]
     << "\nf_nid = " << iso_f[2] << "\nn_nid = " << iso_n[2] << "\nc_ad_nid = " << iso_c[2]
     << "\nf_niv = " << iso_f[3] << "\nn_niv = " << iso_n[3] << "\nc_ad_niv = " << iso_c[3] << "\nr = " << r << "\nn_t = " << tensor_tilt() << "\nmodel = " << model << "\n";
  for (const auto& p : Models::find(model).parameters) fp << p.name << " = " << model_param(p.name) << "\n";
}

CosmologyParameters CosmologyParameters::Planck2018() {
  CosmologyParameters p;
  p.h        = 0.6766;
  p.OmegaB   = 0.02242 / (p.h * p.h);
  p.OmegaCDM = 0.11933 / (p.h * p.h);
  p.OmegaK   = 0.0;
  p.Neff     = 3.046;
  p.TCMB     = 2.7255;
  p.Yp       = 0.2454;
  p.z_reion  = 7.67;
  p.A_s      = std::exp(3.047) * 1e-10;
  p.n_s      = 0.9665;
  p.k_pivot  = 0.05;
  return p;
}

} // namespace FML::COSMOLOGY
