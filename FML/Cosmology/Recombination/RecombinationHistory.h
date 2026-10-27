#ifndef FML_COSMOLOGY_RECOMBINATION_RECOMBINATIONHISTORY_H
#define FML_COSMOLOGY_RECOMBINATION_RECOMBINATIONHISTORY_H

// =============================================================================
// Recombination history: free electron fraction, optical depth, visibility function.
//
// Physics (two recombination models, options.model):
//   * RECFAST (default; Seager, Sasselov & Scott 1999, version 1.5 with Wong, Moss & Scott 2008 and
//     Rubino-Martin et al. 2010): effective three-level atoms for hydrogen and neutral helium, evolved
//     together with the baryon temperature:
//       x_H:  Peebles-type equation with the case-B rate of Pequignot et al. (fudge 1.125) and the
//             double-Gaussian correction to the Lyman-alpha escape (K_H),
//       x_He: He I singlet with the 2s-1s two-photon rate, the Sobolev escape of the 2^1P line, its
//             continuum-opacity (H photoionization) correction, and the triplet channel (fudge 0.86),
//     Saha for H, He I and He II while everything is close to equilibrium, then Saha for H with the helium
//     equation, then the full system once x_H < 0.995 (as RECFAST). n_He/n_H = Yp/(3.9715 (1 - Yp)).
//   * Peebles (the course model): X_e = n_e / n_H (can exceed 1 when helium is ionized; max 1 + 2 f_He).
//   * Saha equilibrium for H+, He+ and He++ (solved together for n_e) while the hydrogen
//     ionization fraction x_H > Xe_saha_limit (0.99); then the Peebles equation for x_H,
//     with the remaining (small) helium ionization still from Saha.
//     options.helium = false: hydrogen only, helium neutral throughout.
//   * n_H = (1 - Yp) n_b, n_He = f_He n_H, f_He = Yp / (4 (1 - Yp)).
//   * Baryon temperature: evolved with Compton coupling to the photons (default), or
//     T_b = T_CMB(x) if options.evolve_Tb = false.
//   * Reionization (if options.reionization and params.reionization), CAMB parametrization:
//       X_e = X_e^rec + (1 + f_He - X_e^rec) [1 + tanh((y_re - y)/dy)]/2
//                     + f_He [1 + tanh((z_He - z)/dz_He)]/2,
//       y = (1+z)^{3/2}, dy = (3/2) sqrt(1 + z_re) dz_re.
//
// Optical depth  tau(x) = int_x^0 c n_e sigma_T / H dx',  tau' = -c n_e sigma_T / H
// Visibility     g(x)   = -tau' e^{-tau},   int g dx = 1
//
// Usage:
//   RecombinationHistory rec(cosmo);
//   rec.solve();
//   rec.info();
//   rec.output("data/recombination.txt");
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>

#include <string>

namespace FML::COSMOLOGY {

struct RecombinationOptions {
  enum class Model { Recfast, Peebles };
  Model  model          = Model::Recfast;
  // Physics switches
  bool   helium         = true;   // Helium ionization
  bool   reionization   = true;   // Reionization (also requires params.reionization)
  double peebles_fudge  = 1.0;    // Peebles: multiplies the case-B recombination rate alpha^(2)
  double Xe_saha_limit  = 0.99;   // Peebles: switch from Saha to Peebles when x_H drops below this
  double recfast_fudge_H  = 1.125;  // RECFAST: hydrogen fudge factor (1.14 - 0.015 with the Gaussian correction)
  double recfast_fudge_He = 0.86;   // RECFAST: helium fudge factor (2^1P continuum opacity)
  double recfast_relerr   = 1e-10;  // RECFAST: relative tolerance of the ODE solver
  bool   evolve_Tb      = true;   // Evolve T_b (else T_b = T_gamma)

  // Grid. tau is integrated backwards from x_end, which must therefore be 0 (today).
  double x_start  = -20.0;
  double x_end    = 0.0;
  int    n_points = 40000;
};

class RecombinationHistory {
  public:
    explicit RecombinationHistory(const BackgroundCosmology& cosmo,
                                  RecombinationOptions options = RecombinationOptions());
    virtual ~RecombinationHistory() = default;

    void solve();

    // Free electrons
    double Xe_of_x(double x) const;           // n_e / n_H (total, including reionization)
    double ne_of_x(double x) const;           // [1/m^3]
    double nH_of_x(double x) const;           // Hydrogen number density [1/m^3]
    double nb_of_x(double x) const;           // Baryon number density n_b = rho_b / m_H [1/m^3]
    double Tb_of_x(double x) const;           // Baryon temperature [K]

    // Optical depth and derivatives
    double tau_of_x(double x) const;
    double dtaudx_of_x(double x) const;
    double ddtauddx_of_x(double x) const;

    // Visibility function and derivatives
    double g_tilde_of_x(double x) const;
    double dgdx_tilde_of_x(double x) const;
    double ddgddx_tilde_of_x(double x) const;

    // Photon-baryon fluid
    double R_of_x(double x) const;            // R = 4 rho_gamma / (3 rho_b)
    double cs_of_x(double x) const;           // Sound speed [m/s]
    double sound_horizon_of_x(double x) const; // s(x) = int c_s dt/a  (comoving) [m]

    // Characteristic epochs
    double x_last_scattering() const;         // tau(x) = 1
    double x_visibility_peak() const;         // max of g_tilde
    double x_Xe_half() const;                 // X_e = 0.5 (Peebles)
    double x_Xe_half_saha() const;            // X_e = 0.5 if only Saha were used

    // Saha prediction for X_e (useful for comparison and as initial condition)
    double Xe_saha_of_x(double x) const;
    // n_He / n_H (0 without helium); RECFAST uses m_He/m_H = 3.9715, the Peebles model 4
    double f_He() const { return options.helium ? f_He_atoms() : 0.0; }
    double f_He_atoms() const {
      return Yp / ((options.model == RecombinationOptions::Model::Recfast ? 3.9715 : 4.0) * (1.0 - Yp));
    }
    bool   has_reionization() const { return reion; }
    double tau_reionization() const;          // optical depth from z = 0 to z = 50
    // Optical depth of the reionization model alone (X_e from the tanh steps, without the residual
    // ionization of recombination) for a given z_reion, from z = 0 to 50. Needs no solve(); differs from
    // tau_reionization() by ~1e-4. Used to sample tau instead of z_reion.
    double tau_reionization_model(double z_reion) const;

    const BackgroundCosmology& get_cosmo() const { return cosmo; }
    const RecombinationOptions& get_options() const { return options; }
    bool is_solved() const { return solved; }

    virtual void info() const;
    void output(const std::string& filename, double x_min = -12.0, double x_max = 0.0,
                int n_points = 5000) const;

  protected:
    const BackgroundCosmology& cosmo;
    RecombinationOptions options;
    double Yp;

    // Saha equilibrium for H and He at T = T_gamma. If xH_fixed >= 0 the hydrogen fraction
    // is taken as given (Peebles regime) and only helium is in Saha equilibrium.
    struct SahaResult { double Xe, xH, xHeII, xHeIII; };
    SahaResult saha(double x, double xH_fixed = -1.0) const;

    // Reionization: X_e = Xe_rec + (1 + f_He - Xe_rec) T_H + f_He T_He; returns X_e and dX_e/dx
    void reionize(double x, double Xe_rec, double dXe_rec, double& Xe, double& dXe) const;
    bool reion = false;

    // RECFAST: dx_H/dx and dx_He/dx (x_He = n_HeII/n_He) at total X_e and baryon temperature Tb
    double dxHdx_recfast(double x, double xH, double Xe, double Tb) const;
    double dxHedx_recfast(double x, double xHe, double xH, double Xe, double Tb) const;
    // Right-hand side of the Peebles equation dX_e/dx at baryon temperature Tb
    virtual double dXedx_peebles(double x, double Xe, double Tb) const;
    // dT_b/dx = -2 T_b + (Gamma_C / H) (T_gamma - T_b)
    virtual double dTbdx(double x, double Xe, double Tb) const;

  private:
    bool solved = false;
    Spline1D log_Xe_spline;
    Spline1D log_Tb_spline;
    Spline1D tau_spline;
    Spline1D ddtauddx_spline;
    Spline1D g_spline;
    Spline1D dgdx_spline;     // spline of g' (for g'')
    Spline1D sound_horizon_spline;

    std::vector<double> dlogXedx;  // d ln X_e / dx on the grid (exact in the Peebles regime)

    void solve_Xe(const std::vector<double>& x_array);
    void solve_Xe_recfast(const std::vector<double>& x_array, std::vector<double>& Xe, std::vector<double>& dXedx,
                          std::vector<double>& Tb, size_t& i_ode);
    void finish_Xe(const std::vector<double>& x_array, std::vector<double>& Xe, std::vector<double>& dXedx,
                   const std::vector<double>& Tb, size_t i_spline_end);
    void solve_tau(const std::vector<double>& x_array);
    void solve_sound_horizon(const std::vector<double>& x_array);
    void require_solved(const char* func) const;
};

} // namespace FML::COSMOLOGY

#endif
