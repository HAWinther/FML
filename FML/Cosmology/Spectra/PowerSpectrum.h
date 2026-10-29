#ifndef FML_COSMOLOGY_SPECTRA_POWERSPECTRUM_H
#define FML_COSMOLOGY_SPECTRA_POWERSPECTRUM_H

// =============================================================================
// CMB power spectra TT, EE, TE (line of sight integration) and matter power spectrum.
//
//   Theta_ell(k)   = int dx S_T(k,x) j_ell(k (eta0 - eta(x)))
//   Theta^E_ell(k) = sqrt((ell+2)!/(ell-2)!) int dx S_E(k,x) j_ell(k (eta0 - eta(x)))
//   C_ell^XY       = 4 pi int dk/k P_R(k) X_ell(k) Y_ell(k)
//
// With spatial curvature K (k^2 = Laplacian eigenvalue, q^2 = k^2 + K, nu = q/sqrt|K|,
// chi = sqrt|K| (eta0 - eta), r = sin_K(chi)/sqrt|K|, s_2 = sqrt(1 - 3K/k^2)):
//   Theta_ell(k) = int dx [ S_T0 Phi_ell + S_T1 (sqrt|K|/k) Phi_ell'
//                          + S_T2 (3 (|K|/k^2) Phi_ell'' + Phi_ell) / (2 s_2) ],
//     S_T0 = g (Theta_0 + Psi) + e^{-tau} (Psi' - Phi'),  S_T1 = -g v_b,  S_T2 = g Pi / 2,
//   Theta^E_ell(k) = sqrt((ell+2)!/(ell-2)!) int dx (3 g Pi / 4) Phi_ell / (s_2 (k r)^2),
// with Phi_ell = Phi^nu_ell(chi) the hyperspherical Bessel functions (' = d/dchi). These are the
// radial functions of CLASS (Lesgourgues & Tram 2014). In the flat limit they reduce to j_ell,
// j_ell' and (3 j_ell'' + j_ell)/2, and integrating the last two by parts gives S_T. The C_ell
// integral keeps the form above in k; in a closed universe the modes are discrete, k_nu =
// sqrt(K (nu^2 - 1)) with integer nu >= 3, and the sum over nu is done as CLASS does it
// (trapezoid on integer nu plus the endpoint correction for the first mode).
// EE and TE are computed when the perturbations include polarization.
// Several statistically independent components (add_component before solve, e.g. the isocurvature
// decomposition of Isocurvature.h): each is a Perturbations solution normalized to the same P_R(k), and every
// quadratic quantity (C_ell, C_ell^phiphi, P(k)) is the sum over components. Transfer functions (get_Theta_ell,
// get_Delta_M, transfer.txt) are those of the first component.
// Tensor modes (Tensors.h, set_tensors before solve): their C_ell are added to the unlensed TT, EE, TE, give the
// unlensed BB, and are lensed together with the scalar spectra.
//   P_R(k)       = A_s (k / k_pivot)^{n_s - 1 + alpha_s/2 ln(k/k_pivot)}   (primordial curvature power)
//
// The transfer functions from Perturbations are normalised per unit curvature
// perturbation (Phi = 2/3 initially without neutrinos, matching CLASS's R = 1).
//
// Lensing potential (options.lensing), single source plane at the peak of the visibility function:
//   phi_ell(k)     = int dx [-(c/Hp) r(chi_* - chi) / (r(chi_*) r(chi)) (Psi - Phi)] j_ell(k chi)
//   C_ell^phiphi   = 4 pi int dk/k P_R(k) phi_ell(k)^2
// (Phi_ell^nu and sin_K in a curved universe), followed by the lensed spectra from Lensing::lens.
//
// Nonlinear (options.nonlinear = Halofit, Halofit.h): P_NL = P_L * ratio(x, k) from Takahashi et al. (2012); the
// Weyl potential in the Limber part of C_ell^phiphi is scaled by sqrt(P_NL/P_L) (as CLASS), so the lensed
// spectra include nonlinear lensing.
//
// Matter power spectrum of the comoving-gauge density contrast of CDM + baryons (+ massive neutrinos):
//   Delta_M(k,x) = sum_i rho_i [delta_i - 3 (1 + w_i) Hp v_i/(ck)] / sum_i rho_i,   P(k,x) = 2 pi^2 / k^3 P_R(k)
//   Delta_M^2
//
// Two-point functions (PowerSpectrumTwoPoint.cpp, numerics in TwoPoint.h):
//   * CMB angular correlation functions C^TT(theta), xi_+, xi_-, xi_X from the (lensed) C_ell.
//   * Matter correlation function xi(r) = int dk k^2/(2 pi^2) P(k) j_0(kr), and sigma_R / sigma_8.
//   * Angular power spectrum of the matter (or galaxy, with a linear bias) density projected with a
//     redshift distribution n(z), W(chi) = n(z) dz/dchi normalised to 1:
//       exact:   C_ell = 4 pi int dk/k P_R(k) Delta_ell(k)^2,  Delta_ell(k) = int dchi W(chi) b Delta_M(k, chi) j_ell(k
//       chi) Limber:  C_ell = int dchi W(chi)^2 b^2 / r(chi)^2 P((ell + 1/2)/r(chi), chi)
//     and its angular correlation function w(theta). Density term only (no redshift-space distortions,
//     lensing magnification or GR corrections), linear theory. The exact projection needs a flat universe.
//   Beyond the perturbation k-range Delta_M is extrapolated: Delta_M ~ k^2 below k_min (superhorizon) and
//   Delta_M ~ k^2 T_EH(k) above k_max, with the no-wiggle transfer function of Eisenstein & Hu (1998)
//   anchored at k_max (P(k) within ~1.5% of CLASS up to 20/Mpc from k_max = 0.3/Mpc; it misses the residual
//   BAO wiggles, so set PerturbationOptions::k_max ~ 1/Mpc when those matter).
//
// Numerics:
//   * j_ell(z) tabulated once per ell on a uniform z-grid (UniformSpline1D).
//   * LOS integral on an x-grid that is the union of a uniform grid in eta (resolving the
//     Bessel oscillations up to k_max) and a fine uniform grid in x around recombination
//     (resolving the visibility function); trapezoidal rule. Skips the region where
//     j_ell is negligible.
//   * k-grid for the LOS/C_ell integrals: dk = min(2 pi / (eta0 N_osc), k dlnk).
//   * C_ell computed for a list of ell and splined in ell.
//   * Defaults are converged to < 1e-4 in D_ell (doubling all LOS resolutions changes D_ell
//     by ~6e-5); the perturbation grids are converged to ~1e-3.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Spectra/TwoPoint.h>

#include <memory>
#include <string>
#include <vector>

namespace FML::COSMOLOGY {

    class Tensors;
    class Halofit;

    struct PowerSpectrumOptions {
        int ell_max = 2000;

        // CMB lensing (Lensing.h): computes the lensing potential C_ell^phiphi and the lensed TT, EE, TE, BB
        // up to ell_max. The unlensed spectra and C_ell^phiphi are then computed up to
        // ell_max + lensing_delta_ell (default max(500, ell_max/3)), since lensing moves power in ell. Lensed TT,
        // EE, TE are then accurate to < 1e-3 up to ell_max; lensed BB to 1% up to ell_max - 500 and ~4% at ell_max
        // (lensing_delta_ell = 1500: 1% at ell_max). The k-range must reach the largest ell: k_max r_* > ell.
        bool lensing = false;
        int lensing_delta_ell = -1;
        // C_ell^phiphi from the full LOS integral below this ell and the Limber approximation above it.
        // Limber needs the Weyl potential at k = (ell + 1/2)/chi, beyond the perturbation k_max at low z;
        // there it is extrapolated with the asymptotic CDM shape Phi ~ ln(beta k)/k^2 (beta from the
        // logarithmic slope at k_max).
        int lensing_limber_ell = 30;
        // Nonlinear matter power for the lensing potential and P_NL(k) (Halofit.h)
        enum class Nonlinear { None, Halofit };
        Nonlinear nonlinear = Nonlinear::None;
        std::vector<int> ells; // ell values where C_ell is computed (default list if empty)
        int ell_step = 25;     // spacing of the default list above ell = 100

        // LOS k-grid
        double k_min = 0.0; // 0 -> use the perturbation k-range
        double k_max = 0.0;
        double n_k_per_oscillation = 4.0; // points per 2 pi / eta0 in k
        double dlnk_max = 0.02;           // max spacing in ln k (small k)

        // LOS x-grid
        double x_start_los = -8.5;          // source is negligible before this
        double n_eta_per_oscillation = 5.0; // points per Bessel oscillation at k_max (uniform in eta)
        double dx_recomb = 0.003;           // spacing in x around recombination
        double x_recomb_min = -7.6, x_recomb_max = -6.2;

        // Bessel tables
        double bessel_points_per_2pi = 32.0;

        // Curved universes always use the hyperspherical Bessel functions Phi^nu_ell(chi) with the full
        // radial functions (see line_of_sight_integration_radial). Setting this also uses that code path
        // for a flat universe (Phi = j_ell), which tests it against the integrated-by-parts flat path.
        bool use_radial_functions = false;

        // Curved universes: for modes with nu = q/sqrt|K| >= flat_approximation_nu, Phi^nu_ell(chi) is
        // approximated by a rescaled spherical Bessel function (as in CLASS, Lesgourgues & Tram 2014),
        //   Phi^nu_ell(chi) ~ A(chi) j_ell(chi sqrt(l(l+1)) / chi_tp),  chi_tp = asinh or asin(sqrt(l(l+1))/nu),
        // which matches the turning point and is accurate to ~1e-3 for nu >> ell. The flat j_ell, j_ell'
        // are tabulated once, so these modes cost the same as in a flat universe. The approximation is
        // accurate near the turning point but drifts in phase far beyond it (chi - chi_tp large). Such
        // oblique projections matter less for high-nu modes, which are also Silk damped: with the CLASS
        // default nu >= 4000 the error in C_ell is ~5e-4 (Omega_K = 0.05, ell_max = 2000); nu >= 2000 gives
        // ~4e-3, nu >= 1000 ~1%. For |Omega_K| >~ 0.1 no mode reaches nu = 4000 at k_max = 0.3/Mpc.
        // A very large value (e.g. 1e30) switches the approximation off.
        double flat_approximation_nu = 4000.0;
        // Curved projection: modes with q < q_max use a subsampled x-grid (every floor(q_max/q)-th point
        // outside the recombination window), with at most this spacing in x
        double dx_max_subsampled = 0.01;

        // Which part of the temperature source to integrate (requires
        // PerturbationOptions::store_source_components for anything but All)
        Perturbations::SourceTerm source_term = Perturbations::SourceTerm::All;
    };

    class PowerSpectrum {
      public:
        PowerSpectrum(const BackgroundCosmology & cosmo,
                      const RecombinationHistory & rec,
                      const Perturbations & pert,
                      PowerSpectrumOptions options = PowerSpectrumOptions());
        ~PowerSpectrum();

        // Add the (solved) tensor spectra to the CMB spectra; call before solve(). nullptr: scalars only.
        void set_tensors(const Tensors * t) { tensors = t; }
        // An additional independent component (same k-grid; see above); call before solve()
        void add_component(const Perturbations & p);
        int n_components() const { return int(comps.size()); }
        const Tensors * get_tensors() const { return tensors; }

        void solve();
        // Only what the matter outputs need (the Halofit table): for runs without C_ell. solve() calls it.
        void solve_matter();

        double primordial_power_spectrum(double k) const; // P_R(k), dimensionless, k in 1/m

        // C_ell (dimensionless) and D_ell = ell (ell+1) C_ell / 2 pi T_CMB^2 in muK^2
        double get_Cell_TT(double ell) const;
        double get_Dell_TT(double ell) const;
        double get_Dell_EE(double ell) const; // require polarization
        double get_Dell_TE(double ell) const;
        double get_Dell_BB(double ell) const; // unlensed BB (tensors; 0 without)
        // Scalar contributions alone (= the above without tensors)
        double get_Dell_TT_scalar(double ell) const;
        double get_Dell_EE_scalar(double ell) const;
        double get_Dell_TE_scalar(double ell) const;
        bool has_polarization() const { return pol; }

        // Lensing (options.lensing): lensing potential and lensed spectra, integer ell in [2, ell_max]
        bool has_lensing() const { return options.lensing; }
        double get_Cell_phiphi(double ell) const; // C_ell^phiphi (dimensionless)
        double get_Dell_TT_lensed(int ell) const; // muK^2
        double get_Dell_EE_lensed(int ell) const;
        double get_Dell_TE_lensed(int ell) const;
        double get_Dell_BB_lensed(int ell) const;

        // Transfer functions Theta_ell(k), Theta^E_ell(k) for one of the ells in the list
        double get_Theta_ell(int ell, double k) const;
        double get_ThetaE_ell(int ell, double k) const;
        const std::vector<int> & get_ells() const { return ells; }
        const std::vector<double> & get_k_los() const { return k_los; }
        const std::vector<double> & get_x_los() const { return x_los; }

        // Matter power spectrum [m^3] at time x and wavenumber k [1/m]: total matter (CDM + baryons
        // + massive neutrinos), and CDM + baryons only
        double get_matter_power_spectrum(double x, double k) const;
        double get_matter_power_spectrum_cb(double x, double k) const;
        double get_Delta_M(double x, double k, int comp = 0) const;
        double get_Delta_cb(double x, double k, int comp = 0) const;
        // Nonlinear total-matter power (Halofit; = linear, extrapolated, if options.nonlinear = None) [m^3]
        double get_matter_power_spectrum_nonlinear(double x, double k) const;

        // -------------------------------------------------------------------------
        // Two-point functions (do not need solve() except the CMB ones). Units: SI (k in 1/m, r in m).
        // -------------------------------------------------------------------------
        double get_Delta_M_extrapolated(double x, double k, int comp = 0) const; // Delta_M for any k > 0
        double get_matter_power_spectrum_extrapolated(double x, double k) const;
        // Matter correlation function xi(r) at x for the radii r [m]; optional Gaussian smoothing R [m].
        // Linear theory; for r < ~10 Mpc it depends on the extrapolation beyond k_max.
        std::vector<double> get_xi_matter(double x, const std::vector<double> & r, double R_smooth = 0.0) const;
        double get_sigma_R(double x, double R) const; // rms of Delta_M in top-hat spheres of radius R [m]
        double get_sigma8(double x = 0.0) const;      // R = 8 Mpc/h
        // Angular power spectrum of the projected matter/galaxy density for the given ells (C_ell, dimensionless)
        // Auto: exact for ell < ell_limber, Limber above (Limber error ~0.4% at ell = 100, ~0.1-0.2% at 200, for
        // sigma_z = 0.1)
        enum class Projection { Auto, Exact, Limber };
        std::vector<double> get_angular_cl_matter(const TwoPoint::RedshiftWindow & window,
                                                  const std::vector<int> & ell_list,
                                                  Projection method = Projection::Auto,
                                                  int ell_limber = 200) const;
        // Angular correlation function w(theta) of the projected density [theta in rad]. Uses C_ell up to
        // ell_max: exact/Limber up to 2000, Limber on a log-spaced grid above (interpolated in log C_ell), and a
        // cosine taper over the top 20% of ell to avoid ringing from the truncation. ell_max ~ 30000 is converged
        // for theta >~ 0.01 deg (linear theory).
        std::vector<double> get_angular_correlation_matter(const TwoPoint::RedshiftWindow & window,
                                                           const std::vector<double> & theta,
                                                           int ell_max = 30000) const;
        // CMB angular correlation functions [muK^2] at theta [rad] (lensed requires options.lensing),
        // optionally smoothed with a Gaussian beam (fwhm in radians)
        TwoPoint::CMBCorrelation
        get_cmb_correlation(const std::vector<double> & theta, bool lensed = false, double fwhm = 0.0) const;

        void info() const;
        void output_cells(const std::string & filename) const; // ell, D_ell^TT, D_ell^EE, D_ell^TE [muK^2]
        // ell, lensed D_ell^TT, EE, TE, BB [muK^2], [ell(ell+1)]^2 C_ell^phiphi / 2pi
        void output_cells_lensed(const std::string & filename) const;
        void output_matter_power(const std::string & filename,
                                 double x = 0.0) const; // k [h/Mpc], P_m, P_cb [(Mpc/h)^3]
        void output_transfer(const std::string & filename, const std::vector<int> & ells_out) const;
        // theta [deg], C^TT, xi_+, xi_-, xi_X [muK^2] for theta = 0..10 deg in steps of 0.01 deg and 10..180 in 0.1 deg
        void output_cmb_correlation(const std::string & filename, bool lensed = false, double fwhm = 0.0) const;
        // r [Mpc/h], xi(r), r^2 xi [(Mpc/h)^2] for r = 1..200 Mpc/h
        void output_matter_correlation(const std::string & filename, double x = 0.0, double R_smooth = 0.0) const;
        // ell, C_ell, ell (ell+1) C_ell / 2pi for ell = 2..ell_max; and theta [deg], w(theta) for 0.01..30 deg
        // (w from get_angular_correlation_matter)
        void output_angular_matter(const std::string & file_cl,
                                   const std::string & file_w,
                                   const TwoPoint::RedshiftWindow & window,
                                   int ell_max = 2000) const;

      private:
        const BackgroundCosmology & cosmo;
        const RecombinationHistory & rec;
        const Perturbations & pert;
        PowerSpectrumOptions options;
        const Tensors * tensors = nullptr;
        std::vector<const Perturbations *> comps; // comps[0] = &pert
        const Perturbations * cur = nullptr;      // the component of the current line-of-sight integration
        // per component: Theta, ThetaE, ThetaPhi
        std::vector<std::vector<std::vector<double>>> Theta_c, ThetaE_c, ThetaPhi_c;
        std::unique_ptr<Halofit> halofit; // built in solve() when options.nonlinear = Halofit
        // tensor D_ell [muK^2] at (non-integer) ell: 0 = TT, 1 = EE, 2 = BB, 3 = TE; linear between integers
        double tensor_dell(int which, double ell) const;

        std::vector<int> ells;
        std::vector<double> k_los, x_los;
        std::vector<UniformSpline1D> bessel;       // j_ell(z) per ell in the list
        std::vector<UniformSpline1D> bessel_z2;    // j_ell(z)/z^2 (E-mode integrand; finite at z = 0)
        std::vector<UniformSpline1D> bessel_d;     // j_ell'(z) (flat approximation in curved universes)
        std::vector<std::vector<double>> Theta;    // Theta[iell][ik] on k_los
        std::vector<std::vector<double>> ThetaE;   // Theta^E[iell][ik] on k_los
        std::vector<std::vector<double>> ThetaPhi; // lensing potential transfer function phi_ell(k)
        Spline1D log_Cell_pp_spline;               // log C_ell^phiphi
        std::vector<double> lensed_tt, lensed_ee, lensed_te, lensed_bb; // lensed C_ell, index = ell
        int ell_max_lensed = 0;
        std::vector<double> cells, cells_EE, cells_TE;                // C_ell for ell in the list
        Spline1D log_Dell_spline, log_Dell_EE_spline, Dell_TE_spline; // splines in ell (D_ell/T^2)
        bool pol = false;
        bool solved = false;
        bool matter_solved = false;

        double interp_k(const std::vector<double> & f, double k) const;
        size_t ell_index(int ell) const;

        void make_ell_list();
        void make_k_grid();
        void make_x_grid();
        void make_bessel_tables();
        void make_bessel_tables_flat_approximation(double z_max);
        void line_of_sight_integration();
        void line_of_sight_integration_radial();
        void compute_cells();
        void compute_lensing();
        // Lensing potential source for the LOS integral, -(c/Hp) r(chi_* - chi)/(r(chi_*) r(chi)) (Psi - Phi),
        // times the trapezoidal weight w; zero before recombination (x < x_*)
        double lensing_source(double x, double k, double w, double deta) const;
        double weyl_potential(double x, double k, int comp = 0) const; // Psi - Phi, extrapolated beyond k_max
        double cell_phiphi_limber(int ell) const;
        double eta_star = 0.0; // conformal time of the source plane (visibility peak)
        void require_solved(const char * f) const;
    };

} // namespace FML::COSMOLOGY

#endif
