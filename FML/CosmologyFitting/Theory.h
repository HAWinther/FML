#ifndef FML_COSMOLOGYFITTING_THEORY_H
#define FML_COSMOLOGYFITTING_THEORY_H

// =============================================================================
// Theory predictions for one point in parameter space, computed lazily: the background always, the
// recombination history only when a quantity needs it (sound horizons at z_* and at the drag epoch).
//
// Distances (all in Mpc):  D_M(z) = transverse comoving distance,  D_H(z) = c/H(z),
//                          D_V(z) = [z D_M^2 D_H]^{1/3}
// Drag epoch: tau_b(z_d) = 1 with the baryon optical depth tau_b(x) = int_x^0 (-tau' R) dx' (R = 4 rho_gamma/
// (3 rho_b)), as CAMB and CLASS; r_d = r_s(z_d). Photon decoupling: tau(z_*) = 1, r_* = r_s(z_*).
// Planck distance priors (Chen, Huang & Wang 2019): R = sqrt(Omega_m H0^2) D_M(z_*)/c,  l_A = pi D_M(z_*)/r_*,
// evaluated as in that paper (their eqs. 1-6), since the published values depend on it: radiation
// Omega_r = Omega_m/(1 + z_eq), z_eq = 2.5e4 omega_m (T/2.7 K)^-4, massive neutrinos counted as matter,
// r_s = int c da/(a^2 H sqrt(3 (1 + 31500 omega_b (T/2.7 K)^-4 a))); everything else in H(z) (dark energy,
// modified expansion) is that of the model. With our exact radiation and neutrinos l_A would be 0.3 (3 sigma)
// higher. z_* is our tau = 1 redshift (theirs is from the Planck chains, i.e. CAMB's).
//
// The recombination history is computed without reionization (z_* and z_d refer to recombination, as in CAMB
// and CLASS). Fast approximations (Options::sound_horizon = Fit), which need only the background: z_* from Hu &
// Sugiyama (1996), r_* from the integral of c_s with that z_*, and r_d from Aubourg et al. (2015, eq. 16; for
// standard early-universe physics only, 0.02% for LCDM with N_eff = 3.046). The Hu-Sugiyama z_* is ~2 too high,
// which shifts l_A by ~1 (10 sigma of the Planck distance prior): use the fit only for quick tests.
//
// CMB spectra (cmb()): the full pipeline of ./cmb (recombination with reionization, perturbations, lensed
// line-of-sight spectra) with the precision of the RunSettings (use precision = fast for fits), up to
// Options::cmb_ell_max. Lensing is always on. With r > 0 the tensor modes are added (Tensors.h; lensed
// together with the scalars, and BB includes the primordial B modes).
//
// theta_MC() = 100 r_s(z_*)/D_M(z_*) with the Hu-Sugiyama z_* and r_s from the background with CosmoMC's baryon
// loading R = 3e4 a omega_b (CAMB's CosmomcTheta, a cheap proxy of the acoustic angle used as a sampling
// coordinate; equal to CAMB's cosmomc_theta to ~2e-6 for the same H(z)), theta_star() the same with the tau = 1
// redshift and the recombination sound horizon (100 theta_* of CAMB/CLASS).
//
// Nuisance parameters of the likelihoods (e.g. the Planck calibration A_planck) are passed with the theory.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace FML::COSMOLOGY {
    class EinsteinBoltzmannSolver;
} // namespace FML::COSMOLOGY

namespace FML::COSMOLOGY::FITTING {


    struct TheoryOptions {
        enum class SoundHorizon { Recombination, Fit };
        SoundHorizon sound_horizon = SoundHorizon::Recombination;
        int rec_points = 4000; // recombination grid (x from rec_x_start to 0); a normal run uses 40000
        double rec_x_start = -11.0;
        double rec_relerr = 1e-8; // RECFAST ODE tolerance (1e-10 in a normal run; r_d changes by < 1e-8, 2x faster)
        int cmb_ell_max = 2508;   // highest multipole of the lensed CMB spectra (cmb())
        // Does any likelihood need the CMB spectra? If not, the matter power spectrum for sigma8(z), fsigma8(z) is
        // computed without the C_ell (compute_cells = false)
        bool cmb = true;
        // The redshifts of sigma8(z) and fsigma8(z) (computed together, with the perturbations, the first time one
        // of them, or the CMB, is needed)
        std::vector<double> growth_z;
        // The redshifts of the linear P_cb(k) tables (pk_cb) on pk_k_grid() (computed with the growth)
        std::vector<double> pk_z;
        // The redshifts of the nonlinear total-matter P(k) tables (pk_nl, on pk_nl_k_grid(); the nonlinear method of
        // the RunSettings, e.g. nonlinear = hmcode2016) and the linear growth factor D(z) (computed with the growth)
        std::vector<double> pk_nl_z;
    };

    // Lensed CMB spectra D_ell = ell (ell+1) C_ell / 2 pi [muK^2], index = ell (0 and 1 unused), and
    // [ell (ell+1)]^2 C_ell^phiphi / 2 pi
    struct CMBSpectra {
        int ell_max = 0;
        std::vector<double> tt, ee, te, bb, pp;
    };

    class Theory {
      public:
        using Options = TheoryOptions;

        // Builds the model and solves the background (throws std::exception for unphysical parameters)
        explicit Theory(const RunSettings & settings, Options options = Options());

        const BackgroundCosmology & background() const { return *bg; }
        const RecombinationHistory & recombination() const; // solved on first use
        const RunSettings & settings() const { return s; }

        double D_M(double z) const;
        double D_H(double z) const;
        double D_V(double z) const;
        double d_L(double z) const { return (1.0 + z) * D_M(z); }

        double z_star() const;
        double r_star() const; // sound horizon at z_*
        double z_drag() const;
        double r_drag() const;     // sound horizon at the drag epoch
        double R_shift() const;    // sqrt(Omega_m H0^2) D_M(z_*)/c   (Chen et al. convention, see above)
        double l_A() const;        // pi D_M(z_*)/r_s(z_*)          (Chen et al. convention)
        double omega_b() const;    // Omega_b h^2
        double omega_m() const;    // (Omega_b + Omega_cdm + Omega_ncdm) h^2
        double theta_MC() const;   // 100 r_s/D_M at the Hu-Sugiyama z_* (background only)
        double theta_star() const; // 100 r_*/D_M(z_*)

        // CMB (computed on first use)
        const CMBSpectra & cmb() const;
        const RecombinationHistory & recombination_reionized() const; // as in ./cmb (with reionization)
        double tau_reio() const; // optical depth to z_reion + 8 delta_z_reion (CLASS tau_reio)
        double sigma8() const;   // linear sigma_8 today (accuracy limited by pert_k_max, see Theory.cpp)
        // At the redshifts of Options::growth_z (throws for others): sigma_8 of the CDM + baryons and f sigma_8 =
        // sigma_8,vd^2/sigma_8 (CAMB's definition; PowerSpectrum::get_fsigma8), linear
        double sigma8_cb(double z) const;
        double fsigma8(double z) const;
        // sigma_8 of the velocity divergence (in units of -aH) of the CDM + baryons at the redshifts of growth_z:
        // f = sigma8_theta / sigma8_cb is the growth rate of the ShapeFit compression
        double sigma8_theta(double z) const;
        // The linear CDM + baryon power spectrum [(Mpc/h)^3] on pk_k_grid() [h/Mpc] at the redshifts of pk_z (h of the
        // model). The grid is that of cosmoprimo's BAO filters: 1024 points log-spaced from 1e-7 to 100 h/Mpc
        static const std::vector<double> & pk_k_grid();
        const std::vector<double> & pk_cb(double z) const;
        // The nonlinear total-matter power spectrum [(Mpc/h)^3] on pk_nl_k_grid() [h/Mpc] at the redshifts of pk_nl_z
        // (= linear without a nonlinear method). The grid: 400 points log-spaced from 1e-5 to 100 h/Mpc
        static const std::vector<double> & pk_nl_k_grid();
        const std::vector<double> & pk_nl(double z) const;
        // At the redshifts of pk_nl_z: the linear growth of the total matter on large scales, D(z)/D(0) =
        // sqrt(P_L(k, z)/P_L(k, 0)) at k = 1e-4 h/Mpc (as CosmoSIS' intrinsic-alignment module), and the comoving
        // distance chi(z) [Mpc/h] (radial; for the flat models of cosmic-shear likelihoods = D_M)
        double growth_D(double z) const;
        // Sigma(z) = G_light/G_* of the lensing (Weyl) potential in the quasi-static limit, k^2 (Phi + Psi) = -8 pi G_*
        // a^2 Sigma rho Delta (QuasiStaticGravity.h; G_* that of the density parameters): g/phi(a) for jbd,
        // 1 + Sigma0 Omega_DE(a)/Omega_DE,0 for musigma, mu for cubic_galileon, 1/(1 + f_R) for fr; 1 for the models
        // without modified lensing (GR, dark energy, DGP, conformally coupled scalars). Throws for horndeski and
        // horndeski_table (not implemented)
        double lensing_Sigma(double z) const;
        double chi_mpch(double z) const { return bg->comoving_distance_of_x(-std::log1p(z)) / Constants::Mpc * s.cosmo.h; }

        // Nuisance parameters
        void set_nuisance(const std::string & name, double value) { nuis[name] = value; }
        double nuisance(const std::string & name, double default_value) const {
            const auto it = nuis.find(name);
            return it == nuis.end() ? default_value : it->second;
        }

        ~Theory();

      private:
        RunSettings s;
        Options opt;
        // The solver (background on construction, recombination with reionization and the CMB on first use; the
        // perturbations are freed once the spectra are extracted) and its background
        mutable std::unique_ptr<EinsteinBoltzmannSolver> ebs;
        const BackgroundCosmology * bg = nullptr;
        mutable std::unique_ptr<RecombinationHistory> rec;
        mutable double zs = -1.0, rs = -1.0, zd = -1.0, rd = -1.0;
        void compute_star() const;
        void compute_drag() const;
        // int_z^inf c_s dz/H [Mpc], from the background only (cosmomc: CAMB's R = 3e4 a omega_b)
        double sound_horizon_bg(double z, bool cosmomc = false) const;
        double E_chen(double z) const; // H/H0 with the radiation and neutrinos of Chen et al.
        double D_M_chen(double z) const;
        double r_s_chen(double z) const;
        mutable double R_c = -1.0, lA_c = -1.0;
        mutable std::unique_ptr<CMBSpectra> spectra;
        mutable double sigma8_ = 0.0;
        mutable bool growth_done = false;
        mutable std::vector<double> sigma8cb_z, fsigma8_z, sigma8th_z; // at opt.growth_z
        mutable std::vector<std::vector<double>> pk_tables;           // at opt.pk_z
        mutable std::vector<std::vector<double>> pk_nl_tables;        // at opt.pk_nl_z
        mutable std::vector<double> growth_D_z;                       // at opt.pk_nl_z
        size_t pk_nl_index(double z) const;
        void compute_growth() const;                        // from the solved power spectrum
        void solve_matter() const;                          // the perturbations without the CMB (opt.cmb = false)
        size_t growth_index(double z) const;
        void ensure_growth() const;
        std::map<std::string, double> nuis;
    };

} // namespace FML::COSMOLOGY::FITTING

#endif
