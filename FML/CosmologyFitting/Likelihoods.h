#ifndef FML_COSMOLOGYFITTING_LIKELIHOODS_H
#define FML_COSMOLOGYFITTING_LIKELIHOODS_H

// =============================================================================
// Likelihoods of background data for the MCMC (all return ln L, up to a constant):
//
//   sn            Binned supernova luminosity distances of the course (external/supernovadata.txt: z, d_L, sigma
//                 in Gpc, from the JLA sample, Betoule et al. 2014). The distances assume a fixed absolute
//                 magnitude, so they constrain h; with free_amplitude the overall scale A (d_L -> A d_L) is
//                 marginalized analytically (flat prior on A), which leaves only the shape of d_L(z).
//   desi_dr2      DESI DR2 BAO: D_V/r_d, D_M/r_d, D_H/r_d with the full covariance (external/bao/)
//   desi_dr1      DESI DR1 BAO
//   planck_dp     Planck 2018 distance priors (TT,TE,EE+lowE, LCDM; Chen, Huang & Wang 2019): R, l_A and
//                 omega_b with their correlations (n_s marginalized). Valid for models with standard
//                 early-universe physics (not EDE).
//   bbn           omega_b = 0.02218 +- 0.00055 (BBN, Schoeneberg 2024, as used by DESI)
//
// Supernovae Ia with full covariances (magnitudes m_i = 5 log10(d_L/10 pc) + M, d_L = (1 + z_hel) D_M(z_cmb);
// the offset M is marginalized analytically with a flat prior: -2 ln L = a - b^2/e + ln(e/2pi), a = D^T C^-1 D,
// b = sum C^-1 D, e = sum C^-1, D = data - model at M = 0; as Cobaya/CosmoMC):
//   pantheonplus  Pantheon+ (Brout et al. 2022), 1590 light curves with z_HD > 0.01, STAT+SYS covariance
//   desy5         DES-SN5YR (DES Collaboration 2024), 1829 SNe, covariance covsys + diag(MUERR_FINAL^2)
//   union3        Union3 (Rubin et al. 2023, UNITY1.5), 22 binned distance moduli with covariance
// Data in external/sn/ (Union3 included; python/fetch_sn_data.py downloads Pantheon+ and DES-SN5YR).
//
// CMB likelihoods (need the full Boltzmann solution; use precision = fast):
//   planck_lite_TTTEEE   Planck 2018 high-ell plik_lite (Planck 2018 V, arXiv:1907.12875): binned, foreground-
//                        marginalized TT (30 <= ell <= 2508), TE and EE (30 <= ell <= 1996), 613 bins with
//                        their covariance. Nuisance parameter A_planck (sample it with gauss_A_planck = 1,
//                        0.0025; fixed to 1 if not sampled): C_ell^model -> C_ell / A_planck^2.
//   planck_lite_TT       TT only (215 bins)
//   planck_lite_TTTEEE_lowl, planck_lite_TT_lowl   ... plus the low-ell TT (2 <= ell <= 29, Commander)
//                        compressed into two Gaussian bins (Prince & Dunkley 2019, arXiv:1909.05869)
//                        The low-ell EE likelihood (which fixes tau) is not included: sample tau_reio with a
//                        Gaussian prior instead (gauss_tau_reio = 0.0506, 0.0086, the Planck 2018 lowE value).
//                        Data: external/planck2018_lite/ (from planck-lite-py).
//   planck_lensing       Planck 2018 lensing (Planck 2018 VIII, arXiv:1807.06210): 9 bandpowers of
//                        [L(L+1)]^2 C_L^phiphi / 2pi, 8 <= L <= 400 (conservative range, T+P SMICA reconstruction),
//                        Gaussian with their covariance, as CMBlikes/Cobaya (planck_2018_lensing.native): binned
//                        theory + linear correction for the dependence of the N0/N1 biases on the true CMB and
//                        lensing spectra, sum_L [W_TT D_TT + W_EE D_EE + W_TE D_TE + W_PP PP] - (its fiducial value),
//                        with the lensed CMB spectra divided by A_planck^2 (nuisance parameter, as plik_lite; share
//                        it when combining). Data: external/planck2018_lensing/.
//   planck_lensing_marged  The same, marginalized over the CMB spectra (Cobaya planck_2018_lensing.CMBMarged):
//                        PP-only correction and a larger covariance; for use without CMB power-spectrum data.
//   cmb_mock             Simulated full-sky-like data: the exact Gaussian likelihood of TT, EE, TE (ell =
//                        2..cmb_mock_lmax) for a fiducial lensed spectrum (cmb_mock_file, a cells_lensed.txt of
//                        ./cmb) with white noise (cmb_mock_noise [muK arcmin], polarization sqrt(2) higher),
//                        a Gaussian beam (cmb_mock_beam [arcmin] FWHM) and sky fraction cmb_mock_fsky:
//                        -2 ln L = sum (2l+1) f_sky [tr(D C^-1) + ln det C - ln det D - 2], D = data + noise,
//                        C = theory + noise (zero at the fiducial). For forecasts and tests. With cmb_mock_bb =
//                        true the (lensed) BB spectrum is added as an independent block,
//                        (2l+1) f_sky [D/C + ln(C/D) - 1], e.g. to forecast r (the file needs the BB column of
//                        cells_lensed.txt).
//
// Each likelihood says whether it needs the recombination history (for r_d or z_*), so that the theory
// only computes what is needed.
// =============================================================================

#include <FML/Cosmology/Common.h>
#include <FML/CosmologyFitting/Theory.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::FITTING {

class Likelihood {
  public:
    virtual ~Likelihood() = default;
    virtual std::string name() const = 0;
    virtual bool needs_recombination() const = 0;
    virtual double loglike(const Theory& t) const = 0;
    virtual int n_data() const = 0;
    virtual bool needs_cmb() const { return false; }
    virtual int cmb_ell_max() const { return 0; }
    virtual std::vector<std::string> nuisance_parameters() const { return {}; }
};

// Gaussian likelihood -chi^2/2 with a fixed covariance (inverted once)
class GaussianData {
  public:
    void set(const std::vector<double>& mean, const std::vector<std::vector<double>>& cov);
    double chi2(const std::vector<double>& model) const;
    size_t size() const { return mu.size(); }
    const std::vector<double>& mean() const { return mu; }

  private:
    std::vector<double> mu;
    std::vector<std::vector<double>> icov;
};

class SupernovaLikelihood : public Likelihood {
  public:
    explicit SupernovaLikelihood(const std::string& file = Utils::fml_file("CosmologyFitting/external/supernovadata.txt"), bool free_amplitude = false);
    std::string name() const override { return free_amp ? "sn (free amplitude)" : "sn"; }
    bool needs_recombination() const override { return false; }
    double loglike(const Theory& t) const override;
    int n_data() const override { return int(z.size()); }

  private:
    std::vector<double> z, dL, sigma;   // Gpc
    bool free_amp;
};

class SNIaLikelihood : public Likelihood {
  public:
    // magnitudes (any constant offset) at z_cmb (for D_M) and z_hel (the 1 + z of d_L), covariance n x n (row-major)
    SNIaLikelihood(std::string name, std::vector<double> z_cmb, std::vector<double> z_hel, std::vector<double> mag,
                   const std::vector<double>& cov);
    static std::unique_ptr<SNIaLikelihood> pantheonplus(const std::string& dir = Utils::fml_file("CosmologyFitting/external/sn/PantheonPlus"));
    static std::unique_ptr<SNIaLikelihood> desy5(const std::string& dir = Utils::fml_file("CosmologyFitting/external/sn/DESY5"));
    static std::unique_ptr<SNIaLikelihood> union3(const std::string& dir = Utils::fml_file("CosmologyFitting/external/sn/Union3"));
    std::string name() const override { return label; }
    bool needs_recombination() const override { return false; }
    double loglike(const Theory& t) const override;
    int n_data() const override { return int(zc.size()); }
    double loglike_mu(const std::vector<double>& mu_model) const;   // for model distance moduli at the data points
    // best-fitting offset M for given model distance moduli (b/e)
    double offset(const std::vector<double>& mu_model) const;
    const std::vector<double>& z_cmb() const { return zc; }

  private:
    std::string label;
    std::vector<double> zc, zh, m;
    std::vector<double> icov, rowsum;   // C^-1 (row-major) and its row sums
    double e = 0.0;                     // sum of C^-1
};

class BAOLikelihood : public Likelihood {
  public:
    enum class Quantity { DV_over_rd, DM_over_rd, DH_over_rd };
    BAOLikelihood(const std::string& label, const std::string& mean_file, const std::string& cov_file);
    std::string name() const override { return label; }
    bool needs_recombination() const override { return true; }
    double loglike(const Theory& t) const override;
    int n_data() const override { return int(z.size()); }
    std::vector<double> model(const Theory& t) const;

  private:
    std::string label;
    std::vector<double> z;
    std::vector<Quantity> q;
    GaussianData data;
};

class PlanckDistancePriors : public Likelihood {
  public:
    PlanckDistancePriors();
    std::string name() const override { return "planck_dp"; }
    bool needs_recombination() const override { return true; }
    double loglike(const Theory& t) const override;
    int n_data() const override { return 3; }

  private:
    GaussianData data;
};

class BBNPrior : public Likelihood {
  public:
    std::string name() const override { return "bbn"; }
    bool needs_recombination() const override { return false; }
    double loglike(const Theory& t) const override;
    int n_data() const override { return 1; }
};

class PlanckLiteLikelihood : public Likelihood {
  public:
    // data_dir: external/planck2018_lite; tt_only: TT, else TTTEEE; low_ell: add the two low-ell TT bins
    PlanckLiteLikelihood(const std::string& data_dir, bool tt_only, bool low_ell);
    std::string name() const override;
    bool needs_recombination() const override { return false; }
    bool needs_cmb() const override { return true; }
    int cmb_ell_max() const override { return 2508; }
    std::vector<std::string> nuisance_parameters() const override { return {"A_planck"}; }
    double loglike(const Theory& t) const override;
    int n_data() const override { return int(data.size()); }
    // ln L for D_ell [muK^2] indexed by ell (TT, TE, EE; at least up to 2508) and the calibration A_planck
    double loglike(const std::vector<double>& Dtt, const std::vector<double>& Dte, const std::vector<double>& Dee,
                   double A_planck = 1.0) const;

  private:
    bool tt_only, low_ell;
    struct Bin { int lmin, lmax; std::vector<double> w; };   // C_ell^bin = sum_l w_l C_l
    std::vector<Bin> bins_tt, bins_pol;                       // TT (incl. low ell), TE = EE binning
    GaussianData data;
};

class PlanckLensingLikelihood : public Likelihood {
  public:
    // data_dir: external/planck2018_lensing; cmb_marged: the CMB-marginalized variant
    PlanckLensingLikelihood(const std::string& data_dir, bool cmb_marged);
    std::string name() const override { return marged ? "planck_lensing_marged" : "planck_lensing"; }
    bool needs_recombination() const override { return false; }
    bool needs_cmb() const override { return true; }
    int cmb_ell_max() const override { return lmax; }
    std::vector<std::string> nuisance_parameters() const override {
      return marged ? std::vector<std::string>{} : std::vector<std::string>{"A_planck"};
    }
    double loglike(const Theory& t) const override;
    int n_data() const override { return int(data.size()); }
    // ln L for D_ell [muK^2] (TT, TE, EE) and PP = [L(L+1)]^2 C_L^phiphi / 2pi, indexed by ell up to 2500
    double loglike(const std::vector<double>& Dtt, const std::vector<double>& Dte, const std::vector<double>& Dee,
                   const std::vector<double>& pp, double A_planck = 1.0) const;
    std::vector<double> binned_theory(const std::vector<double>& Dtt, const std::vector<double>& Dte,
                                      const std::vector<double>& Dee, const std::vector<double>& pp,
                                      double A_planck = 1.0) const;   // incl. the linear correction

  private:
    bool marged;
    static constexpr int lmin = 2, lmax = 2500;
    std::vector<std::vector<double>> w_pp;              // [bin][L - lmin] binning of PP
    std::vector<std::vector<double>> c_tt, c_ee, c_te, c_pp;   // [bin][L - lmin] linear correction
    std::vector<double> fid_corr;
    GaussianData data;
};

class CMBMockLikelihood : public Likelihood {
  public:
    CMBMockLikelihood(const std::string& fiducial_file, int lmax, double fsky, double noise_muK_arcmin,
                      double beam_fwhm_arcmin, bool use_bb = false);
    std::string name() const override { return "cmb_mock"; }
    bool needs_recombination() const override { return false; }
    bool needs_cmb() const override { return true; }
    int cmb_ell_max() const override { return lmax; }
    double loglike(const Theory& t) const override;
    int n_data() const override { return (bb ? 4 : 3) * (lmax - 1); }
    double loglike(const std::vector<double>& Dtt, const std::vector<double>& Dte, const std::vector<double>& Dee,
                   const std::vector<double>& Dbb = {}) const;

  private:
    int lmax;
    double fsky;
    bool bb;
    std::vector<double> dTT, dEE, dTE, dBB, nT, nP;   // C_ell of the data (fiducial + noise) and noise, index ell
};

namespace Likelihoods {
  // By name (throws for an unknown name): sn, sn_free_amplitude, desi_dr2, desi_dr1, planck_dp, bbn,
  // planck_lite_TTTEEE, planck_lite_TT, planck_lite_TTTEEE_lowl, planck_lite_TT_lowl, planck_lensing,
  // planck_lensing_marged, cmb_mock.
  // Options (cmb_mock): cmb_mock_file (default data/cmb_mock/cells_lensed.txt), cmb_mock_lmax (2500),
  // cmb_mock_fsky (0.6), cmb_mock_noise (33 muK arcmin), cmb_mock_beam (7 arcmin), cmb_mock_bb (false)
  std::unique_ptr<Likelihood> create(const std::string& name, const std::map<std::string, std::string>& options = {});
  std::vector<std::string> names();
  std::vector<std::string> option_names();
}

} // namespace FML::COSMOLOGY::FITTING

#endif
