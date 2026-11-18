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
//   boss_dr12     BOSS DR12 consensus BAO + full shape (Alam et al. 2017, arXiv:1607.03155; "final consensus"):
//                 D_M (r_d,fid/r_d) [Mpc], H (r_d/r_d,fid) [km/s/Mpc] and f sigma_8 at z = 0.38, 0.51, 0.61 with the
//                 9x9 covariance (r_d,fid = 147.78 Mpc). f sigma_8 from the linear perturbations (Theory::fsigma8)
//   boss_dr12_bao BOSS DR12 consensus BAO only (D_M and H at the same redshifts)
//   eboss_dr16_lrg  SDSS DR16 BAO + RSD of the luminous red galaxies (eBOSS 2020, arXiv:2007.08991): BOSS DR12 at
//                 z = 0.38, 0.51 and eBOSS LRG at z = 0.698, D_M/r_d, D_H/r_d, f sigma_8 (9x9)
//   eboss_dr16_qso  SDSS DR16 quasars at z = 1.48: D_M/r_d, D_H/r_d, f sigma_8
//                 (do not combine boss_dr12 with eboss_dr16_lrg: the low redshift bins are the same data)
//   desi_dr1_shapefit  DESI DR1 full shape, ShapeFit compression + post-reconstruction BAO (DESI 2024 V,
//                 arXiv:2411.12021), as DESI's Cobaya likelihood desi_shapefit_bao_all: for BGS, LRG1-3, ELG2 and QSO the
//                 BAO scalings q_iso, q_ap, the amplitude df = f sqrt(A_p) / fiducial and the slope dm = m - m_fid, and
//                 q_par, q_per for Lyman-alpha (26 values, block-diagonal covariance). Relative to the DESI fiducial
//                 cosmology (Planck 2018 LCDM, computed here with the same precision as the model). f = sigma_8,
//                 thetatheta / sigma_8,cb; A_p = P_cb(k_p)/s^3 and m = dln P_nowiggle/dln k at k_p = 0.03 h/Mpc / s,
//                 s = h r_d / (h r_d)_fid; the no-wiggle P(k) from cosmoprimo's "peakaverage" filter (ported:
//                 envelopes of the extrema of P/P_EH,nowiggle at the fiducial positions rescaled by s)
//                 (do not combine with desi_dr1: the BAO are the same data)
//   desi_dr1_fs   DESI DR1 full shape (P_0, P_2 of BGS, LRG x 3, ELG, QSO, rotated, with the window and systematics),
//                 one-loop REPT as DESI's own likelihood, bias parameters desi_<SAMPLE>_b1p, _b2p, _bsp, counterterms and
//                 stochastic terms marginalized analytically (option desi_fs_solve = marg | best): see DESIFullShape.h.
//                 desi_dr1_fs_<SAMPLE> for one sample. Data: python/convert_desi_dr1_fullshape.py
//   kids1000_cosebis  KiDS-1000 cosmic shear, COSEBIs (Asgari et al. 2021, arXiv:2007.15633): E_n, n = 1..5, of the
//                 15 pairs of 5 tomographic bins, theta = 0.5' - 300', with the release's covariance (which includes the
//                 shear-calibration uncertainty), SOM n(z) and the correlated Gaussian prior on the photo-z shifts
//                 (included in ln L). Theory: Limber shear + NLA intrinsic alignments (FML/WeakLensing/CosmicShear.h) on
//                 the nonlinear total-matter P(k, z) of the run (nonlinear = hmcode2016 with hmcode2016_eta0_from_A =
//                 true and hmcode2016_A sampled, as KiDS; or jbd_boost etc.; modified lensing through
//                 Theory::lensing_Sigma, e.g. G/phi for Brans-Dicke). Nuisance parameters kids_A_IA (0 if not
//                 sampled) and kids_dz1..kids_dz5 (the prior mean if not sampled), n_i(z) -> n_i(z - kids_dz_i).
//                 Data: external/kids1000/ (python/convert_kids1000.py from the KiDS data release).
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
//   desdovekie    DES-Dovekie (DES-SN5YR recalibrated, Popovic et al. 2025), 1820 SNe, inverse total covariance
//   union3        Union3 (Rubin et al. 2023, UNITY1.5), 22 binned distance moduli with covariance
// Data in external/sn/ (Union3 included; python/fetch_sn_data.py downloads Pantheon+, DES-SN5YR and DES-Dovekie).
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
//   planck_lowl_TT, planck_lowl_EE, planck_camspec_npipe_TTTEEE (_TT, _TE, _EE, _TTTE)   Planck 2018 low-ell TT
//                        (Commander) and EE (SimAll) and the high-ell CamSpec PR4 (NPIPE) likelihood, ported from
//                        Cobaya: see PlanckLikelihoods.h. Data: python/fetch_planck_data.py.
//   act_dr6_lensing_actplanck, act_dr6_lensing (+ _extended)   ACT DR6 (+ Planck PR4) lensing with the CMB-dependent
//                        corrections, as act_dr6_lenslike: see PlanckLikelihoods.h. Data:
//                        python/convert_act_dr6_lensing.py. Needs the CMB spectra to ell = 2999.
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
#include <mutex>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::FITTING {

    class Likelihood {
      public:
        virtual ~Likelihood() = default;
        virtual std::string name() const = 0;
        virtual bool needs_recombination() const = 0;
        virtual double loglike(const Theory & t) const = 0;
        virtual int n_data() const = 0;
        virtual bool needs_cmb() const { return false; }
        virtual int cmb_ell_max() const { return 0; }
        virtual std::vector<std::string> nuisance_parameters() const { return {}; }
        // The redshifts at which the likelihood needs Theory::fsigma8 or sigma8_cb (TheoryOptions::growth_z)
        virtual std::vector<double> growth_redshifts() const { return {}; }
        // The redshifts at which the likelihood needs Theory::pk_cb (TheoryOptions::pk_z)
        virtual std::vector<double> pk_redshifts() const { return {}; }
        // The redshifts at which the likelihood needs Theory::pk_nl and growth_D (TheoryOptions::pk_nl_z)
        virtual std::vector<double> pk_nl_redshifts() const { return {}; }
        // Does the likelihood need Theory::pk_theta (at pk_redshifts)?
        virtual bool needs_pk_theta() const { return false; }
        // > 0: the perturbations must be solved at least up to this k [h/Mpc] (TheoryOptions::pk_k_max)
        virtual double pk_k_max() const { return 0.0; }
    };

    // Gaussian likelihood -chi^2/2 with a fixed covariance (inverted once)
    class GaussianData {
      public:
        void set(const std::vector<double> & mean, const std::vector<std::vector<double>> & cov);
        double chi2(const std::vector<double> & model) const;
        size_t size() const { return mu.size(); }
        const std::vector<double> & mean() const { return mu; }

      private:
        std::vector<double> mu;
        std::vector<std::vector<double>> icov;
    };

    class SupernovaLikelihood : public Likelihood {
      public:
        explicit SupernovaLikelihood(
            const std::string & file = Utils::fml_file("CosmologyFitting/external/supernovadata.txt"),
            bool free_amplitude = false);
        // (GetDist parameter names: no spaces)
        std::string name() const override { return free_amp ? "sn_free_amplitude" : "sn"; }
        bool needs_recombination() const override { return false; }
        double loglike(const Theory & t) const override;
        int n_data() const override { return int(z.size()); }

      private:
        std::vector<double> z, dL, sigma; // Gpc
        bool free_amp;
    };

    class SNIaLikelihood : public Likelihood {
      public:
        // magnitudes (any constant offset) at z_cmb (for D_M) and z_hel (the 1 + z of d_L), covariance n x n
        // (row-major)
        SNIaLikelihood(std::string name,
                       std::vector<double> z_cmb,
                       std::vector<double> z_hel,
                       std::vector<double> mag,
                       const std::vector<double> & cov);
        // the same with the inverse covariance C^-1 given (n x n, row-major)
        struct InverseCovariance {};
        SNIaLikelihood(std::string name,
                       std::vector<double> z_cmb,
                       std::vector<double> z_hel,
                       std::vector<double> mag,
                       std::vector<double> inverse_cov,
                       InverseCovariance);
        static std::unique_ptr<SNIaLikelihood>
        pantheonplus(const std::string & dir = Utils::fml_file("CosmologyFitting/external/sn/PantheonPlus"));
        static std::unique_ptr<SNIaLikelihood>
        desy5(const std::string & dir = Utils::fml_file("CosmologyFitting/external/sn/DESY5"));
        static std::unique_ptr<SNIaLikelihood>
        desdovekie(const std::string & dir = Utils::fml_file("CosmologyFitting/external/sn/DES-Dovekie"));
        static std::unique_ptr<SNIaLikelihood>
        union3(const std::string & dir = Utils::fml_file("CosmologyFitting/external/sn/Union3"));
        std::string name() const override { return label; }
        bool needs_recombination() const override { return false; }
        double loglike(const Theory & t) const override;
        int n_data() const override { return int(zc.size()); }
        double loglike_mu(const std::vector<double> & mu_model) const; // for model distance moduli at the data points
        // best-fitting offset M for given model distance moduli (b/e)
        double offset(const std::vector<double> & mu_model) const;
        const std::vector<double> & z_cmb() const { return zc; }

      private:
        std::string label;
        std::vector<double> zc, zh, m;
        std::vector<double> icov, rowsum; // C^-1 (row-major) and its row sums
        double e = 0.0;                   // sum of C^-1
        void set_row_sums();
    };

    class BAOLikelihood : public Likelihood {
      public:
        enum class Quantity { DV_over_rd, DM_over_rd, DH_over_rd, H_rd, fsigma8 };
        // Mean file: lines "z value quantity" with quantity DV_over_rs, DM_over_rs, DH_over_rs (distances over r_d,
        // times rs_fid if rs_fid > 0: then in Mpc), bao_Hz_rs (H(z) r_d / rs_fid in km/s/Mpc; needs rs_fid > 0) or
        // f_sigma8 (CAMB's f sigma_8). Covariance in the same order (Cobaya/CosmoMC format)
        BAOLikelihood(const std::string & label,
                      const std::string & mean_file,
                      const std::string & cov_file,
                      double rs_fid = 0.0);
        std::string name() const override { return label; }
        bool needs_recombination() const override { return true; }
        double loglike(const Theory & t) const override;
        int n_data() const override { return int(z.size()); }
        std::vector<double> growth_redshifts() const override;
        std::vector<double> model(const Theory & t) const;

      private:
        std::string label;
        double rs_fid = 0.0;
        std::vector<double> z;
        std::vector<Quantity> q;
        GaussianData data;
    };

    // cosmoprimo's PeakAveragePowerSpectrumBAOFilter (as in DESI's DR1 ShapeFit and full-shape likelihoods): the
    // no-wiggle P(k) is P_EH times the mean of the two envelopes of P/P_EH through the BAO maxima and minima, found
    // once in the fiducial P(k) at z = 0 and moved by 1/s, s = h r_d / (h r_d)_fid (see Likelihoods.cpp)
    class PeakAverageBAOFilter {
      public:
        PeakAverageBAOFilter() = default;
        // fiducial: a Theory with 0 in pk_z
        explicit PeakAverageBAOFilter(const Theory & fiducial);
        // The no-wiggle spectrum on Theory::pk_k_grid() of P (on that grid) of a model with these settings (for its
        // Eisenstein-Hu shape) and rescaling s
        std::vector<double> nowiggle(const std::vector<double> & P, const RunSettings & settings, double s) const;

      private:
        // indices into Theory::pk_k_grid() of the low-k padding, the extrema (maxima, minima) and the high-k padding
        std::vector<int> nodes[2];
        int npad_low[2] = {0, 0}, npeaks[2] = {0, 0}, npad_high[2] = {0, 0};
    };

    class DESIShapeFitLikelihood : public Likelihood {
      public:
        enum class Quantity { qiso, qap, qpar, qper, df, dm };
        DESIShapeFitLikelihood(const std::string & mean_file, const std::string & cov_file);
        std::string name() const override { return "desi_dr1_shapefit"; }
        bool needs_recombination() const override { return true; }
        double loglike(const Theory & t) const override;
        int n_data() const override { return int(z.size()); }
        std::vector<double> growth_redshifts() const override { return zs; }
        std::vector<double> pk_redshifts() const override { return zs; }
        std::vector<double> model(const Theory & t) const;

        // The ShapeFit slope m and amplitude f sqrt(A_p) of a theory at z (in pk_z and growth_z) for the BAO-filter
        // rescaling s = h r_d / (h r_d)_fid (the fiducial is set up from t's settings on first use)
        std::pair<double, double> shapefit(const Theory & t, double z, double s) const;
        // The DESI fiducial cosmology (cosmoprimo's fiducial.DESI) with the precision of `settings`
        static RunSettings fiducial_settings(const RunSettings & settings);

      private:
        std::vector<double> z, zs; // the data redshifts; those of the ShapeFit samples
        std::vector<Quantity> q;
        GaussianData data;
        // The fiducial: D_M/r_d, D_H/r_d at the data redshifts, h r_d, m and f sqrt(A_p) at zs, and the BAO filter
        struct Fiducial {
            std::vector<double> dm_rd, dh_rd, m, fA;
            double hrd = 0.0;
            PeakAverageBAOFilter filter;
        };
        mutable Fiducial fid;
        mutable std::once_flag fid_once;
        void setup_fiducial(const Theory & t) const;
        std::pair<double, double> shapefit_impl(const Theory & t, double z, double s) const;
    };

    class KiDS1000COSEBIsLikelihood : public Likelihood {
      public:
        explicit KiDS1000COSEBIsLikelihood(const std::string & dir);
        ~KiDS1000COSEBIsLikelihood() override;
        std::string name() const override { return "kids1000_cosebis"; }
        bool needs_recombination() const override { return false; }
        double loglike(const Theory & t) const override; // -chi^2/2 of the data and of the delta_z prior
        int n_data() const override { return int(data.size()); }
        std::vector<std::string> nuisance_parameters() const override {
            return {"kids_A_IA", "kids_dz1", "kids_dz2", "kids_dz3", "kids_dz4", "kids_dz5"};
        }
        std::vector<double> pk_nl_redshifts() const override { return zgrid; }
        // The model E_n (75, the order of the data) and the chi^2 of the data alone
        std::vector<double> model(const Theory & t) const;
        double chi2_data(const Theory & t) const { return data.chi2(model(t)); }
        double chi2_dz_prior(const Theory & t) const;

      private:
        GaussianData data;
        std::vector<int> pair_index, n_index; // of each data point
        std::vector<double> zmid;            // n(z) nodes
        std::vector<std::vector<double>> nz; // [bin][node]
        std::vector<double> dz_mean;
        std::vector<std::vector<double>> dz_icov;
        std::vector<double> zgrid; // redshifts of the P(k, z) table
        struct Impl;
        std::unique_ptr<Impl> impl; // the COSEBI filters
    };

    class PlanckDistancePriors : public Likelihood {
      public:
        PlanckDistancePriors();
        std::string name() const override { return "planck_dp"; }
        bool needs_recombination() const override { return true; }
        double loglike(const Theory & t) const override;
        int n_data() const override { return 3; }

      private:
        GaussianData data;
    };

    class BBNPrior : public Likelihood {
      public:
        std::string name() const override { return "bbn"; }
        bool needs_recombination() const override { return false; }
        double loglike(const Theory & t) const override;
        int n_data() const override { return 1; }
    };

    class PlanckLiteLikelihood : public Likelihood {
      public:
        // data_dir: external/planck2018_lite; tt_only: TT, else TTTEEE; low_ell: add the two low-ell TT bins
        PlanckLiteLikelihood(const std::string & data_dir, bool tt_only, bool low_ell);
        std::string name() const override;
        bool needs_recombination() const override { return false; }
        bool needs_cmb() const override { return true; }
        int cmb_ell_max() const override { return 2508; }
        std::vector<std::string> nuisance_parameters() const override { return {"A_planck"}; }
        double loglike(const Theory & t) const override;
        int n_data() const override { return int(data.size()); }
        // ln L for D_ell [muK^2] indexed by ell (TT, TE, EE; at least up to 2508) and the calibration A_planck
        double loglike(const std::vector<double> & Dtt,
                       const std::vector<double> & Dte,
                       const std::vector<double> & Dee,
                       double A_planck = 1.0) const;

      private:
        bool tt_only, low_ell;
        struct Bin {
            int lmin, lmax;
            std::vector<double> w;
        }; // C_ell^bin = sum_l w_l C_l
        std::vector<Bin> bins_tt, bins_pol; // TT (incl. low ell), TE = EE binning
        GaussianData data;
    };

    class PlanckLensingLikelihood : public Likelihood {
      public:
        // data_dir: external/planck2018_lensing; cmb_marged: the CMB-marginalized variant
        PlanckLensingLikelihood(const std::string & data_dir, bool cmb_marged);
        std::string name() const override { return marged ? "planck_lensing_marged" : "planck_lensing"; }
        bool needs_recombination() const override { return false; }
        bool needs_cmb() const override { return true; }
        int cmb_ell_max() const override { return lmax; }
        std::vector<std::string> nuisance_parameters() const override {
            return marged ? std::vector<std::string>{} : std::vector<std::string>{"A_planck"};
        }
        double loglike(const Theory & t) const override;
        int n_data() const override { return int(data.size()); }
        // ln L for D_ell [muK^2] (TT, TE, EE) and PP = [L(L+1)]^2 C_L^phiphi / 2pi, indexed by ell up to 2500
        double loglike(const std::vector<double> & Dtt,
                       const std::vector<double> & Dte,
                       const std::vector<double> & Dee,
                       const std::vector<double> & pp,
                       double A_planck = 1.0) const;
        std::vector<double> binned_theory(const std::vector<double> & Dtt,
                                          const std::vector<double> & Dte,
                                          const std::vector<double> & Dee,
                                          const std::vector<double> & pp,
                                          double A_planck = 1.0) const; // incl. the linear correction

      private:
        bool marged;
        static constexpr int lmin = 2, lmax = 2500;
        std::vector<std::vector<double>> w_pp;                   // [bin][L - lmin] binning of PP
        std::vector<std::vector<double>> c_tt, c_ee, c_te, c_pp; // [bin][L - lmin] linear correction
        std::vector<double> fid_corr;
        GaussianData data;
    };

    class CMBMockLikelihood : public Likelihood {
      public:
        CMBMockLikelihood(const std::string & fiducial_file,
                          int lmax,
                          double fsky,
                          double noise_muK_arcmin,
                          double beam_fwhm_arcmin,
                          bool use_bb = false);
        std::string name() const override { return "cmb_mock"; }
        bool needs_recombination() const override { return false; }
        bool needs_cmb() const override { return true; }
        int cmb_ell_max() const override { return lmax; }
        double loglike(const Theory & t) const override;
        int n_data() const override { return (bb ? 4 : 3) * (lmax - 1); }
        double loglike(const std::vector<double> & Dtt,
                       const std::vector<double> & Dte,
                       const std::vector<double> & Dee,
                       const std::vector<double> & Dbb = {}) const;

      private:
        int lmax;
        double fsky;
        bool bb;
        std::vector<double> dTT, dEE, dTE, dBB, nT, nP; // C_ell of the data (fiducial + noise) and noise, index ell
    };

    namespace Likelihoods {
        // By name (throws for an unknown name): sn, sn_free_amplitude, desi_dr2, desi_dr1, boss_dr12, boss_dr12_bao,
        // eboss_dr16_lrg, eboss_dr16_qso, planck_dp, bbn,
        // planck_lite_TTTEEE, planck_lite_TT, planck_lite_TTTEEE_lowl, planck_lite_TT_lowl, planck_lensing,
        // planck_lensing_marged, planck_lowl_TT, planck_lowl_EE, planck_camspec_npipe_TTTEEE (and _TT, _TE, _EE, _TTTE),
        // act_dr6_lensing, act_dr6_lensing_actplanck (and _extended), desi_dr1_fs (and desi_dr1_fs_<SAMPLE>),
        // cmb_mock (and pantheonplus, desy5, desdovekie, union3, desi_dr1_shapefit, kids1000_cosebis: see above).
        // Options (desi_dr1_fs): desi_fs_solve = marg (default) or best. Options (cmb_mock): cmb_mock_file (default data/cmb_mock/cells_lensed.txt), cmb_mock_lmax (2500),
        // cmb_mock_fsky (0.6), cmb_mock_noise (33 muK arcmin), cmb_mock_beam (7 arcmin), cmb_mock_bb (false)
        std::unique_ptr<Likelihood> create(const std::string & name,
                                           const std::map<std::string, std::string> & options = {});
        std::vector<std::string> names();
        std::vector<std::string> option_names();
    } // namespace Likelihoods

} // namespace FML::COSMOLOGY::FITTING

#endif
