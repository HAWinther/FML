#ifndef FML_COSMOLOGYFITTING_PLANCKLIKELIHOODS_H
#define FML_COSMOLOGYFITTING_PLANCKLIKELIHOODS_H

// =============================================================================
// Planck likelihoods ported from Cobaya's native (python) versions (data: python/fetch_planck_data.py):
//
//   planck_lowl_TT   Planck 2018 low-ell TT, 2 <= ell <= 29 (Commander Gibbs likelihood; Planck 2018 V,
//                    arXiv:1907.12875), as Cobaya's planck_2018_lowl.TT: each D_ell / A_planck^2 is mapped to a
//                    Gaussianized variable x_ell by a cubic spline (scipy's InterpolatedUnivariateSpline = not-a-knot)
//                    of the tabulated transformation, ln L = sum ln(dx/dD) - (x - mu)^T C^-1 (x - mu)/2 - offset (the
//                    offset makes ln L = 0 at the tabulated mu_sigma spectrum). Outside the tabulated range: ln L =
//                    -inf. Data: external/planck2018_lowT/.
//   planck_lowl_EE   Planck 2018 low-ell EE, 2 <= ell <= 29 (SimAll; the clik file simall_100x143_offlike5_EE_
//                    Aplanck_B), as Cobaya's planck_2018_lowl.EE: ln L = sum_ell table[int(D_ell / A_planck^2 / 1e-4),
//                    ell] (steps of 1e-4 muK^2 up to 0.3 muK^2; beyond: ln L = -inf). Data: external/planck2018_lowE/.
//   planck_camspec_npipe_TTTEEE   High-ell CamSpec PR4 (NPIPE) 12.6cl (Rosenberg, Gratton & Efstathiou 2022,
//                    arXiv:2205.10869), as Cobaya's planck_NPIPE_highl_CamSpec.TTTEEE: TT of 143x143 (30-2000),
//                    217x217 (500-2500) and 143x217 (500-2500), TE and EE (30-2000), 9915 multipoles with the full
//                    covariance (Cholesky-factorized once, ~1 min). Model: (D_ell + foreground) / cal with the
//                    power-law foreground residuals amp_X (ell/1500)^n_X (Cobaya's default use_fg_residual_model = 0),
//                    cal = A_planck^2 for TT, A_planck^2 calTE for TE, A_planck^2 calEE for EE.
//                    Nuisance parameters (Cobaya's names; value if not sampled): amp_143 (10), amp_217 (20),
//                    amp_143x217 (10) with flat priors [0, 50]; n_143, n_217, n_143x217 (1) flat in [0, 5]; calTE,
//                    calEE (1) with Gaussian priors 1 +- 0.01 (gauss_calTE, gauss_calEE); A_planck (1, gauss 0.0025).
//                    Data: external/planck_npipe_camspec/.
//   planck_camspec_npipe_TT, planck_camspec_npipe_TE, planck_camspec_npipe_EE, planck_camspec_npipe_TTTE ... the
//                    same with only those spectra (cut covariance).
//
// and the ACT DR6 lensing likelihood (act_dr6_lenslike 1.2.1, Madhavacheril et al. 2023, arXiv:2304.05203; Qu et
// al. 2023, arXiv:2304.05202) as its Cobaya likelihood with the CMB-dependent corrections:
//   act_dr6_lensing_actplanck   ACT DR6 + Planck PR4 lensing (variant actplanck_baseline), as used with CMB spectra
//   act_dr6_lensing             ACT DR6 alone (act_baseline); _extended for the extended L range of either
//                    Gaussian in the bandpowers of C_L^kk with model = M_kk C^kk + sum_X M_X C^X + offset (X = TT, EE,
//                    BB, TE; C^kk = [L(L+1)]^2 C_L^phiphi / 4, the lensed C_ell in muK^2): the binning and the linear
//                    corrections for the dependence of the normalization and of N1 on the spectra, precomputed by
//                    python/convert_act_dr6_lensing.py, which also applies the Hartlap factor to the inverse
//                    covariance. Needs the spectra up to L ~ 3000. Data: external/act_dr6_lensing/.
// =============================================================================

#include <FML/CosmologyFitting/Likelihoods.h>

#include <string>
#include <vector>

namespace FML::COSMOLOGY::FITTING {

    // A cubic spline with not-a-knot end conditions (the interpolating cubic B-spline of scipy's
    // InterpolatedUnivariateSpline with k = 3), value and first derivative; x strictly increasing, n >= 4
    class NotAKnotSpline {
      public:
        NotAKnotSpline() = default;
        NotAKnotSpline(std::vector<double> x, std::vector<double> y);
        double operator()(double x) const;
        double derivative(double x) const;

      private:
        std::vector<double> xs, ys, M; // second derivatives at the nodes
        size_t interval(double x) const;
    };

    class PlanckLowlTTLikelihood : public Likelihood {
      public:
        explicit PlanckLowlTTLikelihood(const std::string & data_dir, int lmin = 2, int lmax = 29);
        std::string name() const override { return "planck_lowl_TT"; }
        bool needs_recombination() const override { return false; }
        bool needs_cmb() const override { return true; }
        int cmb_ell_max() const override { return lmax; }
        std::vector<std::string> nuisance_parameters() const override { return {"A_planck"}; }
        double loglike(const Theory & t) const override;
        int n_data() const override { return lmax - lmin + 1; }
        // D_ell = l(l+1)C_l/2pi [muK^2] indexed by ell
        double loglike(const std::vector<double> & Dtt, double A_planck = 1.0) const;

      private:
        int lmin, lmax;
        std::vector<NotAKnotSpline> spline;
        std::vector<double> lo, hi, mu;
        std::vector<std::vector<double>> icov;
        double offset = 0.0;
    };

    class PlanckLowlEELikelihood : public Likelihood {
      public:
        explicit PlanckLowlEELikelihood(const std::string & data_dir);
        std::string name() const override { return "planck_lowl_EE"; }
        bool needs_recombination() const override { return false; }
        bool needs_cmb() const override { return true; }
        int cmb_ell_max() const override { return lmax; }
        std::vector<std::string> nuisance_parameters() const override { return {"A_planck"}; }
        double loglike(const Theory & t) const override;
        int n_data() const override { return lmax - lmin + 1; }
        double loglike(const std::vector<double> & Dee, double A_planck = 1.0) const;

      private:
        static constexpr int lmin = 2, lmax = 29;
        static constexpr double step = 1e-4;
        std::vector<std::vector<double>> table; // [step][ell - lmin]
    };

    class CamSpecNPIPELikelihood : public Likelihood {
      public:
        // spectra: a subset of "143x143 217x217 143x217 TE EE" (the 100x100 spectrum is not used, as Cobaya)
        CamSpecNPIPELikelihood(const std::string & data_dir,
                               const std::vector<std::string> & spectra,
                               std::string name);
        std::string name() const override { return label; }
        bool needs_recombination() const override { return false; }
        bool needs_cmb() const override { return true; }
        int cmb_ell_max() const override { return 2500; }
        std::vector<std::string> nuisance_parameters() const override;
        double loglike(const Theory & t) const override;
        int n_data() const override { return int(data.size()); }

        struct Nuisance {
            double A_planck = 1.0, calTE = 1.0, calEE = 1.0;
            double amp_143 = 10.0, amp_217 = 20.0, amp_143x217 = 10.0;
            double n_143 = 1.0, n_217 = 1.0, n_143x217 = 1.0;
        };
        // chi^2 for D_ell [muK^2] indexed by ell (up to 2500)
        double chi2(const std::vector<double> & Dtt,
                    const std::vector<double> & Dte,
                    const std::vector<double> & Dee,
                    const Nuisance & p) const;

      private:
        std::string label;
        struct Block {
            std::string name;
            int lmin, lmax; // used range
        };
        std::vector<Block> blocks;      // in the order of the data vector
        std::vector<double> data;       // the used multipoles
        std::vector<double> L;          // Cholesky factor (lower, row-major) of the covariance
        bool has_tt = false, has_te = false, has_ee = false;
    };

    class ACTDR6LensingLikelihood : public Likelihood {
      public:
        // variant: act_baseline, act_extended, actplanck_baseline, actplanck_extended
        ACTDR6LensingLikelihood(const std::string & data_dir, const std::string & variant, std::string name);
        std::string name() const override { return label; }
        bool needs_recombination() const override { return false; }
        bool needs_cmb() const override { return true; }
        int cmb_ell_max() const override { return lmax; }
        double loglike(const Theory & t) const override;
        int n_data() const override { return int(data.size()); }
        // ln L for the lensed C_ell [muK^2, no ell factors] and C_L^kk, indexed by L (up to cmb_ell_max())
        double loglike_cl(const std::vector<double> & cl_tt,
                          const std::vector<double> & cl_ee,
                          const std::vector<double> & cl_bb,
                          const std::vector<double> & cl_te,
                          const std::vector<double> & cl_kk) const;

      private:
        std::string label;
        int lmax = 0;
        std::vector<double> data, offset;
        std::vector<std::vector<double>> cinv, m_kk, m_tt, m_ee, m_bb, m_te; // [bin][L]
    };

} // namespace FML::COSMOLOGY::FITTING

#endif
