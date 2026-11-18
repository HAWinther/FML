#ifndef FML_COSMOLOGYFITTING_DESIFULLSHAPE_H
#define FML_COSMOLOGYFITTING_DESIFULLSHAPE_H

// =============================================================================
// DESI DR1 full-shape power-spectrum likelihood (DESI 2024 V, arXiv:2411.12021; DESI 2024 VII, arXiv:2411.12022), as
// DESI's Cobaya likelihood desi_fs_bao_all with observable_name = spectrum-poles-rotated and its theory
// reptvelocileptors (github.com/cosmodesi/desi-kp-cosmological-likelihoods, dr1/cobaya):
//
//   desi_dr1_fs           BGS (z 0.1-0.4), LRG (0.4-0.6, 0.6-0.8, 0.8-1.1), ELG (1.1-1.6), QSO (0.8-2.1): the rotated
//                         P_0, P_2 (k = 0.02-0.2 h/Mpc, 72 values per sample) with the rotated window matrix and the
//                         covariance including the rotation, HOD and photometric systematics
//   desi_dr1_fs_<NAME>    one sample, NAME = BGS_z0, LRG_z0, LRG_z1, LRG_z2, ELG_z1, QSO_z0
//
// Theory: one-loop EPT with IR resummation (velocileptors' REPT, PerturbationTheory/EPTRedshiftSpace.h) of the linear
// CDM + baryon P(k) at the first effective redshift, with the loops scaled by D^4, the linear terms from P_cb(k, z) and
// its no-wiggle part, and the scale-dependent growth rate f(k) = sqrt(P_thetatheta/P_cb) at each redshift
// (ept_fullresum_varyDz_nu_fftw); D = sqrt(P_cb(1 h/Mpc, z)/P_cb(1 h/Mpc, z_0)). The no-wiggle P(k) from the
// peak-average BAO filter (PeakAverageBAOFilter), AP with q_par = E_fid/E, q_per = D_M h/(D_M h)_fid relative to
// cosmoprimo's fiducial DESI cosmology (Planck 2018). Nothing in it assumes a model: for modified gravity or coupled
// dark matter the linear P_cb, P_thetatheta and growth carry the model (with EdS loop kernels).
//
// Nuisance parameters per sample in DESI's "physical" basis (desi_<NAME>_b1p, _b2p, _bsp; b3p = 0): b1p = (1 + b1L)
// sigma8(z), b2p = b2L sigma8^2, bsp = bsL sigma8^2 with sigma8 = sigma_8,cb(z) (DESI's priors: b1p flat in [0, 3], b2p,
// bsp Gaussian 0 +- 5). The counterterms alpha0p, alpha2p (Gaussian 0 +- 12.5) and the stochastic sn0p (0 +- 2), sn2p
// (0 +- 5) enter linearly and are marginalized analytically (alpha4p = alpha6p = sn4p = 0): with solve = marg (default,
// as DESI's likelihood) ln L includes -ln det(F)/2 of the posterior Fisher matrix F of those parameters, with solve =
// best it is the maximum over them (for best fits and profiles, Cobaya's solve: 'best').
//
// Data: external/desi_dr1_fs/ (python/convert_desi_dr1_fullshape.py from DESI's likelihood files).
// =============================================================================

#include <FML/CosmologyFitting/Likelihoods.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::FITTING {

    class DESIDR1FullShapeLikelihood : public Likelihood {
      public:
        // samples: a subset of BGS_z0, LRG_z0, LRG_z1, LRG_z2, ELG_z1, QSO_z0 (kept in DESI's order); marginalize: solve
        // = marg (true) or best (false)
        DESIDR1FullShapeLikelihood(const std::string & data_dir,
                                   std::vector<std::string> samples,
                                   std::string name,
                                   bool marginalize = true);
        std::string name() const override { return label; }
        bool needs_recombination() const override { return true; }
        double loglike(const Theory & t) const override;
        int n_data() const override;
        std::vector<std::string> nuisance_parameters() const override;
        std::vector<double> growth_redshifts() const override { return zeff; }
        std::vector<double> pk_redshifts() const override { return zeff; }
        bool needs_pk_theta() const override { return true; }
        double pk_k_max() const override { return kmax_pert; }

        // The window-convolved theory P_0, P_2 of sample i for the nuisance parameters (b1p, b2p, bsp) and the
        // marginalized ones (alpha0p, alpha2p, sn0p, sn2p); and ln L of the sample with them marginalized
        struct SampleResult {
            std::vector<double> theory; // window x poles at the given linear parameters
            double loglike = 0.0;       // marginalized (or maximized) over the linear parameters
            std::vector<double> best_linear;
        };
        SampleResult sample(const Theory & t, size_t i, const std::vector<double> & bias, const std::vector<double> & linear = {}) const;
        // The multipoles P_0, P_2, P_4 [3 x n_k_theory] at the theory k of the window for sample i (before the window)
        std::vector<double> poles(const Theory & t, size_t i, const std::vector<double> & bias, const std::vector<double> & linear) const;
        const std::vector<std::string> & samples() const { return names; }
        // The inputs of the REPT: the linear P_cb, its no-wiggle part and P_thetatheta at each zeff on DESI's input
        // k-grid [h/Mpc], and the AP parameters
        struct LinearInputs {
            std::vector<double> k;
            std::vector<std::vector<double>> pdd, pnw, ptt; // [z][k]
            std::vector<double> qpar, qper;
        };
        LinearInputs linear_inputs(const Theory & t) const;
        // ln L of each sample for given REPT inputs, sigma_8,cb and f = sigma_8,thetatheta/sigma_8,cb at zeff, and
        // biases (b1p, b2p, bsp) per sample (for tests against DESI's pipeline with the same inputs)
        std::vector<double> loglike_samples(const LinearInputs & in,
                                            const std::vector<double> & sigma8,
                                            const std::vector<double> & f,
                                            const std::vector<std::vector<double>> & bias) const;
        const std::vector<double> & k_theory() const { return kth; }

        // > 0: the minimum k_max [h/Mpc] of the perturbations. 0 (default): that of the run (0.3/Mpc), beyond which P(k)
        // is extrapolated: ln L changes by < 0.003 with 3 h/Mpc
        double kmax_pert = 0.0;

      private:
        std::string label;
        bool marg;
        std::vector<std::string> names;
        std::vector<double> zeff, shotnoise, kth;
        std::vector<std::vector<double>> data;                 // [sample][72]
        std::vector<std::vector<std::vector<double>>> window;  // [sample][72][3 nk]
        std::vector<std::vector<std::vector<double>>> icov;    // [sample][72][72]
        // The fiducial (cosmoprimo's DESI cosmology): D_M h, D_H h at zeff, h r_d and the BAO filter (set up on first
        // use with the precision of the theory)
        struct Fiducial {
            std::vector<double> dmh, dhh;
            double hrd = 0.0;
            PeakAverageBAOFilter filter;
        };
        mutable Fiducial fid;
        mutable std::once_flag fid_once;
        void setup_fiducial(const Theory & t) const;
        // The multipole tables of all samples for a theory (kept with the theory): [sample][ell 0, 2, 4][k][19]
        struct Tables;
        const Tables & tables(const Theory & t) const;
        Tables make_tables(const LinearInputs & in) const;
        std::vector<double> poles_impl(const Tables & T, size_t i, double s8, double f, const std::vector<double> & bias,
                                       const std::vector<double> & linear) const;
        SampleResult sample_impl(const Tables & T, size_t i, double s8, double f, const std::vector<double> & bias,
                                 const std::vector<double> & linear) const;
    };

} // namespace FML::COSMOLOGY::FITTING

#endif
