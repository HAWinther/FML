#ifndef HALOMODEL_INCLUDE
#define HALOMODEL_INCLUDE

#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>
#include <FML/Math/Math.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/Timing/Timings.h>
#include <FML/SphericalCollapse/SphericalCollapse.h>
#include <cmath>
#include <tuple>
#include <iostream>
#include <fstream>

namespace FML {
  namespace COSMOLOGY {
    namespace HALOMODEL {

      extern int _ACCURACY_BOOST;

      class HaloModel;

      using Timer = FML::UTILS::Timings;
      using DVector = FML::SOLVERS::ODESOLVER::DVector;
      using DVector2D = FML::SOLVERS::ODESOLVER::DVector2D;
      using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
      using ODEFunctionJacobian = FML::SOLVERS::ODESOLVER::ODEFunctionJacobian;
      using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
      using Spline = FML::INTERPOLATION::SPLINE::Spline;
      using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
      using Func1 = std::function<double(double)>;
      using Func2 = std::function<double(double,double)>;
      using SphericalCollapse = FML::COSMOLOGY::SPHERICALCOLLAPSE::SphericalCollapse;
      using SphericalCollapseModel = FML::COSMOLOGY::SPHERICALCOLLAPSE::SphericalCollapseModel;

      // Typedefs for providing functions used by the halomodel
      using RhoFunction = std::function<double(double,double,HaloModel *hm)>;
      using cofMFunction = std::function<double(double,HaloModel *hm)>;
      using HaloMultiplicityFunction = std::function<double(double,HaloModel *hm)>;
      using HaloBiasFunction = std::function<double(double,HaloModel *hm)>;
      using WindowFunction = std::function<double(double)>;

      // Fiducial functions (NFW profile, ST mass-function, ...)
      RhoFunction get_nfw_halo_density_profile();
      cofMFunction get_nfw_halo_mass_concentration_relation();
      HaloMultiplicityFunction get_sheth_tormen_halo_multiplicity_function();
      HaloBiasFunction get_sheth_tormen_halo_bias_function();
      WindowFunction get_tophat_window_fourier();

      //===========================================================
      /// @brief Class for computing the standard halomodel predictions
      /// for \f$P(k,z)\f$ plus \f$n(M,z)\f$ or just doing spherical collapse. 
      /// Contains the HMCode modifications so one can run both the standard halomodel
      /// and HMCode by setting a flag. 
      /// The mass-consentration relation, PDF and bias and halo density profile 
      /// can easily be modified if wanted by providing a lambda functions defining these
      /// functions..
      ///
      /// The cosmology/gravity model is specified through the spherical-collapse model. 
      /// This currently only handles scale-independent growth.
      //===========================================================
      class HaloModel {
        public:
          Timer timer;
          SphericalCollapseModel spcmodel;
          double OmegaM{};
          double fnu{};

          //===========================================
          // The linear power-spectrum
          // and the time x=log(a) it is given at
          // (fiducial is z=0)
          //===========================================
          Spline logDelta_of_logk_spline{"logDelta_lin"};;
          double xinput_pofk{};

          //===========================================
          // Values computed at current redshift 
          // after running compute_at_redshift
          //===========================================
          double DeltaVir{};
          double deltac{};
          double sigma8{};
          double sigmav{};
          double sigmav100{};
          double xcollapse{std::numeric_limits<double>::quiet_NaN()};
          double rnl_nu{};
          double neff{};

          //=====================================================
          // Halo massfunction modelling
          // The fiducial option is Sheth-Tormen 1999
          //=====================================================
          std::string halo_massfunction = "ShethTormen1999";
          HaloMultiplicityFunction halo_multiplicity_function = get_sheth_tormen_halo_multiplicity_function();
          HaloBiasFunction halo_bias_function = get_sheth_tormen_halo_bias_function();

          // ST parameters
          double p_ST = 0.3;
          double a_ST = 0.707;
          double A_ST = 0.2162;

          //=====================================================
          // The density profile of halos
          // The fiducial option is NFW 
          // with the possibillity of using the HMCode modifications
          // Halo concentration. The fiducial option is Bullock 2001
          //=====================================================
          std::string halo_profile = "NFW";
          std::string cofM_model = "Bullock2001";
          RhoFunction halo_density_profile = get_nfw_halo_density_profile();
          cofMFunction halo_mass_concentration_relation = get_nfw_halo_mass_concentration_relation();

          //===========================================
          // Range for P(k) calculation
          //===========================================
          double kmin_pofk = 1e-5;
          double kmax_pofk = 100.0;
          int npts_pofk = 1000;

          //===========================================
          // Modified halomodel parameters
          //===========================================
          bool hmcode{true};
          bool verbose{true};
          double hmcode_kstar = 0.01;          // 0.0 fiducial but 0.01 to avoid messing up lowk
          double hmcode_alpha = 1.0;           // 1.0 fifucial
          double hmcode_eta = 0.0;             // 0.0 fiducial
          double hmcode_cmin = 4.0;            // 4.0 fiducial
          double hmcode_f = 0.0;               // 0.0 fiducial
          double cofM_0    = 4.75;             // For non-Bullock c(M)
          double cofM_M0   = 1e14;             // For non-Bullock c(M)
          double cofM_beta = 0.13;             // For non-Bullock c(M)

          //===========================================
          // Spherical collapse (accuracy)
          //===========================================
          double xini_spherical_collapse = std::log(1e-5);
          int npts_spherical_collapse = 1000;

          //===========================================
          // For globally setting accuract parameters instead
          // of relying on fiducial parameters inside methods 
          // to set arrays
          //===========================================
          std::shared_ptr<DVector> fiducial_logR_array{nullptr};
          std::shared_ptr<DVector> fiducial_logk_array{nullptr};

          //===========================================
          // Parameters for simply rescaling some
          // of the halomodel parameters from their
          // fiducial value
          //===========================================
          double DeltaVir_multiplier = 1.0;
          double cofM_multiplier = 1.0;
          double deltac_multiplier = 1.0;

          //===========================================
          // Splines we generate as we run
          //===========================================
          Spline growthfactor_of_x_spline{"Growthfactor"};
          Spline growthrate_of_x_spline{"Growthrate"};;
          Spline deltac_of_x_spline{"deltac"};;
          Spline DeltaVir_of_x_spline{"DeltaVir"};;
          Spline xta_of_x_spline{"xta"};;
          Spline xnl_of_x_spline{"xnl"};;
          Spline xvir_of_x_spline{"xvir"};;
          Spline delta_ini_of_x_spline{"deltaini"};;
          Spline logsigma_of_logR_spline{"logsigma"};;
          Spline formationredshift_of_logM_spline{"zf"};;
          Spline lognu_of_logR_spline{"lognuoflogR"};;
          Spline lognu_of_logM_spline{"lognuoflogM"};;
          Spline logR_of_lognu_spline{"logRoflognu"};;
          Spline dndlogM_of_logM_spline{"dndlogM"};;
          Spline n_of_logM_spline{"nofM"};;
          Spline logDeltaHM_onehalo_of_logk_spline{"logDelta_1h"};;
          Spline logDeltaHM_twohalo_of_logk_spline{"logDelta_2h"};;
          Spline logDeltaHM_full_of_logk_spline{"logDelta"};;
          Spline2D y_of_logR_and_logk_spline{"y"};;

          //===========================================
          // Constructor
          //===========================================
          HaloModel(const HaloModel&) = default;
          HaloModel & operator=(const HaloModel&) = default;
          HaloModel() = default;
          HaloModel(
              SphericalCollapseModel spcmodel, 
              Spline logDelta_of_logk_spline, 
              double xinput_pofk = 0.0, 
              bool hmcode = false,
              bool verbose = false,
              double xini = std::log(1e-5), 
              int npts = 1000);
          ~HaloModel() = default;

          //===========================================
          // Smoothing filter for computation of
          // sigma(R), sigmav(R), sigmav100(R)
          //===========================================
          WindowFunction fourier_window_function = get_tophat_window_fourier();

          //=====================================================
          // Initialize a run. Runs in the constructor
          //=====================================================
          void init();

          //=====================================================
          // Compute everything at redshift z
          //=====================================================
          void compute_at_redshift(double zcollapse);

          //=====================================================
          // Output P(k) over whole range computed
          //=====================================================
          void output_pofk(
              std::string filename,
              double kmin = 1e-5,
              double kmax = 100,
              int npts_k = 1000) const;

          //=====================================================
          // Output deltac(z) and DeltaVir(z)
          //=====================================================
          void output_deltac(std::string filename,
              double zmin = 0.0,
              double zmax = 10.0,
              int npts_z = 100) const;

          //=====================================================
          // Output mass-function
          //=====================================================
          void output_nofM(
              std::string filename,
              double Mmin = 1e8,
              double Mmax = 1e16,
              int npts_M = 100) const;

          //=====================================================
          // Show some info about what we computed
          //=====================================================
          void info() const;

          double get_DeltaVir_hmcode(double x) const;
          double get_deltac_hmcode(double x) const;
          double get_eta_hmcode(double x) const;
          double get_kstar_hmcode(double x) const;
          double get_f_hmcode(double x) const;
          double get_cmin_hmcode(double x) const;
          double get_alpha_hmcode() const;
      };

      // Take in a file with format [k, P(k)] in (h/Mpc, Mpc^3/h^3) units
      // Read and make a spline of the dimensionless power-spectrum
      // normalization is used if we by any chance want to scale P(k)
      void read_pofk_file(
          const std::string filename_pofk,
          Spline & logDelta_of_logk_spline,
          double normalization = 1.0,
          int col_k = 0,
          int col_pofk = 1);

      // Assumes k in (h/Mpc), P(k) in (Mpc/h)^3,
      // R is in Mpc/h. File has format [k, P(k)]
      void compute_sigma(
          const std::function<double(double)> logDelta_of_logk,
          const WindowFunction window_of_kR,
          Spline & logsigma_of_logR_spline,
          const DVector *fiducial_logR_array = nullptr,
          const DVector *fiducial_logk_array = nullptr);

      // M in Msun/h and R given in Mpc/h
      double RvirofM(
          double M,
          double DeltaVir, 
          double OmegaM, 
          double x);

      // R in Mpc/h to mass in Msun/h
      double RofM(
          double M, 
          double OmegaM, 
          double x);

      // M in Msun/h and R given in Mpc/h
      double MofRvir(
          double Rvir, 
          double DeltaVir,
          double OmegaM, 
          double x);

      // M in Msun/h and R given in Mpc/h
      double MofR(
          double R,
          double OmegaM,
          double x);

      // rho_function is 4pi*rho(r,M)r^3/M as function of (r,M)
      DVector compute_rho_fourier_single(
          const std::function<double(double, double)> & rho_function,
          const DVector & k_array, 
          const double M,
          const double Rvir); 

      // rho_function is 4pi*rho(r,M)r^3/M as function of (r,M)
      void compute_rho_fourier(
          const std::function<double(double, double)> & rho_function,
          Spline2D & y_of_logR_and_logk_spline,
          const double DeltaVir, 
          const double OmegaM, 
          const double x,
          const DVector *fiducial_logR_array = nullptr,
          const DVector *fiducial_logk_array = nullptr);

      void compute_lognu_of_logR(
          const double deltac, 
          const Spline & logsigma_of_logR_spline,
          Spline & lognu_of_logR_spline,
          Spline & logR_of_lognu_spline,
          const DVector *fiducial_logR_array = nullptr);

      void compute_massfunction(
          const Spline & logR_of_lognu_spline,
          const Spline & lognu_of_logR_spline,
          Spline & dndlogM_of_logM_spline,
          Spline & n_of_logM_spline,
          const std::function<double(double)> & pdf_of_nu,
          const double OmegaM,
          const double x);

      void compute_one_and_two_halo_terms(
          const Spline & logR_of_lognu_spline, 
          const Spline2D & y_of_logR_and_logk_spline,
          Spline & logDelta_onehalo_of_logk_spline, 
          Spline & logDelta_twohalo_of_logk_spline,
          Spline & logDelta_full_of_logk_spline,
          const std::function<double(double)> & logDeltaLin_of_logk,
          const std::function<double(double)> & pdf_of_nu,
          const std::function<double(double)> & bias_of_nu,
          const double eta_hmcode = 0.0,
          const double kmin = 1e-5,
          const double kmax = 100.0,
          const int npts_k = 100);

      // Computes zf in factor = (1+zf(M))/(1+zcollapse) used in Bullock 2001 c(M) = A*factor
      void compute_formation_redshift_factor(
          const Spline & growthfactor_of_x_spline,
          const Spline & logsigma_of_logR_spline,
          Spline & formationredshift_of_logM_spline,
          double deltac,
          double OmegaM,
          double xcollapse,
          double massfraction = 0.01);

      // Compute sigmav
      double compute_sigmav(
          const std::function<double(double)> logDelta_of_logk,
          const WindowFunction window_of_kR,
          double R = 0.0,
          const DVector *fiducial_logk_array = nullptr);

      // Return (R,neff) where nu(R)=1 and neff=-3-dsigma^2/dlogR
      std::pair<double,double> compute_neff(
          const Spline & logsigma_of_logR_spline,
          const Spline & lognu_of_logR_spline);

    }
  }
}

#endif
