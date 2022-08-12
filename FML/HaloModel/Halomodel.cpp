#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>
#include <FML/Math/Math.h>
#include <FML/FileUtils/FileUtils.h>
#include "Halomodel.h"

namespace FML {

  namespace COSMOLOGY {

    namespace HALOMODEL {

      //================================================
      // 2 for <1% accuracy. Runtime is quadratic in this
      //================================================
      int _ACCURACY_BOOST = 1;

      // Extend a logY(logx) spline assuming 
      // Y ~ x^n close to the endpoints
      // Useful for extending e.g. P(k) needed
      // for halomodel calculations
      void powerlaw_extend_logY_of_logx_spline(
          Spline & logY_of_logx_spline,
          double xmin_new,
          double xmax_new,
          int npts_new,
          std::string label = "logY(log(x)) extrapolated"){

        // Fetch logx-array from spline
        const auto logx_array = logY_of_logx_spline.get_x_data();
        const double logxmin = logx_array.front();
        const double logxmax = logx_array.back();

        // Compute slope near end-points
        double slope_min  = logY_of_logx_spline.deriv_x(logxmin*0.99);
        double slope_max = logY_of_logx_spline.deriv_x(logxmax*1.01);

        // Extrapolation function
        auto logY_extrapolated = [&](double logx){
          if(logx < logxmin){
            return logY_of_logx_spline(logxmin) + (logx-logxmin) * slope_min;
          } else if(logx > logxmax){
            return logY_of_logx_spline(logxmax) + (logx-logxmax) * slope_max;
          }
          return logY_of_logx_spline(logx);
        };

        // Extend spline to one covering a larger region
        const auto logx_array_new = FML::MATH::linspace(std::log(xmin_new), std::log(xmax_new), npts_new);
        DVector logY_array_new;
        for(const auto logx : logx_array_new){
          logY_array_new.push_back( logY_extrapolated(logx) );
        }
        logY_of_logx_spline = Spline(logx_array_new, logY_array_new, label);
      }

      //================================================
      // Take in a file with format [k, P(k)] in (h/Mpc, Mpc^3/h^3) units
      // Read and make a spline of the dimensionless power-spectrum
      //================================================
      void read_pofk_file(
          const std::string filename_pofk,
          Spline & logDelta_of_logk_spline,
          double normalization,
          int col_k,
          int col_pofk){

        // Read file
        auto data = FML::FILEUTILS::loadtxt(filename_pofk);

        // Extract data, transform to Delta, normalize and spline
        DVector logk_array, logDelta_array;
        for(size_t i = 0; i < data.size(); i++){
          double k = data[i][col_k];
          double pofk = data[i][col_pofk];
          double Delta_over_pofk = (k*k*k) / (2.0*M_PI*M_PI);
          logk_array.push_back( std::log(k) );
          logDelta_array.push_back( std::log( pofk * Delta_over_pofk * normalization ) );
        }

        // Make spline
        logDelta_of_logk_spline = Spline(logk_array, logDelta_array, "logDelta(logk)");

        // Extrapolate assuming ~ k^n at each end
        // and remake spline
        const double kmin = 1e-12;
        const double kmax = 1e12;
        const int npts_spline = 1000;
        powerlaw_extend_logY_of_logx_spline(
            logDelta_of_logk_spline,
            kmin,
            kmax,
            npts_spline,
            "logDelta(logk) extrapolated");
      }

      //================================================
      // Computes neff = -3 - d/dlogR sigma^2
      // at the point where nu == deltac/sigma == 1
      //================================================
      std::pair<double,double> compute_neff(
          const Spline & logsigma_of_logR_spline,
          const Spline & lognu_of_logR_spline){
        const double logRmin = std::log(1e-4);
        const double logRmax = std::log(1e4);
        const double epsilon = 1e-10;

        const double lognu_goal = std::log(1.0);
        auto condition = [&](double logR){
          return lognu_of_logR_spline(logR) - lognu_goal;
        };
        const double logR = FML::MATH::find_root_bisection(condition, {logRmin, logRmax}, epsilon);
        const double neff = -3.0 - 2.0 * logsigma_of_logR_spline.deriv_x(logR);
        return std::pair(std::exp(logR), neff);
      }

      //================================================
      // Computes F(R) = 1/3 * Int W(kR)^2 Delta(k)/k^2 dlogk
      // sigmav = F(0) and sigmav100 = F(100)
      //================================================
      double compute_sigmav(
          const Func1 logDelta_of_logk,
          const WindowFunction window_of_kR,
          double R,
          const DVector *fiducial_logk_array){
        auto integrand = [&](double logk){
          const double W = window_of_kR(std::exp(logk)*R);
          return W*W*std::exp(logDelta_of_logk(logk))*std::exp(-2*logk)/3.0;
        };

        const double kmin = 1e-8, kmax = 1e4;
        const int npts_k = 1000;
        auto logk_array = FML::MATH::linspace(std::log(kmin), std::log(kmax), npts_k);
        if(fiducial_logk_array != nullptr)
          logk_array = *fiducial_logk_array;
        auto k_array = logk_array;
        for(auto & k : k_array) k = std::exp(k);
        double integral = 0.0;
        for(const auto logk : logk_array){
          integral += integrand(logk);
        }
        integral *= (logk_array[1]-logk_array[0]);
        return std::sqrt(integral);
      }

      //================================================
      // Computes the variance Int Delta(k)W^2(kR) dlogk
      //================================================
      void compute_sigma(
          const Func1 logDelta_of_logk,
          const WindowFunction window_of_kR,
          Spline & logsigma_of_logR_spline,
          const DVector *fiducial_logR_array,
          const DVector *fiducial_logk_array){

        //================================================
        // Radius array
        // If not provided then we generate it
        //================================================
        const double Rmin = 1e-8, Rmax = 1e8;
        const int npts_R = 500;
        auto logR_array = FML::MATH::linspace(std::log(Rmin), std::log(Rmax), npts_R);
        if(fiducial_logR_array != nullptr)
          logR_array = *fiducial_logR_array;
        auto R_array = logR_array;
        for(auto & R : R_array) R = std::exp(R);

        //================================================
        // k integration array. Linear in logk
        // If not provided then we generate it
        //================================================
        const double kmin = 1e-8, kmax = 1e8;
        const int npts_k = 500;
        auto logk_array = FML::MATH::linspace(std::log(kmin), std::log(kmax), npts_k);
        if(fiducial_logk_array != nullptr)
          logk_array = *fiducial_logk_array;
        auto k_array = logk_array;
        for(auto & k : k_array) k = std::exp(k);

        //================================================
        // Compute and spline sigma(R)
        //================================================
        DVector logsigma_array;
        for(const auto R : R_array){

          auto integrand = [&logDelta_of_logk,&window_of_kR](double logk, double R){
            const double k = std::exp(logk);
            const double W = window_of_kR(k*R);
            return std::exp(logDelta_of_logk(logk)) * W * W;
          };

          double integral = 0.0;
          for(const auto logk : logk_array){
            integral += integrand(logk, R);
          }
          integral *= (logk_array[1]-logk_array[0]);
          logsigma_array.push_back(0.5*std::log(integral));
        }
        logsigma_of_logR_spline = Spline(logR_array, logsigma_array, "sigma(R)");
      }

      //================================================
      // M in Msun/h and R given in Mpc/h
      //================================================
      double RvirofM(
          double M,
          double DeltaVir, 
          double OmegaM, 
          double x){
        return RofM(M, OmegaM, x) * std::pow(DeltaVir, -1.0/3.0);
      }

      //================================================
      // R in Mpc/h to mass in Msun/h
      //================================================
      double RofM(
          double M, 
          double OmegaM, 
          double x){
        const double Mpl2overMsunH0overh = 2.493e21;
        const double H0overh_in_hmpc = 1.0 / 2997.92458;
        const double rho_mean_in_msunh_over_mpch3 = 3.0 * OmegaM * std::exp(-3.0*x) 
          * std::pow(H0overh_in_hmpc,3) * Mpl2overMsunH0overh;
        return std::pow(M / (4.0 * M_PI / 3.0 * rho_mean_in_msunh_over_mpch3), 1.0/3.0);
      }

      //================================================
      // M in Msun/h and R given in Mpc/h
      //================================================
      double MofRvir(
          double Rvir, 
          double DeltaVir,
          double OmegaM, 
          double x){
        return MofR(Rvir, OmegaM, x) * DeltaVir;
     }

      //================================================
      // M in Msun/h and R given in Mpc/h
      //================================================
      double MofR(
          double R,
          double OmegaM,
          double x){
        const double Mpl2overMsunH0overh = 2.493e21;
        const double H0overh_in_hmpc = 1.0 / 2997.92458;
        const double rho_mean_in_msunh_over_mpch3 = 3.0 * OmegaM * std::exp(-3.0*x) 
          * std::pow(H0overh_in_hmpc,3) * Mpl2overMsunH0overh;
        return 4.0 * M_PI / 3.0 * rho_mean_in_msunh_over_mpch3 * std::pow(R, 3);
      }

      //================================================
      // Computes the fourier transform y(k,R) of rho_function
      // rho_function is 4pi*rho(r,M)r^3/M as function of (r,M)
      // for a single M
      //================================================
      DVector compute_rho_fourier_single(
          const Func2 & rho_function,
          const DVector & k_array, 
          const double M,
          const double Rvir){ 

        int accuracy_boost = _ACCURACY_BOOST;

        //================================================
        // Determine lower integration limits
        // by requiring rho(rmin) < 1e-6 rho(Rvir/10)
        //================================================
        double rho_start =  rho_function(Rvir/10.0,M); 
        int power_min = 1;
        while(power_min < 5) {
          ++power_min;
          double rho = rho_function(Rvir/std::pow(10,power_min),M);
          if(rho < 1e-4 * rho_start) break;
        };

        //================================================
        // Determine upper integration limit
        //================================================
        int power_max = 0;
        while(power_max < 7) {
          ++power_max;
          double rho = rho_function(Rvir*std::pow(2,power_max),M);
          if(rho == 0.0) break;
        }

        //================================================
        // Radius integration array
        //================================================
        const double rmin = Rvir/std::pow(10,power_min), rmax = Rvir*std::pow(2,power_max);
        const int npts_r = (power_max+power_min) * 50 * accuracy_boost;
        const auto logr_array = FML::MATH::linspace(std::log(rmin), std::log(rmax), npts_r);
        auto r_array = logr_array;
        for(auto & r : r_array) r = std::exp(r);

        DVector y_array;
        for(const auto k : k_array){

          auto integrand = [&rho_function](double k, double r, double M){
            return rho_function(r, M) * std::sin(k*r)/(k*r);
          };

          // Integrate up to get the dimensionless y = rho(k,M)/M
          double integral = 0.0;
          for(const auto r : r_array){
            integral += integrand(k, r, M);
          }
          integral *= (logr_array[1] - logr_array[0]);
          y_array.push_back(integral);
        }

        // Ensure normalization in case rho is not properly normalized
        // y(k,M) -> 1 as k -> 0
        const double norm = y_array[0];
        for(auto & y : y_array) y /= norm;

        return y_array;
      }

      //================================================
      // Computes the fourier transform y(k,R) of rho_function
      // rho_function is 4pi*rho(r,M)r^3/M as function of (r,M)
      // for all M
      //================================================
      void compute_rho_fourier(
          const Func2 & rho_function,
          Spline2D & y_of_logR_and_logk_spline,
          const double DeltaVir, 
          const double OmegaM, 
          const double x,
          const DVector *fiducial_logR_array,
          const DVector *fiducial_logk_array){

        int accuracy_boost = _ACCURACY_BOOST;

        //================================================
        // k array
        //================================================
        const double kmin = 1e-8, kmax = 1e8;
        const int npts_k = 75 * accuracy_boost;
        auto logk_array = FML::MATH::linspace(std::log(kmin), std::log(kmax), npts_k);
        if(fiducial_logk_array != nullptr)
          logk_array = *fiducial_logk_array;
        auto k_array = logk_array;
        for(auto & k : k_array) k = std::exp(k);

        //================================================
        // R array
        //================================================
        const double Rmin = 1e-8, Rmax = 1e8;
        const int npts_R = 75 * accuracy_boost;
        auto logR_array = FML::MATH::linspace(std::log(Rmin), std::log(Rmax), npts_R);
        if(fiducial_logR_array != nullptr)
          logR_array = *fiducial_logR_array;
        auto R_array = logR_array;
        for(auto & R : R_array) R = std::exp(R);

        DVector2D yofkandM_array(R_array.size(), DVector(k_array.size()));
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic,1)
#endif
        for(size_t i = 0; i < R_array.size(); i++){
          const double R = R_array[i];
          const double M = MofR(R, OmegaM, x);
          const double Rvir = R / std::pow(DeltaVir,1.0/3.0);
          const auto result = compute_rho_fourier_single(rho_function, k_array, M, Rvir);
          yofkandM_array[i] = result;
        }
        y_of_logR_and_logk_spline = Spline2D(logR_array, logk_array, yofkandM_array, "y(logR,logk)");
      }

      //================================================
      // Computes a spline of lognu(logR) and the 
      // inverse logR(lognu)
      //================================================
      void compute_lognu_of_logR(
          const double deltac, 
          const Spline & logsigma_of_logR_spline,
          Spline & lognu_of_logR_spline,
          Spline & logR_of_lognu_spline,
          const DVector *fiducial_logR_array){

        //================================================
        // Radius array
        //================================================
        const double logRmin = logsigma_of_logR_spline.get_xrange().first;
        const double logRmax = logsigma_of_logR_spline.get_xrange().second;
        const int npts_R = 500;
        auto logR_array = FML::MATH::linspace(logRmin, logRmax, npts_R);
        if(fiducial_logR_array != nullptr)
          logR_array = *fiducial_logR_array;

        DVector lognu_array;
        for(const auto logR : logR_array){
          const double sigma = std::exp(logsigma_of_logR_spline(logR));
          const double nu = deltac/sigma;
          lognu_array.push_back(std::log(nu));
        }

        lognu_of_logR_spline = Spline(logR_array, lognu_array, "lognu(logR)");
        logR_of_lognu_spline = Spline(lognu_array, logR_array, "logR(lognu)");
      }

      //================================================
      // Computes the mass-function dndlogM(logM)
      // and n(dlogM)
      //================================================
      void compute_massfunction(
          const Spline & logR_of_lognu_spline,
          const Spline & lognu_of_logR_spline,
          Spline & dndlogM_of_logM_spline,
          Spline & n_of_logM_spline,
          const Func1 & pdf_of_nu,
          const double OmegaM,
          const double x){

        // Units of (h/Mpc)^3 (same as 1/R^3)
        auto dndlogM_of_lognu = [&](double lognu){
          const double nu = std::exp(lognu);
          const double logR = logR_of_lognu_spline(lognu);
          const double R = std::exp(logR);
          const double f = pdf_of_nu(nu);
          const double dnudlogR = nu * lognu_of_logR_spline.deriv_x(logR);
          return f * std::abs(dnudlogR) / (4.0 * M_PI * R * R * R);
        };

        //================================================
        // M array
        //================================================
        const double Mmin = 1e5, Mmax = 1e18;
        const int npts_M = 500;
        const auto logM_array = FML::MATH::linspace(std::log(Mmin),std::log(Mmax),npts_M);
        const double dlogM = logM_array[1]-logM_array[0];

        DVector dndlogM_array(npts_M), n_array(npts_M, 0.0), lognu_array(npts_M,0.0);
        for(int i = npts_M-1; i >= 0; i--){
          const double logM = logM_array[i];
          const double M = std::exp(logM);
          const double R = RofM(M, OmegaM, x);
          const double logR = std::log(R);
          const double lognu = lognu_of_logR_spline(logR);
          const double dndlogM = dndlogM_of_lognu(lognu);
          dndlogM_array[i] = dndlogM;
          n_array[i] = dndlogM * dlogM;
          if(i < npts_M-1)
            n_array[i] += n_array[i+1];
        }
        dndlogM_of_logM_spline = Spline(logM_array, dndlogM_array, "dndlogM(logM)");
        n_of_logM_spline = Spline(logM_array, n_array, "n(logM)");
      }

      //================================================
      // Evaluates the one and two halo integrals 
      // P1h(k) and P2h(k) and returns splines of
      // DeltaP = k^3P(k)/2pi^2 for both terms and
      // the sum Delta1h+Delta2h
      //================================================
      void compute_one_and_two_halo_terms(
          const Spline & logR_of_lognu_spline, 
          const Spline2D & y_of_logR_and_logk_spline,
          Spline & logDelta_onehalo_of_logk_spline, 
          Spline & logDelta_twohalo_of_logk_spline,
          Spline & logDelta_full_of_logk_spline,
          const Func1 & logDeltaLin_of_logk,
          const Func1 & pdf_of_nu,
          const Func1 & bias_of_nu,
          const double eta_hmcode,
          const double kmin,
          const double kmax,
          const int npts_k){

        int accuracy_boost = _ACCURACY_BOOST;

        //================================================
        // nu array
        //================================================
        const double numin = 1e-10, numax = 10;
        const int npts_nu = 200 * accuracy_boost;
        const auto lognu_array = FML::MATH::linspace(std::log(numin),std::log(numax),npts_nu);

        //================================================
        // k array
        //================================================
        const auto logk_array = FML::MATH::linspace(std::log(kmin), std::log(kmax), npts_k);
        auto k_array = logk_array;
        for(auto & k : k_array) k = std::exp(k);

        auto integrand_onehalo = [&logR_of_lognu_spline,&y_of_logR_and_logk_spline,&eta_hmcode,&pdf_of_nu](double lognu, double logk){
          const double nu = std::exp(lognu);
          const double logR = logR_of_lognu_spline(lognu);
          const double y = y_of_logR_and_logk_spline(logR, logk + lognu*eta_hmcode);
          const double f = pdf_of_nu(nu);
          const double Moverrhomean = 4.0 * M_PI / 3.0 * std::exp(3.0*logR);
          return Moverrhomean * f * (y*y) * nu;
        };

        auto integrand_twohalo = [&eta_hmcode,&logR_of_lognu_spline,&y_of_logR_and_logk_spline,&pdf_of_nu,&bias_of_nu](double lognu, double logk){
          const double nu = std::exp(lognu);
          const double logR = logR_of_lognu_spline(lognu);
          const double y = y_of_logR_and_logk_spline(logR, logk + lognu*eta_hmcode);
          const double f = pdf_of_nu(nu);
          const double bias = bias_of_nu(nu);
          return bias * f * y * nu;
        };

        // This should integrate to 1
        auto integrand_norm_onehalo = [&pdf_of_nu](double lognu){
          const double nu = std::exp(lognu);
          const double f = pdf_of_nu(nu);
          return f * nu;
        };

        // This should integrate to 1
        auto integrand_norm_twohalo = [&pdf_of_nu,&bias_of_nu](double lognu){
          const double nu = std::exp(lognu);
          const double f = pdf_of_nu(nu);
          const double bias = bias_of_nu(nu);
          return bias * f * nu;
        };

        // Compute Int fdnu and Int bf dnu
        // Both which should be unity
        double normone = 0.0;
        double normtwo = 0.0;
        for(const auto lognu : lognu_array){
          normone += integrand_norm_onehalo(lognu);
          normtwo += integrand_norm_twohalo(lognu);
        }
        normone *= (lognu_array[1]-lognu_array[0]);
        normtwo *= (lognu_array[1]-lognu_array[0]);
        // Sanity check Int f dnu = 1 and Int f b dnu = 1
        if(std::abs(normone-1.0) > 0.001 or std::abs(normtwo-1.0) > 0.001){
          // Can happen normally if massfunction is not normalized but in any case Int bf / Int f == 1
          std::cout << "Warning: Unity check failed for Int f(nu)dnu = " << normone << " Int b(nu)f(nu)dnu = " << normtwo << "\n";
          std::cout << "Could be that massfunction is not normalized. In that case this should be unity: " << normtwo/normone << "\n";
        }

        DVector logDelta_onehalo_array;
        DVector logDelta_twohalo_array;
        DVector logDelta_full_array;
        for(const auto logk : logk_array){
          const double k = std::exp(logk);

          // Integrate up one and two halo integrals
          double onehalo = 0.0;
          double twohalo = 0.0;
          double full = 0.0;
          for(const auto lognu : lognu_array){
            onehalo += integrand_onehalo(lognu, logk);
            twohalo += integrand_twohalo(lognu, logk);
          }
          onehalo *= (lognu_array[1]-lognu_array[0]);
          twohalo *= (lognu_array[1]-lognu_array[0]);

          // Normalize
          onehalo *= (k*k*k)/(2.0*M_PI*M_PI) / normone;
          twohalo *= 1.0 / normtwo;
          twohalo = twohalo * twohalo * std::exp(logDeltaLin_of_logk(logk));
          full = onehalo + twohalo;

          logDelta_onehalo_array.push_back(std::log(onehalo));
          logDelta_twohalo_array.push_back(std::log(twohalo));
          logDelta_full_array.push_back(std::log(full));

        }

        logDelta_onehalo_of_logk_spline = Spline(logk_array, logDelta_onehalo_array, "logDelta1h(logk)");
        logDelta_twohalo_of_logk_spline = Spline(logk_array, logDelta_twohalo_array, "logDelta2h(logk)");
        logDelta_full_of_logk_spline = Spline(logk_array, logDelta_full_array, "logDelta(logk)");
      }

      //================================================
      // Computes the formation redshift
      // D(zf)/D(z) sigma(M * massfrac) = deltac
      // where the fiducial choice is massfrac = 0.01.
      // If zf < z then we put zf=z
      // This is used in the c(M) relation
      // c(M) = cmin (1+zf(M))/(1+zcollapse) used in Bullock 2001
      //================================================
      void compute_formation_redshift_factor(
          const Spline & growthfactor_of_x_spline,
          const Spline & logsigma_of_logR_spline,
          Spline & formationredshift_of_logM_spline,
          double deltac,
          double OmegaM,
          double xcollapse,
          double massfraction){

        // Solve D(zf)/D(z) sigma(massfraction * M, z) == deltac(z) for zf
        // where z is collapse redshift
        auto compute_formation_redshift = [&](double M){
          const double Mform = massfraction * M;
          const double R = RofM(Mform, OmegaM, xcollapse);
          auto condition = [&](double x){
            const double xminspline = growthfactor_of_x_spline.get_xrange().first;
            double growthratio = growthfactor_of_x_spline(x) / growthfactor_of_x_spline(xcollapse);
            if(x < xminspline){
              growthratio = growthfactor_of_x_spline(xminspline) / growthfactor_of_x_spline(xcollapse);
              growthratio *= std::exp(x-xminspline);
            }
            const double sigma = std::exp(logsigma_of_logR_spline(std::log(R)));
            return growthratio * sigma - deltac;
          };
          double xformation = xcollapse;
          try{
            xformation = FML::MATH::find_root_bisection(condition, {std::log(1e-10), xcollapse}, 1e-8);
          } catch(...){
          }
          return xformation;
        };

        //========================================
        // Make R array
        //========================================
        const double Rmin = 1e-8, Rmax = 1e8;
        const int npts_R = 10000;
        const auto logR_array = FML::MATH::linspace(std::log(Rmin), std::log(Rmax), npts_R);

        // Compute formation-factor (1+zf(m))/(1+zcollapse) for all radii
        DVector formationtime_array;
        DVector logM_array;
        for(const auto logR : logR_array){
          const double M = MofR(std::exp(logR), OmegaM, xcollapse);
          const double xformation = compute_formation_redshift(M);
          formationtime_array.push_back(std::exp(-xformation)-1.0);
          logM_array.push_back(std::log(M));
        }
        formationredshift_of_logM_spline = Spline(logM_array, formationtime_array, "zf(logM)");
      }

      //===========================================
      // Constructor
      //===========================================
      HaloModel::HaloModel(
          SphericalCollapseModel spcmodel, 
          Spline logDelta_of_logk_spline,
          double xinput_pofk, 
          bool hmcode,
          bool verbose,
          double xini, 
          int npts) :
        spcmodel(spcmodel),
        OmegaM(spcmodel.OmegaM),
        logDelta_of_logk_spline(logDelta_of_logk_spline),
        xinput_pofk(xinput_pofk),
        hmcode(hmcode),
        verbose(verbose),
        xini_spherical_collapse(xini),
        npts_spherical_collapse(npts) {
          init(); 
        }

      //===========================================
      // Halo massfunction modelling
      // Get the fiducial option that is ST
      //===========================================
      HaloMultiplicityFunction get_sheth_tormen_halo_multiplicity_function(){
        return [&](double nu, HaloModel *hm){
          const double A_ST = hm->A_ST;
          const double p_ST = hm->p_ST;
          const double a_ST = hm->a_ST;
          return A_ST * (1.0 + std::pow(a_ST*nu*nu,-p_ST)) * std::exp(-a_ST*nu*nu/2.0);
        };
      }
      HaloBiasFunction get_sheth_tormen_halo_bias_function(){
        return [&](double nu, HaloModel *hm){
          const double deltac = hm->deltac;
          const double p_ST = hm->p_ST;
          const double a_ST = hm->a_ST;
          return 1.0 + (a_ST*nu*nu-1.0)/deltac + 2.0*p_ST/deltac/(1.0+std::pow(a_ST*nu*nu,p_ST));
        };
      }

      //===========================================
      // Fourier smoothing filter
      //===========================================
      WindowFunction get_tophat_window_fourier(){
        return [](double kR){
          if(kR < 1e-6) return 1.0 - kR*kR/10.;
          return 3.0/(kR*kR*kR)*(std::sin(kR)-kR*std::cos(kR));
        };
      }

      //=====================================================
      // Initialize a run. Runs in the constructor
      //=====================================================
      void HaloModel::init(){

        /*
        // Make a linear logR-array we can use (to ensure we have range and accuracy set just one place)
        const double Rmin = 1e-8, Rmax = 1e5;
        const int npts_R = 150 * _ACCURACY_BOOST;
        DVector logR_array = FML::MATH::linspace(std::log(Rmin), std::log(Rmax), npts_R);
        fiducial_logR_array = std::make_shared<DVector>(logR_array);

        // Make a linear logk-array we can use (to ensure we have range and accuracy set just one place)
        const double kmin = 1e-5, kmax = 1e6;
        const int npts_k = 150 * _ACCURACY_BOOST;
        DVector logk_array = FML::MATH::linspace(std::log(kmin), std::log(kmax), npts_k);
        fiducial_logk_array = std::make_shared<DVector>(logk_array);
        */

        if(verbose){
          std::cout << "\n#=====================================================\n";
          std::cout << "# Initializing the halomodel\n";
          std::cout << "#=====================================================\n";
          std::cout << "# OmegaM   " << OmegaM << "\n";
          std::cout << "# HMCode?  " << std::boolalpha << hmcode << "\n";
        }

        //=====================================================
        // Spherical collapse evolution
        //=====================================================
        if(not hmcode){
          if(verbose)
            std::cout << "Performing spherical collapse and computing growth\n";
          timer.StartTiming("SphericalCollapse");

          SphericalCollapse sc(
              spcmodel,
              verbose,
              npts_spherical_collapse,
              xini_spherical_collapse);

          sc.run_at_all_redshifts(
              deltac_of_x_spline,
              DeltaVir_of_x_spline,
              xta_of_x_spline,
              xnl_of_x_spline,
              xvir_of_x_spline,
              delta_ini_of_x_spline,
              growthfactor_of_x_spline,
              growthrate_of_x_spline);

          timer.EndTiming("SphericalCollapse");
        } else {
          if(verbose)
            std::cout << "Computing growth\n";
          FML::COSMOLOGY::SPHERICALCOLLAPSE::compute_growthfactor(
              spcmodel,
              growthfactor_of_x_spline,
              growthrate_of_x_spline);
        }

      }

      //=====================================================
      // Compute everything at a given redshift, i.e.
      // fills all the splines that depend on redshift
      // with the correct data
      //=====================================================
      void HaloModel::compute_at_redshift(double zcollapse){
        xcollapse = std::log(1.0/(1.0 + zcollapse));

        if(verbose){
          std::cout << "\n#=====================================================\n";
          std::cout << "# Compute at z = " << zcollapse << "\n";
          std::cout << "#=====================================================\n";
        }

        //=====================================================
        // Functions related to the variance of the smoothed
        // density field: smoothing filter and P(k)
        //=====================================================
        const double pofk_scaling_factor = std::pow(growthfactor_of_x_spline(xcollapse) / growthfactor_of_x_spline(xinput_pofk), 2);
        const Func1 logDeltaLin_of_logk = [&](double logk){
          return logDelta_of_logk_spline(logk) + std::log(pofk_scaling_factor);
        };

        //=====================================================
        // Density field variance when smoothed: sigma(logR)
        //=====================================================
        timer.StartTiming("sigma");
        if(verbose){
          std::cout << "Computing sigmas\n";
        }
        compute_sigma(
            logDeltaLin_of_logk,
            fourier_window_function,
            logsigma_of_logR_spline,
            fiducial_logR_array.get(),
            fiducial_logk_array.get());
        timer.EndTiming("sigma");

        //=====================================================
        // Set OmegaM and sigma8 at current redshift
        //=====================================================
        sigma8    = std::exp(logsigma_of_logR_spline(std::log(8.0)));
        sigmav    = compute_sigmav(logDeltaLin_of_logk, fourier_window_function, 0.0);
        sigmav100 = compute_sigmav(logDeltaLin_of_logk, fourier_window_function, 100.0);

        //=====================================================
        // Extract deltac and DeltaVir
        // If HMCode we use their fits
        //=====================================================
        if(verbose){
          std::cout << "Computing DeltaVir and deltac\n";
        }
        if(hmcode){
          DeltaVir     = get_DeltaVir_hmcode(xcollapse);
          deltac       = get_deltac_hmcode(xcollapse);
        } else {
          DeltaVir = DeltaVir_of_x_spline(xcollapse);
          deltac   = deltac_of_x_spline(xcollapse);
        }
        DeltaVir *= DeltaVir_multiplier;
        deltac *= deltac_multiplier;

        //=====================================================
        // nu(R) relation and the non-linear scale nu(R)==1
        //=====================================================
        if(verbose){
          std::cout << "Computing nu-R relation\n";
        }
        timer.StartTiming("nu");
        compute_lognu_of_logR(
            deltac, 
            logsigma_of_logR_spline, 
            lognu_of_logR_spline, 
            logR_of_lognu_spline,
            fiducial_logR_array.get());
        timer.EndTiming("nu");

        auto neff_data = compute_neff(logsigma_of_logR_spline, lognu_of_logR_spline);
        rnl_nu = neff_data.first;
        neff = neff_data.second;

        //=====================================================
        // Set the HMCode parameters
        //=====================================================
        if(hmcode){
          if(verbose)
            std::cout << "# Applying HMCode fits\n";
          cofM_model   = "Bullock2001";
          hmcode_cmin  = get_cmin_hmcode(xcollapse);
          hmcode_f     = get_f_hmcode(xcollapse);
          hmcode_kstar = get_kstar_hmcode(xcollapse);
          hmcode_eta   = get_eta_hmcode(xcollapse);
          hmcode_alpha = get_alpha_hmcode();
        }
        hmcode_cmin *= cofM_multiplier;
        cofM_0 *= cofM_multiplier;

        if(verbose) {
          std::cout << "# f        = " << hmcode_f << "\n";
          std::cout << "# kstar    = " << hmcode_kstar << "\n";
          std::cout << "# eta      = " << hmcode_eta << "\n";
          std::cout << "# alpha    = " << hmcode_alpha << "\n";
          std::cout << "# cmin     = " << hmcode_cmin << "\n";
          std::cout << "# deltac   = " << deltac << "\n";
          std::cout << "# DeltaVir = " << DeltaVir << "\n";
        }

        //=====================================================
        // The mass consentration relation as in Bullock 2001
        // needed if we want this NFW c(M,z) relation
        //=====================================================
        if(verbose)
          std::cout << "Computing formation redshifts\n";
        compute_formation_redshift_factor(
            growthfactor_of_x_spline,
            logsigma_of_logR_spline,
            formationredshift_of_logM_spline,
            deltac,
            OmegaM,
            xcollapse);

        //=====================================================
        // The halo density profile
        // The dimensionless combination 4pi r^3 rho(r,M) / M
        //=====================================================
        const Func2 rho_function = [&](double r, double M){
          return halo_density_profile(r, M, this);
        };

        //=====================================================
        // Fourier transform of halo density profiles
        //=====================================================
        if(verbose){
          std::cout << "Computing fourier transform of rho\n";
        }
        timer.StartTiming("rho");
        compute_rho_fourier(
            rho_function,
            y_of_logR_and_logk_spline,
            DeltaVir, 
            OmegaM, 
            xcollapse,
            fiducial_logR_array.get(),
            fiducial_logk_array.get());
        timer.EndTiming("rho");

        //=====================================================
        // Functions related to the mass-function
        //=====================================================
        const Func1 pdf_of_nu = [&](double nu){ 
          return halo_multiplicity_function(nu, this); 
        };
        const Func1 bias_of_nu = [&](double nu){ 
          return halo_bias_function(nu, this); 
        };

        //=====================================================
        // Prediction for the mass-function
        //=====================================================
        if(verbose){
          std::cout << "Computing massfunction\n";
        }
        timer.StartTiming("n");
        compute_massfunction(
            logR_of_lognu_spline, 
            lognu_of_logR_spline, 
            dndlogM_of_logM_spline,
            n_of_logM_spline,
            pdf_of_nu,
            OmegaM, 
            xcollapse);
        timer.EndTiming("n");

        //=====================================================
        // Prediction for the power-spectrum
        //=====================================================
        if(verbose){
          std::cout << "Computing halomodel pofk\n";
        }
        timer.StartTiming("Pofk");
        compute_one_and_two_halo_terms(
            logR_of_lognu_spline, 
            y_of_logR_and_logk_spline, 
            logDeltaHM_onehalo_of_logk_spline,
            logDeltaHM_twohalo_of_logk_spline,
            logDeltaHM_full_of_logk_spline,
            logDeltaLin_of_logk,
            pdf_of_nu,
            bias_of_nu,
            hmcode_eta,
            kmin_pofk,
            kmax_pofk,
            npts_pofk);
        timer.EndTiming("Pofk");

        //=====================================================
        // Process these in case of modifications
        //=====================================================
        if(hmcode or hmcode_f != 0.0 or hmcode_kstar > 0.0 or hmcode_alpha != 1.0){
          auto logk_array = logDeltaHM_onehalo_of_logk_spline.get_x_data();
          DVector logDeltaHM_onehalo_of_logk, logDeltaHM_twohalo_of_logk, logDeltaHM_full_of_logk;
          if(verbose)
            std::cout << "Adjusting haloterms: kstar = " << hmcode_kstar << " alpha = " << hmcode_alpha << " f = " << hmcode_f << "\n";
          for(auto logk : logk_array){
            const double k = std::exp(logk);
            double Delta_onehalo = std::exp(logDeltaHM_onehalo_of_logk_spline(logk));
            double Delta_twohalo = std::exp(logDeltaHM_twohalo_of_logk_spline(logk));
            double Delta_full    = std::exp(logDeltaHM_full_of_logk_spline(logk));
            double Delta_lin     = std::exp(logDeltaLin_of_logk(logk));

            // In HMCode we just use the linear power-spectrum as the twohalo term
            if(hmcode)
              Delta_twohalo = Delta_lin;

            // (BAO) damping of twohalo term
            if(hmcode_f != 0.0){
              Delta_twohalo *= std::max(1e-12,(1.0 - hmcode_f * std::pow( std::tanh(k * sigmav / std::sqrt(hmcode_f)), 2) ));
            }

            // Damping of onehalo term at low k
            if(hmcode_kstar > 0.0){
              double k_over_kstar = std::min(k/hmcode_kstar, 20.0);
              double factor = k_over_kstar < 1e-5 ? k_over_kstar*k_over_kstar : 1.0 - std::exp(-k_over_kstar*k_over_kstar);
              Delta_onehalo *= factor;
            }

            // Set a minimum value as we are going to spline the log of this shit
            if(Delta_onehalo < 1e-30) Delta_onehalo = 1e-30;
            if(Delta_twohalo < 1e-30) Delta_twohalo = 1e-30;
            if(Delta_full< 1e-30)     Delta_full = 1e-30;

            // Combine two terms using an l_alpha norm
            if(hmcode_alpha != 1.0)
              Delta_full = std::pow(std::pow(Delta_onehalo, hmcode_alpha) + std::pow(Delta_twohalo, hmcode_alpha), 1.0/hmcode_alpha);
            else
              Delta_full = Delta_onehalo + Delta_twohalo;

            logDeltaHM_onehalo_of_logk.push_back(std::log(Delta_onehalo));
            logDeltaHM_twohalo_of_logk.push_back(std::log(Delta_twohalo));
            logDeltaHM_full_of_logk.push_back(std::log(Delta_full));
          }

          // Remake splines
          logDeltaHM_onehalo_of_logk_spline = Spline(logk_array, logDeltaHM_onehalo_of_logk, "logDelta1h(logk)");
          logDeltaHM_twohalo_of_logk_spline = Spline(logk_array, logDeltaHM_twohalo_of_logk, "logDelta2h(logk)");
          logDeltaHM_full_of_logk_spline    = Spline(logk_array, logDeltaHM_full_of_logk, "logDelta(logk)");
        }

      }

      //=====================================================
      // c(M) relation for NFW halos with options of using
      // Bullock 2001 or simple fit
      //=====================================================
      cofMFunction get_nfw_halo_mass_concentration_relation(){
        return [&](double M, HaloModel *hm){
          if(hm->cofM_model == "Bullock2001"){
            const double zf = hm->formationredshift_of_logM_spline(std::log(M));
            const double z  = std::exp(-hm->xcollapse)-1;
            return hm->hmcode_cmin * std::max(1.0, (1.0+zf)/(1+z) );
          }
          // Simple c = c0/(1+z) * (M/M0)^(-beta) power-law model
          return hm->cofM_0 * std::pow(M / hm->cofM_M0, -hm->cofM_beta) * std::exp(hm->xcollapse);
        };
      }

      //=====================================================
      // The density profile 4pi r^3 rho(r,M)/M for NFW halos
      // with option of applying the HMCode modifications
      //=====================================================
      RhoFunction get_nfw_halo_density_profile(){
        return [&](double r, double M, HaloModel *hm){
          const double DeltaVir = hm->DeltaVir;
          const double OmegaM = hm->OmegaM;
          const double x = hm->xcollapse;
          const double c = hm->halo_mass_concentration_relation(M,hm);
          const double R = RofM(M, OmegaM, x);
          double Rvir = R / std::pow(DeltaVir,1.0/3.0);

          // The modification Rvir *= nu^eta is for the halo bloating effect
          // Fiducial value is 0.0 which has no effect
          // This can be included *either* here or when doing the 1h 2h integrals 
          // Rvir *= std::pow(std::exp(hm->lognu_of_logR_spline(std::log(R))), hm->hmcode_eta);
          // but not both. Currently we do this in the y-integral

          const double rs = Rvir / c;
          if(r > Rvir) return 0.0;
          return std::pow(r/rs, 3) / ( (r/rs) * (1 + r/rs) * (1 + r/rs) ) / (std::log(1 + c) - c/(1 + c));
        };
      }

      void HaloModel::output_deltac(
          std::string filename,
          double zmin,
          double zmax,
          int npts_z) const {

        const auto x_array = FML::MATH::linspace(std::log(1.0/(1.0+zmax)), std::log(1.0/(1.0+zmin)), npts_z);;
        std::ofstream fp(filename);

        if(hmcode){

          fp << "#  z       deltaC(z)      DeltaVir(z)     kstar(z)    fdamp(z)     eta(z)     cmin(z) [HMCode]\n";
          for(const auto x : x_array){
            const double z = std::exp(-x)-1;
            fp << std::setw(10) << z                      << "  ";
            fp << std::setw(10) << get_deltac_hmcode(x)   << "  ";
            fp << std::setw(10) << get_DeltaVir_hmcode(x) << "  ";
            fp << std::setw(10) << get_kstar_hmcode(x)    << "  ";
            fp << std::setw(10) << get_f_hmcode(x)        << "  ";
            fp << std::setw(10) << get_eta_hmcode(x)      << "  ";
            fp << std::setw(10) << get_cmin_hmcode(x)     << "  ";
            fp << "\n";

          } 

        } else {

          fp << "#  z     deltaC(z)      DeltaVir(z)\n";
          for(const auto x : x_array){
            const double z = std::exp(-x)-1;
            fp << std::setw(10) << z                       << "  ";
            fp << std::setw(10) << deltac_of_x_spline(x)   << "  ";
            fp << std::setw(10) << DeltaVir_of_x_spline(x) << "  ";
            fp << "\n";
          }

        }
      }

      //=====================================================
      // Output dndlogM over whole range computed
      //=====================================================
      void HaloModel::output_nofM(
          std::string filename,
          double Mmin,
          double Mmax,
          int npts_M) const {
        const auto logM_array = FML::MATH::linspace(std::log(Mmin), std::log(Mmax), npts_M);;
        std::ofstream fp(filename);
        fp << "#  M (Msun/h)    dlogndlogM      n (h/Mpc)^3      R (Mpc/h)    Rvir (Mpc/h)   nu  [z = " << std::to_string(std::exp(-xcollapse)-1) << "]\n";
        for(const auto logM : logM_array){
          const double M = std::exp(logM);
          const double R = RofM(M, this->OmegaM, xcollapse);
          const double Rvir = RvirofM(M, DeltaVir, this->OmegaM, xcollapse);
          const double logR = std::log(R);
          const double lognu = lognu_of_logR_spline(logR);
          const double nu = std::exp(lognu);
          const double dlogndlogM = dndlogM_of_logM_spline(logM);
          const double n = n_of_logM_spline(logM);
          fp << std::setw(10) << M          << "  ";
          fp << std::setw(10) << dlogndlogM << "  ";
          fp << std::setw(10) << n          << "  ";
          fp << std::setw(10) << R          << "  ";
          fp << std::setw(10) << Rvir       << "  ";
          fp << std::setw(10) << nu         << "  ";
          fp << "\n";
        }
      }

      //=====================================================
      // Output P(k) over whole range computed
      //=====================================================
      void HaloModel::output_pofk(
          std::string filename,
          double kmin,
          double kmax,
          int npts_k) const {

#ifdef HMCODETESTING
        Spline logDeltaNL_of_logk_spline;
        read_pofk_file("pofk_nl_hmcode.txt",logDeltaNL_of_logk_spline);
#endif

        //const auto logk_array = logDeltaHM_full_of_logk_spline.get_x_data();
        const auto logk_array = FML::MATH::linspace(std::log(kmin),std::log(kmax), npts_k);
        std::ofstream fp(filename);
        fp << "#  k (h/Mpc)     P_NL(k)     P_Lin(k)     P_1h(k)   P_2h(k)  (Mpc/h)^3  [z = " << std::to_string(std::exp(-xcollapse)-1) << "]\n";
        for(const auto logk : logk_array){
          const double k = std::exp(logk);
          const double Delta_over_P = 2.0 * M_PI * M_PI / (k*k*k);
          const double scaling = std::pow(growthfactor_of_x_spline(xcollapse) / growthfactor_of_x_spline(xinput_pofk), 2);
          fp << std::setw(10) << k << " ";
          fp << std::setw(10) << Delta_over_P * std::exp(logDeltaHM_full_of_logk_spline(logk))    << " ";
          fp << std::setw(10) << Delta_over_P * std::exp(logDelta_of_logk_spline(logk)) * scaling << " ";
          fp << std::setw(10) << Delta_over_P * std::exp(logDeltaHM_onehalo_of_logk_spline(logk)) << " ";
          fp << std::setw(10) << Delta_over_P * std::exp(logDeltaHM_twohalo_of_logk_spline(logk)) << " ";
#ifdef HMCODETESTING
          fp << std::exp(logDeltaHM_full_of_logk_spline(logk))/std::exp(logDeltaNL_of_logk_spline(logk)) << " ";
#endif
          fp << "\n";
        }
      }

      //=====================================================
      // Show some info about what we have computed
      //=====================================================
      void HaloModel::info() const {
        const bool extraverbose = true;

        std::cout << "\n#=====================================================\n";
        std::cout << "# Halomodel info ( z = " << (isnan(xcollapse) ?  " not yet set " : std::to_string(std::exp(-xcollapse)-1.0)) << " )\n";
        std::cout << "#=====================================================\n";
        std::cout << "# OmegaM            : " << OmegaM << "\n";
        std::cout << "# fnu               : " << fnu << "\n";
        std::cout << "# DeltaVirModifier  : " << DeltaVir_multiplier << "\n";
        std::cout << "# cofMModifier      : " << cofM_multiplier << "\n";
        std::cout << "# deltacModifier    : " << deltac_multiplier << "\n";
        std::cout << "# Massfunction      : " << halo_massfunction << "\n";
        if(halo_massfunction == "ShethTormen1999")
          std::cout << "#   with a : " << a_ST << " p : " << p_ST << "\n";
        std::cout << "# Density profile   : " << halo_profile << "\n";
        if(halo_profile == "NFW")
          std::cout << "#   with c(M)     : " << cofM_model << "\n";
        if(hmcode)
          std::cout << "# Running with HMCode settings\n";

        if(isnan(xcollapse)){
          // Compute sigma8 to be able to output it for debug purposes
          const Func1 logDeltaLin_of_logk = [&](double logk){
            return logDelta_of_logk_spline(logk);
          };
          Spline _logsigma_of_logR_spline;
          compute_sigma(
              logDeltaLin_of_logk,
              fourier_window_function,
              _logsigma_of_logR_spline,
              fiducial_logR_array.get(),
              fiducial_logk_array.get());
          double sigma8ic = std::exp(_logsigma_of_logR_spline(std::log(8.0)));
          std::cout << "# sigma8 of provided Plin at z = " << std::exp(-xinput_pofk)-1 << " is " << sigma8ic << "\n";
          std::cout << "# NB: compute_at_redshift has not yet been run so not much more computed\n";
          return;
        }

        std::cout << "# Current z         : " << std::exp(-xcollapse)-1.0 << "\n";
        std::cout << "# deltaC            : " << deltac   << "\n";
        std::cout << "# DeltaVir          : " << DeltaVir << "\n";

        if(xta_of_x_spline and xvir_of_x_spline and xnl_of_x_spline){
          std::cout << "# znl               : " << std::exp(-xnl_of_x_spline(xcollapse))-1.0       << "\n";
          std::cout << "# zturnaround       : " << std::exp(-xta_of_x_spline(xcollapse))-1.0       << "\n";
          std::cout << "# zvir              : " << std::exp(-xvir_of_x_spline(xcollapse))-1.0      << "\n";
        }

        if(logsigma_of_logR_spline)
          std::cout << "# sigma8            : " << std::exp(logsigma_of_logR_spline(std::log(8.0))) << "\n";
        std::cout << "# sigmav            : " << sigmav     << " (Mpc/h)\n";
        std::cout << "# sigmav100         : " << sigmav100  << " (Mpc/h)\n";
        std::cout << "# rnl_nu            : " << rnl_nu     << " (Mpc/h)\n";
        std::cout << "# knl_nu            : " << 1.0/rnl_nu << " (h/Mpc)\n";
        std::cout << "# neff              : " << neff       << "\n";

        if(extraverbose){
          spcmodel.info(); 
          timer.PrintAllTimings();
        }
      }

      //=====================================================
      // HMCode fits
      //=====================================================
      double HaloModel::get_deltac_hmcode(double x) const{
        double Dofx_over_Dofx_sigma = growthfactor_of_x_spline(x)/growthfactor_of_x_spline(xcollapse);
        double sigma8ofx = sigma8 * Dofx_over_Dofx_sigma;
        double OmegaMofx = spcmodel.OmegaMofx(x);
        double deltacofx = 1.59 + 0.0314*std::log(sigma8ofx);
        deltacofx *= 1.0 + 0.0123*std::log10(OmegaMofx);
        deltacofx *= (1.+0.262*fnu); 
        return deltacofx;
      }

      double HaloModel::get_DeltaVir_hmcode(double x) const{
        double DeltaVirofx = 418.0 * std::pow(spcmodel.OmegaMofx(x),-0.352);
        DeltaVirofx *= (1.+0.916*fnu);
        return DeltaVirofx;
      }

      double HaloModel::get_eta_hmcode(double x) const{
        double Dofx_over_Dofx_sigma = growthfactor_of_x_spline(x)/growthfactor_of_x_spline(xcollapse);
        double sigma8ofx = sigma8 * Dofx_over_Dofx_sigma;
        double etaofx = 0.603-0.3*sigma8ofx;
        return etaofx;
      }

      double HaloModel::get_f_hmcode(double x) const{
        double Dofx_over_Dofx_sigma = growthfactor_of_x_spline(x)/growthfactor_of_x_spline(xcollapse);
        double sigmav100ofx = sigmav100 * Dofx_over_Dofx_sigma;
        double fofx = std::max(std::min(0.0095*std::pow(sigmav100ofx, 1.37), 0.99), 1e-3);
        return fofx;
      }

      double HaloModel::get_kstar_hmcode(double x) const{
        double Dofx_over_Dofx_sigma = growthfactor_of_x_spline(x)/growthfactor_of_x_spline(xcollapse);
        double sigmavofx = sigmav * Dofx_over_Dofx_sigma;
        double kstarofx = 0.584/sigmavofx;
        return kstarofx;
      }

      double HaloModel::get_cmin_hmcode([[maybe_unused]] double x) const{
        return 3.13;
      }

      double HaloModel::get_alpha_hmcode() const{
        // NB: this is only at the current redshift
        return 3.24*std::pow(1.85,neff);
      }

    }
  }
}
