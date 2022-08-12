#include <FML/Math/Math.h>
#include <FML/SphericalCollapse/SphericalCollapse.h>

using SphericalCollapseModel = FML::COSMOLOGY::SPHERICALCOLLAPSE::SphericalCollapseModel;
using SphericalCollapse = FML::COSMOLOGY::SPHERICALCOLLAPSE::SphericalCollapse;
using Spline = FML::INTERPOLATION::SPLINE::Spline;
using Func1 = FML::COSMOLOGY::SPHERICALCOLLAPSE::Func1;

int main(){
  const bool verbose = true;

  //===================================================
  // Define cosmological parameters and functions
  // Cosmology functions E = H(x)/H0, x = log(a)
  //===================================================
  const double w0 = -1.0;
  const double wa = 0.0;
  const double mu0 = 0.0;
  const double mua = 0.0;
  const double OmegaM = 0.3;
  const double OmegaLambda = 1-OmegaM;

  const Func1 Eofx = [OmegaM,OmegaLambda,w0,wa](double x) -> double { 
    const double a = std::exp(x);
    return std::sqrt( 
        OmegaM * std::exp(-3.0*x) + 
        OmegaLambda * std::exp(3.0 * wa * (a - 1) - 3 * (1 + w0 + wa) * x)
        ); 
  };
  const Func1 OmegaMofx = [OmegaM,Eofx](double x) -> double { 
    return OmegaM*std::exp(-3.0*x)/std::pow(Eofx(x),2);
  };
  const Func1 logEprimeofx = [OmegaM,OmegaLambda,w0,wa,Eofx](double x) -> double { 
    const double E = Eofx(x);    
    const double a = std::exp(x);   
    return 1.0 / (2.0 * E * E) * (-3.0 * OmegaM / (a * a * a) +
        OmegaLambda * std::exp(3.0 * wa * (a - 1) - 3 * (1 + w0 + wa) * x) *
        (3.0 * wa * a - 3.0 * (1 + w0 + wa)));
  };
  const Func1 muofx = [mu0,mua,Eofx](double x) -> double { 
    const double E = Eofx(x);
    const double a = std::exp(x);
    return 1.0 + (mu0 + mua*(1.0-a)) / (E*E); 
  };
  const Func1 wofx = [w0,wa](double x) -> double { 
    const double a = std::exp(x);
    return w0 + wa*(1-a); 
  };

  //===================================================
  // Set up the spherical collapse model
  //===================================================
  SphericalCollapseModel scm(
      Eofx, 
      OmegaMofx, 
      logEprimeofx,
      muofx,
      wofx);
  scm.info();

  //===================================================
  // Run spherical collapse for this model
  //===================================================
  SphericalCollapse sc(
      scm,
      verbose);
  Spline deltac_of_x_spline, 
         DeltaVir_of_x_spline, 
         xta_of_x_spline, 
         xvir_of_x_spline, 
         xnl_of_x_spline, 
         delta_ini_of_x_spline,
         growthfactor_of_x_spline, 
         growthrate_of_x_spline;
  sc.run_at_all_redshifts(
      deltac_of_x_spline,
      DeltaVir_of_x_spline,
      xta_of_x_spline,
      xnl_of_x_spline,
      xvir_of_x_spline,
      delta_ini_of_x_spline,
      growthfactor_of_x_spline,
      growthrate_of_x_spline);
}
