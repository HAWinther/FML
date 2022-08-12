#include <FML/Math/Math.h>
#include <FML/SphericalCollapse/SphericalCollapse.h>
#include <FML/HaloModel/Halomodel.h>

using SphericalCollapseModel = FML::COSMOLOGY::SPHERICALCOLLAPSE::SphericalCollapseModel;
using Func1 = FML::COSMOLOGY::SPHERICALCOLLAPSE::Func1;
using HaloModel = FML::COSMOLOGY::HALOMODEL::HaloModel;
using Spline = FML::INTERPOLATION::SPLINE::Spline;

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
  const double OmegaM = 0.295;
  const double OmegaLambda = 1-OmegaM;
  const Func1 Eofx = [OmegaM,OmegaLambda,w0,wa](double x) -> double { 
    const double a = std::exp(x);
    return std::sqrt( OmegaM * std::exp(-3.0*x) + 
        OmegaLambda * std::exp(3.0 * wa * (a - 1) - 3 * (1 + w0 + wa) * x)); 
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
    return 1.0 + (mu0+mua*(1.0-a)) / (E*E); 
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
  // Fetch linear P(k,z=0)
  //===================================================
  const double pofk_normalization = 1.0;
  std::string filename_pofk = "pofk_lin.txt";
  const double xinputpofk = 0.0;
  Spline logDelta_of_logk_spline;
  FML::COSMOLOGY::HALOMODEL::read_pofk_file(
      filename_pofk,
      logDelta_of_logk_spline,
      pofk_normalization);

  FML::COSMOLOGY::HALOMODEL::_ACCURACY_BOOST = 1;

  //===================================================
  // Set up the halomodel
  //===================================================
  double z = 0.0;
  bool hmcode = true;
  std::string label = "testrun";
  HaloModel hm(
      scm,
      logDelta_of_logk_spline,
      xinputpofk,
      hmcode,
      verbose);
  hm.verbose = true;
  hm.compute_at_redshift(z);
 
  // Output to file
  hm.output_pofk  ( "pofk_" + label + "_z" + std::to_string(z) + ".txt");
  hm.output_nofM  ( "nofM_" + label + "_z" + std::to_string(z) + ".txt");
  hm.output_deltac( "sph_"  + label + "_z" + std::to_string(z) + ".txt");
  
  hm.info();
}


