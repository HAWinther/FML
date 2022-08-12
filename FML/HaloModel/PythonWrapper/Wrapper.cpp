#include <FML/Spline/Spline.h>
#include <FML/HaloModel/Halomodel.h>

using SphericalCollapseModel = FML::COSMOLOGY::SPHERICALCOLLAPSE::SphericalCollapseModel;
using HaloModel = FML::COSMOLOGY::HALOMODEL::HaloModel;
using Func1 = FML::COSMOLOGY::HALOMODEL::Func1;
using Spline = FML::INTERPOLATION::SPLINE::Spline;
using DVector = FML::SOLVERS::ODESOLVER::DVector;

// Allocate memory
HaloModel **get(){
  HaloModel **hm = new HaloModel*(); 
  *hm = new HaloModel();
  return hm;
}

void get_deltac(HaloModel **hmp, int nz, double *z, int ndeltac, double *deltac){
  HaloModel *hm = *hmp;
  for(int i = 0; i < nz; i++){
    double x = std::log(1.0/(1+z[i]));
    deltac[i] = hm->hmcode ? hm->get_deltac_hmcode(x) : hm->deltac_of_x_spline(x); 
  }
}

void get_DeltaVir(HaloModel **hmp, int nz, double *z, int nDeltaVir, double *DeltaVir){
  HaloModel *hm = *hmp;
  for(int i = 0; i < nz; i++){
    double x = std::log(1.0/(1+z[i]));
    DeltaVir[i] = hm->hmcode ? hm->get_DeltaVir_hmcode(x) : hm->DeltaVir_of_x_spline(x); 
  }
}

// Free up memory
void free(HaloModel **hm){
  delete *hm;
}

// Fetch computed P(k,z) and return it
void get_pofk(HaloModel **hmp, int nk, double * k, int npofk_lin, double * pofk_lin, int npofk, double *pofk){
  HaloModel *hm = *hmp;
  if(nk != npofk_lin or nk != npofk)
    throw std::runtime_error("Arrays k and pofk must be compatible\n");
  for(int i = 0; i < nk; i++){
    const double factor = (2.0*M_PI*M_PI)/std::pow(k[i],3);
    const double scaling = std::pow(hm->growthfactor_of_x_spline(hm->xcollapse) / hm->growthfactor_of_x_spline(hm->xinput_pofk), 2);
    pofk[i]     = std::exp(hm->logDeltaHM_full_of_logk_spline(std::log(k[i])))*factor;
    pofk_lin[i] = scaling * std::exp(hm->logDelta_of_logk_spline(std::log(k[i])))*factor;
  }
}

void get_pofk_1h_2h(HaloModel **hmp, int nk, double * k, int npofk_1h, double * pofk_1h, int npofk_2h, double *pofk_2h){
  HaloModel *hm = *hmp;
  if(nk != npofk_1h or nk != npofk_2h)
    throw std::runtime_error("Arrays k and pofk must be compatible\n");
  for(int i = 0; i < nk; i++){
    const double factor = (2.0*M_PI*M_PI)/std::pow(k[i],3);
    pofk_1h[i]     = std::exp(hm->logDeltaHM_onehalo_of_logk_spline(std::log(k[i])))*factor;
    pofk_2h[i]     = std::exp(hm->logDeltaHM_twohalo_of_logk_spline(std::log(k[i])))*factor;
  }
}

// Fetch massfunction dlogndlogM and n
void get_nofM(HaloModel **hmp, int nm, double * M, int ndnofm, double *dnofm, int nnofm, double * nofm){
  HaloModel *hm = *hmp;
  if(nm != nnofm or nm != ndnofm)
    throw std::runtime_error("Arrays M and nofM must be compatible\n");
  for(int i = 0; i < nm; i++){
    dnofm[i] = hm->dndlogM_of_logM_spline(std::log(M[i]));
    nofm[i]  = hm->n_of_logM_spline(std::log(M[i]));
  }
}

// Compute P(k,z) 
void calc(HaloModel **hmp, double z){
  HaloModel *hm = *hmp;
  const double zcollapse = std::exp(-hm->xcollapse)-1.0;
  if(std::fabs(z-zcollapse) < 1e-8)
    return;
  hm->compute_at_redshift(z);
}

// Show some info
void info(HaloModel **hmp){
  HaloModel *hm = *hmp;
  hm->info();
}

// Output to file
void output(HaloModel **hmp, std::string label){
  HaloModel *hm = *hmp;
  const double z = std::exp(-hm->xcollapse)-1.0;
  hm->output_pofk("pofk_" + label + "_z" + std::to_string(z) + ".txt");
  hm->output_nofM("nofM_" + label + "_z" + std::to_string(z) + ".txt");
  hm->output_deltac("sph_" + label + "_z" + std::to_string(z) + ".txt");
}

void init(
    HaloModel **hmp, 
    std::string filename_pofk, 
    double OmegaM,
    double w0,
    double wa,
    double mu0,
    double mua,
    bool verbose,
    bool hmcode,
    bool sigma8norm,
    double deltac_multiplier,
    double DeltaVir_multiplier,
    double cofM_multiplier){

  HaloModel *hm = *hmp;
  const double OmegaLambda = 1.0-OmegaM;
  const double xini_spherical_collapse = std::log(1e-5);
  const double xpofk = 0.0;
  const int npts_spherical_collapse = 1000;

  // Cosmology functions E = H(x)/H0, x = log(a)
  const Func1 Eofx = [OmegaM,OmegaLambda,w0,wa](double x) -> double { 
    const double a = std::exp(x);
    return std::sqrt( 
        OmegaM * std::exp(-3.0*x) + 
        OmegaLambda * std::exp(3.0 * wa * (a - 1) - 3 * (1 + w0 + wa) * x)
        ); 
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
  const Func1 muofx_lcdm = []([[maybe_unused]] double x) -> double { 
    return 1.0;
  };
  const Func1 wofx = [w0,wa]([[maybe_unused]] double x) -> double { 
    const double a = std::exp(x);
    return w0 + wa*(1-a);
  };
  const Func1 OmegaMofx = [OmegaM,Eofx](double x) -> double { 
    const double E = Eofx(x);
    return OmegaM*std::exp(-3*x)/(E*E); 
  };

  // Set up the model
  SphericalCollapseModel model(
      Eofx,
      OmegaMofx,
      logEprimeofx,
      muofx,
      wofx);
  
  SphericalCollapseModel lcdmmodel(
      Eofx,
      OmegaMofx,
      logEprimeofx,
      muofx_lcdm,
      wofx);

  // We read in P(k,z=zi) in LCDM and then scale it back
  // and forward again to get the linear P(k,z=zi) of our mu-sims
  Spline Dofx_spline, fofx_spline;
  Spline Dlcdmofx_spline, flcdmofx_spline;
  FML::COSMOLOGY::SPHERICALCOLLAPSE::compute_growthfactor(model, Dofx_spline, fofx_spline);
  FML::COSMOLOGY::SPHERICALCOLLAPSE::compute_growthfactor(lcdmmodel, Dlcdmofx_spline, flcdmofx_spline);
  const double xlinearpofk = 0.0;
  const double xscaleback = std::log(1.0/50.0);
  const double pofk_normalization = (sigma8norm ? 1.0 :  
      std::pow(Dlcdmofx_spline(xscaleback) / Dlcdmofx_spline(xlinearpofk), 2) /
      std::pow(Dofx_spline(xscaleback) / Dofx_spline(xlinearpofk), 2));
  Spline logDelta_of_logk_spline;
  FML::COSMOLOGY::HALOMODEL::read_pofk_file(
      filename_pofk,
      logDelta_of_logk_spline,
      pofk_normalization);

  // Set cosmology and Plin
  hm->spcmodel = model;
  hm->OmegaM = OmegaM;
  hm->fnu = 0.0;
  hm->logDelta_of_logk_spline = logDelta_of_logk_spline;
  hm->xinput_pofk = xpofk;
  hm->xini_spherical_collapse = xini_spherical_collapse;
  hm->npts_spherical_collapse = npts_spherical_collapse;
  hm->hmcode = hmcode;
  hm->verbose = verbose;
  
  // Initialize
  hm->init();
  hm->deltac_multiplier   = deltac_multiplier;
  hm->DeltaVir_multiplier = DeltaVir_multiplier;
  hm->cofM_multiplier     = cofM_multiplier;
}

//================================================
// Read a file with (a, mu(a)) and spline up mu(x)
//================================================
void read_and_spline_mu_file(
    const std::string filename_mu,
    Spline & mu_of_x_spline){

  const auto mudata = FML::FILEUTILS::read_regular_ascii(filename_mu, 2, {0,1}, 0);

  DVector xmu, mu;
  bool negativemu = false;
  for (auto & line : mudata) {
    xmu.push_back(std::log(line[0]));
    mu.push_back(line[1]);
    if(line[1] < 0.0) negativemu = true;
  }

  if(negativemu){
    std::cout << "Error: mu is negative, could be a problem\n";
    exit(1);
  }

  mu_of_x_spline = Spline(xmu,mu);
}
