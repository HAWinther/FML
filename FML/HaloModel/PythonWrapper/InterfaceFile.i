%module HaloModelCXX
%include <std_string.i>
%include <std_shared_ptr.i>
%{
  #define SWIG_FILE_WITH_INIT
  #include <FML/HaloModel/Halomodel.h>

  using HaloModel = FML::COSMOLOGY::HALOMODEL::HaloModel;

  extern HaloModel **get();
  extern void calc(HaloModel **hm, double z);
  extern void free(HaloModel **hm);
  extern void info(HaloModel **hm);
  extern void init(
    HaloModel **hm, 
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
    double cofM_multiplier);
 extern void get_pofk(HaloModel **hm, int nk, double * k, int npofk_lin, double * pofk_lin, int npofk, double *pofk);
 extern void get_nofM(HaloModel **hmp, int nm, double * M, int ndnofm, double *dnofm, int nnofm, double * nofm);
 extern void get_DeltaVir(HaloModel **hmp, int nz, double *z, int nDeltaVir, double *DeltaVir);
 extern void get_deltac(HaloModel **hmp, int nz, double *z, int ndeltac, double *deltac);
 extern void get_pofk_1h_2h(HaloModel **hmp, int nk, double * k, int npofk_1h, double * pofk_1h, int npofk_2h, double *pofk_2h);
%}

#include <FML/HaloModel/Halomodel.h>

%include "numpy.i"
%init %{
import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) {(int nk, double *k), (int npofk_lin, double *pofk_lin), (int npofk, double *pofk)};
%apply (int DIM1, double* IN_ARRAY1) {(int nm, double *M), (int ndnofm, double *dnofm), (int nnofm, double *nofm)};
%apply (int DIM1, double* IN_ARRAY1) {(int nk, double *k), (int npofk_1h, double *pofk_1h), (int npofk_2h, double *pofk_2h)};
%apply (int DIM1, double* IN_ARRAY1) {(int nz, double *z), (int nDeltaVir, double *DeltaVir)};
%apply (int DIM1, double* IN_ARRAY1) {(int nz, double *z), (int ndeltac, double *deltac)};

extern HaloModel **get();
extern void calc(HaloModel **hm, double z);
extern void free(HaloModel **hm);
extern void info(HaloModel **hm);
extern void init(
    HaloModel **hm, 
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
    double cofM_multiplier);
extern void get_pofk(HaloModel **hm, int nk, double * k, int npofk_lin, double * pofk_lin, int npofk, double *pofk);
extern void get_nofM(HaloModel **hmp, int nm, double * M, int ndnofm, double *dnofm, int nnofm, double * nofm);
extern void get_DeltaVir(HaloModel **hmp, int nz, double *z, int nDeltaVir, double *DeltaVir);
extern void get_deltac(HaloModel **hmp, int nz, double *z, int ndeltac, double *deltac);
extern void get_pofk_1h_2h(HaloModel **hmp, int nk, double * k, int npofk_1h, double * pofk_1h, int npofk_2h, double *pofk_2h);
