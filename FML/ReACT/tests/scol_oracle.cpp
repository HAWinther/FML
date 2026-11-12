// Calls upstream ReACT's spherical collapse directly (SCOL::maxP_zeta and SCOL::myscol of ACTio-ReACTio a36dda02):
// the reference values of FML/tests/test_screened_collapse.cpp (FML's ScreenedSphericalCollapse.h).
//
// Build against the backend (FML/ReACT/build_backend.py) and the upstream source:
//   g++ -O2 -std=c++17 scol_oracle.cpp -I<ACTio-ReACTio>/reactions/src -I<build-dir>/deps/include \
//       -L<backend>/lib -lfml_react_backend -lgsl -lgslcblas -Wl,-rpath,<backend>/lib -o scol_oracle
// Usage: scol_oracle model a OmegaM Omega_nu R_th sigma2_8 dsigma2dR_8 parameter
//   model 1 GR, 2 Hu-Sawicki f(R) (parameter |fR0|), 3 nDGP (parameter Omega_rc); R_th in Mpc/h; sigma^2 and
//   d sigma^2 / dR at R = 8 Mpc/h at the collapse redshift
// Prints the most probable environment zeta, delta_c, a_vir and delta(a_vir).
//
// Upstream's delta_ini bracket for modified gravity is [1.1 zeta_i / a, 4.4 zeta_i / a] (zeta_i the initial
// environment): outside it delta_c is pinned at 1.1 zeta / a or 4.4 zeta / a (e.g. 5% low at z = 2.3 for
// sigma_8 = 0.8), so the reference uses z <= 1.
#include "SCOL.h"
#include "SpecialFunctions.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: %s model a OmegaM Omega_nu R_th sigma2_8 dsigma2dR_8 parameter\n", argv[0]);
        return 1;
    }
    const int model = atoi(argv[1]);
    const double a = atof(argv[2]), Om = atof(argv[3]), Onu = atof(argv[4]), Rth = atof(argv[5]);
    const double s1 = atof(argv[6]), s2 = atof(argv[7]), p0 = atof(argv[8]);
    double extpars[maxpars] = {0};
    extpars[0] = p0;
    double pars[8] = {a, Om, Onu, 0, 0, 30, 0, 0};
    IOW iow;
    iow.initnorm(pars, extpars, model); // the linear growth used for delta_c (Dl_spt)
    SCOL scol;
    double out[3];
    const double zeta = scol.maxP_zeta(s1, s2, Om - Onu, 1. / a - 1.);
    scol.myscol(out, a, Om - Onu, Onu, Rth, s1, s2, extpars, model);
    printf("ORACLE zeta=%.10e deltac=%.10e avir=%.10e delta_avir=%.10e\n", zeta, out[0], out[1], out[2]);
}
