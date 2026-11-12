// ReACT halo-model oracle: the per-redshift quantities of compute_reaction_nu_ext (pyreact/react_wrapper.cpp,
// ACTio-ReACTio a36dda02) for tabulated target spectra: delta_c(M), sigma(M), a_vir(M), delta(a_vir) of the real and
// pseudo collapse, the one-halo terms, E, k*, the one-loop P(k0) and the reaction. The reference of
// FML/tests/test_halomodel_reaction.cpp (HaloModel/HaloModelReaction.h) and PerturbationTheory/tests/test_mg_oneloop.cpp:
// tests/reference/react/halomodel_<name>.txt.
//
// Build against ACTio-ReACTio built as a shared library (build_backend.py in this folder), its staged Copter headers
// and the upstream source:
//   g++ -O2 -std=c++17 -fopenmp halomodel_oracle.cpp -I<build-dir>/react-build/include \
//       -I<ACTio-ReACTio>/reactions/src -I<build-dir>/deps/include -L<backend>/lib -lfml_react_backend \
//       -lgsl -lgslcblas -Wl,-rpath,<backend>/lib -o halomodel_oracle
// Usage: halomodel_oracle ../<name>_linear.txt model parameter0 parameter1 Omega_nu mass_loop
//   (model 1 GR, 2 f(R), 3 nDGP, 5 CPL; the cosmology of generate_reference.py; prints lines Z, M, K)
#include <Copter/HALO.h>
#include <Copter/Cosmology.h>
#include <Copter/LinearPS.h>
#include <Copter/SpecialFunctions.h>
#include <Copter/SPT.h>
#include <Copter/BeyondLCDM.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
extern Spline delta_col, a_vir, delta_avir, mysig, delta_colp, a_virp, delta_avirp, mysigp;
extern double kstar, bigE;
int main(int argc, char ** argv) {
    // args: linear-file model param0 param1 Omega_nu mass_loop
    const char * file = argv[1];
    int mod = atoi(argv[2]);
    double extp[maxpars] = {0};
    extp[0] = atof(argv[3]);
    extp[1] = atof(argv[4]);
    const double onu = atof(argv[5]);
    const int mass_loop = atoi(argv[6]);
    const double h = 0.67, ns = 0.965, Om = 0.317 + onu, Ob = 0.05, As = 2.1e-9, ps = 0.05;
    std::vector<std::vector<double>> rows;
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        std::vector<double> r; double v;
        while (s >> v) r.push_back(v);
        rows.push_back(r);
    }
    const int nk = rows.size(), nz = 4;
    double zvals[4] = {2.0, 1.0, 0.5, 0.0}; // decreasing as upstream; file columns are z = 0, 0.5, 1, 2
    array ki(nk), Tm(nk), Tcb(nk), Tnu(nk), Tl(nk);
    for (int j = 0; j < nk; j++) ki[j] = rows[j][0];
    double pars[8] = {1., Om, onu, 0, 0, double(mass_loop), 0, 0};
    IOW iow;
    iow.initnorm(pars, extp, mod);
    std::vector<Cosmology> cos; cos.reserve(16);
    std::vector<LinearPS_as> lps; lps.reserve(16);
    std::vector<HALO> halo; halo.reserve(4);
    std::vector<SPT> spt; spt.reserve(4);
    auto T = [&](double P, double k) { return std::sqrt(P / (2. * M_PI * M_PI * As * k * pow(h, 4) * pow(k * h / ps, ns - 1.))); };
    for (int i = 0; i < nz; i++) {
        const int col = 3 - i; // z index in the file
        for (int j = 0; j < nk; j++) {
            Tm[j] = T(rows[j][1 + 3 * col], ki[j]);
            Tcb[j] = T(rows[j][2 + 3 * col], ki[j]);
            Tnu[j] = onu < 1e-10 ? Tm[j] : (Om * Tm[j] - (Om - onu) * Tcb[j]) / onu;
            Tl[j] = T(rows[j][3 + 3 * col], ki[j]);
        }
        cos.push_back(Cosmology(h, ns, Om, Ob, As, ps, ki, Tm));
        cos.push_back(Cosmology(h, ns, Om, Ob, As, ps, ki, Tcb));
        cos.push_back(Cosmology(h, ns, Om, Ob, As, ps, ki, Tnu));
        cos.push_back(Cosmology(h, ns, Om, Ob, As, ps, ki, Tl));
        for (int q = 0; q < 4; q++) lps.push_back(LinearPS_as(cos[4 * i + q], 0.));
        halo.push_back(HALO(cos[4 * i], lps[4 * i], lps[4 * i + 1], lps[4 * i + 2], lps[4 * i + 3], 1e-2));
        spt.push_back(SPT(cos[4 * i + 1], lps[4 * i + 1], 1e-2));
    }
    const bool modg = (mod == 2 || mod == 3);
    double ploopr[4] = {0}, ploopp[4] = {0};
    const double k0 = K0HMR;
    if (modg) {
        double pkz0[4], pkz0p[4], pkz1[4][50], pkz2[4][1600], pkz1p[4][50], pkz2p[4][1600];
        double KMAX = QMAXp / k0, KMIN = QMINp / k0;
        for (int i = 0; i < nz; i++) {
            pkz0[i] = lps[i * 4 + 1](k0);
            pkz0p[i] = lps[i * 4](k0);
            for (int j = 0; j < 50; j++) {
                double k2 = k0 * KMIN * exp(j * log(KMAX / KMIN) / 49.);
                pkz1[i][j] = lps[i * 4 + 1](k2);
                pkz1p[i][j] = lps[i * 4](k2);
                for (int jj = 0; jj < 32; jj++) {
                    double karg = sqrt(k2 * k2 + k0 * k0 - 2. * k2 * k0 * x32[jj]);
                    pkz2[i][jj * 50 + j] = lps[i * 4 + 1](karg);
                    pkz2p[i][jj * 50 + j] = lps[i * 4](karg);
                }
            }
        }
        spt[0].ploop_init_nu(pkz0, pkz1, pkz2, pkz0p, pkz1p, pkz2p, ploopr, ploopp, zvals, nz, pars, extp, k0, mod);
    }
    double sf[4];
    for (int i = 0; i < nz; i++) sf[i] = 1. / (1. + zvals[i]);
    Spline mysr = CubicSpline(nz, sf, ploopr), mysp = CubicSpline(nz, sf, ploopp);
    const double ks[] = {0.01, 0.03, 0.06, 0.1, 0.3, 1.0, 3.0, 10.0};
    for (int j = 0; j < nz; j++) {
        pars[0] = 1. / (1. + zvals[j]);
        iow.initnorm(pars, extp, mod);
        halo[j].scol_init(pars, extp, true, mod);
        halo[j].scol_initp(pars, true);
        halo[j].react_init_nu_multiz(pars, mysr, mysp, true, modg);
        printf("Z %g a %.10e ploopr %.10e ploopp %.10e E %.10e kstar %.10e gde %.10e\n", zvals[j], pars[0], mysr(pars[0]), mysp(pars[0]), bigE, kstar, g_de);
        for (double lm : {6.0, 9.0, 12.0, 13.0, 14.0, 15.0})
            printf("M %g %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", lm, delta_col(lm), mysig(lm), a_vir(lm), delta_avir(lm), delta_colp(lm), mysigp(lm), a_virp(lm), delta_avirp(lm));
        for (double k : ks)
            printf("K %g %.10e %.10e %.10e\n", k, halo[j].one_halo(k, pars), halo[j].one_halop(k, pars), halo[j].reaction_nu(k, pars, true));
    }
}
