// =============================================================================
// Tests for PowerSpectrum. Run with `make test`. Uses ell_max = 1000 and a reduced
// perturbation k-grid to keep the run time short.
// =============================================================================

#include <algorithm>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <string>
#include <vector>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

int main() {
    CosmologyParameters p;
    p.Neff = 0.0;
    BackgroundCosmology cosmo(p);
    cosmo.solve();
    RecombinationHistory rec(cosmo);
    rec.solve();
    PerturbationOptions po;
    po.n_k = 250;
    Perturbations pert(cosmo, rec, po);
    pert.solve();

    PowerSpectrumOptions so;
    so.ell_max = 1000;
    PowerSpectrum ps(cosmo, rec, pert, so);
    ps.solve();

    std::cout << "Primordial spectrum\n";
    check_rel("P_R(k_pivot) = A_s", ps.primordial_power_spectrum(p.k_pivot / Mpc), p.A_s, 1e-14);
    check_rel("P_R slope = n_s - 1",
              ps.primordial_power_spectrum(2.0 * p.k_pivot / Mpc) / p.A_s,
              std::pow(2.0, p.n_s - 1.0),
              1e-14);

    std::cout << "CMB temperature spectrum\n";
    {
        double l_peak = 0.0, D_peak = 0.0;
        for (double l = 100; l <= 400; l += 1.0)
            if (ps.get_Dell_TT(l) > D_peak) {
                D_peak = ps.get_Dell_TT(l);
                l_peak = l;
            }
        check("first acoustic peak at 190 < ell < 230", l_peak > 190 && l_peak < 230);
        check("first peak higher than the Sachs-Wolfe plateau by > 4x", D_peak > 4.0 * ps.get_Dell_TT(10));
    }
    {
        bool positive = true;
        for (int l = 2; l <= 1000; l++)
            positive &= ps.get_Cell_TT(l) > 0.0;
        check("C_ell > 0 for all ell", positive);
        check_rel("D_ell and C_ell consistent",
                  ps.get_Dell_TT(500),
                  500.0 * 501.0 * ps.get_Cell_TT(500) / (2.0 * M_PI) * std::pow(p.TCMB * 1e6, 2),
                  1e-12);
    }
    {
        // Theta_ell(k) for low ell peaks near k (eta0 - eta_*) ~ ell
        const int ell = 100;
        double k_peak = 0.0, T_max = 0.0;
        for (double k : ps.get_k_los())
            if (std::fabs(ps.get_Theta_ell(ell, k)) > T_max) {
                T_max = std::fabs(ps.get_Theta_ell(ell, k));
                k_peak = k;
            }
        const double D = cosmo.eta0() - cosmo.eta_of_x(rec.x_last_scattering());
        check_close("Theta_100(k) peaks near k D_* ~ ell", k_peak * D / ell, 1.0, 0.3);
    }

    std::cout << "Polarization spectra\n";
    {
        bool positive = true;
        for (int l = 2; l <= 1000; l++)
            positive &= ps.get_Dell_EE(l) > 0.0;
        check("C_ell^EE > 0 for all ell", positive);
        // Cauchy-Schwarz: |C^TE| <= sqrt(C^TT C^EE)
        bool cs = true;
        for (int l = 2; l <= 1000; l++)
            cs &= std::fabs(ps.get_Dell_TE(l)) <= std::sqrt(ps.get_Dell_TT(l) * ps.get_Dell_EE(l)) * (1 + 1e-6);
        check("|C^TE| <= sqrt(C^TT C^EE) for all ell", cs);
        int sign_changes = 0;
        for (int l = 3; l <= 1000; l++)
            sign_changes += (ps.get_Dell_TE(l) * ps.get_Dell_TE(l - 1) < 0);
        check("TE oscillates (>= 5 sign changes up to ell = 1000)", sign_changes >= 5);
        // E-modes trace the velocity: EE peaks are out of phase with TT (first EE peak near the first TT trough)
        double l_EE = 0.0, D_EE = 0.0;
        for (double l = 300; l <= 450; l += 1.0)
            if (ps.get_Dell_EE(l) > D_EE) {
                D_EE = ps.get_Dell_EE(l);
                l_EE = l;
            }
        double l_TT = 0.0, D_TT = 1e30;
        for (double l = 300; l <= 450; l += 1.0)
            if (ps.get_Dell_TT(l) < D_TT) {
                D_TT = ps.get_Dell_TT(l);
                l_TT = l;
            }
        check_close("EE peak near the first TT trough (ell ratio)", l_EE / l_TT, 1.0, 0.1);
        check("polarization fraction: EE << TT at the first peak", ps.get_Dell_EE(210) < 0.01 * ps.get_Dell_TT(210));
    }

    std::cout << "Numerical convergence (doubled LOS resolution in k and x)\n";
    {
        PowerSpectrumOptions so2 = so;
        so2.n_k_per_oscillation *= 2;
        so2.n_eta_per_oscillation *= 2;
        so2.dx_recomb /= 2;
        so2.bessel_points_per_2pi *= 2;
        PowerSpectrum ps2(cosmo, rec, pert, so2);
        ps2.solve();
        double max_diff = 0.0, max_diff_E = 0.0;
        for (int l : {2, 10, 50, 200, 500, 800, 1000}) {
            max_diff = std::max(max_diff, std::fabs(ps2.get_Dell_TT(l) / ps.get_Dell_TT(l) - 1.0));
            max_diff_E = std::max(max_diff_E, std::fabs(ps2.get_Dell_EE(l) / ps.get_Dell_EE(l) - 1.0));
        }
        check_close("max relative change in D_ell^TT", max_diff, 0.0, 1e-3);
        check_close("max relative change in D_ell^EE", max_diff_E, 0.0, 1e-3);
    }

    std::cout << "Matter power spectrum\n";
    {
        const double k1 = 1e-4 / Mpc, k2 = 2e-4 / Mpc;
        const double slope =
            std::log(ps.get_matter_power_spectrum(0.0, k2) / ps.get_matter_power_spectrum(0.0, k1)) / std::log(2.0);
        check_close("large scales: P(k) ~ k^{n_s}", slope, p.n_s, 0.01);

        // Turnover near the horizon scale at matter-radiation equality, k_eq = Hp(x_eq)/c
        const double k_eq = cosmo.Hp_of_x(cosmo.x_matter_radiation_equality()) / c;
        double k_peak = 0.0, P_max = 0.0;
        for (double k : Utils::logspace(1e-3 / Mpc, 0.2 / Mpc, 400))
            if (ps.get_matter_power_spectrum(0.0, k) > P_max) {
                P_max = ps.get_matter_power_spectrum(0.0, k);
                k_peak = k;
            }
        check("P(k) peaks within a factor 2 of k_eq", k_peak > 0.5 * k_eq && k_peak < 2.0 * k_eq);
        check_rel("P(k) scales as Delta_M^2",
                  ps.get_matter_power_spectrum(0.0, 0.05 / Mpc),
                  2.0 * M_PI * M_PI / std::pow(0.05 / Mpc, 3) * ps.primordial_power_spectrum(0.05 / Mpc) *
                      std::pow(ps.get_Delta_M(0.0, 0.05 / Mpc), 2),
                  1e-14);
    }

    ps.info();
    std::cout << "Radial-function projection (used for curved universes)\n";
    {
        // Flat: the full radial functions j_ell, j_ell', (3 j_ell'' + j_ell)/2 with the split sources must
        // reproduce the integrated-by-parts source S_T with j_ell
        PowerSpectrumOptions s1;
        s1.ell_max = 400;
        PowerSpectrum a(cosmo, rec, pert, s1);
        a.solve();
        s1.use_radial_functions = true;
        PowerSpectrum b(cosmo, rec, pert, s1);
        b.solve();
        double dtt = 0.0, dee = 0.0, dte = 0.0, temax = 0.0;
        for (int l = 2; l <= 400; l++)
            temax = std::max(temax, std::fabs(a.get_Dell_TE(l)));
        for (int l = 2; l <= 400; l++) {
            dtt = std::max(dtt, std::fabs(b.get_Dell_TT(l) / a.get_Dell_TT(l) - 1.0));
            dee = std::max(dee, std::fabs(b.get_Dell_EE(l) / a.get_Dell_EE(l) - 1.0));
            dte = std::max(dte, std::fabs(b.get_Dell_TE(l) - a.get_Dell_TE(l)) / temax);
        }
        check_close("flat: radial functions = integrated by parts, TT (ell <= 400)", dtt, 0.0, 3e-4);
        check_close("flat: radial functions = integrated by parts, EE (ell <= 400)", dee, 0.0, 3e-4);
        check_close("flat: radial functions = integrated by parts, TE/max|TE|", dte, 0.0, 1e-4);
    }
    {
        // Curved universes: geometry of the acoustic peaks and the closed-universe mode sum
        auto first_peak = [](const PowerSpectrum & ps) {
            double l_peak = 0.0, D_peak = 0.0;
            for (double l = 100; l <= 350; l += 1.0)
                if (ps.get_Dell_TT(l) > D_peak) {
                    D_peak = ps.get_Dell_TT(l);
                    l_peak = l;
                }
            return l_peak;
        };
        // ell_A = pi r(chi_*) / r_s(z_*) with r the transverse distance (= sin_K)
        auto ell_A = [](const BackgroundCosmology & bc, const RecombinationHistory & rc) {
            const double xs = rc.x_last_scattering();
            return M_PI * bc.transverse_distance_of_x(xs) / rc.sound_horizon_of_x(xs);
        };
        PowerSpectrumOptions s1;
        s1.ell_max = 400;
        PowerSpectrum flat(cosmo, rec, pert, s1);
        flat.solve();
        const double lA0 = ell_A(cosmo, rec), lp0 = first_peak(flat);
        for (double OmK : {0.05, -0.05}) {
            const std::string lab = std::string("Omega_K = ") + (OmK > 0 ? "+0.05: " : "-0.05: ");
            CosmologyParameters pc = p;
            pc.OmegaK = OmK;
            BackgroundCosmology cc(pc);
            cc.solve();
            RecombinationHistory rc(cc);
            rc.solve();
            PerturbationOptions oc;
            oc.n_k = 150;
            oc.k_max = 0.05 / Mpc;
            Perturbations pp(cc, rc, oc);
            pp.solve();
            PowerSpectrum ps(cc, rc, pp, s1);
            ps.solve();
            {
                // Flat approximation against the exact recurrence. The modes here reach nu ~ 1000, so lower the
                // threshold to use it: the error must be small and shrink as the threshold rises.
                auto max_err = [&](double nu_flat) {
                    PowerSpectrumOptions sa = s1, sx = s1;
                    sa.flat_approximation_nu = nu_flat;
                    sx.flat_approximation_nu = 1e30;
                    PowerSpectrum a(cc, rc, pp, sa), ex(cc, rc, pp, sx);
                    a.solve();
                    ex.solve();
                    double d = 0.0;
                    for (int l = 2; l <= 400; l++)
                        d = std::max(d, std::fabs(a.get_Dell_TT(l) / ex.get_Dell_TT(l) - 1.0));
                    return d;
                };
                const double e1 = max_err(600.0), e2 = max_err(900.0);
                check_close(lab + "flat approximation (nu >= 600) vs exact recurrence, TT", e1, 0.0, 0.03);
                check(lab + "flat approximation error decreases with a higher nu threshold", e2 < 0.5 * e1);
            }
            const double ratio_peak = first_peak(ps) / lp0, ratio_lA = ell_A(cc, rc) / lA0;
            check_rel(lab + "first peak moves with ell_A = pi r(chi_*)/r_s", ratio_peak, ratio_lA, 0.015);
            check(lab + (OmK > 0 ? "open: peak at higher ell" : "closed: peak at lower ell"),
                  OmK > 0 ? ratio_peak > 1.03 : ratio_peak < 0.97);
            bool positive = true, cs = true;
            for (int l = 2; l <= 400; l++) {
                positive &= ps.get_Dell_TT(l) > 0.0 && ps.get_Dell_EE(l) > 0.0;
                cs &= std::fabs(ps.get_Dell_TE(l)) <= std::sqrt(ps.get_Dell_TT(l) * ps.get_Dell_EE(l)) * (1 + 1e-6);
            }
            check(lab + "C_ell^TT, C_ell^EE > 0 and |C^TE| <= sqrt(C^TT C^EE)", positive && cs);
            const auto & kl = ps.get_k_los();
            const double K = cc.get_K();
            if (K > 0) {
                // Closed: only integer nu, and a uniform integer stride once the linear spacing takes over
                bool integer = true;
                std::vector<double> nus;
                for (double k : kl) {
                    const double nu = std::sqrt(k * k / K + 1.0);
                    integer &= std::fabs(nu - std::round(nu)) < 1e-6 && nu >= 3.0 - 1e-9;
                    nus.push_back(std::round(nu));
                }
                check(lab + "closed: LOS modes have integer nu >= 3", integer);
                const size_t n = nus.size();
                bool uniform = true;
                for (size_t i = n / 2; i + 2 < n; i++)
                    uniform &= (nus[i + 1] - nus[i]) == (nus[n / 2 + 1] - nus[n / 2]);
                check(lab + "closed: uniform stride in nu at large nu", uniform);
            } else {
                check_rel(lab + "open: LOS modes start at k ~ sqrt|K| (q -> 0)", kl.front(), std::sqrt(-K), 1e-3);
            }
        }
    }

    std::cout << "CMB lensing (flat model above)\n";
    {
        PowerSpectrumOptions sl;
        sl.ell_max = 600;
        sl.lensing = true;
        PowerSpectrum lp(cosmo, rec, pert, sl);
        lp.solve();
        // C_phiphi: full line of sight vs Limber (used above ell = 30)
        PowerSpectrumOptions sn = sl;
        sn.lensing_limber_ell = 100000;
        PowerSpectrum ln(cosmo, rec, pert, sn);
        ln.solve();
        double dlim = 0.0;
        for (int l : {40, 60, 100, 200})
            dlim = std::max(dlim, std::fabs(lp.get_Cell_phiphi(l) / ln.get_Cell_phiphi(l) - 1.0));
        check_close("C_phiphi: Limber vs full line of sight, ell = 40-200", dlim, 0.0, 1e-2);
        double lpk = 0.0, Dpk = 0.0;
        for (int l = 2; l <= 600; l++) {
            const double D = std::pow(l * (l + 1.0), 2) * lp.get_Cell_phiphi(l);
            if (D > Dpk) {
                Dpk = D;
                lpk = l;
            }
        }
        check("[l(l+1)]^2 C_phiphi peaks at ell = 20-80", lpk >= 20 && lpk <= 80);
        double low = 0.0;
        for (int l = 2; l <= 30; l++)
            low = std::max(low, std::fabs(lp.get_Dell_TT_lensed(l) / lp.get_Dell_TT(l) - 1.0));
        check_close("lensed TT = unlensed at ell <= 30 (to 1e-3)", low, 0.0, 1e-3);
        bool bb = true;
        for (int l = 2; l <= 600; l++)
            bb &= lp.get_Dell_BB_lensed(l) > 0.0;
        check("lensed BB > 0", bb);
        check("lensing lowers the first peak", lp.get_Dell_TT_lensed(220) < lp.get_Dell_TT(220));
    }

    std::cout << "Multipole list and k-range\n";
    {
        // every ell up to 30 is computed (EE has the reionization feature there; a spline between sparse nodes
        // was off by 2.5% at ell = 17)
        const auto & ells = ps.get_ells();
        bool all = true;
        for (int l = 2; l <= 30; l++)
            all &= std::find(ells.begin(), ells.end(), l) != ells.end();
        check("the default ell list has every ell = 2..30", all);
        // ell beyond k_max r_* (here k_max = 0.3/Mpc, k_max r_* ~ 4200): Theta_ell = 0 and C_ell = 0 gave NaN
        // spectra at every ell; now a clear error
        PowerSpectrumOptions sbig = so;
        sbig.ell_max = 4500;
        PowerSpectrum psb(cosmo, rec, pert, sbig);
        bool threw = false;
        try {
            psb.solve();
        } catch (const std::invalid_argument & e) {
            threw = std::string(e.what()).find("k-range") != std::string::npos;
        }
        check("ell_max beyond the k-range gives an error (not NaN spectra)", threw);
    }

    return Test::summary();
}
