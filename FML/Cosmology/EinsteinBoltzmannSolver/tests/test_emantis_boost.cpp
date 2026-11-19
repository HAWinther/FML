// =============================================================================
// nonlinear = emantis (model fr, fR_n = 1): P_fR = P_GR B, with P_GR the nonlinear P(k) of the GR universe with the
// same parameters (emantis_gr) and B the e-MANTIS f(R) boost (FML/Emulator/EMantis). Run with `make test`.
//
//   * P_NL(fR) / P_NL(GR, the same method) = B of the emulator at z = 0 - 2.5 and k = 0.01 - 9 h/Mpc (the table:
//     1e-3); above z = 3 it follows the linear P_fR / P_GR from z = 3 (continuous)
//   * fR0 = 0: GR
//   * Errors: fR_n != 1, |fR0| outside 1e-7 - 1e-4, another model, parameters outside the range (unless
//     emantis_extrapolate_cosmo, which clamps them)
// =============================================================================

#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/EinsteinBoltzmannSolver.h>
#include <FML/Emulator/EMantis/EMantis.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace FML::COSMOLOGY;
using namespace Constants;
using FML::TESTS::check;
using FML::TESTS::check_close;

int main() {
    const double h = 0.68, OmegaB = 0.049, OmegaCDM = 0.26, A_s = 2.1e-9, n_s = 0.965;
    const std::vector<std::string> base{"h=" + std::to_string(h),
                                        "OmegaB=" + std::to_string(OmegaB),
                                        "OmegaCDM=" + std::to_string(OmegaCDM),
                                        "A_s=2.1e-9",
                                        "n_s=" + std::to_string(n_s),
                                        "compute_cells=false",
                                        "precision=fast",
                                        "emantis_gr=hmcode2020"};
    auto run = [&](std::vector<std::string> extra) {
        auto o = base;
        o.insert(o.end(), extra.begin(), extra.end());
        auto e = std::make_unique<EinsteinBoltzmannSolver>(EinsteinBoltzmannSolver::from_file("", o));
        e->set_verbose(false);
        e->solve();
        return e;
    };
    auto throws = [&](std::vector<std::string> extra) {
        try {
            run(extra);
        } catch (const std::exception &) {
            return true;
        }
        return false;
    };

    const auto gr = run({"nonlinear=hmcode2020"});
    const auto fr = run({"model=fr", "fR0=1e-5", "fR_n=1", "nonlinear=emantis"});
    const auto & pg = gr->power_spectrum();
    const auto & pf = fr->power_spectrum();

    // The emulator itself, with the same parameters
    FML::EMULATOR::EMANTIS::NonLinearMGBoostEmulator emu("fR");
    FML::EMULATOR::EMANTIS::Parameters q;
    q.Omega_b = gr->background().rhoB(0.0);
    q.Omega_m = gr->background().rhoB(0.0) + gr->background().rhoCDM(0.0) + gr->background().rhoNcdm(0.0);
    q.h = h;
    q.n_s = n_s;
    q.A_s = A_s;
    q.logfR0 = 5.0;
    const std::vector<double> kh = {0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 9.0};
    double err = 0.0, Bmax = 0.0;
    for (double z : {0.0, 0.5, 1.0, 1.7, 2.5}) {
        const double x = -std::log1p(z);
        const auto B = emu.boost(q, 1.0 / (1.0 + z), kh);
        for (size_t i = 0; i < kh.size(); i++) {
            const double k = kh[i] * h / Mpc;
            const double r =
                pf.get_matter_power_spectrum_nonlinear(x, k) / pg.get_matter_power_spectrum_nonlinear(x, k);
            err = std::max(err, std::fabs(r / B[i] - 1.0));
            Bmax = std::max(Bmax, B[i]);
        }
    }
    check_close(err, 0.0, 1e-3, "P_NL(fR) / P_NL(GR) = the e-MANTIS boost (max B = " + std::to_string(Bmax) + ")");
    {
        // Above z = 3: B(z = 3) times the growth of the linear P_fR / P_GR since then; continuous at z = 3
        const double x3 = -std::log(4.0);
        auto ratio = [&](double x, double k) {
            return pf.get_matter_power_spectrum_nonlinear(x, k) / pg.get_matter_power_spectrum_nonlinear(x, k);
        };
        auto L = [&](double x, double k) {
            return pf.get_matter_power_spectrum_extrapolated(x, k) / pg.get_matter_power_spectrum_extrapolated(x, k);
        };
        double e4 = 0.0, jump = 0.0;
        for (double kk : kh) {
            const double k = kk * h / Mpc;
            for (double z : {4.0, 8.0})
                e4 = std::max(e4, std::fabs(ratio(-std::log1p(z), k) /
                                                (ratio(x3, k) * L(-std::log1p(z), k) / L(x3, k)) - 1.0));
            jump = std::max(jump, std::fabs(ratio(x3 - 1e-6, k) / ratio(x3 + 1e-6, k) - 1.0));
        }
        check_close(e4, 0.0, 1e-10, "z > 3: B = B(z = 3) (P_fR,lin / P_GR,lin) / (the same at z = 3)");
        check_close(jump, 0.0, 1e-4, "B is continuous at z = 3");
    }
    {
        const auto f0 = run({"model=fr", "fR0=0", "fR_n=1", "nonlinear=emantis"});
        double e0 = 0.0;
        for (double k : kh)
            e0 = std::max(e0, std::fabs(f0->power_spectrum().get_matter_power_spectrum_nonlinear(0.0, k * h / Mpc) /
                                            pg.get_matter_power_spectrum_nonlinear(0.0, k * h / Mpc) -
                                        1.0));
        check_close(e0, 0.0, 1e-6, "fR0 = 0: P_NL = that of GR");
    }

    check(throws({"model=fr", "fR0=1e-5", "fR_n=2", "nonlinear=emantis"}), "fR_n = 2 throws");
    check(throws({"model=fr", "fR0=1e-3", "fR_n=1", "nonlinear=emantis"}), "|fR0| = 1e-3 throws");
    check(throws({"nonlinear=emantis"}), "model = lcdm throws");
    check(throws({"model=fr", "fR0=1e-5", "fR_n=1", "nonlinear=emantis", "h=0.9"}),
          "h = 0.9 (outside the range) throws");
    {
        const auto c =
            run({"model=fr", "fR0=1e-5", "fR_n=1", "nonlinear=emantis", "h=0.9", "emantis_extrapolate_cosmo=true"});
        const double r = c->power_spectrum().get_matter_power_spectrum_nonlinear(0.0, 1.0 * 0.9 / Mpc) /
                         c->power_spectrum().get_matter_power_spectrum_extrapolated(0.0, 1.0 * 0.9 / Mpc);
        check(std::isfinite(r) && r > 1.0, "emantis_extrapolate_cosmo = true clamps h = 0.9 to the range");
    }
    return FML::TESTS::summary("emantis");
}
