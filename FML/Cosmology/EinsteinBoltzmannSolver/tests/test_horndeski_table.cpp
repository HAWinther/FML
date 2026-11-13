// =============================================================================
// Tests of the model horndeski_table (Perturbations/PerturbationModel_HorndeskiTable.h): a reduced Horndeski theory
// from the table of SimulationCode/scripts/horndeski_frontend.py (input/horndeski there), its background and the
// alpha-parameterization perturbations with the alphas derived from the Lagrangian. Run with `make test` (from the
// folder above).
//
//   * The cubic Galileon from its Lagrangian (the frontend, started on the tracker) reproduces the model
//     cubic_galileon (its own background and field perturbations, PerturbedFieldEquation_CubicGalileon.h): H(a), the
//     growth of delta_cdm from z = 49 to 0 and Psi today
//   * Brans-Dicke (omega = 50, at rest early, G4 = phi/2): on small scales the full solution has the quasi-static
//     G_eff/G = (4 + 2 omega) / ((3 + 2 omega) phi) of the frontend's table (the force of SimulationCode)
//   * A table made for another cosmology (Omega_m) is refused
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/BackgroundModel_HorndeskiTable.h>
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <cstdio>
#include <string>

using namespace FML::COSMOLOGY;
using namespace Constants;
using FML::TESTS::check;

namespace {
    const std::string dir = "../../SimulationCode/input/horndeski/";

    struct Run {
        BackgroundCosmology cosmo;
        RecombinationHistory rec;
        Perturbations pert;
        Run(const CosmologyParameters & p, const PerturbationOptions & o)
            : cosmo(p), rec((cosmo.solve(), cosmo)), pert(cosmo, rec, o) {
            rec.solve();
            pert.solve();
        }
    };

    // The cosmology of the tables in input/horndeski (that of SimulationCode's regression tests)
    CosmologyParameters params(const std::string & model, const std::string & file = "") {
        CosmologyParameters p;
        p.h = 0.671;
        p.OmegaB = 0.049;
        p.OmegaCDM = 0.2685;
        p.Neff = 3.046;
        p.TCMB = 2.7255;
        p.model = model;
        p.model_file = file.empty() ? "" : dir + file;
        return p;
    }

    double rel(double a, double b) { return std::fabs(a / b - 1.0); }
} // namespace

int main() {
    PerturbationOptions o;
    o.k_min = 0.01 / Mpc;
    o.k_max = 2.0 / Mpc;
    const double x49 = std::log(1.0 / 50.0);

    // Cubic Galileon: the table vs the model cubic_galileon
    {
        Run t(params("horndeski_table", "cubic_galileon_gravity.txt"), o);
        Run g(params("cubic_galileon"), o);
        double dE = 0.0;
        for (double x = -12.0; x <= 0.0; x += 0.1)
            dE = std::max(dE, rel(t.cosmo.H_of_x(x), g.cosmo.H_of_x(x)));
        std::printf("cubic Galileon: max |H_table / H - 1| = %.2e\n", dE);
        check(dE < 1e-5, "cubic Galileon: H(a) of the table = the model's");
        double dD = 0.0, dPsi = 0.0;
        for (double kk : {0.05, 0.2, 1.0}) {
            const double k = kk / Mpc;
            const double Dt = t.pert.get_delta_cdm(0.0, k) / t.pert.get_delta_cdm(x49, k);
            const double Dg = g.pert.get_delta_cdm(0.0, k) / g.pert.get_delta_cdm(x49, k);
            const double Pt = t.pert.get_Psi(0.0, k) / t.pert.get_Psi(x49, k);
            const double Pg = g.pert.get_Psi(0.0, k) / g.pert.get_Psi(x49, k);
            std::printf(
                "  k = %.2f/Mpc: growth z = 49 -> 0 table %.6f, cubic_galileon %.6f; Psi(0)/Psi(49) %.6f, %.6f\n",
                kk,
                Dt,
                Dg,
                Pt,
                Pg);
            dD = std::max(dD, rel(Dt, Dg));
            dPsi = std::max(dPsi, rel(Pt, Pg));
        }
        check(dD < 1e-5, "cubic Galileon: growth of delta_cdm from the table = the model's (1e-5)");
        check(dPsi < 1e-5, "cubic Galileon: Psi today from the table = the model's (1e-5)");
    }

    // Brans-Dicke: the full solution on small scales vs the quasi-static G_eff/G of the table
    {
        const auto table = BACKGROUND::read_horndeski_table(dir + "brans_dicke_gravity.txt");
        const double omega = table.header.at("omega");
        Run t(params("horndeski_table", "brans_dicke_gravity.txt"), o);
        const double k = 1.0 / Mpc;
        double worst = 0.0;
        for (double a : {0.25, 0.5, 1.0}) {
            const double x = std::log(a);
            // phi at a from the table (column 9)
            size_t i = 0;
            while (i + 1 < table.a().size() and table.a()[i + 1] < a)
                i++;
            const double f = (a - table.a()[i]) / (table.a()[i + 1] - table.a()[i]);
            const double phi = table.col[8][i] + f * (table.col[8][i + 1] - table.col[8][i]);
            const double mu_qs = (4.0 + 2.0 * omega) / ((3.0 + 2.0 * omega) * phi);
            const double H0a2 = std::pow(t.cosmo.get_H0() * a, 2);
            const double drho =
                H0a2 * (t.cosmo.rhoCDM(x) * t.pert.get_delta_cdm(x, k) + t.cosmo.rhoB(x) * t.pert.get_delta_b(x, k));
            const double mu = -c * c * k * k * t.pert.get_Psi(x, k) / (1.5 * drho);
            std::printf(
                "Brans-Dicke a = %.2f, k = 1/Mpc: -k^2 Psi / (4 pi G a^2 delta rho) = %.6f, quasi-static %.6f\n",
                a,
                mu,
                mu_qs);
            worst = std::max(worst, rel(mu, mu_qs));
        }
        check(worst < 1e-4, "Brans-Dicke: the full solution has the quasi-static G_eff of the table (1e-4)");
    }

    // A table made for another cosmology is refused
    {
        auto p = params("horndeski_table", "cubic_galileon_gravity.txt");
        p.OmegaCDM = 0.25;
        bool refused = false;
        try {
            BackgroundCosmology b(p);
            b.solve();
        } catch (const std::exception & e) {
            refused = std::string(e.what()).find("was made for") != std::string::npos;
        }
        check(refused, "a table made for another Omega_m is refused");
    }
    return FML::TESTS::summary("EBS horndeski_table");
}
