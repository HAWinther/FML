//==========================================================
// Tests for FML::COSMOLOGY::HALOMODEL::HaloModelReaction (the halo-model reaction of ReACT, HaloModelReaction.h):
//   - against ReACT (ACTio-ReACTio a36dda02, HALO.cpp through ReACT/tests/halomodel_oracle.cpp; reference
//     ../ReACT/tests/reference/halomodel_<name>.txt) for the linear spectra of the ReACT module's fixtures (GR,
//     f(R) |fR0| = 1e-5, nDGP Omega_rc = 0.1, CPL w0 = -0.9, wa = 0.1 with sum m_nu = 0.3 eV; z = 0, 0.5, 1, 2):
//     delta_c(M) and sigma(M) of the real and pseudo halos, g_de, E, k* (from ReACT's one-loop P(k0)), the
//     one-halo terms and the reaction
//   - identical real and pseudo cosmologies (massless GR) give R = 1 exactly
//   - modified gravity needs k*; inconsistent one-loop input falls back to E = k* = 1 (as ReACT); invalid input
// Run with: make run (in this folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/HaloModel/HaloModelReaction.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using FML::TESTS::check_local;
using namespace FML::COSMOLOGY::HALOMODEL;
using namespace FML::COSMOLOGY::SPHERICALCOLLAPSE;
using Spline = FML::INTERPOLATION::SPLINE::Spline;

namespace {
    const std::string refdir = "../ReACT/tests/reference/";

    std::vector<std::vector<std::string>> read(const std::string & file) {
        std::ifstream in(file);
        if (!in)
            throw std::runtime_error("Missing " + file + " (run from FML/tests)");
        std::vector<std::vector<std::string>> rows;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() or line[0] == '#')
                continue;
            std::istringstream s(line);
            std::vector<std::string> r;
            std::string t;
            while (s >> t)
                r.push_back(t);
            rows.push_back(r);
        }
        return rows;
    }
    std::string fmt(double v) {
        std::ostringstream s;
        s.precision(2);
        s << v;
        return s.str();
    }

    // The fixture's spectra at column iz (z = 0, 0.5, 1, 2) as log-log splines
    HaloModelReaction::LinearSpectra spectra(const std::vector<std::vector<std::string>> & rows, int iz) {
        std::vector<double> lk, lP, lPcb, lPl;
        for (const auto & r : rows) {
            lk.push_back(std::log(std::stod(r[0])));
            lP.push_back(std::log(std::stod(r[1 + 3 * iz])));
            lPcb.push_back(std::log(std::stod(r[2 + 3 * iz])));
            lPl.push_back(std::log(std::stod(r[3 + 3 * iz])));
        }
        auto sp = std::make_shared<Spline>(lk, lP), spc = std::make_shared<Spline>(lk, lPcb),
             spl = std::make_shared<Spline>(lk, lPl);
        HaloModelReaction::LinearSpectra S;
        S.P = [sp](double k) { return std::exp((*sp)(std::log(k))); };
        S.Pcb = [spc](double k) { return std::exp((*spc)(std::log(k))); };
        S.Pcb_lcdm = [spl](double k) { return std::exp((*spl)(std::log(k))); };
        return S;
    }
} // namespace

int main() {
    struct Case {
        std::string name;
        int model;
        double p0, p1;
    };
    const double Ocb = 0.317;
    const double Onu_cpl = std::stod(read(refdir + "cpl_nu_cosmology.txt")[0][0]);

    std::cout << "Against ReACT (HALO.cpp)\n";
    for (const Case & c : {Case{"gr", 1, 0, 0}, Case{"fr", 2, 1e-5, 0}, Case{"dgp", 3, 0.1, 0}, Case{"cpl_nu", 5, -0.9, 0.1}}) {
        const auto lin = read(refdir + c.name + "_linear.txt");
        const auto ref = read(refdir + "halomodel_" + c.name + ".txt");
        const double Onu = c.model == 5 ? Onu_cpl : 0.0, Om = Ocb + Onu;
        HaloModelReaction::Cosmology cosmo;
        cosmo.OmegaM = Om;
        cosmo.OmegaNu = Onu;
        if (c.model == 5) {
            const double w0 = c.p0, wa = c.p1;
            auto de = [=](double x) { return (1 - Om) * std::exp(-3 * (1 + w0 + wa) * x - 3 * wa * (1 - std::exp(x))); };
            cosmo.Eofx = [=](double x) { return std::sqrt(Om * std::exp(-3 * x) + de(x)); };
            cosmo.logEprimeofx = [=](double x) {
                const double m = Om * std::exp(-3 * x);
                return 0.5 * (-3 * m + de(x) * (-3 * (1 + w0 + wa) + 3 * wa * std::exp(x))) / (m + de(x));
            };
        }
        HaloModelReaction::FifthForceOfRadius F;
        if (c.model == 2)
            F = [=](double R) { return fifth_force_hu_sawicki(c.p0, 1.0, Om, R); };
        if (c.model == 3) {
            SphericalCollapseModel m;
            m.OmegaM = Om;
            m.set_model_lcdm();
            F = [=](double) { return fifth_force_ndgp(c.p0, m); };
        }
        double eE = 0, ek = 0, eg = 0, edc = 0, esg = 0, e1h = 0, eR = 0, Rmax = 0;
        size_t row = 0;
        while (row < ref.size()) {
            const auto & Z = ref[row++];
            const double z = std::stod(Z[1]), a = std::stod(Z[3]);
            const int iz = z == 0 ? 0 : z == 0.5 ? 1 : z == 1 ? 2 : 3;
            HaloModelReaction hm(cosmo, a, spectra(lin, iz), F);
            if (hm.modified_gravity())
                hm.set_one_loop(std::stod(Z[5]), std::stod(Z[7]));
            eE = std::max(eE, std::abs(hm.E() / std::stod(Z[9]) - 1));
            ek = std::max(ek, std::abs(hm.kstar() / std::stod(Z[11]) - 1));
            eg = std::max(eg, std::abs(hm.g_de() / std::stod(Z[13]) - 1));
            while (row < ref.size() and ref[row][0] != "Z") {
                const auto & r = ref[row++];
                const double x = std::stod(r[1]);
                if (r[0] == "M") {
                    edc = std::max({edc, std::abs(hm.deltac(x) / std::stod(r[2]) - 1),
                                    std::abs(hm.deltac_pseudo(x) / std::stod(r[6]) - 1)});
                    esg = std::max({esg, std::abs(hm.sigma(x) / std::stod(r[3]) - 1),
                                    std::abs(hm.sigma_pseudo(x) / std::stod(r[7]) - 1)});
                } else {
                    e1h = std::max({e1h, std::abs(hm.one_halo(x) / std::stod(r[2]) - 1),
                                    std::abs(hm.one_halo_pseudo(x) / std::stod(r[3]) - 1)});
                    eR = std::max(eR, std::abs(hm.reaction(x) - std::stod(r[4])));
                    Rmax = std::max(Rmax, std::abs(std::stod(r[4]) - 1));
                }
            }
        }
        std::cout << "  " << c.name << ": delta_c " << edc << ", sigma " << esg << ", g_de " << eg << ", E " << eE
                  << ", k* " << ek << ", P_1h " << e1h << ", |R - R_ReACT| " << eR << " (max |R - 1| " << Rmax << ")\n";
        check_local(edc < 2e-4 and esg < 2e-4, c.name + ": delta_c(M), sigma(M) (real and pseudo) = ReACT");
        check_local(eg < 1e-6 and eE < 2e-4 and ek < 5e-3, c.name + ": g_de, E, k* = ReACT");
        check_local(e1h < 3e-3, c.name + ": one-halo terms (real and pseudo) = ReACT to " + fmt(e1h));
        check_local(eR < 2e-3, c.name + ": reaction = ReACT to " + fmt(eR) + " (k = 0.01 - 10 h/Mpc)");
    }

    std::cout << "Identical real and pseudo cosmologies\n";
    {
        const auto lin = read(refdir + "gr_linear.txt");
        HaloModelReaction::Cosmology cosmo;
        cosmo.OmegaM = Ocb;
        HaloModelReaction hm(cosmo, 1.0, spectra(lin, 0));
        double d = 0;
        for (double k : {0.02, 0.1, 1.0, 10.0})
            d = std::max(d, std::abs(hm.reaction(k) - 1));
        check_local(d < 1e-12, "massless GR with P_cb^LCDM = P: R = 1");
        check_local(not hm.modified_gravity() and hm.E() == 1.0, "GR: E = 1");
    }

    std::cout << "Modified gravity needs k*; fallback; invalid input\n";
    {
        const auto lin = read(refdir + "fr_linear.txt");
        HaloModelReaction::Cosmology cosmo;
        cosmo.OmegaM = Ocb;
        auto F = [=](double R) { return fifth_force_hu_sawicki(1e-5, 1.0, Ocb, R); };
        HaloModelReaction hm(cosmo, 1.0, spectra(lin, 0), F);
        check_local(FML::TESTS::throws([&] { hm.reaction(0.1); }), "reaction without k* throws for modified gravity");
        check_local(hm.reaction(0.005) == 1.0, "R = 1 below k = 0.01 h/Mpc");
        const bool ok = hm.set_one_loop(1e3, 1e5);
        check_local(not ok and hm.E() == 1.0 and hm.kstar() == 1.0, "inconsistent one-loop input: E = k* = 1 (as ReACT)");
        check_local(FML::TESTS::throws([&] { HaloModelReaction(cosmo, 1.5, spectra(lin, 0)); }), "a > 1 rejected");
        auto bad = cosmo;
        bad.OmegaNu = 0.5;
        check_local(FML::TESTS::throws([&] { HaloModelReaction(bad, 1.0, spectra(lin, 0)); }), "OmegaNu >= OmegaM rejected");
        auto S = spectra(lin, 0);
        S.Pcb_lcdm = nullptr;
        check_local(FML::TESTS::throws([&] { HaloModelReaction(cosmo, 1.0, S); }), "missing LCDM cold spectrum rejected");
    }
    return FML::TESTS::summary("Halo-model reaction");
}
