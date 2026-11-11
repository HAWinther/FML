// HMcode-2016 (the HMcode2016 prescription of FML/HaloModel/Halomodel.h, built from a BackgroundCosmology with
// FML/HaloModel/HaloModelBackground.h) against CAMB mead2016 with identical linear spectra (LCDM, sum(m_nu) = 0.3 eV
// and w0wa with curvature), plus the massive-neutrino one-halo factor of the Standard prescription.
// Fixtures: python/generate_hmcode2016_reference.py (CAMB 2.0.5). Run from the folder above.
#include <FML/HaloModel/HaloModelBackground.h>
#include <FML/HaloModel/HMcode2016.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/tests/TestHelpers.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace FML::COSMOLOGY;
using namespace FML::COSMOLOGY::Constants;
using FML::TESTS::check;
using FML::TESTS::check_rel;
namespace HM = FML::COSMOLOGY::HALOMODEL;

namespace {
    std::vector<std::vector<double>> read(const std::string & name) {
        const std::string file = "tests/reference/nonlinear2016/" + name + ".txt";
        std::ifstream in(file);
        if (!in)
            throw std::runtime_error("Missing HMcode-2016 reference: " + file);
        std::vector<std::vector<double>> rows;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream s(line);
            std::vector<double> row;
            double v;
            while (s >> v)
                row.push_back(v);
            rows.push_back(row);
        }
        return rows;
    }
} // namespace

int main() {
    Utils::verbose = false;
    const double h = 0.67, length = Mpc / h, volume = length * length * length;
    const double xs[3] = {-std::log(3.0), -std::log(2.0), 0.0};
    for (const std::string name : {"lcdm", "nu", "w0wa"}) {
        CosmologyParameters p;
        if (name == "nu")
            p.m_nu = 0.3;
        if (name == "w0wa") {
            p.model = "w0wa";
            p.model_params = {{"w0", -0.8}, {"wa", 0.2}};
            p.OmegaK = 0.02;
        }
        BackgroundCosmology bg(p);
        const auto lin = read(name + "_linear"), ref = read(name + "_hmcode2016");
        // The linear spectra are given at z = 2, 1, 0 (the halo model only evaluates them at the redshift computed)
        std::vector<double> lnk;
        for (const auto & r : lin)
            lnk.push_back(std::log(r[0]));
        std::vector<Spline1D> sm(3), scb(3);
        for (int iz = 0; iz < 3; iz++) {
            std::vector<double> a, b;
            for (const auto & r : lin) {
                a.push_back(std::log(r[1 + 2 * iz]));
                b.push_back(std::log(r[2 + 2 * iz]));
            }
            sm[iz].create(lnk, a, "P_m");
            scb[iz].create(lnk, b, "P_cb");
        }
        auto node = [&](double x) {
            for (int i = 0; i < 3; i++)
                if (std::abs(x - xs[i]) < 1e-12)
                    return i;
            throw std::runtime_error("P(k) requested at a redshift without a fixture");
        };
        auto Pm = [&](double x, double k) { return volume * std::exp(sm[node(x)](std::log(k * length))); };
        auto Pcb = [&](double x, double k) { return volume * std::exp(scb[node(x)](std::log(k * length))); };
        const bool nu = p.m_nu > 0;
        const auto cosmology = HM::cosmology_from_background(bg, Pm, nu ? HM::LinearPower(Pcb) : HM::LinearPower());

        HM::HaloModelOptions options;
        options.prescription = HM::Prescription::HMcode2016;
        options.kmin = 1e-3;
        options.kmax = 20.0;
        options.npts_k = 300;
        HM::HaloModel hm(cosmology, options);
        for (int iz = 0; iz < 3; iz++) {
            const double z = 2 - iz;
            hm.compute_at_redshift(z);
            double error = 0;
            for (const auto & row : ref)
                error = std::max(error, std::abs(hm.pofk(row[0]) / row[1 + iz] - 1));
            const auto & s = hm.state();
            std::cout << name << " z=" << z << " max error=" << error << " deltac=" << s.deltac
                      << " DeltaVir=" << s.DeltaVir << " sigma8_cb=" << s.sigma8 << " neff=" << s.neff
                      << " A=" << s.concentration_A << "\n";
            check(error < 0.001, name + " z=" + std::to_string(int(z)) + ": CAMB mead2016 within 0.1%");
            check_rel(hm.deltac_of_z(z), s.deltac, 1e-5, name + ": deltac_of_z at the current redshift");
            if (iz == 2) {
                // The HMcode2016 class (the solver's interface) gives the same
                std::vector<double> k;
                for (const auto & row : ref)
                    k.push_back(row[0]);
                const auto direct = HM::HMcode2016::spectrum(bg, Pm, Pcb, 0.0, k);
                double diff = 0;
                for (size_t i = 0; i < k.size(); i++)
                    diff = std::max(diff, std::abs(direct.nonlinear[i] / ref[i][3] - 1));
                check(diff < 0.001, name + ": HMcode2016::spectrum within 0.1% of CAMB mead2016 (" +
                                        std::to_string(diff) + ")");
            }
        }

        if (nu) {
            // Standard prescription: the one-halo term is (1 - f_nu)^2 times that without the neutrino factor
            HM::HaloModelOptions standard;
            standard.kmin = 0.01;
            standard.kmax = 10.0;
            standard.npts_k = 100;
            auto massless = cosmology;
            massless.f_nu = 0.0;
            HM::HaloModel a(cosmology, standard), b(massless, standard);
            a.compute_at_redshift(0.0);
            b.compute_at_redshift(0.0);
            check_rel(a.pofk_onehalo(1.0) / b.pofk_onehalo(1.0), std::pow(1 - cosmology.f_nu, 2), 1e-10,
                      "Standard: one-halo term times (1 - f_nu)^2");
            check_rel(a.pofk_twohalo(0.1), b.pofk_twohalo(0.1), 1e-12, "Standard: two-halo term unchanged");
            check(a.state().sigma8 > a.state().sigma8_m, "sigma8 of the cold matter above that of the total matter");
        }
    }
    {
        const auto settings = ParameterFile::read("", {"nonlinear=hmcode2016"});
        check(settings.ps.nonlinear == PowerSpectrumOptions::Nonlinear::HMcode2016, "parse nonlinear = hmcode2016");
        CosmologyParameters p;
        p.model = "musigma";
        BackgroundCosmology bg(p);
        auto pl = [](double, double) { return 1.0; };
        check(FML::TESTS::throws([&] { HM::HMcode2016::spectrum(bg, pl, pl, 0, {1.0}); }),
              "HMcode2016 rejects modified-gravity backgrounds");
    }
    return FML::TESTS::summary("HMcode-2016");
}
