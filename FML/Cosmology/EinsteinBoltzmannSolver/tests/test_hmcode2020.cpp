// Identical-linear-spectrum comparisons with CAMB HMcode-2020 and CLASS Takahashi+Bird.
// Fixtures: python/generate_nonlinear_reference.py. No external Python packages needed to run.
#include <FML/Cosmology/Spectra/HMcode.h>
#include <FML/Cosmology/Spectra/Halofit.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/tests/TestHelpers.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace FML::COSMOLOGY;
using namespace FML::COSMOLOGY::Constants;
using FML::TESTS::check;
using FML::TESTS::check_rel;

namespace {
    std::vector<std::vector<double>> read(const std::string & name) {
        const std::string file = "tests/reference/nonlinear2020/" + name + ".txt";
        std::ifstream in(file);
        if (!in)
            throw std::runtime_error("Missing nonlinear reference: " + file);
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
    Spline1D spline(const std::vector<std::vector<double>> & rows, int column) {
        std::vector<double> lnk, lnp;
        for (const auto & row : rows) {
            lnk.push_back(std::log(row[0]));
            lnp.push_back(std::log(row[column]));
        }
        Spline1D s;
        s.create(lnk, lnp, "external linear power");
        return s;
    }
}

int main() {
    Utils::verbose = false;
    const double h = 0.67, length = Mpc / h, volume = length * length * length;
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
        const auto input = read(name + "_linear"), ref = read(name + "_hmcode");
        std::vector<double> k;
        for (const auto & row : ref)
            k.push_back(row[0]);
        for (int iz = 0; iz < 3; ++iz) {
            const double z = 2 - iz, x = -std::log1p(z);
            const auto sm = spline(input, 1 + 2 * iz), scb = spline(input, 2 + 2 * iz);
            auto pm = [&](double, double kk) { return volume * std::exp(sm(std::log(kk * length))); };
            auto pcb = [&](double, double kk) { return volume * std::exp(scb(std::log(kk * length))); };
            const auto result = HMcode::spectrum(bg, pm, pcb, x, k);
            double error = 0;
            for (size_t i = 0; i < k.size(); ++i) {
                error = std::max(error, std::abs(result.nonlinear[i] / ref[i][iz + 1] - 1));
            }
            check(std::all_of(result.nonlinear.begin(), result.nonlinear.end(), [](double v) {
                      return std::isfinite(v) && v > 0;
                  }), name + ": finite positive nonlinear power");
            std::cout << name << " z=" << z << " max error=" << error << " sigma8_cb=" << result.sigma8_cb
                      << " dc=" << result.delta_c << " Dv=" << result.Delta_v << " neff=" << result.n_eff << "\n";
            check(error < 0.001, name + " z=" + std::to_string(int(z)) + ": CAMB HMcode-2020 within 0.1%");
            if (name == "lcdm" && iz == 2) {
                const auto fine = HMcode::spectrum(bg, pm, pcb, x, k, 512, 0.0075);
                double change = 0;
                for (size_t i = 0; i < k.size(); ++i)
                    change = std::max(change, std::abs(fine.nonlinear[i] / result.nonlinear[i] - 1));
                std::cout << "HMcode convergence: " << change << "\n";
                check(change < 0.001, "HMcode mass/k integrals converged within 0.1%");
            }
        }
    }

    // Bird on exactly the CLASS input grid, including the h-dependent quasi-linear correction.
    {
        CosmologyParameters p;
        p.m_nu = 0.3;
        BackgroundCosmology bg(p);
        const auto input = read("bird_linear"), ref = read("bird_halofit");
        const double fnu = bg.rhoNcdm(0) / (bg.rhoB(0) + bg.rhoCDM(0) + bg.rhoNcdm(0));
        for (int iz = 0; iz < 3; ++iz) {
            const double z = 2 - iz, x = -std::log1p(z);
            std::vector<double> lnk, dl, nl, plain;
            for (const auto & row : input) {
                lnk.push_back(std::log(row[0]));
                dl.push_back(std::pow(row[0], 3) * row[iz + 1] / (2 * M_PI * M_PI));
            }
            const double om = (bg.rhoB(x) + bg.rhoCDM(x) + bg.rhoNcdm(x)) / bg.E2_of_x(x);
            double ks, ne, C;
            Halofit::nonlinear(lnk, dl, om, bg.OmegaDE_of_x(x), -1, nl, ks, ne, C, fnu, h);
            std::vector<double> lognl;
            for (double d : nl)
                lognl.push_back(std::log(d));
            Spline1D s;
            s.create(lnk, lognl, "Bird Delta^2");
            double error = 0;
            for (const auto & row : ref) {
                const double kh = row[0], k = kh * h;
                const double ours = std::exp(s(std::log(k))) * 2 * M_PI * M_PI / std::pow(kh, 3);
                error = std::max(error, std::abs(ours / row[iz + 1] - 1));
            }
            std::cout << "Bird z=" << z << " max CLASS error=" << error << "\n";
            check(error < 0.001, "Bird: CLASS within 0.1%");
            Halofit::nonlinear(lnk, dl, om, bg.OmegaDE_of_x(x), -1, plain, ks, ne, C);
            Halofit::nonlinear(lnk, dl, om, bg.OmegaDE_of_x(x), -1, nl, ks, ne, C, 0, h);
            check(plain == nl, "fnu=0: Bird exactly reproduces Takahashi");
        }
    }
    for (const auto & option : {std::string("halofit_bird"), std::string("hmcode2020")}) {
        const auto settings = ParameterFile::read("", {"nonlinear=" + option});
        check(settings.ps.nonlinear != PowerSpectrumOptions::Nonlinear::None, "parse " + option);
    }
    {
        CosmologyParameters p;
        p.model = "musigma";
        BackgroundCosmology bg(p);
        auto pl = [](double, double) { return 1.0; };
        check(FML::TESTS::throws([&] { HMcode::spectrum(bg, pl, pl, 0, {1.0}); }),
              "HMcode rejects modified-gravity models without a nonlinear prescription");
    }
    return FML::TESTS::summary("HMcode-2020 and Bird");
}
