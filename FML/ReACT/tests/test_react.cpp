#include <FML/ReACT/ReACT.h>
#include <FML/tests/TestHelpers.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

using FML::COSMOLOGY::ReACT;
using namespace FML::TESTS;

namespace {
    ReACT::Input synthetic() {
        ReACT::Input in;
        in.z = {0, 0.5, 1, 2};
        for (int i = 0; i < 64; ++i)
            in.k.push_back(1e-5 * std::pow(1e7, i / 63.0));
        in.total = in.cold = in.lcdm_cold = ReACT::Grid(4, std::vector<double>(64, 1));
        return in;
    }
    std::vector<std::vector<double>> read(const std::string & name) {
        std::ifstream f(std::string(REACT_REFERENCE_DIR) + "/" + name + ".txt");
        if (!f)
            throw std::runtime_error("Missing ReACT reference: " + name);
        std::vector<std::vector<double>> rows;
        std::string line;
        while (std::getline(f, line)) {
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
}

int main() {
    const ReACT::Options gr;
    const auto good = synthetic();
    check(!throws([&] { ReACT::validate(good, gr); }), "valid input is accepted");
    auto bad = good;
    bad.z = {2, 1, 0.5, 0};
    check(throws([&] { ReACT::validate(bad, gr); }), "decreasing redshifts rejected before upstream call");
    bad = good;
    bad.total[0].pop_back();
    check(throws([&] { ReACT::validate(bad, gr); }), "array shape mismatch rejected");
    bad = good;
    bad.z.back() = 3;
    check(throws([&] { ReACT::validate(bad, gr); }), "redshifts above 2.5 rejected");
    bad = good;
    bad.cosmology.Omega_nu = 0.4;
    check(throws([&] { ReACT::validate(bad, gr); }), "unphysical neutrino density rejected");
    bad = good;
    bad.cold[0][0] = std::numeric_limits<double>::quiet_NaN();
    check(throws([&] { ReACT::validate(bad, gr); }), "nonfinite input rejected");
    bad = good;
    bad.cosmology.A_s = 1e-8;
    check(throws([&] { ReACT::validate(bad, gr); }), "upstream primordial-amplitude restriction checked");
    auto opt = gr;
    opt.model = ReACT::Model::HuSawicki;
    check(throws([&] { ReACT::validate(good, opt); }), "fR0=0 requires explicit GR model");
    opt.parameters[0] = 1e-5;
    opt.parameters[19] = 1;
    check(throws([&] { ReACT::validate(good, opt); }), "unused extended parameters rejected");
    ReACT::Result table;
    table.k = good.k;
    table.z = good.z;
    table.reaction = good.total;
    for (size_t iz = 0; iz < table.z.size(); ++iz)
        for (size_t ik = 0; ik < table.k.size(); ++ik)
            table.reaction[iz][ik] = std::exp(0.01 * std::log(table.k[ik]) - 0.2 * std::log1p(table.z[iz]));
    ReACT interpolation(table);
    const double x = -std::log(1.7), k = 0.37;
    check_rel(interpolation.reaction(x, k), std::exp(0.01 * std::log(k) + 0.2 * x), 1e-12,
              "redshift reversal and logarithmic interpolation match a known analytic table");
    check(throws([&] { interpolation.reaction(-std::log(4.0), k); }), "table refuses high-z extrapolation");
    check_rel(interpolation.reaction(0, 1e-6), 1, 0, "reaction=1 below k_min");
    check_rel(interpolation.reaction(0, 1e3), interpolation.reaction(0, table.k.back()), 1e-12,
              "reaction held at upper k endpoint");
    check(throws([&] { interpolation.reaction(0, 0); }), "zero k rejected");
    if (!ReACT::available()) {
        check(throws([&] { ReACT::compute(good, gr); }), "missing backend gives an exception");
        std::cout << "Backend checks not enabled; rebuild with USE_REACT=true for official-reference tests\n";
        return summary("ReACT (backend disabled)");
    }

    for (const std::string name : {"gr", "fr", "dgp", "cpl_nu"}) {
        ReACT::Input in;
        in.z = good.z;
        ReACT::Options options;
        options.mass_loop = 10;
        if (name == "fr") {
            options.model = ReACT::Model::HuSawicki;
            options.parameters[0] = 1e-5;
        } else if (name == "dgp") {
            options.model = ReACT::Model::NormalDGP;
            options.parameters[0] = 0.1;
        } else if (name == "cpl_nu") {
            options.model = ReACT::Model::CPL;
            options.parameters[0] = -0.9;
            options.parameters[1] = 0.1;
            const auto meta = read("cpl_nu_cosmology");
            in.cosmology.Omega_nu = meta[0][0];
            in.cosmology.Omega_m += in.cosmology.Omega_nu;
        }
        const auto spectra = read(name + "_linear");
        for (const auto & row : spectra)
            in.k.push_back(row[0]);
        in.total = in.cold = in.lcdm_cold = ReACT::Grid(4, std::vector<double>(in.k.size()));
        for (size_t iz = 0; iz < in.z.size(); ++iz)
            for (size_t ik = 0; ik < in.k.size(); ++ik) {
                in.total[iz][ik] = spectra[ik][1 + iz * 3];
                in.cold[iz][ik] = spectra[ik][2 + iz * 3];
                in.lcdm_cold[iz][ik] = spectra[ik][3 + iz * 3];
            }
        const auto result = ReACT::compute(in, options);
        const auto ref = read(name);
        double error = 0, linear_error = 0, gr_error = 0;
        for (size_t iz = 0; iz < in.z.size(); ++iz)
            for (size_t ik = 0; ik < in.k.size(); ++ik) {
                error = std::max(error, std::abs(result.reaction[iz][ik] / ref[ik][1 + 3 * iz] - 1));
                error = std::max(error, std::abs(result.pseudo[iz][ik] / ref[ik][3 + 3 * iz] - 1));
                linear_error = std::max(linear_error, std::abs(result.linear[iz][ik] / in.total[iz][ik] - 1));
                if (name == "gr" && in.k[ik] <= 10)
                    gr_error = std::max(gr_error, std::abs(result.reaction[iz][ik] - 1));
            }
        std::cout << name << ": upstream difference=" << error << ", linear roundtrip=" << linear_error << "\n";
        check(error < 1e-8, name + ": C++ interface matches official Python interface");
        check(linear_error < 1e-8, name + ": input target linear power preserved (no double growth rescaling)");
        if (name == "gr")
            check(gr_error < 0.01, "raw GR reaction within 1% of unity for k <= 10 h/Mpc (10 mass samples)");
    }
    return summary("ReACT (official backend)");
}
