#include <FML/AxionHMcode/AxionHMcode.h>
#include <FML/tests/TestHelpers.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

using A = FML::COSMOLOGY::AxionHMcode;
using namespace FML::TESTS;
namespace {
    std::vector<std::vector<double>> read(const std::string & name) {
        std::ifstream f(std::string(AXION_REFERENCE_DIR) + "/" + name + ".txt");
        if (!f) throw std::runtime_error("Missing axion reference: " + name);
        std::vector<std::vector<double>> rows;
        std::string s;
        while (std::getline(f, s)) {
            if (s.empty() || s[0] == '#') continue;
            std::istringstream in(s);
            std::vector<double> row;
            double x;
            while (in >> x) row.push_back(x);
            rows.push_back(row);
        }
        return rows;
    }
}
int main() {
    check_rel(A::beta2(2, .1), .87, 1e-14, "Dome beta2 calibration node");
    check_rel(A::beta2(2.5, .15), (.87 + .59 + .93 + .66) / 4, 1e-14, "Dome beta2 bilinear interpolation");
    check(throws([] { A::beta2(0, .1); }), "beta2 refuses implicit redshift extrapolation");
    check(std::isfinite(A::beta2(0, .1, true)), "beta2 explicit extrapolation");
    check_rel(A::jeans_mass(2, .31, .674, 1e-24) / A::jeans_mass(2, .31, .674, 4e-24), 8, 1e-12,
              "Jeans mass scales as m_axion^-3/2");
    A::Cosmology c;
    c.Omega_b = .05; c.Omega_axion = .0317; c.Omega_cdm = .2353; c.h = .67;
    A::Options o;
    o.n_mass = 65;
    A::Input p;
    const auto linear = read("linear");
    for (const auto & row : linear) { p.k.push_back(row[0]); p.cold.push_back(row[1]); p.axion.push_back(row[2]); }
    check(!throws([&] { A::validate(c, p, o); }), "valid Dome inputs accepted");
    auto bad = p; bad.cold.pop_back();
    check(throws([&] { A::validate(c, bad, o); }), "mismatched linear arrays rejected");
    auto wrong = c; wrong.z = 0;
    check(throws([&] { A::validate(wrong, p, o); }), "Dome z=0 needs explicit extrapolation");
    wrong = c; wrong.mass_eV = 1e-24;
    check(throws([&] { A::validate(wrong, p, o); }), "Dome mass extrapolation must be explicit");
    bad = p; bad.axion[0] = -1;
    check(throws([&] { A::validate(c, bad, o); }), "negative axion power rejected");
    for (const auto & name : {std::string("basic"), std::string("dome")}) {
        o.version = name == "basic" ? A::Version::Basic : A::Version::Dome2024;
        const auto ref = read(name);
        std::vector<double> k;
        for (const auto & row : ref) k.push_back(row[0]);
        const auto result = A::spectrum(c, p, k, o);
        double err = 0;
        for (size_t i = 0; i < k.size(); ++i) {
            for (int j = 0; j < 4; ++j) {
                const auto & out = j == 0 ? result.total : j == 1 ? result.cold : j == 2 ? result.cold_axion : result.axion;
                err = std::max(err, std::abs(out[i] / ref[i][j + 1] - 1));
            }
        }
        double norm = 0;
        for (const auto & h : result.haloes) norm = std::max(norm, std::abs(h.normalization_error));
        std::cout << name << " max component error=" << err << " Fc=" << result.clustered_fraction
                  << " Mcut=" << result.cutoff_mass << " beta=" << result.beta2 << " profile error=" << norm
                  << " dc=" << result.delta_c << " Dv=" << result.Delta_vir << "\n";
        check(err < 1e-4, name + ": native component spectra agree with reference within 0.01%");
        const auto diagnostics = read(name + "_diagnostics")[0];
        check_rel(result.cutoff_mass, diagnostics[0], 1e-7, name + ": cutoff mass");
        check_rel(result.clustered_fraction, diagnostics[2], 1e-7, name + ": clustered fraction");
        check_rel(result.alpha_cold, o.version == A::Version::Basic ? 1 : diagnostics[3], 1e-7, name + ": transition parameter");
        check(norm < 1e-5, name + ": every axion profile has the specified mass");
        check(result.clustered_fraction >= 0 && result.clustered_fraction <= 1, name + ": physical clustered fraction");
        const double f = c.Omega_axion / (c.Omega_b + c.Omega_cdm + c.Omega_axion);
        double residual = 0;
        for (size_t i = 0; i < k.size(); ++i)
            residual = std::max(residual, std::abs((std::pow(1 - f, 2) * result.cold[i] +
                2 * f * (1 - f) * result.cold_axion[i] + f * f * result.axion[i]) / result.total[i] - 1));
        check(residual < 1e-14, "component spectra reconstruct the total");
    }
    {
        o.version = A::Version::Dome2024;
        o.n_mass = 257;
        const std::vector<double> k{.001, .1, .5, 1, 3, 10};
        const auto normal = A::spectrum(c, p, k, o);
        o.n_mass = 513; o.n_radius = 4000;
        const auto fine = A::spectrum(c, p, k, o);
        double change = 0;
        for (size_t i = 0; i < k.size(); ++i) change = std::max(change, std::abs(fine.total[i] / normal.total[i] - 1));
        std::cout << "Default versus doubled mass/radius resolution: " << change << "\n";
        check(change < 5e-4, "default total spectrum converged within 0.05% in mass and radial resolution");
        o.n_mass = 65; o.n_radius = 2000;
        o.full_two_halo = true; o.halo_bloating = true; o.two_halo_damping = true;
        o.jeans_density = A::JeansDensity::OmegaMatter;
        const auto alternate = A::spectrum(c, p, {p.k.front(), .1, 1}, o);
        check_rel(alternate.total[0], alternate.linear_total[0], 1e-3, "optional full two-halo model approaches linear total power");
        check(std::isfinite(alternate.total[2]) && alternate.total[2] > 0, "alternate density convention and optional halo terms are finite");
        o.full_two_halo = o.halo_bloating = o.two_halo_damping = false;
    }
    // Zero axions: exact reduction to the configured cold halo model, without divisions by Omega_a or Fc.
    c.Omega_cdm += c.Omega_axion; c.Omega_axion = 0;
    const auto zero = A::spectrum(c, p, {.001, .1, 1}, o);
    check(zero.total == zero.cold && zero.clustered_fraction == 0 && zero.haloes.empty(), "exact zero-axion limit");
    return summary("native AxionHMcode");
}
