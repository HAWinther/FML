//==========================================================
// Tests of the shared linear input (InitialConditions/LinearInput.h), the species IC helpers (SpeciesIC.h) and the
// conventions of a table TransferProvider (TransferProviders/CAMBFileTransferProvider.h) on small synthetic CAMB
// tables with known values. The comparisons with CAMB and CLASS are in test_camb.cpp and TransferProviders/tests
//==========================================================
#include <FML/InitialConditions/LinearInput.h>
#include <FML/InitialConditions/SpeciesIC.h>
#include <FML/TransferProviders/CAMBFileTransferProvider.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::check_local;
using namespace FML::INITIALCONDITIONS;
template <class F>
void rejects(F && f, const char * label) {
    check_local(FML::TESTS::throws<std::exception>(f), label);
}
int main() {
    const std::string prefix = "test_transfers_" + std::to_string(FML::ThisTask);
    const double h = .67;
    // T_cdm = 1e4 (4 + log a) / (1 + k) at a = 1, 1/11, 1/21, 1/31 and k = 0.001 2^i h/Mpc (cubic in log a, so the
    // 4-point Lagrange growth rate is exact: 1 / (4 + log a)). Columns (CAMB, no header): k CDM
    // baryon photon nu mass_nu total no_nu total_de Weyl v_CDM v_b v_b-v_c, and a copy with only the first 7 (old CAMB)
    const auto t_of = [](double k, double z) { return 1e4 * (4 - std::log(1 + z)) / (1 + k); };
    std::vector<std::string> files;
    for (int ncol : {13, 7}) {
        std::ofstream info(prefix + "_" + std::to_string(ncol) + ".info");
        info << ". 4\n";
        for (int j = 0; j < 4; j++) {
            const auto name = prefix + "_" + std::to_string(ncol) + "_" + std::to_string(j) + ".dat";
            files.push_back(name);
            info << name << " " << j * 10 << "\n";
            std::ofstream out(name);
            out.precision(17);
            for (int i = 0; i < 12; i++) {
                const double k = .001 * std::pow(2., i), t = t_of(k, j * 10.);
                const std::vector<double> row{k, t, .8 * t, 0, 0, .5 * t, .9 * t, .9 * t, .9 * t, 0, .6 * t, .4 * t, 0};
                for (int c = 0; c < ncol; c++)
                    out << row[c] << (c + 1 < ncol ? " " : "\n");
            }
        }
    }
    const auto make = [&](int ncol, RangePolicy policy) {
        CAMBFileTransferProvider::Settings s;
        s.infofile = prefix + "_" + std::to_string(ncol) + ".info";
        s.h = h;
        s.Omegab = .05;
        s.OmegaCDM = .25;
        s.OmegaMNu = .01;
        s.A_s = 2.1e-9;
        s.n_s = .965;
        s.k_pivot_mpc = .05;
        s.range_policy = policy;
        return std::make_shared<CAMBFileTransferProvider>(s);
    };
    auto provider = make(13, RangePolicy::SourceExtrapolation);

    // At the nodes of the tables the values are those of the files
    bool normal = true, velocity = true, cb = true, total = true, rates = true;
    for (int i : {1, 5, 9})
        for (int j : {0, 1, 2}) {
            const double k = .001 * std::pow(2., i), z = 10. * j, a = 1. / (1 + z), k2 = (k * h) * (k * h);
            const double expected = t_of(k, z) * k2;
            normal = normal && std::abs(provider->density_transfer(Species::CDM, k, a) / expected - 1) < 1e-12 &&
                     std::abs(provider->density_over_k_mpc2(Species::CDM, k, a) / t_of(k, z) - 1) < 1e-12;
            velocity = velocity &&
                       std::abs(provider->velocity_divergence_transfer(Species::CDM, k, a) / (-.6 * expected) - 1) <
                           1e-12;
            cb = cb && std::abs(provider->density_transfer(Species::CDMBaryon, k, a) /
                                    ((.25 + .05 * .8) / .30 * expected) -
                                1) < 1e-12;
            total = total && std::abs(provider->density_transfer(Species::TotalMatter, k, a) / (.9 * expected) - 1) <
                                 1e-12;
            for (double aa : {a, a * 0.9})
                rates = rates && std::abs(provider->density_growth_rate(Species::CDM, k, aa) * (4 + std::log(aa)) -
                                          1) < 1e-10;
        }
    check_local(normal, "density = T k^2 with k in 1/Mpc at the nodes of the tables");
    check_local(velocity, "Newtonian velocity = -T_v k^2 (sign and normalization)");
    check_local(cb, "CDM+baryons weighted by Omegab and OmegaCDM");
    check_local(total, "total matter from the total column");
    check_local(rates, "growth rate (4-point Lagrange in log a) exact for delta cubic in log a");
    const double k = .1;
    check_local(std::abs(provider->primordial_power(k) /
                             (2 * M_PI * M_PI / (k * k * k) * 2.1e-9 * std::pow(k * h / .05, .965 - 1)) -
                         1) < 1e-14,
                "primordial normalization in (Mpc/h)^3");
    check_local(provider->density_gauge() == Gauge::SynchronousCDM && provider->velocity_gauge() == Gauge::Newtonian,
                "density and velocity gauges are separate");

    auto strict = make(13, RangePolicy::Reject);
    rejects([&] { strict->density_transfer(Species::CDM, 3, 1); }, "strict k coverage is enforced");
    rejects([&] { strict->density_transfer(Species::CDM, .1, .001); }, "strict time coverage is enforced");
    rejects([&] { provider->density_transfer(Species::CDM, 0, 1); }, "nonpositive query coordinates are rejected");
    rejects([&] { provider->velocity_divergence_transfer(Species::MassiveNeutrino, .1, 1); },
            "missing velocities are not synthesized from density derivatives");
    auto old_format = make(7, RangePolicy::SourceExtrapolation);
    check_local(!old_format->has_velocity(Species::CDM) && provider->has_velocity(Species::CDM) &&
                    old_format->density_transfer(Species::CDM, .1, .5) == provider->density_transfer(Species::CDM, .1, .5),
                "velocity capabilities reflect available columns");

    {
        std::ofstream p(prefix + ".pk"), t(prefix + ".tk");
        p << "# k P\n";
        t << "# k T\n";
        for (double kk : {.01, .1, 1.}) {
            p << kk << " 100\n";
            t << kk << " -200\n";
        }
    }
    auto growth = [](double) { return .02; };
    auto primordial = [](double) { return 2.; };
    auto power = prepare_linear_input({LinearInputType::PowerSpectrum, prefix + ".pk", 1., h}, growth, primordial);
    bool simple = true;
    for (size_t i = 0; i < power.k.size(); i++) {
        double kk = power.k[i] * h;
        simple = simple && power.power[i] == .02 * .02 * 100 &&
                 power.transfer[i] == std::sqrt(power.power[i] / 2) / (kk * kk);
    }
    check_local(simple, "simple P(k,z=0) works without transfer providers or a Boltzmann solver");
    auto transfer =
        prepare_linear_input({LinearInputType::TransferFunction, prefix + ".tk", 1., h}, growth, primordial);
    check_local(transfer.transfer[0] == -4. && transfer.power[0] == 2. * std::pow(-4. * (.01 * h) * (.01 * h), 2),
                "single-transfer input preserves its sign and normalization");
    auto sampled = prepare_linear_input(
        {LinearInputType::TransferInfoFile, "", 1. / 11, h}, growth, primordial, provider.get());
    bool same = std::abs(sampled.k.front() / provider->range().kmin_hmpc - 1) < 1e-12 &&
                std::abs(sampled.k.back() / provider->range().kmax_hmpc - 1) < 1e-12;
    for (size_t i = 0; i < sampled.k.size(); i++)
        same = same &&
               std::abs(sampled.transfer[i] /
                            (provider->density_over_k_mpc2(Species::CDMBaryon, sampled.k[i], 1. / 11) * .02) -
                        1) < 1e-12 &&
               std::abs(sampled.power[i] / (provider->power(Species::CDMBaryon, sampled.k[i], 1. / 11) * (.02 * .02)) -
                        1) < 1e-12;
    check_local(same, "table preparation samples the provider's CDM+baryon transfer and power");
    auto species = two_species_ic_splines(*provider, [](double, double) { return 1.; }, 9., .25 / .30, .05 / .30);
    check_local(std::abs(species.ratio_cdm(std::log(.1)) - .30 / (.25 + .05 * .8)) < 1e-12 &&
                    std::abs(species.f_cdm(std::log(.1)) - 1) < 1e-12,
                "shared species ratios and relative growth are consistent");
    rejects([&] { two_species_ic_splines(*provider, [](double, double) { return 1.; }, 9., .5, .5); },
            "inconsistent species weights are rejected");
    auto mass = two_species_mass_contrast_today(*provider, 9.);
    check_local(std::abs(.25 * mass.first(std::log(.1)) + .05 * mass.second(std::log(.1))) < 1e-12,
                "mass relative modes preserve weighted matter density");

    for (auto & f : files)
        std::remove(f.c_str());
    for (auto ext : {".pk", ".tk", "_13.info", "_7.info"})
        std::remove((prefix + ext).c_str());
    return FML::TESTS::summary("transfer");
}
