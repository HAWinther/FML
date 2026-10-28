#include <FML/InitialConditions/CAMBTransferProvider.h>
#include <FML/InitialConditions/LinearInput.h>
#include <FML/InitialConditions/SpeciesIC.h>
#include <FML/InitialConditions/InternalTransferLoader.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
using namespace FML::INITIALCONDITIONS;
int failures = 0;
void check(bool ok, const char * label) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << '\n';
    if (!ok)
        failures++;
}
template <class F>
void rejects(F fn, const char * label) {
    bool caught = false;
    try {
        fn();
    } catch (const std::exception &) {
        caught = true;
    }
    check(caught, label);
}
int main() {
    const std::string prefix = "test_transfers_" + std::to_string(FML::ThisTask);
    std::vector<std::string> files;
    {
        std::ofstream info(prefix + ".info");
        info << ". 4\n";
        for (int j = 0; j < 4; j++) {
            const auto name = prefix + std::to_string(j) + ".dat";
            files.push_back(name);
            info << name << " " << j * 10 << "\n";
            std::ofstream out(name);
            out.precision(17);
            for (int i = 0; i < 12; i++) {
                double k = .001 * std::pow(2., i), t = 1e4 / (1 + j * 10.) / (1 + k);
                out << k << " " << t << " " << .8 * t << " 0 0 " << .5 * t << " " << .9 * t << " " << .9 * t << " "
                    << .9 * t << " 0 " << .6 * t << " " << .4 * t << " 0\n";
            }
        }
    }
    auto raw = std::make_shared<FML::FILEUTILS::LinearTransferData>(.05, .25, .01, .05, 2.1e-9, .965, .67);
    raw->read_transfer(prefix + ".info");
    CAMBTransferProvider provider(raw);
    bool exact = true, normal = true, rates = true, velocity = true;
    for (double k : {.003, .03, .3})
        for (double a : {1., 1. / 11, 1. / 21}) {
            exact =
                exact &&
                provider.density_over_k_mpc2(Species::CDMBaryon, k, a) == raw->get_cdm_baryon_transfer_function(k, a) &&
                provider.power(Species::CDMBaryon, k, a) == raw->get_cdm_baryon_power_spectrum(k, a) &&
                provider.power(Species::MassiveNeutrino, k, a) == raw->get_massive_neutrino_power_spectrum(k, a);
            double expected = raw->get_cdm_transfer_function(k, a) * (k * .67) * (k * .67);
            normal = normal && provider.density_transfer(Species::CDM, k, a) == expected &&
                     provider.primordial_power(k) == raw->get_primordial_power_spectrum(k);
            rates = rates && provider.density_growth_rate(Species::CDM, k, a) == raw->get_cdm_growth_rate(k, a);
            velocity =
                velocity &&
                std::abs(provider.velocity_divergence_transfer(Species::CDM, k, a) / (-.6 * expected) - 1) < 1e-12;
        }
    check(exact, "CAMB transfers and legacy power arithmetic are unchanged");
    check(normal, "physical k^2 and primordial normalization use the declared units");
    check(rates, "density derivative prescription is retained");
    check(velocity, "direct Newtonian velocity sign and normalization are explicit");
    check(provider.density_gauge() == Gauge::SynchronousCDM && provider.velocity_gauge() == Gauge::Newtonian,
          "density and velocity gauges are separate");
    CAMBTransferProvider strict(raw, RangePolicy::Reject);
    rejects([&] { strict.density_transfer(Species::CDM, 3, 1); }, "strict k coverage is enforced");
    rejects([&] { strict.density_transfer(Species::CDM, .1, .001); }, "strict time coverage is enforced");
    rejects([&] { provider.density_transfer(Species::CDM, 0, 1); }, "nonpositive query coordinates are rejected");
    rejects([&] { provider.velocity_divergence_transfer(Species::MassiveNeutrino, .1, 1); },
            "missing velocities are not synthesized from density derivatives");
    raw->transfer_col_vcdm = -1;
    check(!provider.has_velocity(Species::CDM), "velocity capabilities reflect available columns");
    raw->transfer_col_vcdm = 10;
    {
        std::ofstream p(prefix + ".pk"), t(prefix + ".tk");
        p << "# k P\n";
        t << "# k T\n";
        for (double k : {.01, .1, 1.}) {
            p << k << " 100\n";
            t << k << " -200\n";
        }
    }
    auto growth = [](double) { return .02; };
    auto primordial = [](double) { return 2.; };
    auto power = prepare_linear_input({LinearInputType::PowerSpectrum, prefix + ".pk", 1., .67}, growth, primordial);
    bool simple = true;
    for (size_t i = 0; i < power.k.size(); i++) {
        double k = power.k[i] * .67;
        simple =
            simple && power.power[i] == .02 * .02 * 100 && power.transfer[i] == std::sqrt(power.power[i] / 2) / (k * k);
    }
    check(simple, "simple P(k,z=0) works without transfer providers or a Boltzmann solver");
    auto transfer =
        prepare_linear_input({LinearInputType::TransferFunction, prefix + ".tk", 1., .67}, growth, primordial);
    check(transfer.transfer[0] == -4. && transfer.power[0] == 2. * std::pow(-4. * (.01 * .67) * (.01 * .67), 2),
          "single-transfer input preserves its sign and normalization");
    auto sampled =
        prepare_linear_input({LinearInputType::TransferInfoFile, "", 1. / 11, .67}, growth, primordial, &provider);
    bool same = true;
    for (size_t i = 0; i < sampled.k.size(); i++)
        same = same && sampled.transfer[i] == raw->get_cdm_baryon_transfer_function(sampled.k[i], 1. / 11) * .02 &&
               sampled.power[i] == raw->get_cdm_baryon_power_spectrum(sampled.k[i], 1. / 11) * (.02 * .02);
    check(same, "shared table preparation retains exact legacy sampled amplitudes");
    auto species = two_species_ic_splines(provider, [](double, double) { return 1.; }, 9., .25 / .30, .05 / .30);
    check(std::abs(species.ratio_cdm(std::log(.1)) - .30 / (.25 + .05 * .8)) < 1e-12 &&
              std::abs(species.f_cdm(std::log(.1)) - 1) < 1e-12,
          "shared species ratios and relative growth are consistent");
    rejects([&] { two_species_ic_splines(provider, [](double, double) { return 1.; }, 9., .5, .5); },
            "inconsistent species weights are rejected");
    auto mass = two_species_mass_contrast_today(provider, 9.);
    check(std::abs(.25 * mass.first(std::log(.1)) + .05 * mass.second(std::log(.1))) < 1e-12,
          "mass relative modes preserve weighted matter density");
#ifndef USE_INTERNAL_COSMOLOGY
    rejects([] { load_internal_transfers({}); },
            "ordinary builds report the optional internal feature without linking a Boltzmann solver");
#endif
    raw.reset();
    check(std::isfinite(provider.power(Species::CDMBaryon, .1, 1.)), "providers retain ownership of transfer data");
    for (auto & f : files)
        std::remove(f.c_str());
    for (auto ext : {".pk", ".tk", ".info"})
        std::remove((prefix + ext).c_str());
    FML::SumOverTasks(&failures);
    return failures ? 1 : 0;
}
