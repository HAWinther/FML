#include <FML/InitialConditions/InternalTransferProvider.h>
#include <FML/InitialConditions/LinearInput.h>
#include <FML/InitialConditions/SpeciesIC.h>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace FML::INITIALCONDITIONS;
int main(int argc, char ** argv) {
    FML::COSMOLOGY::Utils::verbose = false;
    FML::COSMOLOGY::RunSettings settings;
    settings.cosmo.h = .67;
    settings.cosmo.OmegaB = .05;
    settings.cosmo.OmegaCDM = .267;
    settings.cosmo.m_nu = 0;
    settings.cosmo.OmegaK = 0;
    settings.cosmo.A_s = 2.1e-9;
    settings.cosmo.n_s = .965;
    settings.cosmo.Yp = .245;
    settings.pert.k_min = 1e-5 / FML::COSMOLOGY::Constants::Mpc;
    settings.pert.k_max = .2 / FML::COSMOLOGY::Constants::Mpc;
    settings.pert.n_k = 180;
    settings.pert.n_x_late = 1000;
    settings.rec.n_points = 20000;
    InternalTransferProvider synchronous(settings);
    auto native = synchronous.with_density_gauge(Gauge::Newtonian);
    int failures = 0;
    auto check = [&](bool ok, const char * label) {
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << '\n';
        if (!ok)
            failures++;
    };
    bool converted = true, velocity = true, weighted = true, power = true, reference_matches = true;
    std::ifstream reference(FML::COSMOLOGY::Utils::fml_file(
        "Cosmology/EinsteinBoltzmannSolver/tests/reference/transfer_provider_camb.txt"));
    std::string header;
    std::getline(reference, header);
    std::getline(reference, header);
    auto bg = FML::COSMOLOGY::BackgroundCosmology(settings.cosmo);
    bg.solve();
    std::ofstream output;
    if (argc > 1) {
        output.open(argv[1]);
        output << std::setprecision(17);
    }
    for (double a : {1., 1. / 10, 1. / 50})
        for (double k : {.001, .01, .1}) {
            const double q =
                bg.Hp_of_x(std::log(a)) / (FML::COSMOLOGY::Constants::c * k * .67 / FML::COSMOLOGY::Constants::Mpc);
            const double theta = native.velocity_divergence_transfer(Species::CDM, k, a);
            const double deltaC = synchronous.density_transfer(Species::CDM, k, a),
                         deltaB = synchronous.density_transfer(Species::Baryon, k, a);
            for (auto s : {Species::CDM, Species::Baryon}) {
                const double expected = native.density_transfer(s, k, a) + 3 * q * q * theta;
                converted = converted && std::abs(synchronous.density_transfer(s, k, a) - expected) <
                                             1e-10 * std::max(1., std::abs(expected));
                velocity = velocity && native.velocity_divergence_transfer(s, k, a) ==
                                           synchronous.velocity_divergence_transfer(s, k, a);
            }
            const double cb = (.267 * deltaC + .05 * deltaB) / .317;
            weighted = weighted && std::abs(cb / synchronous.density_transfer(Species::CDMBaryon, k, a) - 1) < 1e-13;
            power = power && synchronous.power(Species::CDMBaryon, k, a) == cb * cb * synchronous.primordial_power(k);
            double rk, ra, rc, rb, rtC, rtB;
            if (!(reference >> rk >> ra >> rc >> rb >> rtC >> rtB))
                reference_matches = false;
            else
                reference_matches =
                    reference_matches && std::abs(rk / k - 1) < 1e-12 && std::abs(ra / a - 1) < 1e-12 &&
                    std::abs(deltaC / rc - 1) < 1e-3 && std::abs(deltaB / rb - 1) < 1e-3 &&
                    std::abs(theta / rtC - 1) < 1e-3 &&
                    std::abs(synchronous.velocity_divergence_transfer(Species::Baryon, k, a) / rtB - 1) < 1e-3;
            if (output)
                output << k << " " << a << " " << deltaC << " " << deltaB << " " << theta << " "
                       << synchronous.velocity_divergence_transfer(Species::Baryon, k, a) << " "
                       << synchronous.density_growth_rate(Species::CDM, k, a) << '\n';
        }
    check(reference_matches,
          "gauge-converted densities and direct velocities agree with independent CAMB values within 0.1%");
    check(converted, "CDM-rest-frame conversion uses the CDM time shift for every species");
    check(velocity, "changing the density gauge leaves explicitly Newtonian velocities unchanged");
    check(weighted, "combined density preserves background species weights");
    check(power, "converted spectra preserve primordial normalization");
    bool wrong = false;
    try {
        prepare_linear_input(
            {LinearInputType::TransferInfoFile, "", 1., .67},
            [](double) { return 1.; },
            [](double) { return 1.; },
            &native);
    } catch (const std::invalid_argument &) {
        wrong = true;
    }
    check(wrong, "native Newtonian density cannot enter the legacy IC prescription silently");
    bool range = false;
    try {
        synchronous.density_transfer(Species::CDM, 10, 1);
    } catch (const std::out_of_range &) {
        range = true;
    }
    check(range, "internal transfer range is strict");
    auto sampled = prepare_linear_input(
        {LinearInputType::TransferInfoFile, "", 1. / 50, .67},
        [](double) { return 1.; },
        [&](double k) { return synchronous.primordial_power(k); },
        &synchronous);
    check(sampled.k.size() == 1000 && std::abs(sampled.k.back() / synchronous.range().kmax_hmpc - 1) < 1e-12,
          "converted IC table covers both endpoints without extrapolation");
    bool noNu = !synchronous.has_density(Species::MassiveNeutrino);
    check(noNu, "unsupported massive-neutrino ICs are not advertised");
    auto species = two_species_ic_splines(synchronous, [](double, double) { return 1.; }, 49., .267 / .317, .05 / .317);
    check(std::isfinite(species.f_baryon(std::log(.01))), "converted provider prepares separate CDM/baryon ICs");
    const double before = synchronous.power(Species::CDMBaryon, .01, 1.),
                 density_before = synchronous.density_transfer(Species::CDMBaryon, .01, 1.);
    synchronous.rescale_primordial_power(4.);
    check(std::abs(synchronous.power(Species::CDMBaryon, .01, 1.) / before - 4.) < 1e-12 &&
              synchronous.density_transfer(Species::CDMBaryon, .01, 1.) == density_before,
          "primordial rescaling changes power without changing unit-curvature transfers");
    settings.cosmo.m_nu = .1;
    bool rejected = false;
    try {
        InternalTransferProvider bad(settings);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "unsupported cosmologies fail before solving");
    return failures ? 1 : 0;
}
