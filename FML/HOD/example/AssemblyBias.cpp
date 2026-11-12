#include <FML/Global/Global.h>
#include <FML/Global/MPICounts.h>
#include <FML/HOD/AssemblyBias.h>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char ** argv) {
    using namespace FML::HOD;
    std::string prefix = "hod_assembly";
    HeavisideAssemblyBias bias{.5, .4, .6};
    Options options;
    options.seed = 9183;
    options.boxsize = 500;
    options.central_velocity_bias = .2;
    FML::COMMUNICATION::collective_try(
        [&] {
            if (argc != 1 && argc != 2 && argc != 6)
                throw std::invalid_argument(
                    "Usage: assembly_bias [prefix [central_strength satellite_strength split seed]]");
            if (argc >= 2)
                prefix = argv[1];
            if (argc == 6) {
                bias.central_strength = std::stod(argv[2]);
                bias.satellite_strength = std::stod(argv[3]);
                bias.split = std::stod(argv[4]);
                options.seed = std::stoull(argv[5]);
            }
            (void)bias.decorate_marginal({.5, 1}, .5);
        },
        "Read HOD assembly example options");
    std::array<double, 3> lower{bias.split, bias.central_strength, bias.satellite_strength}, upper = lower;
    FML::MinArrayOverTasks(lower.data(), lower.size());
    FML::MaxArrayOverTasks(upper.data(), upper.size());
    auto min_seed = options.seed, max_seed = options.seed;
    FML::MinOverTasks(&min_seed);
    FML::MaxOverTasks(&max_seed);
    FML::COMMUNICATION::collective_try(
        [&] {
            if (lower != upper || min_seed != max_seed)
                throw std::invalid_argument("HOD example options differ across ranks");
        },
        "Agree HOD assembly example options");
    // A deliberately synthetic calibration: c is uniform on [4,12] at each of four fixed masses.
    // Its analytic conditional percentile is q=(c-4)/8, common to all rank decompositions.
    // This is not an empirical percentile estimate from each rank's halo subset.
    constexpr int bins = 4, per_bin = 1000;
    const std::uint64_t first_id = std::uint64_t(1) << 60;
    std::vector<Halo> halos;
    for (int bin = 0; bin < bins; ++bin)
        for (int j = 0; j < per_bin; ++j) {
            Halo h;
            h.id = first_id + std::uint64_t(bin * per_bin + j);
            if (h.id % FML::NTasks != std::uint64_t(FML::ThisTask))
                continue;
            const double q = (j + .5) / per_bin;
            h.mass = std::pow(10., 12.5 + .5 * bin);
            h.concentration = 4 + 8 * q;
            h.radius = .3 * std::cbrt(h.mass / 1e13);
            h.velocity_dispersion = 100;
            Stream random(83192, h.id, 0x48414C4F);
            for (int d = 0; d < 3; ++d) {
                h.position[d] = options.boxsize * random.uniform();
                h.velocity[d] = 100 * random.normal();
            }
            halos.push_back(h);
        }
    PreparedCatalog prepared(std::move(halos));
    Zheng07 baseline;
    auto model = [&](const Halo & h) { return bias(baseline(h), (h.concentration - 4) / 8); };
    Mock mock;
    FML::COMMUNICATION::collective_try([&] { mock = populate(prepared, model, options); }, "Populate decorated HOD");
    // Explicit reductions for the example's summaries; populate itself has no collectives or gathers.
    std::array<long double, bins * 8> totals{};
    FML::COMMUNICATION::collective_try(
        [&] {
            const std::string suffix = "." + std::to_string(FML::ThisTask) + ".csv";
            std::ofstream means(prefix + "-means" + suffix), galaxies(prefix + "-galaxies" + suffix);
            means << std::setprecision(17)
                  << "halo_id,bin,mass,percentile,baseline_central,baseline_satellites,central,satellites,draw_central,"
                     "draw_satellites\n";
            galaxies << std::setprecision(17) << "halo_id,ordinal,central,x,y,z,vx,vy,vz\n";
            for (size_t i = 0; i < prepared.halos().size(); ++i) {
                const auto & h = prepared.halos()[i];
                const int bin = int((h.id - first_id) / per_bin);
                const auto base = baseline(h);
                const auto decorated = model(h);
                const auto & draw = mock.occupations[i];
                const double q = (h.concentration - 4) / 8;
                means << h.id << ',' << bin << ',' << h.mass << ',' << q << ',' << base.central_probability << ','
                      << base.central_probability * base.satellites_if_central << ',' << decorated.central_probability
                      << ',' << decorated.satellite_mean << ',' << draw.central << ',' << draw.satellites << '\n';
                auto * t = totals.data() + 8 * bin;
                t[0] += 1;
                t[1] += base.central_probability;
                t[2] += base.central_probability * base.satellites_if_central;
                t[3] += decorated.central_probability;
                t[4] += decorated.satellite_mean;
                t[5] += draw.central;
                t[6] += draw.satellites;
                t[7] += q > bias.split;
            }
            for (const auto & g : mock.galaxies) {
                galaxies << g.halo_id << ',' << g.ordinal << ',' << g.central;
                for (double x : g.position)
                    galaxies << ',' << x;
                for (double v : g.velocity)
                    galaxies << ',' << v;
                galaxies << '\n';
            }
            means.close();
            galaxies.close();
            if (!means || !galaxies)
                throw std::runtime_error("Cannot write HOD example shards");
        },
        "Write HOD assembly example shards");
    FML::SumArrayOverTasks(totals.data(), totals.size());
    FML::COMMUNICATION::collective_try(
        [&] {
            if (!FML::ThisTask) {
                std::ofstream summary(prefix + "-summary.csv"), config(prefix + "-config.txt");
                summary << std::setprecision(17)
                        << "bin,halos,baseline_centrals,baseline_satellites,decorated_centrals,decorated_satellites,"
                           "drawn_centrals,drawn_satellites,high_halos\n";
                for (int bin = 0; bin < bins; ++bin) {
                    summary << bin;
                    for (int j = 0; j < 8; ++j)
                        summary << ',' << totals[8 * bin + j];
                    summary << '\n';
                }
                config << std::setprecision(17) << "synthetic-uniform-concentration-v1\nseed=" << options.seed
                       << "\nsplit=" << bias.split << "\ncentral_strength=" << bias.central_strength
                       << "\nsatellite_strength=" << bias.satellite_strength
                       << "\nrequire_central=" << options.require_central << "\nboxsize=" << options.boxsize << "\n";
                summary.close();
                config.close();
                if (!summary || !config)
                    throw std::runtime_error("Cannot write HOD example summary");
                std::cout << "Wrote rank-local means/galaxies and global fixed-mass summaries. Population first "
                             "moments are conserved in expectation.\n"
                          << "Percentiles are calibrated analytically for this synthetic catalogue, not for a "
                             "cosmological halo sample.\n";
            }
        },
        "Write HOD assembly example summary");
}
