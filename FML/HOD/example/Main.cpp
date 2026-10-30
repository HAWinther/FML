#include <FML/CountsInCells/SpatialStatistics.h>
#include <FML/Global/ChunkedExchange.h>
#include <FML/HOD/HOD.h>
#include <fstream>
#include <iomanip>
#include <iostream>
int main() {
    const double box = 1;
    std::vector<FML::HOD::Halo> halos;
    for (int i = 0; i < 256; ++i)
        if (i % FML::NTasks == FML::ThisTask) {
            FML::HOD::Halo h;
            h.id = 1000000 + i;
            h.mass = 1e13 * (1 + i % 8);
            h.radius = .02;
            h.concentration = 6;
            h.velocity_dispersion = 180;
            h.position = {(i % 8 + .5) / 8, ((i / 8) % 8 + .5) / 8, (i / 64 + .5) / 4};
            halos.push_back(h);
        }
    FML::HOD::PreparedCatalog prepared(std::move(halos));
    FML::HOD::Zheng07 occupation;
    occupation.m1 = 5e13;
    FML::HOD::Options options;
    options.boxsize = box;
    options.seed = 32971;
    auto mock = FML::HOD::populate(prepared, occupation, options);
    // Explicit replication for this small example. The statistics backend does not
    // implicitly gather a distributed tracer catalogue; its query samples are distributed.
    unsigned long long total = mock.galaxies.size();
    FML::SumOverTasks(&total);
    std::vector<FML::HOD::Galaxy> galaxies;
    FML::assert_mpi(total <= galaxies.max_size(), "Example catalogue too large");
    galaxies.reserve(size_t(total));
    std::vector<std::vector<FML::HOD::Galaxy>> outgoing(FML::NTasks, mock.galaxies);
    FML::COMMUNICATION::exchange_records_chunked(outgoing, [&](const FML::HOD::Galaxy * data, size_t n) noexcept {
        galaxies.insert(galaxies.end(), data, data + n);
    });
    std::sort(galaxies.begin(), galaxies.end(), [](const auto & a, const auto & b) {
        return std::tie(a.halo_id, a.ordinal) < std::tie(b.halo_id, b.ordinal);
    });
    using namespace FML::COUNTS_IN_CELLS;
    SpatialStatistics statistics(galaxies.data(), galaxies.size(), box);
    const size_t nq = 10000, start = nq * FML::ThisTask / FML::NTasks, stop = nq * (FML::ThisTask + 1) / FML::NTasks;
    const auto queries = uniform_queries(stop - start, Bounds{}, 9173, start);
    const std::vector<double> radii{.02, .04, .06, .08, .1, .15, .2};
    const auto knn = statistics.knn_cdf(queries, {1, 2, 4, 8}, radii);
    Options counts_options;
    counts_options.histogram_limit = 20;
    const auto counts = statistics.counts_in_cells(queries, radii, counts_options);
    if (FML::ThisTask == 0) {
        std::ofstream galaxy_file("hod_galaxies.txt"), knn_file("knn_cdf.txt"), count_file("counts_summary.txt"),
            pdf_file("counts_pdf.txt");
        if (!galaxy_file || !knn_file || !count_file || !pdf_file)
            throw std::runtime_error("Cannot create example output files");
        galaxy_file
            << std::setprecision(17)
            << "# halo_id ordinal central x y z vx vy vz; positions in box units, velocities in supplied units\n";
        for (const auto & g : galaxies) {
            galaxy_file << g.halo_id << ' ' << g.ordinal << ' ' << g.central;
            for (double x : g.position)
                galaxy_file << ' ' << x;
            for (double v : g.velocity)
                galaxy_file << ' ' << v;
            galaxy_file << '\n';
        }
        knn_file << std::setprecision(17) << "# radius F1 F2 F4 F8; query seed=9173, samples=10000\n";
        count_file << std::setprecision(17)
                   << "# radius mean variance F2 F3 F4 P0 poisson_subtracted_variance delta_underflow_count "
                      "delta_overflow_count\n";
        pdf_file << std::setprecision(17) << "# radius N probability; N=21 denotes the explicit N>20 overflow bin\n";
        for (size_t r = 0; r < radii.size(); ++r) {
            knn_file << radii[r];
            for (const auto & cdf : knn.cdf)
                knn_file << ' ' << cdf[r];
            knn_file << '\n';
            const auto & c = counts.cells[r];
            count_file << c.radius << ' ' << c.mean << ' ' << c.variance;
            for (size_t q = 2; q <= 4; ++q)
                count_file << ' ' << c.factorial_moments[q];
            count_file << ' ' << c.probability[0] << ' ' << c.poisson_subtracted_variance << ' ' << c.density_underflow
                       << ' ' << c.density_overflow << '\n';
            for (size_t n = 0; n < c.probability.size(); ++n)
                pdf_file << c.radius << ' ' << n << ' ' << c.probability[n] << '\n';
        }
        std::cout << "Populated " << total << " galaxies from 256 halos; summarized " << knn.queries
                  << " independent-position volume queries\n";
    }
}
