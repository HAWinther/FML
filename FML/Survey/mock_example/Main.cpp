// A survey mock from a (here uniform, unclustered) periodic box: an RA-Dec footprint, a target n(z), redshift-space
// distortions from the velocities, and randoms. Writes mock_data.txt and mock_randoms.txt (one file per MPI task).
// With a simulation: fill the MockObject list with objects_from_particles or objects_from_hod (SurveyMocks.h).
#include <FML/Global/Global.h>
#include <FML/Survey/SurveyMocks.h>

#include <cmath>
#include <iostream>
#include <random>

using namespace FML::SURVEY::MOCKS;

int main() {
    // Every task owns a different set of objects (positions in [0, L) Mpc/h, velocities in km/s, unique ids)
    const double L = 1000.0;
    const size_t n_local = 200000;
    std::mt19937_64 g(1234 + FML::ThisTask);
    std::uniform_real_distribution<double> u(0.0, L);
    std::normal_distribution<double> v(0.0, 300.0);
    std::vector<MockObject> objects(n_local);
    for (size_t i = 0; i < n_local; i++) {
        objects[i].position = {u(g), u(g), u(g)};
        objects[i].velocity = {v(g), v(g), v(g)};
        objects[i].id = std::uint64_t(FML::ThisTask) * n_local + i;
    }

    MockOptions o;
    o.boxsize = L;
    o.observer = {0.5 * L, 0.5 * L, 0.5 * L};
    o.z_min = 0.05;
    o.z_max = 0.2;
    o.hubble_over_c = [](double z) { return std::sqrt(0.3 * std::pow(1 + z, 3) + 0.7) / 2997.92458; }; // h/Mpc
    o.footprint = ra_dec_box(120, 240, 0, 60);
    o.target_nbar = [](double z) { return 5e-5 * (1 + 5 * z); }; // (h/Mpc)^3
    o.random_factor = 20;

    const auto data = make_survey(objects, o);
    const auto randoms = make_randoms(o, data.parent_nbar);
    double nd = double(data.galaxies.size()), nr = double(randoms.size());
    FML::SumOverTasks(&nd);
    FML::SumOverTasks(&nr);
    if (FML::ThisTask == 0)
        std::cout << "Mock: " << nd << " galaxies, " << nr << " randoms (parent density " << data.parent_nbar
                  << " (h/Mpc)^3" << (data.repeated ? ", the box is repeated" : "") << ")\n";
    write_catalogue("mock_data.txt", data.galaxies);
    write_catalogue("mock_randoms.txt", randoms);
    // For SurveyPower/SurveyWindow/SurveyReconstruction (SurveyMesh.h):
    //   auto points = to_catalog_points<FML::SURVEY::CatalogPoint>(data.galaxies);
}
