// End-to-end check of SurveyMocks.h with the Survey analysis: a Zel'dovich (1LPT) box at z = 0.1 with velocities
// f a H Psi, a mock (half the sky, 0.05 < z < 0.12, constant n(z)) with randoms, P0/P2/P4 with SurveyPower, and the
// linear Kaiser model (b = 1) convolved with the window of the randoms (SurveyWindow, with the integral constraint).
// The Zel'dovich damping of the measured power is a few percent at k = 0.1 h/Mpc. The particles start on a lattice
// (no Poisson noise), so the mock's shot noise is 1/nbar - 1/n_parent, not the 1/nbar the estimator subtracts: the
// 1/n_parent is added back to the measured P0 below.
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/Survey/SurveyMocks.h>
#include <FML/Survey/SurveyPower.h>
#include <FML/Survey/SurveyWindow.h>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace FML::SURVEY;

struct Particle {
    double Pos[3], Vel[3];
    long long id;
    double * get_pos() { return Pos; }
    double * get_vel() { return Vel; }
    long long get_id() const { return id; }
    void set_id(long long i) { id = i; }
    constexpr int get_ndim() const { return 3; }
};

int main() {
    const double box = 1000.0, z_eff = 0.1, Om = 0.3;
    const int npart_1d = 128;
    auto E = [Om](double z) { return std::sqrt(Om * std::pow(1 + z, 3) + 1 - Om); };
    const double a = 1 / (1 + z_eff), f = std::pow(Om * std::pow(1 + z_eff, 3) / (E(z_eff) * E(z_eff)), 0.55);
    // Linear P(k) at z_eff: BBKS shape (Gamma = 0.2), P(0.1 h/Mpc) = 2000 (Mpc/h)^3 (mildly nonlinear in 1LPT)
    auto T = [](double k) {
        const double q = k / 0.2;
        return std::log(1 + 2.34 * q) / (2.34 * q) *
               std::pow(1 + 3.89 * q + std::pow(16.1 * q, 2) + std::pow(5.46 * q, 3) + std::pow(6.71 * q, 4), -0.25);
    };
    const double norm = 2000.0 / (0.1 * T(0.1) * T(0.1));
    auto Plin = [&](double k) { return k > 0 ? norm * k * T(k) * T(k) : 0.0; };

    // Zel'dovich particles (positions in [0, 1), velocities in km/s)
    FML::PARTICLE::MPIParticles<Particle> part;
    auto rng = std::make_shared<FML::RANDOM::RandomGenerator>();
    rng->set_seed(123);
    FML::NBODY::NBodyInitialConditions<3, Particle>(
        part,
        npart_1d,
        1.5,
        npart_1d,
        false,
        rng.get(),
        [](double) { return 1.0; },
        [&](double kB) { return Plin(kB / box) / (box * box * box); },
        1,
        "gaussian",
        0.0,
        box,
        z_eff,
        {100.0 * box * E(z_eff) * a * f});

    // The mock: observer in the middle, the northern half of the sky, 0.05 < z < 0.12, n = 1e-3 (h/Mpc)^3
    using namespace FML::SURVEY::MOCKS;
    const auto objects = objects_from_particles(part.get_particles_ptr(), part.get_npart(), box, 1.0);
    MockOptions o;
    o.boxsize = box;
    o.observer = {box / 2, box / 2, box / 2};
    o.z_min = 0.05;
    o.z_max = 0.12;
    o.hubble_over_c = [&](double z) { return E(z) / (speed_of_light_kms / 100); };
    o.footprint = ra_dec_box(0, 360, 0, 90);
    const double nbar = 1e-3;
    o.target_nbar = [nbar](double) { return nbar; };
    o.random_factor = 10;
    const auto mock = make_survey(objects, o);
    const auto ran = make_randoms(o, mock.parent_nbar);
    auto data = to_catalog_points<CatalogPoint>(mock.galaxies);
    auto randoms = to_catalog_points<CatalogPoint>(ran);
    const double P0_fkp = 5000;
    for (auto * v : {&data, &randoms})
        for (auto & p : *v)
            p.weight = fkp_weight(p.nbar, P0_fkp);

    // P_ell (local line of sight) and the window-convolved Kaiser model
    Geometry g;
    g.nmesh = 128;
    g.boxsize = 1600; // the window needs the survey (diameter 700 Mpc/h) below half the mesh box
    g.minimum = {-800, -800, -800};
    PowerOptions po;
    for (int i = 0; i <= 10; i++)
        po.edges.push_back(0.01 * i);
    const auto P = estimate_power(data, randoms, g, po);
    std::vector<double> sep;
    for (int i = 0; i <= 150; i++)
        sep.push_back(5.0 * i);
    const auto W = estimate_window(randoms, g, sep, P.alpha, P.i2);
    std::vector<double> in_edges;
    for (int i = 0; i <= 120; i++)
        in_edges.push_back(0.0025 * i);
    const auto M = make_binned_window_matrix(W, in_edges, po.edges);
    std::vector<double> theory;
    for (int ell : {0, 2, 4})
        for (size_t b = 0; b + 1 < in_edges.size(); b++) {
            const double pk = Plin(0.5 * (in_edges[b] + in_edges[b + 1]));
            theory.push_back(ell == 0 ? (1 + 2 * f / 3 + f * f / 5) * pk :
                             ell == 2 ? (4 * f / 3 + 4 * f * f / 7) * pk :
                                        8 * f * f / 35 * pk);
        }
    const auto conv = M.apply(theory);
    double nd = double(data.size());
    FML::SumOverTasks(&nd);
    if (FML::ThisTask == 0) {
        std::cout << "\n# Mock: " << nd << " galaxies; f = " << f << "\n";
        std::cout << "# k [h/Mpc]   P0 measured (+1/n_parent) / model   P2 measured / model\n";
        const size_t nb = po.edges.size() - 1;
        for (size_t b = 0; b < nb; b++)
            std::cout << std::setw(8) << P.k[b] << std::setw(12) << P.power[0][b] + 1 / mock.parent_nbar
                      << std::setw(12) << conv[b] << std::setw(12) << P.power[1][b] << std::setw(12) << conv[nb + b]
                      << "\n";
    }
}
