#include <FML/Baryonification/Baryonification.h>
#include <FML/Global/Global.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

struct Particle {
    double pos[3];
    double * get_pos() { return pos; }
    int get_ndim() const { return 3; }
};

int main() {
    using namespace FML::BARYONIFICATION;
    // Comoving Mpc/h and Msun/h. The radius must match the catalogue's M200 definition.
    const HaloProperties halo{1e14, 0.7, 5.0};
    const ModelParameters model;
    const double boxsize = 100.0, mean_density = 8.3e10;
    const auto profiles = make_halo_profiles(halo, model, mean_density);
    const auto mapping = profiles.mapping();
    Baryonifier field({{{50.0, 50.0, 50.0}, mapping}}, boxsize);

    // Sample the original spherical mass distribution. In production use a gravity-only snapshot.
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> uniform(0, 1);
    std::vector<Particle> particles(10000);
    for (auto & p : particles) {
        const double r = profiles.source.radius(uniform(rng) * profiles.source.total_mass());
        const double z = 2 * uniform(rng) - 1, phi = 2 * detail::pi * uniform(rng);
        p = {
            {50 + r * std::sqrt(1 - z * z) * std::cos(phi), 50 + r * std::sqrt(1 - z * z) * std::sin(phi), 50 + r * z}};
    }
    const auto original = particles;
    field.apply(particles.data(), particles.size());

    if (FML::ThisTask == 0) {
        std::ofstream profiles_file("profiles.csv"), particle_file("particles.csv");
        if (!profiles_file || !particle_file)
            throw std::runtime_error("Cannot open example output files");
        profiles_file << std::setprecision(17) << "r,M_source,M_target,M_gas,M_stars,M_collisionless,displacement\n";
        for (double r : profiles.reference.radii())
            profiles_file << r << ',' << profiles.source.mass(r) << ',' << profiles.target.mass(r) << ','
                          << profiles.gas.mass(r) << ',' << profiles.central_stars.mass(r) << ','
                          << profiles.collisionless.mass(r) << ',' << mapping.displacement(r) << '\n';
        particle_file << std::setprecision(17)
                      << "x_original,y_original,z_original,x_corrected,y_corrected,z_corrected\n";
        for (size_t i = 0; i < particles.size(); ++i)
            particle_file << original[i].pos[0] << ',' << original[i].pos[1] << ',' << original[i].pos[2] << ','
                          << particles[i].pos[0] << ',' << particles[i].pos[1] << ',' << particles[i].pos[2] << '\n';
        std::cout << "Wrote profiles.csv and particles.csv (positions only; no hydrodynamical calibration).\n";
    }
}
