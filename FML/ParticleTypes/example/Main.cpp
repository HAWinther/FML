#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <iostream>
#include <random>

template <int NDIM>
class TestParticle {
    double pos[NDIM];
    double vel[NDIM];
    size_t id = 10;
   
    //double mass = 2.0;

  public:
    
    constexpr int get_ndim() { return NDIM; }
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    size_t get_id() { return id; }
    void set_id(size_t _id) { id = _id; }

    // If the mass is not set here then the fiducial value 1 is used by algorithms
    //double get_mass() { return mass; }
    //void set_mass(double _mass) { mass = _mass; }
};

int main() {
    using Particle = TestParticle<3>;

    Particle p;

    // Show info about the particle
    FML::PARTICLE::info<Particle>();

    std::cout << "The dimension of the particle is: " << FML::PARTICLE::GetNDIM(p) << "\n";
    std::cout << "The size of the particle is: " << FML::PARTICLE::GetSize(p) << " bytes\n";

    // Check if we have get_pos in the class
    if constexpr (FML::PARTICLE::has_get_pos<Particle>()) {
        auto pos = FML::PARTICLE::GetPos(p);
        std::cout << "Particle class has position x = " << pos[0] << "\n";
    } else {
        std::cout << "Particle class do not have position\n";
    }

    // Check if we have get_vel in the class
    if constexpr (FML::PARTICLE::has_get_vel<Particle>()) {
        auto vel = FML::PARTICLE::GetVel(p);
        std::cout << "Particle class has velocity vx = " << vel[0] << "\n";
    } else {
        std::cout << "Particle class do not have velocity\n";
    }

    // Check if we have set_id and get_id in the class
    if constexpr (FML::PARTICLE::has_get_id<Particle>() and FML::PARTICLE::has_set_id<Particle>()) {
        auto id = FML::PARTICLE::GetID(p);
        std::cout << "Particle class has id = " << id << "\n";
    } else {
        std::cout << "Particle class do not have position\n";
    }

    // Check if we have set_mass and get_mass in the class
    if constexpr (FML::PARTICLE::has_get_mass<Particle>() and FML::PARTICLE::has_set_mass<Particle>()) {
        std::cout << "Particle class has mass\n";
    } else {
        std::cout << "Particle class do not have mass\n";
    }

    //... but GetMass exist even if we don't define it (fiducial value = 1.0)
    auto mass = FML::PARTICLE::GetMass(p);
    std::cout << "Mass of particle: " << mass << "\n";
}
