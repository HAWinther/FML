#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>

//==================================================
// A minimal particle class
//==================================================
const int NDIM = 2;
struct Particle {
    double Pos[NDIM];
    constexpr int get_ndim() const { return NDIM; }
    double * get_pos() { return Pos; }
};

int main() {

    //==================================================
    // Make a regular particle grid with Npart_1D^NDIM
    // particles
    //==================================================
    FML::PARTICLE::MPIParticles<Particle> part;
    const int Npart_1D = 6;
    const double buffer_factor = 2.0;
    part.create_particle_grid(Npart_1D, buffer_factor, FML::xmin_domain, FML::xmax_domain);

    //==================================================
    // Print particles on task 0
    //==================================================
    if (FML::ThisTask == 0)
        for (auto && p : part) {
            auto pos = FML::PARTICLE::GetPos(p);
            std::cout << "Task " << FML::ThisTask << " has x = " << pos[0] << " y = " << pos[1] << "\n";
        }

    // Show info
    part.info();

    //==================================================
    // Alternative ways of looping through all active particles
    // for(auto i = 0; i < part.get_npart(); i++) or
    // for(auto it = part.begin(); it != part.end(); ++it) or
    // for(auto & p : part)
    // Don't do for(auto &p :  part.get_particles()) as this will iterate over *all* particles
    // allocated, even uninitalized buffer particles, which is not what we usually want!
    //==================================================

    //==================================================
    // Change x for some of the particles to trigger communication needs below
    //==================================================
    auto & pptr = part.get_particles();
    int i = 0;
    while (i < 5) {
        auto pos = FML::PARTICLE::GetPos(pptr[i++]);
        pos[0] = FML::uniform_random();
    }

    //==================================================
    // Test writing and reading from file
    // each task will write a output.X file with X = ThisTask
    //==================================================
    // part.dump_to_file("output");
    // part.free();
    // part.load_from_file("output");

    //==================================================
    // Send particles that have crossed boundaries
    //==================================================
    if (FML::ThisTask == 0)
        std::cout << "Communicating particles...\n";
    part.communicate_particles();

    // Show info
    part.info();

    //==================================================
    // Mix it up again
    //==================================================
    pptr = part.get_particles();
    i = 0;
    while (i < 5) {
        auto pos = FML::PARTICLE::GetPos(pptr[i++]);
        pos[0] = FML::uniform_random();
    }

    //==================================================
    // Make MPI particles from already existing particles
    //==================================================
    FML::PARTICLE::MPIParticles<Particle> part2;
    const bool all_tasks_have_the_same_particles = false;
    auto nallocate_local = part.get_npart() * 2;
    part2.create(part.get_particles_ptr(),
                 part.get_npart(),
                 nallocate_local,
                 FML::xmin_domain,
                 FML::xmax_domain,
                 all_tasks_have_the_same_particles);

    //==================================================
    // Print the particles we have
    //==================================================
    if (FML::ThisTask == 0)
        for (auto && p : part2) {
            auto pos = FML::PARTICLE::GetPos(p);
            std::cout << "Task " << FML::ThisTask << " has x = " << pos[0] << " y = " << pos[1] << "\n";
        }
}
