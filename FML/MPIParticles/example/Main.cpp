#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>

//==================================================
// An example particle class
//==================================================
const int NDIM = 2;
struct Particle {
    double Pos[NDIM];
    double Vel[NDIM];

    int get_particle_byte_size() { return 2 * sizeof(double) * NDIM; }

    int get_ndim() { return NDIM; }

    double * get_pos() { return Pos; }

    double * get_vel() { return Pos; }

    // Methods needed for commmunication of particle
    void append_to_buffer(char * buffer) {
        // Print what we append just to see what is happening
        std::cout << "Sending x = " << Pos[0] << " from task " << FML::ThisTask << "\n";
        int bytes = sizeof(double) * NDIM;
        std::memcpy(buffer, Pos, bytes);
        buffer += bytes;

        bytes = sizeof(double) * NDIM;
        std::memcpy(buffer, Vel, bytes);
    }

    void assign_from_buffer(char * buffer) {
        int bytes = sizeof(double) * NDIM;
        std::memcpy(Pos, buffer, bytes);
        buffer += bytes;

        bytes = sizeof(double) * NDIM;
        std::memcpy(Vel, buffer, bytes);
        // Print what we assign just to see what is happening
        std::cout << "Recieving x = " << Pos[0] << " at task " << FML::ThisTask << "\n";
    }
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
    // Print particles
    //==================================================
    if (FML::ThisTask == 0)
        for (auto & p : part)
            std::cout << "Task " << FML::ThisTask << " has " << p.get_pos()[0] << " " << p.get_pos()[1] << "\n";

    //==================================================
    // Alternative ways of looping through all active particles
    // for(auto i = 0; i < part.get_npart(); i++) or
    // for(auto it = part.begin(); it != part.end(); it++)
    // Don't do for(auto &p :  part.get_particles()) as this will iterator over *all* particles
    // allocated, even uninitalized buffer particles, which is not what we usually want!
    //==================================================

    //==================================================
    // Change x for some of the particles to trigger communication needs below
    //==================================================
    auto & p = part.get_particles();
    int i = 0;
    while (i < 5) {
        p[i++].Pos[0] = FML::uniform_random();
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

    //==================================================
    // Mix it up again
    //==================================================
    i = 0;
    while (i < 5) {
        p[i++].Pos[0] = FML::uniform_random();
    }

    //==================================================
    // Make MPI particles from already existing particles
    //==================================================
    FML::PARTICLE::MPIParticles<Particle> q;
    const bool all_tasks_have_the_same_particles = false;
    auto nallocate_local = part.get_npart() * 2;
    q.create(part.get_particles_ptr(),
             part.get_npart(),
             nallocate_local,
             FML::xmin_domain,
             FML::xmax_domain,
             all_tasks_have_the_same_particles);

    // Show info about the particles
    q.info();

    //==================================================
    // Print the particles we have
    //==================================================
    if (FML::ThisTask == 0)
        for (auto && ps : q) {
            auto pos = ps.get_pos();
            std::cout << "Task " << FML::ThisTask << " has " << pos[0] << " " << pos[1] << "\n";
        }
}
