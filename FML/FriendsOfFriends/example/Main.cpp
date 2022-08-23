#include <FML/FileUtils/FileUtils.h>
#include <FML/FriendsOfFriends/FoF.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ParticleTypes/SimpleParticle.h>

#include <fstream>
#include <vector>

//==================================================================
// Set up a simple particle compatible with MPIParticles
//==================================================================

template <int NDIM>
struct Particle {
    double x[NDIM];
    Particle() = default;
    Particle(double * _x) { std::memcpy(x, _x, NDIM * sizeof(double)); }
    constexpr int get_ndim() const { return NDIM; }
    double * get_pos() { return x; }

    // If you want the algorithm to store the FoFID in the particle add these methods
    // For particles not belonging to a group we set it equal to FML::FOF::no_FoF_ID (which is SIZE_MAX)
    // size_t FoFID;
    // void set_fofid(size_t _FoFID){ FoFID = _FoFID; }
    // size_t get_fofid(){ return FoFID; }
};

int main() {
    const int NDIM = 3;

    if (FML::ThisTask == 0)
        std::cout << "Reading particles from file\n";

    //==================================================================
    // Read ascii file with [x,y,z]
    //==================================================================
    const double box = 1024.0;
    const std::string filename = "../../../TestData/particles_B1024.txt";
    const int ncols = 3;
    const int nskip_header = 0;
    const std::vector<int> cols_to_keep{0, 1, 2};
    auto data = FML::FILEUTILS::read_regular_ascii(filename, ncols, cols_to_keep, nskip_header);

    // Create particles and scale to [0,1)
    std::vector<Particle<NDIM>> part;
    part.reserve(data.size());
    for (auto & pos : data) {
        for (auto & x : pos) {
            x /= box;
            if (x < 0.0)
                x += 1.0;
            if (x >= 1.0)
                x -= 1.0;
        }
        if (pos[0] >= FML::xmin_domain and pos[0] < FML::xmax_domain)
            part.push_back(Particle<NDIM>(pos.data()));
    }

    //==================================================================
    // Create MPI particles by letting each task keep only the particles that falls in its domain
    //==================================================================
    FML::PARTICLE::MPIParticles<Particle<NDIM>> p;
    const bool all_tasks_have_the_same_particles = false;
    const auto nalloc_per_task = not all_tasks_have_the_same_particles ? part.size() : part.size() / FML::NTasks * 2;
    p.create(part.data(),
             part.size(),
             nalloc_per_task,
             FML::xmin_domain,
             FML::xmax_domain,
             all_tasks_have_the_same_particles);
    p.info();

    //==================================================================
    // Select the class which determines what data we bin up over the particles
    // This is the fiducial one which is just position and velocity (if the particle has it)
    //==================================================================
    using FoFHalo = FML::FOF::FoFHalo<Particle<NDIM>, NDIM>;

    //==================================================================
    // Do Friend of Friend finding
    //==================================================================
    const double linking_length = 0.30;
    const int n_min_FoF_group = 20;
    const bool periodic_box = true;
    const double Buffersize_over_Boxsize = 3.0/box;
    const int Ngrid_max = 512;
    std::vector<FoFHalo> LocalFoFGroups;
    FML::FOF::FriendsOfFriends<Particle<NDIM>, NDIM>(
        p.get_particles_ptr(), 
        p.get_npart(), 
        linking_length, 
        n_min_FoF_group, 
        periodic_box, 
        Buffersize_over_Boxsize,
        LocalFoFGroups, 
        Ngrid_max);

    // Mass of particle
    const double NumPartDMtot = 256*256*256;
    const double OmegaM = 0.3;
    const double MplMpl_over_H0Msunh = 2.49264e21;
    const double HubbleLengthInMpch = 2997.92458;
    double massofparticle = 3.0 * OmegaM * MplMpl_over_H0Msunh * std::pow(box / HubbleLengthInMpch, 3) /
      double(NumPartDMtot);

    //==================================================================
    // Output (as currently written task 0 has all the halos at this point)
    // (We don't read velocities to these will just be 0)
    //==================================================================
    // Append to the same file task-by-task
    for(int i = 0; i < FML::NTasks; i++){
      if(i == FML::ThisTask){
        std::ofstream fp("fof.txt", (i == 0 ? std::ios_base::out : std::ios_base::app));
        for(auto & h : LocalFoFGroups){
          fp << std::setw(15) << h.id     << " ";
          fp << std::setw(15) << h.np     << " ";
          fp << std::setw(15) << h.np*massofparticle << " ";
          fp << std::setw(15) << h.pos[0]*box << " ";
          fp << std::setw(15) << h.pos[1]*box << " ";
          fp << std::setw(15) << h.pos[2]*box << " ";
          fp << std::setw(15) << h.vel[0] << " ";
          fp << std::setw(15) << h.vel[1] << " ";
          fp << std::setw(15) << h.vel[2] << " ";
          fp << std::setw(15) << h.vel_rms[0] << " ";
          fp << std::setw(15) << h.vel_rms[1] << " ";
          fp << std::setw(15) << h.vel_rms[2] << " ";
          fp << "\n";
        }
        fp.close();
      }
      MPI_Barrier(MPI_COMM_WORLD);  
    }
}
