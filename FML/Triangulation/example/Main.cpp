#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Triangulation/PeriodicDelaunay.h>
#include <FML/FileUtils/FileUtils.h>

//=======================================================================================
// The simple particle in 3D with pos, mass and volume
//=======================================================================================
const int NDIM = CGAL_NDIM;
struct Particle {
    double Pos[NDIM];
    double volume{0.0};
    Particle() = default;
    constexpr int get_ndim() const { return NDIM; }
    double get_volume() const { return volume; }
    void set_volume(double _volume) { volume = _volume; }
    double * get_pos() { return Pos; }
};

int main() {

    //=======================================================================================
    // Read particles from file
    //=======================================================================================
    FML::PARTICLE::MPIParticles<Particle> part;
    
    const double boxsize = 1024.0;
    std::string filename = "../../../TestData/particles_B1024.txt";
    auto data = FML::FILEUTILS::loadtxt(filename);
    std::vector<Particle> partvec;
    partvec.reserve(data.size());
    for(auto & row : data){
      Particle p;
      auto * pos = p.get_pos();
      pos[0] = row[0] / boxsize;
      pos[1] = row[1] / boxsize;
      pos[2] = row[2] / boxsize;
      assert(pos[0] < 1.0 and pos[0] >= 0.0);
      assert(pos[1] < 1.0 and pos[1] >= 0.0);
      assert(pos[2] < 1.0 and pos[2] >= 0.0);
      partvec.push_back(p);
    }

    part.create(partvec.data(), partvec.size(), partvec.size(), FML::xmin_domain, FML::xmax_domain, true);

    //=======================================================================================
    // Do a Delaunay tesselation, compute voronoi volumes, locate density minima (or maximima, see h-file)
    // Assign particles to the local density minima using the Delaunay links and bin up data
    // according to what is defined in the class WatershedBasins. In the end all the info about the groups are on task 0
    // This only works in 3D currently, the missing piece is to compute voronoi volumes (i.e. area)
    // from the tesselation in 2D
    //
    // This is basically the ZOBOV void finder without the additional watershed merging of zones
    // (and we use the Delaunay tesselation instead of the Voronoi tesselation to propagate
    // the particles to their density minima)
    //=======================================================================================

    const double random_fraction = 0.33;
    const double buffer_fraction = 0.33;
    using WatershedBasin = FML::TRIANGULATION::WatershedBasin<Particle, NDIM>;
    std::vector<WatershedBasin> watershed_groups;
    FML::TRIANGULATION::WatershedDensity(
        part.get_particles_ptr(), part.get_npart(), watershed_groups, buffer_fraction, random_fraction);

    // Output the resulting (what is here basically a zobov void) catalogue
    if (FML::ThisTask == 0) {
        std::ofstream fp("groups.txt");
        for (size_t i = 0; i < watershed_groups.size(); i++) {
            double mean_density = double(partvec.size());
            double mass = watershed_groups[i].mass;
            double volume = watershed_groups[i].volume;
            // double volume_min = watershed_groups[i].volume_min;
            double density_avg = mass / volume;
            double density_min = watershed_groups[i].density_min;
            double delta_avg = density_avg / mean_density - 1.0;
            double delta_min = density_min / mean_density - 1.0;
            double radius = std::pow(3.0 * volume / (4.0 * M_PI), 0.33333) * boxsize;
            fp << std::setw(6)  << i << " ";
            fp << std::setw(10) << boxsize * watershed_groups[i].pos_barycenter[0] << " ";
            fp << std::setw(10) << boxsize * watershed_groups[i].pos_barycenter[1] << " ";
            fp << std::setw(10) << boxsize * watershed_groups[i].pos_barycenter[2] << " ";
            fp << std::setw(10) << boxsize * watershed_groups[i].pos_min[0] << " ";
            fp << std::setw(10) << boxsize * watershed_groups[i].pos_min[1] << " ";
            fp << std::setw(10) << boxsize * watershed_groups[i].pos_min[2] << " ";
            fp << std::setw(15) << volume << " ";
            fp << std::setw(10) << radius << " ";
            fp << std::setw(15) << delta_avg << " ";
            fp << std::setw(15) << delta_min << " ";
            fp << std::setw(10) << watershed_groups[i].ningroup << "\n";
        }
    }

    // Other examples:

    //=======================================================================================
    // Compute tesselation from MPIParticles and get volume of the particles
    // FML::TRIANGULATION::MPIPeriodicDelaunay<Particle> d;
    // const double random_fraction = 0.5;
    // const double buffer_fraction = 1.0;
    // d.create(part.get_particles_ptr(), part.get_npart(), buffer_fraction, random_fraction);
    // // Compute Voronoi volumes
    // std::vector<double> volumes;
    // d.VoronoiVolume(volumes);
    // // Assign them to particles
    // for(auto & p : part) p.set_volume(volumes.pop_front());
    //=======================================================================================

    //=======================================================================================
    // More general: assign data to the vertices when doing the tesselation.
    // Define a vertex data struct and give the function the assigns the data.
    // Note that p might be null for which we have a boundary/random particle. Example:
    // struct VertexData{ double mass; };
    // FML::TRIANGULATION::MPIPeriodicDelaunay<struct VertexData> d;
    // std::function<void(struct VertexData *, Particle *)> vertex_assignment_function = [](VertexData *v, Particle *p){
    //    if(p) v->mass = p->get_mass();
    // };
    // d.create(part.get_particles_ptr(), part.get_npart(), buffer_fraction, random_fraction,
    // vertex_assignment_function);
    // // We now have the mass assigned to the vertices of the tesselation and can do stuff with it
    //=======================================================================================
}
