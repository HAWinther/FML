#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Triangulation/PeriodicDelaunay.h>

//=======================================================================================
// The simple particle in 3D with pos, mass and volume
//=======================================================================================
const int NDIM = CGAL_NDIM;
struct Particle {
    double Pos[NDIM];
    double volume{0.0};
    Particle() = default;
    int get_ndim() { return NDIM; }
    double get_volume() { return volume; }
    void set_volume(double _volume) { volume = _volume; }
    double * get_pos() { return Pos; }
};

int main() {

    //=======================================================================================
    // Read particles from file
    //=======================================================================================
    FML::PARTICLE::MPIParticles<Particle> part;
    std::ifstream fp("../../../TestData/particles_B1024.txt");
    if (!fp)
        exit(1);
    const int ntot = 351265;
    const double boxsize = 1024.0;
    std::vector<Particle> partvec(ntot);
    for (int i = 0; i < ntot; i++) {
        auto * pos = partvec[i].get_pos();
        fp >> pos[0];
        fp >> pos[1];
        fp >> pos[2];
        pos[0] /= boxsize;
        pos[1] /= boxsize;
        pos[2] /= boxsize;
        assert(pos[0] < 1.0 and pos[0] >= 0.0);
        assert(pos[1] < 1.0 and pos[1] >= 0.0);
        assert(pos[2] < 1.0 and pos[2] >= 0.0);
    }
    fp.close();
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
        for (size_t i = 0; i < watershed_groups.size(); i++) {
            double mean_density = double(ntot);
            double mass = watershed_groups[i].mass;
            double volume = watershed_groups[i].volume;
            // double volume_min = watershed_groups[i].volume_min;
            double density_avg = mass / volume;
            double density_min = watershed_groups[i].density_min;
            double delta_avg = density_avg / mean_density - 1.0;
            double delta_min = density_min / mean_density - 1.0;
            double radius = std::pow(3.0 * volume / (4.0 * M_PI), 0.33333) * boxsize;
            std::cout << i << " " << boxsize * watershed_groups[i].pos_barycenter[0] << " "
                      << boxsize * watershed_groups[i].pos_barycenter[1] << " "
                      << boxsize * watershed_groups[i].pos_barycenter[2] << " "
                      << boxsize * watershed_groups[i].pos_min[0] << " " << boxsize * watershed_groups[i].pos_min[1]
                      << " " << boxsize * watershed_groups[i].pos_min[2] << " " << volume << " " << radius << " "
                      << delta_avg << " " << delta_min << " " << watershed_groups[i].ningroup << "\n";
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
