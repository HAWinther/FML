#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Triangulation/VoronoiTessellation.h>
#include <FML/ExampleData/ExampleData.h>

//=======================================================================================
// The simple particle in 3D with pos, mass and volume
//=======================================================================================
const int NDIM = 3;
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
    // Make particles: a toy clustered box with voids to find (every task makes the same particles). To use your
    // own, read them e.g. with FML::FILEUTILS::loadtxt (ascii [x, y, z])
    //=======================================================================================
    FML::PARTICLE::MPIParticles<Particle> part;

    const double boxsize = 1024.0;
    const auto data = FML::EXAMPLEDATA::clustered_box(50000, boxsize);
    std::vector<Particle> partvec;
    partvec.reserve(data.particles.size());
    for (auto & q : data.particles) {
        Particle p;
        auto * pos = p.get_pos();
        for (int idim = 0; idim < 3; idim++)
            pos[idim] = q.pos[idim] / boxsize;
        partvec.push_back(p);
    }

    part.create(partvec.data(), partvec.size(), partvec.size(), FML::PARTICLE::Input::Replicated);

    //=======================================================================================
    // Do a Delaunay tessellation, compute voronoi volumes, locate density minima (or maximima, see h-file)
    // Assign particles to the local density minima using the Delaunay links and bin up data
    // according to what is defined in the class WatershedBasins. In the end all the info about the groups are on task 0
    // This is basically the ZOBOV void finder without the additional watershed merging of zones
    // (and we use the Delaunay tessellation instead of the Voronoi tessellation to propagate
    // the particles to their density minima)
    //=======================================================================================

    // FML's own tessellation (no CGAL). With CGAL (USE_CGAL=true, Triangulation/PeriodicDelaunay.h) the same is
    // FML::TRIANGULATION::WatershedDensity(p, n, watershed_groups, buffer_fraction, random_fraction)
    using WatershedBasin = FML::TRIANGULATION::WatershedBasin<Particle, NDIM>;
    std::vector<WatershedBasin> watershed_groups;
    FML::TRIANGULATION::WatershedDensityNative(part.get_particles_ptr(), part.get_npart(), watershed_groups);

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
            double radius = std::pow(3.0 * volume / (4.0 * M_PI), 1.0 / 3.0) * boxsize;
            fp << std::setw(6) << i << " ";
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
    // The tessellation itself: the Voronoi volume and the neighbours of the particles (the particles can be on any
    // task; neighbours on other tasks are boundary particles whose values come from their owners)
    // FML::TRIANGULATION::PeriodicVoronoiTessellation<NDIM> d;
    // d.create(part.get_particles_ptr(), part.get_npart());
    // std::vector<double> volumes;
    // d.VoronoiVolume(volumes);
    // for (size_t i = 0; i < part.get_npart(); i++) part[i].set_volume(volumes[i]);
    // d.for_each_neighbor(0, [&](size_t k) { /* k < npart: local particle, else boundary particle k - npart */ });
    // const auto volume_boundary = d.communicate_boundary_values(volumes);
    //
    // The faces (area, centroid, normal), moving points (update) and point location: FML::DELAUNAY::PeriodicVoronoi
    // in Triangulation/PeriodicVoronoi.h (see Triangulation/DELAUNAY.md)
    //=======================================================================================
}
