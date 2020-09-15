#include <FML/RamsesUtils/RamsesUtils.h>

//====================================================
// Simple particle type to store the data in
// If you want to save memory and only want to read positions say
// then you can remove the v[] etc. from the class
// and just leave the related set_* methods blank (or
// retuning a nullptr for *get_vel)
//====================================================

template <int NDIM>
class RamsesParticle {
  private:
    double x[NDIM];
    double v[NDIM];
    double mass;
    int id;

  public:
    RamsesParticle() = default;
    ~RamsesParticle() = default;

    int get_ndim() { return NDIM; }

    // If any of these methods are not present then we will ignore it when
    // reading the file. So if you only have say get_pos then we will only
    // store the positions
    double * get_pos() { return x; }
    double * get_vel() { return v; }
    double get_mass() { return mass; };
    void set_mass(double m) { mass = m; };
    int get_id() { return id; };
    void set_id(int i) { id = i; };
    //int get_level() { return 0; };
    //void set_level(int l) { (void)l; };
    //char get_family() { return 1; };
    //void set_family(char f) { (void)f; };
    //char get_tag() { return 0;; };
    //void set_tag(char t) { (void)t; };
};

//====================================================
// Examples on how to use the RamsesReader class
//====================================================

int main() {
    const int NDIM = 3;
    using Particle = RamsesParticle<NDIM>;
    using RamsesReader = FML::FILEUTILS::RAMSES::RamsesReader;

    // Set up the Ramses reader
    // output_number is X in output_0000X
    // keep_only_particles_in_domain means each task only keeps particles in its own domain
    // so after we are done the particles are distributed among tasks
    std::string outfolder = "../../../TestData/";
    const int output_number = 8;
    const bool keep_only_particles_in_domain = true;
    const bool verbose = true;
    RamsesReader reader(outfolder, output_number, keep_only_particles_in_domain, verbose);

    // The fiducial file format is POS,VEL,MASS,ID,LEVEL,FAMILY,TAG, but
    // if the format is different one can set it here and we can also set if we want
    // to store the resulting data or not
    std::vector<std::string> fileformat{"POS", "VEL", "MASS", "ID"};
    reader.set_file_format(fileformat);

    // Read all particles and store it in p
    std::vector<Particle> part;
    reader.read_ramses(part);

    std::cout << FML::ThisTask << " has " << part.size() << " particles\n";

    // Read a single file and store it in q
    // std::vector<Particle> q;
    // reader.read_ramses_single(1, q);
}
