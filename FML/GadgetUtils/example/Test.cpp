#include <FML/FileUtils/FileUtils.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/Global/Global.h>

//=========================================================
// Read an ascii file with particles. Write it as a gadget file
// and then read the file back in and check that we get the same
// result
//=========================================================

struct Particle {
    double pos[3];
    double vel[3];
    int id{0};

    Particle() {}
    Particle(double * x, double * v) {
        for (int idim = 0; idim < 3; idim++)
            pos[idim] = x[idim];
        if (v != nullptr)
            for (int idim = 0; idim < 3; idim++)
                vel[idim] = v[idim];
    }

    //=========================================================
    // Method the GadgetReader needs to access
    //=========================================================
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    int * get_id() { return &id; }
    void set_id(long long int _id) { id = _id; }
};

using GadgetReader = FML::FILEUTILS::GADGET::GadgetReader;
using GadgetWriter = FML::FILEUTILS::GADGET::GadgetWriter;

int main() {

    if (FML::ThisTask == 0)
        std::cout << "Reading particles from file\n";

    //=========================================================
    // Read ascii file with [x,y,z]
    //=========================================================
    const double box = 1024.0;
    const std::string filename = "../../../TestData/particles_B1024.txt";
    const int ncols = 3;
    const int nskip_header = 0;
    const std::vector<int> cols_to_keep{0, 1, 2};
    auto data = FML::FILEUTILS::read_regular_ascii(filename, ncols, cols_to_keep, nskip_header);

    //=========================================================
    // Create particles and scale to [0,1)
    //=========================================================
    std::vector<Particle> part;
    for (auto & pos : data) {
        for (auto & x : pos)
            x /= box;
        part.push_back(Particle(pos.data(), nullptr));
    }
    for (int i = 0; i < part.size(); i++)
        part[i].set_id(i);

    //=========================================================
    // Normalization from units in file to Mpc/h
    //=========================================================
    const double gadget_pos_norm = 1.0;

    //=========================================================
    // Write to file
    //=========================================================
    GadgetWriter gw;
    const int NumFiles = 1;
    const double aexp = 1.0;
    const double Boxsize = box;
    const double OmegaM = 0.3;
    const double OmegaLambda = 0.7;
    const double h = 0.7;
    const double posnorm = 1.0; // 1.0 for Mpc/h, 1000 for kpc/h
    gw.write_gadget_single("test.0", part, part.size(), NumFiles, aexp, Boxsize, OmegaM, OmegaLambda, h, posnorm);

    //=========================================================
    // The number of dimensions
    //=========================================================
    const int ndim = 3;

    //=========================================================
    // Set up reader (if posnorm=1.0, ndim=3 then we don't need to provide these numbers)
    //=========================================================
    GadgetReader g(gadget_pos_norm, ndim);

    //=========================================================
    // Container to store it in. Particles will be added to the back of part
    //=========================================================
    part.clear();

    //=========================================================
    // Read a gadget file and fill the data in part
    //=========================================================
    std::string fileprefix = "test";
    g.read_gadget(fileprefix, part, true);

    //=========================================================
    // The positions are in [0,1]
    //=========================================================
    std::cout << "\nPositions in [0,1]:\n";
    for (int i = 0; i < 10; i++)
        std::cout << part[i].pos[0] << " " << part[i].pos[1] << " " << part[i].pos[2] << "\n";

    //=========================================================
    // The velocities are peculiar in km/s
    //=========================================================
    std::cout << "\nPeculiar velocities in km/s:\n";
    for (int i = 0; i < 10; i++)
        std::cout << part[i].vel[0] << " " << part[i].vel[1] << " " << part[i].vel[2] << "\n";

    //=========================================================
    // Ids
    //=========================================================
    std::cout << "\nIDs:\n";
    for (int i = 0; i < 10; i++)
        std::cout << part[i].id << "\n";
}
