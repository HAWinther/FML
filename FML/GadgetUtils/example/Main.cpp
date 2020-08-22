#include <FML/GadgetUtils/GadgetUtils.h>

//=========================================================
// A simple 3D particle
//=========================================================
struct Particle {
    double pos[3];
    double vel[3];
    int id;

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

    //=========================================================
    // Normalization from units in file to Mpc/h
    //=========================================================
    const double gadget_pos_norm = 1.0;

    //=========================================================
    // The number of dimensions
    //=========================================================
    const int ndim = 3;

    //=========================================================
    // Set up reader (if posnorm=1.0, ndim=3 then we don't
    // need to provide these numbers)
    //=Y========================================================
    GadgetReader g(gadget_pos_norm, ndim);

    //=========================================================
    // Container to store it in. Particles will be added to the back of part
    //=========================================================
    std::vector<Particle> part;

    //=========================================================
    // Read a gadget file and fill the data in part
    //=========================================================
    std::string fileprefix = "../../../TestData/gadget";
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

    //=========================================================
    // Write to file
    //=========================================================
    GadgetWriter gw;
    gw.write_gadget_single("test.0", part, part.size(), 1, 1.0, 130.0, 0.3, 0.7, 0.7, 1.0);
}
