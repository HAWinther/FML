#include <FML/GadgetUtils/GadgetUtils.h>

//=========================================================
// A simple 3D particle
//=========================================================
struct Particle {
    double pos[3];
    double vel[3];
    int id;

    int get_ndim() { return 3; }

    //=========================================================
    // Method the GadgetReader looks for and uses to store
    // data if they are present
    //=========================================================
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    int get_id() { return id; }
    void set_id(long long int _id) { id = _id; }
};

using GadgetReader = FML::FILEUTILS::GADGET::GadgetReader;
using GadgetWriter = FML::FILEUTILS::GADGET::GadgetWriter;

int main() {

    //=========================================================
    // Set up reader (if ndim!=3 then we need to provide this)
    //=Y========================================================
    GadgetReader g;

    // If we have elements stored in a different order or some fields missing set it here
    // std::vector<std::string> fields_in_file = {"POS", "VEL", "ID"};
    // g.set_fields_in_file(fields_in_file);

    //=========================================================
    // Container to store it in. Particles will be added to the back of part
    //=========================================================
    std::vector<Particle> part;

    //=========================================================
    // Read gadget files and fill the data in part
    // only_keep_part_in_domain is for MPI -> only store particles
    // that belong to the current task
    // buffer_factor : If only_keep_part_in_domain then how much more space 
    // to allocate compared to the average among tasks
    // NB: if we go over we will reallocate automatically!
    //=========================================================
    const std::string fileprefix = "../../../TestData/gadget";
    const bool only_keep_part_in_domain = true;
    const double buffer_factor = 1.0;
    const bool verbose = false;
    g.read_gadget(fileprefix, part, buffer_factor, only_keep_part_in_domain, verbose);
    std::cout << "Task " << FML::ThisTask << " has " << part.size() << " particles. Capacity: " << part.capacity()
              << std::endl;

    auto header = g.get_header();
    size_t NumPartTotal = part.size();
    FML::SumOverTasks(&NumPartTotal);

    //=========================================================
    // The positions are in [0,1]
    //=========================================================
    if (FML::ThisTask == 0) {
        std::cout << "\nPositions in [0,1]:\n";
        for (int i = 0; i < 10; i++)
            std::cout << part[i].pos[0] << " " << part[i].pos[1] << " " << part[i].pos[2] << "\n";
    }

    //=========================================================
    // The velocities are peculiar in km/s
    //=========================================================
    if (FML::ThisTask == 0) {
        std::cout << "\nPeculiar velocities in km/s:\n";
        for (int i = 0; i < 10; i++)
            std::cout << part[i].vel[0] << " " << part[i].vel[1] << " " << part[i].vel[2] << "\n";
    }

    //=========================================================
    // Ids
    //=========================================================
    if (FML::ThisTask == 0) {
        std::cout << "\nIDs:\n";
        for (int i = 0; i < 10; i++)
            std::cout << part[i].id << "\n";
    }

    //=========================================================
    // Write gadget files
    // pos_norm: convert from user position to position in [0,box)
    // vel_norm: convert from user velocity to sqrt(a)*dxdt in km/s 
    //=========================================================
    GadgetWriter gw;
    const double scale_factor = header.time;
    const double boxsize = header.BoxSize;
    const double OmegaM = header.Omega0;
    const double OmegaLambda = header.OmegaLambda;
    const double h = header.HubbleParam;
    const int nfiles = FML::NTasks;
    const double pos_norm = boxsize;
    const double vel_norm = 1.0 / std::sqrt(scale_factor);
    gw.write_gadget_single("test." + std::to_string(FML::ThisTask),
                           part.data(),
                           part.size(),
                           NumPartTotal,
                           nfiles,
                           scale_factor,
                           boxsize,
                           OmegaM,
                           OmegaLambda,
                           h,
                           pos_norm,
                           vel_norm);
}
