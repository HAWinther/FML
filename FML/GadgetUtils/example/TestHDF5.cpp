#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/GadgetUtils/GadgetUtilsHDF5.h>
#include <FML/Global/Global.h>

#include <cmath>
#include <iostream>
#include <vector>

//=========================================================
// Tests of reading and writing GADGET HDF5 files:
//  1. Write particles of several types (with MPI: one file per task), read them back
//     (keeping only the particles in the local domain) and check everything
//  2. Read the same files with a particle without family: only PartType1 is read
//  3. Write the same dark matter particles as binary and HDF5 GADGET files and check
//     that reading them gives the same particles
// Compile with USE_HDF5=true. Run with or without MPI
//=========================================================

using namespace FML::FILEUTILS::GADGET;

struct ParticleWithFamily {
    double pos[3];
    double vel[3];
    long long int id;
    int family;
    int get_ndim() const { return 3; }
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    long long int get_id() const { return id; }
    void set_id(long long int _id) { id = _id; }
    int get_family() const { return family; }
    void set_family(int f) { family = f; }
};

struct Particle {
    double pos[3];
    double vel[3];
    long long int id;
    int get_ndim() const { return 3; }
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    long long int get_id() const { return id; }
    void set_id(long long int _id) { id = _id; }
};

// The particles are a function of the ID so we can check them after reading
const double boxsize = 500.0;
const double aexp = 0.5;
const double vel_kms_per_unit = 123.0;
double expected_pos(long long id, int idim) { return std::fmod(0.6180339887 * id * (idim + 1) + 0.1 * idim, 1.0); }
double expected_vel(long long id, int idim) { return std::sin(0.37 * id + idim); }
int expected_family(long long id) { return id % 7 == 0 ? 0 : (id % 7 == 1 ? 4 : 1); }

template <class T>
std::vector<T> make_particles(size_t n_per_task, bool families) {
    std::vector<T> part;
    for (size_t i = 0; i < n_per_task; i++) {
        T p;
        p.id = (long long)(FML::ThisTask * n_per_task + i);
        for (int idim = 0; idim < 3; idim++) {
            p.pos[idim] = expected_pos(p.id, idim);
            p.vel[idim] = expected_vel(p.id, idim);
        }
        if constexpr (std::is_same_v<T, ParticleWithFamily>)
            p.family = families ? expected_family(p.id) : 1;
        part.push_back(p);
    }
    return part;
}

int nfailed = 0;
void check(bool ok, const std::string & what) {
    int failed = ok ? 0 : 1;
    FML::SumOverTasks(&failed);
    if (FML::ThisTask == 0)
        std::cout << (failed == 0 ? "PASS " : "FAIL ") << what << "\n";
    if (failed)
        nfailed++;
}

// Check that particles read back are what we wrote and are in the local domain
template <class T>
bool particles_ok(std::vector<T> & part, bool only_dm, double vel_norm) {
    for (auto & p : part) {
        if (only_dm and expected_family(p.id) != 1)
            return false;
        if constexpr (std::is_same_v<T, ParticleWithFamily>)
            if (p.family != expected_family(p.id))
                return false;
        if (not(p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain))
            return false;
        for (int idim = 0; idim < 3; idim++) {
            double dx = std::fabs(p.pos[idim] - expected_pos(p.id, idim));
            dx = std::min(dx, 1.0 - dx);
            if (dx > 1e-6 or std::fabs(p.vel[idim] - expected_vel(p.id, idim) * vel_norm) > 1e-5)
                return false;
        }
    }
    return true;
}

// Read a snapshot given on the command line and print what we read
int read_and_print(const std::string & fileprefix) {
    std::vector<ParticleWithFamily> part;
    GadgetHDF5Reader gr;
    gr.read_gadget(fileprefix, part, 1.5, true, FML::ThisTask == 0);
    size_t ntot = part.size();
    FML::SumOverTasks(&ntot);
    if (FML::ThisTask == 0) {
        std::cout << "Read " << ntot << " particles\n";
        for (size_t i = 0; i < std::min(part.size(), size_t(3)); i++) {
            auto & p = part[i];
            std::cout << "id " << p.id << " type " << p.family << " pos " << p.pos[0] << " " << p.pos[1] << " "
                      << p.pos[2] << " vel " << p.vel[0] << " " << p.vel[1] << " " << p.vel[2] << "\n";
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1)
        return read_and_print(argv[1]);
    const size_t n_per_task = 10000;
    const double pos_norm = boxsize;
    const double vel_norm = vel_kms_per_unit / std::sqrt(aexp); // to sqrt(a) dx/dt in km/s
    const int nfiles = FML::NTasks;
    const std::string folder = "output_testhdf5";
    FML::create_folder(folder);

    //=========================================================
    // 1. Several particle types, one file per task
    //=========================================================
    {
        auto part = make_particles<ParticleWithFamily>(n_per_task, true);
        GadgetHDF5Writer gw;
        gw.write_gadget_single(folder + "/multi." + std::to_string(FML::ThisTask) + ".hdf5",
                               part.data(),
                               part.size(),
                               n_per_task * FML::NTasks,
                               nfiles,
                               aexp,
                               boxsize,
                               0.3,
                               0.7,
                               0.7,
                               pos_norm,
                               vel_norm,
                               {0.1, 0.8, 0.0, 0.0, 0.1, 0.0});
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        GadgetHDF5Reader gr;
        std::vector<ParticleWithFamily> part_read;
        gr.set_chunk_size(1000); // to test reading in chunks
        gr.read_gadget(folder + "/multi", part_read, 1.5, true, false);
        size_t ntot = part_read.size();
        FML::SumOverTasks(&ntot);
        auto header = gr.get_header();
        check(ntot == n_per_task * FML::NTasks, "Read back all the particles (" + std::to_string(ntot) + ")");
        check(particles_ok(part_read, false, vel_kms_per_unit), "Positions, velocities, IDs and types are correct");
        check(std::fabs(header.time - aexp) < 1e-12 and std::fabs(header.BoxSize - boxsize) < 1e-12 and
                  header.num_files == nfiles,
              "Header (Time, BoxSize, NumFilesPerSnapshot) is correct");
        check(header.mass[0] > 0 and std::fabs(header.mass[1] / header.mass[0] - 8.0) < 1e-10,
              "MassTable follows OmegaFamilyOverOmegaM");

        //=========================================================
        // 2. Particle without family only reads PartType1
        //=========================================================
        std::vector<Particle> part_dm;
        gr.read_gadget(folder + "/multi", part_dm, 1.5, true, false);
        size_t ndm = part_dm.size(), ndm_expected = 0;
        for (auto & p : part)
            ndm_expected += expected_family(p.id) == 1;
        FML::SumOverTasks(&ndm);
        FML::SumOverTasks(&ndm_expected);
        check(ndm == ndm_expected and particles_ok(part_dm, true, vel_kms_per_unit),
              "Particle without family reads only PartType1 (" + std::to_string(ndm) + ")");
    }

    //=========================================================
    // 3. Binary and HDF5 give the same particles
    //=========================================================
    {
        auto part = make_particles<Particle>(n_per_task, false);
        GadgetWriter gw;
        gw.write_gadget_single(folder + "/bin." + std::to_string(FML::ThisTask), part.data(), part.size(),
                               n_per_task * FML::NTasks, nfiles, aexp, boxsize, 0.3, 0.7, 0.7, pos_norm, vel_norm);
        GadgetHDF5Writer gw5;
        gw5.write_gadget_single(folder + "/h5." + std::to_string(FML::ThisTask) + ".hdf5", part.data(), part.size(),
                                n_per_task * FML::NTasks, nfiles, aexp, boxsize, 0.3, 0.7, 0.7, pos_norm, vel_norm);
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        std::vector<Particle> a, b;
        GadgetReader gr;
        gr.read_gadget(folder + "/bin", a, 1.5, true, false);
        GadgetHDF5Reader gr5;
        gr5.read_gadget(folder + "/h5", b, 1.5, true, false);
        bool same = a.size() == b.size();
        for (size_t i = 0; same and i < a.size(); i++) {
            same = same and a[i].id == b[i].id;
            for (int idim = 0; idim < 3; idim++)
                same = same and std::fabs(a[i].pos[idim] - b[i].pos[idim]) < 1e-6 and
                       std::fabs(a[i].vel[idim] - b[i].vel[idim]) < 1e-5 * (1.0 + std::fabs(a[i].vel[idim]));
        }
        check(same, "Binary and HDF5 files give the same particles (up to float rounding)");
        auto ha = gr.get_header(), hb = gr5.get_header();
        check(ha.mass[1] == hb.mass[1] and ha.npartTotal[1] == hb.npartTotal[1] and ha.Omega0 == hb.Omega0 and
                  ha.HubbleParam == hb.HubbleParam,
              "Binary and HDF5 headers agree");
        check(is_hdf5_snapshot(folder + "/h5") and not is_hdf5_snapshot(folder + "/bin"),
              "is_hdf5_snapshot detects the format");
    }

    if (FML::ThisTask == 0)
        std::cout << (nfailed == 0 ? "All tests passed\n" : "Some tests FAILED\n");
    return nfailed == 0 ? 0 : 1;
}
