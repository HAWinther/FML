//==========================================================
// Tests for FML::FILEUTILS::GADGET (GadgetUtils/GadgetUtils.h, GadgetUtilsHDF5.h):
//   - dark matter: every task writes one file, read back (each task keeps its x-domain): all particles, positions,
//     peculiar velocities (file = v / sqrt(a)) and IDs, the header (totals, mass, time, box)
//   - several particle types in one particle class (family): the MassTable mass of each type
//     (Omega_type rho_crit Box^3 / N_type) and the type of every particle after reading with MPI
//   - IDs larger than 2^31 (8 byte IDs) and small IDs (4 byte IDs as before)
//   - files with few particles (<= 288) and files with no particles
//   - write_gadget_species with per-particle masses: the MASS block is read back (set_mass)
//   - one reader for a big-endian and then a little-endian snapshot, a format 2 (SNAP) file with an extra block
//   - HDF5 (with USE_HDF5): BoxSize with 3 elements (SWIFT), one reader for snapshots with different numbers of
//     files, the MassTable of each type
// Temporary files are written to the current folder and deleted.
// Run with: make run (in this folder) or make test (from the top folder). One test: ./test_gadget <name>
//==========================================================
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/GadgetUtils/GadgetUtilsHDF5.h>
#include <FML/Global/Global.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace FML::FILEUTILS::GADGET;

int fails = 0;
void check(bool ok, std::string what) {
    // All tasks must agree
    int bad = ok ? 0 : 1;
    FML::SumOverTasks(&bad);
    ok = bad == 0;
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os << x;
    return os.str();
}
void barrier() {
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

const double box = 500.0;
const double aexp = 0.25;
const double OmegaM = 0.3;
// rho_crit = 2.77536627e11 h^2 Msun / Mpc^3: the mass of a particle in 1e10 Msun/h
double expected_mass(double fraction, size_t n) { return 27.7536627 * OmegaM * fraction * box * box * box / double(n); }

// The particles are a function of the ID so we can check them after reading
double expected_pos(long long id, int idim) { return std::fmod(0.6180339887 * double(id % 100000 + 1) * (idim + 1) + 0.1 * idim, 1.0); }
double expected_vel(long long id, int idim) { return 100.0 * std::sin(0.37 * double(id % 100000) + idim); } // km/s
int expected_family(long long id) { return id % 5 == 0 ? 0 : (id % 5 == 1 ? 3 : 1); }

struct Particle {
    double pos[3];
    double vel[3];
    long long id;
    constexpr int get_ndim() const { return 3; }
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    long long get_id() const { return id; }
    void set_id(long long _id) { id = _id; }
};
struct ParticleWithFamily : Particle {
    int family{-1};
    double mass{-1.0};
    int get_family() const { return family; }
    void set_family(int f) { family = f; }
    double get_mass() const { return mass; }
    void set_mass(double m) { mass = m; }
};

// Particles [n0, n1) with IDs id_offset + i
template <class T>
std::vector<T> make_particles(size_t n0, size_t n1, long long id_offset = 0) {
    std::vector<T> part;
    for (size_t i = n0; i < n1; i++) {
        T p{};
        p.id = id_offset + (long long)i;
        for (int idim = 0; idim < 3; idim++) {
            p.pos[idim] = expected_pos(p.id, idim);
            p.vel[idim] = expected_vel(p.id, idim);
        }
        if constexpr (std::is_same_v<T, ParticleWithFamily>)
            p.family = expected_family(p.id);
        part.push_back(p);
    }
    return part;
}
// The particles of this task when ntotal are split over the tasks
template <class T>
std::vector<T> make_task_particles(size_t ntotal, long long id_offset = 0) {
    return make_particles<T>(ntotal * FML::ThisTask / FML::NTasks, ntotal * (FML::ThisTask + 1) / FML::NTasks, id_offset);
}

// Check particles read back: the number over all tasks, in the domain, positions, velocities, IDs (sum and range)
template <class T>
bool check_particles(std::vector<T> & part, size_t ntotal, long long id_offset, std::string & info) {
    long long n = part.size(), nbad = 0;
    double maxdpos = 0.0, maxdvel = 0.0, idsum = 0.0;
    for (auto & p : part) {
        if (p.id < id_offset or p.id >= id_offset + (long long)ntotal) {
            nbad++;
            continue;
        }
        if (not(p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain))
            nbad++;
        for (int idim = 0; idim < 3; idim++) {
            double dx = std::fabs(p.pos[idim] - expected_pos(p.id, idim));
            maxdpos = std::max(maxdpos, std::min(dx, 1.0 - dx));
            maxdvel = std::max(maxdvel, std::fabs(p.vel[idim] - expected_vel(p.id, idim)));
        }
        idsum += double(p.id - id_offset);
    }
    FML::SumOverTasks(&n);
    FML::SumOverTasks(&nbad);
    FML::SumOverTasks(&idsum);
    FML::MaxOverTasks(&maxdpos);
    FML::MaxOverTasks(&maxdvel);
    info = "n = " + num(n) + "/" + num(ntotal) + " bad = " + num(nbad) + " max|dx| = " + num(maxdpos) +
           " max|dv| = " + num(maxdvel);
    return (size_t)n == ntotal and nbad == 0 and maxdpos < 1e-6 and maxdvel < 1e-3 and
           idsum == double(ntotal) * double(ntotal - 1) / 2.0;
}

// Raw binary GADGET writing (independent of the writer) for big-endian and format 2 files
template <class U>
U to_endian(U u, bool big) {
    return big ? swap_endian(u) : u;
}
void raw_block(std::ofstream & fp, const std::vector<char> & data, bool big) {
    int n = to_endian(int(data.size()), big);
    fp.write((char *)&n, 4);
    fp.write(data.data(), data.size());
    fp.write((char *)&n, 4);
}
void raw_label(std::ofstream & fp, std::string label, size_t bytes) {
    std::vector<char> data(8, ' ');
    std::memcpy(data.data(), label.c_str(), std::min(size_t(4), label.size()));
    int n = int(bytes) + 8;
    std::memcpy(data.data() + 4, &n, 4);
    raw_block(fp, data, false);
}
// One file with all the particles of type 1: 4 byte IDs, big endian or format 2 (with an extra AGE block)
void write_raw_file(std::string filename, size_t n, bool big, bool format2) {
    std::ofstream fp(filename, std::ios::binary);
    GadgetHeader h;
    h.npart[1] = n;
    h.npartTotal[1] = n;
    h.mass[1] = expected_mass(1.0, n);
    h.time = aexp;
    h.redshift = 1.0 / aexp - 1.0;
    h.num_files = 1;
    h.BoxSize = box;
    if (big) {
        swap_endian_vector(h.npart, 6);
        swap_endian_vector(h.npartTotal, 6);
        swap_endian_vector(h.mass, 6);
        h.time = swap_endian(h.time);
        h.redshift = swap_endian(h.redshift);
        h.num_files = swap_endian(h.num_files);
        h.BoxSize = swap_endian(h.BoxSize);
    }
    std::vector<char> data(sizeof(h));
    std::memcpy(data.data(), &h, sizeof(h));
    if (format2)
        raw_label(fp, "HEAD", data.size());
    raw_block(fp, data, big);
    auto part = make_particles<Particle>(0, n);
    std::vector<float> pos, vel, age;
    std::vector<int> ids;
    for (auto & p : part) {
        for (int idim = 0; idim < 3; idim++) {
            pos.push_back(to_endian(float(p.pos[idim] * box), big));
            vel.push_back(to_endian(float(p.vel[idim] / std::sqrt(aexp)), big));
        }
        ids.push_back(to_endian(int(p.id), big));
        age.push_back(1.0f);
    }
    auto block = [&](std::string label, const void * ptr, size_t bytes) {
        if (format2)
            raw_label(fp, label, bytes);
        std::vector<char> d(bytes);
        std::memcpy(d.data(), ptr, bytes);
        raw_block(fp, d, big);
    };
    block("POS ", pos.data(), pos.size() * 4);
    if (format2)
        block("AGE ", age.data(), age.size() * 4);
    block("VEL ", vel.data(), vel.size() * 4);
    block("ID  ", ids.data(), ids.size() * 4);
}

// The size of a block in a binary file (the n'th block, 0 = header)
int block_size(std::string filename, int n) {
    std::ifstream fp(filename, std::ios::binary);
    int bytes = 0;
    for (int i = 0; i <= n; i++) {
        fp.read((char *)&bytes, 4);
        if (i < n)
            fp.seekg(bytes + 4, std::ios::cur);
    }
    return bytes;
}

int main(int argc, char ** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    auto run = [&](std::string name) { return only == "" or only == name; };

    // A unique prefix for the temporary files (the same on all tasks)
    long long pid = FML::ThisTask == 0 ? (long long)getpid() : 0;
    FML::SumOverTasks(&pid);
    const std::string prefix = "tmp_test_gadget_" + std::to_string(pid) + "_";
    const std::string mytask = std::to_string(FML::ThisTask);
    std::vector<std::string> my_files;
    const double vel_norm = 1.0 / std::sqrt(aexp); // peculiar km/s -> GADGET sqrt(a) dx/dt

    //==========================================================
    // Dark matter, one file per task, read back with domain decomposition
    //==========================================================
    if (run("dm")) {
        const size_t ntotal = 3000;
        auto part = make_task_particles<Particle>(ntotal);
        GadgetWriter gw;
        const std::string fileprefix = prefix + "dm";
        gw.write_gadget_single(fileprefix + "." + mytask, part.data(), part.size(), ntotal, FML::NTasks, aexp, box,
                               OmegaM, 0.7, 0.7, box, vel_norm);
        my_files.push_back(fileprefix + "." + mytask);
        barrier();
        std::vector<Particle> read;
        GadgetReader g;
        g.read_gadget(fileprefix, read, 1.5, true, false);
        std::string info;
        bool ok = check_particles(read, ntotal, 0, info);
        check(ok, "DM round trip with " + num(FML::NTasks) + " file(s): " + info);
        auto h = g.get_header();
        check(h.npartTotal[1] == ntotal and h.num_files == FML::NTasks and h.time == aexp and h.BoxSize == box and
                  std::fabs(h.mass[1] / expected_mass(1.0, ntotal) - 1.0) < 1e-5,
              "DM header: npartTotal = " + num(h.npartTotal[1]) + " mass = " + num(h.mass[1]) + " (expected " +
                  num(expected_mass(1.0, ntotal)) + ")");
        // Small IDs are written with 4 bytes (as before)
        const size_t nfile = part.size();
        check(block_size(fileprefix + "." + mytask, 3) == int(4 * nfile) and
                  block_size(fileprefix + "." + mytask, 1) == int(12 * nfile),
              "DM file layout: POS 12 bytes and ID 4 bytes per particle");
    }

    //==========================================================
    // Several types in one particle class: MassTable and the type of each particle after reading with MPI
    //==========================================================
    if (run("types")) {
        const size_t ntotal = 5000; // 1000 of type 0, 3000 of type 1, 1000 of type 3
        auto part = make_task_particles<ParticleWithFamily>(ntotal);
        const std::string fileprefix = prefix + "types";
        GadgetWriter gw;
        gw.write_gadget_single(fileprefix + "." + mytask, part.data(), part.size(), ntotal, FML::NTasks, aexp, box,
                               OmegaM, 0.7, 0.7, box, vel_norm, {0.2, 0.7, 0.0, 0.1, 0.0, 0.0});
        my_files.push_back(fileprefix + "." + mytask);
        // Default fractions: the totals must still be right
        const std::string fileprefix2 = prefix + "typesdefault";
        gw.write_gadget_single(fileprefix2 + "." + mytask, part.data(), part.size(), ntotal, FML::NTasks, aexp, box,
                               OmegaM, 0.7, 0.7, box, vel_norm);
        my_files.push_back(fileprefix2 + "." + mytask);
        barrier();

        std::vector<ParticleWithFamily> read;
        GadgetReader g;
        g.read_gadget(fileprefix, read, 1.5, true, false);
        auto h = g.get_header();
        const double m0 = expected_mass(0.2, 1000), m1 = expected_mass(0.7, 3000), m3 = expected_mass(0.1, 1000);
        check(std::fabs(h.mass[0] / m0 - 1) < 1e-5 and std::fabs(h.mass[1] / m1 - 1) < 1e-5 and
                  std::fabs(h.mass[3] / m3 - 1) < 1e-5,
              "MassTable of each type: " + num(h.mass[0]) + " " + num(h.mass[1]) + " " + num(h.mass[3]) +
                  " (expected " + num(m0) + " " + num(m1) + " " + num(m3) + ")");
        check(h.npartTotal[0] == 1000 and h.npartTotal[1] == 3000 and h.npartTotal[3] == 1000,
              "npartTotal of each type");
        long long nbadtype = 0;
        for (auto & p : read)
            if (p.family != expected_family(p.id))
                nbadtype++;
        FML::SumOverTasks(&nbadtype);
        std::string info;
        bool ok = check_particles(read, ntotal, 0, info);
        check(ok and nbadtype == 0,
              "types read with " + num(FML::NTasks) + " task(s): wrong type for " + num(nbadtype) + " particles, " + info);

        GadgetReader g2;
        std::vector<ParticleWithFamily> read2;
        g2.read_gadget(fileprefix2, read2, 1.5, true, false);
        auto h2 = g2.get_header();
        check(h2.npartTotal[0] == 1000 and h2.npartTotal[1] == 3000 and h2.npartTotal[3] == 1000 and
                  std::fabs(h2.mass[1] / expected_mass(1.0, 3000) - 1) < 1e-5,
              "types with the default fractions: npartTotal = " + num(h2.npartTotal[0]) + " " + num(h2.npartTotal[1]) +
                  " " + num(h2.npartTotal[3]) + " mass[1] = " + num(h2.mass[1]));
    }

    //==========================================================
    // IDs larger than 2^31
    //==========================================================
    if (run("ids")) {
        const size_t ntotal = 1000;
        const long long offset = 3000000000LL;
        auto part = make_task_particles<Particle>(ntotal, offset);
        const std::string fileprefix = prefix + "ids";
        GadgetWriter gw;
        gw.write_gadget_single(fileprefix + "." + mytask, part.data(), part.size(), ntotal, FML::NTasks, aexp, box,
                               OmegaM, 0.7, 0.7, box, vel_norm);
        my_files.push_back(fileprefix + "." + mytask);
        barrier();
        std::vector<Particle> read;
        GadgetReader g;
        g.read_gadget(fileprefix, read, 1.5, true, false);
        long long minid = read.size() > 0 ? read[0].id : offset;
        for (auto & p : read)
            minid = std::min(minid, p.id);
        std::string info;
        bool ok = check_particles(read, ntotal, offset, info);
        check(ok, "IDs >= 3e9 round trip (smallest ID read " + num(double(minid)) + "): " + info);
    }

    //==========================================================
    // Few particles per file and files with no particles
    //==========================================================
    if (run("small")) {
        // 7 particles per task, and the first task writes no particles
        const size_t nper = 7;
        const size_t ntotal = nper * (FML::NTasks - 1);
        const size_t n0 = FML::ThisTask == 0 ? 0 : nper * (FML::ThisTask - 1);
        const size_t n1 = FML::ThisTask == 0 ? 0 : nper * FML::ThisTask;
        auto part = make_particles<Particle>(n0, n1);
        const std::string fileprefix = prefix + "small";
        GadgetWriter gw;
        gw.write_gadget_single(fileprefix + "." + mytask, part.data(), part.size(), ntotal, FML::NTasks, aexp, box,
                               OmegaM, 0.7, 0.7, box, vel_norm);
        my_files.push_back(fileprefix + "." + mytask);
        // And one file (written by task 0) with 5 particles
        const std::string fileprefix5 = prefix + "five";
        if (FML::ThisTask == 0) {
            auto five = make_particles<Particle>(0, 5);
            gw.write_gadget_single(fileprefix5 + ".0", five.data(), five.size(), 5, 1, aexp, box, OmegaM, 0.7, 0.7,
                                   box, vel_norm);
            my_files.push_back(fileprefix5 + ".0");
        }
        barrier();
        std::vector<Particle> read;
        GadgetReader g;
        g.read_gadget(fileprefix5, read, 1.5, true, false);
        std::string info;
        bool ok = check_particles(read, 5, 0, info);
        check(ok, "file with 5 particles: " + info);
        g.read_gadget(fileprefix, read, 1.5, true, false);
        ok = check_particles(read, ntotal, 0, info);
        check(ok, num(FML::NTasks) + " files with 7 particles, the first with none: " + info);
    }

    //==========================================================
    // write_gadget_species with per-particle masses: the MASS block
    //==========================================================
    if (run("mass")) {
        const size_t ncdm = 2000, nnu = 1000;
        auto cdm = make_task_particles<Particle>(ncdm);
        auto nu = make_task_particles<Particle>(nnu, ncdm);
        auto relmass = [](long long id) { return 1.0 + 0.01 * double(id % 7); };
        auto s1 = make_gadget_species(1, cdm.data(), cdm.size(), ncdm, 0.9);
        s1.get_mass = [&](size_t i) { return relmass(cdm[i].id); };
        auto s3 = make_gadget_species(3, nu.data(), nu.size(), nnu, 0.1);
        const std::string fileprefix = prefix + "mass";
        GadgetWriter gw;
        gw.write_gadget_species(fileprefix + "." + mytask, {s3, s1}, FML::NTasks, aexp, box, OmegaM, 0.7, 0.7, box,
                                vel_norm);
        my_files.push_back(fileprefix + "." + mytask);
        barrier();
        std::vector<ParticleWithFamily> read;
        GadgetReader g;
        g.read_gadget(fileprefix, read, 1.5, true, false);
        auto h = g.get_header();
        double maxerr = 0.0;
        long long nbad = 0;
        for (auto & p : read) {
            if (p.family == 1)
                maxerr = std::max(maxerr, std::fabs(p.mass / (expected_mass(0.9, ncdm) * relmass(p.id)) - 1.0));
            else if (not(p.family == 3 and p.mass == -1.0))
                nbad++;
        }
        FML::MaxOverTasks(&maxerr);
        FML::SumOverTasks(&nbad);
        std::string info;
        bool ok = check_particles(read, ncdm + nnu, 0, info);
        check(ok and maxerr < 1e-5 and nbad == 0 and h.mass[1] == 0.0 and
                  std::fabs(h.mass[3] / expected_mass(0.1, nnu) - 1) < 1e-5,
              "MASS block round trip: max relative error of the masses " + num(maxerr) + ", " + info);
    }

    //==========================================================
    // Big-endian and little-endian files with the same reader, format 2
    //==========================================================
    if (run("formats")) {
        const size_t n = 400;
        const std::string be = prefix + "bigendian", le = prefix + "littleendian", f2 = prefix + "format2";
        if (FML::ThisTask == 0) {
            write_raw_file(be + ".0", n, true, false);
            write_raw_file(le + ".0", n, false, false);
            write_raw_file(f2 + ".0", n, false, true);
            my_files.push_back(be + ".0");
            my_files.push_back(le + ".0");
            my_files.push_back(f2 + ".0");
        }
        barrier();
        GadgetReader g;
        std::vector<Particle> read;
        std::string info;
        g.read_gadget(be, read, 1.5, true, false);
        bool ok = check_particles(read, n, 0, info);
        check(ok, "big-endian file: " + info);
        g.read_gadget(le, read, 1.5, true, false);
        ok = check_particles(read, n, 0, info);
        check(ok, "little-endian file after a big-endian one (same reader): " + info);
        GadgetReader g2;
        g2.read_gadget(f2, read, 1.5, true, false);
        ok = check_particles(read, n, 0, info);
        check(ok, "format 2 file (with an extra AGE block): " + info);
    }

#ifdef USE_HDF5
    //==========================================================
    // HDF5
    //==========================================================
    if (run("hdf5")) {
        // MassTable of each type
        const size_t ntotal = 5000;
        auto part = make_task_particles<ParticleWithFamily>(ntotal);
        const std::string fileprefix = prefix + "h5types";
        GadgetHDF5Writer hw(3);
        hw.write_gadget_single(fileprefix + "." + mytask + ".hdf5", part.data(), part.size(), ntotal, FML::NTasks,
                               aexp, box, OmegaM, 0.7, 0.7, box, vel_norm, {0.2, 0.7, 0.0, 0.1, 0.0, 0.0});
        my_files.push_back(fileprefix + "." + mytask + ".hdf5");
        barrier();
        GadgetHDF5Reader hr(3);
        std::vector<ParticleWithFamily> read;
        hr.read_gadget(fileprefix, read, 1.5, true, false);
        auto h = hr.get_header();
        const double m0 = expected_mass(0.2, 1000), m1 = expected_mass(0.7, 3000), m3 = expected_mass(0.1, 1000);
        std::string info;
        bool ok = check_particles(read, ntotal, 0, info);
        check(ok and std::fabs(h.mass[0] / m0 - 1) < 1e-5 and std::fabs(h.mass[1] / m1 - 1) < 1e-5 and
                  std::fabs(h.mass[3] / m3 - 1) < 1e-5,
              "HDF5 MassTable of each type: " + num(h.mass[0]) + " " + num(h.mass[1]) + " " + num(h.mass[3]) + ", " +
                  info);

        // One reader for a snapshot with 1 file and then one with 2 files (written by task 0)
        const std::string one = prefix + "h5one", two = prefix + "h5two";
        if (FML::ThisTask == 0) {
            auto p0 = make_particles<Particle>(0, 300);
            auto p1 = make_particles<Particle>(300, 600);
            auto s = make_gadget_species(1, p0.data(), p0.size(), 300, 1.0);
            hw.write_gadget_species(one + ".0.hdf5", {s}, 1, aexp, box, OmegaM, 0.7, 0.7, box, vel_norm);
            s.npart_total = 600;
            hw.write_gadget_species(two + ".0.hdf5", {s}, 2, aexp, box, OmegaM, 0.7, 0.7, box, vel_norm);
            auto s1 = make_gadget_species(1, p1.data(), p1.size(), 600, 1.0);
            hw.write_gadget_species(two + ".1.hdf5", {s1}, 2, aexp, box, OmegaM, 0.7, 0.7, box, vel_norm);
            for (auto f : {one + ".0.hdf5", two + ".0.hdf5", two + ".1.hdf5"})
                my_files.push_back(f);
        }
        barrier();
        GadgetHDF5Reader hr2(3);
        std::vector<Particle> readp;
        hr2.read_gadget(one, readp, 1.5, true, false);
        ok = check_particles(readp, 300, 0, info);
        hr2.read_gadget(two, readp, 1.5, true, false);
        ok = check_particles(readp, 600, 0, info) and ok;
        check(ok, "HDF5 reader used for a 1-file and then a 2-file snapshot: " + info);

        // SWIFT: BoxSize with 3 elements
        const std::string swift = prefix + "h5swift";
        if (FML::ThisTask == 0) {
            using namespace FML::FILEUTILS::GADGET::HDF5;
            Handle file(H5Fcreate((swift + ".hdf5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose, swift);
            {
                Handle hg(H5Gcreate2(file, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "Header");
                const unsigned long long npart[6] = {0, 400, 0, 0, 0, 0};
                const double mass[6] = {0, 0, 0, 0, 0, 0}, boxsize[3] = {box, box, box};
                const int nfiles = 1;
                write_attribute(hg, "NumPart_ThisFile", H5T_STD_U64LE, H5T_NATIVE_ULLONG, npart, 6);
                write_attribute(hg, "NumPart_Total", H5T_STD_U64LE, H5T_NATIVE_ULLONG, npart, 6);
                write_attribute(hg, "MassTable", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, mass, 6);
                write_attribute(hg, "Time", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &aexp, 1);
                write_attribute(hg, "BoxSize", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, boxsize, 3);
                write_attribute(hg, "NumFilesPerSnapshot", H5T_STD_I32LE, H5T_NATIVE_INT, &nfiles, 1);
            }
            Handle g(H5Gcreate2(file, "PartType1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "PartType1");
            auto p = make_particles<Particle>(0, 400);
            std::vector<double> x, v;
            std::vector<unsigned long long> ids;
            for (auto & q : p) {
                for (int idim = 0; idim < 3; idim++) {
                    x.push_back(q.pos[idim] * box);
                    v.push_back(q.vel[idim] * vel_norm);
                }
                ids.push_back(q.id);
            }
            write_dataset(g, "Coordinates", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, x.data(), 400, 3);
            write_dataset(g, "Velocities", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, v.data(), 400, 3);
            write_dataset(g, "ParticleIDs", H5T_STD_U64LE, H5T_NATIVE_ULLONG, ids.data(), 400, 1);
            my_files.push_back(swift + ".hdf5");
        }
        barrier();
        GadgetHDF5Reader hr3(3);
        hr3.read_gadget(swift, readp, 1.5, true, false);
        ok = check_particles(readp, 400, 0, info);
        check(ok and hr3.get_header().BoxSize == box, "HDF5 file with a 3 element BoxSize (SWIFT): " + info);
    }
#endif

    // Clean up
    barrier();
    for (auto & f : my_files)
        std::remove(f.c_str());

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All tests passed\n" : num(fails) + " test(s) failed\n");
    return fails == 0 ? 0 : 1;
}
