//==========================================================
// Tests for FML::FILEUTILS::RAMSES (RamsesUtils/RamsesUtils.h) with small synthetic RAMSES snapshots written here:
//   - all particles read with keep_only_particles_in_domain = true (each on its task) and = false (all particles)
//   - positions, velocities (km/s from unit_l / unit_t), masses, ids; the boxsize from the units for both values of
//     Mpc in cm used by RAMSES (3.08e24 and 3.0857e24)
//   - a CPU file without particles, a particle at x = 1, no header file (part_file_descriptor.txt instead)
//   - read_ramses_single returns only the particles in the domain
// The header is also included in test_ramses_odr.cpp (it must be usable from several translation units)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/RamsesUtils/RamsesUtils.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace FML::FILEUTILS::RAMSES;

int ramses_odr_check(); // defined in test_ramses_odr.cpp

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}

struct Particle {
    double pos[3];
    double vel[3];
    double mass;
    int id;
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    void set_mass(double m) { mass = m; }
    void set_id(long long int i) { id = int(i); }
    constexpr int get_ndim() const { return 3; }
};

// Fortran unformatted record
template <class T>
void record(std::ofstream & fp, const std::vector<T> & v) {
    int n = int(v.size() * sizeof(T));
    fp.write((char *)&n, sizeof(int));
    if (n > 0)
        fp.write((char *)v.data(), n);
    fp.write((char *)&n, sizeof(int));
}

// The particle with global index i
double xpos(int i) { return i == 0 ? 1.0 : std::fmod(0.137 * i + 0.01, 1.0); } // particle 0 at x = 1
double ypos(int i) { return std::fmod(0.311 * i + 0.02, 1.0); }
double zpos(int i) { return std::fmod(0.719 * i + 0.03, 1.0); }

// Snapshot with ncpu files (file 2 empty), npart particles, POS VEL MASS ID LEVEL FAMILY TAG
void write_snapshot(const std::string & dir, int ncpu, int npart, double Mpc_cm, bool header) {
    mkdir(dir.substr(0, dir.find('/')).c_str(), 0755);
    mkdir(dir.c_str(), 0755);
    const double boxsize = 500.0, h = 0.7, aexp = 0.5;
    const double unit_l = boxsize * Mpc_cm * aexp / h;
    const double unit_t = aexp * aexp * Mpc_cm / (100.0 * h * 1e5);
    std::ofstream info(dir + "/info_00001.txt");
    info << "ncpu        =          " << ncpu << "\nndim        =          3\nlevelmin    =          5\n"
         << "levelmax    =          5\nngridmax    =      10000\nnstep_coarse=         23\n \n";
    info.precision(15);
    info << std::scientific << "boxlen      =  " << 1.0 << "\ntime        =  " << -1.0 << "\naexp        =  " << aexp
         << "\nH0          =  " << 100 * h << "\nomega_m     =  " << 0.3 << "\nomega_l     =  " << 0.7
         << "\nomega_k     =  " << 0.0 << "\nomega_b     =  " << 0.05 << "\nunit_l      =  " << unit_l
         << "\nunit_d      =  " << 1.0 << "\nunit_t      =  " << unit_t << "\n";
    info.close();
    if (header) {
        std::ofstream hf(dir + "/header_00001.txt");
        hf << " Total number of particles\n " << npart << "\n Total number of dark matter particles\n " << npart
           << "\n Total number of star particles\n 0\n Total number of sink particles\n 0\n Particle fields\n"
           << "pos vel mass iord level family tag \n";
    } else {
        std::ofstream df(dir + "/part_file_descriptor.txt");
        df << "# version:  1\n# ivar, variable_name, variable_type\n 1, position_x, d\n 2, position_y, d\n"
           << " 3, position_z, d\n 4, velocity_x, d\n 5, velocity_y, d\n 6, velocity_z, d\n 7, mass, d\n"
           << " 8, identity, i\n 9, levelp, i\n10, family, b\n11, tag, b\n";
    }
    int start = 0;
    for (int icpu = 0; icpu < ncpu; icpu++) {
        const int n = icpu == 1 ? 0 : (icpu == ncpu - 1 ? npart - start : npart / (ncpu - 1));
        char name[256];
        snprintf(name, sizeof(name), "%s/part_00001.out%05d", dir.c_str(), icpu + 1);
        std::ofstream fp(name, std::ios::binary);
        record(fp, std::vector<int>{ncpu});
        record(fp, std::vector<int>{3});
        record(fp, std::vector<int>{n});
        record(fp, std::vector<int>{1, 2, 3, 4});
        record(fp, std::vector<int>{0});
        record(fp, std::vector<double>{0.0});
        record(fp, std::vector<double>{0.0});
        record(fp, std::vector<int>{0});
        std::vector<double> x(n), y(n), z(n), v(n), m(n);
        std::vector<int> id(n), level(n, 5);
        std::vector<char> fam(n, 1), tag(n, 0);
        for (int j = 0; j < n; j++) {
            x[j] = xpos(start + j);
            y[j] = ypos(start + j);
            z[j] = zpos(start + j);
            m[j] = 1.0 / npart;
            id[j] = start + j + 1;
        }
        record(fp, x);
        record(fp, y);
        record(fp, z);
        for (int d = 0; d < 3; d++) {
            for (int j = 0; j < n; j++)
                v[j] = 0.001 * (d + 1) * (start + j); // code units
            record(fp, v);
        }
        record(fp, m);
        record(fp, id);
        record(fp, level);
        record(fp, fam);
        record(fp, tag);
        start += n;
    }
}

int main() {
    check(ramses_odr_check() == 1, "RamsesUtils.h can be included in several translation units");

    const int npart = 103, ncpu = 4;
    const std::string dir_old = "test_ramses_tmp/output_00001", dir_new = "test_ramses_tmp2/output_00001";
    if (FML::ThisTask == 0) {
        write_snapshot(dir_old, ncpu, npart, 3.08e24, true);
        write_snapshot(dir_new, ncpu, npart, 3.0857e24, false);
    }
#ifdef USE_MPI
    MPI_Barrier(FML::communicator);
#endif

    for (const std::string & dir : {dir_old, dir_new}) {
        const std::string label =
            dir == dir_old ? " (Mpc = 3.08e24 cm, header file)" : " (Mpc = 3.0857e24 cm, part_file_descriptor.txt)";
        // All particles over all tasks, each in its domain
        bool ok = true;
        try {
            RamsesReader reader(dir, 1.0, true, false);
            std::vector<Particle> p;
            reader.read_ramses(p);
            long long int n = p.size();
            FML::SumOverTasks(&n);
            bool fields_ok = true;
            for (auto & q : p) {
                const int i = q.id - 1;
                const double x = xpos(i) >= 1.0 ? 0.0 : xpos(i);
                fields_ok = fields_ok and q.pos[0] >= FML::xmin_domain and q.pos[0] < FML::xmax_domain and
                            std::fabs(q.pos[0] - x) < 1e-14 and std::fabs(q.pos[1] - ypos(i)) < 1e-14 and
                            std::fabs(q.mass - 1.0 / npart) < 1e-14;
                // v = v_code * unit_l / unit_t = v_code * 100 boxsize / aexp km/s
                for (int d = 0; d < 3; d++)
                    fields_ok = fields_ok and std::fabs(q.vel[d] - 0.001 * (d + 1) * i * 100 * 500.0 / 0.5) < 1e-9;
            }
            int bad = not fields_ok;
            FML::SumOverTasks(&bad);
            check(n == npart and bad == 0 and std::fabs(reader.get_boxsize() - 500.0) < 1e-9,
                  "all " + std::to_string(npart) + " particles (one at x = 1, an empty CPU file), fields, boxsize " +
                      std::to_string(reader.get_boxsize()) + label);
        } catch (std::exception & e) {
            ok = false;
            check(false, std::string("reading failed: ") + e.what() + label);
        }
        (void)ok;
    }

    // keep_only_particles_in_domain = false: all particles on every task
    {
        RamsesReader reader(dir_old, 1.0, false, false);
        std::vector<Particle> p;
        reader.read_ramses(p);
        std::set<int> ids;
        for (auto & q : p)
            ids.insert(q.id);
        int bad = not(p.size() == size_t(npart) and ids.size() == size_t(npart));
        FML::SumOverTasks(&bad);
        check(bad == 0, "keep_only_particles_in_domain = false: all particles on every task (an empty vector before)");
    }

    // read_ramses_single: only the particles in the domain (no extra zero particles)
    {
        RamsesReader reader(dir_old, 1.0, true, false);
        std::vector<Particle> p;
        reader.read_ramses_single(0, p);
        bool ok = true;
        for (auto & q : p)
            ok = ok and q.id > 0 and q.pos[0] >= FML::xmin_domain and q.pos[0] < FML::xmax_domain;
        int bad = not ok;
        FML::SumOverTasks(&bad);
        check(bad == 0, "read_ramses_single returns only the particles in the domain");
    }

#ifdef USE_MPI
    MPI_Barrier(FML::communicator);
#endif
    if (FML::ThisTask == 0) {
        for (const std::string & dir : {dir_old, dir_new}) {
            for (int icpu = 0; icpu < ncpu; icpu++) {
                char name[256];
                snprintf(name, sizeof(name), "%s/part_00001.out%05d", dir.c_str(), icpu + 1);
                std::remove(name);
            }
            for (auto f : {"/info_00001.txt", "/header_00001.txt", "/part_file_descriptor.txt"})
                std::remove((dir + f).c_str());
            std::remove(dir.c_str());
            std::remove(dir.substr(0, dir.find('/')).c_str());
        }
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All RAMSES tests passed\n" : std::to_string(fails) + " RAMSES test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
