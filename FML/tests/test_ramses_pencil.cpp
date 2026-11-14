// The RAMSES particle reader with a pencil FML::domain keeps the particles of the pencil of the task (it
// needs x and y): a small synthetic snapshot read on slabs and on pencils gives the same particles
#include <FML/Global/MPICounts.h>
#include <FML/Global/Global.h>
#include <FML/RamsesUtils/RamsesUtils.h>
#include <FML/tests/TestHelpers.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using FML::TESTS::check;

namespace {
    struct Part {
        double pos[3];
        double vel[3];
        double * get_pos() { return pos; }
        double * get_vel() { return vel; }
        constexpr int get_ndim() const { return 3; }
    };
    const std::string dir = "test_ramses_pencil_snap/output_00001/";
    const int ncpu = 3, npart_per_file = 500;

    void record(FILE * fp, const void * data, int bytes) {
        fwrite(&bytes, 4, 1, fp);
        fwrite(data, 1, size_t(bytes), fp);
        fwrite(&bytes, 4, 1, fp);
    }
    void write_snapshot() {
        std::filesystem::create_directories(dir);
        FILE * fp = fopen((dir + "info_00001.txt").c_str(), "w");
        fprintf(fp,
                "ncpu        =  %d\nndim        =  3\nlevelmin    =  6\nlevelmax    =  6\nngridmax    =  1000\n"
                "nstep_coarse=  1\nboxlen      =  1.0\ntime        =  1.0\naexp        =  1.0\nH0          =  70.0\n"
                "omega_m     =  0.3\nomega_l     =  0.7\nomega_k     =  0.0\nomega_b     =  0.0\nunit_l      =  1e7\n"
                "unit_d      =  1.0\nunit_t      =  1.0\n",
                ncpu);
        fclose(fp);
        fp = fopen((dir + "part_file_descriptor.txt").c_str(), "w");
        fprintf(fp, "# version:  1\n# ivar, variable_name, variable_type\n  1, position_x, d\n  2, position_y, d\n"
                    "  3, position_z, d\n  4, velocity_x, d\n  5, velocity_y, d\n  6, velocity_z, d\n");
        fclose(fp);
        std::mt19937 gen(11);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (int cpu = 1; cpu <= ncpu; cpu++) {
            char name[64];
            std::snprintf(name, sizeof(name), "part_00001.out%05d", cpu);
            fp = fopen((dir + name).c_str(), "wb");
            int i = ncpu, ndim = 3, n = npart_per_file, seed[4] = {1, 2, 3, 4}, zero = 0, two[2] = {0, 0};
            record(fp, &i, 4);
            record(fp, &ndim, 4);
            record(fp, &n, 4);
            record(fp, seed, 16);
            record(fp, &zero, 4);
            record(fp, two, 8);
            record(fp, two, 8);
            record(fp, &zero, 4);
            std::vector<double> x(n);
            for (int d = 0; d < 6; d++) {
                for (auto & v : x)
                    v = U(gen);
                record(fp, x.data(), int(8 * n));
            }
            fclose(fp);
        }
    }
    // All the particles (sorted) and whether every task has only particles of its domain
    std::vector<std::array<double, 6>> read(bool & owned) {
        FML::FILEUTILS::RAMSES::RamsesReader reader(dir, 1.0, true, false);
        std::vector<Part> p;
        reader.read_ramses(p);
        owned = true;
        std::vector<std::array<double, 6>> local;
        for (auto & q : p) {
            owned = owned and FML::domain.owner(q.pos) == FML::ThisTask;
            local.push_back({q.pos[0], q.pos[1], q.pos[2], q.vel[0], q.vel[1], q.vel[2]});
        }
        auto all = FML::COMMUNICATION::gather_records_checked(local, true);
        std::sort(all.begin(), all.end());
        int good = owned ? 1 : 0;
        FML::MinOverTasks(&good);
        owned = good == 1;
        return all;
    }
} // namespace

int main() {
    if (FML::ThisTask == 0)
        write_snapshot();
#ifdef USE_MPI
    MPI_Barrier(FML::communicator);
#endif
    bool owned_slab;
    const auto ref = read(owned_slab);
    check(owned_slab and ref.size() == size_t(ncpu * npart_per_file), "slabs: all the particles, each on its task");
    const FML::Domain slab = FML::domain;
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        FML::set_domain(FML::Domain::uniform_pencil(p0, FML::NTasks / p0));
        bool owned;
        const auto res = read(owned);
        FML::set_domain(slab);
        check(owned and res == ref, "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0) +
                                        ": the particles of the slabs, each on the task of its pencil");
    }
#ifdef USE_MPI
    MPI_Barrier(FML::communicator);
#endif
    if (FML::ThisTask == 0)
        std::filesystem::remove_all("test_ramses_pencil_snap");
    return FML::TESTS::summary("RAMSES reader on pencils");
}
