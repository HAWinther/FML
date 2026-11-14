// Regression tests for the fixes from the Nov 2026 library review (parts 1 and 2): copy_fourier_modes on odd meshes,
// empty slice ranges, coordinates in the left extra slices, power, init_mpi with MPI running, dump/load, growing the
// particle capacity in communicate_particles and the temporary file of MPIParticles::dump_to_file. Run with 1, 2, 4 tasks
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/CopyFourierModes.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <cstdio>
#include <iostream>

struct Particle {
    double pos[3];
    double * get_pos() { return pos; }
    const double * get_pos() const { return pos; }
    constexpr int get_ndim() const { return 3; }
};

static int nfail = 0;
static void check(bool ok, const std::string & what) {
    int bad = ok ? 0 : 1;
    FML::SumOverTasks(&bad);
    if (FML::ThisTask == 0)
        std::cout << (bad == 0 ? "[PASS] " : "[FAIL] ") << what << "\n";
    nfail += bad > 0;
}

int main() {
    using namespace FML::GRID;
    // 1. copy_fourier_modes on odd meshes keeps every mode of the smaller grid
    for (auto [nsrc, ndst] : std::vector<std::pair<int, int>>{{5, 7}, {7, 5}, {4, 8}, {8, 6}}) {
        if (nsrc % FML::NTasks or ndst % FML::NTasks) continue;
        FFTWGrid<3> src(nsrc), dst(ndst);
        src.set_grid_status_real(false);
        src.fill_fourier_grid(1.0);
        copy_fourier_modes<3>(src, dst, [](const std::array<int, 3> &) { return 1.0; });
        double nonzero = 0;
        for (auto && fi : dst.get_fourier_range())
            nonzero += std::abs(dst.get_fourier_from_index(fi)) > 0;
        FML::SumOverTasks(&nonzero);
        const int m = std::min(nsrc, ndst), k = (m - 1) / 2;
        const double expected = double(2 * k + 1) * (2 * k + 1) * (k + 1);
        check(nonzero == expected, "copy_fourier_modes " + std::to_string(nsrc) + " -> " + std::to_string(ndst) +
                                       ": " + std::to_string(int(nonzero)) + " modes kept (expected " +
                                       std::to_string(int(expected)) + ")");
    }

    // 2. get_real_range(0, 0) is empty, get_real_range() is everything
    {
        FFTWGrid<3> g(4 * FML::NTasks);
        size_t n00 = 0, nall = 0, nf00 = 0;
        for (auto && i : g.get_real_range(0, 0)) { (void)i; n00++; }
        for (auto && i : g.get_real_range()) { (void)i; nall++; }
        for (auto && i : g.get_fourier_range(0, 0)) { (void)i; nf00++; }
        check(n00 == 0 and nf00 == 0 and nall == size_t(g.get_local_nx()) * 16 * FML::NTasks * FML::NTasks,
              "get_real_range(0,0) and get_fourier_range(0,0) are empty, get_real_range() is the whole grid");
        FFTWGrid<3> empty;
        size_t ne = 0;
        for (auto && i : empty.get_real_range()) { (void)i; ne++; }
        check(ne == 0, "get_real_range() on a default-constructed grid is empty");
    }

    // 3. get_coord_from_index in the left extra slices
    {
        FFTWGrid<3> g(4 * FML::NTasks, 2, 1);
        bool ok = true;
        for (int ix : {-2, -1, 0, int(g.get_local_nx())})
            for (int iy : {0, 3})
                for (int iz : {0, 2}) {
                    const auto idx = g.get_index_real({ix, iy, iz});
                    ok = ok and g.get_coord_from_index(idx) == std::array<int, 3>{ix, iy, iz};
                }
        check(ok, "get_coord_from_index inverts get_index_real in the left and right extra slices");
    }

    // 4. power
    static_assert(FML::power(3, 5) == 243 and FML::power(2, 62) == (1LL << 62) and FML::power(7, 0) == 1 and
                  FML::power(2, -1) == 0);
    check(FML::power(1, 1000000000) == 1, "power is O(log b) (power(1, 1e9) returns at once)");

    // 5. init_mpi when MPI is already running is a no-op (no second MPI_Init)
    FML::init_mpi(nullptr, nullptr);
    check(true, "init_mpi with MPI already initialized");

    // 6. A dump with trailing data (as one written with double and read with float) is rejected: tested by hand
    //    (assert_mpi aborts); here the round trip with the same precision still works
    {
        FFTWGrid<3> g(4 * FML::NTasks, 1, 1);
        for (auto && i : g.get_real_range()) g.set_real_from_index(i, double(i));
        g.dump_to_file("review_fix_grid");
        FFTWGrid<3> h;
        h.load_from_file("review_fix_grid");
        bool ok = true;
        for (auto && i : g.get_real_range()) ok = ok and h.get_real_from_index(i) == g.get_real_from_index(i);
        check(ok, "FFTWGrid dump/load round trip");
        std::remove(("review_fix_grid." + std::to_string(FML::ThisTask)).c_str());
    }

    // 7. MPIParticles: receiving more particles than the capacity grows the storage instead of aborting;
    //    the checkpoint is written through a temporary file
    {
        FML::PARTICLE::MPIParticles<Particle> mp;
        std::vector<Particle> part(10);
        for (size_t i = 0; i < part.size(); i++) {
            // all particles belong to task 0's slab
            part[i].pos[0] = FML::xmin_domain + 1e-3 * (i + 1) * (FML::xmax_domain - FML::xmin_domain);
            part[i].pos[1] = part[i].pos[2] = 0.5;
        }
        mp.create(part.data(), part.size(), 10, FML::PARTICLE::Input::Distributed);
        // move every particle to task 0
        for (auto & p : mp) p.pos[0] = 0.1 / FML::NTasks;
        mp.communicate_particles();
        size_t n = mp.get_npart(), ntot = n;
        FML::SumOverTasks(&ntot);
        check(ntot == size_t(10 * FML::NTasks) and (FML::ThisTask != 0 or n == ntot),
              "communicate_particles grows the capacity when too many particles arrive");
        mp.dump_to_file("review_fix_parts");
        std::ifstream tmp("review_fix_parts." + std::to_string(FML::ThisTask) + ".tmp");
        std::ifstream fin("review_fix_parts." + std::to_string(FML::ThisTask));
        check(not tmp.good() and fin.good(), "MPIParticles::dump_to_file renames the temporary file");
        std::remove(("review_fix_parts." + std::to_string(FML::ThisTask)).c_str());
    }
    if (FML::ThisTask == 0)
        std::cout << (nfail ? "SOME TESTS FAILED\n" : "All review-fix tests passed\n");
    return nfail ? 1 : 0;
}
