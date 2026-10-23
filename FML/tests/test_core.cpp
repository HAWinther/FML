//==========================================================
// Regression tests for core FML functionality
// Run with: make test (from the top folder)
//
// Usage: ./test_core            run all checks
//        ./test_core alloc      must abort: allocation limit
//        ./test_core unwrapped  must abort: particle outside [0,1)
//==========================================================
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/MPIParticles/MPIParticles.h>
#ifdef USE_GSL
#include <FML/Spline/Spline.h>
#endif
#include <cstdlib>

struct Particle {
    double pos[3];
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};

int main(int argc, char ** argv) {
    int fails = 0;
    auto check = [&](bool ok, std::string what) {
        if (FML::ThisTask == 0)
            std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
        if (not ok)
            fails++;
    };
    std::string mode = argc > 1 ? argv[1] : "all";

    if (mode == "all") {
        // 1. reallocate actually reallocates
        FML::GRID::FFTWGrid<3> g;
        g.reallocate(16, 0, 0);
        g.reallocate(16, 1, 1); // and again with different padding
        check(g.get_nmesh() == 16 and g.get_n_extra_slices_left() == 1 and bool(g), "FFTWGrid::reallocate");

        // 2. index computation does not overflow for large Nmesh (pure arithmetic, no allocation needed)
        {
            const long long Nmesh = 2048;
            // Emulate get_index_real formula for last cell on a single task
            std::array<int, 3> c{int(Nmesh - 1), int(Nmesh - 1), int(Nmesh - 1)};
            long long expected = ((Nmesh - 1) * Nmesh + (Nmesh - 1)) * (2 * (Nmesh / 2 + 1)) + (Nmesh - 1);
            long long got = (FML::IndexIntType(c[0]) * int(Nmesh) + c[1]) * (2 * (int(Nmesh) / 2 + 1)) + c[2];
            check(got == expected and got > (1LL << 31), "index formula (Nmesh=2048) no overflow");
        }
        // Use the real method on a small grid to check it is still consistent
        {
            FML::GRID::FFTWGrid<3> h(8);
            bool ok = true;
            for (auto && ri : h.get_real_range()) {
                auto coord = h.get_coord_from_index(ri);
                ok = ok and (h.get_index_real(coord) == ri);
            }
            check(ok, "get_index_real <-> get_coord_from_index roundtrip");
        }

        // 5. Vector math functions compile and work
        FML::Vector<double> v{0.0, 1.0, 2.0};
        auto e = FML::exp(v);
        auto l = FML::log(e);
        check(std::abs(e[1] - std::exp(1.0)) < 1e-14 and std::abs(l[2] - 2.0) < 1e-14, "FML::exp/log on Vector");

        // 6. Spline used from more threads than it was created for and nested regions
#if defined(USE_GSL) && defined(USE_OMP)
        {
            omp_set_num_threads(1);
            FML::Vector<double> x(100), y(100);
            for (int i = 0; i < 100; i++) {
                x[i] = i / 99.0;
                y[i] = x[i] * x[i];
            }
            FML::INTERPOLATION::SPLINE::Spline s(x, y, "test");
            omp_set_num_threads(64);
            omp_set_max_active_levels(2);
            double maxerr = 0.0;
#pragma omp parallel num_threads(64) reduction(max : maxerr)
            {
#pragma omp parallel num_threads(4) reduction(max : maxerr)
                for (int i = 0; i < 1000; i++) {
                    double xx = (i % 997) / 997.0;
                    maxerr = std::max(maxerr, std::abs(s(xx) - xx * xx));
                }
            }
            check(maxerr < 1e-4, "Spline eval with more threads than at creation + nested OMP");
        }
#endif

        // 7. Particle communication (normal case) keeps all particles
        {
            const size_t n = 1000;
            std::vector<Particle> part(n);
            std::srand(1234 + FML::ThisTask);
            for (auto & p : part)
                for (int d = 0; d < 3; d++)
                    p.pos[d] = std::rand() / (RAND_MAX + 1.0);
            FML::PARTICLE::MPIParticles<Particle> mp;
            mp.create(part.data(), n, 2 * n * FML::NTasks, FML::xmin_domain, FML::xmax_domain, false);
            bool ok = mp.get_npart_total() == n * size_t(FML::NTasks);
            for (size_t i = 0; i < mp.get_npart(); i++) {
                double xx = mp[i].pos[0];
                ok = ok and xx >= FML::xmin_domain and xx < FML::xmax_domain;
            }
            int iok = ok ? 0 : 1;
            FML::SumOverTasks(&iok);
            check(iok == 0, "MPIParticles create/communicate keeps all particles in correct domain");
        }
    }

    // 3. Allocation limit: should abort cleanly with message, not write out of bounds
    if (mode == "alloc") {
        const size_t n = 100;
        std::vector<Particle> part(n);
        for (auto & p : part)
            p.pos[0] = p.pos[1] = p.pos[2] = 0.5 * (FML::xmin_domain + FML::xmax_domain);
        FML::PARTICLE::MPIParticles<Particle> mp;
        mp.create(part.data(), n, n / 2, FML::xmin_domain, FML::xmax_domain, false);
    }

    // 4. Unwrapped particle: should give a clear error
    if (mode == "unwrapped") {
        std::vector<Particle> part(10);
        for (auto & p : part)
            p.pos[0] = p.pos[1] = p.pos[2] = 0.5 * (FML::xmin_domain + FML::xmax_domain);
        FML::PARTICLE::MPIParticles<Particle> mp;
        mp.create(part.data(), 10, 100, FML::xmin_domain, FML::xmax_domain, false);
        mp[0].pos[0] = 1.3;
        mp.communicate_particles();
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return fails;
}
