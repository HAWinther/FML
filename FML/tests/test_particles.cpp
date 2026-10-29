//==========================================================
// Tests for the particle containers and particle reflection (MPIParticles/MPIParticles.h,
// ParticleTypes/ReflectOnParticleMethods.h, ParticleTypes/SimpleGalaxy.h, ParticlesInBoxes/ParticlesInBoxes.h
// and with USE_HEALPIX Chunkpix/Chunkpix.h and the chunkpix maps of the COLASolver lightcone):
//  - create_particle_grid with Npart_1D not divisible by NTasks (or < NTasks) has room for a uniform shift
//  - create with only one task having particles (batched read) and with x = 1.0 / -1e-17
//  - dump_to_file / load_from_file with more particles than fit in the buffer
//  - particles with custom (variable size) serialisation methods compile, communicate and round trip
//  - ParticlesInBoxes with negative positions and with no particles
//  - SimpleGalaxy compiles, detection of overloaded (const and non-const) get_pos, swapping Eulerian and
//    Lagrangian positions for particles that cannot be constexpr
//  - shift_particles always gives positions in [0,1) and the positions can be restored exactly
// Run with 1, 2 and 4 MPI tasks (OMP_NUM_THREADS=2). Healpix tests need -DUSE_HEALPIX and
// -I<healpix_cxx include dir> ... -lhealpix_cxx -lcfitsio
//==========================================================
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/ParticleTypes/SimpleGalaxy.h>
#include <FML/ParticlesInBoxes/ParticlesInBoxes.h>
#ifdef USE_HEALPIX
#include <FML/Chunkpix/Chunkpix.h>
#include <FML/COLASolver/src/Lightcone/Lightcone_Healpix.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}

// True only if ok on all tasks
bool all_tasks(bool ok) {
    int bad = ok ? 0 : 1;
    FML::SumOverTasks(&bad);
    return bad == 0;
}

struct Particle {
    double pos[3];
    long long id;
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};

// Variable size serialisation (only n of the extra values are communicated)
struct VarParticle {
    double pos[3];
    int n{0};
    double extra[4];
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
    int get_particle_byte_size() { return int(sizeof(pos) + sizeof(int) + n * sizeof(double)); }
    void append_to_buffer(char * b) {
        std::memcpy(b, pos, sizeof(pos));
        b += sizeof(pos);
        std::memcpy(b, &n, sizeof(int));
        b += sizeof(int);
        std::memcpy(b, extra, n * sizeof(double));
    }
    void assign_from_buffer(char * b) {
        std::memcpy(pos, b, sizeof(pos));
        b += sizeof(pos);
        std::memcpy(&n, b, sizeof(int));
        b += sizeof(int);
        std::memcpy(extra, b, n * sizeof(double));
    }
};

// Overloaded (const and non-const) get_pos
struct Overloaded {
    double pos[3];
    double * get_pos() { return pos; }
    const double * get_pos() const { return pos; }
    constexpr int get_ndim() const { return 3; }
};

// Not a literal type and no get_q
struct NoQ {
    double pos[3];
    std::string name;
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};

// get_q, but cannot be constexpr (non-constexpr default member initializer)
struct WithQ {
    double pos[3];
    double q[3];
    double w = std::sqrt(2.0);
    double * get_pos() { return pos; }
    double * get_q() { return q; }
    constexpr int get_ndim() const { return 3; }
};

static bool in_domain(double x, double xmin, double xmax) { return x >= xmin and x < xmax; }

//==========================================================
// 1. create_particle_grid capacity
//==========================================================
void test_lattice() {
    for (int N : {5, 3, 7}) {
        FML::PARTICLE::MPIParticles<Particle> mp;
        mp.create_particle_grid(N, 1.0, FML::xmin_domain, FML::xmax_domain);
        long long id = 0, idsum0 = 0;
        for (auto & p : mp) {
            p.id = std::llround(p.pos[0] * N) * N * N + std::llround(p.pos[1] * N) * N + std::llround(p.pos[2] * N);
            idsum0 += p.id;
            id++;
        }
        FML::SumOverTasks(&idsum0);
        const size_t needed = size_t(std::ceil(N * (FML::xmax_domain - FML::xmin_domain) - 1e-9)) * N * N;
        check(all_tasks(mp.get_particles().size() >= needed),
              "create_particle_grid N = " + std::to_string(N) +
                  ": capacity >= the particles of the most lattice "
                  "slices the domain can hold");

        // Uniform shift by half a lattice spacing in x then communicate
        for (auto & p : mp) {
            p.pos[0] += 0.5 / N;
            if (p.pos[0] >= 1.0)
                p.pos[0] -= 1.0;
        }
        mp.communicate_particles();
        size_t n = mp.get_npart();
        FML::SumOverTasks(&n);
        long long idsum = 0;
        bool ok = true;
        for (auto & p : mp) {
            idsum += p.id;
            ok = ok and in_domain(p.pos[0], FML::xmin_domain, FML::xmax_domain);
        }
        FML::SumOverTasks(&idsum);
        check(all_tasks(ok) and n == size_t(N * N * N) and idsum == idsum0,
              "create_particle_grid N = " + std::to_string(N) +
                  ": uniform shift + communicate_particles keeps all "
                  "particles");
    }
}

//==========================================================
// 2. create: batched read and boundary positions
//==========================================================
void test_create() {
    {
        const size_t ntot = 10000;
        const size_t n = FML::ThisTask == 0 ? ntot : 0;
        std::mt19937 gen(1);
        std::uniform_real_distribution<double> U(0, 1);
        std::vector<Particle> part(n);
        long long idsum0 = 0;
        for (size_t i = 0; i < n; i++) {
            part[i] = Particle{{U(gen), U(gen), U(gen)}, (long long)i};
            idsum0 += i;
        }
        FML::SumOverTasks(&idsum0);
        const size_t nallocate = FML::NTasks == 1 ? ntot : size_t(1.25 * ntot / FML::NTasks);
        FML::PARTICLE::MPIParticles<Particle> mp;
        mp.create(part.data(), n, nallocate, FML::xmin_domain, FML::xmax_domain, false);
        long long idsum = 0;
        bool ok = true;
        for (auto & p : mp) {
            idsum += p.id;
            ok = ok and in_domain(p.pos[0], FML::xmin_domain, FML::xmax_domain);
        }
        FML::SumOverTasks(&idsum);
        check(all_tasks(ok) and mp.get_npart_total() == ntot and idsum == idsum0,
              "create with all particles on task 0 (batched read, nallocate < particles to read)");
    }
    {
        std::vector<Particle> part(4);
        part[0] = Particle{{0.5, 0.5, 0.5}, 0};
        part[1] = Particle{{1.0, 0.5, 0.5}, 1};
        part[2] = Particle{{-1e-17, 0.5, 0.5}, 2};
        part[3] = Particle{{0.0, 0.5, 0.5}, 3};
        FML::PARTICLE::MPIParticles<Particle> mp;
        mp.create(part.data(), 4, 10, FML::xmin_domain, FML::xmax_domain, true);
        bool ok = true;
        for (auto & p : mp)
            ok = ok and in_domain(p.pos[0], FML::xmin_domain, FML::xmax_domain) and p.pos[0] >= 0.0 and p.pos[0] < 1.0;
        check(all_tasks(ok) and mp.get_npart_total() == 4,
              "create (all tasks same particles) keeps particles at x = 1.0 and x = -1e-17 (inside [0,1))");
    }
}

//==========================================================
// 3. dump_to_file / load_from_file
//==========================================================
template <class T, class Equal>
void dump_load_roundtrip(FML::PARTICLE::MPIParticles<T> & mp, size_t buffer_bytes, std::string label, Equal equal) {
    const std::string prefix = "test_particles_dump_" + label;
    mp.dump_to_file(prefix, buffer_bytes);
    FML::PARTICLE::MPIParticles<T> mp2;
    mp2.load_from_file(prefix);
    bool ok = mp2.get_npart() == mp.get_npart() and mp2.get_particles().size() == mp.get_particles().size() and
              mp2.get_npart_total() == mp.get_npart_total();
    for (size_t i = 0; ok and i < mp.get_npart(); i++)
        ok = equal(mp[i], mp2[i]);

    // The file must contain exactly the particles in use (header + chunks of at most buffer_bytes)
    size_t expected = sizeof(int) + 3 * sizeof(size_t) + 2 * sizeof(double) * FML::NTasks;
    size_t inbuffer = 0;
    for (size_t i = 0; i < mp.get_npart(); i++) {
        size_t bytes = FML::PARTICLE::GetSize(mp[i]);
        if (i == 0 or inbuffer + bytes > buffer_bytes) {
            expected += 2 * sizeof(size_t);
            inbuffer = 0;
        }
        expected += bytes;
        inbuffer += bytes;
    }
    const std::string filename = prefix + "." + std::to_string(FML::ThisTask);
    std::ifstream f(filename, std::ios::binary | std::ios::ate);
    const size_t filesize = f.good() ? size_t(f.tellg()) : 0;
    f.close();
    std::remove(filename.c_str());
    check(all_tasks(ok and filesize == expected),
          "dump_to_file/load_from_file round trip (" + label +
              ", more particles than fit in the buffer, file has "
              "only the particles in use)");
}

void test_dump() {
    FML::PARTICLE::MPIParticles<Particle> mp;
    mp.create_particle_grid(10, 1.0, FML::xmin_domain, FML::xmax_domain);
    long long id = 1000 * FML::ThisTask;
    for (auto & p : mp)
        p.id = id++;
    dump_load_roundtrip(mp, 150 * sizeof(Particle), "lattice", [](Particle & a, Particle & b) {
        return a.id == b.id and a.pos[0] == b.pos[0] and a.pos[1] == b.pos[1] and a.pos[2] == b.pos[2];
    });
}

//==========================================================
// 4. Custom (variable size) serialisation
//==========================================================
void test_custom_serialisation() {
#ifndef TEST_PARTICLES_OLD_API
    // Direct use of AppendToBuffer / AssignFromBuffer (was ambiguous)
    {
        VarParticle a{{0.1, 0.2, 0.3}, 2, {1.0, 2.0, 3.0, 4.0}}, b{};
        std::vector<char> buf(FML::PARTICLE::GetSize(a));
        char * bp = buf.data();
        FML::PARTICLE::AppendToBuffer(a, bp);
        FML::PARTICLE::AssignFromBuffer(b, bp);
        check(FML::PARTICLE::GetSize(a) == int(3 * sizeof(double) + sizeof(int) + 2 * sizeof(double)) and b.n == 2 and
                  b.extra[0] == 1.0 and b.extra[1] == 2.0 and b.pos[2] == 0.3,
              "AppendToBuffer/AssignFromBuffer use the particle's own methods");
        Particle p{{0.1, 0.2, 0.3}, 42}, q{};
        std::vector<char> buf2(sizeof(Particle));
        FML::PARTICLE::AppendToBuffer(p, buf2.data());
        FML::PARTICLE::AssignFromBuffer(q, buf2.data());
        check(q.id == 42 and q.pos[1] == 0.2, "AppendToBuffer/AssignFromBuffer fiducial (memcpy) methods");
    }

    // Communicate and dump/load with variable size particles
    std::mt19937 gen(5 + FML::ThisTask);
    std::uniform_real_distribution<double> U(0, 1);
    std::vector<VarParticle> v(500);
    double sum0 = 0.0;
    int nextra0 = 0;
    for (auto & q : v) {
        q.pos[0] = U(gen);
        q.pos[1] = U(gen);
        q.pos[2] = U(gen);
        q.n = gen() % 5;
        for (int k = 0; k < q.n; k++) {
            q.extra[k] = U(gen);
            sum0 += q.extra[k];
        }
        nextra0 += q.n;
    }
    FML::SumOverTasks(&sum0);
    FML::SumOverTasks(&nextra0);
    FML::PARTICLE::MPIParticles<VarParticle> mv;
    mv.create(v.data(), v.size(), 500 * FML::NTasks + 500, FML::xmin_domain, FML::xmax_domain, false);
    for (int it = 0; it < 3; it++) {
        for (auto & q : mv)
            q.pos[0] = U(gen);
        // Force multiple messages and record splits, with rank-dependent caps using their collective minimum.
        mv.communicate_particles(97 + FML::ThisTask);
    }
    if (FML::NTasks > 1) {
        bool rejected_cap = false;
        try {
            mv.communicate_particles(FML::ThisTask == 0 ? 0 : 97);
        } catch (const std::overflow_error &) {
            rejected_cap = true;
        }
        check(rejected_cap, "one-rank zero migration message cap is rejected collectively before mutation");
        mv.communicate_particles(97);
    }
    double sum = 0.0;
    int nextra = 0;
    bool ok = true;
    for (auto & q : mv) {
        for (int k = 0; k < q.n; k++)
            sum += q.extra[k];
        nextra += q.n;
        ok = ok and in_domain(q.pos[0], FML::xmin_domain, FML::xmax_domain);
    }
    FML::SumOverTasks(&sum);
    FML::SumOverTasks(&nextra);
    check(all_tasks(ok) and nextra == nextra0 and std::fabs(sum - sum0) < 1e-10 * sum0 and
              mv.get_npart_total() == 500 * size_t(FML::NTasks),
          "Variable size particles with custom serialisation: create + communicate_particles");
    dump_load_roundtrip(mv, 120 * sizeof(VarParticle), "custom", [](VarParticle & a, VarParticle & b) {
        bool eq = a.n == b.n and a.pos[0] == b.pos[0] and a.pos[1] == b.pos[1] and a.pos[2] == b.pos[2];
        for (int k = 0; eq and k < a.n; k++)
            eq = a.extra[k] == b.extra[k];
        return eq;
    });
#else
    check(false, "Custom serialisation methods (does not compile with the old code: ambiguous overloads)");
#endif
}

//==========================================================
// 5. ParticlesInBoxes
//==========================================================
void test_particles_in_boxes() {
    {
        std::vector<Particle> v{Particle{{0.1, -0.3, 0.1}, 0}};
        FML::PARTICLE::ParticlesInBoxes<Particle> grid;
        bool threw = false;
        try {
            grid.create(v, 4);
        } catch (std::runtime_error &) {
            threw = true;
        }
        check(threw, "ParticlesInBoxes: a position y = -0.3 is an error (was a wrapped cell index)");
    }
    {
        std::vector<Particle> v{Particle{{0.1, -1e-17, 0.99}, 0}, Particle{{0.6, 0.6, 0.6}, 1}};
        FML::PARTICLE::ParticlesInBoxes<Particle> grid;
        grid.create(v, 4);
        auto & cells = grid.get_cells();
        check(cells.size() == 64 and cells[0 * 16 + 0 * 4 + 3].get_np() == 1 and
                  cells[2 * 16 + 2 * 4 + 2].get_np() == 1,
              "ParticlesInBoxes: y = -1e-17 (round-off) is accepted (first cell)");

        // Recreate with no particles: no stale cells/Ngrid
        std::vector<Particle> none;
        grid.create(none, 2);
        size_t n = 0;
        for (auto & c : grid.get_cells())
            n += c.get_np() + c.get_part().size();
        check(grid.get_ngrid() == 2 and grid.get_npart() == 0 and grid.get_cells().size() == 8 and n == 0,
              "ParticlesInBoxes: create with no particles resets the grid");
    }
}

//==========================================================
// 6. Particle reflection
//==========================================================
void test_reflection() {
#ifndef TEST_PARTICLES_OLD_API
    {
        SimpleGalaxy g(1.0, 2.0, 3.0);
        g.set_RA(4.0);
        g.set_DEC(5.0);
        g.set_z(6.0);
        g.set_weight(0.5);
        check(FML::PARTICLE::GetRA(g) == 4.0 and FML::PARTICLE::GetDEC(g) == 5.0 and
                  FML::PARTICLE::GetRedshift(g) == 6.0 and FML::PARTICLE::GetWeight(g) == 0.5 and
                  FML::PARTICLE::has_set_RA<SimpleGalaxy>(),
              "SimpleGalaxy compiles and its setters/getters work");
    }
#else
    check(false, "SimpleGalaxy (does not compile with the old code)");
#endif

    {
        bool ok = FML::PARTICLE::has_get_pos<Overloaded>() and FML::PARTICLE::has_get_ndim<Overloaded>();
        if constexpr (FML::PARTICLE::has_get_pos<Overloaded>()) {
            Overloaded o{{0.25, 0.5, 0.75}};
            const Overloaded & co = o;
            FML::PARTICLE::GetPos(o)[1] = 0.125;
            ok = ok and FML::PARTICLE::GetPos(co)[1] == 0.125 and FML::PARTICLE::GetPos(co)[2] == 0.75;
            // And it works in MPIParticles
            std::vector<Overloaded> v{o};
            FML::PARTICLE::MPIParticles<Overloaded> mp;
            mp.create(v.data(), v.size(), 2, FML::xmin_domain, FML::xmax_domain, true);
            ok = ok and mp.get_npart_total() == 1;
        }
        check(ok, "Detection of overloaded (const and non-const) get_pos");
    }

#ifndef TEST_PARTICLES_OLD_API
    {
        NoQ a[1]{};
        a[0].pos[0] = 0.5;
        FML::PARTICLE::swap_eulerian_and_lagrangian_positions(a, 1);
        WithQ b[1]{};
        b[0].pos[0] = 0.1;
        b[0].q[0] = 0.2;
        FML::PARTICLE::swap_eulerian_and_lagrangian_positions(b, 1);
        check(a[0].pos[0] == 0.5 and b[0].pos[0] == 0.2 and b[0].q[0] == 0.1,
              "swap_eulerian_and_lagrangian_positions for non-literal particles with and without get_q");
    }
#else
    check(false, "swap_eulerian_and_lagrangian_positions without get_q (does not compile with the old code)");
#endif
}

//==========================================================
// 7. shift_particles
//==========================================================
void test_shift() {
    struct P {
        double pos[3];
        double * get_pos() { return pos; }
        constexpr int get_ndim() const { return 3; }
    };
    {
        P p{{0.5, -1e-17, 0.3}};
        FML::PARTICLE::shift_particles<P, 3>(&p, 1, {0.0, 0.0, 0.0}, 1.0, 1);
        P q{{0.5, 0.5, 0.5}};
        FML::PARTICLE::shift_particles<P, 3>(&q, 1, {0.0, -0.5 - 1e-17, 0.0}, 1.0, 2);
        check(p.pos[1] >= 0.0 and p.pos[1] < 1.0 and q.pos[1] >= 0.0 and q.pos[1] < 1.0,
              "shift_particles wraps y = -1e-17 into [0,1) (was exactly 1.0)");
    }
    std::mt19937 g(1);
    std::uniform_real_distribution<double> U(0, 1);
    const size_t n = 200000;
    std::vector<P> v(n);
    for (auto & p : v)
        for (int d = 0; d < 3; d++)
            p.pos[d] = U(g);
    v[0].pos[1] = 0.0;
    v[1].pos[2] = std::nextafter(1.0, 0.0);
    const std::vector<P> v0 = v;
    bool in_range = true, exact = true;
    for (int wrap_type : {1, 2}) {
        for (double s : {0.5 / 64, -0.5 / 64, 0.25, 1e-17}) {
            const std::array<double, 3> shift{s, s, s};
#ifndef TEST_PARTICLES_OLD_API
            auto saved = FML::PARTICLE::save_particle_positions<P, 3>(v.data(), n);
#endif
            FML::PARTICLE::shift_particles<P, 3>(v.data(), n, shift, +1.0, wrap_type);
            for (auto & p : v)
                for (int d = (wrap_type == 2 ? 1 : 0); d < 3; d++)
                    in_range = in_range and p.pos[d] >= 0.0 and p.pos[d] < 1.0;
#ifndef TEST_PARTICLES_OLD_API
            FML::PARTICLE::restore_particle_positions<P, 3>(v.data(), n, saved);
#else
            FML::PARTICLE::shift_particles<P, 3>(v.data(), n, shift, -1.0, wrap_type);
#endif
            for (size_t i = 0; i < n; i++)
                for (int d = 0; d < 3; d++)
                    exact = exact and v[i].pos[d] == v0[i].pos[d];
        }
    }
    check(in_range, "shift_particles results are always in [0,1)");
    check(exact, "Positions are restored exactly after shift_particles (save/restore_particle_positions)");
}

//==========================================================
// 8. Chunkpix (needs Healpix)
//==========================================================
#ifdef USE_HEALPIX
void test_chunkpix() {
    {
        // More than 2^16 chunks (parameterfile.lua suggests plc_nside_chunks = 256)
        FML::CHUNKPIX::Chunkpix<double> c;
        c.init(512, 256);
        const long npix = 12L * 512 * 512;
        // indices in the ordering of the chunks (NEST for nside = 2^k); the full map is RING
        Healpix_Base hb(512, c.ordering(), SET_NSIDE);
        auto to_ring = [&](long i) { return c.ordering() == NEST ? long(hb.nest2ring(int(i))) : i; };
        c.increase_ipix_count(npix - 1, 1.0);
        c.increase_ipix_count(65536L * 4 + 1, 0.25);
        c.increase_ipix_count(65536L * 4 + 1, 0.25);
        Healpix_Map<double> full;
        c.reconstruct_full_map(full);
        double sum = 0.0;
        for (long i = 0; i < npix; i++)
            sum += full[i];
        check(c.ordering() == NEST and full.Scheme() == RING and full.Npix() == npix and
                  full[to_ring(npix - 1)] == 1.0 and full[to_ring(65536 * 4 + 1)] == 0.5 and sum == 1.5,
              "Chunkpix with 786432 chunks and fractional weights (NEST chunks, RING map)");
    }
    {
        FML::CHUNKPIX::Chunkpix<double> c;
        c.init(16, 2);
        for (int ch = 0; ch < 24; ch++)
            c.increase_ipix_count(ch * (12 * 256 / 48), 1.0);
        check(std::fabs(c.chunk_sparsity() - 0.5) < 1e-14, "Chunkpix chunk_sparsity");
        bool threw = false;
        try {
            c.init(12, 8);
        } catch (std::runtime_error &) {
            threw = true;
        }
        check(threw, "Chunkpix: nside not a multiple of nside_chunks is an error");
    }
    {
        // The point of the chunks: a map of part of the sky only allocates about that fraction of them. A cap of
        // 5% of the sky (as a task's slab of particles in a lightcone shell): NEST chunks (compact patches)
        // allocate ~5-10%; RING chunks (runs along the iso-latitude rings) allocated most of them
        const long nside = 256;
        FML::CHUNKPIX::Chunkpix<double> c;
        c.init(nside, 8);
        Healpix_Base hb(int(nside), c.ordering(), SET_NSIDE);
        Healpix_Map<double> direct(int(nside), RING, SET_NSIDE);
        direct.fill(0.0);
        std::mt19937 gen(3);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        const double cos_max = 1.0 - 2.0 * 0.05;   // cap around the x-axis with fsky = 0.05
        for (int i = 0; i < 200000; i++) {
            const double mu = cos_max + (1.0 - cos_max) * U(gen), ph = 2.0 * M_PI * U(gen), s = std::sqrt(1 - mu * mu);
            const vec3 v(mu, s * std::cos(ph), s * std::sin(ph));
            c.increase_ipix_count(hb.vec2pix(v), 1.0);
            direct[direct.vec2pix(v)] += 1.0;
        }
        const double sparsity = c.chunk_sparsity();
        Healpix_Map<double> full;
        c.reconstruct_full_map(full);
        bool same = full.Scheme() == RING;
        for (long i = 0; same and i < 12 * nside * nside; i++)
            same = full[i] == direct[i];
        check(same, "Chunkpix map = the map filled directly (RING)");
        check(sparsity < 0.12, "Chunkpix: a 5% sky cap allocates < 12% of the chunks (" + std::to_string(sparsity) + ")");
    }
    {
        // nside not a power of two: RING chunks (NEST is not defined), same map
        FML::CHUNKPIX::Chunkpix<double> c;
        c.init(48, 8);
        Healpix_Base hb(48, c.ordering(), SET_NSIDE);
        Healpix_Map<double> direct(48, RING, SET_NSIDE);
        direct.fill(0.0);
        std::mt19937 gen(5);
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        for (int i = 0; i < 1000; i++) {
            const vec3 v(U(gen), U(gen), U(gen));
            c.increase_ipix_count(hb.vec2pix(v), 0.5);
            direct[direct.vec2pix(v)] += 0.5;
        }
        Healpix_Map<double> full;
        c.reconstruct_full_map(full);
        bool same = c.ordering() == RING;
        for (long i = 0; same and i < 12 * 48 * 48; i++)
            same = full[i] == direct[i];
        check(same, "Chunkpix with nside = 48 (RING chunks) = the map filled directly");
    }
    {
        // OnionSlice with chunkpix: the chunks are made into full maps before the sum over tasks and the result
        // is the same as without chunkpix
        OnionSlice<double, 3> a, b;
        a.init(0.0, 1.0, 0.5, 1.0, 16, true, 4);
        b.init(0.0, 1.0, 0.5, 1.0, 16, false, 0);
        std::mt19937 gen(7 + FML::ThisTask);
        std::uniform_real_distribution<double> U(-1, 1);
        for (int i = 0; i < 100; i++) {
            double pos[3] = {U(gen), U(gen), U(gen)};
            a.add_particle_to_map(a.get_pixel_index(pos));
            b.add_particle_to_map(b.get_pixel_index(pos));
        }
        bool ok = true;
        try {
            a.reduce_over_tasks();
            b.reduce_over_tasks();
            ok = a.get_npix() == b.get_npix() and a.get_npix() == 12 * 16 * 16;
            if (FML::ThisTask == 0) {
                double total = 0.0;
                for (long i = 0; ok and i < a.get_npix(); i++) {
                    ok = a.get_map_data()[i] == b.get_map_data()[i];
                    total += a.get_map_data()[i];
                }
                ok = ok and total == 100.0 * FML::NTasks;
            }
        } catch (std::exception &) {
            ok = false;
        }
        check(all_tasks(ok), "OnionSlice with chunkpix: reduce_over_tasks gives the same map as without chunkpix");
    }
}
#endif

// Optional argument: run only the test group with this name (lattice, create, dump, custom, boxes, reflection,
// shift, chunkpix, migration)
template <class T> long long migration_id(const T &p) {
    if constexpr (std::is_same_v<T, VarParticle>)
        return std::llround(p.extra[0]);
    else
        return p.id;
}

template <class T>
void migration_case(std::vector<double> edges, int mode, bool empty_source, const std::string &label) {
    constexpr int per_rank = 12;
    const int ranks = FML::NTasks, rank = FML::ThisTask;
    auto count = [&](int source) {
        return edges[source] < edges[source + 1] && !(empty_source && source == 0) ? per_rank : 0;
    };
    auto position = [&](int source, int i) {
        if (mode == 4) {
            double edge = edges[(source + 1) % ranks];
            return i % 4 == 0 ? 0. : i % 4 == 1 ? 1. : i % 4 == 2 ? edge : std::nextafter(edge, 0.);
        }
        int dest = source;
        if (mode == 1)
            dest = i % 3 == 0 ? source : (source + (i % 3 == 1 ? ranks - 1 : 1)) % ranks;
        if (mode == 2 && source == 0 && i == 0)
            dest = (source + 2) % ranks;
        if (mode == 3)
            dest = (source * 5 + i * 3) % ranks;
        while (edges[dest] == edges[dest + 1])
            dest = (dest + 1) % ranks;
        return .5 * (edges[dest] + edges[dest + 1]);
    };
    auto normalized = [&](double x) {
        if (ranks > 1 && x == 1.)
            return std::nextafter(1., 0.);
        if (ranks > 1 && x == 0.)
            return std::numeric_limits<double>::min();
        return x;
    };
    // Independent linear interval scan for the truth catalogue.
    auto owner = [&](double x) {
        if (ranks == 1)
            return 0;
        for (int t = 0; t < ranks; ++t)
            if (x >= edges[t] && x < edges[t + 1])
                return t;
        return -1;
    };
    std::vector<T> initial(size_t(count(rank)));
    for (int i = 0; i < count(rank); ++i) {
        auto &p = initial[i];
        long long id = per_rank * rank + i;
        p.pos[0] = .5 * (edges[rank] + edges[rank + 1]);
        p.pos[1] = .125 * (i % 7);
        p.pos[2] = .25 * (i % 3);
        if constexpr (std::is_same_v<T, VarParticle>) {
            p.n = 1 + i % 4;
            for (int k = 0; k < p.n; ++k)
                p.extra[k] = k ? 1.25 * id + k : double(id);
        } else
            p.id = id;
    }
    using Container = FML::PARTICLE::MPIParticles<T>;
    using Route = FML::PARTICLE::MigrationRoute;
    Container automatic, general;
    automatic.create(initial.data(), initial.size(), size_t(per_rank * ranks), edges[rank], edges[rank + 1], false);
    general.create(initial.data(), initial.size(), size_t(per_rank * ranks), edges[rank], edges[rank + 1], false);
    for (auto &p : automatic)
        p.pos[0] = position(int(migration_id(p) / per_rank), int(migration_id(p) % per_rank));
    for (auto &p : general)
        p.pos[0] = position(int(migration_id(p) / per_rank), int(migration_id(p) % per_rank));
    automatic.communicate_particles(37 + rank);
    general.communicate_particles(37 + rank, rank == 0);
    std::vector<long long> actual, reference, expected;
    bool good = true, moved = false, distant = false;
    for (int source = 0; source < ranks; ++source)
        for (int i = 0; i < count(source); ++i) {
            int dest = owner(normalized(position(source, i)));
            good &= dest >= 0;
            moved |= dest != source;
            distant |= dest != source && dest != (source + 1) % ranks && dest != (source + ranks - 1) % ranks;
            if (dest == rank)
                expected.push_back(per_rank * source + i);
        }
    auto inspect = [&](Container &container, std::vector<long long> &ids) {
        for (auto &p : container) {
            long long id = migration_id(p);
            ids.push_back(id);
            if (id < 0 || id >= per_rank * ranks) {
                good = false;
                continue;
            }
            int source = int(id / per_rank), i = int(id % per_rank);
            good &=
                p.pos[0] == normalized(position(source, i)) && p.pos[1] == .125 * (i % 7) && p.pos[2] == .25 * (i % 3);
            if constexpr (std::is_same_v<T, VarParticle>) {
                good &= p.n == 1 + i % 4;
                for (int k = 0; k < p.n; ++k)
                    good &= p.extra[k] == (k ? 1.25 * id + k : double(id));
            }
        }
    };
    inspect(automatic, actual);
    inspect(general, reference);
    good &= actual == reference;
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    good &= actual == expected;
    const Route route = ranks == 1 ? Route::Serial
                        : !moved   ? Route::NoTraffic
                        : distant  ? Route::General
                                   : Route::Neighbors;
    good &= automatic.get_last_migration_route() == route &&
            general.get_last_migration_route() == (ranks == 1 ? Route::Serial : Route::General);
    check(all_tasks(good), label + ": exact IDs, fields, receive order and collective route match the reference");
}

struct ExtendedPositionParticle {
    long double pos[3]{};
    long long id = 0;
    long double *get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};
void test_extended_boundary_migration() {
    if (FML::NTasks == 1)
        return;
    ExtendedPositionParticle p{{.5L * (FML::xmin_domain + FML::xmax_domain), .25L, .75L}, FML::ThisTask};
    FML::PARTICLE::MPIParticles<ExtendedPositionParticle> particles;
    particles.create(&p, 1, size_t(FML::NTasks), FML::xmin_domain, FML::xmax_domain, false);
    const long double just_below = std::nextafter(static_cast<long double>(1. / FML::NTasks), 0.L);
    if (FML::ThisTask == 1)
        particles[0].pos[0] = just_below;
    particles.communicate_particles(13);
    bool good = particles.get_npart() == size_t(FML::ThisTask == 0 ? 2 : FML::ThisTask == 1 ? 0 : 1);
    for (auto &q : particles) {
        good &= q.pos[0] >= static_cast<long double>(FML::xmin_domain) &&
                q.pos[0] < static_cast<long double>(FML::xmax_domain);
        if (q.id == 1)
            good &= q.pos[0] == just_below;
    }
    check(all_tasks(good), "extended-precision boundary positions are not narrowed during ownership lookup");
}

void test_migration_routing() {
    test_extended_boundary_migration();
    std::vector<double> edges(FML::NTasks + 1);
    for (int t = 0; t <= FML::NTasks; ++t)
        edges[t] = double(t) / FML::NTasks;
    migration_case<Particle>(edges, 0, false, "no-traffic migration");
    migration_case<Particle>(edges, 1, false, "two-sided neighbor and periodic seam migration");
    migration_case<Particle>(edges, 2, false, "single-rank move and collective fallback decision");
    migration_case<VarParticle>(edges, 1, false, "variable-size neighbor records with split messages");
    migration_case<VarParticle>(edges, 3, false, "arbitrary variable-size destinations");
    migration_case<Particle>(edges, 4, false, "exact and adjacent slab/box boundaries");
    if (FML::NTasks > 1)
        migration_case<Particle>(edges, 1, true, "initially empty rank participates in neighbor exchange");
    for (double &x : edges)
        x *= x;
    migration_case<Particle>(edges, 1, false, "nonuniform slabs");
    if (FML::NTasks >= 3) {
        edges[1] = 0.;
        migration_case<VarParticle>(edges, 3, false, "empty slab is skipped by binary ownership lookup");
    }
}

int main(int argc, char ** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    auto run = [&](std::string name, void (*f)()) {
        if (only.empty() or only == name)
            f();
    };
    run("migration", test_migration_routing);
    run("lattice", test_lattice);
    run("create", test_create);
    run("dump", test_dump);
    run("custom", test_custom_serialisation);
    run("boxes", test_particles_in_boxes);
    run("reflection", test_reflection);
    run("shift", test_shift);
#ifdef USE_HEALPIX
    run("chunkpix", test_chunkpix);
#endif
    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All particle tests passed\n" :
                                   std::to_string(fails) + " particle test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
