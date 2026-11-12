// Tests of FML/Analysis: the context (expansion history, unit conversions), the SO definitions, FoF + SO + the
// overlap flags on particles with known halos, the mass function, the redistribution of moved particles and
// baryonification (particle count, ownership, hosts). Run with 1 or more MPI tasks: the results must not depend
// on the number of tasks.

#include <FML/Analysis/AnalysisContext.h>
#include <FML/Analysis/CosmicWeb.h>
#include <FML/Analysis/Minkowski.h>
#include <FML/Analysis/Baryonification.h>
#include <FML/Analysis/Halos.h>
#include <FML/Analysis/ParticleUtils.h>
#include <FML/Analysis/PowerSpectrum.h>
#include <FML/Analysis/Voids.h>
#include <FML/Analysis/DTFE.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <random>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::check_local;
using FML::TESTS::num;
namespace AN = FML::ANALYSIS;
int analysis_odr_check();

namespace {
    struct Particle {
        double pos[3];
        double vel[3];
        double mass{1.0};
        double * get_pos() { return pos; }
        double * get_vel() { return vel; }
        double get_mass() const { return mass; }
        int get_ndim() const { return 3; }
    };

    /// Uniform background (nback particles) plus Gaussian blobs (center, sigma in box units, number of
    /// particles). Every task makes the same set and keeps the particles in its domain
    std::vector<Particle> make_particles(size_t nback,
                                        const std::vector<std::array<double, 3>> & centers,
                                        const std::vector<double> & sigmas,
                                        const std::vector<size_t> & counts) {
        std::mt19937_64 rng(1234);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::normal_distribution<double> gauss(0.0, 1.0);
        std::vector<Particle> all;
        for (size_t i = 0; i < nback; i++)
            all.push_back({{uni(rng), uni(rng), uni(rng)}, {100.0, 0.0, 0.0}, 1.0});
        for (size_t h = 0; h < centers.size(); h++)
            for (size_t i = 0; i < counts[h]; i++) {
                Particle p{{0, 0, 0}, {0.0, 0.0, 0.0}, 1.0};
                for (int d = 0; d < 3; d++) {
                    double x = centers[h][d] + sigmas[h] * gauss(rng);
                    x -= std::floor(x);
                    p.pos[d] = x >= 1.0 ? 0.0 : x;
                    p.vel[d] = 50.0 * gauss(rng);
                }
                all.push_back(p);
            }
        std::vector<Particle> local;
        for (auto & p : all)
            if (p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain)
                local.push_back(p);
        return local;
    }

    AN::AnalysisContext make_context(double boxsize, double a) {
        AN::AnalysisContext c;
        c.boxsize = boxsize;
        c.scale_factor = a;
        c.background = AN::Background::LCDM(0.3, 0.7);
        c.mass_to_msunh = 1e10;
        c.velocity_to_kms = 1.0;
        return c;
    }
} // namespace

int main() {
    check_local(analysis_odr_check() == 1, "the headers are usable from several translation units");
    //=========================================================================
    // Context
    //=========================================================================
    {
        const auto b = AN::Background::LCDM(0.3, 0.7);
        check_local(std::abs(b.E(1.0) - 1.0) < 1e-14, "flat LCDM: E(1) = 1");
        check_local(std::abs(b.E(0.5) - std::sqrt(0.3 * 8 + 0.7)) < 1e-14, "LCDM E(a = 0.5)");
        check_local(std::abs(b.Omega_m(0.5) - 2.4 / 3.1) < 1e-14, "Omega_m(a)");
        const auto br = AN::Background::LCDM(0.3, 0.6999, 1e-4);
        check_local(std::abs(br.E(1e-3) - std::sqrt(1e-4 * 1e12 + 0.3 * 1e9 + 0.6999)) < 1e-6 * br.E(1e-3),
                    "LCDM with radiation");

        // A table of the same LCDM: exact at the nodes, close in between
        const std::string table = "test_analysis_hubble_" + std::to_string(FML::ThisTask) + ".txt";
        {
            std::ofstream out(table);
            out << "# a H/H0\n";
            for (int i = 0; i <= 400; i++) {
                const double a = std::exp(std::log(0.01) * (1.0 - i / 400.0));
                out.precision(17);
                out << a << " " << b.E(a) << "\n";
            }
        }
        const auto bt = AN::Background::from_table(table, 0.3);
        double err = 0.0;
        for (double a : {0.0101, 0.1, 0.33, 0.5, 0.77, 0.999})
            err = std::max(err, std::abs(bt.E(a) / b.E(a) - 1.0));
        check_local(err < 1e-4, "E(a) from a table (max rel. error " + num(err) + ")");
        check_local(FML::TESTS::throws<std::runtime_error>([&] { bt.E(2.0); }), "a outside the table is an error");
        std::remove(table.c_str());

        const auto c = make_context(500.0, 0.5);
        c.validate();
        check_local(std::abs(c.velocity_to_displacement() - 1.0 / (100.0 * 0.5 * b.E(0.5) * 500.0)) < 1e-18,
                    "velocity_to_displacement = 1 / (100 a E(a) Box)");
        // The particle mass of an Omega_m = 0.3 simulation with N = 256^3 in 500 Mpc/h: total mass from rho_crit
        const double rho_crit = AN::AnalysisContext::critical_density0_msunh_mpch3();
        check_local(std::abs(rho_crit / 2.775e11 - 1.0) < 1e-3, "rho_crit = 2.775e11 h^2 Msun/Mpc^3 (" + num(rho_crit, 8) + ")");
        const double total_mass = 0.3 * rho_crit * std::pow(500.0, 3) / 1e10;
        check_local(std::abs(c.Omega_of_particles(total_mass) - 0.3) < 1e-12, "Omega from the particle mass");
    }

    //=========================================================================
    // SO definitions
    //=========================================================================
    {
        check_local(AN::parse_so_definition("200m").Delta == 200.0 and AN::parse_so_definition("500c").reference == 'c' and
                        AN::parse_so_definition("vir").virial,
                    "parse SO definitions");
        bool all_throw = true;
        for (std::string bad : {"", "m", "200", "200x", "abc", "-200m", "2e2q", "200mm"})
            all_throw = all_throw and FML::TESTS::throws<std::runtime_error>([&] { AN::parse_so_definition(bad); });
        check_local(all_throw, "invalid SO definitions are rejected");

        const auto c = make_context(500.0, 1.0);
        check_local(std::abs(AN::so_overdensity("200m", c, 0.3) - 200.0) < 1e-12, "200m = 200 x mean");
        check_local(std::abs(AN::so_overdensity("200c", c, 0.3) - 200.0 / 0.3) < 1e-10, "200c = 200 / Omega");
        const double x = -0.7, Dvir = 18 * M_PI * M_PI + 82 * x - 39 * x * x;
        check_local(std::abs(AN::so_overdensity("vir", c, 0.3) - Dvir / 0.3) < 1e-10, "vir (Bryan & Norman 1998)");
        // Particles that are only part of the matter (Omega_p = 0.25 of Omega_m = 0.3), at a = 0.5
        const auto c2 = make_context(500.0, 0.5);
        const double E2 = c2.background.E(0.5) * c2.background.E(0.5);
        check_local(std::abs(AN::so_overdensity("200c", c2, 0.25) - 200.0 / (0.25 / (0.125 * E2))) < 1e-10,
                    "200c relative to the particles' own density");
    }

    //=========================================================================
    // FoF + SO + overlap on particles with three blobs; the small one is inside the largest
    //=========================================================================
    {
        const std::vector<std::array<double, 3>> centers{{0.3, 0.3, 0.3}, {0.7, 0.6, 0.2}, {0.3 + 0.012, 0.3, 0.3}};
        const std::vector<double> sigmas{0.004, 0.003, 0.0006};
        const std::vector<size_t> counts{4000, 2000, 150};
        auto parts = make_particles(32768, centers, sigmas, counts);
        auto context = make_context(100.0, 1.0);
        AN::HaloSettings settings;
        settings.linking_length = 0.2;
        settings.nmin_per_halo = 20;
        settings.buffer_length_mpch = 6.0;
        settings.so_definitions = {"200m", "500c"};
        const auto cat = AN::find_halos<3>(parts.data(), parts.size(), settings, context);
        long long nhalos = cat.halos.size(), noverlap = 0, nbig = 0;
        double mass_big = 0.0, mass_mid = 0.0;
        for (auto & h : cat.halos) {
            noverlap += h.so_overlap;
            if (h.np > 3000) {
                nbig++;
                mass_big = h.so_mass[0];
            }
            if (h.np > 1500 and h.np < 3000)
                mass_mid = h.so_mass[0];
        }
        FML::SumOverTasks(&nhalos);
        FML::SumOverTasks(&noverlap);
        FML::SumOverTasks(&nbig);
        FML::SumOverTasks(&mass_big);
        FML::SumOverTasks(&mass_mid);
        // The mean particle mass is 1e10 Msun/h; the blobs are far above the mean density at 200m
        check(nbig == 1, "FoF: one large halo (the blob plus its satellite are linked or the satellite is separate)");
        check(mass_big > 0.9 * 4000e10 and mass_big < 1.2 * 4150e10, "SO 200m mass of the large blob (" + num(mass_big) + ")");
        check(mass_mid > 0.9 * 2000e10 and mass_mid < 1.2 * 2000e10, "SO 200m mass of the second blob (" + num(mass_mid) + ")");
        check(nhalos >= 2 and nhalos <= 4, "FoF: number of halos (" + num(nhalos) + ")");
        check(std::abs(cat.particle_mass_msunh - 1e10) < 1e-3, "mean particle mass in Msun/h");
        check(std::abs(cat.so_thresholds[1] - 500.0 / cat.Omega_particles) < 1e-9, "500c threshold");

        // Mass function: n(>M) at the smallest bin counts all halos above its center, dn/dlnM integrates to it
        const auto mf = AN::compute_mass_function<3>(cat.halos,
                                                     [](const AN::HaloRecord<3> & h) { return h.mass; },
                                                     20,
                                                     1e11,
                                                     1e15,
                                                     context.boxsize);
        double integral = 0.0;
        for (auto v : mf.dndlnM)
            integral += v * std::log(1e4) / 20.0;
        check(std::abs(integral * std::pow(100.0, 3) - double(nhalos)) < 1e-9,
              "mass function: the bins hold all the halos");

        // Overlap flags: a halo inside the radius of a more massive one is flagged, isolated ones are not
        std::vector<AN::HaloRecord<3>> fake(3);
        for (auto & h : fake) {
            h.so_mass = {1.0};
            h.so_radius = {0.01};
        }
        fake[0].id = 0, fake[0].so_pos = {0.5, 0.5, 0.5}, fake[0].so_mass = {10.0};
        fake[1].id = 1, fake[1].so_pos = {0.505, 0.5, 0.5};
        fake[2].id = 2, fake[2].so_pos = {0.999, 0.5, 0.5}, fake[2].so_radius = {0.02}, fake[2].so_mass = {5.0};
        std::vector<AN::HaloRecord<3>> mine;
        if (FML::ThisTask == 0)
            mine = fake;
        AN::flag_overlapping_so_halos<3>(mine);
        if (FML::ThisTask == 0)
            check_local(not mine[0].so_overlap and mine[1].so_overlap and not mine[2].so_overlap,
                        "overlap flags (inside a more massive halo)");
        // Periodic: a small halo across the boundary from a massive one
        if (FML::ThisTask == 0) {
            mine = fake;
            mine[1].so_pos = {0.012, 0.5, 0.5};
        }
        AN::flag_overlapping_so_halos<3>(mine);
        if (FML::ThisTask == 0)
            check_local(mine[1].so_overlap, "overlap flags across the periodic boundary");
    }

    //=========================================================================
    // Redistribution: shift every particle by half the box and check ownership and count
    //=========================================================================
    {
        auto parts = make_particles(20000, {}, {}, {});
        FML::PARTICLE::MPIParticles<Particle> mp;
        mp.create(parts, parts.size(), [](Particle &) { return true; });
        const size_t ntotal = mp.get_npart_total();
        for (size_t i = 0; i < mp.get_npart(); i++) {
            double x = mp[i].pos[0] + 0.5;
            mp[i].pos[0] = x >= 1.0 ? x - 1.0 : x;
        }
        AN::redistribute_particles(mp);
        bool owned = true;
        for (size_t i = 0; i < mp.get_npart(); i++)
            owned = owned and mp[i].pos[0] >= FML::xmin_domain and mp[i].pos[0] < FML::xmax_domain;
        size_t n = mp.get_npart();
        FML::SumOverTasks(&n);
        check(n == ntotal, "redistribution keeps every particle");
        check(owned, "redistribution: every particle is on the task owning its position");
    }

    //=========================================================================
    // Baryonification of two blobs in 100 Mpc/h: hosts found, particles conserved and owned, P(k) changes only on
    // small scales
    //=========================================================================
    {
        const std::vector<std::array<double, 3>> centers{{0.3, 0.3, 0.3}, {0.7, 0.6, 0.2}};
        auto parts = make_particles(65536, centers, {0.006, 0.005}, {6000, 4000});
        // Mass so that Omega = 0.3 in 100 Mpc/h
        const double ntot = 65536 + 10000;
        const double m = 0.3 * AN::AnalysisContext::critical_density0_msunh_mpch3() * 1e6 / ntot / 1e10;
        for (auto & p : parts)
            p.mass = m;
        FML::PARTICLE::MPIParticles<Particle> mp;
        mp.create(parts, size_t(parts.size() * 1.25) + 16, [](Particle &) { return true; });
        auto context = make_context(100.0, 1.0);
        AN::BaryonificationSettings s;
        s.max_radius_over_r200 = 10.0;
        s.min_halo_mass_msunh = 1e13;
        s.fof_buffer_mpch = 8.0; // R200c of the blobs is ~3 Mpc/h
        AN::PowerSpectrumSettings ps;
        ps.nmesh = 32;
        const auto before = AN::measure_power_spectrum<3>(mp.get_particles_ptr(), mp.get_npart(), mp.get_npart_total(), ps, context);
        const auto result = AN::baryonify(mp, s, context, 0.15);
        const auto after = AN::measure_power_spectrum<3>(mp.get_particles_ptr(), mp.get_npart(), mp.get_npart_total(), ps, context);
        bool owned = true;
        for (size_t i = 0; i < mp.get_npart(); i++)
            owned = owned and mp[i].pos[0] >= FML::xmin_domain and mp[i].pos[0] < FML::xmax_domain;
        check(mp.get_npart_total() == size_t(ntot), "baryonification keeps every particle");
        check(owned, "baryonification: every particle is on the task owning its new position");
        check(result.hosts.size() == 2, "baryonification: two hosts (" + num(result.hosts.size()) + ")");
        check(std::abs(result.Delta_200c - 200.0 / 0.3) < 1e-6, "baryonification: 200c from the particle mass");
        check(std::abs(after.pofk[0] / before.pofk[0] - 1.0) < 1e-3, "P(k) unchanged on the largest scale");
        double change = 0.0;
        for (int i = 0; i < after.n; i++)
            change = std::max(change, std::abs(after.pofk[i] / before.pofk[i] - 1.0));
        check(change > 1e-3, "P(k) changed on small scales (max " + num(change) + ")");
        check(FML::TESTS::throws<std::exception>([&] { AN::baryonify(mp, s, context, 1.5); }),
              "baryonify rejects an invalid baryon fraction");
    }

    //=========================================================================
    // Voids: a spherical hole of radius 0.15 (box units) in a uniform particle distribution
    //=========================================================================
    {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<Particle> local;
        for (size_t i = 0; i < 60000; i++) {
            Particle p{{uni(rng), uni(rng), uni(rng)}, {0, 0, 0}, 1.0};
            double r2 = 0.0;
            for (int d = 0; d < 3; d++)
                r2 += (p.pos[d] - 0.5) * (p.pos[d] - 0.5);
            if (r2 < 0.15 * 0.15)
                continue;
            if (p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain)
                local.push_back(p);
        }
        AN::VoidSettings vs;
        vs.nmesh = 32;
        auto distance_to_hole = [](const FML::VOIDFINDER::Void & v) {
            double r2 = 0.0;
            for (int d = 0; d < 3; d++) {
                double dx = v.pos[d] - 0.5;
                dx -= std::round(dx);
                r2 += dx * dx;
            }
            return std::sqrt(r2);
        };
        auto largest = [](const std::vector<FML::VOIDFINDER::Void> & voids) {
            size_t imax = 0;
            for (size_t i = 0; i < voids.size(); i++)
                if (voids[i].radius > voids[imax].radius)
                    imax = i;
            return imax;
        };
        const auto voxel = AN::find_voxel_voids(local.data(), local.size(), vs);
        const bool ok_voxel = not voxel.empty() and distance_to_hole(voxel[largest(voxel)]) < 0.05;
        check(ok_voxel, "voxel voids: the largest void is the hole");
        AN::SphericalVoidSettings ss;
        const auto sph = AN::find_spherical_voids(local.data(), local.size(), ss, vs);
        const bool ok_sph = not sph.empty() and distance_to_hole(sph[largest(sph)]) < 0.05 and
                            sph[largest(sph)].radius > 0.1 and sph[largest(sph)].radius < 0.2;
        check(ok_sph, "spherical voids: the largest void is the hole, with a radius near 0.15");
        auto bad = vs;
        bad.merging = "sometimes";
        check(FML::TESTS::throws<std::runtime_error>([&] { bad.validate(); }), "invalid void settings are rejected");
#ifdef USE_CGAL
        AN::ZobovSettings zs;
        const auto zobov = AN::find_zobov_voids(local.data(), local.size(), zs, vs);
        check(not zobov.empty() and distance_to_hole(zobov[largest(zobov)]) < 0.05,
              "ZOBOV: the largest void is the hole");
        // DTFE of a uniform distribution with a uniform velocity: density ~1 on average, velocity in km/s
        for (auto & p : local)
            p.vel[0] = 10.0, p.vel[1] = p.vel[2] = 0.0;
        AN::DTFESettings ds;
        ds.nmesh = 16;
        ds.buffer_fraction = 1.0; // the tetrahedra across the hole need particles beyond half a slab with 4 tasks
        auto context = make_context(100.0, 1.0);
        context.velocity_to_kms = 2.0;
        const auto fields = AN::compute_dtfe_fields(local.data(), local.size(), ds, context);
        double vmean = 0.0, cells = 0.0;
        for (int ix = 0; ix < fields.density.get_local_nx(); ix++)
            for (int iy = 0; iy < 16; iy++)
                for (int iz = 0; iz < 16; iz++) {
                    vmean += fields.velocity.velocity[0].get_real({ix, iy, iz});
                    cells += 1.0;
                }
        FML::SumOverTasks(&vmean);
        FML::SumOverTasks(&cells);
        check(std::abs(vmean / cells - 20.0) < 1e-6, "DTFE velocity in km/s (" + num(vmean / cells) + ")");
#endif
    }

    //=========================================================================
    // Relaxed FoF (fof_relaxed): the catalogue equals RelaxedFriendsOfFriends run directly, a hot clump (far too
    // fast to be bound) is rejected while standard FoF keeps it, the extra catalogue columns, the settings checks
    //=========================================================================
    {
        const std::array<double, 3> hot{0.8, 0.8, 0.8};
        const std::vector<std::array<double, 3>> centers{{0.3, 0.3, 0.3}, {0.7, 0.6, 0.2}, hot};
        auto parts = make_particles(32768, centers, {0.004, 0.003, 0.0006}, {4000, 2000, 150});
        for (auto & p : parts) {
            double r2 = 0.0;
            for (int d = 0; d < 3; d++)
                r2 += (p.pos[d] - hot[d]) * (p.pos[d] - hot[d]);
            if (r2 < 0.005 * 0.005)
                for (auto & v : p.vel)
                    v *= 200.0; // sigma_3D ~ 17000 km/s, V_std(1.5e12 Msun/h) ~ 125 km/s
        }
        const auto context = make_context(100.0, 1.0);
        AN::HaloSettings settings;
        settings.buffer_length_mpch = 6.0;
        settings.relaxed = true;
        auto copy = parts;
        const auto cat = AN::find_halos<3>(parts.data(), parts.size(), settings, context);
        const auto direct = FML::FOF::RelaxedFriendsOfFriends<Particle, 3>(
            copy.data(), copy.size(), settings.relaxed_parameters(), 0.0, 1.0, 1e10, 1.0, true, 0.06);

        long long n_cat = cat.halos.size(), n_direct = direct.size(), np_cat = 0, np_direct = 0, bad = 0, near_hot = 0;
        for (auto & h : cat.halos) {
            np_cat += h.np;
            bad += (h.rfof_bin < 1 or h.rfof_bin > 6 or h.rfof_pass != h.rfof_bin - 1 or
                    not(h.rfof_vdisp_ratio < 7.02) or not(h.rfof_linking_length >= 0.2 and h.rfof_linking_length <= 0.238 + 1e-12));
            double r2 = 0.0;
            for (int d = 0; d < 3; d++) {
                double dx = h.pos[d] / 100.0 - hot[d];
                r2 += dx * dx;
            }
            near_hot += r2 < 0.01 * 0.01;
        }
        for (auto & h : direct)
            np_direct += h.np;
        for (auto * v : {&n_cat, &n_direct, &np_cat, &np_direct, &bad, &near_hot})
            FML::SumOverTasks(v);
        check(n_cat == n_direct and np_cat == np_direct and n_cat >= 2,
              "relaxed FoF: the catalogue equals RelaxedFriendsOfFriends (" + num(n_cat) + " halos)");
        check(bad == 0, "relaxed FoF: pass, bin, linking length and sigma_v/V_std of every halo");
        check(near_hot == 0, "relaxed FoF: the hot clump is rejected");
        check(cat.relaxed and cat.nmin_per_halo == 8, "relaxed FoF: the catalogue records the settings");

        AN::HaloSettings standard;
        standard.buffer_length_mpch = 6.0;
        standard.nmin_per_halo = 20;
        auto copy2 = parts;
        const auto cat_std = AN::find_halos<3>(copy2.data(), copy2.size(), standard, context);
        long long hot_std = 0;
        for (auto & h : cat_std.halos) {
            double r2 = 0.0;
            for (int d = 0; d < 3; d++) {
                double dx = h.pos[d] / 100.0 - hot[d];
                r2 += dx * dx;
            }
            hot_std += r2 < 0.01 * 0.01;
        }
        FML::SumOverTasks(&hot_std);
        check(hot_std == 1, "standard FoF keeps the hot clump");

        // The text catalogue: 4 more columns
        const std::string file = "test_analysis_rfof_halos.txt";
        AN::write_halo_catalogue_ascii<3>(file, cat);
        if (FML::ThisTask == 0) {
            std::ifstream in(file);
            std::string header, row;
            std::getline(in, header);
            std::getline(in, row);
            std::stringstream ss(row);
            int ncols = 0;
            std::string word;
            while (ss >> word)
                ncols++;
            check_local(header.find("rfof: pass  bin  linking_length  sigma_v/V_std") != std::string::npos and
                            ncols == 11 + 4,
                        "relaxed FoF: the catalogue has the 4 rfof columns (" + num(ncols) + " columns)");
            std::remove(file.c_str());
        }

        AN::HaloSettings bad_mode;
        bad_mode.relaxed = true;
        bad_mode.relaxed_mode = "fast";
        AN::HaloSettings bad_edges;
        bad_edges.relaxed = true;
        bad_edges.relaxed_bin_edges = {20, 10};
        check_local(FML::TESTS::throws<std::runtime_error>([&] { bad_mode.validate(); }) and
                        FML::TESTS::throws<std::runtime_error>([&] { bad_edges.validate(); }),
                    "relaxed FoF: invalid fof_relaxed_mode and bin edges are rejected");
        check_local(FML::TESTS::throws<std::runtime_error>([&] {
                        AN::find_fof_halos<3, Particle>(copy.data(), copy.size(), settings, 100.0, 0.2, 20);
                    }),
                    "relaxed FoF without the context is rejected");
    }

    //=========================================================================
    // Bispectrum (bispectrum = true): measure_bispectrum equals compute_bispectrum, k in h/Mpc, the file
    //=========================================================================
    {
        auto parts = make_particles(8000, {{0.3, 0.3, 0.3}, {0.7, 0.6, 0.2}}, {0.02, 0.03}, {3000, 2000});
        auto copy = parts;
        const auto context = make_context(200.0, 1.0);
        AN::BispectrumSettings settings;
        settings.nmesh = 32;
        settings.nbins = 6;
        size_t ntot = parts.size();
        FML::SumOverTasks(&ntot);
        const auto bofk = AN::measure_bispectrum<3>(parts.data(), parts.size(), ntot, settings, context);
        using Binning = FML::CORRELATIONFUNCTIONS::BispectrumBinning<3>;
        Binning direct(0.0, 2.0 * M_PI * 16, 6, Binning::LINEAR_SPACING);
        direct.subtract_shotnoise = false;
        FML::CORRELATIONFUNCTIONS::compute_bispectrum(32, copy.data(), copy.size(), ntot, direct, "CIC", true);
        direct.scale(200.0);
        auto b = bofk;
        double maxdiff = 0.0, maxb = 0.0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                for (int k = 0; k < 6; k++) {
                    maxdiff = std::max(maxdiff, std::abs(b.get_spectrum(i, j, k) - direct.get_spectrum(i, j, k)));
                    maxb = std::max(maxb, std::abs(b.get_spectrum(i, j, k)));
                }
        check(maxdiff <= 1e-12 * maxb and maxb > 0.0,
              "bispectrum: measure_bispectrum equals compute_bispectrum (relative difference " + num(maxdiff / maxb) + ")");
        // The bins: linear from 0 to the Nyquist frequency pi nmesh / Box (h/Mpc)
        const double dk = M_PI * 32 / 200.0 / 6;
        bool kok = true;
        for (int i = 1; i < 6; i++)
            kok = kok and b.kmean[i] > i * dk and b.kmean[i] < (i + 1) * dk;
        check(kok, "bispectrum: the mean k of the bins in h/Mpc");
        // Clustered particles (two blobs): B > 0 for the equilateral triangles of the largest scales
        check(b.get_spectrum(1, 1, 1) > 0.0 and b.get_reduced_spectrum(1, 1, 1) > 0.0,
              "bispectrum: positive equilateral B and Q for clustered particles");

        const std::string file = "test_analysis_bispectrum.txt";
        AN::write_bispectrum<3>(file, bofk, "# test\n");
        if (FML::ThisTask == 0) {
            std::ifstream in(file);
            std::string line;
            int rows = 0, comments = 0;
            bool ncol_ok = true;
            while (std::getline(in, line)) {
                if (line.empty())
                    continue;
                if (line[0] == '#') {
                    comments++;
                    continue;
                }
                std::stringstream ss(line);
                std::string word;
                int ncols = 0;
                while (ss >> word)
                    ncols++;
                ncol_ok = ncol_ok and ncols == 9;
                rows++;
            }
            check_local(rows == 216 and ncol_ok and comments == 3,
                        "bispectrum: the file has nbins^3 rows of SimulationCode's 9 columns");
            std::remove(file.c_str());
        }

        AN::BispectrumSettings bad;
        bad.nbins = 0;
        AN::BispectrumSettings bad_method;
        bad_method.density_assignment_method = "XYZ";
        check_local(FML::TESTS::throws<std::runtime_error>([&] { bad.validate(); }) and
                        FML::TESTS::throws<std::runtime_error>([&] { bad_method.validate(); }),
                    "bispectrum: invalid settings are rejected");
    }

    //=========================================================================
    // Cosmic web (T-web) and Minkowski functionals on a Gaussian random field: the known fractions at lambda_th = 0
    // (0.08, 0.42, 0.42, 0.08; Doroshkevich 1970) and the Gaussian-field Minkowski functionals; the particle front
    // ends equal the grid functions; the files; the settings checks
    //=========================================================================
    {
        // White noise on a 96^3 grid (box 96 Mpc/h: 1 Mpc/h cells), the same on any number of tasks
        const int N = 96;
        const double box = 96.0;
        FML::GRID::FFTWGrid<3> noise(N);
        auto hash = [](uint64_t x) {
            x += 0x9e3779b97f4a7c15ULL;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            return x ^ (x >> 31);
        };
        for (int ix = 0; ix < noise.get_local_nx(); ix++)
            for (int iy = 0; iy < N; iy++)
                for (int iz = 0; iz < N; iz++) {
                    const uint64_t i = (uint64_t(noise.get_local_x_start() + ix) * N + iy) * N + iz;
                    const double u1 = (double(hash(2 * i) >> 11) + 0.5) * 0x1.0p-53;
                    const double u2 = double(hash(2 * i + 1) >> 11) * 0x1.0p-53;
                    noise.set_real({ix, iy, iz}, std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2));
                }

        AN::CosmicWebSettings web;
        web.smoothing_scale_mpch = 3.0;
        web.lambda_thresholds = {0.0, 0.005, 0.01}; // the smoothed field has sigma ~ 0.03
        const auto fr = AN::cosmic_web_from_delta(noise, web, box);
        const auto & v = fr.volume_fraction[0];
        check(std::abs(v[0] - 0.08) < 0.012 and std::abs(v[1] - 0.42) < 0.02 and std::abs(v[2] - 0.42) < 0.02 and
                  std::abs(v[3] - 0.08) < 0.012,
              "T-web of a Gaussian field: fractions at lambda_th = 0 (" + num(v[0], 3) + ", " + num(v[1], 3) + ", " +
                  num(v[2], 3) + ", " + num(v[3], 3) + ") vs (0.08, 0.42, 0.42, 0.08)");
        bool sums = true, more_voids = true;
        for (size_t i = 0; i < fr.volume_fraction.size(); i++) {
            double sv = 0.0, sm = 0.0;
            for (int k = 0; k < 4; k++) {
                sv += fr.volume_fraction[i][k];
                sm += fr.mass_fraction[i][k];
            }
            sums = sums and std::abs(sv - 1.0) < 1e-12 and std::abs(sm - 1.0) < 1e-9;
            if (i > 0)
                more_voids = more_voids and fr.volume_fraction[i][0] > fr.volume_fraction[i - 1][0] and
                             fr.volume_fraction[i][3] < fr.volume_fraction[i - 1][3];
        }
        std::string info;
        for (size_t i = 0; i < fr.volume_fraction.size(); i++) {
            double sv = 0.0, sm = 0.0;
            for (int k = 0; k < 4; k++) {
                sv += fr.volume_fraction[i][k];
                sm += fr.mass_fraction[i][k];
            }
            info += " [" + num(sv - 1.0, 3) + " " + num(sm - 1.0, 3) + " V " + num(fr.volume_fraction[i][0], 4) + " K " +
                    num(fr.volume_fraction[i][3], 4) + "]";
        }
        check(sums and more_voids, "T-web: the volume and mass fractions sum to 1; more voids, fewer knots for a "
                                   "higher threshold" + info);

        AN::MinkowskiSettings mink;
        mink.smoothing_scale_mpch = 4.0;
        mink.threshold_min = -2.0;
        mink.threshold_max = 2.0;
        mink.nthresholds = 9;
        const auto mf = AN::minkowski_from_delta(noise, mink, box);
        // sigma_1^2 / sigma_0^2 of smoothed white noise: 3 / (2 R^2) (up to the grid)
        const double ratio = mf.sigma1 * mf.sigma1 / (mf.sigma0 * mf.sigma0) * 2.0 * 16.0 / 3.0;
        check(std::abs(ratio - 1.0) < 0.05, "Minkowski: sigma_1^2 / sigma_0^2 = 3 / (2 R^2) for smoothed white noise (" +
                                                num(ratio, 4) + ")");
        std::array<double, 4> worst{}, amp{};
        for (size_t i = 0; i < mf.nu.size(); i++)
            for (int k = 0; k < 4; k++) {
                worst[k] = std::max(worst[k], std::abs(mf.measured[i][k] - mf.gaussian[i][k]));
                amp[k] = std::max(amp[k], std::abs(mf.gaussian[i][k]));
            }
        check(worst[0] < 0.01 and worst[1] < 0.06 * amp[1] and worst[2] < 0.1 * amp[2] and worst[3] < 0.12 * amp[3],
              "Minkowski functionals of a Gaussian field = the Gaussian predictions (max |diff| / max |G|: " +
                  num(worst[0] / amp[0], 2) + ", " + num(worst[1] / amp[1], 2) + ", " + num(worst[2] / amp[2], 2) +
                  ", " + num(worst[3] / amp[3], 2) + ")");
        bool euler_ok = true;
        for (size_t i = 0; i < mf.nu.size(); i++)
            euler_ok = euler_ok and std::abs(mf.measured[i][3] * box * box * box - mf.euler_characteristic[i]) < 1e-6;
        check(euler_ok, "Minkowski: V3 = Euler characteristic / volume");

        // The particle front ends: the same as the grid functions on the density of the particles
        auto parts = make_particles(20000, {{0.3, 0.3, 0.3}, {0.7, 0.6, 0.2}}, {0.03, 0.05}, {4000, 3000});
        const auto context = make_context(200.0, 1.0);
        AN::CosmicWebSettings pweb;
        pweb.nmesh = 32;
        AN::MinkowskiSettings pmink;
        pmink.nmesh = 32;
        pmink.smoothing_scale_mpch = 20.0;
        const auto web_p = AN::measure_cosmic_web(parts.data(), parts.size(), pweb, context);
        const auto mink_p = AN::measure_minkowski_functionals(parts.data(), parts.size(), pmink, context);
        const auto dgrid = AN::density_contrast_grid(parts.data(), parts.size(), 32, "CIC");
        const auto web_g = AN::cosmic_web_from_delta(dgrid, pweb, 200.0);
        const auto mink_g = AN::minkowski_from_delta(dgrid, pmink, 200.0);
        check(web_p.volume_fraction == web_g.volume_fraction and web_p.mass_fraction == web_g.mass_fraction and
                  mink_p.measured == mink_g.measured and mink_p.euler_characteristic == mink_g.euler_characteristic,
              "measure_cosmic_web / measure_minkowski_functionals equal the grid functions on the particle density");

        const std::string f1 = "test_analysis_cosmicweb.txt", f2 = "test_analysis_minkowski.txt";
        AN::write_cosmic_web(f1, fr);
        AN::write_minkowski_functionals(f2, mf);
        if (FML::ThisTask == 0) {
            auto count_rows = [](const std::string & file, int & ncols) {
                std::ifstream in(file);
                std::string line;
                int rows = 0;
                while (std::getline(in, line)) {
                    if (line.empty() or line[0] == '#')
                        continue;
                    std::stringstream ss(line);
                    std::string w;
                    ncols = 0;
                    while (ss >> w)
                        ncols++;
                    rows++;
                }
                return rows;
            };
            int c1 = 0, c2 = 0;
            const int r1 = count_rows(f1, c1), r2 = count_rows(f2, c2);
            check_local(r1 == 3 and c1 == 9 and r2 == 9 and c2 == 13,
                        "the cosmic-web (3 x 9) and Minkowski (9 x 13) files");
            std::remove(f1.c_str());
            std::remove(f2.c_str());
        }

        AN::CosmicWebSettings bad_web;
        bad_web.lambda_thresholds = {};
        AN::MinkowskiSettings bad_mink;
        bad_mink.threshold_variable = "sigma";
        AN::MinkowskiSettings bad_range;
        bad_range.threshold_max = -5.0;
        check_local(FML::TESTS::throws<std::runtime_error>([&] { bad_web.validate(); }) and
                        FML::TESTS::throws<std::runtime_error>([&] { bad_mink.validate(); }) and
                        FML::TESTS::throws<std::runtime_error>([&] { bad_range.validate(); }),
                    "cosmic-web and Minkowski settings checks");
    }

    return FML::TESTS::summary("Analysis");
}
