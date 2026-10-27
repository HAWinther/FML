#ifndef HALOFINDING_HEADER
#define HALOFINDING_HEADER

#include <FML/FriendsOfFriends/FoF.h>
#include <FML/FriendsOfFriends/SphericalOverdensity.h>
#include <FML/GadgetUtils/GadgetUtilsHDF5.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Spline/Spline.h>

#include "Species.h"

#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

//=============================================================================
// Friends-of-friends halos at the outputs (fof = true): a halo catalogue (text or HDF5) and the mass function
// with the Tinker et al. 2008 (Delta = 200 mean) and Watson et al. 2013 (FoF) fits for comparison.
//
// Optionally (fof_so_definitions, e.g. {"200m", "200c", "vir"}) the spherical overdensity masses around the FoF halos:
// shrinking sphere center, then the radius where the mean enclosed density of all the CDM+baryon particles is Delta
// times the reference density (m: the mean CDM+baryon density, c: the critical density, vir: Bryan & Norman 1998),
// with a mass function per definition and the Tinker et al. 2008 fit for the same Delta.
//
// The halos are made of the CDM+baryon particles (the neutrino particles, if any, are not included). With CDM and
// baryons as separate particles both are linked together (particles with masses Omega_i / OmegaCB) and the linking
// length is in units of the mean separation of the particles of one species. Masses are in Msun/h from OmegaCB.
//=============================================================================

template <int NDIM, class T>
class NBodySimulation;

namespace COLASOLVER {

    /// Equal-mass CDM+baryon particle mass in Msun/h (boxsize in comoving Mpc/h).
    inline double halo_particle_mass_msunh(double omega_cb, double boxsize, double H0_hmpc, size_t npart_total) {
        const double MplMpl_over_H0Msunh = 2.49264e21;
        return 3.0 * omega_cb * MplMpl_over_H0Msunh * std::pow(boxsize * H0_hmpc, 3) / double(npart_total);
    }

    /// A halo as written to the catalogue (code units: positions in [0,1), velocities in code units, mass in units
    /// of the mass of a CDM+baryon particle of one species)
    template <int NDIM>
    struct HaloRecord {
        long long int id;
        long long int np;
        double mass;
        std::array<double, NDIM> pos, vel, vel_rms;
        // Spherical overdensity (if fof_so_definitions is not empty): center, velocity (inside the radius of the first
        // definition), and mass, radius, number of particles for each definition
        std::array<double, NDIM> so_pos{}, so_vel{}, so_vel_rms{};
        std::vector<double> so_mass, so_radius;
        std::vector<long long int> so_npart;
        bool so_truncated{false}; // The sphere reached the search radius (the buffer)
        bool so_overlap{false};   // The SO center is inside the radius (first definition) of a more massive halo
    };

    /// The particle used to link CDM and baryons together: position, velocity and mass
    template <int NDIM>
    struct FoFParticle {
        double pos[NDIM];
        double vel[NDIM];
        double mass;
        double * get_pos() { return pos; }
        double * get_vel() { return vel; }
        double get_mass() const { return mass; }
        constexpr int get_ndim() const { return NDIM; }
    };

    /// Run the FoF finder and return the halos found on this task
    template <int NDIM, class P>
    std::vector<HaloRecord<NDIM>> find_fof_halos(P * part,
                                                size_t npart,
                                                const FoFSettings & settings,
                                                double boxsize,
                                                double linking_length,
                                                int nmin_per_halo,
                                                const std::vector<double> & so_thresholds = {},
                                                double total_mass = 0.0) {
        std::vector<FML::FOF::FoFHalo<P, NDIM>> groups;
        FML::FOF::FriendsOfFriends<P, NDIM>(part,
                                            npart,
                                            linking_length,
                                            nmin_per_halo,
                                            true,
                                            settings.buffer_length_mpch / boxsize,
                                            groups,
                                            settings.nmesh_max);
        std::vector<HaloRecord<NDIM>> halos;
        halos.reserve(groups.size());
        for (auto & g : groups) {
            HaloRecord<NDIM> h;
            h.id = (long long int)g.id;
            h.np = (long long int)g.np;
            h.mass = g.mass;
            for (int idim = 0; idim < NDIM; idim++) {
                h.pos[idim] = g.pos[idim];
                h.vel[idim] = g.vel[idim];
                h.vel_rms[idim] = g.vel_rms[idim];
            }
            halos.push_back(h);
        }

        // Spherical overdensity around the FoF halos. The first shrinking sphere has 1.5 times the radius of a
        // sphere with the FoF mass and 200 times the mean density
        if (not so_thresholds.empty()) {
            std::vector<std::array<double, NDIM>> centers;
            std::vector<double> initial_radius;
            for (auto & h : halos) {
                centers.push_back(h.pos);
                initial_radius.push_back(1.5 * std::cbrt(3.0 * h.mass / (4.0 * M_PI * 200.0 * total_mass)));
            }
            auto so = FML::FOF::SphericalOverdensity<P, NDIM>(
                part, npart, total_mass, centers, initial_radius, so_thresholds, settings.buffer_length_mpch / boxsize);
            for (size_t i = 0; i < halos.size(); i++) {
                auto & h = halos[i];
                h.so_pos = so[i].pos;
                h.so_vel = so[i].vel;
                h.so_vel_rms = so[i].vel_rms;
                h.so_mass = so[i].mass;
                h.so_radius = so[i].radius;
                h.so_npart = so[i].npart;
                h.so_truncated = so[i].truncated;
            }
        }
        return halos;
    }

    /// The overdensity of a SO definition in units of the mean CDM+baryon density at scale factor a: "<Delta>m"
    /// (or b: mean CDM+baryon density), "<Delta>c" (critical density) or "vir" (Bryan & Norman 1998)
    inline double so_overdensity(const std::string & definition, const Cosmology & cosmo, double a) {
        const double OmegaCB_a = cosmo.get_Omegab(a) + cosmo.get_OmegaCDM(a);
        if (definition == "vir") {
            const double x = cosmo.get_OmegaM(a) - 1.0;
            const double Delta_c = 18.0 * M_PI * M_PI + 82.0 * x - 39.0 * x * x;
            return Delta_c / OmegaCB_a;
        }
        if (definition.size() < 2)
            throw std::runtime_error("Unknown SO definition [" + definition + "] (examples: 200m, 500c, vir)");
        const char type = definition.back();
        double Delta = 0.0;
        try {
            Delta = std::stod(definition.substr(0, definition.size() - 1));
        } catch (...) {
            throw std::runtime_error("Unknown SO definition [" + definition + "] (examples: 200m, 500c, vir)");
        }
        if (not(Delta > 0.0) or (type != 'm' and type != 'b' and type != 'c'))
            throw std::runtime_error("Unknown SO definition [" + definition + "] (examples: 200m, 500c, vir)");
        return type == 'c' ? Delta / OmegaCB_a : Delta;
    }

    /// The Tinker et al. 2008 multiplicity function for an overdensity Delta (with respect to the mean density) in
    /// 200 <= Delta <= 3200: the parameters of their Table 2 interpolated in log(Delta) and their redshift
    /// dependence (eqs. 5-8). Returns 0 outside this range
    inline double f_tinker2008_delta(double sigma, double z, double Delta) {
        static const double table_Delta[9] = {200, 300, 400, 600, 800, 1200, 1600, 2400, 3200};
        static const double table_A[9] = {0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260};
        static const double table_a[9] = {1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66};
        static const double table_b[9] = {2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41};
        static const double table_c[9] = {1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44};
        if (Delta < 200.0 * (1.0 - 1e-10) or Delta > 3200.0 * (1.0 + 1e-10))
            return 0.0;
        int i = 0;
        while (i < 7 and Delta > table_Delta[i + 1])
            i++;
        const double t = std::log(Delta / table_Delta[i]) / std::log(table_Delta[i + 1] / table_Delta[i]);
        auto interp = [&](const double * v) { return v[i] + t * (v[i + 1] - v[i]); };
        const double alpha = std::pow(10.0, -std::pow(0.75 / std::log10(Delta / 75.0), 1.2));
        const double A = interp(table_A) * std::pow(1.0 + z, -0.14);
        const double a = interp(table_a) * std::pow(1.0 + z, -0.06);
        const double b = interp(table_b) * std::pow(1.0 + z, -alpha);
        const double c = interp(table_c);
        return A * (std::pow(sigma / b, -a) + 1.0) * std::exp(-c / (sigma * sigma));
    }

    /// Flag the halos whose SO center lies inside the radius (of the first definition) of a more massive halo (on
    /// any task). Positions and radii in box units
    template <int NDIM>
    void flag_overlapping_so_halos(std::vector<HaloRecord<NDIM>> & halos) {
        // All halos (center, radius, mass) on all tasks
        std::vector<double> local;
        for (auto & h : halos) {
            for (int idim = 0; idim < NDIM; idim++)
                local.push_back(h.so_pos[idim]);
            local.push_back(h.so_radius.empty() ? 0.0 : h.so_radius[0]);
            local.push_back(h.so_mass.empty() ? 0.0 : h.so_mass[0]);
            local.push_back(double(h.id));
        }
        const int stride = NDIM + 3;
        std::vector<double> all = local;
#ifdef USE_MPI
        int n = int(local.size());
        std::vector<int> counts(FML::NTasks), displs(FML::NTasks, 0);
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
        for (int i = 1; i < FML::NTasks; i++)
            displs[i] = displs[i - 1] + counts[i - 1];
        all.resize(displs.back() + counts.back());
        MPI_Allgatherv(local.data(), n, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       FML::communicator);
#endif
        const size_t nall = all.size() / stride;
        double rmax = 0.0;
        for (size_t j = 0; j < nall; j++)
            rmax = std::max(rmax, all[j * stride + NDIM]);
        if (nall == 0 or rmax <= 0.0)
            return;
        // Cells of size >= rmax
        const int ncell = std::max(1, std::min(256, int(1.0 / rmax)));
        auto cell_of = [&](const double * x, int idim) { return std::min(int(x[idim] * ncell), ncell - 1); };
        auto key_of = [&](const std::array<int, NDIM> & c) {
            long long int k = 0;
            for (int idim = 0; idim < NDIM; idim++)
                k = k * ncell + ((c[idim] % ncell) + ncell) % ncell;
            return k;
        };
        std::map<long long int, std::vector<size_t>> cells;
        for (size_t j = 0; j < nall; j++) {
            std::array<int, NDIM> c;
            for (int idim = 0; idim < NDIM; idim++)
                c[idim] = cell_of(&all[j * stride], idim);
            cells[key_of(c)].push_back(j);
        }
        for (auto & h : halos) {
            const double m = h.so_mass.empty() ? 0.0 : h.so_mass[0];
            std::array<int, NDIM> c0;
            for (int idim = 0; idim < NDIM; idim++)
                c0[idim] = cell_of(h.so_pos.data(), idim);
            const int nr = std::min(1, ncell / 2);
            std::array<int, NDIM> off;
            off.fill(-nr);
            bool done = false;
            std::map<long long int, bool> visited;
            while (not done and not h.so_overlap) {
                std::array<int, NDIM> c;
                for (int idim = 0; idim < NDIM; idim++)
                    c[idim] = c0[idim] + off[idim];
                const auto key = key_of(c);
                if (not visited[key]) {
                    visited[key] = true;
                    auto it = cells.find(key);
                    if (it != cells.end())
                        for (auto j : it->second) {
                            const double * o = &all[j * stride];
                            const double mo = o[NDIM + 1];
                            // More massive (ties: the lower ID is the main halo)
                            if (not(mo > m or (mo == m and o[NDIM + 2] < double(h.id))))
                                continue;
                            double r2 = 0.0;
                            for (int idim = 0; idim < NDIM; idim++) {
                                double d = h.so_pos[idim] - o[idim];
                                d -= std::round(d);
                                r2 += d * d;
                            }
                            if (r2 < o[NDIM] * o[NDIM])
                                h.so_overlap = true;
                        }
                }
                int idim = 0;
                while (idim < NDIM) {
                    if (++off[idim] <= nr)
                        break;
                    off[idim] = -nr;
                    idim++;
                }
                done = idim == NDIM;
            }
        }
    }

    /// sigma(R) of a linear power-spectrum P(k) (k in h/Mpc, R in Mpc/h) with a top-hat filter
    inline double sigma_of_R(const std::function<double(double)> & pofk, double R, double kmin, double kmax) {
        const int n = 2000;
        const double dlogk = std::log(kmax / kmin) / n;
        double sum = 0.0;
        for (int i = 0; i <= n; i++) {
            const double k = kmin * std::exp(i * dlogk);
            const double x = k * R;
            const double w = x < 1e-3 ? 1.0 - x * x / 10.0 : 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
            const double integrand = k * k * k * pofk(k) * w * w / (2.0 * M_PI * M_PI);
            sum += integrand * ((i == 0 or i == n) ? 0.5 : 1.0);
        }
        return std::sqrt(sum * dlogk);
    }

    /// The multiplicity functions f(sigma) with dn/dlnM = f(sigma) rho / M |dln sigma / dln M|
    inline double f_tinker2008(double sigma, double z) {
        // Delta = 200 with respect to the mean density (Tinker et al. 2008, Table 2 and eqs. 5-8)
        const double Delta = 200.0;
        const double alpha = std::pow(10.0, -std::pow(0.75 / std::log10(Delta / 75.0), 1.2));
        const double A = 0.186 * std::pow(1.0 + z, -0.14);
        const double a = 1.47 * std::pow(1.0 + z, -0.06);
        const double b = 2.57 * std::pow(1.0 + z, -alpha);
        const double c = 1.19;
        return A * (std::pow(sigma / b, -a) + 1.0) * std::exp(-c / (sigma * sigma));
    }
    inline double f_watson2013_fof(double sigma) {
        // The universal FoF (b = 0.2) fit of Watson et al. 2013 (eq. 12)
        const double A = 0.282, alpha = 2.163, beta = 1.406, gamma = 1.210;
        return A * (std::pow(beta / sigma, alpha) + 1.0) * std::exp(-gamma / (sigma * sigma));
    }

} // namespace COLASOLVER

/// Locate FoF halos (and their spherical overdensity masses), write the halo catalogue and the mass functions
template <int NDIM, class T>
void compute_fof_halos(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {
    using Halo = COLASOLVER::HaloRecord<NDIM>;

    const std::string redshiftstring = redshift_to_string(redshift);
    const auto & settings = sim.get_parameters().fof;
    const double H0_hmpc = sim.get_gravity_model()->H0_hmpc;
    const double boxsize = sim.get_parameters().simulation.boxsize;
    const auto cosmo = sim.get_cosmology();
    const double OmegaCB = cosmo->get_Omegab() + cosmo->get_OmegaCDM();
    const double a = 1.0 / (1.0 + redshift);

    // The spherical overdensities in units of the mean CDM+baryon density
    const auto & so_definitions = settings.so_definitions;
    std::vector<double> so_thresholds;
    for (auto & def : so_definitions)
        so_thresholds.push_back(COLASOLVER::so_overdensity(def, *cosmo, a));
    const bool do_so = not so_thresholds.empty();
    if (do_so and FML::ThisTask == 0) {
        std::cout << "# Spherical overdensity masses around the FoF halos (Delta in units of the mean CDM+baryon "
                     "density):";
        for (size_t i = 0; i < so_definitions.size(); i++)
            std::cout << " " << so_definitions[i] << ": " << so_thresholds[i];
        std::cout << "\n";
    }

    //=============================================================
    // Halo finding with the CDM+baryon particles
    //=============================================================
    std::vector<Halo> halos;
    long long int npart_total = 0;
    if (not sim.has_baryons()) {
        auto & part = sim.get_particles();
        npart_total = (long long int)part.get_npart_total();
        halos = COLASOLVER::find_fof_halos<NDIM, T>(part.get_particles_ptr(),
                                                    part.get_npart(),
                                                    settings,
                                                    boxsize,
                                                    settings.linking_length,
                                                    settings.nmin_per_halo,
                                                    so_thresholds,
                                                    double(npart_total));
    } else {
        // CDM and baryons linked together. The particle masses are Omega_i / OmegaCB in units of the mean mass of all
        // the particles (times 1 + eps with the mass-encoded relative mode). The linking length is in units of the mean separation of one species and the minimum number
        // of particles is that of one species (so the halos are the same as with one species)
        using P = COLASOLVER::FoFParticle<NDIM>;
        std::vector<P> fofpart;
        long long int n_one_species = 0;
        sim.for_each_cb_species([&](auto & particles, const COLASOLVER::SpeciesInfo &) {
            npart_total += (long long int)particles.get_npart_total();
        });
        sim.for_each_cb_species([&](auto & particles, const COLASOLVER::SpeciesInfo & info) {
            n_one_species = (long long int)particles.get_npart_total();
            const double mass = info.weight_cb * double(npart_total) / double(particles.get_npart_total());
            for (auto & p : particles) {
                P q;
                const auto * pos = FML::PARTICLE::GetPos(p);
                const auto * vel = FML::PARTICLE::GetVel(p);
                for (int idim = 0; idim < NDIM; idim++) {
                    q.pos[idim] = pos[idim];
                    q.vel[idim] = vel[idim];
                }
                // Times the particle mass 1 + eps (particle_baryon_relative_mode = mass, mass_ini; 1 otherwise)
                q.mass = mass * FML::PARTICLE::GetMass(p);
                fofpart.push_back(q);
            }
        });
        const double linking_length =
            settings.linking_length * std::cbrt(double(npart_total) / double(n_one_species));
        const int nmin = int(std::lround(settings.nmin_per_halo * double(npart_total) / double(n_one_species)));
        halos = COLASOLVER::find_fof_halos<NDIM, P>(fofpart.data(),
                                                    fofpart.size(),
                                                    settings,
                                                    boxsize,
                                                    linking_length,
                                                    nmin,
                                                    so_thresholds,
                                                    double(npart_total));
    }
    if (do_so)
        COLASOLVER::flag_overlapping_so_halos<NDIM>(halos);

    // Convert to physical units
    // Code masses are in units of the mean mass -> Msun/h
    const double MplMpl_over_H0Msunh = 2.49264e21;
    const double mass_norm = COLASOLVER::halo_particle_mass_msunh(OmegaCB, boxsize, H0_hmpc, size_t(npart_total));
    // Code positions are in [0,1) -> Mpc/h
    const double pos_norm = boxsize;
    // Code velocities are a^2 dxdt / ((H0 Box) -> peculiar km/s
    const double vel_norm = 100.0 * boxsize * (1.0 + redshift);
    for (auto & g : halos) {
        g.mass *= mass_norm;
        for (int idim = 0; idim < NDIM; idim++) {
            g.pos[idim] *= pos_norm;
            g.vel[idim] *= vel_norm;
            g.vel_rms[idim] *= vel_norm;
            g.so_pos[idim] *= pos_norm;
            g.so_vel[idim] *= vel_norm;
            g.so_vel_rms[idim] *= vel_norm;
        }
        for (auto & m : g.so_mass)
            m *= mass_norm;
        for (auto & r : g.so_radius)
            r *= pos_norm;
    }

    //=============================================================
    // The mass functions and the fits
    //=============================================================
    const int nbins = settings.massfunction_nbins;
    const double massmin = settings.massfunction_Mmin;
    const double massmax = settings.massfunction_Mmax;
    const double dlogM = std::log(massmax / massmin) / double(nbins);
    std::vector<double> logM(nbins, 0.0);
    for (int i = 0; i < nbins; i++)
        logM[i] = std::log(massmin) + dlogM * (i + 0.5);

    // n(>M) and dn/dlnM of a set of masses
    auto mass_function = [&](const std::function<double(const Halo &)> & mass_of,
                             std::vector<double> & nofM,
                             std::vector<double> & dnofM) {
        nofM.assign(nbins, 0.0);
        dnofM.assign(nbins, 0.0);
        for (auto & g : halos) {
            int index = int(std::log(mass_of(g) / massmin) / dlogM);
            if (index >= 0 and index < nbins) {
                dnofM[index] += 1.0;
            }
        }
        FML::SumArrayOverTasks(dnofM.data(), nbins);
        nofM[nbins - 1] = dnofM[nbins - 1];
        for (int i = nbins - 2; i >= 0; i--) {
            nofM[i] = nofM[i + 1] + dnofM[i];
        }
        for (int i = 0; i < nbins; i++) {
            nofM[i] /= std::pow(boxsize, NDIM);
            dnofM[i] /= std::pow(boxsize, NDIM) * dlogM;
        }
    };

    // sigma(M) and dln sigma / dln M of the linear CDM+baryon P(k) at this redshift (if we have it): the fits are
    // dn/dlnM = f(sigma) rho / M |dln sigma / dln M|
    std::vector<double> sigma_of_M(nbins, 0.0), dlnsigma_dlnM(nbins, 0.0);
    const auto & power_initial_spline = sim.get_power_initial_spline();
    const auto grav = sim.get_gravity_model();
    const bool have_fits = power_initial_spline and NDIM == 3;
    // rho_cb in Msun/h / (Mpc/h)^3
    const double rho = 3.0 * OmegaCB * MplMpl_over_H0Msunh * std::pow(H0_hmpc, 3);
    if (have_fits) {
        const double zini = sim.get_parameters().ic.initial_redshift;
        auto pofk = [&](double k) {
            const double D = grav->get_D_1LPT(1.0 / (1.0 + redshift), k / grav->H0_hmpc);
            const double Dini = grav->get_D_1LPT(1.0 / (1.0 + zini), k / grav->H0_hmpc);
            return power_initial_spline(k) * std::pow(D / Dini, 2);
        };
        const auto krange = power_initial_spline.get_xrange();
        auto R_of_M = [&](double M) { return std::cbrt(3.0 * M / (4.0 * M_PI * rho)); };
        if (FML::ThisTask == 0)
            std::cout << "# Mass function fits: sigma(R = 8 Mpc/h, z = " << redshift
                      << ") of the linear CDM+baryon P(k) = "
                      << COLASOLVER::sigma_of_R(pofk, 8.0, krange.first, krange.second) << "\n";
        for (int i = 0; i < nbins; i++) {
            const double M = std::exp(logM[i]);
            const double eps = 0.01;
            sigma_of_M[i] = COLASOLVER::sigma_of_R(pofk, R_of_M(M), krange.first, krange.second);
            const double sp = COLASOLVER::sigma_of_R(pofk, R_of_M(M * std::exp(eps)), krange.first, krange.second);
            const double sm = COLASOLVER::sigma_of_R(pofk, R_of_M(M * std::exp(-eps)), krange.first, krange.second);
            dlnsigma_dlnM[i] = (std::log(sp) - std::log(sm)) / (2.0 * eps);
        }
    }
    auto fit = [&](int i, double f) { return f * rho / std::exp(logM[i]) * std::fabs(dlnsigma_dlnM[i]); };

    // The FoF mass function with the Tinker (Delta = 200 mean) and Watson (FoF) fits
    {
        std::vector<double> nofM, dnofM;
        mass_function([](const Halo & g) { return g.mass; }, nofM, dnofM);
        std::vector<double> dndlogM_tinker(nbins, 0.0), dndlogM_watson(nbins, 0.0);
        if (have_fits)
            for (int i = 0; i < nbins; i++) {
                dndlogM_tinker[i] = fit(i, COLASOLVER::f_tinker2008(sigma_of_M[i], redshift));
                dndlogM_watson[i] = fit(i, COLASOLVER::f_watson2013_fof(sigma_of_M[i]));
            }
        if (FML::ThisTask == 0) {
            std::string filename = snapshot_folder + "/massfunc_z" + redshiftstring + ".txt";
            std::ofstream fp(filename.c_str());
            if (not fp.is_open()) {
                std::cout << "Warning: Cannot write massfunction to file, failed to open [" << filename << "]\n";
            } else {
                fp << "#  M (Msun/h)         n(M) (h/Mpc)^3         dndlogM(M) (h/Mpc)^3   dndlogM Tinker08 "
                      "(Delta=200m)   dndlogM Watson13 (FoF)\n";
                for (int i = 0; i < nbins; i++) {
                    fp << std::setw(15) << std::exp(logM[i]) << " ";
                    fp << std::setw(15) << nofM[i] << " ";
                    fp << std::setw(15) << dnofM[i] << " ";
                    fp << std::setw(15) << dndlogM_tinker[i] << " ";
                    fp << std::setw(15) << dndlogM_watson[i] << " ";
                    fp << "\n";
                }
            }
        }
    }

    // The SO mass functions with the Tinker fit for the same Delta
    for (size_t id = 0; id < so_definitions.size(); id++) {
        std::vector<double> nofM, dnofM;
        mass_function([id](const Halo & g) { return g.so_mass[id]; }, nofM, dnofM);
        std::vector<double> dndlogM_tinker(nbins, 0.0);
        if (have_fits)
            for (int i = 0; i < nbins; i++)
                dndlogM_tinker[i] =
                    fit(i, COLASOLVER::f_tinker2008_delta(sigma_of_M[i], redshift, so_thresholds[id]));
        if (FML::ThisTask == 0) {
            std::string filename = snapshot_folder + "/massfunc_so" + so_definitions[id] + "_z" + redshiftstring + ".txt";
            std::ofstream fp(filename.c_str());
            fp << "# Spherical overdensity " << so_definitions[id] << " (Delta = " << so_thresholds[id]
               << " times the mean CDM+baryon density) around the FoF halos. The Tinker fit is only given for 200 <= "
                  "Delta <= 3200\n";
            fp << "#  M (Msun/h)         n(M) (h/Mpc)^3         dndlogM(M) (h/Mpc)^3   dndlogM Tinker08\n";
            for (int i = 0; i < nbins; i++) {
                fp << std::setw(15) << std::exp(logM[i]) << " ";
                fp << std::setw(15) << nofM[i] << " ";
                fp << std::setw(15) << dnofM[i] << " ";
                fp << std::setw(15) << dndlogM_tinker[i] << " ";
                fp << "\n";
            }
        }
    }

    //=============================================================
    // The halo catalogue
    //=============================================================
    const int nso = int(so_definitions.size());
    if (settings.output_fileformat == "HDF5") {
#ifdef USE_HDF5
        // Gather all the halos on task 0 and write one file
        std::vector<double> local;
        for (auto & g : halos) {
            local.push_back(double(g.id));
            local.push_back(double(g.np));
            local.push_back(g.mass);
            for (int idim = 0; idim < NDIM; idim++)
                local.push_back(g.pos[idim]);
            for (int idim = 0; idim < NDIM; idim++)
                local.push_back(g.vel[idim]);
            for (int idim = 0; idim < NDIM; idim++)
                local.push_back(g.vel_rms[idim]);
            if (do_so) {
                for (int idim = 0; idim < NDIM; idim++)
                    local.push_back(g.so_pos[idim]);
                for (int idim = 0; idim < NDIM; idim++)
                    local.push_back(g.so_vel[idim]);
                for (int idim = 0; idim < NDIM; idim++)
                    local.push_back(g.so_vel_rms[idim]);
                local.push_back(g.so_truncated ? 1.0 : 0.0);
                local.push_back(g.so_overlap ? 1.0 : 0.0);
                for (int id = 0; id < nso; id++) {
                    local.push_back(g.so_mass[id]);
                    local.push_back(g.so_radius[id]);
                    local.push_back(double(g.so_npart[id]));
                }
            }
        }
        const int stride = 3 + 3 * NDIM + (do_so ? 3 * NDIM + 2 + 3 * nso : 0);
        int n = int(local.size());
        std::vector<int> counts(FML::NTasks, n), displs(FML::NTasks, 0);
#ifdef USE_MPI
        MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, FML::communicator);
#endif
        for (int i = 1; i < FML::NTasks; i++)
            displs[i] = displs[i - 1] + counts[i - 1];
        std::vector<double> all(FML::ThisTask == 0 ? displs.back() + counts.back() : 0);
#ifdef USE_MPI
        MPI_Gatherv(local.data(), n, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
                    FML::communicator);
#else
        all = local;
#endif
        if (FML::ThisTask == 0) {
            namespace H5 = FML::FILEUTILS::GADGET::HDF5;
            const size_t nhalos = all.size() / stride;
            // Column c of all halos (ncols consecutive values)
            auto column = [&](int c, int ncols) {
                std::vector<double> v(nhalos * ncols);
                for (size_t i = 0; i < nhalos; i++)
                    for (int j = 0; j < ncols; j++)
                        v[i * ncols + j] = all[i * stride + c + j];
                return v;
            };
            auto to_ll = [](const std::vector<double> & v) {
                std::vector<long long int> r(v.size());
                for (size_t i = 0; i < v.size(); i++)
                    r[i] = (long long int)v[i];
                return r;
            };
            const std::string filename = snapshot_folder + "/halos_z" + redshiftstring + ".hdf5";
            H5::Handle file(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose, filename);
            H5::Handle header(H5Gcreate2(file, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "Header");
            const double attrs_d[] = {redshift, boxsize, settings.linking_length, mass_norm};
            const char * names_d[] = {"Redshift", "BoxSize", "LinkingLength", "MeanParticleMass"};
            for (int i = 0; i < 4; i++)
                H5::write_attribute(header, names_d[i], H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, &attrs_d[i], 1);
            const long long int nh = (long long int)nhalos, nmin = settings.nmin_per_halo;
            H5::write_attribute(header, "NumHalos", H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, &nh, 1);
            H5::write_attribute(header, "MinParticlesPerHalo", H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, &nmin, 1);
            auto write_d = [&](hid_t loc, const std::string & name, int c, int ncols) {
                auto v = column(c, ncols);
                H5::write_dataset(loc, name, H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, v.data(), nhalos, ncols);
            };
            auto write_ll = [&](hid_t loc, const std::string & name, int c) {
                auto v = to_ll(column(c, 1));
                H5::write_dataset(loc, name, H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, v.data(), nhalos, 1);
            };
            write_ll(file, "ID", 0);
            write_ll(file, "NumParticles", 1);
            write_d(file, "Mass", 2, 1);
            write_d(file, "Position", 3, NDIM);
            write_d(file, "Velocity", 3 + NDIM, NDIM);
            write_d(file, "VelocityDispersion", 3 + 2 * NDIM, NDIM);
            if (do_so) {
                const int c0 = 3 + 3 * NDIM;
                H5::Handle so(H5Gcreate2(file, "SO", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "SO");
                write_d(so, "Position", c0, NDIM);
                write_d(so, "Velocity", c0 + NDIM, NDIM);
                write_d(so, "VelocityDispersion", c0 + 2 * NDIM, NDIM);
                write_ll(so, "Truncated", c0 + 3 * NDIM);
                write_ll(so, "Overlap", c0 + 3 * NDIM + 1);
                for (int id = 0; id < nso; id++) {
                    const int c = c0 + 3 * NDIM + 2 + 3 * id;
                    H5::Handle g(H5Gcreate2(so, so_definitions[id].c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                                 H5Gclose,
                                 "SO/" + so_definitions[id]);
                    H5::write_attribute(g, "Delta", H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, &so_thresholds[id], 1);
                    write_d(g, "Mass", c, 1);
                    write_d(g, "Radius", c + 1, 1);
                    write_ll(g, "NumParticles", c + 2);
                }
            }
            std::cout << "Wrote " << nhalos << " halos to " << filename << "\n";
        }
        return;
#endif
    }

    // Text catalogue: append to the same file task by task
    for (int i = 0; i < FML::NTasks; i++) {
        if (i == FML::ThisTask) {
            std::cout << "Found " << halos.size() << " halos on task " << FML::ThisTask << "\n";
            std::string filename = snapshot_folder + "/halos_z" + redshiftstring + ".txt";
            std::ofstream fp(filename.c_str(), (i == 0 ? std::ios_base::out : std::ios_base::app));
            if (not fp.is_open()) {
                std::cout << "Warning: Cannot write halos to file, failed to open [" << filename << "]\n";
            } else {
                if (i == 0) {
                    fp << "#            np     ";
                    fp << "mass (Msun/h)       ";
                    fp << "pos[] (Mpc/h)                                   ";
                    fp << "vel[] (peculiar km/s)                           ";
                    fp << "vel_rms[]  (peculiar km/s)                      ";
                    if (do_so) {
                        fp << "SO: pos[] (Mpc/h)  vel[] (km/s)  vel_rms[] (km/s)  truncated  overlap";
                        for (auto & def : so_definitions)
                            fp << "  M_" << def << " (Msun/h)  R_" << def << " (Mpc/h)  N_" << def;
                    }
                    fp << "\n";
                }
                for (auto & g : halos) {
                    if (g.np > 0) {
                        fp << std::setw(15) << g.np << " ";
                        fp << std::setw(15) << g.mass << " ";
                        for (int idim = 0; idim < NDIM; idim++)
                            fp << std::setw(15) << g.pos[idim] << " ";
                        for (int idim = 0; idim < NDIM; idim++)
                            fp << std::setw(15) << g.vel[idim] << " ";
                        for (int idim = 0; idim < NDIM; idim++)
                            fp << std::setw(15) << g.vel_rms[idim] << " ";
                        if (do_so) {
                            for (int idim = 0; idim < NDIM; idim++)
                                fp << std::setw(15) << g.so_pos[idim] << " ";
                            for (int idim = 0; idim < NDIM; idim++)
                                fp << std::setw(15) << g.so_vel[idim] << " ";
                            for (int idim = 0; idim < NDIM; idim++)
                                fp << std::setw(15) << g.so_vel_rms[idim] << " ";
                            fp << std::setw(3) << int(g.so_truncated) << " " << std::setw(3) << int(g.so_overlap) << " ";
                            for (int id = 0; id < nso; id++)
                                fp << std::setw(15) << g.so_mass[id] << " " << std::setw(15) << g.so_radius[id] << " "
                                   << std::setw(10) << g.so_npart[id] << " ";
                        }
                        fp << "\n";
                    }
                }
            }
            fp.close();
        }
#ifdef USE_MPI
        MPI_Barrier(FML::communicator);
#endif
    }
}

#endif
