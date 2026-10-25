#ifndef HALOFINDING_HEADER
#define HALOFINDING_HEADER

#include <FML/FriendsOfFriends/FoF.h>
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
#include <string>
#include <vector>

//=============================================================================
// Friends-of-friends halos at the outputs (fof = true): a halo catalogue (text or HDF5) and the mass function
// with the Tinker et al. 2008 (Delta = 200 mean) and Watson et al. 2013 (FoF) fits for comparison.
//
// The halos are made of the CDM+baryon particles (the neutrino particles, if any, are not included). With CDM and
// baryons as separate particles both are linked together (particles with masses Omega_i / OmegaCB) and the linking
// length is in units of the mean separation of the particles of one species. Masses are in Msun/h from OmegaCB.
//=============================================================================

template <int NDIM, class T>
class NBodySimulation;

namespace COLASOLVER {

    /// A halo as written to the catalogue (code units: positions in [0,1), velocities in code units, mass in units
    /// of the mass of a CDM+baryon particle of one species)
    template <int NDIM>
    struct HaloRecord {
        long long int id;
        long long int np;
        double mass;
        std::array<double, NDIM> pos, vel, vel_rms;
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
                                                int nmin_per_halo) {
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
        return halos;
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

/// Locate FoF halos, write the halo catalogue and the mass function
template <int NDIM, class T>
void compute_fof_halos(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {
    using Halo = COLASOLVER::HaloRecord<NDIM>;

    const std::string redshiftstring = redshift_to_string(redshift);
    const auto & settings = sim.get_parameters().fof;
    const double H0_hmpc = sim.get_gravity_model()->H0_hmpc;
    const double boxsize = sim.get_parameters().simulation.boxsize;
    const auto cosmo = sim.get_cosmology();
    const double OmegaCB = cosmo->get_Omegab() + cosmo->get_OmegaCDM();

    //=============================================================
    // Halo finding with the CDM+baryon particles
    //=============================================================
    std::vector<Halo> halos;
    long long int npart_total = 0;
    if (not sim.has_baryons()) {
        auto & part = sim.get_particles();
        npart_total = (long long int)part.get_npart_total();
        halos = COLASOLVER::find_fof_halos<NDIM, T>(
            part.get_particles_ptr(), part.get_npart(), settings, boxsize, settings.linking_length, settings.nmin_per_halo);
    } else {
        // CDM and baryons linked together. The particle masses are Omega_i / OmegaCB in units of the mean mass of all
        // the particles. The linking length is in units of the mean separation of one species and the minimum number
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
                q.mass = mass;
                fofpart.push_back(q);
            }
        });
        const double linking_length =
            settings.linking_length * std::cbrt(double(npart_total) / double(n_one_species));
        const int nmin = int(std::lround(settings.nmin_per_halo * double(npart_total) / double(n_one_species)));
        halos = COLASOLVER::find_fof_halos<NDIM, P>(
            fofpart.data(), fofpart.size(), settings, boxsize, linking_length, nmin);
    }

    // Convert to physical units
    // Code masses are in units of the mean mass -> Msun/h
    const double MplMpl_over_H0Msunh = 2.49264e21;
    const double mass_norm = 3.0 * OmegaCB * MplMpl_over_H0Msunh * std::pow(boxsize * H0_hmpc, 3) / double(npart_total);
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
        }
    }

    //=============================================================
    // The mass function and the fits
    //=============================================================
    const int nbins = settings.massfunction_nbins;
    const double massmin = settings.massfunction_Mmin;
    const double massmax = settings.massfunction_Mmax;
    std::vector<double> dnofM(nbins, 0.0), nofM(nbins, 0.0), logM(nbins, 0.0);
    const double dlogM = std::log(massmax / massmin) / double(nbins);
    for (auto & g : halos) {
        int index = int(std::log(g.mass / massmin) / dlogM);
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
        logM[i] = std::log(massmin) + dlogM * (i + 0.5);
        nofM[i] /= std::pow(boxsize, NDIM);
        dnofM[i] /= std::pow(boxsize, NDIM) * dlogM;
    }

    // dn/dlnM from the fits with the linear CDM+baryon P(k) at this redshift (zero if we do not have it)
    std::vector<double> dndlogM_tinker(nbins, 0.0), dndlogM_watson(nbins, 0.0);
    const auto & power_initial_spline = sim.get_power_initial_spline();
    const auto grav = sim.get_gravity_model();
    if (power_initial_spline and NDIM == 3) {
        const double zini = sim.get_parameters().ic.initial_redshift;
        auto pofk = [&](double k) {
            const double D = grav->get_D_1LPT(1.0 / (1.0 + redshift), k / grav->H0_hmpc);
            const double Dini = grav->get_D_1LPT(1.0 / (1.0 + zini), k / grav->H0_hmpc);
            return power_initial_spline(k) * std::pow(D / Dini, 2);
        };
        const auto krange = power_initial_spline.get_xrange();
        // rho_cb in Msun/h / (Mpc/h)^3
        const double rho = 3.0 * OmegaCB * MplMpl_over_H0Msunh * std::pow(H0_hmpc, 3);
        auto R_of_M = [&](double M) { return std::cbrt(3.0 * M / (4.0 * M_PI * rho)); };
        if (FML::ThisTask == 0)
            std::cout << "# Mass function fits: sigma(R = 8 Mpc/h, z = " << redshift
                      << ") of the linear CDM+baryon P(k) = "
                      << COLASOLVER::sigma_of_R(pofk, 8.0, krange.first, krange.second) << "\n";
        for (int i = 0; i < nbins; i++) {
            const double M = std::exp(logM[i]);
            const double eps = 0.01;
            const double sigma = COLASOLVER::sigma_of_R(pofk, R_of_M(M), krange.first, krange.second);
            const double sp = COLASOLVER::sigma_of_R(pofk, R_of_M(M * std::exp(eps)), krange.first, krange.second);
            const double sm = COLASOLVER::sigma_of_R(pofk, R_of_M(M * std::exp(-eps)), krange.first, krange.second);
            const double dlnsigma_dlnM = (std::log(sp) - std::log(sm)) / (2.0 * eps);
            dndlogM_tinker[i] = COLASOLVER::f_tinker2008(sigma, redshift) * rho / M * std::fabs(dlnsigma_dlnM);
            dndlogM_watson[i] = COLASOLVER::f_watson2013_fof(sigma) * rho / M * std::fabs(dlnsigma_dlnM);
        }
    }

    if (FML::ThisTask == 0) {
        std::string filename = snapshot_folder + "/massfunc_z" + redshiftstring + ".txt";
        std::ofstream fp(filename.c_str());
        if (not fp.is_open()) {
            std::cout << "Warning: Cannot write massfunction to file, failed to open [" << filename << "]\n";
        } else {
            fp << "#  M (Msun/h)         n(M) (h/Mpc)^3         dndlogM(M) (h/Mpc)^3   dndlogM Tinker08 (Delta=200m)   "
                  "dndlogM Watson13 (FoF)\n";
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

    //=============================================================
    // The halo catalogue
    //=============================================================
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
        }
        const int stride = 3 + 3 * NDIM;
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
            std::vector<long long int> id(nhalos), np(nhalos);
            std::vector<double> mass(nhalos), pos(nhalos * NDIM), vel(nhalos * NDIM), vel_rms(nhalos * NDIM);
            for (size_t i = 0; i < nhalos; i++) {
                const double * h = &all[i * stride];
                id[i] = (long long int)h[0];
                np[i] = (long long int)h[1];
                mass[i] = h[2];
                for (int idim = 0; idim < NDIM; idim++) {
                    pos[i * NDIM + idim] = h[3 + idim];
                    vel[i * NDIM + idim] = h[3 + NDIM + idim];
                    vel_rms[i * NDIM + idim] = h[3 + 2 * NDIM + idim];
                }
            }
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
            H5::write_dataset(file, "ID", H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, id.data(), nhalos, 1);
            H5::write_dataset(file, "NumParticles", H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, np.data(), nhalos, 1);
            H5::write_dataset(file, "Mass", H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, mass.data(), nhalos, 1);
            H5::write_dataset(file, "Position", H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, pos.data(), nhalos, NDIM);
            H5::write_dataset(file, "Velocity", H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, vel.data(), nhalos, NDIM);
            H5::write_dataset(
                file, "VelocityDispersion", H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, vel_rms.data(), nhalos, NDIM);
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
