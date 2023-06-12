#ifndef ANALYZEANDOUTPUT_HEADER
#define ANALYZEANDOUTPUT_HEADER

#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/FriendsOfFriends/FoF.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

template <int NDIM, class T>
class NBodySimulation;

template <int NDIM, class T>
void output_pofk_for_every_step(NBodySimulation<NDIM, T> & sim) {

    //=============================================================
    // Fetch parameters
    //=============================================================
    const auto output_folder = sim.output_folder;
    const auto simulation_name = sim.simulation_name;
    const auto ic_initial_redshift = sim.ic_initial_redshift;
    const auto & pofk_cb_every_step = sim.pofk_cb_every_step;
    const auto & pofk_total_every_step = sim.pofk_total_every_step;
    const auto & grav = sim.grav;
    const auto & grav_ic = sim.grav_ic;
    const auto & transferdata = sim.transferdata;
    const auto & power_initial_spline = sim.power_initial_spline;
    const auto & ic_use_gravity_model_GR = sim.ic_use_gravity_model_GR;

    //=============================================================
    // Output all CMB+baryon Pofk
    //=============================================================
    for (auto & p : pofk_cb_every_step) {
        auto redshift = p.first;
        auto binning = p.second;
        auto pofk_cb = [&](double k) {
            double pofk_ini = power_initial_spline(k);
            double D = grav->get_D_1LPT(1.0 / (1.0 + redshift), k / grav->H0_hmpc);
            double Dini = grav->get_D_1LPT(1.0 / (1.0 + ic_initial_redshift), k / grav->H0_hmpc);
            return pofk_ini * std::pow(D / Dini, 2);
        };

        std::stringstream stream;
        stream << std::fixed << std::setprecision(3) << redshift;
        std::string redshiftstring = stream.str();
        std::string filename = output_folder;
        filename =
            filename + (filename == "" ? "" : "/") + "pofk_" + simulation_name + "_cb_z" + redshiftstring + ".txt";

        std::ofstream fp(filename.c_str());
        fp << "# k  (h/Mpc)    Pcb(k)  (Mpc/h)^3    Pcb_linear(k) (Mpc/h)^3   ShotnoiseSubtracted = " << std::boolalpha << sim.pofk_subtract_shotnoise << "\n";
        for (int i = 0; i < binning.n; i++) {
            fp << std::setw(15) << binning.kbin[i] << " ";
            fp << std::setw(15) << binning.pofk[i] << " ";
            fp << std::setw(15) << pofk_cb(binning.kbin[i]) << " ";
            fp << "\n";
        }
    }

    //=============================================================
    // Output all total Pofk
    //=============================================================
    for (auto & p : pofk_total_every_step) {
        auto redshift = p.first;
        auto binning = p.second;
        auto pofk_total = [&](double k) {
            double fac = 1.0;  
            // If ic_use_gravity_model_GR then transferdata is that of GR so
            // we need to rescale it with growthfactors
            if(ic_use_gravity_model_GR){ 
              double D = grav->get_D_1LPT(1.0 / (1.0 + redshift), k / grav->H0_hmpc);
              double Dini = grav->get_D_1LPT(1.0 / (1.0 + ic_initial_redshift), k / grav->H0_hmpc);
              double D_GR = grav_ic->get_D_1LPT(1.0 / (1.0 + redshift), k / grav->H0_hmpc);
              double Dini_GR = grav_ic->get_D_1LPT(1.0 / (1.0 + ic_initial_redshift), k / grav->H0_hmpc);
              fac = std::pow((D/Dini) / (D_GR/Dini_GR), 2);
            }
            if (transferdata)
                return fac * transferdata->get_total_power_spectrum(k, 1.0 / (1.0 + redshift));
            // If we don't have transfer data we don't have this info so just output 0.0
            return 0.0;
        };

        std::stringstream stream;
        stream << std::fixed << std::setprecision(3) << redshift;
        std::string redshiftstring = stream.str();
        std::string filename = output_folder;
        filename =
            filename + (filename == "" ? "" : "/") + "pofk_" + simulation_name + "_total_z" + redshiftstring + ".txt";

        std::ofstream fp(filename.c_str());
        fp << "# k  (h/Mpc)    P(k)  (Mpc/h)^3    P_linear(k)  (Mpc/h)^3  ShotnoiseSubtracted = " << std::boolalpha << sim.pofk_subtract_shotnoise << "\n";
        for (int i = 0; i < binning.n; i++) {
            fp << std::setw(15) << binning.kbin[i] << " ";
            fp << std::setw(15) << binning.pofk[i] << " ";
            fp << std::setw(15) << pofk_total(binning.kbin[i]) << " ";
            fp << "\n";
        }
    }
}

template <int NDIM, class T>
void output_fml(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << redshift;
    std::string redshiftstring = stream.str();

    // Output particles in internal format
    std::string fileprefix = snapshot_folder + "/" + "fml_z" + redshiftstring;
    auto & part = sim.part;
    part.dump_to_file(fileprefix);
}

template <int NDIM, class T>
void output_gadget(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << redshift;
    std::string redshiftstring = stream.str();

    //=============================================================
    // Fetch parameters
    //=============================================================
    const auto simulation_boxsize = sim.simulation_boxsize;
    const auto & cosmo = sim.cosmo;
    auto & part = sim.part;

    const double scale_factor = 1.0 / (1.0 + redshift);
    const int nfiles = FML::NTasks;
    const double pos_norm = simulation_boxsize;
    const double vel_norm = 100 * simulation_boxsize / std::pow(scale_factor, 1.5);
    const std::string fileprefix = snapshot_folder + "/" + "gadget_z" + redshiftstring;

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Output in gadget format\n";
        std::cout << "# fileprefix  : " << fileprefix << "\n";
        std::cout << "#=====================================================\n";
    }

    FML::FILEUTILS::GADGET::GadgetWriter gw;
    gw.write_gadget_single(fileprefix + "." + std::to_string(FML::ThisTask),
                           part.get_particles_ptr(),
                           part.get_npart(),
                           part.get_npart_total(),
                           nfiles,
                           scale_factor,
                           simulation_boxsize,
                           cosmo->get_OmegaM(),
                           cosmo->get_OmegaLambda(),
                           cosmo->get_h(),
                           pos_norm,
                           vel_norm);
}

template <int NDIM, class T>
void compute_bispectrum(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << redshift;
    std::string redshiftstring = stream.str();

    //=============================================================
    // Fetch parameters
    //=============================================================
    const double simulation_boxsize = sim.simulation_boxsize;
    const int bispectrum_nmesh = sim.bispectrum_nmesh;
    const int bispectrum_nbins = sim.bispectrum_nbins;
    const std::string bispectrum_density_assignment_method = sim.bispectrum_density_assignment_method;
    const bool bispectrum_interlacing = sim.bispectrum_interlacing;
    const bool bispectrum_subtract_shotnoise = sim.bispectrum_subtract_shotnoise;
    auto & part = sim.part;

    const double kmin = 0.0;
    const double kmax = 2.0 * M_PI * bispectrum_nmesh / 2;
    const auto bin_type = FML::CORRELATIONFUNCTIONS::BispectrumBinning<NDIM>::LINEAR_SPACING;
    FML::CORRELATIONFUNCTIONS::BispectrumBinning<NDIM> bofk(kmin, kmax, bispectrum_nbins, bin_type);
    bofk.subtract_shotnoise = bispectrum_subtract_shotnoise;

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Computing bispectrum of particles\n";
        std::cout << "# kmin (h/Mpc)                         : " << kmin / simulation_boxsize << "\n";
        std::cout << "# kmax (h/Mpc)                         : " << kmax / simulation_boxsize << "\n";
        std::cout << "# deltak (h/Mpc)                       : "
                  << (kmax - kmin) / simulation_boxsize / bispectrum_nbins << "\n";
        std::cout << "# bispectrum_nmesh                     : " << bispectrum_nmesh << "\n";
        std::cout << "# bispectrum_nbins                     : " << bispectrum_nbins << "\n";
        std::cout << "# bispectrum_density_assignment_method : " << bispectrum_density_assignment_method << "\n";
        std::cout << "# bispectrum_interlacing               : " << bispectrum_interlacing << "\n";
        std::cout << "# bispectrum_subtract_shotnoise        : " << bispectrum_subtract_shotnoise << "\n";
        std::cout << "#=====================================================\n";
    }

    FML::CORRELATIONFUNCTIONS::compute_bispectrum(bispectrum_nmesh,
                                                  part.get_particles_ptr(),
                                                  part.get_npart(),
                                                  part.get_npart_total(),
                                                  bofk,
                                                  bispectrum_density_assignment_method,
                                                  bispectrum_interlacing);
    bofk.scale(simulation_boxsize);

    // Output to file
    if (FML::ThisTask == 0) {
        std::string filename = snapshot_folder + "/bispectrum_z" + redshiftstring + ".txt";
        std::ofstream fp(filename.c_str());
        if (not fp.is_open()) {
            std::cout << "Warning: Cannot write bispectrum to file, failed to open [" << filename << "]\n";
        } else {
            fp << "#  i1    i2   i3       k1  (h/Mpc)       k2 (h/Mpc)       k3 (h/Mpc)       B(k1,k2,k3) "
                  "(Mpc/h)^6       Q(k1,k2,k3)       N(k1,k2,k3) \n";
            for (int i = 0; i < bispectrum_nbins; i++) {
                for (int j = 0; j < bispectrum_nbins; j++) {
                    for (int k = 0; k < bispectrum_nbins; k++) {
                        double k1 = bofk.kmean[i];
                        double k2 = bofk.kmean[j];
                        double k3 = bofk.kmean[k];
                        double B123 = bofk.get_spectrum(i, j, k);
                        double Q123 = bofk.get_reduced_spectrum(i, j, k);
                        double N123 = bofk.get_bincount(i, j, k);
                        fp << std::setw(3) << i << " ";
                        fp << std::setw(3) << j << " ";
                        fp << std::setw(3) << k << " ";
                        fp << std::setw(15) << k1 << " ";
                        fp << std::setw(15) << k2 << " ";
                        fp << std::setw(15) << k3 << " ";
                        fp << std::setw(15) << B123 << " ";
                        fp << std::setw(15) << Q123 << " ";
                        fp << std::setw(15) << N123 << "\n";
                    }
                }
            }
        }
    }
}

template <int NDIM, class T>
void compute_power_spectrum_multipoles(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << redshift;
    std::string redshiftstring = stream.str();

    //=============================================================
    // Fetch parameters
    //=============================================================
    const double simulation_boxsize = sim.simulation_boxsize;
    const int pofk_multipole_nmesh = sim.pofk_multipole_nmesh;
    const std::string pofk_multipole_density_assignment_method = sim.pofk_multipole_density_assignment_method;
    const bool pofk_multipole_interlacing = sim.pofk_multipole_interlacing;
    const bool pofk_multipole_subtract_shotnoise = sim.pofk_multipole_subtract_shotnoise;
    const int pofk_multipole_ellmax = sim.pofk_multipole_ellmax;
    const auto & transferdata = sim.transferdata;
    const auto & power_initial_spline = sim.power_initial_spline;
    const auto & grav = sim.grav;
    const auto & cosmo = sim.cosmo;
    auto & part = sim.part;

    const double a = 1.0 / (1.0 + redshift);
    const double velocity_to_displacement = 1.0 / (a * a * cosmo->HoverH0_of_a(a));

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Computing power-spectrum multipoles of particles\n";
        std::cout << "# pofk_multipole_nmesh                     : " << pofk_multipole_nmesh << "\n";
        std::cout << "# pofk_multipole_density_assignment_method : " << pofk_multipole_density_assignment_method
                  << "\n";
        std::cout << "# pofk_multipole_interlacing               : " << pofk_multipole_interlacing << "\n";
        std::cout << "# pofk_multipole_subtract_shotnoise        : " << pofk_multipole_subtract_shotnoise << "\n";
        std::cout << "#=====================================================\n";
    }

    std::vector<FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>> Pells(pofk_multipole_ellmax + 1,
                                                                             pofk_multipole_nmesh / 2);
    Pells[0].subtract_shotnoise = pofk_multipole_subtract_shotnoise;
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum_multipoles<NDIM, T>(pofk_multipole_nmesh,
                                                                          part,
                                                                          velocity_to_displacement,
                                                                          Pells,
                                                                          pofk_multipole_density_assignment_method,
                                                                          pofk_multipole_interlacing);
    for (auto & pl : Pells)
        pl.scale(simulation_boxsize);

    //=============================================================
    // Compute linear predictions
    //=============================================================
    auto pofk_cb = [&](double k) {
        if (power_initial_spline) {
            double pofk_ini = sim.power_initial_spline(k);
            double D = grav->get_D_1LPT(1.0 / (1.0 + redshift), k / grav->H0_hmpc);
            double Dini = grav->get_D_1LPT(1.0 / (1.0 + sim.ic_initial_redshift), k / grav->H0_hmpc);
            return pofk_ini * std::pow(D / Dini, 2);
        } else if (transferdata) {
            return transferdata->get_cdm_baryon_power_spectrum(k, 1.0 / (1.0 + redshift));
        }
        return 0.0;
    };
    auto kvals = Pells[0].kbin;
    auto pofk_cb_linear = kvals;
    for (auto & v : pofk_cb_linear) {
        v = pofk_cb(v);
    }
    FML::INTERPOLATION::SPLINE::Spline pofk_cb_linear_spline(kvals, pofk_cb_linear, "Pcb(k) linear spline");

    // Output to file
    if (FML::ThisTask == 0) {
        std::string filename = snapshot_folder + "/pofk_multipole_z" + redshiftstring + ".txt";
        std::ofstream fp(filename.c_str());
        if (not fp.is_open()) {
            std::cout << "Warning: Cannot write power-spectrum multipoles to file, failed to open [" << filename
                      << "]\n";
        } else {
            fp << "#  k  (h/Mpc)          P0(k)  (Mpc/h)^3          P2(k)  (Mpc/h)^3       P4(k)  (Mpc/h)^3   ...   "
                  "P0_kaiser    P2_kaiser    P4_kaiser  ShotnoiseSubtracted = " << std::boolalpha << sim.pofk_multipole_subtract_shotnoise << "\n";
            for (int i = 0; i < Pells[0].n; i++) {
                const double k = Pells[0].kbin[i];
                const double f = grav->get_f_1LPT(1.0 / (1.0 + redshift), k / grav->H0_hmpc);
                fp << std::setw(15) << k << " ";
                for (size_t ell = 0; ell < Pells.size(); ell += 2)
                    fp << std::setw(15) << Pells[ell].pofk[i] << " ";

                // Kaiser approximation
                fp << std::setw(15) << pofk_cb_linear_spline(k) * (1.0 + 2.0 / 3.0 * f + f * f / 5.0) << " ";
                fp << std::setw(15) << pofk_cb_linear_spline(k) * (4.0 / 3.0 * f + 4.0 / 7.0 * f * f) << " ";
                fp << std::setw(15) << pofk_cb_linear_spline(k) * (8.0 / 35.0 * f * f) << " ";
                fp << "\n";
            }
        }
    }

    // Store the data in the class
    sim.pofk_multipoles_every_output.push_back({redshift, Pells});
}

template <int NDIM, class T>
void compute_power_spectrum(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << redshift;
    std::string redshiftstring = stream.str();

    //=============================================================
    // Fetch parameters
    //=============================================================
    const double simulation_boxsize = sim.simulation_boxsize;
    const int pofk_nmesh = sim.pofk_nmesh;
    const std::string pofk_density_assignment_method = sim.pofk_density_assignment_method;
    const bool pofk_interlacing = sim.pofk_interlacing;
    const bool pofk_subtract_shotnoise = sim.pofk_subtract_shotnoise;
    const auto & transferdata = sim.transferdata;
    const auto & power_initial_spline = sim.power_initial_spline;
    const auto & grav = sim.grav;
    auto & part = sim.part;

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Computing power-spectrum of particles\n";
        std::cout << "# pofk_nmesh                     : " << pofk_nmesh << "\n";
        std::cout << "# pofk_density_assignment_method : " << pofk_density_assignment_method << "\n";
        std::cout << "# pofk_interlacing               : " << pofk_interlacing << "\n";
        std::cout << "# pofk_subtract_shotnoise        : " << pofk_subtract_shotnoise << "\n";
        std::cout << "#=====================================================\n";
    }

    //=============================================================
    // Compute baryon+CDM power-spectrum
    //=============================================================
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> pofk_cb_binning(pofk_nmesh / 2);
    pofk_cb_binning.subtract_shotnoise = pofk_subtract_shotnoise;
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum<NDIM, T>(pofk_nmesh,
                                                               part.get_particles_ptr(),
                                                               part.get_npart(),
                                                               part.get_npart_total(),
                                                               pofk_cb_binning,
                                                               pofk_density_assignment_method,
                                                               pofk_interlacing);
    pofk_cb_binning.scale(simulation_boxsize);

    /* ...or do it this way for which we can compute the total power-spectrum by adding on the neutrinos
    const auto nleftright =
    FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(pofk_density_assignment_method); const int nleft
    = nleftright.first; const int nright = nleftright.second + (pofk_interlacing ? 1 : 0); FML::GRID::FFTWGrid<NDIM>
    density_grid_fourier(pofk_nmesh, nleft, nright);
    FML::INTERPOLATION::particles_to_fourier_grid(part.get_particles_ptr(),
                                                  part.get_npart(),
                                                  part.get_npart_total(),
                                                  density_grid_fourier,
                                                  pofk_density_assignment_method,
                                                  pofk_interlacing);
    FML::INTERPOLATION::deconvolve_window_function_fourier<NDIM>(density_grid_fourier, pofk_density_assignment_method);
    FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum(density_grid_fourier, pofk_cb_binning);
    pofk_cb_binning.scale(simulation_boxsize);
    */

    //=============================================================
    // Compute linear predictions
    //=============================================================
    auto pofk_cb = [&](double k) {
        if (power_initial_spline) {
            double pofk_ini = power_initial_spline(k);
            double D = grav->get_D_1LPT(1.0 / (1.0 + redshift), k / grav->H0_hmpc);
            double Dini = grav->get_D_1LPT(1.0 / (1.0 + sim.ic_initial_redshift), k / grav->H0_hmpc);
            return pofk_ini * std::pow(D / Dini, 2);
        } else if (transferdata) {
            return transferdata->get_cdm_baryon_power_spectrum(k, 1.0 / (1.0 + redshift));
        }
        return 0.0;
    };
    auto kvals = pofk_cb_binning.kbin;
    auto pofk_cb_linear = kvals;
    for (auto & v : pofk_cb_linear) {
        v = pofk_cb(v);
    }
    FML::INTERPOLATION::SPLINE::Spline pofk_cb_linear_spline(kvals, pofk_cb_linear, "Pcb(k) linear spline");

    // Output to file
    if (FML::ThisTask == 0) {
        std::string filename = snapshot_folder + "/pofk_z" + redshiftstring + ".txt";
        std::ofstream fp(filename.c_str());
        if (not fp.is_open()) {
            std::cout << "Warning: Cannot write power-spectrum to file, failed to open [" << filename << "]\n";
        } else {
            fp << "#  k  (h/Mpc)          P(k)  (Mpc/h)^3     P_linear(k) (Mpc/h)^3\n";
            for (int i = 0; i < pofk_cb_binning.n; i++) {
                const double k = pofk_cb_binning.kbin[i];
                fp << std::setw(15) << k << " ";
                fp << std::setw(15) << pofk_cb_binning.pofk[i] << " ";
                fp << std::setw(15) << pofk_cb_linear_spline(k) << " ";
                fp << "\n";
            }
        }
    }

    // Store the data in the class
    sim.pofk_every_output.push_back({redshift, pofk_cb_binning});
}

template <int NDIM, class T>
void compute_fof_halos(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << redshift;
    std::string redshiftstring = stream.str();

    //=============================================================
    // Fetch parameters
    //=============================================================
    const double H0_hmpc = sim.grav->H0_hmpc;
    const double simulation_boxsize = sim.simulation_boxsize;
    const double fof_linking_length = sim.fof_linking_length;
    const int fof_nmin_per_halo = sim.fof_nmin_per_halo;
    const int fof_nmesh_max = sim.fof_nmesh_max;
    const double fof_buffer_length_mpch = sim.fof_buffer_length_mpch;
    const auto & cosmo = sim.cosmo;
    auto & part = sim.part;

    //=============================================================
    // Halo finding
    //=============================================================
    using FoFHalo = FML::FOF::FoFHalo<T, NDIM>;
    const bool periodic_box = true;
    std::vector<FoFHalo> FoFGroups;
    FML::FOF::FriendsOfFriends<T, NDIM>(part.get_particles_ptr(),
                                        part.get_npart(),
                                        fof_linking_length,
                                        fof_nmin_per_halo,
                                        periodic_box,
                                        fof_buffer_length_mpch / simulation_boxsize,
                                        FoFGroups,
                                        fof_nmesh_max);

    // Convert to physical units
    // Code masses are in units of the mean mass -> Msun/h
    const double MplMpl_over_H0Msunh = 2.49264e21;
    const double mass_norm = 3.0 * cosmo->get_OmegaM() * MplMpl_over_H0Msunh *
                             std::pow(simulation_boxsize * H0_hmpc, 3) / double(part.get_npart_total());
    // Code positions are in [0,1) -> Mpc/h
    const double pos_norm = simulation_boxsize;
    // Code velocities are a^2 dxdt / ((H0 Box) -> peculiar km/s
    const double vel_norm = 100.0 * simulation_boxsize * (1.0 + redshift);
    for (auto & g : FoFGroups) {
        g.mass *= mass_norm;
        for (int idim = 0; idim < NDIM; idim++) {
            g.pos[idim] *= pos_norm;
            g.vel[idim] *= vel_norm;
            g.vel_rms[idim] *= vel_norm;
        }
    }

    // Compute mass-function
    const int nbins = 30;
    const double massmin = 1e10;
    const double massmax = 1e16;
    std::vector<double> dnofM(nbins,0.0), nofM(nbins,0.0), logM(nbins,0.0);
    const double dlogM = std::log(massmax / massmin) / double(nbins);
    for (auto & g : FoFGroups) {
        int index = int(std::log(g.mass / massmin) / dlogM);
        if (index >= 0 and index < nbins) {
            dnofM[index] += 1.0;
        }
    }
    
    // Sum over tasks
    FML::SumArrayOverTasks(dnofM.data(), nbins);

    // Integrate up to get n(M)
    nofM[nbins - 1] = dnofM[nbins - 1];
    for (int i = nbins - 2; i >= 0; i--) {
        nofM[i] = nofM[i + 1] + dnofM[i];
    }

    // Normalize
    for (int i = 0; i < nbins; i++) {
        logM[i] = std::log(massmin) + dlogM * (i + 0.5);
        nofM[i] /= std::pow(simulation_boxsize, NDIM);
        dnofM[i] /= std::pow(simulation_boxsize, NDIM) * dlogM;
    }

    // Output mass-function
    if (FML::ThisTask == 0) {
        std::string filename = snapshot_folder + "/massfunc_z" + redshiftstring + ".txt";
        std::ofstream fp(filename.c_str());
        if (not fp.is_open()) {
            std::cout << "Warning: Cannot write massfunction to file, failed to open [" << filename << "]\n";
        } else {
            fp << "#  M (Msun/h)         n(M) (h/Mpc)^3         dndlogM(M) (h/Mpc)^3\n";
            for (int i = 0; i < nbins; i++) {
                fp << std::setw(15) << std::exp(logM[i]) << " ";
                fp << std::setw(15) << nofM[i] << " ";
                fp << std::setw(15) << dnofM[i] << " ";
                fp << "\n";
            }
        }
    }

    // Output to file
    for(int i = 0; i < FML::NTasks; i++){
      if(i == FML::ThisTask){
        std::cout << "Found " << FoFGroups.size() << " halos on task " << FML::ThisTask << "\n";
        std::string filename = snapshot_folder + "/halos_z" + redshiftstring + ".txt";
        std::ofstream fp(filename.c_str(), (i == 0 ? std::ios_base::out : std::ios_base::app));
        if (not fp.is_open()) {
          std::cout << "Warning: Cannot write halos to file, failed to open [" << filename << "]\n";
          break;
        } else {
          if(i == 0){
            fp << "#            np     ";
            fp << "mass (Msun/h)       ";
            fp << "pos[] (Mpc/h)                                   ";
            fp << "vel[] (peculiar km/s)                           ";
            fp << "vel_rms[]  (peculiar km/s)                      ";
            fp << "\n";
          }
          for (auto & g : FoFGroups) {
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
      MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
}

#endif
