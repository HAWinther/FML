#ifndef ANALYZEANDOUTPUT_HEADER
#define ANALYZEANDOUTPUT_HEADER

#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/ComputePowerSpectra/HybridBiasSpectra.h>
#include <FML/ComputePowerSpectra/HybridBiasOneLoop.h>
#include <FML/ComputePowerSpectra/HybridBiasOneLoopMG.h>
#include <FML/ComputePowerSpectra/HybridBiasZeldovich.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/FriendsOfFriends/FoF.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/GadgetUtils/GadgetUtilsHDF5.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <FML/Spline/Spline.h>

#include "GravityModel_DGP.h"
#include "GravityModel_fofr.h"
#include "Species.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

//=============================================================================
// On the fly analysis and output of the simulation. These functions only use
// the public interface of NBodySimulation (get_parameters(), get_particles(), ...)
//=============================================================================

template <int NDIM, class T>
class NBodySimulation;

/// The redshift formatted as used in filenames
inline std::string redshift_to_string(double redshift) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << redshift;
    return stream.str();
}

template <int NDIM, class T>
void output_pofk_for_every_step(const NBodySimulation<NDIM, T> & sim) {

    // Only task 0 outputs
    if(FML::ThisTask > 0) 
      return;

    //=============================================================
    // Fetch parameters
    //=============================================================
    const auto & params = sim.get_parameters();
    const auto ic_initial_redshift = params.ic.initial_redshift;
    const auto & pofk_cb_every_step = sim.get_pofk_cb_every_step();
    const auto & pofk_total_every_step = sim.get_pofk_total_every_step();
    const auto grav = sim.get_gravity_model();
    const auto & cosmo = grav->get_cosmo();
    const auto grav_ic = sim.get_gravity_model_ic();
    const auto transferdata = sim.get_transferdata();
    const auto & power_initial_spline = sim.get_power_initial_spline();
    const auto ic_use_gravity_model_GR = params.ic.use_gravity_model_GR;
    const auto pofk_subtract_shotnoise = params.pofk.subtract_shotnoise;

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
    
        // Get non-linear boost (if this is implemented for the model)
        // Currently we only have EuclidEmulator2 predictions for LCDM and w0waCDM
        FML::INTERPOLATION::SPLINE::Spline non_linear_boost_of_k =
            cosmo->get_nonlinear_matter_power_spectrum_boost(redshift);

        std::string filename =
            params.output.path("pofk_" + params.simulation.name + "_cb_z" + redshift_to_string(redshift) + ".txt");

        std::ofstream fp(filename.c_str());
        fp << "# k  (h/Mpc)    Pcb(k)  (Mpc/h)^3    Pcb_linear(k) (Mpc/h)^3    Pcb_NL(k) (Mpc/h)^3   ShotnoiseSubtracted = " << std::boolalpha << pofk_subtract_shotnoise << "\n";
        for (int i = 0; i < binning.n; i++) {
            double boost = non_linear_boost_of_k ? non_linear_boost_of_k(binning.kbin[i]) : 1.0;
            fp << std::setw(15) << binning.kbin[i] << " ";
            fp << std::setw(15) << binning.pofk[i] << " ";
            fp << std::setw(15) << pofk_cb(binning.kbin[i]) << " ";
            fp << std::setw(15) << pofk_cb(binning.kbin[i]) * boost << " ";
            fp << "\n";
        }
    }

    //=============================================================
    // The species when there are several (cdm, baryon, nu): P(k) of each, with linear theory from the transfer
    // functions P_i(k,z) = P_cb(k,z) (T_i(k,z) / T_cb(k,z))^2
    //=============================================================
    for (auto & [label, list] : sim.get_pofk_species_every_step()) {
        for (auto & p : list) {
            const double redshift = p.first;
            const auto & binning = p.second;
            const double a = 1.0 / (1.0 + redshift);
            std::string filename =
                params.output.path("pofk_" + params.simulation.name + "_" + label + "_z" + redshift_to_string(redshift) + ".txt");
            std::ofstream fp(filename.c_str());
            fp << "# k  (h/Mpc)    P_" << label << "(k)  (Mpc/h)^3    P_" << label
               << "_linear(k) (Mpc/h)^3   ShotnoiseSubtracted = " << std::boolalpha << pofk_subtract_shotnoise << "\n";
            for (int i = 0; i < binning.n; i++) {
                const double k = binning.kbin[i];
                double plin = 0.0;
                if (transferdata) {
                    const double Ti = COLASOLVER::species_transfer_function(label, *transferdata, k, a);
                    const double D = grav->get_D_1LPT(a, k / grav->H0_hmpc);
                    const double Dini = grav->get_D_1LPT(1.0 / (1.0 + ic_initial_redshift), k / grav->H0_hmpc);
                    plin = power_initial_spline(k) * std::pow(D / Dini, 2) *
                           std::pow(Ti / transferdata->get_cdm_baryon_transfer_function(k, a), 2);
                }
                fp << std::setw(15) << k << " " << std::setw(15) << binning.pofk[i] << " " << std::setw(15) << plin
                   << "\n";
            }
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

        std::string filename =
            params.output.path("pofk_" + params.simulation.name + "_total_z" + redshift_to_string(redshift) + ".txt");

        // Get non-linear boost (if this is implemented for the model)
        // Currently we only have EuclidEmulator2 predictions for LCDM and w0waCDM
        FML::INTERPOLATION::SPLINE::Spline non_linear_boost_of_k =
            cosmo->get_nonlinear_matter_power_spectrum_boost(redshift);

        std::ofstream fp(filename.c_str());
        fp << "# k  (h/Mpc)    P(k)  (Mpc/h)^3    P_linear(k)  (Mpc/h)^3     P_NL(k)  ShotnoiseSubtracted = " << std::boolalpha << pofk_subtract_shotnoise << "\n";
        for (int i = 0; i < binning.n; i++) {
            double boost = non_linear_boost_of_k ? non_linear_boost_of_k(binning.kbin[i]) : 1.0;
            fp << std::setw(15) << binning.kbin[i] << " ";
            fp << std::setw(15) << binning.pofk[i] << " ";
            fp << std::setw(15) << pofk_total(binning.kbin[i]) << " ";
            fp << std::setw(15) << pofk_total(binning.kbin[i]) * boost << " ";
            fp << "\n";
        }
    }
}

/// The particle species for the GADGET writers (the types and masses are set in Species.h): CDM (or CDM+baryons)
/// as type 1, baryons as type 2, neutrinos as type 3 (with their delta-f weights as an extra field in HDF5). Particles
/// with a mass (particle_baryon_relative_mode = mass, mass_ini) have their masses written (MASS block, Masses)
template <int NDIM, class T>
std::vector<FML::FILEUTILS::GADGET::GadgetSpecies> gadget_species(NBodySimulation<NDIM, T> & sim) {
    std::vector<FML::FILEUTILS::GADGET::GadgetSpecies> species;
    sim.for_each_species([&](auto & particles, const COLASOLVER::SpeciesInfo & info) {
        using P = std::remove_reference_t<decltype(*particles.get_particles_ptr())>;
        auto * p = particles.get_particles_ptr();
        auto sp = FML::FILEUTILS::GADGET::make_gadget_species(
            info.gadget_type, p, particles.get_npart(), particles.get_npart_total(), info.mass_fraction);
        if constexpr (FML::PARTICLE::has_get_deltaf_weight<P>())
            sp.extra_scalars.push_back({"Weights", [p](size_t i) { return double(p[i].get_deltaf_weight()); }});
        // The mass-encoded baryon-CDM relative mode: per-particle masses (MassTable 0 for these types)
        if constexpr (FML::PARTICLE::has_get_mass<P>())
            sp.get_mass = [p](size_t i) { return double(FML::PARTICLE::GetMass(p[i])); };
        species.push_back(sp);
    });
    return species;
}

template <int NDIM, class T>
void output_gadget(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {

    const std::string redshiftstring = redshift_to_string(redshift);
    const auto simulation_boxsize = sim.get_parameters().simulation.boxsize;
    const auto cosmo = sim.get_cosmology();
    auto & part = sim.get_particles();

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
    if (sim.get_species().size() > 1) {
        // CDM (or CDM+baryons) as type 1, baryons as type 2 and neutrinos as type 3 (collisionless types: type 0 is
        // SPH gas in GADGET)
        const double OmegaM = cosmo->get_OmegaM();
        gw.write_gadget_species(fileprefix + "." + std::to_string(FML::ThisTask),
                                gadget_species(sim),
                                nfiles,
                                scale_factor,
                                simulation_boxsize,
                                OmegaM,
                                cosmo->get_OmegaLambda(),
                                cosmo->get_h(),
                                pos_norm,
                                vel_norm);
        return;
    }
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

/// Output the particles as GADGET HDF5 files: one file per task, <prefix>.<task>.hdf5
/// (or <prefix>.hdf5 if we only have one task)
template <int NDIM, class T>
void output_gadget_hdf5([[maybe_unused]] NBodySimulation<NDIM, T> & sim,
                        [[maybe_unused]] double redshift,
                        [[maybe_unused]] std::string snapshot_folder) {
#ifdef USE_HDF5
    const std::string redshiftstring = redshift_to_string(redshift);
    const auto simulation_boxsize = sim.get_parameters().simulation.boxsize;
    const auto cosmo = sim.get_cosmology();
    auto & part = sim.get_particles();

    const double scale_factor = 1.0 / (1.0 + redshift);
    const int nfiles = FML::NTasks;
    const double pos_norm = simulation_boxsize;
    const double vel_norm = 100 * simulation_boxsize / std::pow(scale_factor, 1.5);
    const std::string fileprefix = snapshot_folder + "/" + "gadget_z" + redshiftstring;
    const std::string filename =
        nfiles == 1 ? fileprefix + ".hdf5" : fileprefix + "." + std::to_string(FML::ThisTask) + ".hdf5";

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Output in gadget HDF5 format\n";
        std::cout << "# fileprefix  : " << fileprefix << "\n";
        std::cout << "#=====================================================\n";
    }

    FML::FILEUTILS::GADGET::GadgetHDF5Writer gw(NDIM);
    if (sim.get_species().size() > 1) {
        // CDM (or CDM+baryons) as PartType1, baryons as PartType2 and neutrinos as PartType3 (with their delta-f
        // weights as PartType3/Weights)
        const double OmegaM = cosmo->get_OmegaM();
        gw.write_gadget_species(filename,
                                gadget_species(sim),
                                nfiles,
                                scale_factor,
                                simulation_boxsize,
                                OmegaM,
                                cosmo->get_OmegaLambda(),
                                cosmo->get_h(),
                                pos_norm,
                                vel_norm);
        return;
    }
    gw.write_gadget_single(filename,
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
#else
    throw std::runtime_error("output_gadget_hdf5: compile with USE_HDF5=true to write HDF5 files");
#endif
}

template <int NDIM, class T>
void compute_bispectrum(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {

    const std::string redshiftstring = redshift_to_string(redshift);
    const auto & settings = sim.get_parameters().bispectrum;
    const double simulation_boxsize = sim.get_parameters().simulation.boxsize;
    const int bispectrum_nmesh = settings.nmesh;
    const int bispectrum_nbins = settings.nbins;
    const std::string bispectrum_density_assignment_method = settings.density_assignment_method;
    const bool bispectrum_interlacing = settings.interlacing;
    const bool bispectrum_subtract_shotnoise = settings.subtract_shotnoise;
    auto & part = sim.get_particles();

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

/// The hybrid bias (Lagrangian bias basis) spectra P_ab(k) of the operators 1, delta_L, delta_L^2, s^2 and
/// nabla^2 delta_L advected with the CDM+baryon particles (written to file). delta_L is the linear field of the initial
/// conditions scaled to the output redshift with the 1LPT growth factor D(k,z) / D(k,zini) of the gravity model
template <int NDIM, class T>
void compute_hybrid_bias_spectra(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {
    using namespace FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
    const auto & settings = sim.get_parameters().hybridbias;
    const double boxsize = sim.get_parameters().simulation.boxsize;
    const double ic_initial_redshift = sim.get_parameters().ic.initial_redshift;
    const int Npart_1D = sim.get_parameters().particle.Npart_1D;
    const auto grav = sim.get_gravity_model();
    const auto transferdata = sim.get_transferdata();
    const auto & power_initial_spline = sim.get_power_initial_spline();
    auto & part = sim.get_particles();
    const double a = 1.0 / (1.0 + redshift);
    const double aini = 1.0 / (1.0 + ic_initial_redshift);

    HybridBiasOptions options;
    options.noperators = settings.noperators;
    options.density_assignment_method = settings.density_assignment_method;
    options.interlacing = settings.interlacing;
    options.subtract_shotnoise = settings.subtract_shotnoise;
    options.smoothing_scale = settings.smoothing_scale / boxsize;
    options.smoothing_method = settings.smoothing_filter;
    // The spherical cut of the linear field (in units of 1/box)
    options.kcut = settings.kcut_over_nyquist * M_PI * Npart_1D;

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Computing hybrid bias spectra\n";
        std::cout << "# hybridbias_nmesh                     : " << settings.nmesh << "\n";
        std::cout << "# hybridbias_noperators                : " << settings.noperators << "\n";
        std::cout << "# hybridbias_density_assignment_method : " << settings.density_assignment_method << "\n";
        std::cout << "# hybridbias_interlacing               : " << settings.interlacing << "\n";
        std::cout << "# hybridbias_subtract_shotnoise        : " << settings.subtract_shotnoise << "\n";
        std::cout << "# hybridbias_smoothing_scale (Mpc/h)   : " << settings.smoothing_scale << "\n";
        std::cout << "# hybridbias_kcut_over_nyquist         : " << settings.kcut_over_nyquist << "\n";
        std::cout << "# hybridbias_zeldovich                 : " << settings.zeldovich << "\n";
        std::cout << "# hybridbias_oneloop                   : " << settings.oneloop << "\n";
        std::cout << "#=====================================================\n";
    }

    // delta_L(k,z) = D(k,z) / D(k,zini) delta_L(k,zini)
    auto growth_factor_of_kBox = [&](double kBox) {
        if (kBox <= 0.0)
            return 1.0;
        const double k = kBox / boxsize;
        return grav->get_D_1LPT(a, k / grav->H0_hmpc) / grav->get_D_1LPT(aini, k / grav->H0_hmpc);
    };

    HybridBiasSpectra<NDIM> spectra;
    FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::compute_hybrid_bias_spectra<NDIM, T>(part.get_particles_ptr(),
                                                                              part.get_npart(),
                                                                              part.get_npart_total(),
                                                                              sim.get_hybridbias_linear_field(),
                                                                              Npart_1D,
                                                                              settings.nmesh,
                                                                              options,
                                                                              spectra,
                                                                              growth_factor_of_kBox);
    spectra.scale(boxsize);

    // The linear P_cb(k,z) for comparison (= P_{delta,delta} on large scales)
    std::vector<double> pofk_linear(spectra.pofk[0].n, 0.0);
    for (int i = 0; i < spectra.pofk[0].n; i++) {
        const double k = spectra.pofk[0].kbin[i];
        if (power_initial_spline) {
            const double D = grav->get_D_1LPT(a, k / grav->H0_hmpc);
            const double Dini = grav->get_D_1LPT(aini, k / grav->H0_hmpc);
            pofk_linear[i] = power_initial_spline(k) * std::pow(D / Dini, 2);
        } else if (transferdata) {
            pofk_linear[i] = transferdata->get_cdm_baryon_power_spectrum(k, a);
        }
    }

    std::ostringstream header;
    header << "# Hybrid bias spectra at z = " << redshift << " (boxsize " << boxsize << " Mpc/h, Npart_1D " << Npart_1D
           << ", nmesh " << settings.nmesh << ", " << settings.density_assignment_method
           << (settings.interlacing ? " with interlacing" : "") << ", shot noise subtracted: " << std::boolalpha
           << settings.subtract_shotnoise << ")\n";
    header << "# The tracer power spectrum is P_tt = sum_ab b_a b_b P_ab with b = (1, b1, b2, bs, bnabla) for the "
              "operators (1, delta_L, delta_L^2 - <delta_L^2>, s^2 - <s^2>, nabla^2 delta_L)\n";
    spectra.write(snapshot_folder + "/hybridbias_z" + redshift_to_string(redshift) + ".txt",
                  header.str(),
                  {{"Pcb_linear(k)", pofk_linear}});

    //=============================================================
    // The LPT predictions (Zel'dovich and/or one-loop) with the same linear P(k,z) and the same cut and smoothing of
    // the operators
    //=============================================================
    if (not settings.zeldovich and not settings.oneloop)
        return;
    if (not power_initial_spline) {
        if (FML::ThisTask == 0)
            std::cout << "Warning: no linear P(k) so we cannot compute the LPT predictions of the hybrid bias spectra\n";
        return;
    }
    const auto krange = power_initial_spline.get_xrange();
    auto pofk_linear_of_k = [&](double k) {
        if (k < krange.first or k > krange.second)
            return 0.0;
        const double D = grav->get_D_1LPT(a, k / grav->H0_hmpc);
        const double Dini = grav->get_D_1LPT(aini, k / grav->H0_hmpc);
        return power_initial_spline(k) * std::pow(D / Dini, 2);
    };
    const double kcut = options.kcut / boxsize;
    const double smoothing_scale = settings.smoothing_scale;
    const std::string filter = settings.smoothing_filter;
    auto W_operators = [&](double k) {
        if (k >= kcut)
            return 0.0;
        if (smoothing_scale <= 0.0)
            return 1.0;
        const double kR = k * smoothing_scale;
        if (filter == "gaussian")
            return std::exp(-0.5 * kR * kR);
        if (filter == "sharpk")
            return kR < 1.0 ? 1.0 : 0.0;
        // tophat
        return kR < 1e-4 ? 1.0 : 3.0 * (std::sin(kR) - kR * std::cos(kR)) / (kR * kR * kR);
    };
    // The displacement field of the IC has the modes of the IC grid (we approximate this by a spherical cut)
    const double kcut_displacement = M_PI * sim.get_parameters().ic.nmesh / boxsize;
    auto W_displacement = [&](double k) { return k < kcut_displacement ? 1.0 : 0.0; };

    // In the bins at low k with few modes we average the prediction over the modes in the bin (as the measurement
    // does) instead of evaluating it at the mean k of the bin (this matters for the steep nabla^2 spectra). The
    // prediction is smooth, so we evaluate it on a log-spaced table spanning these modes and spline it
    const auto & kbin = spectra.pofk[0].kbin;
    const int nbins = int(kbin.size());
    const auto modes = modes_in_bins<NDIM>(spectra.pofk[0], 2.0 * M_PI * 16);
    double kmode_min = std::numeric_limits<double>::max(), kmode_max = 0.0;
    for (int i = 0; i < nbins; i++)
        for (auto & m : modes[i]) {
            kmode_min = std::min(kmode_min, m.first / boxsize);
            kmode_max = std::max(kmode_max, m.first / boxsize);
        }
    // The k-values to evaluate the prediction at (in h/Mpc): first the table, then the centers of the other bins
    const int ntable = kmode_max > 0.0 ? 40 : 0;
    std::vector<double> kall;
    for (int j = 0; j < ntable; j++)
        kall.push_back(std::exp(std::log(kmode_min * 0.999) +
                                std::log(kmode_max * 1.001 / (kmode_min * 0.999)) * j / double(ntable - 1)));
    std::vector<int> jbin(nbins, -1);
    for (int i = 0; i < nbins; i++)
        if (modes[i].empty() and kbin[i] > 0.0) {
            jbin[i] = int(kall.size());
            kall.push_back(kbin[i]);
        }
    // Each task computes a part of the k-values
    std::vector<double> klocal;
    std::vector<int> jlocal;
    for (int j = 0; j < int(kall.size()); j++)
        if (j % FML::NTasks == FML::ThisTask) {
            klocal.push_back(kall[j]);
            jlocal.push_back(j);
        }

    // Bin up the prediction at the local k-values and write it to file
    auto write_prediction = [&](const std::vector<std::vector<double>> & prediction,
                                const std::string & label,
                                const std::string & description,
                                double kcut_displacement_used) {
        auto theory = spectra;
        for (size_t p = 0; p < theory.pofk.size(); p++) {
            std::vector<double> all(kall.size(), 0.0);
            for (size_t j = 0; j < jlocal.size(); j++)
                all[jlocal[j]] = prediction[p][j];
            FML::SumArrayOverTasks(all.data(), int(all.size()));
            std::vector<double> logk_table(ntable), p_table(ntable);
            for (int j = 0; j < ntable; j++) {
                logk_table[j] = std::log(kall[j]);
                p_table[j] = all[j];
            }
            FML::INTERPOLATION::SPLINE::Spline p_spline;
            if (ntable > 0)
                p_spline = FML::INTERPOLATION::SPLINE::Spline(logk_table, p_table, "Hybrid bias prediction");
            for (int i = 0; i < nbins; i++) {
                double value = 0.0;
                if (not modes[i].empty()) {
                    double wsum = 0.0;
                    for (auto & m : modes[i]) {
                        value += m.second * p_spline(std::log(m.first / boxsize));
                        wsum += m.second;
                    }
                    value /= wsum;
                } else if (jbin[i] >= 0) {
                    value = all[jbin[i]];
                }
                theory.pofk[p].pofk[i] = value;
            }
            theory.shotnoise[p] = 0.0;
        }
        std::ostringstream header;
        header << "# " << description << " of the hybrid bias spectra at z = " << redshift
               << " (linear P_cb(k,z), operators cut at k = " << kcut << " h/Mpc"
               << (smoothing_scale > 0.0 ? " and smoothed with R = " + std::to_string(smoothing_scale) + " Mpc/h " + filter
                                         : std::string(""))
               << ", displacement cut at k = " << kcut_displacement_used << " h/Mpc). Same columns as hybridbias_z"
               << redshift_to_string(redshift)
               << ".txt (the shot noise is not included; for k < 16 k_fundamental the prediction is averaged over "
                  "the modes in each bin)\n";
        theory.write(snapshot_folder + "/hybridbias_" + label + "_z" + redshift_to_string(redshift) + ".txt",
                     header.str(),
                     {{"Pcb_linear(k)", pofk_linear}});
    };

    ZeldovichOptions zoptions;
    zoptions.noperators = settings.noperators;
    zoptions.kmin_integration = std::max(1e-5, krange.first);
    if (settings.zeldovich) {
        zoptions.kmax_integration = std::min(std::max(kcut, kcut_displacement), krange.second);
        auto zeldovich = zeldovich_hybrid_bias_spectra(pofk_linear_of_k, W_operators, W_displacement, klocal, zoptions);
        write_prediction(zeldovich, "zeldovich", "Zel'dovich prediction", kcut_displacement);
    }
    if (settings.oneloop) {
        // One-loop LPT: the same filter for the operators and the displacement (the operator cut)
        if (FML::ThisTask == 0 and kcut_displacement > kcut * 1.0001)
            std::cout << "Note: the one-loop prediction uses the operator cut k = " << kcut
                      << " h/Mpc also for the displacement field (the IC have modes up to " << kcut_displacement
                      << " h/Mpc)\n";
        zoptions.kmax_integration = std::min(kcut, krange.second);
        if (settings.oneloop_kernels == "model") {
            // The scale- and time-dependent LPT kernels of the gravity model (flat LCDM background)
            const double OmegaM = sim.get_cosmology()->get_OmegaM();
            MGModel model = mgmodel_gr(OmegaM);
            std::string description = "LCDM";
            if (auto fofr = std::dynamic_pointer_cast<GravityModelFofR<NDIM>>(grav)) {
                model = mgmodel_fofr(OmegaM, fofr->get_fofr0(), fofr->get_nfofr(), fofr->get_use_screening());
            } else if (auto dgp = std::dynamic_pointer_cast<GravityModelDGP<NDIM>>(grav)) {
                model = mgmodel_dgp(OmegaM, dgp->get_rcH0(), dgp->get_use_screening());
            } else if (grav->get_name() != "GR") {
                const double H0_hmpc = grav->H0_hmpc;
                model = mgmodel_mu(OmegaM, [grav, H0_hmpc](double k, double aa) {
                    return grav->GeffOverG(aa, k / H0_hmpc);
                });
            }
            description = model.name();
            if (FML::ThisTask == 0)
                std::cout << "One-loop prediction with the LPT kernels of " << description << "\n";
            // The loop functions are computed for all the outputs at the first output (the kernels are integrated
            // from early times once)
            static std::shared_ptr<MGLoopTables> mg_tables;
            static std::vector<double> mg_tables_redshifts;
            if (not mg_tables) {
                mg_tables_redshifts = sim.get_parameters().output.redshifts;
                std::vector<double> aouts;
                std::vector<std::function<double(double)>> pofks;
                for (double z : mg_tables_redshifts) {
                    const double aa = 1.0 / (1.0 + z);
                    aouts.push_back(aa);
                    pofks.push_back([&, aa](double k) {
                        if (k < krange.first or k > krange.second)
                            return 0.0;
                        const double D = grav->get_D_1LPT(aa, k / grav->H0_hmpc);
                        const double Dini = grav->get_D_1LPT(aini, k / grav->H0_hmpc);
                        return power_initial_spline(k) * std::pow(D / Dini, 2);
                    });
                }
                mg_tables = std::make_shared<MGLoopTables>(mg_loop_tables(model, aouts, pofks, W_operators, zoptions));
            }
            int io = -1;
            for (size_t i = 0; i < mg_tables_redshifts.size(); i++)
                if (std::abs(mg_tables_redshifts[i] - redshift) < 1e-6 * (1.0 + redshift))
                    io = int(i);
            std::vector<std::vector<double>> oneloop;
            if (io >= 0)
                oneloop = oneloop_hybrid_bias_spectra_mg_from_tables(
                    *mg_tables, io, pofk_linear_of_k, W_operators, klocal, zoptions);
            else
                oneloop = oneloop_hybrid_bias_spectra_mg(
                    model, {1.0 / (1.0 + redshift)}, {pofk_linear_of_k}, W_operators, klocal, zoptions)[0];
            write_prediction(oneloop,
                             "oneloop",
                             "One-loop LPT (CLEFT, no counterterms; the LPT kernels of " + description + ") prediction",
                             kcut);
        } else {
            auto oneloop = oneloop_hybrid_bias_spectra(pofk_linear_of_k, W_operators, klocal, zoptions);
            write_prediction(oneloop,
                             "oneloop",
                             "One-loop LPT (CLEFT, no counterterms; EdS kernels) prediction",
                             kcut);
        }
    }
}

/// Returns P_ell(k) for ell = 0,1,...,ellmax (also written to file)
template <int NDIM, class T>
std::vector<FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>>
compute_power_spectrum_multipoles(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {

    const std::string redshiftstring = redshift_to_string(redshift);
    const auto & settings = sim.get_parameters().pofk_multipole;
    const double simulation_boxsize = sim.get_parameters().simulation.boxsize;
    const double ic_initial_redshift = sim.get_parameters().ic.initial_redshift;
    const int pofk_multipole_nmesh = settings.nmesh;
    const std::string pofk_multipole_density_assignment_method = settings.density_assignment_method;
    const bool pofk_multipole_interlacing = settings.interlacing;
    const bool pofk_multipole_subtract_shotnoise = settings.subtract_shotnoise;
    const int pofk_multipole_ellmax = settings.ellmax;
    const auto transferdata = sim.get_transferdata();
    const auto & power_initial_spline = sim.get_power_initial_spline();
    const auto grav = sim.get_gravity_model();
    const auto cosmo = sim.get_cosmology();
    auto & part = sim.get_particles();

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
            double pofk_ini = power_initial_spline(k);
            double D = grav->get_D_1LPT(1.0 / (1.0 + redshift), k / grav->H0_hmpc);
            double Dini = grav->get_D_1LPT(1.0 / (1.0 + ic_initial_redshift), k / grav->H0_hmpc);
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
                  "P0_kaiser    P2_kaiser    P4_kaiser  ShotnoiseSubtracted = " << std::boolalpha << pofk_multipole_subtract_shotnoise << "\n";
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

    return Pells;
}

/// Returns the baryon+CDM P(k) (also written to file)
template <int NDIM, class T>
FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>
compute_power_spectrum(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {

    const std::string redshiftstring = redshift_to_string(redshift);
    const auto & settings = sim.get_parameters().pofk;
    const double simulation_boxsize = sim.get_parameters().simulation.boxsize;
    const double ic_initial_redshift = sim.get_parameters().ic.initial_redshift;
    const int pofk_nmesh = settings.nmesh;
    const std::string pofk_density_assignment_method = settings.density_assignment_method;
    const bool pofk_interlacing = settings.interlacing;
    const bool pofk_subtract_shotnoise = settings.subtract_shotnoise;
    const auto transferdata = sim.get_transferdata();
    const auto & power_initial_spline = sim.get_power_initial_spline();
    const auto grav = sim.get_gravity_model();
    const auto & cosmo = grav->get_cosmo();
    auto & part = sim.get_particles();

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
    // Separate CDM and baryon particles: P(k) of each and of delta_cb = w_c delta_c + w_b delta_b
    std::vector<std::pair<std::string, FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>>> pofk_species;
    if (sim.get_species().size() == 1) {
        FML::CORRELATIONFUNCTIONS::compute_power_spectrum<NDIM, T>(pofk_nmesh,
                                                                   part.get_particles_ptr(),
                                                                   part.get_npart(),
                                                                   part.get_npart_total(),
                                                                   pofk_cb_binning,
                                                                   pofk_density_assignment_method,
                                                                   pofk_interlacing);
        pofk_cb_binning.scale(simulation_boxsize);
    } else {
        const auto nleftright =
            FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(pofk_density_assignment_method);
        auto delta_of = [&](auto & particles) {
            FML::GRID::FFTWGrid<NDIM> grid(pofk_nmesh, nleftright.first, nleftright.second + (pofk_interlacing ? 1 : 0));
            if (pofk_interlacing) {
                FML::INTERPOLATION::particles_to_fourier_grid_interlacing(particles.get_particles_ptr(),
                                                                          particles.get_npart(),
                                                                          particles.get_npart_total(),
                                                                          grid,
                                                                          pofk_density_assignment_method);
            } else {
                FML::INTERPOLATION::particles_to_grid(particles.get_particles_ptr(),
                                                      particles.get_npart(),
                                                      particles.get_npart_total(),
                                                      grid,
                                                      pofk_density_assignment_method);
                grid.fftw_r2c();
            }
            FML::INTERPOLATION::deconvolve_window_function_fourier<NDIM>(grid, pofk_density_assignment_method);
            return grid;
        };
        auto bin = [&](const FML::GRID::FFTWGrid<NDIM> & grid, double shotnoise) {
            FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> binning(pofk_nmesh / 2);
            FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum<NDIM>(grid, binning);
            if (pofk_subtract_shotnoise)
                for (int i = 0; i < binning.n; i++)
                    binning.pofk[i] -= shotnoise;
            binning.scale(simulation_boxsize);
            return binning;
        };
        // delta_cb = sum_i w_i delta_i over the CDM and baryon species (and P(k) of each when there are several)
        FML::GRID::FFTWGrid<NDIM> delta_cb;
        int icb = 0;
        double weight_first = 1.0;
        sim.for_each_cb_species([&](auto & particles, const COLASOLVER::SpeciesInfo & info) {
            auto delta = delta_of(particles);
            if (sim.has_baryons())
                pofk_species.push_back({info.name, bin(delta, sim.species_shotnoise(info))});
            if (icb == 0) {
                delta_cb = std::move(delta);
                weight_first = info.weight_cb;
            } else {
                FML::GRID::linear_combination_fourier<NDIM>(delta_cb, delta, icb == 1 ? weight_first : 1.0, info.weight_cb);
            }
            icb++;
        });
        pofk_cb_binning = bin(delta_cb, sim.cb_shotnoise());
        // Neutrino particles: P_nu(k) and the total P(k) of delta_m = (1 - f_nu) delta_cb + f_nu delta_nu
        if (sim.has_neutrino_particles()) {
            // With force_neutrino_energy_density: the energy-density contrast of the neutrinos, and the total
            // is the energy-density contrast of all matter, ((1 - f_nu) delta_cb + f_nu e delta_nu) / (1 - f_nu +
            // f_nu e) with e = <E/m>(a)
            const double a = 1.0 / (1.0 + redshift);
            typename std::remove_reference_t<decltype(sim.get_neutrinos())>::ScopedEnergyWeights energy_weights(
                sim.get_neutrinos(), a, sim.neutrino_energy_density());
            auto delta_nu = delta_of(sim.get_neutrino_particles());
            if (FML::ThisTask == 0)
                delta_nu.set_fourier_from_index(0, 0.0);
            const double fnu = cosmo->get_fMNu();
            const double nu_shot = sim.get_neutrinos().shotnoise();
            pofk_species.push_back({"nu", bin(delta_nu, nu_shot)});
            if (sim.neutrino_energy_density()) {
                const double e = sim.neutrino_energy_factor(a);
                const double norm = 1.0 / (1.0 - fnu + fnu * e);
                FML::GRID::linear_combination_fourier<NDIM>(delta_cb, delta_nu, (1.0 - fnu) * norm, fnu * e * norm);
                const double shot = ((1.0 - fnu) * (1.0 - fnu) * sim.cb_shotnoise() + fnu * fnu * e * e * nu_shot) *
                                    norm * norm;
                pofk_species.push_back({"total", bin(delta_cb, shot)});
            } else {
                FML::GRID::linear_combination_fourier<NDIM>(delta_cb, delta_nu, 1.0 - fnu, fnu);
                pofk_species.push_back({"total", bin(delta_cb, sim.total_shotnoise())});
            }
        }
    }

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
            double Dini = grav->get_D_1LPT(1.0 / (1.0 + ic_initial_redshift), k / grav->H0_hmpc);
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
        
    // The shot noise of the (weighted) neutrino particles (collective, so computed on all tasks)
    double nu_shotnoise = 0.0;
    if (sim.has_neutrino_particles()) {
        typename std::remove_reference_t<decltype(sim.get_neutrinos())>::ScopedEnergyWeights energy_weights(
            sim.get_neutrinos(), 1.0 / (1.0 + redshift), sim.neutrino_energy_density());
        nu_shotnoise = sim.get_neutrinos().shotnoise() * std::pow(simulation_boxsize, NDIM);
    }

    // Output to file
    if (FML::ThisTask == 0) {
    
        // Get non-linear boost (if this is implemented for the model)
        // Currently we only have EuclidEmulator2 predictions for LCDM and w0waCDM
        FML::INTERPOLATION::SPLINE::Spline non_linear_boost_of_k =
            cosmo->get_nonlinear_matter_power_spectrum_boost(redshift);

        std::string filename = snapshot_folder + "/pofk_z" + redshiftstring + ".txt";
        std::ofstream fp(filename.c_str());
        if (not fp.is_open()) {
            std::cout << "Warning: Cannot write power-spectrum to file, failed to open [" << filename << "]\n";
        } else {
            fp << "#  k  (h/Mpc)          P(k)  (Mpc/h)^3     P_linear(k)   P_NL(k) (Mpc/h)^3\n";
            for (int i = 0; i < pofk_cb_binning.n; i++) {
                const double k = pofk_cb_binning.kbin[i];
                double boost = non_linear_boost_of_k ? non_linear_boost_of_k(k) : 1.0;
                fp << std::setw(15) << k << " ";
                fp << std::setw(15) << pofk_cb_binning.pofk[i] << " ";
                fp << std::setw(15) << pofk_cb_linear_spline(k) << " ";
                fp << std::setw(15) << pofk_cb_linear_spline(k) * boost << " ";
                fp << "\n";
            }
        }

        // The species (linear theory: P_cb (T_i / T_cb)^2 from the transfer functions)
        const double a = 1.0 / (1.0 + redshift);
        for (auto & sp : pofk_species) {
            std::string fname = snapshot_folder + "/pofk_" + sp.first + "_z" + redshiftstring + ".txt";
            std::ofstream fps(fname.c_str());
            fps << "#  k  (h/Mpc)          P_" << sp.first << "(k)  (Mpc/h)^3     P_" << sp.first << "_linear(k)";
            if (sp.first == "nu")
                fps << "   [shot noise of the neutrino particles sum w^2/N^2 V = " << nu_shotnoise
                    << " (Mpc/h)^3, subtracted: " << std::boolalpha << pofk_subtract_shotnoise << "]";
            fps << "\n";
            for (int i = 0; i < sp.second.n; i++) {
                const double k = sp.second.kbin[i];
                double ratio = 1.0;
                if (transferdata) {
                    const double Ti = COLASOLVER::species_transfer_function(sp.first, *transferdata, k, a);
                    ratio = std::pow(Ti / transferdata->get_cdm_baryon_transfer_function(k, a), 2);
                }
                fps << std::setw(15) << k << " " << std::setw(15) << sp.second.pofk[i] << " " << std::setw(15)
                    << pofk_cb_linear_spline(k) * ratio << "\n";
            }
        }
    }

    return pofk_cb_binning;
}

// The FoF halos (compute_fof_halos)
#include "HaloFinding.h"

#endif
