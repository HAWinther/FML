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
#include "OutputUtils.h"
#include "SnapshotOutput.h"
#include "SnapshotPowerSpectrum.h"

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

template <int NDIM, class T>
void output_pofk_for_every_step(const NBodySimulation<NDIM, T> & sim) {

    // Only task 0 outputs
    if (FML::ThisTask > 0)
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
    const auto transfers = sim.get_transfers();
    const auto ic_use_gravity_model_GR = params.ic.use_gravity_model_GR;
    const auto pofk_subtract_shotnoise = params.pofk.subtract_shotnoise;

    //=============================================================
    // Output all CMB+baryon Pofk
    //=============================================================
    for (auto & p : pofk_cb_every_step) {
        auto redshift = p.first;
        auto binning = p.second;
        auto pofk_cb = [&](double k) { return sim.linear_pofk_cb(k, 1.0 / (1.0 + redshift)); };

        // Get non-linear boost (if this is implemented for the model)
        // Currently we only have EuclidEmulator2 predictions for LCDM and w0waCDM
        FML::INTERPOLATION::SPLINE::Spline non_linear_boost_of_k =
            cosmo->get_nonlinear_matter_power_spectrum_boost(redshift);

        std::string filename =
            params.output.path("pofk_" + params.simulation.name + "_cb_z" + redshift_to_string(redshift) + ".txt");

        std::ofstream fp(filename.c_str());
        fp << "# k  (h/Mpc)    Pcb(k)  (Mpc/h)^3    Pcb_linear(k) (Mpc/h)^3    Pcb_NL(k) (Mpc/h)^3   "
              "ShotnoiseSubtracted = "
           << std::boolalpha << pofk_subtract_shotnoise << "\n";
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
    for (auto & [species, list] : sim.get_pofk_species_every_step()) {
        const std::string label = SIMCODE::species_name(species);
        for (auto & p : list) {
            const double redshift = p.first;
            const auto & binning = p.second;
            const double a = 1.0 / (1.0 + redshift);
            std::string filename = params.output.path("pofk_" + params.simulation.name + "_" + label + "_z" +
                                                      redshift_to_string(redshift) + ".txt");
            std::ofstream fp(filename.c_str());
            fp << "# k  (h/Mpc)    P_" << label << "(k)  (Mpc/h)^3    P_" << label
               << "_linear(k) (Mpc/h)^3   ShotnoiseSubtracted = " << std::boolalpha << pofk_subtract_shotnoise << "\n";
            for (int i = 0; i < binning.n; i++) {
                const double k = binning.kbin[i];
                double plin = 0.0;
                if (transfers)
                    plin =
                        sim.linear_pofk_cb(k, a) * sim.species_linear_ratio(SIMCODE::species_transfer(species), k, a);
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
            // If ic_use_gravity_model_GR then transfers is that of GR so
            // we need to rescale it with growthfactors
            if (ic_use_gravity_model_GR) {
                double D = grav->get_D_1LPT(1.0 / (1.0 + redshift), k / grav->H0_hmpc);
                double Dini = grav->get_D_1LPT(1.0 / (1.0 + ic_initial_redshift), k / grav->H0_hmpc);
                double D_GR = grav_ic->get_D_1LPT(1.0 / (1.0 + redshift), k / grav->H0_hmpc);
                double Dini_GR = grav_ic->get_D_1LPT(1.0 / (1.0 + ic_initial_redshift), k / grav->H0_hmpc);
                fac = std::pow((D / Dini) / (D_GR / Dini_GR), 2);
            }
            if (transfers)
                return fac * transfers->power(SIMCODE::TransferSpecies::TotalMatter, k, 1.0 / (1.0 + redshift));
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
        fp << "# k  (h/Mpc)    P(k)  (Mpc/h)^3    P_linear(k)  (Mpc/h)^3     P_NL(k)  ShotnoiseSubtracted = "
           << std::boolalpha << pofk_subtract_shotnoise << "\n";
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
    sim.for_each_species([&](auto & particles, const SIMCODE::SpeciesInfo & info) {
        using P = std::remove_reference_t<decltype(*particles.get_particles_ptr())>;
        auto * p = particles.get_particles_ptr();
        auto sp = FML::FILEUTILS::GADGET::make_gadget_species(info.gadget_type,
                                                              p,
                                                              particles.get_npart(),
                                                              particles.get_npart_total(),
                                                              info.mass_fraction);
        if constexpr (FML::PARTICLE::has_get_deltaf_weight<P>())
            sp.extra_scalars.push_back({"Weights", [p](size_t i) { return double(p[i].get_deltaf_weight()); }});
        // The mass-encoded baryon-CDM relative mode: per-particle masses (MassTable 0 for these types). The particle
        // type can have a mass without using it (ParticleFiducial): then all masses are 1 and are not written
        if constexpr (FML::PARTICLE::has_get_mass<P>())
            if (sim.particles_have_masses())
                sp.get_mass = [p](size_t i) { return double(FML::PARTICLE::GetMass(p[i])); };
        species.push_back(sp);
    });
    return species;
}

/// Output the particles as GADGET files (output_fileformat = GADGET or HDF5, one file per task; see
/// SnapshotOutput.h): all the species with their types when there are several, else the particles as type 1
template <int NDIM, class T>
void output_particles(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {
    const auto & params = sim.get_parameters();
    const auto format = params.output.fileformat;
    const std::string fileprefix = snapshot_folder + "/" + "gadget_z" + redshift_to_string(redshift);
    const auto header =
        SIMCODE::snapshot_header(*sim.get_cosmology(), 1.0 / (1.0 + redshift), params.simulation.boxsize);

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Output in gadget " << (format == SIMCODE::ParticleFileFormat::HDF5 ? "HDF5 " : "")
                  << "format\n";
        std::cout << "# fileprefix  : " << fileprefix << "\n";
        std::cout << "#=====================================================\n";
    }

    // Several species: CDM (or CDM+baryons) as type 1, baryons as type 2 and neutrinos as type 3 (collisionless
    // types: type 0 is SPH gas in GADGET; in HDF5 the neutrinos have their delta-f weights as PartType3/Weights)
    if (sim.get_species().size() > 1) {
        SIMCODE::write_snapshot_species<NDIM>(format, fileprefix, gadget_species(sim), header);
        return;
    }
    auto & part = sim.get_particles();
    SIMCODE::write_snapshot<NDIM>(format,
                                  fileprefix,
                                  part.get_particles_ptr(),
                                  part.get_npart(),
                                  part.get_npart_total(),
                                  header);
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

/// The k-values the LPT predictions of the hybrid bias spectra are computed at, and how they are binned like the
/// measurement. In the bins at low k with few modes we average the prediction over the modes in the bin (as the
/// measurement does) instead of evaluating it at the mean k of the bin (this matters for the steep nabla^2 spectra).
/// The prediction is smooth, so we evaluate it on a log-spaced table spanning these modes and spline it. Each task
/// computes a part of the k-values (klocal)
template <int NDIM>
struct HybridBiasPredictionGrid {
    double boxsize;
    int nbins{0};
    std::vector<std::vector<std::pair<double, double>>> modes;
    int ntable{0};
    std::vector<double> kall;
    std::vector<int> jbin;
    std::vector<double> klocal;
    std::vector<int> jlocal;

    HybridBiasPredictionGrid(const FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> & binning, double boxsize)
        : boxsize(boxsize) {
        using FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::modes_in_bins;
        const auto & kbin = binning.kbin;
        nbins = int(kbin.size());
        modes = modes_in_bins<NDIM>(binning, 2.0 * M_PI * 16);
        double kmode_min = std::numeric_limits<double>::max(), kmode_max = 0.0;
        for (int i = 0; i < nbins; i++)
            for (auto & m : modes[i]) {
                kmode_min = std::min(kmode_min, m.first / boxsize);
                kmode_max = std::max(kmode_max, m.first / boxsize);
            }
        // The k-values to evaluate the prediction at (in h/Mpc): first the table, then the centers of the other bins
        ntable = kmode_max > 0.0 ? 40 : 0;
        for (int j = 0; j < ntable; j++)
            kall.push_back(std::exp(std::log(kmode_min * 0.999) +
                                    std::log(kmode_max * 1.001 / (kmode_min * 0.999)) * j / double(ntable - 1)));
        jbin.assign(nbins, -1);
        for (int i = 0; i < nbins; i++)
            if (modes[i].empty() and kbin[i] > 0.0) {
                jbin[i] = int(kall.size());
                kall.push_back(kbin[i]);
            }
        // Each task computes a part of the k-values
        for (int j = 0; j < int(kall.size()); j++)
            if (j % FML::NTasks == FML::ThisTask) {
                klocal.push_back(kall[j]);
                jlocal.push_back(j);
            }
    }

    /// Put the prediction at the local k-values (prediction[p][j] for spectrum p) into the bins of theory
    void bin(const std::vector<std::vector<double>> & prediction,
             FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::HybridBiasSpectra<NDIM> & theory) const {
        for (size_t p = 0; p < theory.pofk.size(); p++) {
            std::vector<double> all(kall.size(), 0.0);
            for (size_t j = 0; j < jlocal.size(); j++)
                all[jlocal[j]] = prediction[p][j];
            FML::SumArrayOverTasks(all.data(), all.size());
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
    }
};

/// The model of the one-loop LPT kernels of the gravity model (hybridbias_oneloop_kernels = model): its theory (f(R)
/// and DGP with their screening if the simulation has it, other theories through their mu(k, a)) on the run's
/// background; models without a shared theory (mu BZ/table) through GeffOverG
template <int NDIM, class T>
FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::MGModel hybrid_bias_mg_model(NBodySimulation<NDIM, T> & sim) {
    using namespace FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
    const auto grav = sim.get_gravity_model();
    auto cosmo = sim.get_cosmology();
    auto qs = grav->get_quasi_static_gravity();
    bool screening = false;
    if (auto scr = std::dynamic_pointer_cast<GravityModelScreened<NDIM>>(grav))
        screening = scr->get_use_screening();
    MGModel model = mgmodel_from_theory(cosmo->background(), qs, screening);
    if (not qs and grav->get_name() != "GR") {
        const double H0_hmpc = grav->H0_hmpc;
        model.type = MGModel::MU;
        model.mu_of_k_a = [grav, H0_hmpc](double k, double aa) { return grav->GeffOverG(aa, k / H0_hmpc); };
    }
    return model;
}

/// Measure the hybrid bias spectra of the CDM+baryon particles (see compute_hybrid_bias_spectra) and write them to
/// file with the linear P_cb(k,z). Returns the spectra; options: the options used
template <int NDIM, class T>
FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::HybridBiasSpectra<NDIM>
measure_hybrid_bias_spectra(NBodySimulation<NDIM, T> & sim,
                            double redshift,
                            const std::string & snapshot_folder,
                            FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::HybridBiasOptions & options,
                            std::vector<double> & pofk_linear) {
    using namespace FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
    const auto & settings = sim.get_parameters().hybridbias;
    const double boxsize = sim.get_parameters().simulation.boxsize;
    const double ic_initial_redshift = sim.get_parameters().ic.initial_redshift;
    const int Npart_1D = sim.get_parameters().particle.Npart_1D;
    const auto grav = sim.get_gravity_model();
    auto & part = sim.get_particles();
    const double a = 1.0 / (1.0 + redshift);
    const double aini = 1.0 / (1.0 + ic_initial_redshift);

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
    pofk_linear.assign(spectra.pofk[0].n, 0.0);
    for (int i = 0; i < spectra.pofk[0].n; i++) {
        const double k = spectra.pofk[0].kbin[i];
        pofk_linear[i] = sim.linear_pofk_cb(k, a);
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
    return spectra;
}

/// The one-loop LPT prediction of the hybrid bias spectra at the k-values klocal with the kernels of model (the loop
/// functions for all the outputs are computed at the first output, as the kernels are integrated from early times,
/// and kept by the simulation)
template <int NDIM, class T, class FP, class FW, class FD>
std::vector<std::vector<double>>
hybrid_bias_oneloop_mg(NBodySimulation<NDIM, T> & sim,
                       const FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::MGModel & model,
                       double redshift,
                       std::pair<double, double> krange,
                       FP && pofk_linear_of_k,
                       FW && W_operators,
                       FD && W_displacement,
                       const std::vector<double> & klocal,
                       const FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::ZeldovichOptions & zoptions) {
    using namespace FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
    // The loop functions are computed for all the outputs at the first output (the kernels are integrated
    // from early times once) and kept by the simulation
    auto & mg_tables = sim.get_hybridbias_mg_tables();
    const auto & mg_tables_redshifts = sim.get_parameters().output.redshifts;
    if (not mg_tables) {
        std::vector<double> aouts;
        std::vector<std::function<double(double)>> pofks;
        for (double z : mg_tables_redshifts) {
            const double aa = 1.0 / (1.0 + z);
            aouts.push_back(aa);
            pofks.push_back([&, aa](double k) {
                if (k < krange.first or k > krange.second)
                    return 0.0;
                return sim.linear_pofk_cb(k, aa);
            });
        }
        mg_tables = std::make_shared<MGLoopTables>(mg_loop_tables(model, aouts, pofks, W_operators, W_displacement, zoptions));
    }
    int io = -1;
    for (size_t i = 0; i < mg_tables_redshifts.size(); i++)
        if (std::abs(mg_tables_redshifts[i] - redshift) < 1e-6 * (1.0 + redshift))
            io = int(i);
    std::vector<std::vector<double>> oneloop;
    if (io >= 0)
        oneloop = oneloop_hybrid_bias_spectra_mg_from_tables(
            *mg_tables, io, pofk_linear_of_k, W_operators, W_displacement, klocal, zoptions);
    else
        oneloop = oneloop_hybrid_bias_spectra_mg(model,
                                                 {1.0 / (1.0 + redshift)},
                                                 {pofk_linear_of_k},
                                                 W_operators,
                                                 W_displacement,
                                                 klocal,
                                                 zoptions)[0];
    return oneloop;
}

/// The hybrid bias (Lagrangian bias basis) spectra P_ab(k) of the operators 1, delta_L, delta_L^2, s^2 and
/// nabla^2 delta_L advected with the CDM+baryon particles (written to file). delta_L is the linear field of the initial
/// conditions scaled to the output redshift with the 1LPT growth factor D(k,z) / D(k,zini) of the gravity model
template <int NDIM, class T>
void compute_hybrid_bias_spectra(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {
    using namespace FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
    const auto & settings = sim.get_parameters().hybridbias;
    const double boxsize = sim.get_parameters().simulation.boxsize;
    const auto & power_initial_spline = sim.get_power_initial_spline();
    const double a = 1.0 / (1.0 + redshift);

    HybridBiasOptions options;
    std::vector<double> pofk_linear;
    const auto spectra = measure_hybrid_bias_spectra(sim, redshift, snapshot_folder, options, pofk_linear);

    //=============================================================
    // The LPT predictions (Zel'dovich and/or one-loop) with the same linear P(k,z) and the same cut and smoothing of
    // the operators
    //=============================================================
    if (not settings.zeldovich and not settings.oneloop)
        return;
    if (not power_initial_spline) {
        if (FML::ThisTask == 0)
            std::cout
                << "Warning: no linear P(k) so we cannot compute the LPT predictions of the hybrid bias spectra\n";
        return;
    }
    const auto krange = power_initial_spline.get_xrange();
    auto pofk_linear_of_k = [&](double k) {
        if (k < krange.first or k > krange.second)
            return 0.0;
        return sim.linear_pofk_cb(k, a);
    };
    const double kcut = options.kcut / boxsize;
    const double smoothing_scale = settings.smoothing_scale;
    const std::string filter = settings.smoothing_filter;
    // The filter of the smoothing of the operators (FML::GRID::smoothing_filter_fourier_space) as a function of k^2
    const std::function<double(double)> W_of_k2 =
        smoothing_scale > 0.0 ? FML::GRID::smoothing_filter_of_k2<NDIM>(smoothing_scale, filter) : nullptr;
    auto W_operators = [&](double k) {
        if (k >= kcut)
            return 0.0;
        return W_of_k2 ? W_of_k2(k * k) : 1.0;
    };
    // The displacement field of the IC has the modes of the IC grid (we approximate this by a spherical cut)
    const double kcut_displacement = M_PI * sim.get_parameters().ic.nmesh / boxsize;
    auto W_displacement = [&](double k) { return k < kcut_displacement ? 1.0 : 0.0; };

    // The k-values the predictions are computed at (each task a part of them) and their binning
    const HybridBiasPredictionGrid<NDIM> grid(spectra.pofk[0], boxsize);

    // Bin up the prediction at the local k-values and write it to file
    auto write_prediction = [&](const std::vector<std::vector<double>> & prediction,
                                const std::string & label,
                                const std::string & description,
                                double kcut_displacement_used) {
        auto theory = spectra;
        grid.bin(prediction, theory);
        std::ostringstream header;
        header << "# " << description << " of the hybrid bias spectra at z = " << redshift
               << " (linear P_cb(k,z), operators cut at k = " << kcut << " h/Mpc"
               << (smoothing_scale > 0.0 ?
                       " and smoothed with R = " + std::to_string(smoothing_scale) + " Mpc/h " + filter :
                       std::string(""))
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
        auto zeldovich =
            zeldovich_hybrid_bias_spectra(pofk_linear_of_k, W_operators, W_displacement, grid.klocal, zoptions);
        write_prediction(zeldovich, "zeldovich", "Zel'dovich prediction", kcut_displacement);
    }
    if (settings.oneloop) {
        // One-loop LPT: the operators with their cut and smoothing, the displacement with the modes of the IC
        zoptions.kmax_integration = std::min(std::max(kcut, kcut_displacement), krange.second);
        if (settings.oneloop_kernels == "model") {
            const MGModel model = hybrid_bias_mg_model(sim);
            const std::string description = model.name();
            if (FML::ThisTask == 0)
                std::cout << "One-loop prediction with the LPT kernels of " << description << "\n";
            const auto oneloop = hybrid_bias_oneloop_mg(sim,
                                                        model,
                                                        redshift,
                                                        krange,
                                                        pofk_linear_of_k,
                                                        W_operators,
                                                        W_displacement,
                                                        grid.klocal,
                                                        zoptions);
            write_prediction(oneloop,
                             "oneloop",
                             "One-loop LPT (CLEFT, no counterterms; the LPT kernels of " + description + ") prediction",
                             kcut_displacement);
        } else {
            auto oneloop =
                oneloop_hybrid_bias_spectra(pofk_linear_of_k, W_operators, W_displacement, grid.klocal, zoptions);
            write_prediction(oneloop,
                             "oneloop",
                             "One-loop LPT (CLEFT, no counterterms; EdS kernels) prediction",
                             kcut_displacement);
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
    const int pofk_multipole_nmesh = settings.nmesh;
    const std::string pofk_multipole_density_assignment_method = settings.density_assignment_method;
    const bool pofk_multipole_interlacing = settings.interlacing;
    const bool pofk_multipole_subtract_shotnoise = settings.subtract_shotnoise;
    const int pofk_multipole_ellmax = settings.ellmax;
    const auto transfers = sim.get_transfers();
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
    auto pofk_cb = [&](double k) { return sim.linear_pofk_cb(k, 1.0 / (1.0 + redshift)); };
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
                  "P0_kaiser    P2_kaiser    P4_kaiser  ShotnoiseSubtracted = "
               << std::boolalpha << pofk_multipole_subtract_shotnoise << "\n";
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

/// The P(k) of a particle species (its name, transfer functions for linear theory and the binning)
template <int NDIM>
struct SpeciesPofk {
    std::string name;
    SIMCODE::TransferSpecies transfer;
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> binning;
};

/// The CDM+baryon P(k) of a simulation with several species (delta_cb = w_c delta_c + w_b delta_b) and the P(k) of
/// the species: CDM and baryons (separate particles), the neutrino particles and the total matter
template <int NDIM, class T>
FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>
measure_species_power_spectra(NBodySimulation<NDIM, T> & sim,
                              double redshift,
                              std::vector<SpeciesPofk<NDIM>> & pofk_species) {
    const auto & settings = sim.get_parameters().pofk;
    const double simulation_boxsize = sim.get_parameters().simulation.boxsize;
    const int pofk_nmesh = settings.nmesh;
    const std::string pofk_density_assignment_method = settings.density_assignment_method;
    const bool pofk_interlacing = settings.interlacing;
    const bool pofk_subtract_shotnoise = settings.subtract_shotnoise;
    const auto & cosmo = sim.get_gravity_model()->get_cosmo();
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> pofk_cb_binning(pofk_nmesh / 2);
    const auto nleftright = FML::INTERPOLATION::get_extra_slices(pofk_density_assignment_method);
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
    sim.for_each_cb_species([&](auto & particles, const SIMCODE::SpeciesInfo & info) {
        auto delta = delta_of(particles);
        if (sim.has_baryons())
            pofk_species.push_back(
                {info.name, SIMCODE::species_transfer(info.species), bin(delta, sim.species_shotnoise(info))});
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
            sim.get_neutrinos(),
            a,
            sim.neutrino_energy_density());
        auto delta_nu = delta_of(sim.get_neutrino_particles());
        if (FML::ThisTask == 0)
            delta_nu.set_fourier_from_index(0, 0.0);
        const double fnu = cosmo->get_fMNu();
        const double nu_shot = sim.get_neutrinos().shotnoise();
        pofk_species.push_back({SIMCODE::species_name(SIMCODE::ParticleSpecies::MassiveNeutrino),
                                SIMCODE::TransferSpecies::MassiveNeutrino,
                                bin(delta_nu, nu_shot)});
        if (sim.neutrino_energy_density()) {
            const double e = sim.neutrino_energy_factor(a);
            const double norm = 1.0 / (1.0 - fnu + fnu * e);
            FML::GRID::linear_combination_fourier<NDIM>(delta_cb, delta_nu, (1.0 - fnu) * norm, fnu * e * norm);
            const double shot =
                ((1.0 - fnu) * (1.0 - fnu) * sim.cb_shotnoise() + fnu * fnu * e * e * nu_shot) * norm * norm;
            pofk_species.push_back({"total", SIMCODE::TransferSpecies::TotalMatter, bin(delta_cb, shot)});
        } else {
            FML::GRID::linear_combination_fourier<NDIM>(delta_cb, delta_nu, 1.0 - fnu, fnu);
            pofk_species.push_back(
                {"total", SIMCODE::TransferSpecies::TotalMatter, bin(delta_cb, sim.total_shotnoise())});
        }
    }

    return pofk_cb_binning;
}

/// Write the CDM+baryon P(k) (pofk_z*.txt, with linear theory and the nonlinear boost if the model has one) and the
/// P(k) of the species (pofk_<name>_z*.txt, with linear theory P_cb (T_i / T_cb)^2 from the transfer functions)
template <int NDIM, class T>
void write_power_spectra(NBodySimulation<NDIM, T> & sim,
                         double redshift,
                         const std::string & snapshot_folder,
                         const FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> & pofk_cb_binning,
                         const std::vector<SpeciesPofk<NDIM>> & pofk_species) {
    const std::string redshiftstring = redshift_to_string(redshift);
    const double simulation_boxsize = sim.get_parameters().simulation.boxsize;
    const bool pofk_subtract_shotnoise = sim.get_parameters().pofk.subtract_shotnoise;
    const auto & cosmo = sim.get_gravity_model()->get_cosmo();
    //=============================================================
    // Compute linear predictions
    //=============================================================
    auto pofk_cb = [&](double k) { return sim.linear_pofk_cb(k, 1.0 / (1.0 + redshift)); };
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
            sim.get_neutrinos(),
            1.0 / (1.0 + redshift),
            sim.neutrino_energy_density());
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
            std::string fname = snapshot_folder + "/pofk_" + sp.name + "_z" + redshiftstring + ".txt";
            std::ofstream fps(fname.c_str());
            fps << "#  k  (h/Mpc)          P_" << sp.name << "(k)  (Mpc/h)^3     P_" << sp.name << "_linear(k)";
            if (sp.transfer == SIMCODE::TransferSpecies::MassiveNeutrino)
                fps << "   [shot noise of the neutrino particles sum w^2/N^2 V = " << nu_shotnoise
                    << " (Mpc/h)^3, subtracted: " << std::boolalpha << pofk_subtract_shotnoise << "]";
            fps << "\n";
            for (int i = 0; i < sp.binning.n; i++) {
                const double k = sp.binning.kbin[i];
                const double ratio = sim.species_linear_ratio(sp.transfer, k, a);
                fps << std::setw(15) << k << " " << std::setw(15) << sp.binning.pofk[i] << " " << std::setw(15)
                    << pofk_cb_linear_spline(k) * ratio << "\n";
            }
        }
    }
}

/// Returns the baryon+CDM P(k) (also written to file)
template <int NDIM, class T>
FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>
compute_power_spectrum(NBodySimulation<NDIM, T> & sim, double redshift, std::string snapshot_folder) {
    const auto & settings = sim.get_parameters().pofk;
    const double simulation_boxsize = sim.get_parameters().simulation.boxsize;
    const int pofk_nmesh = settings.nmesh;
    const std::string pofk_density_assignment_method = settings.density_assignment_method;
    const bool pofk_interlacing = settings.interlacing;
    const bool pofk_subtract_shotnoise = settings.subtract_shotnoise;
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
    std::vector<SpeciesPofk<NDIM>> pofk_species;
    if (sim.get_species().size() == 1) {
        pofk_cb_binning = SIMCODE::measure_snapshot_power_spectrum<NDIM>(part.get_particles_ptr(),
                                                                         part.get_npart(),
                                                                         part.get_npart_total(),
                                                                         settings,
                                                                         simulation_boxsize);
    } else {
        pofk_cb_binning = measure_species_power_spectra(sim, redshift, pofk_species);
    }

    write_power_spectra(sim, redshift, snapshot_folder, pofk_cb_binning, pofk_species);
    return pofk_cb_binning;
}

// The FoF halos (compute_fof_halos)
#include "HaloFinding.h"

#endif
