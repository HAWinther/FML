#include <FML/Cosmology/BackgroundCosmology/BackgroundCosmology.h>
#include <FML/Cosmology/LinearPerturbations/Perturbations.h>
#include <FML/Cosmology/LinearPowerSpectra/PowerSpectrum.h>
#include <FML/Cosmology/RecombinationHistory/RecombinationHistory.h>
#include <FML/Global/Global.h>

void read_parameters(FML::UTILS::ParameterMap & p);

int main() {

    auto & Units = FML::COSMOLOGY::Constants;

    FML::COSMOLOGY::Constants = FML::UTILS::ConstantsAndUnits("SI");

    // Time the run
    FML::UTILS::Timings timer;
    timer.StartTiming("CMFB");

    //===================================================================
    // 0) Set up parameter container and read in parameters
    // (right now set in the method at the end of main)
    //===================================================================
    FML::UTILS::ParameterMap p;
    read_parameters(p);

    //===================================================================
    // 1) Solve the background cosmology
    //===================================================================
    auto lcdm = std::make_shared<FML::COSMOLOGY::BackgroundCosmology>(p);
    lcdm->solve();
    lcdm->info();

    // Output background quantities
    if (FML::ThisTask == 0) {
        lcdm->output("output/background.txt");
    }

    //===================================================================
    // 2) Solve the recombination history (Saha+Peebles or use recfast)
    //===================================================================
    auto rec = std::make_shared<FML::COSMOLOGY::RecombinationHistory>(lcdm, p);
    rec->solve();
    rec->info();

    // Output recombination quantities
    if (FML::ThisTask == 0) {
        rec->output("output/recombination.txt");
    }

    //===================================================================
    // 3) Integrate the perturbations. Splines up transfer and source functions
    //===================================================================
    auto pert = std::make_shared<FML::COSMOLOGY::Perturbations>(lcdm, rec, p);
    pert->solve();
    pert->info();

    // Output the newtonian gauge perturbations for a given value of k
    if (FML::ThisTask == 0) {
        const double k = 0.01 / Units.Mpc;
        pert->output_perturbations(k, "output/perturbations.txt");
    }

    // Output transfer functions at z = 0.0
    if (FML::ThisTask == 0) {
        const double z = 0.0;
        pert->output_transfer(std::log(1.0 / (1.0 + z)), "output/transfer.txt");
    }

    //===================================================================
    // 4) Compute power-spectra and correlation functions
    //===================================================================
    auto power = std::make_shared<FML::COSMOLOGY::PowerSpectrum>(lcdm, rec, pert, p);
    power->solve();
    power->info();

    // Output Cells (tons of other things we can output also)
    if (FML::ThisTask == 0) {
        power->output_angular_power_spectra("output/cell.txt");
    }

    // Output matter power-spectrum at z = 0.0
    if (FML::ThisTask == 0) {
        const double z = 0.0;
        power->output_matter_power_spectrum(std::log(1.0 / (1.0 + z)), "output/pofk.txt");
    }

    // Output correlation functions at z = 0.0 (requires FFTW to give any output and a high ketamax to be accurate)
    if (FML::ThisTask == 0) {
        const double z = 0.0;
        power->output_correlation_function(std::log(1.0 / (1.0 + z)), "output/corr.txt");
    }

    // Print the timing
    timer.EndTiming("CMFB");
    if (FML::ThisTask == 0)
        timer.PrintAllTimings();
}

// Set the parameters of the run
void read_parameters(FML::UTILS::ParameterMap & p) {
    auto && pmap = p.get_map();

    // Some units needed below
    auto & Units = FML::COSMOLOGY::Constants;
    const double Kelvin = Units.K;
    const double Mpc = Units.Mpc;

    // Add cosmological parameters
    pmap["CosmologyLabel"] = std::string("LCDM");
    pmap["PhysicalParameters"] = true; // Supply physical parameters? i.e. Omegah^2 instead of Omega's
    pmap["OmegaBh2"] = 0.0245;         // Baryon density OmegaB h^2
    pmap["OmegaCDMh2"] = 0.10976;      // CDM density OmegaCDM h^2
    pmap["OmegaKh2"] = 0.0;            // Curvature density OmegaK h^2
    pmap["OmegaLambdah2"] = 0.35574;   // Dark energy density OmegaLambda h^2
    pmap["TCMB"] = 2.7255 * Kelvin;    // CMB temperature in your temperature units
    pmap["h"] = 0.7;                   // Hubble constant H0 / (100km/s/Mpc)
    pmap["Neff"] = 3.046;              // Effective number of neutrinos

    // Add recombination parameters
    pmap["userecfast"] = false;         // Use recfast? (Must be compiled and linked to work)
    pmap["RecFudgeFactor"] = 1.14;      // Fudgefactor in Peebles equation in Recfast
    pmap["Yp"] = 0.24;                  // Helium abundance
    pmap["reionization"] = true;        // Include reionization?
    pmap["z_reion"] = 11.0;             // Reionization redshift
    pmap["delta_z_reion"] = 0.5;        // Reionization width
    pmap["helium_reionization"] = true; // Double Helium reionization?
    pmap["z_helium_reion"] = 3.5;       // Double Helium reionization redshift
    pmap["delta_z_helium_reion"] = 0.5; // Double Helium reionization width

    // Add perturbation parameters
    pmap["polarization"] = true;    // Include photon polarization?
    pmap["neutrinos"] = true;       // Include massless neutrinos?
    pmap["n_ell_theta"] = 12;       // Number of temperature multipoles to keep in the Boltzmann hierarchy
    pmap["n_ell_nu"] = 12;          // Number of Nu multipoles to keep in the Boltzmann hierarchy
    pmap["keta_min"] = 0.1;         // Range we integrate perturbations over
    pmap["keta_max"] = 8000.0;      // Integrate k from k_min*eta0 -> k_max*eta0. Also used for power-spectrum
    pmap["k_max_pert"] = 0.0 / Mpc; // If we want to much higher kmax for perturbations than Cell's use this
                                    // (remove or use 0.0 to let keta_max determine the max)

    // Add power spectrum parameters
    pmap["A_s"] = 2e-9;                            // Primordial amplitude
    pmap["n_s"] = 0.96;                            // Spectral indez
    pmap["kpivot"] = 0.05 / Mpc;                   // Pivot scale
    pmap["ell_max"] = 4000;                        // Maximum ell to compute Cell's for
    pmap["CellOutputRedshift"] = 0.0;              // Redshift to compute Cell at
    pmap["compute_temperature_cells"] = true;      // Compute Cell TT?
    pmap["compute_polarization_cells"] = not true; // Compute Cell EE (and TE if both are true)?
    pmap["compute_lensing_cells"] = not true;      // Compute Cell lensing potential?
    pmap["compute_neutrino_cells"] = not true;     // Compute neutrino Cell?
    pmap["compute_corr_function"] = true;          // Copmute correlation functions?

    // Accuracy settings (NB: keta_max and n_ell's also impact the accuracy)
    pmap["pert_integration_nk_per_logint"] = 25;  // Number of k's per logarithmic interval, 100 for high accuracy
    pmap["pert_spline_all_ells"] = false;         // Spline all the Theta_ell(k,x) etc. multipoles otherwise just 0,1,2
    pmap["pert_x_initial"] = -15.0;               // When to start the integration x = log(aini)
    pmap["pert_delta_x"] = 0.05;                  // Sampling in x for splines of perturbation
    pmap["bessel_nsamples_per_osc"] = 16;         // Sampling of bessel functions
    pmap["los_integration_nsamples_per_osc"] = 8; // Sampling of line of sight integrals
    pmap["los_integration_loga_nsamples"] = 300;  // Samping of line of sight integrals
    pmap["cell_nsamples_per_osc"] = 32;           // Sampling of Cell integration

    // Show all that we have in the map
    if (FML::ThisTask == 0)
        p.info();
}

