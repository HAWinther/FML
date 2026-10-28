#ifndef FML_COSMOLOGY_EINSTEINBOLTZMANNSOLVER_PARAMETERFILE_H
#define FML_COSMOLOGY_EINSTEINBOLTZMANNSOLVER_PARAMETERFILE_H

// =============================================================================
// Parameter files: all cosmological, precision and output settings of a run in one text file.
//
// Format: one "name = value" per line; '#' starts a comment; blank lines are ignored. Names are
// case sensitive. Booleans: true/false, yes/no, on/off, 1/0. Lists: comma separated. Parameters that
// are not given keep their default; an unknown name is an error (catches typos). Command-line
// arguments of the form name=value override the file.
//
//   RunSettings s = ParameterFile::read("params/default.ini", {"h=0.7", "lensing=true"});
//   ParameterFile::write(s, "out/params_used.ini");   // complete, commented file with every value
//
// The complete list of parameters with their defaults and descriptions is generated from the table in
// ParameterFile.cpp: `./cmb --write-defaults params/default.ini`. Units in the file: k in 1/Mpc, angles
// in degrees, everything else as in the code (x = ln a). `preset = planck2018` loads that parameter set
// before the other cosmological parameters are applied. `model = <name>` selects a model from the
// registry (Model.h); its parameters (with their defaults) are then valid names too, and a parameter of
// another model is an error.
// =============================================================================

#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Perturbations/Tensors.h>

#include <string>
#include <utility>
#include <vector>

namespace FML::COSMOLOGY {

    struct RunSettings {
        // Physics and precision
        CosmologyParameters cosmo = CosmologyParameters::fiducial();
        RecombinationOptions rec;
        PerturbationOptions pert;
        PowerSpectrumOptions ps;
        TensorOptions tensor;                      // used when r > 0
        double bg_x_start = -20.0, bg_x_end = 5.0; // background ODE range and resolution
        int bg_n_points = 5000;

        // Output (written to output_dir)
        std::string output_dir = "data";
        bool write_background = true, write_recombination = true;
        std::vector<double> perturbation_k_output = {0.001, 0.01, 0.1}; // [1/Mpc]; empty = none
        bool compute_cells = true; // line-of-sight integration, cells.txt (cells_lensed.txt)
        bool write_matter_power = true;
        std::vector<double> matter_power_z = {0.0};                // redshifts for matter_power[_z<z>].txt
        std::vector<int> transfer_ells = {6, 100, 200, 500, 1000}; // empty = no transfer.txt
        // Two-point functions
        bool cmb_correlation = false;              // cmb_correlation_unlensed.txt (+ _lensed.txt with lensing)
        double cmb_correlation_beam_fwhm = 0.0;    // [deg]; > 0: also *_beam<fwhm>deg.txt
        bool matter_correlation = false;           // matter_correlation.txt, xi(r) at z = 0
        double matter_correlation_smoothing = 0.0; // Gaussian smoothing R [Mpc/h]
        std::vector<double> angular_z = {};        // Gaussian n(z) centres for angular_cl/w_z<z>.txt
        double angular_sigma_z = 0.1, angular_bias = 1.0;
        int angular_ell_max = 2000;
    };

    namespace ParameterFile {
        // Read a parameter file (empty name: defaults only) and apply "name=value" overrides
        RunSettings read(const std::string & filename, const std::vector<std::string> & overrides = {});
        // Write every parameter with its current value and description (a complete, re-readable file)
        void write(const RunSettings & s, const std::string & filename);
        // Names of all general parameters (for documentation and tests), and those of one model
        std::vector<std::string> names();
        // Precision presets ("default": none, "fast": for parameter fits): the (name, value) pairs they set
        std::vector<std::pair<std::string, std::string>> precision_preset(const std::string & name);
        std::vector<std::string> model_parameter_names(const std::string & model);
    } // namespace ParameterFile

} // namespace FML::COSMOLOGY

#endif
