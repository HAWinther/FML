#ifndef FML_COSMOLOGY_EINSTEINBOLTZMANNSOLVER_H
#define FML_COSMOLOGY_EINSTEINBOLTZMANNSOLVER_H

// =============================================================================
// EinsteinBoltzmannSolver: the whole solver in one object.
//
// The stages, each built on the previous ones:
//   Background     BackgroundCosmology   - H(x), eta(x), t(x), distances (the model of settings.cosmo)
//   Recombination  RecombinationHistory  - X_e(x), tau(x), visibility function g(x)
//   Perturbations  Perturbations         - Phi, Psi, delta, v, Theta_ell (x, k), source functions (with
//                                          isocurvature modes: one solution per independent component)
//   Spectra        PowerSpectrum         - C_ell (lensed; + Tensors when r > 0), P(k), two-point functions
//                                          (only P(k) when compute_cells = false)
//
// Usage:
//   EinsteinBoltzmannSolver ebs(ParameterFile::read("params/planck2018.ini", {"lensing=true"}));
//   ebs.solve();                                   // all stages (or solve(Stage::Perturbations), ...)
//   double s8 = ebs.power_spectrum().get_sigma8();
//   ebs.output();                                  // the files enabled in the settings, in settings.output_dir
//
// solve(last) runs the stages up to last that are not done yet, so the stages can also be run one at a time.
// reset(first) frees the stages from first on (e.g. the perturbations once the spectra have been extracted), and
// set_verbose(false) stops the printing of the parameters and of each stage's info.
// The stage objects live as long as the solver; the accessors throw if their stage has not been solved.
// The object can be moved (the stages are on the heap and keep their references to each other) but not copied.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Perturbations/Tensors.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>

#include <memory>
#include <string>
#include <vector>

namespace FML::COSMOLOGY {

    class EinsteinBoltzmannSolver {
      public:
        enum class Stage { Background = 0, Recombination = 1, Perturbations = 2, Spectra = 3 };

        explicit EinsteinBoltzmannSolver(RunSettings settings = RunSettings{});
        /// From a parameter file (empty name: defaults) with "name=value" overrides
        static EinsteinBoltzmannSolver from_file(const std::string & filename,
                                                 const std::vector<std::string> & overrides = {});

        EinsteinBoltzmannSolver(EinsteinBoltzmannSolver &&) noexcept = default;
        EinsteinBoltzmannSolver & operator=(EinsteinBoltzmannSolver &&) noexcept = default;
        EinsteinBoltzmannSolver(const EinsteinBoltzmannSolver &) = delete;
        EinsteinBoltzmannSolver & operator=(const EinsteinBoltzmannSolver &) = delete;
        ~EinsteinBoltzmannSolver();

        /// Run the stages up to (and including) last that are not solved yet; prints each stage's info
        void solve(Stage last = Stage::Spectra);
        bool is_solved(Stage stage) const { return int(stage) < n_solved; }
        /// Free the stages from first on (they can be solved again)
        void reset(Stage first);
        /// Print the parameters and each stage's info while solving (default true)
        void set_verbose(bool v) { verbose = v; }

        /// Write every output enabled in the settings for the solved stages to settings.output_dir (also the
        /// parameters: params_used.ini, params.txt)
        void output() const;

        const RunSettings & settings() const { return s; }
        const BackgroundCosmology & background() const;
        const RecombinationHistory & recombination() const;
        /// The first perturbation solution (it contains the adiabatic mode)
        const Perturbations & perturbations() const;
        /// All independent components (isocurvature modes; one element without them)
        std::vector<const Perturbations *> perturbation_components() const;
        const PowerSpectrum & power_spectrum() const;
        /// nullptr unless the C_ell were computed with r > 0
        const Tensors * tensors() const { return tens.get(); }

      private:
        RunSettings s;
        int n_solved = 0;
        bool verbose = true;
        std::unique_ptr<BackgroundCosmology> cosmo;
        std::unique_ptr<RecombinationHistory> rec;
        std::vector<std::unique_ptr<Perturbations>> pert; // pert[0]: adiabatic (+ first isocurvature component)
        std::unique_ptr<Tensors> tens;
        std::unique_ptr<PowerSpectrum> power;

        void require(Stage stage, const char * what) const;
    };

} // namespace FML::COSMOLOGY

#endif
