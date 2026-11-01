#ifndef FML_COSMOLOGY_BACKGROUND_MODEL_H
#define FML_COSMOLOGY_BACKGROUND_MODEL_H

// =============================================================================
// The registry of the built-in models (names, parameters with defaults and descriptions) and, TRANSITIONAL, the old
// models as a list of species and a theory of gravity.
//
// A run selects a model by name (CosmologyParameters::model, "model = lcdm" in a parameter file); the
// model's own parameters are in CosmologyParameters::model_params (name -> value, defaults from the
// registry). The registry (registry, find, parameter, validate) gives the parameter file, params_used.ini and the
// --write-defaults output their models and parameters. BackgroundCosmology(params) builds the model from the
// registries of FML/Background and FML/Perturbations (same names and parameters).
//
// TRANSITIONAL: Model, the build functions and Models::build (the old classes, Cosmology/Models/) are no longer used by
// the solver; they remain as the references of the tests of FML/Background and FML/Perturbations.
//
// Adding a model: its background and perturbations in FML/Background and FML/Perturbations (with their registry
// entries), and a ModelInfo entry (name, description, parameters with defaults) here.
// =============================================================================

#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Background/Gravity.h>
#include <FML/Cosmology/Background/Species.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace FML::COSMOLOGY {

    struct Model {
        std::string name;
        std::vector<std::shared_ptr<const Species>> species; // in summation order
        std::shared_ptr<Gravity> gravity;
    };

    struct ModelParameter {
        std::string name;
        double default_value;
        std::string description;
    };

    struct ModelInfo {
        std::string name, description;
        std::vector<ModelParameter> parameters;
        std::function<Model(const CosmologyParameters &)> build;
    };

    namespace Models {
        // All built-in models
        const std::vector<ModelInfo> & registry();
        // The entry for a name (throws std::invalid_argument for an unknown model)
        const ModelInfo & find(const std::string & name);
        // Value of a model parameter: params.model_params[name] or the registry default
        double parameter(const CosmologyParameters & params, const std::string & name);
        // Throws std::invalid_argument for an unknown model or parameters that do not belong to it
        void validate(const CosmologyParameters & params);
        // Build the model selected by params.model
        Model build(const CosmologyParameters & params);

        // Building blocks
        // Baryons, CDM, photons, massless neutrinos and (if m_nu > 0) one massive neutrino species
        std::vector<std::shared_ptr<const Species>> standard_species(const CosmologyParameters & params);
        // LCDM: standard species + cosmological constant (Omega_Lambda from closure), GR
        Model lcdm(const CosmologyParameters & params);
        // mu-Sigma (Planck/DES): LCDM species, MuSigmaGravity
        Model musigma(const CosmologyParameters & params);
        // DGP: standard species + Lambda (from E(0) = 1), DGPGravity
        Model dgp(const CosmologyParameters & params);
        // Quintessence: standard species + QuintessenceSpecies (exponential potential), GR
        Model quintessence(const CosmologyParameters & params);
        // Jordan-Brans-Dicke: standard species + Lambda (from E(0) = 1), BransDickeGravity
        Model jbd(const CosmologyParameters & params);
        // Hu-Sawicky f(R): LCDM species, HuSawickyGravity
        Model hu_sawicky(const CosmologyParameters & params);
        // Coupled quintessence: CoupledCDM + CoupledField (exponential potential), GR
        Model coupled_quintessence(const CosmologyParameters & params);
        // Dark scattering: standard species (CDM without own perturbations) + FluidDarkEnergy (w, cs2, xi), GR
        Model dark_scattering(const CosmologyParameters & params);
        // Cubic Galileon on the tracker: standard species + CubicGalileonSpecies, CubicGalileonGravity
        Model cubic_galileon(const CosmologyParameters & params);
        // Horndeski (alphas, hi_class parameterizations): standard species + Lambda or CPL dark energy,
        // HorndeskiAlphaGravity
        Model horndeski(const CosmologyParameters & params);
        // w0-wa dark energy: standard species + CPLDarkEnergy (smooth), GR
        Model w0wa(const CosmologyParameters & params);
        // Early dark energy: standard species + EarlyDarkEnergySpecies (axion-like) + Lambda (closure), GR
        Model ede(const CosmologyParameters & params);
        // Interacting dark energy: standard species with InteractingCDM + InteractingDarkEnergy (fluid), GR
        Model ide(const CosmologyParameters & params);
    } // namespace Models

} // namespace FML::COSMOLOGY

#endif
