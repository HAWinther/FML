#ifndef MODELS_HEADER
#define MODELS_HEADER

#include "Cosmology.h"
#include "Cosmology_DGP.h"
#include "Cosmology_JBD.h"
#include "Cosmology_LCDM.h"
#include "Cosmology_w0wa.h"
#include "Cosmology_Quintessence.h"
#include "Cosmology_EDE.h"
#include "Cosmology_Tabulated.h"

#include "GravityModel.h"
#include "GravityModel_DGP.h"
#include "GravityModel_GR.h"
#include "GravityModel_Geff.h"
#include "GravityModel_JBD.h"
#include "GravityModel_fofr.h"
#include "GravityModel_mbeta.h"
#include "GravityModel_mu.h"
#include "GravityModel_symmetron.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

//=============================================================================
// The cosmologies and gravity models that can be selected in the parameterfile
// (cosmology_model and gravity_model). To add a new model: write the class
// (inheriting from Cosmology or GravityModel), include its header above and add
// a line to the list below. The parameters it reads go in ParameterSchema.h
//=============================================================================

namespace SIMCODE {

    using CosmologyFactory = std::function<std::shared_ptr<Cosmology>()>;
    template <int NDIM>
    using GravityModelFactory = std::function<std::shared_ptr<GravityModel<NDIM>>(std::shared_ptr<Cosmology>)>;

    /// All the cosmologies: (name in the parameterfile, how to create it)
    inline const std::vector<std::pair<std::string, CosmologyFactory>> & cosmology_models() {
        static const std::vector<std::pair<std::string, CosmologyFactory>> models{
            {"LCDM", []() { return std::make_shared<CosmologyLCDM>(); }},
            {"w0waCDM", []() { return std::make_shared<Cosmologyw0waCDM>(); }},
            {"DGP", []() { return std::make_shared<CosmologyDGP>(); }},
            {"JBD", []() { return std::make_shared<CosmologyJBD>(); }},
            {"Quintessence", []() { return std::make_shared<CosmologyQuintessence>(); }},
            {"EDE", []() { return std::make_shared<CosmologyEDE>(); }},
            {"Tabulated", []() { return std::make_shared<CosmologyTabulated>(); }},
        };
        return models;
    }

    /// All the gravity models: (name in the parameterfile, how to create it)
    template <int NDIM>
    const std::vector<std::pair<std::string, GravityModelFactory<NDIM>>> & gravity_models() {
        static const std::vector<std::pair<std::string, GravityModelFactory<NDIM>>> models{
            {"GR", [](auto cosmo) { return std::make_shared<GravityModelGR<NDIM>>(cosmo); }},
            {"f(R)", [](auto cosmo) { return std::make_shared<GravityModelFofR<NDIM>>(cosmo); }},
            {"DGP", [](auto cosmo) { return std::make_shared<GravityModelDGP<NDIM>>(cosmo); }},
            {"JBD", [](auto cosmo) { return std::make_shared<GravityModelJBD<NDIM>>(cosmo); }},
            {"Symmetron", [](auto cosmo) { return std::make_shared<GravityModelSymmetron<NDIM>>(cosmo); }},
            {"Geff", [](auto cosmo) { return std::make_shared<GravityModelGeff<NDIM>>(cosmo); }},
            {"mbeta", [](auto cosmo) { return std::make_shared<GravityModelmbeta<NDIM>>(cosmo); }},
            {"mu", [](auto cosmo) { return std::make_shared<GravityModelMu<NDIM>>(cosmo); }},
        };
        return models;
    }

    /// The names in a list of (name, factory)
    template <class List>
    std::vector<std::string> model_names(const List & models) {
        std::vector<std::string> names;
        for (auto & model : models)
            names.push_back(model.first);
        return names;
    }

    /// Find a model by name. Throws (listing the available models) if it does not exist
    template <class List>
    auto find_model(const List & models, const std::string & name, const std::string & what) {
        for (auto & model : models)
            if (model.first == name)
                return model.second;
        std::string available;
        for (auto & model : models)
            available += " " + model.first;
        throw std::runtime_error("Unknown " + what + " [" + name + "]. Available:" + available);
    }

    inline std::shared_ptr<Cosmology> make_cosmology(const std::string & name) {
        return find_model(cosmology_models(), name, "cosmology")();
    }

    template <int NDIM>
    std::shared_ptr<GravityModel<NDIM>> make_gravity_model(const std::string & name, std::shared_ptr<Cosmology> cosmo) {
        return find_model(gravity_models<NDIM>(), name, "gravity model")(cosmo);
    }

} // namespace SIMCODE

#endif
