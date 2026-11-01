#ifndef FML_PERTURBATIONS_PERTURBATIONMODEL_H
#define FML_PERTURBATIONS_PERTURBATIONMODEL_H

// =====================================================================================================================
// PerturbationModel: the linear perturbations of a model, built on its background (a BACKGROUND::ComponentBackground,
// held as a shared_ptr so that it outlives everything built from it).
//
//   species_perturbations(ctx)         which background species get which SpeciesPerturbation (one object per run).
//                                      Baryons and photons are not included (they stay in the core of the solver);
//                                      smooth species (Lambda, ...) get none.
//   field_equation_perturbations(ctx)  the PerturbedFieldEquation of the model's field equations
// ctx must be a context of this model's background (context(settings) makes one).
//
// A model finds the background objects it needs (typed) once in its constructor; nothing is looked up or cast when
// the perturbations are evaluated. StandardSpeciesPerturbations does that for the standard species, which models
// reuse:
//   every CDM species                     SpeciesPerturbation_CDM
//   every massless-neutrino species       SpeciesPerturbation_MasslessNeutrinos, unless rho today = 0, or
//                                         settings.neutrinos = false (then with a warning: inconsistent model), as
//                                         the solver
//   every massive-neutrino species        SpeciesPerturbation_MassiveNeutrinos (a MassiveNeutrinoSpecies, or one
//                                         wrapped in a ScaledSpecies, e.g. the Brans-Dicke density definitions)
// The models: PerturbationModel_LCDM.h, PerturbationModel_BransDicke.h; the registry: PerturbationPresets.h.
// =====================================================================================================================

#include <FML/Background/ComponentBackground.h>
#include <FML/Background/Species.h>
#include <FML/Perturbations/PerturbationTypes.h>
#include <FML/Perturbations/PerturbedFieldEquation.h>
#include <FML/Perturbations/SpeciesPerturbation.h>
#include <FML/Perturbations/SpeciesPerturbation_CDM.h>
#include <FML/Perturbations/SpeciesPerturbation_MassiveNeutrinos.h>
#include <FML/Perturbations/SpeciesPerturbation_MasslessNeutrinos.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    using SpeciesPerturbationList = std::vector<std::unique_ptr<SpeciesPerturbation>>;

    class PerturbationModel {
      public:
        explicit PerturbationModel(BACKGROUND::ComponentBackgroundPtr background) : bg(std::move(background)) {
            if (not bg)
                throw std::invalid_argument("[PerturbationModel] No background");
        }
        virtual ~PerturbationModel() = default;
        virtual std::string name() const = 0;

        const BACKGROUND::ComponentBackground & background() const { return *bg; }
        const BACKGROUND::ComponentBackgroundPtr & background_ptr() const { return bg; }
        /// A context of this model's background
        PerturbationContext context(PerturbationSettings settings = {}) const {
            return PerturbationContext(*bg, settings);
        }

        virtual SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const = 0;
        virtual std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const = 0;

      protected:
        BACKGROUND::ComponentBackgroundPtr bg;

        void check(const PerturbationContext & ctx) const {
            if (&ctx.background != bg.get())
                throw std::invalid_argument("[" + name() + "] The context is not one of this model's background");
        }
    };

    using PerturbationModelPtr = std::shared_ptr<const PerturbationModel>;

    /// The perturbations of the standard species of a background (see the header)
    class StandardSpeciesPerturbations {
      public:
        explicit StandardSpeciesPerturbations(const BACKGROUND::ComponentBackground & bg) {
            using BACKGROUND::Component;
            for (const auto & s : bg.species()) {
                if (s->component() == Component::CDM)
                    entries.push_back({Kind::CDM, s.get(), nullptr, 0.0});
                else if (s->component() == Component::MasslessNeutrinos)
                    entries.push_back({Kind::Massless, s.get(), nullptr, 0.0});
                else if (s->component() == Component::MassiveNeutrinos) {
                    // A MassiveNeutrinoSpecies, or one wrapped in a ScaledSpecies (rescaled densities)
                    double factor = 1.0;
                    const BACKGROUND::Species * inner = s.get();
                    if (const auto * scaled = dynamic_cast<const BACKGROUND::ScaledSpecies *>(inner)) {
                        factor = scaled->factor();
                        inner = &scaled->species();
                    }
                    const auto * typed = dynamic_cast<const BACKGROUND::MassiveNeutrinoSpecies *>(inner);
                    if (not typed)
                        throw std::invalid_argument("[StandardSpeciesPerturbations] A massive neutrino species that "
                                                    "is not a (scaled) MassiveNeutrinoSpecies");
                    entries.push_back({Kind::Massive, s.get(), typed, factor});
                }
            }
        }

        /// Append the perturbations of the standard species (in the order of the background's species) to list
        void add_to(const PerturbationContext & ctx, SpeciesPerturbationList & list) const {
            for (const auto & e : entries) {
                switch (e.kind) {
                    case Kind::CDM:
                        list.push_back(std::make_unique<SpeciesPerturbation_CDM>(*e.species, ctx));
                        break;
                    case Kind::Massless:
                        if (not(e.species->rho(0.0) > 0.0))
                            break;
                        if (not ctx.settings.neutrinos) {
                            std::cerr << "[Perturbations] warning: Omega_nu > 0 but neutrino perturbations are "
                                         "switched off (inconsistent model)\n";
                            break;
                        }
                        list.push_back(std::make_unique<SpeciesPerturbation_MasslessNeutrinos>(*e.species, ctx));
                        break;
                    case Kind::Massive:
                        list.push_back(
                            std::make_unique<SpeciesPerturbation_MassiveNeutrinos>(*e.massive, ctx, e.factor));
                        break;
                }
            }
        }

      private:
        enum class Kind { CDM, Massless, Massive };
        struct Entry {
            Kind kind;
            const BACKGROUND::Species * species;
            const BACKGROUND::MassiveNeutrinoSpecies * massive; ///< the massive neutrinos (possibly wrapped)
            double factor;                                      ///< the ScaledSpecies factor (1 if not wrapped)
        };
        std::vector<Entry> entries;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
