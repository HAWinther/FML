#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_DARKSCATTERING_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_DARKSCATTERING_H

// =====================================================================================================================
// Dark scattering (Background/BackgroundModel_DarkScattering.h: wCDM): the standard species except the CDM
// (StandardSpeciesPerturbations), and the CDM with the dark-energy fluid in one block
// (SpeciesPerturbation_DarkScattering, SpeciesPerturbation_DarkEnergyFluid.h), in general relativity. Parameters as the
// Einstein-Boltzmann solver: xi = sigma_D / m_c in barn/GeV (>= 0; 0 is wCDM with dark-energy perturbations) and the
// rest-frame sound speed cs2 > 0; w (from the background) != -1.
//   make_dark_scattering_perturbations(p, w, xi_barn_per_GeV, cs2, r)
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundModel_DarkScattering.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Perturbations/PerturbationModel.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Perturbations/SpeciesPerturbation_DarkEnergyFluid.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_DarkScattering : public PerturbationModel {
      public:
        PerturbationModel_DarkScattering(BACKGROUND::ComponentBackgroundPtr background,
                                         double xi_barn_per_GeV,
                                         double cs2)
            : PerturbationModel(std::move(background)), cdm(find(*bg, BACKGROUND::Component::CDM)),
              de(find(*bg, BACKGROUND::Component::DarkEnergy)), standard(*bg, {cdm}), cs2(cs2),
              xi(xi_barn_per_GeV * 1e-28 / FML::UTILS::SI::GeV_over_c2) {
            if (not bg->field_equations_as<BACKGROUND::GeneralRelativity>())
                throw std::invalid_argument("[PerturbationModel_DarkScattering] The background is not general "
                                            "relativity");
            if (std::abs(1.0 + de->w(0.0)) < 1e-6)
                throw std::invalid_argument("dark energy fluid: w = -1 has no perturbations (use lcdm)");
            if (not(cs2 > 0.0))
                throw std::invalid_argument("dark energy fluid: need cs2 > 0");
            if (not(xi >= 0.0))
                throw std::invalid_argument("dark scattering: need xi >= 0");
        }
        std::string name() const override { return "dark_scattering"; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            list.push_back(std::make_unique<SpeciesPerturbation_DarkScattering>(*cdm, *de, cs2, xi, ctx));
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_GeneralRelativity>(ctx);
        }

      private:
        const BACKGROUND::Species * cdm;
        const BACKGROUND::Species * de;
        StandardSpeciesPerturbations standard;
        double cs2, xi; ///< xi in m^2/kg

        /// The one species of a component
        static const BACKGROUND::Species * find(const BACKGROUND::ComponentBackground & bg, BACKGROUND::Component c) {
            const BACKGROUND::Species * found = nullptr;
            for (const auto & s : bg.species())
                if (s->component() == c) {
                    if (found)
                        throw std::invalid_argument("[PerturbationModel_DarkScattering] Need one CDM and one dark "
                                                    "energy species");
                    found = s.get();
                }
            if (not found)
                throw std::invalid_argument("[PerturbationModel_DarkScattering] Need a CDM and a dark energy species");
            return found;
        }
    };

    inline std::shared_ptr<const PerturbationModel_DarkScattering>
    make_dark_scattering_perturbations(const BACKGROUND::BackgroundParameters & p,
                                       double w,
                                       double xi_barn_per_GeV,
                                       double cs2 = 1.0,
                                       BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_DarkScattering>(
            BACKGROUND::make_dark_scattering(p, w, r), xi_barn_per_GeV, cs2);
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
