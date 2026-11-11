#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_AXION_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONMODEL_AXION_H

// =====================================================================================================================
// Mixed ultralight-axion dark matter (Background/BackgroundModel_Axion.h): the standard species
// (StandardSpeciesPerturbations, with the CDM share of the dark matter), the axion field (SpeciesPerturbation_ScalarField.h,
// named "axion": the Klein-Gordon equation, then, where the background replaces the oscillating field by a fluid, the
// effective fluid with w = 0 and the sound speed c_s^2 = k^2 / (k^2 + 4 m^2 a^2) (c = 1) of the WKB average, as
// axionCAMB) and a cosmological constant, in general relativity. Outputs delta_phi_axion, delta_axion, q_axion.
//   make_axion_perturbations(p, log10_m_eV, ax_fraction, switch_mH, r)
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundModel_Axion.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Perturbations/PerturbationModel.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Perturbations/SpeciesPerturbation_ScalarField.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_Axion : public PerturbationModel {
      public:
        explicit PerturbationModel_Axion(BACKGROUND::ComponentBackgroundPtr background)
            : PerturbationModel(std::move(background)), standard(*bg) {
            if (not bg->field_equations_as<BACKGROUND::GeneralRelativity>())
                throw std::invalid_argument("[PerturbationModel_Axion] The background is not general relativity");
            for (const auto & s : bg->species())
                if (const auto * a = dynamic_cast<const BACKGROUND::AxionSpecies *>(s.get()))
                    field = a;
            if (not field)
                throw std::invalid_argument("[PerturbationModel_Axion] The background has no axion");
        }
        std::string name() const override { return "axion"; }
        const BACKGROUND::AxionSpecies & axion() const { return *field; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            list.push_back(std::make_unique<SpeciesPerturbation_ScalarField>(
                *field, "axion", ctx, std::vector<std::string>{"delta_phi_axion", "delta_axion", "q_axion"}));
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_GeneralRelativity>(ctx);
        }

      private:
        StandardSpeciesPerturbations standard;
        const BACKGROUND::AxionSpecies * field = nullptr;
    };

    inline std::shared_ptr<const PerturbationModel_Axion>
    make_axion_perturbations(const BACKGROUND::BackgroundParameters & p,
                             double log10_m_eV,
                             double ax_fraction,
                             double switch_mH = 0.0,
                             BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_Axion>(
            BACKGROUND::make_axion(p, log10_m_eV, ax_fraction, switch_mH, r));
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
