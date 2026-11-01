#ifndef FML_PERTURBATIONS_PERTURBATIONMODEL_W0WA_H
#define FML_PERTURBATIONS_PERTURBATIONMODEL_W0WA_H

// =====================================================================================================================
// w0wa: the standard species (StandardSpeciesPerturbations) in general relativity, with smooth w0wa dark energy (the
// background: Background/BackgroundModel_w0wa.h). As in the Einstein-Boltzmann solver the dark energy has no
// perturbations, which is gauge-inconsistent for w != -1 (~2e-3 in P(k) on large scales): meant for background data.
// A warning is printed (once per process) unless w0 = -1, wa = 0. A constant-w fluid with perturbations is
// dark_scattering with ds_xi = 0 (PerturbationModel_DarkScattering.h).
//   make_w0wa_perturbations(p, w0, wa, r)
// =====================================================================================================================

#include <FML/Background/BackgroundModel_w0wa.h>
#include <FML/Background/ComponentBackground.h>
#include <FML/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Perturbations/PerturbationModel.h>
#include <FML/Perturbations/PerturbedFieldEquation_GeneralRelativity.h>

#include <atomic>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbationModel_w0wa : public PerturbationModel {
      public:
        explicit PerturbationModel_w0wa(BACKGROUND::ComponentBackgroundPtr background)
            : PerturbationModel(std::move(background)), standard(*bg) {
            if (not bg->field_equations_as<BACKGROUND::GeneralRelativity>())
                throw std::invalid_argument("[PerturbationModel_w0wa] The background is not general relativity");
            for (const auto & s : bg->species())
                if (const auto * d = dynamic_cast<const BACKGROUND::CPLDarkEnergy *>(s.get()))
                    de = d;
            if (not de)
                throw std::invalid_argument("[PerturbationModel_w0wa] The background has no w0wa dark energy");
            static std::atomic<bool> warned{false}; // once per process (parameter fits build many models)
            if ((de->get_w0() != -1.0 or de->get_wa() != 0.0) and not warned.exchange(true))
                std::cerr << "Warning: w0wa: the dark energy has no perturbations (gauge-inconsistent for w != -1, "
                             "~2e-3 in P(k) on large scales); for a constant-w fluid with perturbations use "
                             "dark_scattering with ds_xi = 0\n";
        }
        std::string name() const override { return "w0wa"; }

        SpeciesPerturbationList species_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            SpeciesPerturbationList list;
            standard.add_to(ctx, list);
            return list;
        }
        std::unique_ptr<PerturbedFieldEquation>
        field_equation_perturbations(const PerturbationContext & ctx) const override {
            check(ctx);
            return std::make_unique<PerturbedFieldEquation_GeneralRelativity>(ctx);
        }

      private:
        StandardSpeciesPerturbations standard;
        const BACKGROUND::CPLDarkEnergy * de = nullptr;
    };

    inline std::shared_ptr<const PerturbationModel_w0wa>
    make_w0wa_perturbations(const BACKGROUND::BackgroundParameters & p, double w0, double wa, BACKGROUND::BackgroundRange r = {}) {
        return std::make_shared<const PerturbationModel_w0wa>(BACKGROUND::make_w0wa(p, w0, wa, r));
    }

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
