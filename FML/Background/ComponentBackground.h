#ifndef FML_BACKGROUND_COMPONENTBACKGROUND_H
#define FML_BACKGROUND_COMPONENTBACKGROUND_H

// =====================================================================================================================
// ComponentBackground: a Background made of species (Species.h) and field equations (FieldEquations.h).
//
//   E^2(x) = field_equations.E2(x, *this)        (GR: sum of the species densities + Omega_k e^{-2x})
//   Omega(c, x) = sum of rho_i of the species of component c / E^2     (Curvature: Omega_k e^{-2x} / E^2)
//
// The species list must close the budget, E(0) = 1 (the presets put the dark energy or the theory's free parameter
// there); the constructor checks it. The models (LCDM, w0wa, DGP, ...) are presets (Presets.h) that assemble species
// and field equations; a new model is a new Species and/or FieldEquations plus a preset, not a new Background.
// standard_species(p) gives the species every model has (baryons, CDM, photons, massless and massive neutrinos).
// =====================================================================================================================

#include <FML/Background/Background.h>
#include <FML/Background/FieldEquations.h>
#include <FML/Background/Species.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::BACKGROUND {

    class ComponentBackground : public Background {
      public:
        using SpeciesList = std::vector<std::shared_ptr<const Species>>;

        ComponentBackground(std::string name,
                            BackgroundParameters params,
                            SpeciesList species_list,
                            std::shared_ptr<FieldEquations> field_equations,
                            BackgroundRange r = {})
            : label(std::move(name)), params(std::move(params)), list(std::move(species_list)),
              eqs(std::move(field_equations)) {
            const std::string msg = "[" + label + "] ";
            this->params.validate();
            if (not eqs)
                throw std::invalid_argument(msg + "No field equations");
            for (const auto & s : list)
                if (not s)
                    throw std::invalid_argument(msg + "Null species");
            range = r;
            eqs->init(*this);
            const double E0 = E(0.0);
            if (not(std::abs(E0 - 1.0) < 1e-8))
                throw std::invalid_argument(msg + "E(0) = " + std::to_string(E0) +
                                            ": the species and the field equations do not close the budget");
            finalize();
        }

        // ---- Background ---------------------------------------------------------------------------------------------
        std::string name() const override { return label; }
        const BackgroundParameters & parameters() const override { return params; }
        double E(double x) const override { return std::sqrt(eqs->E2(x, *this)); }
        double dlogEdx(double x) const override { return 0.5 * eqs->dE2dx(x, *this) / eqs->E2(x, *this); }
        double Omega(Component c, double x) const override {
            if (c == Component::Curvature)
                return rho_k(x) / eqs->E2(x, *this);
            return rho(c, x) / eqs->E2(x, *this);
        }
        std::string description() const override {
            std::ostringstream s;
            s << eqs->info();
            for (const auto & sp : list)
                s << "; " << sp->info();
            return s.str();
        }

        // ---- Components -----------------------------------------------------------------------------------------
        const SpeciesList & species() const { return list; }
        const FieldEquations & field_equations() const { return *eqs; }
        /// The field equations as a given type (nullptr if they are not), e.g. to reach a theory's scalar field
        template <class T>
        const T * field_equations_as() const {
            return dynamic_cast<const T *>(eqs.get());
        }

        /// Sum over all species of rho_i / rho_c0 (curvature not included) and its x-derivative
        double rho_total(double x) const {
            double sum = 0.0;
            for (const auto & s : list)
                sum += s->rho(x);
            return sum;
        }
        double drho_total(double x) const {
            double sum = 0.0;
            for (const auto & s : list)
                sum += s->drhodx(x);
            return sum;
        }
        /// Sum over all species of p_i / (rho_c0 c^2)
        double p_total(double x) const {
            double sum = 0.0;
            for (const auto & s : list)
                sum += s->p(x);
            return sum;
        }
        /// Sum over the species of one component
        double rho(Component c, double x) const {
            double sum = 0.0;
            for (const auto & s : list)
                if (s->component() == c)
                    sum += s->rho(x);
            return sum;
        }
        double p(Component c, double x) const {
            double sum = 0.0;
            for (const auto & s : list)
                if (s->component() == c)
                    sum += s->p(x);
            return sum;
        }
        double drhodx(Component c, double x) const {
            double sum = 0.0;
            for (const auto & s : list)
                if (s->component() == c)
                    sum += s->drhodx(x);
            return sum;
        }
        /// p / rho of a component (0 if there is none)
        double w(Component c, double x) const {
            const double r = rho(c, x);
            return r != 0.0 ? p(c, x) / r : 0.0;
        }
        /// The curvature "density" Omega_k e^{-2x} and its x-derivative
        double rho_k(double x) const { return params.Omega_k * std::exp(-2.0 * x); }
        double drho_k(double x) const { return -2.0 * rho_k(x); }

      private:
        std::string label;
        BackgroundParameters params;
        SpeciesList list;
        std::shared_ptr<FieldEquations> eqs;
    };

    using BackgroundPtr = std::shared_ptr<const Background>;
    using ComponentBackgroundPtr = std::shared_ptr<const ComponentBackground>;

    inline ComponentBackground::SpeciesList standard_species(const BackgroundParameters & p) {
        p.validate();
        ComponentBackground::SpeciesList s{
            std::make_shared<PowerLawSpecies>("baryons", Component::Baryons, p.Omega_b, 3.0),
            std::make_shared<PowerLawSpecies>("cdm", Component::CDM, p.Omega_cdm, 3.0),
            std::make_shared<PowerLawSpecies>("photons", Component::Photons, p.Omega_gamma(), 4.0),
            std::make_shared<PowerLawSpecies>("massless_nu", Component::MasslessNeutrinos, p.Omega_nu_massless(), 4.0),
        };
        for (double m : p.m_nu_eV)
            s.push_back(std::make_shared<MassiveNeutrinoSpecies>(m, p));
        return s;
    }

    /// Sum of rho_i(0) / rho_c0 of a list of species
    inline double density_today(const ComponentBackground::SpeciesList & s) {
        double sum = 0.0;
        for (const auto & sp : s)
            sum += sp->rho(0.0);
        return sum;
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
