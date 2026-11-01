#ifndef FML_COSMOLOGY_BACKGROUND_FUNCTIONBACKGROUND_H
#define FML_COSMOLOGY_BACKGROUND_FUNCTIONBACKGROUND_H

// =====================================================================================================================
// FunctionBackground: a Background from user functions of x = ln a. Bridges the callback interfaces of FML (spherical
// collapse, the halo model, LPT and N-body time stepping take E(x) or H(a)/H0 callbacks) and is handy in tests.
//
//   E(x)        required
//   dlogEdx(x)  optional (default: 4th-order central difference of log E)
//   Omega(c, x) optional (default: baryons and CDM as Omega_i0 e^{-3x} / E^2, curvature as Omega_k e^{-2x} / E^2,
//               other components throw: give Omega if they are needed)
// E(0) must be 1, and x_min must be in matter or radiation domination (see Background::finalize).
// =====================================================================================================================

#include <FML/Cosmology/Background/Background.h>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::BACKGROUND {

    class FunctionBackground : public Background {
      public:
        using Function = std::function<double(double)>;
        using OmegaFunction = std::function<double(Component, double)>;

        FunctionBackground(std::string name,
                           BackgroundParameters params,
                           Function E_of_x,
                           Function dlogEdx_of_x = nullptr,
                           OmegaFunction Omega_of_x = nullptr,
                           BackgroundRange r = {})
            : label(std::move(name)), params(std::move(params)), Ef(std::move(E_of_x)),
              dlogEf(std::move(dlogEdx_of_x)), Omegaf(std::move(Omega_of_x)) {
            this->params.validate();
            if (not Ef)
                throw std::invalid_argument("[" + label + "] E(x) is required");
            range = r;
            if (not(std::abs(Ef(0.0) - 1.0) < 1e-8))
                throw std::invalid_argument("[" + label + "] E(0) = " + std::to_string(Ef(0.0)) + ", not 1");
            finalize();
        }

        std::string name() const override { return label; }
        const BackgroundParameters & parameters() const override { return params; }
        double E(double x) const override { return Ef(x); }
        double dlogEdx(double x) const override {
            if (dlogEf)
                return dlogEf(x);
            const double d = 1e-3;
            const auto L = [&](double xx) { return std::log(Ef(xx)); };
            return (8.0 * (L(x + d) - L(x - d)) - (L(x + 2 * d) - L(x - 2 * d))) / (12.0 * d);
        }
        double Omega(Component c, double x) const override {
            if (Omegaf)
                return Omegaf(c, x);
            const double E2 = Ef(x) * Ef(x);
            switch (c) {
                case Component::Baryons:
                    return params.Omega_b * std::exp(-3.0 * x) / E2;
                case Component::CDM:
                    return params.Omega_cdm * std::exp(-3.0 * x) / E2;
                case Component::Curvature:
                    return params.Omega_k * std::exp(-2.0 * x) / E2;
                case Component::MassiveNeutrinos:
                    if (params.m_nu_eV.empty())
                        return 0.0;
                    [[fallthrough]];
                default:
                    throw std::logic_error("[" + label + "] Omega(" + component_name(c) +
                                           ") is not available: give the Omega function");
            }
        }

      private:
        std::string label;
        BackgroundParameters params;
        Function Ef, dlogEf;
        OmegaFunction Omegaf;
    };

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
