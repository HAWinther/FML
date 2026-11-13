#ifndef FML_HYDRODYNAMICS_EQUATION_OF_STATE_HEADER
#define FML_HYDRODYNAMICS_EQUATION_OF_STATE_HEADER
#include <cmath>
#include <stdexcept>
namespace FML::HYDRODYNAMICS {
    /// rho, specific internal energy u, pressure and c_s use one consistent unit system.
    /// Shared gas primitive; no SPH, gravity, cosmology or Riemann-solver dependency.
    class IdealGas {
        double gamma_;

      public:
        explicit IdealGas(double gamma = 5. / 3) : gamma_(gamma) {
            if (!std::isfinite(gamma) || gamma <= 1)
                throw std::invalid_argument("[IdealGas] Require finite gamma > 1");
        }
        double gamma() const { return gamma_; }
        double pressure(double rho, double u) const { return ((gamma_ - 1) * u) * rho; }
        double sound_speed_squared(double, double u) const { return gamma_ * ((gamma_ - 1) * u); }
        double specific_energy(double rho, double pressure) const { return (pressure / rho) / (gamma_ - 1); }
        double entropy_proxy(double rho, double u) const { return (gamma_ - 1) * u / std::pow(rho, gamma_ - 1); }
    };
} // namespace FML::HYDRODYNAMICS
#endif
