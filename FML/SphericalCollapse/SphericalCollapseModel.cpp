#include "SphericalCollapseModel.h"

namespace FML {
  namespace COSMOLOGY {
    namespace SPHERICALCOLLAPSE {

      //================================================
      // Some some general info about the cosmology
      //================================================
      void SphericalCollapseModel::info() const {
        std::cout << "\n#=====================================================\n";
        std::cout << "# SphericalCollapseModel model info:\n";
        std::cout << "#=====================================================\n";
        std::cout << "OmegaM : " << OmegaM << "\n";
        auto z_array = FML::MATH::linspace(0.0,5.0,10);
        for(auto z : z_array){
          const double x = -std::log(1+z);
          std::cout << " z:     " << std::setw(10) << z;
          std::cout << " E(z):  " << std::setw(10) << Eofx(x);
          std::cout << " mu(z): " << std::setw(10) << muofx(x);
          std::cout << " w(z):  " << std::setw(10) << wofx(x) << "\n"; 
        }
        std::cout << "#=====================================================\n";
      }

      //================================================
      // Virial condition. Virialization when this quantity is zero
      //================================================
      double SphericalCollapseModel::virial_condition(
          double delta, 
          double delta_prime, 
          double roverR, 
          double x) const {
        const double E = Eofx(x);
        const double mu = muofx(x);
        const double OmegaM_x = OmegaMofx(x);
        // The force from the smooth (non-matter) components: sum_i OmegaDE_i(x) (1 + 3w_i) from the background
        // acceleration (a''/a = -H^2/2 sum_i Omega_i (1+3w_i) = H^2 (1 + dlogE/dloga)). For one dark energy fluid this is
        // OmegaDE (1 + 3w); curvature does not contribute (before 1 - OmegaM was used for OmegaDE, which counted
        // curvature as dark energy: -4% in DeltaVir for an open model with OmegaM = 0.3)
        const double OmegaDE_one_plus_3w = -2.0 * (1.0 + logEprimeofx(x)) - OmegaM_x;
        const double drdx_over_r = 1.0 - delta_prime / 3.0 / (1.0+delta);
        double EpoverM = -3.0/10.0 * std::pow(E * roverR, 2) * (
            OmegaDE_one_plus_3w +
            OmegaM_x * (1.0 + (mu - 1.0) * delta / (1.0+delta)) * (1.0+delta) 
            );
        double EkoverM = 3.0/10.0 * std::pow(E * roverR, 2) * std::pow(drdx_over_r,2);
        return 2.0*EkoverM + EpoverM;
      }

      //================================================
      // Define non-linear when this quantity is zero
      //================================================
      double SphericalCollapseModel::nonlinear_condition(
          double delta, 
          [[maybe_unused]] double delta_prime, 
          [[maybe_unused]] double roverR, 
          [[maybe_unused]] double x) const {
        return delta-1.0;
      }

    }
  }
}
