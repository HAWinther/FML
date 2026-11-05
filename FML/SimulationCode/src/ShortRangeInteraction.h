#ifndef SIMCODE_SHORT_RANGE_INTERACTION_H
#define SIMCODE_SHORT_RANGE_INTERACTION_H
#include "ShortRangeSettings.h"
namespace SIMCODE {
    // Phi=-G M/r [mu0+(muinf-mu0) exp(-r/lambda)] outside the spline support.
    // range_mpch is COMOVING Mpc/h; zero denotes infinite range, not zero force range.
    struct ShortRangeInteraction {
        double mu0{1}, muinf{1}, range_mpch{0};
        void validate() const {
            if (!std::isfinite(mu0) || !std::isfinite(muinf) || !std::isfinite(range_mpch) || range_mpch < 0 ||
                (range_mpch > 0 ? !(mu0 > 0) : muinf < 0))
                throw std::invalid_argument("Invalid model short-range coupling or comoving range");
            if (range_mpch > 0 && !std::isfinite((muinf - mu0) / mu0))
                throw std::invalid_argument("Unrepresentable relative Yukawa coupling");
        }
        double prefactor() const { return range_mpch == 0 ? muinf : mu0; }
        FML::SHORT_RANGE_GRAVITY::Kernel
        kernel(const ShortRangeSettings & settings, double boxsize, double normalization = 1) const {
            validate();
            auto options = settings.options(boxsize, normalization * prefactor());
            if (range_mpch == 0 || muinf == mu0)
                return FML::SHORT_RANGE_GRAVITY::Kernel(options);
            return FML::SHORT_RANGE_GRAVITY::Kernel(
                options,
                FML::SHORT_RANGE_GRAVITY::NewtonYukawa{(muinf - mu0) / mu0, range_mpch / boxsize});
        }
        double fourier_multiplier(double kbox2, double boxsize) const {
            if (range_mpch == 0)
                return muinf;
            const double x = std::sqrt(kbox2) * (range_mpch / boxsize);
            const double ratio = x > 1 ? 1 / (1 + (1 / x) * (1 / x)) : x * x / (1 + x * x);
            return mu0 + (muinf - mu0) * ratio;
        }
    };
} // namespace SIMCODE
#endif
