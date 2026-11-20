#ifndef FML_FORCE_OPTIONS_H
#define FML_FORCE_OPTIONS_H
#include <stdexcept>
#include <string>
namespace FML::NBODY {
    /// Enum listing options we have for what fourier space kernel to use for 1/D^2
    enum GreensFunctionLaplaceOperatorKernels : int {
        // Continuous kernel: -1/k^2 - Poor mans Green's solver
        CONTINUOUS_GREENS_FUNCTION,
        // This kernel corresponds to using the symmetric 2-point formula for D = (phi(i+1)-phi(i-1)) to get D^2 =
        // [phi(i+1)+phi(i-1)-2phi(i)]/h^2
        // This is also the discrete Green's function described by Hockney & Eastwood (1988), used by FastPM
        // (Feng et al. 2016, https://arxiv.org/abs/1603.00476) and GLAM. Compare the GADGET-2 force discussion:
        // https://arxiv.org/abs/astro-ph/0505010.
        DISCRETE_GREENS_FUNCTION_FROM_TWO_POINT,
        // This kernel corresponds to using the symmetric 4-point formula for D to get D^2
        DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT,
    };

    /// Enum lising options we have for what kernel to use for the gradient D
    enum GradientKernels : int {
        // Continuous kernel D = ik_vec
        CONTINUOUS_GRADIENT,
        // This kernel corresponds to using the symmetric 2-point formula for D = (phi(i+1)-phi(i-1))
        DISCRETE_TWO_POINT_GRADIENT,
        // This kernel corresponds to using the symmetric 4-point formula for D
        DISCRETE_FOUR_POINT_GRADIENT,
    };


    /// Per-call force discretization. No process-wide mutable defaults.
    ///
    /// Warning: with a force grid an integer factor finer than the particle grid (e.g. Nmesh = 2 Npart_1D) the
    /// particles of the initial lattice sit exactly on grid nodes. With CIC and the default continuous gradient
    /// this gives spurious forces as long as the displacements are small compared to a force cell, and the
    /// growth of the large-scale modes can be wrong by tens of per cent (a factor of several in a test at very
    /// small amplitude). Use gradient_kernel = DISCRETE_TWO_POINT_GRADIENT (or TSC, or offset the lattice by half
    /// a force cell and use DISCRETE_FOUR_POINT_GRADIENT as SimulationCode does). See ForceOptions.md.
    class LatticeGreensFunction;

    struct ForceOptions {
        GreensFunctionLaplaceOperatorKernels laplace_kernel{CONTINUOUS_GREENS_FUNCTION};
        GradientKernels gradient_kernel{CONTINUOUS_GRADIENT};
        bool deconvolve{false};
        // The number of window functions W(k) in the force that deconvolve divides out: 2 for assignment and
        // interpolation with the kernel, 1 if one of them is exact (NUFFT), 0 if both are
        int deconvolve_window_power{2};
        // If set, the Green's function of every force mode is multiplied by the lattice-calibrated factor (see
        // LatticeGreensFunction.h): the linear force on particles near a lattice is then exact. Not owned
        const LatticeGreensFunction * lattice_greens{nullptr};
        void validate() const {
            if (laplace_kernel < CONTINUOUS_GREENS_FUNCTION ||
                laplace_kernel > DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT || gradient_kernel < CONTINUOUS_GRADIENT ||
                gradient_kernel > DISCRETE_FOUR_POINT_GRADIENT)
                throw std::invalid_argument("[ForceOptions] Invalid force kernel");
        }
    };
    inline GreensFunctionLaplaceOperatorKernels laplace_kernel_from_name(std::string kernel) {
        if (kernel == "" or kernel == "fiducial") {
            return CONTINUOUS_GREENS_FUNCTION;
        } else if (kernel == "continuous") {
            return CONTINUOUS_GREENS_FUNCTION;
        } else if (kernel == "discrete_2pt") {
            return DISCRETE_GREENS_FUNCTION_FROM_TWO_POINT;
        } else if (kernel == "discrete_4pt") {
            return DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT;
        } else {
            throw std::runtime_error("Error in laplace_kernel_from_name. Unknown kernel (continuous, "
                                     "discrete_2pt, discrete_4pt) got" +
                                     kernel);
        }
    }
    inline GradientKernels gradient_kernel_from_name(std::string kernel) {
        if (kernel == "" or kernel == "fiducial") {
            return CONTINUOUS_GRADIENT;
        } else if (kernel == "continuous") {
            return CONTINUOUS_GRADIENT;
        } else if (kernel == "discrete_2pt") {
            return DISCRETE_TWO_POINT_GRADIENT;
        } else if (kernel == "discrete_4pt") {
            return DISCRETE_FOUR_POINT_GRADIENT;
        } else {
            throw std::runtime_error("Error in gradient_kernel_from_name. Unknown kernel (continuous, "
                                     "discrete_2pt, discrete_4pt) got" +
                                     kernel);
        }
    }
} // namespace FML::NBODY
#endif
