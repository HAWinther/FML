#ifndef FML_STEPFROMBIGBANG_FORCES_H
#define FML_STEPFROMBIGBANG_FORCES_H
// =====================================================================================================================
// Ready-made force functions for StepFromBigBang, for a single species of particles in a model where the force is
//
//   F = dv/dlna = kappa0(a) G,   G = -grad laplace^-1 [ mu(a, k) filter(k) delta ]   (box units)
//
// with kappa0(a) = a^2 E (3/2) Omega_m(a) (kappa0_from_background) and mu = G_eff/G (1 by default). Callers with more
// physics (linear massive neutrinos in the density, several species, a fifth force with screening) write their own
// force function with the same signature (SimulationCode does).
//
//   make_force_nufft : the exact force of the particles up to the mesh's Nyquist frequency (NUFFT): no assignment
//                      window, no aliasing, no lattice artefacts. The right choice near the lattice (early times).
//   make_force_pm    : particle-mesh with a mass assignment (NGP .. PQS), optional interlacing and deconvolution.
// =====================================================================================================================

#include <FML/Cosmology/Background/Background.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/NBody/NBody.h>
#include <FML/NUFFT/NUFFT.h>
#include <FML/StepFromBigBang/StepFromBigBang.h>
#include <FML/Units/Units.h>

#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace FML::STEPFROMBIGBANG {

    /// kappa0(a) = a^2 E (3/2) Omega_m(a) from a background (Omega_m: all clustering matter)
    inline std::function<double(double)> kappa0_from_background(const FML::COSMOLOGY::BACKGROUND::Background & bg) {
        return [&bg](double a) {
            const double x = std::log(a), E = bg.E(x);
            return a * a * E * 1.5 * bg.Omega_m(x);
        };
    }

    /// mu(a, kBox) from mu(x = ln a, koverH0) (e.g. a QuasiStaticGravity) and the box size in Mpc/h
    inline std::function<double(double, double)> mu_of_a_kBox(std::function<double(double, double)> mu_x_koverH0,
                                                               double boxsize) {
        return [mu_x_koverH0, boxsize](double a, double kBox) {
            return mu_x_koverH0(std::log(a), kBox / boxsize * FML::UTILS::SI::c_over_H0_mpch);
        };
    }

    /// The exact (band-limited) force with the NUFFT on nmesh^NDIM modes
    template <int NDIM, class P>
    typename StepFromBigBang<NDIM, P>::ForceFunction
    make_force_nufft(int nmesh,
                     std::function<double(double)> kappa0,
                     std::function<double(double, double)> mu_of_a_kBox = nullptr,
                     double epsilon = 1e-12,
                     double sigma = 2.0) {
        using Particles = typename StepFromBigBang<NDIM, P>::Particles;
        using Filter = typename StepFromBigBang<NDIM, P>::Filter;
        return [=](Particles & part, double a, const Filter & filter, const std::vector<Target> & targets) {
            std::array<std::vector<FML::GRID::FloatType>, NDIM> F;
            std::function<double(double)> filt = nullptr;
            if (filter or mu_of_a_kBox)
                filt = [&](double kBox) {
                    double r = mu_of_a_kBox ? mu_of_a_kBox(a, kBox) : 1.0;
                    if (filter)
                        r *= filter(kBox);
                    return r;
                };
            FML::NUFFT::compute_force_nufft<NDIM, P>(
                part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), nmesh, 1.0, F, epsilon, sigma, filt);
            const double k0 = kappa0(a);
            auto * p = part.get_particles_ptr();
            for (size_t i = 0; i < part.get_npart(); i++) {
                double f[NDIM];
                for (int d = 0; d < NDIM; d++)
                    f[d] = -k0 * F[d][i]; // compute_force_nufft with norm 1 returns +grad laplace^-1 delta
                add_to_targets<NDIM>(p[i], f, targets);
            }
        };
    }

    /// Particle-mesh force on an nmesh^NDIM grid with the given mass assignment (interpolation with the same kernel),
    /// optional interlacing (two grids shifted by half a cell) and deconvolution of the window
    template <int NDIM, class P>
    typename StepFromBigBang<NDIM, P>::ForceFunction
    make_force_pm(int nmesh,
                  std::function<double(double)> kappa0,
                  std::function<double(double, double)> mu_of_a_kBox = nullptr,
                  std::string assignment = "CIC",
                  bool interlacing = false,
                  bool deconvolve = false) {
        using Particles = typename StepFromBigBang<NDIM, P>::Particles;
        using Filter = typename StepFromBigBang<NDIM, P>::Filter;
        return [=](Particles & part, double a, const Filter & filter, const std::vector<Target> & targets) {
            const auto nextra = FML::INTERPOLATION::get_extra_slices(assignment);
            FML::GRID::FFTWGrid<NDIM> density(nmesh, nextra.first, nextra.second);
            FML::INTERPOLATION::particles_to_fourier_grid<NDIM, P>(
                part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), density, assignment, interlacing);
            if (deconvolve)
                FML::INTERPOLATION::deconvolve_window_function_fourier<NDIM>(density, assignment);
            if (filter or mu_of_a_kBox) {
                FML::GRID::for_each_fourier_mode<NDIM>(density, [&](size_t i, double kBox) {
                    double r = mu_of_a_kBox ? mu_of_a_kBox(a, kBox) : 1.0;
                    if (filter)
                        r *= filter(kBox);
                    density.set_fourier_from_index(i, density.get_fourier_from_index(i) * FML::GRID::FloatType(r));
                });
            }
            std::array<FML::GRID::FFTWGrid<NDIM>, NDIM> force_real;
            FML::NBODY::compute_force_from_density_fourier<NDIM>(density, force_real, assignment, 1.0);
            std::vector<std::vector<FML::GRID::FloatType>> F(NDIM);
            for (int d = 0; d < NDIM; d++)
                FML::INTERPOLATION::interpolate_grid_to_particle_positions<NDIM, P>(
                    force_real[d], part.get_particles_ptr(), part.get_npart(), F[d], assignment);
            const double k0 = kappa0(a);
            auto * p = part.get_particles_ptr();
            for (size_t i = 0; i < part.get_npart(); i++) {
                double f[NDIM];
                for (int d = 0; d < NDIM; d++)
                    f[d] = -k0 * F[d][i]; // as the NUFFT: norm 1 gives +grad laplace^-1 delta
                add_to_targets<NDIM>(p[i], f, targets);
            }
        };
    }

    /// The growth model of a background (and optionally a quasi-static theory of gravity): the background must outlive
    /// the model. With scale-dependent growth (a theory with mu(k)), settings.scale_dependent must be true.
    inline GrowthModel growth_from_background(const FML::COSMOLOGY::BACKGROUND::Background & bg,
                                              FML::COSMOLOGY::GROWTH::LPTGrowthSettings settings,
                                              FML::COSMOLOGY::PERTURBATIONS::QuasiStaticGravityPtr qs = nullptr) {
        const auto eq = FML::COSMOLOGY::GROWTH::LPTGrowthEquations::from_background(bg, qs);
        auto lpt = std::make_shared<FML::COSMOLOGY::GROWTH::LPTGrowthFactors>(eq, settings);
        return growth_from_LPTGrowthFactors(lpt, eq);
    }

    //==================================================================================================================
    /// One call: make the particles from the linear density field delta_fourier (at a_ref; with scale-dependent growth
    /// the field of each k is moved with D1(k, a)) and take the steps from settings.a_start to settings.a_end.
    /// part is (re)created. Returns the number of force evaluations.
    //==================================================================================================================
    template <int NDIM, class P>
    int generate_particles_from_big_bang(FML::PARTICLE::MPIParticles<P> & part,
                                         int Npart_1D,
                                         double buffer_factor,
                                         const FML::GRID::FFTWGrid<NDIM> & delta_fourier,
                                         double a_ref,
                                         const GrowthModel & growth,
                                         typename StepFromBigBang<NDIM, P>::ForceFunction force,
                                         const Settings & settings,
                                         double lattice_shift = 0.0) {
        StepFromBigBang<NDIM, P> sfbb(growth, force, settings);
        sfbb.make_start_particles(part, Npart_1D, buffer_factor, delta_fourier, a_ref, lattice_shift);
        sfbb.run(part);
        return sfbb.number_of_forces();
    }

} // namespace FML::STEPFROMBIGBANG
#endif
