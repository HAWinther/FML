#ifndef INITIALCONDITIONS_HEADER
#define INITIALCONDITIONS_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/CAMBUtils/CAMBReader.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/RandomFields/NonLocalGaussianRandomField.h>
#include <FML/RandomGenerator/RandomGenerator.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"
#include "GravityModel.h"
#include "SimulationParameters.h"

#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <memory>
#include <string>
#include <vector>

//=============================================================================
// Generating the initial density field delta(k,zini) that the particles are
// displaced from (with LPT) to create the initial conditions
//=============================================================================

namespace COLASOLVER {

    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using RandomGenerator = FML::RANDOM::RandomGenerator;
    using GSLRandomGenerator = FML::RANDOM::GSLRandomGenerator;

    /// The random number generator to use for the initial conditions
    inline std::shared_ptr<RandomGenerator> make_random_generator(const ICSettings & ic) {
        std::shared_ptr<RandomGenerator> rng;
        if (ic.random_generator == RandomGeneratorType::GSL)
            rng = std::make_shared<GSLRandomGenerator>();
        else
            rng = std::make_shared<RandomGenerator>();
        rng->set_seed(ic.random_seed);
        return rng;
    }

    // The library helpers for loops over the fourier modes
    using FML::GRID::transform_fourier_modes;

    //=============================================================================
    /// Generate a gaussian or non-gaussian (non-local fNL) random field delta(k,zini)
    /// in fourier space with the power-spectrum power_initial_spline
    //=============================================================================
    template <int NDIM>
    void generate_initial_density_field(FFTWGrid<NDIM> & delta_ini_fourier,
                                        const ICSettings & ic,
                                        double boxsize,
                                        const Spline & power_initial_spline,
                                        const Spline & power_primordial_spline,
                                        std::shared_ptr<Cosmology> cosmo,
                                        std::shared_ptr<GravityModel<NDIM>> grav_ic,
                                        RandomGenerator * rng) {

        if (ic.random_field_type == ICRandomFieldType::Gaussian) {

            auto Pofk_of_kBox_over_volume = [&](double kBox) {
                return power_initial_spline(kBox / boxsize) / std::pow(boxsize, NDIM);
            };
            FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier<NDIM>(delta_ini_fourier,
                                                                                rng,
                                                                                Pofk_of_kBox_over_volume,
                                                                                ic.fix_amplitude);

        } else if (ic.random_field_type == ICRandomFieldType::NonGaussian) {

            // The power-spectrum of the Bardeen potential (-Phi) at the given redshift
            const double afnl = 1.0 / (1.0 + ic.fnl_redshift);
            const double H0Box = boxsize * grav_ic->H0_hmpc;
            const double factor = 1.5 * cosmo->get_OmegaM() / afnl;
            auto Pofk_of_kBox_over_volume_primordial = [&](double kBox) {
                return power_primordial_spline(kBox / boxsize) / std::pow(boxsize, NDIM) *
                       std::pow(factor * grav_ic->get_D_1LPT(afnl, kBox / H0Box) /
                                    grav_ic->get_D_1LPT(1.0, kBox / H0Box),
                                2);
            };
            // The product of this and the function above is the initial power-spectrum for delta
            auto Pofk_of_kBox_over_Pofk_primordal = [&](double kBox) {
                return power_initial_spline(kBox / boxsize) / std::pow(boxsize, NDIM) /
                       Pofk_of_kBox_over_volume_primordial(kBox);
            };
            FML::RANDOM::NONGAUSSIAN::generate_nonlocal_gaussian_random_field_fourier_cosmology(
                delta_ini_fourier,
                rng,
                Pofk_of_kBox_over_Pofk_primordal,
                Pofk_of_kBox_over_volume_primordial,
                ic.fix_amplitude,
                ic.fnl,
                ic.fnl_type);
        }
    }

    /// Keep the phases of delta(k), but set the amplitudes to sqrt(P(k,zini)): |delta(k)| -> sqrt(P(k))
    template <int NDIM>
    void
    whiten_and_set_power_spectrum(FFTWGrid<NDIM> & delta_fourier, const Spline & power_initial_spline, double boxsize) {
        auto sqrt_pofk = [&](double kBox) {
            return std::sqrt(power_initial_spline(kBox / boxsize) / std::pow(boxsize, NDIM));
        };
        transform_fourier_modes<NDIM>(delta_fourier, [&](double kmag, std::complex<FML::GRID::FloatType> value) {
            // Whiten the field
            auto absvalue = std::abs(value);
            if (absvalue > 0.0)
                value /= absvalue;
            // Multiply by power-spectrum
            value *= sqrt_pofk(kmag);
            return value;
        });
    }

    /// delta(k) -> -delta(k) (for paired simulations)
    template <int NDIM>
    void reverse_phases(FFTWGrid<NDIM> & delta_fourier) {
        transform_fourier_modes<NDIM>(delta_fourier, [](double, std::complex<FML::GRID::FloatType> value) {
            return FML::GRID::FloatType(-1.0) * value;
        });
    }

    /// The 1LPT potential phi_1LPT(k) = delta(k) / k^2  (i.e. D^2 phi_1LPT = -delta)
    template <int NDIM>
    FFTWGrid<NDIM> compute_phi_1LPT_from_delta(const FFTWGrid<NDIM> & delta_fourier) {
        FFTWGrid<NDIM> phi_1LPT_fourier = delta_fourier;
        auto Local_x_start = phi_1LPT_fourier.get_local_x_start();
        auto Local_nx = phi_1LPT_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            double kmag2;
            std::array<double, NDIM> kvec;
            for (auto && fourier_index : phi_1LPT_fourier.get_fourier_range(islice, islice + 1)) {
                if (Local_x_start == 0 and fourier_index == 0)
                    continue; // DC mode k = 0
                phi_1LPT_fourier.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                auto value = phi_1LPT_fourier.get_fourier_from_index(fourier_index) / FML::GRID::FloatType(kmag2);
                phi_1LPT_fourier.set_fourier_from_index(fourier_index, value);
            }
        }
        // Deal with DC mode
        if (Local_x_start == 0)
            phi_1LPT_fourier.set_fourier_from_index(0, 0.0);
        return phi_1LPT_fourier;
    }


    //=============================================================================
    /// Make the particles of a species i from the CDM+baryon density field delta_cb(k,zini): delta_i(k) =
    /// delta_cb(k) ratio(kB) with ratio = T_i / T_cb, LPT displacements and velocities (1LPT velocity
    /// velocity_norm_1LPT(kB) Psi, the rest from velocity_norms; see FML::NBODY::NBodyInitialConditions). With
    /// LPT_order >= 2 the 2LPT term is that of the total field delta_cb. The particles start from the lattice
    /// offset by lattice_shift (in units of the box) in each direction: the fields are evaluated at the offset
    /// positions and the particles moved there. id_offset is added to the IDs. With store_cb_LPT_fields the LPT
    /// displacement fields stored in the particles (for COLA) are those of delta_cb, i.e. all the species share the
    /// COLA frame of the total CDM+baryon field. With mass_contrast_of_kBox the particles get the masses
    /// m = 1 + eps(q) with eps(k) = delta_cb(k) mass_contrast(kB) (the mass-encoded baryon-CDM relative mode).
    /// Returns delta_i(k) (shifted).
    //=============================================================================
    template <int NDIM, class P>
    FFTWGrid<NDIM> generate_species_from_delta_cb(FML::PARTICLE::MPIParticles<P> & particles,
                                                  const FFTWGrid<NDIM> & delta_cb_fourier,
                                                  const std::function<double(double)> & ratio_of_kBox,
                                                  std::function<double(double)> velocity_norm_1LPT_of_kBox,
                                                  const std::vector<double> & velocity_norms,
                                                  int LPT_order,
                                                  int Npart_1D,
                                                  double allocation_factor,
                                                  double boxsize,
                                                  double zini,
                                                  double lattice_shift,
                                                  long long int id_offset,
                                                  bool store_cb_LPT_fields = false,
                                                  std::function<double(double)> mass_contrast_of_kBox = nullptr) {
        std::array<double, NDIM> shift_vec;
        shift_vec.fill(lattice_shift);

        // delta_i(k) on the (offset) lattice: f(q) -> f(q + shift) so that the particle created at q gets the
        // displacement at q + shift (and is then moved there)
        FFTWGrid<NDIM> delta = delta_cb_fourier;
        const bool need_total = LPT_order >= 2 or store_cb_LPT_fields;
        FFTWGrid<NDIM> delta_total;
        if (need_total)
            delta_total = delta_cb_fourier;
        FML::GRID::multiply_fourier_by_function_of_k<NDIM>(delta, ratio_of_kBox);
        if (lattice_shift != 0.0) {
            FML::GRID::shift_grid_fourier<NDIM>(delta, shift_vec, +1.0);
            if (need_total)
                FML::GRID::shift_grid_fourier<NDIM>(delta_total, shift_vec, +1.0);
        }

        // The mass contrast eps(k) = delta_cb(k) mass_contrast(kB) (on the same offset lattice)
        FFTWGrid<NDIM> mass_contrast;
        if (mass_contrast_of_kBox) {
            mass_contrast = delta_cb_fourier;
            FML::GRID::multiply_fourier_by_function_of_k<NDIM>(mass_contrast, mass_contrast_of_kBox);
            if (lattice_shift != 0.0)
                FML::GRID::shift_grid_fourier<NDIM>(mass_contrast, shift_vec, +1.0);
        }

        std::vector<FFTWGrid<NDIM>> no_potentials;
        FML::NBODY::NBodyInitialConditions<NDIM, P>(particles,
                                                    Npart_1D,
                                                    allocation_factor,
                                                    delta,
                                                    no_potentials,
                                                    LPT_order,
                                                    boxsize,
                                                    zini,
                                                    velocity_norms,
                                                    velocity_norm_1LPT_of_kBox,
                                                    LPT_order >= 2 ? &delta_total : nullptr,
                                                    store_cb_LPT_fields ? &delta_total : nullptr,
                                                    mass_contrast_of_kBox ? &mass_contrast : nullptr);

        for (auto & p : particles) {
            auto * pos = FML::PARTICLE::GetPos(p);
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] += lattice_shift;
                if (pos[idim] >= 1.0)
                    pos[idim] -= 1.0;
            }
            if constexpr (FML::PARTICLE::has_get_q<P>()) {
                auto * q = FML::PARTICLE::GetLagrangianPos(p);
                for (int idim = 0; idim < NDIM; idim++) {
                    q[idim] += lattice_shift;
                    if (q[idim] >= 1.0)
                        q[idim] -= 1.0;
                }
            }
            if constexpr (FML::PARTICLE::has_set_id<P>())
                FML::PARTICLE::SetID(p, FML::PARTICLE::GetID(p) + id_offset);
        }
        particles.communicate_particles();
        return delta;
    }


    //=============================================================================
    /// The ratios T_i / T_cb and the growth rates f_i of CDM and baryons at zini for the initial conditions with
    /// separate CDM and baryon particles: delta_i(k) = delta_cb(k) R_i(k,zini) with R_i = T_i / T_cb from the
    /// transfer functions (w_c R_c + w_b R_b = 1). The 1LPT velocity uses f_i = f_cb + dlog(R_i)/dlna where f_cb is
    /// the growth rate of the gravity model (so the total CDM+baryon velocity is exactly the growing mode, as for one
    /// species, and the relative velocity comes from the transfer functions). Splines of log(k) (k in h/Mpc).
    //=============================================================================
    struct TwoSpeciesICSplines {
        FML::INTERPOLATION::SPLINE::Spline ratio_cdm, ratio_baryon, f_cdm, f_baryon;
    };
    template <int NDIM>
    TwoSpeciesICSplines two_species_ic_splines(const FML::FILEUTILS::LinearTransferData & transferdata,
                                               const GravityModel<NDIM> & grav_ic,
                                               double zini,
                                               double weight_cdm,
                                               double weight_baryon) {
        using DVector = FML::INTERPOLATION::SPLINE::DVector;
        const double aini = 1.0 / (1.0 + zini);
        if (aini < 1.0 / (1.0 + transferdata.get_zmax_splines()) * (1.0 - 1e-10))
            throw std::runtime_error("particle_baryons: the transfer functions only go up to z = " +
                                     std::to_string(transferdata.get_zmax_splines()) + " (need zini)");

        const double kmin = transferdata.get_kmin_hmpc_splines();
        const double kmax = transferdata.get_kmax_hmpc_splines();
        const int nk = 512;
        DVector logk(nk), ratio_cdm(nk), ratio_baryon(nk), f_cdm(nk), f_baryon(nk);
        double max_err = 0.0;
        // dlog(R_i)/dlna by finite differences over +-dx in log(a) (one-sided at the end of the table)
        const double dx = 0.05;
        const double xini = std::log(aini);
        const double xmin_table = std::log(1.0 / (1.0 + transferdata.get_zmax_splines()));
        const double xlo = std::max(xini - dx, xmin_table), xhi = xini + dx;
        auto ratios = [&](double k, double a) {
            const double Tcb = transferdata.get_cdm_baryon_transfer_function(k, a);
            return std::pair<double, double>(transferdata.get_cdm_transfer_function(k, a) / Tcb,
                                             transferdata.get_baryon_transfer_function(k, a) / Tcb);
        };
        for (int i = 0; i < nk; i++) {
            logk[i] = std::log(kmin) + std::log(kmax / kmin) * i / double(nk - 1);
            const double k = std::exp(logk[i]);
            const auto r = ratios(k, aini), rlo = ratios(k, std::exp(xlo)), rhi = ratios(k, std::exp(xhi));
            ratio_cdm[i] = r.first;
            ratio_baryon[i] = r.second;
            const double fcb = grav_ic.get_f_1LPT(aini, k / grav_ic.H0_hmpc);
            f_cdm[i] = fcb + std::log(rhi.first / rlo.first) / (xhi - xlo);
            f_baryon[i] = fcb + std::log(rhi.second / rlo.second) / (xhi - xlo);
            max_err = std::max(max_err, std::fabs(weight_cdm * ratio_cdm[i] + weight_baryon * ratio_baryon[i] - 1.0));
        }
        TwoSpeciesICSplines s;
        s.ratio_cdm.create(logk, ratio_cdm, "T_cdm/T_cb(k,zini)");
        s.ratio_baryon.create(logk, ratio_baryon, "T_b/T_cb(k,zini)");
        s.f_cdm.create(logk, f_cdm, "f_cdm(k,zini)");
        s.f_baryon.create(logk, f_baryon, "f_b(k,zini)");
        if (FML::ThisTask == 0) {
            std::cout << "# Two species IC at z = " << zini << ": T_b/T_c = ";
            for (double k : {0.01, 0.1, 1.0})
                std::cout << s.ratio_baryon(std::log(k)) / s.ratio_cdm(std::log(k)) << " (k = " << k << ") ";
            std::cout << "\n# f_cdm, f_b at k = 0.1: " << s.f_cdm(std::log(0.1)) << " " << s.f_baryon(std::log(0.1))
                      << ". Max |w_c T_c/T_cb + w_b T_b/T_cb - 1| = " << max_err << "\n";
        }
        if (max_err > 1e-2)
            throw std::runtime_error("particle_baryons: OmegaCDM and Omegab do not match the transfer functions "
                                     "(w_c T_c + w_b T_b != T_cb)");
        return s;
    }

    //=============================================================================
    /// The mass contrast of particle_baryon_relative_mode = mass: eps_i(k) = (T_i - T_cb)(k,z=0) / T_cb(k,zini), the
    /// linear baryon-CDM relative mode today per unit delta_cb(zini) (Hahn, Rampf & Uhlemann 2021). Splines of log(k)
    /// (k in h/Mpc) for CDM and baryons (w_c eps_c + w_b eps_b = 0)
    //=============================================================================
    inline std::pair<FML::INTERPOLATION::SPLINE::Spline, FML::INTERPOLATION::SPLINE::Spline>
    two_species_mass_contrast_today(const FML::FILEUTILS::LinearTransferData & transferdata, double zini) {
        using DVector = FML::INTERPOLATION::SPLINE::DVector;
        const double aini = 1.0 / (1.0 + zini);
        const double kmin = transferdata.get_kmin_hmpc_splines();
        const double kmax = transferdata.get_kmax_hmpc_splines();
        const int nk = 512;
        DVector logk(nk), eps_cdm(nk), eps_baryon(nk);
        for (int i = 0; i < nk; i++) {
            logk[i] = std::log(kmin) + std::log(kmax / kmin) * i / double(nk - 1);
            const double k = std::exp(logk[i]);
            const double Tcb_ini = transferdata.get_cdm_baryon_transfer_function(k, aini);
            const double Tcb_today = transferdata.get_cdm_baryon_transfer_function(k, 1.0);
            eps_cdm[i] = (transferdata.get_cdm_transfer_function(k, 1.0) - Tcb_today) / Tcb_ini;
            eps_baryon[i] = (transferdata.get_baryon_transfer_function(k, 1.0) - Tcb_today) / Tcb_ini;
        }
        FML::INTERPOLATION::SPLINE::Spline sc(logk, eps_cdm, "(T_cdm-T_cb)(k,z=0)/T_cb(k,zini)");
        FML::INTERPOLATION::SPLINE::Spline sb(logk, eps_baryon, "(T_b-T_cb)(k,z=0)/T_cb(k,zini)");
        return {sc, sb};
    }

} // namespace COLASOLVER

#endif
