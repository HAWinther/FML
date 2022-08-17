#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FriendsOfFriends/FoF.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParticleTypes/SimpleParticle.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/Spline/Spline.h>

//=====================================================
//
// This example shows how to make a very simple PM N-body solver:
// Initial conditions (1LPT, 2LPT or 3LPT) from a given P(k) to get either a gaussian or a non-local non-gaussian
// initial density field (with or without the amplitude fixing trick) Computing forces (using any grid-assignment order
// you want) and using this to do time-stepping. This is done with a second order leapfrog (though 4th order is also
// availiable)
//
// We also implement the COLA method on top of this with a choice of what LPT order (1, 2 or 3) you want to use.
//
// With the result we compute:
// * Power-spectrum
// * Redshift space multipoles P_ell(k) for ell=0,2,4
// * Bispectrum
// * A FoF halo catalogue
//
// This is just meant to illustrate how one can do this so its not the cleanest code below and all parameters
// are just set as global variables below
//
//=====================================================

//=====================================================
// Parameters for the simulation
//=====================================================

const int NDIM = 3;                                  // Number of dimensions we work in
const double box = 1000.0;                           // Boxsizein Mpc/h
const double zini = 20.0;                            // Starting redshift
const double aini = 1.0 / (1.0 + zini);              // Starting scalefactor
const double aend = 1.0;                             // Final scalefactor
const int nsteps = 20;                               // Number of time-steps equidistributed in a
const bool fix_amplitude = true;                     // Amplitude fixed "gaussian" random field for IC
const std::string type_of_random_field = "gaussian"; // Type of random field: gaussian, local, equilateral, orthogonal
const double fNL = 100.0;                            // If non-local non-gaussianity the value of fNL
const unsigned int random_seed = 1234;               // A random seed
const int Nmesh = 256;                               // Grid-size for binning particles to a grid and computing forces
const int Nmesh_IC = 128;                            // Grid-size for making the IC
const int Npart_1D = 128;                            // Number of particles per dimension
const double buffer_factor = 1.5;                    // Extra allocation factor for particles
const bool COLA = true;                              // Use the COLA method. We use the LPT order we are able to
const int LPT_order_IC = 3;                          // The LPT order to use when making the ICi
                                                     //
const int n_min_FoF_group = 20;                      // Minimum number of particles per halo
const int Ngrid_max_FoF = 256; // Maximum gridsize for speeding up FoF linking (bigger better, but more memory)
const double FoFBoundarySizeMpch =
    3.0; // Size of boundary we communicate to do FoF linking (should be bigger than largest halo expected)
const double linking_length = 0.2;              // Halo linking length
                                                //
const int nbin_bispectrum = 16;                 // Number of bins for computing the bispectrum
const double apofk = 1.0;                       // Redshift for which P(k) is at
const std::string filename = "pofk.txt";        // File with k (h/Mpc), P(k)  (Mpc/h)^3
const std::string interpolation_method = "CIC"; // The grid-assignment method
const double OmegaM = 1.0;                      // The matter density parameter at z = 0
const double h = 0.7;                           // Hubble parameter h = H0/(100km/s/Mpc) at z = 0
const double A_s = 2e-9;                        // Primordial amplitude
const double n_s = 0.96;                        // Primordial spectral index
const double kpivot = 0.05 / h;                 // Pivot scale in h/Mpc

FML::INTERPOLATION::SPLINE::Spline D_1LPT_of_loga;
FML::INTERPOLATION::SPLINE::Spline D_2LPT_of_loga;
FML::INTERPOLATION::SPLINE::Spline D_3LPTa_of_loga;
FML::INTERPOLATION::SPLINE::Spline D_3LPTb_of_loga;

double HoverH0(double a) { return std::sqrt(OmegaM / (a * a * a) + 1.0 - OmegaM); };

// Type aliases for easier use below
template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;
using Particle = COLAParticle<NDIM>;
using DVector = FML::SOLVERS::ODESOLVER::DVector;
using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
using FoFHalo = FML::FOF::FoFHalo<Particle, NDIM>;

int main() {

    //========================================================
    // If COLA then the particles must have get_D_nLPT method(s)
    // This determines the LPT order we use with COLA
    // The IC method will store the relevant stuff in the particles
    //========================================================
    int LPT_order_COLA = 0;
    if (COLA) {
        if constexpr (FML::PARTICLE::has_get_D_1LPT<Particle>())
            LPT_order_COLA = 1;
        if constexpr (FML::PARTICLE::has_get_D_2LPT<Particle>())
            LPT_order_COLA = 2;
        if constexpr (FML::PARTICLE::has_get_D_3LPTa<Particle>() or FML::PARTICLE::has_get_D_3LPTb<Particle>())
            LPT_order_COLA = 3;
        LPT_order_COLA = std::min(LPT_order_COLA, LPT_order_IC);
    }

    //========================================================
    // We need a spline of the LPT growth factor(s)
    // Normalized such that D1LPT(zini) = 1
    //========================================================
    FML::COSMOLOGY::LPT::compute_LPT_growth_factors_LCDM(
        OmegaM, zini, HoverH0, D_1LPT_of_loga, D_2LPT_of_loga, D_3LPTa_of_loga, D_3LPTb_of_loga);

    //============================================================
    // Read k, P(k) (File has units of h/Mpc, (Mpc/h)^3 but what matters is that it has the same
    // units as the boxsize)
    //============================================================
    std::ifstream fp(filename.c_str());
    if (!fp.is_open())
        throw std::runtime_error("Cannot open file: " + filename);
    std::vector<double> logk;
    std::vector<double> logpofk;
    for (;;) {
        double kin, pofkin;
        fp >> kin;
        if (fp.eof())
            break;
        fp >> pofkin;
        logk.push_back(std::log(kin));
        logpofk.push_back(std::log(pofkin));
    }
    // Make a spline and the P(k) function needed below
    FML::INTERPOLATION::SPLINE::Spline logpofk_spline(logk, logpofk);

    //============================================================
    // Power spectrum scaled back to the initial time using growth-factors
    // For non-gaussian initial conditions we also need the primordial
    // power-spectrum
    //============================================================
    const double DiniOverD = D_1LPT_of_loga(std::log(aini)) / D_1LPT_of_loga(std::log(apofk));
    auto Pofk_of_kBox_over_volume = [&](double kBox) {
        const double logk = std::log(kBox / box);
        const double volume = std::pow(box, NDIM);
        return DiniOverD * DiniOverD * std::exp(logpofk_spline(logk)) / volume;
    };
    auto Pofk_of_kBox_over_volume_primordial = [&](double kBox) {
        const double k = kBox / box;
        return 2.0 * M_PI * M_PI / (kBox * kBox * kBox) * A_s * std::pow(k / kpivot, n_s - 1.0);
    };
    auto Pofk_of_kBox_over_Pofk_primordial = [&](double kBox) {
        return Pofk_of_kBox_over_volume(kBox) / Pofk_of_kBox_over_volume_primordial(kBox);
    };

    //============================================================
    // Hubble function and growth-rate(s) needed to compute IC
    //============================================================
    auto H_over_H0_of_loga = [&](double loga) { return HoverH0(std::exp(loga)); };
    std::function<double(double)> growth_rate_f_1LPT_of_loga = [&](double loga) {
        return D_1LPT_of_loga.deriv_x(loga) / D_1LPT_of_loga(loga);
    };
    std::function<double(double)> growth_rate_f_2LPT_of_loga = [&](double loga) {
        return D_2LPT_of_loga.deriv_x(loga) / D_2LPT_of_loga(loga);
    };
    std::function<double(double)> growth_rate_f_3LPTa_of_loga = [&](double loga) {
        return D_3LPTa_of_loga.deriv_x(loga) / D_3LPTa_of_loga(loga);
    };
    std::function<double(double)> growth_rate_f_3LPTb_of_loga = [&](double loga) {
        return D_3LPTb_of_loga.deriv_x(loga) / D_3LPTb_of_loga(loga);
    };
    std::vector<std::function<double(double)>> growth_rate_f_of_loga;
    growth_rate_f_of_loga.push_back(growth_rate_f_1LPT_of_loga);
    growth_rate_f_of_loga.push_back(growth_rate_f_2LPT_of_loga);
    growth_rate_f_of_loga.push_back(growth_rate_f_3LPTa_of_loga);
    growth_rate_f_of_loga.push_back(growth_rate_f_3LPTb_of_loga);

    //============================================================
    // Set up random number generator
    //============================================================
    std::shared_ptr<FML::RANDOM::RandomGenerator> rng = std::make_shared<FML::RANDOM::RandomGenerator>();
    rng->set_seed(random_seed);

    //============================================================
    // Generate initial conditions
    //============================================================

    std::vector<double> velocity_norms(4);
    for (int i = 0; i < 4; i++) {
        velocity_norms[i] =
            100.0 * box * H_over_H0_of_loga(std::log(aini)) * aini * growth_rate_f_of_loga[i](std::log(aini));
    }

    FML::PARTICLE::MPIParticles<Particle> part;
    FML::NBODY::NBodyInitialConditions<NDIM, Particle>(part,
                                                       Npart_1D,
                                                       buffer_factor,
                                                       Nmesh_IC,
                                                       fix_amplitude,
                                                       rng.get(),
                                                       Pofk_of_kBox_over_Pofk_primordial,
                                                       Pofk_of_kBox_over_volume_primordial,
                                                       LPT_order_IC,
                                                       type_of_random_field,
                                                       fNL,
                                                       box,
                                                       zini,
                                                       velocity_norms);

    //============================================================
    // In the COLA frame v=0 so reset velocities
    //============================================================
    if (COLA) {
        // Loop over all active particles
        for (auto & p : part) {
            auto * vel = FML::PARTICLE::GetVel(p);
            for (int idim = 0; idim < NDIM; idim++) {
                vel[idim] = 0.0;
            }
        }
    }

    //============================================================
    // Instead of stepping in dt_code = dt H0/a^2 we have unit time-steps in a. Here we
    // compute the integral of dt_code/da over one time-step
    //============================================================

    // The prefactor to delta in the Poisson equation
    auto poisson_factor = [&](double a) { return 1.5 * OmegaM * a; };

    auto compute_vel_timestep = [&](double alow, double ahigh) {
        // For the kick step we have an extra 'a' coming from the force via the Poisson equation
        const double amid = (ahigh + alow) / 2.;
        ODEFunction deriv = [&](double a, [[maybe_unused]] const double * t, double * dtda) {
            dtda[0] = 1.0 / (a * a * a * HoverH0(a)) * poisson_factor(a) / poisson_factor(amid);
            return GSL_SUCCESS;
        };

        // Solve the integral
        DVector tini{0.0};
        DVector avec{alow, ahigh};
        ODESolver ode;
        ode.solve(deriv, avec, tini);
        return ode.get_final_data()[0];
    };

    auto compute_pos_timestep = [&](double alow, double ahigh) {
        ODEFunction deriv = [&](double a, [[maybe_unused]] const double * t, double * dtda) {
            dtda[0] = 1.0 / (a * a * a * HoverH0(a));
            return GSL_SUCCESS;
        };
        // Solve the integral
        DVector tini{0.0};
        DVector avec{alow, ahigh};
        ODESolver ode;
        ode.solve(deriv, avec, tini);
        return ode.get_final_data()[0];
    };

    //============================================================
    // Main timestepping loop
    //============================================================
    for (int i = 0; i <= nsteps; i++) {

        // Scale factor for velocity and positions. Equidistance steps in 'a'
        const double delta_aexp = (aend - aini) / double(nsteps);
        const double apos_old = aini + delta_aexp * i;
        const double apos_new = (i == nsteps) ? aend : apos_old + delta_aexp;
        const double avel_old = (i == 0) ? aini : apos_old - 0.5 * delta_aexp;
        const double avel_new = (i == 0) ? aini + 0.5 * delta_aexp : (i == nsteps ? aend : apos_old + 0.5 * delta_aexp);

        // Timesteps: first step we displace pos and vel by 1/2 timestep to do leapfrog
        // Last time-step we synchronize back by only moving vel
        const double delta_time_pos = compute_pos_timestep(apos_old, apos_new);
        const double delta_time_vel = compute_vel_timestep(avel_old, avel_new);
        const double norm_poisson_equation = poisson_factor(apos_old);

        // Particles -> density field(x) -> density fielk(k)
        auto nleftright = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(interpolation_method);
        FFTWGrid<NDIM> density_grid_fourier(Nmesh, nleftright.first, nleftright.second);
        FML::INTERPOLATION::particles_to_grid(part.get_particles_ptr(),
                                              part.get_npart(),
                                              part.get_npart_total(),
                                              density_grid_fourier,
                                              interpolation_method);
        density_grid_fourier.fftw_r2c();

        // Here we can bin up a power-spectrum if we want (its basically free)
        // Here we can also add contribution from radiation on large scales and linear massless neutrinos etc...

        // Density field -> force
        std::array<FFTWGrid<NDIM>, NDIM> force_real;
        FML::NBODY::compute_force_from_density_fourier<NDIM>(
            density_grid_fourier, force_real, interpolation_method, norm_poisson_equation);

        // Update velocity of particles (frees force)
        FML::NBODY::KickParticles<NDIM>(force_real, part, delta_time_vel, interpolation_method);

        // Add on COLA displacement and velocity
        if (COLA) {
            const double D1new = D_1LPT_of_loga(std::log(apos_new));
            const double D1old = D_1LPT_of_loga(std::log(apos_old));
            const double D1ini = D_1LPT_of_loga(std::log(aini));
            const double fac1_pos = (D1new - D1old) / D1ini;
            const double fac1_vel = -norm_poisson_equation * D1old / D1ini * delta_time_vel;

            const double D2new = D_2LPT_of_loga(std::log(apos_new));
            const double D2old = D_2LPT_of_loga(std::log(apos_old));
            const double D2ini = D_2LPT_of_loga(std::log(aini));
            const double fac2_pos = (D2new - D2old) / D2ini;
            const double fac2_vel = -norm_poisson_equation * (D2old - D1old * D1old) / D2ini * delta_time_vel;

            const double D3anew = D_3LPTa_of_loga(std::log(apos_new));
            const double D3aold = D_3LPTa_of_loga(std::log(apos_old));
            const double D3aini = D_3LPTa_of_loga(std::log(aini));
            const double fac3a_pos = (D3anew - D3aold) / D3aini;
            const double fac3a_vel =
                -norm_poisson_equation * (D3aold - 2.0 * D1old * D1old * D1old) / D3aini * delta_time_vel;

            const double D3bnew = D_3LPTb_of_loga(std::log(apos_new));
            const double D3bold = D_3LPTb_of_loga(std::log(apos_old));
            const double D3bini = D_3LPTb_of_loga(std::log(aini));
            const double fac3b_pos = (D3bnew - D3bold) / D3bini;
            const double fac3b_vel =
                -norm_poisson_equation * (D3bold + D1old * D1old * D1old - D1old * D2old) / D3bini * delta_time_vel;

            // Loop over all active particles
            for (auto & p : part) {
                auto * pos = FML::PARTICLE::GetPos(p);
                auto * vel = FML::PARTICLE::GetVel(p);

                if (LPT_order_COLA >= 1) {
                    auto * D1 = FML::PARTICLE::GetD_1LPT(p);
                    for (int idim = 0; idim < NDIM; idim++) {
                        pos[idim] += D1[idim] * fac1_pos;
                        vel[idim] += D1[idim] * fac1_vel;
                    }
                }

                if (LPT_order_COLA >= 2) {
                    auto * D2 = FML::PARTICLE::GetD_2LPT(p);
                    for (int idim = 0; idim < NDIM; idim++) {
                        pos[idim] += D2[idim] * fac2_pos;
                        vel[idim] += D2[idim] * fac2_vel;
                    }
                }

                if constexpr (FML::PARTICLE::has_get_D_3LPTa<Particle>())
                    if (LPT_order_COLA >= 3) {
                        auto * D3a = FML::PARTICLE::GetD_3LPTa(p);
                        for (int idim = 0; idim < NDIM; idim++) {
                            pos[idim] += D3a[idim] * fac3a_pos;
                            vel[idim] += D3a[idim] * fac3a_vel;
                        }
                    }

                if constexpr (FML::PARTICLE::has_get_D_3LPTb<Particle>())
                    if (LPT_order_COLA >= 3) {
                        auto * D3b = FML::PARTICLE::GetD_3LPTb(p);
                        for (int idim = 0; idim < NDIM; idim++) {
                            pos[idim] += D3b[idim] * fac3b_pos;
                            vel[idim] += D3b[idim] * fac3b_vel;
                        }
                    }
            }
        }

        // Move particles (this does communication)
        FML::NBODY::DriftParticles<NDIM, Particle>(part, delta_time_pos);

        if (FML::ThisTask == 0)
            std::cout << i << " / " << nsteps << " Positions at a: " << apos_new << " Velocities at a: " << avel_new
                      << "\n";

        // Print info
        part.info();
    }

    //============================================================
    // For COLA we must add on the LPT velocity as we solve in
    // the COLA frame
    //============================================================

    if (COLA) {
        const double D1end = D_1LPT_of_loga(std::log(aend));
        const double D1ini = D_1LPT_of_loga(std::log(aini));
        const double D2end = D_2LPT_of_loga(std::log(aend));
        const double D2ini = D_2LPT_of_loga(std::log(aini));
        const double D3aend = D_3LPTa_of_loga(std::log(aend));
        const double D3aini = D_3LPTa_of_loga(std::log(aini));
        const double D3bend = D_3LPTb_of_loga(std::log(aend));
        const double D3bini = D_3LPTb_of_loga(std::log(aini));
        const double vfac_1LPT =
            D1end / D1ini * growth_rate_f_1LPT_of_loga(std::log(aend)) * aend * aend * HoverH0(aend);
        const double vfac_2LPT =
            D2end / D2ini * growth_rate_f_2LPT_of_loga(std::log(aend)) * aend * aend * HoverH0(aend);
        const double vfac_3LPTa =
            D3aend / D3aini * growth_rate_f_3LPTa_of_loga(std::log(aend)) * aend * aend * HoverH0(aend);
        const double vfac_3LPTb =
            D3bend / D3bini * growth_rate_f_3LPTb_of_loga(std::log(aend)) * aend * aend * HoverH0(aend);

        // Here we add on the COLA velocity kick
        for (auto & p : part) {
            auto * vel = FML::PARTICLE::GetVel(p);

            if constexpr (FML::PARTICLE::has_get_D_1LPT<Particle>())
                if (LPT_order_COLA >= 1) {
                    auto * D1 = FML::PARTICLE::GetD_1LPT(p);
                    for (int idim = 0; idim < NDIM; idim++) {
                        vel[idim] += D1[idim] * vfac_1LPT;
                    }
                }

            if constexpr (FML::PARTICLE::has_get_D_2LPT<Particle>())
                if (LPT_order_COLA >= 2) {
                    auto * D2 = FML::PARTICLE::GetD_2LPT(p);
                    for (int idim = 0; idim < NDIM; idim++) {
                        vel[idim] += D2[idim] * vfac_2LPT;
                    }
                }

            if constexpr (FML::PARTICLE::has_get_D_3LPTa<Particle>())
                if (LPT_order_COLA >= 3) {
                    auto * D3a = FML::PARTICLE::GetD_3LPTa(p);
                    for (int idim = 0; idim < NDIM; idim++) {
                        vel[idim] += D3a[idim] * vfac_3LPTa;
                    }
                }

            if constexpr (FML::PARTICLE::has_get_D_3LPTb<Particle>())
                if (LPT_order_COLA >= 3) {
                    auto * D3b = FML::PARTICLE::GetD_3LPTb(p);
                    for (int idim = 0; idim < NDIM; idim++) {
                        vel[idim] += D3b[idim] * vfac_3LPTb;
                    }
                }
        }
    }

    //============================================================
    // Compute power-spectrum at final redshift
    //============================================================

    if (FML::ThisTask == 0) {
        std::cout << "Compute power spectrum\n";
    }

    const bool interlacing = true;
    const std::string density_assignment = "CIC";
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> pofk(Nmesh / 2);
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum<NDIM>(Nmesh,
                                                            part.get_particles_ptr(),
                                                            part.get_npart(),
                                                            part.get_npart_total(),
                                                            pofk,
                                                            density_assignment,
                                                            interlacing);
    pofk.scale(box);
    if (FML::ThisTask == 0) {
        std::ofstream fp("pofk_out.txt");
        fp << "\n# k   (h/Mpc)      P(k)  (Mpc/h)^3\n";
        for (int i = 0; i < pofk.n; i++) {
            fp << pofk.kbin[i] << " " << pofk.pofk[i] << "\n";
        }
    }

    //============================================================
    // Compute power-spectrum multipoles at final redshift
    //============================================================

    // Factor to convert from velocity v_code = a^2 dxdt / (H0 Box) to the RSD shift (dxdt / H) in units of the box
    const double velocity_to_displacement = 1.0 / (aend * aend * HoverH0(aend));
    const int ellmax = 4;
    std::vector<FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>> Pells(1 + ellmax, Nmesh / 2);
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum_multipoles<NDIM>(
        Nmesh, part, velocity_to_displacement, Pells, density_assignment, interlacing);

    // To physical units
    for (size_t ell = 0; ell < Pells.size(); ell++) {
        Pells[ell].scale(box);
    }

    // Output: k, P0, P0_kaiser P2, P2_kaiser, P4, P4_kaiser
    if (FML::ThisTask == 0) {
        FML::assert_mpi(Pells[0].n == pofk.n, "We require P(k) and Pell(k) to have the same bins");
        std::ofstream fp("pofk_multipoles_out.txt");
        const double f = growth_rate_f_1LPT_of_loga(std::log(aend));
        for (int i = 0; i < Pells[0].n; i++) {
            fp << Pells[0].k[i] << " ";
            for (size_t ell = 0; ell < Pells.size(); ell += 2) {
                fp << Pells[ell].pofk[i] << " ";

                // Output the Kaiser limit relation between P(k) and P_ell(k)
                if (ell == 0)
                    fp << pofk.pofk[i] * (1.0 + 2. * f / 3. + f * f / 5.) << " ";
                if (ell == 2)
                    fp << pofk.pofk[i] * (4. * f / 3. + 4. * f * f / 7.) << " ";
                if (ell == 4)
                    fp << pofk.pofk[i] * (8. * f * f / 35.) << " ";
            }
            fp << "\n";
        }
        fp << "\n";
    }

    //============================================================
    // Compute bispectrum
    //============================================================
    FML::CORRELATIONFUNCTIONS::BispectrumBinning<NDIM> bofk(
        0.0,
        2.0 * M_PI * Nmesh / 2,
        nbin_bispectrum,
        FML::CORRELATIONFUNCTIONS::BispectrumBinning<NDIM>::LINEAR_SPACING);
    FML::CORRELATIONFUNCTIONS::compute_bispectrum(Nmesh,
                                                  part.get_particles_ptr(),
                                                  part.get_npart(),
                                                  part.get_npart_total(),
                                                  bofk,
                                                  density_assignment,
                                                  interlacing);
    bofk.scale(box);

    if (FML::ThisTask == 0) {
        std::ofstream fp("bofk_out.txt");
        fp << "# k1         k2         k3         B123         Q123         N123\n";
        for (int i = 0; i < nbin_bispectrum; i++) {
            for (int j = 0; j < nbin_bispectrum; j++) {
                for (int k = 0; k < nbin_bispectrum; k++) {
                    double k1 = bofk.kmean[i];
                    double k2 = bofk.kmean[j];
                    double k3 = bofk.kmean[k];
                    double B123 = bofk.get_spectrum(i, j, k);
                    double Q123 = bofk.get_reduced_spectrum(i, j, k);
                    double N123 = bofk.get_bincount(i, j, k);
                    fp << k1 << " " << k2 << " " << k3 << " " << B123 << " " << Q123 << " " << N123 << "\n";
                }
            }
        }
    }

    //============================================================
    // Locate halos
    //============================================================

    part.communicate_particles();
    std::vector<FoFHalo> FoFGroups;
    double bufferlength = FoFBoundarySizeMpch / box;
    FML::FOF::FriendsOfFriends<Particle, NDIM>(part.get_particles_ptr(),
                                               part.get_npart(),
                                               linking_length,
                                               n_min_FoF_group,
                                               true,
                                               bufferlength,
                                               FoFGroups,
                                               Ngrid_max_FoF);

    // Output halos task by task
    for (int i = 0; i < FML::NTasks; i++) {
        if (i == FML::ThisTask) {
            std::cout << "\n# Found " << FoFGroups.size() << " halos on task " << FML::ThisTask << "\n";
            std::ofstream fp("halo_out.txt");
            for (auto & g : FoFGroups) {
                if (g.np > 0) {
                    fp << g.np << " ";
                    for (int idim = 0; idim < NDIM; idim++)
                        fp << g.pos[idim] * box << " ";
                    fp << "\n";
                }
            }
        }
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

#ifdef MEMORY_LOGGING
    // Show full info about allocations we as function of time
    FML::MemoryLog::get()->print();
#endif
}
