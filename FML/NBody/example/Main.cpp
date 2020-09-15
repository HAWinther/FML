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
// This example shows how to make a very simple PM N-body
// The only input is a power-spectrum
//
// We also implement the (1LPT or 2LPT) COLA method (just a few more lines)
// and how to compute P(k) and a halo catlogue from
// the final result.
//
// This is just meant to illustrate how one can do this
//
//=====================================================

//=====================================================
// Parameters for the simulation
//=====================================================

const int Ndim = 3;
const double box = 1000.0;
const double zini = 20.0;
const double aini = 1.0 / (1.0 + zini);
const double aend = 1.0;
const int nsteps = 10;
const bool fix_amplitude = true;
const unsigned int random_seed = 1234;
const double OmegaM = 1.0;
const int Nmesh = 256;
const int Npart_1D = 128;
const double buffer_factor = 2;
const bool COLA = true;
const int LPT_order = 3;
const double apofk = 1.0;
const int n_min_FoF_group = 20;
const double fof_distance = 0.2 / double(Npart_1D);
const std::string filename = "pofk.txt";
const std::string interpolation_method = "CIC";
FML::INTERPOLATION::SPLINE::Spline D_1LPT_of_loga;
FML::INTERPOLATION::SPLINE::Spline D_2LPT_of_loga;
auto HoverH0 = [&](double a) { return std::sqrt(OmegaM / (a * a * a) + 1.0 - OmegaM); };

// Type aliases for easier use below
template <class T>
using MPIParticles = FML::PARTICLE::MPIParticles<T>;
template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;
using Particle = COLAParticle<Ndim>;
using DVector = FML::SOLVERS::ODESOLVER::DVector;
using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
using FoFHalo = FML::FOF::FoFHalo<Particle, Ndim>;

int main() {

    // If COLA then the particles must have get_D_nLPT method(s)
    if (COLA) {
        if (LPT_order >= 1)
            FML::assert_mpi(FML::PARTICLE::has_get_D_1LPT<Particle>(),
                            "For 1LPT COLA particles must have get_D_1LPT method");
        if (LPT_order >= 2)
            FML::assert_mpi(FML::PARTICLE::has_get_D_2LPT<Particle>(),
                            "For 2LPT COLA particles must have get_D_2LPT method");
    }

    //========================================================
    // We need the 1LPT and 2LPT growth factor(s)
    //========================================================
    auto compute_growth_factor = [&]() {
        const int npts = 1000;
        const double aini = std::min(0.01, 1.0 / (1.0 + zini));
        const double aend = 1.0;

        // The ODE ddDddx + ( 2 + H'/ ) dDdx = 1.5 OmegaM / (H^2 a^3) D
        ODEFunction deriv = [&](double x, const double * y, double * dydx) {
            const double a = std::exp(x);
            const double H = HoverH0(a);
            const double dlogHdx = 1.0 / (2.0 * H * H) * (-3.0 * OmegaM / (a * a * a));
            const double D1 = y[0];
            const double dD1dx = y[1];
            const double D2 = y[2];
            const double dD2dx = y[3];
            dydx[0] = dD1dx;
            dydx[1] = 1.5 * OmegaM / (H * H * a * a * a) * D1 - (2.0 + dlogHdx) * dD1dx;
            dydx[2] = dD2dx;
            dydx[3] = 1.5 * OmegaM / (H * H * a * a * a) * (D2 - D1 * D1) - (2.0 + dlogHdx) * dD2dx;
            return GSL_SUCCESS;
        };

        // The initial conditions D1 ~ a and D2 = -3/7D1^2  for growing mode in EdS
        DVector yini{1.0, 1.0, -3.0 / 7.0, -6.0 / 7.0};
        DVector xarr(npts);
        for (int i = 0; i < npts; i++)
            xarr[i] = std::log(aini) + std::log(aend / aini) * i / double(npts);

        // Solve the ODE
        ODESolver ode;
        ode.solve(deriv, xarr, yini);
        auto D1 = ode.get_data_by_component(0);
        auto D2 = ode.get_data_by_component(2);

        // Spline it up
        D_1LPT_of_loga.create(xarr, D1, "D1(loga) Spline");
        D_2LPT_of_loga.create(xarr, D2, "D2(loga) Spline");
    };
    compute_growth_factor();

    //============================================================
    // Read P(k) from file
    //============================================================

    // Read k, P(k) (File has units of h/Mpc, (Mpc/h)^3 but what matters is that it has the same
    // units as the boxsize)
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
    //============================================================
    const double DiniOverD = D_1LPT_of_loga(std::log(aini)) / D_1LPT_of_loga(std::log(apofk));
    std::function<double(double)> Pofk_of_kBox_over_volume = [&](double kBox) {
        const double logk = std::log(kBox / box);
        const double volume = std::pow(box, Ndim);
        return DiniOverD * DiniOverD * std::exp(logpofk_spline(logk)) / volume;
    };

    //============================================================
    // Hubble function and growth-rate needed to compute IC
    //============================================================
    std::function<double(double)> H_over_H0_of_loga = [&](double loga) { return HoverH0(std::exp(loga)); };
    std::function<double(double)> growth_rate_f_1LPT_of_loga = [&](double loga) {
        return D_1LPT_of_loga.deriv_x(loga) / D_1LPT_of_loga(loga);
    };
    std::function<double(double)> growth_rate_f_2LPT_of_loga = [&](double loga) {
        return D_2LPT_of_loga.deriv_x(loga) / D_2LPT_of_loga(loga);
    };
    std::function<double(double)> growth_rate_f_3LPT_of_loga = [&](double loga) {
        return 3.0 * D_1LPT_of_loga.deriv_x(loga) / D_1LPT_of_loga(loga);
    };
    std::vector<std::function<double(double)>> growth_rate_f_of_loga;
    growth_rate_f_of_loga.push_back(growth_rate_f_1LPT_of_loga);
    growth_rate_f_of_loga.push_back(growth_rate_f_2LPT_of_loga);
    growth_rate_f_of_loga.push_back(growth_rate_f_3LPT_of_loga);

    //============================================================
    // Set up random number generator
    //============================================================
    std::shared_ptr<FML::RANDOM::RandomGenerator> rng = std::make_shared<FML::RANDOM::RandomGenerator>();
    rng->set_seed(random_seed);

    //============================================================
    // Generate initial conditions
    //============================================================
    FML::PARTICLE::MPIParticles<Particle> part;
    FML::NBODY::NBodyInitialConditions<Ndim, Particle>(part,
                                                       Npart_1D,
                                                       buffer_factor,
                                                       Nmesh,
                                                       fix_amplitude,
                                                       rng.get(),
                                                       Pofk_of_kBox_over_volume,
                                                       LPT_order,
                                                       box,
                                                       zini,
                                                       H_over_H0_of_loga,
                                                       growth_rate_f_of_loga);

    // In the COLA frame v=0 so reset velocities
    if (COLA) {
        for (auto & p : part.get_particles()) {
            auto * vel = FML::PARTICLE::GetVel(p);
            for (int idim = 0; idim < Ndim; idim++) {
                vel[idim] = 0.0;
            }
        }
    }

    //============================================================
    // Instead of stepping in dt_code = dt H0/a^2 we have unit time-steps in a. Here we
    // compute the integral of dt_code/da over one time-step
    //============================================================
    auto compute_timestep = [&](double alow, double ahigh, bool kick_step) {
        ODEFunction deriv = [&](double a, [[maybe_unused]] const double * t, double * dtda) {
            // For the kick step we have an extra 'a' coming from the Poisson equation
            const double aa = kick_step ? (ahigh + alow) / 2. : a;
            dtda[0] = 1.0 / (a * a * aa * HoverH0(a));
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
    for (int i = 0; i < nsteps; i++) {

        // Scale factor for velocity and positions. Equidistance steps in 'a'
        const double delta_aexp = (aend - aini) / double(nsteps);
        const double apos_old = aini + delta_aexp * i;
        const double apos_new = apos_old + delta_aexp;
        const double avel_old = i == 0 ? aini : apos_old - 0.5 * delta_aexp;
        const double avel_new = i == 0 ? aini + 0.5 * delta_aexp : apos_old + 0.5 * delta_aexp;

        // Timestep: first step we displace pos and vel by 1/2 timestep to do leapfrog
        const double delta_time_pos = compute_timestep(apos_old, apos_new, false);
        const double delta_time_vel = compute_timestep(avel_old, avel_new, true);
        const double norm_poisson_equation = 1.5 * OmegaM * apos_old;

        // Particles -> density field
        auto nleftright = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(interpolation_method);
        FFTWGrid<Ndim> density_grid_real(Nmesh, nleftright.first, nleftright.second);
        FML::INTERPOLATION::particles_to_grid<Ndim, Particle>(part.get_particles().data(),
                                                              part.get_npart(),
                                                              part.get_npart_total(),
                                                              density_grid_real,
                                                              interpolation_method);

        // Density field -> force
        std::array<FFTWGrid<Ndim>, Ndim> force_real;
        FML::NBODY::compute_force_from_density(density_grid_real, force_real, norm_poisson_equation);

        // Update velocity of particles (frees force)
        FML::NBODY::KickParticles(force_real, part, delta_time_vel, interpolation_method);

        // Add on COLA displacement and velocity
        if (COLA) {
            const double D1new = D_1LPT_of_loga(std::log(apos_new));
            const double D1old = D_1LPT_of_loga(std::log(apos_old));
            const double D1ini = D_1LPT_of_loga(std::log(aini));
            const double fac1_pos = (D1new - D1old) / D1ini;
            const double fac1_vel = -1.5 * OmegaM * apos_old * D1old / D1ini * delta_time_vel;

            const double D2new = D_2LPT_of_loga(std::log(apos_new));
            const double D2old = D_2LPT_of_loga(std::log(apos_old));
            const double D2ini = D_2LPT_of_loga(std::log(aini));
            const double fac2_pos = (D2new - D2old) / D2ini;
            const double fac2_vel = -1.5 * OmegaM * apos_old * (D2old - D1old * D1old) / D2ini * delta_time_vel;

            for (auto & p : part.get_particles()) {
                auto * pos = FML::PARTICLE::GetPos(p);
                auto * vel = FML::PARTICLE::GetVel(p);

                if (LPT_order >= 1) {
                    auto * D1 = FML::PARTICLE::GetD_1LPT(p);
                    for (int idim = 0; idim < Ndim; idim++) {
                        pos[idim] += D1[idim] * fac1_pos;
                        vel[idim] += D1[idim] * fac1_vel;
                    }
                }

                if (LPT_order >= 2) {
                    auto * D2 = FML::PARTICLE::GetD_2LPT(p);
                    for (int idim = 0; idim < Ndim; idim++) {
                        pos[idim] += D2[idim] * fac2_pos;
                        vel[idim] += D2[idim] * fac2_vel;
                    }
                }
            }

            // Print info
            part.info();
        }

        // Move particles (this does communication)
        FML::NBODY::DriftParticles<Ndim, Particle>(part, delta_time_pos);

        if (FML::ThisTask == 0)
            std::cout << i + 1 << " / " << nsteps << " Positions at a: " << apos_new << " Velocities at a: " << avel_new
                      << "\n";
    }
    // NB: velocities are here da/2 out of sync with positions (we don't bother fixing this in this example)

    if (FML::ThisTask == 0) {
        std::cout << "Compute power spectrum\n";
    }

    // Compute power-spectrum at final redshift
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<Ndim> pofk(Nmesh / 2);
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum_interlacing<Ndim>(
        Nmesh, part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), pofk, "CIC");
    pofk.scale(box);
    if (FML::ThisTask == 0) {
        std::ofstream fp("pofk_out.txt");
        fp << "\n# k   (h/Mpc)      P(k)  (Mpc/h)^3\n";
        for (int i = 0; i < pofk.n; i++) {
            fp << pofk.kbin[i] << " " << pofk.pofk[i] << "\n";
        }
    }

    if (FML::ThisTask == 0) {
        std::cout << "Locate halos\n";
    }

    // Locate halos
    part.communicate_particles();
    std::vector<FoFHalo> FoFGroups;
    FML::FOF::merging_in_parallel_default = true;
    FML::FOF::FriendsOfFriends<Particle, Ndim>(
        part.get_particles_ptr(), part.get_npart(), fof_distance, n_min_FoF_group, true, FoFGroups);

    // Sort by position
    std::sort(FoFGroups.begin(), FoFGroups.end(), [](const FoFHalo & a, const FoFHalo & b) -> bool {
        return a.pos[0] > b.pos[0];
    });

    if (FML::ThisTask == 0) {
        std::cout << "\n# Found " << FoFGroups.size() << " halos\n";
        std::ofstream fp("halo_out.txt");
        for (auto & g : FoFGroups) {
            if (g.np > 0) {
                fp << g.np << " ";
                for (int idim = 0; idim < Ndim; idim++)
                    fp << g.pos[idim] * box << " ";
                fp << "\n";
            }
        }
    }

#ifdef MEMORY_LOGGING
   FML::MemoryLog::get()->print();
#endif
}
