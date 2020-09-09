#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FriendsOfFriends/FoF.h>
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
// We also implement the COLA method (just a few more lines)
// and how to compute P(k) and a halo catlogue from
// the final result.
//
// This is just meant to illustrate how one can do this
// We only do 1LPT, adding 2LPT is straight forward,
// just copy the code from LPT/example
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
const int nsteps = 20;
const bool fix_amplitude = true;
const unsigned int random_seed = 1234;
const double OmegaM = 1.0;
const int Nmesh = 256;
const int Npart_1D = 128;
const double buffer_factor = 2;
const bool COLA = true;
const double apofk = 1.0;
const int n_min_FoF_group = 20;
const double fof_distance = 0.2 / double(Npart_1D);
const std::string filename = "pofk.txt";
const std::string interpolation_method = "CIC";
FML::INTERPOLATION::SPLINE::Spline D_1LPT_of_loga;
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

void generate_delta(FFTWGrid<Ndim> & delta, std::string filename_pofk) {
    // Set up random number generator
    FML::RANDOM::RandomGenerator * rng = new FML::RANDOM::RandomGenerator;
    rng->set_seed(random_seed);

    // Read k, P(k) (File has units of h/Mpc, (Mpc/h)^3 but what matters is that it has the same
    // units as the boxsize)
    std::ifstream fp(filename_pofk.c_str());
    if (!fp.is_open())
        throw std::runtime_error("Cannot open file: " + filename_pofk);
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

    // Power spectrum scaled back to the initial time using growth-factors
    const double DiniOverD = D_1LPT_of_loga(std::log(aini)) / D_1LPT_of_loga(std::log(apofk));
    std::function<double(double)> Powspec = [&](double kBox) {
        const double logk = std::log(kBox / box);
        const double volume = std::pow(box, Ndim);
        return DiniOverD * DiniOverD * std::exp(logpofk_spline(logk)) / volume;
    };

    // Make a random field in fourier space
    FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier(delta, rng, Powspec, fix_amplitude);
}

void generate_particles(FFTWGrid<Ndim> & delta, MPIParticles<Particle> & part) {
    // Set up a regular grid with particles
    part.create_particle_grid(Npart_1D, buffer_factor, FML::xmin_domain, FML::xmax_domain);

    // Generate the 1LPT potential phi_1LPT = delta(k)/k^2
    FFTWGrid<Ndim> phi_1LPT;
    FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(delta, phi_1LPT);
    delta.free();

    // Generate displacement field Psi = Dphi (3+3 FFTs)
    std::vector<FFTWGrid<Ndim>> Psi_1LPT_vector;
    FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector(phi_1LPT, Psi_1LPT_vector);
    phi_1LPT.free();

    // Interpolate 1LPT displacement field to particle positions
    std::vector<std::vector<FML::GRID::FloatType>> displacements_1LPT(Ndim);
    for (int idim = Ndim - 1; idim >= 0; idim--) {
        FML::INTERPOLATION::interpolate_grid_to_particle_positions( // CIC here?
            Psi_1LPT_vector[idim],
            part.get_particles_ptr(),
            part.get_npart(),
            displacements_1LPT[idim],
            interpolation_method);
        Psi_1LPT_vector[idim].free();
    }

    // Set the velocities to be the dimensionless v_code = a^2 dx/dt  / (H0 Box)
    const double growth_rate1 = D_1LPT_of_loga.deriv_x(std::log(aini)) / D_1LPT_of_loga(std::log(aini));
    const double vfac_1LPT = aini * aini * HoverH0(aini) * growth_rate1;

    // Add 1LPT displacement to particle position
    double max_disp_1LPT = 0.0;
    double max_vel_1LPT = 0.0;
    auto * part_ptr = part.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_disp_1LPT, max_vel_1LPT)
#endif
    for (size_t ind = 0; ind < part.get_npart(); ind++) {
        auto * pos = part_ptr[ind].get_pos();
        auto * vel = part_ptr[ind].get_vel();
        auto * D = part_ptr[ind].get_D();
        for (int idim = Ndim - 1; idim >= 0; idim--) {
            const double dpos_1LPT = displacements_1LPT[idim][ind];
            pos[idim] += dpos_1LPT;
            if (COLA) {
                // With COLA: in the COLA frame the velocity is 0 and store disp field
                D[idim] = dpos_1LPT;
                vel[idim] = 0.0;
            } else {
                vel[idim] = vfac_1LPT * dpos_1LPT;
            }
            if (std::fabs(dpos_1LPT) > max_disp_1LPT)
                max_disp_1LPT = std::fabs(dpos_1LPT);
            if (std::fabs(vfac_1LPT * dpos_1LPT) > max_vel_1LPT)
                max_vel_1LPT = std::fabs(vfac_1LPT * dpos_1LPT);
            if (pos[idim] >= 1.0)
                pos[idim] -= 1.0;
            if (pos[idim] < 0.0)
                pos[idim] += 1.0;
        }
    }

    // Output the maximum displacment and velocity
    FML::MaxOverTasks(&max_disp_1LPT);
    FML::MaxOverTasks(&max_vel_1LPT);
    if (FML::ThisTask == 0)
        std::cout << "IC maximum displacements: " << max_disp_1LPT * box << " Mpc/h\n";
    if (FML::ThisTask == 0)
        std::cout << "IC maximum velocity: " << max_vel_1LPT * 100.0 * box / aini << " km/s peculiar\n";

    // Communicate particles (they might have left the current task)
    part.communicate_particles();
}

int main() {

    // We need the LPT growth factor(s). Here only 1LPT:
    auto compute_growth_factor = [&]() {
        const int npts = 1000;
        const double aini = std::min(0.01, 1.0 / (1.0 + zini));
        const double aend = 1.0;

        // The ODE ddDddx + ( 2 + H'/ ) dDdx = 1.5 OmegaM / (H^2 a^3) D
        ODEFunction deriv = [&](double x, const double * y, double * dydx) {
            const double a = std::exp(x);
            const double H = HoverH0(a);
            const double dlogHdx = 1.0 / (2.0 * H * H) * (-3.0 * OmegaM / (a * a * a));
            const double D = y[0];
            const double dDdx = y[1];
            dydx[0] = dDdx;
            dydx[1] = 1.5 * OmegaM / (H * H * a * a * a) * D - (2.0 + dlogHdx) * dDdx;
            return GSL_SUCCESS;
        };

        // The initial conditions D = a for growing mode in EdS
        DVector yini{1.0, 1.0};
        DVector xarr(npts);
        for (int i = 0; i < npts; i++)
            xarr[i] = std::log(aini) + std::log(aend / aini) * i / double(npts);

        // Solve the ODE
        ODESolver ode;
        ode.solve(deriv, xarr, yini);
        auto D = ode.get_data_by_component(0);

        // Spline it up
        D_1LPT_of_loga.create(xarr, D, "D(loga) Spline");
    };
    compute_growth_factor();

    // Generate density field
    auto nextra = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(interpolation_method);
    FFTWGrid<Ndim> delta(Nmesh, nextra.first, nextra.second);
    generate_delta(delta, filename);

    // Make a regular (Lagrangian) particle grid with Npart_1D^NDIM particles in total
    MPIParticles<Particle> part;
    generate_particles(delta, part);

    // Instead of stepping in dt_code = dt H0/a^2 we have unit time-steps in a. Here we
    // compute the integral of dt_code/da over one time-step
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

    // Main timestepping loop
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
            const double Dnew = D_1LPT_of_loga(std::log(apos_new));
            const double Dold = D_1LPT_of_loga(std::log(apos_old));
            const double Dini = D_1LPT_of_loga(std::log(aini));
            const double fac_pos = (Dnew - Dold) / Dini;
            const double fac_vel = -1.5 * OmegaM * apos_old * Dold / Dini * delta_time_vel;
            for (auto & p : part.get_particles()) {
                auto * pos = p.get_pos();
                auto * vel = p.get_vel();
                auto * D = p.get_D();
                for (int idim = 0; idim < Ndim; idim++) {
                    pos[idim] += D[idim] * fac_pos;
                    vel[idim] += D[idim] * fac_vel;
                }
            }
        }

        // Move particles (this does communication)
        FML::NBODY::DriftParticles<Ndim, Particle>(part, delta_time_pos);

        if (FML::ThisTask == 0)
            std::cout << i + 1 << " / " << nsteps << " Positions at a: " << apos_new << " Velocities at a: " << avel_new
                      << "\n";
    }
    // NB: velocities are here da/2 out of sync with positions (we don't bother fixing this in this example)

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

    // Locate halos
    part.communicate_particles();
    std::vector<FoFHalo> FoFGroups;
    FML::FOF::merging_in_parallel_default = true;
    FML::FOF::FriendsOfFriends<Particle, Ndim>(
        part.get_particles_ptr(), part.get_npart(), fof_distance, n_min_FoF_group, true, FoFGroups);

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
}
