#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/ParticleTypes/SimpleParticle.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/Spline/Spline.h>

//=====================================================
//
// This example shows how to make a very simple and
// naive N-body solver from a given P(k).
//
// We generate particles using 1LPPT and step from the
// initial time til z = 0 and compute the power-spectrum
// and compare to the input.
//
// This is just meant to illustrate how one can do this
//
//=====================================================

//=====================================================
// The dimension we are working in (e.g. Ndim=2 is nice
// to use for testing as its much faster) and the
// boxsize in your physical units (say Mpc/h)
// This only enters in converting the power-spectum
// to dimensionless units
//=====================================================

const int Ndim = 3;
const double box = 1000.0;
const double z_ini = 20.0;
const bool fix_amplitude = true;
const double OmegaM = 1.0;
const int Nmesh = 256;
const int Npart_1D = 256;
const double buffer_factor = 2;
const int nsteps = 40;
std::string interpolation_method = "CIC";
std::string filename = "pofk.txt";

//=====================================================
// Type aliases for easier use below
//=====================================================
template <class T>
using MPIParticles = FML::PARTICLE::MPIParticles<T>;
template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;
using Particle = SimpleParticle<Ndim>;

//=====================================================
// To generate a gaussian random field we need a RNG generator
// and a power-spectrum. The P(k) we read in is at z=0
//=====================================================
void generate_delta(FFTWGrid<Ndim> & delta, std::string filename_pofk) {
    FML::RANDOM::RandomGenerator * rng = new FML::RANDOM::RandomGenerator;

    // Read k, P(k) in units of h/Mpc, (Mpc/h)^3
    std::ifstream fp(filename_pofk.c_str());
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
    // We scale this back to the original time using growth-factors
    // which we just approximate as D = a
    FML::INTERPOLATION::SPLINE::Spline logpofk_spline(logk, logpofk);
    std::function<double(double)> Powspec = [&](double kBox) {
        const double D = 1.0 / (1.0 + z_ini);
        const double logk = std::log(kBox / box);
        const double volume = std::pow(box, Ndim);
        return D * D * std::exp(logpofk_spline(logk)) / volume;
    };

    // Make a random field in fourier space
    FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier(delta, rng, Powspec, fix_amplitude);
}

void generate_particles(FFTWGrid<Ndim> & delta, MPIParticles<Particle> & part) {
    //=====================================================
    // Set up a regular grid with particles
    //=====================================================
    part.create_particle_grid(Npart_1D, buffer_factor, FML::xmin_domain, FML::xmax_domain);
    part.info();

    //=====================================================
    // Generate the 1LPT potential phi_1LPT = delta(k)/k^2
    //=====================================================
    FFTWGrid<Ndim> phi_1LPT;
    FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(delta, phi_1LPT);
    delta.free();

    //=====================================================
    // Generate displacement field Psi = Dphi
    // 3+3 FFTS
    //=====================================================
    std::vector<FFTWGrid<Ndim>> Psi_1LPT_vector;
    FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector(phi_1LPT, Psi_1LPT_vector);
    phi_1LPT.free();

    //=====================================================
    // Interpolate 1LPT displacement field to particle positions
    // (Note: if Npart_1D = Nmesh then we can assign directly
    // without interpolation using the disp fields as the
    // cell is at the particle position and save time though
    // we don't do this here)
    //=====================================================
    std::vector<std::vector<FML::GRID::FloatType>> displacements_1LPT(Ndim);
    for (int idim = Ndim - 1; idim >= 0; idim--) {
        if (FML::ThisTask == 0)
            std::cout << "Assigning particles for idim = " << idim << "\n";
        FML::INTERPOLATION::interpolate_grid_to_particle_positions(
            Psi_1LPT_vector[idim], part.get_particles_ptr(), part.get_npart(), displacements_1LPT[idim], "CIC");
        Psi_1LPT_vector[idim].free();
    }

    //=====================================================
    // Set the velocities to be the dimensionless v_code = a^2 dx/dt  / (H0 Box)
    // For simplicity for the growth rate we just use the approx D1 = a => f1 = 1 and f2 = 2
    // (easy to compute the exact values by just solving the ODE or using the approx f1 ~ OmegaM(a)^0.55)
    //=====================================================
    const double a_ini = 1.0 / (1.0 + z_ini);
    const double HoverH0 = std::sqrt(OmegaM / (a_ini * a_ini * a_ini) + 1.0 - OmegaM);
    const double growth_rate1 = std::pow(OmegaM / (a_ini * a_ini * a_ini * HoverH0 * HoverH0), 0.55);
    const double vfac_1LPT = a_ini * a_ini * HoverH0 * growth_rate1;

    //=====================================================
    // Add 1LPT displacement to particle position
    // Note the -3/7 factor. This is because growth factors
    // are defined to be == 1 at the initial time, however
    // the physically relevant solution has D2 = -3/7 D1^2
    // so we need to put this in by hand
    //=====================================================

    double max_disp_1LPT = 0.0;
    double max_vel_1LPT = 0.0;

    auto * part_ptr = part.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_disp_1LPT, max_vel_1LPT)
#endif
    for (size_t ind = 0; ind < part.get_npart(); ind++) {
        auto * pos = part_ptr[ind].get_pos();
        auto * vel = part_ptr[ind].get_vel();
        for (int idim = Ndim - 1; idim >= 0; idim--) {
            const double dpos_1LPT = displacements_1LPT[idim][ind];
            pos[idim] += dpos_1LPT;
            vel[idim] = vfac_1LPT * dpos_1LPT;
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

    //=====================================================
    // Output the maximum displacment and velocity
    //=====================================================
    FML::MaxOverTasks(&max_disp_1LPT);
    FML::MaxOverTasks(&max_vel_1LPT);
    if (FML::ThisTask == 0)
        std::cout << "Maximum displacements: " << max_disp_1LPT * box << " Mpc/h\n";
    if (FML::ThisTask == 0)
        std::cout << "Maximum velocity: " << max_vel_1LPT * 100.0 * box << " km/s comoving\n";

    //=====================================================
    // Communicate particles (they might have left the
    // current task)
    //=====================================================
    part.communicate_particles();
}

int main() {

    //=====================================================
    // Generate density field
    //=====================================================
    auto nextra = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(interpolation_method);
    FFTWGrid<Ndim> delta(Nmesh, nextra.first, nextra.second);
    generate_delta(delta, filename);

    //===============================Y======================
    // Make a regular (Lagrangian) particle grid with
    // Npart_1D^NDIM particles in total
    //=====================================================
    MPIParticles<Particle> part;
    generate_particles(delta, part);

    // Start the main time-stepping loop
    double aini = 1.0 / (1.0 + z_ini);
    double aend = 1.0;
    const double delta_aexp = (aend - aini) / double(nsteps);
    auto HoverH0 = [&](double a) { return std::sqrt(OmegaM / (a * a * a) + 1.0 - OmegaM); };

    for (int i = 0; i < nsteps; i++) {

        // Scale factor for velocity and positions
        double apos_old = aini + delta_aexp * i;
        double apos_new = apos_old + delta_aexp;
        double avel_old = i == 0 ? aini : apos_old - 0.5 * delta_aexp;
        double avel_new = i == 0 ? aini + 0.5 * delta_aexp : apos_old + 0.5 * delta_aexp;

        // Timestep: first step we displace pos and vel by 1/2 timestep to do leapfrog
        double delta_time_pos = (apos_new - apos_old) / (HoverH0(avel_new) * std::pow(avel_new, 3));
        double delta_time_vel = (avel_new - avel_old) / (HoverH0(apos_old) * std::pow(apos_old, 3));
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

        // Move particles (this does communication)
        FML::NBODY::DriftParticles<Ndim, Particle>(part, delta_time_pos);

        if (FML::ThisTask == 0)
            std::cout << i + 1 << " / " << nsteps << " done! a: " << apos_new << "\n";
    }
    // Velocities are here da/2 out of sync with positions (we don't bother fixing this in this example)

    // Compute power-spectrum
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<Ndim> pofk(Nmesh / 2);
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum_interlacing<Ndim>(
        Nmesh, part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), pofk, "PQS");
    pofk.scale(box);
    if (FML::ThisTask == 0) {
        for (int i = 0; i < pofk.n; i++) {
            const double k = pofk.kbin[i];
            std::cout << k << " " << pofk.pofk[i] << "\n";
        }
    }
}

