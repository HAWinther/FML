#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/MemoryLogging/MemoryLogging.h>
#include <FML/ParticleTypes/SimpleParticle.h>
#include <FML/RandomFields/GaussianRandomField.h>

//=====================================================
//
// In this example we do the following
//
// 1. Generate a gaussian random field
// delta from a given power-spectrum.
//
// 2. Compute the LPT displacement field to
// second order
//
// 3. Generate particle positions
// and velocity (we use v_code = dx/dt / (H0 Box) )
// (The velocity and 2LPT is cosmology specific)
//
// 4. Compute the power-spectrum of this particle
// distribution and the cross spectrum with the
// initial density field and compare it to the input P(k)
//
// So this test is basically a very simple IC generator
// for N-body simulations.
//
// Relation between code units and physical units here:
// dt_code = H0 dt/a^2
// x_code = x/Box
// v_code = a^2 dx/dt / (H0 Box) (see velfac below)
// Thus positions are in [0,1) and the velocities are
// comoving velocities in units of (100 * box) km/s
//
//=====================================================

//=====================================================
// The dimension we are working in (e.g. Ndim=2 is nice
// to use for testing as its much faster) and the
// boxsize in your physical units (say Mpc/h)
//=====================================================
const int Ndim = 3;
const double box = 100.0;

//=====================================================
// Type aliases for easier use below
//=====================================================
template <class T>
using MPIParticles = FML::PARTICLE::MPIParticles<T>;
template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;
using Particle = SimpleParticle<Ndim>;

//=====================================================
// Power-spectrum at the time we want to generate the
// IC in physical units. Just a random example
// (i.e. Units of L^NDIM where L is the units of box)
//=====================================================
double power_spectrum(double k) {
    if (k == 0.0)
        return 0.0;
    return 1e-4 / std::pow(k, Ndim);
}

//=====================================================
// To generate a gaussian random field we need a RNG generator
// and a power-spectrum
//=====================================================
void generate_delta(FFTWGrid<Ndim> & delta) {
    FML::RANDOM::RandomGenerator * rng = new FML::RANDOM::RandomGenerator;
    std::function<double(double)> Powspec;

    // Fix amplitude (so only random phases) or do a normal GRF if false
    const bool fix_amplitude = true;

    // Function P(kBox / Box) / Volume that the method below requires
    const double volume = std::pow(box, Ndim);
    Powspec = [&](double kBox) -> double { return power_spectrum(kBox / box) / volume; };

    // Make a random field in fourier space
    FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier(delta, rng, Powspec, fix_amplitude);
}

int main() {
#ifdef MEMORY_LOGGING
    auto * mem = FML::MemoryLog::get();
#endif

    //=====================================================
    // Setting: what grid to use to generate the IC on and
    // how many particles to generate
    //=====================================================
    const int Nmesh = 128;
    const int Npart_1D = 128;
    const double buffer_factor = 1.25;

    //=====================================================
    // Generate density field (the extra slices we use in this grid
    // propagates to the grids we generate below)
    // interpolation_method is how we interpolate from the grid
    // to the particles (use NGP or CIC)
    //=====================================================
    std::string interpolation_method = "CIC";
    const auto nextra = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(interpolation_method);
    FFTWGrid<Ndim> delta(Nmesh, nextra.first, nextra.second);
    generate_delta(delta);

    //=====================================================
    // Generate the 1LPT potential phi_1LPT = delta(k)/k^2
    //=====================================================
    FFTWGrid<Ndim> phi_1LPT;
    FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(delta, phi_1LPT);

    //=====================================================
    // Generate the 2LPT potential phi_2LPT
    // 7 FFTS
    //=====================================================
    FFTWGrid<Ndim> phi_2LPT;
    FML::COSMOLOGY::LPT::compute_2LPT_potential_fourier(delta, phi_2LPT);

    //=====================================================
    // Generate displacement field Psi = Dphi
    // 3+3 FFTS
    //=====================================================
    std::array<FFTWGrid<Ndim>, Ndim> Psi_1LPT_vector;
    FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<Ndim>(phi_1LPT, Psi_1LPT_vector);
    phi_1LPT.free();

    std::array<FFTWGrid<Ndim>, Ndim> Psi_2LPT_vector;
    FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<Ndim>(phi_2LPT, Psi_2LPT_vector);
    phi_2LPT.free();

    //=====================================================
    // Make a regular (Lagrangian) particle grid with
    // Npart_1D^NDIM particles in total
    //=====================================================
    MPIParticles<Particle> part;
    part.create_particle_grid(Npart_1D, buffer_factor, FML::xmin_domain, FML::xmax_domain);
    part.info();

    //=====================================================
    // Interpolate 1LPT displacement field to particle positions
    // (Note: if Npart_1D = Nmesh then we can assign directly
    // without interpolation using the disp fields as the
    // cell is at the particle position and save time though
    // we don't do this here)
    //=====================================================
    std::array<std::vector<FML::GRID::FloatType>, Ndim> displacements_1LPT;
    FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<Ndim, Particle>(
        Psi_1LPT_vector, part.get_particles_ptr(), part.get_npart(), displacements_1LPT, interpolation_method);
    for (int idim = 0; idim < Ndim; idim++) {
        Psi_1LPT_vector[idim].free();
    }

    //=====================================================
    // Interpolate 2LPT displacement field to particle positions
    //=====================================================
    std::array<std::vector<FML::GRID::FloatType>, Ndim> displacements_2LPT;
    FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<Ndim, Particle>(
        Psi_2LPT_vector, part.get_particles_ptr(), part.get_npart(), displacements_2LPT, interpolation_method);
    for (int idim = 0; idim < Ndim; idim++) {
        Psi_2LPT_vector[idim].free();
    }

    //=====================================================
    // Set the velocities to be the dimensionless v_code = a^2 dxdt / (H0 Box)
    // For simplicity for the growth rate we just use the approx D1 = a => f1 = 1 and f2 = 2
    // valid for OmegaM = 1
    // (easy to compute the exact values by just solving the ODE or using the approx f1 ~ OmegaM(a)^0.55)
    //=====================================================
    const double a_ini = 1.0 / (1.0 + 50.0);
    const double OmegaM = 1.0;
    const double HoverH0 = std::sqrt(OmegaM / (a_ini * a_ini * a_ini) + 1.0 - OmegaM);
    const double growth_rate1 = 1.0;
    const double growth_rate2 = 2.0;
    const double vfac_1LPT = a_ini * a_ini * HoverH0 * growth_rate1;
    const double vfac_2LPT = a_ini * a_ini * HoverH0 * growth_rate2;

    //=====================================================
    // Add 1LPT displacement to particle position
    // Add 2LPT displacement to particle position
    //=====================================================

    double max_disp_1LPT = 0.0;
    double max_disp_2LPT = 0.0;
    double max_vel_1LPT = 0.0;
    double max_vel_2LPT = 0.0;

    auto * part_ptr = part.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_disp_1LPT, max_disp_2LPT, max_vel_1LPT, max_vel_2LPT)
#endif
    for (size_t ind = 0; ind < part.get_npart(); ind++) {
        auto * pos = part_ptr[ind].get_pos();
        auto * vel = part_ptr[ind].get_vel();
        for (int idim = Ndim - 1; idim >= 0; idim--) {
            const double dpos_1LPT = displacements_1LPT[idim][ind];
            const double dpos_2LPT = displacements_2LPT[idim][ind];
            pos[idim] += dpos_1LPT + dpos_2LPT;
            vel[idim] = vfac_1LPT * dpos_1LPT + vfac_2LPT * dpos_2LPT;
            if (std::fabs(dpos_1LPT) > max_disp_1LPT)
                max_disp_1LPT = std::fabs(dpos_1LPT);
            if (std::fabs(dpos_2LPT) > max_disp_2LPT)
                max_disp_2LPT = std::fabs(dpos_2LPT);
            if (std::fabs(vfac_1LPT * dpos_1LPT) > max_vel_1LPT)
                max_vel_1LPT = std::fabs(vfac_1LPT * dpos_1LPT);
            if (std::fabs(vfac_2LPT * dpos_2LPT) > max_vel_2LPT)
                max_vel_2LPT = std::fabs(vfac_2LPT * dpos_2LPT);
            if (pos[idim] >= 1.0)
                pos[idim] -= 1.0;
            if (pos[idim] < 0.0)
                pos[idim] += 1.0;
        }
    }

    //=====================================================
    // We no longer need Psi
    //=====================================================
    for (int idim = 0; idim < Ndim; idim++) {
        displacements_1LPT[idim].clear();
        displacements_1LPT[idim].shrink_to_fit();
        displacements_2LPT[idim].clear();
        displacements_2LPT[idim].shrink_to_fit();
    }

    //=====================================================
    // Output the maximum displacment and velocity
    //=====================================================
    FML::MaxOverTasks(&max_disp_1LPT);
    FML::MaxOverTasks(&max_disp_2LPT);
    FML::MaxOverTasks(&max_vel_1LPT);
    FML::MaxOverTasks(&max_vel_2LPT);
    if (FML::ThisTask == 0)
        std::cout << "Maximum displacements: " << max_disp_1LPT * Nmesh << " grid cells\n";
    if (FML::ThisTask == 0)
        std::cout << "Maximum displacements: " << max_disp_2LPT * Nmesh << " grid cells\n";
    if (FML::ThisTask == 0)
        std::cout << "Maximum velocity: " << max_vel_1LPT * 100.0 * box / a_ini << " km/s peculiar\n";
    if (FML::ThisTask == 0)
        std::cout << "Maximum velocity: " << max_vel_2LPT * 100.0 * box / a_ini << " km/s peculiar\n";

    //=====================================================
    // Communicate particles (they might have left the
    // current task)
    //=====================================================
    part.communicate_particles();

    //=====================================================
    // Lets test that it works as expected
    // From particles to density field. Compute P(k) and cross spectrum wrt the random field used
    // to generate particles. This shows that with interlacing and a high order kernel (PQS is 5th order)
    // we are able to reconstruct the initial density field from the particles to
    // high precision
    //=====================================================
    const bool interlacing = true;
    const std::string density_assignment_method = "PQS";
    const auto nlr = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(density_assignment_method);
    FFTWGrid<Ndim> delta_from_part(Nmesh, nlr.first, nlr.second);
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<Ndim> pofk_cross(Nmesh / 2);
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<Ndim> pofk_part(Nmesh / 2);
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<Ndim> pofk_ini(Nmesh / 2);
    FML::INTERPOLATION::particles_to_fourier_grid(part.get_particles_ptr(),
                                                  part.get_npart(),
                                                  part.get_npart_total(),
                                                  delta_from_part,
                                                  density_assignment_method,
                                                  interlacing);
    FML::INTERPOLATION::deconvolve_window_function_fourier<Ndim>(delta_from_part, density_assignment_method);
    FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum(delta_from_part, pofk_part);
    FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum(delta, pofk_ini);
    FML::CORRELATIONFUNCTIONS::bin_up_cross_power_spectrum(delta, delta_from_part, pofk_cross);

    //=====================================================
    // Convert P(k) to physical units: inherits the same
    // units as the boxsize so h/Mpc for k and (Mpc/h)^NDIM for P(k)
    //=====================================================
    pofk_part.scale(box);
    pofk_ini.scale(box);
    pofk_cross.scale(box);

    //=====================================================
    // Output to screen
    //=====================================================
    if (FML::ThisTask == 0) {
        std::cout << "\n#  k (h/Mpc)   CorrelationCoeff      P_part / P       P / P_analytic\n";
        for (int i = 0; i < pofk_ini.n; i++) {
            double k = pofk_ini.kbin[i];
            double cross_corr_coeff = pofk_cross.pofk[i] / std::sqrt(pofk_part.pofk[i] * pofk_ini.pofk[i]);
            std::cout << k << " " << cross_corr_coeff << " " << pofk_part.pofk[i] / pofk_ini.pofk[i] << " "
                      << pofk_ini.pofk[i] / power_spectrum(k) << "\n";
        }
    }

    //=====================================================
    // Free the last of the memory before printing the
    // memory status (this is not needed)
    //=====================================================
    part.free();

#ifdef MEMORY_LOGGING
    // Print memory summary
    mem->print();
#endif
}
