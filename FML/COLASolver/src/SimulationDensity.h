#ifndef SIMULATIONDENSITY_HEADER
#define SIMULATIONDENSITY_HEADER

//=============================================================================
// Part of NBodySimulation (included at the end of Simulation.h): the density field the force is computed from.
//
//   delta_m(k) = delta_cb(k)                                  (CDM+baryon particles)
//              + one of (the parameter validation does not allow combinations):
//                  the linear N-body gauge source             (force_nbody_gauge)
//                  the neutrino particles                     (particle_neutrinos)
//                  linear massive neutrinos                   (force_linear_massive_neutrinos)
//
// The P(k) of the particles (and of each species and the total) is binned up at every step on the way.
//=============================================================================

/// Assign the particles to a grid, Fourier transform it and add the other contributions to delta_m (see above).
/// Also bins up P(k) at every step (if pofk_every_step)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::compute_density_field_fourier(FFTWGrid<NDIM> & density_grid_fourier, double a) {
    const auto & force = current_force;
    const bool subtract_shotnoise = params.pofk.subtract_shotnoise;

    if (FML::ThisTask == 0) {
        std::cout << (has_baryons() ? "Adding the CDM and baryon particles to the densityfield\n" :
                                      "Adding particles (baryons+CDM) to the densityfield\n");
    }
    timer.StartTiming("DensityAssignment+FFT");
    assign_cb_particles_to_grid(density_grid_fourier, a);
    timer.EndTiming("DensityAssignment+FFT");

    // P(k) of the CDM+baryon particles. With force interlacing the particles might be shifted relative to the grid:
    // then we have the linear fields below in the shifted frame and do not bin up P(k) again
    const bool every_step = params.pofk.every_step and not particles_shifted();
    const bool deconvolve = not(force.density_method == "nufft" or force.density_method == "sheet_nufft");
    if (every_step) {
        timer.StartTiming("PofkEveryStep");
        bin_up_power_spectrum_every_step(density_grid_fourier,
                                         deconvolve,
                                         subtract_shotnoise ? cb_shotnoise() : 0.0,
                                         a,
                                         pofk_cb_every_step);
        timer.EndTiming("PofkEveryStep");
    }

    // The other contributions to delta_m
    bool has_total = true;
    double total_shotnoise_value = 0.0;
    if (nbody_gauge) {
        add_nbody_gauge_source(density_grid_fourier, a);
    } else if (has_neutrino_particles()) {
        add_neutrino_particles(density_grid_fourier, a);
        if (subtract_shotnoise)
            total_shotnoise_value = total_shotnoise(a);
    } else if (force.linear_massive_neutrinos and cosmo->get_fMNu() > 0.0 and transferdata) {
        add_linear_neutrinos(density_grid_fourier, a);
    } else {
        has_total = false;
    }

    // P(k) of the total. NB: with linear neutrinos / the N-body gauge source the density field is W delta_cb +
    // delta_extra so the deconvolution of the window W is not fully correct for the last part at large k. One way
    // to improve this is to bin up <(W delta_cb)^2>, <delta_extra^2> and the cross term and deconvolve them separately
    if (has_total and every_step)
        bin_up_power_spectrum_every_step(density_grid_fourier,
                                         deconvolve,
                                         total_shotnoise_value,
                                         a,
                                         pofk_total_every_step);

    initial_density_field_shifted.free();
}

/// The CDM+baryon density delta_cb(k): particles to grid and Fourier transform (or NUFFT / sheet interpolation).
/// With CDM and baryons as separate particles delta_cb = w_c delta_c + w_b delta_b (and P(k) of each species)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::assign_cb_particles_to_grid(FFTWGrid<NDIM> & density_grid_fourier, double a) {
    const auto & force = current_force;
    if (force.density_method == "nufft") {
        // The exact Fourier modes (no window function, no aliasing) for |m_i| < Nmesh/2
        FML::NUFFT::particles_to_fourier_grid_nufft<NDIM>(part.get_particles_ptr(),
                                                          part.get_npart(),
                                                          part.get_npart_total(),
                                                          density_grid_fourier,
                                                          force.nufft_epsilon);
        return;
    }
    if (force.density_method == "sheet" or force.density_method == "sheet_nufft") {
        if constexpr (FML::PARTICLE::has_get_q<T>()) {
            FML::INTERPOLATION::particles_to_fourier_grid_sheet<NDIM>(
                part.get_particles_ptr(),
                part.get_npart(),
                params.particle.Npart_1D,
                force.sheet_refinement,
                density_grid_fourier,
                force.density_method == "sheet" ? force.density_assignment_method : std::string("NUFFT"),
                force.nufft_epsilon);
        } else {
            FML::assert_mpi(false, "force_density_method = sheet needs particles with a Lagrangian position q");
        }
        return;
    }

    // Each CDM+baryon species to a grid. The first one goes into density_grid_fourier and the others are added
    // with their weights
    int ncb = 0;
    for (auto & info : species)
        ncb += info.in_cb() ? 1 : 0;
    const bool bin_species = ncb > 1 and params.pofk.every_step;
    int icb = 0;
    double weight_first = 1.0;
    for_each_cb_species([&](MPIParticles & particles, const COLASOLVER::SpeciesInfo & info) {
        auto assign = [&](FFTWGrid<NDIM> & grid) {
            FML::INTERPOLATION::particles_to_grid(particles.get_particles_ptr(),
                                                  particles.get_npart(),
                                                  particles.get_npart_total(),
                                                  grid,
                                                  force.density_assignment_method);
            grid.fftw_r2c();
            if (bin_species)
                bin_up_power_spectrum_every_step(grid,
                                                 true,
                                                 params.pofk.subtract_shotnoise ? species_shotnoise(info) : 0.0,
                                                 a,
                                                 pofk_species_every_step[info.name]);
        };
        if (icb == 0) {
            assign(density_grid_fourier);
            weight_first = info.weight_cb;
        } else {
            FFTWGrid<NDIM> grid(density_grid_fourier.get_nmesh(),
                                density_grid_fourier.get_n_extra_slices_left(),
                                density_grid_fourier.get_n_extra_slices_right());
            assign(grid);
            FML::GRID::linear_combination_fourier<NDIM>(density_grid_fourier,
                                                        grid,
                                                        icb == 1 ? weight_first : 1.0,
                                                        info.weight_cb);
        }
        icb++;
    });
}

/// N-body gauge: add the linear source from radiation, neutrinos and GR
/// delta_extra(k,a) = delta_cb(k,aini) T_extra(k,a) / T_cb(k,aini)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::add_nbody_gauge_source(FFTWGrid<NDIM> & density_grid_fourier, double a) {
    if (FML::ThisTask == 0)
        std::cout << "Adding the N-body gauge linear source (radiation, neutrinos, GR) to the densityfield\n";
    const double boxsize = params.simulation.boxsize;
    const double aini = 1.0 / (1.0 + params.ic.initial_redshift);
    const Spline ratio = nbody_gauge->ratio_spline(a, aini, *transferdata);
    const double logkmin = std::log(nbody_gauge->get_kmin_hmpc());
    const double logkmax = std::log(nbody_gauge->get_kmax_hmpc());

    // The force is normalized to OmegaM = OmegaCB + OmegaMNu so the source is (1-fMNu)(delta_cb + delta_extra)
    const double fMNu = cosmo->get_fMNu();
    const auto & delta_ini_grid = initial_density_in_particle_frame();
    FML::GRID::for_each_fourier_mode<NDIM>(density_grid_fourier, [&](size_t fourier_index, double kmag) {
        const double logk = std::min(std::max(std::log(kmag / boxsize), logkmin), logkmax);
        const double r = kmag > 0.0 ? ratio(logk) : 0.0;
        auto delta_ini = delta_ini_grid.get_fourier_from_index(fourier_index);
        auto deltaCB = density_grid_fourier.get_fourier_from_index(fourier_index);
        auto delta = (deltaCB + delta_ini * FML::GRID::FloatType(r)) * FML::GRID::FloatType(1.0 - fMNu);
        density_grid_fourier.set_fourier_from_index(fourier_index, delta);
    });
}

/// Neutrino particles: delta_m = (1 - f_nu) delta_cb + f_nu delta_nu (and P_nu(k))
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::add_neutrino_particles(FFTWGrid<NDIM> & density_grid_fourier, double a) {
    if (FML::ThisTask == 0)
        std::cout << "Adding the neutrino particles to the densityfield\n";
    FFTWGrid<NDIM> density_nu(density_grid_fourier.get_nmesh(),
                              density_grid_fourier.get_n_extra_slices_left(),
                              density_grid_fourier.get_n_extra_slices_right());
    {
        // With force_neutrino_energy_density the field is the energy-density contrast (weights w E / (m <E/m>))
        typename COLASOLVER::NeutrinoParticles<NDIM>::ScopedEnergyWeights energy_weights(
            neutrinos,
            a,
            params.force.neutrino_energy_density);
        neutrinos.density_fourier(density_nu, current_force.density_assignment_method);
        if (params.pofk.every_step and not particles_shifted())
            bin_up_power_spectrum_every_step(density_nu,
                                             true,
                                             params.pofk.subtract_shotnoise ? neutrinos.shotnoise() : 0.0,
                                             a,
                                             pofk_species_every_step["nu"]);
    }
    // The source of the Poisson equation (normalized to the non-relativistic matter density Omega_M a^-3):
    // (1 - f_nu) delta_cb + f_nu <E/m>(a) delta_nu
    const double fMNu = cosmo->get_fMNu();
    FML::GRID::linear_combination_fourier<NDIM>(density_grid_fourier,
                                                density_nu,
                                                1.0 - fMNu,
                                                fMNu * neutrino_energy_factor(a));
}

/// Linear massive neutrinos: delta_m = (1 - f_nu) delta_cb + f_nu delta_cb(k,zini) T_nu(k,z) / T_cb(k,zini)
/// (needs the transfer functions, i.e. transferinfofile). The first time we store delta_cb(k,zini)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::add_linear_neutrinos(FFTWGrid<NDIM> & density_grid_fourier, double a) {
    const double redshift = 1.0 / a - 1.0;
    const double ic_initial_redshift = params.ic.initial_redshift;
    if (not initial_density_field_fourier) {
        initial_density_field_fourier = density_grid_fourier;
        initial_density_field_fourier.add_memory_label("density(k,zini)");
        FML::assert_mpi(std::fabs(redshift - ic_initial_redshift) < 1e-3, "This should not happen");
    }

    if (FML::ThisTask == 0) {
        std::cout << "Adding on the massive neutrinos to the densityfield using linear theory\n";
    }

    // delta_cb(zini,k) -> delta_nu(z,k) using the transfer functions
    const double aini = 1.0 / (1.0 + ic_initial_redshift);
    const double koverkBox = 1.0 / params.simulation.boxsize;
    auto norm = [&](double kBox) {
        double T_mnu = transferdata->get_massive_neutrino_transfer_function(kBox * koverkBox, a);
        double T_cb_ini = transferdata->get_cdm_baryon_transfer_function(kBox * koverkBox, aini);
        return (T_mnu / T_cb_ini);
    };

    // The total matter density-field deltaM = (OmegaCB deltaCB + OmegaMNu deltaMNu)/OmegaM. With
    // force_neutrino_energy_density the neutrino term is f_nu <E/m>(a) delta_nu (delta_nu from the transfer functions
    // is the energy-density contrast)
    const double fMNu = cosmo->get_fMNu();
    const double fMNu_source = fMNu * neutrino_energy_factor(a);
    const auto & delta_ini_grid = initial_density_in_particle_frame();
    FML::GRID::for_each_fourier_mode<NDIM>(density_grid_fourier, [&](size_t fourier_index, double kmag) {
        auto delta_ini = delta_ini_grid.get_fourier_from_index(fourier_index);
        auto deltaCB = density_grid_fourier.get_fourier_from_index(fourier_index);
        auto deltaM =
            deltaCB * FML::GRID::FloatType(1.0 - fMNu) + delta_ini * FML::GRID::FloatType(norm(kmag) * fMNu_source);
        density_grid_fourier.set_fourier_from_index(fourier_index, deltaM);
    });
}

/// The initial density field in the frame of the particles (with force interlacing the particles are shifted by
/// density_shift relative to the grid: delta_s(k) = exp(-ik.s) delta(k))
template <int NDIM, class T>
const FML::GRID::FFTWGrid<NDIM> & NBodySimulation<NDIM, T>::initial_density_in_particle_frame() {
    if (not particles_shifted())
        return initial_density_field_fourier;
    if (not initial_density_field_shifted) {
        initial_density_field_shifted = initial_density_field_fourier;
        FML::GRID::shift_grid_fourier<NDIM>(initial_density_field_shifted, density_shift, -1.0);
    }
    return initial_density_field_shifted;
}

/// Bin up P(k) of a density field (deconvolving the window of the force density assignment), subtract the shot noise
/// (if pofk_subtract_shotnoise; shotnoise in units of the box volume) and add it to the list
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::bin_up_power_spectrum_every_step(
    const FFTWGrid<NDIM> & grid,
    bool deconvolve,
    double shotnoise,
    double a,
    std::vector<std::pair<double, PowerSpectrumBinning>> & list) {
    PowerSpectrumBinning pofk(grid.get_nmesh() / 2);
    pofk.subtract_shotnoise = params.pofk.subtract_shotnoise;
    if (deconvolve)
        FML::CORRELATIONFUNCTIONS::bin_up_deconvolved_power_spectrum(grid,
                                                                     pofk,
                                                                     current_force.density_assignment_method);
    else
        FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum(grid, pofk);
    if (params.pofk.subtract_shotnoise and shotnoise != 0.0)
        for (int i = 0; i < pofk.n; i++)
            pofk.pofk[i] -= shotnoise;
    pofk.scale(params.simulation.boxsize);
    list.push_back({1.0 / a - 1.0, pofk});
}

#endif
