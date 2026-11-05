#ifndef SIMULATIONSTEP_HEADER
#define SIMULATIONSTEP_HEADER

//=============================================================================
// Part of NBodySimulation (included at the end of Simulation.h): the time-stepping (run, the force settings of a
// step, the PM workspace, the kick and drift, the lightcone, the interlaced force) and the analysis at the outputs
//=============================================================================

//=============================================================================
// Running the simulation
//=============================================================================

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::run() {
    timer.StartTiming("Timestepping");

    const auto & output_redshifts = params.output.redshifts;
    const double zini = params.ic.initial_redshift;

    // Only the spectra of the Zel'dovich field of the initial conditions (no time stepping)
    if (params.hybridbias.enabled and params.hybridbias.zeldovich_field_only) {
        for (double redshift : output_redshifts) {
            const std::string snapshot_folder =
                params.output.path("snapshot_" + params.simulation.name + "_z" + redshift_to_string(redshift));
            if (not FML::create_folder(snapshot_folder))
                throw std::runtime_error("Failed to create snapshot folder [" + snapshot_folder + "]");
            if (FML::ThisTask == 0)
                std::cout << "\n# Zel'dovich field only: hybrid bias spectra of the Zel'dovich field at z = "
                          << redshift << "\n";
            timer.StartTiming("Hybrid bias spectra");
            std::vector<double> pofk_linear;
            const auto zfield = compute_hybrid_bias_zeldovich_field(*this, redshift, snapshot_folder, &pofk_linear);
            // The Zel'dovich prediction (hybridbias_zeldovich; no one-loop here) in the same bins
            write_hybrid_bias_predictions(*this,
                                          redshift,
                                          snapshot_folder,
                                          zfield,
                                          hybrid_bias_options(*this),
                                          pofk_linear,
                                          false);
            timer.EndTiming("Hybrid bias spectra");
        }
        timer.EndTiming("Timestepping");
        timer.EndTiming("The whole simulation");
        if (FML::ThisTask == 0)
            timer.PrintAllTimings();
        return;
    }

    const auto stepper = time_stepping();

    // If we only give the total number of steps then compute how many steps between outputs
    std::vector<int> nsteps = params.timestep.nsteps;
    if (nsteps.size() == 1) {
        nsteps = stepper.steps_between_outputs(zini, output_redshifts, nsteps[0]);
        // An output with no steps would write the particles of the previous output at another redshift
        for (size_t i = 0; i < nsteps.size(); i++)
            if (not(nsteps[i] > 0 or (i == 0 and output_redshifts[0] == zini)))
                throw std::runtime_error("Splitting timestep_nsteps over the outputs gives 0 steps before output z = " +
                                         std::to_string(output_redshifts[i]) +
                                         ": use more steps or give the number of steps for every output");
    }
    stepper.print_steps(zini, output_redshifts, nsteps);

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Starting main time-stepping loop\n";
        std::cout << "#=====================================================\n\n";
    }

    int istep_total = 0;
    for (size_t ioutput = 0; ioutput < output_redshifts.size(); ioutput++) {

        // The steps to take till the next output
        const double amin = (ioutput == 0) ? 1.0 / (1.0 + zini) : 1.0 / (1.0 + output_redshifts[ioutput - 1]);
        const double amax = 1.0 / (1.0 + output_redshifts[ioutput]);
        const auto steps = stepper.steps(amin, amax, nsteps[ioutput]);

        if (nsteps[ioutput] > 0) {
            for (int istep = 0; istep <= nsteps[ioutput]; istep++) {
                // The last step is just a sync step so don't count towards the total
                if (istep < nsteps[ioutput])
                    istep_total++;

                if (FML::ThisTask == 0) {
                    std::cout << std::setprecision(5) << "\n";
                    std::cout << "#=====================================================\n";
                    std::cout << "# Step " << istep_total << " (" << istep << " / " << nsteps[ioutput]
                              << " till output " << ioutput << ")\n";
                    std::cout << "# Pos: " << steps[istep].apos << " -> " << steps[istep].apos_new
                              << "  Vel: " << steps[istep].avel << " -> " << steps[istep].avel_new << "\n";
                    std::cout << "#=====================================================\n";
                }

                // Reorder the particles in memory by position (faster density assignment and interpolation)
                const int sort_every = params.particle.sort_every_nsteps;
                if (sort_every > 0 and istep < nsteps[ioutput] and istep_total % sort_every == 0) {
                    timer.StartTiming("SortParticles");
                    const int nblocks = std::max(1, params.force.nmesh / 4);
                    for_each_species(
                        [&](auto & particles, const SIMCODE::SpeciesInfo &) { particles.sort_by_position(nblocks); });
                    timer.EndTiming("SortParticles");
                }

                step(steps[istep]);

                // Most plans are made by now (initial conditions and the first step): save them already so that
                // they are kept also if the run does not finish
                if (istep_total == 1 and istep == 0)
                    save_fftw_wisdom_if_new_plans();
            }
        }

        analyze_and_output(ioutput, output_redshifts[ioutput]);
        save_fftw_wisdom_if_new_plans();
    }
    timer.EndTiming("Timestepping");

    // Lightcone create kappa maps
    if (lightcone->lightcone_active()) {
        lightcone->create_weak_lensing_maps();
    }
    save_fftw_wisdom_if_new_plans();

    timer.EndTiming("The whole simulation");
    if (FML::ThisTask == 0) {
        timer.PrintAllTimings();
    }

#ifdef MEMORY_LOGGING
    // Simulation is over, output the memory usage (of what we log)
    FML::MemoryLog::get()->print();
#endif
}

/// Choose the force settings for a step where the force is evaluated at scale factor a: the early-time
/// settings (force_early_method) for z > force_early_until_redshift, otherwise the normal ones
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::set_force_settings_for_step(double a) {
    const auto & f = params.force;
    current_force = f;
    const bool early = f.early_method != "none" and 1.0 / a - 1.0 > f.early_until_redshift;
    if (early and f.early_method == "nufft") {
        current_force.density_method = "nufft";
        current_force.interpolation_method = "nufft";
        current_force.interlacing_order = 1;
        current_force.deconvolve_window = false;
    }
    if (early and f.early_method == "interlaced") {
        current_force.density_method = "assignment";
        current_force.interpolation_method = "assignment";
        current_force.interlacing_order = f.early_interlacing_order;
        current_force.density_assignment_method = f.early_density_assignment_method;
        current_force.deconvolve_window = true;
    }
    force_options.deconvolve = current_force.deconvolve_window;
    early_force = early;
    if (not early)
        initial_density_field.end_of_early_force();
    if (early and FML::ThisTask == 0)
        std::cout << "Using the early-time force (" << f.early_method << ") at z = " << 1.0 / a - 1.0 << "\n";
}

/// The density grid of the PM step (the grid of the previous step if it has the same size and extra slices,
/// otherwise a new one). Its content is left over from the previous step: the density assignment overwrites it
template <int NDIM, class T>
auto NBodySimulation<NDIM, T>::pm_density_workspace(int nmesh, int nleft, int nright) -> FFTWGrid<NDIM> & {
    auto & grid = pm_density_grid;
    if (grid.get_nmesh() != nmesh or grid.get_n_extra_slices_left() != nleft or
        grid.get_n_extra_slices_right() != nright) {
        grid = FFTWGrid<NDIM>(nmesh, nleft, nright);
        grid.add_memory_label("NBodySimulation::pm_density_grid");
    }
    grid.set_grid_status_real(true);
    return grid;
}

/// A grid of the step kept between steps (allocated when the layout changes; the content is that of the last use)
template <int NDIM, class T>
auto NBodySimulation<NDIM, T>::pm_workspace(const std::string & slot, int nmesh, int nleft, int nright)
    -> FFTWGrid<NDIM> & {
    auto & grid = pm_extra_grids[slot];
    if (grid.get_nmesh() != nmesh or grid.get_n_extra_slices_left() != nleft or
        grid.get_n_extra_slices_right() != nright) {
        grid = FFTWGrid<NDIM>(nmesh, nleft, nright);
        grid.add_memory_label("NBodySimulation::pm_workspace " + slot);
    }
    grid.set_grid_status_real(true);
    return grid;
}

/// Free the PM density and force grids and the other grids kept between steps (they are allocated again at the next
/// step)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::release_pm_workspace() {
    pm_density_grid.free();
    for (auto & grid : pm_force_grids)
        grid.free();
    pm_extra_grids.clear();
    if (grav)
        grav->release_workspace();
    if (lightcone)
        lightcone->release_potential_memory();
}

/// One kick-drift step: density -> force -> kick -> COLA kick -> lightcone -> COLA drift -> drift
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::step(const Step & s) {
    const auto & sim = params.simulation;
    set_force_settings_for_step(s.apos);
    const auto & force = current_force;

    // Force interlacing: average the force over copies of the particles shifted relative to the grid
    const auto nleftright = FML::INTERPOLATION::get_extra_slices(force.density_assignment_method);
    if (force.interlacing_order > 1 and s.delta_time_kick != 0.0 and not sim.enforce_LPT_trajectories) {
        kick_with_interlaced_force(s);
        return drift_and_lightcone(s);
    }

    // Compute total density field
    FFTWGrid<NDIM> & density_grid_fourier = pm_density_workspace(force.nmesh, nleftright.first, nleftright.second);
    if (s.delta_time_kick != 0.0) {
        timer.StartTiming("ComputeDensityField");
        compute_density_field_fourier(density_grid_fourier, s.apos);
        timer.EndTiming("ComputeDensityField");
        // The potential for the lightcone (from the same density field as the force)
        if (lightcone->needs_potential(s.apos, s.apos_new)) {
            timer.StartTiming("LightconePotential");
            lightcone->compute_potential(density_grid_fourier,
                                         s.apos,
                                         current_force.density_assignment_method,
                                         current_force.deconvolve_window);
            timer.EndTiming("LightconePotential");
        }
    }

    // Compute forces and kick particles (updates velocity)
    std::array<FFTWGrid<NDIM>, NDIM> & force_real = pm_force_grids;
    if (s.delta_time_kick != 0.0 and not sim.enforce_LPT_trajectories) {
        timer.StartTiming("ComputeForce");
        grav->compute_force(s.apos,
                            grav->H0_hmpc * sim.boxsize,
                            density_grid_fourier,
                            force.density_assignment_method,
                            force_real,
                            force_options);
        timer.EndTiming("ComputeForce");

        timer.StartTiming("Kick");
        if (force.interpolation_method == "nufft") {
            // Exact evaluation of the Fourier series of the force at the particle positions (one species)
            SIMCODE::kick_particles_nufft<NDIM>(part,
                                                force_real,
                                                s.delta_time_kick,
                                                s.velocity_scale,
                                                force.nufft_epsilon);
        } else {
            for_each_species([&](auto & particles, const SIMCODE::SpeciesInfo & info) {
                SIMCODE::kick_particles<NDIM>(particles,
                                              info,
                                              force_real,
                                              s.apos,
                                              s.delta_time_kick,
                                              s.velocity_scale,
                                              force.density_assignment_method,
                                              sim.boxsize);
            });
        }
        // The delta-f weights depend on the momenta
        if (has_neutrino_particles())
            part_neutrinos.update_weights();
        timer.EndTiming("Kick");
    }
    drift_and_lightcone(s);
}

/// The rest of a step after the (PM) kick: COLA kick-drift, lightcone and drift
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::drift_and_lightcone(const Step & s) {
    const auto & sim = params.simulation;
    if (sim.enforce_LPT_trajectories and FML::ThisTask == 0) {
        std::cout << "Enforcing LPT trajectories so we do not compute forces and apply (non COLA) KICK/DRIFT "
                     "operators\n";
    }

    // For COLA we can do the kick and drift at the same time. The lightcone drifts the particles from apos with
    // their true velocity (which includes the LPT velocity), so then the COLA drift (the LPT displacement over
    // the step) is done after the lightcone
    const bool make_lightcone = lightcone->lightcone_active() and s.delta_time_drift != 0.0;
    if (cola.enabled) {
        timer.StartTiming("COLA");
        // The COLA kick subtracts the LPT force from the COLA velocities (which get the PM force). When enforcing the
        // LPT trajectories there is no PM kick, so there must be no COLA kick either: the COLA velocities stay zero
        // and the particles have the LPT velocity (with it the velocities were close to the initial LPT velocity)
        const double delta_time_kick_cola = sim.enforce_LPT_trajectories ? 0.0 : s.delta_time_kick;
        for_each_cb_species([&](MPIParticles & particles, const SIMCODE::SpeciesInfo &) {
            if (make_lightcone)
                cola.kick(particles, s.apos, s.apos_new, delta_time_kick_cola);
            else
                cola.kick_drift(particles, s.apos, s.apos_new, delta_time_kick_cola, s.delta_time_drift);
        });
        if (not make_lightcone and not cola.scaledependent and sim.enforce_LPT_trajectories)
            part.communicate_particles();
        timer.EndTiming("COLA");
    }

    // Build the lightcone (with the true velocities of the particles)
    if (make_lightcone) {
        ScopedLPTVelocity<NDIM, T> true_velocities(cola, part, s.avel_new, &timer);
        timer.StartTiming("Lightcone");
        // The potential of this step (if not computed with the force, e.g. with force interlacing: from the
        // CDM+baryon density only)
        if (lightcone->needs_potential(s.apos, s.apos_new) and not lightcone->has_potential(s.apos)) {
            const auto nleftright = FML::INTERPOLATION::get_extra_slices(current_force.density_assignment_method);
            FFTWGrid<NDIM> & density =
                pm_workspace("lightcone density", current_force.nmesh, nleftright.first, nleftright.second);
            assign_cb_particles_to_grid(density, s.apos, current_force);
            timer.StartTiming("LightconePotential");
            lightcone->compute_potential(
                density, s.apos, current_force.density_assignment_method, current_force.deconvolve_window);
            timer.EndTiming("LightconePotential");
        }
        lightcone->create_lightcone(
            part, s.apos, s.apos_new, s.avel_new, s.delta_time_drift, current_force.density_assignment_method);
        timer.EndTiming("Lightcone");
    }
    if (cola.enabled and make_lightcone) {
        timer.StartTiming("COLA");
        for_each_cb_species(
            [&](MPIParticles & particles, const SIMCODE::SpeciesInfo &) { cola.drift(particles, s.apos, s.apos_new); });
        if (not cola.scaledependent and sim.enforce_LPT_trajectories)
            part.communicate_particles();
        timer.EndTiming("COLA");
    }

    // Drift particles (updates positions)
    if (s.delta_time_drift != 0.0 and not sim.enforce_LPT_trajectories) {
        timer.StartTiming("Drift");
        for_each_species([&](auto & particles, const SIMCODE::SpeciesInfo & info) {
            SIMCODE::drift_particles<NDIM>(particles, info, s.apos, s.apos_new, s.delta_time_drift, sim.boxsize);
        });
        timer.EndTiming("Drift");
    }

    // Show info about particles and memory use
    for_each_species([&](auto & particles, const SIMCODE::SpeciesInfo &) { particles.info(); });
    FML::print_system_memory_use();
}

/// The kick with force interlacing: the force is computed for force_interlacing_order copies of the particles
/// shifted by a fraction of a cell relative to the grid (see FML::INTERPOLATION::generate_interlacing_shifts),
/// interpolated to the shifted particles and averaged. This cancels the leading aliasing terms of both the
/// density assignment and the force interpolation
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::kick_with_interlaced_force(const Step & s) {
    const auto & sim = params.simulation;
    const auto & force = current_force;
    const auto nleftright = FML::INTERPOLATION::get_extra_slices(force.density_assignment_method);
    const auto shifts =
        FML::INTERPOLATION::generate_interlacing_shifts<NDIM>(force.interlacing_order, 1.0 / double(force.nmesh));
    // With one task the shifted particles can be wrapped in all directions. Otherwise not in x (they stay on
    // their task and the grid has one extra slice on the right for the ones that end up outside the domain)
    const int wrap_type = FML::NTasks == 1 ? 1 : 2;

    auto * p = part.get_particles_ptr();
    const size_t npart = part.get_npart();
    std::array<std::vector<FML::GRID::FloatType>, NDIM> force_sum;
    for (auto & f : force_sum)
        f.assign(npart, 0.0);
    // The positions are restored exactly after each shift (shifting back changes them in the last digit)
    const auto saved_pos = FML::PARTICLE::save_particle_positions<T, NDIM>(p, npart);
    for (auto & shift : shifts) {
        FML::PARTICLE::shift_particles<T, NDIM>(p, npart, shift, +1.0, wrap_type);
        density_shift = shift;

        FFTWGrid<NDIM> & density_grid_fourier =
            pm_density_workspace(force.nmesh, nleftright.first, nleftright.second + 1);
        timer.StartTiming("ComputeDensityField");
        compute_density_field_fourier(density_grid_fourier, s.apos);
        timer.EndTiming("ComputeDensityField");

        timer.StartTiming("ComputeForce");
        std::array<FFTWGrid<NDIM>, NDIM> & force_real = pm_force_grids;
        grav->compute_force(s.apos,
                            grav->H0_hmpc * sim.boxsize,
                            density_grid_fourier,
                            force.density_assignment_method,
                            force_real,
                            force_options);
        timer.EndTiming("ComputeForce");

        timer.StartTiming("Kick");
        for (int idim = 0; idim < NDIM; idim++)
            force_real[idim].communicate_boundaries();
        const double norm = 1.0 / double(shifts.size());
        auto add_force = [&](size_t i, const std::array<FML::GRID::FloatType, NDIM> & f) {
            for (int idim = 0; idim < NDIM; idim++)
                force_sum[idim][i] += f[idim] * norm;
            return 0.0;
        };
        FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions_apply<NDIM, T>(
            force_real,
            p,
            npart,
            force.density_assignment_method,
            add_force);
        timer.EndTiming("Kick");

        FML::PARTICLE::restore_particle_positions<T, NDIM>(p, npart, saved_pos);
    }
    density_shift = {};

    // Kick: v -> velocity_scale * v - F dt
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < npart; i++) {
        auto * vel = FML::PARTICLE::GetVel(p[i]);
        for (int idim = 0; idim < NDIM; idim++)
            vel[idim] = vel[idim] * s.velocity_scale - force_sum[idim][i] * s.delta_time_kick;
    }
}

/// Do all the on the fly analysis that is turned on and write the output
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::analyze_and_output(int ioutput, double redshift) {

    // The analysis allocates its own grids
    release_pm_workspace();

    // Create snapshot folder
    const std::string snapshot_folder =
        params.output.path("snapshot_" + params.simulation.name + "_z" + redshift_to_string(redshift));
    if (not FML::create_folder(snapshot_folder)) {
        throw std::runtime_error("Failed to create snapshot folder [" + snapshot_folder + "]");
    }

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Analyze and output\n";
        std::cout << "#=====================================================\n";
        std::cout << "Doing output " << ioutput << " at redshift z = " << redshift << "\n";
    }

    // In the COLA frame the velocities are relative to LPT. While this object exists the
    // particles have their true velocities
    ScopedLPTVelocity<NDIM, T> true_velocities(cola, part, 1.0 / (1.0 + redshift), &timer);
    std::optional<ScopedLPTVelocity<NDIM, T>> true_velocities_baryons;
    if (has_baryons())
        true_velocities_baryons.emplace(cola, part_baryon, 1.0 / (1.0 + redshift), &timer);

    // Every step we (might) have computed Pcb(k) and possibly Ptotal(k). Output these to file
    // (only the steps since the last output: the files of earlier steps are already written)
    if (pofk_cb_every_step.size() > 0 or pofk_total_every_step.size() > 0) {
        output_pofk_for_every_step(*this);
        pofk_cb_every_step.clear();
        pofk_total_every_step.clear();
        pofk_species_every_step.clear();
    }

    struct Analysis {
        std::string name;
        bool enabled;
        std::function<void()> run;
    };
    const std::vector<Analysis> analyses{
        {"Baryonification",
         params.baryonification.enabled,
         [&]() {
             if constexpr (NDIM == 3)
                 SIMCODE::output_baryonified_snapshot(
                     SIMCODE::copy_baryonification_particles(part.get_particles_ptr(), part.get_npart()),
                     params,
                     cosmo,
                     redshift,
                     snapshot_folder);
         }},
        {"Power-spectrum",
         params.pofk.enabled,
         [&]() { compute_power_spectrum(*this, redshift, snapshot_folder); }},
        {"Power-spectrum multipoles",
         params.pofk_multipole.enabled,
         [&]() { compute_power_spectrum_multipoles(*this, redshift, snapshot_folder); }},
        {"FOF", params.fof.enabled, [&]() { compute_fof_halos(*this, redshift, snapshot_folder); }},
        {"Bispectrum", params.bispectrum.enabled, [&]() { compute_bispectrum(*this, redshift, snapshot_folder); }},
        {"Hybrid bias spectra",
         params.hybridbias.enabled,
         [&]() { compute_hybrid_bias_spectra(*this, redshift, snapshot_folder); }},
        {"Output particles",
         params.output.particles,
         [&]() { output_particles(*this, redshift, snapshot_folder); }},
    };
    for (auto & analysis : analyses) {
        if (not analysis.enabled)
            continue;
        timer.StartTiming(analysis.name);
        analysis.run();
        timer.EndTiming(analysis.name);
    }
}

#endif
