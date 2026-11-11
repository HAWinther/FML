#ifndef SIMULATIONINIT_HEADER
#define SIMULATIONINIT_HEADER

//=============================================================================
// Part of NBodySimulation (included at the end of Simulation.h): reading the parameters, the initial conditions
// (the background and growth output, the initial power spectrum, the particles of each species and their velocities,
// the N-body gauge source, reading the IC from file) and the FFTW plans
//=============================================================================

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::read_parameters(const ParameterMap & param, const Parameters & _params) {
    parameters = std::make_shared<ParameterMap>(param);
    params = _params;
    // Library callers can construct the simulation without Main.cpp's model checks.
    if (params.force.short_range.enabled) {
        params.validate();
        if (cosmo->get_OmegaMNu() != 0.0 || SIMCODE::get_parameter<bool>(param, "force_use_finite_difference_force"))
            throw std::invalid_argument(
                "Short-range integration currently excludes massive neutrinos and finite-difference forces");
        grav->validate_short_range_support();
        grav->short_range_interaction(params.ic.a_initial());
        if (!FML::PARTICLE::has_get_id<T>())
            throw std::invalid_argument("Short-range integration requires particle IDs");
    }

    // We cannot use COLA if the particle type is not compatible with it
    if (params.simulation.use_cola and not FML::PARTICLE::has_get_D_1LPT<T>()) {
        throw std::runtime_error("Set to use COLA, but particle-type does not even have 1LPT growthfactors!");
    }

    force_options.laplace_kernel = FML::NBODY::laplace_kernel_from_name(params.force.greens_function_kernel);

    // The fiducial gradient kernel: the continuous kernel (D = ik) is the most accurate when the force grid is
    // the same as the particle grid. When the force grid is finer the regular particle lattice (of the IC) puts
    // power just below the Nyquist frequency of the grid where ik is large, which gives large spurious forces
    // (tens of percent errors in the growth of individual large-scale modes and too much small-scale power).
    // The finite difference kernels (which vanish at the Nyquist frequency) suppress this. Together with the
    // half-cell lattice offset (ic_lattice_offset = fiducial) the four-point kernel gives the most accurate
    // nonlinear P(k) for force grids up to 3 times the particle grid and the two-point kernel for finer grids
    // (see the README)
    std::string gradient_kernel = params.force.gradient_kernel;
    if ((gradient_kernel == "fiducial" or gradient_kernel == "") and params.force.nmesh > params.particle.Npart_1D) {
        gradient_kernel = params.force.nmesh <= 3 * params.particle.Npart_1D ? "discrete_4pt" : "discrete_2pt";
        if (FML::ThisTask == 0)
            std::cout << "NB: force_nmesh > particle_Npart_1D so we use force_gradient_kernel = " << gradient_kernel
                      << " (set force_gradient_kernel to continuous, discrete_2pt or discrete_4pt to change this)\n";
    }
    force_options.gradient_kernel = FML::NBODY::gradient_kernel_from_name(gradient_kernel);
    force_options.deconvolve = params.force.deconvolve_window;
    current_force = params.force;

    // The (experimental) finite difference force: only the stencils 2, 4 and 6 are implemented and the gradient is
    // the finite difference stencil, so an explicitly chosen Fourier gradient kernel would be ignored
    if (SIMCODE::get_parameter<bool>(param, "force_use_finite_difference_force")) {
        const int stencil_order = SIMCODE::get_parameter<int>(param, "force_finite_difference_stencil_order");
        if (not FML::NBODY::is_valid_finite_difference_stencil_order(stencil_order))
            throw std::runtime_error("Parameterfile: force_finite_difference_stencil_order = " +
                                     std::to_string(stencil_order) + " is not implemented. Valid options are: 2 4 6");
        if (params.force.gradient_kernel != "fiducial" and params.force.gradient_kernel != "")
            throw std::runtime_error("Parameterfile: force_gradient_kernel = " + params.force.gradient_kernel +
                                     " is not used with force_use_finite_difference_force = true (the gradient is "
                                     "the finite difference stencil set by force_finite_difference_stencil_order). "
                                     "Use force_gradient_kernel = fiducial");
    }

    cola.enabled = params.simulation.use_cola;
    cola.scaledependent = params.simulation.use_scaledependent_cola;
    cola.LPT_order = params.simulation.cola_LPT_order;
    cola.aini = params.ic.a_initial();
    cola.H0Box = grav->H0_hmpc * params.simulation.boxsize;
    cola.grav = grav;
}

//=============================================================================
// Initial conditions
//=============================================================================

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::init() {

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Create initial conditions\n";
        std::cout << "#=====================================================\n";
    }

    // Empty output_folder means to output to current directory. Otherwise make folder
    if (params.output.folder != "") {
        if (not FML::create_folder(params.output.folder)) {
            throw std::runtime_error("Cannot create output directory [" + params.output.folder + "]");
        }
    }

    setup_fftw();

    output_background_and_growth_functions();

    // The particle species (CDM+baryons as one, or CDM and baryons, and massive neutrinos)
    species = SIMCODE::make_species_list(params.particle, *cosmo, NDIM);
    if (FML::ThisTask == 0 and species.size() > 1) {
        std::cout << "# Particle species (name, Npart_1D, GADGET type, Omega_i / OmegaM, weight in delta_cb):\n";
        for (auto & info : species)
            std::cout << "#   " << info.name << " " << info.Npart_1D << " " << info.gadget_type << " "
                      << info.mass_fraction << " " << info.weight_cb << "\n";
    }
    if (has_neutrino_particles()) {
        const auto & masses = cosmo->get_neutrino_masses_eV();
        const double Tnu0_eV = cosmo->get_neutrino_temperature_eV(1.0);
        part_neutrinos.init(cosmo->get_massive_neutrino_species(),
                            params.simulation.boxsize,
                            params.particle.neutrino_deltaf);
        if (FML::ThisTask == 0) {
            std::cout << "# Massive neutrinos as particles (delta-f weights: " << std::boolalpha
                      << params.particle.neutrino_deltaf << "). m_nu =";
            for (double m : masses)
                std::cout << " " << m;
            std::cout << " eV (one set of particles for all species), T_nu0 = " << Tnu0_eV
                      << " eV, f_nu = " << cosmo->get_fMNu() << "\n";
        }
    }

    // If we have a MG model and want exactly the same IC as for LCDM
    // we can supply LCDM P(k) and use the GR growth factors to scale it back
    if (params.ic.use_gravity_model_GR) {
        grav_ic = std::make_shared<GravityModelGR<NDIM>>(cosmo);
        grav_ic->set_transfers(grav->get_transfers()); // (loaded once in Main.cpp)
        grav_ic->read_parameters(*parameters);
        grav_ic->init();
    } else {
        grav_ic = grav;
    }

    // A modified gravity model with a fitted nonlinear boost over GR (f(R)) also needs the linear growth of GR for the
    // nonlinear predictions in the P(k) outputs (nonlinear_over_linear_prediction)
    if (grav->nonlinear_boost_over_GR(0.0)) {
        if (params.ic.use_gravity_model_GR) {
            grav_gr = grav_ic;
        } else {
            grav_gr = std::make_shared<GravityModelGR<NDIM>>(cosmo);
            grav_gr->set_transfers(grav->get_transfers());
            grav_gr->read_parameters(*parameters);
            grav_gr->init();
        }
    }

    setup_initial_power_spectrum();

    generate_initial_conditions();

    // The N-body gauge source needs the initial density field of the particles
    if (params.force.nbody_gauge)
        setup_nbody_gauge();

    // In the COLA frame v=0 so reset velocities. With CDM and baryons as separate particles the initial conditions
    // already have the velocities relative to the shared COLA frame (the baryon-CDM relative velocity)
    if (cola.enabled and not has_baryons() and not ic_velocities_relative_to_cola_frame) {
        cola_initialize_velocities<NDIM, T>(part);
    }

    // Set up the lightcone
    std::vector<std::string> particle_types_lightcone{"cb"};
    lightcone = std::make_shared<Lightcone<NDIM, T>>(cosmo);
    lightcone->set_parameters(params);
    lightcone->init(particle_types_lightcone);
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::output_background_and_growth_functions() {
    if (FML::ThisTask != 0)
        return;
    const std::string cosmoprefix = params.output.path("cosmology_" + params.simulation.name);
    const std::string gravprefix = params.output.path("gravitymodel_" + params.simulation.name);
    cosmo->output(cosmoprefix + ".txt");
    grav->output(gravprefix + "_k0.00001.txt", 1e-5 / grav->H0_hmpc);
    grav->output(gravprefix + "_k0.0001.txt", 1e-4 / grav->H0_hmpc);
    grav->output(gravprefix + "_k0.001.txt", 1e-3 / grav->H0_hmpc);
    grav->output(gravprefix + "_k0.01.txt", 1e-2 / grav->H0_hmpc);
    grav->output(gravprefix + "_k0.1.txt", 1e-1 / grav->H0_hmpc);
    grav->output(gravprefix + "_k1.0.txt", 1e+0 / grav->H0_hmpc);
    grav->output(gravprefix + "_k10.0.txt", 1e+1 / grav->H0_hmpc);
}

//=============================================================================
/// Read the input P(k) / T(k), bring it to the initial redshift, make splines
/// and (optionally) normalize it to a given sigma8
//=============================================================================
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::setup_initial_power_spectrum() {
    const auto & ic = params.ic;
    const double boxsize = params.simulation.boxsize;

    auto table = SIMCODE::read_initial_power_spectrum<NDIM>(ic, cosmo, grav, grav_ic, transfers);
    auto make_splines = [&](std::string label) {
        power_initial_spline.create(table.k, table.power, "P(k,zini)" + label);
        power_primordial_spline.create(table.k, table.power_primordial, "P_primordial(k)" + label);
    };
    make_splines("");

    // The splines are constant beyond the table, so modes of the IC grid with k > k_max of the input would silently get
    // P(k_max). Only a warning: the power there is often negligible (and a table only slightly too short is common)
    const double kmax_grid = std::sqrt(double(NDIM)) * M_PI * ic.nmesh / boxsize;
    if (FML::ThisTask == 0 and ic.random_field_type != SIMCODE::ICRandomFieldType::ReadParticles and
        not table.k.empty() and table.k.back() < kmax_grid)
        std::cout << "WARNING: the input P(k) / T(k) only goes to k = " << table.k.back()
                  << " h/Mpc, but the IC grid has modes up to sqrt(NDIM) pi ic_nmesh / simulation_boxsize = "
                  << kmax_grid << " h/Mpc. P(k) is taken to be constant (P(k_max)) beyond the table\n";

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Testing splines: \n";
        std::cout << "#=====================================================\n";
        for (double k : {0.01, 0.1, 1.0, 10.0})
            std::cout << "k = " << std::setw(5) << k << ": P(k,zini) = " << power_initial_spline(k) << " (Mpc/h)^3\n";
        FML::assert_mpi(power_initial_spline(0.01) == power_initial_spline(0.01), "NaN detected");
    }

    // sigma(R,z) from the current P(k,zini)
    auto sigma_of_R = [&](double R_mpch, double redshift) {
        return SIMCODE::sigma_of_R<NDIM>(power_initial_spline, grav_ic, boxsize, ic.initial_redshift, redshift, R_mpch);
    };
    const double sigma8_today = sigma_of_R(8.0, 0.0);

    // Normalize P(k) to a given sigma8 value. We then need to update the splines,
    // the transfer data and the cosmology
    double pofk_norm_factor = 1.0;
    if (ic.sigma8_normalization) {
        const double sigma8 = sigma_of_R(8.0, ic.sigma8_redshift);
        pofk_norm_factor = std::pow(ic.sigma8 / sigma8, 2);
    }
    if (pofk_norm_factor != 1.0) {
        table.scale(pofk_norm_factor);
        make_splines(" sigma8 normalized");

        // Update the transfer functions: the one object (loaded in Main.cpp) of the gravity models (grav and grav_ic),
        // the initial conditions and the analysis
        if (transfers)
            transfers->rescale_primordial_power(pofk_norm_factor);
        cosmo->set_As(cosmo->get_As() * pofk_norm_factor);

        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "Updated cosmology and gravitymodel to reflect change of As:\n";
            std::cout << "#=====================================================\n";
            cosmo->info();
            grav->info();
        }
    }

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Sigma(R = 8 Mpc/h, z = 0.0 ) : " << std::setw(15) << sigma8_today << "\n";
        if (ic.sigma8_normalization) {
            std::cout << "# We will normalize to Sigma8 = " << std::setw(15) << ic.sigma8 << " at z = " << std::setw(15)
                      << ic.sigma8_redshift << "\n";
            std::cout << "# New Sigma(R = 8 Mpc/h, z = 0.0 ) : " << std::setw(15)
                      << sigma8_today * std::sqrt(pofk_norm_factor) << "\n";
            std::cout << "# Sigma(R = Box/ic_nmesh, z = zini) : " << std::setw(15)
                      << sigma_of_R(boxsize / ic.nmesh, ic.initial_redshift) << "\n";
        }
        std::cout << "#=====================================================\n";
    }
}

//=============================================================================
/// Create the particles: either generate the initial conditions from a random
/// field (gaussian / non-gaussian / phases taken from a file) or read them from file
//=============================================================================
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::generate_initial_conditions() {
    // If we simply read IC from file (useful for testing)
    if (params.ic.random_field_type == SIMCODE::ICRandomFieldType::ReadParticles) {
        read_ic();
        return;
    }

    const FFTWGrid<NDIM> delta_ini_fourier = make_initial_density_field();

    // Massive neutrino particles (made from the same random field)
    if (has_neutrino_particles())
        generate_neutrino_particles(delta_ini_fourier);

    timer.StartTiming("InitialConditions");
    if (has_baryons())
        generate_initial_conditions_two_species(delta_ini_fourier);
    else
        generate_initial_conditions_one_species(delta_ini_fourier);
    apply_lattice_offset();
    timer.EndTiming("InitialConditions");
}

/// The initial CDM+baryon density field delta(k, zini) (and the linear field for the hybrid bias spectra)
template <int NDIM, class T>
FML::GRID::FFTWGrid<NDIM> NBodySimulation<NDIM, T>::make_initial_density_field() {
    const auto & ic = params.ic;
    const double boxsize = params.simulation.boxsize;

    auto nextra = FML::INTERPOLATION::get_extra_slices("CIC");
    if (ic.random_field_type == SIMCODE::ICRandomFieldType::ReadPhases) {
        // If read_phases then delta_ini is computed from particles
        nextra = FML::INTERPOLATION::get_extra_slices(ic.reconstruct_assigment_method);
    }
    FFTWGrid<NDIM> delta_ini_fourier(ic.nmesh, nextra.first, nextra.second);
    delta_ini_fourier.add_memory_label("delta_ini_fourier(k)");
    delta_ini_fourier.set_grid_status_real(false);

    timer.StartTiming("InitialDensityField");
    auto rng = SIMCODE::make_random_generator(ic);
    SIMCODE::generate_initial_density_field<NDIM>(delta_ini_fourier,
                                                  ic,
                                                  boxsize,
                                                  power_initial_spline,
                                                  power_primordial_spline,
                                                  cosmo,
                                                  grav_ic,
                                                  rng.get());
    timer.EndTiming("InitialDensityField");

    // Take the phases from the particles in a file, but use our P(k) for the amplitudes
    if (ic.random_field_type == SIMCODE::ICRandomFieldType::ReadPhases) {
        SIMCODE::delta_from_gadget_particles<NDIM, T>(delta_ini_fourier, params, *cosmo);
        FML::RANDOM::GAUSSIAN::whiten_and_set_power_spectrum<NDIM>(delta_ini_fourier, [&](double kBox) {
            return power_initial_spline(kBox / boxsize) / std::pow(boxsize, NDIM);
        });
    }

    // Reverse the phases? For doing pair-fixed simulations
    if (ic.reverse_phases) {
        FML::RANDOM::GAUSSIAN::reverse_phases<NDIM>(delta_ini_fourier);
    }

    // Keep the linear field for the hybrid bias spectra (the Lagrangian operators are formed at the lattice points so
    // we only need the modes the lattice resolves)
    if (params.hybridbias.enabled) {
        hybridbias_linear_field_fourier = FFTWGrid<NDIM>(params.particle.Npart_1D);
        hybridbias_linear_field_fourier.add_memory_label("delta_L(k,zini) for hybrid bias");
        FML::NUFFT::copy_fourier_modes<NDIM>(delta_ini_fourier,
                                             hybridbias_linear_field_fourier,
                                             [](const std::array<int, NDIM> &) { return 1.0; });
    }
    return delta_ini_fourier;
}

/// The massive neutrino particles from the initial density field (and the transfer functions)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::generate_neutrino_particles(const FFTWGrid<NDIM> & delta_ini_fourier) {
    const auto & ic = params.ic;
    timer.StartTiming("InitialConditionsNeutrinos");
    FML::assert_mpi(transfers != nullptr, "particle_neutrinos needs the transfer functions (camb or class)");
    const auto & nu = SIMCODE::find_species(species, SIMCODE::ParticleSpecies::MassiveNeutrino);
    part_neutrinos.generate(delta_ini_fourier,
                            *transfers,
                            ic.initial_redshift,
                            cosmo->HoverH0_of_a(ic.a_initial()),
                            nu.Npart_1D,
                            params.particle.allocation_factor,
                            ic.random_seed,
                            nu.id_offset);
    timer.EndTiming("InitialConditionsNeutrinos");
}

//=============================================================================
// The IC velocities. The velocity of LPT order n is fac f_n Psi_n with fac = a^2 H / H0 at zini and f_n the growth
// rate of the IC (grav_ic). With COLA the velocities in the particles are relative to the COLA frame, which has the
// growth rates of the gravity model (grav): at k = 0 for scale-independent COLA, f(k) for scale-dependent COLA, and
// only the LPT orders up to simulation_cola_LPT_order. With ic_use_gravity_model_GR and a modified gravity model the
// IC have the growth rate of GR (grav_ic) but the COLA frame that of the gravity model (grav).
//=============================================================================

/// fac {f_1LPT, f_2LPT, f_3LPTa, f_3LPTb} of the IC at k = 0
template <int NDIM, class T>
std::vector<double> NBodySimulation<NDIM, T>::ic_velocity_norms(double fac) const {
    const double aini = params.ic.a_initial();
    return {fac * grav_ic->get_f_1LPT(aini),
            fac * grav_ic->get_f_2LPT(aini),
            fac * grav_ic->get_f_3LPTa(aini),
            fac * grav_ic->get_f_3LPTb(aini)};
}

/// The IC have the growth of GR (ic_use_gravity_model_GR) and the COLA frame that of the modified gravity model
template <int NDIM, class T>
bool NBodySimulation<NDIM, T>::cola_frame_growth_differs() const {
    return cola.enabled and grav_ic != grav and grav_ic->get_name() != grav->get_name();
}

/// The scale-dependent 2LPT velocity norm (nullptr if it is that of k = 0): f_2LPT(k, zini) Psi_2LPT with k the
/// wavenumber of the 2LPT field (as in the COLA frame of scale-dependent COLA, which therefore needs nothing extra).
/// Using f_2LPT(k = 0) for all k gave velocities that differed from those of scale-dependent COLA by 4.3e-4 rms for
/// 0.4 eV neutrinos. With scale-independent COLA the frame has f_2LPT(0) Psi_2LPT and the particles get the residual.
/// In the N-body gauge only the 1LPT growth rate differs from that of the gravity model (on horizon scales, from the
/// transfer functions); the 2LPT velocity is that of the gravity model in both cases (as in the COLA frame). With
/// simulation_cola_LPT_order = 1 the frame has no 2LPT velocity and the particles get all of it. With
/// ic_use_gravity_model_GR and modified gravity the residual relative to the frame of the gravity model. The same for
/// all species (the 2LPT displacement is that of the total CDM+baryon field)
template <int NDIM, class T>
std::function<double(double)> NBodySimulation<NDIM, T>::ic_velocity_norm_2LPT_of_kBox(double fac) const {
    const auto & ic = params.ic;
    const double boxsize = params.simulation.boxsize;
    const double aini = ic.a_initial();
    const bool cola_frame_has_2LPT = cola.enabled and cola.LPT_order >= 2;
    if (not(ic.LPT_order >= 2 and
            ((grav_ic->is_growth_scaledependent() and (not cola_frame_has_2LPT or not cola.scaledependent)) or
             (cola_frame_has_2LPT and cola_frame_growth_differs()))))
        return nullptr;
    if (FML::ThisTask == 0 and grav_ic->is_growth_scaledependent())
        std::cout << "# Scale-dependent 2LPT growth rate at zini f_2(k) = "
                  << grav_ic->get_f_2LPT(aini, 0.01 / grav_ic->H0_hmpc) << " (k = 0.01 h/Mpc), "
                  << grav_ic->get_f_2LPT(aini, 0.1 / grav_ic->H0_hmpc) << " (k = 0.1 h/Mpc), "
                  << grav_ic->get_f_2LPT(aini, 1.0 / grav_ic->H0_hmpc) << " (k = 1 h/Mpc). Scale-independent value "
                  << grav_ic->get_f_2LPT(aini) << "\n";
    const double f2_frame0 = cola_frame_has_2LPT ? grav->get_f_2LPT(aini) : 0.0;
    const bool frame_scaledependent = cola_frame_has_2LPT and cola.scaledependent;
    auto g = grav;
    auto g_ic = grav_ic;
    return [g, g_ic, fac, aini, boxsize, f2_frame0, frame_scaledependent](double kBox) {
        const double f2_frame = frame_scaledependent ? g->get_f_2LPT(aini, kBox / boxsize / g->H0_hmpc) : f2_frame0;
        return fac * (g_ic->get_f_2LPT(aini, kBox / boxsize / g_ic->H0_hmpc) - f2_frame);
    };
}

/// The scale-dependent 1LPT velocity norm of one species (nullptr if it is that of k = 0). With COLA the velocities
/// it gives are relative to the COLA frame.
/// N-body gauge: the growth rate (and so the 1LPT velocity) is scale-dependent on large scales; we take it from the
/// (N-body gauge) transfer functions.
/// Scale-dependent growth (massive neutrinos with the transfer functions, modified gravity): the 1LPT velocity is
/// f(k, zini) Psi (the growth rate at k = 0 for all k overestimated the velocity: for 0.4 eV neutrinos f = 0.998
/// instead of f_cb = 0.976 for k >> k_fs, which gave +1.7% in P_cb(z = 0) on linear scales). With scale-dependent COLA
/// the COLA frame already has this velocity; with scale-independent COLA it has f(k = 0) Psi and the particles get the
/// residual (f(k) - f(0)) Psi. With ic_use_gravity_model_GR the particles get the residual (f_IC(k) - f_frame(k)) Psi
/// for the orders the frame has. Zeroing the velocities (or taking the frame growth rate to be that of the IC) started
/// COLA with the velocity of the gravity model and the run without COLA with that of GR
template <int NDIM, class T>
std::function<double(double)> NBodySimulation<NDIM, T>::ic_velocity_norm_1LPT_of_kBox(double fac) {
    const double boxsize = params.simulation.boxsize;
    const double aini = params.ic.a_initial();
    auto g = grav;
    auto g_ic = grav_ic;
    const bool cola_enabled = cola.enabled;
    const bool cola_scaledependent = cola.scaledependent;

    if (params.force.nbody_gauge) {
        FML::assert_mpi(transfers != nullptr, "force_nbody_gauge needs the transfer functions");
        const double kmin = transfers->range().kmin_hmpc;
        const double kmax = transfers->range().kmax_hmpc;
        const auto & tr = *transfers;
        const Spline growth_rate_of_logk = SIMCODE::spline_of_logk(
            kmin,
            kmax,
            [&](double logk) {
                return tr.density_growth_rate(SIMCODE::TransferSpecies::CDMBaryon, std::exp(logk), aini);
            },
            "f_cb(k,zini)");
        if (FML::ThisTask == 0)
            std::cout << "# N-body gauge: growth rate at zini f(k) = " << growth_rate_of_logk(std::log(kmin))
                      << " (k = " << kmin << " h/Mpc), " << growth_rate_of_logk(std::log(0.1)) << " (k = 0.1 h/Mpc). "
                      << "Scale-independent value " << grav_ic->get_f_1LPT(aini) << "\n";
        // With COLA the velocities are relative to the COLA frame, which has the growth rate of the gravity model
        // (at k = 0 for scale-independent COLA) for 1LPT and all of the 2LPT velocity. Zeroing the velocities (as for
        // scale-independent growth) would replace f_nbg(k) by f(0) (0.75% too large velocity on the largest scales)
        return
            [g, growth_rate_of_logk, fac, aini, boxsize, kmin, kmax, cola_enabled, cola_scaledependent](double kBox) {
                const double logk = SIMCODE::clamped_logk(kBox / boxsize, kmin, kmax);
                const double f_frame = not cola_enabled    ? 0.0 :
                                       cola_scaledependent ? g->get_f_1LPT(aini, kBox / boxsize / g->H0_hmpc) :
                                                             g->get_f_1LPT(aini);
                return fac * (growth_rate_of_logk(logk) - f_frame);
            };
    }

    if (not((grav_ic->is_growth_scaledependent() and not(cola_enabled and cola_scaledependent)) or
            cola_frame_growth_differs()))
        return nullptr;
    if (FML::ThisTask == 0 and cola_frame_growth_differs())
        std::cout << "# The IC have the growth rate of GR at zini f = " << grav_ic->get_f_1LPT(aini)
                  << ", the COLA frame that of " << grav->get_name()
                  << " f(k) = " << grav->get_f_1LPT(aini, 0.01 / grav->H0_hmpc) << " (k = 0.01 h/Mpc), "
                  << grav->get_f_1LPT(aini, 1.0 / grav->H0_hmpc) << " (k = 1 h/Mpc): the particles get the "
                  << "difference\n";
    if (FML::ThisTask == 0 and grav_ic->is_growth_scaledependent())
        std::cout << "# Scale-dependent growth rate at zini f(k) = "
                  << grav_ic->get_f_1LPT(aini, 0.01 / grav_ic->H0_hmpc) << " (k = 0.01 h/Mpc), "
                  << grav_ic->get_f_1LPT(aini, 0.1 / grav_ic->H0_hmpc) << " (k = 0.1 h/Mpc), "
                  << grav_ic->get_f_1LPT(aini, 1.0 / grav_ic->H0_hmpc) << " (k = 1 h/Mpc). Scale-independent value "
                  << grav_ic->get_f_1LPT(aini) << "\n";
    const double f0 = cola_enabled ? grav->get_f_1LPT(aini) : 0.0;
    const bool frame_scaledependent = cola_enabled and cola_scaledependent;
    return [g, g_ic, fac, aini, boxsize, f0, frame_scaledependent](double kBox) {
        const double f_frame = frame_scaledependent ? g->get_f_1LPT(aini, kBox / boxsize / g->H0_hmpc) : f0;
        return fac * (g_ic->get_f_1LPT(aini, kBox / boxsize / g_ic->H0_hmpc) - f_frame);
    };
}

/// Scale-dependent COLA: the LPT potentials phi_1LPT and phi_2LPT (k, zini) of the CDM+baryon field (the COLA
/// frame). phi_2LPT_ini_fourier: the 2LPT potential if the IC generator returned it (else it is computed here)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::store_cola_potentials(const FFTWGrid<NDIM> & delta_cb_fourier,
                                                     FFTWGrid<NDIM> * phi_2LPT_ini_fourier) {
    if (not(cola.enabled and cola.scaledependent))
        return;
    if (FML::PARTICLE::has_get_D_1LPT<T>() and cola.LPT_order >= 1) {
        cola.phi_1LPT_ini_fourier = FFTWGrid<NDIM>();
        FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier<NDIM>(delta_cb_fourier, cola.phi_1LPT_ini_fourier);
        cola.phi_1LPT_ini_fourier.add_memory_label("phi_1LPT(k,zini)");
    }
    if (FML::PARTICLE::has_get_D_2LPT<T>() and cola.LPT_order >= 2 and params.ic.LPT_order >= 2) {
        if (FML::ThisTask == 0)
            std::cout << "Storing phi_2LPT(k,zini) for scale-dependent COLA\n";
        if (phi_2LPT_ini_fourier)
            cola.phi_2LPT_ini_fourier = std::move(*phi_2LPT_ini_fourier);
        else
            FML::COSMOLOGY::LPT::compute_2LPT_potential_fourier(delta_cb_fourier, cola.phi_2LPT_ini_fourier);
        cola.phi_2LPT_ini_fourier.add_memory_label("phi_2LPT(k,zini)");
    }
}

/// The initial conditions with one CDM+baryon species: the particles displaced from a regular grid with LPT, the
/// velocities from the growth rates (see ic_velocity_norm_1LPT_of_kBox and ic_velocity_norm_2LPT_of_kBox)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::generate_initial_conditions_one_species(const FFTWGrid<NDIM> & delta_cb_fourier) {
    const auto & ic = params.ic;
    const double aini = ic.a_initial();
    const double fac = aini * aini * cosmo->HoverH0_of_a(aini);
    std::vector<double> velocity_norms = ic_velocity_norms(fac);

    // The IC generator returns the 2LPT potential if the vector has room for it (for scale-dependent COLA)
    const bool store_2LPT = cola.enabled and cola.scaledependent and FML::PARTICLE::has_get_D_2LPT<T>() and
                            cola.LPT_order >= 2 and ic.LPT_order >= 2;
    std::vector<FFTWGrid<NDIM>> phi_nLPT_potentials;
    if (store_2LPT)
        phi_nLPT_potentials.resize(1);

    const auto velocity_norm_1LPT_of_kBox = ic_velocity_norm_1LPT_of_kBox(fac);

    // With COLA the velocities are set relative to the COLA frame (instead of zeroing them after the IC, see init())
    // if the scale-dependent 1LPT velocity differs from the frame's (velocity_norm_1LPT_of_kBox), or if the IC have
    // LPT orders the frame does not have (simulation_cola_LPT_order: 1LPT, 2LPT). Those (2LPT with
    // simulation_cola_LPT_order = 1, 3LPT) keep their full IC velocity: zeroing all the velocities
    // (cola_initialize_velocities) dropped them (1.7% rms in the velocity at z = 20 with ic_LPT_order = 2 and
    // simulation_cola_LPT_order = 1)
    if (cola.enabled and (velocity_norm_1LPT_of_kBox or ic.LPT_order > cola.LPT_order))
        ic_velocities_relative_to_cola_frame = true;
    if (ic_velocities_relative_to_cola_frame) {
        // velocity_norms = {1LPT, 2LPT, 3LPTa, 3LPTb}: the frame has the first cola.LPT_order (<= 2) of them
        for (int i = 0; i < cola.LPT_order; i++)
            velocity_norms[i] = 0.0;
    }

    const auto velocity_norm_2LPT_of_kBox = ic_velocity_norm_2LPT_of_kBox(fac);
    // (with COLA the velocities are already relative to the frame, see above)
    FML::assert_mpi(not velocity_norm_2LPT_of_kBox or not cola.enabled or ic_velocities_relative_to_cola_frame,
                    "Internal error: scale-dependent 2LPT velocity with COLA must be relative to the COLA frame");

    FML::NBODY::NBodyInitialConditions<NDIM, T>(part,
                                                params.particle.Npart_1D,
                                                params.particle.allocation_factor,
                                                delta_cb_fourier,
                                                phi_nLPT_potentials,
                                                ic.LPT_order,
                                                params.simulation.boxsize,
                                                ic.initial_redshift,
                                                velocity_norms,
                                                velocity_norm_1LPT_of_kBox,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                velocity_norm_2LPT_of_kBox);

    store_cola_potentials(delta_cb_fourier, store_2LPT ? &phi_nLPT_potentials[0] : nullptr);
}

/// Offset the particle lattice from the nodes of the force grid. With a force grid finer than the particle
/// grid the particles would otherwise start exactly on grid nodes where CIC responds to |displacement|
/// (a rectified response) and this gives large spurious forces.
/// This is a uniform translation of all the particles (all species) so it does not change the physics
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::apply_lattice_offset() {
    const auto & ic = params.ic;
    const bool offset_lattice = ic.lattice_offset == "half_force_cell" or
                                (ic.lattice_offset == "fiducial" and params.force.nmesh > params.particle.Npart_1D);
    if (not offset_lattice)
        return;
    const double shift = 0.5 / double(params.force.nmesh);
    // Scale-dependent COLA interpolates its LPT potentials (of the lattice before the offset) at q - shift
    cola.lattice_offset = shift;
    if (FML::ThisTask == 0)
        std::cout << "Offsetting the particle lattice by half a force grid cell (" << shift * params.simulation.boxsize
                  << " Mpc/h) in each direction\n";
    for_each_species([&](auto & particles, const SIMCODE::SpeciesInfo &) {
        using P = std::remove_reference_t<decltype(*particles.get_particles_ptr())>;
        for (auto & p : particles) {
            auto * pos = FML::PARTICLE::GetPos(p);
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] += shift;
                if (pos[idim] >= 1.0)
                    pos[idim] -= 1.0;
            }
            if constexpr (FML::PARTICLE::has_get_q<P>()) {
                auto * q = FML::PARTICLE::GetLagrangianPos(p);
                for (int idim = 0; idim < NDIM; idim++) {
                    q[idim] += shift;
                    if (q[idim] >= 1.0)
                        q[idim] -= 1.0;
                }
            }
        }
        particles.communicate_particles();
    });
}

/// The initial conditions with separate CDM and baryon particles (see two_species_ic_splines and
/// FML::NBODY::generate_species_from_delta_cb). The 2LPT displacement is that of the total (CDM+baryon)
/// field for both. By default (particle_baryon_lattice = same) the baryons start from the same lattice as the CDM.
/// With "offset" their lattice is offset by half a particle spacing in each direction. Their IDs follow those of the
/// CDM.
/// NB: interleaved (offset) lattices give a large spurious growth of the baryon-CDM difference on scales k > k_Ny/8
/// of the particles (and a few % too little CDM+baryon power at k ~ k_Ny/2) in this PM code; with the same lattice the
/// difference follows linear theory on all scales and the CDM+baryon P(k) is that of one species (see the README)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::generate_initial_conditions_two_species(const FFTWGrid<NDIM> & delta_cb_fourier) {
    const auto & ic = params.ic;
    const double boxsize = params.simulation.boxsize;
    const double aini = ic.a_initial();
    FML::assert_mpi(transfers != nullptr,
                    "particle_baryons needs the transfer functions (camb, class, internal or tables)");
    const auto & cdm = SIMCODE::find_species(species, SIMCODE::ParticleSpecies::CDM);
    const auto & baryon = SIMCODE::find_species(species, SIMCODE::ParticleSpecies::Baryon);
    const auto splines = SIMCODE::two_species_ic_splines<NDIM>(*transfers,
                                                               *grav_ic,
                                                               ic.initial_redshift,
                                                               cdm.weight_cb,
                                                               baryon.weight_cb);

    const double fac = aini * aini * cosmo->HoverH0_of_a(aini);
    const std::vector<double> velocity_norms = ic_velocity_norms(fac);
    // The 2LPT velocity with scale-dependent growth (the same for both species, see ic_velocity_norm_2LPT_of_kBox)
    const auto velocity_norm_2LPT = ic_velocity_norm_2LPT_of_kBox(fac);
    const double kmin = transfers->range().kmin_hmpc;
    const double kmax = transfers->range().kmax_hmpc;
    auto clamp_logk = [&](double kBox) { return SIMCODE::clamped_logk(kBox / boxsize, kmin, kmax); };
    // The baryon-CDM relative mode: in the displacement (each species its own displacement and velocity), the
    // masses m = 1 + eps (mass: shared CDM+baryon flow, eps = the linear relative mode today; mass_ini: shared
    // displacement, eps at zini and each species its own velocity)
    const std::string & relative_mode = params.particle.baryon_relative_mode;
    const bool mass_mode = relative_mode == "mass" or relative_mode == "mass_ini";
    std::pair<Spline, Spline> eps_today;
    if (relative_mode == "mass") {
        eps_today = SIMCODE::two_species_mass_contrast_today(*transfers, ic.initial_redshift);
        if (FML::ThisTask == 0) {
            std::cout << "# Mass-encoded relative mode: eps_b / delta_cb(zini) = ";
            for (double k : {0.01, 0.1, 1.0})
                std::cout << eps_today.second(std::log(k)) << " (k = " << k << ") ";
            std::cout << "\n";
        }
    }
    auto make_species = [&](MPIParticles & particles,
                            const SIMCODE::SpeciesInfo & info,
                            const Spline & ratio,
                            const Spline & growth_rate,
                            const Spline * eps_of_logk,
                            double shift) {
        if (FML::ThisTask == 0)
            std::cout << "# Making the " << info.name << " particles\n";
        // With COLA the species share the COLA frame of the total CDM+baryon field and the velocities we set are
        // relative to it: v_i - v_frame = a^2 H [(f_i - f_frame / R_i) Psi_i] for 1LPT (Psi_i = R_i Psi_cb) and zero
        // for 2LPT (the same for all species; all of the 2LPT velocity with simulation_cola_LPT_order = 1 as the frame
        // then has none, see ic_velocity_norm_2LPT_of_kBox). The frame velocity is f_frame Psi_cb with
        // f_frame = f_cb(k) for scale-dependent COLA, but the growth rate at k = 0 for scale-independent COLA (the
        // frame adds back grav->get_f_1LPT(a), as for one species). Using f_cb(k) there with scale-dependent growth
        // gave the velocities (f_cb(k) - f(0)) Psi_cb too little (-1.75% rms for 0.4 eV neutrinos). The frame has the
        // growth rate of the gravity model (grav), also with ic_use_gravity_model_GR where the IC have that of GR
        // (grav_ic)
        std::vector<double> residual_velocity_norms = velocity_norms;
        for (int i = 0; i < cola.LPT_order; i++)
            residual_velocity_norms[i] = 0.0;
        auto f_frame = [&](double logk) {
            return cola.scaledependent ? grav->get_f_1LPT(aini, std::exp(logk) / grav->H0_hmpc) :
                                         grav->get_f_1LPT(aini);
        };
        std::function<double(double)> velocity_norm_1LPT = [&](double kBox) {
            const double logk = clamp_logk(kBox);
            if (not cola.enabled)
                return fac * growth_rate(logk);
            return fac * (growth_rate(logk) - f_frame(logk) / ratio(logk));
        };
        std::function<double(double)> displacement_ratio = [&](double kBox) { return ratio(clamp_logk(kBox)); };
        std::function<double(double)> mass_contrast = nullptr;
        if (mass_mode) {
            // Both species are displaced by Psi_cb. mass: the CDM+baryon velocity f_cb Psi_cb. mass_ini: the velocity
            // of the species v_i = f_i R_i Psi_cb. With COLA relative to the frame velocity f_frame Psi_cb
            displacement_ratio = [](double) { return 1.0; };
            velocity_norm_1LPT = [&](double kBox) {
                const double logk = clamp_logk(kBox);
                const double f_cb = grav_ic->get_f_1LPT(aini, std::exp(logk) / grav_ic->H0_hmpc);
                const double f_i = relative_mode == "mass" ? f_cb : growth_rate(logk) * ratio(logk);
                return fac * (cola.enabled ? f_i - f_frame(logk) : f_i);
            };
            if (relative_mode == "mass")
                mass_contrast = [&](double kBox) { return (*eps_of_logk)(clamp_logk(kBox)); };
            else
                mass_contrast = [&](double kBox) { return ratio(clamp_logk(kBox)) - 1.0; };
        }
        FML::NBODY::generate_species_from_delta_cb<NDIM, T>(particles,
                                                            delta_cb_fourier,
                                                            displacement_ratio,
                                                            velocity_norm_1LPT,
                                                            cola.enabled ? residual_velocity_norms : velocity_norms,
                                                            ic.LPT_order,
                                                            info.Npart_1D,
                                                            params.particle.allocation_factor,
                                                            boxsize,
                                                            ic.initial_redshift,
                                                            shift,
                                                            info.id_offset,
                                                            cola.enabled,
                                                            mass_contrast,
                                                            velocity_norm_2LPT);
    };
    make_species(part, cdm, splines.ratio_cdm, splines.f_cdm, &eps_today.first, 0.0);
    const double baryon_shift = params.particle.baryon_lattice == "offset" ? 0.5 / double(baryon.Npart_1D) : 0.0;
    make_species(part_baryon, baryon, splines.ratio_baryon, splines.f_baryon, &eps_today.second, baryon_shift);

    // Scale-dependent COLA: the LPT potentials of the total CDM+baryon field (the shared COLA frame)
    store_cola_potentials(delta_cb_fourier, nullptr);
    if (cola.enabled and cola.scaledependent) {
        // The baryons on the offset lattice: the potentials of the shifted field, as for their IC (see
        // ColaState::ShiftedSpecies)
        if (baryon_shift != 0.0) {
            if (FML::ThisTask == 0)
                std::cout << "Storing the LPT potentials of the offset baryon lattice for scale-dependent COLA\n";
            typename ColaState<NDIM, T>::ShiftedSpecies sp;
            sp.part = &part_baryon;
            sp.shift = baryon_shift;
            FFTWGrid<NDIM> delta_shifted = delta_cb_fourier;
            std::array<double, NDIM> shift_vec;
            shift_vec.fill(baryon_shift);
            FML::GRID::shift_grid_fourier<NDIM>(delta_shifted, shift_vec, +1.0);
            if (cola.phi_1LPT_ini_fourier) {
                sp.phi_1LPT_ini_fourier = FFTWGrid<NDIM>();
                FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier<NDIM>(delta_shifted, sp.phi_1LPT_ini_fourier);
                sp.phi_1LPT_ini_fourier.add_memory_label("phi_1LPT(k,zini) baryons");
            }
            if (cola.phi_2LPT_ini_fourier) {
                FML::COSMOLOGY::LPT::compute_2LPT_potential_fourier(delta_shifted, sp.phi_2LPT_ini_fourier);
                sp.phi_2LPT_ini_fourier.add_memory_label("phi_2LPT(k,zini) baryons");
            }
            cola.shifted_species.push_back(std::move(sp));
        }
    }
}

/// Read the N-body gauge source and check it
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::setup_nbody_gauge() {
    const auto & force = params.force;
    // (the models are checked in check_parameters_against_models)
    FML::assert_mpi(transfers != nullptr, "force_nbody_gauge needs the transfer functions");

    nbody_gauge = std::make_shared<SIMCODE::NBodyGaugeSource>();
    if (params.ic.from_internal_solver()) {
        if (FML::ThisTask == 0)
            std::cout << "# Running in the N-body gauge. The linear source from the Einstein-Boltzmann solver\n";
        nbody_gauge->from_transfers(*transfers);
    } else {
        if (FML::ThisTask == 0)
            std::cout << "# Running in the N-body gauge. Reading the linear source [" << force.nbody_gauge_filename
                      << "]\n";
        nbody_gauge->read(force.nbody_gauge_filename);
        nbody_gauge->check_cosmology(*cosmo, params.ic.initial_redshift, params.simulation.boxsize, force.nmesh);
    }

    if (FML::ThisTask == 0) {
        const double aini = params.ic.a_initial();
        for (double a : {aini, 0.5, 1.0}) {
            const auto ratio = nbody_gauge->ratio_spline(a, aini, *transfers);
            std::cout << "# N-body gauge source: delta_extra(k,z) / delta_cb(k,zini) at z = " << 1.0 / a - 1.0 << ":";
            for (double k : {1e-3, 1e-2, 1e-1})
                std::cout << " " << ratio(std::log(k)) << " (k = " << k << ")";
            std::cout << "\n";
        }
    }

    // delta_cb(k,zini) is stored from the density field of the first force evaluation (add_nbody_gauge_source) so that
    // it has the same window as the density fields it is added to
}

/// Read the particles from gadget files. With COLA we also reconstruct the LPT displacement fields
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::read_ic() {
    SIMCODE::read_ic_from_gadget<NDIM, T>(part,
                                          params,
                                          cosmo,
                                          grav_ic,
                                          cola.phi_1LPT_ini_fourier,
                                          cola.phi_2LPT_ini_fourier);

    // With the LPT fields reconstructed from the density field the COLA frame velocity is only an approximation to the
    // velocities in the file (smoothed, from the Eulerian density). Keep the velocities of the file: store them
    // relative to the frame instead of replacing them by it. (The exact reconstruction reproduces them, so nothing to
    // do there)
    if (cola.enabled and not params.ic.reconstruct_exact) {
        cola.add_on_LPT_velocity(part, params.ic.a_initial(), -1.0);
        ic_velocities_relative_to_cola_frame = true;
    }
}

//=============================================================================
// FFTW plans. With simulation_fftw_planner = measure (or slower) FFTW times
// candidate algorithms the first time each grid size is transformed. The plans
// (wisdom) are written to the output folder so that later runs can read them
// with simulation_fftw_wisdom_file and skip the planning
//=============================================================================

template <int NDIM, class T>
std::string NBodySimulation<NDIM, T>::fftw_wisdom_output_filename() const {
    return params.output.path("fftw_wisdom_" + params.simulation.name + ".txt");
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::setup_fftw() {
    const auto & planner = params.simulation.fftw_planner;
    const auto & wisdom_file = params.simulation.fftw_wisdom_file;
    FML::GRID::set_fftw_planner_flag(FML::GRID::fftw_planner_flag_from_name(planner));
    if (wisdom_file != "") {
        if (not FML::GRID::load_fftw_wisdom(wisdom_file))
            throw std::runtime_error("Cannot read FFTW wisdom from simulation_fftw_wisdom_file = [" + wisdom_file +
                                     "] (missing file, or wisdom from another FFTW version or precision)");
    }
    if (FML::ThisTask == 0) {
        std::cout << "# FFTW planner: " << planner << " with " << FML::GRID::get_fftw_nthreads() << " thread(s)";
        if (wisdom_file != "")
            std::cout << ", wisdom read from " << wisdom_file;
        if (planner != "estimate")
            std::cout << ". Plans made are written to " << fftw_wisdom_output_filename();
        std::cout << "\n";
    }
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::save_fftw_wisdom_if_new_plans() {
    // Plans made with FFTW_ESTIMATE are not worth keeping (no measuring was done)
    if (params.simulation.fftw_planner == "estimate")
        return;
    // Collective: all tasks must agree on whether to save (they make the same plans, but be safe)
    size_t nplans = FML::GRID::get_number_of_fftw_plans_made();
    FML::MaxOverTasks(&nplans);
    if (nplans == fftw_plans_in_saved_wisdom)
        return;
    const auto filename = fftw_wisdom_output_filename();
    if (FML::GRID::save_fftw_wisdom(filename)) {
        fftw_plans_in_saved_wisdom = nplans;
        if (FML::ThisTask == 0)
            std::cout << "# Wrote FFTW wisdom (" << nplans << " plans made in this run) to " << filename << "\n";
    } else if (FML::ThisTask == 0) {
        std::cout << "# Warning: could not write FFTW wisdom to " << filename << "\n";
    }
}

#endif
