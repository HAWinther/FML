------------------------------------------------------------
-- Simulation parameter file
-- This is a LUA script so you can write simple code
-- and use parameter names to set other parameters
-- See ReadParameters.h for which parameters are read
-- and what the standard value is
-- Include other paramfile into this: dofile("param.lua")
------------------------------------------------------------

-- Don't allow any parameters to take optional values?
-- If so we have to provide everything. If not most of the
-- parameters below don't have to be provided and fiduicial values
-- are used instead (see ReadParameters.h for fiducial values)
all_parameters_must_be_in_file = true

------------------------------------------------------------
-- Simulation options
------------------------------------------------------------
-- Label
simulation_name = "TestSim"
-- Boxsize of simulation in Mpc/h
simulation_boxsize = 512.0

------------------------------------------------------------
-- COLA
------------------------------------------------------------
-- Use the COLA method
simulation_use_cola = true
-- If gravity model has scaledependent growth. If this is false
-- then we use the k=0 limit of the growth factors when doing COLA
simulation_use_scaledependent_cola = true

------------------------------------------------------------
-- Choose the cosmology 
------------------------------------------------------------
-- Cosmology: LCDM, w0waCDM, DGP, JBD, ... add your own ...
cosmology_model = "LCDM"
--- CDM density
cosmology_OmegaCDM = 0.2637
--- Baryon density
cosmology_Omegab = 0.049
-- Massive neutrino density
cosmology_OmegaMNu = 0.0048
-- Curvature parameter (-k/H0^2)
cosmology_OmegaK = 0.0
-- Effective number of relativistic species
cosmology_Neffective = 3.046
-- Temperature of CMB today
cosmology_TCMB_kelvin = 2.7255
-- Hubble paramster
cosmology_h = 0.671
-- Primodial amplitude
cosmology_As = 2.215e-9 
-- Spectral index
cosmology_ns = 0.966
-- Pivot scale in 1/Mpc
cosmology_kpivot_mpc = 0.05

-- The w0wa parametrization
if cosmology_model == "w0waCDM" then 
  cosmology_w0 = -1.0
  cosmology_wa = 0.0
end

-- DGP self-accelerating model
if cosmology_model == "DGP" then 
  cosmology_dgp_OmegaRC = 0.11642
end

-- Jordan-Brans-Dicke. Connection to the hi-class implementation:
-- If hi-class is run with M_pl_today_smg = ... and normalize_G_NR = no then we should use:
-- cosmology_JBD_wBD = 100.0
-- cosmology_JBD_GeffG_today = (4+2*cosmology_JBD_wBD)/(3+2*cosmology_JBD_wBD) / M_pl_today_smg
-- cosmology_JBD_density_parameter_definition = "hi-class"
-- If we run hi-class with normalize_G_NR = yes then we need
-- cosmology_JBD_wBD = 100.0
-- cosmology_JBD_GeffG_today = 1.0
-- cosmology_JBD_density_parameter_definition = "hi-class"
if cosmology_model == "JBD" then 
  -- The JBD parameter (wBD -> infty gives us GR)
  cosmology_JBD_wBD = 100.0 
  -- The value of G today ("should" by unity). We have G_N = G / phi_* where phi_* = (4+2w)/(3+2w).
  -- G_today = G/phi(a=1) so GeffG_today = phi_*/phi(a=1) so if GeffG_today != 1.0
  -- we effectively have a different Newtons constant
  cosmology_JBD_GeffG_today = 1.0
  -- Density parameter definition Omega = 8pi G_* rho / 3H0^2
  -- Different choices for G_* (Gbare, Gnewton, Gtoday, hi-class)
  cosmology_JBD_density_parameter_definition = "hi-class"
end

------------------------------------------------------------
-- Choose the gravity model
------------------------------------------------------------
-- Gravity model: GR, DGP, f(R), JBD, Symmetron, Geff, ... add your own ...
gravity_model = "GR"

-- General Geff/G(a) models (mu-parametrization)
if gravity_model == "Geff" then 
  -- File with the format [a, Geff/G(a)]
  gravity_model_geff_geffofa_filename = "GeffoverG_of_a.txt"
end

-- (m(a),beta(a)) models
if gravity_model == "mbeta" then 
  
  -- The parameters defining m(a) and beta(a)
  -- In this example implementation we have beta(a) = beta0 a^n, m(a) = m0*H0*a^m
  -- where (beta0, n, m0, m) are the parameters
  gravity_model_mbeta_params = {0.5, 0.0, 1e3, -2.0}

  -- Approximate screening model (otherwise linear)
  gravity_model_screening = true
  -- Screening efficiency (1.0 is standard)
  -- This can be used to increase or decrease the amount of screening if using gravity_model_screening
  gravity_model_screening_efficiency = 1.0
  -- Combine screeneed solution with linear solution to enforce correct
  -- linear evolution on large scales
  gravity_model_screening_enforce_largescale_linear = true
  -- The fourier scale for which we use the linear solution for k < k*
  -- and the screened solution for k > k*
  gravity_model_screening_linear_scale_hmpc = 0.1

end

-- Hu-Sawicky f(R) model
if gravity_model == "f(R)" then 
  -- f_R0 value
  gravity_model_fofr_fofr0 = 1e-5
  -- The index n
  gravity_model_fofr_nfofr = 1.0
  -- Solve exact f(R) equation using the multigridsolver (slow and mainly for testing)
  -- This option takes precedent over the approximate screening model below if both are set
  gravity_model_fofr_exact_solution = false
  -- Approximate screening model (otherwise linear)
  gravity_model_screening = true
  -- Screening efficiency (1.0 is standard)
  -- This can be used to increase or decrease the amount of screening if using gravity_model_screening
  gravity_model_screening_efficiency = 1.0
  -- Combine screeneed solution with linear solution to enforce correct
  -- linear evolution on large scales
  gravity_model_screening_enforce_largescale_linear = true
  -- The fourier scale for which we use the linear solution for k < k*
  -- and the screened solution for k > k*
  gravity_model_screening_linear_scale_hmpc = 0.1

  -- Options for the multigrid solver in case we solve exact equation:
  multigrid_solver_residual_convergence = 1e-7
  -- How many Newton-Gauss-Seidel sweeps to do every level
  multigrid_nsweeps = 5
  -- In some cases the multigrid solver fails if we are not close to the
  -- solution before starting multigrid. Increase this if so
  multigrid_nsweeps_first_step = 5
end

-- Symmetron model
if gravity_model == "Symmetron" then 
  -- Symmetry breaking scalefactor (no fifth-force for a < assb)
  gravity_model_symmetron_assb = 0.333
  -- Coupling strength relative to gravity
  gravity_model_symmetron_beta = 1.0
  -- Range of force in background today
  gravity_model_symmetron_L_mpch = 1.0
  -- Solve exact symmetron equation using the multigridsolver (slow and mainly for testing)
  -- This option takes precedent over the approximate screening model below if both are set
  gravity_model_symmetron_exact_solution = false
  -- Approximate screening model (otherwise linear)
  gravity_model_screening = false
  -- Screening efficiency (1.0 is standard)
  -- This can be used to increase or decrease the amount of screening if using gravity_model_screening
  gravity_model_screening_efficiency = 1.0
  -- Combine screeneed solution with linear solution to enforce correct
  -- linear evolution on large scales
  gravity_model_screening_enforce_largescale_linear = false
  -- The fourier scale for which we use the linear solution for k < k*
  -- and the screened solution for k > k*
  gravity_model_screening_linear_scale_hmpc = 0.1

  -- Options for the multigrid solver in case we solve exact equation:
  multigrid_solver_residual_convergence = 1e-7
  -- How many Newton-Gauss-Seidel sweeps to do every level
  multigrid_nsweeps = 5
  -- In some cases the multigrid solver fails if we are not close to the
  -- solution before starting multigrid. Increase this if so
  multigrid_nsweeps_first_step = 5
end

-- DGP model (pick LCDM as the cosmology to get the normal branch)
-- For the self accelerating branch rcH0 must have a negative sign
if gravity_model == "DGP" then 
  -- The cross-over scale rc*H0/c
  gravity_model_dgp_rcH0overc = 1.0
  -- Solve exact DGP equation using the multigridsolver (slow and mainly for testing)
  -- This option takes precedent over the approximate screening model below if both are set
  -- NB: this is not always easy to get to work, its a tricky equation, and might saturate at
  -- a given residual level (and then you will have to reduce the epsilon and deem that to be converged)
  gravity_model_dgp_exact_solution = false
  -- Approximate screening model (otherwise linear)
  gravity_model_screening = true
  -- For screening approx: smoothing filter for density (tophat, gaussian, sharpk)
  gravity_model_dgp_smoothing_filter = "tophat"
  -- For screening approx: smoothing scale R/boxsize
  gravity_model_dgp_smoothing_scale_over_boxsize = 0.0 / simulation_boxsize
  -- Combine screeneed solution with linear solution to enforce correct
  -- linear evolution on large scales
  gravity_model_screening_enforce_largescale_linear = true
  -- The fourier scale for which we use the linear solution for k < k*
  -- and the screened solution for k > k*
  gravity_model_screening_linear_scale_hmpc = 0.1
  
  -- Options for the multigrid solver in case we solve exact equation:
  multigrid_solver_residual_convergence = 1e-4
  -- How many Newton-Gauss-Seidel sweeps to do every level
  multigrid_nsweeps = 2
  -- In some cases the multigrid solver fails if we are not close to the
  -- solution before starting multigrid. Increase this if so
  multigrid_nsweeps_first_step = 2
end

------------------------------------------------------------
-- Particles
------------------------------------------------------------
-- Number of CDM+b particles per dimension
particle_Npart_1D = 128
-- Factor of how many more particles to allocate space
particle_allocation_factor = 1.25

------------------------------------------------------------
-- Output
------------------------------------------------------------
-- List of output redshifts
output_redshifts = {0.0}
-- Output particles?
output_particles = true
-- Fileformat: GADGET, FML
output_fileformat = "GADGET"
-- Output folder
output_folder = "output"

------------------------------------------------------------
-- Time-stepping
------------------------------------------------------------
-- Number of steps between the outputs (in output_redshifts). 
-- If only one number in the list then its the total number of steps 
timestep_nsteps = {10}
-- The time-stepping method: Quinn, Tassev
timestep_method = "Quinn"
-- For Tassev: the nLPT parameter
timestep_cola_nLPT = -2.5
-- The time-stepping algorithm: KDK
timestep_algorithm = "KDK"

-- Spacing of the time-steps in 'a' is: linear, logarithmic, powerlaw
timestep_scalefactor_spacing = "linear"
if timestep_scalefactor_spacing == "powerlaw" then
  timestep_spacing_power = 1.0
end

------------------------------------------------------------
-- Initial conditions
------------------------------------------------------------
-- The random seed
ic_random_seed = 1234
-- The random generator (GSL or MT19937). Fiducial GSL is gsl_rng_ranlxd1 (as used in the 2LPTIC code for comparison)
ic_random_generator = "GSL"
-- Fix amplitude when generating the gaussian random field
ic_fix_amplitude = true
-- Mirror the phases (for amplitude-fixed simulations)
ic_reverse_phases = false
-- Type of IC: gaussian, nongaussian, read_particles, read_phases
-- read_particles   (read GADGET file and use that for sim - reconstruct LPT fields if COLA) 
-- read_phases      (read GADGET file and use that to set the phases for the sim)
ic_random_field_type = "gaussian"
-- The grid-size used to generate the IC
ic_nmesh = particle_Npart_1D
-- For MG: input LCDM P(k) and use GR to scale back and ensure same IC as for LCDM
-- ONLY use this if you don't have the P(k) for the MG model or if you want to consider PMG/PLCDM run with same IC at high z
ic_use_gravity_model_GR = false
-- The LPT order to use for the IC
ic_LPT_order = 2
-- The type of input: 
-- powerspectrum    (file with [k (h/Mph) , P(k) (Mpc/h)^3)])
-- transferfunction (file with [k (h/Mph) , T(k)  Mpc^2)]
-- transferinfofile (file containing paths to a bunch of T(k,z) files from CAMB)
ic_type_of_input = "transferinfofile"
ic_type_of_input_fileformat = "CAMB" -- Format for transferinfofile: CAMB, CLASS (run this with format=camb), AXIONCAMB. Easy to add more in CAMBReader.h
-- Path to the input (NB: for using the example files update the path at the top of the file below)
ic_input_filename = "input/transfer_infofile_lcdm_nu0.2.txt"
-- The redshift of the P(k), T(k) we give as input
ic_input_redshift = 0.0
-- The initial redshift of the simulation
ic_initial_redshift = 20.0
-- Normalize wrt sigma8? Otherwise use normalization in input + As etc.
-- If ic_use_gravity_model_GR then this is the sigma8 value is a corresponding GR universe!
ic_sigma8_normalization = false
-- Redshift of sigma8 value to normalize wrt
ic_sigma8_redshift = 0.0
-- The sigma8 value to normalize wrt
ic_sigma8 = 0.83

if ic_random_field_type == "nongaussian" then
  -- Type of non-gaussian IC: local, equilateral, orthogonal
  ic_fnl_type = "local"
  -- The fNL value
  ic_fnl = 100.0
  -- The redshift of which to apply the non-gaussian potential
  ic_fnl_redshift = ic_initial_redshift
end

-- For reading IC from an external file
-- If COLA then we reconstruct the LPT fields 
if ic_random_field_type == "read_particles" or ic_random_field_type == "read_phases" then
  -- Path to GADGET files
  ic_reconstruct_gadgetfilepath = "output/snapshot_TestSim_z20.000/gadget_z20.000"
  -- COLA settings to (naively) reconstruct the LPT fields:
  -- Density assignment method: NGP, CIC, TSC, PCS, PQS
  -- We use ic_nmesh to set the grid to compute the density field on
  -- NB: this should be equal to the nmesh used to generate the IC (i.e. NpartTot^1/3)
  ic_reconstruct_assigment_method = "PCS"
  ic_reconstruct_interlacing = true
  -- Smoothing filter to remove small-scale modes (only relevant if for
  -- some reason you want ic_nmesh to be larger than the grid it was created on)
  ic_reconstruct_smoothing_filter = "sharpk"
  ic_reconstruct_dimless_smoothing_scale = 1.0 /(2.0 * math.pi * ic_nmesh / 2)
end

------------------------------------------------------------
-- Force calculation
------------------------------------------------------------
-- Grid to use for computing PM forces
force_nmesh = 128
-- Density assignment method: NGP, CIC, TSC, PCS, PQS
force_density_assignment_method = "CIC"
-- The kernel to use for D^2 when solving the Poisson equation
-- Options: (fiducial = continuous, discrete_2pt, discrete_4pt)
force_greens_function_kernel = "fiducial"
-- The kernel to use for D when computing forces (with fourier)
-- Options: (fiducial = continuous, discrete_2pt, discrete_4pt)
force_gradient_kernel = "fiducial"
-- Include the effects of massive neutrinos when computing
-- the density field (density of mnu is the linear prediction)
-- Requires: transferinfofile above (we need all T(k,z))
force_linear_massive_neutrinos = true

-- Experimental feature: Use finite difference on the gravitational 
-- potential to compute forces instead of using Fourier transforms.
force_use_finite_difference_force = false
force_finite_difference_stencil_order = 4

------------------------------------------------------------
-- On the fly analysis
------------------------------------------------------------

------------------------------------------------------------
-- Halofinding
------------------------------------------------------------
-- Do halofinding every output?
fof = true
-- Minimum number of particles per halo
fof_nmin_per_halo = 20
-- The linking length (distance in units of mean particle separation)
fof_linking_length = 0.2
-- Limit the maximum grid to use to bin particles to
-- to speed up the fof linking. 0 means we let the code choose this
fof_nmesh_max = 0
-- The size of the buffer region larger than largest halo, 2-3Mpc/h should be fine)
fof_buffer_length_mpch = 3.0

------------------------------------------------------------
-- Power-spectrum evaluation
------------------------------------------------------------
-- Compute power-spectrum when we output
pofk = true
-- Gridsize to use for this
pofk_nmesh = 128
-- Use interlaced grids for alias reduction?
pofk_interlacing = true
-- Subtract shotnoise?
pofk_subtract_shotnoise = false
-- Density assignment method: NGP, CIC, TSC, PCS, PQS, ...
pofk_density_assignment_method = "PCS"

------------------------------------------------------------
-- Power-spectrum multipole evaluation
------------------------------------------------------------
-- Compute redshift space multipoles P_ell(k) when outputting
pofk_multipole = false
-- Gridsize to use for this
pofk_multipole_nmesh = 128
-- Use interlaced grids for alias reduction?
pofk_multipole_interlacing = true
-- Subtract shotnoise for P0?
pofk_multipole_subtract_shotnoise = false
-- Maximum ell we want P_ell for
pofk_multipole_ellmax = 4
-- Density assignment method: NGP, CIC, TSC, PCS, PQS
pofk_multipole_density_assignment_method = "PCS"

------------------------------------------------------------
-- Bispectrum evaluation
------------------------------------------------------------
-- Compute the bispectrum when we output?
bispectrum = false
-- Gridsize to use for this
bispectrum_nmesh = 128
-- Number of bins in k. NB: we need to store nbins grids and 
-- do nbins^3 integrals so both memory and computationally expensive
bispectrum_nbins = 10
-- Use interlaced grids for alias reduction?
bispectrum_interlacing = true
-- Subtract shotnoise?
bispectrum_subtract_shotnoise = false
-- Density assignment method: NGP, CIC, TSC, PCS, PQS
bispectrum_density_assignment_method = "PCS"

