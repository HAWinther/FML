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
simulation_name = "LCDM_nu0.2"
-- Boxsize of simulation in Mpc/h
simulation_boxsize = 1000.0

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
-- Cosmology: LCDM, w0waCDM, DGP, ... add your own ...
cosmology_model = "LCDM"
--- CDM density
cosmology_OmegaCDM = 0.2637
--- Baryon density
cosmology_Omegab = 0.049
-- Massive neutrino density
cosmology_OmegaMNu = 0.0048
-- Dark energy density (a CC)
cosmology_OmegaLambda = 0.682407079
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

------------------------------------------------------------
-- Choose the gravity model
------------------------------------------------------------
-- Gravity model: GR, DGP, f(R), ... add your own ...
gravity_model = "GR"

-- Hu-Sawicky f(R) model
if gravity_model == "f(R)" then 
  -- f_R0 value
  gravity_model_fofr_fofr0 = 1e-5
  -- The index n
  gravity_model_fofr_nfofr = 1.0
  -- Approximate screening model (otherwise linear)
  gravity_model_screening = true
  -- Combine screeneed solution with linear solution to enforce correct
  -- linear evolution on large scales
  gravity_model_screening_enforce_largescale_linear = true
  -- The fourier scale for which we use the linear solution for k < k*
  -- and the screened solution for k > k*
  gravity_model_screening_linear_scale_hmpc = 0.1
end

-- DGP model (pick LCDM as the cosmology to get the normal branch)
-- For the self accelerating branch rcH0 must have a negative sign
if gravity_model == "DGP" then 
  -- The cross-over scale rc*H0/c
  gravity_model_dgp_rcH0overc = 1.0
  -- Approximate screening model (otherwise linear)
  gravity_model_screening = true
  -- For screening approx: smoothing filter for density (tophat, gaussian, sharpk)
  gravity_model_dgp_smoothing_filter = "tophat"
  -- For screening approx: smoothing scale R/boxsize
  gravity_model_dgp_smoothing_scale_over_boxsize = 0.0
  -- Combine screeneed solution with linear solution to enforce correct
  -- linear evolution on large scales
  gravity_model_screening_enforce_largescale_linear = true
  -- The fourier scale for which we use the linear solution for k < k*
  -- and the screened solution for k > k*
  gravity_model_screening_linear_scale_hmpc = 0.1
end

------------------------------------------------------------
-- Particles
------------------------------------------------------------
-- Number of CDM+b particles per dimension
particle_Npart_1D = 256
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
timestep_nsteps = {20}
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
-- Type of IC: gaussian, nongaussian, reconstruct_from_particles, read_from_file
-- All of these generate the IC in fourier space. We can also generate a white noise
-- field in real-space and fourier transform it
ic_random_field_type = "gaussian"
-- The grid-size used to generate the IC
ic_nmesh = particle_Npart_1D
-- For MG: input LCDM P(k) and use GR to scale back and ensure same IC as for LCDM
ic_use_gravity_model_GR = true
-- The LPT order to use for the IC
ic_LPT_order = 2
-- The type of input: 
-- powerspectrum    (file with [k (h/Mph) , P(k) (Mpc/h)^3)])
-- transferfunction (file with [k (h/Mph) , T(k)  Mpc^2)]
-- transferinfofile (file containing paths to a bunch of T(k,z) files from CAMB)
-- reconstruct_from_particles (see below)
-- read_particles   (if not COLA read GADGET file and use that for sim) 
ic_type_of_input = "transferinfofile"
-- Path to the input
ic_input_filename = "../../input/transfer_infofile_lcdm_nu0.2.txt"
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

-- For reconstructing the initial density field from particles
-- (needed for COLA when we don't just need the positions, but also displacement fields)
if ic_random_field_type == "reconstruct_from_particles" then
  -- Path to gadget files
  ic_reconstruct_gadgetfilepath = "output/gadget"
  -- Density assignment method: NGP (not a good idea), CIC, TSC, PCS, PQS
  ic_reconstruct_assigment_method = "CIC"
  -- Smoothing filter: tophat, gaussian, sharpk
  ic_reconstruct_smoothing_filter = "sharpk"
  -- Smoothing scale R/box (for killing off modes not in the IC if we have a bigger grid)
  ic_reconstruct_dimless_smoothing_scale = 1.0 / (128 * math.pi)
  -- Want interlaced grids? Probably not
  ic_reconstruct_interlacing = false
end

-- For reading IC from an external file (for usual N-body)
if ic_random_field_type == "read_particles" then
  -- Path to GADGET files
  ic_reconstruct_gadgetfilepath = "path/gadget"
end

------------------------------------------------------------
-- Force calculation
------------------------------------------------------------
-- Grid to use for computing PM forces
force_nmesh = 128
-- Density assignment method: NGP, CIC, TSC, PCS, PQS
force_density_assignment_method = "CIC"
-- The kernel to use when solving the Poisson equation
force_kernel = "continuous_greens_function"
-- Include the effects of massive neutrinos when computing
-- the density field (density of mnu is the linear prediction)
-- Requires: transferinfofile above (we need all T(k,z))
force_linear_massive_neutrinos = true

------------------------------------------------------------
-- On the fly analysis
------------------------------------------------------------

------------------------------------------------------------
-- Halofinding
------------------------------------------------------------
-- Do halofinding every output?
fof = false
-- Minimum number of particles per halo
fof_nmin_per_halo = 20
-- The fof distance (rmin / boxsize) used when linking
fof_linking_length = 0.2 / particle_Npart_1D
-- Limit the maximum grid to use to bin particles to
-- to speed up the fof linking. 0 means we let the code choose this
fof_nmesh_max = 0

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

