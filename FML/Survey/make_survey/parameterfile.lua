-- make_survey parameter file (Lua). Lengths in Mpc/h, velocities in km/s, angles in degrees.

-- Input: "ascii" (x y z vx vy vz [id] per line, a periodic box), "gadget" (a GADGET snapshot, e.g. from
-- COLASolver; file prefix without the .N) or "lightcone_hdf5" (COLASolver HDF5 lightcone files; needs USE_HDF5)
input_format = "ascii"
input_files = {"galaxies.txt"}
boxsize = 1000.0
velocity_factor = 1.0          -- multiplies the input velocities (to get km/s)
-- parent_nbar = 1e-3          -- density of the simulation (required for lightcones; default N/L^3)

-- Distances: flat or curved w0waCDM background, H(z)/c = 100 E(z)/c [h/Mpc] (no radiation)
Omega_m = 0.3
Omega_lambda = 0.7
w0 = -1.0
wa = 0.0

-- Geometry
observer = {500.0, 500.0, 500.0}
pole = {0.0, 0.0, 1.0}         -- survey pole (Dec = 90) in the box frame
ra_zero = {1.0, 0.0, 0.0}      -- RA = 0 towards this direction
-- remap = {5,2,1, 2,1,0, 0,0,1}  -- volume-preserving cuboid remapping (periodic box input only)

-- Redshifts and selection
z_min = 0.05
z_max = 0.2
redshift_space = true
footprint = "radec"            -- "full", "radec", "healpix_ascii", "healpix_fits", "mangle" (polygon format)
footprint_radec = {120.0, 240.0, 0.0, 60.0}   -- ra_min, ra_max, dec_min, dec_max
-- footprint_file = "mask.txt"  -- Healpix map or MANGLE .ply file
apply_completeness = true
nz = "constant"                -- "none", "constant", "table_nbar" (z n(z) [(h/Mpc)^3]), "table_dndz" (z dN/dz per deg^2)
nz_value = 3e-4
-- nz_file = "nz.txt"

-- Randoms and output
random_factor = 20
seed = 1
random_seed = 2
output_format = "ascii"        -- "ascii" or "hdf5" (USE_HDF5)
output_data = "mock_data.txt"
output_randoms = "mock_randoms.txt"
