#===================================================================
# FML: tests
#
# make run   (or 'make test' in the top folder) builds and runs the
# regression tests. With MPI they are run for each task count in NP_LIST
#===================================================================

USE_MPI             ?= true
USE_OMP             ?= true
USE_FFTW            ?= true
USE_GSL             ?= true
USE_LUA             ?= true

TARGETS := test_core test_grid_tools test_fof test_so test_fnl test_kdtree test_paircount test_survey test_pofk test_reconstruction test_multigrid test_hybridbias test_sphericalcollapse test_halomodel test_gadget test_ramses test_camb test_numcore test_fftlog test_emulator test_utils test_interpolation test_grids test_particles test_lpt test_nbody
TARGETS += test_baryonification test_numericalmethods test_interlacing_offset test_streamed_kick test_lightcone test_random_fields
TARGETS += test_cosmicweb

include ../Makefile.config

ifeq ($(USE_LUA),true)
TARGETS += test_lua
endif

FML_MODULES := Global Spline ODESolver Math SphericalCollapse HaloModel FileUtils GadgetUtils FFTLog ParameterMap
EXTRA_VPATH += $(FML_ROOT)/Baryonification/tests

TEST_OBJS = test_core.o Global.o
ifeq ($(USE_GSL),true)
TEST_OBJS += Spline.o
endif

test_random_fields: test_random_fields.o Global.o Spline.o ODESolver.o ParameterMap.o
	$(LINK_EXE)

test_lightcone: test_lightcone.o Global.o Spline.o ParameterMap.o
	$(LINK_EXE)

test_core: $(TEST_OBJS)
	$(LINK_EXE)

test_grid_tools: test_grid_tools.o Global.o
	$(LINK_EXE)

test_cosmicweb: test_cosmicweb.o Global.o
	$(LINK_EXE)

test_fof: test_fof.o Global.o
	$(LINK_EXE)

test_so: test_so.o Global.o
	$(LINK_EXE)

test_kdtree: test_kdtree.o test_kdtree_odr.o Global.o
	$(LINK_EXE)

test_baryonification: test_baryonification.o test_baryonification_odr.o Global.o
	$(LINK_EXE)

PAIRCOUNT_OBJS = test_paircount.o test_paircount_odr.o Global.o
ifeq ($(USE_GSL),true)
PAIRCOUNT_OBJS += Spline.o
endif

test_paircount: $(PAIRCOUNT_OBJS)
	$(LINK_EXE)

# GalaxiesToBox needs the ODE solver and splines (GSL)
test_survey: test_survey.o Global.o Spline.o ODESolver.o
	$(LINK_EXE)

# NonLocalGaussianRandomField.h includes the cosmology helpers that use splines and the ODE solver with GSL
FNL_OBJS = test_fnl.o Global.o
ifeq ($(USE_GSL),true)
FNL_OBJS += Spline.o ODESolver.o
endif

test_fnl: $(FNL_OBJS)
	$(LINK_EXE)

# ComputePowerSpectrum.h includes the LPT and cosmology helpers (splines and the ODE solver with GSL)
POFK_OBJS = test_pofk.o test_pofk_odr.o Global.o
ifeq ($(USE_GSL),true)
POFK_OBJS += Spline.o ODESolver.o
endif

test_pofk: $(POFK_OBJS)
	$(LINK_EXE)

# Spherical collapse (splines and the ODE solver with GSL)
test_sphericalcollapse: test_sphericalcollapse.o Global.o Spline.o ODESolver.o Math.o SphericalCollapse.o SphericalCollapseModel.o
	$(LINK_EXE)

# Halo model (reads the linear P(k) and the HMCode reference in ../HaloModel/example)
test_halomodel: test_halomodel.o Global.o Spline.o ODESolver.o Math.o SphericalCollapse.o SphericalCollapseModel.o Halomodel.o FileUtils.o
	$(LINK_EXE)

# Reconstruction, Hessian (GSL eigensolver) and smoothing
test_reconstruction: test_reconstruction.o Global.o Spline.o ODESolver.o
	$(LINK_EXE)

test_multigrid: test_multigrid.o Global.o
	$(LINK_EXE)

test_hybridbias: test_hybridbias.o Global.o
	$(LINK_EXE)

# GadgetUtils (binary GADGET format; the HDF5 part only with USE_HDF5=true)
test_gadget: test_gadget.o Global.o GadgetUtils.o
	$(LINK_EXE)

# RamsesUtils (writes small synthetic RAMSES snapshots)
test_ramses: test_ramses.o test_ramses_odr.o Global.o
	$(LINK_EXE)

# CAMBUtils (reads the tables in ../COLASolver/input)
test_camb: test_camb.o test_camb_odr.o Global.o Spline.o
	$(LINK_EXE)

# Numerical core: splines, ODE solver and math functions (GSL)
test_numericalmethods: test_numericalmethods.o NumericalMethods.o
	$(LINK_EXE)

test_numcore: test_numcore.o Global.o Spline.o ODESolver.o Math.o
	$(LINK_EXE)

# FFTLog (FFTW)
test_fftlog: test_fftlog.o Global.o FFTLog.o
	$(LINK_EXE)

# EuclidEmulator2 (GSL; the data file is ../Emulator/EuclidEmulator2.dat)
test_emulator: test_emulator.o test_emulator_odr.o Global.o
	$(LINK_EXE)

# Small utilities: FileUtils, RandomGenerator, FoFBinning, MemoryLogging, Timings, ParameterMap
test_utils: test_utils.o test_units_odr.o Global.o FileUtils.o ParameterMap.o
	$(LINK_EXE)

# Particle <-> grid interpolation
test_interlacing_offset: test_interlacing_offset.o Global.o
	$(LINK_EXE)

test_interpolation: test_interpolation.o Global.o
	$(LINK_EXE)

# LuaFileParser (needs Lua: USE_LUA=true, the default here as for COLASolver)
test_lua: test_lua.o test_lua_odr.o Global.o
	$(LINK_EXE)

# FFTWGrid and MPIGrid
test_grids: test_grids.o Global.o
	$(LINK_EXE)

# MPIParticles, particle reflection, ParticlesInBoxes (Chunkpix only with USE_HEALPIX)
test_particles: test_particles.o Global.o
	$(LINK_EXE)

# LPT displacement fields and random fields
test_lpt: test_lpt.o test_lpt_odr.o Global.o Spline.o ODESolver.o
	$(LINK_EXE)

# NBody forces, drift and PGD
test_streamed_kick: test_streamed_kick.o Global.o
	$(LINK_EXE)

test_nbody: test_nbody.o test_nbody_odr.o Global.o Spline.o ODESolver.o
	$(LINK_EXE)

MPIRUN  ?= mpirun
NP_LIST ?= 1 2 4

ifeq ($(USE_MPI),true)
RUN  = $(MPIRUN) -np $$np
RUN1 = $(MPIRUN) -np 1
else
NP_LIST = 1
RUN  =
RUN1 =
endif

# Idle OpenMP threads sleep instead of spinning: with more threads than cores (e.g. 4 tasks x 2 threads on a laptop)
# spinning threads starve the MPI tasks and the many short parallel regions of e.g. the multigrid solver become
# ~100 times slower (test_multigrid on 4 tasks: 270 s -> 2 s on 2 cores)
export OMP_WAIT_POLICY = passive

# Each run must pass; the two "must abort" runs must fail with the expected message
.PHONY: run
run: $(TARGETS)
	@set -e; for np in $(NP_LIST); do \
	  echo "== Running test_core with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_core; \
	  echo "== Running test_grid_tools with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_grid_tools; \
	  echo "== Running test_cosmicweb with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_cosmicweb; \
	  echo "== Running test_fof with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_fof; \
	  echo "== Running test_so with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_so; \
	  echo "== Running test_fnl with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_fnl; \
	  echo "== Running test_kdtree with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_kdtree; \
	  echo "== Running test_paircount with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_paircount; \
	  echo "== Running test_survey with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_survey; \
	  echo "== Running test_pofk with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_pofk; \
	  echo "== Running test_reconstruction with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_reconstruction; \
	  echo "== Running test_multigrid with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_multigrid; \
	  echo "== Running test_sphericalcollapse with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_sphericalcollapse; \
	  echo "== Running test_halomodel with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_halomodel; \
	  echo "== Running test_gadget with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_gadget; \
	  echo "== Running test_ramses with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_ramses; \
	  echo "== Running test_camb with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_camb; \
	  echo "== Running test_numcore with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_numcore; \
	  echo "== Running test_fftlog with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_fftlog; \
	  echo "== Running test_emulator with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_emulator; \
	  echo "== Running test_utils with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_utils; \
	  echo "== Running test_interpolation with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_interpolation; \
	  echo "== Running test_interlacing_offset with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_interlacing_offset; \
	  echo "== Running test_grids with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_grids; \
	  echo "== Running test_particles with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_particles; \
	  echo "== Running test_lpt with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_lpt; \
	  echo "== Running test_random_fields with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_random_fields; \
	  echo "== Running test_lightcone with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_lightcone; \
	  echo "== Running test_nbody with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_nbody; \
	  echo "== Running test_streamed_kick with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_streamed_kick; \
	  echo "== Running test_baryonification with $$np task(s)"; \
	  OMP_NUM_THREADS=2 $(RUN) ./test_baryonification; \
	done
	@echo "== Running test_hybridbias"
	@OMP_NUM_THREADS=2 $(RUN1) ./test_hybridbias
	@echo "== Checking that bad input aborts with a clear message"
	@! $(RUN1) ./test_core alloc > alloc.log 2>&1 && grep -q "Reached allocation limit" alloc.log && echo "[PASS] allocation limit aborts"
ifeq ($(USE_MPI),true)
	@! $(MPIRUN) -np 2 ./test_core unwrapped > unwrapped.log 2>&1 && grep -q "outside \[0,1)" unwrapped.log && echo "[PASS] unwrapped particle aborts"
endif
	@rm -f alloc.log unwrapped.log
	@for m in "truncated:File is truncated or unreadable" "decomposition:does not match the current domain decomposition" "dumpfail:Failed to open file for writing"; do \
	  ! $(RUN1) ./test_grids $${m%%:*} > grids.log 2>&1 && grep -qF "$${m#*:}" grids.log && echo "[PASS] test_grids $${m%%:*} gives an error" || exit 1; \
	done; rm -f grids.log
	@for m in "fdslices:needs at least stencil_order/2 extra slices" "stencil:Unknown stencil_order"; do \
	  ! $(RUN1) ./test_nbody $${m%%:*} > nbody.log 2>&1 && grep -qF "$${m#*:}" nbody.log && echo "[PASS] test_nbody $${m%%:*} gives an error" || exit 1; \
	done; rm -f nbody.log
	@! $(RUN1) ./test_interpolation unknown > interp.log 2>&1 && grep -qF "Unknown density assignment method" interp.log && echo "[PASS] test_interpolation unknown method gives an error"
ifeq ($(USE_MPI),true)
	@! $(MPIRUN) -np 4 ./test_interpolation thin > interp.log 2>&1 && grep -qF "Fewer slices on a task" interp.log && echo "[PASS] test_interpolation too thin slabs give an error"
endif
	@rm -f interp.log
	@! $(RUN1) ./test_interlacing_offset negative > offset.log 2>&1 && grep -qF "Assignment offsets must be" offset.log && echo "[PASS] negative assignment offset gives an error"
ifeq ($(USE_MPI),true)
	@! $(MPIRUN) -np 2 ./test_interlacing_offset halo > offset.log 2>&1 && grep -qF "one additional right slice" offset.log && echo "[PASS] shifted deposition checks the right halo"
endif
	@rm -f offset.log
ifeq ($(USE_LUA),true)
	@for np in $(NP_LIST); do echo "== Running test_lua with $$np task(s)"; $(RUN) ./test_lua || exit 1; done
	@for m in "bool:expected true or false" "int:expected an integer" "numarray:Element 2 of mixed" "strarray:Element 2 of badstrs"; do \
	  ! $(RUN1) ./test_lua $${m%%:*} > lua.log 2>&1 && grep -qF "$${m#*:}" lua.log && echo "[PASS] test_lua $${m%%:*} gives an error" || exit 1; \
	done; rm -f lua.log
endif
	@for m in "spline_npts:needs at least 3 points" "spline2d_deriv:eval_deriv] (2,1)" "spline2d_monotone:is not monotone" "ode_nojacobian:requires the jacobian"; do \
	  ! $(RUN1) ./test_numcore $${m%%:*} > numcore.log 2>&1 && grep -qF "$${m#*:}" numcore.log && echo "[PASS] test_numcore $${m%%:*} gives an error" || exit 1; \
	done; rm -f numcore.log
