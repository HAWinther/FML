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

TARGETS := test_core test_grid_tools test_fof test_so test_fnl test_kdtree test_paircount test_survey test_pofk test_reconstruction test_multigrid test_hybridbias test_sphericalcollapse test_halomodel test_gadget test_ramses test_camb test_numcore test_fftlog test_emulator

include ../Makefile.config

FML_MODULES := Global Spline ODESolver Math SphericalCollapse HaloModel FileUtils GadgetUtils FFTLog

TEST_OBJS = test_core.o Global.o
ifeq ($(USE_GSL),true)
TEST_OBJS += Spline.o
endif

test_core: $(TEST_OBJS)
	$(LINK_EXE)

test_grid_tools: test_grid_tools.o Global.o
	$(LINK_EXE)

test_fof: test_fof.o Global.o
	$(LINK_EXE)

test_so: test_so.o Global.o
	$(LINK_EXE)

test_kdtree: test_kdtree.o test_kdtree_odr.o Global.o
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
test_numcore: test_numcore.o Global.o Spline.o ODESolver.o Math.o
	$(LINK_EXE)

# FFTLog (FFTW)
test_fftlog: test_fftlog.o Global.o FFTLog.o
	$(LINK_EXE)

# EuclidEmulator2 (GSL; the data file is ../Emulator/EuclidEmulator2.dat)
test_emulator: test_emulator.o test_emulator_odr.o Global.o
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
	done
	@echo "== Running test_hybridbias"
	@OMP_NUM_THREADS=2 $(RUN1) ./test_hybridbias
	@echo "== Checking that bad input aborts with a clear message"
	@! $(RUN1) ./test_core alloc > alloc.log 2>&1 && grep -q "Reached allocation limit" alloc.log && echo "[PASS] allocation limit aborts"
ifeq ($(USE_MPI),true)
	@! $(MPIRUN) -np 2 ./test_core unwrapped > unwrapped.log 2>&1 && grep -q "outside \[0,1)" unwrapped.log && echo "[PASS] unwrapped particle aborts"
endif
	@rm -f alloc.log unwrapped.log
	@for m in "spline_npts:needs at least 3 points" "spline2d_deriv:eval_deriv] (2,1)" "spline2d_monotone:is not monotone" "ode_nojacobian:requires the jacobian"; do \
	  ! $(RUN1) ./test_numcore $${m%%:*} > numcore.log 2>&1 && grep -qF "$${m#*:}" numcore.log && echo "[PASS] test_numcore $${m%%:*} gives an error" || exit 1; \
	done; rm -f numcore.log
