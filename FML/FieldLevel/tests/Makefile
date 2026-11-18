USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := test_pm_assignment test_particle_mesh_observer test_map_optimizer test_particle_mesh_observation test_particle_mesh_checkpoint test_particle_mesh_adjoint test_particle_mesh_posterior
TARGETS += test_survey_checkpoint_mpi test_survey_likelihood_mpi test_bao_gradients test_bao_validation test_field_level test_field_level_safety test_framework test_framework_resolution test_framework_mpi test_framework_lpt_mpi test_framework_reconstruction_mpi test_framework_checkpoint_mpi test_particle_mesh
ifeq ($(USE_MPI),true)
TARGETS += test_framework_linear_grid_mpi test_framework_lpt_grid_mpi test_survey_grid_mpi test_checkpoint_grid_mpi
endif
EXTRA_CLEAN += test_framework_linear_grid_mpi test_framework_lpt_grid_mpi test_survey_grid_mpi test_checkpoint_grid_mpi
TARGETS += test_eulerian_metric test_eulerian_checkpoint test_compact_eulerian_lpt test_eulerian_bias test_eulerian_lpt test_fixed_ic test_periodic_catalog
include ../../Makefile.config
FML_MODULES := Global
ifeq ($(USE_MPI),true)
PM_GRID_OBJECT := PencilGrid.o
endif
test_pm_assignment: test_pm_assignment.o Global.o
	$(LINK_EXE)
test_particle_mesh: test_particle_mesh.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
test_particle_mesh_adjoint: test_particle_mesh_adjoint.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
test_particle_mesh_posterior: test_particle_mesh_posterior.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
test_particle_mesh_observation: test_particle_mesh_observation.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
# Do not reuse a VPATH object compiled with another module's MPI/precision flags.
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_survey_checkpoint_mpi: test_survey_checkpoint_mpi.o Global.o
	$(LINK_EXE)
test_survey_likelihood_mpi: test_survey_likelihood_mpi.o Global.o
	$(LINK_EXE)
test_framework_nonlinear_mpi: test_framework_nonlinear_mpi.o Global.o
	$(LINK_EXE)
test_bao_gradients: test_bao_gradients.o Global.o
	$(LINK_EXE)
test_bao_validation: test_bao_validation.o Global.o
	$(LINK_EXE)
test_field_level: test_field_level.o Global.o
	$(LINK_EXE)
test_field_level_safety: test_field_level_safety.o Global.o
	$(LINK_EXE)
test_framework: test_framework.o Global.o
	$(LINK_EXE)
test_framework_resolution: test_framework_resolution.o Global.o
	$(LINK_EXE)
test_framework_mpi: test_framework_mpi.o Global.o
	$(LINK_EXE)
test_framework_lpt_mpi: test_framework_lpt_mpi.o Global.o
	$(LINK_EXE)
test_framework_reconstruction_mpi: test_framework_reconstruction_mpi.o Global.o
	$(LINK_EXE)
test_framework_checkpoint_mpi: test_framework_checkpoint_mpi.o Global.o
	$(LINK_EXE)
MPIRUN ?= mpirun
NP_LIST ?= 1 2 4
ifeq ($(USE_MPI),true)
SAFETY_RUN = $(MPIRUN) -np $$np
SINGLE_RUN = $(MPIRUN) -np 1
else
NP_LIST = 1
SAFETY_RUN =
SINGLE_RUN =
endif
.PHONY: run safety survey-checkpoint-mpi survey-mpi framework-nonlinear-mpi bao-gradients bao-validation framework framework-mpi framework-lpt-mpi framework-reconstruction-mpi framework-checkpoint-mpi
# Each prepare/resume is a separate process; each rank count gets its own directory.
# Set TMPDIR to a shared filesystem for multi-node MPI test jobs.
survey-checkpoint-mpi: test_survey_checkpoint_mpi
	@set -e; survey_tmp=$$(mktemp -d "$${TMPDIR:-/tmp}/fml-survey-checkpoint.XXXXXX"); trap 'rm -rf "$$survey_tmp"' EXIT; \
	for np in $(NP_LIST); do \
	  $(SAFETY_RUN) ./test_survey_checkpoint_mpi "$$survey_tmp/np$$np" prepare; \
	  $(SAFETY_RUN) ./test_survey_checkpoint_mpi "$$survey_tmp/np$$np" resume; \
	done

survey-mpi: test_survey_likelihood_mpi
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_survey_likelihood_mpi; done
framework-nonlinear-mpi: test_framework_nonlinear_mpi
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_framework_nonlinear_mpi; done

bao-gradients: test_bao_gradients
	$(SINGLE_RUN) ./test_bao_gradients
BAO_SAMPLES ?= 600
BAO_MOCKS ?= 8
BAO_OUTPUT ?= bao_validation.tsv
BAO_MODEL ?= 1lpt
bao-validation: test_bao_validation
	$(SINGLE_RUN) ./test_bao_validation "$(BAO_OUTPUT)" $(BAO_SAMPLES) $(BAO_MOCKS) $(BAO_MODEL)
safety: test_field_level_safety
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_field_level_safety; done
framework: test_framework test_framework_resolution
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_framework; done
	$(SINGLE_RUN) ./test_framework_resolution
framework-mpi: test_framework_mpi
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_framework_mpi; done
framework-lpt-mpi: test_framework_lpt_mpi
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_framework_lpt_mpi; done
framework-reconstruction-mpi: test_framework_reconstruction_mpi
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_framework_reconstruction_mpi; done
framework-checkpoint-mpi: test_framework_checkpoint_mpi
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_framework_checkpoint_mpi; done
run: $(TARGETS) survey-checkpoint-mpi survey-mpi bao-gradients safety framework framework-mpi framework-lpt-mpi framework-reconstruction-mpi framework-checkpoint-mpi
	$(SINGLE_RUN) ./test_field_level
	$(SINGLE_RUN) ./test_bao_validation

.PHONY: particle-mesh
particle-mesh: test_particle_mesh
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_particle_mesh; done
run: particle-mesh

.PHONY: particle-mesh-adjoint
particle-mesh-adjoint: test_particle_mesh_adjoint
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_particle_mesh_adjoint; done
run: particle-mesh-adjoint

.PHONY: particle-mesh-posterior
particle-mesh-posterior: test_particle_mesh_posterior
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_particle_mesh_posterior; done
run: particle-mesh-posterior

.PHONY: particle-mesh-observation
particle-mesh-observation: test_particle_mesh_observation
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_particle_mesh_observation; done
run: particle-mesh-observation

# Generic distributed linear backend (the pencil grid requires MPI/FFTW).
PencilGrid.o: $(FML_ROOT)/PencilDomain/PencilGrid.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
test_framework_linear_grid_mpi: test_framework_linear_grid_mpi.o PencilGrid.o Global.o
	$(LINK_EXE)
.PHONY: framework-linear-grid-mpi
framework-linear-grid-mpi: test_framework_linear_grid_mpi
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_framework_linear_grid_mpi; done
ifeq ($(USE_MPI),true)
run: framework-linear-grid-mpi
endif

.PHONY: framework-lpt-grid-mpi
test_framework_lpt_grid_mpi: test_framework_lpt_grid_mpi.o PencilGrid.o Global.o
	$(LINK_EXE)
framework-lpt-grid-mpi: test_framework_lpt_grid_mpi
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_framework_lpt_grid_mpi; done
ifeq ($(USE_MPI),true)
run: framework-lpt-grid-mpi
endif

.PHONY: survey-grid-mpi checkpoint-grid-mpi
test_survey_grid_mpi: test_survey_grid_mpi.o PencilGrid.o Global.o
	$(LINK_EXE)
test_checkpoint_grid_mpi: test_checkpoint_grid_mpi.o PencilGrid.o Global.o
	$(LINK_EXE)
survey-grid-mpi: test_survey_grid_mpi
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_survey_grid_mpi; done
checkpoint-grid-mpi: test_checkpoint_grid_mpi
	@set -e; grid_tmp=$$(mktemp -d "$${TMPDIR:-/tmp}/fml-grid-checkpoint.XXXXXX"); trap 'rm -rf "$$grid_tmp"' EXIT; \
	for np in $(NP_LIST); do \
	  $(SAFETY_RUN) ./test_checkpoint_grid_mpi "$$grid_tmp/np$$np" prepare; \
	  $(SAFETY_RUN) ./test_checkpoint_grid_mpi "$$grid_tmp/np$$np" resume; \
	done
ifeq ($(USE_MPI),true)
run: survey-grid-mpi checkpoint-grid-mpi
endif

.PHONY: pm-hmc-smoke
pm-hmc-smoke:
	$(MAKE) -C ../validation/pm/hmc pm_hmc
	python3 ../validation/pm/hmc/smoke.py --exe ../validation/pm/hmc/pm_hmc --mpi $(USE_MPI) --mpirun "$(MPIRUN)" --np-list "$(NP_LIST)"
run: pm-hmc-smoke

PM_CHECKPOINT_BACKENDS := slab
ifeq ($(USE_MPI),true)
PM_CHECKPOINT_BACKENDS += pencil
endif
.PHONY: particle-mesh-checkpoint
test_particle_mesh_checkpoint: test_particle_mesh_checkpoint.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
particle-mesh-checkpoint: test_particle_mesh_checkpoint
	@set -e; pm_tmp=$$(mktemp -d "$${TMPDIR:-/tmp}/fml-pm-checkpoint.XXXXXX"); trap 'rm -rf "$$pm_tmp"' EXIT; \
	for backend in $(PM_CHECKPOINT_BACKENDS); do for np in $(NP_LIST); do for dense in 0 1; do \
	  $(SAFETY_RUN) ./test_particle_mesh_checkpoint "$$pm_tmp/$$backend-np$$np-dense$$dense" prepare $$backend 0 $$dense; \
	  $(SAFETY_RUN) ./test_particle_mesh_checkpoint "$$pm_tmp/$$backend-np$$np-dense$$dense" resume $$backend 0 $$dense; \
	done; \
	  $(SAFETY_RUN) ./test_particle_mesh_checkpoint "$$pm_tmp/$$backend-np$$np-compensated" prepare $$backend 0 1 1; \
	  $(SAFETY_RUN) ./test_particle_mesh_checkpoint "$$pm_tmp/$$backend-np$$np-compensated" resume $$backend 0 1 1; \
	  $(SAFETY_RUN) ./test_particle_mesh_checkpoint "$$pm_tmp/$$backend-np$$np-observed" prepare $$backend 0 1 1 4; \
	  $(SAFETY_RUN) ./test_particle_mesh_checkpoint "$$pm_tmp/$$backend-np$$np-observed" resume $$backend 0 1 1 4; \
	done; done
run: particle-mesh-checkpoint

.PHONY: map-optimizer
test_map_optimizer: test_map_optimizer.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
map-optimizer: test_map_optimizer
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_map_optimizer; done
run: map-optimizer

.PHONY: particle-mesh-observer
test_particle_mesh_observer: test_particle_mesh_observer.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
particle-mesh-observer: test_particle_mesh_observer
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_particle_mesh_observer; done
run: particle-mesh-observer

.PHONY: pm-assignment
pm-assignment: test_pm_assignment
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_pm_assignment; done
run: pm-assignment

run: eulerian-bias
.PHONY: eulerian-bias
test_eulerian_bias: test_eulerian_bias.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
eulerian-bias: test_eulerian_bias
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_eulerian_bias; done

run: eulerian-lpt
.PHONY: eulerian-lpt
test_eulerian_lpt: test_eulerian_lpt.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
eulerian-lpt: test_eulerian_lpt
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_eulerian_lpt; done

run: fixed-ic
.PHONY: fixed-ic
test_fixed_ic: test_fixed_ic.o Global.o
	$(LINK_EXE)
fixed-ic: test_fixed_ic
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_fixed_ic; done

run: periodic-catalog
.PHONY: periodic-catalog
test_periodic_catalog: test_periodic_catalog.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
periodic-catalog: test_periodic_catalog
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_periodic_catalog; done

.PHONY: framework-nufft-grid-mpi
framework-nufft-grid-mpi: test_framework_lpt_grid_mpi
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_framework_lpt_grid_mpi --pswf; done
ifeq ($(USE_MPI),true)
run: framework-nufft-grid-mpi
endif

.PHONY: compact-eulerian-lpt
test_compact_eulerian_lpt: test_compact_eulerian_lpt.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
compact-eulerian-lpt: test_compact_eulerian_lpt
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_compact_eulerian_lpt; done
run: compact-eulerian-lpt

.PHONY: eulerian-checkpoint
test_eulerian_checkpoint: test_eulerian_checkpoint.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
eulerian-checkpoint: test_eulerian_checkpoint
	@set -e; checkpoint_tmp=$$(mktemp -d "$${TMPDIR:-/tmp}/fml-eulerian-checkpoint.XXXXXX"); trap 'rm -rf "$$checkpoint_tmp"' EXIT; \
	for np in $(NP_LIST); do \
	  $(SAFETY_RUN) ./test_eulerian_checkpoint "$$checkpoint_tmp/np$$np" prepare; \
	  $(SAFETY_RUN) ./test_eulerian_checkpoint "$$checkpoint_tmp/np$$np" resume; \
	done
run: eulerian-checkpoint

.PHONY: eulerian-metric
test_eulerian_metric: test_eulerian_metric.o Global.o $(PM_GRID_OBJECT)
	$(LINK_EXE)
eulerian-metric: test_eulerian_metric
	@set -e; for np in $(NP_LIST); do $(SAFETY_RUN) ./test_eulerian_metric; done
run: eulerian-metric
