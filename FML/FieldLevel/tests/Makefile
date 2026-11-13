USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := test_survey_likelihood_mpi test_bao_gradients test_bao_validation test_field_level test_field_level_safety test_framework test_framework_resolution test_framework_mpi test_framework_lpt_mpi test_framework_reconstruction_mpi test_framework_checkpoint_mpi
include ../../Makefile.config
FML_MODULES := Global
# Do not reuse a VPATH object compiled with another module's MPI/precision flags.
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
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
.PHONY: run safety survey-mpi framework-nonlinear-mpi bao-gradients bao-validation framework framework-mpi framework-lpt-mpi framework-reconstruction-mpi framework-checkpoint-mpi
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
run: $(TARGETS) survey-mpi bao-gradients safety framework framework-mpi framework-lpt-mpi framework-reconstruction-mpi framework-checkpoint-mpi
	$(SINGLE_RUN) ./test_field_level
	$(SINGLE_RUN) ./test_bao_validation
