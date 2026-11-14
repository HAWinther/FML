USE_PENCIL_GRID ?= false
USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := survey_checkpoint_mpi survey_mpi fieldlevel calibration framework framework_mpi checkpoint_mpi
ifeq ($(USE_PENCIL_GRID),true)
ifneq ($(USE_MPI),true)
$(error USE_PENCIL_GRID=true requires USE_MPI=true)
endif
OPTIONS += -DFML_FIELDLEVEL_PENCIL_GRID
PENCIL_OBJECT := PencilGrid.o
endif
include ../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
fieldlevel: Main.o Global.o
	$(LINK_EXE)
calibration: Calibration.o Global.o
	$(LINK_EXE)

framework: Framework.o Global.o
	$(LINK_EXE)

framework_mpi: FrameworkMPI.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

checkpoint_mpi: CheckpointMPI.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

survey_mpi: SurveyMPI.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

survey_checkpoint_mpi: SurveyCheckpointMPI.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

PencilGrid.o: $(FML_ROOT)/PencilDomain/PencilGrid.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
