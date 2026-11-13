USE_MPI ?= false
USE_OMP ?= true
USE_FFTW ?= true
TARGETS := fieldlevel calibration framework framework_mpi checkpoint_mpi
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

framework_mpi: FrameworkMPI.o Global.o
	$(LINK_EXE)

checkpoint_mpi: CheckpointMPI.o Global.o
	$(LINK_EXE)
