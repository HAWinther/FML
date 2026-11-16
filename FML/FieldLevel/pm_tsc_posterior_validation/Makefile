USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
USE_PENCIL_GRID ?= false
TARGETS := pm_tsc_posterior pm_tsc_diagnostics pm_tsc_joint
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
PencilGrid.o: $(FML_ROOT)/PencilDomain/PencilGrid.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
pm_tsc_posterior: Main.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)
pm_tsc_diagnostics: Diagnostics.o
	$(LINK_EXE)

JointMain.o: $(FML_ROOT)/FieldLevel/pm_tsc_joint_validation/Main.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
pm_tsc_joint: JointMain.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)
