USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
USE_PENCIL_GRID ?= false
TARGETS := pm_stationarity_capture pm_stationarity_probe
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
pm_stationarity_capture: Capture.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)
pm_stationarity_probe: Probe.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)
