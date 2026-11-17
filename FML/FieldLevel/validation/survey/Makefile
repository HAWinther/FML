USE_PENCIL_GRID ?= false
USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := survey_validate survey_reference
ifeq ($(USE_PENCIL_GRID),true)
ifneq ($(USE_MPI),true)
$(error USE_PENCIL_GRID=true requires USE_MPI=true)
endif
OPTIONS += -DFML_FIELDLEVEL_PENCIL_GRID
PENCIL_OBJECT := PencilGrid.o
endif
include ../../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
survey_validate: Main.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

Reference.o: Main.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC) -DACTIVE_MODE_REFERENCE
survey_reference: Reference.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

PencilGrid.o: $(FML_ROOT)/PencilDomain/PencilGrid.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
