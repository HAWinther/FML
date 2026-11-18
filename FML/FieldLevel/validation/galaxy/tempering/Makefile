USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
USE_PENCIL_GRID ?= false
TARGETS := tempering_pilot tempering_test tempering_diagnostics tempering_mock_test tempering_prior_test
ifeq ($(USE_PENCIL_GRID),true)
ifneq ($(USE_MPI),true)
$(error USE_PENCIL_GRID=true requires USE_MPI=true)
endif
OPTIONS += -DFML_FIELDLEVEL_PENCIL_GRID
PENCIL_OBJECT := PencilGrid.o
endif
include ../../../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
PencilGrid.o: $(FML_ROOT)/PencilDomain/PencilGrid.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
tempering_pilot: Main.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

tempering_test: Test.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)

tempering_diagnostics: Diagnostics.o
	$(LINK_EXE)

tempering_mock_test: MockTest.o
	$(LINK_EXE)

tempering_prior_test: PriorTest.o Global.o $(PENCIL_OBJECT)
	$(LINK_EXE)
