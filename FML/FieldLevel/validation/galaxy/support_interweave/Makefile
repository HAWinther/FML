USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := support_test support_pilot
include ../../../../Makefile.config
INC += -I$(FML_ROOT)/FieldLevel/validation/galaxy/tempering
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
PencilGrid.o: $(FML_ROOT)/PencilDomain/PencilGrid.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
SupportMain.cpp: make_driver.py $(FML_ROOT)/FieldLevel/validation/galaxy/tempering/Main.cpp
	python3 make_driver.py $(FML_ROOT)/FieldLevel/validation/galaxy/tempering/Main.cpp $@
support_pilot: SupportMain.o Global.o
	$(LINK_EXE)
support_test: Test.o Global.o
	$(LINK_EXE)
support_test_mpi: Test.o Global.o PencilGrid.o
	$(LINK_EXE)
