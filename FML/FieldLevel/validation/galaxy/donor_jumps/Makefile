USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := donor_test donor_probe donor_pilot
include ../../../../Makefile.config
INC += -I$(FML_ROOT)/FieldLevel/validation/galaxy/tempering
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
DonorMain.cpp: make_driver.py $(FML_ROOT)/FieldLevel/validation/galaxy/tempering/Main.cpp
	python3 make_driver.py $(FML_ROOT)/FieldLevel/validation/galaxy/tempering/Main.cpp $@ --support-generator $(FML_ROOT)/FieldLevel/validation/galaxy/support_interweave/make_driver.py
donor_test: Test.o Global.o
	$(LINK_EXE)
donor_probe: Probe.o Global.o
	$(LINK_EXE)
donor_pilot: DonorMain.o Global.o
	$(LINK_EXE)
