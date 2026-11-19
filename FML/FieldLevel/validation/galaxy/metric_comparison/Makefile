USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := gn_tempering
include ../../../../Makefile.config
INC += -I$(FML_ROOT)/FieldLevel/validation/galaxy/tempering
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
GNMain.cpp: make_driver.py $(FML_ROOT)/FieldLevel/validation/galaxy/tempering/Main.cpp
	python3 make_driver.py $(FML_ROOT)/FieldLevel/validation/galaxy/tempering/Main.cpp $@
gn_tempering: GNMain.o Global.o
	$(LINK_EXE)
