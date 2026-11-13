USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := survey_validate survey_reference
include ../../Makefile.config
FML_MODULES := Global
Global.o: $(FML_ROOT)/Global/Global.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
survey_validate: Main.o Global.o
	$(LINK_EXE)

Reference.o: Main.cpp
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC) -DACTIVE_MODE_REFERENCE
survey_reference: Reference.o Global.o
	$(LINK_EXE)
