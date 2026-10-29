USE_MPI ?= false
USE_OMP ?= false
USE_FFTW ?= true
TARGETS := test_field_level
include ../../Makefile.config
FML_MODULES := Global
test_field_level: test_field_level.o Global.o
	$(LINK_EXE)
.PHONY: run
run: $(TARGETS)
	./test_field_level
