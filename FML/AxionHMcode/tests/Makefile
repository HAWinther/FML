USE_MPI ?= false
USE_OMP ?= false
USE_GSL ?= true
TARGETS := test_axionhmcode
include ../../Makefile.config
FML_MODULES := AxionHMcode Global
OPTIONS += -DAXION_REFERENCE_DIR=\"$(FML_ROOT)/AxionHMcode/tests/reference\"
test_axionhmcode: test_axionhmcode.o AxionHMcode.o Global.o
	$(LINK_EXE)
.PHONY: run
run: test_axionhmcode
	./test_axionhmcode
