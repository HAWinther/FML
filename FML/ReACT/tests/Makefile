USE_MPI ?= false
USE_OMP ?= false
USE_GSL ?= true
TARGETS := test_react
include ../../Makefile.config
FML_MODULES := ReACT Spline Global
OPTIONS += -DSPLINE_THROW_ON_ERROR -DREACT_REFERENCE_DIR=\"$(FML_ROOT)/ReACT/tests/reference\"
test_react: test_react.o ReACT.o Spline.o Global.o
	$(LINK_EXE)
.PHONY: run
run: test_react
	./test_react
