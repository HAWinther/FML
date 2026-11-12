#===================================================================
# FML: LatinHyperCube/tests
#
#   make       build every test_*.cpp
#   make run   build and run them
#===================================================================

USE_OMP             ?= false
USE_MPI             ?= false

TARGETS := $(patsubst %.cpp,%,$(wildcard test_*.cpp))

include ../../Makefile.config

FML_MODULES := Global

$(TARGETS): %: %.o Global.o
	$(LINK_EXE)

.PHONY: run
run: $(TARGETS)
	@status=0; for t in $(TARGETS); do echo "=== $$t ==="; ./$$t || status=1; done; exit $$status
