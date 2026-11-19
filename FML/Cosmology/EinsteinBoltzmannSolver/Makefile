#===================================================================
# FML: Cosmology/EinsteinBoltzmannSolver
#
#   make             ./cmb (the solver: ./cmb params/planck2018.ini)
#   make milestones  ./milestones (the data of the report), make benchmark  ./benchmark
#   make test        build and run the tests in tests/ (from this folder)
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_OMP=false USE_DEBUG=true
#===================================================================

USE_OMP             ?= true
USE_GSL             ?= true
USE_MPI             ?= false

TARGETS := cmb

include ../../Makefile.config
include Cosmology.mk

FML_MODULES := $(COSMO_MODULES) $(COSMO_FML_MODULES)
EXTRA_CLEAN := milestones benchmark build

# The objects go to build/: this folder is also a source folder of the solver, and objects here would be found
# through VPATH (and reused with the wrong flags) by the other programs that link the solver (CosmologyFitting, ...)
OBJ := $(CURDIR)/build
$(OBJ)/%.o: %.cpp
	@mkdir -p $(OBJ)
	$(CC) $(DEPFLAGS) -c -o $@ $< $(OPTIONS) $(INC)
-include $(wildcard $(OBJ)/*.d)

cmb: $(addprefix $(OBJ)/,Main.o $(COSMO_OBJS))
	$(LINK_EXE)

milestones: $(addprefix $(OBJ)/,Milestones.o $(COSMO_OBJS))
	$(LINK_EXE)

benchmark: $(addprefix $(OBJ)/,Benchmark.o $(COSMO_OBJS))
	$(LINK_EXE)

.PHONY: test
test:
	@$(MAKE) --no-print-directory -C tests run
