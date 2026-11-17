#===================================================================
# FML: MultigridSolver/examples
#
# Compiler, library paths and how the options below are used are set
# in ../../Makefile.config. Override for your machine in Makefile.local
# there, or on the command line, e.g. make USE_MPI=false USE_DEBUG=true
#===================================================================

USE_MPI             ?= true
USE_OMP             ?= true
USE_FFTW            ?= true
USE_MEMORYLOG       ?= true

TARGETS := mgsolver

include ../../Makefile.config

FML_MODULES := Global MultigridSolver FileUtils
EXTRA_CLEAN := contsolver poissonsolver fofrsolver symm dgp recon amrpoisson amrdgp amrsymmetron amrdilaton

mgsolver: GenericExample.o Global.o
	$(LINK_EXE)

# Not built by default (make <name>)
contsolver: ContinuityExample.o Global.o
	$(LINK_EXE)

poissonsolver: PoissonExample.o Global.o
	$(LINK_EXE)

# Multigrid on an adaptively refined grid (AMRMultiGridSolver.h): mpirun -np 4 ./amrpoisson [2|3]
amrpoisson: AMRPoissonExample.o Global.o
	$(LINK_EXE)

# DGP on an adaptively refined grid (AMRDGPSolver.h) against the exact spherical solution: mpirun -np 4 ./amrdgp
amrdgp: AMRDGPExample.o Global.o
	$(LINK_EXE)

# The symmetron on an adaptively refined grid (AMRSymmetronSolver.h): mpirun -np 4 ./amrsymmetron [quick]
amrsymmetron: AMRSymmetronExample.o Global.o
	$(LINK_EXE)

# The dilaton on an adaptively refined grid (AMRDilatonSolver.h): mpirun -np 4 ./amrdilaton [quick]
amrdilaton: AMRDilatonExample.o Global.o
	$(LINK_EXE)

fofrsolver: FofrExample.o Global.o
	$(LINK_EXE)

symm: SymmetronExample.o Global.o
	$(LINK_EXE)

dgp: dgpExample.o Global.o
	$(LINK_EXE)

recon: Reconstruction.o Global.o FileUtils.o
	$(LINK_EXE)
