USE_MPI ?= false
USE_OMP ?= true
TARGETS := phase_halos
include ../../Makefile.config
FML_MODULES := Global
phase_halos: Main.o Global.o
	$(LINK_EXE)
