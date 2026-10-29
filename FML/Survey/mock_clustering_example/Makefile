# End-to-end check of the survey mocks: Zel'dovich box -> mock and randoms -> P_ell (SurveyPower) against the
# window-convolved Kaiser model (SurveyWindow).   make && mpirun -np 2 ./mock_clustering
USE_MPI ?= true
USE_OMP ?= true
USE_FFTW ?= true
USE_GSL ?= true
TARGETS := mock_clustering
include ../../Makefile.config
FML_MODULES := Global Spline ODESolver
mock_clustering: Main.o Global.o Spline.o ODESolver.o
	$(LINK_EXE)
