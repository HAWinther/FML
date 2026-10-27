#===================================================================
# The Einstein-Boltzmann solver as a list of FML modules and objects,
# for the Makefiles of the programs and tests that link it. Include
# after Makefile.config (needs FML_ROOT).
#===================================================================

COSMO_MODULES := Cosmology/Background Cosmology/Models Cosmology/Recombination \
                 Cosmology/Perturbations Cosmology/Spectra Cosmology/EinsteinBoltzmannSolver
# All .cpp files of the solver (ParameterFile.cpp from EinsteinBoltzmannSolver, not its programs)
COSMO_OBJS := $(notdir $(patsubst %.cpp,%.o,$(wildcard $(addsuffix /*.cpp,$(addprefix $(FML_ROOT)/,$(filter-out Cosmology/EinsteinBoltzmannSolver,$(COSMO_MODULES))))))) \
              ParameterFile.o

# FML modules the solver uses (FML_MODULES must include them: $(COSMO_FML_MODULES))
COSMO_FML_MODULES := Spline ODESolver Math
COSMO_OBJS += Spline.o ODESolver.o NumericalMethods.o
# Errors in splines and ODEs are exceptions also with MPI (a failing parameter point in an MCMC is rejected)
OPTIONS += -DSPLINE_THROW_ON_ERROR -DODESOLVER_THROW_ON_ERROR

# Data files shipped with FML are found relative to the FML folder (Utils::fml_file)
OPTIONS += -DFML_ROOT_DIR=\"$(FML_ROOT)\"
