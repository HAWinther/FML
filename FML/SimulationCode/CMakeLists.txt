#===================================================================
# FML: SimulationCode (CMake build)
#
# An alternative to the Makefile (both build the same nbody executable
# with the same options). Out-of-source build:
#
#   cmake -S FML/SimulationCode -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ./build/nbody parameterfile.lua
#
# Options (defaults as in the Makefile), e.g. -DFML_USE_MPI=OFF:
#   NDIM_SPACE                3      Number of dimensions of space
#   FML_USE_MPI               ON     MPI (also FFTW-MPI)
#   FML_USE_OMP               ON     OpenMP
#   FML_USE_FFTW_THREADS      ON     Threads in FFTW (with OpenMP)
#   FML_USE_MEMORYLOG         ON     Log the allocations in the library
#   FML_USE_HDF5              OFF    GADGET HDF5 snapshots, lightcones and halo catalogues
#   FML_USE_HEALPIX           OFF    Weak lensing maps (HEALPIX_ROOT, CFITSIO_ROOT as hints)
#   FML_USE_CHUNKPIX          OFF    Store the HEALPix maps as chunks (needs FML_USE_HEALPIX)
#   FML_USE_MINIMAL_PARTICLES OFF    Compile for all 7 particle types (see src/Particles.h)
#   FML_FFTW_PRECISION        double double, float or long (long double)
#   FML_NATIVE_ARCH           ON     -march=native (turn off when building for another machine)
#   FML_USE_SANITIZER         OFF    Address sanitizer
#   FML_BUILD_TESTS           ON     The tests in tests/ (run with ctest)
#
# Libraries not in a standard place are found with the usual hints, e.g.
#   -DCMAKE_PREFIX_PATH=$HOME/local  (FFTW, GSL, Lua, HDF5, ...)
#   -DFFTW_ROOT=/path/to/fftw -DGSL_ROOT_DIR=... -DLUA_DIR=... -DHDF5_ROOT=...
#===================================================================

cmake_minimum_required(VERSION 3.16)
project(FML_SimulationCode LANGUAGES C CXX)

# The source folder has the Makefile build (objects and nbody): keep the CMake build out of it
if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(FATAL_ERROR "Build out of source, e.g. cmake -S . -B build (then remove CMakeCache.txt and CMakeFiles/ here)")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# The Makefile uses -O3 (CMake's Release is -O3 -DNDEBUG)

#-------------------------------------------------------------------
# Options
#-------------------------------------------------------------------
set(NDIM_SPACE 3 CACHE STRING "Number of dimensions of space")
option(FML_USE_MPI "Use MPI" ON)
option(FML_USE_OMP "Use OpenMP" ON)
option(FML_USE_FFTW_THREADS "Use threads in FFTW (with OpenMP)" ON)
option(FML_USE_MEMORYLOG "Log the allocations in the library" ON)
option(FML_USE_HDF5 "GADGET HDF5 output (needs the HDF5 C library)" OFF)
option(FML_USE_HEALPIX "Weak lensing maps (needs HEALPix C++ and CFITSIO)" OFF)
option(FML_USE_CHUNKPIX "Store the HEALPix maps as chunks" OFF)
option(FML_USE_MINIMAL_PARTICLES "Compile for all 7 particle types and pick the smallest for a run" OFF)
set(FML_FFTW_PRECISION double CACHE STRING "FFTW precision: double, float or long")
set_property(CACHE FML_FFTW_PRECISION PROPERTY STRINGS double float long)
option(FML_NATIVE_ARCH "Compile with -march=native" ON)
option(FML_USE_SANITIZER "Address sanitizer" OFF)
option(FML_BUILD_TESTS "Build the tests in tests/" ON)

if(FML_USE_CHUNKPIX AND NOT FML_USE_HEALPIX)
  message(FATAL_ERROR "FML_USE_CHUNKPIX needs FML_USE_HEALPIX")
endif()

# This folder is FML/SimulationCode: headers are included as <FML/...>
get_filename_component(FML_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
get_filename_component(FML_INCLUDE "${FML_ROOT}/.." ABSOLUTE)

#-------------------------------------------------------------------
# External libraries
#-------------------------------------------------------------------
find_package(GSL REQUIRED)
find_package(Lua REQUIRED)
if(FML_USE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()
if(FML_USE_OMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# FFTW (no CMake module of its own: library names by precision)
if(FML_FFTW_PRECISION STREQUAL "double")
  set(_fftw_suffix "")
elseif(FML_FFTW_PRECISION STREQUAL "float")
  set(_fftw_suffix "f")
elseif(FML_FFTW_PRECISION STREQUAL "long")
  set(_fftw_suffix "l")
else()
  message(FATAL_ERROR "FML_FFTW_PRECISION must be double, float or long (not ${FML_FFTW_PRECISION})")
endif()
find_path(FFTW_INCLUDE_DIR fftw3.h HINTS ${FFTW_ROOT} $ENV{FFTW_ROOT} $ENV{FFTW_DIR} PATH_SUFFIXES include)
find_library(FFTW_LIBRARY fftw3${_fftw_suffix} HINTS ${FFTW_ROOT} $ENV{FFTW_ROOT} $ENV{FFTW_DIR} PATH_SUFFIXES lib lib64)
if(NOT FFTW_INCLUDE_DIR OR NOT FFTW_LIBRARY)
  message(FATAL_ERROR "FFTW (fftw3${_fftw_suffix}) not found: set FFTW_ROOT or CMAKE_PREFIX_PATH")
endif()
set(FFTW_LIBRARIES "")
if(FML_USE_MPI)
  find_library(FFTW_MPI_LIBRARY fftw3${_fftw_suffix}_mpi HINTS ${FFTW_ROOT} $ENV{FFTW_ROOT} $ENV{FFTW_DIR}
               PATH_SUFFIXES lib lib64)
  if(NOT FFTW_MPI_LIBRARY)
    message(FATAL_ERROR "FFTW-MPI (fftw3${_fftw_suffix}_mpi) not found (or use -DFML_USE_MPI=OFF)")
  endif()
  list(APPEND FFTW_LIBRARIES ${FFTW_MPI_LIBRARY})
endif()
set(_use_fftw_threads OFF)
if(FML_USE_OMP AND FML_USE_FFTW_THREADS)
  find_library(FFTW_THREADS_LIBRARY fftw3${_fftw_suffix}_threads HINTS ${FFTW_ROOT} $ENV{FFTW_ROOT} $ENV{FFTW_DIR}
               PATH_SUFFIXES lib lib64)
  if(NOT FFTW_THREADS_LIBRARY)
    message(FATAL_ERROR "FFTW threads (fftw3${_fftw_suffix}_threads) not found (or use -DFML_USE_FFTW_THREADS=OFF)")
  endif()
  list(APPEND FFTW_LIBRARIES ${FFTW_THREADS_LIBRARY})
  set(_use_fftw_threads ON)
endif()
list(APPEND FFTW_LIBRARIES ${FFTW_LIBRARY})
add_library(fml_fftw INTERFACE)
target_include_directories(fml_fftw INTERFACE ${FFTW_INCLUDE_DIR})
target_link_libraries(fml_fftw INTERFACE ${FFTW_LIBRARIES})

if(FML_USE_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS C)
endif()

if(FML_USE_HEALPIX)
  find_path(HEALPIX_INCLUDE_DIR healpix_map.h HINTS ${HEALPIX_ROOT} $ENV{HEALPIX}
            PATH_SUFFIXES include include/healpix_cxx healpix_cxx)
  find_library(HEALPIX_LIBRARY healpix_cxx HINTS ${HEALPIX_ROOT} $ENV{HEALPIX} PATH_SUFFIXES lib)
  # libsharp is part of libhealpix_cxx in newer versions of HEALPix
  find_library(SHARP_LIBRARY sharp HINTS ${HEALPIX_ROOT} $ENV{HEALPIX} PATH_SUFFIXES lib)
  find_path(CFITSIO_INCLUDE_DIR fitsio.h HINTS ${CFITSIO_ROOT} PATH_SUFFIXES include include/cfitsio)
  find_library(CFITSIO_LIBRARY cfitsio HINTS ${CFITSIO_ROOT} PATH_SUFFIXES lib)
  if(NOT HEALPIX_INCLUDE_DIR OR NOT HEALPIX_LIBRARY OR NOT CFITSIO_LIBRARY)
    message(FATAL_ERROR "HEALPix C++ / CFITSIO not found: set HEALPIX_ROOT and CFITSIO_ROOT")
  endif()
  add_library(fml_healpix INTERFACE)
  target_include_directories(fml_healpix INTERFACE ${HEALPIX_INCLUDE_DIR} ${CFITSIO_INCLUDE_DIR})
  target_link_libraries(fml_healpix INTERFACE ${HEALPIX_LIBRARY})
  if(SHARP_LIBRARY)
    target_link_libraries(fml_healpix INTERFACE ${SHARP_LIBRARY})
  endif()
  target_link_libraries(fml_healpix INTERFACE ${CFITSIO_LIBRARY})
endif()

#-------------------------------------------------------------------
# Compile settings shared by nbody and the tests
#-------------------------------------------------------------------
add_library(fml_settings INTERFACE)
target_include_directories(fml_settings INTERFACE ${FML_INCLUDE})
target_compile_definitions(fml_settings INTERFACE
  NDIM_SPACE=${NDIM_SPACE}
  USE_FFTW USE_GSL
  # Errors in splines and ODEs are exceptions (as in Cosmology/EinsteinBoltzmannSolver/Cosmology.mk)
  SPLINE_THROW_ON_ERROR ODESOLVER_THROW_ON_ERROR
  # Data files shipped with FML are found relative to the FML folder (Utils::fml_file)
  FML_ROOT_DIR="${FML_ROOT}")
target_compile_options(fml_settings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
target_link_libraries(fml_settings INTERFACE fml_fftw GSL::gsl)
if(FML_NATIVE_ARCH)
  target_compile_options(fml_settings INTERFACE -march=native)
endif()
if(FML_USE_MPI)
  target_compile_definitions(fml_settings INTERFACE USE_MPI)
  target_link_libraries(fml_settings INTERFACE MPI::MPI_CXX)
endif()
if(FML_USE_OMP)
  target_compile_definitions(fml_settings INTERFACE USE_OMP)
  target_link_libraries(fml_settings INTERFACE OpenMP::OpenMP_CXX)
endif()
if(_use_fftw_threads)
  target_compile_definitions(fml_settings INTERFACE USE_FFTW_THREADS)
endif()
if(FML_FFTW_PRECISION STREQUAL "float")
  target_compile_definitions(fml_settings INTERFACE SINGLE_PRECISION_FFTW)
elseif(FML_FFTW_PRECISION STREQUAL "long")
  target_compile_definitions(fml_settings INTERFACE LONG_DOUBLE_PRECISION_FFTW)
endif()
if(FML_USE_MEMORYLOG)
  target_compile_definitions(fml_settings INTERFACE MEMORY_LOGGING)
endif()
if(FML_USE_SANITIZER)
  target_compile_options(fml_settings INTERFACE -g -fsanitize=address)
  target_link_options(fml_settings INTERFACE -fsanitize=address)
endif()
if(FML_USE_HDF5)
  target_compile_definitions(fml_settings INTERFACE USE_HDF5)
  target_link_libraries(fml_settings INTERFACE HDF5::HDF5)
endif()
if(FML_USE_HEALPIX)
  target_compile_definitions(fml_settings INTERFACE USE_HEALPIX)
  target_link_libraries(fml_settings INTERFACE fml_healpix)
  if(FML_USE_CHUNKPIX)
    target_compile_definitions(fml_settings INTERFACE USE_CHUNKPIX)
  endif()
endif()

#-------------------------------------------------------------------
# The FML library sources and FML's Einstein-Boltzmann solver (for
# ic_type_of_input = internal), compiled once for nbody and the tests
#-------------------------------------------------------------------
set(FML_CORE_SOURCES
  ${FML_ROOT}/Global/Global.cpp
  ${FML_ROOT}/Spline/Spline.cpp
  ${FML_ROOT}/ODESolver/ODESolver.cpp
  ${FML_ROOT}/ParameterMap/ParameterMap.cpp
  ${FML_ROOT}/FileUtils/FileUtils.cpp
  ${FML_ROOT}/Math/Math.cpp
  ${FML_ROOT}/Math/NumericalMethods.cpp
  ${FML_ROOT}/GadgetUtils/GadgetUtils.cpp)
# All .cpp files of the solver modules (as in Cosmology.mk) plus ParameterFile.cpp and EinsteinBoltzmannSolver.cpp
# (not the solver's own programs). CONFIGURE_DEPENDS: re-run the glob when files are added
file(GLOB FML_COSMO_SOURCES CONFIGURE_DEPENDS
  ${FML_ROOT}/Cosmology/Background/*.cpp
  ${FML_ROOT}/Cosmology/Recombination/*.cpp
  ${FML_ROOT}/Cosmology/Perturbations/*.cpp
  ${FML_ROOT}/Cosmology/Spectra/*.cpp)
list(APPEND FML_COSMO_SOURCES
  ${FML_ROOT}/ReACT/ReACT.cpp
  ${FML_ROOT}/HaloModel/HMcode2020.cpp
  ${FML_ROOT}/HaloModel/HMcode2016.cpp
  ${FML_ROOT}/HaloModel/Halomodel.cpp
  ${FML_ROOT}/HaloModel/HaloModelTools.cpp
  ${FML_ROOT}/AxionHMcode/AxionHMcode.cpp
  ${FML_ROOT}/SphericalCollapse/SphericalCollapse.cpp
  ${FML_ROOT}/SphericalCollapse/SphericalCollapseModel.cpp
  ${FML_ROOT}/Cosmology/EinsteinBoltzmannSolver/ParameterFile.cpp
  ${FML_ROOT}/Cosmology/EinsteinBoltzmannSolver/EinsteinBoltzmannSolver.cpp)

add_library(fml_core STATIC ${FML_CORE_SOURCES} ${FML_COSMO_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/TransferInput.cpp)
target_link_libraries(fml_core PUBLIC fml_settings)

#-------------------------------------------------------------------
# nbody: the simulation is compiled once per particle type, each in
# its own file (compiled in parallel with cmake --build -j)
#-------------------------------------------------------------------
set(RUN_SIMULATION_SOURCES src/RunSimulation_COLAScaleDepWithMass.cpp)
if(FML_USE_MINIMAL_PARTICLES)
  list(APPEND RUN_SIMULATION_SOURCES
    src/RunSimulation_COLAWithMass.cpp src/RunSimulation_BasicWithMass.cpp src/RunSimulation_COLAScaleDep.cpp
    src/RunSimulation_COLA.cpp src/RunSimulation_BasicQ.cpp src/RunSimulation_Basic.cpp)
endif()

add_executable(nbody src/Main.cpp ${RUN_SIMULATION_SOURCES} src/BaryonificationOutput.cpp)
target_link_libraries(nbody PRIVATE fml_core)
target_compile_definitions(nbody PRIVATE USE_LUA $<$<BOOL:${FML_USE_MINIMAL_PARTICLES}>:USE_MINIMAL_PARTICLES>)
target_include_directories(nbody PRIVATE ${LUA_INCLUDE_DIR})
target_link_libraries(nbody PRIVATE ${LUA_LIBRARIES} ${CMAKE_DL_LIBS})

#-------------------------------------------------------------------
# Tests (as tests/Makefile; run with ctest --test-dir build)
#-------------------------------------------------------------------
if(FML_BUILD_TESTS)
  enable_testing()
  # Explicit, longer accuracy experiment (not part of the default build/CTest).
  add_executable(benchmark_clustered_force EXCLUDE_FROM_ALL tests/benchmark_clustered_force.cpp)
  target_link_libraries(benchmark_clustered_force PRIVATE fml_core)
  set_target_properties(benchmark_clustered_force PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_executable(benchmark_short_range_dynamics EXCLUDE_FROM_ALL tests/benchmark_short_range_dynamics.cpp src/BaryonificationOutput.cpp)
  target_link_libraries(benchmark_short_range_dynamics PRIVATE fml_core ${LUA_LIBRARIES} ${CMAKE_DL_LIBS})
  target_compile_definitions(benchmark_short_range_dynamics PRIVATE USE_LUA)
  target_include_directories(benchmark_short_range_dynamics PRIVATE ${LUA_INCLUDE_DIR})
  set_target_properties(benchmark_short_range_dynamics PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_executable(test_short_range_simulation EXCLUDE_FROM_ALL tests/test_short_range_simulation.cpp src/BaryonificationOutput.cpp)
  target_link_libraries(test_short_range_simulation PRIVATE fml_core ${LUA_LIBRARIES} ${CMAKE_DL_LIBS})
  target_compile_definitions(test_short_range_simulation PRIVATE USE_LUA)
  target_include_directories(test_short_range_simulation PRIVATE ${LUA_INCLUDE_DIR})
  set_target_properties(test_short_range_simulation PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  set(_tests test_baryonification_parameters test_cola_frame test_models test_lightcone_crossing
             test_short_range_force
             test_gravity_short_range)
  foreach(t IN LISTS _tests)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE fml_core)
    set_target_properties(${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    # Run as one task (also with MPI: a singleton MPI process needs no mpirun)
    add_test(NAME ${t} COMMAND ${t} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endif()
