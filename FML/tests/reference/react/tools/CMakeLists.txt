# Build the unmodified upstream ACTio-ReACTio (a36dda02) as a shared library, only to regenerate the ReACT reference
# values of the tests (see README.md here). Not part of FML.
cmake_minimum_required(VERSION 3.15)
project(FMLReACTBackend LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(REACT_SOURCE "" CACHE PATH "ACTio-ReACTio checkout (a36dda02db6b537a4ddcc7bf044c446c10a58373)")
set(SUNDIALS_ROOT "" CACHE PATH "SUNDIALS 4.1.0 installation")
if(NOT EXISTS "${REACT_SOURCE}/pyreact/react_wrapper.cpp")
    message(FATAL_ERROR "Set REACT_SOURCE to the ACTio-ReACTio checkout")
endif()
execute_process(COMMAND git -C "${REACT_SOURCE}" rev-parse HEAD OUTPUT_VARIABLE react_revision OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT react_revision STREQUAL "a36dda02db6b537a4ddcc7bf044c446c10a58373")
    message(FATAL_ERROR "Unsupported ReACT revision: ${react_revision}; the FML ABI is pinned to a36dda02")
endif()
find_package(GSL REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_path(SUNDIALS_INCLUDE_DIR cvode/cvode.h HINTS "${SUNDIALS_ROOT}/include" REQUIRED)
foreach(component cvode nvecserial sunlinsoldense sunmatrixdense)
    find_library(SUNDIALS_${component} NAMES sundials_${component} HINTS "${SUNDIALS_ROOT}/lib" REQUIRED)
endforeach()
set(src "${REACT_SOURCE}/reactions/src")
set(copter_sources Closure.cpp BSPT.cpp BSPTN.cpp Common.cpp ConsistentSPT.cpp CorrelationFunction.cpp
    Cosmology.cpp Datafile.cpp FlowingWithTime.cpp GrowthFunction.cpp InterpolatedPS.cpp Kaiser.cpp
    LagrangianResummation.cpp LargeN.cpp LinearPS.cpp MonteCarlo.cpp NoWigglePS.cpp PowerSpectrum.cpp
    Quadrature.cpp RPT.cpp SPT.cpp SimpleRG.cpp SpecialFunctions.cpp Spline.cpp Timer.cpp array.cpp
    cfg.c pstring.cpp rng.c SCOL.cpp HALO.cpp BeyondLCDM.cpp)
list(TRANSFORM copter_sources PREPEND "${src}/")
# The upstream wrapper expects installed Copter headers. Stage them in the build tree without changing
# the source checkout or installing anything system-wide.
file(GLOB copter_headers "${src}/*.h" "${src}/*.inl")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/Copter")
file(COPY ${copter_headers} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/include/Copter")
add_library(fml_react_backend SHARED ${copter_sources} "${REACT_SOURCE}/pyreact/react_wrapper.cpp")
target_include_directories(fml_react_backend PRIVATE "${src}" "${REACT_SOURCE}/reactions"
    "${CMAKE_CURRENT_BINARY_DIR}/include" "${SUNDIALS_INCLUDE_DIR}")
target_compile_definitions(fml_react_backend PRIVATE DATADIR="${REACT_SOURCE}/reactions/data")
target_link_libraries(fml_react_backend PRIVATE GSL::gsl GSL::gslcblas OpenMP::OpenMP_CXX
    ${SUNDIALS_cvode} ${SUNDIALS_nvecserial} ${SUNDIALS_sunlinsoldense} ${SUNDIALS_sunmatrixdense})
install(TARGETS fml_react_backend LIBRARY DESTINATION lib)
install(FILES "${REACT_SOURCE}/LICENSE" DESTINATION share/fml-react RENAME REACT-LICENSE)
install(FILES "${REACT_SOURCE}/reactions/COPYING" DESTINATION share/fml-react RENAME COPTER-COPYING)
