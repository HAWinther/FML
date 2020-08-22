#ifndef FFTWGLOBAL_HEADER
#define FFTWGLOBAL_HEADER

#include <complex>

#ifdef USE_FFTW
#include <fftw3.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#ifdef USE_FFTW
#include <fftw3-mpi.h>
#endif
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

//==========================================================================
// This file contains type definition and macros for avoiding alot of
// ifdefs wrt float/double/long double FFTW transfors
// It also contains a singleton for automatically initializing FFTW
// with MPI and threads
//==========================================================================

#ifdef SINGLE_PRECISION_FFTW
using FloatType = float;
#else
#ifdef LONG_DOUBLE_PRECISION_FFTW
using FloatType = long double;
#else
using FloatType = double;
#endif
#endif
using ComplexType = std::complex<FloatType>;

#ifdef USE_FFTW
#ifdef SINGLE_PRECISION_FFTW
using my_fftw_complex = fftwf_complex;
using my_fftw_plan = fftwf_plan;
#define SET_FFTW_NTHREADS fftwf_plan_with_nthreads
#define INIT_FFTW_THREADS fftwf_init_threads
#ifdef USE_MPI
#define INIT_FFTW_MPI fftwf_mpi_init
#define CLEANUP_FFTW_MPI fftwf_mpi_cleanup
#define MAKE_PLAN_R2C fftwf_mpi_plan_dft_r2c
#define MAKE_PLAN_C2R fftwf_mpi_plan_dft_c2r
#define MPI_FFTW_LOCAL_SIZE fftwf_mpi_local_size
#else
#define MAKE_PLAN_R2C fftwf_plan_dft_r2c
#define MAKE_PLAN_C2R fftwf_plan_dft_c2r
#endif
#define EXECUTE_FFT fftwf_execute
#define DESTROY_PLAN fftwf_destroy_plan
#else // Single precision
#ifdef LONG_DOUBLE_PRECISION_FFTW
using my_fftw_complex = fftwl_complex;
using my_fftw_plan = fftwl_plan;
#define SET_FFTW_NTHREADS fftwl_plan_with_nthreads
#define INIT_FFTW_THREADS fftwl_init_threads
#ifdef USE_MPI
#define INIT_FFTW_MPI fftwl_mpi_init
#define CLEANUP_FFTW_MPI fftwl_mpi_cleanup
#define MAKE_PLAN_R2C fftwl_mpi_plan_dft_r2c
#define MAKE_PLAN_C2R fftwl_mpi_plan_dft_c2r
#define MPI_FFTW_LOCAL_SIZE fftwl_mpi_local_size
#else
#define MAKE_PLAN_R2C fftwl_plan_dft_r2c
#define MAKE_PLAN_C2R fftwl_plan_dft_c2r
#endif
#define EXECUTE_FFT fftwl_execute
#define DESTROY_PLAN fftwl_destroy_plan
#else // Long double precision
using my_fftw_complex = fftw_complex;
using my_fftw_plan = fftw_plan;
#define SET_FFTW_NTHREADS fftw_plan_with_nthreads
#define INIT_FFTW_THREADS fftw_init_threads
#ifdef USE_MPI
#define INIT_FFTW_MPI fftw_mpi_init
#define CLEANUP_FFTW_MPI fftw_mpi_cleanup
#define MAKE_PLAN_R2C fftw_mpi_plan_dft_r2c
#define MAKE_PLAN_C2R fftw_mpi_plan_dft_c2r
#define MPI_FFTW_LOCAL_SIZE fftw_mpi_local_size
#else
#define MAKE_PLAN_R2C fftw_plan_dft_r2c
#define MAKE_PLAN_C2R fftw_plan_dft_c2r
#endif
#define EXECUTE_FFT fftw_execute
#define DESTROY_PLAN fftw_destroy_plan
#endif // Double precision
#endif
#endif

#endif
