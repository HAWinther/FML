// This extension deliberately owns no MPI/FFTW/OpenMP runtime lifecycle.
#if defined(USE_MPI) || defined(USE_OMP) || defined(USE_FFTW)
#error "fml-analysis is a serial extension; do not inject FML parallel/FFT flags"
#endif
#include <FML/Global/Global.cpp>
