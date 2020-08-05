#ifndef FFTLOG_HEADER
#define FFTLOG_HEADER

//====================================================================
// 
// This version is adapted from https://github.com/slosar/FFTLog
// which again is based on code in Copter by JWG Carlson 
// https://github.com/jwgcarlson/Copter
// and the original implementation by Andrew Hamilton
// http://casa.colorado.edu/~ajsh/FFTLog/
//
// Requires the FFTW3 library
//
//====================================================================

#include <vector>
#include <cassert>
#include <complex>
#include <cmath>
#include <cstring>
#include <fftw3.h>

namespace FML {
  namespace SOLVERS {
    namespace FFTLog {
  
      using CDouble = std::complex<double>;
      using CVector = std::vector<CDouble>;
      using DVector = std::vector<double>;

      //==========================================================================
      // Compute the correlation function xi(r) from a power spectrum P(k), sampled
      // at logarithmically spaced points k[j]
      //==========================================================================
      std::pair<DVector,DVector> ComputeCorrelationFunction(const DVector &k, const DVector &pk);

      //==========================================================================
      // Compute the power spectrum P(k) from a correlation function xi(r), sampled
      // at logarithmically spaced points r[i]
      //==========================================================================
      std::pair<DVector,DVector> ComputePowerSpectrum(const DVector &r, const DVector &xi);

      //==========================================================================
      // Compute the function
      //   \xi_l^m(r) = \int_0^\infty \frac{dk}{2\pi^2} k^m j_l(kr) P(k)
      // The usual 2-point correlation function xi(r) is just xi_0^2(r)
      // The input k-values must be logarithmically spaced.
      // The resulting xi_l^m(r) will be evaluated at the dual r-values
      //   r[0] = 1/k[N-1], ..., r[N-1] = 1/k[0]
      //==========================================================================
      std::pair<DVector,DVector> ComputeXiLM(int ell, int m, const DVector &k, const DVector &pk);

      //==========================================================================
      // Compute the discrete Hankel transform of the function a(r). See the FFTLog
      // documentation for a description of exactly what this function computes.
      // If u is NULL, the transform coefficients will be computed and discarded
      // afterwards. If you plan on performing many consecutive transforms, it is
      // more efficient to pre-compute the u coefficients.
      //==========================================================================
      std::pair<DVector, CVector> DiscreteHankelTransform(
          const DVector &r, 
          const CVector &a, 
          double mu,
          double q = 0, 
          double kcrc = 1, 
          bool noring = true, 
          CVector* u = nullptr);

      //==========================================================================
      // Pre-compute the coefficients that appear in the FFTLog implementation of
      // the discrete Hankel transform. The parameters N, mu, and q here are the
      // same as for the function DiscreteHankelTransform.  The parameter L is defined
      // to be N times the logarithmic spacing of the input array, i.e.
      //   L = N * log(r[N-1]/r[0])/(N-1)
      //==========================================================================
      CVector ComputeCoefficients(int N, double mu, double q, double L, double kcrc);
    }
  }
}

#endif
