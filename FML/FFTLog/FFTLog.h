#ifndef FFTLOG_HEADER
#define FFTLOG_HEADER

#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <fftw3.h>
#include <memory>
#include <vector>

namespace FML {
    namespace SOLVERS {

        //====================================================================
        ///
        /// This namespace deals with performing Hankel transforms. We can
        /// for example use this to convert between a power-spectrum and the
        /// correlation function. NB: this is only implemented in 3D. Its not hard to
        /// generalize, but not done as I can't see any use for it.
        ///
        /// This version is adapted from https://github.com/slosar/FFTLog
        /// which again is based on code in Copter by JWG Carlson
        /// https://github.com/jwgcarlson/Copter
        /// and the original implementation by Andrew Hamilton
        /// http://casa.colorado.edu/~ajsh/FFTLog/
        ///
        /// Requires the FFTW3 library.
        ///
        //====================================================================

        namespace FFTLog {

            using CDouble = std::complex<double>;
            using CVector = std::vector<CDouble>;
            using DVector = std::vector<double>;

            /// Reusable aligned FFT buffers, plans and one cached coefficient set.
            /// Prepare/destroy serially; use a separate workspace per concurrent caller.
            class Workspace {
              public:
                struct Statistics {
                    size_t plan_builds = 0;
                    size_t coefficient_builds = 0;
                };
                explicit Workspace(unsigned planner_flags = FFTW_ESTIMATE);
                ~Workspace();
                Workspace(Workspace &&) noexcept;
                Workspace & operator=(Workspace &&) noexcept;
                Workspace(const Workspace &) = delete;
                Workspace & operator=(const Workspace &) = delete;

                void prepare(size_t n);
                void clear() noexcept;
                size_t size() const noexcept;
                size_t buffer_bytes() const noexcept;
                Statistics statistics() const noexcept;

              private:
                struct Impl;
                std::unique_ptr<Impl> impl;
                unsigned planner_flags;
                DVector execute(const DVector &, double, double, double, bool, const CVector *);
                friend std::pair<DVector, CVector> DiscreteHankelTransform(
                    const DVector &, const CVector &, Workspace &, double, double, double, bool, CVector *);
                friend std::pair<DVector, DVector> ComputeXiLM(
                    int, int, const DVector &, const DVector &, Workspace &, double);
            };

            std::pair<DVector, DVector>
            ComputeCorrelationFunction(const DVector & k, const DVector & pk, Workspace & workspace);
            std::pair<DVector, DVector>
            ComputePowerSpectrum(const DVector & r, const DVector & xi, Workspace & workspace);
            std::pair<DVector, DVector>
            ComputeXiLM(int ell, int m, const DVector & k, const DVector & pk, Workspace & workspace, double q = 0.0);
            std::pair<DVector, CVector> DiscreteHankelTransform(const DVector & r,
                                                             const CVector & a,
                                                             Workspace & workspace,
                                                             double mu,
                                                             double q = 0,
                                                             double kcrc = 1,
                                                             bool noring = true,
                                                             CVector * u = nullptr);

            //==========================================================================
            /// @brief Compute the correlation function xi(r) from a power spectrum P(k), sampled
            /// at logarithmically spaced points k[j]
            //==========================================================================
            std::pair<DVector, DVector> ComputeCorrelationFunction(const DVector & k, const DVector & pk);

            //==========================================================================
            /// @brief Compute the power spectrum P(k) from a correlation function xi(r), sampled
            /// at logarithmically spaced points r[i]
            //==========================================================================
            std::pair<DVector, DVector> ComputePowerSpectrum(const DVector & r, const DVector & xi);

            //==========================================================================
            /// @brief Compute the function
            ///   \f$ \xi_l^m(r) = \int_0^\infty \frac{dk}{2\pi^2} k^m j_l(kr) P(k) \f$
            /// The usual 2-point correlation function xi(r) is just \f$ \xi_0^2(r) \f$
            /// The input k-values must be logarithmically spaced.
            /// The resulting \f$ \xi_l^m(r) \f$ will be evaluated at the dual r-values
            ///   \f$ r[i] \approx 1/k[N-1-i] \f$ (times a factor close to 1 chosen to reduce ringing; the returned
            ///   r-array is the right one)
            /// q is the FFTLog bias: the transform is done for \f$ k^{-q} \times \f$ the integrand. q = 0 is fine for
            /// \f$ \ell = 0 \f$; for higher \f$ \ell \f$ a negative q (e.g. q = -1 for \f$ \ell = 2,4 \f$, m = 2) can
            /// be much more accurate at small r.
            //==========================================================================
            std::pair<DVector, DVector>
            ComputeXiLM(int ell, int m, const DVector & k, const DVector & pk, double q = 0.0);

            //==========================================================================
            /// @brief Compute the discrete Hankel transform of the function a(r). See the FFTLog
            /// documentation for a description of exactly what this function computes.
            /// If u is NULL, the transform coefficients will be computed and discarded
            /// afterwards. For repeated transforms, the Workspace overload reuses plans,
            /// aligned buffers and coefficients. Supplied u bypasses the noring adjustment.
            //==========================================================================
            std::pair<DVector, CVector> DiscreteHankelTransform(const DVector & r,
                                                                const CVector & a,
                                                                double mu,
                                                                double q = 0,
                                                                double kcrc = 1,
                                                                bool noring = true,
                                                                CVector * u = nullptr);

            //==========================================================================
            /// @brief Pre-compute the coefficients that appear in the FFTLog implementation of
            /// the discrete Hankel transform. The parameters N, mu, and q here are the
            /// same as for the function DiscreteHankelTransform.  The parameter L is defined
            /// to be N times the logarithmic spacing of the input array, i.e.
            ///   \f$ L = N * log(r[N-1]/r[0])/(N-1) \f$
            //==========================================================================
            CVector ComputeCoefficients(int N, double mu, double q, double L, double kcrc);
        } // namespace FFTLog
    } // namespace SOLVERS
} // namespace FML

#endif
