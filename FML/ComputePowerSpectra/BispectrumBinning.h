#ifndef BISPECTRUMBINNING_HEADER
#define BISPECTRUMBINNING_HEADER
#include <FML/Global/Global.h>
#include <cmath>
#include <vector>

namespace FML {
    namespace CORRELATIONFUNCTIONS {

        /// For storing the results from the bispectrum
        /// Currently only linear spaced bins.
        template <int N>
        class BispectrumBinning {
          public:
            int n{0};
            double kmin{0.0};
            double kmax{0.0};

            // This the the bispectrum, the volume factor and the power-spectrum
            std::vector<double> B123;
            std::vector<double> N123;
            std::vector<double> pofk;

            // k is linear spaced from kmin to kmax
            // klow is lower bin edge, khigh is higher bin edge
            // kbin is center of bin and kmean is the mean of wavenumbers
            // added to the bin. k = kbin except for the first and last bin
            std::vector<double> k;
            std::vector<double> kbin;
            std::vector<double> klow;
            std::vector<double> khigh;
            std::vector<double> kmean;

            BispectrumBinning() = default;
            BispectrumBinning(int nbins);
            BispectrumBinning(double kmin, double kmax, int nbins);

            // k *= 1/boxsize bofk *= boxsize^(2 NDIM)
            void scale(const double boxsize);

            // Free up memory
            void free();

            // Zero the arrays
            void reset();

            // The spectrum for k1,k2,k3 with k1 = k[ i ] etc.
            double get_spectrum(int i, int j, int k);

            // The reduced bispectrum B123 / (P1P2 + cyc)
            double get_reduced_spectrum(int i, int j, int k);

            // Return N123
            double get_bincount(int i, int j, int k);

            // The (k1,k2,k3) corresponding to a given index in the B123 vector
            std::array<int, 3> get_coord_from_index(size_t index);
            size_t get_index_from_coord(const std::array<int, 3> & ik);

            // Symmetry: we only need to compute ik1 <= ik2 <= ...
            // This function tells algorithms which configurations to compute and
            // which to set by using symmetry from the configs we have computed
            bool compute_this_configuration(const std::array<int, 3> & ik);

            // This is just to make it easier to add binnings
            // of several spectra... just for testing
            int nbinnings = 0;
            void combine(BispectrumBinning & rhs);
        };

        template <int N>
        BispectrumBinning<N>::BispectrumBinning(int nbins) : BispectrumBinning(0.0, 2.0 * M_PI * (nbins - 1), nbins) {}

        template <int N>
        BispectrumBinning<N>::BispectrumBinning(double kmin, double kmax, int nbins) {
            n = nbins;
            B123.resize(n * n * n);
            N123.resize(n * n * n);
            pofk.resize(n, 0.0);
            k.resize(n, 0.0);
            kbin.resize(n, 0.0);
            kmean.resize(n, 0.0);
            klow.resize(n, 0.0);
            khigh.resize(n, 0.0);
            this->kmin = kmin;
            this->kmax = kmax;
            for (size_t i = 0; i < k.size(); i++)
                k[i] = kmin + (kmax - kmin) * i / double(n);

            for (int i = 0; i < n; i++) {
                if (i == 0) {
                    klow[i] = std::min(k[0], 0.0);
                    khigh[i] = k[0] + (k[1] - k[0]) / 2.0;
                } else if (i < n - 1) {
                    klow[i] = khigh[i - 1];
                    khigh[i] = k[i] + (k[i + 1] - k[i]) / 2.0;
                } else {
                    klow[i] = khigh[i - 1];
                    khigh[i] = k[nbins - 1];
                }
                kbin[i] = (klow[i] + khigh[i]) / 2.0;
            }
        }

        template <int N>
        void BispectrumBinning<N>::scale(const double boxsize) {
            for (size_t i = 0; i < k.size(); i++) {
                k[i] *= 1.0 / boxsize;
                kbin[i] *= 1.0 / boxsize;
                klow[i] *= 1.0 / boxsize;
                khigh[i] *= 1.0 / boxsize;
                kmean[i] *= 1.0 / boxsize;
            }
            double scale = std::pow(boxsize, 2 * N);

            // Bispectrum
            for (auto & b : B123)
                b *= scale;

            // In principle we should scale N123 by 1/boxsize^3
            // however this is only used internally and then in  
            // dimensionless units so we omit this here

            // Power-spectrum
            scale = std::pow(boxsize, N);
            for (auto & p : pofk)
                p *= scale;
        }

        template <int N>
        void BispectrumBinning<N>::free() {
            B123.clear();
            B123.shrink_to_fit();
            N123.clear();
            N123.shrink_to_fit();
        }

        template <int N>
        void BispectrumBinning<N>::reset() {
            std::fill(B123.begin(), B123.end(), 0.0);
            std::fill(N123.begin(), N123.end(), 0.0);
            std::fill(pofk.begin(), pofk.end(), 0.0);
        }

        template <int N>
        double BispectrumBinning<N>::get_spectrum(int i, int j, int k) {
            assert_mpi(i >= 0 and j >= 0 and k >= 0, "[BispectrumBinning::get_spectrum] i,j,k has to be >= 0\n");
            assert_mpi(i < n and j < n and k < n, "[BispectrumBinning::get_spectrum] i,j,k has to be < n\n");
            std::array<int, 3> ik{i, j, k};
            return B123[(ik[0] * n + ik[1]) * n + ik[2]];
        }

        template <int N>
        std::array<int, 3> BispectrumBinning<N>::get_coord_from_index(size_t index) {
            std::array<int, 3> ik;
            for (int ii = 3 - 1, npow = 1; ii >= 0; ii--, npow *= n) {
                ik[ii] = index / npow % n;
            }
            return ik;
        }

        template <int N>
        size_t BispectrumBinning<N>::get_index_from_coord(const std::array<int, 3> & ik) {
            size_t index = 0;
            for (int i = 0; i < 3; i++)
                index = index * n + ik[i];
            return index;
        }

        // Symmetry: we only need to compute ik1 <= ik2 <= ...
        template <int N>
        bool BispectrumBinning<N>::compute_this_configuration(const std::array<int, 3> & ik) {
            double ksum = 0.0;
            for (int ii = 1; ii < 3; ii++) {
                if (ik[ii - 1] > ik[ii])
                    return false;
                ksum += khigh[ik[ii - 1]];
            }

            // No valid 'triangles' if k1+k2+... < kN so just set too zero right away
            if (ksum < klow[ik[3 - 1]])
                return false;
            return true;
        }

        template <int N>
        double BispectrumBinning<N>::get_reduced_spectrum(int i, int j, int k) {
            double B = get_spectrum(i, j, k);
            double p1 = pofk[i];
            double p2 = pofk[j];
            double p3 = pofk[k];
            double pij = p1 * p2 + p2 * p3 + p3 * p1;
            return B / pij;
        }

        template <int N>
        double BispectrumBinning<N>::get_bincount(int i, int j, int k) {
            assert_mpi(i >= 0 and j >= 0 and k >= 0, "[BispectrumBinning::get_spectrum] i,j,k has to be >= 0\n");
            assert_mpi(i < n and j < n and k < n, "[BispectrumBinning::get_spectrum] i,j,k has to be < n\n");
            std::array<int, 3> ik{i, j, k};
            return N123[(ik[0] * n + ik[1]) * n + ik[2]];
        }

        template <int N>
        void BispectrumBinning<N>::combine(BispectrumBinning & rhs) {
            assert_mpi(n == rhs.n, "[BispectrumBinning::combine] Incompatible binnings\n");
            if (nbinnings == 0) {
                B123 = rhs.B123;
                N123 = rhs.N123;
                pofk = rhs.pofk;
                kmean = rhs.kmean;
            } else {
                for (size_t i = 0; i < B123.size(); i++)
                    B123[i] = (B123[i] * nbinnings + rhs.B123[i]) / (nbinnings + 1.0);
                for (size_t i = 0; i < N123.size(); i++)
                    N123[i] += rhs.N123[i];
                for (size_t i = 0; i < pofk.size(); i++)
                    pofk[i] = (pofk[i] * nbinnings + rhs.pofk[i]) / (nbinnings + 1.0);
            }
            nbinnings++;
        }
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif
