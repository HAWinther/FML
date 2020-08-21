#ifndef BISPECTRUMBINNING_HEADER
#define BISPECTRUMBINNING_HEADER
#include <vector>
#include <cmath>
#include <FML/Global/Global.h>

namespace FML {
  namespace CORRELATIONFUNCTIONS {

    // For storing the results from the bispectrum
    // Currently only linear bins
    template<int N> 
      class BispectrumBinning {
        public:
          int n{0};
          double kmin{0.0};
          double kmax{0.0};

          std::vector<double> B123;
          std::vector<double> N123;
          std::vector<double> pofk;
          std::vector<double> k;
          std::vector<double> kbin;

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

          // This is just to make it easier to add binnings
          // of several spectra... just for testing
          int nbinnings = 0;
          void combine(BispectrumBinning &rhs);
      };

    template<int N>
      BispectrumBinning<N>::BispectrumBinning(int nbins) : BispectrumBinning(0.0, 2.0*M_PI*(nbins-1), nbins) {}

    template<int N>
      BispectrumBinning<N>::BispectrumBinning(double kmin, double kmax, int nbins){
        n = nbins;
        B123.resize(n*n*n);
        N123.resize(n*n*n);
        pofk.resize(n);
        k.resize(n);
        kbin.resize(n);
        this->kmin = kmin;
        this->kmax = kmax;
        for(size_t i = 0; i < k.size(); i++)
          k[i] = kmin + (kmax - kmin) * i / double(n);
      }

    template<int N>
      void BispectrumBinning<N>::scale(const double boxsize){
        for(size_t i = 0; i < k.size(); i++){
          k[i] *= 1.0 / boxsize;
          kbin[i] *= 1.0 / boxsize;
        }
        double scale = std::pow(boxsize, 2*N);

        // Bispectrum
        for(auto &b : B123)
          b *= scale;

        // Power-spectrum
        scale = std::pow(boxsize, N);
        for(auto &p : pofk)
          p *= scale;
      }

    template<int N>
      void BispectrumBinning<N>::free(){
        B123.clear();
        B123.shrink_to_fit();
        N123.clear();
        N123.shrink_to_fit();
      }

    template<int N>
      void BispectrumBinning<N>::reset(){
        std::fill(B123.begin(), B123.end(), 0.0);
        std::fill(N123.begin(), N123.end(), 0.0);
        std::fill(pofk.begin(), pofk.end(), 0.0);
        std::fill(kbin.begin(), kbin.end(), 0.0);
      }

    template<int N>
      double BispectrumBinning<N>::get_spectrum(int i, int j, int k){
        assert_mpi(i >= 0 and j >= 0 and k >= 0,
            "[BispectrumBinning::get_spectrum] i,j,k has to be >= 0\n");
        assert_mpi(i < n and j < n and k < n,
            "[BispectrumBinning::get_spectrum] i,j,k has to be < n\n");
        return B123[(i*n + j)*n + k];
      }

    template<int N>
      double BispectrumBinning<N>::get_reduced_spectrum(int i, int j, int k){
        double B = get_spectrum(i,j,k);
        double p1 = pofk[i];
        double p2 = pofk[j];
        double p3 = pofk[k];
        double pij = p1*p2 + p2*p3 + p3*p1;
        return B/pij;
      }

    template<int N>
      double BispectrumBinning<N>::get_bincount(int i, int j, int k){
        assert_mpi(i >= 0 and j >= 0 and k >= 0,
            "[BispectrumBinning::get_spectrum] i,j,k has to be >= 0\n");
        assert_mpi(i < n and j < n and k < n,
            "[BispectrumBinning::get_spectrum] i,j,k has to be < n\n");
        return N123[(i*n + j)*n + k];
      }

    template<int N>
      void BispectrumBinning<N>::combine(BispectrumBinning &rhs){
        assert_mpi(n == rhs.n,
            "[BispectrumBinning::combine] Incompatible binnings\n");
        if(nbinnings == 0){
          B123 = rhs.B123;
          N123 = rhs.N123;
          pofk = rhs.pofk;
          kbin = rhs.kbin;
        } else {
          for(size_t i = 0; i < B123.size(); i++)
            B123[i] = (B123[i]*nbinnings + rhs.B123[i])/(nbinnings+1.0);
          for(size_t i = 0; i < N123.size(); i++)
            N123[i] += rhs.N123[i];
          for(size_t i = 0; i < pofk.size(); i++)
            pofk[i] = (pofk[i]*nbinnings + rhs.pofk[i])/(nbinnings+1.0);
        }
        nbinnings++;
      }
  }
}
#endif
