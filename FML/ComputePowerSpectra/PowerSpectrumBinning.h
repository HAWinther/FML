#ifndef POWERSPECTRUMBINNING_HEADER
#define POWERSPECTRUMBINNING_HEADER
#include <vector>
#include <cmath>
#include <FML/Global/Global.h>

namespace FML {
  namespace CORRELATIONFUNCTIONS {

    // This class is thread-safe, but cannot be created or normalized inside a OMP parallel region
    // Summation over MPI tasks is done in normalize()
    template<int N> 
      class PowerSpectrumBinning {
        public:
          enum BinningType {
            LINEAR_SPACING,
            LOG_SPACING
          };

          int n;
          int bin_type;
          double kmin;
          double kmax;

          std::vector<double> k;
          std::vector<double> count; 
          std::vector<double> pofk; 
          std::vector<double> kbin; 

#ifdef USE_OMP
          // Temporary storage for OMP parallization
          std::vector< std::vector<double> > count_thread; 
          std::vector< std::vector<double> > pofk_thread; 
          std::vector< std::vector<double> > kbin_thread; 
#endif

          PowerSpectrumBinning();
          PowerSpectrumBinning(const int n);
          PowerSpectrumBinning(const double kmin, const double kmax, const int n, const int bin_type);

          // Add two binnings together
          PowerSpectrumBinning & operator +=(const PowerSpectrumBinning & rhs);

          // Reset everything. Call before starting to bim
          void reset();

          // Add a new point to a bin
          void add_to_bin(double kvalue, double power, double weight = 1.0);

          // Normalize (i.e. find mean in each bin) Do summation over MPI tasks
          void normalize();

          // Scale to physical units: k *= 1/Box and pofk *= Box^N
          void scale(const double boxsize);

          // For controlling the bins
          int get_bin_index(const double kvalue, const double kmin, const double kmax, const int n, const int bin_type);
          double get_k_from_bin_index(const int index, const double kmin, const double kmax, const int n, const int bin_type);

          // Print some info
          void print_info();

          // Combine with another binning (just for testing to make it easier to bin up over realisations)
          int nbinnings = 0;
          void combine(PowerSpectrumBinning &rhs);
      };
          
    template<int N> 
      void PowerSpectrumBinning<N>::combine(PowerSpectrumBinning &rhs){
        if(nbinnings == 0){
          count = rhs.count;
          pofk = rhs.pofk;
        } else {
          for(int i = 0; i < n; i++)
            count[i] += rhs.count[i];
          for(int i = 0; i < n; i++)
            pofk[i] = (pofk[i]*nbinnings + rhs.pofk[i])/(nbinnings+1.0);
        }
        nbinnings++;
      }

    template<int N> 
      PowerSpectrumBinning<N> & PowerSpectrumBinning<N>::operator +=(const PowerSpectrumBinning<N> & rhs){
        assert(k.size() == rhs.k.size());
        assert(count.size() == rhs.count.size());
        assert(pofk.size() == rhs.pofk.size());
        assert(kbin.size() == rhs.kbin.size());
        assert(bin_type == rhs.bin_type);
        assert(kmin == rhs.kmin);
        assert(kmax == rhs.kmax);
        for(int i = 0; i < n; i++){
          count[i] += rhs.count[i];
          pofk[i] += rhs.pofk[i];
          kbin[i] += rhs.kbin[i];
        }
        return *this;
      }

    template<int N> 
      PowerSpectrumBinning<N>::PowerSpectrumBinning() : n(0) {}

    template<int N> 
      PowerSpectrumBinning<N>::PowerSpectrumBinning(int nbins)
      : PowerSpectrumBinning(2.0 * M_PI, 2.0 * M_PI * nbins, nbins, LINEAR_SPACING) {}

    template<int N> 
      PowerSpectrumBinning<N>::PowerSpectrumBinning(
          const double kmin, 
          const double kmax,                                       
          const int n, 
          const int bin_type)
      : 
        n(n), 
        bin_type(bin_type),
        kmin(kmin), 
        kmax(kmax), 
        k(std::vector<double>(n, 0.0)), 
        count(std::vector<double>(n, 0.0)), 
        pofk(std::vector<double>(n, 0.0)),
        kbin(std::vector<double>(n, 0.0)) 
    {
      for (int i = 0; i < n; i++)
        k[i] = get_k_from_bin_index(i, kmin, kmax, n, bin_type);

#ifdef USE_OMP
      assert_mpi(omp_get_thread_num() == 0, 
          "[PowerSpectrumBinning] You cannot create a binning inside a parallel region\n");
      count_thread.resize(NThreads);
      pofk_thread.resize(NThreads);
      kbin_thread.resize(NThreads);
      for(int i = 0; i < NThreads; i++){
        count_thread[i] = std::vector<double>(n, 0.0);
        pofk_thread[i]  = std::vector<double>(n, 0.0);
        kbin_thread[i]  = std::vector<double>(n, 0.0);
      }
#endif
    }

    template<int N> 
      void PowerSpectrumBinning<N>::print_info() {
        if(ThisTask == 0){
          printf("\n==================================\n");
          printf("PowerSpectrumBinning Info:\n");
          printf("nbins:  %i\n", n);
          printf("kmin:   %10.5e\n", kmin);
          printf("kmax:   %10.5e\n", kmax);
          printf("BinType %i\n", bin_type);
          printf("Data: \n");
          for (int i = 0; i < n; i++)
            printf("%10.5e  %10.5e\n", k[i], pofk[i]);
          printf("==================================\n");
        }
      }

    template<int N> 
      void PowerSpectrumBinning<N>::add_to_bin(
          double kvalue, 
          double power,
          double weight) {

        // Do not include zero-mode
        if(kvalue == 0.0) return;
        int index = get_bin_index(kvalue, kmin, kmax, n, bin_type);
#ifdef USE_OMP
        const int myid = NThreads == 1 ? 0 : omp_get_thread_num();
        if (0 <= index && index < n) {
          count_thread[myid][index] += weight;
          pofk_thread[myid][index] += power * weight;
          kbin_thread[myid][index] += kvalue * weight;
        }
#else
        if (0 <= index && index < n) {
          count[index] += weight;
          pofk[index] += power * weight;
          kbin[index] += kvalue * weight;
        }
#endif
      }

    template<int N> 
      void PowerSpectrumBinning<N>::reset() {
        for (int i = 0; i < n; i++) {
          count[i] = pofk[i] = kbin[i] = 0.0;
        }
#ifdef USE_OMP
        for (int id = 0; id < NThreads; id++) {
          for (int i = 0; i < n; i++) {
            count_thread[id][i] = pofk_thread[id][i] = kbin_thread[id][i] = 0.0;
          }
        }
#endif
      }

    template<int N> 
      void PowerSpectrumBinning<N>::normalize() {

#ifdef USE_OMP
        assert_mpi(omp_get_thread_num() == 0, 
            "[PowerSpectrumBinning::normalize] This method can only be run by the main thread\n");
        for (int id = 0; id < NThreads; id++) {
          for (int i = 0; i < n; i++) {
            count[i] += count_thread[id][i];
            pofk[i]  += pofk_thread[id][i];
            kbin[i]  += kbin_thread[id][i];
          }
        }
#endif

#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, pofk.data(),  n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, count.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, kbin.data(),  n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

        for (int i = 0; i < n; i++) {
          if (count[i] > 0) {
            pofk[i] /= count[i];
            kbin[i] /= count[i];
          }
        }
      }

    template<int N> 
      void PowerSpectrumBinning<N>::scale(const double boxsize) {
        for (int i = 0; i < n; i++) {
          k[i] *= 1.0/boxsize;
          kbin[i] *= 1.0/boxsize;
          pofk[i] *= std::pow(boxsize,N);
        }
      }

    // This method can return values out of bounds
    template<int N> 
      int PowerSpectrumBinning<N>::get_bin_index(const double kvalue, const double kmin,
          const double kmax, const int n,
          const int bin_type) {
        int index = -1;
        if (bin_type == LINEAR_SPACING) {
          index = int((kvalue - kmin) / (kmax - kmin) * (n-1) + 0.5);
        } else if (bin_type == LOG_SPACING) {
          if(kvalue <= 0.0) return -1;
          index = int(std::log(kvalue / kmin) / std::log(kmax / kmin) * (n-1) + 0.5);
        } else {
          assert_mpi(false,
              "[PowerSpectrumBinning::get_bin_index] Unknown binning type\n");
        }
        return index;
      }

    template<int N> 
      double PowerSpectrumBinning<N>::get_k_from_bin_index(const int index,
          const double kmin,
          const double kmax,
          const int n,
          const int bin_type) {
        double kvalue = 0.0;
        if (bin_type == LINEAR_SPACING) {
          kvalue = kmin + (kmax - kmin) / double(n-1) * index;
        } else if (bin_type == LOG_SPACING) {
          kvalue = std::exp(std::log(kmin) + std::log(kmax / kmin) / double(n-1) * index);
        } else {
          assert_mpi(false,
              "[PowerSpectrumBinning::get_k_from_bin_index] Unknown binning type\n");
        }
        return kvalue;
      }
  }
}
#endif