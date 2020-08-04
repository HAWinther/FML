#include "ComputePowerSpectrum.h"

namespace FML {

  namespace CORRELATIONFUNCTIONS {

    PowerSpectrumBinning::PowerSpectrumBinning() : PowerSpectrumBinning(128) {}

    PowerSpectrumBinning::PowerSpectrumBinning(const int n)
      : PowerSpectrumBinning(2.0 * M_PI, 2.0 * M_PI * n, n, LINEAR_SPACING) {}

    PowerSpectrumBinning::PowerSpectrumBinning(
        const double kmin, 
        const double kmax,                                       
        const int n, 
        const int bin_type)
      : 
        kmin(kmin), 
        kmax(kmax), 
        n(n), 
        bin_type(bin_type),
        k(std::vector<double>(n, 0.0)), 
        pofk(std::vector<double>(n, 0.0)),
        count(std::vector<double>(n, 0.0)), 
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

    void PowerSpectrumBinning::print_info() {
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

    void PowerSpectrumBinning::add_to_bin(
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

    void PowerSpectrumBinning::reset() {
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

    void PowerSpectrumBinning::normalize() {

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

    void PowerSpectrumBinning::scale(const double kscale, const double pofkscale) {
      for (int i = 0; i < n; i++) {
        k[i] *= kscale;
        kbin[i] *= kscale;
        pofk[i] *= pofkscale;
      }
    }

    // This method can return values out of bounds
    int PowerSpectrumBinning::get_bin_index(const double kvalue, const double kmin,
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

    double PowerSpectrumBinning::get_k_from_bin_index(const int index,
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

