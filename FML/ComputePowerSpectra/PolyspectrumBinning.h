#ifndef POLYSPECTRUMBINNING_HEADER
#define POLYSPECTRUMBINNING_HEADER

//=====================================================================
//=====================================================================

namespace FML {
  namespace CORRELATIONFUNCTIONS {

    // For storing the results from general polyspectrum
    // Currently only linear bins
    template<int N, int order>
      class PolyspectrumBinning {
        public:
          int n;
          double kmin;
          double kmax;

          std::vector<double> P123;
          std::vector<double> N123;
          std::vector<double> pofk;
          std::vector<double> k;
          std::vector<double> kbin;

          PolyspectrumBinning() : n(0) {}
          PolyspectrumBinning(double kmin, double kmax, int nbins){
            n = nbins;
            size_t ntot = FML::power(size_t(n),order);
            P123.resize(ntot);
            N123.resize(ntot);
            pofk.resize(n);
            k.resize(n);
            kbin.resize(n);
            this->kmin = kmin;
            this->kmax = kmax;
            for(int i = 0; i < n; i++)
              k[i] = kmin + (kmax - kmin) * i / double(n);
          }

          // kscale = 1/boxsize polyofkscale = boxsize^((order-1) NDIM)
          void scale(const double boxsize){
            for(int i = 0; i < n; i++){
              k[i] *= 1.0 / boxsize;
              kbin[i] *= 1.0 / boxsize;
            }
            double scale = std::pow(boxsize, N * (order-1));
            for(auto &p : P123)
              p *= scale;
          }

          void free(){
            P123.clear();
            P123.shrink_to_fit();
            N123.clear();
            N123.shrink_to_fit();
          }

          double get_spectrum(std::vector<int> &ik){
            assert(ik.size() == order);
            size_t index = 0;
            for(int i = 0; i < order; i++)
              index = index * n + ik[i];
            return P123[index];
          }
          
          double get_reduced_spectrum(std::vector<int> &ik){
            double P = get_spectrum(ik);
            std::vector<double> pi(n);
            
            for(int i = 0; i < order; i++)
              pi[i] = pofk[ik[i]];
           
            if(order == 2){
              // 1 * delta(i,j) for order = 2
              return P / pofk[ik[0]];
            } else if(order == 3 or order == 4){
              // P(k1,..) / (P1P2 + cyc) for order = 3 or 4
              // So = 0 and = 1 if gaussian
              double pipj = pi[n-1] * pi[0];
              for(int i = 0; i < order-1; i++)
                pipj += pi[i]*pi[i+1];
              return P / pipj;
            }

            // The general case
            // <d1d2...d2n> / (Pi1Pi2...Pin + cyc) ~ 1 if gaussian 
            // This will never be used so we don't implement it
            assert(false);
            return 0.0;
          }

          double get_bincount(std::vector<int> & ik){
            assert(ik.size() == order);
            size_t index = 0;
            for(int i = 0; i < order; i++)
              index = index * n + ik[i];
            return N123[index];
          }
      };
  }
}

#endif
