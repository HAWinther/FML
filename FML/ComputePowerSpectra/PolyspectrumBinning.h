#ifndef POLYSPECTRUMBINNING_HEADER
#define POLYSPECTRUMBINNING_HEADER
#include <FML/Global/Global.h>
#include <cmath>
#include <vector>

namespace FML {
    namespace CORRELATIONFUNCTIONS {

        // For storing the results from general polyspectrum
        // Currently only linear bins
        template <int N, int ORDER>
        class PolyspectrumBinning {
          public:
            int n{0};
            double kmin{0.0};
            double kmax{0.0};

            std::vector<double> P123;
            std::vector<double> N123;
            std::vector<double> pofk;
            std::vector<double> k;
            std::vector<double> kbin;

            PolyspectrumBinning() = default;
            PolyspectrumBinning(double _kmin, double _kmax, int nbins);
            PolyspectrumBinning(int nbins);

            // k *= 1/boxsize and P *= boxsize^((ORDER-1) NDIM)
            void scale(const double boxsize);

            // Clears the arrays
            void reset();

            // Frees up the arrays
            void free();

            // Get the polyspectra at (k[ik1], k[ik2], ...)
            double get_spectrum(std::vector<int> & ik);

            // Get the reduced polyspectrum: P / fac
            // where fac = P(k1) for order=2 and P(k1)P(k2) + cyc for ORDER=3,4
            double get_reduced_spectrum(std::vector<int> & ik);

            // Returns N123
            double get_bincount(std::vector<int> & ik);
        };

        template <int N, int ORDER>
        PolyspectrumBinning<N, ORDER>::PolyspectrumBinning(double _kmin, double _kmax, int nbins) {
            n = nbins;
            kmin = _kmin;
            kmax = _kmax;
            size_t ntot = FML::power(size_t(n), ORDER);
            P123.resize(ntot, 0.0);
            N123.resize(ntot, 0.0);
            pofk.resize(n, 0.0);
            k.resize(n, 0.0);
            kbin.resize(n, 0.0);
            for (int i = 0; i < n; i++)
                k[i] = kmin + (kmax - kmin) * i / double(n);
        }

        template <int N, int ORDER>
        PolyspectrumBinning<N, ORDER>::PolyspectrumBinning(int nbins)
            : PolyspectrumBinning(0.0, 2.0 * M_PI * (nbins - 1), nbins){};

        template <int N, int ORDER>
        void PolyspectrumBinning<N, ORDER>::scale(const double boxsize) {
            for (int i = 0; i < n; i++) {
                k[i] *= 1.0 / boxsize;
                kbin[i] *= 1.0 / boxsize;
            }
            double scale = std::pow(boxsize, N * (ORDER - 1));
            for (auto & p : P123)
                p *= scale;
        }

        template <int N, int ORDER>
        void PolyspectrumBinning<N, ORDER>::reset() {
            std::fill(P123.begin(), P123.end(), 0.0);
            std::fill(N123.begin(), N123.end(), 0.0);
            std::fill(kbin.begin(), kbin.end(), 0.0);
        }

        template <int N, int ORDER>
        void PolyspectrumBinning<N, ORDER>::free() {
            P123.clear();
            P123.shrink_to_fit();
            N123.clear();
            N123.shrink_to_fit();
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_spectrum(std::vector<int> & ik) {
            assert_mpi(ik.size() == ORDER, "[PolyspectrumBinning::get_spectrum] ik != ORDER has the wrong size\n");
            size_t index = 0;
            for (int i = 0; i < ORDER; i++)
                index = index * n + ik[i];
            return P123[index];
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_reduced_spectrum(std::vector<int> & ik) {
            double P = get_spectrum(ik);
            std::vector<double> pi(n);

            for (int i = 0; i < ORDER; i++)
                pi[i] = pofk[ik[i]];

            if (ORDER == 2) {
                // 1 * delta(i,j) for ORDER = 2
                return P / pofk[ik[0]];
            } else if (ORDER == 3 or ORDER == 4) {
                // P(k1,..) / (P1P2 + cyc) for ORDER = 3 or 4
                // So = 0 and = 1 if gaussian
                double pipj = pi[n - 1] * pi[0];
                for (int i = 0; i < ORDER - 1; i++)
                    pipj += pi[i] * pi[i + 1];
                return P / pipj;
            }

            // The general case
            // <d1d2...d2n> / (Pi1Pi2...Pin + cyc) ~ 1 if gaussian
            // This will never be used so we don't implement it
            assert_mpi(false, "[PolyspectrumBinning::get_reduced_spectrum] Not implemented\n");

            return 0.0;
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_bincount(std::vector<int> & ik) {
            assert_mpi(ik.size() == ORDER, "[PolyspectrumBinning::get_bincount] ik != ORDER has the wrong size\n");
            size_t index = 0;
            for (int i = 0; i < ORDER; i++)
                index = index * n + ik[i];
            return N123[index];
        }
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML

#endif
