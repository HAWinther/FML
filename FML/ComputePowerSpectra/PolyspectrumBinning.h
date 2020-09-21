#ifndef POLYSPECTRUMBINNING_HEADER
#define POLYSPECTRUMBINNING_HEADER
#include <FML/Global/Global.h>
#include <array>
#include <cmath>
#include <vector>

namespace FML {
    namespace CORRELATIONFUNCTIONS {

        /// Class for storing the results from general polyspectrum
        template <int N, int ORDER>
        class PolyspectrumBinning {
          public:
            enum BinningType { LINEAR_SPACING, LOG_SPACING };

            int n{0};
            int bin_type{LINEAR_SPACING};
            double kmin{0.0};
            double kmax{0.0};

            // The polyspectrum, the volume factor and the power-spectrum
            std::vector<double> P123;
            std::vector<double> N123;
            std::vector<double> pofk;

            // klow is lower bin edge, khigh is higher bin edge
            // kbin is center of bin and kmean is the mean of wavenumbers
            // added to the bin
            std::vector<double> kbin;
            std::vector<double> klow;
            std::vector<double> khigh;
            std::vector<double> kmean;

            // If N123 is already computed or not
            bool bincount_is_set{false};

            PolyspectrumBinning() = default;
            PolyspectrumBinning(double _kmin, double _kmax, int nbins, int bin_type = LINEAR_SPACING);
            PolyspectrumBinning(int nbins, int Nmesh, int bin_type = LINEAR_SPACING);

            // k *= 1/boxsize and P *= boxsize^((ORDER-1) NDIM)
            void scale(const double boxsize);

            // Clears the arrays
            void reset();

            // Frees up the arrays
            void free();

            // Get the polyspectra at (k[ik1], k[ik2], ...)
            double get_spectrum(const std::array<int, ORDER> & ik);
            double get_spectrum(int i, int j);
            double get_spectrum(int i, int j, int k);
            double get_spectrum(int i, int j, int k, int l);

            // The coord (k1,k2,...) corresponding to a given index in the P123 and N123 array
            // These methods determine how the data is stored and fetched
            std::array<int, ORDER> get_coord_from_index(size_t index);
            size_t get_index_from_coord(const std::array<int, ORDER> & ik);

            // Symmetry: we only need to compute ik1 <= ik2 <= ...
            // This function tells algorithms which configurations to compute and
            // which to set by using symmetry from the configs we have computed
            bool compute_this_configuration(const std::array<int, ORDER> & ik);

            // Get the reduced polyspectrum: P / fac
            // where fac = sqrt(P(k1)P(k2)) for order=2 and P(k1)P(k2) + cyc for ORDER=3,4
            double get_reduced_spectrum(const std::array<int, ORDER> & ik);
            double get_reduced_spectrum(int i, int j);
            double get_reduced_spectrum(int i, int j, int k);
            double get_reduced_spectrum(int i, int j, int k, int l);

            // Returns N123
            double get_bincount(const std::array<int, ORDER> & ik);
            double get_bincount(int i, int j);
            double get_bincount(int i, int j, int k);
            double get_bincount(int i, int j, int k, int l);

            // Copy over N123 (useful to save computations if we do many calculations with
            // the same setup)
            void set_bincount(std::vector<double> & N123_external);

            // This is just to make it easier to add binnings
            // of several spectra... just for testing
            int nbinnings = 0;
            void combine(PolyspectrumBinning & rhs);
        };

        template <int N, int ORDER>
        PolyspectrumBinning<N, ORDER>::PolyspectrumBinning(int Nmesh, int nbins, int bin_type)
            : PolyspectrumBinning(0.0, 2.0 * M_PI * Nmesh / 2.0, nbins, bin_type){};

        template <int N, int ORDER>
        PolyspectrumBinning<N, ORDER>::PolyspectrumBinning(double _kmin, double _kmax, int nbins, int bin_type) {
            n = nbins;
            kmin = _kmin;
            kmax = _kmax;
            size_t ntot = size_t(FML::power(n, ORDER));
            P123.resize(ntot, 0.0);
            N123.resize(ntot, 0.0);
            pofk.resize(n, 0.0);
            kbin.resize(n, 0.0);
            klow.resize(n, 0.0);
            khigh.resize(n, 0.0);
            kmean.resize(n, 0.0);

            // k[0] is leftmost bin edge and k[n-1] is rightmost bin edge and otherwise center of bin
            // Used to set kbin below
            std::vector<double> k(n);
            for (int i = 0; i < n; i++) {
                if (bin_type == LINEAR_SPACING) {
                    k[i] = kmin + (kmax - kmin) * i / double(n - 1);
                } else if (bin_type == LOG_SPACING) {
                    assert_mpi(kmin > 0.0, "[PolyspectrumBinning] For log spacing we cannot have kmin = 0.0");
                    k[i] = std::exp(std::log(kmin) + std::log(kmax / kmin) * i / double(n - 1));
                } else {
                    // Unknown binning type
                    assert_mpi(false, "[PolyspectrumBinning] Unknown bintype (not LINEAR_SPACING or LOG_SPACING)");
                }
            }

            // We require that klow = 0 in the smallest bin
            for (int i = 0; i < n; i++) {
                if (i == 0) {
                    klow[i] = std::min(k[0], 0.0);
                    khigh[i] = k[0] + (k[1] - k[0]) / 2.0;
                    if (bin_type == LOG_SPACING)
                        khigh[i] = std::exp(std::log(k[0]) + std::log(k[1] / k[0]) / 2.0);
                } else if (i < n - 1) {
                    klow[i] = khigh[i - 1];
                    khigh[i] = k[i] + (k[i + 1] - k[i]) / 2.0;
                    if (bin_type == LOG_SPACING)
                        khigh[i] = std::exp(std::log(k[i]) + std::log(k[i + 1] / k[i]) / 2.0);
                } else {
                    klow[i] = khigh[i - 1];
                    khigh[i] = k[nbins - 1];
                }
                kbin[i] = (klow[i] + khigh[i]) / 2.0;
            }
        }

        template <int N, int ORDER>
        std::array<int, ORDER> PolyspectrumBinning<N, ORDER>::get_coord_from_index(size_t index) {
            std::array<int, ORDER> ik;
            for (int ii = ORDER - 1, npow = 1; ii >= 0; ii--, npow *= n) {
                ik[ii] = index / npow % n;
            }
            return ik;
        }

        template <int N, int ORDER>
        size_t PolyspectrumBinning<N, ORDER>::get_index_from_coord(const std::array<int, ORDER> & ik) {
            size_t index = 0;
            for (int i = 0; i < ORDER; i++)
                index = index * n + ik[i];
            return index;
        }

        // Symmetry: we only need to compute ik1 <= ik2 <= ...
        template <int N, int ORDER>
        bool PolyspectrumBinning<N, ORDER>::compute_this_configuration(const std::array<int, ORDER> & ik) {
            double ksum = 0.0;
            for (int ii = 1; ii < ORDER; ii++) {
                if (ik[ii - 1] > ik[ii]) {
                    return false;
                }
                ksum += khigh[ik[ii - 1]];
            }

            // No valid 'triangles' if k1+k2+... < kN so can just set too zero right away
            // Saves a bit of time computing, but not significantly so its fine to remove this just to be sure
            if (ksum < klow[ik[ORDER - 1]] and ORDER > 2)
                return false;

            return true;
        }

        template <int N, int ORDER>
        void PolyspectrumBinning<N, ORDER>::scale(const double boxsize) {
            for (int i = 0; i < n; i++) {
                kbin[i] *= 1.0 / boxsize;
                klow[i] *= 1.0 / boxsize;
                khigh[i] *= 1.0 / boxsize;
                kmean[i] *= 1.0 / boxsize;
            }

            // Scale polyspectrum
            double scale = std::pow(boxsize, N * (ORDER - 1));
            for (auto & p : P123)
                p *= scale;
            
            // Scale power-spectrum
            scale = std::pow(boxsize, N);
            for (auto & p : pofk)
              p *= scale;

            // In principle we should scale N123 by 1/boxsize^ORDER
            // however this is only used internally and then in
            // dimensionless units so we omit this here
        }

        template <int N, int ORDER>
        void PolyspectrumBinning<N, ORDER>::reset() {
            std::fill(P123.begin(), P123.end(), 0.0);
            std::fill(N123.begin(), N123.end(), 0.0);
            std::fill(pofk.begin(), pofk.end(), 0.0);
            std::fill(kmean.begin(), kmean.end(), 0.0);
            nbinnings = 0;
        }

        template <int N, int ORDER>
        void PolyspectrumBinning<N, ORDER>::free() {
            // We only bother to free the things that take up memory
            P123.clear();
            P123.shrink_to_fit();
            N123.clear();
            N123.shrink_to_fit();
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_spectrum(int i, int j) {
            assert_mpi(ORDER == 2, "[PolyspectrumBinning::get_spectrum] This method can only be called for ORDER=2\n");
            std::array<int, 2> ik{i, j};
            size_t index = ik[0] * n + ik[1];
            return P123[index];
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_spectrum(int i, int j, int k) {
            assert_mpi(ORDER == 3, "[PolyspectrumBinning::get_spectrum] This method can only be called for ORDER=3\n");
            std::array<int, 3> ik{i, j, k};
            size_t index = (ik[0] * n + ik[1]) * n + ik[2];
            return P123[index];
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_spectrum(int i, int j, int k, int l) {
            assert_mpi(ORDER == 4, "[PolyspectrumBinning::get_spectrum] This method can only be called for ORDER=4\n");
            std::array<int, 4> ik{i, j, k, l};
            size_t index = ((ik[0] * n + ik[1]) * n + ik[2]) * n + ik[3];
            return P123[index];
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_spectrum(const std::array<int, ORDER> & ik) {
            assert_mpi(ik.size() == ORDER, "[PolyspectrumBinning::get_spectrum] ik != ORDER has the wrong size\n");
            return P123[get_index_from_coord(ik)];
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_reduced_spectrum(int i, int j) {
            assert_mpi(ORDER == 2,
                       "[PolyspectrumBinning::get_reduced_spectrum] This method can only be called for ORDER=2\n");
            std::array<int, 2> ik{i, j};
            return get_reduced_spectrum(ik);
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_reduced_spectrum(int i, int j, int k) {
            assert_mpi(ORDER == 3,
                       "[PolyspectrumBinning::get_reduced_spectrum] This method can only be called for ORDER=3\n");
            std::array<int, 3> ik{i, j, k};
            return get_reduced_spectrum(ik);
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_reduced_spectrum(int i, int j, int k, int l) {
            assert_mpi(ORDER == 4,
                       "[PolyspectrumBinning::get_reduced_spectrum] This method can only be called for ORDER=4\n");
            std::array<int, 4> ik{i, j, k, l};
            return get_reduced_spectrum(ik);
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_reduced_spectrum(const std::array<int, ORDER> & ik) {
            const double P = get_spectrum(ik);
            std::vector<double> pi(n);

            for (int i = 0; i < ORDER; i++)
                pi[i] = pofk[ik[i]];

            if (ORDER == 2) {
                // For gaussian delta(i,j) for ORDER = 2
                return P / std::sqrt(pofk[ik[0]] * pofk[ik[1]]);
            } else if (ORDER == 3 or ORDER == 4) {
                // P(k1,..) / (P1P2 + cyc) for ORDER = 3 or 4
                // So = 0 and = 1 if gaussian
                double pipj = pi[ORDER - 1] * pi[0];
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
        double PolyspectrumBinning<N, ORDER>::get_bincount(const std::array<int, ORDER> & ik) {
            assert_mpi(ik.size() == ORDER, "[PolyspectrumBinning::get_bincount] ik != ORDER has the wrong size\n");
            return N123[get_index_from_coord(ik)];
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_bincount(int i, int j) {
            assert_mpi(ORDER == 2, "[PolyspectrumBinning::get_bincount] This method can only be called for ORDER=2\n");
            std::array<int, 2> ik{i, j};
            return get_bincount(ik);
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_bincount(int i, int j, int k) {
            assert_mpi(ORDER == 3, "[PolyspectrumBinning::get_bincount] This method can only be called for ORDER=3\n");
            std::array<int, 3> ik{i, j, k};
            return get_bincount(ik);
        }

        template <int N, int ORDER>
        double PolyspectrumBinning<N, ORDER>::get_bincount(int i, int j, int k, int l) {
            assert_mpi(ORDER == 4, "[PolyspectrumBinning::get_bincount] This method can only be called for ORDER=4\n");
            std::array<int, 4> ik{i, j, k, l};
            return get_bincount(ik);
        }

        template <int N, int ORDER>
        void PolyspectrumBinning<N, ORDER>::combine(PolyspectrumBinning<N, ORDER> & rhs) {
            assert_mpi(n == rhs.n and bin_type == rhs.bin_type,
                       "[PolyspectrumBinning::combine] Incompatible binnings\n");
            if (nbinnings == 0) {
                P123 = rhs.P123;
                N123 = rhs.N123;
                pofk = rhs.pofk;
                kmean = rhs.kmean;
            } else {
                for (size_t i = 0; i < P123.size(); i++)
                    P123[i] = (P123[i] * nbinnings + rhs.P123[i]) / (nbinnings + 1.0);
                for (size_t i = 0; i < N123.size(); i++)
                    N123[i] += rhs.N123[i];
                for (size_t i = 0; i < pofk.size(); i++)
                    pofk[i] = (pofk[i] * nbinnings + rhs.pofk[i]) / (nbinnings + 1.0);
            }
            nbinnings++;
        }

        template <int N, int ORDER>
        void PolyspectrumBinning<N, ORDER>::set_bincount(std::vector<double> & N123_external) {
            assert_mpi(N123_external.size() == N123.size(), "[set_bincount] Incompatible size of N123 array");
            N123 = N123_external;
            bincount_is_set = true;
        }
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML

#endif
