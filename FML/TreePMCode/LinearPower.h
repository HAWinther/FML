#ifndef FML_TREEPMCODE_LINEARPOWER_HEADER
#define FML_TREEPMCODE_LINEARPOWER_HEADER

#include <FML/Global/Global.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace FML::NBODY::TREEPM {

    //==========================================================================================
    /// Linear matter power spectrum at z = 0 (k in h/Mpc, P in (Mpc/h)^3): a CAMB-format file (k, P), or the
    /// Eisenstein & Hu (1998) no-wiggle P(k) normalized to sigma8
    //==========================================================================================
    class LinearPower {
        std::vector<double> logk, logP;
        double OmegaM, OmegaB, h, ns, A{1.0};
        bool from_file{false};

        // Eisenstein & Hu (1998) no-wiggle transfer function, k in h/Mpc
        double T_EH(double k) const {
            const double om = OmegaM * h * h, ob = OmegaB * h * h, fb = OmegaB / OmegaM, theta = 2.7255 / 2.7;
            const double s = 44.5 * std::log(9.83 / om) / std::sqrt(1.0 + 10.0 * std::pow(ob, 0.75)); // Mpc
            const double alpha = 1.0 - 0.328 * std::log(431.0 * om) * fb + 0.38 * std::log(22.3 * om) * fb * fb;
            const double kMpc = k * h;
            const double Gamma = OmegaM * h * (alpha + (1.0 - alpha) / (1.0 + std::pow(0.43 * kMpc * s, 4)));
            const double q = k * theta * theta / Gamma;
            const double L0 = std::log(2.0 * M_E + 1.8 * q), C0 = 14.2 + 731.0 / (1.0 + 62.5 * q);
            return L0 / (L0 + C0 * q * q);
        }
        double sigma_R(double R) const {
            // sigma^2 = int dlogk k^3 P(k) / (2 pi^2) W(kR)^2
            double sum = 0.0;
            const int n = 4000;
            const double lk1 = std::log(1e-5), lk2 = std::log(1e3), dlk = (lk2 - lk1) / n;
            for (int i = 0; i <= n; i++) {
                const double k = std::exp(lk1 + i * dlk), x = k * R;
                const double W = x < 1e-4 ? 1.0 : 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
                sum += (i == 0 or i == n ? 0.5 : 1.0) * k * k * k * (*this)(k) * W * W;
            }
            return std::sqrt(sum * dlk / (2.0 * M_PI * M_PI));
        }

      public:
        LinearPower(const std::string & filename, double OmegaM_, double OmegaB_, double h_, double ns_, double sigma8)
            : OmegaM(OmegaM_), OmegaB(OmegaB_), h(h_), ns(ns_) {
            if (not filename.empty()) {
                std::ifstream fp(filename);
                assert_mpi(fp.good(), ("Cannot open power spectrum file " + filename).c_str());
                std::string line;
                while (std::getline(fp, line)) {
                    if (line.empty() or line[0] == '#')
                        continue;
                    std::stringstream ss(line);
                    double k, P;
                    if (ss >> k >> P and k > 0 and P > 0) {
                        logk.push_back(std::log(k));
                        logP.push_back(std::log(P));
                    }
                }
                assert_mpi(logk.size() > 2, "Too few points in the power spectrum file");
                from_file = true;
            }
            if (sigma8 > 0.0)
                A *= std::pow(sigma8 / sigma_R(8.0), 2);
        }
        double operator()(double k) const {
            if (k <= 0.0)
                return 0.0;
            if (not from_file) {
                const double T = T_EH(k);
                return A * std::pow(k, ns) * T * T;
            }
            // Power-law extrapolation outside the table
            const double lk = std::log(k);
            size_t i = std::upper_bound(logk.begin(), logk.end(), lk) - logk.begin();
            i = std::min(std::max(i, size_t(1)), logk.size() - 1);
            const double w = (lk - logk[i - 1]) / (logk[i] - logk[i - 1]);
            return A * std::exp(logP[i - 1] + w * (logP[i] - logP[i - 1]));
        }
    };

} // namespace FML::NBODY::TREEPM
#endif
