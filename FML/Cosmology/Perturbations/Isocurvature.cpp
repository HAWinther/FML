#include <FML/Cosmology/Perturbations/Isocurvature.h>

#include <cmath>
#include <stdexcept>

namespace FML::COSMOLOGY {

    namespace Isocurvature {
        const char * const mode_names[4] = {"cdi", "bi", "nid", "niv"};

        bool active(const CosmologyParameters & p) {
            for (double f : p.iso_f)
                if (f != 0.0)
                    return true;
            return false;
        }

        std::vector<PerturbationOptions> components(const CosmologyParameters & p, const PerturbationOptions & base) {
            std::vector<int> on; // active modes
            for (int m = 0; m < 4; m++)
                if (p.iso_f[m] != 0.0)
                    on.push_back(m);
            const size_t n = on.size() + 1;
            // correlation matrix (adiabatic first) and its Cholesky factor
            std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0)), L(n, std::vector<double>(n, 0.0));
            for (size_t i = 0; i < n; i++)
                C[i][i] = 1.0;
            for (size_t i = 1; i < n; i++)
                C[0][i] = C[i][0] = p.iso_c[on[i - 1]];
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j <= i; j++) {
                    double s = C[i][j];
                    for (size_t k = 0; k < j; k++)
                        s -= L[i][k] * L[j][k];
                    if (i == j) {
                        if (!(s > 0.0))
                            throw std::invalid_argument(
                                "isocurvature: the correlations c_ad_* must satisfy sum c^2 < 1");
                        L[i][i] = std::sqrt(s);
                    } else {
                        L[i][j] = s / L[j][j];
                    }
                }
            std::vector<PerturbationOptions> out;
            for (size_t j = 0; j < n; j++) {
                PerturbationOptions o = base;
                o.ic_adiabatic = L[0][j];
                o.iso_amplitude = {{0.0, 0.0, 0.0, 0.0}};
                o.iso_tilt = {{0.0, 0.0, 0.0, 0.0}};
                for (size_t i = 1; i < n; i++) {
                    const int m = on[i - 1];
                    o.iso_amplitude[m] = L[i][j] * p.iso_f[m];
                    o.iso_tilt[m] = 0.5 * (p.iso_n[m] - p.n_s);
                }
                out.push_back(o);
            }
            return out;
        }
    } // namespace Isocurvature

} // namespace FML::COSMOLOGY
