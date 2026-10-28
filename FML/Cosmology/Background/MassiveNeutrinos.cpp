#include <FML/Cosmology/Background/MassiveNeutrinos.h>
#include <FML/Cosmology/Common.h>

#include <gsl/gsl_integration.h>

#include <cmath>
#include <stdexcept>

namespace FML::COSMOLOGY {

    namespace MassiveNeutrinos {

        Quadrature::Quadrature(int n) {
            if (n < 1)
                throw std::invalid_argument("MassiveNeutrinos::Quadrature: need n >= 1");
            gsl_integration_fixed_workspace * ws =
                gsl_integration_fixed_alloc(gsl_integration_fixed_laguerre, n, 0.0, 1.0, 0.0, 0.0);
            const double * x = gsl_integration_fixed_nodes(ws);
            const double * wl = gsl_integration_fixed_weights(ws);
            q.assign(x, x + n);
            w.resize(n);
            // Laguerre weight e^{-q}: int F f0 dq = int e^{-q} [F e^q f0] dq, and e^q f0 = 1/(1 + e^{-q})
            for (int i = 0; i < n; i++)
                w[i] = wl[i] / (1.0 + std::exp(-x[i]));
            gsl_integration_fixed_free(ws);
        }

        double dlnf0_dlnq(double q) { return -q / (1.0 + std::exp(-q)); }

        namespace {
            const Quadrature & bg() {
                static const Quadrature quad(48);
                return quad;
            }
        } // namespace

        double I_rho(double M) {
            const auto & Q = bg();
            double s = 0.0;
            for (int i = 0; i < Q.size(); i++) {
                const double q = Q.q[i];
                s += Q.w[i] * q * q * std::sqrt(q * q + M * M);
            }
            return s;
        }

        double I_p(double M) {
            const auto & Q = bg();
            double s = 0.0;
            for (int i = 0; i < Q.size(); i++) {
                const double q = Q.q[i];
                s += Q.w[i] * q * q * q * q / (3.0 * std::sqrt(q * q + M * M));
            }
            return s;
        }

        double dI_p_dlnM(double M) {
            const auto & Q = bg();
            double s = 0.0;
            for (int i = 0; i < Q.size(); i++) {
                const double q = Q.q[i], e = std::sqrt(q * q + M * M);
                s -= Q.w[i] * q * q * q * q * M * M / (3.0 * e * e * e);
            }
            return s;
        }

        double I_pp(double M) {
            const auto & Q = bg();
            double s = 0.0;
            for (int i = 0; i < Q.size(); i++) {
                const double q = Q.q[i], e = std::sqrt(q * q + M * M);
                s += Q.w[i] * q * q * q * q * q * q / (3.0 * e * e * e);
            }
            return s;
        }

        double M0(double m_eV, double T_nu0) { return m_eV * Constants::eV / (Constants::k_b * T_nu0); }
    } // namespace MassiveNeutrinos

} // namespace FML::COSMOLOGY
