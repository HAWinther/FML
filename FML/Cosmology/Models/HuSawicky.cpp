#include <FML/Cosmology/Models/HuSawicky.h>
#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    using Constants::c;
    using Constants::Mpc;

    HuSawickyGravity::HuSawickyGravity(double fR0_in, double n_in) : MuSigmaGravity(0.0, 0.0), fR0(fR0_in), n(n_in) {
        if (!(fR0 >= 0.0))
            throw std::invalid_argument("Hu-Sawicky: need fR0 = |f_R0| >= 0");
        if (!(n > 0.0))
            throw std::invalid_argument("Hu-Sawicky: need n > 0");
    }

    double HuSawickyGravity::R(double x, const BackgroundCosmology & bg) const {
        double t = 0.0; // rho - 3p (radiation drops out)
        for (const auto & s : bg.species())
            t += s->rho(x) - 3.0 * s->p(x);
        const double H0 = bg.get_H0();
        return 3.0 * H0 * H0 * t / (c * c);
    }

    double HuSawickyGravity::fR(double x, const BackgroundCosmology & bg) const {
        return -fR0 * std::pow(R(0.0, bg) / R(x, bg), n + 1.0);
    }

    TensorCoefficients HuSawickyGravity::tensor(double x, const BackgroundCosmology & bg) const {
        // d f_R/dx = -(n + 1) f_R dlnR/dx; dlnR/dx by a central difference
        const double h = 1e-4;
        const double dlnR = (std::log(R(x + h, bg)) - std::log(R(x - h, bg))) / (2.0 * h);
        const double f = fR(x, bg);
        return {1.0 + f, -(n + 1.0) * f * dlnR / (1.0 + f), 0.0};
    }

    double HuSawickyGravity::fRR(double x, const BackgroundCosmology & bg) const {
        return -(n + 1.0) * fR(x, bg) / R(x, bg);
    }

    double HuSawickyGravity::compton_wavelength(double x, const BackgroundCosmology & bg) const {
        return std::sqrt(3.0 * fRR(x, bg) / (1.0 + fR(x, bg)));
    }

    void
    HuSawickyGravity::values(double x, double k, const BackgroundCosmology & bg, double & mu, double & Sigma) const {
        const double Rx = R(x, bg), f_R = -fR0 * std::pow(R(0.0, bg) / Rx, n + 1.0), f_RR = -(n + 1.0) * f_R / Rx;
        const double ka = k * std::exp(-x);
        const double Q = f_RR * ka * ka / (1.0 + f_R); // = (k/a)^2 lambda_C^2 / 3
        Sigma = 1.0 / (1.0 + f_R);
        mu = (1.0 + 4.0 * Q) / (1.0 + 3.0 * Q) * Sigma;
    }

    // d(2 Sigma - mu)/dx at fixed k by central differences (all terms are smooth functions of x)
    HuSawickyGravity::MuSigma HuSawickyGravity::mu_sigma(double x, double k, const BackgroundCosmology & bg) const {
        MuSigma m;
        values(x, k, bg, m.mu, m.Sigma);
        const double h = 1e-4;
        double mp, sp, mm, sm;
        values(x + h, k, bg, mp, sp);
        values(x - h, k, bg, mm, sm);
        m.dg = ((2.0 * sp - mp) - (2.0 * sm - mm)) / (2.0 * h);
        return m;
    }

    void HuSawickyGravity::init_background(const BackgroundCosmology & bg) {
        MuSigmaGravity::init_background(bg);
        lambdaC0_Mpc = compton_wavelength(0.0, bg) / Mpc;
    }

    std::string HuSawickyGravity::info() const {
        std::ostringstream os;
        os << "Hu-Sawicky f(R) (|f_R0| = " << fR0 << ", n = " << n << ", Compton wavelength today " << lambdaC0_Mpc
           << " Mpc; LCDM background, quasi-static)";
        return os.str();
    }

} // namespace FML::COSMOLOGY
