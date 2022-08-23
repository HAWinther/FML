#include <FML/Math/Math.h>
#include <cmath>
#include <iostream>

int main() {

    //==============================================
    // Python linspace
    //==============================================
    auto xarr = FML::MATH::linspace(0.0, 1.0, 11);
    for (auto x : xarr)
        std::cout << x << " ";
    std::cout << std::endl;

    //==============================================
    // Find a root of a function
    //==============================================
    std::function<double(double)> function = [](double x) -> double { return x * x - x - 1; };
    std::pair<double, double> range = {1.0, 2.0};
    auto root = FML::MATH::find_root_bisection(function, range);
    std::cout << "Root " << root << " = " << (1.0 + std::sqrt(5.0)) / 2.0 << "\n";

    //==============================================
    // Sherical bessel function
    //==============================================
    int ell = 0;
    double x = 1.0;
    std::cout << "Sph.Bessel " << FML::MATH::j_ell(ell, x) << " = " << sin(x) / x << "\n";
    
    //==============================================
    // Legendre polyomials
    //==============================================
    double mu = 1./3.;
    auto Pell = FML::MATH::legendre_ell_of_mu_vector(mu, 2);
    std::cout << "Legendre P0(" << mu << ") = " << Pell[0] << " = "<< 1.0 << " )\n";
    std::cout << "Legendre P1(" << mu << ") = " << Pell[1] << " = " << mu << " )\n";
    std::cout << "Legendre P2(" << mu << ") = " << Pell[2] << " = " << (3*mu*mu-1)/2. << " )\n";

    //==============================================
    // Airy function
    //==============================================
#ifdef USE_GSL
    std::cout << "Airy " << FML::MATH::Airy_Ai(0.0) << " = " << 1.0 / std::pow(3., 2. / 3.) / std::tgamma(2. / 3.)
              << "\n";
#endif

    //==============================================
    // Evaluate Continued fraction converging to pi
    // (b0 + a1/(b1 + a2 /( ... )))
    //==============================================
    std::function<double(int)> a = [](int i) -> double { return (2.0 * i - 1) * (2.0 * i - 1); };
    std::function<double(int)> b = [](int i) -> double {
        if (i == 0)
            return 3.0;
        return 6.0;
    };
    const double eps = 1e-6;
    const int maxsteps = 100;
    auto res = FML::MATH::GeneralizedLentzMethod(a, b, eps, maxsteps);
    std::cout << "Pi = " << res.first << " Converged? " << res.second << "\n";
}
