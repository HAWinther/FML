#include <FML/FFTLog/FFTLog.h>
#include <iostream>

using DVector = FML::SOLVERS::FFTLog::DVector;

DVector generate_pofk(DVector & k) {

    //=====================================================
    // Generate something that looks like the
    // matter power-spectrum with a something to give
    // a very sharp BAO peak as a test function to transform
    //=====================================================

    // Fitting formula for the transfer function
    auto bbks_fit = [&](double k) -> double {
        const double keq = 0.01;
        const double arg = k / keq;
        double T = log(1.0 + 0.171 * arg) / (0.171 * arg) *
                   pow(1 + 0.284 * arg + pow(1.18 * arg, 2) + pow(0.399 * arg, 3) + pow(0.490 * arg, 4), -0.25);
        T *= (1.0 + 0.01 * sin(k * 100.));
        return T;
    };

    // Some thing that looks like the power-spectrum
    auto pk_func = [&](double k) -> double {
        double T = bbks_fit(k);
        const double ns = 0.96;
        return 1e6 * k * T * T * pow(k / 0.05, ns - 1);
    };

    // Fill the vector with P(k)
    DVector pk(k);
    for (auto & x : pk)
        x = pk_func(x);

    return pk;
}

int main() {

    //=====================================================
    // Example on how to use FFTLog to compute
    // the correlation function from a power-spectrum
    // The other way around is completely analogous
    //=====================================================

    // Generate a k-array
    const double kmin = 1e-5;
    const double kmax = 1e3;
    const int npts = 4096;
    DVector k_array(npts);
    for (size_t i = 0; i < npts; i++)
        k_array[i] = exp(log(kmin) + (log(kmax / kmin) * i / double(npts)));

    // Generate a power-spectrum array
    DVector pk_array = generate_pofk(k_array);

    // Call FFTLog
    auto res = FML::SOLVERS::FFTLog::ComputeCorrelationFunction(k_array, pk_array);

    // Output correlation function
    auto r = res.first;
    auto xi = res.second;
    for (size_t i = 0; i < r.size(); i++)
        std::cout << r[i] << " " << xi[i] << "\n";
}
