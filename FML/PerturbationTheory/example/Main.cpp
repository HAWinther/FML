#include <FML/Global/Global.h>
#include <FML/PerturbationTheory/FastOneLoop.h>
#include <iomanip>
#include <iostream>
int main() {
    using namespace FML::PERTURBATION;
    Vector k(1024), p(1024);
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = std::exp(std::log(1e-5) + double(i) / (k.size() - 1) * std::log(1e7));
        p[i] = k[i] * std::exp(-k[i] * k[i]);
    }
    FastEdS workspace;
    auto result = workspace.evaluate(k, p, .1, 2.1);
    if (FML::ThisTask == 0) {
        std::cout
            << std::setprecision(10)
            << "# Gaussian-cutoff toy; EdS kernels; explicit finite support\n# max direct spot-check relative error="
            << result.max_reference_relative_error << "\n# k Plinear P22 P13 Ploop Ptotal\n";
        for (size_t i = 0; i < result.k.size(); i += 12)
            std::cout << result.k[i] << ' ' << result.linear[i] << ' ' << result.p22[i] << ' ' << result.p13[i] << ' '
                      << result.loop[i] << ' ' << result.total[i] << '\n';
    }
    // Real applications: supply P_linear(k,z) in consistent length/volume units,
    // extend the sampled support, vary spacing/padding and validate the requested range.
    // For kernels with scale-dependent growth use direct_one_loop with explicit callbacks.
}
