#include <FML/Global/Global.h>
#include <FML/PerturbationTheory/OneLoop.h>
#include <iostream>
int main() {
    using namespace FML::PERTURBATION;
    Kernels kernels;
    kernels.name = "analytic constant kernels";
    kernels.f2 = [](double, double, double) { return 1.; };
    kernels.f3_collapsed = [](double, double, double) { return 0.; };
    Vector k{.2, .7, .95};
    auto result = direct_one_loop(
        k, [](double) { return 1.; }, 1e-4, 1, {}, kernels);
    bool ok = true;
    for (size_t i = 0; i < k.size(); ++i) {
        double exact = (4 + k[i]) * (2 - k[i]) * (2 - k[i]) / (48 * detail::pi * detail::pi);
        ok &= std::abs(result.p22[i] / exact - 1) < 1e-5 && result.p13[i] == 0;
    }
    int failed = !ok;
    FML::MaxOverTasks(&failed);
    if (!FML::ThisTask)
        std::cout
            << (failed ? "[FAIL] " : "[PASS] ")
            << "FFT-independent custom-kernel quadrature matches the analytic intersection volume of two unit balls\n";
    return failed;
}
