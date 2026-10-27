// Standalone regression tests: shared helpers must not depend on cosmology or Global.cpp.
#include <FML/FileUtils/DataPath.h>
#include <FML/Math/NumericalMethods.h>
#include <FML/Timing/Timer.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char * name) {
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
        failures += !ok;
    };
    auto close = [](double a, double b) { return std::fabs(a - b) < 1e-12; };
    auto throws = [](const auto & fn) {
        try { fn(); } catch (const std::invalid_argument &) { return true; }
        return false;
    };
    namespace M = FML::MATH;
    check(M::linspace(2, 4, 0).empty() && M::linspace(2, 4, 1) == std::vector<double>{2} &&
          M::linspace(4, 2, 3) == std::vector<double>({4, 3, 2}), "existing linspace conventions");
    check(M::logspace(-2, 2, 3) == std::vector<double>({0.01, 1, 100}),
          "existing logspace takes exponents");
    const auto geo = M::geomspace(0.01, 100, 3);
    check(geo.front() == 0.01 && geo.back() == 100 && close(geo[1], 1) &&
          close(M::geomspace(100, 0.01, 3)[1], 1), "geomspace takes physical endpoints");
    check(throws([] { M::geomspace(0, 1, 3); }) && throws([] { M::geomspace(1, 2, 1); }),
          "geomspace rejects invalid limits and counts");
    check(close(M::trapz({0, 1, 3}, {0, 1, 3}), 4.5) && M::trapz({}, {}) == 0 &&
          throws([] { M::trapz({0}, {}); }), "trapezoidal quadrature and mismatched arrays");
    std::vector<double> js(301);
    M::spherical_bessel_j_array(300, 2.0, js.data());
    check(close(js[0], std::sin(2.0) / 2.0) &&
          close(js[1], std::sin(2.0) / 4.0 - std::cos(2.0) / 2.0) && std::isfinite(js[300]),
          "Miller recurrence at small x and large order");
    M::spherical_bessel_j_array(300, 500.0, js.data());
    check(close(js[0], std::sin(500.0) / 500.0), "upward recurrence at large x");
#ifdef USE_GSL
    check(close(M::integrate([](double x) { return x * x; }, 0, 1), 1.0 / 3.0),
          "shared adaptive quadrature");
    check(close(M::find_root([](double x) { return x * x - 2; }, 2, 0), std::sqrt(2.0)),
          "shared Brent solver with reversed endpoints");
    const double j = M::spherical_bessel_j(1, 2.0);
    check(close(j, std::sin(2.0) / 4.0 - std::cos(2.0) / 2.0) &&
          M::spherical_bessel_j(1, -2.0) == -j && M::spherical_bessel_j(0, 0) == 1 &&
          M::spherical_bessel_j(1000, 0.1) == 0, "shared scalar Bessel parity and underflow");
#endif
    // Save/restore the environment so this test also works with a user's FML_DIR setting.
    const char * previous = std::getenv("FML_DIR");
    const bool had_previous = previous != nullptr;
    const std::string saved = previous ? previous : "";
    setenv("FML_DIR", "/test/fml", 1);
    check(FML::FILEUTILS::fml_file("data/table") == "/test/fml/data/table", "environment data path");
    setenv("FML_DIR", "", 1);
#ifdef FML_ROOT_DIR
    check(FML::FILEUTILS::fml_file("data/table") == std::string(FML_ROOT_DIR) + "/data/table",
          "compiled data path fallback");
#else
    check(FML::FILEUTILS::fml_file("data/table") == "data/table", "relative data path fallback");
#endif
    if (had_previous) setenv("FML_DIR", saved.c_str(), 1); else unsetenv("FML_DIR");
    FML::UTILS::Timer timer("standalone");
    std::ostringstream output;
    auto * original = std::cout.rdbuf(output.rdbuf());
    timer.print(false);
    const bool silent = output.str().empty();
    timer.print();
    std::cout.rdbuf(original);
    check(silent && output.str().find("[Timer standalone]") == 0 && timer.elapsed() >= 0,
          "standalone stopwatch and explicit output control");
    return failures ? 1 : 0;
}
