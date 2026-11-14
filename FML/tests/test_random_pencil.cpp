// Gaussian random fields on pencils equal those on the slabs: the mode-keyed (Philox) backend generated
// directly on the PencilGrid, and the NGenIC backend (generated on slabs and redistributed); 2D and 3D
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/RandomGenerator/RandomGenerator.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <string>

using FML::TESTS::check;
using FML::TESTS::num;
using namespace FML::TESTS::GRIDS;
namespace G = FML::RANDOM::GAUSSIAN;

namespace {
    template <int N>
    void compare(int p0, int p1, G::RandomFieldBackend backend, bool fixed) {
        const int n = FML::NTasks % 3 == 0 ? 12 : 16;
        auto power = [](double k) { return 1.0 / (1.0 + k * k); };
        G::RandomFieldOptions o;
        o.backend = backend;
        o.seed = 1234;
        FML::RANDOM::RandomGenerator rng1, rng2;
        rng1.set_seed(77);
        rng2.set_seed(77);
        FML::GRID::FFTWGrid<N> slab(n);
        G::generate_gaussian_random_field_fourier<N>(slab, &rng1, power, fixed, o);
        FML::GRID::PencilGrid<N> pen(n, 0, 0, FML::PencilDomain::uniform(p0, p1));
        G::generate_gaussian_random_field_fourier<N>(pen, &rng2, power, fixed, o);
        const double d = max_rel_diff(global_fourier(pen), global_fourier(slab));
        check(d == 0.0, std::to_string(N) + "D pencils " + std::to_string(p0) + "x" + std::to_string(p1) + ", " +
                            (backend == G::RandomFieldBackend::Philox ? "Philox" : "NGenIC") +
                            (fixed ? " fixed" : "") + ": the field of the slabs (max diff " + num(d, 2) + ")");
    }
} // namespace

int main() {
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        for (auto b : {G::RandomFieldBackend::Philox, G::RandomFieldBackend::NGenIC})
            compare<3>(p0, FML::NTasks / p0, b, b == G::RandomFieldBackend::Philox);
    }
    compare<2>(FML::NTasks, 1, G::RandomFieldBackend::Philox, false);
    return FML::TESTS::summary("random fields on pencils");
}
