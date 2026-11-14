// The sparse scatter of mesh-node contributions to their owners (FieldLevel DistributedAssignment) on a
// PencilGrid: the deposited field and the gathered values equal those on the slabs
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FieldLevel/DistributedAssignment.h>
#include <FML/Global/Global.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
using namespace FML::TESTS::GRIDS;
namespace D = FML::FIELDLEVEL::FRAMEWORK::DETAIL;

namespace {
    const int n = 12;
    // Contributions from every task to nodes all over the mesh (also colliding)
    auto requests = [](auto add) {
        for (int i = 0; i < 300; i++) {
            const std::uint64_t g = std::uint64_t(i) * 7 + std::uint64_t(FML::ThisTask) * 131;
            add((g * 37 + 11) % (n * n * n), 1 + std::sin(.1 * g));
            add((n * n * n - 1 - g) % (n * n * n), -.3);
        }
    };
    template <class G>
    std::pair<std::vector<double>, std::vector<double>> run(G & grid) {
        D::DistributedAssignmentT<G> a(grid);
        a.deposit(grid, requests);
        auto field = global_real(grid);
        // The reverse: the value of a known field at the nodes we asked for
        fill<3>(grid, [](auto x) { return std::cos(3 * x[0] + 5 * x[1] + 7 * x[2]); });
        const auto back = a.reverse(grid);
        // (in the order of the owners: look them up by node)
        std::vector<double> values;
        requests([&](std::uint64_t node, double) { values.push_back(a.at(back, node)); });
        return {field, values};
    }
} // namespace

int main() {
    FML::GRID::FFTWGrid<3> slab(n);
    const auto ref = run(slab);
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        const int p1 = FML::NTasks / p0;
        if (FML::NTasks % p0 or n % p0 or n % p1)
            continue;
        FML::GRID::PencilGrid<3> pen(n, 0, 0, FML::PencilDomain::uniform(p0, p1));
        const auto res = run(pen);
        double d1 = max_rel_diff(res.first, ref.first), d2 = max_rel_diff(res.second, ref.second);
        int same_size = res.second.size() == ref.second.size();
        FML::MinOverTasks(&same_size);
        check(d1 < 1e-14 and d2 == 0.0 and same_size, "pencils " + std::to_string(p0) + "x" + std::to_string(p1) +
                                                           ": deposited field and reverse values = slabs (max diff " +
                                                           num(std::max(d1, d2), 2) + ")");
    }
    return FML::TESTS::summary("distributed assignment on pencils");
}
