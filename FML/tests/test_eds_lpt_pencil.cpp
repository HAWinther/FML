// The arbitrary order (EdS) LPT computed natively on pencils (PencilGrid work grids, owner-routed mode
// copies, Hermitian planes exchanged between the owners of mirrored modes) gives the orders of the slabs
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/ArbitraryOrder.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <numeric>
#include <string>

using FML::TESTS::check;
using FML::TESTS::num;
using namespace FML::TESTS::GRIDS;
namespace L = FML::COSMOLOGY::LPT;

namespace {
    // A few modes (and their mirrors on the kz = 0 plane), set by integer wave-vector
    template <class Grid>
    void fixture(Grid & g) {
        const int n = g.get_nmesh();
        g.set_grid_status_real(false);
        for (FML::IndexIntType irow = 0; irow < g.get_local_nrows_fourier(); irow++)
            for (auto && index : g.get_fourier_range_of_rows(irow, irow + 1)) {
                auto k = g.get_fourier_integer_wavevector_from_index(index);
                for (auto & v : k)
                    v = ((v % n) + n) % n;
                ComplexType value(0);
                if (k == std::array<int, 3>{1, 0, 0})
                    value = {.12, .03};
                if (k == std::array<int, 3>{n - 1, 0, 0})
                    value = {.12, -.03};
                if (k == std::array<int, 3>{0, 1, 1})
                    value = {-.08, .04};
                if (k == std::array<int, 3>{1, n - 1, 2})
                    value = {.06, -.02};
                g.set_fourier_from_index(index, value);
            }
    }
} // namespace

template <class Grid>
double max_diff(const std::vector<L::EdSLPTVectorT<Grid>> & orders, const std::vector<std::vector<ComplexType>> & ref) {
    double d = 0.0;
    size_t k = 0;
    for (auto & order : orders)
        for (auto & g : order)
            d = std::max(d, max_rel_diff(global_fourier(g), ref[k++]));
    return d;
}
template <class Vector>
std::vector<std::vector<ComplexType>> modes_of(const std::vector<Vector> & orders) {
    std::vector<std::vector<ComplexType>> m;
    for (auto & order : orders)
        for (auto & g : order)
            m.push_back(global_fourier(g));
    return m;
}
// Add a (small or large) non-Hermitian part to the mode (2, 1, 0) without changing its mirror (n-2, n-1, 0)
template <class Grid>
void break_hermitian(Grid & g, double amount) {
    for (auto && index : g.get_fourier_range()) {
        auto k = g.get_fourier_integer_wavevector_from_index(index);
        if (k[0] == 2 and k[1] == 1 and k[2] == 0)
            g.set_fourier_from_index(index, g.get_fourier_from_index(index) + ComplexType(amount, 0));
        if (k[0] == -2 and k[1] == -1 and k[2] == 0)
            g.set_fourier_from_index(index, g.get_fourier_from_index(index) + ComplexType(0.01, 0));
    }
}

int main() {
    const int n = FML::NTasks % 3 == 0 ? 12 : 16;
    L::EdSLPTOptions o;
    o.max_order = 4;
    o.input_mode_limit.fill(2);
    o.max_owned_bytes_per_rank = size_t(512) * 1024 * 1024;
    const double tolerance = sizeof(FML::GRID::FloatType) == 4 ? 1e-4 : 1e-11;

    // The slab reference: the orders, the evaluation and a slightly non-Hermitian input (corrected)
    FML::GRID::FFTWGrid<3> slab(n);
    fixture(slab);
    const auto ref = L::compute_eds_lpt(slab, o);
    const auto ref_modes = modes_of(ref.orders_fourier);
    const auto ref_eval = ref.evaluate(1.3, size_t(512) * 1024 * 1024);
    std::vector<std::vector<ComplexType>> ref_eval_modes;
    for (auto * v : {&ref_eval.displacement_fourier, &ref_eval.ddisplacement_dln_a_fourier})
        for (auto & g : *v)
            ref_eval_modes.push_back(global_fourier(g));
    auto slab_perturbed = slab;
    break_hermitian(slab_perturbed, 0.01 + 1e-16);
    const auto ref_corrected = L::compute_eds_lpt(slab_perturbed, o);

    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        const int p1 = FML::NTasks / p0;
        if (FML::NTasks % p0 or n % p0 or n % p1)
            continue;
        const std::string label = "pencils " + std::to_string(p0) + "x" + std::to_string(p1) + ": ";
        FML::GRID::PencilGrid<3> pen(n, 0, 0, FML::PencilDomain::uniform(p0, p1));
        fixture(pen);

        auto result = L::compute_eds_lpt(pen, o);
        const double d = result.orders_fourier.size() == 4 ? max_diff(result.orders_fourier, ref_modes) : 1e100;
        check(d < tolerance, label + "EdS LPT orders 1-4 = slabs (max rel diff " + num(d, 2) + ")");
        const auto & r = result.report;
        check(r.fft_count == r.planned_fft_count and r.remap_count == r.planned_remap_count and
                  r.accounted_owned_peak_bytes <= r.owned_peak_bound_bytes and r.work_mesh % std::lcm(std::lcm(2, p0), p1) == 0,
              label + "transform/remap counts as planned, memory within the preflight bound (work mesh " +
                  std::to_string(r.work_mesh) + ")");
        // The work grids are pencils: an order has the layout of the input
        check(result.orders_fourier[0][0].get_domain().process_grid() == std::array<int, 2>{p0, p1} and
                  result.orders_fourier[0][0].get_local_range(1) == pen.get_local_range(1),
              label + "the orders are decomposed as the input");

        // Streaming gives the same orders
        std::vector<L::EdSLPTVectorT<FML::GRID::PencilGrid<3>>> streamed;
        L::for_each_eds_lpt_order(pen, o, [&](int, const L::EdSLPTVectorT<FML::GRID::PencilGrid<3>> & fields) {
            streamed.push_back(fields);
        });
        check(max_diff(streamed, modes_of(result.orders_fourier)) == 0.0, label + "streamed orders = retained");

        // The evaluation
        const auto eval = result.evaluate(1.3, size_t(512) * 1024 * 1024);
        double de = 0;
        size_t k = 0;
        for (auto * v : {&eval.displacement_fourier, &eval.ddisplacement_dln_a_fourier})
            for (auto & g : *v)
                de = std::max(de, max_rel_diff(global_fourier(g), ref_eval_modes[k++]));
        check(de < tolerance, label + "evaluation = slabs (max rel diff " + num(de, 2) + ")");

        // A slightly non-Hermitian kz = 0 plane is corrected as on slabs, a clearly non-Hermitian one is rejected
        auto perturbed = pen;
        break_hermitian(perturbed, 0.01 + 1e-16);
        const auto corrected = L::compute_eds_lpt(perturbed, o);
        check(corrected.report.max_hermitian_correction == ref_corrected.report.max_hermitian_correction and
                  max_diff(corrected.orders_fourier, modes_of(ref_corrected.orders_fourier)) < tolerance,
              label + "Hermitian correction as on slabs (" + num(corrected.report.max_hermitian_correction, 2) + ")");
        auto broken = pen;
        break_hermitian(broken, 0.5);
        check(FML::TESTS::throws([&] { L::compute_eds_lpt(broken, o); }), label + "non-Hermitian input rejected");
    }

    // Pencils of any mesh (here odd): the default decomposition with the edges on cells. Order 1 is the Zeldovich
    // displacement i k / k^2 delta of the (band-limited) input
    {
        const int nodd = 11;
        FML::GRID::PencilGrid<3> pen(nodd);
        fixture(pen);
        L::EdSLPTOptions oo = o;
        oo.max_order = 3;
        const auto result = L::compute_eds_lpt(pen, oo);
        double d = 0;
        for (int i = 0; i < 3; i++)
            for (auto && index : pen.get_fourier_range()) {
                std::array<double, 3> kvec;
                double k2;
                pen.get_fourier_wavevector_and_norm2_by_index(index, kvec, k2);
                const ComplexType expected = k2 > 0 ? ComplexType(0, kvec[i] / k2) * pen.get_fourier_from_index(index) : 0.0;
                d = std::max(d, double(std::abs(result.orders_fourier[0][i].get_fourier_from_index(index) - expected)));
            }
        FML::MaxOverTasks(&d);
        check(d < tolerance * 1e-3 and result.report.fft_count == result.report.planned_fft_count,
              "odd mesh " + std::to_string(nodd) + " on the default pencils: order 1 = Zeldovich (" + num(d, 2) + ")");
    }
    return FML::TESTS::summary("EdS LPT on pencils");
}
