#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Performance/Counters.h>
#include <iostream>
#include <thread>
using namespace FML::PERFORMANCE;
int main() {
    using Grid = FML::GRID::FFTWGrid<3>;
    const int mesh = FML::NTasks == 3 ? 12 : 8;
    Grid field(mesh, 1, 2);
    field.fill_fourier_grid({3, 1});
    field.set_grid_status_real(false);
    const Count bytes = Count(field.get_ntot_fourier_alloc()) * sizeof(FML::GRID::ComplexType);
    reset();
    Grid copied(field);
    copied = field;
    Grid & self = copied;
    copied = self;
    Grid moved(std::move(copied));
    auto empty = field.clone_empty();
    bool good = true;
    for (auto i : empty.get_fourier_range())
        good &= empty.get_fourier_from_index(i) == FML::GRID::ComplexType(0) &&
                moved.get_fourier_from_index(i) == field.get_fourier_from_index(i);
    auto s = snapshot();
    if (enabled) {
        Count halo = Count(field.get_ntot_fourier_alloc() - field.get_ntot_fourier()) * sizeof(FML::GRID::ComplexType);
        good &= s[Counter::GridAllocations] == 2 && s[Counter::GridCopiedBytes] == 2 * bytes + halo;
        good &= s[Counter::GridCopyOperations] == 3 && !s.overflow;
        reset();
        std::vector<std::thread> workers;
        for (int t = 0; t < 4; ++t)
            workers.emplace_back([] {
                for (int i = 0; i < 1000; ++i)
                    record(Counter::ForwardFFTs);
            });
        for (auto & t : workers)
            t.join();
        good &= snapshot()[Counter::ForwardFFTs] == 4000;
        reset();
        record(Counter::GridCopiedBytes, std::numeric_limits<Count>::max());
        record(Counter::GridCopiedBytes);
        good &= snapshot().overflow && snapshot()[Counter::GridCopiedBytes] == std::numeric_limits<Count>::max();
    } else
        for (auto value : s.values)
            good &= value == 0;
    reset();
    good &= !snapshot().overflow;
#ifdef USE_FFTW
    Grid transform(mesh);
    transform.fill_real_grid([](std::array<double, 3> &) { return 2.; });
    reset();
    transform.fftw_r2c();
    transform.fftw_c2r();
    if (enabled)
        good &= snapshot()[Counter::ForwardFFTs] == 1 && snapshot()[Counter::InverseFFTs] == 1;
    if (enabled)
        good &= snapshot()[Counter::FFTPlans] == 2;
    reset();
    transform.fftw_r2c();
    transform.fftw_c2r();
    if (enabled)
        good &= snapshot()[Counter::FFTPlans] == 0;
    for (auto i : transform.get_real_range())
        good &= std::abs(transform.get_real_from_index(i) - 2) < 1e-5;
#endif
    int bad = !good;
    FML::MaxOverTasks(&bad);
    if (!FML::ThisTask)
        std::cout << (bad ? "FAIL" : "PASS") << " counter/allocator/copy/move/clone/FFT checks (enabled=" << enabled
                  << ")\n";
    return bad;
}
