#include <FML/Morphology/VoxelMorphology.h>
#include <iomanip>
#include <iostream>
// Minimal real-field protocol, also implemented by FFTWGrid<3>.
struct Field {
    int n = 32;
    int get_nmesh() const { return n; }
    int get_local_x_start() const { return n * FML::ThisTask / FML::NTasks; }
    int get_local_nx() const { return n * (FML::ThisTask + 1) / FML::NTasks - get_local_x_start(); }
    double get_real(std::array<int, 3> c) const {
        return std::cos(2 * M_PI * (c[0] + get_local_x_start() + .5) / n) + std::cos(2 * M_PI * (c[1] + .5) / n) +
               std::cos(2 * M_PI * (c[2] + .5) / n);
    }
};
int main() {
    auto results = FML::MORPHOLOGY::excursions(Field{}, {-1, 0, 1}, 100);
    if (!FML::ThisTask) {
        std::cout << std::setprecision(10)
                  << "# periodic closed voxel union; L=100; >= threshold\n# threshold volume surface_area "
                     "integrated_mean_curvature Euler\n";
        for (const auto & r : results)
            std::cout << r.threshold << ' ' << r.measures.volume << ' ' << r.measures.surface_area << ' '
                      << r.measures.integrated_mean_curvature << ' ' << r.measures.euler_characteristic << '\n';
    }
}
