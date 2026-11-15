// Tests of the Visualization module: grid projections (slabs and pencils, every axis, periodic slabs) against direct
// sums, particle projections (NGP/CIC weights, mass, the slab), colour maps, and files that check_png.py reads back
// with Python's zlib (PNG) and parses (npy).
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <FML/Visualization/Visualization.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace FML::VISUALIZATION;
using FML::TESTS::check;

namespace {
    constexpr int nmesh = 12;
    double field(int ix, int iy, int iz) { return 1.0 + ix + 0.5 * iy * iy + 0.01 * iz * iz * iz + 0.1 * ix * iz; }

    // The projection computed directly from the formula
    Image reference(int axis, double start, double thickness, bool average) {
        Image image(nmesh, nmesh);
        int ncells = 0;
        for (int i = 0; i < nmesh; i++) {
            double d = double(i) / nmesh - start; // the cells are at i / nmesh
            d -= std::floor(d);
            if (thickness >= 1.0 or d < thickness) {
                ncells++;
                for (int a = 0; a < nmesh; a++)
                    for (int b = 0; b < nmesh; b++) {
                        int c[3];
                        c[axis] = i;
                        c[axis == 0 ? 1 : 0] = a;
                        c[axis == 2 ? 1 : 2] = b;
                        image(a, b) += field(c[0], c[1], c[2]);
                    }
            }
        }
        if (average)
            for (auto & v : image.data)
                v /= ncells;
        return image;
    }

    double max_diff(const Image & a, const Image & b) {
        if (a.nx != b.nx or a.ny != b.ny)
            return 1e100;
        double d = 0.0;
        for (size_t i = 0; i < a.data.size(); i++)
            d = std::max(d, std::fabs(a.data[i] - b.data[i]) / (1.0 + std::fabs(b.data[i])));
        return d;
    }

    template <template <int> class Grid>
    void test_grid(Grid<3> & grid, const std::string & name) {
        for (auto && index : grid.get_real_range()) {
            auto coord = grid.get_coord_from_index(index);
            for (int d = 0; d < 3; d++)
                coord[d] += int(grid.get_local_range(d)[0]);
            grid.set_real_from_index(index, field(coord[0], coord[1], coord[2]));
        }
        struct Case {
            double start, thickness;
            bool average;
        };
        for (int axis = 0; axis < 3; axis++)
            for (auto c : {Case{0.0, 1.0, true}, Case{0.0, 1.0, false}, Case{0.3, 0.25, true}, Case{0.9, 0.3, false}}) {
                const auto image = project_grid(grid, axis, c.start, c.thickness, c.average);
                const double d = max_diff(image, reference(axis, c.start, c.thickness, c.average));
                check(d < 1e-6, name + ": axis " + std::to_string(axis) + ", slab [" + std::to_string(c.start) + ", +" +
                                    std::to_string(c.thickness) + ")" + (c.average ? " mean" : " sum") +
                                    " = direct (diff " + std::to_string(d) + ")");
            }
        bool threw = false;
        try {
            project_grid(grid, 2, 0.51 / nmesh, 0.2 / nmesh);
        } catch (const std::exception &) {
            threw = true;
        }
        check(threw, name + ": a slab with no cell centre throws");
    }

    struct Particle {
        double pos[3];
        double mass{1.0};
        double * get_pos() { return pos; }
        double get_mass() const { return mass; }
        constexpr int get_ndim() const { return 3; }
    };
    struct Particle2D {
        double pos[2];
        double * get_pos() { return pos; }
        constexpr int get_ndim() const { return 2; }
    };
} // namespace

int main() {
    // ---------------- Grids ----------------
    {
        FML::GRID::FFTWGrid<3> grid(nmesh);
        test_grid(grid, "FFTWGrid");
    }
#ifdef USE_MPI
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        const int p1 = FML::NTasks / p0;
        if (FML::NTasks % p0 or nmesh % p0 or nmesh % p1)
            continue;
        FML::GRID::PencilGrid<3> grid(nmesh, 0, 0, FML::PencilDomain::uniform(p0, p1));
        test_grid(grid, "PencilGrid " + std::to_string(p0) + "x" + std::to_string(p1));
    }
#endif
    {
        FML::GRID::FFTWGrid<2> grid(nmesh);
        for (auto && index : grid.get_real_range()) {
            auto coord = grid.get_coord_from_index(index);
            grid.set_real_from_index(index, 1.0 + coord[0] + grid.get_local_range(0)[0] + 100.0 * coord[1]);
        }
        const auto image = project_grid(grid);
        double d = 0.0;
        for (int ix = 0; ix < nmesh; ix++)
            for (int iy = 0; iy < nmesh; iy++)
                d = std::max(d, std::fabs(image(ix, iy) - (1.0 + ix + 100.0 * iy)));
        check(d < 1e-4, "FFTWGrid<2>: the image is the grid");
    }

    // ---------------- Particles ----------------
    {
        const int npix = 8;
        // One particle per task, at known places: task t puts it at the centre of pixel (t, 2t) (NGP and CIC: all of
        // it in that pixel), a second one at the corner between pixels (CIC: a quarter in each of four, wrapping
        // round the edge for the last task), and one outside the slab
        std::vector<Particle> part(3);
        const int t = FML::ThisTask % npix;
        part[0] = {{(t + 0.5) / npix, ((2 * t) % npix + 0.5) / npix, 0.55}, 2.0};
        part[1] = {{0.0, 4.0 / npix, 0.52}, 1.0};
        part[2] = {{0.3, 0.3, 0.1}, 5.0};
        const auto ngp = project_particles(part.data(), part.size(), npix, 2, 0.5, 0.1, "NGP");
        const auto cic = project_particles(part.data(), part.size(), npix, 2, 0.5, 0.1, "CIC");
        // Total mass in the slab: 3 per task; the mean pixel value = 3 NTasks / npix^2
        const double mean = 3.0 * FML::NTasks / (npix * npix);
        Image expect_ngp(npix, npix), expect_cic(npix, npix);
        for (int task = 0; task < FML::NTasks; task++) {
            const int s = task % npix;
            expect_ngp(s, (2 * s) % npix) += 2.0 / mean;
            expect_cic(s, (2 * s) % npix) += 2.0 / mean;
            expect_ngp(0, 4) += 1.0 / mean;
            for (int a : {npix - 1, 0})
                for (int b : {3, 4})
                    expect_cic(a, b) += 0.25 / mean;
        }
        check(max_diff(ngp, expect_ngp) < 1e-12, "particles NGP: mass in the right pixels, slab respected, mean 1");
        check(max_diff(cic, expect_cic) < 1e-12, "particles CIC: weights (centre, corner, periodic wrap)");

        // Random particles in the whole box: the CIC image sums to npix^2 (mass conserved) on any number of tasks
        std::mt19937 gen(1234 + FML::ThisTask);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<Particle> random(1000);
        for (auto & p : random)
            p = {{u(gen), u(gen), u(gen)}, 0.5 + u(gen)};
        for (int axis = 0; axis < 3; axis++) {
            const auto image = project_particles(random.data(), random.size(), 32, axis);
            double sum = 0.0;
            for (auto v : image.data)
                sum += v;
            check(std::fabs(sum / (32.0 * 32.0) - 1.0) < 1e-12 and image.min() >= 0.0,
                  "particles CIC, axis " + std::to_string(axis) + ": mass conserved, positive");
        }
        std::vector<Particle2D> p2(1);
        p2[0] = {{0.5 / 4, 2.5 / 4}};
        const auto image2 = project_particles(p2.data(), p2.size(), 4, 0, 0.0, 1.0, "NGP");
        check(std::fabs(image2(0, 2) - 16.0) < 1e-12, "2D particles: the plane itself");
    }

    // ---------------- Colour maps and scales ----------------
    {
        bool ok = true;
        for (const auto & name : colormap_names()) {
            const RGB a = colormap(name, 0.0), b = colormap(name + "_r", 1.0);
            ok = ok and a.r == b.r and a.g == b.g and a.b == b.b;
        }
        const RGB v0 = colormap("viridis", 0.0), v1 = colormap("viridis", 1.0), g = colormap("grey", 0.5);
        ok = ok and v0.r == 68 and v0.g == 1 and v0.b == 84 and v1.r == 253 and v1.g == 231 and v1.b == 37;
        ok = ok and g.r == g.g and g.g == g.b and std::abs(int(g.r) - 128) <= 1;
        bool threw = false;
        try {
            colormap("no_such_map", 0.5);
        } catch (const std::exception &) {
            threw = true;
        }
        check(ok and threw, "colour maps: viridis endpoints, grey, reversed maps, unknown name throws");

        Image image(4, 1);
        image.data = {1.0, 10.0, 100.0, 1000.0};
        const auto rgb = to_rgb(image, "grey", ColorScale::logarithmic(1.0, 1000.0));
        check(rgb[0] == 0 and rgb[9] == 255 and std::abs(int(rgb[3]) - 85) <= 1 and std::abs(int(rgb[6]) - 170) <= 1,
              "log scale: decades evenly spaced");
        const auto [lo, hi] = ColorScale::automatic_symmetric(100.0).limits(Image(2, 1, -3.0));
        check(lo == -3.0 and hi == 3.0, "symmetric scale: [-max|v|, max|v|]");
        const auto small = resize(image, 2, 1);
        check(small(0, 0) == 5.5 and small(1, 0) == 550.0, "resize: the mean of the merged pixels");
    }

    // ---------------- Files for check_png.py ----------------
    if (FML::ThisTask == 0) {
        // A test pattern (smooth with sharp edges and noise) and its raw bytes, for an exact comparison
        const int w = 301, h = 157;
        std::vector<uint8_t> rgb(size_t(w) * h * 3);
        std::mt19937 gen(7);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int c = 0; c < 3; c++)
                    rgb[(size_t(y) * w + x) * 3 + c] =
                        uint8_t(x > 200 ? gen() : (x + 2 * y + 60 * c + (x / 16 + y / 16) % 2 * 90) & 255);
        write_png("test_pattern.png", w, h, rgb);
        std::ofstream("test_pattern.raw", std::ios::binary)
            .write(reinterpret_cast<const char *>(rgb.data()), std::streamsize(rgb.size()));
        std::vector<uint8_t> grey(size_t(w) * h);
        for (size_t i = 0; i < grey.size(); i++)
            grey[i] = uint8_t((i * 7) % 251);
        write_png_grey("test_grey.png", w, h, grey);
        std::ofstream("test_grey.raw", std::ios::binary)
            .write(reinterpret_cast<const char *>(grey.data()), std::streamsize(grey.size()));

        // An image with a known orientation: bright at the top left (ix = 0, iy = ny-1), and its .npy values
        Image image(40, 20);
        for (int iy = 0; iy < image.ny; iy++)
            for (int ix = 0; ix < image.nx; ix++)
                image(ix, iy) = 1.0 + ix + 1000.0 * (ix == 0 and iy == image.ny - 1);
        write_png("test_orientation.png", image, "grey", ColorScale::linear(1.0, 41.0));
        write_npy("test_orientation.npy", image);
    }
    return FML::TESTS::summary("visualization");
}
