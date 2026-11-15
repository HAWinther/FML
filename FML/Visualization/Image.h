#ifndef FML_VISUALIZATION_IMAGE_H
#define FML_VISUALIZATION_IMAGE_H

// =====================================================================================================================
// A 2D image of values (e.g. a projected density) and how to turn it into a picture:
//
//   FML::VISUALIZATION::Image image = ...;                        // e.g. from project_grid (Projection.h)
//   write_png("density.png", image, "magma", ColorScale::automatic_log());
//   write_png("force.png", force_image, "coolwarm", ColorScale::automatic_symmetric());
//   write_npy("density.npy", image);                              // the values, for numpy: np.load (shape ny, nx)
//
// image(ix, iy) is the value at column ix (left to right) and row iy (bottom to top): iy = 0 is the bottom row of the
// picture, as in a plot with the y-axis up. The writers are not collective: call them on one task.
// =====================================================================================================================

#include <FML/Visualization/Colormaps.h>
#include <FML/Visualization/PNG.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML {
    namespace VISUALIZATION {

        /// A 2D array of values: value (ix, iy) is data[iy * nx + ix], iy = 0 the bottom row
        struct Image {
            int nx{0};
            int ny{0};
            std::vector<double> data;

            Image() = default;
            Image(int _nx, int _ny, double value = 0.0) : nx(_nx), ny(_ny), data(size_t(_nx) * size_t(_ny), value) {}

            double & operator()(int ix, int iy) { return data[size_t(iy) * nx + ix]; }
            double operator()(int ix, int iy) const { return data[size_t(iy) * nx + ix]; }
            bool empty() const { return data.empty(); }

            /// Smallest, largest and mean of the finite values
            double min() const {
                double v = std::numeric_limits<double>::infinity();
                for (auto x : data)
                    if (std::isfinite(x))
                        v = std::min(v, x);
                return v;
            }
            double max() const {
                double v = -std::numeric_limits<double>::infinity();
                for (auto x : data)
                    if (std::isfinite(x))
                        v = std::max(v, x);
                return v;
            }
            double mean() const {
                double sum = 0.0;
                size_t n = 0;
                for (auto x : data)
                    if (std::isfinite(x)) {
                        sum += x;
                        n++;
                    }
                return n > 0 ? sum / double(n) : 0.0;
            }
        };

        /// How values map to [0, 1] for the colour map: linear or logarithmic between vmin and vmax (values outside are
        /// clamped; for log, values <= 0 get the colour of vmin), or limits from percentiles of the image
        struct ColorScale {
            bool log{false};
            bool automatic{true};
            bool symmetric{false};
            double vmin{0.0};
            double vmax{1.0};
            double low_percentile{1.0};
            double high_percentile{99.9};

            /// Fixed limits
            static ColorScale linear(double vmin, double vmax) { return {false, false, false, vmin, vmax, 0, 0}; }
            static ColorScale logarithmic(double vmin, double vmax) { return {true, false, false, vmin, vmax, 0, 0}; }
            /// Limits from percentiles of the (positive, for log) values. A density: automatic_log()
            static ColorScale automatic_linear(double low = 1.0, double high = 99.9) {
                return {false, true, false, 0, 0, low, high};
            }
            static ColorScale automatic_log(double low = 1.0, double high = 99.9) {
                return {true, true, false, 0, 0, low, high};
            }
            /// Linear in [-m, m] with m the percentile of |value| (signed fields, with a diverging map: coolwarm)
            static ColorScale automatic_symmetric(double high = 99.5) { return {false, true, true, 0, 0, 0, high}; }

            /// The limits for an image (vmin, vmax as given, or from the percentiles)
            std::pair<double, double> limits(const Image & image) const {
                if (not automatic)
                    return {vmin, vmax};
                std::vector<double> v;
                v.reserve(image.data.size());
                for (auto x : image.data)
                    if (std::isfinite(x) and (not log or x > 0.0))
                        v.push_back(symmetric ? std::fabs(x) : x);
                if (v.empty())
                    return {log ? 1.0 : 0.0, log ? 10.0 : 1.0};
                auto percentile = [&](double p) {
                    const size_t k = size_t(std::clamp(p, 0.0, 100.0) / 100.0 * double(v.size() - 1) + 0.5);
                    std::nth_element(v.begin(), v.begin() + k, v.end());
                    return v[k];
                };
                if (symmetric) {
                    const double m = percentile(high_percentile);
                    return {-m, m};
                }
                return {percentile(low_percentile), percentile(high_percentile)};
            }
        };

        /// The pixels of an image coloured with a colour map (colormap_names()): width * height * 3 bytes, rows from
        /// the top (as PNG wants)
        inline std::vector<uint8_t> to_rgb(const Image & image,
                                           const std::string & colormap_name = "magma",
                                           const ColorScale & scale = ColorScale::automatic_log()) {
            auto [lo, hi] = scale.limits(image);
            if (scale.log) {
                if (not(lo > 0.0) or not(hi > 0.0))
                    throw std::runtime_error("[FML::VISUALIZATION::to_rgb] A log scale needs positive limits");
                lo = std::log(lo);
                hi = std::log(hi);
            }
            const double range = hi > lo ? hi - lo : 1.0;
            colormap(colormap_name, 0.0); // check the name before the loop
            std::vector<uint8_t> rgb(size_t(image.nx) * size_t(image.ny) * 3);
            for (int iy = 0; iy < image.ny; iy++)
                for (int ix = 0; ix < image.nx; ix++) {
                    double v = image(ix, iy);
                    double t;
                    if (not std::isfinite(v))
                        t = 0.0;
                    else if (scale.log)
                        t = v > 0.0 ? (std::log(v) - lo) / range : 0.0;
                    else
                        t = (v - lo) / range;
                    const RGB c = colormap(colormap_name, t);
                    const size_t i = (size_t(image.ny - 1 - iy) * image.nx + ix) * 3;
                    rgb[i] = c.r;
                    rgb[i + 1] = c.g;
                    rgb[i + 2] = c.b;
                }
            return rgb;
        }

        /// Write an image as a PNG with a colour map and a colour scale. Not collective: call it on one task
        inline void write_png(const std::string & filename,
                              const Image & image,
                              const std::string & colormap_name = "magma",
                              const ColorScale & scale = ColorScale::automatic_log()) {
            write_png(filename, image.nx, image.ny, to_rgb(image, colormap_name, scale));
        }

        /// Write the values of an image as a numpy .npy file (float64, shape (ny, nx), row 0 = iy = 0, the bottom row:
        /// plt.imshow(np.load(f), origin="lower")). Not collective: call it on one task
        inline void write_npy(const std::string & filename, const Image & image) {
            std::ofstream fp(filename, std::ios::binary);
            if (not fp)
                throw std::runtime_error("[FML::VISUALIZATION::write_npy] Cannot open [" + filename + "]");
            std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + std::to_string(image.ny) +
                                 ", " + std::to_string(image.nx) + "), }";
            // Magic (6) + version (2) + header length (2) + header, padded with spaces to a multiple of 64 with '\n'
            while ((10 + header.size() + 1) % 64 != 0)
                header += ' ';
            header += '\n';
            const char magic[8] = {char(0x93), 'N', 'U', 'M', 'P', 'Y', 1, 0};
            fp.write(magic, 8);
            const uint16_t len = uint16_t(header.size());
            const char l[2] = {char(len & 0xFF), char(len >> 8)};
            fp.write(l, 2);
            fp.write(header.data(), std::streamsize(header.size()));
            for (double v : image.data) { // little endian
                uint64_t bits;
                std::memcpy(&bits, &v, 8);
                char b[8];
                for (int k = 0; k < 8; k++)
                    b[k] = char((bits >> (8 * k)) & 0xFF);
                fp.write(b, 8);
            }
            if (not fp)
                throw std::runtime_error("[FML::VISUALIZATION::write_npy] Failed to write [" + filename + "]");
        }

        /// The image at another size: the mean of the pixels that fall in a new pixel when shrinking, the nearest
        /// pixel when enlarging (each axis separately)
        inline Image resize(const Image & image, int nx, int ny) {
            if (nx <= 0 or ny <= 0 or image.empty())
                throw std::runtime_error("[FML::VISUALIZATION::resize] Empty image or size");
            Image out(nx, ny);
            for (int jy = 0; jy < ny; jy++) {
                int y0 = int((long long)jy * image.ny / ny), y1 = int((long long)(jy + 1) * image.ny / ny);
                if (y1 <= y0)
                    y1 = y0 + 1;
                for (int jx = 0; jx < nx; jx++) {
                    int x0 = int((long long)jx * image.nx / nx), x1 = int((long long)(jx + 1) * image.nx / nx);
                    if (x1 <= x0)
                        x1 = x0 + 1;
                    double sum = 0.0;
                    for (int iy = y0; iy < y1; iy++)
                        for (int ix = x0; ix < x1; ix++)
                            sum += image(ix, iy);
                    out(jx, jy) = sum / double((x1 - x0) * (y1 - y0));
                }
            }
            return out;
        }
    } // namespace VISUALIZATION
} // namespace FML
#endif
