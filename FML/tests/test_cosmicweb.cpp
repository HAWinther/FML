// Analytic CosmicWeb tensors and thread-count consistency. FFT/MPI calls remain
// outside OpenMP regions; the same tests run without OpenMP and in float builds.
#include <FML/CosmicWeb/CosmicWeb.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace CW = FML::COSMICWEB;
using Grid = FML::GRID::FFTWGrid<3>;
using Float = FML::GRID::FloatType;
using Tensor = std::array<Grid, 6>;
constexpr int mesh = 16;
constexpr int ii[6] = {0, 0, 0, 1, 1, 2};
constexpr int jj[6] = {0, 1, 2, 1, 2, 2};
const double tolerance = 128 * std::numeric_limits<Float>::epsilon();
int failures = 0;

void check(bool ok, const std::string & label) {
    int bad = !ok;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (FML::ThisTask == 0)
        std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
}
void set_threads(int n) {
#ifdef USE_OMP
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}
std::vector<int> thread_counts() {
#ifdef USE_OMP
    return {1, 2, 4};
#else
    return {1};
#endif
}
bool near(double a, double b, double tol) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tol;
}
bool same_grid(const Grid & a, const Grid & b, double tol) {
    if (a.get_nmesh() != b.get_nmesh())
        return false;
    for (auto index : a.get_real_range())
        if (!near(a.get_real_from_index(index), b.get_real_from_index(index), tol))
            return false;
    return true;
}
bool same_result(const CW::CosmicWeb & a, const CW::CosmicWeb & b, double tol) {
    if (a.get_nmesh() != b.get_nmesh() || a.get_local_nx() != b.get_local_nx() ||
        a.get_local_x_start() != b.get_local_x_start())
        return false;
    for (int ix = 0; ix < a.get_local_nx(); ++ix)
        for (int iy = 0; iy < a.get_nmesh(); ++iy)
            for (int iz = 0; iz < a.get_nmesh(); ++iz)
                if (a.get_classification({ix, iy, iz}) != b.get_classification({ix, iy, iz}))
                    return false;
    for (int k = 0; k < 3; ++k) {
        if (a.eigenvalues[k].get_nmesh() != b.eigenvalues[k].get_nmesh())
            return false;
        if (a.eigenvalues[k].get_nmesh() && !same_grid(a.eigenvalues[k], b.eigenvalues[k], tol))
            return false;
    }
    for (int k = 0; k < 4; ++k)
        if (!near(a.volume_fraction[k], b.volume_fraction[k], tol) ||
            !near(a.mass_fraction[k], b.mass_fraction[k], tol))
            return false;
    return true;
}

bool compact_layout(const CW::CosmicWeb & result) {
    const size_t cells = size_t(result.get_local_nx()) * result.get_nmesh() * result.get_nmesh();
    return result.has_byte_classification() && result.classification.get_nmesh() == 0 &&
           result.classification.get_ntot_fourier_alloc() == 0 && result.classification_bytes.size_bytes() == cells &&
           result.classification_bytes.get_data().size() == cells;
}

void test_storage_lifecycle(const Tensor & tensor, const Grid & density, double threshold,
                            const CW::CosmicWeb & reference) {
    CW::CosmicWeb result = reference;
    CW::classify_tensor(tensor, threshold, result, CW::ClassificationStorage::Bytes, true, &density);
    check(compact_layout(result) && same_result(result, reference, 0),
          "switching grid storage to bytes releases the FFT buffer and preserves every result");
    auto converted = result.classification_as_grid();
    check(same_grid(converted, reference.classification, 0) && compact_layout(result),
          "explicit byte-to-FFTW conversion is exact and leaves byte storage compact");
    auto copied = result.classification_bytes;
    auto moved = std::move(copied);
    check(copied.get_nmesh() == 0 && copied.get_local_nx() == 0 && copied.get_data().empty() &&
              same_grid(moved.to_fftw_grid(), reference.classification, 0),
          "byte-grid copies and moves preserve data and reset moved-from layout metadata");
    bool bounds = true;
    for (const auto coord : {std::array<int, 3>{-1, 0, 0}, {0, -1, 0}, {0, 0, mesh},
                             {int(moved.get_local_nx()), 0, 0}}) {
        try {
            (void)moved.get_real(coord);
            bounds = false;
        } catch (const std::out_of_range &) {}
    }
    check(bounds, "byte-grid coordinate access rejects all slab boundaries outside the stored domain");
    size_t flat = 0;
    bool dense = true;
    for (int ix = 0; ix < result.get_local_nx(); ++ix)
        for (int iy = 0; iy < mesh; ++iy)
            for (int iz = 0; iz < mesh; ++iz)
                dense = dense && result.classification_bytes.get_data()[flat++] == result.get_classification({ix, iy, iz});
    check(dense && flat == result.classification_bytes.size_bytes(),
          "byte labels are tightly packed in local x/y/z order with no FFT row padding");
    // The original function-pointer signature remains available as well as old call syntax.
    void (*legacy)(const Tensor &, double, CW::CosmicWeb &, bool, const Grid *) = &CW::classify_tensor;
    legacy(tensor, threshold, result, true, &density);
    check(!result.has_byte_classification() && result.classification_bytes.get_data().capacity() == 0 &&
              same_result(result, reference, 0),
          "switching bytes back to the legacy grid releases byte storage and preserves the original API");
    CW::classify_tensor(tensor, threshold, result, CW::ClassificationStorage::Bytes, false, &density);
    bool no_eigenvalues = true;
    for (const auto & grid : result.eigenvalues)
        no_eigenvalues = no_eigenvalues && grid.get_nmesh() == 0;
    check(compact_layout(result) && no_eigenvalues,
          "byte mode without requested eigenvalues allocates neither FFT labels nor eigenvalue grids");
    CW::ByteClassificationGrid empty;
    check(empty.get_data().empty() && empty.to_fftw_grid().get_nmesh() == 0,
          "default byte grids remain empty when explicitly converted");
}

// Frozen scalar classification/counting order from before the parallelization.
CW::CosmicWeb serial_reference(const Tensor & tensor, double threshold, const Grid * density = nullptr) {
    CW::CosmicWeb result;
    result.classification = Grid(mesh);
    for (auto & grid : result.eigenvalues)
        grid = Grid(mesh);
    double volume[4]{}, mass[4]{};
    for (auto index : tensor[0].get_real_range()) {
        double e[3];
        CW::symmetric_eigenvalues(tensor[0].get_real_from_index(index), tensor[1].get_real_from_index(index),
                                 tensor[2].get_real_from_index(index), tensor[3].get_real_from_index(index),
                                 tensor[4].get_real_from_index(index), tensor[5].get_real_from_index(index), e);
        int type = 0;
        for (int d = 0; d < 3; ++d) {
            type += e[d] > threshold;
            result.eigenvalues[d].set_real_from_index(index, Float(e[d]));
        }
        result.classification.set_real_from_index(index, Float(type));
        volume[type] += 1;
        if (density)
            mass[type] += density->get_real_from_index(index);
    }
    FML::SumArrayOverTasks(volume, 4);
    FML::SumArrayOverTasks(mass, 4);
    const double total_volume = volume[0] + volume[1] + volume[2] + volume[3];
    const double total_mass = mass[0] + mass[1] + mass[2] + mass[3];
    for (int k = 0; k < 4; ++k) {
        result.volume_fraction[k] = volume[k] / total_volume;
        result.mass_fraction[k] = total_mass > 0 ? mass[k] / total_mass : 0;
    }
    return result;
}

void test_classification_and_vweb() {
    Tensor tensor, shear;
    Grid density(mesh), divergence(mesh);
    for (int c = 0; c < 6; ++c) {
        tensor[c] = Grid(mesh);
        shear[c] = Grid(mesh);
    }
    constexpr double threshold = 0.125, normalization = -0.5;
    const double rotation[3][3] = {{0.6, -0.64, 0.48}, {0.8, 0.48, -0.36}, {0, 0.6, 0.8}};
    for (auto index : density.get_real_range()) {
        auto coord = density.get_coord_from_index(index);
        coord[0] += int(density.get_local_x_start());
        const int type = (coord[0] + coord[1] + coord[2]) % 4;
        double eigenvalues[3];
        for (int d = 0; d < 3; ++d)
            eigenvalues[d] = threshold + (d < type ? (3 - d) * 0.25 : -(d + 1) * 0.25);
        for (int c = 0; c < 6; ++c) {
            double value = 0;
            for (int d = 0; d < 3; ++d)
                value += rotation[ii[c]][d] * rotation[jj[c]][d] * eigenvalues[d];
            tensor[c].set_real_from_index(index, Float(value));
            shear[c].set_real_from_index(index, Float(value / normalization - (ii[c] == jj[c] ? 0.25 : 0)));
        }
        density.set_real_from_index(index, Float(0.125 + 0.01 * (coord[0] + 3 * coord[1] + 5 * coord[2]) + 0.001 * type));
        divergence.set_real_from_index(index, Float(0.75));
    }
    const auto reference = serial_reference(tensor, threshold, &density);
    bool analytic = true;
    for (auto index : density.get_real_range()) {
        auto coord = density.get_coord_from_index(index);
        coord[0] += int(density.get_local_x_start());
        const int type = (coord[0] + coord[1] + coord[2]) % 4;
        analytic = analytic && reference.classification.get_real_from_index(index) == type;
        for (int d = 0; d < 3; ++d) {
            const double expected = threshold + (d < type ? (3 - d) * 0.25 : -(d + 1) * 0.25);
            analytic = analytic && near(reference.eigenvalues[d].get_real_from_index(index), expected, tolerance);
        }
    }
    check(analytic, "rotated tensors recover independent analytic eigenvalues and all four web classes");
    CW::CosmicWeb first, first_vweb;
    for (int threads : thread_counts()) {
        set_threads(threads);
        CW::CosmicWeb actual, vweb;
        CW::classify_tensor(tensor, threshold, actual, true, &density);
        check(same_result(actual, reference, 1e-12),
              "classification and weighted fractions match the previous scalar calculation, threads=" + std::to_string(threads));
        if (threads == 1)
            first = actual;
        check(same_result(actual, first, 0), "classification, eigenvalues and fractions are thread-count independent");
        CW::CosmicWeb compact;
        CW::classify_tensor(tensor, threshold, compact, CW::ClassificationStorage::Bytes, true, &density);
        check(compact_layout(compact) && same_result(compact, actual, 0),
              "byte classification matches grid labels, eigenvalues and fractions exactly");
        void (*legacy_vweb)(const Tensor &, const Grid &, double, double, CW::CosmicWeb &, bool) = &CW::VWeb;
        legacy_vweb(shear, divergence, normalization, threshold, vweb, true);
        bool vweb_ok = same_grid(vweb.classification, reference.classification, 0);
        for (int d = 0; d < 3; ++d)
            vweb_ok = vweb_ok && same_grid(vweb.eigenvalues[d], reference.eigenvalues[d], tolerance);
        for (int k = 0; k < 4; ++k)
            vweb_ok = vweb_ok && vweb.volume_fraction[k] == 0.25 && vweb.mass_fraction[k] == 0;
        check(vweb_ok, "V-web shear/divergence assembly agrees with the analytic tensor");
        if (threads == 1)
            first_vweb = vweb;
        check(same_result(vweb, first_vweb, 0), "V-web outputs are thread-count independent");
        CW::VWeb(shear, divergence, normalization, threshold, compact, CW::ClassificationStorage::Bytes, true);
        check(compact_layout(compact) && same_result(compact, vweb, 0), "V-web supports identical compact output");
    }
    test_storage_lifecycle(tensor, density, threshold, first);
    for (auto & grid : tensor)
        grid.fill_real_grid(0);
    CW::classify_tensor(tensor, 0, first, false);
    bool ties = first.volume_fraction[CW::VOID] == 1 && first.mass_fraction[CW::VOID] == 0;
    for (auto & grid : first.eigenvalues)
        ties = ties && grid.get_nmesh() == 0;
    check(ties, "threshold ties use strict comparison and reused results release unrequested eigenvalues");
    CW::classify_tensor(tensor, -0.001, first);
    check(first.volume_fraction[CW::KNOT] == 1, "a negative threshold classifies a zero tensor as knots");
}

struct Wave {
    std::array<int, 3> mode;
    double amplitude, phase;
};
const std::array<Wave, 3> waves{{{{1, 2, 0}, 0.2, 0}, {{0, 1, 1}, 0.15, 0.7}, {{1, -1, 1}, 0.05, -0.4}}};
double wave_value(const Wave & wave, const std::array<double, 3> & x) {
    double phase = wave.phase;
    for (int d = 0; d < 3; ++d)
        phase += 2 * M_PI * wave.mode[d] * x[d];
    return wave.amplitude * std::cos(phase);
}

void test_tidal_tensor_and_tweb() {
    Grid delta(mesh), density(mesh);
    delta.fill_real_grid([](std::array<double, 3> & x) {
        double value = 0.1; // The tidal tensor must remove the DC component.
        for (const auto & wave : waves)
            value += wave_value(wave, x);
        return Float(value);
    });
    const Grid original = delta;
    for (auto index : density.get_real_range())
        density.set_real_from_index(index, Float(1.0 + delta.get_real_from_index(index)));
    for (double radius : {0.0, 0.03}) {
        set_threads(1);
        Tensor reference_tensor;
        CW::tidal_tensor(delta, radius, reference_tensor);
        const auto reference = serial_reference(reference_tensor, 0.05, &density);
        CW::CosmicWeb first;
        for (int threads : thread_counts()) {
            set_threads(threads);
            Tensor tensor;
            CW::tidal_tensor(delta, radius, tensor);
            bool analytic = true, consistent = true;
            for (auto index : delta.get_real_range()) {
                const auto x = delta.get_real_position(delta.get_coord_from_index(index));
                for (int c = 0; c < 6; ++c) {
                    double expected = 0;
                    for (const auto & wave : waves) {
                        const double norm = wave.mode[0] * wave.mode[0] + wave.mode[1] * wave.mode[1] + wave.mode[2] * wave.mode[2];
                        expected += wave.mode[ii[c]] * wave.mode[jj[c]] / norm *
                                    std::exp(-0.5 * 4 * M_PI * M_PI * norm * radius * radius) * wave_value(wave, x);
                    }
                    analytic = analytic && near(tensor[c].get_real_from_index(index), expected, tolerance);
                    consistent = consistent && near(tensor[c].get_real_from_index(index),
                                                     reference_tensor[c].get_real_from_index(index), tolerance);
                }
            }
            check(analytic && consistent, "all six tidal components match smoothed plane waves, threads=" + std::to_string(threads));
            CW::TWebOptions options;
            options.smoothing_scale = radius;
            options.lambda_threshold = 0.05;
            options.store_eigenvalues = true;
            CW::CosmicWeb actual;
            CW::TWeb(delta, options, actual);
            check(same_result(actual, reference, tolerance), "T-web agrees with scalar classification and unsmoothed mass weights");
            if (threads == 1)
                first = actual;
            check(same_result(actual, first, tolerance), "T-web outputs agree across thread counts");
            options.classification_storage = CW::ClassificationStorage::Bytes;
            CW::CosmicWeb compact;
            CW::TWeb(delta, options, compact);
            check(compact_layout(compact) && same_result(compact, actual, tolerance),
                  "T-web byte option preserves labels, eigenvalues and unsmoothed mass fractions");
        }
    }
    check(same_grid(delta, original, 0), "tidal tensor and T-web leave the input density unchanged");
}

int main() {
    test_classification_and_vweb();
    test_tidal_tensor_and_tweb();
    if (FML::NTasks == 1) {
        Grid layout(7);
        CW::ByteClassificationGrid bytes(layout);
        for (int ix = 0; ix < 7; ++ix)
            for (int iy = 0; iy < 7; ++iy)
                for (int iz = 0; iz < 7; ++iz)
                    bytes.set_real({ix, iy, iz}, std::uint8_t((ix + iy + iz) % 4));
        const auto grid = bytes.to_fftw_grid();
        bool exact = bytes.size_bytes() == 343;
        for (int ix = 0; ix < 7; ++ix)
            for (int iy = 0; iy < 7; ++iy)
                for (int iz = 0; iz < 7; ++iz)
                    exact = exact && grid.get_real({ix, iy, iz}) == (ix + iy + iz) % 4;
        check(exact, "odd-mesh byte rows convert exactly to padded FFTW rows");
    }
    // Exercise the particle overload without conflating storage with the threaded
    // deposition fallback's variable floating-point summation order.
    set_threads(1);
    struct Particle {
        double pos[3];
        double * get_pos() { return pos; }
        constexpr int get_ndim() const { return 3; }
    };
    Grid layout(mesh);
    std::vector<Particle> particles;
    for (int ix = 0; ix < layout.get_local_nx(); ++ix)
        for (int iy = 0; iy < mesh; ++iy)
            for (int iz = 0; iz < mesh; ++iz) {
                const int global_x = ix + int(layout.get_local_x_start());
                if ((global_x + iy + iz) % 3)
                    particles.push_back({{(global_x + 0.25) / mesh, (iy + 0.25) / mesh, (iz + 0.25) / mesh}});
            }
    // The original four-field aggregate initialization must retain its meaning.
    CW::TWebOptions options{0.03, 0.05, false, "NGP"};
    CW::CosmicWeb regular, compact;
    CW::TWeb(particles.data(), particles.size(), mesh, options, regular);
    options.classification_storage = CW::ClassificationStorage::Bytes;
    CW::TWeb(particles.data(), particles.size(), mesh, options, compact);
    check(compact_layout(compact) && same_result(compact, regular, tolerance),
          "particle T-web propagates byte storage and retains legacy option initialization");
    if (FML::ThisTask == 0)
        std::cout << (failures ? "COSMICWEB TESTS FAILED\n" : "ALL COSMICWEB TESTS PASSED\n");
    return failures ? 1 : 0;
}
