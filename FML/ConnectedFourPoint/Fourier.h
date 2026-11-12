#ifndef FML_CONNECTED_FOUR_POINT_FOURIER_H
#define FML_CONNECTED_FOUR_POINT_FOURIER_H
#include <FML/ConnectedFourPoint/ConnectedFourPoint.h>
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <map>
namespace FML::FOURPOINT {
    using FourierGrid = FML::GRID::FFTWGrid<3>;
    using Configuration = std::array<int, 4>;
    struct ShellOptions {
        double boxsize{1};
        // Physical inverse-length units. Half-open [low, high); DC and even-mesh Nyquist planes excluded.
        std::vector<double> k_edges;
    };
    struct ShellResult {
        Configuration bins;
        FourPointResult moment; // Dimensionless shell-integrated fourth moment and Wick subtraction.
        std::array<double, 4> mode_counts;
        double closing_quartets{};                 // Ordered full signed-wavevector tuples, not independent samples.
        double raw{}, disconnected{}, connected{}; // V^3 * moment / closing_quartets; length^9.
        int work_mesh{};
        bool valid{}; // Empty shells/no closing quartets return NaN normalized values.
    };
    namespace detail {
        inline bool
        uses_mode(const FourierGrid & grid, const std::array<double, 3> & wave, double k2, double low, double high) {
            return FML::CORRELATIONFUNCTIONS::detail::polyspectrum_uses_mode<3, 4>(grid,
                                                                                   wave,
                                                                                   k2,
                                                                                   low * low,
                                                                                   high * high,
                                                                                   true);
        }
        inline FourierGrid shell(const FourierGrid & input, const ShellOptions & o, int bin, int mesh, bool geometry) {
            FourierGrid field(input.get_nmesh());
            field.set_grid_status_real(false);
            const double low = o.k_edges[bin] * o.boxsize, high = o.k_edges[bin + 1] * o.boxsize;
            for (auto index : input.get_fourier_range()) {
                std::array<double, 3> wave;
                double k2;
                input.get_fourier_wavevector_and_norm2_by_index(index, wave, k2);
                const auto value = geometry ? FML::GRID::ComplexType(1) : input.get_fourier_from_index(index);
                field.set_fourier_from_index(index,
                                             uses_mode(input, wave, k2, low, high) ? value : FML::GRID::ComplexType{});
            }
            if (mesh != input.get_nmesh())
                field = FML::GRID::resize_fourier_grid(field, mesh);
            field.fftw_c2r();
            return field;
        }
        inline long double average_product(const std::array<const FourierGrid *, 4> & fields) {
            long double sum = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : sum)
#endif
            for (int x = 0; x < fields[0]->get_local_nx(); ++x)
                for (auto index : fields[0]->get_real_range(x, x + 1)) {
                    long double product = 1;
                    for (auto f : fields)
                        product *= f->get_real_from_index(index);
                    sum += product;
                }
            FML::SumOverTasks(&sum);
            const long double n = fields[0]->get_nmesh();
            return sum / (n * n * n);
        }
    } // namespace detail
    /// Collective periodic shell estimator. Input is FML-normalized Fourier density contrast.
    /// References are exact ensemble covariances of the four dimensionless shell-filtered fields,
    /// one row-major 4x4 matrix per requested tuple. No particle-noise/window correction is applied.
    inline std::vector<ShellResult>
    connected_shell_four_point(const FourierGrid & input,
                               const ShellOptions & o,
                               const std::vector<Configuration> & configurations,
                               const std::vector<std::array<double, 16>> & reference_covariances = {}) {
        bool valid = input.get_nmesh() > 0 && !const_cast<FourierGrid &>(input).get_grid_status_real() &&
                     std::isfinite(o.boxsize) && o.boxsize > 0 && o.k_edges.size() >= 2 && !configurations.empty() &&
                     (reference_covariances.empty() || reference_covariances.size() == configurations.size());
        double maximum = 0;
        for (size_t i = 0; i < o.k_edges.size(); ++i)
            valid = valid && std::isfinite(o.k_edges[i]) && o.k_edges[i] >= 0 &&
                    (!i || o.k_edges[i] > o.k_edges[i - 1]) && std::isfinite(std::pow(o.k_edges[i] * o.boxsize, 2));
        for (auto cfg : configurations)
            for (int bin : cfg) {
                const bool in_range = bin >= 0 && size_t(bin) + 1 < o.k_edges.size();
                valid = valid && in_range;
                if (in_range)
                    maximum = std::max(maximum, o.k_edges[bin + 1] * o.boxsize);
            }
        const long double volume = (long double)o.boxsize * o.boxsize * o.boxsize;
        valid = valid && std::isfinite(volume * volume * volume) && volume > 0;
        for (const auto & c : reference_covariances)
            valid = valid && detail::valid_covariance(c);
        require(valid, "[FourPoint] Invalid grid/shell options/reference");
        std::vector<double> signature{double(input.get_nmesh()),
                                      o.boxsize,
                                      double(o.k_edges.size()),
                                      double(configurations.size()),
                                      double(reference_covariances.size())};
        signature.insert(signature.end(), o.k_edges.begin(), o.k_edges.end());
        for (auto cfg : configurations)
            for (int b : cfg)
                signature.push_back(b);
        for (auto c : reference_covariances)
            signature.insert(signature.end(), c.begin(), c.end());
        agree(signature);
        valid = true;
        for (auto index : input.get_fourier_range()) {
            const auto z = input.get_fourier_from_index(index);
            valid = valid && std::isfinite(z.real()) && std::isfinite(z.imag());
        }
        require(valid, "[FourPoint] Nonfinite Fourier input");
        const int mesh =
            FML::CORRELATIONFUNCTIONS::detail::polyspectrum_work_mesh<4>(input.get_nmesh(), {maximum}, true);
        std::vector<ShellResult> result;
        for (size_t c = 0; c < configurations.size(); ++c) {
            ShellResult out;
            out.bins = configurations[c];
            out.work_mesh = mesh;
            std::map<int, FourierGrid> fields;
            // Geometry phase and density phase never retain both sets of shell fields.
            for (int bin : out.bins)
                if (!fields.count(bin))
                    fields.emplace(bin, detail::shell(input, o, bin, mesh, true));
            std::array<const FourierGrid *, 4> selected;
            for (int i = 0; i < 4; ++i)
                selected[i] = &fields.at(out.bins[i]);
            const auto quartets = detail::average_product(selected);
            require(std::isfinite(quartets) && quartets >= -0.5L, "[FourPoint] Invalid quartet count");
            out.closing_quartets = double(std::round(quartets));
            fields.clear();
            std::map<int, double> counts;
            for (int bin : out.bins)
                if (!fields.count(bin)) {
                    double count = 0;
                    for (auto index : input.get_fourier_range()) {
                        std::array<double, 3> wave;
                        double k2;
                        input.get_fourier_wavevector_and_norm2_by_index(index, wave, k2);
                        if (detail::uses_mode(input,
                                              wave,
                                              k2,
                                              o.k_edges[bin] * o.boxsize,
                                              o.k_edges[bin + 1] * o.boxsize))
                            count += FML::GRID::fourier_mode_multiplicity(int(index % (input.get_nmesh() / 2 + 1)),
                                                                          input.get_nmesh());
                    }
                    FML::SumOverTasks(&count);
                    counts[bin] = count;
                    fields.emplace(bin, detail::shell(input, o, bin, mesh, false));
                }
            for (int i = 0; i < 4; ++i) {
                selected[i] = &fields.at(out.bins[i]);
                out.mode_counts[i] = counts.at(out.bins[i]);
            }
            const size_t count = size_t(selected[0]->get_local_nx()) * mesh * mesh;
            auto sample = [&](size_t row) {
                const std::array<int, 3> coord{int(row / (size_t(mesh) * mesh)),
                                               int(row / mesh % mesh),
                                               int(row % mesh)};
                const auto index = selected[0]->get_index_real(coord);
                std::array<FML::GRID::FloatType, 4> values;
                for (int i = 0; i < 4; ++i)
                    values[i] = selected[i]->get_real_from_index(index);
                return values;
            };
            std::optional<std::array<double, 16>> reference;
            if (!reference_covariances.empty())
                reference = reference_covariances[c];
            out.moment = zero_mean_four_point(count, sample, reference);
            out.valid = out.closing_quartets > 0;
            const long double factor = out.valid ? volume * volume * volume / out.closing_quartets :
                                                   std::numeric_limits<long double>::quiet_NaN();
            out.raw = double(factor * out.moment.fourth);
            out.disconnected = double(factor * out.moment.disconnected);
            out.connected = double(factor * out.moment.connected);
            require(!out.valid ||
                        (std::isfinite(out.raw) && std::isfinite(out.disconnected) && std::isfinite(out.connected)),
                    "[FourPoint] Normalized moment overflow");
            result.push_back(out);
        }
        return result;
    }
} // namespace FML::FOURPOINT
#endif
