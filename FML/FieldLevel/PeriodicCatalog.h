#ifndef FML_FIELD_LEVEL_PERIODIC_CATALOG_H
#define FML_FIELD_LEVEL_PERIODIC_CATALOG_H
#include <FML/FieldLevel/Framework.h>
#include <FML/NUFFT/NUFFT.h>
#include <cstring>

namespace FML::FIELDLEVEL::FRAMEWORK {
    struct CatalogEntry {
        std::array<double, 3> position{}; // Physical coordinates; arbitrary periodic images accepted.
        double weight{1};
    };
    struct PeriodicCatalogSettings {
        double boxsize{1}, epsilon{1e-8}, upsampling{2};
        std::array<double, 3> origin{};
        size_t max_input_rows{262144}, max_received_rows{1048576}, message_bytes{8 * 1024 * 1024};
    };
    struct PeriodicCatalogStatistics {
        FML::COMMUNICATION::Count rows{};
        double weight_scale{}, effective_count{}, number_density{}, poisson_power{}, mode_variance{};
        long double sum_scaled_weights{}, sum_squared_scaled_weights{};
    };
    // Collective streaming observation adapter. One fine grid + owned real accumulator;
    // one FFT at finish, no replicated catalog or global mode gather. All points are routed internally.
    template <class GridType = FML::GRID::FFTWGrid<3>>
    class PeriodicCatalogT {
      public:
        using Grid = GridType;

      private:
        using Count = FML::COMMUNICATION::Count;
        struct Point {
            std::array<double, 3> x;
            double weight;
            double * get_pos() { return x.data(); }
        };
        const PeriodicCatalogSettings config_;
        const int mesh_;
        FML::NUFFT::PSWFKernel<3> kernel_;
        const int fine_mesh_;
        Grid output_, fine_;
        FML::GRID::CellOwners<3> owners_;
        std::vector<double> accumulated_, inverse_kernel_;
        Count local_rows_{};
        long double local_sum_{}, local_square_{};
        double scale_{};
        bool finished_{false}, failed_{false};
        PeriodicCatalogStatistics statistics_{};
        static void require(bool ok, const char * message) {
            int bad = !ok;
            FML::MaxOverTasks(&bad);
            if (bad)
                throw std::invalid_argument(message);
        }
        template <class F>
        static void local(F && f) {
            FML::COMMUNICATION::collective_try(std::forward<F>(f), "[PeriodicCatalog] Local preparation");
        }
        static PeriodicCatalogSettings checked(const Grid & prototype, PeriodicCatalogSettings c) {
            static_assert(std::tuple_size<decltype(prototype.get_fourier_integer_wavevector_from_index(0))>::value == 3,
                          "Periodic catalog requires a three-dimensional grid");
            bool valid = prototype.get_nmesh() >= 4 && prototype.get_nmesh() % 2 == 0 &&
                         prototype.get_nmesh() <= INT_MAX / 128 && std::isfinite(c.boxsize) && c.boxsize > 0 &&
                         std::isfinite(std::pow(c.boxsize, 3)) && std::pow(c.boxsize, 3) > 0 &&
                         std::isfinite(c.epsilon) &&
                         c.epsilon >= 32 * std::numeric_limits<FML::GRID::FloatType>::epsilon() && c.epsilon < 1 &&
                         std::isfinite(c.upsampling) && c.upsampling >= 1.25 && c.upsampling <= 8 &&
                         c.max_input_rows > 0 && c.max_received_rows > 0 && c.message_bytes >= sizeof(Point);
            for (double x : c.origin)
                valid &= std::isfinite(x);
            require(valid, "[PeriodicCatalog] Invalid grid, box, tolerance, upsampling or buffer limits");
            const double width =
                std::ceil(std::log(.18 * 1.4 * 1.4 / c.epsilon) / (M_PI * std::sqrt(1 - 1 / c.upsampling)) + 1);
            require(width <= 16, "[PeriodicCatalog] Requested kernel width exceeds NUFFT support");
            for (double x : {double(prototype.get_nmesh()),
                             c.boxsize,
                             c.epsilon,
                             c.upsampling,
                             c.origin[0],
                             c.origin[1],
                             c.origin[2]}) {
                double lo = x, hi = x;
                FML::MinOverTasks(&lo);
                FML::MaxOverTasks(&hi);
                require(lo == hi, "[PeriodicCatalog] Replicated configuration differs");
            }
            for (Count x : {Count(c.max_input_rows), Count(c.max_received_rows), Count(c.message_bytes)}) {
                Count lo = x, hi = x;
                FML::MinOverTasks(&lo);
                FML::MaxOverTasks(&hi);
                require(lo == hi, "[PeriodicCatalog] Buffer controls differ across ranks");
            }
            return c;
        }
        static int choose_fine(const Grid & prototype, const PeriodicCatalogSettings & c, int extra) {
            const int n = prototype.get_nmesh();
            int fine = int(std::ceil(c.upsampling * n));
            if (fine % 2)
                ++fine;
            for (; fine <= 64 * n; fine += 2) {
                // Unlike the slab kernel's convenience fine_nmesh, pencils need no divisibility by NTasks.
                bool good = fits_grid_like(prototype, fine) && fine >= 2 * extra;
                for (int d = 0; d < 3; ++d) {
                    auto r = prototype.get_local_range(d);
                    const auto count =
                        std::llround(double(r[0] + r[1]) * fine / n) - std::llround(double(r[0]) * fine / n);
                    good &= count > 0 && (r[1] == n || count >= extra);
                }
                int bad = !good;
                FML::MaxOverTasks(&bad);
                if (!bad)
                    return fine;
            }
            throw std::invalid_argument("[PeriodicCatalog] No supported fine mesh for this domain/kernel");
        }
        std::vector<Point> route(std::vector<std::vector<Point>> & outgoing) const {
            std::vector<Count> send, receive;
            std::vector<size_t> offsets;
            local([&] {
                send.resize(FML::NTasks);
                receive.resize(FML::NTasks);
                offsets.resize(FML::NTasks);
                for (int p = 0; p < FML::NTasks; ++p)
                    send[p] = outgoing[p].size();
            });
#ifdef USE_MPI
            MPI_Alltoall(send.data(),
                         1,
                         MPI_UNSIGNED_LONG_LONG,
                         receive.data(),
                         1,
                         MPI_UNSIGNED_LONG_LONG,
                         FML::communicator);
#else
            receive = send;
#endif
            Count total = 0;
            bool valid = true;
            for (int p = 0; p < FML::NTasks; ++p) {
                offsets[p] = size_t(total);
                valid &= FML::COMMUNICATION::add_count(
                    total,
                    receive[p],
                    std::min(Count(config_.max_received_rows), Count(std::vector<Point>().max_size())));
            }
            require(valid, "[PeriodicCatalog] Received batch exceeds configured capacity; reduce the input batch");
            std::vector<Point> result;
            local([&] { result.resize(size_t(total)); });
            const size_t self = FML::ThisTask;
            std::copy(outgoing[self].begin(), outgoing[self].end(), result.begin() + offsets[self]);
#ifdef USE_MPI
            const Count capacity = std::min(config_.message_bytes, size_t(INT_MAX)) / sizeof(Point);
            for (int step = 1; step < FML::NTasks; ++step) {
                const int dest = int((static_cast<long long>(FML::ThisTask) + step) % FML::NTasks);
                const int source = int((static_cast<long long>(FML::ThisTask) + FML::NTasks - step) % FML::NTasks);
                Count rounds = std::max(send[dest], receive[source]) / capacity +
                               (std::max(send[dest], receive[source]) % capacity != 0);
                FML::MaxOverTasks(&rounds);
                for (Count r = 0; r < rounds; ++r) {
                    Count start = r * capacity;
                    Count ns = start < send[dest] ? std::min(capacity, send[dest] - start) : 0;
                    Count nr = start < receive[source] ? std::min(capacity, receive[source] - start) : 0;
                    MPI_Sendrecv(ns ? outgoing[dest].data() + size_t(start) : nullptr,
                                 int(ns * sizeof(Point)),
                                 MPI_BYTE,
                                 dest,
                                 0,
                                 nr ? result.data() + offsets[source] + size_t(start) : nullptr,
                                 int(nr * sizeof(Point)),
                                 MPI_BYTE,
                                 source,
                                 0,
                                 FML::communicator,
                                 MPI_STATUS_IGNORE);
                }
            }
#endif
            return result;
        }
        void add(const std::vector<CatalogEntry> & entries) {
            require(entries.size() <= config_.max_input_rows,
                    "[PeriodicCatalog] Input batch exceeds configured capacity");
            Count new_rows = local_rows_;
            require(FML::COMMUNICATION::add_count(new_rows, entries.size()),
                    "[PeriodicCatalog] Local row count overflow");
            double maximum = 0;
            bool valid = true;
            for (const auto & e : entries) {
                valid &= std::isfinite(e.weight) && e.weight >= 0;
                for (double x : e.position)
                    valid &= std::isfinite(x);
                maximum = std::max(maximum, e.weight);
            }
            require(valid, "[PeriodicCatalog] Catalog coordinates/weights must be finite; weights nonnegative");
            FML::MaxOverTasks(&maximum);
            if (maximum == 0) {
                local_rows_ = new_rows;
                return;
            }
            const double next_scale = std::max(scale_, maximum);
            std::vector<std::vector<Point>> outgoing;
            long double sum = 0, square = 0;
            local([&] {
                outgoing.resize(FML::NTasks);
                for (const auto & e : entries) {
                    Point p;
                    for (int d = 0; d < 3; ++d) {
                        long double u =
                            std::fmod((long double)e.position[d] - config_.origin[d], (long double)config_.boxsize) /
                            config_.boxsize;
                        if (u < 0)
                            u += 1;
                        p.x[d] = double(u);
                        if (p.x[d] >= 1)
                            p.x[d] = 0;
                        if (!std::isfinite(p.x[d]))
                            throw std::invalid_argument("Unrepresentable periodic coordinate");
                    }
                    p.weight = e.weight / next_scale;
                    sum += p.weight;
                    square += (long double)p.weight * p.weight;
                    if (p.weight == 0)
                        continue;
                    std::array<int, 3> cell;
                    for (int d = 0; d < 3; ++d)
                        cell[d] = std::min(fine_mesh_ - 1, int(p.x[d] * fine_mesh_));
                    const int owner = owners_.owner(cell);
                    if (owner < 0 || owner >= FML::NTasks)
                        throw std::invalid_argument("Catalog point has no grid owner");
                    outgoing[owner].push_back(p);
                }
            });
            auto points = route(outgoing);
            std::vector<double> weights;
            local([&] {
                weights.reserve(points.size());
                for (auto p : points)
                    weights.push_back(p.weight);
            });
            fine_.set_grid_status_real(true);
            FML::NUFFT::spread_to_fine_grid<3, Point>(points.data(), points.size(), weights, kernel_, fine_);
            const double ratio = scale_ == 0 ? 0 : scale_ / next_scale;
            size_t q = 0;
            for (auto i : fine_.get_real_range()) {
                accumulated_[q] = accumulated_[q] * ratio + fine_.get_real_from_index(i);
                ++q;
            }
            local_sum_ = local_sum_ * ratio + sum;
            local_square_ = local_square_ * (long double)ratio * ratio + square;
            scale_ = next_scale;
            local_rows_ = new_rows;
        }

      public:
        PeriodicCatalogT(const Grid & prototype, PeriodicCatalogSettings c)
            : config_(checked(prototype, c)), mesh_(prototype.get_nmesh()),
              kernel_(config_.epsilon, config_.upsampling),
              fine_mesh_(choose_fine(prototype, config_, kernel_.extra_slices())),
              output_(make_grid_like(prototype, mesh_, 0, 0)),
              fine_(make_grid_like(prototype, fine_mesh_, kernel_.extra_slices(), kernel_.extra_slices())),
              owners_(fine_) {
            local([&] {
                size_t cells = 0;
                for (auto i : fine_.get_real_range()) {
                    (void)i;
                    ++cells;
                }
                accumulated_.assign(cells, 0.);
                inverse_kernel_.resize(mesh_ / 2 + 1);
                for (size_t n = 0; n < inverse_kernel_.size(); ++n) {
                    inverse_kernel_[n] = 1. / kernel_.fourier(2 * M_PI * n / fine_mesh_);
                    if (!std::isfinite(inverse_kernel_[n]))
                        throw std::invalid_argument("Singular NUFFT kernel transform");
                }
            });
            reset();
        }
        PeriodicCatalogT(const PeriodicCatalogT &) = delete;
        PeriodicCatalogT(PeriodicCatalogT &&) = delete;
        void reset() {
            std::fill(accumulated_.begin(), accumulated_.end(), 0.);
            output_.fill_fourier_grid(FML::GRID::ComplexType(0));
            output_.set_grid_status_real(false);
            fine_.fill_real_grid(0.);
            fine_.set_grid_status_real(true);
            local_rows_ = 0;
            local_sum_ = local_square_ = 0;
            scale_ = 0;
            statistics_ = {};
            finished_ = failed_ = false;
        }
        int fine_mesh() const { return fine_mesh_; }
        int kernel_width() const { return kernel_.width; }
        void add_batch(const std::vector<CatalogEntry> & entries) {
            require(!finished_ && !failed_, "[PeriodicCatalog] Reset before adding to a finished/failed stream");
            try {
                add(entries);
            } catch (...) {
                failed_ = true;
                throw;
            }
        }
        const PeriodicCatalogStatistics & finish() {
            require(!failed_, "[PeriodicCatalog] Reset a failed stream before finishing");
            if (finished_)
                return statistics_;
            try {
                long double local_values[2] = {local_sum_, local_square_}, values[2];
#ifdef USE_MPI
                MPI_Allreduce(local_values, values, 2, MPI_LONG_DOUBLE, MPI_SUM, FML::communicator);
#else
                std::copy(local_values, local_values + 2, values);
#endif
                Count count = 0;
                bool valid = true;
                for (auto n : FML::GatherFromTasks(&local_rows_))
                    valid &= FML::COMMUNICATION::add_count(count, n);
                require(valid && count > 0 && values[0] > 0 && values[1] > 0 && std::isfinite(values[0]) &&
                            std::isfinite(values[1]),
                        "[PeriodicCatalog] Empty, zero-weight or unrepresentable catalog");
                statistics_.rows = count;
                statistics_.weight_scale = scale_;
                statistics_.sum_scaled_weights = values[0];
                statistics_.sum_squared_scaled_weights = values[1];
                statistics_.mode_variance = double(values[1] / values[0] / values[0]);
                statistics_.effective_count = 1. / statistics_.mode_variance;
                statistics_.poisson_power = std::pow(config_.boxsize, 3) * statistics_.mode_variance;
                statistics_.number_density = double(count) / std::pow(config_.boxsize, 3);
                require(statistics_.mode_variance > 0 && std::isfinite(statistics_.mode_variance) &&
                            statistics_.poisson_power > 0 && std::isfinite(statistics_.poisson_power) &&
                            statistics_.number_density > 0 && std::isfinite(statistics_.number_density) &&
                            std::isfinite(statistics_.effective_count),
                        "[PeriodicCatalog] Unrepresentable catalog density/noise normalization");
                size_t q = 0;
                for (auto i : fine_.get_real_range())
                    fine_.set_real_from_index(i, accumulated_[q++]);
                fine_.fftw_r2c();
                const double prefactor = std::pow(double(fine_mesh_), 3) / double(values[0]);
                FML::GRID::copy_fourier_modes<3>(fine_, output_, [&](auto n) {
                    return prefactor * inverse_kernel_[std::abs(n[0])] * inverse_kernel_[std::abs(n[1])] *
                           inverse_kernel_[std::abs(n[2])];
                });
                bool finite = true;
                for (auto i : output_.get_fourier_range()) {
                    const auto n = output_.get_fourier_integer_wavevector_from_index(i);
                    bool excluded = n[0] == 0 && n[1] == 0 && n[2] == 0;
                    for (int v : n)
                        excluded |= std::abs(v) == mesh_ / 2;
                    if (excluded)
                        output_.set_fourier_from_index(i, 0.);
                    const auto v = output_.get_fourier_from_index(i);
                    finite &= std::isfinite(v.real()) && std::isfinite(v.imag());
                }
                require(finite, "[PeriodicCatalog] Nonfinite density modes");
                finished_ = true;
                return statistics_;
            } catch (...) {
                failed_ = true;
                throw;
            }
        }
        const Grid & fourier_grid() const {
            if (!finished_ || failed_)
                throw std::logic_error("[PeriodicCatalog] Finish before reading density modes");
            return output_;
        }
        template <class ModeList>
        Modes modes(const ModeList & requested) const {
            require(finished_ && !failed_, "[PeriodicCatalog] Finish before exporting modes");
            bool valid = true;
            for (const auto & m : requested) {
                if (m.fourier_index >= size_t(output_.get_ntot_fourier())) {
                    valid = false;
                    continue;
                }
                const auto n = output_.get_fourier_integer_wavevector_from_index(m.fourier_index);
                for (int d = 0; d < 3; ++d)
                    valid &= n[d] == m.n[d] && std::abs(n[d]) < mesh_ / 2;
                valid &= n[0] != 0 || n[1] != 0 || n[2] != 0;
            }
            require(valid,
                    "[PeriodicCatalog] Requested modes do not match this grid/layout or include excluded planes");
            Modes result;
            local([&] {
                result.reserve(requested.size());
                for (const auto & m : requested)
                    result.push_back(output_.get_fourier_from_index(m.fourier_index));
            });
            return result;
        }
    };
    using PeriodicCatalog = PeriodicCatalogT<>;
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif
