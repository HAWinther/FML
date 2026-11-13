#ifndef FML_HYDRODYNAMICS_CARTESIAN_GRID_HEADER
#define FML_HYDRODYNAMICS_CARTESIAN_GRID_HEADER
#include <FML/Hydrodynamics/Euler.h>
#include <FML/Global/MPICounts.h>
#include <FML/Global/ChunkedExchange.h>
#include <vector>
namespace FML::HYDRODYNAMICS {
    enum class Boundary { Periodic, Outflow, Reflecting };
    namespace detail {
        namespace C = FML::COMMUNICATION;
        template <size_t N>
        void agree(const std::array<double, N> & value, const char * message) {
#ifdef USE_MPI
            std::array<double, N> lo{}, hi{};
            MPI_Allreduce(value.data(), lo.data(), int(N), MPI_DOUBLE, MPI_MIN, FML::communicator);
            MPI_Allreduce(value.data(), hi.data(), int(N), MPI_DOUBLE, MPI_MAX, FML::communicator);
            if (lo != hi)
                throw std::invalid_argument(message);
#else
            (void)value;
            (void)message;
#endif
        }
        inline size_t product(size_t a, size_t b) {
            if (b && a > std::numeric_limits<size_t>::max() / b)
                throw std::overflow_error("[Hydro] Grid product overflow");
            return a * b;
        }
        template <int D>
        size_t count(const std::array<int, D> & shape) {
            size_t n = 1;
            for (auto x : shape)
                n = product(n, size_t(x));
            return n;
        }
        template <int D>
        std::array<int, D> coordinates(size_t index, const std::array<int, D> & shape) {
            std::array<int, D> p{};
            for (int d = D - 1; d >= 0; --d) {
                p[d] = int(index % shape[d]);
                index /= shape[d];
            }
            return p;
        }
        template <int D>
        size_t index(const std::array<int, D> & p, const std::array<int, D> & shape) {
            size_t k = 0;
            for (int d = 0; d < D; ++d)
                k = k * shape[d] + p[d];
            return k;
        }
        inline std::pair<int, bool> boundary_index(int index, int n, Boundary low, Boundary high) {
            bool flip = false;
            while (index < 0 || index >= n) {
                const auto bc = index < 0 ? low : high;
                if (bc == Boundary::Periodic) {
                    index = (index % n + n) % n;
                    break;
                }
                if (bc == Boundary::Outflow) {
                    index = std::clamp(index, 0, n - 1);
                    break;
                }
                index = index < 0 ? -index - 1 : n - 1 - (index - n);
                flip = !flip;
            }
            return {index, flip};
        }
    } // namespace detail
    template <int D>
    struct Layout {
        std::array<int, D> global_shape{}, owned_shape{};
        Vector<D> lengths{}, spacing{};
        int start = 0, width = 0;
        size_t cells = 0;
        Vector<D> center(size_t i) const {
            auto q = detail::coordinates<D>(i, owned_shape);
            q[0] += start;
            Vector<D> x{};
            for (int d = 0; d < D; ++d)
                x[d] = (q[d] + .5) * spacing[d];
            return x;
        }
        size_t global_index(size_t i) const {
            auto q = detail::coordinates<D>(i, owned_shape);
            q[0] += start;
            return detail::index<D>(q, global_shape);
        }
        double cell_volume() const {
            double v = 1;
            for (auto dx : spacing)
                v *= dx;
            return v;
        }
    };
    template <int D>
    class CartesianGrid {
        struct Plan {
            size_t source, destination;
            bool reflect;
        };
        struct Record {
            size_t destination;
            Conserved<D> value;
        };
        Layout<D> layout_;
        std::array<std::array<Boundary, 2>, D> boundaries_;
        std::array<int, D> padded_shape_{};
        std::vector<std::vector<Plan>> plan_;
        std::vector<Conserved<D>> padded_;
        size_t message_bytes_;

      public:
        CartesianGrid(std::array<int, D> shape,
                      Vector<D> lengths,
                      std::array<std::array<Boundary, 2>, D> bc,
                      size_t max_cells,
                      size_t message_bytes)
            : boundaries_(bc), message_bytes_(message_bytes) {
            detail::C::collective_try(
                [&] {
                    layout_.global_shape = shape;
                    layout_.lengths = lengths;
                    layout_.start = int(static_cast<long long>(FML::ThisTask) * shape[0] / FML::NTasks);
                    layout_.width =
                        int(static_cast<long long>(FML::ThisTask + 1) * shape[0] / FML::NTasks) - layout_.start;
                    layout_.owned_shape = shape;
                    layout_.owned_shape[0] = layout_.width;
                    for (int d = 0; d < D; ++d)
                        layout_.spacing[d] = lengths[d] / shape[d];
                    layout_.cells = layout_.width ? detail::count<D>(layout_.owned_shape) : 0;
                    padded_shape_ = layout_.owned_shape;
                    for (auto & x : padded_shape_)
                        x += 4;
                    const size_t n = layout_.width ? detail::count<D>(padded_shape_) : 0;
                    if (n > max_cells || n > std::numeric_limits<std::ptrdiff_t>::max() / sizeof(Conserved<D>))
                        throw std::length_error("[Hydro] Local padded-grid budget exceeded");
                    padded_.resize(n);
                    plan_.resize(FML::NTasks);
                    std::vector<int> ends(FML::NTasks);
                    for (int r = 0; r < FML::NTasks; ++r)
                        ends[r] = int(static_cast<long long>(r + 1) * shape[0] / FML::NTasks);
                    size_t plane = 1;
                    for (int d = 1; d < D; ++d)
                        plane = detail::product(plane, size_t(shape[d]));
                    for (int peer = 0; peer < FML::NTasks; ++peer) {
                        int start = int(static_cast<long long>(peer) * shape[0] / FML::NTasks),
                            width = ends[peer] - start;
                        if (!width)
                            continue;
                        auto dest_shape = shape;
                        dest_shape[0] = width;
                        for (auto & n : dest_shape)
                            n += 4;
                        for (int ghost : {-2, -1, width, width + 1}) {
                            auto mapping = detail::boundary_index(start + ghost, shape[0], bc[0][0], bc[0][1]);
                            int owner = int(std::upper_bound(ends.begin(), ends.end(), mapping.first) - ends.begin());
                            if (owner != FML::ThisTask)
                                continue;
                            for (size_t yz = 0; yz < plane; ++yz) {
                                auto local = detail::coordinates<D>(yz, layout_.owned_shape);
                                local[0] = mapping.first - layout_.start;
                                auto destination = local;
                                destination[0] = ghost;
                                for (auto & q : destination)
                                    q += 2;
                                plan_[peer].push_back({detail::index<D>(local, layout_.owned_shape),
                                                       detail::index<D>(destination, dest_shape),
                                                       mapping.second});
                            }
                        }
                    }
                },
                "Hydro Cartesian halo plan");
        }
        const Layout<D> & layout() const { return layout_; }
        const std::array<int, D> & padded_shape() const { return padded_shape_; }
        const std::vector<Conserved<D>> & padded() const { return padded_; }
        size_t padded_index(std::array<int, D> coordinate) const {
            for (auto & x : coordinate)
                x += 2;
            return detail::index<D>(coordinate, padded_shape_);
        }
        void exchange(const std::vector<Conserved<D>> & owned) {
            std::vector<std::vector<Record>> send;
            detail::C::collective_try(
                [&] {
                    if (owned.size() != layout_.cells)
                        throw std::invalid_argument("[Hydro] Owned grid shape mismatch");
                    for (size_t i = 0; i < owned.size(); ++i)
                        padded_[padded_index(detail::coordinates<D>(i, layout_.owned_shape))] = owned[i];
                    send.resize(FML::NTasks);
                    for (int peer = 0; peer < FML::NTasks; ++peer) {
                        send[peer].reserve(plan_[peer].size());
                        for (const auto & p : plan_[peer]) {
                            auto value = owned[p.source];
                            if (p.reflect)
                                value[1] = -value[1];
                            send[peer].push_back({p.destination, value});
                        }
                    }
                },
                "Hydro halo staging");
            detail::C::exchange_records_chunked(
                send,
                [&](const Record * p, size_t n) noexcept {
                    for (size_t i = 0; i < n; ++i)
                        padded_[p[i].destination] = p[i].value;
                },
                message_bytes_);
            // Extend y/z after x, including corners. Only a normal momentum changes sign.
            for (int axis = 1; axis < D; ++axis)
                for (size_t i = 0; i < padded_.size(); ++i) {
                    auto q = detail::coordinates<D>(i, padded_shape_);
                    int at = q[axis] - 2;
                    if (at >= 0 && at < layout_.owned_shape[axis])
                        continue;
                    const auto mapped = detail::boundary_index(at,
                                                               layout_.owned_shape[axis],
                                                               boundaries_[axis][0],
                                                               boundaries_[axis][1]);
                    auto source = q;
                    source[axis] = mapped.first + 2;
                    auto value = padded_[detail::index<D>(source, padded_shape_)];
                    if (mapped.second)
                        value[1 + axis] = -value[1 + axis];
                    padded_[i] = value;
                }
        }
    };
} // namespace FML::HYDRODYNAMICS
#endif
