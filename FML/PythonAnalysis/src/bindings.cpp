#include <FML/CountsInCells/SpatialStatistics.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#if defined(USE_MPI) || defined(USE_OMP) || defined(USE_FFTW)
#error "fml-analysis is a serial extension"
#endif
namespace py = pybind11;
namespace Spatial = FML::COUNTS_IN_CELLS;
using RealArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

RealArray real_array(py::handle object, int ndim, const char *name) {
    auto array = py::array::ensure(object);
    if (!array || (array.dtype().kind() != 'i' && array.dtype().kind() != 'u' && array.dtype().kind() != 'f'))
        throw py::type_error(std::string(name) + " must be a real integer/float array");
    if (array.ndim() != ndim)
        throw py::value_error(std::string(name) + " has the wrong number of dimensions");
    auto result = RealArray::ensure(array);
    if (!result)
        throw py::type_error(std::string(name) + " cannot be converted to a float64 array");
    return result;
}
std::vector<Spatial::Point> points(py::handle object, const char *name) {
    auto a = real_array(object, 2, name);
    if (a.shape(1) != 3)
        throw py::value_error(std::string(name) + " must have shape (n, 3)");
    auto v = a.unchecked<2>();
    std::vector<Spatial::Point> result(size_t(a.shape(0)));
    for (py::ssize_t i = 0; i < a.shape(0); ++i)
        for (int d = 0; d < 3; ++d)
            result[size_t(i)][d] = v(i, d);
    return result;
}
std::vector<double> reals(py::handle object, const char *name) {
    auto a = real_array(object, 1, name);
    return {a.data(), a.data() + a.size()};
}
std::vector<size_t> integers(py::handle object, bool exclusion = false) {
    auto a = py::array::ensure(object);
    if (!a || (a.dtype().kind() != 'i' && a.dtype().kind() != 'u'))
        throw py::type_error("orders/exclude must be integer arrays (no floating conversion)");
    if (a.ndim() != 1)
        throw py::value_error("orders/exclude must be one-dimensional");
    std::vector<size_t> result;
    for (py::ssize_t i = 0; i < a.size(); ++i) {
        py::int_ item(a.attr("__getitem__")(i));
        if (py::cast<bool>(item.attr("__lt__")(0))) {
            if (exclusion && py::cast<bool>(item.attr("__eq__")(-1)))
                result.push_back(SIZE_MAX);
            else
                throw py::value_error("orders must be positive; exclude uses -1 or an original catalogue index");
        } else {
            auto value = PyLong_AsUnsignedLongLong(item.ptr());
            if (PyErr_Occurred())
                throw py::error_already_set();
            if (value > SIZE_MAX)
                throw py::value_error("integer exceeds size_t");
            // SIZE_MAX is reserved internally; only Python -1 denotes no exclusion.
            if (exclusion && value == SIZE_MAX)
                throw py::value_error("exclude index out of range");
            result.push_back(size_t(value));
        }
    }
    return result;
}
std::vector<Spatial::Query> queries(py::handle object, py::handle exclude) {
    auto positions = points(object, "queries");
    std::vector<size_t> excluded;
    if (!exclude.is_none()) {
        excluded = integers(exclude, true);
        if (excluded.size() != positions.size())
            throw py::value_error("exclude length must match queries");
    }
    std::vector<Spatial::Query> result(positions.size());
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = {positions[i], exclude.is_none() ? SIZE_MAX : excluded[i]};
    return result;
}
template <class T> py::array_t<T> array(const std::vector<T> &values) {
    py::array_t<T> result(values.size());
    std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}
template <class T> py::array_t<T> matrix(const std::vector<std::vector<T>> &values, size_t columns) {
    py::array_t<T> result({py::ssize_t(values.size()), py::ssize_t(columns)});
    for (size_t i = 0; i < values.size(); ++i)
        std::copy(values[i].begin(), values[i].end(), result.mutable_data() + i * columns);
    return result;
}
py::dict cell(const Spatial::CellStatistics &s) {
    py::dict result;
#define SCALAR(name) result[#name] = s.name
#define ARRAY(name) result[#name] = array(s.name)
    SCALAR(radius);
    SCALAR(volume);
    SCALAR(reference_number_density);
    SCALAR(queries);
    SCALAR(density_underflow);
    SCALAR(density_overflow);
    SCALAR(density_undefined);
    SCALAR(mean);
    SCALAR(variance);
    SCALAR(mean_delta);
    SCALAR(poisson_subtracted_variance);
    SCALAR(normalized_connected_variance);
    ARRAY(histogram);
    ARRAY(density_histogram);
    ARRAY(probability);
    ARRAY(density_probability);
    ARRAY(density_pdf);
    ARRAY(factorial_moments);
    ARRAY(normalized_factorial_moments);
#undef ARRAY
#undef SCALAR
    return result;
}
struct Catalogue {
    const double boxsize;
    Spatial::SpatialStatistics statistics;
    Catalogue(const FML::KDTREE::pointVec &positions, double box) : boxsize(box), statistics(positions, box) {}
};
PYBIND11_MODULE(_native, m) {
    m.doc() = "Serial FML spatial statistics, API schema 1; owned inputs and NumPy-owned results.";
    py::class_<Catalogue>(m, "Catalogue")
        .def(py::init([](py::object positions, double boxsize) {
                 auto p = points(positions, "positions");
                 FML::KDTREE::pointVec owned;
                 owned.reserve(p.size());
                 for (auto x : p)
                     owned.emplace_back(x.begin(), x.end());
                 py::gil_scoped_release release;
                 return std::make_unique<Catalogue>(owned, boxsize);
             }),
             py::arg("positions"), py::kw_only(), py::arg("boxsize") = 0.0)
        .def("__len__", [](const Catalogue &c) { return c.statistics.size(); })
        .def_readonly("boxsize", &Catalogue::boxsize)
        .def(
            "knn_distances",
            [](const Catalogue &c, py::object positions, py::object orders, py::object exclude) {
                auto q = queries(positions, exclude);
                auto k = integers(orders);
                Spatial::KnnDistances result;
                {
                    py::gil_scoped_release release;
                    result = c.statistics.knn_distances(q, k);
                }
                return matrix(result.distances, q.size());
            },
            py::arg("queries"), py::arg("orders"), py::kw_only(), py::arg("exclude") = py::none())
        .def(
            "sphere_counts",
            [](const Catalogue &c, py::object positions, py::object radii, py::object exclude) {
                auto q = queries(positions, exclude);
                auto r = reals(radii, "radii");
                std::vector<std::vector<size_t>> result;
                {
                    py::gil_scoped_release release;
                    result = c.statistics.sphere_counts(q, r);
                }
                py::array_t<Spatial::Count> out({py::ssize_t(result.size()), py::ssize_t(q.size())});
                for (size_t i = 0; i < result.size(); ++i)
                    std::copy(result[i].begin(), result[i].end(), out.mutable_data() + i * q.size());
                return out;
            },
            py::arg("queries"), py::arg("radii"), py::kw_only(), py::arg("exclude") = py::none())
        .def(
            "knn_cdf",
            [](const Catalogue &c, py::object positions, py::object orders, py::object radii, py::object exclude) {
                auto q = queries(positions, exclude);
                auto k = integers(orders);
                auto r = reals(radii, "radii");
                Spatial::KnnCDF result;
                {
                    py::gil_scoped_release release;
                    result = c.statistics.knn_cdf(q, k, r, Spatial::Reduction::Local);
                }
                py::dict out;
                out["orders"] = array(result.orders);
                out["radii"] = array(result.radii);
                out["hits"] = matrix(result.hits, r.size());
                out["cdf"] = matrix(result.cdf, r.size());
                out["queries"] = result.queries;
                out["excluded_queries"] = result.excluded_queries;
                out["boxsize"] = result.boxsize;
                return out;
            },
            py::arg("queries"), py::arg("orders"), py::arg("radii"), py::kw_only(), py::arg("exclude") = py::none())
        .def(
            "counts_in_cells",
            [](const Catalogue &c, py::object positions, py::object radii, py::object exclude, size_t histogram_limit,
               unsigned factorial_order, py::object delta_edges, py::object number_density, py::object window,
               bool allow_truncated_cells) {
                auto q = queries(positions, exclude);
                auto r = reals(radii, "radii");
                Spatial::Options options;
                options.reduction = Spatial::Reduction::Local;
                options.histogram_limit = histogram_limit;
                options.factorial_order = factorial_order;
                if (!delta_edges.is_none())
                    options.delta_edges = reals(delta_edges, "delta_edges");
                if (!number_density.is_none())
                    options.number_density = number_density.cast<double>();
                if (!window.is_none()) {
                    auto bounds = points(window, "window");
                    if (bounds.size() != 2)
                        throw py::value_error("window must have shape (2, 3)");
                    options.window = Spatial::Bounds{bounds[0], bounds[1]};
                }
                options.allow_truncated_cells = allow_truncated_cells;
                Spatial::CountsInCells result;
                {
                    py::gil_scoped_release release;
                    result = c.statistics.counts_in_cells(q, r, options);
                }
                py::list cells;
                for (const auto &s : result.cells)
                    cells.append(cell(s));
                py::dict out;
                out["cells"] = cells;
                out["delta_edges"] = array(result.delta_edges);
                out["histogram_limit"] = result.histogram_limit;
                out["boxsize"] = result.boxsize;
                out["excluded_queries"] = result.excluded_queries;
                out["uncorrected_clipping"] = result.uncorrected_clipping;
                return out;
            },
            py::arg("queries"), py::arg("radii"), py::kw_only(), py::arg("exclude") = py::none(),
            py::arg("histogram_limit") = 128, py::arg("factorial_order") = 4, py::arg("delta_edges") = py::none(),
            py::arg("number_density") = py::none(), py::arg("window") = py::none(),
            py::arg("allow_truncated_cells") = false);
    m.def(
        "uniform_queries",
        [](size_t number, py::object window, uint64_t seed, uint64_t first_id, uint32_t stream) {
            auto b = points(window, "bounds");
            if (b.size() != 2)
                throw py::value_error("bounds must have shape (2, 3)");
            std::vector<Spatial::Query> result;
            {
                py::gil_scoped_release release;
                result = Spatial::uniform_queries(number, {b[0], b[1]}, seed, first_id, stream);
            }
            py::array_t<double> out({py::ssize_t(number), py::ssize_t(3)});
            for (size_t i = 0; i < number; ++i)
                std::copy(result[i].position.begin(), result[i].position.end(), out.mutable_data() + 3 * i);
            return out;
        },
        py::arg("number"), py::arg("bounds"), py::kw_only(), py::arg("seed"), py::arg("first_id") = 0,
        py::arg("stream") = 0);
}
