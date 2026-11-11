//=====================================================================================================================
// pybind11 bindings of the halo model (Halomodel.h): the module _halomodel used by HaloModelWrapper.py.
// A cosmology with w0wa expansion and mu(a) = 1 + (mu0 + mua (1 - a)) / E^2(a) for the growth and spherical
// collapse, and the linear P(k, z = 0) from a file.
//=====================================================================================================================
#include <FML/HaloModel/Halomodel.h>
#include <FML/SphericalCollapse/SphericalCollapse.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>

namespace py = pybind11;
using namespace FML::COSMOLOGY::HALOMODEL;
using FML::COSMOLOGY::SPHERICALCOLLAPSE::compute_growthfactor;

namespace {
    SphericalCollapseModel make_model(double OmegaM, double w0, double wa, double mu0, double mua) {
        const double OmegaLambda = 1.0 - OmegaM;
        const Func1 E = [=](double x) {
            const double a = std::exp(x);
            return std::sqrt(OmegaM * std::exp(-3.0 * x) +
                             OmegaLambda * std::exp(3.0 * wa * (a - 1) - 3 * (1 + w0 + wa) * x));
        };
        const Func1 logEprime = [=](double x) {
            const double e = E(x), a = std::exp(x);
            return 1.0 / (2.0 * e * e) *
                   (-3.0 * OmegaM / (a * a * a) + OmegaLambda * std::exp(3.0 * wa * (a - 1) - 3 * (1 + w0 + wa) * x) *
                                                      (3.0 * wa * a - 3.0 * (1 + w0 + wa)));
        };
        const Func1 mu = [=](double x) {
            const double e = E(x), a = std::exp(x);
            return 1.0 + (mu0 + mua * (1.0 - a)) / (e * e);
        };
        const Func1 w = [=](double x) { return w0 + wa * (1 - std::exp(x)); };
        const Func1 Om = [=](double x) {
            const double e = E(x);
            return OmegaM * std::exp(-3 * x) / (e * e);
        };
        return SphericalCollapseModel(E, Om, logEprime, mu, w);
    }

    // Apply f elementwise to a numpy array
    template <class F>
    py::array_t<double> map(py::array_t<double, py::array::c_style | py::array::forcecast> in, F && f) {
        py::array_t<double> out(in.request().shape);
        const double * src = in.data();
        double * dst = out.mutable_data();
        for (py::ssize_t j = 0; j < in.size(); j++)
            dst[j] = f(src[j]);
        return out;
    }
} // namespace

PYBIND11_MODULE(_halomodel, m) {
    m.doc() = "FML halo model (HaloModel/Halomodel.h): the standard halo model and HMcode-2016";

    py::class_<HaloModelState>(m, "State")
        .def_readonly("z", &HaloModelState::z)
        .def_readonly("deltac", &HaloModelState::deltac)
        .def_readonly("DeltaVir", &HaloModelState::DeltaVir)
        .def_readonly("sigma8", &HaloModelState::sigma8)
        .def_readonly("sigma8_m", &HaloModelState::sigma8_m)
        .def_readonly("sigmav", &HaloModelState::sigmav)
        .def_readonly("sigmav100", &HaloModelState::sigmav100)
        .def_readonly("rnl", &HaloModelState::rnl)
        .def_readonly("neff", &HaloModelState::neff)
        .def_readonly("kstar", &HaloModelState::kstar)
        .def_readonly("alpha", &HaloModelState::alpha)
        .def_readonly("eta", &HaloModelState::eta)
        .def_readonly("f_damp", &HaloModelState::f_damp)
        .def_readonly("concentration_A", &HaloModelState::concentration_A);

    py::class_<HaloModel>(m, "HaloModel")
        .def(py::init([](const std::string & filename_pofk,
                         double OmegaM,
                         double w0,
                         double wa,
                         double mu0,
                         double mua,
                         bool verbose,
                         bool hmcode,
                         bool sigma8norm,
                         double deltac_multiplier,
                         double DeltaVir_multiplier,
                         double concentration_multiplier) {
                 const auto model = make_model(OmegaM, w0, wa, mu0, mua);
                 // P(k, z = 0) is read as is (sigma8norm) or, for a mu != 1 model, as the LCDM spectrum that is scaled
                 // back to z = 50 with LCDM growth and forward again with the growth of the model
                 double normalization = 1.0;
                 if (not sigma8norm) {
                     const auto lcdm = make_model(OmegaM, w0, wa, 0.0, 0.0);
                     Spline D, f, Dl, fl;
                     compute_growthfactor(model, D, f);
                     compute_growthfactor(lcdm, Dl, fl);
                     const double xs = std::log(1.0 / 50.0);
                     normalization = std::pow(Dl(xs) / Dl(0.0), 2) / std::pow(D(xs) / D(0.0), 2);
                 }
                 HaloModelOptions options;
                 options.prescription = hmcode ? Prescription::HMcode2016 : Prescription::Standard;
                 options.verbose = verbose;
                 options.deltac_multiplier = deltac_multiplier;
                 options.DeltaVir_multiplier = DeltaVir_multiplier;
                 options.concentration_multiplier = concentration_multiplier;
                 return HaloModel(cosmology_from_pofk_file(model, filename_pofk, 0.0, normalization), options);
             }),
             py::arg("filename_pofk"),
             py::arg("OmegaM"),
             py::arg("w0") = -1.0,
             py::arg("wa") = 0.0,
             py::arg("mu0") = 0.0,
             py::arg("mua") = 0.0,
             py::arg("verbose") = false,
             py::arg("hmcode") = true,
             py::arg("sigma8norm") = true,
             py::arg("deltac_multiplier") = 1.0,
             py::arg("DeltaVir_multiplier") = 1.0,
             py::arg("concentration_multiplier") = 1.0)
        .def("compute_at_redshift", &HaloModel::compute_at_redshift, py::arg("z"))
        .def("computed", &HaloModel::computed)
        .def("state", &HaloModel::state, py::return_value_policy::copy)
        .def("info", &HaloModel::info)
        .def("pofk", [](const HaloModel & h, py::array_t<double> k) { return map(k, [&](double x) { return h.pofk(x); }); })
        .def("pofk_linear",
             [](const HaloModel & h, py::array_t<double> k) { return map(k, [&](double x) { return h.pofk_linear(x); }); })
        .def("pofk_onehalo",
             [](const HaloModel & h, py::array_t<double> k) { return map(k, [&](double x) { return h.pofk_onehalo(x); }); })
        .def("pofk_twohalo",
             [](const HaloModel & h, py::array_t<double> k) { return map(k, [&](double x) { return h.pofk_twohalo(x); }); })
        .def("dndlogM", [](const HaloModel & h, py::array_t<double> M) { return map(M, [&](double x) { return h.dndlogM(x); }); })
        .def("n_of_M", [](const HaloModel & h, py::array_t<double> M) { return map(M, [&](double x) { return h.n_of_M(x); }); })
        .def("bias_of_M",
             [](const HaloModel & h, py::array_t<double> M) { return map(M, [&](double x) { return h.bias_of_M(x); }); })
        .def("concentration",
             [](const HaloModel & h, py::array_t<double> M) { return map(M, [&](double x) { return h.concentration(x); }); })
        .def("deltac_of_z",
             [](const HaloModel & h, py::array_t<double> z) { return map(z, [&](double x) { return h.deltac_of_z(x); }); })
        .def("DeltaVir_of_z",
             [](const HaloModel & h, py::array_t<double> z) { return map(z, [&](double x) { return h.DeltaVir_of_z(x); }); })
        .def("output_pofk", &HaloModel::output_pofk, py::arg("filename"), py::arg("kmin") = 1e-5, py::arg("kmax") = 100.0,
             py::arg("npts_k") = 1000)
        .def("output_nofM", &HaloModel::output_nofM, py::arg("filename"), py::arg("Mmin") = 1e8, py::arg("Mmax") = 1e16,
             py::arg("npts_M") = 100)
        .def("output_deltac", &HaloModel::output_deltac, py::arg("filename"), py::arg("zmin") = 0.0,
             py::arg("zmax") = 10.0, py::arg("npts_z") = 100);
}
