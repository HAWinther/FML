// Regression coverage for FL-1/FL-2. MPI builds deliberately test rejection, not distributed inference.
#include <FML/FieldLevel/FieldLevelBAO.h>
#include <FML/tests/TestHelpers.h>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
using namespace FML::FIELDLEVEL;
using FML::TESTS::check;
static Settings settings() {
    Settings st;
    st.nmesh = 8;
    st.kmax = .015;
    return st;
}
template <class F>
static bool invalid(F && f, const std::string & text = {}) {
    try {
        f();
    } catch (const std::invalid_argument & e) {
        return text.empty() || std::string(e.what()).find(text) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}
static double max_error(const Field & a, const Field & b) {
    double error = 0;
    for (size_t i = 0; i < a.size(); ++i)
        error = std::max(error, std::abs(a[i] - b[i]));
    return error;
}
int main(int argc, char ** argv) {
#ifdef USE_MPI
#ifdef NO_AUTO_FML_SETUP
    bool rejected = false;
    try {
        FieldLevelBAO model(settings(), BAOSpectrum::eisenstein_hu());
    } catch (const std::logic_error & e) {
        rejected = std::string(e.what()).find("Initialize FML/MPI") != std::string::npos;
    }
    FML::FMLSetup::init(&argc, &argv);
    check(rejected, "manual-setup host rejects construction before MPI initialization");
#else
    FML::FMLSetup::init(&argc, &argv);
#endif
#else
    FML::FMLSetup::init(&argc, &argv);
#endif
    if (FML::NTasks > 1) {
        for (auto kind : {Model::Linear, Model::LPT1, Model::LPT}) {
            auto st = settings();
            st.model = kind;
            check(
                invalid([&] { FieldLevelBAO model(st, BAOSpectrum::eisenstein_hu()); }, "single-rank FML communicator"),
                "unsupported multi-rank constructor rejects before full-grid indexing");
        }
        // Invalid mesh must not reach FFTW planning before the communicator guard.
        auto bad = settings();
        bad.nmesh = -1;
        check(invalid([&] { FieldLevelBAO model(bad, BAOSpectrum::eisenstein_hu()); }, "single-rank FML communicator"),
              "MPI guard runs before invalid mesh allocation");
        const int original = FML::NTasks;
        FML::NTasks = 1;
        const bool actual = invalid([&] { FieldLevelBAO model(settings(), BAOSpectrum::eisenstein_hu()); },
                                    "single-rank FML communicator");
        FML::NTasks = original;
        check(actual, "guard checks actual communicator size, not cached task count");
        return FML::TESTS::summary("field-level safety (MPI rejection)");
    }
    for (auto mutate : std::vector<std::function<void(Settings &)>>{
             [](auto & s) { s.nmesh = 0; },
             [](auto & s) { s.nmesh = -1; },
             [](auto & s) { s.nmesh = std::numeric_limits<int>::max(); },
             [](auto & s) { s.nmesh_density = -1; },
             [](auto & s) { s.nmesh_density = 1; },
             [](auto & s) { s.boxsize = 0; },
             [](auto & s) { s.bias = std::numeric_limits<double>::infinity(); },
             [](auto & s) { s.kmax = NAN; }}) {
        auto st = settings();
        mutate(st);
        check(invalid([&] { FieldLevelBAO model(st, BAOSpectrum::eisenstein_hu()); }),
              "invalid settings rejected before FFT allocation/indexing");
    }
    for (auto kind : {Model::Linear, Model::LPT1, Model::LPT}) {
        auto st = settings();
        st.model = kind;
        FieldLevelBAO model(st, BAOSpectrum::eisenstein_hu());
        std::mt19937_64 rng(123);
        const auto x = model.white_noise(rng);
        const auto n = model.ncells(), nm = model.modes().size();
        check(n == 512 && !model.modes().empty(), "supported one-rank construction");
        const std::vector<double> identity(nm, 1.);
        const auto reference = model.predict(x, 1.);
        for (size_t length : {size_t(0), size_t(1), n - 1, n + 1}) {
            const Field wrong(length);
            check(invalid([&] { model.filter(wrong, identity); }, "wrong size"),
                  "filter rejects invalid field dimensions");
            check(invalid([&] { model.sampled_from_white(wrong, 1.); }, "wrong size"),
                  "white-to-sampled rejects invalid field dimensions");
            check(invalid([&] { model.white_from_sampled(wrong, 1.); }, "wrong size"),
                  "sampled-to-white rejects invalid field dimensions");
            check(invalid([&] { model.predict(wrong, 1.); }, "wrong size"),
                  "prediction rejects invalid field dimensions");
            check(invalid([&] { model.potential(wrong, 1.); }, "wrong size"),
                  "potential rejects invalid field dimensions");
            Field grad{17};
            double beta_grad = 19;
            check(invalid([&] { model.potential_and_gradient(wrong, 1., grad, beta_grad); }, "wrong size") &&
                      grad == Field{17} && beta_grad == 19,
                  "gradient rejects invalid field before modifying outputs");
        }
        for (size_t length : {size_t(0), nm - 1, nm + 1}) {
            check(invalid([&] { model.filter(x, std::vector<double>(length, 1.)); }, "one entry per mode"),
                  "filter rejects incorrect mode-vector length");
        }
        check(model.predict(x, 1.) == reference, "rejected inputs preserve subsequent valid predictions");
        check(max_error(model.filter(x, identity), x) < 2e-12, "identity filter preserves supported field");
        const auto sampled = model.sampled_from_white(x, 1.03), restored = model.white_from_sampled(sampled, 1.03);
        check(max_error(restored, x) < 2e-12, "valid white/sample conversion round trip");
        // Existing posterior/adjoint path remains valid after the negative calls.
        model.set_data(model.make_mock(x, 1., rng));
        Field grad;
        double gb = 0;
        const auto U = model.potential_and_gradient(x, 1., grad, gb);
        check(grad.size() == n && std::isfinite(U) && std::isfinite(gb),
              "valid posterior gradient survives dimension checks");
    }
    return FML::TESTS::summary("field-level safety");
}
