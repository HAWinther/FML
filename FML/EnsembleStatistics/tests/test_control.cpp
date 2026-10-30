#include <FML/EnsembleStatistics/ControlVariates.h>
#include <FML/RandomGenerator/CounterBasedRandom.h>
#include <iostream>
using namespace FML::ENSEMBLE;
int failures = 0;
void check(bool ok, const std::string & label) {
    int bad = !ok;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (!FML::ThisTask)
        std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
}
bool near(double a, double b, double tolerance = 1e-10) {
    return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}
Metadata meta() { return {"test-cosmology", "two observables, physical units; schema-v1", "truth", "ZA-control"}; }
PhaseKey key(std::uint64_t id) {
    PhaseKey k;
    k.realization = id;
    k.seed = 100000 + id;
    return k;
}
Vector target(const Vector & c) { return {10 + 2 * c[0] - 3 * c[1], -4 + .5 * c[0] + 4 * c[1]}; }
PairedTraining training() {
    PairedTraining t(2, 2, meta());
    for (int i = 0; i < 12; ++i)
        if (i % FML::NTasks == FML::ThisTask) {
            Vector c{double(i % 4) * 2 - 3, double(i / 4) * 2 - 2};
            t.add(target(c), c, key(i), key(i));
        }
    return t.global();
}
void test_moments() {
    RunningCovariance a(2), b(2);
    const std::vector<Vector> rows{{1, 2}, {3, 6}, {5, 4}};
    a.add(rows[0]);
    for (size_t i = 1; i < rows.size(); ++i)
        b.add(rows[i]);
    a.merge(b);
    auto s = a.summary();
    check(s.count == 3 && s.mean == Vector({3, 4}) && near(s.covariance[0], 4) && near(s.covariance[1], 2) &&
              near(s.covariance[3], 4),
          "streaming covariance and Chan merge reproduce hand-computed sample covariance");
    RunningCovariance local(2);
    for (int i = 0; i < 37; ++i)
        if (i % FML::NTasks == FML::ThisTask)
            local.add({1e12 + .125 * i, -2e12 + .25 * i});
    auto global = local.global().summary();
    const double variance = .125 * .125 * 37 * 38 / 12;
    check(global.count == 37 && near(global.mean[0], 1e12 + 2.25) && near(global.covariance[0], variance, 2e-6) &&
              near(global.covariance[1], 2 * variance, 2e-6),
          "global merge handles large offsets, unequal sample counts and empty ranks");
    RunningCovariance root(1);
    if (!FML::ThisTask) {
        root.add({2});
        root.add({4});
    }
    auto merged = root.global().summary();
    check(merged.count == 2 && merged.mean[0] == 3 && merged.covariance[0] == 2,
          "root-only sample input reduces correctly across empty ranks");
    RunningCovariance empty(2);
    auto e = empty.global().summary();
    check(e.count == 0 && std::isnan(e.mean[0]) && std::isnan(e.covariance[0]),
          "empty global ensemble is explicitly undefined");
    empty.add({1, 2});
    check(std::isnan(empty.summary().covariance[0]), "one sample does not imply zero covariance");
    a.reset();
    check(a.count() == 0, "running moments can release accumulated state without changing dimensions");
}
void test_fit() {
    auto t = training();
    FitOptions options;
    options.ridge = 0;
    auto model = t.fit(MeanEstimate::exact({.25, -.5}, meta()), options);
    const auto & b = model.coefficients();
    check(near(b[0], 2) && near(b[1], -3) && near(b[2], .5) && near(b[3], 4),
          "full standardized multivariate fit recovers exact known control coefficients");
    ControlEvaluation e(model);
    for (int i = 0; i < 17; ++i)
        if (i % FML::NTasks == FML::ThisTask) {
            Vector c{.3 * i - 2, .1 * i + 1};
            e.add(target(c), c, key(1000 + i), key(1000 + i));
        }
    auto result = e.global().estimate();
    check(result.count == 17 && near(result.mean[0], 12) && near(result.mean[1], -5.875) &&
              std::abs(result.sample_covariance[0]) < 1e-24,
          "held-out affine observables collapse to the analytically controlled mean");
    options.mode = FitMode::Diagonal;
    auto diagonal = t.fit(MeanEstimate::exact({0, 0}, meta()), options);
    check(near(diagonal.coefficients()[0], 2) && diagonal.coefficients()[1] == 0 && diagonal.coefficients()[2] == 0 &&
              near(diagonal.coefficients()[3], 4),
          "diagonal per-bin control variates keep cross-bin coefficients zero");
    PairedTraining singular(1, 2, meta());
    for (int i = 0; i < 6; ++i) {
        Vector c{double(i), double(i)};
        singular.add({3.0 * i}, c, key(100 + i), key(100 + i));
    }
    options.mode = FitMode::Full;
    bool failed = false;
    try {
        singular.fit(MeanEstimate::exact({0, 0}, meta()), options);
    } catch (const std::domain_error &) {
        failed = true;
    }
    check(failed, "singular unregularized control covariance fails explicitly");
    options.ridge = 1e-6;
    auto regular = singular.fit(MeanEstimate::exact({0, 0}, meta()), options);
    check(near(regular.coefficients()[0] + regular.coefficients()[1], 3, 1e-5),
          "standardized ridge handles collinear control bins");
    PairedTraining zero(1, 2, meta());
    for (int i = 0; i < 8; ++i)
        zero.add({2.0 * i}, {double(i), 7}, key(200 + i), key(200 + i));
    auto z = zero.fit(MeanEstimate::exact({0, 7}, meta()));
    check(near(z.coefficients()[0], 2, 1e-6) && z.coefficients()[1] == 0, "constant controls are ignored safely");
}
void test_mean_uncertainty() {
    const std::vector<Vector> samples{{-1, 2}, {1, 4}, {0, 3}, {2, 1}};
    RunningCovariance local(2);
    std::vector<PhaseKey> keys;
    for (size_t i = 0; i < samples.size(); ++i) {
        keys.push_back(key(2000 + i));
        if (i % FML::NTasks == size_t(FML::ThisTask))
            local.add(samples[i]);
    }
    auto population = local.global();
    auto mean = MeanEstimate::independent(population, keys, meta());
    FitOptions opt;
    opt.ridge = 0;
    auto model = training().fit(mean, opt);
    const auto s = population.summary();
    const auto & b = model.coefficients();
    Vector expected(4);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            for (size_t a = 0; a < 2; ++a)
                for (size_t c = 0; c < 2; ++c)
                    expected[2 * i + j] += b[2 * i + a] * s.covariance[2 * a + c] / samples.size() * b[2 * j + c];
    ControlEvaluation few(model), many(model);
    for (int i = 0; i < 100; ++i)
        if (i % FML::NTasks == FML::ThisTask) {
            Vector c{.25 * i, .4 * i};
            many.add(target(c), c, key(3000 + i), key(3000 + i));
            if (i < 4)
                few.add(target(c), c, key(3000 + i), key(3000 + i));
        }
    auto f = few.global().estimate(), m = many.global().estimate();
    bool floor = true;
    for (size_t i = 0; i < 4; ++i)
        floor &= near(f.covariance_of_mean[i], expected[i]) && near(m.covariance_of_mean[i], expected[i]);
    const auto controlled = target(mean.mean());
    check(floor && near(m.mean[0], controlled[0]) && near(m.mean[1], controlled[1]),
          "independent surrogate-mean uncertainty is propagated and does not shrink with evaluation count");
}
void test_variance_reduction() {
    auto m = meta();
    m.observable_schema = "scalar observable";
    PairedTraining t(1, 1, m);
    auto sample = [](std::uint64_t id) {
        FML::RANDOM::COUNTER_BASED::Stream c(84, id, 1), noise(84, id, 2);
        const double control = c.normal();
        return std::pair<double, double>{5 + 1.7 * control + .1 * noise.normal(), control};
    };
    for (int i = 0; i < 300; ++i)
        if (i % FML::NTasks == FML::ThisTask) {
            const auto p = sample(i);
            t.add({p.first}, {p.second}, key(i), key(i));
        }
    auto model = t.global().fit(MeanEstimate::exact({0}, m));
    ControlEvaluation controlled(model);
    RunningCovariance raw(1);
    for (int i = 10000; i < 12000; ++i)
        if (i % FML::NTasks == FML::ThisTask) {
            const auto p = sample(i);
            raw.add({p.first});
            controlled.add({p.first}, {p.second}, key(i), key(i));
        }
    const auto result = controlled.global().estimate();
    const auto original = raw.global().summary();
    check(result.count == 2000 && std::abs(result.mean[0] - 5) < 8 * std::sqrt(result.covariance_of_mean[0]) &&
              original.covariance[0] > 100 * result.sample_covariance[0],
          "held-out Gaussian ensemble retains the target mean while reducing variance by more than 100x");
}
void test_validation() {
    auto rejects = [](auto call) {
        try {
            call();
        } catch (const std::exception &) {
            return true;
        }
        return false;
    };
    RunningCovariance c(2);
    c.add({1, 2});
    check(rejects([&] { c.add({NAN, 2}); }) && c.count() == 1 && rejects([&] { c.add({1}); }),
          "invalid moment rows do not change accumulated state");
    PairedTraining t(2, 2, meta());
    auto a = key(1), b = a;
    b.seed++;
    check(rejects([&] { t.add({1, 2}, {3, 4}, a, b); }), "mismatched target/control phases are rejected");
    b = a;
    b.generator = PhaseGenerator::NGenIC;
    check(rejects([&] { t.add({1, 2}, {3, 4}, a, b); }), "mismatched phase-generator backends are rejected");
    t.add({1, 2}, {3, 4}, a, a);
    b = a;
    b.realization = 99;
    b.phase_sign = -1;
    check(rejects([&] { t.add({1, 2}, {3, 4}, b, b); }),
          "antithetic members cannot be counted as independent phase blocks");
    auto trained = training();
    auto model = trained.fit(MeanEstimate::exact({0, 0}, meta()));
    check(rejects([&] { model.apply({1, 2}, {3, 4}, key(0), key(0)); }), "evaluation cannot reuse training phases");
    RunningCovariance cheap(2);
    cheap.add({1, 2});
    cheap.add({2, 3});
    check(rejects([&] { trained.fit(MeanEstimate::independent(cheap, {key(0), key(100)}, meta())); }),
          "estimated surrogate mean must be independent of training");
    auto estimated = trained.fit(MeanEstimate::independent(cheap, {key(100), key(101)}, meta()));
    check(rejects([&] { estimated.apply({1, 2}, {3, 4}, key(100), key(100)); }),
          "evaluation cannot reuse surrogate-mean phases");
    ControlEvaluation eval(model);
    eval.add({1, 2}, {3, 4}, key(10000), key(10000));
    check(rejects([&] { eval.add({1, 2}, {3, 4}, key(10000), key(10000)); }), "duplicate evaluation rows are rejected");
    if (FML::NTasks > 1) {
        PairedTraining dup(1, 1, meta());
        dup.add({1}, {1}, key(100), key(100));
        check(rejects([&] { dup.global(); }), "MPI training rejects phase blocks repeated on different ranks");
        auto different = meta();
        different.observable_schema += std::to_string(FML::ThisTask);
        PairedTraining schema(1, 1, different);
        check(rejects([&] { schema.global(); }), "MPI model schemas are checked before data reduction");
    }
    auto paired = meta();
    paired.phase_pair_averaged = true;
    PairedTraining blocked(1, 1, paired);
    auto k = key(99);
    k.phase_sign = 0;
    blocked.add({2}, {1}, k, k);
    check(blocked.summary().count == 1, "already averaged antithetic pairs are explicitly represented as one sample");
}
int main() {
    test_moments();
    test_fit();
    test_mean_uncertainty();
    test_variance_reduction();
    test_validation();
    if (!FML::ThisTask)
        std::cout << (failures ? "Ensemble tests FAILED\n" : "All ensemble tests passed\n");
    return failures ? 1 : 0;
}
