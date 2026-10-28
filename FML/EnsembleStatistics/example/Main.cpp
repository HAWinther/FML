#include <FML/EnsembleStatistics/ControlVariates.h>
#include <FML/RandomGenerator/CounterBasedRandom.h>
#include <iomanip>
#include <iostream>
using namespace FML::ENSEMBLE;
PhaseKey key(std::uint64_t id) {
    PhaseKey k;
    k.realization = id;
    k.seed = 87000 + id;
    return k;
}
// Toy correlated observables; replace with matched N-body and ZA/2LPT/COLA summary vectors.
std::pair<Vector, Vector> sample(std::uint64_t id) {
    FML::RANDOM::COUNTER_BASED::Stream rng(key(id).seed, id);
    double c = rng.normal(), noise = rng.normal();
    return {{5 + 1.7 * c + .1 * noise}, {c}};
}
int main() {
    Metadata metadata{"toy Gaussian model; seed-v1", "one dimensionless scalar; v1", "correlated noisy target",
                      "Gaussian surrogate"};
    PairedTraining training(1, 1, metadata);
    for (int i = 0; i < 300; ++i)
        if (i % FML::NTasks == FML::ThisTask) {
            auto [y, c] = sample(i);
            training.add(y, c, key(i), key(i));
        }
    // A true analytic zero mean exists for this toy. For simulations use an independent
    // cheap ensemble with MeanEstimate::independent and propagate its uncertainty.
    auto model = training.global().fit(MeanEstimate::exact({0}, metadata));
    ControlEvaluation evaluation(model);
    RunningCovariance raw(1);
    for (int i = 1000; i < 3000; ++i)
        if (i % FML::NTasks == FML::ThisTask) {
            auto [y, c] = sample(i);
            raw.add(y);
            evaluation.add(y, c, key(i), key(i));
        }
    auto estimate = evaluation.global().estimate();
    auto baseline = raw.global().summary();
    if (!FML::ThisTask)
        std::cout << std::setprecision(10) << "schema=" << metadata.observable_schema
                  << " metadata=" << metadata.fingerprint() << " model=" << model.fingerprint()
                  << "\ntraining=" << model.training_samples() << " evaluation=" << estimate.count
                  << " beta=" << model.coefficients()[0] << "\nraw_mean=" << baseline.mean[0]
                  << " controlled_mean=" << estimate.mean[0]
                  << " mean_error=" << std::sqrt(estimate.covariance_of_mean[0])
                  << " variance_reduction=" << baseline.covariance[0] / estimate.sample_covariance[0] << "\n";
}
