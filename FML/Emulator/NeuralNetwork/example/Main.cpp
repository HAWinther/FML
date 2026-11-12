// A neural-network emulator of a linear matter power spectrum (the BBKS fit, normalized to sigma8) in
// (Omega_m, sigma8, n_s): a Latin hypercube design, training, the accuracy on test cosmologies, the Jacobian and
// save/load. A real emulator replaces bbks_log_power by simulations or a Boltzmann code.

#include <FML/Emulator/NeuralNetwork/NeuralNetwork.h>
#include <FML/LatinHyperCube/LatinHyperCube.h>

#include <cmath>
#include <cstdio>

using namespace FML::EMULATOR::NEURALNETWORK;
using FML::SAMPLING::LatinHyperCube;
using FML::SAMPLING::LHSOptimization;

const int nk = 30;
const double kmin = 1e-3, kmax = 1.0, h = 0.7; // k in h/Mpc

double kvalue(int i) { return kmin * std::pow(kmax / kmin, double(i) / (nk - 1)); }

// BBKS transfer function with shape parameter Gamma = Omega_m h
double transfer(double k, double omega_m) {
    const double q = k / (omega_m * h);
    return std::log(1.0 + 2.34 * q) / (2.34 * q) *
           std::pow(1.0 + 3.89 * q + std::pow(16.1 * q, 2) + std::pow(5.46 * q, 3) + std::pow(6.71 * q, 4), -0.25);
}

// ln P(k) at the nk wavenumbers for (Omega_m, sigma8, n_s)
std::vector<double> bbks_log_power(const std::vector<double> & p) {
    const double omega_m = p[0], sigma8 = p[1], ns = p[2];
    auto shape = [&](double k) { return std::pow(k, ns) * std::pow(transfer(k, omega_m), 2); };
    // sigma(R = 8 Mpc/h)^2 = int dk k^2 P(k) W(kR)^2 / (2 pi^2), trapezoid in ln k
    double s2 = 0.0;
    const int n = 2000;
    const double a = std::log(1e-5), b = std::log(1e2), dx = (b - a) / n;
    for (int i = 0; i <= n; i++) {
        const double k = std::exp(a + i * dx), x = 8.0 * k;
        const double w = 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
        s2 += (i == 0 or i == n ? 0.5 : 1.0) * k * k * k * shape(k) * w * w * dx / (2.0 * M_PI * M_PI);
    }
    std::vector<double> lnp(nk);
    for (int i = 0; i < nk; i++)
        lnp[size_t(i)] = std::log(sigma8 * sigma8 / s2 * shape(kvalue(i)));
    return lnp;
}

Matrix design(size_t n, uint64_t seed) {
    LatinHyperCube lhs(3, true, 1, LHSOptimization::random_cd, seed);
    return FML::SAMPLING::scale(lhs.random(n), {0.2, 0.7, 0.9}, {0.4, 0.9, 1.0});
}

Matrix simulate(const Matrix & X) {
    Matrix Y;
    for (const auto & x : X)
        Y.push_back(bbks_log_power(x));
    return Y;
}

int main() {
    const Matrix X = design(300, 1), Xval = design(60, 2), Xtest = design(200, 3);
    const Matrix Y = simulate(X), Yval = simulate(Xval), Ytest = simulate(Xtest);

    TrainingOptions opt;
    opt.hidden_layers = {64, 64};
    opt.activation = Activation::gelu;
    opt.max_epochs = 1000;
    opt.batch_size = 16;
    opt.learning_rate = 2e-3;
    opt.restore_best_weights = true;
    opt.verbose_every = 100;
    TrainingHistory history;
    const NeuralNetwork net = train(X, Y, Xval, Yval, opt, &history);
    std::printf("\n%d epochs (%s), best validation loss %.3e at epoch %d, SWA %s\n", history.epochs_run,
                history.stopped_early ? "stopped early" : "max_epochs", history.best_val_loss, history.best_epoch,
                history.swa_applied ? "applied" : "not applied");

    // Relative error of P(k) = exp(ln P) on the test cosmologies
    const Matrix P = net.evaluate(Xtest);
    double max_err = 0.0, sum_err = 0.0;
    for (size_t i = 0; i < Xtest.size(); i++)
        for (int j = 0; j < nk; j++) {
            const double e = std::abs(std::exp(P[i][size_t(j)] - Ytest[i][size_t(j)]) - 1.0);
            max_err = std::max(max_err, e);
            sum_err += e / double(Xtest.size() * nk);
        }
    std::printf("Test set (%zu cosmologies): mean |dP/P| = %.2e, max = %.2e\n", Xtest.size(), sum_err, max_err);

    // The Jacobian: d ln P / d sigma8 is 2 / sigma8 at every k
    const std::vector<double> fiducial{0.3, 0.8, 0.96};
    const Matrix J = net.jacobian(fiducial);
    std::printf("d lnP / d sigma8 at the fiducial cosmology: %.3f (k = %.3g), %.3f (k = %.3g); exact %.3f\n",
                J[5][1], kvalue(5), J[25][1], kvalue(25), 2.0 / 0.8);

    net.save("emulator_bbks.txt");
    const NeuralNetwork loaded = NeuralNetwork::load("emulator_bbks.txt");
    std::printf("Saved to emulator_bbks.txt and read back: %s\n",
                loaded(fiducial) == net(fiducial) ? "the same predictions" : "DIFFERENT predictions");
}
