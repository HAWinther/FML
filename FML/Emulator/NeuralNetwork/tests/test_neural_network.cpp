// =============================================================================
// Tests of the neural-network emulator (NeuralNetwork.h). Run with `make run` in this folder.
//
//   * Against PyTorch (TorchReference.h, from make_torch_reference.py): outputs, losses and gradients for every
//     activation and loss, and the parameters after 3 clipped Adam steps with weight decay; ReduceLROnPlateau
//   * A Lightning checkpoint of an Emulator.py network exported with export_lightning_emulator.py: outputs equal
//     torch's
//   * Gradients and the Jacobian against finite differences (with input/output scaling)
//   * save/load, early stopping, SWA, the CSV reader, errors
//   * Training: fits a smooth 2D -> 2D function on a Latin hypercube design; the same seed gives the same network
// =============================================================================

#include "TorchReference.h"
#include <FML/Emulator/NeuralNetwork/NeuralNetwork.h>
#include <FML/LatinHyperCube/LatinHyperCube.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <cstdio>
#include <fstream>

using namespace FML::EMULATOR::NEURALNETWORK;
using FML::TESTS::check;
using FML::TESTS::num;
using FML::TESTS::throws;

static Matrix rows(const std::vector<double> & flat, size_t ncols) {
    Matrix m(flat.size() / ncols);
    for (size_t i = 0; i < m.size(); i++)
        m[i].assign(flat.begin() + long(i * ncols), flat.begin() + long((i + 1) * ncols));
    return m;
}

static double max_abs_diff(const std::vector<double> & a, const std::vector<double> & b) {
    if (a.size() != b.size())
        return INFINITY;
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

static std::vector<size_t> all_indices(size_t n) {
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; i++)
        idx[i] = i;
    return idx;
}

// The test function: y = (sin(2 x0) x1, exp(-x0^2) + 0.3 x1^2), x in [-1, 1]^2
static Matrix target(const Matrix & X) {
    Matrix Y;
    for (const auto & x : X)
        Y.push_back({std::sin(2.0 * x[0]) * x[1], std::exp(-x[0] * x[0]) + 0.3 * x[1] * x[1]});
    return Y;
}

static Matrix design(size_t n, uint64_t seed) {
    FML::SAMPLING::LatinHyperCube lhs(2, true, 1, FML::SAMPLING::LHSOptimization::none, seed);
    return FML::SAMPLING::scale(lhs.random(n), {-1.0, -1.0}, {1.0, 1.0});
}

int main() {
    //-------------------------------------------------------------------------
    // PyTorch: forward, loss, gradient, clipped Adam with weight decay
    //-------------------------------------------------------------------------
    for (const auto & c : TorchReference::cases) {
        NeuralNetwork net(TorchReference::sizes, activation_from_name(c.activation));
        net.parameters() = c.params;
        const Matrix X = rows(c.X, size_t(TorchReference::sizes.front()));
        const Matrix Y = rows(c.Y, size_t(TorchReference::sizes.back()));
        const std::string tag = c.activation + "/" + c.loss;

        std::vector<double> out;
        for (const auto & y : net.evaluate(X))
            out.insert(out.end(), y.begin(), y.end());
        std::vector<double> grad;
        const Loss loss = loss_from_name(c.loss);
        const double value = batch_loss_and_gradient(net, X, Y, all_indices(X.size()), 0, X.size(), loss, grad);
        const double e_out = max_abs_diff(out, c.output), e_grad = max_abs_diff(grad, c.grad);
        check(e_out < 1e-13 and std::abs(value - c.loss_value) < 1e-13 and e_grad < 1e-13,
              "torch " + tag + ": output, loss, gradient (errors " + num(e_out, 2) + ", " +
                  num(std::abs(value - c.loss_value), 2) + ", " + num(e_grad, 2) + ")");

        Adam adam(net.num_parameters(), TorchReference::lr, TorchReference::weight_decay);
        for (int s = 0; s < TorchReference::adam_steps; s++) {
            batch_loss_and_gradient(net, X, Y, all_indices(X.size()), 0, X.size(), loss, grad);
            clip_gradient_norm(grad, TorchReference::clip);
            adam.step(net.parameters(), grad);
        }
        const double e_adam = max_abs_diff(net.parameters(), c.params_after);
        check(e_adam < 1e-12, "torch " + tag + ": parameters after 3 clipped Adam steps (error " + num(e_adam, 2) + ")");
    }

    {
        ReduceLROnPlateau plateau(0.1, 0.5, 2, 0.02);
        bool same = true;
        for (size_t i = 0; i < TorchReference::plateau_metrics.size(); i++)
            same = same and std::abs(plateau.step(TorchReference::plateau_metrics[i]) - TorchReference::plateau_lrs[i]) <
                                1e-15;
        check(same, "ReduceLROnPlateau: the same learning rates as torch");
    }

    //-------------------------------------------------------------------------
    // An exported Lightning checkpoint (Emulator.py structure, gelu)
    //-------------------------------------------------------------------------
    {
        const NeuralNetwork net = NeuralNetwork::load("data/lightning_emulator.txt");
        std::ifstream in("data/lightning_emulator_expected.txt");
        std::string line;
        double worst = 0.0;
        int npoints = 0;
        while (std::getline(in, line)) {
            if (line.empty() or line[0] == '#')
                continue;
            std::stringstream ss(line);
            std::vector<double> x(4), y(3);
            for (auto & v : x)
                ss >> v;
            for (auto & v : y)
                ss >> v;
            worst = std::max(worst, max_abs_diff(net(x), y));
            npoints++;
        }
        check(npoints == 5 and net.get_activation() == Activation::gelu and net.get_layer_sizes() ==
                                                                               std::vector<int>{4, 16, 16, 3},
              "exported Lightning network: sizes and activation");
        check(worst < 1e-13, "exported Lightning network: outputs equal torch's (error " + num(worst, 2) + ")");
    }

    //-------------------------------------------------------------------------
    // Finite differences: the parameter gradient (all activations and losses) and the Jacobian (with scaling)
    //-------------------------------------------------------------------------
    for (Activation a : {Activation::identity, Activation::relu, Activation::tanh, Activation::sigmoid,
                         Activation::gelu, Activation::silu, Activation::elu, Activation::leaky_relu,
                         Activation::softplus}) {
        NeuralNetwork net({3, 6, 5, 2}, a);
        net.init_glorot_uniform(11);
        for (size_t l = 0; l < net.num_layers(); l++)
            for (int o = 0; o < net.get_layer_sizes()[l + 1]; o++)
                net.biases(l)[o] = 0.1 * (o + 1) - 0.25;
        net.set_input_scaling({0.5, -1.0, 2.0}, {2.0, 0.5, 3.0});
        net.set_output_scaling({1.0, -2.0}, {0.3, 4.0});
        const Matrix Z{{0.3, -0.7, 1.1}, {-1.2, 0.4, 0.05}, {0.9, 0.8, -0.6}};
        const Matrix T{{0.2, -0.1}, {1.0, 0.5}, {-0.4, 0.3}};
        double worst = 0.0;
        for (Loss loss : {Loss::mse, Loss::l1, Loss::huber}) {
            std::vector<double> grad, g2;
            batch_loss_and_gradient(net, Z, T, all_indices(3), 0, 3, loss, grad);
            for (size_t k = 0; k < net.num_parameters(); k++) {
                const double p0 = net.parameters()[k], h = 1e-6;
                net.parameters()[k] = p0 + h;
                const double lp = batch_loss_and_gradient(net, Z, T, all_indices(3), 0, 3, loss, g2);
                net.parameters()[k] = p0 - h;
                const double lm = batch_loss_and_gradient(net, Z, T, all_indices(3), 0, 3, loss, g2);
                net.parameters()[k] = p0;
                worst = std::max(worst, std::abs((lp - lm) / (2 * h) - grad[k]));
            }
        }
        const std::vector<double> x{0.7, -0.6, 3.0};
        const Matrix J = net.jacobian(x);
        double worst_j = 0.0;
        for (int j = 0; j < 3; j++) {
            auto xp = x, xm = x;
            xp[size_t(j)] += 1e-6;
            xm[size_t(j)] -= 1e-6;
            const auto yp = net(xp), ym = net(xm);
            for (int k = 0; k < 2; k++)
                worst_j = std::max(worst_j, std::abs((yp[size_t(k)] - ym[size_t(k)]) / 2e-6 - J[size_t(k)][size_t(j)]));
        }
        check(worst < 1e-7 and worst_j < 1e-6,
              "finite differences " + activation_name(a) + ": gradient " + num(worst, 2) + ", Jacobian " +
                  num(worst_j, 2));
    }

    //-------------------------------------------------------------------------
    // save / load
    //-------------------------------------------------------------------------
    {
        NeuralNetwork net({2, 7, 3}, Activation::silu);
        net.init_normal(0.8, 5);
        net.set_input_scaling({1.0, 2.0}, {0.1, 3.0});
        net.set_output_scaling({-1.0, 0.0, 5.0}, {2.0, 1e-3, 7.0});
        net.save("test_nn_roundtrip.txt");
        const NeuralNetwork back = NeuralNetwork::load("test_nn_roundtrip.txt");
        std::remove("test_nn_roundtrip.txt");
        check(back.parameters() == net.parameters() and back.get_output_std() == net.get_output_std() and
                  back.get_input_mean() == net.get_input_mean() and back(std::vector<double>{0.3, 0.4}) ==
                                                                        net(std::vector<double>{0.3, 0.4}),
              "save/load gives the same network (bitwise)");
        check(throws<std::runtime_error>([] { NeuralNetwork::load("no_such_file.txt"); }), "load: missing file");
    }

    //-------------------------------------------------------------------------
    // Early stopping, SWA
    //-------------------------------------------------------------------------
    {
        EarlyStopping es(3);
        const std::vector<double> m{1.0, 0.5, 0.6, 0.5, 0.4, 0.45, 0.4, 0.41, 0.3};
        int stopped_at = -1;
        for (size_t i = 0; i < m.size() and stopped_at < 0; i++)
            if (es.step(m[i]))
                stopped_at = int(i);
        check(stopped_at == 7, "early stopping after patience checks without improvement (at " +
                                   std::to_string(stopped_at) + ")");
        check(EarlyStopping(5).step(NAN), "early stopping on a non-finite loss");

        const Matrix X = design(60, 1), Y = target(X), Xv = design(20, 2), Yv = target(Xv);
        TrainingOptions opt;
        opt.hidden_layers = {8};
        opt.max_epochs = 40;
        opt.swa_epoch_start = 0.5;
        opt.swa_annealing_epochs = 5;
        opt.early_stopping_patience = 0;
        TrainingHistory h;
        train(X, Y, Xv, Yv, opt, &h);
        check(h.swa_applied and h.epochs_run == 40 and std::abs(h.learning_rate[24] - opt.swa_lr) < 1e-15 and
                  h.learning_rate[19] < opt.swa_lr,
              "SWA: averaged weights used, learning rate annealed to swa_lr");

        opt.early_stopping_patience = 1;
        opt.learning_rate = 0.5; // diverges / stalls: stops early
        train(X, Y, Xv, Yv, opt, &h);
        check(h.stopped_early and not h.swa_applied, "stopped early: SWA not applied (as Lightning)");
    }

    //-------------------------------------------------------------------------
    // Training
    //-------------------------------------------------------------------------
    {
        const Matrix X = design(400, 10), Y = target(X), Xv = design(100, 20), Yv = target(Xv);
        const Matrix Xt = design(200, 30), Yt = target(Xt);
        TrainingOptions opt;
        opt.hidden_layers = {32, 32};
        opt.max_epochs = 300;
        opt.batch_size = 16;
        opt.learning_rate = 3e-3;
        opt.restore_best_weights = true;
        TrainingHistory h;
        const NeuralNetwork net = train(X, Y, Xv, Yv, opt, &h);
        const Matrix P = net.evaluate(Xt);
        double worst = 0.0;
        for (int k = 0; k < 2; k++) {
            double se = 0.0, var = 0.0, mean = 0.0;
            for (const auto & y : Yt)
                mean += y[size_t(k)] / double(Yt.size());
            for (size_t i = 0; i < Yt.size(); i++) {
                se += std::pow(P[i][size_t(k)] - Yt[i][size_t(k)], 2);
                var += std::pow(Yt[i][size_t(k)] - mean, 2);
            }
            worst = std::max(worst, std::sqrt(se / var));
        }
        check(worst < 0.02, "training: test RMS error / std < 2% (" + num(worst, 3) + ", " +
                                std::to_string(h.epochs_run) + " epochs, best val loss " + num(h.best_val_loss, 3) +
                                " at epoch " + std::to_string(h.best_epoch) + ")");

        const NeuralNetwork again = train(X, Y, Xv, Yv, opt);
#ifndef USE_OMP
        check(again.parameters() == net.parameters(), "training: the same seed gives the same network");
#endif
        // Emulator.py's initialization N(0, 1e-3^2) starts on a plateau: with its learning-rate reductions and
        // early stopping the training ends before the loss drops (as in PyTorch)
        TrainingOptions py = opt;
        py.init_std = 1e-3;
        TrainingHistory hp;
        train(X, Y, Xv, Yv, py, &hp);
        check(hp.stopped_early and hp.best_val_loss > 0.5, "init_std = 1e-3: stuck on the initial plateau (val loss " +
                                                               num(hp.best_val_loss, 3) + ")");
        check(throws<std::invalid_argument>([&] { train(X, Y, Matrix{}, Matrix{}, opt); }),
              "training: an empty validation set is rejected");
    }

    //-------------------------------------------------------------------------
    // CSV, errors
    //-------------------------------------------------------------------------
    {
        {
            std::ofstream csv("test_nn.csv");
            csv << ",\"omega_m\", sigma8,h,P0\n0,0.3,0.8,0.7,100.5\n1,0.25,0.9,0.65,-3e2\n\n";
        }
        const Matrix m = read_csv_columns("test_nn.csv", {"P0", "omega_m"});
        std::remove("test_nn.csv");
        check(m == Matrix{{100.5, 0.3}, {-300.0, 0.25}}, "read_csv_columns: selected columns in the order asked");
        check(throws<std::invalid_argument>([] { activation_from_name("swoosh"); }), "unknown activation rejected");
        NeuralNetwork net({2, 3, 1});
        check(throws<std::invalid_argument>([&] { net(std::vector<double>{1.0}); }), "wrong input size rejected");
        check(throws<std::invalid_argument>([&] { net.set_input_scaling({0, 0}, {1, 0}); }), "std 0 rejected");
    }

    return FML::TESTS::summary("NeuralNetwork");
}
