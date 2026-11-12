#ifndef FML_EMULATOR_NEURALNETWORK_HEADER
#define FML_EMULATOR_NEURALNETWORK_HEADER

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef USE_OMP
#include <omp.h>
#endif

namespace FML {
    namespace EMULATOR {

        //=============================================================================================================
        /// A fully connected neural network (multilayer perceptron) for emulators: evaluation, training and a text
        /// file format. Header only, no dependencies (OpenMP if USE_OMP). See README.md.
        ///
        /// The network: input x -> standardized z = (x - input_mean) / input_std -> hidden layers
        /// a = activation(W a + b) -> a linear output layer -> y = output * output_std + output_mean.
        ///
        /// The training follows the PyTorch Lightning emulator of the Sesame pipeline (Emulator.py): Adam with
        /// (L2) weight decay, gradient-norm clipping, ReduceLROnPlateau on the validation loss, early stopping and
        /// stochastic weight averaging, with the same defaults. Two defaults differ: inputs and outputs are
        /// standardized, and the weights start Glorot uniform (Emulator.py: no scaling, N(0, 1e-3^2); set
        /// standardize_inputs = standardize_outputs = false and init_std = 1e-3 for that setup).
        ///
        ///   using namespace FML::EMULATOR::NEURALNETWORK;
        ///   TrainingOptions opt;
        ///   opt.hidden_layers = {64, 64};
        ///   TrainingHistory history;
        ///   NeuralNetwork net = train(X_train, Y_train, X_val, Y_val, opt, &history);
        ///   std::vector<double> y = net(x);
        ///   net.save("emulator.txt");
        ///   auto net2 = NeuralNetwork::load("emulator.txt");      // also files from export_lightning_emulator.py
        //=============================================================================================================
        namespace NEURALNETWORK {

            /// Points: X[i] is point i, X[i][j] its component j
            using Matrix = std::vector<std::vector<double>>;

            /// The activations of the hidden layers (as torch.nn.functional; leaky_relu has slope 0.01, elu
            /// alpha = 1, softplus beta = 1 with PyTorch's threshold 20, gelu is the exact (erf) version)
            enum class Activation { identity, relu, tanh, sigmoid, gelu, silu, elu, leaky_relu, softplus };

            /// The losses (means over all the points and outputs, as torch.nn.MSELoss, L1Loss and HuberLoss with
            /// delta = 1, which equals SmoothL1Loss with beta = 1)
            enum class Loss { mse, l1, huber };

            namespace detail {
                inline std::string lower(std::string s) {
                    std::string r;
                    for (char c : s)
                        if (not std::isspace((unsigned char)c))
                            r += char(std::tolower((unsigned char)c));
                    return r;
                }

                inline double sigmoid(double z) {
                    if (z >= 0.0)
                        return 1.0 / (1.0 + std::exp(-z));
                    const double e = std::exp(z);
                    return e / (1.0 + e);
                }

                inline double activate(Activation a, double z) {
                    switch (a) {
                        case Activation::identity:
                            return z;
                        case Activation::relu:
                            return z > 0.0 ? z : 0.0;
                        case Activation::tanh:
                            return std::tanh(z);
                        case Activation::sigmoid:
                            return sigmoid(z);
                        case Activation::gelu:
                            return 0.5 * z * (1.0 + std::erf(z / std::sqrt(2.0)));
                        case Activation::silu:
                            return z * sigmoid(z);
                        case Activation::elu:
                            return z > 0.0 ? z : std::expm1(z);
                        case Activation::leaky_relu:
                            return z > 0.0 ? z : 0.01 * z;
                        case Activation::softplus:
                            return z > 20.0 ? z : std::log1p(std::exp(z));
                    }
                    return z;
                }

                /// The derivative (at the kinks as PyTorch's backward: relu' = 0 and leaky_relu' = 0.01 at 0)
                inline double activate_derivative(Activation a, double z) {
                    switch (a) {
                        case Activation::identity:
                            return 1.0;
                        case Activation::relu:
                            return z > 0.0 ? 1.0 : 0.0;
                        case Activation::tanh: {
                            const double t = std::tanh(z);
                            return 1.0 - t * t;
                        }
                        case Activation::sigmoid: {
                            const double s = sigmoid(z);
                            return s * (1.0 - s);
                        }
                        case Activation::gelu:
                            return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0))) +
                                   z * std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
                        case Activation::silu: {
                            const double s = sigmoid(z);
                            return s * (1.0 + z * (1.0 - s));
                        }
                        case Activation::elu:
                            return z > 0.0 ? 1.0 : std::exp(z);
                        case Activation::leaky_relu:
                            return z > 0.0 ? 1.0 : 0.01;
                        case Activation::softplus:
                            return z > 20.0 ? 1.0 : sigmoid(z);
                    }
                    return 1.0;
                }

                /// The loss of one residual r = prediction - target and its derivative
                inline double loss_value(Loss l, double r) {
                    const double ar = std::abs(r);
                    switch (l) {
                        case Loss::mse:
                            return r * r;
                        case Loss::l1:
                            return ar;
                        case Loss::huber:
                            return ar < 1.0 ? 0.5 * r * r : ar - 0.5;
                    }
                    return r * r;
                }
                inline double loss_derivative(Loss l, double r) {
                    const double sign = r > 0.0 ? 1.0 : (r < 0.0 ? -1.0 : 0.0);
                    switch (l) {
                        case Loss::mse:
                            return 2.0 * r;
                        case Loss::l1:
                            return sign;
                        case Loss::huber:
                            return std::abs(r) < 1.0 ? r : sign;
                    }
                    return 2.0 * r;
                }

                /// mt19937_64 with portable conversions (the same numbers with every standard library)
                class RNG {
                    std::mt19937_64 gen;
                    bool have_spare{false};
                    double spare{0.0};

                  public:
                    explicit RNG(uint64_t seed = 0) : gen(seed) {}
                    double uniform() { return double(gen() >> 11) * 0x1.0p-53; }
                    double normal() {
                        if (have_spare) {
                            have_spare = false;
                            return spare;
                        }
                        const double u1 = 1.0 - uniform(); // (0, 1]
                        const double u2 = uniform();
                        const double r = std::sqrt(-2.0 * std::log(u1));
                        spare = r * std::sin(2.0 * M_PI * u2);
                        have_spare = true;
                        return r * std::cos(2.0 * M_PI * u2);
                    }
                    uint64_t below(uint64_t range) {
                        const uint64_t threshold = (0 - range) % range;
                        while (true) {
                            const uint64_t r = gen();
                            if (r >= threshold)
                                return r % range;
                        }
                    }
                    template <class T>
                    void shuffle(std::vector<T> & v) {
                        for (size_t i = v.size(); i > 1; --i)
                            std::swap(v[i - 1], v[size_t(below(i))]);
                    }
                };

                inline void check_matrix(const Matrix & m, size_t ncols, const std::string & what) {
                    for (const auto & row : m)
                        if (row.size() != ncols)
                            throw std::invalid_argument("NeuralNetwork: " + what + " has rows of the wrong length");
                }
            } // namespace detail

            /// Activation from its name: the torch.nn.functional names (relu, tanh, sigmoid, gelu, silu, elu,
            /// leaky_relu, softplus) and identity (also linear, none, swish = silu). Throws for other names
            inline Activation activation_from_name(const std::string & name) {
                const std::string n = detail::lower(name);
                if (n == "identity" or n == "linear" or n == "none")
                    return Activation::identity;
                if (n == "relu")
                    return Activation::relu;
                if (n == "tanh")
                    return Activation::tanh;
                if (n == "sigmoid")
                    return Activation::sigmoid;
                if (n == "gelu")
                    return Activation::gelu;
                if (n == "silu" or n == "swish")
                    return Activation::silu;
                if (n == "elu")
                    return Activation::elu;
                if (n == "leaky_relu" or n == "leakyrelu")
                    return Activation::leaky_relu;
                if (n == "softplus")
                    return Activation::softplus;
                throw std::invalid_argument("NeuralNetwork: unknown activation '" + name + "'");
            }

            inline std::string activation_name(Activation a) {
                switch (a) {
                    case Activation::identity:
                        return "identity";
                    case Activation::relu:
                        return "relu";
                    case Activation::tanh:
                        return "tanh";
                    case Activation::sigmoid:
                        return "sigmoid";
                    case Activation::gelu:
                        return "gelu";
                    case Activation::silu:
                        return "silu";
                    case Activation::elu:
                        return "elu";
                    case Activation::leaky_relu:
                        return "leaky_relu";
                    case Activation::softplus:
                        return "softplus";
                }
                return "identity";
            }

            /// Loss from its name: the torch.nn names MSELoss, L1Loss, HuberLoss, SmoothL1Loss, or mse, l1, mae,
            /// huber
            inline Loss loss_from_name(const std::string & name) {
                const std::string n = detail::lower(name);
                if (n == "mse" or n == "mseloss")
                    return Loss::mse;
                if (n == "l1" or n == "l1loss" or n == "mae")
                    return Loss::l1;
                if (n == "huber" or n == "huberloss" or n == "smoothl1loss" or n == "smooth_l1")
                    return Loss::huber;
                throw std::invalid_argument("NeuralNetwork: unknown loss '" + name + "'");
            }

            //=========================================================================================================
            /// The network. The parameters are stored in one array: for each layer l the weights (sizes[l+1] rows of
            /// sizes[l], row-major, as torch.nn.Linear.weight) and then the biases
            //=========================================================================================================
            class NeuralNetwork {
              public:
                /// Work arrays for one evaluation (one per thread)
                struct Workspace {
                    std::vector<std::vector<double>> pre; // pre-activations of layer l (size sizes[l+1])
                    std::vector<std::vector<double>> act; // inputs of layer l (act[0] = z), act[L] = the output
                    std::vector<double> delta, delta_prev;
                };

                NeuralNetwork() = default;

                /// layer_sizes: {inputs, hidden..., outputs}. Parameters zero, no scaling
                explicit NeuralNetwork(const std::vector<int> & layer_sizes, Activation activation = Activation::tanh)
                    : sizes(layer_sizes), activation(activation) {
                    if (sizes.size() < 2)
                        throw std::invalid_argument("NeuralNetwork: we need at least the input and output sizes");
                    for (int s : sizes)
                        if (s < 1)
                            throw std::invalid_argument("NeuralNetwork: layer sizes must be >= 1");
                    size_t off = 0;
                    for (size_t l = 0; l + 1 < sizes.size(); l++) {
                        w_off.push_back(off);
                        off += size_t(sizes[l]) * size_t(sizes[l + 1]);
                        b_off.push_back(off);
                        off += size_t(sizes[l + 1]);
                    }
                    params.assign(off, 0.0);
                    input_mean.assign(sizes.front(), 0.0);
                    input_std.assign(sizes.front(), 1.0);
                    output_mean.assign(sizes.back(), 0.0);
                    output_std.assign(sizes.back(), 1.0);
                }

                /// The number of linear layers (hidden layers + 1)
                size_t num_layers() const { return sizes.empty() ? 0 : sizes.size() - 1; }
                int input_dim() const { return sizes.empty() ? 0 : sizes.front(); }
                int output_dim() const { return sizes.empty() ? 0 : sizes.back(); }
                const std::vector<int> & get_layer_sizes() const { return sizes; }
                Activation get_activation() const { return activation; }
                size_t num_parameters() const { return params.size(); }

                /// All the parameters (layer by layer: weights then biases)
                std::vector<double> & parameters() { return params; }
                const std::vector<double> & parameters() const { return params; }
                /// Weight (o, i) of layer l is weights(l)[o * sizes[l] + i]
                double * weights(size_t l) { return params.data() + w_off.at(l); }
                const double * weights(size_t l) const { return params.data() + w_off.at(l); }
                double * biases(size_t l) { return params.data() + b_off.at(l); }
                const double * biases(size_t l) const { return params.data() + b_off.at(l); }

                /// Weights from N(0, std^2), biases zero (Emulator.py's init_weights uses std = 1e-3)
                void init_normal(double sigma, uint64_t seed) {
                    detail::RNG rng(seed);
                    std::fill(params.begin(), params.end(), 0.0);
                    for (size_t l = 0; l < num_layers(); l++)
                        for (size_t k = 0; k < size_t(sizes[l]) * size_t(sizes[l + 1]); k++)
                            weights(l)[k] = sigma * rng.normal();
                }

                /// Glorot (Xavier) uniform weights, biases zero
                void init_glorot_uniform(uint64_t seed) {
                    detail::RNG rng(seed);
                    std::fill(params.begin(), params.end(), 0.0);
                    for (size_t l = 0; l < num_layers(); l++) {
                        const double a = std::sqrt(6.0 / double(sizes[l] + sizes[l + 1]));
                        for (size_t k = 0; k < size_t(sizes[l]) * size_t(sizes[l + 1]); k++)
                            weights(l)[k] = a * (2.0 * rng.uniform() - 1.0);
                    }
                }

                /// The standardization of the inputs: z = (x - mean) / std
                void set_input_scaling(const std::vector<double> & mean, const std::vector<double> & sigma) {
                    check_scaling(mean, sigma, sizes.front(), "input");
                    input_mean = mean;
                    input_std = sigma;
                }
                /// The scaling of the outputs: y = output * std + mean
                void set_output_scaling(const std::vector<double> & mean, const std::vector<double> & sigma) {
                    check_scaling(mean, sigma, sizes.back(), "output");
                    output_mean = mean;
                    output_std = sigma;
                }
                const std::vector<double> & get_input_mean() const { return input_mean; }
                const std::vector<double> & get_input_std() const { return input_std; }
                const std::vector<double> & get_output_mean() const { return output_mean; }
                const std::vector<double> & get_output_std() const { return output_std; }

                /// Evaluate at one point (in physical units)
                std::vector<double> operator()(const std::vector<double> & x) const {
                    if (x.size() != size_t(input_dim()))
                        throw std::invalid_argument("NeuralNetwork: the input has the wrong size");
                    Workspace ws;
                    std::vector<double> z(x.size()), y(output_dim());
                    for (size_t j = 0; j < x.size(); j++)
                        z[j] = (x[j] - input_mean[j]) / input_std[j];
                    forward_standardized(z.data(), ws);
                    for (size_t k = 0; k < y.size(); k++)
                        y[k] = ws.act.back()[k] * output_std[k] + output_mean[k];
                    return y;
                }

                /// Evaluate at many points (OpenMP over the points if USE_OMP)
                Matrix evaluate(const Matrix & X) const {
                    detail::check_matrix(X, size_t(input_dim()), "the input");
                    Matrix Y(X.size());
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
                    for (long i = 0; i < long(X.size()); i++)
                        Y[size_t(i)] = (*this)(X[size_t(i)]);
                    return Y;
                }

                /// The Jacobian dy_k / dx_j at x (in physical units): J[k][j]
                Matrix jacobian(const std::vector<double> & x) const {
                    if (x.size() != size_t(input_dim()))
                        throw std::invalid_argument("NeuralNetwork: the input has the wrong size");
                    Workspace ws;
                    std::vector<double> z(x.size());
                    for (size_t j = 0; j < x.size(); j++)
                        z[j] = (x[j] - input_mean[j]) / input_std[j];
                    forward_standardized(z.data(), ws);
                    const size_t L = num_layers();
                    Matrix J(size_t(output_dim()), std::vector<double>(x.size()));
                    for (int k = 0; k < output_dim(); k++) {
                        ws.delta.assign(size_t(output_dim()), 0.0);
                        ws.delta[size_t(k)] = 1.0;
                        for (size_t l = L; l-- > 0;) {
                            const size_t nin = size_t(sizes[l]), nout = size_t(sizes[l + 1]);
                            const double * W = weights(l);
                            ws.delta_prev.assign(nin, 0.0);
                            for (size_t o = 0; o < nout; o++)
                                for (size_t i = 0; i < nin; i++)
                                    ws.delta_prev[i] += W[o * nin + i] * ws.delta[o];
                            if (l > 0)
                                for (size_t i = 0; i < nin; i++)
                                    ws.delta_prev[i] *= detail::activate_derivative(activation, ws.pre[l - 1][i]);
                            std::swap(ws.delta, ws.delta_prev);
                        }
                        for (size_t j = 0; j < x.size(); j++)
                            J[size_t(k)][j] = output_std[size_t(k)] * ws.delta[j] / input_std[j];
                    }
                    return J;
                }

                /// The network on standardized inputs z; the (standardized) output is ws.act.back()
                void forward_standardized(const double * z, Workspace & ws) const {
                    const size_t L = num_layers();
                    ws.pre.resize(L);
                    ws.act.resize(L + 1);
                    ws.act[0].assign(z, z + sizes[0]);
                    for (size_t l = 0; l < L; l++) {
                        const size_t nin = size_t(sizes[l]), nout = size_t(sizes[l + 1]);
                        const double * W = weights(l);
                        const double * b = biases(l);
                        ws.pre[l].resize(nout);
                        ws.act[l + 1].resize(nout);
                        const double * a = ws.act[l].data();
                        for (size_t o = 0; o < nout; o++) {
                            double s = b[o];
                            const double * w = W + o * nin;
                            for (size_t i = 0; i < nin; i++)
                                s += w[i] * a[i];
                            ws.pre[l][o] = s;
                            ws.act[l + 1][o] = (l + 1 < L) ? detail::activate(activation, s) : s;
                        }
                    }
                }

                /// One point (standardized z, target t): adds scale * d(sum_k loss(out_k - t_k))/d(params) to grad
                /// and returns sum_k loss(out_k - t_k)
                double accumulate_gradient(
                    const double * z, const double * t, Loss loss, double scale, double * grad, Workspace & ws) const {
                    forward_standardized(z, ws);
                    const size_t L = num_layers();
                    const size_t nout_last = size_t(sizes.back());
                    double total = 0.0;
                    ws.delta.resize(nout_last);
                    for (size_t k = 0; k < nout_last; k++) {
                        const double r = ws.act[L][k] - t[k];
                        total += detail::loss_value(loss, r);
                        ws.delta[k] = scale * detail::loss_derivative(loss, r);
                    }
                    for (size_t l = L; l-- > 0;) {
                        const size_t nin = size_t(sizes[l]), nout = size_t(sizes[l + 1]);
                        const double * W = weights(l);
                        double * gW = grad + w_off[l];
                        double * gb = grad + b_off[l];
                        const double * a = ws.act[l].data();
                        for (size_t o = 0; o < nout; o++) {
                            const double d = ws.delta[o];
                            gb[o] += d;
                            double * g = gW + o * nin;
                            for (size_t i = 0; i < nin; i++)
                                g[i] += d * a[i];
                        }
                        if (l > 0) {
                            ws.delta_prev.assign(nin, 0.0);
                            for (size_t o = 0; o < nout; o++) {
                                const double d = ws.delta[o];
                                const double * w = W + o * nin;
                                for (size_t i = 0; i < nin; i++)
                                    ws.delta_prev[i] += w[i] * d;
                            }
                            for (size_t i = 0; i < nin; i++)
                                ws.delta_prev[i] *= detail::activate_derivative(activation, ws.pre[l - 1][i]);
                            std::swap(ws.delta, ws.delta_prev);
                        }
                    }
                    return total;
                }

                /// Write the network (text, 17 digits: reading it back gives the same network)
                void save(const std::string & filename) const {
                    std::ofstream out(filename);
                    if (not out)
                        throw std::runtime_error("NeuralNetwork::save: cannot open " + filename);
                    out << std::setprecision(17);
                    out << "FML_NEURAL_NETWORK 1\n";
                    out << "activation " << activation_name(activation) << "\n";
                    out << "sizes " << sizes.size();
                    for (int s : sizes)
                        out << " " << s;
                    out << "\n";
                    write_vector(out, "input_mean", input_mean);
                    write_vector(out, "input_std", input_std);
                    write_vector(out, "output_mean", output_mean);
                    write_vector(out, "output_std", output_std);
                    for (size_t l = 0; l < num_layers(); l++) {
                        const size_t nin = size_t(sizes[l]), nout = size_t(sizes[l + 1]);
                        out << "layer " << l << "\n";
                        for (size_t o = 0; o < nout; o++) {
                            for (size_t i = 0; i < nin; i++)
                                out << (i ? " " : "") << weights(l)[o * nin + i];
                            out << "\n";
                        }
                        for (size_t o = 0; o < nout; o++)
                            out << (o ? " " : "") << biases(l)[o];
                        out << "\n";
                    }
                    if (not out)
                        throw std::runtime_error("NeuralNetwork::save: error writing " + filename);
                }

                /// Read a network written by save() or by export_lightning_emulator.py ('#' starts a comment)
                static NeuralNetwork load(const std::string & filename) {
                    std::ifstream in(filename);
                    if (not in)
                        throw std::runtime_error("NeuralNetwork::load: cannot open " + filename);
                    std::stringstream tokens;
                    std::string line;
                    while (std::getline(in, line))
                        tokens << line.substr(0, line.find('#')) << "\n";

                    auto expect = [&](const std::string & word) {
                        std::string w;
                        if (not(tokens >> w) or w != word)
                            throw std::runtime_error("NeuralNetwork::load: expected '" + word + "' in " + filename);
                    };
                    auto read_number = [&]() {
                        double v;
                        if (not(tokens >> v))
                            throw std::runtime_error("NeuralNetwork::load: missing or bad number in " + filename);
                        return v;
                    };
                    expect("FML_NEURAL_NETWORK");
                    if (int(read_number()) != 1)
                        throw std::runtime_error("NeuralNetwork::load: unsupported version in " + filename);
                    expect("activation");
                    std::string act;
                    tokens >> act;
                    expect("sizes");
                    const int nsizes = int(read_number());
                    if (nsizes < 2)
                        throw std::runtime_error("NeuralNetwork::load: bad sizes in " + filename);
                    std::vector<int> s(static_cast<size_t>(nsizes));
                    for (auto & v : s)
                        v = int(read_number());
                    NeuralNetwork net(s, activation_from_name(act));
                    auto read_vector = [&](const std::string & name, size_t n) {
                        expect(name);
                        std::vector<double> v(n);
                        for (auto & x : v)
                            x = read_number();
                        return v;
                    };
                    const auto imean = read_vector("input_mean", size_t(s.front()));
                    const auto istd = read_vector("input_std", size_t(s.front()));
                    const auto omean = read_vector("output_mean", size_t(s.back()));
                    const auto ostd = read_vector("output_std", size_t(s.back()));
                    net.set_input_scaling(imean, istd);
                    net.set_output_scaling(omean, ostd);
                    for (size_t l = 0; l < net.num_layers(); l++) {
                        expect("layer");
                        if (size_t(read_number()) != l)
                            throw std::runtime_error("NeuralNetwork::load: layers out of order in " + filename);
                        const size_t nw = size_t(s[l]) * size_t(s[l + 1]);
                        for (size_t k = 0; k < nw; k++)
                            net.weights(l)[k] = read_number();
                        for (size_t k = 0; k < size_t(s[l + 1]); k++)
                            net.biases(l)[k] = read_number();
                    }
                    return net;
                }

              private:
                std::vector<int> sizes;
                Activation activation{Activation::tanh};
                std::vector<double> params;
                std::vector<size_t> w_off, b_off;
                std::vector<double> input_mean, input_std, output_mean, output_std;

                static void check_scaling(const std::vector<double> & mean,
                                          const std::vector<double> & sigma,
                                          int n,
                                          const std::string & what) {
                    if (mean.size() != size_t(n) or sigma.size() != size_t(n))
                        throw std::invalid_argument("NeuralNetwork: the " + what + " scaling has the wrong size");
                    for (double s : sigma)
                        if (not(s > 0.0) or not std::isfinite(s))
                            throw std::invalid_argument("NeuralNetwork: the " + what + " std must be > 0");
                }
                static void write_vector(std::ostream & out, const std::string & name, const std::vector<double> & v) {
                    out << name;
                    for (double x : v)
                        out << " " << x;
                    out << "\n";
                }
            };

            //=========================================================================================================
            // The pieces of the training (as PyTorch / Lightning; usable on their own)
            //=========================================================================================================

            /// The mean loss over the points idx[begin, end) of standardized (Z, T) and its gradient (overwrites
            /// grad). OpenMP over the points if USE_OMP (one gradient buffer per thread, summed in thread order)
            inline double batch_loss_and_gradient(const NeuralNetwork & net,
                                                  const Matrix & Z,
                                                  const Matrix & T,
                                                  const std::vector<size_t> & idx,
                                                  size_t begin,
                                                  size_t end,
                                                  Loss loss,
                                                  std::vector<double> & grad) {
                const size_t np = net.num_parameters();
                const size_t nb = end - begin;
                const double scale = 1.0 / (double(nb) * double(net.output_dim()));
                grad.assign(np, 0.0);
                double total = 0.0;
#ifdef USE_OMP
                const int nthreads = omp_get_max_threads();
                if (nthreads > 1 and nb >= 2 * size_t(nthreads)) {
                    std::vector<std::vector<double>> g(size_t(nthreads), std::vector<double>(np, 0.0));
                    std::vector<double> part(size_t(nthreads), 0.0);
#pragma omp parallel
                    {
                        const int t = omp_get_thread_num();
                        NeuralNetwork::Workspace ws;
#pragma omp for schedule(static)
                        for (long q = long(begin); q < long(end); q++) {
                            const size_t i = idx[size_t(q)];
                            part[size_t(t)] +=
                                net.accumulate_gradient(Z[i].data(), T[i].data(), loss, scale, g[size_t(t)].data(), ws);
                        }
                    }
                    for (int t = 0; t < nthreads; t++) {
                        total += part[size_t(t)];
                        for (size_t k = 0; k < np; k++)
                            grad[k] += g[size_t(t)][k];
                    }
                    return total * scale;
                }
#endif
                NeuralNetwork::Workspace ws;
                for (size_t q = begin; q < end; q++) {
                    const size_t i = idx[q];
                    total += net.accumulate_gradient(Z[i].data(), T[i].data(), loss, scale, grad.data(), ws);
                }
                return total * scale;
            }

            /// The mean loss over all of standardized (Z, T)
            inline double mean_loss(const NeuralNetwork & net, const Matrix & Z, const Matrix & T, Loss loss) {
                double total = 0.0;
#ifdef USE_OMP
#pragma omp parallel
#endif
                {
                    NeuralNetwork::Workspace ws;
#ifdef USE_OMP
#pragma omp for schedule(static) reduction(+ : total)
#endif
                    for (long i = 0; i < long(Z.size()); i++) {
                        net.forward_standardized(Z[size_t(i)].data(), ws);
                        for (int k = 0; k < net.output_dim(); k++)
                            total += detail::loss_value(loss, ws.act.back()[size_t(k)] - T[size_t(i)][size_t(k)]);
                    }
                }
                return total / (double(Z.size()) * double(net.output_dim()));
            }

            /// torch.nn.utils.clip_grad_norm_: scale grad so that its L2 norm is at most max_norm. Returns the
            /// norm before clipping
            inline double clip_gradient_norm(std::vector<double> & grad, double max_norm) {
                double s = 0.0;
                for (double g : grad)
                    s += g * g;
                const double norm = std::sqrt(s);
                const double coef = max_norm / (norm + 1e-6);
                if (coef < 1.0)
                    for (auto & g : grad)
                        g *= coef;
                return norm;
            }

            /// torch.optim.Adam (weight_decay is L2: added to the gradient, not AdamW)
            class Adam {
              public:
                double lr, beta1, beta2, eps, weight_decay;

                Adam(size_t n, double lr = 1e-3, double weight_decay = 0.0, double beta1 = 0.9, double beta2 = 0.999,
                     double eps = 1e-8)
                    : lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay), m(n, 0.0), v(n, 0.0) {}

                void step(std::vector<double> & p, const std::vector<double> & grad) {
                    t++;
                    const double bc1 = 1.0 - std::pow(beta1, double(t));
                    const double bc2 = 1.0 - std::pow(beta2, double(t));
                    const double step_size = lr / bc1;
                    const double sqrt_bc2 = std::sqrt(bc2);
                    for (size_t k = 0; k < p.size(); k++) {
                        const double g = grad[k] + weight_decay * p[k];
                        m[k] = beta1 * m[k] + (1.0 - beta1) * g;
                        v[k] = beta2 * v[k] + (1.0 - beta2) * g * g;
                        p[k] -= step_size * m[k] / (std::sqrt(v[k]) / sqrt_bc2 + eps);
                    }
                }
                long steps() const { return t; }

              private:
                std::vector<double> m, v;
                long t{0};
            };

            /// torch.optim.lr_scheduler.ReduceLROnPlateau (mode min, threshold_mode rel, cooldown 0)
            class ReduceLROnPlateau {
              public:
                ReduceLROnPlateau(double lr, double factor = 0.1, int patience = 20, double min_lr = 1e-6,
                                  double threshold = 1e-4)
                    : lr(lr), factor(factor), patience(patience), min_lr(min_lr), threshold(threshold) {}

                /// Call once per epoch with the monitored value; returns the learning rate to use
                double step(double metric) {
                    if (metric < best * (1.0 - threshold)) {
                        best = metric;
                        num_bad = 0;
                    } else {
                        num_bad++;
                    }
                    if (num_bad > patience) {
                        const double new_lr = std::max(lr * factor, min_lr);
                        if (lr - new_lr > 1e-8)
                            lr = new_lr;
                        num_bad = 0;
                    }
                    return lr;
                }
                double get_lr() const { return lr; }

              private:
                double lr, factor;
                int patience;
                double min_lr, threshold;
                double best{std::numeric_limits<double>::infinity()};
                int num_bad{0};
            };

            /// Lightning's EarlyStopping (mode min): stop after patience checks without an improvement by more
            /// than min_delta, or at once if the value is not finite
            class EarlyStopping {
              public:
                explicit EarlyStopping(int patience = 30, double min_delta = 0.0)
                    : patience(patience), min_delta(std::abs(min_delta)) {}

                /// Returns true if the training should stop
                bool step(double metric) {
                    if (not std::isfinite(metric))
                        return true;
                    if (metric + min_delta < best) {
                        best = metric;
                        wait = 0;
                        return false;
                    }
                    return ++wait >= patience;
                }

              private:
                int patience;
                double min_delta;
                double best{std::numeric_limits<double>::infinity()};
                int wait{0};
            };

            //=========================================================================================================
            // Training
            //=========================================================================================================

            /// The settings (the defaults are those of Emulator.py and Lightning where those have the setting)
            struct TrainingOptions {
                std::vector<int> hidden_layers{64, 64};
                Activation activation{Activation::tanh};
                Loss loss{Loss::mse};
                int max_epochs{1000};
                size_t batch_size{32};
                double learning_rate{1e-3};
                double weight_decay{0.0};
                double adam_beta1{0.9}, adam_beta2{0.999}, adam_eps{1e-8};
                /// Initial weights: init_std <= 0: Glorot uniform (the default); > 0: N(0, init_std^2). Emulator.py
                /// uses 1e-3, which starts on a plateau (the loss stays at its initial value for ~50 epochs, also
                /// in PyTorch), where the learning-rate reductions and early stopping can end the training
                double init_std{0.0};
                /// Clip the gradient to this L2 norm (<= 0: no clipping)
                double gradient_clip_norm{0.5};
                /// ReduceLROnPlateau on the validation loss
                int lr_patience{20};
                double lr_factor{0.1};
                double lr_min{1e-6};
                double lr_threshold{1e-4};
                /// Early stopping on the validation loss (<= 0: off)
                int early_stopping_patience{30};
                double early_stopping_min_delta{0.0};
                /// Stochastic weight averaging (Lightning's StochasticWeightAveraging): from epoch swa_epoch_start
                /// (a fraction of max_epochs if < 1) the learning rate is cosine-annealed to swa_lr over
                /// swa_annealing_epochs epochs and the weights are averaged at the end of each epoch. As in
                /// Lightning the average replaces the weights only if the training reaches max_epochs (unless
                /// swa_apply_if_stopped_early)
                bool swa{true};
                double swa_lr{1e-2};
                double swa_epoch_start{0.8};
                int swa_annealing_epochs{10};
                bool swa_apply_if_stopped_early{false};
                /// Return the weights of the epoch with the lowest validation loss (if SWA was not applied).
                /// false: the weights after the last epoch (as Lightning's last checkpoint)
                bool restore_best_weights{false};
                /// Standardize the inputs / outputs by the mean and std of the training set (stored in the network)
                bool standardize_inputs{true};
                bool standardize_outputs{true};
                uint64_t seed{1234};
                /// Print the losses every this many epochs (0: never)
                int verbose_every{0};
            };

            /// What happened during the training (the losses are in standardized units if standardize_outputs)
            struct TrainingHistory {
                std::vector<double> train_loss, val_loss, learning_rate;
                int epochs_run{0};
                int best_epoch{-1};
                double best_val_loss{std::numeric_limits<double>::infinity()};
                bool stopped_early{false};
                bool swa_applied{false};
                double final_val_loss{0.0}; // with the weights that are returned
            };

            /// Mean and std (population) of each column; std 0 is replaced by 1
            inline std::pair<std::vector<double>, std::vector<double>> column_mean_std(const Matrix & X) {
                const size_t n = X.size(), d = X.empty() ? 0 : X[0].size();
                std::vector<double> mean(d, 0.0), sigma(d, 0.0);
                for (const auto & x : X)
                    for (size_t j = 0; j < d; j++)
                        mean[j] += x[j] / double(n);
                for (const auto & x : X)
                    for (size_t j = 0; j < d; j++)
                        sigma[j] += (x[j] - mean[j]) * (x[j] - mean[j]) / double(n);
                for (auto & s : sigma) {
                    s = std::sqrt(s);
                    if (not(s > 0.0))
                        s = 1.0;
                }
                return {mean, sigma};
            }

            /// Train a network on (X_train, Y_train), monitoring the loss on (X_val, Y_val)
            inline NeuralNetwork train(const Matrix & X_train,
                                       const Matrix & Y_train,
                                       const Matrix & X_val,
                                       const Matrix & Y_val,
                                       const TrainingOptions & opt = {},
                                       TrainingHistory * history = nullptr) {
                if (X_train.empty() or X_val.empty())
                    throw std::invalid_argument("NeuralNetwork::train: empty training or validation set");
                if (X_train.size() != Y_train.size() or X_val.size() != Y_val.size())
                    throw std::invalid_argument("NeuralNetwork::train: X and Y have different numbers of points");
                if (opt.max_epochs < 1 or opt.batch_size < 1)
                    throw std::invalid_argument("NeuralNetwork::train: max_epochs and batch_size must be >= 1");
                const size_t nin = X_train[0].size(), nout = Y_train[0].size();
                if (nin == 0 or nout == 0)
                    throw std::invalid_argument("NeuralNetwork::train: no inputs or outputs");
                detail::check_matrix(X_train, nin, "X_train");
                detail::check_matrix(X_val, nin, "X_val");
                detail::check_matrix(Y_train, nout, "Y_train");
                detail::check_matrix(Y_val, nout, "Y_val");

                std::vector<int> sizes{int(nin)};
                for (int h : opt.hidden_layers)
                    sizes.push_back(h);
                sizes.push_back(int(nout));
                NeuralNetwork net(sizes, opt.activation);
                if (opt.init_std > 0.0)
                    net.init_normal(opt.init_std, opt.seed);
                else
                    net.init_glorot_uniform(opt.seed);
                if (opt.standardize_inputs) {
                    const auto ms = column_mean_std(X_train);
                    net.set_input_scaling(ms.first, ms.second);
                }
                if (opt.standardize_outputs) {
                    const auto ms = column_mean_std(Y_train);
                    net.set_output_scaling(ms.first, ms.second);
                }

                // The data in standardized units
                auto standardize = [](const Matrix & A, const std::vector<double> & mean,
                                      const std::vector<double> & sigma) {
                    Matrix B(A);
                    for (auto & row : B)
                        for (size_t j = 0; j < row.size(); j++)
                            row[j] = (row[j] - mean[j]) / sigma[j];
                    return B;
                };
                const Matrix Zx = standardize(X_train, net.get_input_mean(), net.get_input_std());
                const Matrix Zy = standardize(Y_train, net.get_output_mean(), net.get_output_std());
                const Matrix Vx = standardize(X_val, net.get_input_mean(), net.get_input_std());
                const Matrix Vy = standardize(Y_val, net.get_output_mean(), net.get_output_std());

                const size_t n = Zx.size();
                std::vector<double> & params = net.parameters();
                std::vector<double> grad(params.size());
                Adam adam(params.size(), opt.learning_rate, opt.weight_decay, opt.adam_beta1, opt.adam_beta2,
                          opt.adam_eps);
                ReduceLROnPlateau plateau(opt.learning_rate, opt.lr_factor, opt.lr_patience, opt.lr_min,
                                          opt.lr_threshold);
                EarlyStopping stopper(opt.early_stopping_patience, opt.early_stopping_min_delta);

                const int swa_start = opt.swa_epoch_start < 1.0 ? int(opt.max_epochs * opt.swa_epoch_start)
                                                                : int(std::lround(opt.swa_epoch_start));
                const int swa_end = opt.max_epochs - 1;
                bool swa_active = false;
                double swa_lr0 = 0.0;
                int swa_steps = 0;
                size_t swa_count = 0;
                std::vector<double> swa_average, best_params(params);

                TrainingHistory h;
                detail::RNG rng(opt.seed ^ 0x9e3779b97f4a7c15ULL);
                std::vector<size_t> order(n);
                for (size_t i = 0; i < n; i++)
                    order[i] = i;

                for (int epoch = 0; epoch < opt.max_epochs; epoch++) {
                    if (opt.swa and epoch == swa_start and swa_start <= swa_end) {
                        swa_active = true;
                        swa_lr0 = adam.lr;
                    }

                    rng.shuffle(order);
                    double train_sum = 0.0;
                    for (size_t begin = 0; begin < n; begin += opt.batch_size) {
                        const size_t end = std::min(n, begin + opt.batch_size);
                        const double l = batch_loss_and_gradient(net, Zx, Zy, order, begin, end, opt.loss, grad);
                        train_sum += l * double(end - begin);
                        if (opt.gradient_clip_norm > 0.0)
                            clip_gradient_norm(grad, opt.gradient_clip_norm);
                        adam.step(params, grad);
                    }
                    const double train_loss = train_sum / double(n);

                    if (swa_active) {
                        // Average the weights, then anneal the learning rate (cos) towards swa_lr
                        if (swa_count == 0)
                            swa_average = params;
                        else
                            for (size_t k = 0; k < params.size(); k++)
                                swa_average[k] += (params[k] - swa_average[k]) / double(swa_count + 1);
                        swa_count++;
                        swa_steps++;
                        const double t =
                            std::min(1.0, double(swa_steps) / double(std::max(1, opt.swa_annealing_epochs)));
                        const double alpha = 0.5 * (1.0 - std::cos(M_PI * t));
                        adam.lr = opt.swa_lr * alpha + swa_lr0 * (1.0 - alpha);
                    }

                    const double val_loss = mean_loss(net, Vx, Vy, opt.loss);
                    h.train_loss.push_back(train_loss);
                    h.val_loss.push_back(val_loss);
                    h.learning_rate.push_back(adam.lr);
                    h.epochs_run = epoch + 1;
                    if (val_loss < h.best_val_loss) {
                        h.best_val_loss = val_loss;
                        h.best_epoch = epoch;
                        if (opt.restore_best_weights)
                            best_params = params;
                    }
                    if (opt.verbose_every > 0 and (epoch % opt.verbose_every == 0 or epoch + 1 == opt.max_epochs))
                        std::cout << "epoch " << std::setw(6) << epoch << "  train " << std::setw(12) << train_loss
                                  << "  val " << std::setw(12) << val_loss << "  lr " << adam.lr << "\n";

                    if (not swa_active)
                        adam.lr = plateau.step(val_loss);
                    if (opt.early_stopping_patience > 0 and stopper.step(val_loss)) {
                        h.stopped_early = epoch + 1 < opt.max_epochs;
                        break;
                    }
                    if (not std::isfinite(val_loss))
                        break;
                }

                const bool reached_end = h.epochs_run == opt.max_epochs;
                if (swa_count > 0 and (reached_end or opt.swa_apply_if_stopped_early)) {
                    params = swa_average;
                    h.swa_applied = true;
                } else if (opt.restore_best_weights) {
                    params = best_params;
                }
                h.final_val_loss = mean_loss(net, Vx, Vy, opt.loss);
                if (history)
                    *history = h;
                return net;
            }

            //=========================================================================================================
            /// Read the given columns of a CSV file with a header line (as pandas writes them; quotes and spaces
            /// around the names are ignored). Returns rows of the values in the order of the columns asked for
            //=========================================================================================================
            inline Matrix read_csv_columns(const std::string & filename, const std::vector<std::string> & columns) {
                std::ifstream in(filename);
                if (not in)
                    throw std::runtime_error("read_csv_columns: cannot open " + filename);
                auto split = [](const std::string & line) {
                    std::vector<std::string> fields;
                    std::string field;
                    std::stringstream ss(line);
                    while (std::getline(ss, field, ','))
                        fields.push_back(field);
                    if (not line.empty() and line.back() == ',')
                        fields.push_back("");
                    return fields;
                };
                auto clean = [](std::string s) {
                    std::string r;
                    for (char c : s)
                        if (c != '"' and c != '\'' and c != '\r')
                            r += c;
                    const size_t a = r.find_first_not_of(" \t"), b = r.find_last_not_of(" \t");
                    return a == std::string::npos ? std::string() : r.substr(a, b - a + 1);
                };
                std::string line;
                if (not std::getline(in, line))
                    throw std::runtime_error("read_csv_columns: empty file " + filename);
                const auto header = split(line);
                std::vector<size_t> index;
                for (const auto & c : columns) {
                    size_t k = 0;
                    while (k < header.size() and clean(header[k]) != c)
                        k++;
                    if (k == header.size())
                        throw std::runtime_error("read_csv_columns: no column '" + c + "' in " + filename);
                    index.push_back(k);
                }
                Matrix rows;
                while (std::getline(in, line)) {
                    if (clean(line).empty())
                        continue;
                    const auto fields = split(line);
                    std::vector<double> row;
                    for (size_t k : index) {
                        if (k >= fields.size())
                            throw std::runtime_error("read_csv_columns: short line in " + filename);
                        row.push_back(std::stod(clean(fields[k])));
                    }
                    rows.push_back(row);
                }
                return rows;
            }

        } // namespace NEURALNETWORK
    }     // namespace EMULATOR
} // namespace FML

#endif
