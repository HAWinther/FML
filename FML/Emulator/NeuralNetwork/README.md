# Neural-network emulator

`NeuralNetwork.h` is a fully connected neural network for emulators, written from scratch in C++. It has no
dependencies (OpenMP is optional) and covers both uses:

- **Evaluate a network trained in PyTorch.** Use this for networks trained with the Sesame pipeline's
  [`Emulator.py`](https://github.com/renmau/Sesame_pipeline/blob/main/Pipeline/neural_network_emulator/Emulator.py):
  export the checkpoint with `export_lightning_emulator.py`, then call `NeuralNetwork::load`.
- **Train in C++.** `train()` uses the same recipe as `Emulator.py`: Adam with weight decay, gradient clipping,
  ReduceLROnPlateau, early stopping and stochastic weight averaging, with the same defaults.

```cpp
#include <FML/Emulator/NeuralNetwork/NeuralNetwork.h>
using namespace FML::EMULATOR::NEURALNETWORK;

TrainingOptions opt;                      // defaults: Emulator.py / Lightning (see below)
opt.hidden_layers = {64, 64};
opt.activation = Activation::gelu;        // or activation_from_name("gelu")
TrainingHistory history;
NeuralNetwork net = train(X_train, Y_train, X_val, Y_val, opt, &history);   // Matrix = vector<vector<double>>

std::vector<double> y = net(x);           // one point, in physical units
Matrix Y = net.evaluate(X);               // many points (OpenMP)
Matrix J = net.jacobian(x);               // dy_k/dx_j, e.g. for HMC or Fisher matrices
net.save("emulator.txt");
NeuralNetwork net2 = NeuralNetwork::load("emulator.txt");

Matrix X = read_csv_columns("train.csv", {"Omega_m", "sigma8"});   // the CSV files Emulator.py reads
```

Use the [Latin hypercube](../../LatinHyperCube/README.md) module to choose the training points. The
[example](example/Main.cpp) does this for a linear power-spectrum emulator in (Omega_m, sigma8, n_s): it trains in a
few seconds, reaches a mean |ΔP/P| of 7e-4 on 200 test cosmologies, and checks the Jacobian against the exact
d ln P / d σ8 = 2/σ8.

## A network trained with Emulator.py

```sh
python3 export_lightning_emulator.py path/to/lightning_logs/version_0 emulator.txt
```

The script takes the run folder (the one `EmulatorEvaluator.load` uses, i.e. the first checkpoint in
`checkpoints/`) or a `.ckpt` file. It reads the layers `hidden_layers.0, 1, …` and `output_layer`, and takes the
activation from the checkpoint or `hparams.yaml`; `--activation` overrides it. It needs `torch`. `Emulator.py` does
not scale its inputs or outputs, so the exported file has mean 0 and std 1. If you scaled the data yourself, pass the
scaling with `--input-scaling` / `--output-scaling FILE` (two lines: the means, then the stds). In C++ the exported
network gives the same outputs as torch in double precision to about 1e-16 (see the tests). Torch itself runs the
model in float32, so expect differences of about 1e-7 relative to a float32 torch run.

## The network and the file format

The network computes x → z = (x − input_mean)/input_std → hidden layers a ← f(W a + b) → a linear output layer
→ y = out · output_std + output_mean. The activations f are `identity, relu, tanh, sigmoid, gelu` (exact erf),
`silu, elu, leaky_relu` (slope 0.01) and `softplus` (with PyTorch's threshold 20). The losses (`Loss::mse, l1,
huber`) are means over the points and the outputs, as `torch.nn.MSELoss`, `L1Loss` and `HuberLoss`. All
parameters live in one array (`parameters()`): for each layer the weights (row-major, as `nn.Linear.weight`), then
the biases.

The file is plain text and `#` starts a comment:

```
FML_NEURAL_NETWORK 1
activation gelu
sizes 4 4 16 16 3            # the number of sizes, then inputs, hidden..., outputs
input_mean ...  input_std ...  output_mean ...  output_std ...
layer 0                      # then sizes[1] rows of sizes[0] weights, then one row of sizes[1] biases
...
```

`save` writes 17 significant digits, so reading a file back gives a bit-identical network.

## Training

| Option | Default | Emulator.py / Lightning |
| --- | --- | --- |
| `hidden_layers`, `activation`, `loss` | {64, 64}, tanh, mse | from `input.yaml` |
| `max_epochs`, `batch_size`, `learning_rate`, `weight_decay` | 1000, 32, 1e-3, 0 | from `input.yaml` |
| Adam β1, β2, ε | 0.9, 0.999, 1e-8 | torch defaults; weight decay is L2 as in `torch.optim.Adam` (not AdamW) |
| `gradient_clip_norm` | 0.5 | `gradient_clip_val=0.5` (norm) |
| `lr_patience`, `lr_factor`, `lr_min`, `lr_threshold` | 20, 0.1, 1e-6, 1e-4 | ReduceLROnPlateau on the validation loss |
| `early_stopping_patience`, `early_stopping_min_delta` | 30, 0 | EarlyStopping on the validation loss |
| `swa`, `swa_lr`, `swa_epoch_start`, `swa_annealing_epochs` | true, 1e-2, 0.8, 10 | StochasticWeightAveraging(swa_lrs=1e-2) |
| `standardize_inputs`, `standardize_outputs` | true | **not done** |
| `init_std` | 0 (Glorot uniform) | **N(0, 1e-3²)**: set `init_std = 1e-3` |
| `restore_best_weights` | false | false: the weights after the last epoch (the last checkpoint) |

Each epoch shuffles the training set and takes one Adam step per batch (the last batch may be smaller). It then
computes the validation loss on the whole validation set and updates the learning-rate schedule and early
stopping. With standardized outputs the losses are in standardized units. Stochastic weight averaging starts at
epoch `swa_epoch_start · max_epochs`. From then on the learning rate is cosine-annealed from its current value to
`swa_lr` over `swa_annealing_epochs` epochs, replacing ReduceLROnPlateau, and the weights are averaged at the end of
each epoch. As in Lightning, the averaged weights are used only if the training reaches `max_epochs`; with the
default patience, early stopping usually ends the training before that. `TrainingHistory` records the losses and
learning rate of every epoch, the best epoch, and whether the training stopped early or used SWA.

**Why two defaults differ from Emulator.py.** With weights drawn from N(0, 1e-3²), the network starts on a plateau:
the loss stays at its initial value for about 50 epochs. PyTorch does exactly the same. ReduceLROnPlateau (patience
20) and early stopping (patience 30) then end the training before it learns anything. Glorot initialisation trains
at once, and standardizing the data does not depend on the units of the parameters or of the emulated quantity.
`tests/test_neural_network.cpp` checks both behaviours.

With `USE_OMP` the points of a batch are split over the threads, so the result depends slightly on the number of
threads. Without OpenMP, the same seed gives the same network on every machine.

## Tests

Run `make run` in `tests/`. The PyTorch reference values in `tests/TorchReference.h` come from
`tests/make_torch_reference.py`, which needs torch. The tests check:

- **Against PyTorch:** the outputs, losses and gradients for every activation and loss, the parameters after three
  clipped Adam steps with weight decay (all within about 1e-15), ReduceLROnPlateau, and a Lightning-style
  checkpoint exported with the script.
- **Finite differences:** the gradient and the Jacobian.
- **The rest:** bitwise save/load, early stopping, SWA, training a smooth 2D → 2D function to 1%, reproducible
  seeds, and the CSV reader.
