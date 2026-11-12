#!/usr/bin/env python3
"""Export an emulator trained with Emulator.py (the PyTorch Lightning network of the Sesame pipeline) to the text
format of FML::EMULATOR::NEURALNETWORK::NeuralNetwork::load.

    python3 export_lightning_emulator.py RUN emulator.txt

RUN is the Lightning log folder of the run (the folder EmulatorEvaluator.load takes: the first checkpoint in
RUN/checkpoints/ is used, the activation is read from RUN/hparams.yaml) or a .ckpt file. The network is the one of
Emulator.py: the layers hidden_layers.0, hidden_layers.1, ... with the activation, then the linear output_layer. The
Python code does not scale the inputs or outputs, so the file has mean 0 and std 1; --input-scaling and
--output-scaling FILE (two lines: the means and the stds, whitespace separated) put a scaling of your own in the file.

Needs torch (and PyYAML if the activation is read from hparams.yaml). A checkpoint is a pickle: load only files you
trust.
"""

import argparse
import glob
import os
import re
import sys

ACTIVATIONS = {"identity", "relu", "tanh", "sigmoid", "gelu", "silu", "elu", "leaky_relu", "softplus"}


def find_checkpoint(path):
    if os.path.isfile(path):
        return path, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), "hparams.yaml")
    ckpts = sorted(glob.glob(os.path.join(path, "checkpoints", "*.ckpt")))
    if not ckpts:
        sys.exit(f"export_lightning_emulator: no checkpoints/*.ckpt in {path}")
    return ckpts[0], os.path.join(path, "hparams.yaml")


def read_activation(checkpoint, hparams_file):
    hp = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    if "activation_function" in hp:
        return str(hp["activation_function"])
    if os.path.isfile(hparams_file):
        try:
            import yaml

            with open(hparams_file) as f:
                data = yaml.safe_load(f) or {}
            if "activation_function" in data:
                return str(data["activation_function"])
        except ImportError:
            with open(hparams_file) as f:
                m = re.search(r"^activation_function:\s*['\"]?([A-Za-z_]+)", f.read(), re.M)
            if m:
                return m.group(1)
    return None


def read_scaling(filename, n, what):
    if filename is None:
        return [0.0] * n, [1.0] * n
    with open(filename) as f:
        lines = [line.split() for line in f if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) != 2 or len(lines[0]) != n or len(lines[1]) != n:
        sys.exit(f"export_lightning_emulator: {filename} must have two lines of {n} numbers ({what} mean, std)")
    return [float(v) for v in lines[0]], [float(v) for v in lines[1]]


def layers_from_state_dict(state):
    """The (weight, bias) of each linear layer, in order: hidden_layers.0, 1, ..., then output_layer"""
    state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
    layers = []
    i = 0
    while f"hidden_layers.{i}.weight" in state:
        layers.append((state[f"hidden_layers.{i}.weight"], state[f"hidden_layers.{i}.bias"]))
        i += 1
    if "output_layer.weight" not in state:
        sys.exit("export_lightning_emulator: no output_layer.weight in the checkpoint (not an Emulator.py network?)")
    layers.append((state["output_layer.weight"], state["output_layer.bias"]))
    return layers


def write_network(filename, activation, layers, input_scaling, output_scaling):
    sizes = [int(layers[0][0].shape[1])] + [int(w.shape[0]) for w, _ in layers]
    for k in range(len(layers)):
        w, b = layers[k]
        if tuple(w.shape) != (sizes[k + 1], sizes[k]) or tuple(b.shape) != (sizes[k + 1],):
            sys.exit(f"export_lightning_emulator: layer {k} has inconsistent shapes {tuple(w.shape)}, {tuple(b.shape)}")
    fmt = lambda values: " ".join(repr(float(v)) for v in values)
    with open(filename, "w") as f:
        f.write("FML_NEURAL_NETWORK 1\n")
        f.write(f"activation {activation}\n")
        f.write(f"sizes {len(sizes)} " + " ".join(str(s) for s in sizes) + "\n")
        f.write("input_mean " + fmt(input_scaling[0]) + "\n")
        f.write("input_std " + fmt(input_scaling[1]) + "\n")
        f.write("output_mean " + fmt(output_scaling[0]) + "\n")
        f.write("output_std " + fmt(output_scaling[1]) + "\n")
        for k, (w, b) in enumerate(layers):
            f.write(f"layer {k}\n")
            for row in w.tolist():
                f.write(fmt(row) + "\n")
            f.write(fmt(b.tolist()) + "\n")
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="the Lightning log folder of the run, or a .ckpt file")
    parser.add_argument("output", help="the file to write")
    parser.add_argument("--activation", help="the hidden activation (default: from the checkpoint/hparams.yaml)")
    parser.add_argument("--input-scaling", help="file with the input means and stds (default: none)")
    parser.add_argument("--output-scaling", help="file with the output means and stds (default: none)")
    args = parser.parse_args()

    import torch

    ckpt_file, hparams_file = find_checkpoint(args.run)
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    layers = [(w.detach().double().cpu(), b.detach().double().cpu()) for w, b in layers_from_state_dict(state)]

    activation = args.activation or read_activation(checkpoint, hparams_file)
    if activation is None:
        sys.exit("export_lightning_emulator: activation not found; give it with --activation")
    if activation not in ACTIVATIONS:
        sys.exit(f"export_lightning_emulator: activation '{activation}' is not supported by FML "
                 f"({', '.join(sorted(ACTIVATIONS))})")

    n_in, n_out = int(layers[0][0].shape[1]), int(layers[-1][0].shape[0])
    sizes = write_network(args.output, activation, layers, read_scaling(args.input_scaling, n_in, "input"),
                          read_scaling(args.output_scaling, n_out, "output"))
    print(f"{ckpt_file} -> {args.output}: sizes {sizes}, activation {activation}")


if __name__ == "__main__":
    main()
