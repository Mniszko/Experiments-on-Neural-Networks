# Experiments on Neural Networks

This repository contains three neural networks coded without specific machine learning libraries. Oscillatory Neural Networks (ONNs) were written in Python with numpy (although I have plans for rewriting them with python JAX or in Julia) and Feed-Forward Neural Network (FFNN) has been written in Julia. ONNs being more computationally inefficient are aimed to be trained on simple XOR dataset and multiplexer dataset - the first one having 2 inputs and 1 output, the second 4 inputs and 2 outputs. FFNN by design learns handwritten digit recognition from [optdigits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)

### Multiplexer

First two bits select specific data bit, second two are data bits. First output bit is the value of the selected bit and second is the value of the other one. I chose multiplexer because it requires low number of neurons and has quite complicated dynamics.

Current implementation is based on [this study](https://arxiv.org/abs/2402.08579), I will try to implement ideas from [here](https://arxiv.org/abs/2311.03260) before switching to oscillator model proper for exciton-polariton condensate networks.

## Usage

before running network trainings I advise to first run "prepar_files_and_dataset.sh" which downloads optdigits dataset and initializes csv files that store important distance and accuracy values.
