# Experiments on Neural Networks

This repository contains three neural networks coded without specific machine learning libraries. Oscillatory Neural Networks (ONNs) were written in Python with numpy (later they will be rewritten in more efficient Julia code) and Feed-Forward Neural Network (FFNN) has been written in Julia. ONNs being more computationally inefficient are aimed to be trained on simple classification tasks with arbitrary numbers of input and output. Given examples are for one/multidimentional XOR datasets and multiplexer (which doesn't learn above cartain accuracy because it tries to predict output values, not classify output and this task needs another kind of architecture). FFNN by design learns handwritten digit recognition from [optdigits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)

Current implementation is based on [this study](https://arxiv.org/abs/2402.08579), I will try to implement ideas from [here](https://arxiv.org/abs/2311.03260) before switching to oscillator model proper for exciton-polariton condensate networks.

## Usage

before running network trainings I advise to first run "prepar_files_and_dataset.sh" which downloads optdigits dataset and initializes csv files that store important distance and accuracy values.
