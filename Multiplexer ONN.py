import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import colorsys
import os
import imageio
import re
import itertools
import pandas as pd
import csv
import sys
# importing custom functions
sys.path.append('./src')
from ONNModule import kuramoto_oscillators, kuramoto_oscillators_nudge, clear_csv, append_to_csv


def main():

    # Generating multiplexer dataset:
    # Generate all possible combinations of 4 binary inputs
    input_combinations = list(itertools.product([0, 1], repeat=4))

    # Define output logic for the multiplexer
    outputs = []
    for combination in input_combinations:
        selector_bits = combination[:2]  # First two bits as selector bits
        data_bits = combination[2:]      # Last two bits as data bits

        # Determine which data bit to select based on selector bits
        selected_data_bit_index = selector_bits[0] * 2 + selector_bits[1]
        output1 = data_bits[selected_data_bit_index % 2]  # Selected data bit
        output2 = data_bits[(selected_data_bit_index + 1) % 2]  # The other data bit

        outputs.append([output1, output2])
    df = pd.DataFrame(input_combinations, columns=['Input1', 'Input2', 'Input3', 'Input4'])
    df['Output1'] = [output[0] for output in outputs]
    df['Output2'] = [output[1] for output in outputs]
    features = np.array(df.iloc[:,0:4])
    labels = np.array(df.iloc[:,4:])
    def convert_binary_to_angle(arr):
        return np.where(arr == 0, -np.pi/2, np.pi/2)

    features_converted = convert_binary_to_angle(features)
    labels_converted = convert_binary_to_angle(labels)

    
    N = int(input("Enter number of (fully connected) neurons (minimum stands at N = 5):\t"))
    if N<6:
        return 1
    
    # initializing network:
    neurons = np.arange(0,N,1)
    outputn = [4,5]
    inputn = [0,1,2,3]
    connections_neuronwise = np.array([[element for element in neurons if element != neuron] for neuron in neurons])
    # weight matrix is defined like this for ease of updating it with gradients using numpy methods
    weights_matrix = np.zeros((N,N))
    for i in range(1,N):
        for j in range(0,i):
            weights_matrix[i][j] = np.random.normal(loc=0, scale=1)
            weights_matrix[j][i] = weights_matrix[i][j]

    # matrix for direct use in elementwise operations during equation solving
    weights = weights_matrix[connections_neuronwise, np.arange(N)[:, None]]
    biases = np.array([np.random.uniform(-0.5, 0.5) for _ in neurons])
    bias_phases = np.array([np.random.uniform(-np.pi, np.pi) for _ in neurons])
    init_phases = np.array([np.random.uniform(-np.pi, np.pi) for _ in neurons])
    phases = np.copy(init_phases)

    distances = []
    costs = []
    accuracies = []

    # Defining training parameters and matrices for use during training

    T = 10
    dt = 0.01
    times = np.arange(0, T+dt, dt)

    beta = np.zeros(N)
    beta[outputn[0]] = 1e-6
    beta[outputn[1]] = 1e-6
    batch_size = len(features_converted)
    random_init_times = 1
    inv_nudge_step = 1/beta[outputn[0]]
    inv_batch_size = 1/batch_size
    inv_random_init_times = 1/random_init_times

    gradient_weights_forward = np.zeros((N,N))
    gradient_weights_backward = np.zeros((N,N))
    weight_gradient = np.zeros((N,N))

    gradient_biases_forward = np.zeros(N)
    gradient_biases_backward = np.zeros(N)
    bias_gradient = np.zeros(N)

    gradient_bias_phases_forward = np.zeros(N)
    gradient_bias_phases_backward = np.zeros(N)
    bias_phase_gradient = np.zeros(N)

    learning_rate = 0.1
    num_of_epochs = int(input("Enter number of epochs:\t"))

    # Training loop

    for epoch in range(num_of_epochs):

        if epoch%50==0:
            print(f"Starting epoch number {epoch}")

        weight_gradient = np.zeros((N,N))
        bias_gradient = np.zeros(N)
        bias_phase_gradient = np.zeros(N)
        
        distance_temp = []
        cost_temp = []
        accuracy_temp = []
        for feature, label in zip(features_converted, labels_converted):
            
            target = np.zeros(N)
            target[outputn[0]] = label[0]
            target[outputn[1]] = label[1]

            # inserting input values and random initialization
            phases = np.array([np.random.uniform(-np.pi, np.pi) for _ in neurons])
            
            for inp in inputn:
                phases[inp] = feature[inp]
                phases[inp] = feature[inp]

            # calculating inference and inference energy
            thetas = odeint(kuramoto_oscillators, phases, times, args=(weights, biases, bias_phases, connections_neuronwise, inputn), full_output=0)

            # applying resulting phases
            phases = thetas[-1]

            for i in range(1,N):
                for j in range(0,i):
                    gradient_weights_forward[i][j] = -np.cos(phases[i]-phases[j])
                    gradient_weights_forward[j][i] = gradient_weights_forward[i][j]
            for neuron in neurons:
                gradient_biases_forward[neuron] = -np.cos(phases[neuron]-bias_phases[neuron])
                gradient_bias_phases_forward[neuron] = -biases[neuron] * np.sin(phases[neuron]-bias_phases[neuron])
            
            # calculating nudge of inference and its energy
            thetas_back = odeint(kuramoto_oscillators_nudge, phases, times, args=(weights, biases, bias_phases, connections_neuronwise, inputn, beta, target), full_output=0)

            phases = thetas_back[-1]

            for i in range(1,N):
                for j in range(0,i):
                    gradient_weights_backward[i][j] = -np.cos(phases[i]-phases[j])
                    gradient_weights_backward[j][i] = gradient_weights_backward[i][j]
            for neuron in neurons:
                gradient_biases_backward[neuron] = -np.cos(phases[neuron]-bias_phases[neuron])
                gradient_bias_phases_backward[neuron] = -biases[neuron] * np.sin(phases[neuron]-bias_phases[neuron])
            
            # calculating gradient
            weight_gradient += gradient_weights_backward - gradient_weights_forward
            bias_gradient += gradient_biases_backward - gradient_biases_forward
            bias_phase_gradient += gradient_bias_phases_backward - gradient_bias_phases_forward

            # calculating distance and cost
            distance_temp.append(2-np.cos(phases[outputn[0]]-label[0])-np.cos(phases[outputn[1]]-label[1]))
            cost_temp.append(-np.log(1+np.cos(phases[outputn[0]]-label[0]))-np.log(1+np.cos(phases[outputn[1]]-label[1])))
            if distance_temp[-1]>2:
                accuracy_temp.append(0)
            elif distance_temp[-1]<2:
                accuracy_temp.append(1)
        
        distances.append(np.mean(distance_temp))
        costs.append(np.mean(cost_temp))
        accuracies.append(np.mean(accuracy_temp))

        # parameter updates
        weight_gradient *= inv_nudge_step * inv_batch_size * inv_random_init_times
        bias_gradient *= inv_nudge_step * inv_batch_size * inv_random_init_times
        bias_phase_gradient *= inv_nudge_step * inv_batch_size * inv_random_init_times

        # normalization (gradient has a strong tendency to have very high values, changing cost function might help)
        #if np.linalg.norm(weight_gradient, ord=2)>5:
        if True:
            weight_gradient /= np.linalg.norm(weight_gradient,ord=2)
            bias_gradient /= np.linalg.norm(bias_gradient,ord=2)
            bias_phase_gradient /= np.linalg.norm(bias_phase_gradient,ord=2)

        weights_matrix -= learning_rate * weight_gradient
        weights = weights_matrix[connections_neuronwise, np.arange(N)[:, None]]
        biases -= learning_rate * bias_gradient
        bias_phases -= learning_rate * bias_phase_gradient

    fig, ax1 = plt.subplots()

    ax1.plot(distances, color='b', label='Distances')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Distances', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(accuracies, color='g', label='Accuracies')
    ax2.set_ylabel('Accuracies', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    plt.title('Distances and Accuracies')

    fig.tight_layout()
    plt.show()

    do_save = input("Do you want to save the distances and accuracies y/n:\t")
    if do_save=="y" or do_save=="yes":
        append_to_csv("multiplexer_FNN_distances.csv", distances)
        append_to_csv("multiplexer_FNN_accuracies.csv", accuracies)

    return 0

if __name__ == "__main__":
    main()



