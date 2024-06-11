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
import argparse
#progress bar
from tqdm import tqdm
# importing custom functions
sys.path.append('./src')
from ONNModule import kuramoto_oscillators, kuramoto_oscillators_nudge, clear_csv, append_to_csv, save_parameters, load_parameters


# works well with N=16 and eta=0.5 (even 600 epochs do the job)
def main():

    parser = argparse.ArgumentParser(description="A program to demonstrate command-line arguments.")

    # Define the arguments
    parser.add_argument('number', type=int, help="An integer number.")
    parser.add_argument('letter', type=str, help="A letter (string or char), if y or yes - output will be saved to a file")
    parser.add_argument('num_of_epochs', type=int, help="number of epochs.")
    parser.add_argument('save_parameters', type=str, help="y/yes to save parameters after finishing each epoch")
    parser.add_argument('load_parameters', type=str, help="y/yes to load parameters from previusly saved file")
    parser.add_argument('learning_rate', type=float, help="learning rate value")

    # Parse the arguments
    args = parser.parse_args()

    N = args.number
    do_save = args.letter
    num_of_epochs = args.num_of_epochs
    save_params = args.save_parameters
    load_params = args.load_parameters
    learning_rate = args.learning_rate

    print(f"Parameters:\nN\t=\t{N}\ndo_save\t=\t{do_save}\nnum_of_epochs\t=\t{num_of_epochs}\nsave_params\t=\t{save_params}\nload_params\t=\t{load_params}")

    # Generate all possible combinations of 4 binary inputs
    input_combinations = list(itertools.product([0, 1], repeat=4))

    # Define output logic for the XOR dataset
    outputs = []
    for combination in input_combinations:
        input1, input2, input3, input4 = combination

        # XOR operation on pairs
        output1 = input1 ^ input2  # XOR of first and second bit
        output2 = input3 ^ input4  # XOR of third and fourth bit

        # Create the label array of 4 numbers
        label = [-np.pi/2] * 4
        label[output1 * 2 + output2] = np.pi/2  # Map to one-hot like encoding with angles
        outputs.append(label)

    # Create DataFrame
    df = pd.DataFrame(input_combinations, columns=['Input1', 'Input2', 'Input3', 'Input4'])
    df[['Label1', 'Label2', 'Label3', 'Label4']] = pd.DataFrame(outputs)

    # Convert inputs
    def convert_binary_to_angle(arr):
        return np.where(arr == 0, -np.pi/2, np.pi/2)

    features = np.array(df.iloc[:, :4])
    features_converted = convert_binary_to_angle(features)
    labels_converted = np.array(df.iloc[:, 4:])
    # labels are set in order (0,0), (0,1), (1,0), (1,1)

    if N<8:
        print("Number of neurons is chosen too small. It has to be greater than or equal to 8")
        return 1

    # initializing network:
    neurons = np.arange(0,N,1)
    # for classification task output has to equal all possible labels (here 4 for every combination of 2 bits)
    outputn = [N-4,N-3,N-2,N-1]
    inputn = [0,1,2,3]
    connections_neuronwise = np.array([[element for element in neurons if element != neuron] for neuron in neurons])
    # weight matrix is defined like this for ease of updating it with gradients using numpy methods
    weights_matrix = np.zeros((N,N))
    for i in range(1,N):
        for j in range(0,i):
            weights_matrix[i][j] = np.random.normal(loc=0, scale=1)
            weights_matrix[j][i] = weights_matrix[i][j]

    # matrix for direct use in elementwise operations during equation solving

    if load_params=="y" or load_params=="yes":
        weights_matrix, weights, biases, bias_phases = load_parameters("saved_parameters")
        print("Parameters loaded...")
    else:
        weights = weights_matrix[connections_neuronwise, np.arange(N)[:, None]]
        biases = np.array([np.random.uniform(-0.5, 0.5) for _ in neurons])
        bias_phases = np.array([np.random.uniform(-np.pi, np.pi) for _ in neurons])

    init_phases = np.array([np.random.uniform(-np.pi, np.pi) for _ in neurons])
    phases = np.copy(init_phases)

    
    #if (save_params=="y") and (load_params!="y"):
    #    clear_csv("./XOR 4x2.csv")
    #    clear_csv("./XOR 4x2 accuracies.csv")

    if len(biases) != N:
        print("Sizes of saved matrices don't match desired number of neurons.")
        return 1

    distances = []
    costs = []
    accuracies = []

    # Defining training parameters and matrices for use during training

    T = 20
    dt = 0.01
    times = np.arange(0, T+dt, dt)

    beta_std = 1e-4
    beta = np.zeros(N)
    beta[outputn[0]] = beta_std
    beta[outputn[1]] = beta_std
    beta[outputn[2]] = beta_std
    beta[outputn[3]] = beta_std
    batch_size = len(features_converted)
    random_init_times = 16
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

    #num_of_epochs = int(input("Enter number of epochs:\t"))

    # Training loop


    #for epoch in range(num_of_epochs):
    for epoch in tqdm(range(num_of_epochs)):

        weight_gradient = np.zeros((N,N))
        bias_gradient = np.zeros(N)
        bias_phase_gradient = np.zeros(N)

        distance_temp = []
        cost_temp = []
        accuracy_temp = []
        num_of_grads = 0
        for feature, label in zip(features_converted, labels_converted):

            target = np.zeros(N)
            target[outputn[0]] = label[0]
            target[outputn[1]] = label[1]
            target[outputn[2]] = label[2]
            target[outputn[3]] = label[3]

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

            # removes jumping gradients (critical value is arbitrary)
            if np.max(gradient_weights_backward - gradient_weights_forward)*inv_nudge_step*inv_random_init_times>1:
                continue

            # calculating gradient
            weight_gradient += gradient_weights_backward - gradient_weights_forward
            bias_gradient += gradient_biases_backward - gradient_biases_forward
            bias_phase_gradient += gradient_bias_phases_backward - gradient_bias_phases_forward
            num_of_grads += 1

            # calculating distance and cost
            distance_temp.append(2-np.cos(phases[outputn[0]]-label[0])-np.cos(phases[outputn[1]]-label[1]))
            cost_temp.append(-np.log(1+np.cos(phases[outputn[0]]-label[0]))-np.log(1+np.cos(phases[outputn[1]]-label[1])))


            accuracy_measure = [1-np.cos(phases[outputn[0]]-label[0]), 1-np.cos(phases[outputn[1]]-label[1])]


            if accuracy_measure[0] < 1 and accuracy_measure[1] < 1:  # Cosines are closer
                accuracy_temp.append(1)
            else:  # Cosines are further
                accuracy_temp.append(0)

        distances.append(np.mean(distance_temp))
        costs.append(np.mean(cost_temp))
        accuracies.append(np.mean(accuracy_temp))

        # parameter updates
        weight_gradient *= inv_nudge_step  * inv_random_init_times / num_of_grads
        bias_gradient *= inv_nudge_step  * inv_random_init_times / num_of_grads
        bias_phase_gradient *= inv_nudge_step  * inv_random_init_times / num_of_grads

        weight_gradient /= np.linalg.norm(weight_gradient,ord=2)
        bias_gradient /= np.linalg.norm(bias_gradient,ord=2)
        bias_phase_gradient /= np.linalg.norm(bias_phase_gradient,ord=2)

        if weight_gradient.max()>1:
            print(f"WARNING: weight_gradient.max() = {weight_gradient.max()}. Gradients may overflow critical value and no further convergence might be possible.")
            print(f"WARNING: all weights normalized automatically")
            weight_gradient /= np.linalg.norm(weight_gradient,ord=2)
            bias_gradient /= np.linalg.norm(bias_gradient,ord=2)
            bias_phase_gradient /= np.linalg.norm(bias_phase_gradient,ord=2)

        # normalization (gradient has a strong tendency to have very high values, changing cost function might help)
        #if np.linalg.norm(weight_gradient, ord=2)>5:
        
        #if True:
        #    weight_gradient /= np.linalg.norm(weight_gradient,ord=2)
        #    bias_gradient /= np.linalg.norm(bias_gradient,ord=2)
        #    bias_phase_gradient /= np.linalg.norm(bias_phase_gradient,ord=2)

        #plt.imshow(weights_matrix, cmap='hot', interpolation='nearest'); plt.show()

        weights_matrix -= learning_rate * weight_gradient
        weights = weights_matrix[connections_neuronwise, np.arange(N)[:, None]]
        biases -= learning_rate * bias_gradient
        bias_phases -= learning_rate * bias_phase_gradient

        if save_params=="y" or save_params=="yes":
            # czy zapisuje nauczone czy nienauczone
            save_parameters(weights_matrix, weights, biases, bias_phases, "saved_parameters")



    """
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
    """


    #do_save = input("Do you want to save the distances and accuracies y/n:\t")
    if do_save=="y" or do_save=="yes":
        append_to_csv("XOR 4x2.csv", distances)
        append_to_csv("XOR 4x2 accuracies.csv", accuracies)

    return 0

if __name__ == "__main__":
    main()



