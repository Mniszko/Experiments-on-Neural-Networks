import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import colorsys
import os
import imageio
import re
import sys
import argparse
# importing custom functions
sys.path.append('./src')
from ONNModule import kuramoto_oscillators, kuramoto_oscillators_nudge, clear_csv, append_to_csv


def main():

    # Create the parser
    parser = argparse.ArgumentParser(description="A program to demonstrate command-line arguments.")
    
    # Define the arguments
    parser.add_argument('number', type=int, help="An integer number.")
    parser.add_argument('letter', type=str, help="A letter (string).")
    parser.add_argument('num_of_epochs', type=int, help="number of epochs.")
    
    # Parse the arguments
    args = parser.parse_args()

    N = args.number
    do_save = args.letter
    num_of_epochs = args.num_of_epochs

    
    #N = int(input("Enter number of (fully connected) neurons (minimum stands at N = 5):\t"))
    if N<3:
        return 1

    # defining neurons and connections
    neurons = np.arange(0,N,1)
    outputn = 2
    inputn = [0,1]
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

    # defining training values, full dataset and gradient matrices for later use
    T = 100
    dt = 0.01
    times = np.arange(0, T+dt, dt)

    beta = np.zeros(N)
    beta[outputn] = 1e-6
    batch_size = 4
    random_init_times = 1
    inv_nudge_step = 1/beta[outputn]
    inv_batch_size = 1/batch_size
    inv_random_init_times = 1/random_init_times

    features_converted = np.array([
        [-np.pi,-np.pi],
        [np.pi,-np.pi],
        [-np.pi,np.pi],
        [np.pi,np.pi]
    ])/2

    labels_converted = np.array([-np.pi,np.pi,np.pi,-np.pi])/2

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

    #num_of_epochs = int(input("Input number of epochs:\t"))


    # training the network
    for epoch in range(num_of_epochs):

        if epoch%100 == 0:
            print(f"Starting epoch number {epoch}")

        weight_gradient = np.zeros((N,N))
        bias_gradient = np.zeros(N)
        bias_phase_gradient = np.zeros(N)
        
        distance_temp = []
        cost_temp = []
        for feature, label in zip(features_converted, labels_converted):
            
            target = np.zeros(N)
            target[outputn] = label

            # inserting input values and random initialization
            phases = np.array([np.random.uniform(-np.pi, np.pi) for _ in neurons])
            phases[inputn[0]] = feature[0]
            phases[inputn[1]] = feature[1]

            # calculating inference and inference energy
            thetas = odeint(kuramoto_oscillators, phases, times, args=(weights, biases, bias_phases, connections_neuronwise, inputn), full_output=0)

            # this should be allright, but maybe I understood something incorrectly
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
            distance_temp.append(1-np.cos(phases[outputn]-label))
            cost_temp.append(-np.log(1+np.cos(phases[outputn]-label)))
        
        distances.append(np.mean(distance_temp))
        costs.append(np.mean(cost_temp))

        # parameter updates
        weight_gradient *= inv_nudge_step * inv_batch_size * inv_random_init_times
        bias_gradient *= inv_nudge_step * inv_batch_size * inv_random_init_times
        bias_phase_gradient *= inv_nudge_step * inv_batch_size * inv_random_init_times

        # normalization (required for first few steps)
        
        #if np.linalg.norm(weight_gradient, ord=2)>5:
        if True:
            #print(f"gradient absolute size exceeded expected value\nL(∇W) = {np.linalg.norm(weight_gradient, ord=2)}")
            weight_gradient /= np.linalg.norm(weight_gradient,ord=2)
            bias_gradient /= np.linalg.norm(bias_gradient,ord=2)
            bias_phase_gradient /= np.linalg.norm(bias_phase_gradient,ord=2)

        """    
        if distances[-1] < 0.1:
            learning_rate = 0.03
        if distances[-1] < 0.01:
            learning_rate = 0.005
        """

        

        weights_matrix -= learning_rate * weight_gradient
        weights = weights_matrix[connections_neuronwise, np.arange(N)[:, None]]
        biases -= learning_rate * bias_gradient
        bias_phases -= learning_rate * bias_phase_gradient

    plt.plot(distances)
    plt.yscale("log")
    plt.grid(True)
    plt.title("Distances")
    plt.savefig("XOR_ONN_distances.png")
    plt.show()
    
    #do_save = input("Do you want to save the distances and accuracies y/n:\t")
    if do_save=="y" or do_save=="yes":
        append_to_csv("XOR_FNN_distances.csv", distances)

    return 0



if __name__ == "__main__":
    main()
