import numpy as np
import csv

def kuramoto_oscillators(theta, t, K, h, psi, coupled_theta, inputn):
    """
    Computes the derivative of theta for the Kuramoto oscillators.
    
    :param theta: np.array, current phase angles
    :param t: float, current time (not used in this function)
    :param K: np.array, coupling matrix
    :param h: np.array, external driving strengths
    :param psi: np.array, external driving phases
    :param coupled_theta: np.array, indices of coupled oscillators
    :param inputn: list, indices of independent oscillators
    :return: np.array, derivatives of theta
    """
    coupled_values = theta[coupled_theta]
    dtheta_dt = np.zeros_like(theta)
    for neuron in range(len(theta)):
        if neuron in inputn:  # handles independent nodes
            dtheta_dt[neuron] = 0
        else:
            sin_diffs = np.sin(theta[neuron] - coupled_values[neuron])
            sin_external = np.sin(theta[neuron] - psi[neuron])
            dtheta_dt[neuron] = - np.sum(sin_diffs * K[neuron]) - h[neuron]*sin_external
    return dtheta_dt

def kuramoto_oscillators_nudge(theta, t, K, h, psi, coupled_theta, inputn, beta, target):
    """
    Computes the derivative of theta for the Kuramoto oscillators with a nudge term.
    
    :param theta: np.array, current phase angles
    :param t: float, current time (not used in this function)
    :param K: np.array, coupling matrix
    :param h: np.array, external driving strengths
    :param psi: np.array, external driving phases
    :param coupled_theta: np.array, indices of coupled oscillators
    :param inputn: list, indices of independent oscillators
    :param beta: np.array, nudge strengths
    :param target: np.array, target phase angles
    :return: np.array, derivatives of theta
    """
    coupled_values = theta[coupled_theta]
    dtheta_dt = np.zeros_like(theta)
    for neuron in range(len(theta)):
        if neuron in inputn:  # handles independent nodes
            dtheta_dt[neuron] = 0
        else:
            sin_diffs = np.sin(theta[neuron] - coupled_values[neuron])
            sin_external = np.sin(theta[neuron] - psi[neuron])
            dtheta_dt[neuron] = - np.sum(sin_diffs * K[neuron]) - h[neuron]*sin_external - beta[neuron] * np.sin(theta[neuron]-target[neuron])/(np.cos(theta[neuron]-target[neuron])+1+1e-8)
    return dtheta_dt

def clear_csv(filename):
    """
    Clears the contents of the CSV file.
    
    :param filename: str, name of the CSV file to clear
    """
    with open(filename, mode='w', newline='') as file:
        pass

def append_to_csv(filename, data_array):
    """
    Appends a single array (row) to the CSV file.
    
    :param filename: str, name of the CSV file
    :param data_array: list, the array to be appended as a row in the CSV
    """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_array)
