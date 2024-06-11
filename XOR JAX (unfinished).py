import jax.numpy as jnp
from jax import jit, grad
from jax.experimental.ode import odeint
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
# importing custom functions
sys.path.append('./src')
from ONNModule import clear_csv, append_to_csv


# JAX defined kuramoto oscillators and nudge
@jit
def kuramoto_oscillators_jax(theta, t, K, h, psi, coupled_theta, inputn):
    """
    Computes the derivative of theta for the Kuramoto oscillators.

    :param theta: jnp.array, current phase angles
    :param t: float, current time (not used in this function)
    :param K: jnp.array, coupling matrix
    :param h: jnp.array, external driving strengths
    :param psi: jnp.array, external driving phases
    :param coupled_theta: jnp.array, indices of coupled oscillators
    :param inputn: jnp.array, indices of independent oscillators
    :return: jnp.array, derivatives of theta
    """
    coupled_values = jnp.take(theta, coupled_theta, axis=0)
    dtheta_dt = jnp.zeros_like(theta)
    for neuron in range(len(theta)):
        is_independent = jnp.any(neuron == inputn)
        sin_diffs = jnp.sin(theta[neuron] - coupled_values[neuron])
        sin_external = jnp.sin(theta[neuron] - psi[neuron])
        dtheta_dt = dtheta_dt.at[neuron].set(
            jnp.where(
                is_independent,
                0,
                -jnp.sum(sin_diffs * K[neuron]) - h[neuron] * sin_external
            )
        )
    return dtheta_dt

@jit
def kuramoto_oscillators_nudge_jax(theta, t, K, h, psi, coupled_theta, inputn, beta, target):
    """
    Computes the derivative of theta for the Kuramoto oscillators with a nudge term.

    :param theta: jnp.array, current phase angles
    :param t: float, current time (not used in this function)
    :param K: jnp.array, coupling matrix
    :param h: jnp.array, external driving strengths
    :param psi: jnp.array, external driving phases
    :param coupled_theta: jnp.array, indices of coupled oscillators
    :param inputn: jnp.array, indices of independent oscillators
    :param beta: jnp.array, nudge strengths
    :param target: jnp.array, target phase angles
    :return: jnp.array, derivatives of theta
    """
    coupled_values = jnp.take(theta, coupled_theta, axis=0)
    dtheta_dt = jnp.zeros_like(theta)
    for neuron in range(len(theta)):
        is_independent = jnp.any(neuron == inputn)
        sin_diffs = jnp.sin(theta[neuron] - coupled_values[neuron])
        sin_external = jnp.sin(theta[neuron] - psi[neuron])
        nudge_term = - beta[neuron] * jnp.sin(theta[neuron] - target[neuron]) / (jnp.cos(theta[neuron] - target[neuron]) + 1 + 1e-8)
        dtheta_dt = dtheta_dt.at[neuron].set(
            jnp.where(
                is_independent,
                0,
                -jnp.sum(sin_diffs * K[neuron]) - h[neuron] * sin_external + nudge_term
            )
        )
    return dtheta_dt



def cosine_similarity(a, b):
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))

def check_accuracy(predictions, labels):
    accuracy_temp = []

    for pred, label in zip(predictions, labels):
        # Compute cosines of predictions and labels
        pred_cos = [jnp.cos(pred[0]), jnp.cos(pred[1])]
        label_cos = [jnp.cos(label[0]), jnp.cos(label[1])]

        # Compute cosine similarity
        cos_sim = cosine_similarity(pred_cos, label_cos)

        # Update accuracy_temp based on cosine similarity
        if cos_sim > 0:  # Cosines are closer
            accuracy_temp.append(1)
        else:  # Cosines are further
            accuracy_temp.append(0)

    return accuracy_temp

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

    if N < 3:
        return 1

    # defining neurons and connections
    neurons = np.arange(0, N, 1)
    outputn = 2
    inputn = [0, 1]
    connections_neuronwise = np.array([[element for element in neurons if element != neuron] for neuron in neurons])

    weights_matrix = np.zeros((N, N))
    for i in range(1, N):
        for j in range(0, i):
            weights_matrix[i][j] = np.random.normal(loc=0, scale=1)
            weights_matrix[j][i] = weights_matrix[i][j]

    weights = jnp.array(weights_matrix[connections_neuronwise, np.arange(N)[:, None]])
    biases = jnp.array([np.random.uniform(-0.5, 0.5) for _ in neurons])
    bias_phases = jnp.array([np.random.uniform(-np.pi, np.pi) for _ in neurons])
    init_phases = jnp.array([np.random.uniform(-np.pi, np.pi) for _ in neurons])
    phases = jnp.copy(init_phases)

    distances = []
    costs = []

    T = 100
    dt = 0.01
    times = jnp.arange(0, T + dt, dt)

    beta = jnp.zeros(N)
    beta = beta.at[outputn].set(1e-6)
    batch_size = 4
    random_init_times = 1
    inv_nudge_step = 1 / beta[outputn]
    inv_batch_size = 1 / batch_size
    inv_random_init_times = 1 / random_init_times

    features_converted = jnp.array([
        [-jnp.pi, -jnp.pi],
        [jnp.pi, -jnp.pi],
        [-jnp.pi, jnp.pi],
        [jnp.pi, jnp.pi]
    ]) / 2

    labels_converted = jnp.array([-jnp.pi, jnp.pi, jnp.pi, -jnp.pi]) / 2

    gradient_weights_forward = jnp.zeros((N, N))
    gradient_weights_backward = jnp.zeros((N, N))
    weight_gradient = jnp.zeros((N, N))

    gradient_biases_forward = jnp.zeros(N)
    gradient_biases_backward = jnp.zeros(N)
    bias_gradient = jnp.zeros(N)

    gradient_bias_phases_forward = jnp.zeros(N)
    gradient_bias_phases_backward = jnp.zeros(N)
    bias_phase_gradient = jnp.zeros(N)

    learning_rate = 0.1

    for epoch in range(num_of_epochs):
        if epoch % 100 == 0:
            print(f"Starting epoch number {epoch}")

        weight_gradient = jnp.zeros((N, N))
        bias_gradient = jnp.zeros(N)
        bias_phase_gradient = jnp.zeros(N)

        distance_temp = []
        cost_temp = []

        for feature, label in zip(features_converted, labels_converted):
            target = jnp.zeros(N)
            target = target.at[outputn].set(label)

            phases = jnp.array([np.random.uniform(-jnp.pi, jnp.pi) for _ in neurons])
            phases = phases.at[inputn[0]].set(feature[0])
            phases = phases.at[inputn[1]].set(feature[1])

            thetas = odeint(kuramoto_oscillators_jax, phases, times, weights, biases, bias_phases, connections_neuronwise, inputn)

            phases = thetas[-1]

            for i in range(1, N):
                for j in range(0, i):
                    gradient_weights_forward = gradient_weights_forward.at[i, j].set(-jnp.cos(phases[i] - phases[j]))
                    gradient_weights_forward = gradient_weights_forward.at[j, i].set(gradient_weights_forward[i, j])
            for neuron in neurons:
                gradient_biases_forward = gradient_biases_forward.at[neuron].set(-jnp.cos(phases[neuron] - bias_phases[neuron]))
                gradient_bias_phases_forward = gradient_bias_phases_forward.at[neuron].set(-biases[neuron] * jnp.sin(phases[neuron] - bias_phases[neuron]))

            thetas_back = odeint(kuramoto_oscillators_nudge_jax, phases, times, weights, biases, bias_phases, connections_neuronwise, inputn, beta, target)

            phases = thetas_back[-1]

            for i in range(1, N):
                for j in range(0, i):
                    gradient_weights_backward = gradient_weights_backward.at[i, j].set(-jnp.cos(phases[i] - phases[j]))
                    gradient_weights_backward = gradient_weights_backward.at[j, i].set(gradient_weights_backward[i, j])
            for neuron in neurons:
                gradient_biases_backward = gradient_biases_backward.at[neuron].set(-jnp.cos(phases[neuron] - bias_phases[neuron]))
                gradient_bias_phases_backward = gradient_bias_phases_backward.at[neuron].set(-biases[neuron] * jnp.sin(phases[neuron] - bias_phases[neuron]))

            weight_gradient += gradient_weights_backward - gradient_weights_forward
            bias_gradient += gradient_biases_backward - gradient_biases_forward
            bias_phase_gradient += gradient_bias_phases_backward - gradient_bias_phases_forward

            distance_temp.append(1 - jnp.cos(phases[outputn] - label))
            cost_temp.append(-jnp.log(1 + jnp.cos(phases[outputn] - label)))

        distances.append(np.mean(distance_temp))
        costs.append(np.mean(cost_temp))

        weight_gradient *= inv_nudge_step * inv_batch_size * inv_random_init_times
        bias_gradient *= inv_nudge_step * inv_batch_size * inv_random_init_times
        bias_phase_gradient *= inv_nudge_step * inv_batch_size * inv_random_init_times

        if True:
            weight_gradient /= jnp.linalg.norm(weight_gradient, ord=2)
            bias_gradient /= jnp.linalg.norm(bias_gradient, ord=2)
            bias_phase_gradient /= jnp.linalg.norm(bias_phase_gradient, ord=2)

        weights_matrix -= learning_rate * weight_gradient
        weights = jnp.array(weights_matrix[connections_neuronwise, np.arange(N)[:, None]])
        biases -= learning_rate * bias_gradient
        bias_phases -= learning_rate * bias_phase_gradient

    plt.plot(distances)
    plt.yscale("log")
    plt.grid(True)
    plt.title("Distances")
    plt.savefig("XOR_ONN_distances.png")
    plt.show()

    if do_save == "y" or do_save == "yes":
        append_to_csv("XOR_FNN_distances.csv", distances)

    predictions = [[0, 1], [1, 0], [1, 1], [0, 0]]  # Replace with your NN predictions
    labels = [[0, 1], [1, 0], [0, 1], [1, 0]]  # Replace with your actual labels (0 or 1)
    label_angles = [[-jnp.pi / 2 if l == 0 else jnp.pi / 2 for l in label] for label in labels]

    accuracy_temp = check_accuracy(predictions, label_angles)
    print(f"Accuracy: {sum(accuracy_temp) / len(accuracy_temp) * 100:.2f}%")

if __name__ == "__main__":
    main()
