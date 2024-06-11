module basicONN

using FileIO, Images
using Plots
using Random
using LinearAlgebra
using Statistics
using DelimitedFiles


function kuramoto_oscillators(theta, K, h, psi, coupled_theta, inputn)
    #=
    Computes the derivative of theta for the Kuramoto oscillators.
    
    :param theta: Array, current phase angles
    :param t: Float64, current time (not used in this function)
    :param K: Array, coupling matrix
    :param h: Array, external driving strengths
    :param psi: Array, external driving phases
    :param coupled_theta: Array, indices of coupled oscillators
    :param inputn: Array, indices of independent oscillators
    :return: Array, derivatives of theta
    =#
    coupled_values = theta[coupled_theta]
    dtheta_dt = zeros(theta)
    for neuron in 1:length(theta)
        if neuron in inputn
            dtheta_dt[neuron] = 0
        else
            sin_diffs = sin.(theta[neuron] .- coupled_values[neuron])
            sin_external = sin(theta[neuron] - psi[neuron])
            dtheta_dt[neuron] = - sum(sin_diffs .* K[neuron, :]) - h[neuron] * sin_external
        end
    end
    return dtheta_dt
end

function kuramoto_oscillators_nudge(theta, K, h, psi, coupled_theta, inputn, beta, target)
    #=
    Computes the derivative of theta for the Kuramoto oscillators with a nudge term.
    
    :param theta: Array, current phase angles
    :param t: Float64, current time (not used in this function)
    :param K: Array, coupling matrix
    :param h: Array, external driving strengths
    :param psi: Array, external driving phases
    :param coupled_theta: Array, indices of coupled oscillators
    :param inputn: Array, indices of independent oscillators
    :param beta: Array, nudge strengths
    :param target: Array, target phase angles
    :return: Array, derivatives of theta
    =#
    coupled_values = theta[coupled_theta]
    dtheta_dt = zeros(theta)
    for neuron in 1:length(theta)
        if neuron in inputn
            dtheta_dt[neuron] = 0
        else
            sin_diffs = sin.(theta[neuron] .- coupled_values[neuron])
            sin_external = sin(theta[neuron] - psi[neuron])
            dtheta_dt[neuron] = - sum(sin_diffs .* K[neuron, :]) - h[neuron] * sin_external - beta[neuron] * sin(theta[neuron] - target[neuron]) / (cos(theta[neuron] - target[neuron]) + 1 + 1e-8)
        end
    end
    return dtheta_dt
end

# ode solving wrapper
GIRK

# equating matrices and vectors to 0

# calculating gradients

# updating weights and biases

# distance and accuracy calculation

# check if gradient exceeds threshold