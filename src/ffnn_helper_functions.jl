module ffnnBarebonesModule

using Plots
using LinearAlgebra
using Statistics
using Random
using Printf

export relu, relu_derivative, softmax, softmax_derivative, cross_entropy_loss, cross_entropy_loss_derivative, normalize_matrix_to_one, update_with_ADAM

# activation functions
function relu(x::AbstractVector)
    return max.(0f0,x)
end

# matrix relu
function relu_derivative(x::AbstractMatrix)
    rows, cols = size(x)
    answer = Array{Float64}(undef, rows, cols)
    for i in 1:rows
        for j in 1:cols
            if x[i, j] < 0
                answer[i, j] = 0.0
            else
                answer[i, j] = 1.0
            end
        end
    end
    return answer
end

# vector relu
function relu_derivative(x::AbstractVector)
    answer = []
    for val in x
        if val < 0
            push!(answer,Float64(0.))
        else
            push!(answer,Float64(1.))
        end
    end
    return answer
end

function softmax(x::AbstractVector)
    # subtracting max value prevents overflow
    exp_x = exp.(x .- maximum(x))
    return exp_x ./ sum(exp_x)
end


function softmax_derivative(x::AbstractArray)
    x = softmax(x)
    return x .* (1 .- x)
end

function cross_entropy_loss(output, label)
    return - dot(label, log.(output)) + dot((1 .- label), log.(1 .- output))
end

function cross_entropy_loss_derivative(output, label)
    - label ./ output + (1 .- label) ./ (1 .- output)
end

function normalize_matrix_to_one(matrix::Matrix{Float64})
    # Compute the sum of absolute values of all elements in the matrix
    sum_abs = sum(abs.(matrix))

    if sum_abs == 0
        return matrix  # Return the original matrix if sum of absolute values is zero to avoid division by zero
    else
        return matrix / sum_abs  # Normalize each element in the matrix
    end
end

function normalize_matrix_to_one(matrix::Vector{Float64})
    # Compute the sum of absolute values of all elements in the matrix
    sum_abs = sum(abs.(matrix))

    if sum_abs == 0
        return matrix  # Return the original matrix if sum of absolute values is zero to avoid division by zero
    else
        return matrix / sum_abs  # Normalize each element in the matrix
    end
end

# adam optimization (for single layer)
function update_with_ADAM(weight_grads::Matrix{Float64}, bias_grads::Vector{Float64}, vw::Matrix{Float64}, sw::Matrix{Float64}, vb::Vector{Float64}, sb::Vector{Float64}, t::Float64, beta1::Float64 = 0.9, beta2::Float64 = 0.999)
    #=
    l - depth of a network
    v, s - preinitialized matrix-like hyperparameters
    t - number of times update was performed in current Epoch
    beta1, beta2 - predefined float hyperparameters
    =#
    vw_corrected = zeros(size(vw))
    sw_corrected = zeros(size(sw))

    vw_corrected = zeros(size(vb))
    sw_corrected = zeros(size(sb))

    @. vw = beta1 * vw + (1 - beta1) * weight_grads
    @. vb = beta1 * vb + (1 - beta1) * bias_grads

    vw_corrected = vw ./ (1 - beta1 ^ t)
    vb_corrected = vb ./ (1 - beta1 ^ t)

    @. sw = beta2 * sw + (1 - beta2) * sqrt(weight_grads)
    @. sb = beta2 * sb + (1 - beta2) * sqrt(bias_grads)

    sw_corrected = sw ./ (1 - beta2 ^ t)
    sb_corrected = sb ./ (1 - beta2 ^ t)

    weight_gradient_adam = vw ./ sqrt.(sw .+ eps(Float64))
    bias_gradient_adam = vb_corrected ./ sqrt.(sb_corrected .+ eps(Float64))

    return weight_gradient_adam, bias_gradient_adam, vw, vb, sw, sb
end
