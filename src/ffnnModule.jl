
module basicNeuralNetwork

using FileIO, Images
using Plots
using Random
using LinearAlgebra
using Statistics
using DelimitedFiles



#neural network structs and functions
export DenseLayer, NeuralNetwork, Optimizer, ADAM_Optimizer, Simple_Subtract, forward_pass, relu, relu_derivative, softmax, softmax_derivative, linear_layer_derivative, linear_layer_identity, CostFunction, SquareCost, CrossEntropy, cost, derivative, backpropagation_pass, changing_weights, clip_gradients!

#extra functions
export normalize_matrix_to_one



function normalize_matrix_to_one(matrix::Matrix{Float64})
    # Compute the sum of absolute values of all elements in the matrix
    sum_abs = sum(abs.(matrix))
    
    if sum_abs == 0
        return matrix  # Return the original matrix if sum of absolute values is zero to avoid division by zero
    else
        return matrix / sum_abs  # Normalize each element in the matrix
    end
end

#=
Optdigits readout

functions for reading dataset, plotting and batching
=#


#=
Fully Connected Feed Forward Neural Network

functions and structures for constructing neural network, its inference, backpropagation and weight updates
=#

#shouldn't be mutable
mutable struct DenseLayer
    weights::Matrix{Float64}
    bias::Vector{Float64}
    activation::Function
    activation_derivative::Function
end

#initialization
function DenseLayer(input_dim::Int, output_dim::Int, activation::Function, activation_derivative::Function, add_bias::Bool=false)
    weights = randn(Float64, output_dim, input_dim) * sqrt(2.0 / input_dim)  # Ensure input_dim and output_dim are Integers
    bias = zeros(Float64, output_dim)
    if add_bias
        bias = randn(Float64, output_dim)
    end
    return DenseLayer(weights, bias, activation, activation_derivative)
end

#shouldn't be mutable
mutable struct NeuralNetwork
    layers::Vector{DenseLayer}
    #activations serve purpose of calculating gradient without repeating inference
    activations::Vector{Vector{Float64}}
end

#initialization
function NeuralNetwork(layers::DenseLayer...)
    return NeuralNetwork(collect(layers), Vector{Vector{Float64}}())
end

#this isn't pure inference pass, for inference another forward_pass function should be constructed that doesn't take into account storing activations for training
function forward_pass(model::NeuralNetwork, input::Vector{Float64})
    empty!(model.activations)
    x = input
    for layer in model.layers
        push!(model.activations, x)
        x = layer.activation(layer.weights * x .+ layer.bias) 
    end
    return x
end

#would work better if polymorphic (like cost function)
function relu(x::AbstractVector)
    return max.(0f0,x)
end

#matrix relu
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

#vector relu
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
    exp_x = exp.(x - maximum(x))
    return exp_x ./ sum(exp_x)
end

function softmax_derivative(x::AbstractArray)
    #softmax(x) would work correctly if this function wasn't working on softmax output already
    x = softmax(x)
    return x .* (1 .- x)
end

linear_layer_identity(x) = x 
linear_layer_derivative(x) = Ones(size(x))

abstract type CostFunction end
struct SquareCost <: CostFunction end
struct CrossEntropy <: CostFunction end

#interface generics
cost(::CostFunction, output, label) = error("Not implemented") #cost
derivative(::CostFunction, output, label) = error("Not implemented") #costs derivative

#functions for square cost
cost(::SquareCost, output, label) = sum((output - label) .^ 2)
derivative(::SquareCost, output, label) = 2 .* (output .- label)

function cost(::CrossEntropy, output, label)
    clamped_output = clamp.(output, eps(Float64), 1.0 - eps(Float64))
    return -dot(label, log.(clamped_output .+ 1e-12))
end

function derivative(::CrossEntropy, output, label)
    clamped_output = clamp.(output, eps(Float64), 1.0 - eps(Float64))
    return -label ./ (clamped_output .+ 1e-12)
end

function backpropagation_pass(model::NeuralNetwork, output::Vector{Float64}, label::Vector{Float64}, costModel::CostFunction)
    delta_L =  model.layers[end].activation_derivative(
        derivative(costModel, output, label)
    )
    delta_l_previous = delta_L
    gradients = Vector{Matrix{Float64}}()
    
    #iterating through full array
    for i in length(model.layers):-1:1
        layer = model.layers[i]
        # activation from l-1 layer - thats why before delta_l redefinition
        activation_vec = model.activations[i]
        grad = delta_l_previous * transpose(activation_vec) #I'm not sure about this matrix multiplication
        push!(gradients, grad)
        delta_l = layer.activation_derivative(transpose(layer.weights) * delta_l_previous)
        delta_l_previous = delta_l
    end
    reverse!(gradients)

    #changing weights
    return gradients
end

function backpropagation_bias_pass(model::NeuralNetwork, output::Vector{Float64}, label::Vector{Float64}, costModel::CostFunction)
    #=
    under development
    =#
    return 0
end

abstract type Optimizer end
struct Simple_Subtract <: Optimizer end

struct ADAM_Optimizer <: Optimizer
    first_moment::Vector{Matrix{Float64}}
    first_decay_rate::Float64
    second_moment::Vector{Matrix{Float64}}
    second_decay_rate::Float64
end

function ADAM_Optimizer(model::NeuralNetwork)
    zero_initialization_1 = [zeros(size(l.weights)) for l in model.layers]
    zero_initialization_2 = [zeros(size(l.weights)) for l in model.layers]
    return ADAM_Optimizer(zero_initialization_1, 0.9, zero_initialization_2, 0.999)
end

function changing_weights(optimizer_model::Optimizer,model::NeuralNetwork, gradients::Vector{Matrix{Float64}}, learning_rate::Float64)
    error("Function not implemented for Optimizer type")
end

function changing_weights(optimizer_model::Simple_Subtract, model::NeuralNetwork, gradients::Vector{Matrix{Float64}}, learning_rate::Float64, timestep::Int64)
    for i in 1:length(model.layers)
        model.layers[i].weights .-= (learning_rate .* gradients[i])
    end 
end

#there are problems with this implementation - mainly that gradients do not add up well with these operators
function changing_weights(optimizer_model::ADAM_Optimizer, model::NeuralNetwork, gradients::Vector{Matrix{Float64}}, learning_rate::Float64, timestep::Int64)
    epsilon = eps(Float64)

    for iter in 1:size(model.layers,1)
        optimizer_model.first_moment[iter] = optimizer_model.first_decay_rate .* optimizer_model.first_moment[iter] .+ (1-optimizer_model.first_decay_rate) .*gradients[iter]

        optimizer_model.second_moment[iter] = optimizer_model.second_decay_rate .* optimizer_model.second_moment[iter] .+ (1-optimizer_model.second_decay_rate) .*gradients[iter].^2

        cor_bias_first_moment = optimizer_model.first_moment[iter]./(1-optimizer_model.first_decay_rate^timestep)

        cor_bias_second_moment = optimizer_model.second_moment[iter]./(1-optimizer_model.second_decay_rate^timestep)

        model.layers[iter].weights .-= learning_rate .* cor_bias_first_moment./(sqrt.(cor_bias_second_moment).+epsilon)
    end
end


end


module mnistLoader


using FileIO, Images
using Plots
using Random
using LinearAlgebra
using Statistics
using DelimitedFiles

#Optdigits 8x8 helper functions exports
export optdigits_data_processing, plot_single_optdigits_digit, create_optdigits_batches_with_labels

#MNIST 28x28 helper functions exports
export read_ubyte_images, read_mnist_labels, create_batches_with_labels, plot_ubyte_image


function optdigits_data_processing(filename::String; do_batch = true, batch_size = 10)
    data = readdlm(filename, ',', Int64)
    features = Float64.(data[:,1:64])/16
    labels = one_hot_optdigits_labels(data[:,65])
    if do_batch
        feature_batches, label_batches = create_optdigits_batches_with_labels(features, labels, batch_size)
        return feature_batches, label_batches
    end
    return features, labels
end

function one_hot_optdigits_labels(labels::Vector{Int64})
    num_labels = size(labels, 1)
    labels_one_hot = Array{Float64}(undef, num_labels, 10)
    fill!(labels_one_hot, 0.0)

    for i in 1:num_labels
        label = labels[i]
        labels_one_hot[i, label + 1] = 1.0
    end
    return labels_one_hot
end

function plot_single_optdigits_digit(features::Matrix{Float64}, labels::Matrix{Float64}; index::Int64=1)
    pixel_values = features[index,:]
    image = reshape(pixel_values, 8, 8)
    label = labels[:,index]
    # Plot the image
    image = rotl90(image)
    plot = heatmap(image, color=:grays, axis=false, colorbar=false, aspect_ratio=:equal, title="Label: $(label)")
    xlims!(plot, (1,8)); ylims!(plot, (1,8))
    return plot
end

function create_optdigits_batches_with_labels(features::Matrix{Float64}, labels::Matrix{Float64}, batch_size::Int64)
    num_samples = size(features, 1)
    permuted_indices = randperm(num_samples)
    shuffled_images = features[permuted_indices, :]
    shuffled_labels = labels[permuted_indices, :]
    image_batches = [shuffled_images[i:min(i + batch_size - 1, num_samples), :] for i in 1:batch_size:num_samples]
    label_batches = [shuffled_labels[i:min(i + batch_size - 1, num_samples), :] for i in 1:batch_size:num_samples]
    #ensures all batches to have the same dimensions
    pop!(image_batches)
    pop!(label_batches)
    return image_batches, label_batches
end

#=
MNIST readout

functions for reading MNIST dataset from ubyte format, making it useful for implementation here, batching and visualising
=#

function read_ubyte_images(filename)
    open(filename) do file
        magic_number = read(file, UInt32)
        num_images = ntoh(read(file, UInt32))
        num_rows = ntoh(read(file, UInt32))
        num_cols = ntoh(read(file, UInt32))
        images = Array{Float64}(undef, num_rows, num_cols, num_images) # Use Float64 array
        for i = 1:num_images
            for j = 1:num_rows
                for k = 1:num_cols
                    single_image = read(file, UInt8)
                    # Convert to Float64 and normalize
                    images[j, k, i] = single_image / 255.0
                end
            end
        end
        return images
    end
end

function read_mnist_labels(filename)
    open(filename) do file
        magic_number = read(file, UInt32)
        num_labels = ntoh(read(file, UInt32))
        labels_uint8 = Array{UInt8}(undef, num_labels)
        read!(file, labels_uint8)
        # Initialize a matrix for one-hot encoded labels
        labels_one_hot = Array{Float64}(undef, 10, num_labels)
        fill!(labels_one_hot, 0.0)  # Fill with 0.0
        
        for i in 1:num_labels
            label = labels_uint8[i]
            labels_one_hot[label + 1, i] = 1.0  # MNIST labels are 0-9, Julia arrays are 1-indexed
        end
        
        return labels_one_hot
    end
end

function create_batches_with_labels(images, labels, batch_size)
    num_images = size(images, 3)
    image_size = size(images, 1) * size(images, 2)
    
    # Flatten images
    flattened_images = reshape(images, image_size, num_images)
    
    # Shuffle images and labels with the same permutation
    permuted_indices = randperm(num_images)
    shuffled_images = flattened_images[:, permuted_indices]
    
    # For one-hot encoded labels, we need to shuffle columns, not rows
    shuffled_labels = labels[:, permuted_indices]
    
    # Split into batches
    image_batches = [shuffled_images[:, i:min(i+batch_size-1, end)] for i in 1:batch_size:num_images]
    label_batches = [shuffled_labels[:, i:min(i+batch_size-1, end)] for i in 1:batch_size:num_images]

    return image_batches, label_batches
end

function plot_ubyte_image(filename, number)
    images = read_ubyte_images(filename)
    image = images[:, :, number]
    image = rotl90(transpose(image))
    plot = heatmap(image, color=:grays, colorbar=false, axis=false, aspect_ratio=:equal)
    xlims!(plot, (0,28)); ylims!(plot, (0,28))
    return plot
end

end