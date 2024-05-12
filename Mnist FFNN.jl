!/usr/bin/env julia
using Plots
using LinearAlgebra
using Statistics
using Random
using Printf
# including functions for loading dataset
include("./src/ffnnModule.jl")
using .mnistLoader
# including basic functions for the network (activation functions, loss functions, optimizers,...)
include("./src/ffnn_helper_functions.jl")
using .ffnnBarebonesModule

pi_sqrt = sqrt(pi)
train_file_path = "./mnist_red/optdigits.tra"
test_file_path = "./mnist_red/optdigits.tes"

function main()
  # creating batches
  batch_size = 64
  feature_batches, label_batches = optdigits_data_processing(train_file_path, batch_size=batch_size)
  test_features, test_labels = optdigits_data_processing(test_file_path, do_batch=false)
  train_features, train_labels = optdigits_data_processing(train_file_path, do_batch=false)

  # initializing layers with He rules
  W1 = randn(Float64, 128, 64) .* sqrt(2/64)
  B1 = fill(0.01, 128)
  activ1 = relu
  activ_der1 = relu_derivative

  W2 = randn(Float64, 64, 128) .* sqrt(2/128)
  B2 = fill(0.01, 64)
  activ2 = relu
  activ_der2 = relu_derivative

  W3 = randn(Float64, 10, 64) .* sqrt(2/64)
  B3 = fill(0.01, 10)
  activ3 = softmax
  activ_der3 = softmax_derivative

  # vectors for plotting
  costs = []
  accuracies = []
  train_accuracies = []


  # hyperparameters
  learning_rate = 0.01
  number_of_epochs = 100


  # training loop
  for epoch in 1:number_of_epochs

      # randomizing batches for each epoch to prevent overfitting
      feature_batches, label_batches = optdigits_data_processing(train_file_path, batch_size=batch_size)
      inference_accuracy = zeros(Float64, size(test_labels, 1))
      inference_accuracy_train = zeros(Float64, size(train_labels, 1))
      timestep = 1.

      # ADAM variables initialization
      vW3 = zeros(size(W3))
      vB3 = zeros(size(B3))
      sW3 = zeros(size(W3))
      sB3 = zeros(size(B3))

      vW2 = zeros(size(W2))
      vB2 = zeros(size(B2))
      sW2 = zeros(size(W2))
      sB2 = zeros(size(B2))

      vW1 = zeros(size(W1))
      vB1 = zeros(size(B1))
      sW1 = zeros(size(W1))
      sB1 = zeros(size(B1))

      # looping through minibatches
      for (feature_batch, label_batch) in zip(feature_batches, label_batches)

          mean_gradient3 = zeros(size(W3))
          mean_gradient2 = zeros(size(W2))
          mean_gradient1 = zeros(size(W1))

          mean_gradient_bias3 = zeros(size(B3))
          mean_gradient_bias2 = zeros(size(B2))
          mean_gradient_bias1 = zeros(size(B1))

          temp_costs = []

          for j in 1:batch_size
              feature = feature_batch[j, :]
              label = label_batch[j, :]
              
              # forward pass
              z1 = W1 * feature + B1
              a1 = activ1(z1) 

              z2 = W2 * a1 + B2
              a2 = activ2(z2)

              z3 = W3 * a2
              output = activ3(z3)

              # loss function
              loss = cross_entropy_loss(output, label)
              # loss_derivative = cross_entropy_loss_derivative(output, label) - very low accuracy for this particular problem
              # loss_derivative = activ3(output - label)
              loss_derivative = activ3(cross_entropy_loss_derivative(output, label))

              # backpropagation pass
              #delta_3 = activ_der3(loss_derivative)
              delta_3 = loss_derivative .* (1 .- loss_derivative)
              delta_2 = activ_der2(transpose(W3) * delta_3)
              delta_1 = activ_der1(transpose(W2) * delta_2)

              mean_gradient3 .+= delta_3 * transpose(a2)
              mean_gradient2 .+= delta_2 * transpose(a1)
              mean_gradient1 .+= delta_1 * transpose(feature)

                  # bias derivative of cost function equals delta * 1
              mean_gradient_bias3 .+= delta_3
              mean_gradient_bias2 .+= delta_2
              mean_gradient_bias1 .+= delta_1

              push!(temp_costs, loss)
          end

          mean_gradient3 ./= batch_size
          mean_gradient2 ./= batch_size
          mean_gradient1 ./= batch_size

          mean_gradient_bias3 ./= batch_size
          mean_gradient_bias2 ./= batch_size
          mean_gradient_bias1 ./= batch_size

          # ADAM optimization reinitialized for each epoch (it could be good idea to test it with one initialization for whole training)
          mean_gradient3, mean_gradient_bias3, vW3, vB3, sW3, sB3 = update_with_ADAM(mean_gradient3, mean_gradient_bias3, vW3, sW3, vB3, sB3, timestep)

          mean_gradient2, mean_gradient_bias2, vW2, vB2, sW2, sB2 = update_with_ADAM(mean_gradient2, mean_gradient_bias2, vW2, sW2, vB2, sB2, timestep)

          mean_gradient1, mean_gradient_bias1, vW1, vB1, sW1, sB1 = update_with_ADAM(mean_gradient1, mean_gradient_bias1, vW1, sW1, vB1, sB1, timestep)

          timestep += 1
          # changing weights
          W3 -= (learning_rate * mean_gradient3)
          B3 -= (learning_rate * mean_gradient_bias3)

          W2 -= (learning_rate * mean_gradient2)
          B2 -= (learning_rate * mean_gradient_bias2)

          W1 -= (learning_rate * mean_gradient1)
          B1 -= (learning_rate * mean_gradient_bias1)
          
          push!(costs, sum(temp_costs)/batch_size)
      end
      # calculate predictions for test dataset
      for j in 1:size(test_labels, 1)
          feature = test_features[j,:]
          label = test_labels[j,:]
          z1 = W1 * feature + B1
          a1 = activ1(z1) 
          z2 = W2 * a1 + B2
          a2 = activ2(z2)
          z3 = W3 * a2
          output = activ3(z3)
          inference_accuracy[j] = Float64(argmax(output) == argmax(label)) 
      end
      push!(accuracies, mean(inference_accuracy))
      
      #calculate predictions for training dataset
      for j in 1:size(train_labels, 1)
          feature = train_features[j,:]
          label = train_labels[j,:]
          z1 = W1 * feature + B1
          a1 = activ1(z1) 
          z2 = W2 * a1 + B2
          a2 = activ2(z2)
          z3 = W3 * a2
          output = activ3(z3)
          inference_accuracy_train[j] = Float64(argmax(output) == argmax(label)) 
      end
      push!(train_accuracies, mean(inference_accuracy_train))
      
      println("epoch: $epoch \t accuracy: $(@sprintf("%.4f%%", 100 * accuracies[end])) \t train data accuracy: $(@sprintf("%.4f%%", 100 * train_accuracies[end]))")
  end
  end

  # plotting
  p_costs = plot(costs, title="Costs Over Updates", label="Cost", xlabel="Epoch", ylabel="Cost")
  p_accuracy = plot(accuracies, title="Accuracy Over Epochs", label="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy [%]")
  plot!(p_accuracy, train_accuracies, label="Training Accuracy")

  combined_plot = plot(p_costs, p_accuracy, layout=(2, 1), size=(600, 800))
  savefig(combined_plot, "combined_plot.png")
  display(combined_plot)


if abspath(PROGRAM_FILE) == @__FILE__
  main()
end



