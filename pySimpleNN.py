# -*- coding: utf-8 -*-
# Copyright Jesper Larsson 2018

"""
CONFIGURATION
"""
target_accuracy = 99.99
training_mode = True
max_training_time = 60 * 5
starting_alpha = 10          # relative change on each iteration, lower values makes us less likely to "overshoot" our target values, but it can take longer to get close to the result we want
dropout_percent = 0.1        # dropout rate
hidden_dim = 4               # dimensions in hidden layers
random_seed = 1


"""
SETUP
"""
import numpy as np
from math import *
from datetime import *

np.random.seed(random_seed)
alpha = starting_alpha


"""
TEST DATA
"""
# Each row is a training sample of 3 input nodes each (3x1 matrix)
input_tests = np.array([
                [0,0,1],   
                [0,1,1],
                [1,0,1],
                [1,1,1],
                [1,0,1],
                [0,0,1],
                [1,0,1],
                [1,0,1],
                ])
# Expected values, T function (transpose) switches columns for rows
expected_outputs = np.array([[0,0,1,1,1,0,1,1]]).T

"""
FUNCTIONS AND CLASSES
"""
# Sigmoid activation function and its derivative (maps any value to non-linear space between 0 and 1)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_slope(x):
    return x*(1-x)
def value_to_slope(output):
    return output*(1-output) # output to slope


"""
CODE START
"""
# Randomize synapses (weight) layers, mean = 0
syn_0 = 2*np.random.random((3,4)) - 1 # 3 inputs (from test data) => 4 outputs (next layer)
syn_1 = 2*np.random.random((4,1)) - 1 # 4 inputs, 1 output (final value)

# Print starting synapses
print("===== INITIAL (RANDOMIZED) WEIGHTS")
print(str(syn_0))
print(str(syn_1))
print("")

# First layer is our input training data
layer_0 = input_tests

# Mainloop
print("===== NETWORK STARTED")
start_time = datetime.now()
last_trace = 0
iteration = 0
while(True):
    iteration += 1
    
    layer_1 = sigmoid(np.dot(layer_0,syn_0))
    layer_2 = sigmoid(np.dot(layer_1,syn_1))

    # Hinton's dropout (only when training) - discards some values at random to avoid descending multiple nodes into the same (local) minimum
    if (training_mode):
        layer_1 *= np.random.binomial([np.ones((len(layer_0),hidden_dim))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

    # Calculate output differences vs expected errors ("Confidence weighted error")
    l2_error = expected_outputs - layer_2
    l2_delta = l2_error * sigmoid_slope(layer_2)

    l1_error = l2_delta.dot(syn_1.T)
    l1_delta = l1_error * sigmoid_slope(layer_1)
    
    # Nudge weights
    syn_1 += alpha * ( layer_1.T.dot(l2_delta) )
    syn_0 += alpha * ( layer_0.T.dot(l1_delta) )

    # Print every now and then
    current_accuracy = 100 * (1 - (np.mean(np.abs(l2_error))))
    uptime = (datetime.now() - start_time).total_seconds()
    if (int(uptime) > int(last_trace)):
        print("  Iteration " + str(iteration) + " accuracy: " + str(current_accuracy) + "%")
        last_trace = uptime
    
    # We're done
    if (current_accuracy >= target_accuracy):
        print("  Achieved target " + str(target_accuracy) + "% accuracy after " + str(iteration) +
              " training steps with average test accuracy: " + str(current_accuracy) +
              "% in " + str(uptime) + "s")
        break

    # Timeout
    if (uptime > max_training_time):
        print("  TIMEOUT after " + str(iteration) +
              " training steps with average test accuracy: " + str(current_accuracy) +
              "% in " + str(uptime) + "s")
        break


output_layer = layer_2
print("")

# Print details
print("===== CALCULATED WEIGHTS")
print(str(syn_0))
print(str(syn_1))
print("")

print("===== FINAL RESULTS")
for i in xrange(len(l1)):
    output_value = output_layer[i][0] # 0 because we only have a single output result
    expected_value = expected_outputs[i][0]
    value_diff = fabs(expected_value - output_value)
    print("  Test " + (str(i + 1)) + ". " + str(output_value) + ". Expected " + str(expected_value) + ". Diff = " + str(value_diff))
print("")

