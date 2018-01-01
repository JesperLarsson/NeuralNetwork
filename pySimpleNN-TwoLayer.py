# -*- coding: utf-8 -*-
# Copyright Jesper Larsson 2018

# TODO:
# multiple layers, needs backpropogation
# multiple output nodes
# synapse bias
# mini-batching
# batch normalization
# hidden layers
# read test data from file (xml/json?)


import numpy as np
from math import *

np.random.seed(1)

# Sigmoid function and its derivative (maps any linear value to non-linear space between 0 and 1)
def nonlin(x):
    return 1/(1+np.exp(-x))
def nonlin_derive(x):
    return x*(1-x)

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

# Randomize synapse layer 0 to start
l0_to_l1_weights = 2*np.random.random((3,1)) - 1
print("Starting weights:")
print(str(l0_to_l1_weights))

# First layer is our input data
l0 = input_tests

# Mainloop
for iter in xrange(100000):
    # Calculate second layer (our output layer)
    l1 = nonlin(np.dot(l0,l0_to_l1_weights))

    # "Confidence weighted error"
    l1_error = expected_outputs - l1
    
    # Nudge synapse weights individually
    l1_delta = l1_error * nonlin_derive(l1)
    l0_to_l1_weights += np.dot(l0.T,l1_delta)

    # Print every now and then
    if iter % 1000 == 0:
        print("  Iteration " + str(iter) + " accuracy: " + str(100 * (1 - (np.mean(np.abs(l1_error))))) + "%")

print("===== FINAL WEIGHTS")
print("Synapse 0:")
print(str(l0_to_l1_weights))
print("=====")

# Print details,
print("RESULTS:")
accumilator = 0.0
for i in xrange(len(l1)):
    output_value = l1[i][0] # 0 because we only have a single output node
    expected_value = expected_outputs[i][0]
    value_diff = fabs(expected_value - output_value)
    accumilator += value_diff
    print("  Test " + (str(i + 1)) + ". " + str(output_value) + ". Expected " + str(expected_value) + ". Diff = " + str(value_diff))
    
accuracy = 100 * (1 - (accumilator / len(l1)))
print("Network final accuracy: " + str(accuracy) + "%")

