# -*- coding: utf-8 -*-
# Copyright Jesper Larsson 2018

# TODO:
# OK - basic statistics
# OK - multiple layers, needs backpropogation
# multiple output nodes
# synapse bias
# mini-batching
# batch normalization
# hidden layers
# read test data from file (xml/json?)


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
CODE START
"""
import numpy as np
from math import *

np.random.seed(1)

# Sigmoid activation function and its derivative (maps any linear value to non-linear space between 0 and 1)
def nonlin(x):
    return 1/(1+np.exp(-x))
def nonlin_derive(x):
    return x*(1-x)

# Randomize synapses (weight) layers
syn_0 = 2*np.random.random((3,4)) - 1 # 3 inputs (from test data) => 4 outputs (next layer)
syn_1 = 2*np.random.random((4,1)) - 1 # 4 inputs, 1 output (final value)

# Print starting synapses
print("===== RANDOM WEIGHTS")
print("Synapse 0:")
print(str(syn_0))
print("Synapse 1:")
print(str(syn_1))
print("=====")

# First layer is our input training data
l0 = input_tests

for iter in xrange(100000):
    # Calculate values and non-linearize them
    l1 = nonlin(np.dot(l0,syn_0))
    l2 = nonlin(np.dot(l1,syn_1))

    # Calculate output differences vs expected errors (ie the "Confidence weighted error")
    l2_error = expected_outputs - l2
    l2_delta = l2_error*nonlin_derive(l2)

    # Remaining layer errors
    l1_error = l2_delta.dot(syn_1.T)
    l1_delta = l1_error * nonlin_derive(l1)
    
    # Nudge synapse weights individually
    syn_1 += l1.T.dot(l2_delta)
    syn_0 += np.dot(l0.T,l1_delta)

    # Print every now and then
    if iter % 1000 == 0:
        print("  Iteration " + str(iter) + " accuracy: " + str(100 * (1 - (np.mean(np.abs(l1_error))))) + "%")

print("===== CALCULATED WEIGHTS")
print("Synapse 0:")
print(str(syn_0))
print("Synapse 1:")
print(str(syn_1))
print("=====")

# Print details,
print("===== FINAL RESULTS")
accumilator = 0.0
output_layer = l2
for i in xrange(len(l1)):
    output_value = output_layer[i][0] # 0 because we only have a single output result
    expected_value = expected_outputs[i][0]
    value_diff = fabs(expected_value - output_value)
    accumilator += value_diff
    print("  Test " + (str(i + 1)) + ". " + str(output_value) + ". Expected " + str(expected_value) + ". Diff = " + str(value_diff))
    
accuracy = 100 * (1 - (accumilator / len(l1)))
print("Network final accuracy: " + str(accuracy) + "%")

