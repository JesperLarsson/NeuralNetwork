# -*- coding: utf-8 -*-
# Copyright Jesper Larsson 2018, github.com/JesperLarsson

"""
CONFIGURATION
"""
# Basic
target_accuracy = 99.9
training_mode = True
randomization_mode = False
max_training_time = 60 * 5

# Algorithm
starting_alpha = 10 # relative change on each iteration, lower values makes us less likely to "overshoot" our target values, but it can take longer to get close to the result we want
dropout_percent = 0.1

# Network layout
hidden_layers = 1 # number of hidden layers
hidden_dimensions = 4
input_dimensions = 3 # input/output dimensions are inherintly tied to the training data
output_dimensions = 1



"""
SETUP
"""
try:
      import numpy as np
except ImportError:
   try:
      # Try to install it via package manager
      print("Installing numpy...")
      import pip
      pip.main(['install', "numpy"])
      import numpy as np
   except ImportError:
      #  install using "pip install numpy" when in path C:\Python3\Scripts
      print("Unable to find numpy, please install it manually by running 'pip install numpy' in your python scripts folder")
      input("Press enter to exit")
      sys.exit(2)
from math import *
from datetime import *

np.random.seed(1) # easier debugging without true seed

alpha = starting_alpha

layer_count = hidden_layers + 2 # + input + output

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


"""    
# Input training data sample
class TrainingCase:
    input_values = None
    expected_result = None

    def __init__(self, _input_values, _expected_result):
        input_values = _input_values
        expected_result = _expected_result
"""

class Synapse:
   weights = None
   name = "N/A"

   def __init__(self, synapse_input_dimensions, synapse_output_dimensions, synapse_name):
      # Randomize synapse weights with mean value = 0
      self.weights = 2*np.random.random((synapse_input_dimensions, synapse_output_dimensions)) - 1
      self.name = synapse_name
      print("  Created synapse " + self.name + "(" + str(synapse_input_dimensions) + "/" + str(synapse_output_dimensions) + " dimensions)")


class Layer:
      neurons = None
      next_layer = None
      name = "N/A"

      def __init__(self, layer_name):
         print("  Created layer " + layer_name)
         self.name = layer_name
    

class Network:
   layers = []
   synapses = []

   # Performs the actual network "magic"
   def network_tick(self):
      layer_0 = self.layers[0]
      layer_1 = self.layers[1]
      layer_2 = self.layers[2]
      syn_0 = self.synapses[0]
      syn_1 = self.synapses[1]

      layer_1.neurons = sigmoid(np.dot(layer_0.neurons,syn_0.weights))
      layer_2.neurons = sigmoid(np.dot(layer_1.neurons,syn_1.weights))

      # Hinton's dropout algorithm
      #   This optimizes away (by luck) some cases where multiple search paths are converging on the same (local) minimum slope
      if (training_mode):
        layer_1.neurons *= np.random.binomial([np.ones((len(layer_0.neurons),hidden_dimensions))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

      # Calculate output differences vs expected errors ("Confidence weighted error")
      l2_error = expected_outputs - layer_2.neurons
      l2_delta = l2_error * sigmoid_slope(layer_2.neurons)

      l1_error = l2_delta.dot(syn_1.weights.T)
      l1_delta = l1_error * sigmoid_slope(layer_1.neurons)

      # Nudge weights
      syn_1.weights += alpha * ( layer_1.neurons.T.dot(l2_delta) )
      syn_0.weights += alpha * ( layer_0.neurons.T.dot(l1_delta) )      

      return l2_error

   def load_test_data(self):
      self.layers[0].neurons = input_tests

   def main_loop_training(self):
      start_time = datetime.now()
      last_trace = 0
      iteration = 0
      while(True):
         iteration += 1

         output_error_rate = self.network_tick()

         # Print intermittently
         current_accuracy = 100 * (1 - (np.mean(np.abs(output_error_rate))))
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
           return
       
       
   def __init__(self):
      # Create layers
      for iter in range(layer_count):
         new_layer = Layer("Layer " + str(iter))
         self.layers.append(new_layer)

         if (iter > 0):
            self.layers[iter - 1].next_layer = new_layer # Set next layer in chain

      # Create synapses between layers
      for iter in range(layer_count - 1):
         print("i =" + str(iter) + "AA" + str((layer_count - 1)))
         synapse_input_dimensions = hidden_dimensions
         synapse_output_dimensions = hidden_dimensions
         synapse_name = "Synapse L" + str(iter) + " => L" + str(iter + 1)

         if (iter == 0):
            # First obj special case
            synapse_input_dimensions = input_dimensions

         if (iter == (layer_count - 2)):
            # Last obj special case
            synapse_output_dimensions = output_dimensions

         new_synapse = Synapse(synapse_input_dimensions, synapse_output_dimensions, synapse_name)
         self.synapses.append(new_synapse)


            


"""
CODE ENTRY POINT
"""
print("===== INITIATING NETWORK")
network = Network()
network.load_test_data()
print("")

# Print starting synapse weights
print("===== STARTING WEIGHTS")
for iter in network.synapses:
   print(iter.name + " = " + str(iter.weights))
print("")

# Start training
print("===== NETWORK TRAINING")
network.main_loop_training()
print("")

# Print details
print("===== CALCULATED WEIGHTS")
for iter in network.synapses:
   print(iter.name + " = " + str(iter.weights))
print("")

print("===== FINAL TRAINING RESULTS")
output_neurons = network.layers[-1].neurons
for i in range(len(output_neurons)):
    output_value = output_neurons[i][0] # 0 because we only have a single output result
    expected_value = expected_outputs[i][0]
    value_diff = fabs(expected_value - output_value)
    print("  Test " + (str(i + 1)) + ". " + str(output_value) + ". Expected " + str(expected_value) + ". Diff = " + str(value_diff))
print("")
