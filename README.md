# About this project #
My personal experimentation with neural networks in Python 3. Requires python package "numpy" which is installed automatically.

# Math and resources #
Youtube series by 3Blue1Brown (highly recommended introduction!):
https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

State-of-the-art articles:
http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html

Python sample code:
http://iamtrask.github.io/

# Sample run #
<pre>
===== INITIATING NETWORK
  Created Layer 0
  Created Layer 1
  Created Layer 2
  Created Synapse L0 => L1(3/4 dimensions)
  Created Synapse L1 => L2(4/1 dimensions)

===== STARTING WEIGHTS
Synapse L0 => L1 = [[-0.16595599  0.44064899 -0.99977125 -0.39533485]
 [-0.70648822 -0.81532281 -0.62747958 -0.30887855]
 [-0.20646505  0.07763347 -0.16161097  0.370439  ]]
Synapse L1 => L2 = [[-0.5910955 ]
 [ 0.75623487]
 [-0.94522481]
 [ 0.34093502]]

===== NETWORK TRAINING
  Iteration 7538 accuracy: 99.8715353033%
  Achieved target 99.9% accuracy after 12386 training steps with average test accuracy: 99.9000004556% in 1.644514s

===== CALCULATED WEIGHTS
Synapse L0 => L1 = [[-3.9382048   4.86032762 -5.13584047 -1.53590338]
 [-0.06942316 -0.40045859  0.32493273 -0.8842848 ]
 [ 1.0711445  -2.67828225  1.26132948 -0.94699561]]
Synapse L1 => L2 = [[-4.590061  ]
 [ 8.24317989]
 [-4.63217834]
 [-1.4270996 ]]

===== TRAINING CASES RESULTS
  Test 1. 0.0010093375003. Expected 0. Diff = 0.0010093375003007463
  Test 2. 0.000878275161492. Expected 0. Diff = 0.0008782751614916322
  Test 3. 0.999048298298. Expected 1. Diff = 0.000951701701774943
  Test 4. 0.998703793417. Expected 1. Diff = 0.0012962065832322622
  Test 5. 0.999048298298. Expected 1. Diff = 0.000951701701774943
  Test 6. 0.0010093375003. Expected 0. Diff = 0.0010093375003007463
  Test 7. 0.999048298298. Expected 1. Diff = 0.000951701701774943
  Test 8. 0.999048298298. Expected 1. Diff = 0.000951701701774943

[Finished in 2.7s]
</pre>
