# About this project #
My personal experimentation with neural networks in Python 3.

Requires python package "numpy" which is installed automatically by the script.

# Math and resources #
Youtube series by 3Blue1Brown (highly recommended introduction!):
https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

State-of-the-art articles:
http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html

Python sample code:
http://iamtrask.github.io/

# Sample run #
<pre>===== INITIATING NETWORK
  Targeting 99.95% accuracy
  Created Layer 0
  Created Layer 1
  Created Layer 2
  Created Synapse L0 => L1(3/4 dimensions)
  Created Synapse L1 => L2(4/1 dimensions)

===== STARTING NETWORK
Layer 0:
  [0 0 1]
  [0 1 1]
  [1 0 1]
  [1 1 1]
  [1 0 1]
  [0 0 1]
  [1 0 1]
  [1 0 1]

Layer 1:
  Uninitialized
Layer 2:
  Uninitialized


===== NETWORK TRAINING
  Iteration 1 accuracy: 53.5456962018%
  Iteration 2 accuracy: 66.1023483643%
  Iteration 3 accuracy: 45.9218211197%
  Iteration 7651 accuracy: 99.8725014286%
  Iteration 15336 accuracy: 99.9102552268%
  Iteration 23112 accuracy: 99.9270426909%
  Iteration 30813 accuracy: 99.9369205791%
  Iteration 38606 accuracy: 99.9437088153%
  Iteration 46357 accuracy: 99.9486829512%
  Achieved target 99.95% accuracy after 48805 training steps in 6.318738s

===== TRAINED NETWORK
Layer 0:
  [0 0 1]
  [0 1 1]
  [1 0 1]
  [1 1 1]
  [1 0 1]
  [0 0 1]
  [1 0 1]
  [1 0 1]

Layer 1:
  [ 0.84411389  0.0644636   0.88462105  0.32044549]
  [ 0.83402052  0.04618485  0.93761861  0.16064069]
  [ 0.0533366   0.99796845  0.02272087  0.08318952]
  [ 0.05089636  0.95702394  0.0311938   0.03627492]
  [ 0.0533366   0.99796845  0.02272087  0.08318952]
  [ 0.84411389  0.0644636   0.88462105  0.32044549]
  [ 0.          0.99796845  0.02272087  0.08318952]
  [ 0.0533366   0.99796845  0.02272087  0.08318952]

Layer 2:
  [ 0.00050045]
  [ 0.00044036]
  [ 0.99952128]
  [ 0.99935613]
  [ 0.99952128]
  [ 0.00050045]
  [ 0.99952128]
  [ 0.99952128]



===== TRAINING CASES RESULTS
  Test 1. 0.00050044603525. Expected 0. Diff = 0.0005004460352499578
  Test 2. 0.00044036471381. Expected 0. Diff = 0.00044036471381042564
  Test 3. 0.999521282535. Expected 1. Diff = 0.0004787174651744719
  Test 4. 0.999356131914. Expected 1. Diff = 0.0006438680859717216
  Test 5. 0.999521282535. Expected 1. Diff = 0.0004787174651744719
  Test 6. 0.00050044603525. Expected 0. Diff = 0.0005004460352499578
  Test 7. 0.999521282535. Expected 1. Diff = 0.0004787174651744719
  Test 8. 0.999521282535. Expected 1. Diff = 0.0004787174651744719

[Finished in 7.0s]
</pre>
