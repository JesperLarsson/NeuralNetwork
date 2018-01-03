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
<pre>
===== INITIATING NETWORK
  Created Layer 0
  Created Layer 1
  Created Layer 2
  Created Layer 3
  Created Synapse L0 => L1(3 inputs / 4 outputs)
  Created Synapse L1 => L2(4 inputs / 4 outputs)
  Created Synapse L2 => L3(4 inputs / 1 outputs)
  Loaded 8 test samples

===== TRAINING
  Targeting 99.9% training accuracy
  Iteration 1 accuracy: 49.5953204776%
  Iteration 2 accuracy: 61.005812321%
  Iteration 3 accuracy: 58.0511004948%
  Iteration 4 accuracy: 45.7878008812%
  Iteration 5 accuracy: 60.6654856987%
  Iteration 4626 accuracy: 99.871164834%
  Achieved target 99.9% accuracy after 7556 training steps in 1.601917s

===== TRAINED NETWORK
Synapse L0 => L1:
  From Node 0:
    To Node 0: 6.87247098714
    To Node 1: -2.33450447067
    To Node 2: -6.33825845689
    To Node 3: -6.17902601138
  From Node 1:
    To Node 0: -2.25544062872
    To Node 1: -0.942461360904
    To Node 2: -1.03804862961
    To Node 3: 0.991880876389
  From Node 2:
    To Node 0: -2.96107027521
    To Node 1: -1.93706308911
    To Node 2: -2.52751747586
    To Node 3: 1.7085207643

Synapse L1 => L2:
  From Node 0:
    To Node 0: -3.47160392848
    To Node 1: 1.69951625589
    To Node 2: -3.25272176995
    To Node 3: 2.0963043105
  From Node 1:
    To Node 0: -1.23582669204
    To Node 1: -0.991980884234
    To Node 2: -0.931802917158
    To Node 3: -1.11660057653
  From Node 2:
    To Node 0: 0.323522901846
    To Node 1: -1.23911770804
    To Node 2: 0.0751844515155
    To Node 3: -1.62528140311
  From Node 3:
    To Node 0: 2.12850254128
    To Node 1: -3.5502934681
    To Node 2: 1.69112633601
    To Node 3: -4.22028789943

Synapse L2 => L3:
  From Node 0:
    To Node 0: -4.72337782069
  From Node 1:
    To Node 0: 3.90499068584
  From Node 2:
    To Node 0: -3.96758846579
  From Node 3:
    To Node 0: 4.78269492698



===== TRAINING CASES RESULTS
  Test 1. 0.00134816413095. Expected 0. Diff = 0.0013481641309522696
  Test 2. 0.000771435251446. Expected 0. Diff = 0.0007714352514456539
  Test 3. 0.999218054792. Expected 1. Diff = 0.0007819452081336831
  Test 4. 0.998595780445. Expected 1. Diff = 0.001404219555143471
  Test 5. 0.999218054792. Expected 1. Diff = 0.0007819452081336831
  Test 6. 0.00134816413095. Expected 0. Diff = 0.0013481641309522696
  Test 7. 0.999218054792. Expected 1. Diff = 0.0007819452081336831
  Test 8. 0.999218054792. Expected 1. Diff = 0.0007819452081336831

[Finished in 2.4s]
</pre>
