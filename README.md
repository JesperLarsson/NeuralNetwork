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
  Iteration 7582 accuracy: 99.8719132381%
  Iteration 15222 accuracy: 99.9099083241%
  Iteration 22775 accuracy: 99.9265060572%
  Iteration 29973 accuracy: 99.9360339093%
  Iteration 37603 accuracy: 99.9429556359%
  Iteration 44381 accuracy: 99.9475391936%
  Iteration 51935 accuracy: 99.951543007%
  Iteration 59458 accuracy: 99.9547365928%
  Iteration 67132 accuracy: 99.9574320237%
  Iteration 74796 accuracy: 99.9596858341%
  Iteration 82372 accuracy: 99.9616000299%
  Iteration 90098 accuracy: 99.963300218%
  Iteration 97752 accuracy: 99.9647812464%
  Iteration 105445 accuracy: 99.9661032368%
  Iteration 111981 accuracy: 99.9671175682%
  Iteration 118614 accuracy: 99.9680561819%
  Iteration 125532 accuracy: 99.9689570995%
  Iteration 130919 accuracy: 99.969606309%
  Iteration 137177 accuracy: 99.9703126727%
  Iteration 144074 accuracy: 99.971039007%
  Iteration 151347 accuracy: 99.9717496801%
  Iteration 158555 accuracy: 99.9724029928%
  Iteration 165863 accuracy: 99.9730238531%
  Iteration 173027 accuracy: 99.9735929977%
  Iteration 180124 accuracy: 99.9741218215%
  Iteration 185870 accuracy: 99.9745284472%
  Iteration 193004 accuracy: 99.975006217%
  Iteration 200364 accuracy: 99.9754741654%
  Iteration 207651 accuracy: 99.9759125472%
  Iteration 213863 accuracy: 99.9762684676%
  Iteration 221383 accuracy: 99.9766783935%
  Iteration 228831 accuracy: 99.977062818%
  Iteration 235259 accuracy: 99.9773815368%
  Iteration 242664 accuracy: 99.9777327637%
  Iteration 248527 accuracy: 99.9779992641%
  Iteration 255495 accuracy: 99.9783040104%
  Iteration 262759 accuracy: 99.9786095428%
  Iteration 270304 accuracy: 99.9789124504%
  Iteration 278002 accuracy: 99.9792090124%
  Iteration 285362 accuracy: 99.9794813083%
  Iteration 292522 accuracy: 99.9797358346%
  Iteration 300224 accuracy: 99.9800002283%
  Iteration 307859 accuracy: 99.9802522763%
  Iteration 315399 accuracy: 99.9804918845%
  Iteration 322939 accuracy: 99.980722875%
  Iteration 330403 accuracy: 99.980943461%
  Iteration 337972 accuracy: 99.9811598237%
  Iteration 345505 accuracy: 99.9813687845%
  Iteration 353090 accuracy: 99.9815716592%
  Iteration 360753 accuracy: 99.9817701377%
  Iteration 368475 accuracy: 99.9819631414%
  Iteration 376184 accuracy: 99.9821507631%
  Iteration 383892 accuracy: 99.9823323174%
  Iteration 391492 accuracy: 99.9825058024%
  Iteration 399130 accuracy: 99.9826758558%
  Iteration 406687 accuracy: 99.9828388937%
  Iteration 414296 accuracy: 99.9829986766%
  Iteration 421912 accuracy: 99.983153905%
  Iteration 429434 accuracy: 99.9833031151%
  Iteration 437092 accuracy: 99.9834510115%
  Iteration 444796 accuracy: 99.983596324%
  Iteration 452301 accuracy: 99.9837340892%
  Iteration 459833 accuracy: 99.9838692584%
  Iteration 467535 accuracy: 99.9840038122%
  Iteration 475167 accuracy: 99.9841333936%
  Iteration 482381 accuracy: 99.9842533901%
  Iteration 489904 accuracy: 99.9843759152%
  Iteration 497546 accuracy: 99.9844975763%
  Iteration 504813 accuracy: 99.9846101805%
  Iteration 512436 accuracy: 99.984726029%
  Iteration 519727 accuracy: 99.9848344956%
  Iteration 527424 accuracy: 99.9849462502%
  Iteration 535129 accuracy: 99.9850560919%
  Iteration 542538 accuracy: 99.9851595084%
  Iteration 550173 accuracy: 99.9852639438%
  Iteration 557870 accuracy: 99.9853670849%
  Iteration 565343 accuracy: 99.9854649498%
  Iteration 572914 accuracy: 99.9855620181%
  Iteration 580546 accuracy: 99.9856581556%
  Iteration 588149 accuracy: 99.9857517712%
  Iteration 595746 accuracy: 99.9858438407%
  Iteration 603404 accuracy: 99.9859346652%
  Iteration 611022 accuracy: 99.986023071%
  Iteration 618672 accuracy: 99.9861103721%
  Iteration 626331 accuracy: 99.9861962577%
  Iteration 633994 accuracy: 99.9862806716%
  Iteration 641549 accuracy: 99.9863623997%
  Iteration 648971 accuracy: 99.9864411238%
  Iteration 656660 accuracy: 99.9865215336%
  Iteration 664336 accuracy: 99.9866005051%
  Iteration 671968 accuracy: 99.9866772141%
  Iteration 679613 accuracy: 99.9867529765%
  Iteration 687262 accuracy: 99.9868275098%
  Iteration 694840 accuracy: 99.9869000077%
  Iteration 702238 accuracy: 99.9869695368%
  Iteration 709720 accuracy: 99.9870390138%
  Iteration 717384 accuracy: 99.9871089236%
  Iteration 725087 accuracy: 99.9871778454%
  Iteration 732801 accuracy: 99.9872462603%
  Iteration 740485 accuracy: 99.9873131977%
  Iteration 747933 accuracy: 99.9873770238%
  Iteration 755443 accuracy: 99.9874402968%
  Iteration 762909 accuracy: 99.9875022324%
  Iteration 769837 accuracy: 99.9875590151%
  Iteration 776761 accuracy: 99.9876150576%
  Iteration 784170 accuracy: 99.9876741539%
  Iteration 791800 accuracy: 99.9877339561%
  Iteration 799524 accuracy: 99.9877938545%
  Iteration 807197 accuracy: 99.9878524244%
  Iteration 814760 accuracy: 99.9879091594%
  Iteration 822307 accuracy: 99.9879652882%
  Iteration 829841 accuracy: 99.9880203985%
  Iteration 837361 accuracy: 99.9880748735%
  Iteration 844838 accuracy: 99.9881280338%
  Iteration 852218 accuracy: 99.9881801442%
  Iteration 859792 accuracy: 99.988232735%
  Iteration 867502 accuracy: 99.9882855719%
  Iteration 875012 accuracy: 99.9883363065%
  Iteration 882263 accuracy: 99.9883846396%
  Iteration 889662 accuracy: 99.9884335358%
  Iteration 897181 accuracy: 99.9884824688%
  Iteration 904874 accuracy: 99.9885319197%
  Iteration 912542 accuracy: 99.9885805628%
  Iteration 920255 accuracy: 99.9886288179%
  Iteration 927973 accuracy: 99.9886764029%
  Iteration 935573 accuracy: 99.988722817%
  Iteration 943172 accuracy: 99.9887687907%
  Iteration 950812 accuracy: 99.9888143917%
  Iteration 958246 accuracy: 99.9888581146%
  Iteration 965402 accuracy: 99.9888998017%
  Iteration 973014 accuracy: 99.9889436906%
  Iteration 980655 accuracy: 99.9889871983%
  Iteration 988331 accuracy: 99.9890304279%
  Iteration 996034 accuracy: 99.9890733147%
  Iteration 1003460 accuracy: 99.9891141963%
  Iteration 1010758 accuracy: 99.9891538628%
  Iteration 1018283 accuracy: 99.9891943396%
  Iteration 1025970 accuracy: 99.9892352166%
  Iteration 1033578 accuracy: 99.9892752799%
  Iteration 1041303 accuracy: 99.9893154828%
  Iteration 1048735 accuracy: 99.9893537885%
  Iteration 1055765 accuracy: 99.9893895368%
  Iteration 1062305 accuracy: 99.9894224671%
  Iteration 1068420 accuracy: 99.989452981%
  Iteration 1073915 accuracy: 99.9894803028%
  Iteration 1079990 accuracy: 99.9895102171%
  Iteration 1086534 accuracy: 99.9895421225%
  Iteration 1093504 accuracy: 99.9895758469%
  Iteration 1100065 accuracy: 99.9896072492%
  Iteration 1106702 accuracy: 99.9896387848%
  Iteration 1113178 accuracy: 99.989669124%
  Iteration 1119632 accuracy: 99.9896991736%
  Iteration 1126668 accuracy: 99.9897315793%
  Iteration 1133199 accuracy: 99.9897614135%
  Iteration 1139807 accuracy: 99.9897914081%
  Iteration 1145915 accuracy: 99.9898188547%
  Iteration 1153310 accuracy: 99.989851777%
  Iteration 1160266 accuracy: 99.9898825231%
  Iteration 1167204 accuracy: 99.9899128722%
  Iteration 1174059 accuracy: 99.9899426207%
  Iteration 1180980 accuracy: 99.9899724245%
  Iteration 1187256 accuracy: 99.9899991037%
  Achieved target 99.99% accuracy after 1187471 training steps with average test accuracy: 99.9900000027% in 162.038376s

===== CALCULATED WEIGHTS
Synapse L0 => L1 = [[ -4.57002734e+00   5.19660583e+00  -5.46968908e+00  -1.78982996e+00]
 [ -3.63490563e-03  -2.61097057e-01   3.06855269e-01  -8.45009192e-01]
 [  1.32009876e+00  -3.02855203e+00   1.52444922e+00  -7.86984805e-01]]
Synapse L1 => L2 = [[ -5.62792692]
 [ 10.7820327 ]
 [ -5.81241955]
 [ -1.63505283]]

===== TRAINING CASES RESULTS
  Test 1. 9.81889178769e-05. Expected 0. Diff = 9.818891787692262e-05
  Test 2. 8.88907294826e-05. Expected 0. Diff = 8.889072948263298e-05
  Test 3. 0.999902861676. Expected 1. Diff = 9.713832448043913e-05
  Test 4. 0.999873822079. Expected 1. Diff = 0.00012617792104052672
  Test 5. 0.999902861676. Expected 1. Diff = 9.713832448043913e-05
  Test 6. 9.81889178769e-05. Expected 0. Diff = 9.818891787692262e-05
  Test 7. 0.999902861676. Expected 1. Diff = 9.713832448043913e-05
  Test 8. 0.999902861676. Expected 1. Diff = 9.713832448043913e-05

[Finished in 163.0s]

# Source modifications #
You are free to change and distribute the source according to the GPL license.
