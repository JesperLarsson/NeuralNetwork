# About this project #
Simple experimentation with neural networks in Python 3.

# Math  #
Concepts and math is based on this Youtube series by 3Blue1Brown (highly recommended):
https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

# Sample run #
<pre>
===== INITIAL (RANDOMIZED) WEIGHTS
[[-0.16595599  0.44064899 -0.99977125 -0.39533485]
 [-0.70648822 -0.81532281 -0.62747958 -0.30887855]
 [-0.20646505  0.07763347 -0.16161097  0.370439  ]]
[[-0.5910955 ]
 [ 0.75623487]
 [-0.94522481]
 [ 0.34093502]]

===== NETWORK OUTPUT
  Iteration 0 accuracy: 53.5456962018%
  Iteration 1000 accuracy: 98.8080670526%
  Iteration 2000 accuracy: 99.1895859333%
  Iteration 3000 accuracy: 99.3507256894%
  Iteration 4000 accuracy: 99.4444686204%
  Iteration 5000 accuracy: 99.507430331%
  Iteration 6000 accuracy: 99.5533692441%
  Iteration 7000 accuracy: 99.5887511261%
  Iteration 8000 accuracy: 99.6170626751%
  Iteration 9000 accuracy: 99.6403696394%
  Iteration 10000 accuracy: 99.659983006%
  Iteration 11000 accuracy: 99.6767796639%
  Iteration 12000 accuracy: 99.6913710222%
  Iteration 13000 accuracy: 99.7041979919%
  Iteration 14000 accuracy: 99.7155875238%
  Iteration 15000 accuracy: 99.7257878225%
  Iteration 16000 accuracy: 99.7349911268%
  Iteration 17000 accuracy: 99.743348929%
  Iteration 18000 accuracy: 99.7509824286%
  Iteration 19000 accuracy: 99.7579898876%
  Iteration 20000 accuracy: 99.7644519174%
  Iteration 21000 accuracy: 99.7704353493%
  Iteration 22000 accuracy: 99.7759961181%
  Iteration 23000 accuracy: 99.7811814407%
  Iteration 24000 accuracy: 99.7860314857%
  Iteration 25000 accuracy: 99.7905806691%
  Iteration 26000 accuracy: 99.79485867%
  Iteration 27000 accuracy: 99.7988912355%
  Iteration 28000 accuracy: 99.8027008255%
  Iteration 29000 accuracy: 99.806307132%
  Iteration 30000 accuracy: 99.8097275022%
  Iteration 31000 accuracy: 99.8129772853%
  Iteration 32000 accuracy: 99.816070118%
  Iteration 33000 accuracy: 99.8190181625%
  Iteration 34000 accuracy: 99.8218323051%
  Iteration 35000 accuracy: 99.8245223227%
  Iteration 36000 accuracy: 99.8270970239%
  Iteration 37000 accuracy: 99.829564369%
  Iteration 38000 accuracy: 99.8319315716%
  Iteration 39000 accuracy: 99.8342051865%
  Iteration 40000 accuracy: 99.8363911848%
  Iteration 41000 accuracy: 99.838495019%
  Iteration 42000 accuracy: 99.8405216794%
  Iteration 43000 accuracy: 99.8424757434%
  Iteration 44000 accuracy: 99.8443614178%
  Iteration 45000 accuracy: 99.8461825772%
  Iteration 46000 accuracy: 99.8479427963%
  Iteration 47000 accuracy: 99.8496453794%
  Iteration 48000 accuracy: 99.8512933859%
  Iteration 49000 accuracy: 99.8528896531%
  Iteration 50000 accuracy: 99.854436817%
  Iteration 51000 accuracy: 99.8559373293%
  Iteration 52000 accuracy: 99.8573934745%
  Iteration 53000 accuracy: 99.8588073839%
  Iteration 54000 accuracy: 99.8601810484%
  Iteration 55000 accuracy: 99.8615163304%
  Iteration 56000 accuracy: 99.8628149742%
  Iteration 57000 accuracy: 99.8640786154%
  Iteration 58000 accuracy: 99.8653087894%
  Iteration 59000 accuracy: 99.8665069392%
  Iteration 60000 accuracy: 99.8676744224%
  Iteration 61000 accuracy: 99.8688125176%
  Iteration 62000 accuracy: 99.8699224303%
  Iteration 63000 accuracy: 99.8710052979%
  Iteration 64000 accuracy: 99.8720621948%
  Iteration 65000 accuracy: 99.8730941367%
  Iteration 66000 accuracy: 99.8741020849%
  Iteration 67000 accuracy: 99.8750869495%
  Iteration 68000 accuracy: 99.8760495933%
  Iteration 69000 accuracy: 99.8769908347%
  Iteration 70000 accuracy: 99.8779114506%
  Iteration 71000 accuracy: 99.8788121792%
  Iteration 72000 accuracy: 99.879693722%
  Iteration 73000 accuracy: 99.8805567469%
  Iteration 74000 accuracy: 99.8814018894%
  Iteration 75000 accuracy: 99.8822297552%
  Iteration 76000 accuracy: 99.8830409217%
  Iteration 77000 accuracy: 99.8838359395%
  Iteration 78000 accuracy: 99.8846153345%
  Iteration 79000 accuracy: 99.8853796088%
  Iteration 80000 accuracy: 99.8861292422%
  Iteration 81000 accuracy: 99.8868646938%
  Iteration 82000 accuracy: 99.8875864025%
  Iteration 83000 accuracy: 99.8882947886%
  Iteration 84000 accuracy: 99.8889902549%
  Iteration 85000 accuracy: 99.8896731869%
  Iteration 86000 accuracy: 99.8903439548%
  Iteration 87000 accuracy: 99.8910029132%
  Iteration 88000 accuracy: 99.8916504028%
  Iteration 89000 accuracy: 99.8922867506%
  Iteration 90000 accuracy: 99.8929122706%
  Iteration 91000 accuracy: 99.8935272648%
  Iteration 92000 accuracy: 99.8941320234%
  Iteration 93000 accuracy: 99.8947268258%
  Iteration 94000 accuracy: 99.8953119405%
  Iteration 95000 accuracy: 99.8958876263%
  Iteration 96000 accuracy: 99.8964541322%
  Iteration 97000 accuracy: 99.8970116984%
  Iteration 98000 accuracy: 99.8975605562%
  Iteration 99000 accuracy: 99.8981009287%
  Iteration 100000 accuracy: 99.8986330311%
  Iteration 101000 accuracy: 99.899157071%
  Iteration 102000 accuracy: 99.8996732489%
  Finished on iteration 102641 with accuracy: 99.9000000739% in 5.287s

===== CALCULATED WEIGHTS
[[-2.81566946  4.83652051 -4.98172518  1.09088631]
 [-0.31389172 -0.52933321  0.05022101 -0.30192428]
 [ 0.94116298 -1.90985547  2.1403772   0.33187547]]
[[-3.27015697]
 [ 7.02626373]
 [-6.83782461]
 [ 1.44628531]]

===== FINAL RESULTS
  Test 1. 0.00120247747005. Expected 0. Diff = 0.00120247747005
  Test 2. 0.000925769423551. Expected 0. Diff = 0.000925769423551
  Test 3. 0.999109084247. Expected 1. Diff = 0.000890915753455
  Test 4. 0.998894393287. Expected 1. Diff = 0.00110560671276
  Test 5. 0.999109084247. Expected 1. Diff = 0.000890915753455
  Test 6. 0.00120247747005. Expected 0. Diff = 0.00120247747005
  Test 7. 0.999109084247. Expected 1. Diff = 0.000890915753455
  Test 8. 0.999109084247. Expected 1. Diff = 0.000890915753455
</pre>

# Source modifications #
You are free to change and distribute the source according to the GPL license.
