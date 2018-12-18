### Introduction and Proposal

This is a repository which I will use to develop an exchange format for neural networks based on the MIRACLE algorithm. This my Part II Project at the University of Cambridge.

A full description can be found in my project proposal hosted here:
<a href="https://github.com/TudorParas/TudorParas.github.io/blob/master/pdfs/MIRACLE%20Neural%20Network%20Exchange%20Format%20-%20Final.pdf" rel="Project Proposal"><img src="https://github.com/TudorParas/TudorParas.github.io/blob/master/images/MIRACLE%20thumbnail.png?raw=true"  width="600"></a>


### Progress

In 'examples/linear_regression_experiment/mnist_data' I have uploaded a file in which the MIRACLE algorithm is hardcoded 
for a linear regression model. The model consists of 7840 parameters, and is trained on the MNIST dataset. The uncompressed 
model has an accuracy of 0.925. In the 'out/compression' folder there are two compressions of this model. The 
'trainp_bits10_block15_hash1.mrcl' file has a size of 658 bytes, and after decompression it scores an accuracy of 0.893.
The 'trainp_bits30_block15_hash1.mrcl' has a size of 332 bytes, and after decompression it scores an accuracy of 0.86

The 'run_miracle' script can be used to load these models or to train and compress your own model by tweaking the 
hyperparameters.