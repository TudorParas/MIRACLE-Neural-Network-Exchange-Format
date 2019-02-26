### Introduction and Proposal

This is a repository which I will use to develop an exchange format for neural networks based on the MIRACLE algorithm. This my Part II Project at the University of Cambridge.

A full description can be found in my project proposal hosted here:
<a href="https://github.com/TudorParas/TudorParas.github.io/blob/master/pdfs/MIRACLE%20Neural%20Network%20Exchange%20Format%20-%20Final.pdf" rel="Project Proposal"><img src="https://github.com/TudorParas/TudorParas.github.io/blob/master/images/MIRACLE%20thumbnail.png?raw=true"  width="600"></a>


### How to run

The library exposes 7 functions which should be called in the following order:

1. __create_variable__: Takes a shape as input and returns a tensorflow tensor that can be used to build your Neural Networks. The user can specify a hash group size i that will reduce the compressed size of this layer by a factor of i, but it will affect performance (very hard in some cases.)
2. __create_compression_graph__: This is where the magic of the library happens. This creates the compression, training, and loading graphs. This method should be called after finishing creating the variables. It takes as arguments the user defined loss and the final size of the compressed file in bits (for people who know what they're doing they can also specify the nr of variables in a block and the number of bits used to compress a block).
3. __assign_session__: After creating a TensorFlow session you should use this method to assign the session to the miracle graph. This method exists in order to hide some complexities from the user.
4. __pretrain__: Trains to minimize the loss. Takes as argument the number of iterations and optionally a function f:int->unit that will be executed after every iteration (I use it for printing).
5. __train__: Trains to minimize the loss and also takes into account the KL difference between the trained graph and the prior. Should be run until both the accurarcy and the Mean KL converge.
6. __compress__: Compresses the graph and outputs it to a file. Takes as argument the path file and the number of time we retrain the graph after compressing a block (recomment 10-100).
7. __load__: Loads the graph from a file. Still requires a session to have been assigned to the graph in the same session. Does not require the function 4 to 6 to have been called in the same session.



### Examples

The main evaluation has been done on LeNet5-caffe. This can be found in examples/miracle_graphs/mnist/LeNet5/graphs/lenet5_miracle.py. It achieves a compression rate of 1000x with an accuracy of 99.17%. It also achieves a compression rate of 1500x with an accuracy of 99.02%. These compressed files have been uploaded and can be tested by the user.

The compression rates have been calculated assuming that the 431k variables occupy 4 bytes each.