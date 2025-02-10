# Abstract

This paper shows a small feed-forward neural network for Handwritten Digit
Recognition on the MNIST dataset. The code is written in Matlab. I began with the
process of loading and preprocessing the data, then convert the digit images into normalized vectors and perform transform class labels into one-hot encodings. The neural
network consists of an input layer, a hidden layer, and an output layer. The input layer
has 784 input nodes. The hidden layer uses the ReLU activation function, and the output
layer uses the softmax function. Together, the model can accurately predict the digit of
the handwritten images from 0 to 9. In order to optimize the model, I use forward and
backward passes to compute the loss and gradients to apply mini-batch gradient descent
for parameters. The model can achieve a high accuracy on both validation and test sets,
which shows the effectiveness of the neural network and the optimization algorithm of
our model. The result is shown in the format of a confusion matrix. The complete code is
available in the appendix.

Keywords: MNIST; Neural Network; Handwritten Digit Recognition.
