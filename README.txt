Multilayer Neural Network

----DESCRIPTION----

This program implements a multilayer neural network to classify the images of
0s and 1s from the mnist data set. The data files included are as follows:

1) mnist_train_0_1.csv: pixel data for images used for training the algorithm

2) mnist_test_0_1.csv: pixel data for images used for testing the algorithm

The program takes a large amount of time to run completely. It displays a percent complete
as it works. Once the training and testing is complete the program writes the accuracy percentage
out for the user to evaluate the model.

The program is currently flawed in the training portion. When the program tests the model
it outputs the same class label indiscriminately. When training is occuring only very
small weight changes are occuring which seems to be the problem but I have been unable
to determine the source of this issue.