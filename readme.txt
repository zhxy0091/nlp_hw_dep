Explanation of Files:
- src/network.py: build a complete neural network with two hidden layers and defined properties.
- src/tagger.py: train the neural network by passing the parameters of all properties.
- src/net_properties.py: define properties of the neural network.
- src/vocab.py: create maps of vocabs of words, pos, labels and actions.
- src/depModel.py: load the model and predict the result
- nn_model_1: model for part1
- nn_model_2: model for part2
- nn_model_3: model for part3

Part1:
Unlabeled attachment score 83.79
Labeled attachment score 80.67

To use model for part1, please use the following setting in depModel.py:
self.net_properties = NetProperties(64, 32, 32, 200, 200, 1000, 0.2)
m = DepModel("nn_model_1")




Part2:
Unlabeled attachment score 83.95
Labeled attachment score 80.52

To use model for part2, please use the following setting in depModel.py:
self.net_properties = NetProperties(64, 32, 32, 400, 400, 1000, 0.2)
m = DepModel("nn_model_2")

The accuracy slightly improved for unlabeled attachment, but generally the accuracy are similar with no significant change.
The reason could be that although I increase the dimension of two hidden layers, the epochs number does not change, so it does not have enough epochs to learn parameters with a larger dimension.
Another reason could be the training data size is too small for such a large dimension of hidden layers.





Part3:
Unlabeled attachment score 85.34
Labeled attachment score 82.29

To use model for part3, please use the following setting in depModel.py:
self.net_properties = NetProperties(64, 32, 32, 400, 400, 1000, 0.2)
m = DepModel("nn_model_3")

We can see a significant improvement in accuracy in both unlabeled and labeled attachment comparing to part1 and part2.

I keep everything as the same as the setting in part2, with dimensions of two hidden layers equal to 400, but I add dropout=0.2 during the training. Specifically, the neural network will dropout 20% nodes in each of hidden layer 1 and hidden layer 2.

Reason for Improvement:
Dropout is one way of regularization, to solve the overfitting problem of the neural network. Although the loss is reduced overtime for the training data, the loss on validation data may first decrease and then increase. This is the indicator of overfitting. By adding dropout, I randomly omit 20% nodes in two hidden layers, so we obtain a different neural network for each minibatch of training data. Therefore, the chances of overfitting reduce dramatically.