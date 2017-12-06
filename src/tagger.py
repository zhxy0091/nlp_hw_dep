from network import *
from net_properties import *
from vocab import *

if __name__ == '__main__':
    train_data_file = "data/train.data"
    net_properties = NetProperties(64, 32, 32, 200, 200, 1000)
    vocab = Vocab()
    network = Network(vocab, net_properties)
    network.train(train_data_file, 7)
    network.save("nn_model")
