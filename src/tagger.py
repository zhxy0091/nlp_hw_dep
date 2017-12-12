from network import *
from net_properties import *
from vocab import *

if __name__ == '__main__':
    model_file = "nn_model_3"
    train_data_file = "data/train.data"
    net_properties = NetProperties(64, 32, 32, 400, 400, 1000, 0.2)
    vocab = Vocab()
    network = Network(vocab, net_properties)
    network.train(train_data_file, 7)
    network.save(model_file)
