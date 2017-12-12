import dynet as dynet
import random
import matplotlib.pyplot as plt
import numpy as np


class Network:
    def __init__(self, vocab, properties):
        self.properties = properties
        self.vocab = vocab
        self.model = dynet.Model()
        self.updater = dynet.AdamTrainer(self.model)

        self.word_embedding = self.model.add_lookup_parameters((vocab.num_words(), properties.word_embed_dim))
        self.pos_embedding = self.model.add_lookup_parameters((vocab.num_pos(), properties.pos_embed_dim))
        self.label_embedding = self.model.add_lookup_parameters((vocab.num_label(), properties.label_embed_dim))

        self.transfer = dynet.rectify
        input_dim = 20 * properties.word_embed_dim + 20 * properties.pos_embed_dim + 12 * properties.label_embed_dim

        self.hidden_layer_1 = self.model.add_parameters((properties.hidden_dim_1, input_dim))
        self.hidden_layer_bias_1 = self.model.add_parameters(properties.hidden_dim_1, init=dynet.ConstInitializer(0.2))

        self.hidden_layer_2 = self.model.add_parameters((properties.hidden_dim_2, properties.hidden_dim_1))
        self.hidden_layer_bias_2 = self.model.add_parameters(properties.hidden_dim_2, init=dynet.ConstInitializer(0.2))

        self.output_layer = self.model.add_parameters((vocab.num_action(), properties.hidden_dim_2))
        self.output_bias = self.model.add_parameters(vocab.num_action(), init=dynet.ConstInitializer(0))

    def forward(self, features, dropout=False):
        # extract ids for word, pos and label
        word_ids = [self.vocab.word2id(w) for w in features[:20]]
        pos_ids = [self.vocab.pos2id(p) for p in features[20:40]]
        label_ids = [self.vocab.label2id(l) for l in features[40:52]]

        # extract embedding from features
        word_embeds = [self.word_embedding[wid] for wid in word_ids]
        pos_embeds = [self.pos_embedding[pid] for pid in pos_ids]
        label_embeds = [self.label_embedding[lid] for lid in label_ids]

        # concatenating all features
        embedding_layer = dynet.concatenate(word_embeds + pos_embeds + label_embeds)

        # calculating the hidden layers
        hidden_1 = self.transfer(self.hidden_layer_1.expr() * embedding_layer + self.hidden_layer_bias_1.expr())
        if dropout:
            hidden_1 = dynet.dropout(hidden_1, self.properties.dropout)
        hidden_2 = self.transfer(self.hidden_layer_2.expr() * hidden_1 + self.hidden_layer_bias_2.expr())
        if dropout:
            hidden_2 = dynet.dropout(hidden_2, self.properties.dropout)
        # calculating the output layer
        output = self.output_layer.expr() * hidden_2 + self.output_bias.expr()

        return output

    def train(self, train_data_file, epochs):
        # matplotlib config
        loss_values = []
        plt.ion()
        ax = plt.gca()
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 3])
        plt.title("Loss over time")
        plt.xlabel("Minibatch")
        plt.ylabel("Loss")
        plt.show()

        
        
        for i in range(epochs):
            print 'epoch', (i + 1)

            losses = []  # minibatch loss vector
            train_data = open(train_data_file, 'r').read().strip().split("\n")

            random.shuffle(train_data)  # shuffle the training data.

            for line in train_data:
                fields = line.strip('\n').split(" ")
                features, action, gold_action = fields[:-1], fields[-1], self.vocab.action2id(fields[-1])
                result = self.forward(features, True)

                # getting loss with respect to negative log softmax function and the gold label; and appending to the minibatch losses.
                loss = dynet.pickneglogsoftmax(result, gold_action)
                losses.append(loss)

                if len(losses) >= self.properties.minibatch_size:
                    minibatch_loss = dynet.esum(losses) / len(
                        losses)  # now we have enough loss values to get loss for minibatch
                    minibatch_loss.forward()  # calling dynetnet to run forward computation for all minibatch items
                    minibatch_loss_value = minibatch_loss.value()  # getting float value of the loss for current minibatch

                    # printing info and plotting
                    loss_values.append(minibatch_loss_value)

                    if len(loss_values) % 10 == 0:
                        ax.set_xlim([0, len(loss_values) + 10])
                        ax.plot(loss_values)
                        plt.draw()
                        try:
                            plt.pause(0.0001)
                        except:
                            pass

                    minibatch_loss.backward()  # calling dynetnet to run backpropagation
                    self.updater.update()  # calling dynet to change parameter values with respect to current backpropagation

                    # empty the loss vector and refresh the memory of dynetnet
                    losses = []
                    dynet.renew_cg()

            dynet.renew_cg()  # there are still some minibatch items in the memory but they are smaller than the minibatch size
        print 'finished training!'



    def load(self, filename):
        self.model.populate(filename)

    def save(self, filename):
        self.model.save(filename)