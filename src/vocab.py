import os


class Vocab:
    def __init__(self):
        self.vocabs_actions_file = os.path.abspath("data/vocabs.actions")
        self.vocabs_labels_file = os.path.abspath("data/vocabs.labels")
        self.vocabs_pos_file = os.path.abspath("data/vocabs.pos")
        self.vocabs_word_file = os.path.abspath("data/vocabs.word")
        # string->int
        self.actions_map = {}
        self.labels_map = {}
        self.pos_map = {}
        self.word_map = {}
        self.__create_map(self.actions_map, self.vocabs_actions_file)
        self.__create_map(self.labels_map, self.vocabs_labels_file)
        self.__create_map(self.pos_map, self.vocabs_pos_file)
        self.__create_map(self.word_map, self.vocabs_word_file)

    def __create_map(self, map, file):
        f = open(file, "r")
        for line in f:
            data = line.strip('\n').split(" ")
            map[data[0]] = int(data[1])
        f.close()

    def word2id(self, word):
        if word not in self.word_map:
            word = "<unk>"
        return self.word_map[word]

    def label2id(self, label):
        if label not in self.labels_map:
            label = "<null>"
        return self.labels_map[label]

    def pos2id(self, pos):
        if pos not in self.pos_map:
            pos = "<null>"
        return self.pos_map[pos]

    def action2id(self, action):
        return self.actions_map[action]

    def num_words(self):
        return len(self.word_map)

    def num_pos(self):
        return len(self.pos_map)

    def num_label(self):
        return len(self.labels_map)

    def num_action(self):
        return len(self.actions_map)