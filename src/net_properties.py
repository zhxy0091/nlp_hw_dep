class NetProperties:
    def __init__(self, word_embed_dim, pos_embed_dim, label_embed_dim, hidden_dim_1, hidden_dim_2, minibatch_size):
        self.word_embed_dim = word_embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.label_embed_dim = label_embed_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.minibatch_size = minibatch_size