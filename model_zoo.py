import torch
import bcolz
import pickle
import numpy as np


class TextSentiment(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = torch.nn.EmbeddingBag(
                vocab_size, embed_dim, sparse=True)
        self.fc = torch.nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)


class SimpleBiLSTMBaseline(torch.nn.Module):
    # emb_dim from create_emb()
    def __init__(self, hidden_dim, vocab, pred_num):
        super().__init__()  # don't forget to call this!

        emb_dim = vocab.vectors.size()[1]
        num_linear = 1
        self.embedding = torch.nn.Embedding(len(vocab), emb_dim)
        self.embedding.from_pretrained(vocab.vectors, freeze=True)
        # self.embedding, emb_num, emb_dim = create_emb(vocab)
        self.encoder = torch.nn.LSTM(
                emb_dim, hidden_dim, num_layers=1)
        self.linear_layers = []
        for _ in range(num_linear-1):
            self.linear_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = torch.nn.ModuleList(self.linear_layers)
        self.predictor = torch.nn.Linear(hidden_dim, pred_num)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, seq):
        hdn, _ = self.encoder(self.dropout(self.embedding(seq)))
        feature = hdn[-1, :, :]  # 选择最后一个output
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)

        return preds


class TorchLSTM(torch.nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(TorchLSTM, self).__init__()
        self.drop = torch.nn.Dropout(dropout)
        self.encoder = torch.nn.Embedding(ntoken, ninp)
        self.rnn = torch.nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = torch.nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


def glove_dict():
    glove_path = "./dataset/glove"
    vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

    return {w: vectors[word2idx[w]] for w in words}


def create_emb(target_vocab, trainable=True):
    glove = glove_dict()
    emb_dim = 50
    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for i, word in enumerate(target_vocab.freqs.keys()):
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
    print(f"Found words: {words_found}")

    emb_layer = torch.nn.Embedding(matrix_len, emb_dim)
    weights_tensor = torch.Tensor(weights_matrix)
    emb_layer.from_pretrained(weights_tensor,
                              freeze=False if trainable else True)

    return emb_layer, matrix_len, emb_dim


def main():
    pass


if __name__ == "__main__":
    main()
