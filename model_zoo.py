import torch


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
    def __init__(self, hidden_dim, vocab_size, emb_dim=300,
                 spatial_dropout=0.05, recurrent_dropout=0.1,
                 num_linear=1, pred_num=43):
        super().__init__()  # don't forget to call this!
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.encoder = torch.nn.LSTM(
                emb_dim, hidden_dim, num_layers=1, dropout=recurrent_dropout)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers = torch.nn.ModuleList(self.linear_layers)
        self.predictor = torch.nn.Linear(hidden_dim, pred_num)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)

        return preds


def main():
    pass


if __name__ == "__main__":
    main()
