import tqdm
import torch
import torchtext
import pandas as pd
import numpy as np
import submission


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


def create_dataloader(data_pt, lbl_names):
    def categorical(gt):
        # Convert laebl to multi-one-hot
        label = [1 if cat in gt else 0 for cat in lbl_names]
        return label

    TEXT = torchtext.data.Field()
    LABELS = torchtext.data.Field(
            preprocessing=categorical, sequential=False, use_vocab=False)

    # FIXME check what is the header, True will lose one row?
    # FIXME concatenate reply into text
    train_ds = torchtext.data.TabularDataset(
            path=data_pt, format='json', skip_header=False,
            fields={'text': ('text', TEXT), 'categories': ('label', LABELS)})

    # FIXME: batch_size_fn is similar to collate_fn in Dataloader
    # NOTE: shuffle is True default
    train_iter = torchtext.data.BucketIterator(
            dataset=train_ds, batch_size=32,
            sort_key=lambda x: len(x.text))

    TEXT.build_vocab(train_ds)
    # Is building vocabulary for labels necessary here?
    # LABELS.build_vocab(train_ds)

    return train_ds, train_iter


def model_train(model, device, data_ld, optmzr, crtrn):
    def cal_acc(pred, label):
        pass
    NUM_CLS = 43
    model.to(device)
    train_loss, counter = 0, 0

    # pred_onehot = torch.tensor((BS, NUM_CLS), requires_grad=True, device=device, dtype=torch.float32)
    # pred_onehot = torch.Tensor(BS, NUM_CLS).to(device)
    pbar = tqdm.tqdm(data_ld)
    for batch in pbar:
        counter += 1
        optmzr.zero_grad()
        text = batch.text.to(device)
        label = batch.label.type(torch.FloatTensor).to(device)
        output = model(text)

        # FIXME use number of ground truth as additional info
        # preds_idx = torch.argsort(output, dim=1)[:, -6:]
        # pred_onehot.zero_()
        # pred_onehot.scatter_(1, preds_idx, 1)

        loss = crtrn(output, label)
        loss.backward()
        optmzr.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=f'{train_loss/counter:.6f}')

    return model


def main():
    NUM_CLS = 43
    EMBED_DIM = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = "./dataset/emotion_gif/train_gold.json"
    LABEL_PATH = "./dataset/emotion_gif/categories.json"
    label_names = pd.read_json(LABEL_PATH)[0].to_list()
    dataset, dataloader = create_dataloader(DATA_PATH, label_names)

    VOCAB_SIZE = len(dataset.fields['text'].vocab)
    # EMBED_DIM = 32
    # model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLS).to(device)
    model = SimpleBiLSTMBaseline(
            hidden_dim=500, emb_dim=500, vocab_size=VOCAB_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    model = model_train(model, device, dataloader, optimizer, criterion)

    TEST_PATH = "./dataset/emotion_gif/dev_unlabeled.json"
    SUBMISSION_PATH = "./submit/dev.json"
    test_ds = submission.create_dataloader(TEST_PATH, dataset)

    predictions = submission.model_pred(model, device, test_ds, label_names)
    submission.dump_submission(predictions, TEST_PATH, SUBMISSION_PATH)


if __name__ == "__main__":
    main()
