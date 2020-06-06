import nltk
import random
import torchtext
import pandas as pd
import numpy as np


def create_dataloader(data_pt, lbl_names):
    def lst2tp(gt):
        # label = tuple([1 if cat in gt else 0 for cat in lbl_names])
        label = tuple([lbl_names.index(cat) for cat in gt])
        return label

    def categorical(batch, vocab):
        # Convert laebl to multi-one-hot
        categorical_batch = np.zeros((len(batch), len(lbl_names)))
        for idx, row in enumerate(batch):
            categorical_batch[idx, row] = 1
        return categorical_batch

    tknz_func = nltk.TweetTokenizer().tokenize
    stopwords = nltk.corpus.stopwords.words('english')

    TEXT = torchtext.data.Field(
            tokenize=tknz_func, stop_words=stopwords, lower=True)
    LABELS = torchtext.data.Field(
            preprocessing=lst2tp, sequential=False, use_vocab=False,
            postprocessing=categorical)

    # NOTE skip_header will miss the first row, so disable here
    # FIXME concatenate reply into text
    dataset = torchtext.data.TabularDataset(
            path=data_pt, format='json', skip_header=False,
            fields={'text': ('text', TEXT), 'categories': ('label', LABELS)})

    # Ref: https://pytorch.org/text/data.html#torchtext.data.Dataset.split
    # FIXME: - enable stratified for spliting dataset balancely
    # NOTE:  - set fixed random_state for fixed set of train & val dataset
    random.seed(66)
    (train_ds, val_ds) = dataset.split(
            split_ratio=0.8, stratified=True, random_state=random.getstate())

    # NOTE: shuffle is True default
    train_iter, val_iter = torchtext.data.BucketIterator.splits(
            datasets=(train_ds, val_ds), batch_sizes=(64, 64),
            sort_key=lambda x: len(x.text))

    TEXT.build_vocab(train_ds)
    # Is building vocabulary for labels necessary here?
    # LABELS.build_vocab(train_ds)

    return train_ds, train_iter, val_iter


def main():
    DATA_PATH = "./dataset/emotion_gif/train_gold.json"
    LABEL_PATH = "./dataset/emotion_gif/categories.json"
    label_names = pd.read_json(LABEL_PATH)[0].to_list()
    train_ds, train_dl, val_dl = create_dataloader(DATA_PATH, label_names)


if __name__ == "__main__":
    main()
