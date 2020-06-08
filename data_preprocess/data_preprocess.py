import nltk
import random
import torchtext
import pandas as pd
import numpy as np
import torch
import string


def create_dataloader(data_pt, lbl_names, emb_dim):
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

    def rm_punc(tokens):
        return [s for s in tokens if s not in punctuation]

    tknz_func = nltk.TweetTokenizer().tokenize
    stopwords = nltk.corpus.stopwords.words('english')
    punctuation = string.punctuation + "’" + "..." + "“" + "”" + chr(65039)

    TEXT = torchtext.data.Field(
            tokenize=tknz_func, lower=True, init_token='<s>', eos_token='</s>')
    LABELS = torchtext.data.Field(
            preprocessing=lst2tp, sequential=False, use_vocab=False,
            postprocessing=categorical)

    # NOTE skip_header will miss the first row, so disable here
    # FIXME concatenate reply into text
    dataset = torchtext.data.TabularDataset(
            path=data_pt, format='json', skip_header=False,
            fields={'text': ('text', TEXT), 'categories': ('label', LABELS)})

    # Ref: https://pytorch.org/text/data.html#torchtext.data.Dataset.split
    # NOTE:  - set fixed random_state for fixed set of train & val dataset
    random.seed(666)
    (train_ds, val_ds) = dataset.split(
            split_ratio=0.8, stratified=True, random_state=random.getstate())

    # NOTE: shuffle is True default
    train_iter, val_iter = torchtext.data.BucketIterator.splits(
            datasets=(train_ds, val_ds), batch_sizes=(32, 32),
            sort_key=lambda x: len(x.text))

    # min_freq = 3?
    TEXT.build_vocab(train_ds, vectors=f"glove.twitter.27B.{emb_dim}d",
                     min_freq=2, unk_init=torch.Tensor.normal_)
    print(len(TEXT.vocab))

    return train_ds, train_iter, val_iter


def main():
    DATA_PATH = "./dataset/emotion_gif/train_gold.json"
    LABEL_PATH = "./dataset/emotion_gif/categories.json"
    label_names = pd.read_json(LABEL_PATH)[0].to_list()
    train_ds, train_dl, val_dl = create_dataloader(
            DATA_PATH, label_names, emb_dim=50)
    return train_ds.fields['text'].vocab


if __name__ == "__main__":
    main()
