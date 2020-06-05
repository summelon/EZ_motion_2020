import random
import torchtext
import pandas as pd
import numpy as np


def create_dataloader(data_pt, lbl_names):
    def categorical(gt):
        # Convert laebl to multi-one-hot
        label = [1 if cat in gt else 0 for cat in lbl_names]
        return label

    TEXT = torchtext.data.Field()
    LABELS = torchtext.data.Field(
            preprocessing=categorical, sequential=False, use_vocab=False)

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
            split_ratio=0.8, stratified=False, random_state=random.getstate())

    # NOTE: shuffle is True default
    train_iter, val_iter = torchtext.data.BucketIterator.splits(
            datasets=(train_ds, val_ds), batch_sizes=(32, 32),
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
