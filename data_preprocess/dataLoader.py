import torchtext


def main():
    TEXT = torchtext.data.Field()
    LABELS = torchtext.data.Filed
    DATA_PATH = "./dataset/emotion_gif/training_ds.csv"

    train_ds = torchtext.data.TabularDataset(
            path=DATA_PATH, format='csv',
            fields=[('text', TEXT), ('label', LABELS)])

    # FIXME: batch_size_fn is similar to collate_fn in Dataloader
    # NOTE: shuffle is True default
    train_iter = torchtext.data.BucketIterator(
            dataset=train_ds, batch_size=8,
            sort_key=lambda x: len(x.text), device=0)

    TEXT.build_vocab(train_ds)
    LABELS.build_vocab(train_ds)

    print(list(train_iter)[0].text)


if __name__ == "__main__":
    main()
