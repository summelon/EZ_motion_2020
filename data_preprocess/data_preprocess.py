import torchtext
import pandas as pd


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


def main():
    DATA_PATH = "../dataset/emotion_gif/train_gold.json"
    LABEL_PATH = "../dataset/emotion_gif/categories.json"
    label_names = pd.read_json(LABEL_PATH)[0].to_list()
    dataset, dataloader = create_dataloader(DATA_PATH, label_names)
    print(dataset)


if __name__ == "__main__":
    main()
