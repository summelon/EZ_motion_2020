import os
import pandas as pd
import unidecode
import tqdm
from argparse import ArgumentParser


def get_corpus(path: str):
    cps = pd.read_json(path, lines=True)
    del cps['idx'], cps['mp4']

    return cps


def concat_text_reply(text: str, reply: str) -> str:
    if reply == '':
        doc = text
    else:
        doc = text + " " + reply

    return unidecode.unidecode(doc).replace("\n", " ")


def dump_dataset_json(train_path: str, label_path: str, dump_file: str):
    corpus = get_corpus(train_path)
    label_list = pd.read_json(label_path)[0].to_list()

    dataset = {'text': [], 'label': []}
    pbar = tqdm.tqdm(range(len(corpus)))
    for i in pbar:
        dataset['text'].append(
                concat_text_reply(corpus.text[i], corpus.reply[i]))
        dataset['label'].append(
                [label_list.index(cat) for cat in corpus.categories[i]])

    assert len(dataset['text']) == len(dataset['label']), f"Length different!"
    dataset_df = pd.DataFrame(dataset, columns=dataset.keys())
    dataset_df.to_csv(dump_file, index=False)

    return True


def params_loader():
    parser = ArgumentParser(description=f"Dump tabular dataset include "
                                        f"\"text\" and \"label\" columns")
    parser.add_argument("--dataset_dir", type=str, help=f"Path to dataset")
    parser.add_argument("--dump_file", type=str,
                        help=f"Path/name.csv of dump dataset")
    args, _ = parser.parse_known_args()
    params = {k: v for k, v in vars(args).items() if v is not None}

    return params


def run(params):
    train_path = os.path.join(params['dataset_dir'], "train_gold.json")
    label_path = os.path.join(params['dataset_dir'], "categories.json")
    dump_dataset_json(train_path, label_path, params['dump_file'])

    return f"Dump dataset in {params['dump_file']}"


if __name__ == "__main__":
    p = params_loader()
    print(run(p))
