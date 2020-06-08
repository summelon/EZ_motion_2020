import tqdm
import torch
import nltk
import torchtext
import pandas as pd


def create_dataloader(data_pt, vocab_src, emb_dim):
    tknz_func = nltk.TweetTokenizer().tokenize
    TEXT = torchtext.data.Field(tokenize=tknz_func, lower=True)

    test_ds = torchtext.data.TabularDataset(
            path=data_pt, format='json', skip_header=False,
            fields={'text': ('text', TEXT)})

    test_iter = torchtext.data.BucketIterator(
            dataset=test_ds, batch_size=64, sort_key=lambda x: len(x.text))

    TEXT.build_vocab(vocab_src, vectors=f"glove.twitter.27B.{emb_dim}d",
                     unk_init=torch.Tensor.normal_)

    return test_iter


def model_pred(model, device, data_ld, lbl_list):
    model.eval()
    model = model.to(device)
    preds = []
    with torch.no_grad():
        pbar = tqdm.tqdm(data_ld)
        for batch in pbar:
            text = batch.text.to(device)
            output = model(text)
            result = torch.argsort(output, dim=1)[:, -6:]
            for p in result:
                preds.append([[lbl_list[cat_idx] for cat_idx in result[0]]])

    return preds


def dump_submission(preds, data_pt, dump_file):
    df_preds = pd.DataFrame(preds, columns=["categories"])
    df_orig = pd.read_json(data_pt, lines=True)
    # print(df_preds)
    # print(df_orig.reply)
    df_rstl = pd.concat(
            [df_orig.idx, df_preds.categories, df_orig.reply, df_orig.text],
            axis=1)
    df_rstl.to_json(dump_file, orient='records', lines=True)


def main():
    PATH = "./dataset/emotion_gif/dev_unlabeled.json"
    df = pd.read_json(PATH, lines=True)
    cps = df.text.to_list()
    print(cps)


if __name__ == "__main__":
    main()
