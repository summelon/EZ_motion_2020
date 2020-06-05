import tqdm
import torch
import torchtext
import pandas as pd
import numpy as np

import model_zoo
import submission
import data_preprocess.data_preprocess as preprocess


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
    dataset, dataloader = preprocess.create_dataloader(DATA_PATH, label_names)

    VOCAB_SIZE = len(dataset.fields['text'].vocab)
    # EMBED_DIM = 32
    # model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLS).to(device)
    model = model_zoo.SimpleBiLSTMBaseline(
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
