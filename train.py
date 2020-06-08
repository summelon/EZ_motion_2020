import copy
import tqdm
import torch
import pandas as pd

import model_zoo
import submission
import data_preprocess.data_preprocess as preprocess


def model_train(model, device, t_dl, v_dl, optmzr, crtrn, schdl):
    def cal_acc():
        pass
    model.to(device)
    train_loss, train_correct = 0, 0
    e_counter, gt_counter = 0, 0
    model.train()

    t_pbar = tqdm.tqdm(t_dl)
    for batch in t_pbar:
        optmzr.zero_grad()
        text = batch.text.to(device)
        label_oh = batch.label.type(torch.FloatTensor).to(device)
        output_logits = model(text)

        # FIXME use number of ground truth as additional info
        loss = crtrn(output_logits, label_oh)
        loss.backward()
        optmzr.step()

        label_idx = [torch.nonzero(l, as_tuple=False) for l in label_oh]
        output_idx = torch.argsort(output_logits, dim=1)[:, -6:]
        for l, o in zip(label_idx, output_idx):
            train_correct += sum([1 for elem in o if elem in l])
            gt_counter += l.nelement()

        e_counter += 1
        train_loss += loss.item()
        t_pbar.set_postfix(loss=f"{train_loss/e_counter:.6f}",
                           acc=f"{train_correct/gt_counter:.2%}")
    # schdl.step()

    # Validation ------------------------------------------------------
    val_loss, val_correct = 0, 0
    e_counter, gt_counter = 0, 0
    model.eval()

    v_pbar = tqdm.tqdm(v_dl)
    for batch in v_pbar:
        text = batch.text.to(device)
        label_oh = batch.label.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            output_logits = model(text)
            # FIXME use number of ground truth as additional info
            loss = crtrn(output_logits, label_oh)

        label_idx = [torch.nonzero(l, as_tuple=False) for l in label_oh]
        output_idx = torch.argsort(output_logits, dim=1)[:, -6:]
        for l, o in zip(label_idx, output_idx):
            val_correct += sum([1 for elem in o if elem in l])
            gt_counter += l.nelement()

        e_counter += 1
        val_loss += loss.item()
        v_pbar.set_postfix(loss=f"{val_loss/e_counter:.6f}",
                           acc=f"{val_correct/gt_counter:.2%}")
        tot_acc = val_correct / gt_counter

    return model, tot_acc


def dump(mode, emb_dim, model, device, label_names, dataset):
    if mode == "dev":
        DS_PATH = "./dataset/emotion_gif/dev_unlabeled.json"
        SUBMISSION_PATH = "./submit/dev.json"
    elif mode == "eval":
        DS_PATH = "./dataset/emotion_gif/test_unlabeled.json"
        SUBMISSION_PATH = "./submit/eval.json"

    test_ds = submission.create_dataloader(DS_PATH, dataset, emb_dim)
    predictions = submission.model_pred(model, device, test_ds, label_names)
    submission.dump_submission(predictions, DS_PATH, SUBMISSION_PATH)


def main():
    EMB_DIM = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = "./dataset/emotion_gif/train_gold.json"
    LABEL_PATH = "./dataset/emotion_gif/categories.json"
    label_names = pd.read_json(LABEL_PATH)[0].to_list()
    train_ds, train_dl, val_dl = preprocess.create_dataloader(
                                    DATA_PATH, label_names, emb_dim=EMB_DIM)

    # VOCAB_SIZE = len(train_ds.fields['text'].vocab)
    model = model_zoo.SimpleBiLSTMBaseline(
            hidden_dim=500, vocab=train_ds.fields['text'].vocab, pred_num=43)

    # optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = torch.nn.BCEWithLogitsLoss().to(device)
    # criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    import lossfn
    criterion = lossfn.FocalLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.9)

    best_acc, es_counter = 0, 0
    for epoch in range(1, 1000):
        print(f"\nThis is No.{epoch} epochs--------------------------------")
        print(es_counter)
        model, accuracy = model_train(model, device, train_dl, val_dl,
                                      optimizer, criterion, scheduler)
        if best_acc < accuracy:
            es_counter = 0
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            es_counter += 1
            scheduler.step()
            model.load_state_dict(best_model_wts)
            if es_counter > 10:
                break

    print(f"Best validation accuracy is {best_acc:.2%}")
    torch.save(model.state_dict(), "./model.pt")
    dump(mode="dev", emb_dim=EMB_DIM, model=model, device=device,
         label_names=label_names, dataset=train_ds)
    dump(mode="eval", emb_dim=EMB_DIM, model=model, device=device,
         label_names=label_names, dataset=train_ds)


if __name__ == "__main__":
    main()
