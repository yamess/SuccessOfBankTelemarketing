import torch

from bank_telemarketing.utils import model_performance


def train(model, dataloader, optimizer, criterion, device, pos_weight, cut_point=0.5):
    y_trues = []
    losses = []
    probs = []
    model.train()

    for batch in dataloader:
        x_emb = batch["x_emb"].to(device)
        x_cont = batch["x_cont"].to(device)
        y_true = batch["y"].to(device)

        optimizer.zero_grad()
        logits = model(x_cont, x_emb)
        logits = logits.squeeze(1)

        loss = criterion(logits, y_true.float())
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item() * len(batch))
        probs.extend(torch.sigmoid(logits).detach().tolist())
        y_trues.extend(y_true.detach().cpu().tolist())

    y_preds = [p >= cut_point for p in probs]

    train_performance = model_performance(
        y_true=y_trues,
        y_pred=y_preds,
        losses=losses,
        pos_weight=pos_weight,
    )
    return train_performance


def evaluate(model, dataloader, criterion, device, pos_weight, cut_point=0.5):
    y_trues = []
    losses = []
    probs = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            x_emb = batch["x_emb"].to(device)
            x_cont = batch["x_cont"].to(device)
            y_true = batch["y"].to(device)

            logits = model(x_cont, x_emb)
            logits = logits.squeeze(1)

            loss = criterion(logits, y_true.float())

            losses.append(loss.detach().cpu().item() * len(batch))
            probs.extend(torch.sigmoid(logits).detach().tolist())
            y_trues.extend(y_true.detach().cpu().tolist())

        y_preds = [p >= cut_point for p in probs]

        train_performance = model_performance(
            y_true=y_trues,
            y_pred=y_preds,
            losses=losses,
            pos_weight=pos_weight,
        )
        return train_performance
