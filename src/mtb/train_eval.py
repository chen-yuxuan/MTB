from typing import List, Callable
from logging import getLogger
from tqdm import tqdm, trange

import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score, classification_report


logger = getLogger(__name__)


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    label_names: List[str],
    batch_processor: Callable,
    num_epochs: int = 5,
    lr: float = 3.0e-5,
    device: torch.device = torch.device("cpu"),
) -> float:
    # set loss function, optimizer and model device
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=lr)
    model.to(device)

    for epoch in trange(num_epochs):
        model.train()
        loss_epoch, labels_list, preds_list = 0, [], []
        for batch in tqdm(train_loader, leave=False):
            tokenized, cues = batch_processor(batch)
            tokenized.pop("offset_mapping")
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            labels = torch.tensor(batch["relation_id"], device=device)

            outputs = model(x=tokenized, cues=cues)
            loss = criterion(outputs, labels)
            loss_epoch += loss.item()

            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy().tolist()
            preds_list.extend(preds)
            labels_list.extend(batch["relation_id"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_f1 = f1_score(
            labels_list,
            preds_list,
            labels=range(1, len(label_names)),
            average="micro",
        )
        logger.info(
            "Epoch [{}/{}], Training Loss: {:.4f}, Training F1: {:.4f}.".format(
                epoch + 1, num_epochs, loss_epoch, train_f1
            )
        )
    return train_f1


def eval(
    model: nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    label_names: List[str],
    batch_processor: Callable,
    device: torch.device = torch.device("cpu"),
) -> None:
    model.to(device)
    with torch.no_grad():
        model.eval()
        labels_list, preds_list = [], []
        for batch in tqdm(eval_loader):
            tokenized, cues = batch_processor(batch)
            tokenized.pop("offset_mapping")
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            cues.to(device)

            outputs = model(x=tokenized, cues=cues)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy().tolist()
            preds_list.extend(preds)
            labels_list.extend(batch["relation_id"])

    eval_f1 = f1_score(
        labels_list,
        preds_list,
        labels=range(1, len(label_names)),
        average="micro",
    )

    cls_report = classification_report(
        labels_list,
        preds_list,
        labels=range(0, len(label_names)),
        target_names=label_names,
    )
    with open("classification_report.txt", "a") as f:
        f.write(cls_report)

    return eval_f1
