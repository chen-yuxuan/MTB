from typing import List
from logging import getLogger

import torch
from torch import nn
from torch.optim import Adam


logger = getLogger(__name__)


def trainer(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 5,
    lr: float = 3.0e-5,
    device: torch.device = torch.device("/cpu:0"),
) -> float:
    # set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=lr)

    model.to(device)
