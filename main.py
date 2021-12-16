import logging

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from transformers import AutoTokenizer
import hydra
from omegaconf import DictConfig, OmegaConf

from mtb.data import TACREDDataset
from mtb.model import BERTModel
from mtb.tokenizer import BERTProcessor
from mtb.utils import resolve_relative_path


logger = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="configs")
def main(cfg: DictConfig) -> None:
    """
    Conducts evaluation given the configuration.
    Args:
        cfg: Hydra-format configuration given in a dict.
    """
    resolve_relative_path(cfg)
    print(OmegaConf.to_yaml(cfg))

    # prepare dataset: parse raw dataset and do some simple pre-processing such as
    # convert special tokens and insert entity markers
    entity_marker = True if cfg.variant in ["d", "e", "f"] else False
    train_dataset = TACREDDataset(cfg.train_file, entity_marker=entity_marker)
    val_dataset = TACREDDataset(cfg.val_file, entity_marker=entity_marker)

    # set dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True
    )

    # processor tokenizes and aligns all the tokens in a batch
    processor = BERTProcessor(
        tokenizer_name_or_path=cfg.model,
        entity_marker=entity_marker,
        max_length=cfg.max_length,
    )

    # build model
    model = None

    # set loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(params=model.parameters(), lr=cfg.lr)

    # start the training
    trainer(model, train_dataloader, val_dataloader, lr=cfg.lr)


if __name__ == "__main__":
    main()
